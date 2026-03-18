"""
LLM processing module: chunk-aware memory materialization.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from typing import Any, Dict, Iterable, List, Optional

from config import (
    API_PROXY_KEY,
    API_PROXY_MODEL,
    API_PROXY_URL,
    BEDROCK_LLM_MAX_OUTPUT_TOKENS,
    BEDROCK_REGION,
    GEMINI_API_KEY,
    LLM_PROVIDER,
    LLM_BURST_GAP_SECONDS,
    LLM_BURST_MAX_DURATION_SECONDS,
    LLM_BURST_MAX_PHOTOS,
    LLM_SESSION_HARD_DISTANCE_KM,
    LLM_SESSION_HARD_GAP_SECONDS,
    LLM_SESSION_NEAR_DISTANCE_KM,
    LLM_SESSION_SOFT_GAP_SECONDS,
    LLM_SESSION_STRONG_GAP_SECONDS,
    LLM_MODEL,
    LLM_SLICE_MAX_DENSITY_SCORE,
    LLM_SLICE_HARD_MAX_BURSTS,
    LLM_SLICE_MAX_BURSTS,
    LLM_SLICE_MAX_INFO_SCORE,
    LLM_SLICE_MAX_PHOTOS,
    LLM_SLICE_MAX_RARE_CLUES,
    LLM_SLICE_MIN_PHOTOS,
    LLM_SLICE_OVERLAP_BURSTS,
    MAX_RETRIES,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_LLM_MODEL,
    OPENROUTER_SITE_URL,
    RETRY_DELAY,
)
from models import Event, Relationship
from services.bedrock_runtime import (
    build_bedrock_client,
    build_inference_config,
    build_text_message,
    extract_text_from_converse_response,
    resolve_bedrock_model_candidates,
    should_try_next_bedrock_model,
)


class LLMProcessor:
    """Chunk-aware LLM processor with Bedrock support."""

    def __init__(self):
        self.provider = LLM_PROVIDER
        self.use_proxy = self.provider == "proxy"
        self.use_openrouter = self.provider == "openrouter"
        self.use_bedrock = self.provider == "bedrock"
        self.model = OPENROUTER_LLM_MODEL if self.use_openrouter else LLM_MODEL
        self.requests = None
        self.genai = None
        self.bedrock_client = None
        self.bedrock_model_candidates: List[str] = []
        self.last_memory_contract: Dict[str, Any] | None = None
        self.last_chunk_artifacts: Dict[str, Any] = {}

        if self.use_proxy:
            if not API_PROXY_URL or not API_PROXY_KEY:
                raise ValueError("使用代理服务需要配置 API_PROXY_URL 和 API_PROXY_KEY")
            try:
                import requests
            except ModuleNotFoundError:
                requests = None
            self.requests = requests
            self.proxy_url = API_PROXY_URL
            self.proxy_key = API_PROXY_KEY
            self.proxy_model = API_PROXY_MODEL
            print(f"[LLM] 使用代理服务: {self.proxy_url}")
        elif self.use_openrouter:
            try:
                import requests
            except ModuleNotFoundError:
                requests = None
            self.requests = requests
            self.openrouter_api_key = OPENROUTER_API_KEY or GEMINI_API_KEY
            if not self.openrouter_api_key:
                raise ValueError("使用 OpenRouter 需要配置 OPENROUTER_API_KEY 或 GEMINI_API_KEY")
            self.openrouter_base_url = OPENROUTER_BASE_URL.rstrip("/")
            self.openrouter_site_url = OPENROUTER_SITE_URL
            self.openrouter_app_name = OPENROUTER_APP_NAME
            print(f"[LLM] 使用 OpenRouter: {self.model}")
        elif self.use_bedrock:
            self.bedrock_client = build_bedrock_client(BEDROCK_REGION)
            self.bedrock_model_candidates = resolve_bedrock_model_candidates(
                [self.model],
                BEDROCK_REGION,
            )
            if self.bedrock_model_candidates:
                self.model = self.bedrock_model_candidates[0]
            print(f"[LLM] 使用 Bedrock: {self.model} @ {BEDROCK_REGION}")
        else:
            from google import genai

            self.genai = genai
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            print("[LLM] 使用官方 Gemini API")

    def _coerce_text_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part).strip()
        if isinstance(content, dict):
            text = content.get("text")
            if isinstance(text, str):
                return text.strip()
        raise ValueError(f"无法从内容中提取文本: {type(content).__name__}")

    def _extract_json_payload(self, raw_text: str) -> Dict[str, Any]:
        text = str(raw_text or "").strip()
        text = text.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "").strip()

        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

        decoder = json.JSONDecoder()
        candidates = [text]
        first_json_start = min((idx for idx in (text.find("{"), text.find("[")) if idx != -1), default=-1)
        if first_json_start != -1:
            candidates.append(text[first_json_start:])

        last_error = None
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                last_error = exc
                try:
                    payload, _end = decoder.raw_decode(candidate)
                    if isinstance(payload, dict):
                        return payload
                    return {"items": payload}
                except json.JSONDecodeError:
                    continue

        if last_error:
            raise last_error
        raise ValueError("无法解析 JSON payload")

    def _is_retryable_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        retry_keywords = ["429", "rate limit", "connection", "timeout", "temporarily unavailable", "reset by peer", "throttl", "too many requests"]
        return any(keyword in message for keyword in retry_keywords)

    def _call_with_retries(self, callback):
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                return callback()
            except Exception as exc:
                last_error = exc
                if attempt == MAX_RETRIES - 1 or not self._is_retryable_error(exc):
                    raise
                delay_seconds = RETRY_DELAY * (2 ** attempt)
                print(f"[LLM] 可重试错误，{delay_seconds}s 后重试 ({attempt + 1}/{MAX_RETRIES}): {exc}")
                time.sleep(delay_seconds)
        raise last_error

    def extract_memory_contract(
        self,
        vlm_results: List[Dict[str, Any]],
        face_db: Optional[Dict[str, Any]] = None,
        primary_person_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not vlm_results:
            contract = self._empty_contract()
            self.last_memory_contract = contract
            self.last_chunk_artifacts = {
                "photo_fact_count": 0,
                "raw_session_count": 0,
                "slice_count": 0,
                "slices": [],
                "session_summaries": [],
            }
            return contract

        photo_facts = self._build_photo_fact_buffer(vlm_results)
        bursts = self._build_bursts(photo_facts)
        raw_sessions = self._build_raw_sessions(bursts)
        session_slices = self._build_session_slices(raw_sessions)

        slice_contracts = []
        slice_artifacts = []
        slice_contracts_by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for session_slice in session_slices:
            prompt = self._create_slice_memory_prompt(session_slice["evidence_packet"], primary_person_id)
            response = self._call_with_retries(lambda prompt=prompt: self._call_json_prompt(prompt))
            normalized = self._normalize_memory_contract(response)
            slice_contracts.append(normalized)
            slice_contracts_by_session[session_slice["raw_session_id"]].append(normalized)
            slice_artifacts.append(
                {
                    "slice_id": session_slice["slice_id"],
                    "raw_session_id": session_slice["raw_session_id"],
                    "photo_ids": list(session_slice["photo_ids"]),
                    "burst_ids": list(session_slice["burst_ids"]),
                    "overlap_burst_ids": list(session_slice["overlap_burst_ids"]),
                    "rare_clue_count": session_slice["rare_clue_count"],
                    "photo_count": len(session_slice["facts"]),
                    "burst_count": len(session_slice["bursts"]),
                    "fact_inventory_count": len(session_slice["evidence_packet"]["fact_inventory"]),
                    "change_point_count": len(session_slice["evidence_packet"]["change_points"]),
                    "location_chain": list(session_slice["evidence_packet"]["location_chain"]),
                    "dominant_person_ids": list(session_slice["evidence_packet"]["dominant_person_ids"]),
                    "conflict_count": len(session_slice["evidence_packet"]["conflicts"]),
                    "information_score": float(session_slice.get("information_score") or 0.0),
                    "density_score": float(session_slice.get("density_score") or 0.0),
                    "contract_counts": self._contract_counts(normalized),
                }
            )

        session_contracts = []
        session_artifacts = []
        for raw_session in raw_sessions:
            session_slice_records = [
                artifact
                for artifact in slice_artifacts
                if artifact["raw_session_id"] == raw_session["raw_session_id"]
            ]
            session_prompt = self._create_session_merge_prompt(
                raw_session=raw_session,
                session_slice_records=session_slice_records,
                slice_contracts=slice_contracts_by_session.get(raw_session["raw_session_id"], []),
                primary_person_id=primary_person_id,
            )
            session_contract = self._call_with_retries(lambda prompt=session_prompt: self._call_json_prompt(prompt))
            normalized_session_contract = self._normalize_memory_contract(session_contract)
            session_contracts.append(normalized_session_contract)
            session_artifacts.append(
                {
                    "raw_session_id": raw_session["raw_session_id"],
                    "started_at": raw_session["started_at"],
                    "ended_at": raw_session["ended_at"],
                    "location_chain": list(raw_session["location_chain"]),
                    "photo_count": len(raw_session["facts"]),
                    "burst_count": len(raw_session["bursts"]),
                    "slice_ids": [item["slice_id"] for item in session_slice_records],
                    "continuity_decisions": list(raw_session["continuity_decisions"]),
                    "information_score": round(
                        sum(float(item.get("information_score") or 0.0) for item in session_slice_records),
                        3,
                    ),
                    "density_score": round(
                        sum(float(item.get("density_score") or 0.0) for item in session_slice_records),
                        3,
                    ),
                    "contract_counts": self._contract_counts(normalized_session_contract),
                }
            )

        merge_prompt = self._create_merge_prompt(
            photo_fact_count=len(photo_facts),
            raw_sessions=raw_sessions,
            session_slices=session_slices,
            session_contracts=session_contracts,
            session_artifacts=session_artifacts,
            primary_person_id=primary_person_id,
        )
        merged_contract = self._call_with_retries(lambda: self._call_json_prompt(merge_prompt))
        contract = self._normalize_memory_contract(merged_contract)

        self.last_memory_contract = contract
        self.last_chunk_artifacts = {
            "photo_fact_count": len(photo_facts),
            "burst_count": len(bursts),
            "raw_session_count": len(raw_sessions),
            "session_contract_count": len(session_contracts),
            "slice_count": len(session_slices),
            "bursts": [
                {
                    "burst_id": burst["burst_id"],
                    "photo_ids": list(burst["photo_ids"]),
                    "started_at": burst["started_at"],
                    "ended_at": burst["ended_at"],
                    "location_bucket": burst["location_bucket"],
                    "dominant_person_ids": list(burst["dominant_person_ids"]),
                    "rare_clue_count": len(burst["rare_clues"]),
                    "information_score": float(burst.get("information_score") or 0.0),
                    "photo_density_score": float(burst.get("photo_density_score") or 0.0),
                }
                for burst in bursts
            ],
            "session_summaries": session_artifacts,
            "slices": slice_artifacts,
            "final_contract_counts": self._contract_counts(contract),
        }
        return contract

    def events_from_memory_contract(self, memory_contract: Dict[str, Any]) -> List[Event]:
        events: List[Event] = []
        for index, item in enumerate(memory_contract.get("events", []), start=1):
            started_at = str(item.get("started_at") or "")
            ended_at = str(item.get("ended_at") or started_at)
            date, time_range = self._legacy_time_fields(started_at, ended_at)
            participants = [str(value) for value in item.get("participant_person_ids", []) or item.get("participants", [])]
            tags = [str(value) for value in item.get("event_facets", [])]
            alternative = item.get("alternative_type_candidates", [])
            meta_info = {
                "title": item.get("title", ""),
                "timestamp": f"{started_at} - {ended_at}" if started_at else "",
                "location_context": item.get("location", ""),
                "photo_count": len(item.get("photo_ids", [])),
                "original_image_ids": list(item.get("original_image_ids", []) or item.get("photo_ids", [])),
                "coarse_event_type": item.get("coarse_event_type", "unknown"),
                "alternative_type_candidates": alternative,
            }
            objective_fact = {
                "scene_description": item.get("description", ""),
                "participants": participants,
            }
            events.append(
                Event(
                    event_id=str(item.get("event_id") or f"EVT_{index:03d}"),
                    date=date,
                    time_range=time_range,
                    duration="",
                    title=str(item.get("title") or ""),
                    type=str(item.get("coarse_event_type") or "其他"),
                    participants=participants,
                    location=str(item.get("location") or ""),
                    description=str(item.get("description") or ""),
                    photo_count=len(item.get("photo_ids", [])),
                    confidence=float(item.get("confidence") or 0.0),
                    reason=str(item.get("reason") or ""),
                    narrative_synthesis=str(item.get("narrative_synthesis") or item.get("description") or ""),
                    meta_info=meta_info,
                    objective_fact=objective_fact,
                    evidence_photos=list(item.get("photo_ids", [])),
                    tags=tags,
                    persona_evidence=item.get("persona_evidence", {}) or {},
                    social_dynamics=item.get("social_dynamics", []) or [],
                )
            )
        return events

    def relationships_from_memory_contract(self, memory_contract: Dict[str, Any]) -> List[Relationship]:
        relationships: List[Relationship] = []
        for item in memory_contract.get("relationship_hypotheses", []):
            person_id = str(item.get("person_id") or item.get("target_person_id") or "")
            if not person_id:
                continue
            relationships.append(
                Relationship(
                    person_id=person_id,
                    relationship_type=str(item.get("relationship_type") or "acquaintance"),
                    label=str(item.get("label") or "熟人"),
                    confidence=float(item.get("confidence") or 0.0),
                    evidence={
                        **(item.get("evidence", {}) or {}),
                        "supporting_event_ids": [str(value) for value in item.get("supporting_event_ids", []) or []],
                        "supporting_photo_ids": [str(value) for value in item.get("supporting_photo_ids", []) or []],
                    },
                    reason=str(item.get("reason") or item.get("reason_summary") or ""),
                )
            )
        return relationships

    def profile_markdown_from_memory_contract(
        self,
        memory_contract: Dict[str, Any],
        primary_person_id: Optional[str],
    ) -> str:
        profile_deltas = memory_contract.get("profile_deltas", [])
        uncertainty = memory_contract.get("uncertainty", [])
        if not profile_deltas and not uncertainty:
            return "# Profile\n\n- no durable profile deltas extracted"

        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in profile_deltas:
            profile_key = str(item.get("profile_key") or "general_profile")
            grouped[profile_key].append(item)

        lines = ["# Profile", ""]
        if primary_person_id:
            lines.append(f"- primary_person_id_hint: {primary_person_id}")
            lines.append("")
        for profile_key in sorted(grouped):
            lines.append(f"## {profile_key}")
            lines.append("")
            for item in sorted(grouped[profile_key], key=lambda payload: (str(payload.get("field_key") or ""), -float(payload.get("confidence") or 0.0))):
                summary = str(item.get("summary") or item.get("value_summary") or item.get("field_value") or "")
                confidence = float(item.get("confidence") or 0.0)
                lines.append(f"- {item.get('field_key')}: {summary} (confidence={confidence:.2f})")
            lines.append("")
        if uncertainty:
            lines.append("## uncertainty")
            lines.append("")
            for item in uncertainty:
                lines.append(f"- {item.get('field')}: {item.get('status')} ({item.get('reason')})")
        return "\n".join(lines).strip()

    def extract_events(self, vlm_results: List[Dict], primary_person_id: Optional[str] = None) -> List[Event]:
        contract = self.last_memory_contract
        if contract is None:
            contract = self.extract_memory_contract(vlm_results, primary_person_id=primary_person_id)
        return self.events_from_memory_contract(contract)

    def infer_relationships(
        self,
        vlm_results: List[Dict],
        face_db: Dict,
        primary_person_id: Optional[str],
    ) -> List[Relationship]:
        contract = self.last_memory_contract
        if contract is None:
            contract = self.extract_memory_contract(vlm_results, face_db=face_db, primary_person_id=primary_person_id)
        return self.relationships_from_memory_contract(contract)

    def generate_profile(
        self,
        events: List[Event],
        relationships: List[Relationship],
        primary_person_id: Optional[str],
    ) -> str:
        if self.last_memory_contract is not None:
            return self.profile_markdown_from_memory_contract(self.last_memory_contract, primary_person_id)
        return self._create_legacy_profile(events, relationships, primary_person_id)

    def _build_photo_fact_buffer(self, vlm_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        facts = []
        for item in sorted(vlm_results, key=lambda payload: payload.get("timestamp") or ""):
            analysis = item.get("vlm_analysis", {}) if isinstance(item.get("vlm_analysis"), dict) else {}
            scene = analysis.get("scene", {}) if isinstance(analysis, dict) else {}
            event = analysis.get("event", {}) if isinstance(analysis, dict) else {}
            location = item.get("location") or {}
            place_candidates = []
            for candidate in analysis.get("place_candidates", []) or []:
                if isinstance(candidate, dict):
                    place_candidates.append(
                        {
                            "name": str(candidate.get("name") or "").strip(),
                            "confidence": float(candidate.get("confidence") or 0.0),
                            "reason": str(candidate.get("reason") or "").strip(),
                        }
                    )
                elif candidate:
                    place_candidates.append(
                        {
                            "name": str(candidate).strip(),
                            "confidence": 0.0,
                            "reason": "",
                        }
                    )
            location_name = str(
                (location or {}).get("name")
                or (place_candidates[0]["name"] if place_candidates and place_candidates[0]["name"] else "")
                or scene.get("location_detected")
                or ""
            ).strip()
            person_ids = [str(value) for value in item.get("face_person_ids", []) if value]
            rare_clues = self._collect_rare_clues(analysis)
            facts.append(
                {
                    "photo_id": item.get("photo_id"),
                    "timestamp": item.get("timestamp"),
                    "location": location,
                    "location_name": location_name,
                    "person_ids": person_ids,
                    "scene_hint": str(scene.get("location_detected") or ""),
                    "activity_hint": str(event.get("activity") or ""),
                    "social_hint": str(event.get("social_context") or ""),
                    "summary": str(analysis.get("summary") or ""),
                    "details": list(analysis.get("details", []) or []),
                    "key_objects": list(analysis.get("key_objects", []) or []),
                    "ocr_hits": [str(value) for value in analysis.get("ocr_hits", []) or [] if value],
                    "brands": [str(value) for value in analysis.get("brands", []) or [] if value],
                    "route_plan_clues": [str(value) for value in analysis.get("route_plan_clues", []) or [] if value],
                    "transport_clues": [str(value) for value in analysis.get("transport_clues", []) or [] if value],
                    "health_treatment_clues": [str(value) for value in analysis.get("health_treatment_clues", []) or [] if value],
                    "object_last_seen_clues": [str(value) for value in analysis.get("object_last_seen_clues", []) or [] if value],
                    "place_candidates": place_candidates,
                    "raw_structured_observations": list(analysis.get("raw_structured_observations", []) or []),
                    "uncertainty": list(analysis.get("uncertainty", []) or []),
                    "rare_clues": rare_clues,
                    "raw": item,
                }
            )
        return facts

    def _collect_rare_clues(self, analysis: Dict[str, Any]) -> List[str]:
        candidates: List[str] = []
        for key in ("ocr_hits", "brands", "route_plan_clues", "transport_clues", "health_treatment_clues", "object_last_seen_clues"):
            for value in analysis.get(key, []) or []:
                text = str(value).strip()
                if text and text not in candidates:
                    candidates.append(text)
        for item in analysis.get("raw_structured_observations", []) or []:
            if not isinstance(item, dict):
                continue
            value = str(item.get("value") or "").strip()
            field = str(item.get("field") or "").strip()
            if value and (field or value) not in candidates:
                candidates.append(f"{field}: {value}" if field else value)
        return candidates

    def _build_bursts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not facts:
            return []
        bursts: List[List[Dict[str, Any]]] = [[facts[0]]]
        for fact in facts[1:]:
            current = bursts[-1]
            previous = current[-1]
            gap = self._seconds_between(previous.get("timestamp"), fact.get("timestamp"))
            duration = self._seconds_between(current[0].get("timestamp"), fact.get("timestamp"))
            if (
                gap <= LLM_BURST_GAP_SECONDS
                and duration <= LLM_BURST_MAX_DURATION_SECONDS
                and len(current) < LLM_BURST_MAX_PHOTOS
                and not self._burst_break_required(previous, fact)
            ):
                current.append(fact)
            else:
                bursts.append([fact])
        burst_payloads = []
        for index, group in enumerate(bursts, start=1):
            dominant_person_ids = self._dominant_person_ids_from_facts(group)
            duration_seconds = max(
                1,
                int(self._seconds_between(group[0].get("timestamp"), group[-1].get("timestamp"))),
            )
            information_score = self._estimate_information_score(group)
            density_score = self._estimate_density_score(group, duration_seconds)
            burst_payloads.append(
                {
                    "burst_id": f"burst_{index:04d}",
                    "photo_ids": [item["photo_id"] for item in group],
                    "started_at": group[0]["timestamp"],
                    "ended_at": group[-1]["timestamp"],
                    "facts": group,
                    "location_bucket": self._location_bucket(group[0]),
                    "location_chain": self._unique(self._location_bucket(item) for item in group),
                    "dominant_person_ids": dominant_person_ids,
                    "activity_hints": self._unique(item.get("activity_hint") for item in group),
                    "scene_hints": self._unique(item.get("scene_hint") for item in group),
                    "duration_seconds": duration_seconds,
                    "photo_density_score": density_score,
                    "information_score": information_score,
                    "rare_clues": self._unique(
                        clue
                        for item in group
                        for clue in item.get("rare_clues", [])
                    ),
                    "fact_inventory": self._build_fact_inventory(group),
                }
            )
        return burst_payloads

    def _build_raw_sessions(self, bursts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not bursts:
            return []
        raw_sessions: List[List[Dict[str, Any]]] = [[bursts[0]]]
        continuity_decisions: List[List[Dict[str, Any]]] = [[]]
        for burst in bursts[1:]:
            current = raw_sessions[-1]
            previous = current[-1]
            continuity = self._session_continuity(previous, burst)
            if continuity["same_session"]:
                current.append(burst)
                continuity_decisions[-1].append(continuity)
            else:
                raw_sessions.append([burst])
                continuity_decisions.append([])
        payloads = []
        for index, group in enumerate(raw_sessions, start=1):
            facts = [fact for burst in group for fact in burst["facts"]]
            payloads.append(
                {
                    "raw_session_id": f"raw_session_{index:04d}",
                    "started_at": group[0]["started_at"],
                    "ended_at": group[-1]["ended_at"],
                    "location_bucket": group[0]["location_bucket"],
                    "location_chain": self._unique(
                        location_name
                        for burst in group
                        for location_name in burst.get("location_chain", [])
                    ),
                    "photo_ids": [photo_id for burst in group for photo_id in burst["photo_ids"]],
                    "burst_ids": [burst["burst_id"] for burst in group],
                    "bursts": list(group),
                    "facts": facts,
                    "dominant_person_ids": self._dominant_person_ids_from_facts(facts),
                    "continuity_decisions": continuity_decisions[index - 1],
                }
            )
        return payloads

    def _build_session_slices(self, raw_sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        slices: List[Dict[str, Any]] = []
        slice_index = 1
        for raw_session in raw_sessions:
            bursts = list(raw_session.get("bursts", []))
            if not bursts:
                continue
            start_index = 0
            while start_index < len(bursts):
                end_index = start_index
                selected: List[Dict[str, Any]] = []
                photo_count = 0
                rare_clue_count = 0
                information_score = 0.0
                density_score = 0.0
                while end_index < len(bursts) and len(selected) < LLM_SLICE_HARD_MAX_BURSTS:
                    burst = bursts[end_index]
                    projected_photo_count = photo_count + len(burst["facts"])
                    projected_rare_count = rare_clue_count + len(burst["rare_clues"])
                    projected_information_score = information_score + float(burst.get("information_score") or 0.0)
                    projected_density_score = density_score + float(burst.get("photo_density_score") or 0.0)
                    if selected:
                        if len(selected) >= LLM_SLICE_MAX_BURSTS:
                            break
                        if projected_photo_count > LLM_SLICE_MAX_PHOTOS:
                            break
                        if projected_rare_count > LLM_SLICE_MAX_RARE_CLUES and photo_count >= LLM_SLICE_MIN_PHOTOS:
                            break
                        if projected_information_score > LLM_SLICE_MAX_INFO_SCORE and photo_count >= LLM_SLICE_MIN_PHOTOS:
                            break
                        if projected_density_score > LLM_SLICE_MAX_DENSITY_SCORE and photo_count >= LLM_SLICE_MIN_PHOTOS:
                            break
                    selected.append(burst)
                    photo_count = projected_photo_count
                    rare_clue_count = projected_rare_count
                    information_score = projected_information_score
                    density_score = projected_density_score
                    end_index += 1
                if not selected:
                    selected = [bursts[start_index]]
                    end_index = start_index + 1
                    photo_count = len(selected[0]["facts"])
                    rare_clue_count = len(selected[0]["rare_clues"])
                    information_score = float(selected[0].get("information_score") or 0.0)
                    density_score = float(selected[0].get("photo_density_score") or 0.0)
                overlap_bursts = (
                    [burst["burst_id"] for burst in selected[:LLM_SLICE_OVERLAP_BURSTS]]
                    if start_index > 0
                    else []
                )
                slices.append(
                    self._slice_payload(
                        raw_session=raw_session,
                        bursts=selected,
                        rare_clue_count=rare_clue_count,
                        information_score=information_score,
                        density_score=density_score,
                        slice_index=slice_index,
                        overlap_burst_ids=overlap_bursts,
                    )
                )
                slice_index += 1
                if end_index >= len(bursts):
                    break
                start_index = max(end_index - LLM_SLICE_OVERLAP_BURSTS, start_index + 1)
        return slices

    def _slice_payload(
        self,
        raw_session: Dict[str, Any],
        bursts: List[Dict[str, Any]],
        rare_clue_count: int,
        information_score: float,
        density_score: float,
        slice_index: int,
        overlap_burst_ids: List[str],
    ) -> Dict[str, Any]:
        slice_id = f"slice_{slice_index:04d}"
        facts = [fact for burst in bursts for fact in burst["facts"]]
        evidence_packet = self._build_slice_evidence_packet(
            raw_session=raw_session,
            bursts=bursts,
            facts=facts,
            overlap_burst_ids=overlap_burst_ids,
        )
        evidence_packet["slice_id"] = slice_id
        return {
            "slice_id": slice_id,
            "raw_session_id": raw_session["raw_session_id"],
            "photo_ids": [fact["photo_id"] for fact in facts],
            "burst_ids": [burst["burst_id"] for burst in bursts],
            "overlap_burst_ids": overlap_burst_ids,
            "started_at": facts[0]["timestamp"],
            "ended_at": facts[-1]["timestamp"],
            "location_bucket": self._location_bucket(facts[0]),
            "rare_clue_count": rare_clue_count,
            "information_score": round(float(information_score), 3),
            "density_score": round(float(density_score), 3),
            "bursts": list(bursts),
            "facts": list(facts),
            "evidence_packet": evidence_packet,
        }

    def _build_slice_evidence_packet(
        self,
        *,
        raw_session: Dict[str, Any],
        bursts: List[Dict[str, Any]],
        facts: List[Dict[str, Any]],
        overlap_burst_ids: List[str],
    ) -> Dict[str, Any]:
        location_chain = self._unique(
            location_name
            for burst in bursts
            for location_name in burst.get("location_chain", [])
        )
        rare_clues = self._unique(
            clue
            for fact in facts
            for clue in fact.get("rare_clues", [])
        )
        return {
            "session_id": raw_session["raw_session_id"],
            "slice_id": "",
            "time_range": {
                "start": facts[0]["timestamp"],
                "end": facts[-1]["timestamp"],
            },
            "location_chain": location_chain,
            "dominant_person_ids": self._dominant_person_ids_from_facts(facts),
            "burst_ids": [burst["burst_id"] for burst in bursts],
            "overlap_burst_ids": list(overlap_burst_ids),
            "fact_inventory": self._build_fact_inventory(facts),
            "rare_clues": rare_clues,
            "change_points": self._build_change_points(bursts),
            "conflicts": self._detect_slice_conflicts(facts),
            "slice_budget_metrics": {
                "photo_count": len(facts),
                "burst_count": len(bursts),
                "rare_clue_count": len(rare_clues),
                "information_score": round(sum(float(burst.get("information_score") or 0.0) for burst in bursts), 3),
                "density_score": round(sum(float(burst.get("photo_density_score") or 0.0) for burst in bursts), 3),
            },
            "photo_refs": [fact["photo_id"] for fact in facts],
            "photo_facts": [
                {
                    "photo_id": fact["photo_id"],
                    "timestamp": fact["timestamp"],
                    "location_name": fact.get("location_name", ""),
                    "person_ids": fact.get("person_ids", []),
                    "scene_hint": fact.get("scene_hint", ""),
                    "activity_hint": fact.get("activity_hint", ""),
                    "social_hint": fact.get("social_hint", ""),
                    "summary": fact.get("summary", ""),
                    "rare_clues": fact.get("rare_clues", []),
                }
                for fact in facts
            ],
            "session_context": {
                "raw_session_id": raw_session["raw_session_id"],
                "raw_session_started_at": raw_session["started_at"],
                "raw_session_ended_at": raw_session["ended_at"],
                "raw_session_location_chain": list(raw_session["location_chain"]),
            },
        }

    def _build_fact_inventory(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        aggregated: Dict[tuple[str, str], Dict[str, Any]] = {}

        def add_fact(fact_type: str, value: str, photo_id: str, confidence: float = 0.0) -> None:
            normalized_value = str(value or "").strip()
            if not normalized_value:
                return
            key = (fact_type, normalized_value)
            item = aggregated.setdefault(
                key,
                {
                    "fact_type": fact_type,
                    "value": normalized_value,
                    "support_count": 0,
                    "photo_ids": [],
                    "confidence": 0.0,
                },
            )
            item["support_count"] += 1
            if photo_id not in item["photo_ids"]:
                item["photo_ids"].append(photo_id)
            item["confidence"] = max(float(item["confidence"]), float(confidence or 0.0))

        for fact in facts:
            photo_id = str(fact.get("photo_id") or "")
            add_fact("scene_location", str(fact.get("scene_hint") or ""), photo_id)
            add_fact("activity", str(fact.get("activity_hint") or ""), photo_id)
            add_fact("social_context", str(fact.get("social_hint") or ""), photo_id)
            add_fact("summary", str(fact.get("summary") or ""), photo_id)
            for value in fact.get("key_objects", []) or []:
                add_fact("object", str(value), photo_id)
            for value in fact.get("ocr_hits", []) or []:
                add_fact("ocr", str(value), photo_id, confidence=0.95)
            for value in fact.get("brands", []) or []:
                add_fact("brand", str(value), photo_id, confidence=0.9)
            for value in fact.get("route_plan_clues", []) or []:
                add_fact("route_plan", str(value), photo_id, confidence=0.8)
            for value in fact.get("transport_clues", []) or []:
                add_fact("transport", str(value), photo_id, confidence=0.8)
            for value in fact.get("health_treatment_clues", []) or []:
                add_fact("health_treatment", str(value), photo_id, confidence=0.8)
            for value in fact.get("object_last_seen_clues", []) or []:
                add_fact("object_last_seen", str(value), photo_id, confidence=0.8)
            for candidate in fact.get("place_candidates", []) or []:
                if not isinstance(candidate, dict):
                    continue
                add_fact(
                    "place_candidate",
                    str(candidate.get("name") or ""),
                    photo_id,
                    confidence=float(candidate.get("confidence") or 0.0),
                )
        inventory = list(aggregated.values())
        inventory.sort(key=lambda item: (item["support_count"], item["confidence"], item["fact_type"], item["value"]), reverse=True)
        return inventory

    def _estimate_information_score(self, facts: List[Dict[str, Any]]) -> float:
        weights = {
            "ocr": 3.5,
            "brand": 3.0,
            "route_plan": 3.0,
            "health_treatment": 3.0,
            "object_last_seen": 3.0,
            "place_candidate": 2.5,
            "transport": 2.0,
            "activity": 1.5,
            "scene_location": 1.5,
            "social_context": 1.0,
            "object": 0.8,
            "summary": 0.4,
        }
        score = 0.0
        for item in self._build_fact_inventory(facts):
            fact_type = str(item.get("fact_type") or "")
            support_count = int(item.get("support_count") or 0)
            confidence = float(item.get("confidence") or 0.0)
            weight = weights.get(fact_type, 0.75)
            support_multiplier = min(1.5, 0.5 + (support_count * 0.25))
            confidence_multiplier = 1.0 + min(0.5, confidence * 0.5)
            score += weight * support_multiplier * confidence_multiplier
        rare_clue_count = len(self._unique(clue for fact in facts for clue in fact.get("rare_clues", [])))
        score += rare_clue_count * 1.5
        return round(score, 3)

    def _estimate_density_score(self, facts: List[Dict[str, Any]], duration_seconds: int) -> float:
        if not facts:
            return 0.0
        duration_minutes = max(1.0, float(duration_seconds) / 60.0)
        photos_per_minute = len(facts) / duration_minutes
        return round(min(8.0, photos_per_minute), 3)

    def _build_change_points(self, bursts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        change_points: List[Dict[str, Any]] = []
        seen_rare_clues: set[str] = set()
        for burst in bursts:
            new_clues = [clue for clue in burst.get("rare_clues", []) if clue not in seen_rare_clues]
            if new_clues:
                change_points.append(
                    {
                        "change_type": "rare_clue_emergence",
                        "burst_id": burst["burst_id"],
                        "photo_ids": list(burst["photo_ids"]),
                        "details": new_clues[:5],
                    }
                )
                seen_rare_clues.update(new_clues)

        for left, right in zip(bursts, bursts[1:]):
            if left.get("location_bucket") != right.get("location_bucket"):
                change_points.append(
                    {
                        "change_type": "location_chain_change",
                        "from_burst_id": left["burst_id"],
                        "to_burst_id": right["burst_id"],
                        "details": [left.get("location_bucket"), right.get("location_bucket")],
                    }
                )
            if set(left.get("dominant_person_ids", [])) != set(right.get("dominant_person_ids", [])):
                change_points.append(
                    {
                        "change_type": "dominant_person_change",
                        "from_burst_id": left["burst_id"],
                        "to_burst_id": right["burst_id"],
                        "details": {
                            "from": left.get("dominant_person_ids", []),
                            "to": right.get("dominant_person_ids", []),
                        },
                    }
                )
            if set(left.get("activity_hints", [])) != set(right.get("activity_hints", [])):
                change_points.append(
                    {
                        "change_type": "activity_change",
                        "from_burst_id": left["burst_id"],
                        "to_burst_id": right["burst_id"],
                        "details": {
                            "from": left.get("activity_hints", []),
                            "to": right.get("activity_hints", []),
                        },
                    }
                )
        return change_points

    def _detect_slice_conflicts(self, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        conflicts: List[Dict[str, Any]] = []
        strong_places = [
            candidate["name"]
            for fact in facts
            for candidate in fact.get("place_candidates", [])
            if isinstance(candidate, dict)
            and candidate.get("name")
            and float(candidate.get("confidence") or 0.0) >= 0.7
        ]
        if len(set(strong_places)) > 1:
            conflicts.append(
                {
                    "conflict_type": "multiple_strong_place_candidates",
                    "values": self._unique(strong_places),
                }
            )
        return conflicts

    def _session_continuity(self, left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
        gap_seconds = self._seconds_between(left.get("ended_at"), right.get("started_at"))
        if gap_seconds > LLM_SESSION_HARD_GAP_SECONDS:
            return {
                "same_session": False,
                "score": -99,
                "reason": "hard_split_time_gap",
                "gap_seconds": gap_seconds,
            }

        distance_km = self._distance_km(left.get("facts", [None])[0], right.get("facts", [None])[0])
        if distance_km > LLM_SESSION_HARD_DISTANCE_KM and not self._looks_like_explainable_transition(left, right):
            return {
                "same_session": False,
                "score": -99,
                "reason": "hard_split_distance_jump",
                "gap_seconds": gap_seconds,
                "distance_km": round(distance_km, 3),
            }

        score = 0
        signals: List[str] = []

        if gap_seconds <= LLM_SESSION_STRONG_GAP_SECONDS:
            score += 2
            signals.append("strong_time_continuity")
        elif gap_seconds <= LLM_SESSION_SOFT_GAP_SECONDS:
            score += 1
            signals.append("soft_time_continuity")

        left_bucket = str(left.get("location_bucket") or "")
        right_bucket = str(right.get("location_bucket") or "")
        if left_bucket and right_bucket and left_bucket == right_bucket:
            score += 2
            signals.append("same_location_bucket")
        elif distance_km and distance_km <= LLM_SESSION_NEAR_DISTANCE_KM:
            score += 2
            signals.append("near_distance")
        elif self._looks_like_explainable_transition(left, right):
            score += 1
            signals.append("explainable_transition")

        left_people = set(left.get("dominant_person_ids", []) or [])
        right_people = set(right.get("dominant_person_ids", []) or [])
        shared_people = left_people.intersection(right_people)
        if left_people and right_people and max(len(left_people), len(right_people)) > 0:
            overlap_ratio = len(shared_people) / max(len(left_people), len(right_people))
            if overlap_ratio >= 0.5:
                score += 2
                signals.append("strong_people_continuity")
            elif shared_people:
                score += 1
                signals.append("weak_people_continuity")

        left_scene = set(self._normalized_hint_set(left.get("scene_hints", [])))
        right_scene = set(self._normalized_hint_set(right.get("scene_hints", [])))
        if left_scene.intersection(right_scene):
            score += 1
            signals.append("scene_continuity")

        left_activity = set(self._normalized_hint_set(left.get("activity_hints", [])))
        right_activity = set(self._normalized_hint_set(right.get("activity_hints", [])))
        if left_activity.intersection(right_activity):
            score += 1
            signals.append("activity_continuity")

        if self._rare_clue_conflict(left, right):
            score -= 3
            signals.append("rare_clue_conflict")

        if not shared_people and left_bucket != right_bucket and not left_scene.intersection(right_scene) and not left_activity.intersection(right_activity):
            score -= 3
            signals.append("semantic_rupture")

        if score >= 3:
            decision = "same_session"
            same_session = True
        elif score >= 1:
            decision = "review_needed"
            same_session = True
        else:
            decision = "distinct"
            same_session = False
        return {
            "same_session": same_session,
            "score": score,
            "decision": decision,
            "gap_seconds": gap_seconds,
            "distance_km": round(distance_km, 3),
            "signals": signals,
        }

    def _burst_break_required(self, left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        left_bucket = self._location_bucket(left)
        right_bucket = self._location_bucket(right)
        if left_bucket and right_bucket and left_bucket != right_bucket:
            return True
        left_people = set(left.get("person_ids") or [])
        right_people = set(right.get("person_ids") or [])
        left_activity = str(left.get("activity_hint") or "").strip().lower()
        right_activity = str(right.get("activity_hint") or "").strip().lower()
        if left_people and right_people and not left_people.intersection(right_people) and left_activity and right_activity and left_activity != right_activity:
            return True
        return False

    def _location_bucket(self, fact: Dict[str, Any]) -> str:
        location = fact.get("location") or {}
        if isinstance(location, dict):
            name = str(location.get("name") or "").strip().lower()
            if name:
                return name
        place_candidates = fact.get("place_candidates", []) or []
        for candidate in place_candidates:
            if not isinstance(candidate, dict):
                continue
            name = str(candidate.get("name") or "").strip().lower()
            if name:
                return name
        return str(fact.get("location_name") or "").strip().lower()

    def _seconds_between(self, left: Optional[str], right: Optional[str]) -> float:
        try:
            left_dt = datetime.fromisoformat(str(left))
            right_dt = datetime.fromisoformat(str(right))
        except Exception:
            return 0.0
        return abs((right_dt - left_dt).total_seconds())

    def _distance_km(self, left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> float:
        if not left or not right:
            return 0.0
        left_location = left.get("location") if isinstance(left, dict) else None
        right_location = right.get("location") if isinstance(right, dict) else None
        if not isinstance(left_location, dict) or not isinstance(right_location, dict):
            return 0.0
        if any(key not in left_location for key in ("lat", "lng")) or any(key not in right_location for key in ("lat", "lng")):
            return 0.0
        lat1 = radians(float(left_location["lat"]))
        lon1 = radians(float(left_location["lng"]))
        lat2 = radians(float(right_location["lat"]))
        lon2 = radians(float(right_location["lng"]))
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
        hav = sin(delta_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(delta_lon / 2) ** 2
        return 2 * 6371.0 * asin(sqrt(hav))

    def _dominant_person_ids_from_facts(self, facts: List[Dict[str, Any]]) -> List[str]:
        counts = Counter(
            person_id
            for fact in facts
            for person_id in fact.get("person_ids", []) or []
            if person_id
        )
        return [person_id for person_id, _count in counts.most_common(3)]

    def _normalized_hint_set(self, values: Iterable[str]) -> List[str]:
        normalized: List[str] = []
        for value in values:
            text = str(value or "").strip().lower()
            if text and text not in normalized:
                normalized.append(text)
        return normalized

    def _looks_like_explainable_transition(self, left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        left_bucket = str(left.get("location_bucket") or "")
        right_bucket = str(right.get("location_bucket") or "")
        if left_bucket and right_bucket:
            left_tokens = set(re.split(r"[\s_:/-]+", left_bucket))
            right_tokens = set(re.split(r"[\s_:/-]+", right_bucket))
            if left_tokens.intersection(right_tokens):
                return True
        left_scene = set(self._normalized_hint_set(left.get("scene_hints", [])))
        right_scene = set(self._normalized_hint_set(right.get("scene_hints", [])))
        left_activity = set(self._normalized_hint_set(left.get("activity_hints", [])))
        right_activity = set(self._normalized_hint_set(right.get("activity_hints", [])))
        return bool(left_scene.intersection(right_scene) or left_activity.intersection(right_activity))

    def _rare_clue_conflict(self, left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        left_ocr = {clue.lower() for clue in left.get("rare_clues", []) if clue}
        right_ocr = {clue.lower() for clue in right.get("rare_clues", []) if clue}
        if not left_ocr or not right_ocr:
            return False
        if left_ocr.intersection(right_ocr):
            return False
        left_places = {name.lower() for name in left.get("location_chain", []) if name}
        right_places = {name.lower() for name in right.get("location_chain", []) if name}
        return bool(left_places and right_places and not left_places.intersection(right_places))

    def _create_slice_memory_prompt(self, evidence_packet: Dict[str, Any], primary_person_id: Optional[str]) -> str:
        primary_label = primary_person_id or "authenticated_user"
        return f"""你是 memory materialization LLM。你现在只处理一个 session slice，不要总结整个相册。

规则：
1. 你的目标是把 session-scoped evidence packet 转成可落库的 memory contract。
2. 不能编造城市、人物关系、偏好或长期画像。
3. 你必须优先使用 fact_inventory、rare_clues、change_points、conflicts，而不是只看 summary。
4. 允许输出 uncertainty。
5. “我” 在内部永远对应 authenticated user={primary_label}，但只有在证据足够时才说用户出镜。
6. events 只是 memory 的一部分；细粒度事实必须进入 observations 或 claims。
7. 这个 slice 只是分析窗口，不等于完整现实活动链；不要跨越 packet 边界做过强推断。

请输出严格 JSON，包含以下 6 个顶层字段：
{{
  "events": [
    {{
      "event_id": "字符串",
      "title": "字符串",
      "coarse_event_type": "unknown/social_outing/sightseeing/dining/training/daily_life/travel/work/study/health/shopping/other",
      "event_facets": ["字符串"],
      "alternative_type_candidates": [{{"type": "字符串", "score": 0.0}}],
      "started_at": "ISO时间",
      "ended_at": "ISO时间",
      "location": "字符串",
      "participant_person_ids": ["Person_001"],
      "photo_ids": ["photo_001"],
      "original_image_ids": ["photo_001"],
      "description": "客观描述",
      "narrative_synthesis": "一句话客观归纳",
      "confidence": 0.0,
      "reason": "判断依据"
    }}
  ],
  "observations": [
    {{
      "observation_id": "字符串",
      "category": "ocr|object|brand|dish|price|clothing|style|health|transport|route_plan|scene|activity|place_hint",
      "field_key": "字符串",
      "field_value": "字符串",
      "confidence": 0.0,
      "photo_ids": ["photo_001"],
      "original_image_ids": ["photo_001"],
      "event_id": "可为空",
      "session_id": "{evidence_packet['session_id']}",
      "person_ids": ["Person_001"],
      "evidence_refs": [{{"ref_type": "photo", "ref_id": "photo_001"}}]
    }}
  ],
  "claims": [
    {{
      "claim_id": "字符串",
      "claim_type": "location|identity|brand|dish|price|health|transport|route_plan|preference_signal|culture_signal|object_last_seen",
      "subject": "对象ID或语义主体",
      "predicate": "字符串",
      "object": "字符串",
      "confidence": 0.0,
      "photo_ids": ["photo_001"],
      "original_image_ids": ["photo_001"],
      "event_id": "可为空",
      "session_id": "{evidence_packet['session_id']}",
      "evidence_refs": [{{"ref_type": "photo", "ref_id": "photo_001"}}]
    }}
  ],
  "relationship_hypotheses": [
    {{
      "person_id": "Person_002",
      "relationship_type": "acquaintance|friend|close_friend|colleague|family|partner|co_presence_only",
      "label": "标签",
      "confidence": 0.0,
      "supporting_event_ids": ["event_001"],
      "supporting_photo_ids": ["photo_001"],
      "reason_summary": "原因",
      "reason": "原因",
      "evidence": {{}}
    }}
  ],
  "profile_deltas": [
    {{
      "profile_key": "career_profile|preference_profile|opinion_trajectory_profile|style_profile|identity_trajectory_profile|consumption_profile|aesthetic_profile|recommendation_prior_profile",
      "field_key": "字符串",
      "field_value": "字符串",
      "summary": "增量总结",
      "confidence": 0.0,
      "supporting_event_ids": ["event_001"],
      "supporting_photo_ids": ["photo_001"],
      "evidence_refs": [{{"ref_type": "photo", "ref_id": "photo_001"}}]
    }}
  ],
  "uncertainty": [
    {{
      "field": "字符串",
      "status": "unknown|insufficient_evidence|ambiguous",
      "reason": "为什么"
    }}
  ]
}}

注意：
- 如果只是看到环境中的品牌或物体，不要直接推断用户偏好。
- 如果只有单次暴露，不要直接推断长期兴趣。
- 如果没有稳定跨时段证据，不要输出强关系。
- rare clues 和 OCR / 品牌 / 路线 / 价格 / 地点名 / 物体最后出现线索必须尽量保留。
- `original_image_ids` 必须精确绑定原始图片 ID；没有额外来源时可与 `photo_ids` 相同。

Session-scoped evidence packet:
{json.dumps(evidence_packet, ensure_ascii=False, indent=2)}
"""

    def _create_session_merge_prompt(
        self,
        *,
        raw_session: Dict[str, Any],
        session_slice_records: List[Dict[str, Any]],
        slice_contracts: List[Dict[str, Any]],
        primary_person_id: Optional[str],
    ) -> str:
        primary_label = primary_person_id or "authenticated_user"
        session_summary = {
            "raw_session_id": raw_session["raw_session_id"],
            "started_at": raw_session["started_at"],
            "ended_at": raw_session["ended_at"],
            "location_chain": raw_session["location_chain"],
            "photo_ids": raw_session["photo_ids"],
            "burst_ids": raw_session["burst_ids"],
            "dominant_person_ids": raw_session["dominant_person_ids"],
            "continuity_decisions": raw_session["continuity_decisions"],
        }
        return f"""你是 memory session aggregator。你会收到同一个 raw session 下多个带 overlap 的 slice memory contracts，请先在 session 内部做去重、拼接和保守确认。

规则：
1. 优先比较相邻 slice 的事件与 claims，依据 overlap_burst_ids / photo_ids / 时间连续性 / 人物连续性 / location_chain 做拼接。
2. 不要丢失 rare clues、OCR、brands、place claims、object last-seen claims。
3. 如果证据不足，宁可保守保留多条 event，也不要强行 merge。
4. “我” 对应 authenticated_user={primary_label}，不要把任何 Person_x 直接绑成用户。
5. 输出仍然是完整 6 段 memory contract。

Session summary:
{json.dumps(session_summary, ensure_ascii=False, indent=2)}

Slice packets:
{json.dumps(session_slice_records, ensure_ascii=False, indent=2)}

Slice contracts:
{json.dumps(slice_contracts, ensure_ascii=False, indent=2)}
"""

    def _create_merge_prompt(
        self,
        *,
        photo_fact_count: int,
        raw_sessions: List[Dict[str, Any]],
        session_slices: List[Dict[str, Any]],
        session_contracts: List[Dict[str, Any]],
        session_artifacts: List[Dict[str, Any]],
        primary_person_id: Optional[str],
    ) -> str:
        compact_sessions = [
            {
                "raw_session_id": session["raw_session_id"],
                "started_at": session["started_at"],
                "ended_at": session["ended_at"],
                "location_chain": session["location_chain"],
                "photo_ids": session["photo_ids"],
            }
            for session in raw_sessions
        ]
        compact_slices = [
            {
                "slice_id": session_slice["slice_id"],
                "raw_session_id": session_slice["raw_session_id"],
                "photo_ids": session_slice["photo_ids"],
                "burst_ids": session_slice["burst_ids"],
                "information_score": session_slice.get("information_score"),
                "density_score": session_slice.get("density_score"),
            }
            for session_slice in session_slices
        ]
        primary_label = primary_person_id or "authenticated_user"
        return f"""你是 memory global aggregator。你会收到多个 raw session 的 session-level memory contracts，请做全局去重、合并、谨慎关系修订，并输出最终 memory contract。

规则：
1. 不要丢失高价值 observations / claims。
2. 允许多个 event 合并，但只能在时间/地点/活动证据充分时合并。
3. 关系版本要保守：没有纵向证据就不要输出强关系。
4. “我” 对应 authenticated_user={primary_label}，但不要因为缺少人脸锚点就把任何 Person_x 绑成用户。
5. profile_deltas 只输出增量，不直接下绝对结论。

输出仍然是同样的 6 段 JSON contract，顶层字段必须完整。

输入概览：
- total_photo_facts: {photo_fact_count}
- raw_sessions: {json.dumps(compact_sessions, ensure_ascii=False)}
- session_slices: {json.dumps(compact_slices, ensure_ascii=False)}
- session_artifacts: {json.dumps(session_artifacts, ensure_ascii=False)}
- session_contracts:
{json.dumps(session_contracts, ensure_ascii=False, indent=2)}
"""

    def _empty_contract(self) -> Dict[str, Any]:
        return {
            "events": [],
            "observations": [],
            "claims": [],
            "relationship_hypotheses": [],
            "profile_deltas": [],
            "uncertainty": [],
        }

    def _normalize_memory_contract(self, payload: Any) -> Dict[str, Any]:
        contract = self._empty_contract()
        if isinstance(payload, str):
            payload = self._extract_json_payload(payload)
        if not isinstance(payload, dict):
            return contract
        for key in contract:
            value = payload.get(key, [])
            contract[key] = value if isinstance(value, list) else []

        for index, event in enumerate(contract["events"], start=1):
            if not isinstance(event, dict):
                contract["events"][index - 1] = {}
                event = contract["events"][index - 1]
            event.setdefault("event_id", f"EVT_{index:03d}")
            event.setdefault("title", "")
            event.setdefault("coarse_event_type", "other")
            event.setdefault("event_facets", [])
            event.setdefault("alternative_type_candidates", [])
            event.setdefault("participant_person_ids", [])
            event.setdefault("photo_ids", [])
            event.setdefault("original_image_ids", list(event.get("photo_ids", []) or []))
            event.setdefault("description", "")
            event.setdefault("narrative_synthesis", "")
            event.setdefault("confidence", 0.0)
            event.setdefault("reason", "")
        for key in ("observations", "claims", "relationship_hypotheses", "profile_deltas", "uncertainty"):
            for index, item in enumerate(contract[key], start=1):
                if not isinstance(item, dict):
                    contract[key][index - 1] = {}
        for index, item in enumerate(contract["observations"], start=1):
            if not isinstance(item, dict):
                item = contract["observations"][index - 1]
            item.setdefault("observation_id", f"OBS_{index:03d}")
            item.setdefault("category", "scene")
            item.setdefault("field_key", "")
            item.setdefault("field_value", "")
            item.setdefault("confidence", 0.0)
            item.setdefault("photo_ids", [])
            item.setdefault("original_image_ids", list(item.get("photo_ids", []) or []))
            item.setdefault("event_id", "")
            item.setdefault("session_id", "")
            item.setdefault("person_ids", [])
            item.setdefault("evidence_refs", [])
        for index, item in enumerate(contract["claims"], start=1):
            if not isinstance(item, dict):
                item = contract["claims"][index - 1]
            item.setdefault("claim_id", f"CLM_{index:03d}")
            item.setdefault("claim_type", "other")
            item.setdefault("subject", "")
            item.setdefault("predicate", "")
            item.setdefault("object", "")
            item.setdefault("confidence", 0.0)
            item.setdefault("photo_ids", [])
            item.setdefault("original_image_ids", list(item.get("photo_ids", []) or []))
            item.setdefault("event_id", "")
            item.setdefault("session_id", "")
            item.setdefault("evidence_refs", [])
        for index, item in enumerate(contract["relationship_hypotheses"], start=1):
            if not isinstance(item, dict):
                item = contract["relationship_hypotheses"][index - 1]
            item.setdefault("relationship_id", f"REL_{index:03d}")
            item.setdefault("person_id", item.get("target_person_id", ""))
            item.setdefault("relationship_type", "co_presence_only")
            item.setdefault("label", "")
            item.setdefault("confidence", 0.0)
            item.setdefault("supporting_event_ids", [])
            item.setdefault("supporting_photo_ids", [])
            item.setdefault("reason_summary", "")
            item.setdefault("reason", "")
            item.setdefault("evidence", {})
        for index, item in enumerate(contract["profile_deltas"], start=1):
            if not isinstance(item, dict):
                item = contract["profile_deltas"][index - 1]
            item.setdefault("delta_id", f"DELTA_{index:03d}")
            item.setdefault("profile_key", "general_profile")
            item.setdefault("field_key", "")
            item.setdefault("field_value", "")
            item.setdefault("summary", "")
            item.setdefault("confidence", 0.0)
            item.setdefault("supporting_event_ids", [])
            item.setdefault("supporting_photo_ids", [])
            item.setdefault("evidence_refs", [])
        for index, item in enumerate(contract["uncertainty"], start=1):
            if not isinstance(item, dict):
                item = contract["uncertainty"][index - 1]
            item.setdefault("uncertainty_id", f"UNC_{index:03d}")
            item.setdefault("field", "")
            item.setdefault("status", "unknown")
            item.setdefault("reason", "")
        return contract

    def _contract_counts(self, contract: Dict[str, Any]) -> Dict[str, int]:
        return {key: len(contract.get(key, [])) for key in self._empty_contract()}

    def _legacy_time_fields(self, started_at: str, ended_at: str) -> tuple[str, str]:
        try:
            start_dt = datetime.fromisoformat(started_at)
            end_dt = datetime.fromisoformat(ended_at or started_at)
        except Exception:
            return "", ""
        return start_dt.strftime("%Y-%m-%d"), f"{start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')}"

    def _create_legacy_profile(
        self,
        events: List[Event],
        relationships: List[Relationship],
        primary_person_id: Optional[str],
    ) -> str:
        lines = ["# Profile", ""]
        if primary_person_id:
            lines.append(f"- primary_person_id_hint: {primary_person_id}")
            lines.append("")
        if events:
            lines.append("## recent_events")
            lines.append("")
            for event in events[:10]:
                lines.append(f"- {event.title}: {event.narrative_synthesis or event.description}")
            lines.append("")
        if relationships:
            lines.append("## relationships")
            lines.append("")
            for relationship in relationships[:10]:
                lines.append(f"- {relationship.person_id}: {relationship.label} ({relationship.confidence:.2f})")
        return "\n".join(lines).strip()

    def _unique(self, items: Iterable[Optional[str]]) -> List[str]:
        values: List[str] = []
        for item in items:
            if not item:
                continue
            text = str(item)
            if text not in values:
                values.append(text)
        return values

    def _call_json_prompt(self, prompt: str) -> Dict[str, Any]:
        if self.use_proxy:
            return self._call_llm_via_proxy(prompt)
        if self.use_openrouter:
            return self._call_llm_via_openrouter(prompt)
        if self.use_bedrock:
            return self._call_llm_via_bedrock(prompt)
        return self._call_llm_via_official_api(prompt)

    def _call_markdown_prompt(self, prompt: str) -> str:
        if self.use_proxy:
            return self._call_profile_via_proxy(prompt)
        if self.use_openrouter:
            return self._call_profile_via_openrouter(prompt)
        if self.use_bedrock:
            return self._call_profile_via_bedrock(prompt)
        return self._call_profile_via_official_api(prompt)

    def _call_llm_via_official_api(self, prompt: str) -> dict:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self.genai.types.GenerateContentConfig(response_mime_type="application/json"),
        )
        return self._extract_json_payload(response.text)

    def _call_llm_via_bedrock(self, prompt: str) -> dict:
        candidates = self.bedrock_model_candidates or [self.model]
        last_error: Exception | None = None
        for index, model_id in enumerate(candidates):
            try:
                response = self.bedrock_client.converse(
                    modelId=model_id,
                    messages=build_text_message(prompt),
                    inferenceConfig=build_inference_config(
                        temperature=0.1,
                        max_tokens=BEDROCK_LLM_MAX_OUTPUT_TOKENS,
                        top_p=None,
                    ),
                )
                self.model = model_id
                return self._extract_json_payload(extract_text_from_converse_response(response))
            except Exception as exc:
                last_error = exc
                if index < len(candidates) - 1 and should_try_next_bedrock_model(exc):
                    continue
                raise
        if last_error:
            raise last_error
        raise RuntimeError("未能调用任何 Bedrock LLM 模型")

    def _call_llm_via_proxy(self, prompt: str) -> dict:
        headers = {
            "x-api-key": self.proxy_key,
            "Content-Type": "application/json",
        }
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ]
        }
        url = f"{self.proxy_url}/api/gemini/v1beta/models/{self.proxy_model}:generateContent"
        response = self.requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            response_data = response.json()
            if "candidates" in response_data and response_data["candidates"]:
                candidate = response_data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            return self._extract_json_payload(part["text"])
            return {}
        error_msg = f"代理 API 返回状态码 {response.status_code}"
        if response.text:
            try:
                error_data = response.json()
                error_msg += f": {error_data.get('error', {}).get('message', response.text)}"
            except Exception:
                error_msg += f": {response.text[:200]}"
        raise Exception(error_msg)

    def _openrouter_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.openrouter_site_url,
            "X-Title": self.openrouter_app_name,
        }

    def _extract_openrouter_content(self, response_data: Dict[str, Any]) -> str:
        choices = response_data.get("choices", [])
        if not choices:
            raise ValueError("OpenRouter 未返回 choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not content and message.get("reasoning"):
            content = message["reasoning"]
        return self._coerce_text_content(content)

    def _call_llm_via_openrouter(self, prompt: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8192,
            "temperature": 0.1,
        }
        response = self.requests.post(
            f"{self.openrouter_base_url}/chat/completions",
            json=payload,
            headers=self._openrouter_headers(),
            timeout=60,
        )
        if response.status_code == 200:
            response_data = response.json()
            return self._extract_json_payload(self._extract_openrouter_content(response_data))
        error_msg = f"OpenRouter 返回状态码 {response.status_code}"
        if response.text:
            try:
                error_data = response.json()
                error_msg += f": {error_data.get('error', {}).get('message', response.text)}"
            except Exception:
                error_msg += f": {response.text[:200]}"
        raise Exception(error_msg)

    def _call_profile_via_official_api(self, prompt: str) -> str:
        response = self.client.models.generate_content(model=self.model, contents=prompt)
        return response.text

    def _call_profile_via_bedrock(self, prompt: str) -> str:
        candidates = self.bedrock_model_candidates or [self.model]
        last_error: Exception | None = None
        for index, model_id in enumerate(candidates):
            try:
                response = self.bedrock_client.converse(
                    modelId=model_id,
                    messages=build_text_message(prompt),
                    inferenceConfig=build_inference_config(
                        temperature=0.3,
                        max_tokens=BEDROCK_LLM_MAX_OUTPUT_TOKENS,
                        top_p=None,
                    ),
                )
                self.model = model_id
                return extract_text_from_converse_response(response)
            except Exception as exc:
                last_error = exc
                if index < len(candidates) - 1 and should_try_next_bedrock_model(exc):
                    continue
                raise
        if last_error:
            raise last_error
        raise RuntimeError("未能调用任何 Bedrock profile 模型")

    def _call_profile_via_proxy(self, prompt: str) -> str:
        headers = {
            "x-api-key": self.proxy_key,
            "Content-Type": "application/json",
        }
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ]
        }
        url = f"{self.proxy_url}/api/gemini/v1beta/models/{self.proxy_model}:generateContent"
        response = self.requests.post(url, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            response_data = response.json()
            if "candidates" in response_data and response_data["candidates"]:
                candidate = response_data["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            return part["text"]
            return ""
        error_msg = f"代理 API 返回状态码 {response.status_code}"
        if response.text:
            try:
                error_data = response.json()
                error_msg += f": {error_data.get('error', {}).get('message', response.text)}"
            except Exception:
                error_msg += f": {response.text[:200]}"
        raise Exception(error_msg)

    def _call_profile_via_openrouter(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8192,
            "temperature": 0.4,
        }
        response = self.requests.post(
            f"{self.openrouter_base_url}/chat/completions",
            json=payload,
            headers=self._openrouter_headers(),
            timeout=60,
        )
        if response.status_code == 200:
            return self._extract_openrouter_content(response.json())
        error_msg = f"OpenRouter 返回状态码 {response.status_code}"
        if response.text:
            try:
                error_data = response.json()
                error_msg += f": {error_data.get('error', {}).get('message', response.text)}"
            except Exception:
                error_msg += f": {response.text[:200]}"
        raise Exception(error_msg)
