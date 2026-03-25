"""
LLM processing module: chunk-aware memory materialization.
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from config import (
    API_PROXY_KEY,
    API_PROXY_MODEL,
    API_PROXY_URL,
    BEDROCK_LLM_MAX_OUTPUT_TOKENS,
    BEDROCK_RELATIONSHIP_LLM_FALLBACK_MODEL,
    BEDROCK_RELATIONSHIP_LLM_MODEL,
    BEDROCK_RELATIONSHIP_MAX_OUTPUT_TOKENS,
    BEDROCK_REGION,
    DEFAULT_TASK_VERSION,
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
    OPENROUTER_REASONING_EFFORT,
    OPENROUTER_SITE_URL,
    RELATIONSHIP_FOLLOWS_MAIN_LLM,
    RELATIONSHIP_MAX_CONCURRENCY,
    RELATIONSHIP_MAX_RETRIES,
    RELATIONSHIP_MIN_CO_OCCURRENCE,
    RELATIONSHIP_MIN_DISTINCT_DAYS,
    RELATIONSHIP_MIN_INTIMACY_SCORE,
    RELATIONSHIP_PROVIDER,
    RELATIONSHIP_REQUEST_TIMEOUT_SECONDS,
    RETRY_DELAY,
    TASK_VERSION_V0323,
    TASK_VERSION_V0325,
    TASK_VERSION_V0317_HEAVY,
    V0323_OPENROUTER_MODEL,
    V0325_OPENROUTER_LLM_MODEL,
)
from models import Event, Relationship
from services.bedrock_runtime import (
    build_bedrock_client,
    build_inference_config,
    build_text_message,
    extract_text_from_converse_response,
    should_try_next_bedrock_model,
)


class LLMProcessor:
    """Chunk-aware LLM processor with Bedrock support."""

    @staticmethod
    def _dedupe_bedrock_candidates(candidates: Sequence[str]) -> List[str]:
        return list(dict.fromkeys(str(candidate or "").strip() for candidate in candidates if str(candidate or "").strip()))

    def __init__(self, task_version: str = DEFAULT_TASK_VERSION):
        self.task_version = task_version
        self.use_heavy_pipeline = self.task_version == TASK_VERSION_V0317_HEAVY
        self.provider = "openrouter" if self.task_version in {TASK_VERSION_V0323, TASK_VERSION_V0325} else LLM_PROVIDER
        self.relationship_follows_main_llm = RELATIONSHIP_FOLLOWS_MAIN_LLM
        if self.task_version in {TASK_VERSION_V0323, TASK_VERSION_V0325}:
            self.relationship_provider = self.provider
        else:
            self.relationship_provider = RELATIONSHIP_PROVIDER if not self.relationship_follows_main_llm else self.provider
        self.use_proxy = self.provider == "proxy"
        self.use_openrouter = self.provider == "openrouter"
        self.use_bedrock = self.provider == "bedrock"
        self.relationship_use_proxy = self.relationship_provider == "proxy"
        self.relationship_use_openrouter = self.relationship_provider == "openrouter"
        self.relationship_use_bedrock = self.relationship_provider == "bedrock"
        if self.task_version == TASK_VERSION_V0323:
            self.model = V0323_OPENROUTER_MODEL
        elif self.task_version == TASK_VERSION_V0325:
            self.model = V0325_OPENROUTER_LLM_MODEL
        else:
            self.model = OPENROUTER_LLM_MODEL if self.use_openrouter else LLM_MODEL
        self.relationship_model = self.model
        self.requests = None
        self.genai = None
        self.bedrock_client = None
        self.bedrock_model_candidates: List[str] = []
        self.relationship_bedrock_model_candidates: List[str] = []
        self.last_memory_contract: Dict[str, Any] | None = None
        self.last_chunk_artifacts: Dict[str, Any] = {}
        self._active_json_context: Dict[str, Any] = {}

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
            # Trust explicit configured Bedrock model/profile identifiers directly.
            # This avoids a slow management-catalog lookup during heavy-task resume.
            self.bedrock_model_candidates = self._dedupe_bedrock_candidates(
                [self.model],
            )
            self.relationship_bedrock_model_candidates = self._dedupe_bedrock_candidates(
                [BEDROCK_RELATIONSHIP_LLM_MODEL, BEDROCK_RELATIONSHIP_LLM_FALLBACK_MODEL],
            )
            if self.bedrock_model_candidates:
                self.model = self.bedrock_model_candidates[0]
            print(f"[LLM] 使用 Bedrock: {self.model} @ {BEDROCK_REGION}")
        else:
            from google import genai

            self.genai = genai
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            print("[LLM] 使用官方 Gemini API")

        if self.relationship_use_openrouter and self.requests is None:
            try:
                import requests
            except ModuleNotFoundError:
                requests = None
            self.requests = requests
            self.openrouter_api_key = OPENROUTER_API_KEY or GEMINI_API_KEY
            if not self.openrouter_api_key:
                raise ValueError("relationship 使用 OpenRouter 需要配置 OPENROUTER_API_KEY 或 GEMINI_API_KEY")
            self.openrouter_base_url = OPENROUTER_BASE_URL.rstrip("/")
            self.openrouter_site_url = OPENROUTER_SITE_URL
            self.openrouter_app_name = OPENROUTER_APP_NAME

        if self.relationship_use_openrouter:
            if self.task_version == TASK_VERSION_V0323:
                self.relationship_model = V0323_OPENROUTER_MODEL
            elif self.task_version == TASK_VERSION_V0325:
                self.relationship_model = V0325_OPENROUTER_LLM_MODEL
            else:
                self.relationship_model = OPENROUTER_LLM_MODEL
        elif self.relationship_use_bedrock:
            self.relationship_model = BEDROCK_RELATIONSHIP_LLM_MODEL
        else:
            self.relationship_model = self.model

        if (self.use_bedrock or self.relationship_use_bedrock) and self.bedrock_client is None:
            self.bedrock_client = build_bedrock_client(BEDROCK_REGION)
        if self.use_heavy_pipeline and self.relationship_use_bedrock and not self.relationship_bedrock_model_candidates:
            self.relationship_bedrock_model_candidates = self._dedupe_bedrock_candidates(
                [BEDROCK_RELATIONSHIP_LLM_MODEL, BEDROCK_RELATIONSHIP_LLM_FALLBACK_MODEL],
            )

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

    def _sanitize_json_text(self, raw_text: str) -> str:
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
        return text.strip()

    def _extract_balanced_json_candidate(self, text: str) -> str:
        start = -1
        depth = 0
        in_string = False
        escaped = False
        for index, char in enumerate(text):
            if start == -1:
                if char in "{[":
                    start = index
                    depth = 1
                continue
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "{[":
                depth += 1
            elif char in "}]":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        return ""

    def _try_parse_json_candidate(self, candidate: str) -> tuple[Optional[Dict[str, Any]], Optional[json.JSONDecodeError]]:
        decoder = json.JSONDecoder()
        try:
            payload = json.loads(candidate)
            if isinstance(payload, dict):
                return payload, None
            return {"items": payload}, None
        except json.JSONDecodeError as exc:
            try:
                payload, _end = decoder.raw_decode(candidate)
                if isinstance(payload, dict):
                    return payload, None
                return {"items": payload}, None
            except json.JSONDecodeError:
                return None, exc

    def _insert_missing_commas(self, candidate: str) -> str:
        repaired = candidate
        patterns = (
            r'(?<=[0-9\]}\"])(\s+)(?=("|\{|\[|true|false|null|-?\d))',
            r'(?<=true)(\s+)(?=("|\{|\[|-?\d))',
            r'(?<=false)(\s+)(?=("|\{|\[|-?\d))',
            r'(?<=null)(\s+)(?=("|\{|\[|-?\d))',
        )
        for _ in range(3):
            updated = repaired
            for pattern in patterns:
                updated = re.sub(pattern, r",\1", updated)
            if updated == repaired:
                break
            repaired = updated
        return repaired

    def _normalize_json_string_literals(self, candidate: str) -> str:
        repaired: List[str] = []
        in_string = False
        escaped = False
        length = len(candidate)
        index = 0
        while index < length:
            char = candidate[index]
            if escaped:
                if char in {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}:
                    repaired.append(char)
                else:
                    repaired.append("\\")
                    repaired.append(char)
                escaped = False
                index += 1
                continue
            if char == "\\":
                repaired.append(char)
                escaped = True
                index += 1
                continue
            if char == '"':
                if not in_string:
                    in_string = True
                    repaired.append(char)
                    index += 1
                    continue
                lookahead = index + 1
                while lookahead < length and candidate[lookahead].isspace():
                    lookahead += 1
                next_char = candidate[lookahead] if lookahead < length else ""
                if next_char in {":", ",", "}", "]", ""}:
                    in_string = False
                    repaired.append(char)
                else:
                    repaired.append('\\"')
                index += 1
                continue
            if in_string and char == "\n":
                repaired.append("\\n")
                index += 1
                continue
            if in_string and char == "\r":
                repaired.append("\\r")
                index += 1
                continue
            if in_string and char == "\t":
                repaired.append("\\t")
                index += 1
                continue
            repaired.append(char)
            index += 1
        return "".join(repaired)

    def _escape_internal_quotes(self, candidate: str) -> str:
        return self._normalize_json_string_literals(candidate)

    def _repair_json_candidates(
        self,
        candidate: str,
        error: Optional[json.JSONDecodeError] = None,
    ) -> List[str]:
        repairs: List[str] = []

        def add(item: str) -> None:
            item = str(item or "").strip()
            if item and item not in repairs:
                repairs.append(item)

        translated_quotes = candidate.translate(
            str.maketrans(
                {
                    "“": '"',
                    "”": '"',
                    "‘": "'",
                    "’": "'",
                }
            )
        )
        translated = translated_quotes.translate(
            str.maketrans(
                {
                    "，": ",",
                    "：": ":",
                    "（": "(",
                    "）": ")",
                }
            )
        )
        add(translated_quotes)
        add(self._normalize_json_string_literals(translated_quotes))
        add(re.sub(r",(?=\s*[}\]])", "", translated_quotes))
        add(re.sub(r"\bNone\b", "null", translated_quotes))
        add(re.sub(r"\bTrue\b", "true", translated_quotes))
        add(re.sub(r"\bFalse\b", "false", translated_quotes))
        add(self._insert_missing_commas(translated_quotes))
        add(self._insert_missing_commas(re.sub(r",(?=\s*[}\]])", "", translated_quotes)))
        add(self._insert_missing_commas(self._normalize_json_string_literals(translated_quotes)))
        add(translated)
        add(self._normalize_json_string_literals(translated))
        add(re.sub(r",(?=\s*[}\]])", "", translated))
        add(re.sub(r"\bNone\b", "null", translated))
        add(re.sub(r"\bTrue\b", "true", translated))
        add(re.sub(r"\bFalse\b", "false", translated))
        add(self._insert_missing_commas(translated))
        add(self._insert_missing_commas(re.sub(r",(?=\s*[}\]])", "", translated)))
        add(self._insert_missing_commas(self._normalize_json_string_literals(translated)))

        if error and "Expecting ',' delimiter" in str(error):
            position = int(getattr(error, "pos", -1))
            if 0 <= position < len(candidate):
                add(candidate[:position] + "," + candidate[position:])
                previous_quote = candidate.rfind('"', 0, position)
                if previous_quote != -1 and (previous_quote == 0 or candidate[previous_quote - 1] != "\\"):
                    add(candidate[:previous_quote] + '\\"' + candidate[previous_quote + 1 :])
                    next_quote = candidate.find('"', position)
                    if next_quote != -1 and (next_quote == 0 or candidate[next_quote - 1] != "\\"):
                        add(
                            candidate[:previous_quote]
                            + '\\"'
                            + candidate[previous_quote + 1 : next_quote]
                            + '\\"'
                            + candidate[next_quote + 1 :]
                        )
        return repairs

    def _record_parse_failure(
        self,
        *,
        raw_text: str,
        error: Exception,
        repaired_attempts: List[str],
    ) -> None:
        if not hasattr(self, "last_chunk_artifacts") or not isinstance(self.last_chunk_artifacts, dict):
            self.last_chunk_artifacts = {}
        failures = self.last_chunk_artifacts.setdefault("parse_failures", [])
        failures.append(
            {
                **dict(getattr(self, "_active_json_context", {}) or {}),
                "error": str(error),
                "raw_text_preview": str(raw_text or "")[:4000],
                "raw_text_tail": str(raw_text or "")[-4000:],
                "repair_attempt_preview": [item[:1000] for item in repaired_attempts[:6]],
                "failed_at": datetime.now().isoformat(),
            }
        )

    def _extract_json_payload(self, raw_text: str) -> Dict[str, Any]:
        text = self._sanitize_json_text(raw_text)
        candidates: List[str] = []

        def add_candidate(item: str) -> None:
            item = str(item or "").strip()
            if item and item not in candidates:
                candidates.append(item)

        add_candidate(text)
        first_json_start = min((idx for idx in (text.find("{"), text.find("[")) if idx != -1), default=-1)
        if first_json_start != -1:
            add_candidate(text[first_json_start:])
        balanced_candidate = self._extract_balanced_json_candidate(text)
        if balanced_candidate:
            add_candidate(balanced_candidate)

        last_error: Optional[Exception] = None
        repair_attempts: List[str] = []
        for candidate in candidates:
            payload, error = self._try_parse_json_candidate(candidate)
            if payload is not None:
                return payload
            if error is not None:
                last_error = error
            for repaired in self._repair_json_candidates(candidate, error):
                if repaired not in repair_attempts:
                    repair_attempts.append(repaired)
                payload, repaired_error = self._try_parse_json_candidate(repaired)
                if payload is not None:
                    return payload
                if repaired_error is not None:
                    last_error = repaired_error

        if last_error:
            self._record_parse_failure(
                raw_text=text,
                error=last_error,
                repaired_attempts=repair_attempts,
            )
            raise last_error
        raise ValueError("无法解析 JSON payload")

    def _is_retryable_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        retry_keywords = [
            "429",
            "500",
            "502",
            "503",
            "504",
            "520",
            "522",
            "524",
            "rate limit",
            "connection",
            "timeout",
            "temporarily unavailable",
            "reset by peer",
            "throttl",
            "too many requests",
            "bad gateway",
            "gateway timeout",
            "response ended prematurely",
            "chunkedencodingerror",
            "incomplete read",
            "server disconnected",
            "remote end closed connection",
            "connection aborted",
        ]
        return any(keyword in message for keyword in retry_keywords)

    def _is_json_parse_error(self, exc: Exception) -> bool:
        if isinstance(exc, json.JSONDecodeError):
            return True
        message = str(exc).lower()
        json_error_markers = (
            "unterminated string",
            "expecting value",
            "expecting ',' delimiter",
            "extra data",
            "invalid control character",
            "unterminated string starting at",
            "无法解析 json payload",
        )
        return any(marker in message for marker in json_error_markers)

    def _record_salvage_recovery(
        self,
        *,
        stage: str,
        reason: str,
        payload_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not hasattr(self, "last_chunk_artifacts") or not isinstance(self.last_chunk_artifacts, dict):
            self.last_chunk_artifacts = {}
        recoveries = self.last_chunk_artifacts.setdefault("salvage_recoveries", [])
        recoveries.append(
            {
                **dict(getattr(self, "_active_json_context", {}) or {}),
                "stage": stage,
                "reason": reason,
                "payload_summary": payload_summary or {},
                "recovered_at": datetime.now().isoformat(),
            }
        )

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

    def _call_json_prompt_with_compact_retry(
        self,
        *,
        primary_prompt: str,
        compact_prompt_builder: Optional[Callable[[], str]] = None,
        salvage_builder: Optional[Callable[[], Dict[str, Any]]] = None,
        salvage_stage: str,
    ) -> Dict[str, Any]:
        original_context = dict(getattr(self, "_active_json_context", {}) or {})
        try:
            return self._call_with_retries(lambda prompt=primary_prompt: self._call_json_prompt(prompt))
        except Exception as exc:
            if not self._is_json_parse_error(exc):
                raise

            if compact_prompt_builder is not None:
                compact_context = {**original_context, "recovery_mode": "compact_retry", "initial_error": str(exc)}
                self._active_json_context = compact_context
                try:
                    compact_prompt = compact_prompt_builder()
                    return self._call_with_retries(lambda prompt=compact_prompt: self._call_json_prompt(prompt))
                except Exception as compact_exc:
                    if not self._is_json_parse_error(compact_exc):
                        raise
                    if salvage_builder is None:
                        raise
                    salvage_contract = self._finalize_memory_contract(salvage_builder())
                    self._record_salvage_recovery(
                        stage=salvage_stage,
                        reason=f"compact retry failed after parse error: {compact_exc}",
                        payload_summary=self._contract_counts(salvage_contract),
                    )
                    return salvage_contract
                finally:
                    self._active_json_context = original_context

            if salvage_builder is None:
                raise

            salvage_contract = self._finalize_memory_contract(salvage_builder())
            self._record_salvage_recovery(
                stage=salvage_stage,
                reason=f"primary parse failed without compact retry: {exc}",
                payload_summary=self._contract_counts(salvage_contract),
            )
            return salvage_contract
        finally:
            self._active_json_context = original_context

    def extract_memory_contract(
        self,
        vlm_results: List[Dict[str, Any]],
        face_db: Optional[Dict[str, Any]] = None,
        primary_person_id: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        started_at = time.perf_counter()
        if not vlm_results:
            contract = self._empty_contract()
            self.last_memory_contract = contract
            self.last_chunk_artifacts = {
                "photo_fact_count": 0,
                "raw_event_count": 0,
                "slice_count": 0,
                "slices": [],
                "event_summaries": [],
                "llm_runtime_seconds": 0.0,
            }
            return contract

        photo_facts = self._build_photo_fact_buffer(vlm_results)
        bursts = self._build_bursts(photo_facts)
        raw_sessions = self._build_raw_sessions(bursts)
        session_slices = self._build_session_slices(raw_sessions)
        self.last_chunk_artifacts = {
            "task_version": self.task_version,
            "photo_fact_count": len(photo_facts),
            "burst_count": len(bursts),
            "raw_event_count": len(raw_sessions),
            "slice_count": len(session_slices),
            "slices": [],
            "slice_contract_records": [],
            "event_summaries": [],
            "event_merge_records": [],
            "pre_relationship_contract": None,
            "parse_failures": [],
        }
        self._emit_progress(
            progress_callback,
            {
                "message": "LLM 改写中",
                "substage": "slice_contract",
                "processed_slices": 0,
                "slice_count": len(session_slices),
                "processed_events": 0,
                "event_count": len(raw_sessions),
                "percent": 5,
                "provider": self._active_llm_provider(),
                "model": self._active_llm_model(),
                "last_success_at": None,
                "retry_count": 0,
                "runtime_seconds": round(time.perf_counter() - started_at, 4),
            },
        )

        slice_contracts = []
        slice_artifacts = []
        slice_contract_records: List[Dict[str, Any]] = []
        slice_contracts_by_session: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for session_slice in session_slices:
            self._active_json_context = {
                "stage": "slice_contract",
                "slice_id": session_slice["slice_id"],
                "raw_event_id": session_slice["raw_session_id"],
                "photo_ids": list(session_slice["photo_ids"]),
            }
            prompt = self._create_slice_memory_prompt(session_slice["evidence_packet"], primary_person_id)
            response = self._call_json_prompt_with_compact_retry(
                primary_prompt=prompt,
                compact_prompt_builder=(
                    lambda packet=session_slice["evidence_packet"], pid=primary_person_id: self._create_compact_slice_memory_prompt(packet, pid)
                ),
                salvage_builder=lambda packet=session_slice["evidence_packet"]: self._salvage_slice_contract_from_evidence_packet(packet),
                salvage_stage="slice_contract",
            )
            normalized = self._finalize_memory_contract(response)
            slice_contracts.append(normalized)
            slice_contracts_by_session[session_slice["raw_session_id"]].append(normalized)
            slice_contract_records.append(
                {
                    "slice_id": session_slice["slice_id"],
                    "raw_event_id": session_slice["raw_session_id"],
                    "photo_ids": list(session_slice["photo_ids"]),
                    "contract": deepcopy(normalized),
                }
            )
            slice_artifacts.append(
                {
                    "slice_id": session_slice["slice_id"],
                    "raw_event_id": session_slice["raw_session_id"],
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
            self.last_chunk_artifacts["slices"] = list(slice_artifacts)
            self.last_chunk_artifacts["slice_contract_records"] = list(slice_contract_records)
            self._emit_progress(
                progress_callback,
                {
                    "message": "LLM 改写中",
                    "substage": "slice_contract",
                    "processed_slices": len(slice_artifacts),
                    "slice_count": len(session_slices),
                    "processed_events": 0,
                    "event_count": len(raw_sessions),
                    "percent": self._progress_percent(len(slice_artifacts), max(len(session_slices), 1), start=5, end=65),
                    "provider": self._active_llm_provider(),
                    "model": self._active_llm_model(),
                    "last_success_at": self._iso_now(),
                    "retry_count": len(self.last_chunk_artifacts.get("parse_failures", []) or []),
                    "runtime_seconds": round(time.perf_counter() - started_at, 4),
                },
            )

        session_contracts = []
        session_artifacts = []
        event_merge_records: List[Dict[str, Any]] = []
        for raw_session in raw_sessions:
            self._active_json_context = {
                "stage": "session_merge",
                "raw_event_id": raw_session["raw_session_id"],
                "photo_ids": list(raw_session["photo_ids"]),
            }
            session_slice_records = [
                artifact
                for artifact in slice_artifacts
                if artifact.get("raw_event_id") == raw_session["raw_session_id"]
            ]
            session_prompt = self._create_session_merge_prompt(
                raw_session=raw_session,
                session_slice_records=session_slice_records,
                slice_contracts=slice_contracts_by_session.get(raw_session["raw_session_id"], []),
                primary_person_id=primary_person_id,
            )
            session_contract = self._call_json_prompt_with_compact_retry(
                primary_prompt=session_prompt,
                compact_prompt_builder=None,
                salvage_builder=(
                    lambda contracts=slice_contracts_by_session.get(raw_session["raw_session_id"], []), raw_event_id=raw_session["raw_session_id"]: self._salvage_session_contract_from_slices(
                        raw_event_id=raw_event_id,
                        slice_contracts=contracts,
                    )
                ),
                salvage_stage="session_merge",
            )
            normalized_session_contract = self._finalize_memory_contract(session_contract)
            session_contracts.append(normalized_session_contract)
            event_merge_records.append(
                {
                    "raw_event_id": raw_session["raw_session_id"],
                    "photo_ids": list(raw_session["photo_ids"]),
                    "contract": deepcopy(normalized_session_contract),
                }
            )
            session_artifacts.append(
                {
                    "raw_event_id": raw_session["raw_session_id"],
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
            self.last_chunk_artifacts["event_summaries"] = list(session_artifacts)
            self.last_chunk_artifacts["event_merge_records"] = list(event_merge_records)
            self._emit_progress(
                progress_callback,
                {
                    "message": "LLM 汇总事件窗口",
                    "substage": "event_merge",
                    "processed_slices": len(slice_artifacts),
                    "slice_count": len(session_slices),
                    "processed_events": len(session_artifacts),
                    "event_count": len(raw_sessions),
                    "percent": self._progress_percent(len(session_artifacts), max(len(raw_sessions), 1), start=65, end=92),
                    "provider": self._active_llm_provider(),
                    "model": self._active_llm_model(),
                    "last_success_at": self._iso_now(),
                    "retry_count": len(self.last_chunk_artifacts.get("parse_failures", []) or []),
                    "runtime_seconds": round(time.perf_counter() - started_at, 4),
                },
            )

        merge_prompt = self._create_merge_prompt(
            photo_fact_count=len(photo_facts),
            raw_sessions=raw_sessions,
            session_slices=session_slices,
            session_contracts=session_contracts,
            session_artifacts=session_artifacts,
            primary_person_id=primary_person_id,
        )
        self._emit_progress(
            progress_callback,
            {
                "message": "LLM 生成最终 memory contract",
                "substage": "global_merge",
                "processed_slices": len(slice_artifacts),
                "slice_count": len(session_slices),
                "processed_events": len(session_artifacts),
                "event_count": len(raw_sessions),
                "percent": 96,
                "provider": self._active_llm_provider(),
                "model": self._active_llm_model(),
                "last_success_at": None,
                "retry_count": len(self.last_chunk_artifacts.get("parse_failures", []) or []),
                "runtime_seconds": round(time.perf_counter() - started_at, 4),
            },
        )
        self._active_json_context = {
            "stage": "global_merge",
            "photo_fact_count": len(photo_facts),
            "raw_event_count": len(raw_sessions),
            "slice_count": len(session_slices),
        }
        merged_contract = self._call_json_prompt_with_compact_retry(
            primary_prompt=merge_prompt,
            compact_prompt_builder=None,
            salvage_builder=lambda contracts=session_contracts: self._salvage_global_contract_from_sessions(contracts),
            salvage_stage="global_merge",
        )
        contract = self._finalize_memory_contract(merged_contract)
        recovered_contract = self._recover_contract_if_sparse(
            merged_contract=contract,
            session_contracts=session_contracts,
            session_artifacts=session_artifacts,
        )
        merge_recovered = recovered_contract is not contract
        contract = recovered_contract
        self.last_memory_contract = deepcopy(contract)
        self.last_chunk_artifacts["pre_relationship_contract"] = deepcopy(contract)
        if self.use_heavy_pipeline:
            self._emit_progress(
                progress_callback,
                {
                    "message": "LLM 关系推断中",
                    "substage": "relationship_inference",
                    "processed_slices": len(slice_artifacts),
                    "slice_count": len(session_slices),
                    "processed_events": len(session_artifacts),
                    "event_count": len(raw_sessions),
                    "percent": 98,
                    "provider": self._active_relationship_provider(),
                    "model": self._active_relationship_model(),
                    "candidate_count": 0,
                    "filtered_count": 0,
                    "processed_candidates": 0,
                    "last_success_at": None,
                    "retry_count": 0,
                    "runtime_seconds": round(time.perf_counter() - started_at, 4),
                },
            )
            contract["relationship_hypotheses"] = self._run_heavy_relationship_pass(
                contract=contract,
                photo_facts=photo_facts,
                primary_person_id=primary_person_id,
                progress_callback=progress_callback,
                llm_started_at=started_at,
                slice_count=len(session_slices),
                event_count=len(raw_sessions),
                processed_events=len(session_artifacts),
                processed_slices=len(slice_artifacts),
            )

        self._active_json_context = {}

        self.last_memory_contract = contract
        self.last_chunk_artifacts = {
            "photo_fact_count": len(photo_facts),
            "burst_count": len(bursts),
            "raw_event_count": len(raw_sessions),
            "event_contract_count": len(session_contracts),
            "slice_count": len(session_slices),
            "task_version": self.task_version,
            "merge_recovered_from_sessions": merge_recovered,
            "relationship_provider": self._active_relationship_provider(),
            "relationship_model_candidates": list(self.relationship_bedrock_model_candidates or []),
            "slice_contract_records": slice_contract_records,
            "event_merge_records": event_merge_records,
            "pre_relationship_contract": deepcopy(self.last_chunk_artifacts.get("pre_relationship_contract")),
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
            "event_summaries": session_artifacts,
            "slices": slice_artifacts,
            "final_contract_counts": self._contract_counts(contract),
            "llm_runtime_seconds": round(time.perf_counter() - started_at, 4),
        }
        return contract

    def facts_from_memory_contract(self, memory_contract: Dict[str, Any]) -> List[Event]:
        events: List[Event] = []
        for index, item in enumerate(memory_contract.get("facts", []), start=1):
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
                    event_id=str(item.get("fact_id") or f"FACT_{index:03d}"),
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
                        "supporting_event_ids": [str(value) for value in item.get("supporting_fact_ids", []) or []],
                        "supporting_fact_ids": [str(value) for value in item.get("supporting_fact_ids", []) or []],
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

    def _finalize_memory_contract(self, payload: Any) -> Dict[str, Any]:
        contract = self._normalize_memory_contract(payload)
        self._ensure_profile_deltas(contract)
        return contract

    def _ensure_profile_deltas(self, contract: Dict[str, Any]) -> None:
        profile_deltas = contract.setdefault("profile_deltas", [])
        existing_keys = {
            (
                str(item.get("profile_key") or ""),
                str(item.get("field_key") or ""),
                str(item.get("field_value") or ""),
            )
            for item in profile_deltas
            if isinstance(item, dict)
        }
        synthesized: List[Dict[str, Any]] = []

        def append_delta(
            *,
            profile_key: str,
            field_key: str,
            field_value: str,
            summary: str,
            confidence: float,
            supporting_fact_ids: Optional[List[str]] = None,
            supporting_photo_ids: Optional[List[str]] = None,
            evidence_refs: Optional[List[Dict[str, str]]] = None,
        ) -> None:
            normalized_key = (profile_key, field_key, field_value)
            if not field_value or normalized_key in existing_keys:
                return
            existing_keys.add(normalized_key)
            synthesized.append(
                {
                    "profile_key": profile_key,
                    "field_key": field_key,
                    "field_value": field_value,
                    "summary": summary or field_value,
                    "confidence": round(max(0.0, min(1.0, confidence)), 4),
                    "supporting_fact_ids": list(supporting_fact_ids or []),
                    "supporting_photo_ids": list(supporting_photo_ids or []),
                    "evidence_refs": list(evidence_refs or []),
                }
            )

        for fact in contract.get("facts", [])[:16]:
            fact_id = str(fact.get("fact_id") or "")
            photo_ids = [str(item) for item in fact.get("original_image_ids", fact.get("photo_ids", [])) or [] if item]
            summary = str(fact.get("narrative_synthesis") or fact.get("description") or fact.get("title") or "").strip()
            title = str(fact.get("title") or "").strip()
            if title:
                append_delta(
                    profile_key="recommendation_prior_profile",
                    field_key="recent_activity_signal",
                    field_value=title,
                    summary=summary or title,
                    confidence=float(fact.get("confidence") or 0.0) * 0.75,
                    supporting_fact_ids=[fact_id] if fact_id else [],
                    supporting_photo_ids=photo_ids,
                    evidence_refs=[{"ref_type": "fact", "ref_id": fact_id}] if fact_id else [],
                )
            location = str(fact.get("location") or "").strip()
            if location:
                append_delta(
                    profile_key="identity_trajectory_profile",
                    field_key="place_signal",
                    field_value=location,
                    summary=f"近期活动地点线索: {location}",
                    confidence=max(0.35, float(fact.get("confidence") or 0.0) * 0.68),
                    supporting_fact_ids=[fact_id] if fact_id else [],
                    supporting_photo_ids=photo_ids,
                    evidence_refs=[{"ref_type": "fact", "ref_id": fact_id}] if fact_id else [],
                )

        observation_profile_map = {
            "scene": ("aesthetic_profile", "scene_signal"),
            "activity": ("recommendation_prior_profile", "activity_signal"),
            "place_hint": ("identity_trajectory_profile", "place_signal"),
            "brand": ("preference_profile", "observed_brand_signal"),
            "dish": ("preference_profile", "observed_dish_signal"),
            "price": ("consumption_profile", "price_signal"),
            "clothing": ("style_profile", "style_signal"),
            "style": ("style_profile", "style_signal"),
            "health": ("identity_trajectory_profile", "health_signal"),
            "transport": ("identity_trajectory_profile", "transport_signal"),
            "route_plan": ("recommendation_prior_profile", "route_signal"),
            "object": ("recommendation_prior_profile", "object_signal"),
        }
        for observation in contract.get("observations", [])[:24]:
            category = str(observation.get("category") or "scene").strip()
            profile_key, field_key = observation_profile_map.get(category, ("recommendation_prior_profile", "observation_signal"))
            field_value = str(observation.get("field_value") or "").strip()
            if not field_value:
                continue
            append_delta(
                profile_key=profile_key,
                field_key=field_key,
                field_value=field_value,
                summary=f"{observation.get('field_key') or category}: {field_value}",
                confidence=max(0.28, float(observation.get("confidence") or 0.0) * 0.72),
                supporting_fact_ids=[str(observation.get("fact_id") or "")] if observation.get("fact_id") else [],
                supporting_photo_ids=[str(item) for item in observation.get("original_image_ids", observation.get("photo_ids", [])) or [] if item],
                evidence_refs=observation.get("evidence_refs", []) or [],
            )

        claim_profile_map = {
            "location": ("identity_trajectory_profile", "location_claim"),
            "identity": ("career_profile", "identity_claim"),
            "brand": ("preference_profile", "brand_claim"),
            "dish": ("preference_profile", "dish_claim"),
            "price": ("consumption_profile", "price_claim"),
            "health": ("identity_trajectory_profile", "health_claim"),
            "transport": ("identity_trajectory_profile", "transport_claim"),
            "route_plan": ("recommendation_prior_profile", "route_plan_claim"),
            "preference_signal": ("preference_profile", "preference_signal"),
            "culture_signal": ("aesthetic_profile", "culture_signal"),
            "object_last_seen": ("recommendation_prior_profile", "object_last_seen"),
        }
        for claim in contract.get("claims", [])[:24]:
            claim_type = str(claim.get("claim_type") or "other").strip()
            profile_key, field_key = claim_profile_map.get(claim_type, ("recommendation_prior_profile", "claim_signal"))
            claim_object = str(claim.get("object") or "").strip()
            if not claim_object:
                continue
            predicate = str(claim.get("predicate") or "").strip()
            summary = f"{predicate}: {claim_object}" if predicate else claim_object
            append_delta(
                profile_key=profile_key,
                field_key=field_key,
                field_value=claim_object,
                summary=summary,
                confidence=max(0.32, float(claim.get("confidence") or 0.0) * 0.8),
                supporting_fact_ids=[str(claim.get("fact_id") or "")] if claim.get("fact_id") else [],
                supporting_photo_ids=[str(item) for item in claim.get("original_image_ids", claim.get("photo_ids", [])) or [] if item],
                evidence_refs=claim.get("evidence_refs", []) or [],
            )

        if not synthesized:
            return

        remaining_capacity = max(0, 12 - len(profile_deltas))
        if remaining_capacity <= 0:
            return

        for index, item in enumerate(synthesized[:remaining_capacity], start=1):
            item.setdefault("delta_id", f"SYNTH_DELTA_{index:03d}")
            profile_deltas.append(item)

    def _recover_contract_if_sparse(
        self,
        *,
        merged_contract: Dict[str, Any],
        session_contracts: List[Dict[str, Any]],
        session_artifacts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        core_keys = ("facts", "observations", "claims", "relationship_hypotheses", "profile_deltas")
        if any(merged_contract.get(key) for key in core_keys):
            return merged_contract
        if not any(contract.get(key) for contract in session_contracts for key in core_keys):
            return merged_contract

        recovered = self._merge_contracts_union(session_contracts)
        recovered.setdefault("uncertainty", []).append(
            {
                "field": "global_merge_contract",
                "status": "insufficient_evidence",
                "reason": "global merge returned an empty core contract; recovered from session-level contracts",
                "session_artifact_refs": [
                    str(item.get("raw_event_id") or "")
                    for item in session_artifacts
                    if item.get("raw_event_id")
                ],
            }
        )
        self._ensure_profile_deltas(recovered)
        return recovered

    def _merge_contracts_union(self, contracts: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged = self._empty_contract()
        dedupe_maps: Dict[str, Dict[str, Dict[str, Any]]] = {
            "facts": {},
            "observations": {},
            "claims": {},
            "relationship_hypotheses": {},
            "profile_deltas": {},
            "uncertainty": {},
        }

        identity_fields = {
            "facts": "fact_id",
            "observations": "observation_id",
            "claims": "claim_id",
            "relationship_hypotheses": "relationship_id",
            "profile_deltas": "delta_id",
            "uncertainty": "uncertainty_id",
        }

        for contract in contracts:
            normalized = self._finalize_memory_contract(contract)
            for key, id_field in identity_fields.items():
                bucket = dedupe_maps[key]
                for item in normalized.get(key, []):
                    if not isinstance(item, dict):
                        continue
                    identifier = str(item.get(id_field) or "").strip()
                    if not identifier:
                        identifier = self._fallback_identity_for_contract_item(key, item)
                    if identifier in bucket:
                        bucket[identifier] = self._merge_contract_item(bucket[identifier], item)
                    else:
                        bucket[identifier] = dict(item)

        for key in merged:
            merged[key] = list(dedupe_maps[key].values())
        return self._finalize_memory_contract(merged)

    def _fallback_identity_for_contract_item(self, bucket: str, item: Dict[str, Any]) -> str:
        if bucket == "facts":
            return "|".join(
                [
                    str(item.get("title") or ""),
                    str(item.get("started_at") or ""),
                    str(item.get("ended_at") or ""),
                    str(item.get("location") or ""),
                ]
            )
        if bucket == "observations":
            return "|".join(
                [
                    str(item.get("category") or ""),
                    str(item.get("field_key") or ""),
                    str(item.get("field_value") or ""),
                    ",".join(str(value) for value in item.get("original_image_ids", []) or []),
                ]
            )
        if bucket == "claims":
            return "|".join(
                [
                    str(item.get("claim_type") or ""),
                    str(item.get("predicate") or ""),
                    str(item.get("object") or ""),
                    ",".join(str(value) for value in item.get("original_image_ids", []) or []),
                ]
            )
        if bucket == "relationship_hypotheses":
            return "|".join(
                [
                    str(item.get("person_id") or item.get("target_person_id") or ""),
                    str(item.get("relationship_type") or ""),
                ]
            )
        if bucket == "profile_deltas":
            return "|".join(
                [
                    str(item.get("profile_key") or ""),
                    str(item.get("field_key") or ""),
                    str(item.get("field_value") or ""),
                ]
            )
        return "|".join(
            [
                str(item.get("field") or ""),
                str(item.get("status") or ""),
                str(item.get("reason") or ""),
            ]
        )

    def _merge_contract_item(self, left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(left)
        for key, value in right.items():
            if value in (None, "", [], {}):
                continue
            current = merged.get(key)
            if isinstance(current, list) and isinstance(value, list):
                merged[key] = self._merge_unique_list(current, value)
            elif isinstance(current, dict) and isinstance(value, dict):
                next_value = dict(current)
                for sub_key, sub_value in value.items():
                    if sub_value not in (None, "", [], {}):
                        next_value[sub_key] = sub_value
                merged[key] = next_value
            elif key == "confidence":
                merged[key] = max(float(current or 0.0), float(value or 0.0))
            elif current in (None, "", [], {}):
                merged[key] = value
        return merged

    def _merge_unique_list(self, left: List[Any], right: List[Any]) -> List[Any]:
        values: List[Any] = list(left)
        seen = {json.dumps(item, ensure_ascii=False, sort_keys=True, default=str) for item in left}
        for item in right:
            marker = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
            if marker not in seen:
                seen.add(marker)
                values.append(item)
        return values

    def extract_events(self, vlm_results: List[Dict], primary_person_id: Optional[str] = None) -> List[Event]:
        contract = self.last_memory_contract
        if contract is None:
            contract = self.extract_memory_contract(vlm_results, primary_person_id=primary_person_id)
        return self.facts_from_memory_contract(contract)

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
        if self.use_heavy_pipeline and self.last_memory_contract is not None:
            return self._generate_heavy_profile(events, relationships, primary_person_id)
        if self.last_memory_contract is not None:
            return self.profile_markdown_from_memory_contract(self.last_memory_contract, primary_person_id)
        return self._create_legacy_profile(events, relationships, primary_person_id)

    def _run_heavy_relationship_pass(
        self,
        *,
        contract: Dict[str, Any],
        photo_facts: List[Dict[str, Any]],
        primary_person_id: Optional[str],
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        llm_started_at: Optional[float] = None,
        slice_count: int = 0,
        event_count: int = 0,
        processed_events: int = 0,
        processed_slices: int = 0,
    ) -> List[Dict[str, Any]]:
        people = sorted(
            {
                str(person_id)
                for fact in photo_facts
                for person_id in fact.get("person_ids", []) or []
                if person_id and person_id != primary_person_id
            }
        )
        if not people:
            return []

        candidate_records: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []
        for person_index, person_id in enumerate(people):
            evidence = self._build_relationship_evidence(
                person_id=person_id,
                contract=contract,
                photo_facts=photo_facts,
                primary_person_id=primary_person_id,
            )
            requires_llm = self._relationship_candidate_requires_llm(evidence)
            candidate_records.append(
                {
                    "index": person_index,
                    "person_id": person_id,
                    "evidence": evidence,
                    "requires_llm": requires_llm,
                }
            )
            if not requires_llm:
                results.append(
                    self._co_presence_only_relationship(
                        person_id=person_id,
                        evidence=evidence,
                        reason="weak evidence below relationship LLM thresholds; downgraded to co_presence_only",
                    )
                )

        llm_candidates = [item for item in candidate_records if item["requires_llm"]]
        self.last_chunk_artifacts["relationship_candidates"] = [
            {
                "person_id": item["person_id"],
                "requires_llm": item["requires_llm"],
                "evidence": deepcopy(item["evidence"]),
            }
            for item in candidate_records
        ]

        relationship_runtime_start = llm_started_at if llm_started_at is not None else time.perf_counter()
        if llm_candidates:
            self._preflight_relationship_provider()

        processed_candidates = 0
        relationship_retry_count = 0
        if llm_candidates:
            with ThreadPoolExecutor(max_workers=min(RELATIONSHIP_MAX_CONCURRENCY, len(llm_candidates))) as executor:
                future_map = {
                    executor.submit(
                        self._infer_single_relationship_candidate,
                        person_id=str(item["person_id"]),
                        evidence=deepcopy(item["evidence"]),
                    ): item
                    for item in llm_candidates
                }
                llm_results_by_person: Dict[str, Dict[str, Any]] = {}
                for future in as_completed(future_map):
                    item = future_map[future]
                    person_id = str(item["person_id"])
                    normalized, retry_count = future.result()
                    relationship_retry_count += int(retry_count or 0)
                    llm_results_by_person[person_id] = normalized
                    processed_candidates += 1
                    self._emit_progress(
                        progress_callback,
                        {
                            "message": "LLM 关系推断中",
                            "substage": "relationship_inference",
                            "processed_slices": processed_slices,
                            "slice_count": slice_count,
                            "processed_events": processed_events,
                            "event_count": event_count,
                            "candidate_count": len(candidate_records),
                            "filtered_count": len(llm_candidates),
                            "processed_candidates": processed_candidates,
                            "current_person_id": person_id,
                            "provider": self._active_relationship_provider(),
                            "model": self._active_relationship_model(),
                            "retry_count": relationship_retry_count,
                            "last_success_at": self._iso_now(),
                            "percent": self._progress_percent(processed_candidates, max(len(llm_candidates), 1), start=97, end=99),
                            "runtime_seconds": round(time.perf_counter() - relationship_runtime_start, 4),
                        },
                    )
                for item in llm_candidates:
                    person_id = str(item["person_id"])
                    normalized = llm_results_by_person.get(person_id)
                    if normalized is not None:
                        results.append(normalized)

        results.sort(
            key=lambda item: (
                -float(item.get("confidence") or 0.0),
                -len(item.get("supporting_fact_ids", []) or []),
                str(item.get("person_id") or ""),
            )
        )
        self.last_chunk_artifacts["relationship_stage"] = {
            "provider": self._active_relationship_provider(),
            "model": self._active_relationship_model(),
            "candidate_count": len(candidate_records),
            "filtered_count": len(llm_candidates),
            "processed_candidates": processed_candidates,
            "retry_count": relationship_retry_count,
            "result_count": len(results),
            "completed_at": self._iso_now(),
        }
        return results

    def _relationship_candidate_requires_llm(self, evidence: Dict[str, Any]) -> bool:
        if int(evidence.get("distinct_days") or 0) >= RELATIONSHIP_MIN_DISTINCT_DAYS:
            return True
        if int(evidence.get("co_occurrence_count") or 0) >= RELATIONSHIP_MIN_CO_OCCURRENCE:
            return True
        if bool(evidence.get("contact_types")):
            return True
        if bool(evidence.get("interaction")):
            return True
        if int(evidence.get("exclusive_one_on_one") or 0) > 0:
            return True
        return float(evidence.get("intimacy_score") or 0.0) >= RELATIONSHIP_MIN_INTIMACY_SCORE

    def _co_presence_only_relationship(
        self,
        *,
        person_id: str,
        evidence: Dict[str, Any],
        reason: str,
    ) -> Dict[str, Any]:
        supporting_fact_ids = [
            str(item.get("fact_id") or "")
            for item in evidence.get("shared_facts", [])
            if item.get("fact_id")
        ]
        supporting_photo_ids = self._unique(
            photo_id
            for item in evidence.get("shared_facts", [])
            for photo_id in item.get("original_image_ids", []) or []
        )
        confidence = round(
            min(
                0.45,
                0.12
                + (min(6, int(evidence.get("co_occurrence_count") or 0)) * 0.035)
                + (min(3, int(evidence.get("distinct_days") or 0)) * 0.03),
            ),
            4,
        )
        return {
            "relationship_id": f"REL_{person_id}",
            "person_id": person_id,
            "relationship_type": "co_presence_only",
            "label": "co_presence_only",
            "confidence": confidence,
            "supporting_fact_ids": supporting_fact_ids,
            "supporting_photo_ids": supporting_photo_ids,
            "reason_summary": reason,
            "reason": reason,
            "evidence": {
                "status": "filtered_without_llm",
                "co_occurrence_count": evidence.get("co_occurrence_count", 0),
                "distinct_days": evidence.get("distinct_days", 0),
                "monthly_average": evidence.get("monthly_average", 0),
                "intimacy_score": evidence.get("intimacy_score", 0.0),
                "scenes": evidence.get("scenes", []),
                "contact_types": evidence.get("contact_types", []),
                "interaction": evidence.get("interaction", []),
                "shared_facts": evidence.get("shared_facts", []),
                "supporting_fact_ids": supporting_fact_ids,
                "supporting_photo_ids": supporting_photo_ids,
            },
        }

    def _preflight_relationship_provider(self) -> None:
        if self.relationship_use_openrouter:
            if not self.requests or not getattr(self, "openrouter_api_key", ""):
                raise RuntimeError("relationship provider openrouter unavailable")
            return
        if self.relationship_use_bedrock:
            if not self.bedrock_client or not (self.relationship_bedrock_model_candidates or [BEDROCK_RELATIONSHIP_LLM_MODEL]):
                raise RuntimeError("relationship provider bedrock unavailable")
            return
        if self.relationship_use_proxy:
            if not self.requests:
                raise RuntimeError("relationship provider proxy unavailable")

    def _infer_single_relationship_candidate(
        self,
        *,
        person_id: str,
        evidence: Dict[str, Any],
    ) -> tuple[Dict[str, Any], int]:
        prompt = self._create_relationship_prompt(person_id=person_id, evidence=evidence)
        retry_count = 0
        last_error: Exception | None = None
        for attempt in range(RELATIONSHIP_MAX_RETRIES):
            try:
                payload = self._call_relationship_prompt(prompt)
                normalized = self._normalize_relationship_result(person_id=person_id, payload=payload, evidence=evidence)
                return normalized, retry_count
            except Exception as exc:
                last_error = exc
                if attempt == RELATIONSHIP_MAX_RETRIES - 1 or not self._is_retryable_error(exc):
                    break
                retry_count += 1
                time.sleep(RETRY_DELAY * (attempt + 1))
        if last_error is not None and self._is_json_parse_error(last_error):
            payload = self._call_json_prompt(prompt)
            normalized = self._normalize_relationship_result(person_id=person_id, payload=payload, evidence=evidence)
            return normalized, retry_count
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"relationship inference failed for {person_id}")

    def _build_relationship_evidence(
        self,
        *,
        person_id: str,
        contract: Dict[str, Any],
        photo_facts: List[Dict[str, Any]],
        primary_person_id: Optional[str],
    ) -> Dict[str, Any]:
        relevant_photo_facts = [
            fact
            for fact in photo_facts
            if person_id in (fact.get("person_ids", []) or [])
        ]
        relevant_facts = [
            fact
            for fact in contract.get("facts", [])
            if person_id in (fact.get("participant_person_ids", []) or [])
        ]
        timestamps = [str(fact.get("timestamp") or "") for fact in relevant_photo_facts if fact.get("timestamp")]
        days = sorted({timestamp.split("T", 1)[0] for timestamp in timestamps if "T" in timestamp})
        span_days = 0
        if len(days) >= 2:
            try:
                span_days = abs((datetime.fromisoformat(days[-1]) - datetime.fromisoformat(days[0])).days)
            except Exception:
                span_days = 0
        monthly_avg = round((len(relevant_photo_facts) / max(1.0, (span_days + 1) / 30.0)), 2) if relevant_photo_facts else 0.0
        scenes = self._unique(
            value
            for fact in relevant_facts
            for value in fact.get("event_facets", []) or []
        )
        contact_types = self._unique(
            str(person.get("contact_type") or "")
            for fact in relevant_photo_facts
            for person in fact.get("people", []) or []
            if isinstance(person, dict) and str(person.get("person_id") or "") == person_id and str(person.get("contact_type") or "")
        )
        interactions = self._unique(
            str(item.get("interaction_type") or "")
            for fact in relevant_facts
            for item in fact.get("social_dynamics", []) or []
            if isinstance(item, dict) and str(item.get("target_id") or "") == person_id
        )
        co_appearing_people = Counter(
            other_person_id
            for fact in relevant_photo_facts
            for other_person_id in fact.get("person_ids", []) or []
            if other_person_id and other_person_id != person_id
        )
        exclusive_one_on_one = sum(
            1
            for fact in relevant_facts
            if person_id in (fact.get("participant_person_ids", []) or [])
            and len({pid for pid in fact.get("participant_person_ids", []) or [] if pid}) <= (2 if primary_person_id else 1)
        )
        intimacy_score = 0.0
        intimacy_weights = {
            "kiss": 1.0,
            "hug": 0.9,
            "holding_hands": 0.9,
            "arm_in_arm": 0.7,
            "selfie_together": 0.6,
            "shoulder_lean": 0.65,
            "sitting_close": 0.55,
            "standing_near": 0.35,
            "no_contact": 0.1,
        }
        if contact_types:
            intimacy_score = round(
                sum(intimacy_weights.get(contact_type, 0.2) for contact_type in contact_types) / len(contact_types),
                3,
            )
        shared_facts = [
            {
                "fact_id": str(fact.get("fact_id") or ""),
                "title": str(fact.get("title") or ""),
                "narrative_synthesis": str(fact.get("narrative_synthesis") or fact.get("description") or ""),
                "social_dynamics": [
                    item
                    for item in fact.get("social_dynamics", []) or []
                    if isinstance(item, dict) and str(item.get("target_id") or "") == person_id
                ],
                "original_image_ids": list(fact.get("original_image_ids", []) or []),
            }
            for fact in relevant_facts[:10]
        ]
        return {
            "co_occurrence_count": len(relevant_photo_facts),
            "distinct_days": len(days),
            "monthly_average": monthly_avg,
            "span_days": span_days,
            "scenes": scenes[:12],
            "contact_types": contact_types,
            "interaction": interactions[:12],
            "exclusive_one_on_one": exclusive_one_on_one,
            "co_appearing_third_parties": [
                {"person_id": pid, "count": count}
                for pid, count in co_appearing_people.most_common(5)
            ],
            "intimacy_score": intimacy_score,
            "trend": "stable" if len(days) <= 1 else "longitudinal_observed",
            "shared_facts": shared_facts,
        }

    def _create_relationship_prompt(self, *, person_id: str, evidence: Dict[str, Any]) -> str:
        return f"""You are a gossip-savvy social analyst who reads photo albums like a reality TV producer reads footage. Your job: figure out how {person_id} fits into the authenticated user's life.

# Evidence
- Co-occurrence: {evidence.get("co_occurrence_count", 0)} photos over {evidence.get("distinct_days", 0)} days (monthly avg: {evidence.get("monthly_average", 0)}/mo)
- Scenes: {json.dumps(evidence.get("scenes", []), ensure_ascii=False)}
- Contact types: {json.dumps(evidence.get("contact_types", []), ensure_ascii=False)}
- Interaction: {json.dumps(evidence.get("interaction", []), ensure_ascii=False)}
- Exclusive 1-on-1: {evidence.get("exclusive_one_on_one", 0)}
- Co-appearing third parties: {json.dumps(evidence.get("co_appearing_third_parties", []), ensure_ascii=False)}
- Intimacy score: {evidence.get("intimacy_score", 0.0)}
- Trend: {evidence.get("trend", "stable")}
- Shared facts: {json.dumps(evidence.get("shared_facts", []), ensure_ascii=False)}

# Relationship Types (pick exactly one)
- family
- romantic
- bestie
- close_friend
- friend
- classmate_colleague
- activity_buddy
- acquaintance

# Relationship Status (pick exactly one)
- new
- growing
- stable
- fading
- gone

Output JSON:
{{
  "relationship_type": "friend",
  "status": "stable",
  "confidence": 0.0,
  "reason": "why"
}}
"""

    def _call_relationship_prompt(self, prompt: str) -> Dict[str, Any]:
        if self.relationship_use_openrouter:
            return self._call_llm_via_openrouter(
                prompt,
                model=self.relationship_model,
                max_tokens=BEDROCK_RELATIONSHIP_MAX_OUTPUT_TOKENS,
                timeout=(15, RELATIONSHIP_REQUEST_TIMEOUT_SECONDS),
            )
        if self.relationship_use_proxy:
            return self._call_llm_via_proxy(prompt)
        if not self.relationship_use_bedrock:
            return self._call_json_prompt(prompt)
        if not self.bedrock_client:
            raise RuntimeError("Bedrock relationship client unavailable")
        candidates = self.relationship_bedrock_model_candidates or [BEDROCK_RELATIONSHIP_LLM_MODEL, BEDROCK_RELATIONSHIP_LLM_FALLBACK_MODEL]
        last_error: Exception | None = None
        for index, model_id in enumerate(candidates):
            try:
                response = self.bedrock_client.converse(
                    modelId=model_id,
                    messages=build_text_message(prompt),
                    inferenceConfig=build_inference_config(
                        temperature=0.1,
                        max_tokens=BEDROCK_RELATIONSHIP_MAX_OUTPUT_TOKENS,
                        top_p=None,
                    ),
                )
                self.relationship_model = model_id
                return self._extract_json_payload(extract_text_from_converse_response(response))
            except Exception as exc:
                last_error = exc
                if index < len(candidates) - 1 and should_try_next_bedrock_model(exc):
                    continue
                raise
        if last_error:
            raise last_error
        raise RuntimeError("未能调用任何 relationship LLM 模型")

    def _normalize_relationship_result(
        self,
        *,
        person_id: str,
        payload: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        if isinstance(payload, str):
            payload = self._extract_json_payload(payload)
        payload = payload if isinstance(payload, dict) else {}
        supporting_fact_ids = [
            str(item.get("fact_id") or "")
            for item in evidence.get("shared_facts", [])
            if item.get("fact_id")
        ]
        supporting_photo_ids = self._unique(
            photo_id
            for item in evidence.get("shared_facts", [])
            for photo_id in item.get("original_image_ids", []) or []
        )
        relationship_type = str(payload.get("relationship_type") or "acquaintance").strip() or "acquaintance"
        status = str(payload.get("status") or "stable").strip() or "stable"
        confidence = round(max(0.0, min(1.0, float(payload.get("confidence") or 0.0))), 4)
        reason = str(payload.get("reason") or "").strip()
        return {
            "relationship_id": f"REL_{person_id}",
            "person_id": person_id,
            "relationship_type": relationship_type,
            "label": f"{relationship_type}:{status}",
            "confidence": confidence,
            "supporting_fact_ids": supporting_fact_ids,
            "supporting_photo_ids": supporting_photo_ids,
            "reason_summary": reason,
            "reason": reason,
            "evidence": {
                "status": status,
                "co_occurrence_count": evidence.get("co_occurrence_count", 0),
                "distinct_days": evidence.get("distinct_days", 0),
                "monthly_average": evidence.get("monthly_average", 0),
                "intimacy_score": evidence.get("intimacy_score", 0.0),
                "scenes": evidence.get("scenes", []),
                "contact_types": evidence.get("contact_types", []),
                "interaction": evidence.get("interaction", []),
                "shared_facts": evidence.get("shared_facts", []),
                "supporting_fact_ids": supporting_fact_ids,
                "supporting_photo_ids": supporting_photo_ids,
            },
        }

    def _generate_heavy_profile(
        self,
        events: List[Event],
        relationships: List[Relationship],
        primary_person_id: Optional[str],
    ) -> str:
        prompt = self._create_profile_prompt(events=events, relationships=relationships, primary_person_id=primary_person_id)
        try:
            return self._call_with_retries(lambda: self._call_markdown_prompt(prompt))
        except Exception:
            return self.profile_markdown_from_memory_contract(self.last_memory_contract or self._empty_contract(), primary_person_id)

    def _create_profile_prompt(
        self,
        *,
        events: List[Event],
        relationships: List[Relationship],
        primary_person_id: Optional[str],
    ) -> str:
        event_payload = [
            {
                "fact_id": event.event_id,
                "title": event.title,
                "timestamp": event.meta_info.get("timestamp") if isinstance(event.meta_info, dict) else "",
                "location_context": event.meta_info.get("location_context") if isinstance(event.meta_info, dict) else "",
                "narrative_synthesis": event.narrative_synthesis,
                "social_dynamics": list(event.social_dynamics or []),
                "persona_evidence": dict(event.persona_evidence or {}),
                "original_image_ids": list(event.evidence_photos or []),
            }
            for event in events
        ]
        relationship_payload = [
            {
                "person_id": relationship.person_id,
                "relationship_type": relationship.relationship_type,
                "label": relationship.label,
                "confidence": relationship.confidence,
                "evidence": dict(relationship.evidence or {}),
                "reason": relationship.reason,
            }
            for relationship in relationships
        ]
        primary_label = primary_person_id or "authenticated_user"
        return f"""你是一位世界级的行为分析专家和 FBI 级别的人格画像师，擅长通过行为残迹还原人类灵魂。

请根据 facts 和 relationships 生成《用户全维画像分析报告》。

约束：
1. 必须引用证据，不要写无来源结论。
2. 如果信息模糊，使用“高概率/疑似/待进一步观察”。
3. “我” 对应 authenticated_user={primary_label}。

Facts:
{json.dumps(event_payload, ensure_ascii=False, indent=2)}

Relationships:
{json.dumps(relationship_payload, ensure_ascii=False, indent=2)}
"""

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
                    "people": list(analysis.get("people", []) or []),
                    "relations": list(analysis.get("relations", []) or []),
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
            "event_id": raw_session["raw_session_id"],
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
                    "people": fact.get("people", []),
                    "relations": fact.get("relations", []),
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
        if self.use_heavy_pipeline:
            return self._create_heavy_slice_memory_prompt(evidence_packet, primary_person_id)
        primary_label = primary_person_id or "authenticated_user"
        compact_packet = self._compact_evidence_packet_for_prompt(evidence_packet)
        return f"""你是 memory materialization LLM。你现在只处理一个 event slice，不要总结整个相册。

规则：
1. 你的目标是把 event-scoped evidence packet 转成可落库的 memory contract。
2. 不能编造城市、人物关系、偏好或长期画像。
3. 你必须优先使用 fact_inventory、rare_clues、change_points、conflicts，而不是只看 summary。
4. 允许输出 uncertainty。
5. “我” 在内部永远对应 authenticated user={primary_label}，但只有在证据足够时才说用户出镜。
6. facts 只是 memory 的一部分；细粒度事实必须进入 observations 或 claims。
7. 这个 slice 只是分析窗口，不等于完整现实活动链；不要跨越 packet 边界做过强推断。
8. 只要 evidence packet 与主用户有关，就尽量输出 profile_deltas；不要把画像层完全留空。
9. 所有字符串内部如果需要出现双引号，必须转义为 \\\"，输出必须能被 JSON 解析器直接解析。

请输出严格 JSON，包含以下 6 个顶层字段：
{{
  "facts": [
    {{
      "fact_id": "字符串",
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
      "fact_id": "可为空",
      "event_id": "{evidence_packet['event_id']}",
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
      "fact_id": "可为空",
      "event_id": "{evidence_packet['event_id']}",
      "evidence_refs": [{{"ref_type": "photo", "ref_id": "photo_001"}}]
    }}
  ],
  "relationship_hypotheses": [
    {{
      "person_id": "Person_002",
      "relationship_type": "acquaintance|friend|close_friend|colleague|family|partner|co_presence_only",
      "label": "标签",
      "confidence": 0.0,
      "supporting_fact_ids": ["fact_001"],
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
      "supporting_fact_ids": ["fact_001"],
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

Event-scoped evidence packet:
{json.dumps(compact_packet, ensure_ascii=False, indent=2)}
"""

    def _create_heavy_slice_memory_prompt(self, evidence_packet: Dict[str, Any], primary_person_id: Optional[str]) -> str:
        primary_label = primary_person_id or "authenticated_user"
        compact_packet = self._compact_evidence_packet_for_prompt(evidence_packet)
        return f"""你是一位资深的人类学专家与社会学行为分析师。现在只分析一个切分后的事件窗口，不要跨越这个窗口做推断。

目标：
1. 基于照片级 VLM 结果，提取这个窗口内的原子事实（facts）。
2. 最大限度保留细粒度 observations 与 claims，后续检索只依赖 memory。
3. 只做事件级社会线索总结；真正的人际关系最终判定会由下一层关系 LLM 完成。
4. 允许 uncertainty，禁止把弱线索写成硬结论。

特别规则：
- “我” 永远代表 authenticated_user={primary_label}，但除非证据强，不要把任何 Person_x 直接当作用户本人。
- OCR、品牌、价格、菜品、地点候选、路线线索、服饰材质、物品最后出现线索不能丢。
- 如果只看到商店、海报、广告牌、环境品牌，不要直接推断长期偏好。
- 如果这个窗口有多张逻辑连贯的照片，可以合并为 1-n 个 facts；重点是“事实完整”而不是“事件数尽量少”。
- 所有长文本字段（description / narrative_synthesis / reason / social_clue / summary）中禁止直接出现 ASCII 双引号 `"`；引用 OCR 或原文时，请改用《》或「」。
- OCR 原文、品牌原文、价格原文优先放进 observations / claims，不要在长文本字段里重复堆叠。

请严格输出 JSON，且必须包含以下 6 个顶层字段：
{{
  "facts": [
    {{
      "fact_id": "FACT_xxx",
      "title": "事件标题",
      "coarse_event_type": "social_outing|sightseeing|dining|travel|shopping|daily_life|work|study|health|training|other|unknown",
      "event_facets": ["细粒度标签"],
      "alternative_type_candidates": [{{"type": "travel", "score": 0.7}}],
      "started_at": "ISO时间",
      "ended_at": "ISO时间",
      "location": "地点描述或候选",
      "participant_person_ids": ["Person_001"],
      "photo_ids": ["photo_001"],
      "original_image_ids": ["photo_001"],
      "description": "客观事实描述",
      "narrative_synthesis": "一句话综合描述",
      "confidence": 0.0,
      "reason": "为什么这样判断",
      "social_dynamics": [
        {{
          "target_id": "Person_002",
          "interaction_type": "共同观景/同桌用餐/被拍摄/自拍同框",
          "social_clue": "来自视角、距离、动作、relations 的证据",
          "relation_hypothesis": "仅作弱猜测，如 friend_candidate",
          "confidence": 0.0
        }}
      ],
      "persona_evidence": {{
        "behavioral": ["行为特征"],
        "aesthetic": ["审美特征"],
        "socioeconomic": ["消费/阶层线索"]
      }}
    }}
  ],
  "observations": [],
  "claims": [],
  "relationship_hypotheses": [],
  "profile_deltas": [],
  "uncertainty": []
}}

窗口 evidence packet:
{json.dumps(compact_packet, ensure_ascii=False, indent=2)}
"""

    def _compact_contract_limits(self, evidence_packet: Dict[str, Any]) -> Dict[str, int]:
        metrics = evidence_packet.get("slice_budget_metrics", {}) if isinstance(evidence_packet, dict) else {}
        photo_count = int(metrics.get("photo_count") or len(evidence_packet.get("photo_refs", []) or []))
        rare_clue_count = int(metrics.get("rare_clue_count") or len(evidence_packet.get("rare_clues", []) or []))
        information_score = float(metrics.get("information_score") or 0.0)
        density_score = float(metrics.get("density_score") or 0.0)

        fact_cap = 1
        if photo_count >= 6 or information_score >= 8 or rare_clue_count >= 5:
            fact_cap = 2
        if photo_count >= 12 or information_score >= 16 or rare_clue_count >= 10:
            fact_cap = 3

        observation_cap = 6
        if rare_clue_count >= 5 or information_score >= 8:
            observation_cap = 8
        if rare_clue_count >= 10 or information_score >= 16 or density_score >= 8:
            observation_cap = 10

        claim_cap = 4
        if rare_clue_count >= 5 or information_score >= 10:
            claim_cap = 5
        if rare_clue_count >= 10 or information_score >= 18:
            claim_cap = 6

        return {
            "fact_cap": fact_cap,
            "observation_cap": observation_cap,
            "claim_cap": claim_cap,
            "relationship_cap": 2 if photo_count >= 8 else 1,
            "profile_cap": 6 if information_score >= 10 else 4,
            "uncertainty_cap": 4,
        }

    def _create_compact_slice_memory_prompt(self, evidence_packet: Dict[str, Any], primary_person_id: Optional[str]) -> str:
        primary_label = primary_person_id or "authenticated_user"
        limits = self._compact_contract_limits(evidence_packet)
        compact_packet = self._compact_evidence_packet_for_prompt(evidence_packet)
        return f"""你是 memory materialization LLM 的紧凑恢复模式。上一次该窗口输出过长或 JSON 损坏，这一次必须输出更短、更稳、更易解析的 contract。

恢复原则：
1. 仍然只分析当前窗口，不要跨窗口推断。
2. 优先保留高价值 OCR、品牌、地点候选、价格、路线/计划、物体最后出现线索。
3. 所有长文本字段必须压缩成短句或短语；单字段尽量不超过 60 个汉字。
4. 所有长文本字段禁止直接出现 ASCII 双引号 `"`；引用原文时用《》或「」。
5. “我” 永远代表 authenticated_user={primary_label}，不要把 Person_x 强行认成用户。
6. 如果没有足够证据，宁可放进 uncertainty，也不要补全。
7. `original_image_ids` 必须精确绑定原始图片 ID。

输出上限：
- facts <= {limits['fact_cap']}
- observations <= {limits['observation_cap']}
- claims <= {limits['claim_cap']}
- relationship_hypotheses <= {limits['relationship_cap']}
- profile_deltas <= {limits['profile_cap']}
- uncertainty <= {limits['uncertainty_cap']}

输出格式仍必须是完整 6 段 JSON contract：
{{
  "facts": [],
  "observations": [],
  "claims": [],
  "relationship_hypotheses": [],
  "profile_deltas": [],
  "uncertainty": []
}}

字段约束：
- facts 只保留最重要的原子事实，title/description/narrative_synthesis 要短。
- observations 只保留最可检索的高价值项。
- claims 只保留最可能被 query 直接命中的项。
- relationship_hypotheses 只保留弱共现，不做强关系定型。
- profile_deltas 只保留有明确证据引用的增量。

窗口 evidence packet:
{json.dumps(compact_packet, ensure_ascii=False, indent=2)}
"""

    def _salvage_slice_contract_from_evidence_packet(self, evidence_packet: Dict[str, Any]) -> Dict[str, Any]:
        photo_ids = [str(item) for item in evidence_packet.get("photo_refs", []) or [] if item]
        location_chain = [str(item) for item in evidence_packet.get("location_chain", []) or [] if item]
        dominant_person_ids = [str(item) for item in evidence_packet.get("dominant_person_ids", []) or [] if item]
        fact_inventory = [item for item in evidence_packet.get("fact_inventory", []) or [] if isinstance(item, dict)]
        time_range = evidence_packet.get("time_range", {}) if isinstance(evidence_packet.get("time_range"), dict) else {}
        rare_clues = [str(item) for item in evidence_packet.get("rare_clues", []) or [] if item]

        activity_values = [
            str(item.get("value") or "").strip()
            for item in fact_inventory
            if str(item.get("type") or "").strip() == "activity_hint" and str(item.get("value") or "").strip()
        ]
        scene_values = [
            str(item.get("value") or "").strip()
            for item in fact_inventory
            if str(item.get("type") or "").strip() == "scene_hint" and str(item.get("value") or "").strip()
        ]
        top_activity = activity_values[0] if activity_values else ""
        top_scene = scene_values[0] if scene_values else ""
        location_label = location_chain[0] if location_chain else ""

        coarse_event_type = "other"
        activity_text = " ".join(activity_values).lower()
        if any(keyword in activity_text for keyword in ("餐", "吃", "meal", "dining", "coffee", "cafe", "restaurant")):
            coarse_event_type = "dining"
        elif any(keyword in activity_text for keyword in ("travel", "trip", "旅行", "观光", "sight", "景")):
            coarse_event_type = "travel"
        elif any(keyword in activity_text for keyword in ("walk", "stroll", "逛", "散步", "view")):
            coarse_event_type = "social_outing"
        elif any(keyword in activity_text for keyword in ("work", "办公", "会议")):
            coarse_event_type = "work"

        title_parts = [part for part in (top_activity, top_scene, location_label) if part]
        title = " / ".join(title_parts[:2]) or "恢复的窗口事实"
        description = "由切片证据直接恢复的窗口事实，原始 LLM 输出因 JSON 截断未能完整解析。"

        observations: List[Dict[str, Any]] = []
        for index, clue in enumerate(rare_clues[:6], start=1):
            category = "ocr" if any(char.isdigit() for char in clue) or any(token in clue.lower() for token in ("http", "www", "tel", "t.", "街", "路", "店", "馆")) else "scene"
            observations.append(
                {
                    "observation_id": f"OBS_SALVAGE_{index:03d}",
                    "category": category,
                    "field_key": "rare_clue",
                    "field_value": clue,
                    "confidence": 0.34,
                    "photo_ids": list(photo_ids),
                    "original_image_ids": list(photo_ids),
                    "fact_id": "FACT_SALVAGE_001",
                    "event_id": str(evidence_packet.get("event_id") or ""),
                    "person_ids": list(dominant_person_ids),
                    "evidence_refs": [{"ref_type": "photo", "ref_id": photo_id} for photo_id in photo_ids[:4]],
                }
            )

        claims: List[Dict[str, Any]] = []
        if location_label:
            claims.append(
                {
                    "claim_id": "CLM_SALVAGE_001",
                    "claim_type": "location",
                    "subject": str(evidence_packet.get("event_id") or "slice"),
                    "predicate": "location_candidate",
                    "object": location_label,
                    "confidence": 0.41,
                    "photo_ids": list(photo_ids),
                    "original_image_ids": list(photo_ids),
                    "fact_id": "FACT_SALVAGE_001",
                    "event_id": str(evidence_packet.get("event_id") or ""),
                    "evidence_refs": [{"ref_type": "photo", "ref_id": photo_id} for photo_id in photo_ids[:4]],
                }
            )

        return {
            "facts": [
                {
                    "fact_id": "FACT_SALVAGE_001",
                    "title": title,
                    "coarse_event_type": coarse_event_type,
                    "event_facets": [item for item in (top_activity, top_scene) if item][:4],
                    "alternative_type_candidates": [],
                    "started_at": str(time_range.get("start") or ""),
                    "ended_at": str(time_range.get("end") or time_range.get("start") or ""),
                    "location": location_label,
                    "participant_person_ids": list(dominant_person_ids),
                    "photo_ids": list(photo_ids),
                    "original_image_ids": list(photo_ids),
                    "description": description,
                    "narrative_synthesis": description,
                    "confidence": 0.22,
                    "reason": "slice JSON parse failed; recovered from evidence packet",
                    "social_dynamics": [],
                    "persona_evidence": {},
                }
            ],
            "observations": observations,
            "claims": claims,
            "relationship_hypotheses": [],
            "profile_deltas": [],
            "uncertainty": [
                {
                    "field": "slice_contract",
                    "status": "insufficient_evidence",
                    "reason": "slice output was truncated and recovered from evidence packet",
                }
            ],
        }

    def _salvage_session_contract_from_slices(
        self,
        *,
        raw_event_id: str,
        slice_contracts: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        merged = self._merge_contracts_union(list(slice_contracts))
        merged.setdefault("uncertainty", []).append(
            {
                "field": "session_merge_contract",
                "status": "insufficient_evidence",
                "reason": f"session merge for {raw_event_id} failed to parse and was recovered from slice unions",
            }
        )
        return merged

    def _salvage_global_contract_from_sessions(
        self,
        session_contracts: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        merged = self._merge_contracts_union(list(session_contracts))
        merged.setdefault("uncertainty", []).append(
            {
                "field": "global_merge_contract",
                "status": "insufficient_evidence",
                "reason": "global merge failed to parse and was recovered from session unions",
            }
        )
        return merged

    def _compact_text_for_prompt(self, value: Any, *, limit: int = 180) -> str:
        text = str(value or "").strip()
        if len(text) <= limit:
            return text
        return f"{text[:limit].rstrip()}..."

    def _compact_slice_record_for_prompt(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "slice_id": record.get("slice_id"),
            "raw_event_id": record.get("raw_event_id"),
            "photo_ids": list(record.get("photo_ids", [])[:12]),
            "burst_ids": list(record.get("burst_ids", [])[:8]),
            "overlap_burst_ids": list(record.get("overlap_burst_ids", [])[:4]),
            "rare_clue_count": int(record.get("rare_clue_count") or 0),
            "photo_count": int(record.get("photo_count") or 0),
            "burst_count": int(record.get("burst_count") or 0),
            "fact_inventory_count": int(record.get("fact_inventory_count") or 0),
            "change_point_count": int(record.get("change_point_count") or 0),
            "location_chain": list(record.get("location_chain", [])[:6]),
            "dominant_person_ids": list(record.get("dominant_person_ids", [])[:6]),
            "conflict_count": int(record.get("conflict_count") or 0),
            "information_score": float(record.get("information_score") or 0.0),
            "density_score": float(record.get("density_score") or 0.0),
            "contract_counts": dict(record.get("contract_counts", {}) or {}),
        }

    def _compact_contract_for_prompt(self, contract: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "facts": [
                {
                    "fact_id": item.get("fact_id"),
                    "title": item.get("title"),
                    "coarse_event_type": item.get("coarse_event_type"),
                    "event_facets": list(item.get("event_facets", [])[:8]),
                    "started_at": item.get("started_at"),
                    "ended_at": item.get("ended_at"),
                    "location": item.get("location"),
                    "participant_person_ids": list(item.get("participant_person_ids", [])[:8]),
                    "original_image_ids": list(item.get("original_image_ids", item.get("photo_ids", []))[:8]),
                    "description": self._compact_text_for_prompt(item.get("description"), limit=160),
                    "narrative_synthesis": self._compact_text_for_prompt(item.get("narrative_synthesis"), limit=160),
                    "confidence": item.get("confidence"),
                }
                for item in contract.get("facts", [])[:8]
                if isinstance(item, dict)
            ],
            "observations": [
                {
                    "observation_id": item.get("observation_id"),
                    "category": item.get("category"),
                    "field_key": item.get("field_key"),
                    "field_value": self._compact_text_for_prompt(item.get("field_value"), limit=120),
                    "fact_id": item.get("fact_id"),
                    "original_image_ids": list(item.get("original_image_ids", item.get("photo_ids", []))[:6]),
                    "confidence": item.get("confidence"),
                }
                for item in contract.get("observations", [])[:12]
                if isinstance(item, dict)
            ],
            "claims": [
                {
                    "claim_id": item.get("claim_id"),
                    "claim_type": item.get("claim_type"),
                    "predicate": item.get("predicate"),
                    "object": self._compact_text_for_prompt(item.get("object"), limit=120),
                    "fact_id": item.get("fact_id"),
                    "original_image_ids": list(item.get("original_image_ids", item.get("photo_ids", []))[:6]),
                    "confidence": item.get("confidence"),
                }
                for item in contract.get("claims", [])[:12]
                if isinstance(item, dict)
            ],
            "relationship_hypotheses": [
                {
                    "person_id": item.get("person_id"),
                    "relationship_type": item.get("relationship_type"),
                    "confidence": item.get("confidence"),
                    "supporting_fact_ids": list(item.get("supporting_fact_ids", [])[:6]),
                }
                for item in contract.get("relationship_hypotheses", [])[:6]
                if isinstance(item, dict)
            ],
            "profile_deltas": [
                {
                    "profile_key": item.get("profile_key"),
                    "field_key": item.get("field_key"),
                    "field_value": self._compact_text_for_prompt(item.get("field_value"), limit=100),
                    "confidence": item.get("confidence"),
                }
                for item in contract.get("profile_deltas", [])[:8]
                if isinstance(item, dict)
            ],
            "uncertainty": [
                {
                    "field": item.get("field"),
                    "status": item.get("status"),
                    "reason": self._compact_text_for_prompt(item.get("reason"), limit=120),
                }
                for item in contract.get("uncertainty", [])[:8]
                if isinstance(item, dict)
            ],
        }

    def _compact_evidence_packet_for_prompt(self, evidence_packet: Dict[str, Any]) -> Dict[str, Any]:
        fact_inventory = []
        for item in evidence_packet.get("fact_inventory", [])[:36]:
            if not isinstance(item, dict):
                continue
            fact_inventory.append(
                {
                    "fact_type": item.get("fact_type"),
                    "value": self._compact_text_for_prompt(item.get("value"), limit=120),
                    "support_count": item.get("support_count"),
                    "photo_ids": list(item.get("photo_ids", [])[:6]),
                    "confidence": item.get("confidence"),
                }
            )
        photo_facts = []
        for item in evidence_packet.get("photo_facts", [])[:18]:
            if not isinstance(item, dict):
                continue
            photo_facts.append(
                {
                    "photo_id": item.get("photo_id"),
                    "timestamp": item.get("timestamp"),
                    "location_name": item.get("location_name"),
                    "person_ids": list(item.get("person_ids", [])[:8]),
                    "scene_hint": self._compact_text_for_prompt(item.get("scene_hint"), limit=80),
                    "activity_hint": self._compact_text_for_prompt(item.get("activity_hint"), limit=80),
                    "social_hint": self._compact_text_for_prompt(item.get("social_hint"), limit=80),
                    "rare_clues": list(item.get("rare_clues", [])[:8]),
                }
            )
        return {
            "event_id": evidence_packet.get("event_id"),
            "slice_id": evidence_packet.get("slice_id"),
            "time_range": dict(evidence_packet.get("time_range", {}) or {}),
            "location_chain": list(evidence_packet.get("location_chain", [])[:8]),
            "dominant_person_ids": list(evidence_packet.get("dominant_person_ids", [])[:8]),
            "burst_ids": list(evidence_packet.get("burst_ids", [])[:12]),
            "overlap_burst_ids": list(evidence_packet.get("overlap_burst_ids", [])[:4]),
            "fact_inventory": fact_inventory,
            "rare_clues": list(evidence_packet.get("rare_clues", [])[:24]),
            "change_points": list(evidence_packet.get("change_points", [])[:12]),
            "conflicts": list(evidence_packet.get("conflicts", [])[:8]),
            "slice_budget_metrics": dict(evidence_packet.get("slice_budget_metrics", {}) or {}),
            "photo_refs": list(evidence_packet.get("photo_refs", [])[:24]),
            "photo_facts": photo_facts,
            "session_context": dict(evidence_packet.get("session_context", {}) or {}),
        }

    def _create_session_merge_prompt(
        self,
        *,
        raw_session: Dict[str, Any],
        session_slice_records: List[Dict[str, Any]],
        slice_contracts: List[Dict[str, Any]],
        primary_person_id: Optional[str],
    ) -> str:
        if self.use_heavy_pipeline:
            return self._create_heavy_session_merge_prompt(
                raw_session=raw_session,
                session_slice_records=session_slice_records,
                slice_contracts=slice_contracts,
                primary_person_id=primary_person_id,
            )
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
        compact_slice_packets = [self._compact_slice_record_for_prompt(item) for item in session_slice_records]
        compact_slice_contracts = [self._compact_contract_for_prompt(item) for item in slice_contracts]
        return f"""你是 memory event aggregator。你会收到同一个 raw event 下多个带 overlap 的 slice memory contracts，请先在 event 内部做去重、拼接和保守确认。

规则：
1. 优先比较相邻 slice 的事件与 claims，依据 overlap_burst_ids / photo_ids / 时间连续性 / 人物连续性 / location_chain 做拼接。
2. 不要丢失 rare clues、OCR、brands、place claims、object last-seen claims。
3. 如果证据不足，宁可保守保留多条 event，也不要强行 merge。
4. “我” 对应 authenticated_user={primary_label}，不要把任何 Person_x 直接绑成用户。
5. 输出仍然是完整 6 段 memory contract。

Event summary:
{json.dumps(session_summary, ensure_ascii=False, indent=2)}

Slice packets:
{json.dumps(compact_slice_packets, ensure_ascii=False, indent=2)}

Slice contracts:
{json.dumps(compact_slice_contracts, ensure_ascii=False, indent=2)}
"""

    def _create_heavy_session_merge_prompt(
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
            "dominant_person_ids": raw_session["dominant_person_ids"],
            "continuity_decisions": raw_session["continuity_decisions"],
        }
        compact_slice_packets = [self._compact_slice_record_for_prompt(item) for item in session_slice_records]
        compact_slice_contracts = [self._compact_contract_for_prompt(item) for item in slice_contracts]
        return f"""你是 LP1 的 session 级聚合器。你会收到同一原始事件窗口下多个 slice 的输出，请只在同一 raw_session 内做去重与拼接。

要求：
1. 保留高价值 observations / claims / persona_evidence / social_dynamics。
2. overlap 的 slice 不能重复计数，但也不能把细节吞掉。
3. 相同 fact 若 title 接近、时间连续、地点连续、photo_ids overlap，应合并。
4. relationship_hypotheses 此阶段只允许保留弱共现，不做强关系定型。
5. profile_deltas 只保留有明确证据引用的增量。
6. “我” 代表 authenticated_user={primary_label}。
7. 所有长文本字段禁止直接出现 ASCII 双引号 `"`；引用 OCR 或原文时统一改用《》或「」。

输出仍为完整 6 段 contract。

Session summary:
{json.dumps(session_summary, ensure_ascii=False, indent=2)}

Slice packets:
{json.dumps(compact_slice_packets, ensure_ascii=False, indent=2)}

Slice contracts:
{json.dumps(compact_slice_contracts, ensure_ascii=False, indent=2)}
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
        if self.use_heavy_pipeline:
            return self._create_heavy_merge_prompt(
                photo_fact_count=photo_fact_count,
                raw_sessions=raw_sessions,
                session_slices=session_slices,
                session_contracts=session_contracts,
                session_artifacts=session_artifacts,
                primary_person_id=primary_person_id,
            )
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
        compact_event_contracts = [self._compact_contract_for_prompt(item) for item in session_contracts]
        return f"""你是 memory global aggregator。你会收到多个 raw event 的 event-level memory contracts，请做全局去重、合并、谨慎关系修订，并输出最终 memory contract。

规则：
1. 不要丢失高价值 observations / claims。
2. 允许多个 event 合并，但只能在时间/地点/活动证据充分时合并。
3. 关系版本要保守：没有纵向证据就不要输出强关系。
4. “我” 对应 authenticated_user={primary_label}，但不要因为缺少人脸锚点就把任何 Person_x 绑成用户。
5. profile_deltas 只输出增量，不直接下绝对结论。
6. 所有字符串内部如果需要出现双引号，必须转义为 \\\"，输出必须能被 JSON 解析器直接解析。

输出仍然是同样的 6 段 JSON contract，顶层字段必须完整。

输入概览：
- total_photo_facts: {photo_fact_count}
- raw_events: {json.dumps(compact_sessions, ensure_ascii=False)}
- session_slices: {json.dumps(compact_slices, ensure_ascii=False)}
- event_artifacts: {json.dumps(session_artifacts, ensure_ascii=False)}
- event_contracts:
{json.dumps(compact_event_contracts, ensure_ascii=False, indent=2)}
"""

    def _create_heavy_merge_prompt(
        self,
        *,
        photo_fact_count: int,
        raw_sessions: List[Dict[str, Any]],
        session_slices: List[Dict[str, Any]],
        session_contracts: List[Dict[str, Any]],
        session_artifacts: List[Dict[str, Any]],
        primary_person_id: Optional[str],
    ) -> str:
        primary_label = primary_person_id or "authenticated_user"
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
        compact_event_contracts = [self._compact_contract_for_prompt(item) for item in session_contracts]
        return f"""你是 LP1 的全局聚合器。你的任务是把多个 session 级 contract 合并成最终 memory contract，但不能丢失 slice 中已有的有效内容。

要求：
1. 核心目标是保留事实完整性，而不是把结果压到最少。
2. facts 之间只有在时间/地点/人物/活动证据充分连续时才能合并。
3. observations / claims 若有检索价值，优先保留。
4. relationship_hypotheses 此阶段只保留可从事件层直接观察到的弱共现证据；最终关系判定由下一层关系模型完成。
5. profile_deltas 允许保守增量，不允许把单次暴露写成稳定画像。
6. “我” 代表 authenticated_user={primary_label}，不能因为缺少 face 锚点就阻断用户视角。
7. 所有长文本字段禁止直接出现 ASCII 双引号 `"`；引用 OCR 或原文时统一改用《》或「」。

输出必须是完整 6 段 contract。

输入概览：
- total_photo_facts: {photo_fact_count}
- raw_sessions: {json.dumps(compact_sessions, ensure_ascii=False)}
- session_slices: {json.dumps(compact_slices, ensure_ascii=False)}
- session_artifacts: {json.dumps(session_artifacts, ensure_ascii=False)}
- session_contracts:
{json.dumps(compact_event_contracts, ensure_ascii=False, indent=2)}
"""

    def _empty_contract(self) -> Dict[str, Any]:
        return {
            "facts": [],
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
            if key == "facts":
                value = payload.get("facts", payload.get("events", []))
            else:
                value = payload.get(key, [])
            contract[key] = value if isinstance(value, list) else []

        for index, fact in enumerate(contract["facts"], start=1):
            if not isinstance(fact, dict):
                contract["facts"][index - 1] = {}
                fact = contract["facts"][index - 1]
            fact.setdefault("fact_id", fact.get("event_id") or f"FACT_{index:03d}")
            fact.setdefault("title", "")
            fact.setdefault("coarse_event_type", "other")
            fact.setdefault("event_facets", [])
            fact.setdefault("alternative_type_candidates", [])
            fact.setdefault("participant_person_ids", [])
            fact.setdefault("photo_ids", [])
            fact.setdefault("original_image_ids", list(fact.get("photo_ids", []) or []))
            fact.setdefault("description", "")
            fact.setdefault("narrative_synthesis", "")
            fact.setdefault("confidence", 0.0)
            fact.setdefault("reason", "")
            fact.setdefault("social_dynamics", [])
            fact.setdefault("persona_evidence", {})
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
            item.setdefault("fact_id", item.get("event_id", ""))
            item.setdefault("event_id", item.get("session_id", ""))
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
            item.setdefault("fact_id", item.get("event_id", ""))
            item.setdefault("event_id", item.get("session_id", ""))
            item.setdefault("evidence_refs", [])
        for index, item in enumerate(contract["relationship_hypotheses"], start=1):
            if not isinstance(item, dict):
                item = contract["relationship_hypotheses"][index - 1]
            item.setdefault("relationship_id", f"REL_{index:03d}")
            item.setdefault("person_id", item.get("target_person_id", ""))
            item.setdefault("relationship_type", "co_presence_only")
            item.setdefault("label", "")
            item.setdefault("confidence", 0.0)
            item.setdefault("supporting_fact_ids", item.get("supporting_event_ids", []))
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
            item.setdefault("supporting_fact_ids", item.get("supporting_event_ids", []))
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

    def _emit_progress(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]],
        payload: Dict[str, Any],
    ) -> None:
        if progress_callback:
            progress_callback(payload)

    def _progress_percent(self, processed: int, total: int, *, start: int, end: int) -> int:
        if total <= 0:
            return start
        ratio = min(1.0, max(0.0, processed / total))
        return int(round(start + ((end - start) * ratio)))

    def _iso_now(self) -> str:
        return datetime.now().isoformat()

    def _active_llm_provider(self) -> str:
        return self.provider

    def _active_llm_model(self) -> str:
        return self.model

    def _active_relationship_provider(self) -> str:
        return self.relationship_provider

    def _active_relationship_model(self) -> str:
        if self.relationship_use_bedrock and self.relationship_bedrock_model_candidates:
            return self.relationship_bedrock_model_candidates[0]
        return self.relationship_model

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
            lines.append("## recent_facts")
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

    def _call_json_prompt(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: Optional[tuple[int | float, int | float]] = None,
    ) -> Dict[str, Any]:
        if self.use_proxy:
            return self._call_llm_via_proxy(prompt)
        if self.use_openrouter:
            return self._call_llm_via_openrouter(
                prompt,
                max_tokens=int(max_tokens or 8192),
                response_format=response_format,
                timeout=timeout or (15, 180),
            )
        if self.use_bedrock:
            return self._call_llm_via_bedrock(prompt)
        return self._call_llm_via_official_api(prompt)

    def _call_json_prompt_raw_text(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: Optional[tuple[int | float, int | float]] = None,
    ) -> str:
        response = self._call_json_prompt_raw_response(
            prompt,
            max_tokens=max_tokens,
            response_format=response_format,
            timeout=timeout,
        )
        return str(response.get("text") or "")

    def _call_json_prompt_raw_response(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: Optional[tuple[int | float, int | float]] = None,
    ) -> Dict[str, Any]:
        if self.use_openrouter:
            payload = self._apply_openrouter_reasoning(
                {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": int(max_tokens or 8192),
                    "temperature": 0.1,
                }
            )
            if response_format:
                payload["response_format"] = deepcopy(response_format)
            response_data = self._post_openrouter_chat_completion(
                payload,
                timeout=timeout or (15, 180),
            )
            finish_reason = None
            choices = response_data.get("choices", [])
            if choices:
                finish_reason = choices[0].get("finish_reason")
            return {
                "text": self._extract_openrouter_content(response_data),
                "provider_response_id": response_data.get("id"),
                "provider_finish_reason": finish_reason,
                "provider_usage": response_data.get("usage"),
            }
        payload = self._call_json_prompt(
            prompt,
            max_tokens=max_tokens,
            response_format=response_format,
            timeout=timeout,
        )
        return {
            "text": json.dumps(payload, ensure_ascii=False, default=str),
            "provider_response_id": None,
            "provider_finish_reason": None,
            "provider_usage": None,
        }

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

    def _is_retryable_openrouter_status(self, status_code: int) -> bool:
        return status_code in {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}

    def _build_openrouter_error(self, response) -> str:
        error_msg = f"OpenRouter 返回状态码 {response.status_code}"
        if response.text:
            try:
                error_data = response.json()
                error_msg += f": {error_data.get('error', {}).get('message', response.text)}"
            except Exception:
                error_msg += f": {response.text[:200]}"
        return error_msg

    def _post_openrouter_chat_completion(
        self,
        payload: Dict[str, Any],
        *,
        timeout: tuple[int | float, int | float] = (15, 180),
    ) -> Dict[str, Any]:
        try:
            response = self.requests.post(
                f"{self.openrouter_base_url}/chat/completions",
                json=payload,
                headers=self._openrouter_headers(),
                timeout=timeout,
            )
        except Exception as exc:
            raise RuntimeError(f"OpenRouter 请求失败: {exc}") from exc

        if response.status_code == 200:
            try:
                return response.json()
            except Exception as exc:
                raise RuntimeError(f"OpenRouter 响应解析失败: {exc}") from exc

        error_msg = self._build_openrouter_error(response)
        if self._is_retryable_openrouter_status(int(response.status_code)):
            raise RuntimeError(error_msg)
        raise Exception(error_msg)

    def _extract_openrouter_content(self, response_data: Dict[str, Any]) -> str:
        choices = response_data.get("choices", [])
        if not choices:
            raise ValueError("OpenRouter 未返回 choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not content and message.get("reasoning"):
            content = message["reasoning"]
        return self._coerce_text_content(content)

    def _apply_openrouter_reasoning(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        effort = OPENROUTER_REASONING_EFFORT
        if effort:
            payload["reasoning"] = {"effort": effort}
        return payload

    def _call_llm_via_openrouter(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        max_tokens: int = 8192,
        timeout: tuple[int | float, int | float] = (15, 180),
        response_format: Optional[Dict[str, Any]] = None,
    ) -> dict:
        payload = self._apply_openrouter_reasoning({
            "model": model or self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.1,
        })
        if response_format:
            payload["response_format"] = deepcopy(response_format)
        response_data = self._post_openrouter_chat_completion(payload, timeout=timeout)
        return self._extract_json_payload(self._extract_openrouter_content(response_data))

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
        payload = self._apply_openrouter_reasoning({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8192,
            "temperature": 0.4,
        })
        response_data = self._post_openrouter_chat_completion(payload)
        return self._extract_openrouter_content(response_data)
