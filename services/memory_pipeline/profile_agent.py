from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any, Dict, List, Tuple

from .evidence_utils import build_evidence_payload, extract_ids_from_refs, flatten_ref_buckets
from .profile_tools import (
    analyze_evidence_stats,
    check_subject_ownership,
    extract_metadata_evidence,
    fetch_field_evidence,
    find_counter_evidence,
    get_resolved_facts,
)
from .types import FieldSpec, ProfileState

EVENT_EVIDENCE_KEYS = ("event_id", "date", "title", "location", "description", "narrative_synthesis", "tags", "photo_count")

_VLM_COMMON_KEYS = ("photo_id", "signal")

_VLM_FIELD_KEYS: Dict[str, Tuple[str, ...]] = {
    # identity
    "long_term_facts.identity.name": ("details", "ocr_hits"),
    "long_term_facts.identity.gender": ("people", "subject_role"),
    "long_term_facts.identity.age_range": ("activity", "details", "ocr_hits"),
    "long_term_facts.identity.role": ("activity", "work_signals", "details", "ocr_hits"),
    "long_term_facts.identity.race": ("people", "subject_role"),
    "long_term_facts.identity.nationality": ("location", "details", "ocr_hits", "normalized_places"),
    # social_identity
    "long_term_facts.social_identity.education": ("activity", "location", "details", "ocr_hits", "work_signals"),
    "long_term_facts.social_identity.career": ("activity", "work_signals", "details", "location"),
    "long_term_facts.social_identity.career_phase": ("activity", "work_signals", "details", "location"),
    "long_term_facts.social_identity.professional_dedication": ("activity", "work_signals", "details", "location"),
    "long_term_facts.social_identity.language_culture": ("details", "ocr_hits", "location"),
    "long_term_facts.social_identity.political_preference": ("details", "ocr_hits"),
    # material
    "long_term_facts.material.asset_level": ("brands", "details", "location"),
    "long_term_facts.material.spending_style": ("brands", "details", "ocr_hits"),
    "long_term_facts.material.brand_preference": ("brands", "details", "ocr_hits"),
    "long_term_facts.material.income_model": ("work_signals", "details", "activity"),
    "long_term_facts.material.signature_items": ("brands", "details"),
    # geography
    "long_term_facts.geography.location_anchors": ("location", "normalized_places", "place_candidates"),
    "long_term_facts.geography.mobility_pattern": ("location", "normalized_places", "timestamp"),
    "long_term_facts.geography.cross_border": ("location", "normalized_places"),
    # time
    "long_term_facts.time.life_rhythm": ("timestamp", "activity", "time_keys"),
    "long_term_facts.time.event_cycles": ("timestamp", "activity", "time_keys"),
    "long_term_facts.time.sleep_pattern": ("timestamp", "time_keys"),
    # relationships
    "long_term_facts.relationships.intimate_partner": ("people", "activity", "subject_role"),
    "long_term_facts.relationships.close_circle_size": ("people",),
    "long_term_facts.relationships.social_groups": ("people", "activity"),
    "long_term_facts.relationships.pets": ("activity", "details"),
    "long_term_facts.relationships.parenting": ("activity", "details", "people"),
    "long_term_facts.relationships.living_situation": ("location", "activity", "details"),
    # hobbies
    "long_term_facts.hobbies.interests": ("activity", "activity_tags", "details", "location"),
    "long_term_facts.hobbies.frequent_activities": ("activity", "activity_tags", "location"),
    "long_term_facts.hobbies.solo_vs_social": ("people", "activity", "subject_role"),
    # physiology
    "long_term_facts.physiology.fitness_level": ("activity", "details"),
    "long_term_facts.physiology.diet_mode": ("activity", "details", "location"),
    "long_term_facts.physiology.health_conditions": ("activity", "details"),
    # expression
    "long_term_expression.personality_mbti": ("activity", "details"),
    "long_term_expression.morality": ("activity", "details"),
    "long_term_expression.philosophy": ("activity", "details"),
    "long_term_expression.attitude_style": ("activity", "details"),
    "long_term_expression.aesthetic_tendency": ("activity", "details", "location"),
    "long_term_expression.visual_creation_style": ("activity", "details"),
    # short_term
    "short_term_facts.life_events": ("activity", "details", "timestamp"),
    "short_term_facts.phase_change": ("activity", "details", "timestamp"),
    "short_term_facts.spending_shift": ("brands", "details", "ocr_hits"),
    "short_term_facts.current_displacement": ("location", "normalized_places", "timestamp"),
    "short_term_facts.recent_habits": ("activity", "details", "timestamp"),
    "short_term_facts.recent_interests": ("activity", "activity_tags", "details"),
    "short_term_facts.physiological_state": ("activity", "details"),
    # short_term_expression
    "short_term_expression.current_mood": ("activity", "details", "people"),
    "short_term_expression.mental_state": ("activity", "details", "people"),
    "short_term_expression.motivation_shift": ("activity", "details", "people"),
    "short_term_expression.stress_signal": ("activity", "details", "people"),
    "short_term_expression.social_energy": ("activity", "details", "people"),
}

_VLM_DEFAULT_EXTRA_KEYS = ("activity", "details", "activity_tags", "location")


DOMAIN_SPECS: List[Dict[str, Any]] = [
    {
        "domain_key": "foundation_social_identity",
        "display_name": "Foundation & Social Identity",
        "fields": [
            "long_term_facts.identity.name",
            "long_term_facts.identity.gender",
            "long_term_facts.identity.age_range",
            "long_term_facts.identity.role",
            "long_term_facts.identity.race",
            "long_term_facts.identity.nationality",
            "long_term_facts.social_identity.education",
            "long_term_facts.social_identity.career",
            "long_term_facts.social_identity.career_phase",
            "long_term_facts.social_identity.professional_dedication",
            "long_term_facts.social_identity.language_culture",
        ],
    },
    {
        "domain_key": "wealth_consumption",
        "display_name": "Wealth & Consumption",
        "fields": [
            "long_term_facts.material.asset_level",
            "long_term_facts.material.spending_style",
            "long_term_facts.material.brand_preference",
            "long_term_facts.material.income_model",
            "long_term_facts.material.signature_items",
            "short_term_facts.spending_shift",
        ],
    },
    {
        "domain_key": "spatiotemporal_habits",
        "display_name": "Spatio-Temporal Habits",
        "fields": [
            "long_term_facts.geography.location_anchors",
            "long_term_facts.geography.mobility_pattern",
            "long_term_facts.geography.cross_border",
            "long_term_facts.time.life_rhythm",
            "long_term_facts.time.event_cycles",
            "long_term_facts.time.sleep_pattern",
            "short_term_facts.phase_change",
            "short_term_facts.current_displacement",
            "short_term_facts.recent_habits",
        ],
    },
    {
        "domain_key": "relationships_household",
        "display_name": "Relationships & Household",
        "fields": [
            "long_term_facts.relationships.intimate_partner",
            "long_term_facts.relationships.close_circle_size",
            "long_term_facts.relationships.social_groups",
            "long_term_facts.relationships.pets",
            "long_term_facts.relationships.parenting",
            "long_term_facts.relationships.living_situation",
        ],
    },
    {
        "domain_key": "taste_interests",
        "display_name": "Taste & Interests",
        "fields": [
            "long_term_facts.hobbies.interests",
            "long_term_facts.hobbies.frequent_activities",
            "long_term_facts.hobbies.solo_vs_social",
            "long_term_facts.physiology.fitness_level",
            "long_term_facts.physiology.diet_mode",
            "short_term_facts.life_events",
            "short_term_facts.recent_interests",
        ],
    },
    {
        "domain_key": "visual_expression",
        "display_name": "Visual Expression",
        "fields": [
            "long_term_expression.attitude_style",
            "long_term_expression.aesthetic_tendency",
            "long_term_expression.visual_creation_style",
            "short_term_expression.current_mood",
            "short_term_expression.social_energy",
        ],
    },
    {
        "domain_key": "semantic_expression",
        "display_name": "Semantic Expression",
        "fields": [
            "long_term_expression.personality_mbti",
            "long_term_expression.morality",
            "long_term_expression.philosophy",
            "short_term_expression.mental_state",
            "short_term_expression.motivation_shift",
            "short_term_expression.stress_signal",
        ],
    },
]

class ProfileAgent:
    def __init__(self, field_specs: Dict[str, FieldSpec]) -> None:
        self.field_specs = field_specs

    def run(
        self,
        context: Dict[str, Any],
        structured_profile: Dict[str, Any],
        llm_processor: Any | None = None,
        target_field_keys: set | None = None,
    ) -> Dict[str, Any]:
        profile_state = ProfileState(structured_profile=deepcopy(structured_profile))
        metadata_bundle = extract_metadata_evidence(context)
        profile_state.tool_cache["metadata_bundle"] = metadata_bundle
        profile_state.resolved_facts_summary = get_resolved_facts(profile_state.__dict__)["resolved_facts_summary"]

        for domain_spec in DOMAIN_SPECS:
            self._run_domain(domain_spec, context, profile_state, llm_processor=llm_processor, target_field_keys=target_field_keys)

        return {
            "structured": self._build_traceable_structured_profile(profile_state.structured_profile),
            "field_decisions": profile_state.field_decisions,
            "llm_batch_debug": profile_state.llm_batch_debug,
            "profile_state": profile_state.to_dict(),
        }

    def _run_domain(
        self,
        domain_spec: Dict[str, Any],
        context: Dict[str, Any],
        profile_state: ProfileState,
        llm_processor: Any | None = None,
        target_field_keys: set | None = None,
    ) -> None:
        prompt_units: List[Dict[str, Any]] = []
        for field_key in domain_spec["fields"]:
            if target_field_keys and field_key not in target_field_keys:
                continue
            spec = self.field_specs.get(field_key)
            if spec is None:
                continue

            evidence_bundle = fetch_field_evidence(field_key, context, profile_state=profile_state.__dict__)
            ownership_bundle = check_subject_ownership(field_key, evidence_bundle)
            stats_bundle = analyze_evidence_stats(
                field_key,
                evidence_bundle,
                ownership_bundle=ownership_bundle,
                profile_state=profile_state.__dict__,
            )
            counter_bundle = find_counter_evidence(
                field_key,
                evidence_bundle,
                ownership_bundle=ownership_bundle,
                profile_state=profile_state.__dict__,
            )
            tool_trace = {
                "evidence_bundle": evidence_bundle,
                "stats_bundle": stats_bundle,
                "ownership_bundle": ownership_bundle,
                "counter_bundle": counter_bundle,
            }

            if spec.requires_social_media and not profile_state.tool_cache["metadata_bundle"]["has_social_media_evidence"]:
                final = self._build_silent_null(field_key, evidence_bundle)
                self._record_field_result(
                    domain_spec=domain_spec,
                    batch_name=f"{domain_spec['display_name']}::silent",
                    spec=spec,
                    tool_trace=tool_trace,
                    draft=final,
                    final=final,
                    null_reason="silent_by_missing_social_media",
                    profile_state=profile_state,
                )
                continue

            if spec.requires_protagonist_face and not context.get("primary_person_id"):
                final = self._build_silent_null(field_key, evidence_bundle)
                self._record_field_result(
                    domain_spec=domain_spec,
                    batch_name=f"{domain_spec['display_name']}::silent",
                    spec=spec,
                    tool_trace=tool_trace,
                    draft=final,
                    final=final,
                    null_reason="silent_by_photographer_mode",
                    profile_state=profile_state,
                )
                continue

            deterministic = self._deterministic_field_value(field_key, context)
            if deterministic is not None:
                final = self._build_deterministic_final(field_key, deterministic, evidence_bundle)
                self._record_field_result(
                    domain_spec=domain_spec,
                    batch_name=f"{domain_spec['display_name']}::deterministic",
                    spec=spec,
                    tool_trace=tool_trace,
                    draft=final,
                    final=final,
                    null_reason=None,
                    profile_state=profile_state,
                )
                continue

            prompt_units.append(
                {
                    "field_key": field_key,
                    "spec": spec,
                    "tool_trace": tool_trace,
                }
            )

        if not prompt_units:
            return

        for batch_index, batch in enumerate(self._split_batches(prompt_units), start=1):
            batch_name = f"{domain_spec['display_name']}::batch_{batch_index}"
            response_map, batch_debug = self._run_batch(
                domain_spec=domain_spec,
                batch_name=batch_name,
                batch=batch,
                context=context,
                profile_state=profile_state,
                llm_processor=llm_processor,
            )
            profile_state.llm_batch_debug.append(batch_debug)
            for unit in batch:
                field_key = unit["field_key"]
                spec = unit["spec"]
                tool_trace = unit["tool_trace"]
                field_output = response_map.get(field_key, {})
                draft = self._build_draft(field_key, field_output, tool_trace)
                final, null_reason = self._build_final(field_key, draft, tool_trace, field_output)
                self._record_field_result(
                    domain_spec=domain_spec,
                    batch_name=batch_name,
                    spec=spec,
                    tool_trace=tool_trace,
                    draft=draft,
                    final=final,
                    null_reason=null_reason,
                    profile_state=profile_state,
                )

    def _run_batch(
        self,
        domain_spec: Dict[str, Any],
        batch_name: str,
        batch: List[Dict[str, Any]],
        context: Dict[str, Any],
        profile_state: ProfileState,
        llm_processor: Any | None = None,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        batch_debug = self._build_batch_debug(domain_spec, batch_name, batch, llm_processor)
        if llm_processor is None or not hasattr(llm_processor, "_call_llm_via_official_api"):
            batch_debug["fallback_reason"] = "no_llm_processor"
            batch_debug["used_offline_fallback"] = True
            return self._run_batch_offline(batch), batch_debug
        prompt = self._build_batch_prompt(domain_spec, batch, profile_state)
        try:
            result = llm_processor._call_llm_via_official_api(prompt, response_mime_type="application/json")
        except Exception as exc:
            self._merge_batch_debug(batch_debug, self._consume_llm_call_debug(llm_processor))
            batch_debug["api_call_attempted"] = True
            batch_debug["exception_type"] = exc.__class__.__name__
            batch_debug["exception_message"] = str(exc)
            batch_debug["fallback_reason"] = "exception"
            batch_debug["used_offline_fallback"] = True
            return self._run_batch_offline(batch), batch_debug
        self._merge_batch_debug(batch_debug, self._consume_llm_call_debug(llm_processor))
        batch_debug["api_call_attempted"] = True
        self._merge_batch_debug(batch_debug, self._summarize_raw_result(result))
        recovered = self._recover_batch_result(batch, result)
        batch_debug["recovered_field_count"] = len(recovered)
        batch_debug["recovered_field_keys"] = sorted(recovered.keys())
        if recovered:
            return recovered, batch_debug
        batch_debug["fallback_reason"] = "parse_failure" if not batch_debug["raw_result_parseable"] else "recovery_failure"
        batch_debug["used_offline_fallback"] = True
        return self._run_batch_offline(batch), batch_debug

    def _normalize_batch_result(
        self,
        batch: List[Dict[str, Any]],
        payload: Any,
    ) -> Dict[str, Dict[str, Any]]:
        expected_keys = [unit["field_key"] for unit in batch]
        normalized: Dict[str, Dict[str, Any]] = {}

        if isinstance(payload, list):
            for item in payload:
                if not isinstance(item, dict):
                    continue
                explicit_key = item.get("field_key") or item.get("key") or item.get("name")
                field_output = self._strip_field_identifier(item)
                if explicit_key in expected_keys and isinstance(field_output, dict):
                    normalized[explicit_key] = field_output
            if normalized:
                return normalized
            anonymous_items = [self._strip_field_identifier(item) for item in payload if isinstance(item, dict)]
            anonymous_items = [item for item in anonymous_items if isinstance(item, dict)]
            if anonymous_items and len(anonymous_items) == len(batch):
                for unit, item in zip(batch, anonymous_items):
                    normalized[unit["field_key"]] = item
                return normalized
            return {}

        if not isinstance(payload, dict):
            return {}

        for key, value in payload.items():
            if key in expected_keys and isinstance(value, dict):
                normalized[key] = value

        if normalized:
            return normalized

        # 兼容模型把模板里的字面量 "field_key" 当成真实 key 输出。
        if len(batch) == 1 and isinstance(payload.get("field_key"), dict):
            return {batch[0]["field_key"]: payload["field_key"]}

        explicit_items = []
        for value in payload.values():
            if not isinstance(value, dict):
                continue
            explicit_key = value.get("field_key") or value.get("key") or value.get("name")
            field_output = self._strip_field_identifier(value)
            if explicit_key in expected_keys and isinstance(field_output, dict):
                explicit_items.append((explicit_key, field_output))
        if explicit_items:
            return dict(explicit_items)

        wrapper_keys = {"fields", "results", "outputs", "items", "data", "result"}
        if payload and set(payload.keys()) <= wrapper_keys:
            return {}

        # 兼容弱结构结果（key 非标准但条数与 batch 一致），按 batch 顺序对齐。
        anonymous_items = [self._strip_field_identifier(value) for value in payload.values() if isinstance(value, dict)]
        anonymous_items = [value for value in anonymous_items if isinstance(value, dict)]
        if anonymous_items and len(anonymous_items) == len(batch):
            for unit, item in zip(batch, anonymous_items):
                normalized[unit["field_key"]] = item
            return normalized

        return {}

    def _recover_batch_result(
        self,
        batch: List[Dict[str, Any]],
        raw_result: Any,
    ) -> Dict[str, Dict[str, Any]]:
        payload = self._coerce_json_like_payload(raw_result)
        if payload is None:
            return {}

        for candidate in self._yield_candidate_payloads(payload):
            if len(batch) == 1 and isinstance(candidate, dict) and {"value", "confidence"} <= set(candidate.keys()):
                return {batch[0]["field_key"]: candidate}
            normalized = self._normalize_batch_result(batch, candidate)
            if normalized:
                return normalized
        return {}

    def _coerce_json_like_payload(self, raw_result: Any) -> Any:
        if raw_result is None:
            return None
        if isinstance(raw_result, (dict, list)):
            return raw_result
        if not isinstance(raw_result, str):
            return None

        text = raw_result.strip()
        if not text:
            return None

        for candidate in self._extract_json_candidates(text):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None

    def _extract_json_candidates(self, text: str) -> List[str]:
        candidates = [text]
        fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        candidates.extend(block.strip() for block in fenced_blocks if block.strip())

        first_object = text.find("{")
        last_object = text.rfind("}")
        if 0 <= first_object < last_object:
            candidates.append(text[first_object:last_object + 1])

        first_array = text.find("[")
        last_array = text.rfind("]")
        if 0 <= first_array < last_array:
            candidates.append(text[first_array:last_array + 1])

        deduped: List[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.strip()
            if normalized and normalized not in seen:
                deduped.append(normalized)
                seen.add(normalized)
        return deduped

    def _yield_candidate_payloads(self, payload: Any):
        queue: List[Any] = [payload]
        seen: set[int] = set()
        wrapper_keys = ("fields", "results", "outputs", "items", "data", "result")

        while queue:
            candidate = queue.pop(0)
            candidate_id = id(candidate)
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            yield candidate

            if isinstance(candidate, dict):
                for key in wrapper_keys:
                    nested = candidate.get(key)
                    if isinstance(nested, (dict, list)):
                        queue.append(nested)

    def _strip_field_identifier(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: value
            for key, value in payload.items()
            if key not in {"field_key", "key", "name"}
        }

    def _build_batch_debug(
        self,
        domain_spec: Dict[str, Any],
        batch_name: str,
        batch: List[Dict[str, Any]],
        llm_processor: Any | None,
    ) -> Dict[str, Any]:
        return {
            "domain_name": domain_spec.get("display_name"),
            "batch_name": batch_name,
            "field_keys": [unit.get("field_key") for unit in batch],
            "batch_size": len(batch),
            "llm_processor_class": llm_processor.__class__.__name__ if llm_processor is not None else None,
            "api_call_available": bool(llm_processor and hasattr(llm_processor, "_call_llm_via_official_api")),
            "api_call_attempted": False,
            "http_status_code": None,
            "model": getattr(llm_processor, "model", None),
            "raw_response_preview": "",
            "raw_response_truncated": False,
            "raw_result_type": None,
            "raw_result_parseable": False,
            "parsed_payload_type": None,
            "recovered_field_count": 0,
            "recovered_field_keys": [],
            "used_offline_fallback": False,
            "fallback_reason": None,
            "exception_type": None,
            "exception_message": None,
        }

    def _consume_llm_call_debug(self, llm_processor: Any | None) -> Dict[str, Any]:
        if llm_processor is None or not hasattr(llm_processor, "_consume_last_call_debug"):
            return {}
        try:
            payload = llm_processor._consume_last_call_debug()
        except Exception:
            return {}
        return dict(payload or {}) if isinstance(payload, dict) else {}

    def _merge_batch_debug(self, batch_debug: Dict[str, Any], updates: Dict[str, Any]) -> None:
        for key, value in (updates or {}).items():
            if value in (None, "", [], {}):
                continue
            batch_debug[key] = value

    def _summarize_raw_result(self, raw_result: Any) -> Dict[str, Any]:
        payload = self._coerce_json_like_payload(raw_result)
        preview_source = raw_result
        if isinstance(raw_result, (dict, list)):
            preview_source = json.dumps(raw_result, ensure_ascii=False)
        preview, truncated = _truncate_debug_text(preview_source)
        return {
            "raw_result_type": type(raw_result).__name__ if raw_result is not None else None,
            "raw_response_preview": preview,
            "raw_response_truncated": truncated,
            "raw_result_parseable": payload is not None,
            "parsed_payload_type": type(payload).__name__ if payload is not None else None,
        }

    def _run_batch_offline(self, batch: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        outputs: Dict[str, Dict[str, Any]] = {}
        for unit in batch:
            field_key = unit["field_key"]
            tool_trace = unit["tool_trace"]
            field_output = self._offline_field_output(field_key, tool_trace)
            if field_output:
                outputs[field_key] = field_output
        return outputs

    def _offline_field_output(
        self,
        field_key: str,
        tool_trace: Dict[str, Any],
    ) -> Dict[str, Any]:
        evidence_bundle = tool_trace.get("evidence_bundle", {}) or {}
        stats_bundle = tool_trace.get("stats_bundle", {}) or {}
        counter_bundle = tool_trace.get("counter_bundle", {}) or {}
        supporting_refs = flatten_ref_buckets((evidence_bundle.get("supporting_refs") or {}))
        supporting_ref_ids = self._collect_ref_ids(supporting_refs)
        contradicting_refs = counter_bundle.get("contradicting_refs", []) or []
        contradicting_ref_ids = self._collect_ref_ids(contradicting_refs)

        if field_key == "long_term_facts.material.brand_preference":
            top_brands = ((stats_bundle.get("brand_summary") or {}).get("top_brands") or [])
            ownership_map = {
                item.get("candidate"): item.get("signal")
                for item in (tool_trace.get("ownership_bundle", {}).get("candidate_signals") or [])
                if item.get("candidate")
            }
            selected = [
                item.get("brand_name")
                for item in top_brands
                if item.get("brand_name")
                and int(item.get("source_count", 0) or 0) >= 2
                and ownership_map.get(item.get("brand_name"), "background_or_ambiguous") in {"owned_or_used", "worn"}
            ]
            selected = list(dict.fromkeys(selected))
            if not selected:
                return {}
            has_cross_scene = any(
                int(item.get("event_count", 0) or 0) >= 2 and int(item.get("scene_count", 0) or 0) >= 2
                for item in top_brands
                if item.get("brand_name") in selected
            )
            confidence = 0.72 if has_cross_scene else 0.6
            return {
                "value": selected[:3],
                "confidence": confidence,
                "reasoning": "离线批处理基于 LP1 事件与 VLM 品牌线索统计，选取跨来源重复出现的品牌。",
                "supporting_ref_ids": supporting_ref_ids,
                "contradicting_ref_ids": contradicting_ref_ids,
                "null_reason": None,
            }

        if field_key == "long_term_facts.geography.location_anchors":
            candidates = ((stats_bundle.get("location_summary") or {}).get("top_city_candidates") or [])
            selected = []
            for item in candidates:
                if item.get("primary_role") in {"travel_place", "others_place", "passby_place", "generic_place"}:
                    continue
                city_name = str(item.get("city_name", "") or "")
                if city_name.endswith(("广场", "公园", "博物馆", "展厅", "景区", "门店", "酒店")):
                    continue
                if (
                    int(item.get("event_count", 0) or 0) < 1
                    and int(item.get("photo_count", 0) or 0) < 2
                    and int(item.get("city_hit_count", 0) or 0) < 2
                ):
                    continue
                selected.append(city_name)
            selected = [name for name in list(dict.fromkeys(selected)) if name]
            if not selected:
                return {}
            confidence = 0.7 if len(selected) >= 2 else 0.62
            return {
                "value": selected[:3],
                "confidence": confidence,
                "reasoning": "离线批处理基于城市级归一后的地点桶，保留跨窗口稳定出现的城市锚点。",
                "supporting_ref_ids": supporting_ref_ids,
                "contradicting_ref_ids": contradicting_ref_ids,
                "null_reason": None,
            }

        if field_key == "short_term_facts.recent_interests":
            candidates = ((stats_bundle.get("recent_topic_summary") or {}).get("top_topics") or [])
            selected = []
            for item in candidates:
                topic_name = item.get("topic_name")
                if not topic_name:
                    continue
                if int(item.get("event_count", 0) or 0) < 2:
                    continue
                if float(item.get("novelty_score", 0.0) or 0.0) <= 0:
                    continue
                selected.append(topic_name)
            selected = list(dict.fromkeys(selected))
            if not selected:
                return {}
            return {
                "value": selected[:3],
                "confidence": 0.64,
                "reasoning": "离线批处理基于近期事件主题重复出现与相对长期兴趣的新颖度判断 recent_interests。",
                "supporting_ref_ids": supporting_ref_ids,
                "contradicting_ref_ids": contradicting_ref_ids,
                "null_reason": None,
            }

        if field_key == "long_term_facts.social_identity.education":
            signal_text = self._collect_supporting_text(supporting_refs).lower()
            hit_count = sum(
                signal_text.count(keyword)
                for keyword in ("校园", "学校", "教室", "campus", "classroom", "lecture", "college")
            )
            if hit_count < 2:
                return {}
            return {
                "value": "college_student",
                "confidence": 0.62,
                "reasoning": "离线批处理基于 LP1 事件与 VLM 场景中的校园学习信号，判定为 college_student。",
                "supporting_ref_ids": supporting_ref_ids,
                "contradicting_ref_ids": contradicting_ref_ids,
                "null_reason": None,
            }

        if field_key == "long_term_facts.social_identity.career":
            work_signal_summary = stats_bundle.get("work_signal_summary") or {}
            if int(work_signal_summary.get("total_signal_count", 0) or 0) < 2:
                return {}
            return {
                "value": "student_or_early_career",
                "confidence": 0.56,
                "reasoning": "离线批处理基于工资/考勤/打卡/现场等工作信号摘要，输出保守职业标签。",
                "supporting_ref_ids": supporting_ref_ids,
                "contradicting_ref_ids": contradicting_ref_ids,
                "null_reason": None,
            }

        if field_key == "long_term_facts.material.income_model":
            work_signal_summary = stats_bundle.get("work_signal_summary") or {}
            payroll_hits = int(work_signal_summary.get("payroll_sheet", 0) or 0)
            attendance_hits = int(work_signal_summary.get("attendance_sheet", 0) or 0)
            punch_hits = int(work_signal_summary.get("punch_in_scene", 0) or 0)
            if payroll_hits + attendance_hits + punch_hits < 2:
                return {}
            confidence = 0.62 if payroll_hits >= 1 else 0.56
            return {
                "value": "salary",
                "confidence": confidence,
                "reasoning": "离线批处理基于工资表、考勤表和打卡场景的结构化工作信号，判定收入模式更接近工资制。",
                "supporting_ref_ids": supporting_ref_ids,
                "contradicting_ref_ids": contradicting_ref_ids,
                "null_reason": None,
            }

        return {}

    def _collect_ref_ids(self, refs: List[Dict[str, Any]]) -> List[str]:
        ids: List[str] = []
        for ref in refs:
            ids.extend(_candidate_ref_ids(ref))
        return list(dict.fromkeys(ids))

    def _collect_supporting_text(self, refs: List[Dict[str, Any]]) -> str:
        snippets: List[str] = []
        for ref in refs:
            for key in ("signal", "description", "narrative_synthesis", "location", "activity", "type"):
                value = ref.get(key)
                if isinstance(value, str) and value.strip():
                    snippets.append(value.strip())
        return " | ".join(snippets)

    def _build_batch_prompt(
        self,
        domain_spec: Dict[str, Any],
        batch: List[Dict[str, Any]],
        profile_state: ProfileState,
    ) -> str:
        resolved_facts_summary = profile_state.resolved_facts_summary
        resolved_clause = ""
        if resolved_facts_summary:
            resolved_clause = (
                f"\n# Resolved Facts (已确定的上游字段)\n"
                f"{resolved_facts_summary}\n"
            )

        field_units = []
        for unit in batch:
            spec = unit["spec"]
            tool_trace = unit["tool_trace"]

            field_units.append(
                f"""### Field: {unit['field_key']}
Risk: {spec.risk_level}
High Weight Signals:
{_format_list_for_prompt(spec.strong_evidence)}
COT Steps:
{_format_list_for_prompt(spec.cot_steps)}
{f"Important Hint: {spec.cot_hint}" if spec.cot_hint else ""}
{f"Field Boundary: {spec.field_boundary}" if spec.field_boundary else ""}
{f"Cross-Field Caution: {spec.cross_field_caution}" if spec.cross_field_caution else ""}
Owner Resolution:
{_format_list_for_prompt(spec.owner_resolution_steps)}
Time Reasoning:
{_format_list_for_prompt(spec.time_reasoning_steps)}
Counter Evidence Checks:
{_format_list_for_prompt(spec.counter_evidence_checks)}
Requires Social Media: {spec.requires_social_media}
Evidence Summary:
{self._summarize_evidence(tool_trace['evidence_bundle'], field_key=unit['field_key'])}
Stats Summary:
{self._summarize_stats_bundle(tool_trace['stats_bundle'])}
Ownership Summary:
{self._summarize_ownership_bundle(tool_trace['ownership_bundle'])}
Counter Summary:
{self._summarize_counter_bundle(tool_trace['counter_bundle'])}
"""
            )

        return f"""# Role
你是结构化画像的字段裁决 agent。

# Domain: {domain_spec['display_name']}
Current fields: {[unit['field_key'] for unit in batch]}
{resolved_clause}
# Reasoning Protocol
Step 1: 先基于字段级 COT 形成草案，分析所有可用证据。
Step 2: High Weight Signals / Stats / Counter 只作为证据权重与置信度加权依据，而不是硬门槛。
Step 3: 只要存在任何可靠支持证据，就先输出最佳推断值（value）并用 confidence 表达不确定性；不要因为证据不完美而默认 null。如果证据中包含具体的子类线索（如 top_candidates、detail_snippet 中的品牌名、学校类型、活动子类型），应在推断时充分利用这些线索来区分子类，而不是直接归并为泛类。
Step 4: 同一组字段之间以及 Resolved Facts 中已确定的字段，构成当前字段判断的上下文。你输出的标签应在这个人的已知身份和生活状态下有意义。如果当前字段的草案与已确定字段存在明显冲突或不协调，应在 reasoning 中说明并调整。
Step 5: 只有在完全没有相关证据时，或者全部证据都只能指向他人/背景/噪声而无法回到主角本人时，才输出 null。

# Field Units
{chr(10).join(field_units)}

# Output Contract
严格输出 JSON:
{{
  "fields": {{
    "<必须使用真实字段key，例如 long_term_facts.social_identity.education>": {{
      "value": null,
      "confidence": 0.0,
      "reasoning": "",
      "supporting_ref_ids": [],
      "contradicting_ref_ids": [],
      "null_reason": null
    }}
  }}
}}

注意：
1. `fields` 内每个 key 必须是 `Current fields` 里的真实字段名，不能输出字面量 `field_key`。
2. 多字段 batch 时，`Current fields` 中每个字段都要返回一个对象。
3. **关键指令**：忽略 High Weight Signals 的严格要求。只要有任何可靠的支持证据（即使不完美），就输出该值和相应的 confidence 评分。
4. 如果证据偏弱、时间跨度不足、跨事件重复不够或存在轻微反证，不要直接输出 null，而是降低 confidence。
5. 只有在完全零证据或主体归属明显错误时才输出 null。
6. 证据引用只需要可追溯锚点（event_id/photo_id/person_id/feature_name），不要重复展开整包原始证据。
7. reasoning 中必须引用至少 1 条具体证据细节（如品牌名、地点名、活动子类型、学校名），不得只写概括性描述。"""

    def _split_batches(self, prompt_units: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        batches: List[List[Dict[str, Any]]] = []
        current: List[Dict[str, Any]] = []
        current_limit = 5
        for unit in prompt_units:
            limit = 2 if unit["spec"].risk_level == "P0" else 5
            if current and (len(current) >= current_limit or limit < current_limit):
                batches.append(current)
                current = []
                current_limit = limit
            current_limit = min(current_limit, limit)
            current.append(unit)
        if current:
            batches.append(current)
        return batches

    def _deterministic_field_value(self, field_key: str, context: Dict[str, Any]) -> Tuple[Any, float] | None:
        relationships = context.get("relationships", [])
        groups = context.get("groups", [])
        events = context.get("events", [])

        if field_key == "long_term_facts.relationships.intimate_partner":
            romantic = next((rel.person_id for rel in relationships if rel.relationship_type == "romantic"), None)
            return romantic, 0.88 if romantic else 0.0
        if field_key == "long_term_facts.relationships.close_circle_size":
            return _compute_close_circle_size(relationships), 0.82
        if field_key == "long_term_facts.relationships.social_groups":
            group_names = [group.group_type_candidate for group in groups]
            return (group_names or None), (0.76 if group_names else 0.0)
        if field_key == "long_term_facts.time.sleep_pattern":
            ratio = _compute_late_night_event_ratio(events)
            if ratio >= 0.5:
                return "night_owl", 0.65
            if ratio > 0:
                return "irregular", 0.55
            return None, 0.0
        if field_key == "long_term_facts.relationships.living_situation":
            inferred = _infer_living_situation_from_events_and_relationships(events, relationships)
            if inferred is not None:
                return inferred, 0.78
            return None, 0.0
        return None

    def _build_silent_null(self, field_key: str, evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
        evidence = self._build_tag_evidence(
            evidence_bundle,
            selected_supporting_ids=[],
            selected_contradicting_ids=[],
            constraint_notes=["silent_by_missing_social_media"],
        )
        return {
            "value": None,
            "confidence": 0.0,
            "evidence": evidence,
            "reasoning": f"{field_key} 当前缺少社媒模态，未进入推断；因此静默输出 null。",
        }

    def _build_deterministic_final(
        self,
        field_key: str,
        deterministic: Tuple[Any, float],
        evidence_bundle: Dict[str, Any],
    ) -> Dict[str, Any]:
        value, confidence = deterministic
        evidence = self._build_tag_evidence(evidence_bundle)
        reasoning = _build_reasoning_from_ids(field_key, value, evidence, explicit_reason=None)
        return {
            "value": value,
            "confidence": confidence if value is not None else 0.0,
            "evidence": evidence,
            "reasoning": reasoning,
        }

    def _build_draft(self, field_key: str, field_output: Dict[str, Any], tool_trace: Dict[str, Any]) -> Dict[str, Any]:
        evidence = self._build_tag_evidence(
            tool_trace["evidence_bundle"],
            selected_supporting_ids=field_output.get("supporting_ref_ids", []),
            selected_contradicting_ids=field_output.get("contradicting_ref_ids", []),
        )
        return {
            "value": field_output.get("draft_value", field_output.get("value")),
            "confidence": float(field_output.get("draft_confidence", field_output.get("confidence", 0.0)) or 0.0),
            "evidence": evidence,
        }

    def _build_final(
        self,
        field_key: str,
        draft: Dict[str, Any],
        tool_trace: Dict[str, Any],
        field_output: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str | None]:
        value = draft["value"]
        confidence = draft["confidence"]
        null_reason = field_output.get("null_reason")
        time_summary = (tool_trace.get("stats_bundle", {}) or {}).get("time_summary") or {}
        if (
            field_key in {
                "long_term_facts.time.life_rhythm",
                "long_term_facts.time.event_cycles",
            }
            and null_reason
            and int(time_summary.get("distinct_months", 0) or 0) > 1
            and any(marker in str(null_reason).lower() for marker in ("same day", "single day", "同一天", "单日"))
        ):
            null_reason = "cross_month_but_pattern_not_stable"

        # 检查非日常事件突发干扰
        if (value is not None and
            field_key in ("long_term_facts.material.asset_level",
                         "long_term_facts.material.spending_style",
                         "long_term_facts.material.brand_preference",
                         "long_term_facts.geography.location_anchors",
                         "long_term_facts.geography.mobility_pattern",
                         "long_term_facts.hobbies.interests",
                         "long_term_facts.hobbies.frequent_activities")):
            # 检查证据是否来自非日常事件的浓密拍摄
            evidence_bundle = tool_trace.get("evidence_bundle", {})
            supporting_events = evidence_bundle.get("supporting_refs", {}).get("events", [])
            supporting_vlm = evidence_bundle.get("supporting_refs", {}).get("vlm_observations", [])

            unique_event_ids = {ref.get("event_id") for ref in supporting_events if ref.get("event_id")}
            concentrated_single_event = len(unique_event_ids) <= 1 and (
                len(supporting_vlm) >= 3 or any((ref.get("photo_count") or 0) >= 3 for ref in supporting_events)
            )

            # 检查是否包含非日常关键词
            from .profile_fields import NON_DAILY_EVENT_KEYWORDS, _is_non_daily_event_ref, _contains_any_keyword
            non_daily_signal = any(_is_non_daily_event_ref(ref) for ref in supporting_events) or any(
                _contains_any_keyword(ref.get("signal", ""), NON_DAILY_EVENT_KEYWORDS)
                for ref in supporting_vlm
            )

            if concentrated_single_event and non_daily_signal:
                # 降低 confidence 而非强制 null
                confidence = max(0.0, confidence * 0.75)  # 降低 25%

        constraint_notes = [null_reason] if null_reason else []
        # 添加 constraint_notes 中的信息
        if field_output.get("constraint_notes"):
            constraint_notes.extend(field_output.get("constraint_notes", []))
        evidence = self._build_tag_evidence(
            tool_trace["evidence_bundle"],
            selected_supporting_ids=field_output.get("supporting_ref_ids", []),
            selected_contradicting_ids=field_output.get("contradicting_ref_ids", []),
            constraint_notes=constraint_notes,
        )
        reasoning = _build_reasoning_from_ids(
            field_key,
            value,
            evidence,
            explicit_reason=field_output.get("reasoning"),
        )
        return {
            "value": value,
            "confidence": confidence if value is not None else 0.0,
            "evidence": evidence,
            "reasoning": reasoning,
        }, null_reason

    def _build_tag_evidence(
        self,
        evidence_bundle: Dict[str, Any],
        selected_supporting_ids: List[str] | None = None,
        selected_contradicting_ids: List[str] | None = None,
        constraint_notes: List[str] | None = None,
    ) -> Dict[str, Any]:
        ref_index = evidence_bundle["ref_index"]
        all_supporting = flatten_ref_buckets(evidence_bundle["supporting_refs"])
        all_contradicting = evidence_bundle["ref_index"]

        supporting_refs = self._select_refs(ref_index, selected_supporting_ids) or all_supporting
        contradicting_refs = self._select_refs(ref_index, selected_contradicting_ids)
        ids = extract_ids_from_refs(supporting_refs)
        evidence = build_evidence_payload(
            photo_ids=ids["photo_ids"],
            event_ids=ids["event_ids"],
            person_ids=ids["person_ids"],
            group_ids=ids["group_ids"],
            feature_names=ids["feature_names"],
            supporting_refs=supporting_refs,
            contradicting_refs=contradicting_refs,
        )
        evidence["events"] = self._filter_bucket(evidence_bundle["allowed_refs"]["events"], supporting_refs)
        evidence["relationships"] = self._filter_bucket(evidence_bundle["allowed_refs"]["relationships"], supporting_refs)
        evidence["vlm_observations"] = self._filter_bucket(evidence_bundle["allowed_refs"]["vlm_observations"], supporting_refs)
        evidence["group_artifacts"] = self._filter_bucket(evidence_bundle["allowed_refs"]["group_artifacts"], supporting_refs)
        evidence["feature_refs"] = self._filter_bucket(evidence_bundle["allowed_refs"]["feature_refs"], supporting_refs)
        evidence["constraint_notes"] = list(dict.fromkeys(constraint_notes or []))
        evidence["summary"] = f"profile_agent:{evidence_bundle['field_key']}"
        return evidence

    def _record_field_result(
        self,
        *,
        domain_spec: Dict[str, Any],
        batch_name: str,
        spec: FieldSpec,
        tool_trace: Dict[str, Any],
        draft: Dict[str, Any],
        final: Dict[str, Any],
        null_reason: str | None,
        profile_state: ProfileState,
    ) -> None:
        _assign_tag_object(profile_state.structured_profile, spec.field_key, final)
        profile_state.resolved_tags[spec.field_key] = final
        profile_state.resolved_facts_summary = get_resolved_facts(profile_state.__dict__)["resolved_facts_summary"]
        profile_state.field_decisions.append(
            {
                "field_key": spec.field_key,
                "domain_name": domain_spec["display_name"],
                "batch_name": batch_name,
                "field_spec_snapshot": spec.to_dict(),
                "resolved_facts_summary_at_decision_time": deepcopy(profile_state.resolved_facts_summary),
                "tool_trace": tool_trace,
                "draft": draft,
                "final": final,
                "null_reason": null_reason,
            }
        )

    def _build_traceable_structured_profile(self, payload: Any) -> Any:
        if isinstance(payload, dict):
            if {"value", "confidence", "evidence", "reasoning"} <= set(payload.keys()):
                traceable = deepcopy(payload)
                traceable["evidence"] = self._build_traceable_evidence(payload.get("evidence", {}) or {})
                return traceable
            return {
                key: self._build_traceable_structured_profile(value)
                for key, value in payload.items()
            }
        if isinstance(payload, list):
            return [self._build_traceable_structured_profile(item) for item in payload]
        return deepcopy(payload)

    def _build_traceable_evidence(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        supporting_refs = list(evidence.get("supporting_refs", []) or [])
        contradicting_refs = list(evidence.get("contradicting_refs", []) or [])
        return {
            "photo_ids": list(evidence.get("photo_ids", []) or []),
            "event_ids": list(evidence.get("event_ids", []) or []),
            "person_ids": list(evidence.get("person_ids", []) or []),
            "group_ids": list(evidence.get("group_ids", []) or []),
            "feature_names": list(evidence.get("feature_names", []) or []),
            "supporting_ref_count": len(supporting_refs),
            "contradicting_ref_count": len(contradicting_refs),
            "constraint_notes": list(evidence.get("constraint_notes", []) or []),
            "summary": str(evidence.get("summary", "") or ""),
        }

    def _summarize_evidence(self, evidence_bundle: Dict[str, Any], field_key: str = "") -> Dict[str, Any]:
        compact = evidence_bundle.get("compact") or {}
        supporting = evidence_bundle.get("supporting_refs") or {}

        # Event: 给完整描述字段
        events = [
            {k: ref.get(k) for k in EVENT_EVIDENCE_KEYS if ref.get(k) is not None}
            for ref in (supporting.get("events") or [])
        ]

        # VLM: 公共字段 + 字段专属子字段
        extra_keys = _VLM_FIELD_KEYS.get(field_key, _VLM_DEFAULT_EXTRA_KEYS)
        vlm_keys = _VLM_COMMON_KEYS + extra_keys
        vlm = [
            {k: ref.get(k) for k in vlm_keys if ref.get(k) is not None}
            for ref in (supporting.get("vlm_observations") or [])
        ]

        result: Dict[str, Any] = {
            "summary": compact.get("summary", ""),
            "source_coverage": compact.get("source_coverage", {}),
            "top_candidates": list(compact.get("top_candidates") or []),
            "evidence_ids": compact.get("evidence_ids", {}),
        }
        if events:
            result["events"] = events
        if vlm:
            result["vlm"] = vlm
        for source in ("relationships", "group_artifacts", "feature_refs"):
            refs = supporting.get(source)
            if refs:
                result[source] = refs
        return result

    def _summarize_stats_bundle(self, stats_bundle: Dict[str, Any]) -> Dict[str, Any]:
        summary = {
            key: value
            for key, value in stats_bundle.items()
            if key in {
                "support_count",
                "event_count",
                "photo_count",
                "window_count",
                "cross_event_stability",
                "burst_score",
                "suggested_strong_evidence_met",
            }
        }
        if stats_bundle.get("brand_summary"):
            summary["brand_summary"] = {
                "top_brands": list((stats_bundle["brand_summary"].get("top_brands") or [])[:5]),
                "suppressed_candidates": list((stats_bundle["brand_summary"].get("suppressed_candidates") or [])[:5]),
                "rejected_candidates": list((stats_bundle["brand_summary"].get("rejected_candidates") or [])[:8]),
            }
        if stats_bundle.get("location_summary"):
            summary["location_summary"] = {
                "top_city_candidates": list(((stats_bundle.get("location_summary") or {}).get("top_city_candidates") or [])[:5]),
            }
        if stats_bundle.get("recent_topic_summary"):
            overlap_items = list((((stats_bundle.get("recent_topic_summary") or {}).get("overlap_with_long_term") or {}).items()))[:5]
            summary["recent_topic_summary"] = {
                "top_topics": list(((stats_bundle.get("recent_topic_summary") or {}).get("top_topics") or [])[:5]),
                "overlap_with_long_term": dict(overlap_items),
            }
        if stats_bundle.get("time_summary"):
            time_summary = dict(stats_bundle.get("time_summary") or {})
            summary["time_summary"] = {
                "month_histogram": dict(time_summary.get("month_histogram", {}) or {}),
                "distinct_months": time_summary.get("distinct_months", 0),
                "span_days": time_summary.get("span_days", 0),
                "recurring_pattern_candidates": list(time_summary.get("recurring_pattern_candidates", []) or [])[:5],
                "evidence_event_ids": list(time_summary.get("evidence_event_ids", []) or [])[:8],
            }
        if stats_bundle.get("work_signal_summary"):
            work_signal_summary = dict(stats_bundle.get("work_signal_summary") or {})
            summary["work_signal_summary"] = {
                key: value
                for key, value in work_signal_summary.items()
                if key != "signal_ids"
            }
        return summary

    def _summarize_ownership_bundle(self, ownership_bundle: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "field_key": ownership_bundle.get("field_key"),
            "ownership_signal": ownership_bundle.get("ownership_signal"),
            "candidate_signals": list((ownership_bundle.get("candidate_signals") or [])[:8]),
        }

    def _summarize_counter_bundle(self, counter_bundle: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "field_key": counter_bundle.get("field_key"),
            "conflict_types": list(counter_bundle.get("conflict_types", []) or []),
            "conflict_strength": counter_bundle.get("conflict_strength", 0),
            "conflict_summary": counter_bundle.get("conflict_summary", ""),
            "contradicting_ids": list((counter_bundle.get("contradicting_ids") or [])[:12]),
        }

    def _select_refs(self, ref_index: Dict[str, Dict[str, Any]], ref_ids: List[str] | None) -> List[Dict[str, Any]]:
        if not ref_ids:
            return []
        refs: List[Dict[str, Any]] = []
        for ref_id in ref_ids:
            if ref_id in ref_index:
                refs.append(ref_index[ref_id])
        return _dedupe_refs(refs)

    def _filter_bucket(self, bucket: List[Dict[str, Any]], selected_refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not selected_refs:
            return []
        selected_ids = set()
        for ref in selected_refs:
            selected_ids.update(_candidate_ref_ids(ref))
        filtered = []
        for ref in bucket:
            if selected_ids.intersection(_candidate_ref_ids(ref)):
                filtered.append(ref)
        return filtered

def _assign_tag_object(payload: Dict[str, Any], field_key: str, value: Dict[str, Any]) -> None:
    path = field_key.split(".")
    current = payload
    for part in path[:-1]:
        current = current.setdefault(part, {})
    current[path[-1]] = value


def _build_reasoning_from_ids(
    field_key: str,
    value: Any,
    evidence: Dict[str, Any],
    explicit_reason: str | None,
) -> str:
    evidence_ids = (
        evidence.get("event_ids", [])[:2]
        + evidence.get("photo_ids", [])[:2]
        + evidence.get("person_ids", [])[:1]
        + evidence.get("feature_names", [])[:2]
    )
    evidence_clause = "、".join(evidence_ids) if evidence_ids else "当前证据池"
    if value is None:
        if "silent_by_missing_social_media" in evidence.get("constraint_notes", []):
            return f"{field_key} 当前缺少社媒模态，未进入推断；审查 {evidence_clause} 后静默输出 null。"
        if "null_due_to_expression_conflict_reflection" in evidence.get("constraint_notes", []):
            return f"{field_key} 读取已定稿 facts 后，在事件证据中发现明显冲突；审查 {evidence_clause} 后回退为 null。"
        if "null_due_to_non_daily_event_reflection" in evidence.get("constraint_notes", []):
            return f"{field_key} 的支持信号主要来自非日常高密度事件；审查 {evidence_clause} 后为防止非日常事件干扰回退为 null。"
        return f"{field_key} 在审查 {evidence_clause} 后，当前证据仍不足以形成可审计结论，因此输出 null。"
    if explicit_reason:
        return f"{explicit_reason} 关键证据: {evidence_clause}。"
    return f"{field_key} 主要依据 {evidence_clause} 得到当前结论，且没有发现足以推翻该字段的反证。"


def _format_list_for_prompt(items: List[str]) -> str:
    if not items:
        return "- 无"
    return "\n".join(f"- {item}" for item in items)


def _candidate_ref_ids(ref: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for key in ("event_id", "photo_id", "person_id", "group_id", "feature_name"):
        if ref.get(key):
            ids.append(str(ref[key]))
    for key in ("event_ids", "photo_ids", "person_ids", "group_ids", "feature_names"):
        ids.extend(str(item) for item in ref.get(key, []) or [])
    return list(dict.fromkeys(ids))


def _dedupe_refs(refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    ordered: List[Dict[str, Any]] = []
    for ref in refs:
        key = tuple(_candidate_ref_ids(ref)) or tuple(sorted(ref.items()))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(ref)
    return ordered


def _truncate_debug_text(text: Any, limit: int = 2000) -> Tuple[str, bool]:
    normalized = str(text or "")
    if len(normalized) <= limit:
        return normalized, False
    return normalized[:limit], True


def _compute_late_night_event_ratio(events: List[Any]) -> float:
    if not events:
        return 0.0
    late_night_events = 0
    for event in events:
        time_range = getattr(event, "time_range", "") or ""
        start_time = time_range.split(" - ")[0].strip()
        if start_time[:2].isdigit() and int(start_time[:2]) >= 22:
            late_night_events += 1
    return round(late_night_events / max(len(events), 1), 2)


def _compute_close_circle_size(relationships: List[Any]) -> int:
    close_types = {"romantic", "family", "bestie", "close_friend"}
    person_ids: List[str] = []
    for rel in relationships:
        rel_type = str(getattr(rel, "relationship_type", "") or "")
        confidence = float(getattr(rel, "confidence", 0.0) or 0.0)
        person_id = str(getattr(rel, "person_id", "") or "")
        if not person_id:
            continue
        if rel_type in close_types and confidence >= 0.6:
            person_ids.append(person_id)
    return len(set(person_ids))


def _infer_living_situation_from_events_and_relationships(events: List[Any], relationships: List[Any]) -> str | None:
    shared_home_rel_ids = {
        str(rel.person_id)
        for rel in relationships
        if str(getattr(rel, "relationship_type", "")) in {"romantic", "family"}
        and float(getattr(rel, "confidence", 0.0) or 0.0) >= 0.7
    }
    if not shared_home_rel_ids:
        return None

    home_event_hits = 0
    for event in events:
        location_text = " ".join(
            [
                str(getattr(event, "location", "") or ""),
                str(getattr(event, "title", "") or ""),
                str(getattr(event, "description", "") or ""),
            ]
        ).lower()
        participants = set(getattr(event, "participants", []) or [])
        if not participants.intersection(shared_home_rel_ids):
            continue
        if any(keyword in location_text for keyword in ("home", "room", "宿舍", "家", "apartment", "卧室")):
            home_event_hits += 1
    if home_event_hits >= 2:
        return "shared"
    return None
