from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, Iterable, List

from config import PROFILE_LLM_PROVIDER
from services.memory_pipeline.profile_llm import OpenRouterProfileLLMProcessor

from .types import CaseFact


ALLOWED_ROOT_CAUSE_FAMILIES = (
    "field_reasoning",
    "evidence_packaging",
    "tool_retrieval",
    "tool_selection_policy",
    "orchestration_guardrail",
    "engineering_issue",
    "watch_only",
    "coverage_gap_source",
    "coverage_gap_tool",
)

ROOT_CAUSE_TO_FIX_SURFACE = {
    "field_reasoning": "field_cot",
    "evidence_packaging": "field_cot",
    "tool_retrieval": "tool_rule",
    "tool_selection_policy": "call_policy",
    "orchestration_guardrail": "engineering_issue",
    "engineering_issue": "engineering_issue",
    "watch_only": "watch_only",
    "coverage_gap_source": "call_policy",
    "coverage_gap_tool": "tool_rule",
}


class UpstreamTriageScorer:
    def __init__(self, llm_processor: Any | None = None) -> None:
        self.llm_processor = llm_processor or _build_llm_processor()

    def enrich_case(self, case_fact: CaseFact, similar_patterns: Iterable[Dict[str, Any]] | None = None) -> CaseFact:
        if case_fact.signal_source != "mainline_profile":
            return case_fact
        if case_fact.routing_result == "resolved_ok" or case_fact.accuracy_gap_status == "resolved":
            case_fact.resolution_route = ""
            return case_fact
        if case_fact.causality_route in {"downstream_caused", "downstream_exacerbated"}:
            case_fact.routing_result = "audit_disagreement"
            case_fact.triage_reason = "downstream_backflow_degraded_gt_alignment"
            case_fact.root_cause_family = "watch_only"
            case_fact.fix_surface_confidence = 0.0
            case_fact.business_priority = "high"
            case_fact.resolution_route = "difficult_case"
            case_fact.accuracy_gap_status = case_fact.accuracy_gap_status or "open"
            return case_fact

        coverage_gap = dict((case_fact.tool_usage_summary or {}).get("coverage_gap") or {})
        if coverage_gap.get("has_gap"):
            gap_type = str(coverage_gap.get("gap_type") or "")
            if gap_type == "source_unconfigured":
                chosen = {"root_cause_family": "coverage_gap_source", "fix_surface_confidence": 0.92,
                          "recommended_fix_surface": "call_policy"}
            elif gap_type in {"tool_called_no_hit", "tool_rule_blocked", "index_path_suspect"}:
                chosen = {"root_cause_family": "coverage_gap_tool", "fix_surface_confidence": 0.90,
                          "recommended_fix_surface": "tool_rule"}
            else:
                chosen = None
            if chosen is not None:
                case_fact.root_cause_family = chosen["root_cause_family"]
                case_fact.fix_surface_confidence = float(chosen["fix_surface_confidence"])
                case_fact.tool_usage_summary = {
                    **dict(case_fact.tool_usage_summary or {}),
                    "recommended_fix_surface": chosen["recommended_fix_surface"],
                }
                case_fact.accuracy_gap_status = case_fact.accuracy_gap_status or ("open" if case_fact.badcase_source else "")
                case_fact.resolution_route = resolve_accuracy_gap_route(case_fact)
                return case_fact

        features = extract_upstream_triage_features(case_fact, similar_patterns or [])
        llm_result = self._score_with_llm(case_fact, features)
        heuristic_result = _heuristic_score(case_fact, features)
        chosen = _merge_scoring(heuristic_result, llm_result)

        case_fact.root_cause_family = chosen["root_cause_family"]
        case_fact.fix_surface_confidence = float(chosen["fix_surface_confidence"])
        case_fact.tool_usage_summary = {
            **dict(case_fact.tool_usage_summary or {}),
            **features,
            "recommended_fix_surface": chosen["recommended_fix_surface"],
        }
        case_fact.accuracy_gap_status = case_fact.accuracy_gap_status or ("open" if case_fact.badcase_source else "")
        if chosen["root_cause_family"] == "engineering_issue":
            case_fact.routing_result = "engineering_issue"
            case_fact.business_priority = "high"
            case_fact.resolution_route = "engineering_fix"
            return case_fact
        case_fact.resolution_route = resolve_accuracy_gap_route(case_fact)
        if case_fact.resolution_route == "difficult_case":
            case_fact.business_priority = "medium" if case_fact.comparison_grade == "partial_match" else case_fact.business_priority
        return case_fact

    def _score_with_llm(self, case_fact: CaseFact, features: Dict[str, Any]) -> Dict[str, Any]:
        if self.llm_processor is None:
            return {}

        prompt = (
            "你是上游反思 triage scorer。"
            "只能从这些 root_cause_family 中选择一个："
            + ", ".join(ALLOWED_ROOT_CAUSE_FAMILIES)
            + "。\n"
            "只返回 JSON，字段为 root_cause_family, fix_surface_confidence, reason。\n"
            "若无法稳定判断，输出 watch_only。\n"
            + json.dumps(
                {
                    "case_id": case_fact.case_id,
                    "dimension": case_fact.dimension,
                    "badcase_source": case_fact.badcase_source,
                    "badcase_kind": case_fact.badcase_kind,
                    "decision_trace": case_fact.decision_trace,
                    "tool_usage_summary": case_fact.tool_usage_summary,
                    "features": features,
                    "comparison_result": case_fact.comparison_result,
                    "gt_payload": case_fact.gt_payload,
                },
                ensure_ascii=False,
            )
        )
        try:
            response = self.llm_processor._call_llm_via_official_api(prompt, response_mime_type="application/json")
        except Exception:
            return {}
        if not isinstance(response, dict):
            return {}
        root_cause_family = str(response.get("root_cause_family") or "").strip()
        if root_cause_family not in ALLOWED_ROOT_CAUSE_FAMILIES:
            return {}
        confidence = _safe_float(response.get("fix_surface_confidence"), default=0.0)
        return {
            "root_cause_family": root_cause_family,
            "fix_surface_confidence": confidence,
            "llm_reason": str(response.get("reason") or "").strip(),
        }


def extract_upstream_triage_features(case_fact: CaseFact, similar_patterns: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    tool_usage = dict(case_fact.tool_usage_summary or {})
    evidence_count = len(list(case_fact.evidence_refs or []))
    retrieval_hit_count = int(tool_usage.get("retrieval_hit_count") or evidence_count)
    similar_patterns_list = list(similar_patterns or [])
    return {
        "evidence_count": evidence_count,
        "tool_called": bool(tool_usage.get("tool_called") or tool_usage.get("tool_trace_present") or retrieval_hit_count > 0),
        "retrieval_hit_count": retrieval_hit_count,
        "support_count": max(case_fact.support_count, evidence_count),
        "confidence": case_fact.auto_confidence,
        "gt_mismatch_severity": str((case_fact.comparison_result or {}).get("severity") or ""),
        "history_recurrence": len(similar_patterns_list),
        "has_gt": bool(case_fact.gt_payload),
    }


def retrieve_similar_patterns(case_fact: CaseFact, existing_patterns: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked: List[tuple[int, Dict[str, Any]]] = []
    for payload in existing_patterns:
        score = 0
        if str(payload.get("dimension") or "") == case_fact.dimension:
            score += 3
        if case_fact.root_cause_family and case_fact.root_cause_family in list(payload.get("root_cause_candidates") or []):
            score += 2
        if str(payload.get("triage_reason") or "") == case_fact.triage_reason:
            score += 1
        if score > 0:
            ranked.append((score, payload))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [payload for _, payload in ranked[:5]]


def recommend_fix_surface(pattern_facts: Iterable[CaseFact]) -> Dict[str, Any]:
    facts = list(pattern_facts)
    if not facts:
        return {
            "recommended_option": "watch_only",
            "fix_surface_confidence": 0.0,
            "is_direction_clear": False,
        }

    counter = Counter(_resolved_root_cause(fact) for fact in facts)
    best_root_cause, best_count = counter.most_common(1)[0]
    next_count = counter.most_common(2)[1][1] if len(counter) > 1 else 0
    avg_confidence = sum(float(fact.fix_surface_confidence or 0.0) for fact in facts) / len(facts)
    recommended_option = ROOT_CAUSE_TO_FIX_SURFACE.get(best_root_cause, "watch_only")
    is_direction_clear = (
        recommended_option not in {"engineering_issue", "watch_only"}
        and avg_confidence >= 0.8
        and best_count > next_count
    )
    return {
        "recommended_option": recommended_option,
        "fix_surface_confidence": round(avg_confidence, 4),
        "is_direction_clear": is_direction_clear,
        "root_cause_family": best_root_cause,
    }


def _heuristic_score(case_fact: CaseFact, features: Dict[str, Any]) -> Dict[str, Any]:
    tool_called = bool(features["tool_called"])
    retrieval_hit_count = int(features["retrieval_hit_count"])
    evidence_count = int(features["evidence_count"])
    history_recurrence = int(features["history_recurrence"])
    gt_mismatch_severity = str(features["gt_mismatch_severity"] or "")

    if case_fact.routing_result == "engineering_issue":
        return _score_payload("engineering_issue", 1.0)

    if case_fact.badcase_source == "gt_mismatch_candidate":
        if tool_called and evidence_count >= 2:
            return _score_payload("field_reasoning", 0.9)
        if tool_called and retrieval_hit_count == 0:
            return _score_payload("tool_retrieval", 0.84)
        if not tool_called:
            return _score_payload("tool_selection_policy", 0.85)
        return _score_payload("evidence_packaging", 0.78)

    if case_fact.badcase_source == "empty_output_candidate":
        if tool_called and evidence_count >= 1:
            return _score_payload("field_reasoning", 0.82)
        if tool_called and retrieval_hit_count == 0:
            return _score_payload("tool_retrieval", 0.86)
        if not tool_called:
            return _score_payload("tool_selection_policy", 0.88)

    if history_recurrence >= 2 and evidence_count >= 2:
        return _score_payload("evidence_packaging", 0.76)

    if gt_mismatch_severity == "high":
        return _score_payload("field_reasoning", 0.8)

    return _score_payload("watch_only", 0.45)


def _merge_scoring(heuristic_result: Dict[str, Any], llm_result: Dict[str, Any]) -> Dict[str, Any]:
    if not llm_result:
        return {
            **heuristic_result,
            "recommended_fix_surface": ROOT_CAUSE_TO_FIX_SURFACE.get(heuristic_result["root_cause_family"], "watch_only"),
        }
    heuristic_confidence = float(heuristic_result.get("fix_surface_confidence") or 0.0)
    llm_confidence = float(llm_result.get("fix_surface_confidence") or 0.0)
    chosen = llm_result if llm_confidence >= heuristic_confidence else heuristic_result
    return {
        **chosen,
        "recommended_fix_surface": ROOT_CAUSE_TO_FIX_SURFACE.get(chosen["root_cause_family"], "watch_only"),
    }


def resolve_accuracy_gap_route(case_fact: CaseFact) -> str:
    if case_fact.routing_result == "engineering_issue" or case_fact.root_cause_family == "engineering_issue":
        return "engineering_fix"
    if case_fact.accuracy_gap_status != "open" and not case_fact.badcase_source:
        return ""
    if case_fact.comparison_grade == "partial_match":
        return "difficult_case"
    if case_fact.root_cause_family == "watch_only":
        return "difficult_case"
    if case_fact.root_cause_family in {
        "field_reasoning",
        "evidence_packaging",
        "tool_retrieval",
        "tool_selection_policy",
    } and float(case_fact.fix_surface_confidence or 0.0) >= 0.75:
        return "strategy_fix"
    return "difficult_case"


def _score_payload(root_cause_family: str, fix_surface_confidence: float) -> Dict[str, Any]:
    return {
        "root_cause_family": root_cause_family,
        "fix_surface_confidence": round(fix_surface_confidence, 4),
    }


def _resolved_root_cause(case_fact: CaseFact) -> str:
    if case_fact.root_cause_family:
        return case_fact.root_cause_family
    if case_fact.entity_type == "profile_field":
        return "field_reasoning"
    if case_fact.entity_type in {"primary_person", "relationship_candidate"}:
        return "tool_retrieval"
    return "watch_only"


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_llm_processor() -> Any | None:
    if PROFILE_LLM_PROVIDER != "openrouter":
        return None
    try:
        return OpenRouterProfileLLMProcessor()
    except Exception:
        return None
