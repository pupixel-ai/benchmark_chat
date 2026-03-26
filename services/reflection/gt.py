from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .gt_matcher import compare_profile_field_values
from .types import CaseFact


def load_profile_field_gt(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    payloads: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
    return payloads


def apply_profile_field_gt(case_facts: Iterable[CaseFact], gt_records: Iterable[Dict[str, Any]]) -> Tuple[List[CaseFact], List[Dict[str, Any]]]:
    gt_by_key = {
        _gt_lookup_key(str(record.get("album_id") or ""), str(record.get("field_key") or "")): dict(record)
        for record in gt_records
        if str(record.get("album_id") or "").strip() and str(record.get("field_key") or "").strip()
    }

    updated: List[CaseFact] = []
    comparisons: List[Dict[str, Any]] = []
    for fact in case_facts:
        if fact.signal_source != "mainline_profile" or fact.entity_type != "profile_field":
            updated.append(fact)
            continue

        gt_payload = gt_by_key.get(_gt_lookup_key(fact.album_id, fact.dimension), {})
        updated_fact, comparison_payload = _apply_single_gt(fact, gt_payload)
        updated.append(updated_fact)
        if comparison_payload:
            comparisons.append(comparison_payload)
    return updated, comparisons


def _apply_single_gt(case_fact: CaseFact, gt_payload: Dict[str, Any]) -> Tuple[CaseFact, Dict[str, Any] | None]:
    upstream_value = case_fact.upstream_output.get("value")
    pre_audit_present = bool(case_fact.pre_audit_output)
    pre_audit_value = case_fact.pre_audit_output.get("value") if pre_audit_present else None
    if upstream_value in (None, "", []):
        case_fact.badcase_source = "empty_output_candidate"
        case_fact.badcase_kind = "missing_value"
        case_fact.accuracy_gap_status = "open"
        if not gt_payload:
            return case_fact, None
        comparison_result = _build_comparison_payload(
            upstream_value=upstream_value,
            gt_value=gt_payload.get("gt_value"),
            comparison={
                "grade": "missing_prediction",
                "score": 0.0,
                "method": "rule_missing_prediction",
            },
            confidence=case_fact.auto_confidence,
        )
        pre_audit_comparison = _build_optional_pre_audit_comparison(
            pre_audit_present=pre_audit_present,
            pre_audit_value=pre_audit_value,
            field_key=case_fact.dimension,
            gt_value=gt_payload.get("gt_value"),
            confidence=case_fact.auto_confidence,
        )
        case_fact.gt_payload = dict(gt_payload)
        case_fact.comparison_result = dict(comparison_result)
        case_fact.comparison_grade = "missing_prediction"
        case_fact.comparison_score = 0.0
        case_fact.comparison_method = "rule_missing_prediction"
        _apply_pre_audit_comparison(case_fact, pre_audit_comparison)
        case_fact.causality_route = _resolve_causality_route(
            audit_action_type=case_fact.audit_action_type,
            pre_grade=case_fact.pre_audit_comparison_grade,
            post_grade=case_fact.comparison_grade,
        )
        _apply_causality_routing(case_fact)
        return case_fact, {
            "case_id": case_fact.case_id,
            "user_name": case_fact.user_name,
            "album_id": case_fact.album_id,
            "field_key": case_fact.dimension,
            "gt_payload": gt_payload,
            "comparison_result": comparison_result,
        }

    if not gt_payload:
        return case_fact, None

    comparison = compare_profile_field_values(
        field_key=case_fact.dimension,
        predicted_value=upstream_value,
        gt_value=gt_payload.get("gt_value"),
    )
    grade = str(comparison.get("grade") or "mismatch")
    score = float(comparison.get("score") or 0.0)
    method = str(comparison.get("method") or "rule_no_match")
    is_match = grade in {"exact_match", "close_match"}
    pre_audit_comparison = _build_optional_pre_audit_comparison(
        pre_audit_present=pre_audit_present,
        pre_audit_value=pre_audit_value,
        field_key=case_fact.dimension,
        gt_value=gt_payload.get("gt_value"),
        confidence=case_fact.auto_confidence,
    )
    comparison_payload = {
        "case_id": case_fact.case_id,
        "user_name": case_fact.user_name,
        "album_id": case_fact.album_id,
        "field_key": case_fact.dimension,
        "gt_payload": gt_payload,
        "comparison_result": _build_comparison_payload(
            upstream_value=upstream_value,
            gt_value=gt_payload.get("gt_value"),
            comparison=comparison,
            confidence=case_fact.auto_confidence,
        ),
    }
    case_fact.gt_payload = dict(gt_payload)
    case_fact.comparison_result = dict(comparison_payload["comparison_result"])
    case_fact.comparison_grade = grade
    case_fact.comparison_score = round(score, 4)
    case_fact.comparison_method = method
    _apply_pre_audit_comparison(case_fact, pre_audit_comparison)
    case_fact.causality_route = _resolve_causality_route(
        audit_action_type=case_fact.audit_action_type,
        pre_grade=case_fact.pre_audit_comparison_grade,
        post_grade=case_fact.comparison_grade,
    )
    if is_match:
        case_fact.badcase_source = ""
        case_fact.badcase_kind = ""
        case_fact.routing_result = "resolved_ok"
        case_fact.business_priority = "low"
        case_fact.accuracy_gap_status = "resolved"
        return case_fact, comparison_payload

    case_fact.badcase_source = "gt_mismatch_candidate"
    case_fact.badcase_kind = _resolve_badcase_kind(upstream_value=upstream_value, gt_value=gt_payload.get("gt_value"), grade=grade)
    case_fact.business_priority = "medium" if grade == "partial_match" else "high"
    case_fact.accuracy_gap_status = "open"
    _apply_causality_routing(case_fact)
    return case_fact, comparison_payload


def _gt_lookup_key(album_id: str, field_key: str) -> str:
    return f"{album_id.strip()}::{field_key.strip()}"


def _normalize_value(value: Any) -> str:
    if isinstance(value, list):
        return "|".join(sorted(_normalize_value(item) for item in value if _normalize_value(item)))
    if value is None:
        return ""
    return str(value).strip().lower()


def _mismatch_severity(confidence: float) -> str:
    return "high" if confidence >= 0.6 else "medium"


def _build_optional_pre_audit_comparison(
    *,
    pre_audit_present: bool,
    pre_audit_value: Any,
    field_key: str,
    gt_value: Any,
    confidence: float,
) -> Dict[str, Any]:
    if not pre_audit_present:
        return {}
    if pre_audit_value is None and gt_value in (None, "", []):
        return {}
    comparison = compare_profile_field_values(
        field_key=field_key,
        predicted_value=pre_audit_value,
        gt_value=gt_value,
    )
    return _build_comparison_payload(
        upstream_value=pre_audit_value,
        gt_value=gt_value,
        comparison=comparison,
        confidence=confidence,
    )


def _apply_pre_audit_comparison(case_fact: CaseFact, comparison: Dict[str, Any]) -> None:
    if not comparison:
        return
    case_fact.pre_audit_comparison_result = dict(comparison)
    case_fact.pre_audit_comparison_grade = str(comparison.get("grade") or "")
    case_fact.pre_audit_comparison_score = float(comparison.get("score") or 0.0)
    case_fact.pre_audit_comparison_method = str(comparison.get("method") or "")


def _build_comparison_payload(
    *,
    upstream_value: Any,
    gt_value: Any,
    comparison: Dict[str, Any],
    confidence: float,
) -> Dict[str, Any]:
    grade = str(comparison.get("grade") or "mismatch")
    score = float(comparison.get("score") or 0.0)
    method = str(comparison.get("method") or "rule_no_match")
    return {
        "output_value": upstream_value,
        "gt_value": gt_value,
        "normalized_output": _normalize_value(upstream_value),
        "normalized_gt": _normalize_value(gt_value),
        "is_match": grade in {"exact_match", "close_match"},
        "severity": _mismatch_severity(confidence),
        "grade": grade,
        "score": round(score, 4),
        "method": method,
    }


def _resolve_causality_route(*, audit_action_type: str, pre_grade: str, post_grade: str) -> str:
    if not audit_action_type or not pre_grade:
        return ""
    pre_rank = _comparison_rank(pre_grade)
    post_rank = _comparison_rank(post_grade)
    if _is_resolved_grade(pre_grade) and not _is_resolved_grade(post_grade):
        return "downstream_caused"
    if not _is_resolved_grade(pre_grade) and _is_resolved_grade(post_grade):
        return "downstream_helped"
    if post_rank < pre_rank:
        return "downstream_exacerbated"
    return "upstream_rooted"


def _apply_causality_routing(case_fact: CaseFact) -> None:
    if case_fact.causality_route not in {"downstream_caused", "downstream_exacerbated"}:
        return
    case_fact.routing_result = "audit_disagreement"
    case_fact.business_priority = "high"
    case_fact.resolution_route = "difficult_case"
    case_fact.triage_reason = "downstream_backflow_degraded_gt_alignment"


def _comparison_rank(grade: str) -> int:
    if grade in {"exact_match", "close_match"}:
        return 3
    if grade == "partial_match":
        return 2
    if grade in {"mismatch", "missing_prediction"}:
        return 1
    return 0


def _is_resolved_grade(grade: str) -> bool:
    return grade in {"exact_match", "close_match"}


def _resolve_badcase_kind(*, upstream_value: Any, gt_value: Any, grade: str) -> str:
    if grade == "partial_match" and isinstance(upstream_value, list) and isinstance(gt_value, list):
        upstream_tokens = set(_normalize_value(item) for item in upstream_value if _normalize_value(item))
        gt_tokens = set(_normalize_value(item) for item in gt_value if _normalize_value(item))
        if upstream_tokens < gt_tokens:
            return "underclaim"
        if gt_tokens < upstream_tokens:
            return "overclaim"
        return "wrong_granularity"
    return "wrong_value"
