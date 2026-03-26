from __future__ import annotations

import copy
from typing import Any, Dict


def _dedupe_strings(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def compact_evidence(evidence: Any) -> Dict[str, Any]:
    if not isinstance(evidence, dict):
        return {
            "photo_ids": [],
            "event_ids": [],
            "person_ids": [],
            "group_ids": [],
            "feature_names": [],
            "supporting_ref_count": 0,
            "contradicting_ref_count": 0,
            "constraint_notes": [],
            "summary": "",
        }
    supporting_refs = evidence.get("supporting_refs")
    contradicting_refs = evidence.get("contradicting_refs")
    return {
        "photo_ids": _dedupe_strings(evidence.get("photo_ids")),
        "event_ids": _dedupe_strings(evidence.get("event_ids")),
        "person_ids": _dedupe_strings(evidence.get("person_ids")),
        "group_ids": _dedupe_strings(evidence.get("group_ids")),
        "feature_names": _dedupe_strings(evidence.get("feature_names")),
        "supporting_ref_count": int(
            evidence.get("supporting_ref_count")
            if evidence.get("supporting_ref_count") is not None
            else len(supporting_refs) if isinstance(supporting_refs, list) else 0
        ),
        "contradicting_ref_count": int(
            evidence.get("contradicting_ref_count")
            if evidence.get("contradicting_ref_count") is not None
            else len(contradicting_refs) if isinstance(contradicting_refs, list) else 0
        ),
        "constraint_notes": copy.deepcopy(list(evidence.get("constraint_notes") or [])),
        "summary": str(evidence.get("summary") or ""),
    }


def _looks_like_tag_object(node: Any) -> bool:
    if not isinstance(node, dict):
        return False
    keys = set(node.keys())
    return "value" in keys and "confidence" in keys and bool(keys & {"evidence", "reasoning"})


def compact_structured_profile(node: Any) -> Any:
    if _looks_like_tag_object(node):
        return {
            "value": copy.deepcopy(node.get("value")),
            "confidence": node.get("confidence"),
            "evidence": compact_evidence(node.get("evidence")),
            "reasoning": str(node.get("reasoning") or ""),
        }
    if isinstance(node, dict):
        return {key: compact_structured_profile(value) for key, value in node.items()}
    if isinstance(node, list):
        return [compact_structured_profile(value) for value in node]
    return copy.deepcopy(node)


def compact_lp3_profile(lp3_profile: Any) -> Dict[str, Any]:
    if not isinstance(lp3_profile, dict):
        return {}
    payload = {
        "structured": compact_structured_profile(lp3_profile.get("structured") or {}),
        "summary": copy.deepcopy(lp3_profile.get("summary")),
        "consistency": copy.deepcopy(lp3_profile.get("consistency") or {}),
        "report_markdown": str(lp3_profile.get("report_markdown") or ""),
    }
    if "field_decisions" in lp3_profile:
        payload["field_decisions"] = copy.deepcopy(lp3_profile.get("field_decisions") or [])
    return payload
