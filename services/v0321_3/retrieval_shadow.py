from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence


MEMORY_UNITS_V2_SCHEMA = "memory_units_v2.v1"
MEMORY_EVIDENCE_V2_SCHEMA = "memory_evidence_v2.v1"
PROFILE_TRUTH_V1_SCHEMA = "profile_truth.v1"


def _unique(values: Iterable[Any]) -> List[Any]:
    seen = set()
    result: List[Any] = []
    for value in values:
        if value is None:
            continue
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(value)
    return result


def _clean_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text


def _compact_list(values: Sequence[Any], *, limit: int) -> List[str]:
    cleaned = [_clean_text(item) for item in values]
    return _unique(item for item in cleaned if item)[:limit]


def _build_unit_retrieval_text(event: Dict[str, Any]) -> str:
    parts: List[str] = []
    title = _clean_text(event.get("title"))
    summary = _clean_text(event.get("event_summary"))
    if title:
        parts.append(title)
    if summary and summary != title:
        parts.append(summary)
    places = _compact_list(event.get("place_refs", []) or [], limit=3)
    if places:
        parts.append(f"Places: {', '.join(places)}")
    participants = _compact_list(event.get("participant_person_ids", []) or [], limit=5)
    if participants:
        parts.append(f"Participants: {', '.join(participants)}")
    evidence_values = _compact_list(
        [
            item.get("value_or_text")
            for item in list(event.get("atomic_evidence", []) or [])[:5]
            if isinstance(item, dict)
        ],
        limit=4,
    )
    if evidence_values:
        parts.append(f"Evidence: {'; '.join(evidence_values)}")
    return "\n".join(parts).strip()


def _build_evidence_retrieval_text(
    evidence: Dict[str, Any],
    *,
    parent_event: Dict[str, Any] | None,
) -> str:
    parts: List[str] = []
    value_or_text = _clean_text(evidence.get("value_or_text"))
    evidence_type = _clean_text(evidence.get("evidence_type"))
    if evidence_type or value_or_text:
        parts.append(": ".join(part for part in [evidence_type, value_or_text] if part))
    if parent_event:
        title = _clean_text(parent_event.get("title"))
        summary = _clean_text(parent_event.get("event_summary"))
        places = _compact_list(parent_event.get("place_refs", []) or [], limit=2)
        if title:
            parts.append(f"Event: {title}")
        if summary and summary != title:
            parts.append(summary)
        if places:
            parts.append(f"Places: {', '.join(places)}")
    return "\n".join(parts).strip()


def build_memory_units_v2(
    *,
    user_id: str,
    pipeline_family: str,
    event_revisions: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    units: List[Dict[str, Any]] = []
    for event in event_revisions:
        atomic_evidence = list(event.get("atomic_evidence", []) or [])
        units.append(
            {
                "schema_version": MEMORY_UNITS_V2_SCHEMA,
                "unit_id": event.get("event_revision_id"),
                "user_id": user_id,
                "pipeline_family": pipeline_family,
                "source_type": "event_revision",
                "event_root_id": event.get("event_root_id"),
                "event_revision_id": event.get("event_revision_id"),
                "revision": event.get("revision"),
                "title": _clean_text(event.get("title")),
                "summary": _clean_text(event.get("event_summary")),
                "retrieval_text": _build_unit_retrieval_text(event),
                "display_title": _clean_text(event.get("title")),
                "display_summary": _clean_text(event.get("event_summary")),
                "started_at": event.get("started_at"),
                "ended_at": event.get("ended_at"),
                "place_refs": _compact_list(event.get("place_refs", []) or [], limit=8),
                "participant_person_ids": _compact_list(event.get("participant_person_ids", []) or [], limit=12),
                "depicted_person_ids": _compact_list(event.get("depicted_person_ids", []) or [], limit=12),
                "original_photo_ids": _compact_list(event.get("original_photo_ids", []) or [], limit=256),
                "evidence_ids": _compact_list(
                    [item.get("evidence_id") for item in atomic_evidence if isinstance(item, dict)],
                    limit=128,
                ),
                "supporting_evidence_count": len(atomic_evidence),
                "confidence": event.get("confidence"),
                "status": event.get("status"),
                "sealed_state": event.get("sealed_state"),
            }
        )
    return units


def build_memory_evidence_v2(
    *,
    user_id: str,
    pipeline_family: str,
    atomic_evidence: Sequence[Dict[str, Any]],
    event_revisions: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    event_index = {
        str(event.get("event_revision_id")): event
        for event in event_revisions
        if event.get("event_revision_id")
    }
    records: List[Dict[str, Any]] = []
    for evidence in atomic_evidence:
        parent_event_id = str(evidence.get("root_event_revision_id") or "")
        parent_event = event_index.get(parent_event_id)
        records.append(
            {
                "schema_version": MEMORY_EVIDENCE_V2_SCHEMA,
                "evidence_id": evidence.get("evidence_id"),
                "user_id": user_id,
                "pipeline_family": pipeline_family,
                "source_type": "atomic_evidence",
                "parent_unit_id": parent_event_id or None,
                "event_root_id": parent_event.get("event_root_id") if parent_event else None,
                "event_revision_id": parent_event_id or None,
                "event_title": _clean_text(parent_event.get("title")) if parent_event else "",
                "evidence_type": _clean_text(evidence.get("evidence_type")),
                "value_or_text": _clean_text(evidence.get("value_or_text")),
                "provenance": _clean_text(evidence.get("provenance")),
                "retrieval_text": _build_evidence_retrieval_text(evidence, parent_event=parent_event),
                "original_photo_ids": _compact_list(evidence.get("original_photo_ids", []) or [], limit=128),
                "participant_person_ids": _compact_list(
                    parent_event.get("participant_person_ids", []) or [],
                    limit=12,
                )
                if parent_event
                else [],
                "place_refs": _compact_list(parent_event.get("place_refs", []) or [], limit=8)
                if parent_event
                else [],
                "confidence": evidence.get("confidence"),
            }
        )
    return records


def _split_hints_by_evidence_level(items: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    strong: List[Dict[str, Any]] = []
    weak: List[Dict[str, Any]] = []
    for item in items:
        payload = dict(item)
        evidence_level = str(payload.get("evidence_level") or "")
        if evidence_level == "weak_reference":
            weak.append(payload)
        else:
            strong.append(payload)
    return {"strong": strong, "weak": weak}


def build_profile_truth_v1(
    *,
    user_id: str,
    pipeline_family: str,
    profile_revision: Dict[str, Any],
    profile_input_pack: Dict[str, Any],
    relationship_revisions: Sequence[Dict[str, Any]],
    profile_markdown: str,
) -> Dict[str, Any]:
    identity_signals = dict(profile_input_pack.get("identity_signals") or {})
    lifestyle_signals = dict(profile_input_pack.get("lifestyle_consumption_signals") or {})
    event_persona_clues = dict(profile_input_pack.get("event_persona_clues") or {})
    event_grounded_signals = dict(profile_input_pack.get("event_grounded_signals") or {})
    weak_reference_signals = dict(profile_input_pack.get("reference_media_weak_signals") or {})
    social_patterns = dict(profile_input_pack.get("social_patterns") or {})
    social_clues = list(profile_input_pack.get("social_clues") or [])
    change_points = list(profile_input_pack.get("change_points", []) or [])
    guardrails = dict(profile_input_pack.get("evidence_guardrails") or {})
    key_event_refs = list(profile_input_pack.get("key_event_refs", []) or [])
    key_relationship_refs = list(profile_input_pack.get("key_relationship_refs", []) or [])

    strong_identity: Dict[str, List[Dict[str, Any]]] = {}
    weak_identity: Dict[str, List[Dict[str, Any]]] = {}
    for key, values in identity_signals.items():
        split = _split_hints_by_evidence_level(list(values or []))
        strong_identity[key] = split["strong"]
        weak_identity[key] = split["weak"]

    strong_lifestyle: Dict[str, List[Dict[str, Any]]] = {}
    weak_lifestyle: Dict[str, List[Dict[str, Any]]] = {}
    for key, values in lifestyle_signals.items():
        split = _split_hints_by_evidence_level(list(values or []))
        strong_lifestyle[key] = split["strong"]
        weak_lifestyle[key] = split["weak"]

    relationship_truth = {
        "top_relationships": list(social_patterns.get("top_relationships", []) or []),
        "relationship_summary": dict(social_patterns.get("relationship_summary") or {}),
        "social_style_hints": dict(social_patterns.get("social_style_hints") or {}),
        "relationship_revision_ids": _unique(
            item.get("relationship_revision_id")
            for item in relationship_revisions
            if isinstance(item, dict) and item.get("relationship_revision_id")
        )[:64],
    }

    return {
        "schema_version": PROFILE_TRUTH_V1_SCHEMA,
        "profile_truth_id": f"{profile_revision.get('profile_revision_id')}:truth",
        "user_id": user_id,
        "pipeline_family": pipeline_family,
        "profile_revision_id": profile_revision.get("profile_revision_id"),
        "profile_input_pack_id": profile_input_pack.get("profile_input_pack_id"),
        "primary_person_id": profile_revision.get("primary_person_id"),
        "scope": profile_revision.get("scope"),
        "generation_mode": profile_revision.get("generation_mode"),
        "time_range": dict(profile_input_pack.get("time_range") or {}),
        "baseline_rhythm": dict(profile_input_pack.get("baseline_rhythm") or {}),
        "place_patterns": dict(profile_input_pack.get("place_patterns") or {}),
        "activity_patterns": dict(profile_input_pack.get("activity_patterns") or {}),
        "truth_layers": {
            "strong_identity": strong_identity,
            "weak_identity": weak_identity,
            "strong_lifestyle": strong_lifestyle,
            "weak_lifestyle": weak_lifestyle,
            "event_persona_clues": event_persona_clues,
            "event_grounded_signals": event_grounded_signals,
            "weak_reference_signals": weak_reference_signals,
            "relationship_truth": relationship_truth,
            "social_clues": social_clues,
        },
        "change_points": change_points,
        "key_event_refs": key_event_refs,
        "key_relationship_refs": key_relationship_refs,
        "guardrails": guardrails,
        "report_status": {
            "has_markdown_report": bool(_clean_text(profile_markdown)),
            "report_format": "markdown",
        },
        "original_photo_ids": _compact_list(profile_revision.get("original_photo_ids", []) or [], limit=512),
    }
