from __future__ import annotations

from typing import Any, Dict, Iterable, List


def build_evidence_payload(
    *,
    photo_ids: Iterable[str] | None = None,
    event_ids: Iterable[str] | None = None,
    person_ids: Iterable[str] | None = None,
    group_ids: Iterable[str] | None = None,
    feature_names: Iterable[str] | None = None,
    supporting_refs: Iterable[Dict[str, Any]] | None = None,
    contradicting_refs: Iterable[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    return {
        "photo_ids": _dedupe_strs(photo_ids),
        "event_ids": _dedupe_strs(event_ids),
        "person_ids": _dedupe_strs(person_ids),
        "group_ids": _dedupe_strs(group_ids),
        "feature_names": _dedupe_strs(feature_names),
        "supporting_refs": list(supporting_refs or []),
        "contradicting_refs": list(contradicting_refs or []),
    }


def flatten_ref_buckets(ref_buckets: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for refs in ref_buckets.values():
        flat.extend(refs)
    return flat


def extract_ids_from_refs(refs: Iterable[Dict[str, Any]]) -> Dict[str, List[str]]:
    photo_ids: List[str] = []
    event_ids: List[str] = []
    person_ids: List[str] = []
    group_ids: List[str] = []
    feature_names: List[str] = []

    for ref in refs:
        source_type = str(ref.get("source_type", "") or "")
        source_id = ref.get("source_id")
        if source_type == "photo" and source_id:
            photo_ids.append(str(source_id))
        elif source_type == "event" and source_id:
            event_ids.append(str(source_id))
        elif source_type == "person" and source_id:
            person_ids.append(str(source_id))
        elif source_type == "group" and source_id:
            group_ids.append(str(source_id))
        elif source_type == "feature" and source_id:
            feature_names.append(str(source_id))

        if ref.get("photo_id"):
            photo_ids.append(str(ref["photo_id"]))
        for item in ref.get("photo_ids", []) or []:
            photo_ids.append(str(item))
        if ref.get("event_id"):
            event_ids.append(str(ref["event_id"]))
        for item in ref.get("event_ids", []) or []:
            event_ids.append(str(item))
        if ref.get("person_id"):
            person_ids.append(str(ref["person_id"]))
        for item in ref.get("person_ids", []) or []:
            person_ids.append(str(item))
        if ref.get("group_id"):
            group_ids.append(str(ref["group_id"]))
        for item in ref.get("group_ids", []) or []:
            group_ids.append(str(item))
        if ref.get("feature_name"):
            feature_names.append(str(ref["feature_name"]))
        for item in ref.get("feature_names", []) or []:
            feature_names.append(str(item))

    return {
        "photo_ids": _dedupe_strs(photo_ids),
        "event_ids": _dedupe_strs(event_ids),
        "person_ids": _dedupe_strs(person_ids),
        "group_ids": _dedupe_strs(group_ids),
        "feature_names": _dedupe_strs(feature_names),
    }


def _dedupe_strs(values: Iterable[str] | None) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values or []:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered
