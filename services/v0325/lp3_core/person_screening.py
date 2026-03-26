from __future__ import annotations

from typing import Any, Dict, List

from .types import MemoryState, PersonScreening


def screen_people(state: MemoryState) -> Dict[str, PersonScreening]:
    screenings: Dict[str, PersonScreening] = {}
    appearance_stats = _collect_person_vlm_stats(state.vlm_results)

    for person_id, person_info in state.face_db.items():
        photo_count = _get_attr(person_info, "photo_count") or 0
        stats = appearance_stats.get(person_id, {})
        refs = stats.get("refs", [])
        blocked = []
        person_kind = "real_person"
        memory_value = "candidate"

        if stats.get("mediated_ratio", 0.0) >= 0.8:
            person_kind = "mediated_person"
            memory_value = "block"
            blocked.append("mostly_screen_or_poster")
        elif stats.get("service_ratio", 0.0) >= 0.7:
            person_kind = "service_person"
            memory_value = "block"
            blocked.append("mostly_service_context")
        elif photo_count <= 1 and stats.get("group_only_ratio", 0.0) >= 0.8:
            person_kind = "incidental_person"
            memory_value = "low_value"
        elif photo_count >= 8:
            memory_value = "core"

        screenings[person_id] = PersonScreening(
            person_id=person_id,
            person_kind=person_kind,
            memory_value=memory_value,
            screening_refs=refs,
            block_reasons=blocked,
        )

    return screenings


def _collect_person_vlm_stats(vlm_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for item in vlm_results:
        photo_id = item.get("photo_id")
        summary = str(item.get("vlm_analysis", {}).get("summary", "") or "").lower()
        event_data = item.get("vlm_analysis", {}).get("event", {}) or {}
        social_context = str(event_data.get("social_context", "") or "").lower() if isinstance(event_data, dict) else ""
        scene = item.get("vlm_analysis", {}).get("scene", {}) or {}
        location = str(scene.get("location_detected", "") or "").lower() if isinstance(scene, dict) else ""
        for person in item.get("vlm_analysis", {}).get("people", []) or []:
            person_id = person.get("person_id")
            if not person_id:
                continue
            bucket = stats.setdefault(person_id, {"total": 0, "mediated": 0, "service": 0, "group_only": 0, "refs": []})
            bucket["total"] += 1
            if any(keyword in summary for keyword in ("poster", "screenshot", "screen", "tv", "电视", "海报")):
                bucket["mediated"] += 1
            if any(keyword in social_context for keyword in ("staff", "cashier", "waiter", "服务员", "店员")):
                bucket["service"] += 1
            if any(keyword in location for keyword in ("stadium", "concert", "广场", "street")) and len(item.get("vlm_analysis", {}).get("people", []) or []) >= 4:
                bucket["group_only"] += 1
            bucket["refs"].append({"photo_id": photo_id, "signal": summary[:120], "why": "person_screening"})

    for bucket in stats.values():
        total = max(bucket["total"], 1)
        bucket["mediated_ratio"] = round(bucket["mediated"] / total, 2)
        bucket["service_ratio"] = round(bucket["service"] / total, 2)
        bucket["group_only_ratio"] = round(bucket["group_only"] / total, 2)
    return stats


def _get_attr(person_info: Any, key: str) -> Any:
    if isinstance(person_info, dict):
        return person_info.get(key)
    return getattr(person_info, key, None)
