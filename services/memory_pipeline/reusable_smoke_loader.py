from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List

from models import Event

from .orchestrator import build_memory_state
from .types import MemoryState


@dataclass
class ReusableSmokeLoadResult:
    case_path: Path
    state: MemoryState
    fallback_primary_person_id: str
    runtime_input_paths: Dict[str, Path]
    reference_relationships: List[Dict[str, Any]]
    reference_profile_path: Path | None
    mapping_debug: Dict[str, Any]


def load_reusable_smoke_case(case_dir: str | Path) -> ReusableSmokeLoadResult:
    case_path = Path(case_dir)
    if not case_path.exists():
        raise FileNotFoundError(f"case_dir 不存在: {case_path}")

    face_path = _resolve_required_file(case_path, "*_face_recognition_output.json")
    vlm_path = _resolve_required_file(case_path, "*_vlm_cache.json")
    events_path = _resolve_required_file(case_path, "*_events.json")

    reference_profile_path = _resolve_optional_file(case_path, "*_profile_structured.json")
    reference_state_path = _resolve_optional_file(case_path, "*_face_recognition_state.json")

    face_payload = _load_json_file(face_path)
    vlm_payload = _load_json_file(vlm_path)
    events_payload = _load_json_file(events_path)

    fallback_primary_person_id = _extract_required_primary_person_id(face_payload, face_path)
    face_db = _load_face_db(face_payload, face_path)
    vlm_results = _load_vlm_results(vlm_payload, vlm_path)
    events = _load_events(events_payload, events_path)

    state = build_memory_state(
        photos=[],
        face_db=face_db,
        vlm_results=vlm_results,
    )
    state.events = events
    state.relationships = []

    ignored_reference_inputs: List[Dict[str, Any]] = []
    if reference_profile_path:
        ignored_reference_inputs.append(
            {
                "kind": "profile_structured",
                "path": str(reference_profile_path),
                "reason": "reference_only_not_fed",
            }
        )
    if reference_state_path:
        ignored_reference_inputs.append(
            {
                "kind": "face_recognition_state",
                "path": str(reference_state_path),
                "reason": "reference_only_not_fed",
            }
        )

    ignored_embedded_payloads: List[Dict[str, Any]] = []
    reference_relationships = list(events_payload.get("relationships") or [])
    if reference_relationships:
        ignored_embedded_payloads.append(
            {
                "source": str(events_path),
                "key": "relationships",
                "count": len(reference_relationships),
                "reason": "reference_only_not_fed",
            }
        )
    embedded_face_db = events_payload.get("face_db") or {}
    if embedded_face_db:
        ignored_embedded_payloads.append(
            {
                "source": str(events_path),
                "key": "face_db",
                "count": len(embedded_face_db) if isinstance(embedded_face_db, dict) else 1,
                "reason": "reference_only_not_fed",
            }
        )

    mapping_debug = {
        "case_dir": str(case_path),
        "runtime_inputs": {
            "face_output_path": str(face_path),
            "vlm_cache_path": str(vlm_path),
            "events_path": str(events_path),
        },
        "fallback_primary_person_id": fallback_primary_person_id,
        "ignored_reference_inputs": ignored_reference_inputs,
        "ignored_embedded_payloads": ignored_embedded_payloads,
        "state_counts": {
            "face_db_count": len(face_db),
            "vlm_result_count": len(vlm_results),
            "event_count": len(events),
            "input_relationship_count": 0,
        },
    }

    return ReusableSmokeLoadResult(
        case_path=case_path,
        state=state,
        fallback_primary_person_id=fallback_primary_person_id,
        runtime_input_paths={
            "face": face_path,
            "vlm": vlm_path,
            "events": events_path,
        },
        reference_relationships=reference_relationships,
        reference_profile_path=reference_profile_path,
        mapping_debug=mapping_debug,
    )


def _resolve_required_file(case_path: Path, pattern: str) -> Path:
    matches = sorted(case_path.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"缺少必需输入文件，未匹配到: {pattern}")
    if len(matches) > 1:
        raise ValueError(f"输入文件匹配不唯一: {pattern} -> {matches}")
    return matches[0]


def _resolve_optional_file(case_path: Path, pattern: str) -> Path | None:
    matches = sorted(case_path.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"可选 reference 文件匹配不唯一: {pattern} -> {matches}")
    return matches[0]


def _load_json_file(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"仅支持 dict 顶层 payload: {path}")
    return payload


def _extract_required_primary_person_id(face_payload: Dict[str, Any], face_path: Path) -> str:
    primary_person_id = face_payload.get("primary_person_id")
    if not isinstance(primary_person_id, str) or not primary_person_id.strip():
        raise ValueError(f"face 输出缺少 primary_person_id: {face_path}")
    return primary_person_id.strip()


def _load_face_db(face_payload: Dict[str, Any], face_path: Path) -> Dict[str, Dict[str, Any]]:
    persons = face_payload.get("persons")
    if not isinstance(persons, list) or not persons:
        raise ValueError(f"face 输出缺少 persons[]: {face_path}")

    face_db: Dict[str, Dict[str, Any]] = {}
    for person in persons:
        if not isinstance(person, dict):
            raise ValueError(f"face person 条目必须是对象: {face_path}")
        person_id = str(person.get("person_id") or "").strip()
        if not person_id:
            raise ValueError(f"face person 缺少 person_id: {face_path}")
        face_db[person_id] = {
            "photo_count": int(person.get("photo_count", 0) or 0),
            "first_seen": _parse_datetime(person.get("first_seen")),
            "last_seen": _parse_datetime(person.get("last_seen")),
            "avg_confidence": float(person.get("avg_score", 0.0) or 0.0),
            "avg_quality": float(person.get("avg_quality", 0.0) or 0.0),
            "name": str(person.get("label", "") or ""),
        }
    return face_db


def _load_vlm_results(vlm_payload: Dict[str, Any], vlm_path: Path) -> List[Dict[str, Any]]:
    photos = vlm_payload.get("photos")
    if not isinstance(photos, list) or not photos:
        raise ValueError(f"vlm cache 缺少 photos[]: {vlm_path}")

    vlm_results: List[Dict[str, Any]] = []
    for photo in photos:
        if not isinstance(photo, dict):
            raise ValueError(f"vlm photo 条目必须是对象: {vlm_path}")
        photo_id = str(photo.get("photo_id") or "").strip()
        timestamp = str(photo.get("timestamp") or "").strip()
        analysis = photo.get("vlm_analysis")
        if not photo_id or not timestamp or not isinstance(analysis, dict):
            raise ValueError(f"vlm photo 缺少 photo_id/timestamp/vlm_analysis: {vlm_path}")
        vlm_results.append(
            {
                "photo_id": photo_id,
                "timestamp": timestamp,
                "filename": str(photo.get("filename", "") or ""),
                "location": photo.get("location"),
                "face_person_ids": list(photo.get("face_person_ids", []) or []),
                "media_kind": photo.get("media_kind"),
                "is_reference_like": photo.get("is_reference_like"),
                "sequence_index": photo.get("sequence_index"),
                "vlm_analysis": {
                    "summary": str(analysis.get("summary", "") or ""),
                    "people": list(analysis.get("people", []) or []),
                    "relations": list(analysis.get("relations", []) or []),
                    "scene": dict(analysis.get("scene", {}) or {}),
                    "event": dict(analysis.get("event", {}) or {}),
                    "details": _normalize_list_like(analysis.get("details")),
                    "ocr_hits": list(analysis.get("ocr_hits", []) or []),
                    "brands": list(analysis.get("brands", []) or []),
                    "place_candidates": list(analysis.get("place_candidates", []) or []),
                },
            }
        )
    return vlm_results


def _load_events(events_payload: Dict[str, Any], events_path: Path) -> List[Event]:
    raw_events = events_payload.get("events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError(f"events 文件缺少 events[]: {events_path}")

    events: List[Event] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            raise ValueError(f"event 条目必须是对象: {events_path}")
        event_id = str(raw_event.get("event_id") or "").strip()
        participants = [str(item).strip() for item in list(raw_event.get("participants", []) or []) if str(item).strip()]
        if not event_id:
            raise ValueError(f"event 缺少 event_id: {events_path}")
        events.append(
            Event(
                event_id=event_id,
                date=str(raw_event.get("date", "") or ""),
                time_range=str(raw_event.get("time_range", "") or ""),
                duration=str(raw_event.get("duration", "") or ""),
                title=str(raw_event.get("title", "") or ""),
                type=str(raw_event.get("type", "其他") or "其他"),
                participants=participants,
                location=str(raw_event.get("location", "") or ""),
                description=str(raw_event.get("description", "") or ""),
                photo_count=int(raw_event.get("photo_count", 0) or 0),
                confidence=float(raw_event.get("confidence", 0.0) or 0.0),
                reason=str(raw_event.get("reason", "") or ""),
                narrative=str(raw_event.get("narrative", "") or ""),
                narrative_synthesis=str(raw_event.get("narrative_synthesis", raw_event.get("narrative", "")) or ""),
                meta_info={
                    "source": "reusable_smoke_events",
                    "runtime_input_whitelist": ["face_recognition_output", "vlm_cache", "events.events"],
                },
                objective_fact={
                    "participants": participants,
                    "place_refs": [str(raw_event.get("location", "") or "")] if raw_event.get("location") else [],
                },
                social_interaction=dict(raw_event.get("social_interaction", {}) or {}),
                social_dynamics=list(raw_event.get("social_dynamics", []) or []),
                lifestyle_tags=list(raw_event.get("lifestyle_tags", []) or []),
                tags=list(raw_event.get("tags", []) or []),
                social_slices=list(raw_event.get("social_slices", []) or []),
                persona_evidence=dict(raw_event.get("persona_evidence", {}) or {}),
            )
        )
    return events


def _normalize_list_like(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None
