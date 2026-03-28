from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from models import Event

from .orchestrator import build_memory_state
from .types import MemoryState

PRIMARY_ALIAS_COMPACTS = {
    "主角",
    "主角本人",
    "主角person_id",
    "主角personid",
    "用户",
    "用户本人",
    "用户person_id",
    "用户personid",
    "primary_person_id",
    "primarypersonid",
}
CANONICAL_PERSON_ID_PATTERN = re.compile(r"Person_\d{3}")


def load_precomputed_memory_state(base_dir: str | Path) -> MemoryState:
    base_path = Path(base_dir)
    snapshot_primary_person_id = _load_snapshot_primary_person_id(base_path)
    if _looks_like_bundle_layout(base_path):
        face_file = base_path / "face" / "face_recognition_output.json"
        vlm_file = base_path / "vlm" / "vp1_observations.json"
        event_file = base_path / "lp1" / "lp1_events_compact.json"
    else:
        # 灵活地加载文件（支持通配符）
        face_file = _resolve_file(
            base_path,
            [
                "face_recognition.json",
                "face_recognition_output.json",
                "*_face_recognition_output.json",
                "*_face_recognition.json",
                "face_db.json",
            ],
        )
        vlm_file = _resolve_file_optional(
            base_path,
            [
                "vlm_results.json",
                "vlm_cache.json",
                "vp1_observations.json",
                "*_vlm_cache.json",
                "*_vlm_results.json",
            ],
        )
        event_file = _resolve_file(
            base_path,
            [
                "events.json",
                "final_events.json",
                "lp1_events.json",
                "lp1_events_compact.json",
                "*_events.json",
                "*_events_compact.json",
            ],
        )

    face_payload = json.loads(face_file.read_text(encoding="utf-8"))
    vlm_payload = json.loads(vlm_file.read_text(encoding="utf-8")) if vlm_file else []
    event_payload = json.loads(event_file.read_text(encoding="utf-8"))
    primary_person_id = (
        face_payload.get("primary_person_id")
        or face_payload.get("face_recognition", {}).get("primary_person_id")
        or face_payload.get("metadata", {}).get("primary_person_id")
        or snapshot_primary_person_id
    )
    if _looks_like_bundle_layout(base_path) and not primary_person_id:
        primary_person_id = "主角"

    state = build_memory_state(
        photos=[],
        face_db=_load_face_db(face_payload),
        vlm_results=_load_vlm_results(vlm_payload, primary_person_id=primary_person_id),
    )

    # Load events first
    state.events = _load_events(
        event_payload.get("events", []) if isinstance(event_payload, dict) else event_payload,
        primary_person_id=primary_person_id,
    )

    if primary_person_id:
        state.primary_decision = {
            "mode": "person_id",
            "primary_person_id": primary_person_id,
            "confidence": 0.9,
            "evidence": {
                "photo_ids": [],
                "event_ids": [],
                "person_ids": [primary_person_id],
                "group_ids": [],
                "feature_names": ["precomputed_face_primary_person"],
                "supporting_refs": [],
                "contradicting_refs": [],
            },
            "reasoning": "预计算人脸链路已经稳定识别出该主角，因此直接复用。",
        }
    state.relationships = _load_relationships(event_payload.get("relationships", []) if isinstance(event_payload, dict) else [])
    return state


def _looks_like_bundle_layout(base_path: Path) -> bool:
    return (
        (base_path / "face" / "face_recognition_output.json").exists()
        and (base_path / "vlm" / "vp1_observations.json").exists()
        and (base_path / "lp1" / "lp1_events_compact.json").exists()
    )


def _resolve_file(base_path: Path, patterns: List[str]) -> Path:
    """查找并返回匹配模式的第一个文件"""
    for pattern in patterns:
        if "*" in pattern:
            matches = list(base_path.glob(pattern))
            if matches:
                return matches[0]
        else:
            filepath = base_path / pattern
            if filepath.exists():
                return filepath
    raise FileNotFoundError(f"无法找到文件，尝试的模式: {patterns}")


def _resolve_file_optional(base_path: Path, patterns: List[str]) -> Path | None:
    try:
        return _resolve_file(base_path, patterns)
    except FileNotFoundError:
        return None


def _load_snapshot_primary_person_id(base_path: Path) -> str | None:
    for snapshot_file in ("internal_artifacts.json", "normalized_state_snapshot.json"):
        file_path = base_path / snapshot_file
        if not file_path.exists():
            continue
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        primary_person_id = (
            payload.get("primary_decision", {}).get("primary_person_id")
            if isinstance(payload, dict)
            else None
        )
        if primary_person_id:
            return primary_person_id
    return None


def _load_vlm_results(
    vlm_payload: Dict[str, Any] | List[Dict[str, Any]],
    *,
    primary_person_id: str | None,
) -> List[Dict[str, Any]]:
    """Load VLM analysis results from cache payload"""
    if isinstance(vlm_payload, list):
        photos = vlm_payload
    else:
        photos = vlm_payload.get("photos") or vlm_payload.get("vlm_results") or []
    vlm_results = []
    for photo in photos:
        analysis = dict(photo.get("vlm_analysis", {}) or {})
        normalized_people = []
        for person in analysis.get("people", []) or []:
            if not isinstance(person, dict):
                continue
            normalized_person = dict(person)
            person_id = _extract_canonical_person_id(normalized_person.get("person_id"), primary_person_id)
            if person_id:
                normalized_person["person_id"] = person_id
            normalized_people.append(normalized_person)
        normalized_relations = []
        for relation in analysis.get("relations", []) or []:
            if not isinstance(relation, dict):
                continue
            normalized_relation = dict(relation)
            normalized_relation["subject"] = _normalize_primary_person_reference(
                normalized_relation.get("subject"),
                primary_person_id,
            )
            normalized_relation["object"] = _normalize_primary_person_reference(
                normalized_relation.get("object"),
                primary_person_id,
            )
            normalized_relations.append(normalized_relation)
        analysis["people"] = normalized_people
        analysis["relations"] = normalized_relations
        result = {
            "photo_id": photo.get("photo_id"),
            "timestamp": photo.get("timestamp"),
            "filename": photo.get("filename"),
            "location": photo.get("location"),
            "face_person_ids": _normalize_face_person_ids(photo.get("face_person_ids", []) or []),
            "media_kind": photo.get("media_kind"),
            "is_reference_like": photo.get("is_reference_like"),
            "sequence_index": photo.get("sequence_index"),
            "vlm_analysis": analysis,
        }
        vlm_results.append(result)
    return vlm_results


def _load_relationships(relationships_payload: List[Dict[str, Any]]) -> List[Any]:
    """Load relationship data from event payload - currently placeholder"""
    return []


def _load_face_db(face_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    persons = face_payload.get("face_recognition", {}).get("persons", [])
    if not persons:
        persons = face_payload.get("persons", [])
    if not persons:
        return _load_flat_face_db(face_payload)
    face_db: Dict[str, Dict[str, Any]] = {}
    for person in persons:
        person_id = person.get("person_id")
        if not person_id:
            continue
        face_db[person_id] = {
            "photo_count": int(person.get("photo_count", 0) or 0),
            "first_seen": _parse_datetime(person.get("first_seen")),
            "last_seen": _parse_datetime(person.get("last_seen")),
            "avg_confidence": float(person.get("avg_score", 0.0) or 0.0),
            "avg_quality": float(person.get("avg_quality", 0.0) or 0.0),
            "name": person.get("label", ""),
        }
    return face_db


def _load_flat_face_db(face_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    face_db: Dict[str, Dict[str, Any]] = {}
    for person_id, stats in (face_payload or {}).items():
        if not isinstance(person_id, str) or not isinstance(stats, dict):
            continue
        if not CANONICAL_PERSON_ID_PATTERN.fullmatch(person_id):
            continue
        face_db[person_id] = {
            "photo_count": int(stats.get("photo_count", 0) or 0),
            "first_seen": _parse_datetime(stats.get("first_seen")),
            "last_seen": _parse_datetime(stats.get("last_seen")),
            "avg_confidence": float(
                stats.get("avg_confidence", stats.get("avg_score", 0.0)) or 0.0
            ),
            "avg_quality": float(stats.get("avg_quality", 0.0) or 0.0),
            "name": stats.get("name", stats.get("label", "")),
        }
    return face_db


def _load_events(
    events_payload: List[Dict[str, Any]],
    *,
    primary_person_id: str | None,
) -> List[Event]:
    """Load events from new format (v2.0) with proper participant mapping"""
    events: List[Event] = []
    for raw_event in events_payload:
        if any(key in raw_event for key in ("participant_person_ids", "started_at", "supporting_photo_ids")):
            events.append(_load_compact_event(raw_event, primary_person_id=primary_person_id))
            continue

        # Parse date and time from event metadata
        meta_info = dict(raw_event.get("meta_info", {}) or {})
        objective_fact = dict(raw_event.get("objective_fact", {}) or {})
        date = raw_event.get("date", "")
        time_range = raw_event.get("time_range", "")

        # Normalize participants - convert "主角" to "Person_002" (from precomputed primary_person_id)
        raw_participants = list(raw_event.get("participants", []) or [])
        participants, canonicalized_raw_participants = _normalize_event_participants(
            raw_participants,
            primary_person_id=primary_person_id,
        )

        # Extract persona evidence from structured format
        persona_evidence = dict(raw_event.get("persona_evidence", {}) or {})
        social_dynamics = _normalize_social_dynamics(
            raw_event.get("social_dynamics", []),
            primary_person_id=primary_person_id,
        )
        if canonicalized_raw_participants:
            meta_info["raw_participants"] = canonicalized_raw_participants
            objective_fact["raw_participants"] = canonicalized_raw_participants

        events.append(
            Event(
                event_id=raw_event.get("event_id", ""),
                date=date,
                time_range=time_range,
                duration=raw_event.get("duration", ""),
                title=raw_event.get("title", ""),
                type=raw_event.get("type", "其他"),
                participants=participants,
                location=raw_event.get("location", ""),
                description=raw_event.get("description", ""),
                photo_count=raw_event.get("photo_count", 0),
                confidence=float(raw_event.get("confidence", 0.0) or 0.0),
                reason=raw_event.get("reason", ""),
                narrative_synthesis=raw_event.get("narrative", ""),
                meta_info={
                    **meta_info,
                    "started_at": f"{date}T{time_range.split(' - ')[0]}" if time_range else date,
                    "ended_at": f"{date}T{time_range.split(' - ')[1]}" if time_range and " - " in time_range else date,
                },
                objective_fact={
                    **objective_fact,
                    "participants": participants,
                    "place_refs": [raw_event.get("location", "")] if raw_event.get("location") else [],
                },
                tags=raw_event.get("lifestyle_tags", []),
                social_dynamics=social_dynamics,
                persona_evidence=persona_evidence,
            )
        )
    return events


def _load_compact_event(raw_event: Dict[str, Any], *, primary_person_id: str | None) -> Event:
    meta_info = dict(raw_event.get("meta_info", {}) or {})
    objective_fact = dict(raw_event.get("objective_fact", {}) or {})
    started_at = str(raw_event.get("started_at", "") or "")
    ended_at = str(raw_event.get("ended_at", "") or "")
    supporting_photo_ids = list(raw_event.get("supporting_photo_ids", []) or [])
    raw_participants = list(raw_event.get("participant_person_ids", []) or objective_fact.get("participants", []) or [])
    participants, canonicalized_raw_participants = _normalize_event_participants(
        raw_participants,
        primary_person_id=primary_person_id,
    )
    depicted_person_ids, raw_depicted_person_ids = _normalize_event_participants(
        raw_event.get("depicted_person_ids", []) or [],
        primary_person_id=primary_person_id,
    )
    place_refs = list(raw_event.get("place_refs", []) or objective_fact.get("place_refs", []) or [])
    location = (
        meta_info.get("location_context")
        or (place_refs[0] if place_refs else "")
        or raw_event.get("location", "")
    )
    photo_count = int(meta_info.get("photo_count") or len(supporting_photo_ids) or raw_event.get("photo_count", 0) or 0)
    objective_fact["participants"] = participants
    if place_refs and not objective_fact.get("place_refs"):
        objective_fact["place_refs"] = place_refs
    if canonicalized_raw_participants:
        meta_info["raw_participants"] = canonicalized_raw_participants
        objective_fact["raw_participants"] = canonicalized_raw_participants
    if depicted_person_ids:
        objective_fact["depicted_person_ids"] = depicted_person_ids
    if raw_depicted_person_ids:
        meta_info["raw_depicted_person_ids"] = raw_depicted_person_ids
    trace = {
        "supporting_photo_ids": supporting_photo_ids,
        "anchor_photo_id": raw_event.get("anchor_photo_id"),
        "batch_id": raw_event.get("batch_id"),
        "source_temp_event_id": raw_event.get("source_temp_event_id"),
        "started_at": started_at,
        "ended_at": ended_at,
        "depicted_person_ids": depicted_person_ids,
    }
    meta_info["trace"] = trace
    social_dynamics = _normalize_social_dynamics(
        raw_event.get("social_dynamics", []),
        primary_person_id=primary_person_id,
    )

    return Event(
        event_id=raw_event.get("event_id", ""),
        date=started_at[:10] if started_at else "",
        time_range=_build_time_range(started_at, ended_at),
        duration="",
        title=raw_event.get("title") or meta_info.get("title", ""),
        type=raw_event.get("type", "其他") or "其他",
        participants=participants,
        location=location,
        description=objective_fact.get("scene_description") or raw_event.get("description", ""),
        photo_count=photo_count,
        confidence=float(raw_event.get("confidence", 0.0) or 0.0),
        reason=raw_event.get("reason", ""),
        narrative_synthesis=raw_event.get("narrative_synthesis", ""),
        meta_info=meta_info,
        objective_fact=objective_fact,
        social_dynamics=social_dynamics,
        tags=list(raw_event.get("tags", []) or []),
        persona_evidence=dict(raw_event.get("persona_evidence", {}) or {}),
    )


def _normalize_face_person_ids(values: List[Any]) -> List[str]:
    normalized: List[str] = []
    seen = set()
    for value in values or []:
        candidate = str(value or "").strip()
        if not candidate or candidate in seen:
            continue
        normalized.append(candidate)
        seen.add(candidate)
    return normalized


def _normalize_event_participants(
    participants: List[Any],
    *,
    primary_person_id: str | None,
) -> tuple[List[str], List[str]]:
    raw_values: List[str] = []
    canonical_values: List[str] = []
    seen_raw = set()
    seen_canonical = set()
    for participant in participants or []:
        if not isinstance(participant, str):
            continue
        raw_value = participant.strip()
        if not raw_value:
            continue
        if raw_value not in seen_raw:
            raw_values.append(raw_value)
            seen_raw.add(raw_value)
        canonical = _extract_canonical_person_id(raw_value, primary_person_id)
        if not canonical or canonical in seen_canonical:
            continue
        canonical_values.append(canonical)
        seen_canonical.add(canonical)
    return canonical_values, raw_values


def _normalize_social_dynamics(
    social_dynamics: List[Any],
    *,
    primary_person_id: str | None,
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for item in social_dynamics or []:
        if not isinstance(item, dict):
            continue
        dynamic = dict(item)
        target_id = dynamic.get("target_id")
        canonical_target = _extract_canonical_person_id(target_id, primary_person_id)
        if canonical_target:
            dynamic["target_id"] = canonical_target
        elif isinstance(target_id, str):
            dynamic["target_id"] = _normalize_primary_person_reference(target_id, primary_person_id)
        normalized.append(dynamic)
    return normalized


def _extract_canonical_person_id(raw_value: Any, primary_person_id: str | None) -> str:
    normalized = _normalize_primary_person_reference(raw_value, primary_person_id)
    if not isinstance(normalized, str):
        return ""
    value = normalized.strip()
    if not value:
        return ""
    if primary_person_id and value == primary_person_id:
        return value
    if CANONICAL_PERSON_ID_PATTERN.fullmatch(value):
        return value
    match = CANONICAL_PERSON_ID_PATTERN.search(value)
    return match.group(0) if match else ""


def _normalize_primary_person_reference(raw_value: Any, primary_person_id: str | None) -> Any:
    if not isinstance(raw_value, str):
        return raw_value
    value = raw_value.strip()
    if not value or not primary_person_id:
        return value
    compact = re.sub(r"[\s\-_]+", "", value).replace("【", "").replace("】", "").lower()
    if compact in PRIMARY_ALIAS_COMPACTS or re.fullmatch(r"person0+", compact):
        return primary_person_id
    return value


def _build_time_range(started_at: str, ended_at: str) -> str:
    if not started_at:
        return ""
    start = started_at[11:16] if len(started_at) >= 16 else ""
    end = ended_at[11:16] if ended_at and len(ended_at) >= 16 else start
    return f"{start} - {end}".strip(" -")


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None
