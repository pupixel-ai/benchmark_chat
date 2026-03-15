"""Sequence builders for burst/session/movement/day timeline objects."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from typing import Dict, Iterable, List, Sequence
from uuid import NAMESPACE_URL, uuid5

from memory_module.dto import BurstDTO, DayTimelineDTO, MovementDTO, SessionDTO


BURST_GAP_SECONDS = 90
SESSION_GAP_SECONDS = 4 * 60 * 60
SESSION_DISTANCE_THRESHOLD_KM = 1.5
MOVEMENT_DISTANCE_THRESHOLD_KM = 0.1


@dataclass(slots=True)
class SequenceBundle:
    bursts: List[BurstDTO]
    sessions: List[SessionDTO]
    movements: List[MovementDTO]
    day_timelines: List[DayTimelineDTO]


def build_sequences(
    photos: Sequence,
    photo_uuid_map: Dict[str, str],
    person_uuid_map: Dict[str, str],
    scope_key: str,
) -> SequenceBundle:
    ordered = sorted(photos, key=lambda photo: photo.timestamp)
    if not ordered:
        return SequenceBundle(bursts=[], sessions=[], movements=[], day_timelines=[])

    bursts = _build_bursts(ordered, photo_uuid_map, scope_key)
    sessions = _build_sessions(ordered, bursts, photo_uuid_map, person_uuid_map, scope_key)
    movements = _build_movements(sessions, scope_key)
    day_timelines = _build_day_timelines(sessions, movements, scope_key)
    return SequenceBundle(
        bursts=bursts,
        sessions=sessions,
        movements=movements,
        day_timelines=day_timelines,
    )


def _build_bursts(photos: Sequence, photo_uuid_map: Dict[str, str], scope_key: str) -> List[BurstDTO]:
    groups: List[List] = []
    current_group: List = [photos[0]]

    for photo in photos[1:]:
        gap_seconds = (photo.timestamp - current_group[-1].timestamp).total_seconds()
        if gap_seconds <= BURST_GAP_SECONDS:
            current_group.append(photo)
            continue
        groups.append(current_group)
        current_group = [photo]

    groups.append(current_group)

    bursts: List[BurstDTO] = []
    for index, group in enumerate(groups, start=1):
        burst_id = f"burst_{index:03d}"
        burst_uuid = _stable_uuid(scope_key, "burst", burst_id)
        bursts.append(
            BurstDTO(
                burst_id=burst_id,
                burst_uuid=burst_uuid,
                upstream_ref={"object_type": "burst", "object_id": burst_id},
                photo_ids=[photo.photo_id for photo in group],
                photo_uuids=[photo_uuid_map[photo.photo_id] for photo in group],
                started_at=group[0].timestamp.isoformat(),
                ended_at=group[-1].timestamp.isoformat(),
                duration_seconds=int((group[-1].timestamp - group[0].timestamp).total_seconds()),
            )
        )

    return bursts


def _build_sessions(
    photos: Sequence,
    bursts: Sequence[BurstDTO],
    photo_uuid_map: Dict[str, str],
    person_uuid_map: Dict[str, str],
    scope_key: str,
) -> List[SessionDTO]:
    burst_by_photo_id = {}
    for burst in bursts:
        for photo_id in burst.photo_ids:
            burst_by_photo_id[photo_id] = burst

    groups: List[List] = []
    current_group: List = [photos[0]]

    for photo in photos[1:]:
        previous = current_group[-1]
        gap_seconds = (photo.timestamp - previous.timestamp).total_seconds()
        if gap_seconds <= SESSION_GAP_SECONDS and _same_session(previous.location, photo.location):
            current_group.append(photo)
            continue
        groups.append(current_group)
        current_group = [photo]

    groups.append(current_group)

    sessions: List[SessionDTO] = []
    for index, group in enumerate(groups, start=1):
        session_id = f"session_{index:03d}"
        session_uuid = _stable_uuid(scope_key, "session", session_id)
        burst_ids = _unique(
            burst_by_photo_id[photo.photo_id].burst_id
            for photo in group
            if photo.photo_id in burst_by_photo_id
        )
        face_person_ids = [face.get("person_id") for photo in group for face in photo.faces if face.get("person_id")]
        dominant_face_person_ids = [
            person_id
            for person_id, _count in Counter(face_person_ids).most_common(3)
        ]
        activity_hints = _session_activity_hints(group)
        sessions.append(
            SessionDTO(
                session_id=session_id,
                session_uuid=session_uuid,
                upstream_ref={"object_type": "session", "object_id": session_id},
                day_key=group[0].timestamp.strftime("%Y-%m-%d"),
                photo_ids=[photo.photo_id for photo in group],
                photo_uuids=[photo_uuid_map[photo.photo_id] for photo in group],
                burst_ids=burst_ids,
                started_at=group[0].timestamp.isoformat(),
                ended_at=group[-1].timestamp.isoformat(),
                duration_seconds=int((group[-1].timestamp - group[0].timestamp).total_seconds()),
                location_hint=_location_hint(group),
                dominant_face_person_ids=dominant_face_person_ids,
                dominant_person_uuids=[
                    person_uuid_map[person_id]
                    for person_id in dominant_face_person_ids
                    if person_id in person_uuid_map
                ],
                activity_hints=activity_hints,
                summary_hint=_session_summary_hint(group, activity_hints),
            )
        )

    return sessions


def _build_movements(sessions: Sequence[SessionDTO], scope_key: str) -> List[MovementDTO]:
    movements: List[MovementDTO] = []

    for index, (left, right) in enumerate(zip(sessions, sessions[1:]), start=1):
        from_location = left.location_hint or {}
        to_location = right.location_hint or {}
        distance_km = _distance_km(from_location, to_location)
        if distance_km < MOVEMENT_DISTANCE_THRESHOLD_KM and _location_name(from_location) == _location_name(to_location):
            continue

        movement_id = f"movement_{index:03d}"
        movement_uuid = _stable_uuid(scope_key, "movement", movement_id)
        movements.append(
            MovementDTO(
                movement_id=movement_id,
                movement_uuid=movement_uuid,
                upstream_ref={"object_type": "movement", "object_id": movement_id},
                from_session_id=left.session_id,
                from_session_uuid=left.session_uuid,
                to_session_id=right.session_id,
                to_session_uuid=right.session_uuid,
                started_at=left.ended_at,
                ended_at=right.started_at,
                from_location=from_location,
                to_location=to_location,
                distance_km=round(distance_km, 3),
            )
        )

    return movements


def _build_day_timelines(
    sessions: Sequence[SessionDTO],
    movements: Sequence[MovementDTO],
    scope_key: str,
) -> List[DayTimelineDTO]:
    sessions_by_day: Dict[str, List[SessionDTO]] = defaultdict(list)
    for session in sessions:
        sessions_by_day[session.day_key].append(session)

    movement_lookup: Dict[tuple[str, str], str] = {}
    for movement in movements:
        movement_lookup[(movement.from_session_id, movement.to_session_id)] = movement.movement_id

    timelines: List[DayTimelineDTO] = []
    for index, day_key in enumerate(sorted(sessions_by_day), start=1):
        day_sessions = sorted(sessions_by_day[day_key], key=lambda item: item.started_at)
        movement_ids = []
        for left, right in zip(day_sessions, day_sessions[1:]):
            movement_id = movement_lookup.get((left.session_id, right.session_id))
            if movement_id:
                movement_ids.append(movement_id)

        timeline_id = f"timeline_{index:03d}"
        timeline_uuid = _stable_uuid(scope_key, "timeline", timeline_id)
        timelines.append(
            DayTimelineDTO(
                timeline_id=timeline_id,
                timeline_uuid=timeline_uuid,
                upstream_ref={"object_type": "day_timeline", "object_id": timeline_id},
                day_key=day_key,
                session_ids=[session.session_id for session in day_sessions],
                session_uuids=[session.session_uuid for session in day_sessions],
                movement_ids=movement_ids,
                started_at=day_sessions[0].started_at,
                ended_at=day_sessions[-1].ended_at,
            )
        )

    return timelines


def _session_activity_hints(group: Sequence) -> List[str]:
    hints: List[str] = []
    for photo in group:
        analysis = photo.vlm_analysis if isinstance(photo.vlm_analysis, dict) else {}
        event = analysis.get("event", {}) if isinstance(analysis, dict) else {}
        scene = analysis.get("scene", {}) if isinstance(analysis, dict) else {}
        for candidate in (
            event.get("activity") if isinstance(event, dict) else None,
            event.get("social_context") if isinstance(event, dict) else None,
            scene.get("location_detected") if isinstance(scene, dict) else None,
        ):
            if candidate and candidate not in hints:
                hints.append(str(candidate))
    return hints[:5]


def _session_summary_hint(group: Sequence, activity_hints: Sequence[str]) -> str:
    for photo in group:
        analysis = photo.vlm_analysis if isinstance(photo.vlm_analysis, dict) else {}
        summary = analysis.get("summary") if isinstance(analysis, dict) else ""
        if summary:
            return str(summary)
    if activity_hints:
        return " / ".join(activity_hints[:2])
    return f"{len(group)} photos"


def _location_hint(group: Sequence) -> Dict[str, object]:
    named_locations = [
        photo.location
        for photo in group
        if isinstance(photo.location, dict) and photo.location.get("name")
    ]
    if named_locations:
        counts = Counter(str(location.get("name")) for location in named_locations)
        dominant_name, _count = counts.most_common(1)[0]
        for location in named_locations:
            if location.get("name") == dominant_name:
                return dict(location)

    for photo in group:
        if isinstance(photo.location, dict) and ("lat" in photo.location or "lng" in photo.location):
            return dict(photo.location)

    return {"name": "unknown"}


def _same_session(left: Dict, right: Dict) -> bool:
    left_name = _location_name(left)
    right_name = _location_name(right)
    if left_name and right_name:
        return left_name == right_name

    distance_km = _distance_km(left, right)
    if distance_km and distance_km <= SESSION_DISTANCE_THRESHOLD_KM:
        return True

    if not left_name and not right_name and not distance_km:
        return True

    return False


def _location_name(location: Dict | None) -> str:
    if not isinstance(location, dict):
        return ""
    return str(location.get("name") or "").strip().lower()


def _distance_km(left: Dict | None, right: Dict | None) -> float:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return 0.0

    if "lat" not in left or "lng" not in left or "lat" not in right or "lng" not in right:
        return 0.0

    lat1 = float(left["lat"])
    lon1 = float(left["lng"])
    lat2 = float(right["lat"])
    lon2 = float(right["lng"])
    radius_km = 6371.0

    lat1_r = radians(lat1)
    lat2_r = radians(lat2)
    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)

    hav = (
        sin(delta_lat / 2) ** 2
        + cos(lat1_r) * cos(lat2_r) * sin(delta_lon / 2) ** 2
    )
    return 2 * radius_km * asin(sqrt(hav))


def _unique(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _stable_uuid(scope_key: str, object_type: str, object_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"{scope_key}:{object_type}:{object_id}"))
