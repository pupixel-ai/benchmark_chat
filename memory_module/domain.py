"""Domain entities and aggregates for the memory module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class UserMemoryAggregate:
    user_id: str
    tenant_id: Optional[str]
    person_uuids: List[str] = field(default_factory=list)
    photo_uuids: List[str] = field(default_factory=list)
    session_uuids: List[str] = field(default_factory=list)
    event_uuids: List[str] = field(default_factory=list)
    latest_profile_version: int = 0


@dataclass(slots=True)
class PersonEntity:
    person_uuid: str
    face_person_id: str
    user_id: str
    photo_count: int = 0
    face_count: int = 0
    is_primary: bool = False
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class PhotoEntity:
    photo_uuid: str
    photo_id: str
    user_id: str
    source_hash: str
    captured_at: str
    location: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class FaceEntity:
    face_uuid: str
    face_id: str
    face_person_id: str
    person_uuid: str
    photo_uuid: str
    quality_score: float = 0.0
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class BurstEntity:
    burst_uuid: str
    photo_uuids: List[str] = field(default_factory=list)
    started_at: str = ""
    ended_at: str = ""


@dataclass(slots=True)
class SessionEntity:
    session_uuid: str
    photo_uuids: List[str] = field(default_factory=list)
    burst_uuids: List[str] = field(default_factory=list)
    started_at: str = ""
    ended_at: str = ""
    location_hint: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MovementEntity:
    movement_uuid: str
    from_session_uuid: str
    to_session_uuid: str
    distance_km: float = 0.0


@dataclass(slots=True)
class DayTimelineEntity:
    timeline_uuid: str
    day_key: str
    session_uuids: List[str] = field(default_factory=list)
    movement_uuids: List[str] = field(default_factory=list)


@dataclass(slots=True)
class EventEntity:
    event_uuid: str
    title: str
    event_type: str
    location: str
    started_at: str
    ended_at: str
    session_uuids: List[str] = field(default_factory=list)
    person_uuids: List[str] = field(default_factory=list)
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class RelationshipEntity:
    relationship_id: str
    person_uuid: str
    relationship_type: str
    label: str
    confidence: float
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class ProfileAggregate:
    user_id: str
    profile_version: int
    generated_at: str
    fields: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class SemanticSegmentEntity:
    segment_uuid: str
    segment_type: str
    text: str
    embedding_source: str
    photo_uuid: str
    person_uuid: Optional[str] = None
    session_uuid: Optional[str] = None
    event_uuid: Optional[str] = None
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class IngestionJobEntity:
    ingestion_id: str
    user_id: str
    task_id: Optional[str]
    state: str
    generated_at: str


@dataclass(slots=True)
class DeleteJobEntity:
    delete_job_id: str
    user_id: str
    state: str
    scope: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RebuildJobEntity:
    rebuild_job_id: str
    user_id: str
    state: str
    generated_at: str


@dataclass(slots=True)
class EventCandidate:
    event_uuid: str
    title: str
    confidence: float
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class RelationshipHypothesis:
    relationship_id: str
    person_uuid: str
    relationship_type: str
    confidence: float
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class ProfileEvidenceItem:
    evidence_id: str
    field_key: str
    field_value: str
    confidence: float
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class ProfileAssertionCandidate:
    field_key: str
    field_value: str
    confidence: float
    supporting_event_uuids: List[str] = field(default_factory=list)
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class ConflictMarker:
    conflict_id: str
    field_key: str
    conflict_summary: str
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class MaterializationInputBundle:
    user_id: str
    person_entities: List[PersonEntity] = field(default_factory=list)
    photo_entities: List[PhotoEntity] = field(default_factory=list)
    event_entities: List[EventEntity] = field(default_factory=list)
    relationship_entities: List[RelationshipEntity] = field(default_factory=list)
    profile_evidence_items: List[ProfileEvidenceItem] = field(default_factory=list)
