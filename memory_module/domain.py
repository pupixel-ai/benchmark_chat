"""Domain entities and aggregates for the memory module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class UserMemoryAggregate:
    user_id: str
    tenant_id: Optional[str]
    person_uuids: List[str] = field(default_factory=list)
    session_uuids: List[str] = field(default_factory=list)
    event_uuids: List[str] = field(default_factory=list)
    relationship_uuids: List[str] = field(default_factory=list)
    latest_profile_version: int = 0


@dataclass(slots=True)
class PersonEntity:
    person_uuid: str
    face_person_id: str
    user_id: str
    photo_count: int = 0
    face_count: int = 0
    is_primary_candidate: bool = False
    first_seen_at: str = ""
    last_seen_at: str = ""
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class PlaceAnchorEntity:
    place_uuid: str
    user_id: str
    canonical_name: str
    aliases: List[str] = field(default_factory=list)
    place_type: str = "unknown"
    geo_hash: str = ""
    lat: Optional[float] = None
    lng: Optional[float] = None
    source: str = "derived_from_session"
    confidence: float = 0.0


@dataclass(slots=True)
class SessionEntity:
    session_uuid: str
    user_id: str
    started_at: str
    ended_at: str
    duration_seconds: int = 0
    place_uuid: Optional[str] = None
    participant_count: int = 0
    dominant_person_uuids: List[str] = field(default_factory=list)
    photo_count: int = 0
    representative_photo_ids: List[str] = field(default_factory=list)
    artifact_ref_ids: List[str] = field(default_factory=list)


@dataclass(slots=True)
class DayTimelineEntity:
    timeline_uuid: str
    user_id: str
    day_key: str
    started_at: str
    ended_at: str
    session_uuids: List[str] = field(default_factory=list)


@dataclass(slots=True)
class EventHypothesisEntity:
    event_uuid: str
    user_id: str
    title: str
    normalized_event_type: str
    event_subtype: str
    started_at: str
    ended_at: str
    place_uuid: Optional[str]
    confidence: float
    participant_count: int = 0
    photo_count: int = 0
    representative_photo_ids: List[str] = field(default_factory=list)
    artifact_ref_ids: List[str] = field(default_factory=list)
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class RelationshipHypothesisEntity:
    relationship_uuid: str
    relationship_key: str
    revision: int
    status: str
    anchor_person_uuid: Optional[str]
    target_person_uuid: str
    relationship_type: str
    label: str
    confidence: float
    window_start: str
    window_end: str
    model_version: str
    reason_summary: str
    feature_snapshot: Dict[str, Any] = field(default_factory=dict)
    score_snapshot: Dict[str, float] = field(default_factory=dict)
    inherited_metrics: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)
    prior_revision_uuid: Optional[str] = None


@dataclass(slots=True)
class MoodStateHypothesisEntity:
    mood_uuid: str
    mood_label: str
    mood_score: float
    confidence: float
    session_uuid: Optional[str] = None
    event_uuid: Optional[str] = None
    window_start: str = ""
    window_end: str = ""
    reason_summary: str = ""
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class PrimaryPersonHypothesisEntity:
    primary_person_hypothesis_uuid: str
    user_id: str
    person_uuid: str
    confidence: float
    window_start: str
    window_end: str
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class PeriodHypothesisEntity:
    period_uuid: str
    user_id: str
    period_type: str
    label: str
    window_start: str
    window_end: str
    confidence: float
    reason_summary: str = ""
    artifact_ref_ids: List[str] = field(default_factory=list)
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class ConceptEntity:
    concept_uuid: str
    canonical_name: str
    aliases: List[str] = field(default_factory=list)
    concept_type: str = "unknown"
    scope: str = "canonical"
    status: str = "active"
    version: str = "v1"
    user_id: Optional[str] = None


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
    user_id: str
    photo_uuid: Optional[str] = None
    person_uuid: Optional[str] = None
    session_uuid: Optional[str] = None
    event_uuid: Optional[str] = None
    relationship_uuid: Optional[str] = None
    concept_uuid: Optional[str] = None
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
class ProfileEvidenceItem:
    evidence_id: str
    field_key: str
    field_value: str
    confidence: float
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class MaterializationInputBundle:
    user_id: str
    facts: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    hypotheses: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    materialized_views: Dict[str, Any] = field(default_factory=dict)
    evidence_segments: List[Dict[str, Any]] = field(default_factory=list)
    profile_evidence_items: List[ProfileEvidenceItem] = field(default_factory=list)
