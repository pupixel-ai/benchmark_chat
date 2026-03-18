"""Transport-layer DTOs for the memory module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class ScopeDTO:
    user_id: str
    tenant_id: Optional[str] = None
    source_system: str = "memory_engineering"
    pipeline_version: str = ""
    task_id: Optional[str] = None
    ingestion_id: str = ""
    generated_at: str = ""


@dataclass(slots=True)
class ArtifactRefDTO:
    artifact_type: str
    path: str
    url: Optional[str] = None
    mime_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FaceObservationDTO:
    face_id: str
    face_uuid: str
    face_person_id: str
    person_uuid: str
    upstream_ref: Dict[str, str]
    bbox_xywh: Dict[str, int]
    confidence: float
    similarity: float
    quality_score: float
    quality_flags: List[str] = field(default_factory=list)
    pose_bucket: Optional[str] = None
    pose_yaw: Optional[float] = None
    pose_pitch: Optional[float] = None
    pose_roll: Optional[float] = None
    landmark_detected: bool = False
    landmark_source: Optional[str] = None


@dataclass(slots=True)
class PersonObservationDTO:
    observation_id: str
    upstream_ref: Dict[str, str]
    face_id: Optional[str]
    face_person_id: Optional[str]
    person_uuid: Optional[str]
    appearance: str = ""
    clothing: str = ""
    activity: str = ""
    interaction: str = ""
    expression: str = ""
    confidence: float = 0.0


@dataclass(slots=True)
class VLMPhotoObservationDTO:
    summary: str
    scene: Dict[str, Any] = field(default_factory=dict)
    event: Dict[str, Any] = field(default_factory=dict)
    details: List[str] = field(default_factory=list)
    key_objects: List[str] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PhotoFactDTO:
    photo_id: str
    photo_uuid: str
    upstream_ref: Dict[str, str]
    filename: str
    source_hash: str
    captured_at_original: str
    captured_at_utc: str
    timezone_guess: Optional[str]
    time_confidence: float
    location: Dict[str, Any]
    primary_face_person_id: Optional[str]
    faces: List[FaceObservationDTO] = field(default_factory=list)
    vlm_observation: Optional[VLMPhotoObservationDTO] = None
    people_observations: List[PersonObservationDTO] = field(default_factory=list)
    artifact_refs: List[ArtifactRefDTO] = field(default_factory=list)


@dataclass(slots=True)
class BurstDTO:
    burst_id: str
    burst_uuid: str
    upstream_ref: Dict[str, str]
    photo_ids: List[str]
    photo_uuids: List[str]
    started_at: str
    ended_at: str
    duration_seconds: int


@dataclass(slots=True)
class SessionDTO:
    session_id: str
    session_uuid: str
    upstream_ref: Dict[str, str]
    day_key: str
    photo_ids: List[str]
    photo_uuids: List[str]
    burst_ids: List[str]
    started_at: str
    ended_at: str
    duration_seconds: int
    location_hint: Dict[str, Any]
    dominant_face_person_ids: List[str] = field(default_factory=list)
    dominant_person_uuids: List[str] = field(default_factory=list)
    activity_hints: List[str] = field(default_factory=list)
    summary_hint: str = ""


@dataclass(slots=True)
class MovementDTO:
    movement_id: str
    movement_uuid: str
    upstream_ref: Dict[str, str]
    from_session_id: str
    from_session_uuid: str
    to_session_id: str
    to_session_uuid: str
    started_at: str
    ended_at: str
    from_location: Dict[str, Any]
    to_location: Dict[str, Any]
    distance_km: float = 0.0


@dataclass(slots=True)
class DayTimelineDTO:
    timeline_id: str
    timeline_uuid: str
    upstream_ref: Dict[str, str]
    day_key: str
    session_ids: List[str]
    session_uuids: List[str]
    movement_ids: List[str]
    started_at: str
    ended_at: str


@dataclass(slots=True)
class EventCandidateDTO:
    event_id: str
    event_uuid: str
    upstream_ref: Dict[str, str]
    title: str
    event_type: str
    time_range: str
    started_at: str
    ended_at: str
    location: str
    participant_face_person_ids: List[str] = field(default_factory=list)
    participant_person_uuids: List[str] = field(default_factory=list)
    photo_ids: List[str] = field(default_factory=list)
    photo_uuids: List[str] = field(default_factory=list)
    session_ids: List[str] = field(default_factory=list)
    session_uuids: List[str] = field(default_factory=list)
    description: str = ""
    narrative_synthesis: str = ""
    tags: List[str] = field(default_factory=list)
    persona_evidence: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class PlaceAnchorDTO:
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
    search_text: str = ""


@dataclass(slots=True)
class ConceptDTO:
    concept_uuid: str
    canonical_name: str
    aliases: List[str] = field(default_factory=list)
    concept_type: str = "unknown"
    scope: str = "canonical"
    status: str = "active"
    version: str = "v1"
    user_id: Optional[str] = None
    search_text: str = ""
    description: str = ""
    parent_concepts: List[str] = field(default_factory=list)


@dataclass(slots=True)
class RelationshipHypothesisDTO:
    relationship_uuid: str
    relationship_key: str
    upstream_ref: Dict[str, str]
    anchor_person_uuid: Optional[str]
    target_person_uuid: str
    target_face_person_id: str
    relationship_type: str
    label: str
    confidence: float
    revision: int = 1
    status: str = "draft"
    window_start: str = ""
    window_end: str = ""
    model_version: str = ""
    reason_summary: str = ""
    feature_snapshot: Dict[str, Any] = field(default_factory=dict)
    score_snapshot: Dict[str, float] = field(default_factory=dict)
    inherited_metrics: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)
    prior_revision_uuid: Optional[str] = None


@dataclass(slots=True)
class MoodStateHypothesisDTO:
    mood_uuid: str
    upstream_ref: Dict[str, str]
    mood_label: str
    mood_score: float
    confidence: float
    session_uuid: Optional[str] = None
    event_uuid: Optional[str] = None
    window_start: str = ""
    window_end: str = ""
    model_version: str = ""
    reason_summary: str = ""
    artifact_ref_ids: List[str] = field(default_factory=list)
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class PrimaryPersonHypothesisDTO:
    primary_person_hypothesis_uuid: str
    upstream_ref: Dict[str, str]
    user_id: str
    person_uuid: str
    confidence: float
    window_start: str
    window_end: str
    model_version: str = ""
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class PeriodHypothesisDTO:
    period_uuid: str
    upstream_ref: Dict[str, str]
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
class ProfileEvidenceDTO:
    evidence_id: str
    upstream_ref: Dict[str, str]
    field_key: str
    field_value: str
    category: str
    confidence: float
    supporting_event_ids: List[str] = field(default_factory=list)
    supporting_event_uuids: List[str] = field(default_factory=list)
    supporting_person_uuids: List[str] = field(default_factory=list)
    supporting_session_uuids: List[str] = field(default_factory=list)
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ObservationDTO:
    observation_id: str
    upstream_ref: Dict[str, str]
    category: str
    field_key: str
    field_value: str
    confidence: float
    photo_ids: List[str] = field(default_factory=list)
    event_id: Optional[str] = None
    session_id: Optional[str] = None
    person_ids: List[str] = field(default_factory=list)
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class ClaimDTO:
    claim_id: str
    upstream_ref: Dict[str, str]
    claim_type: str
    subject: str
    predicate: str
    object_value: str
    confidence: float
    photo_ids: List[str] = field(default_factory=list)
    event_id: Optional[str] = None
    session_id: Optional[str] = None
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class ProfileDeltaDTO:
    delta_id: str
    upstream_ref: Dict[str, str]
    profile_key: str
    field_key: str
    field_value: str
    summary: str
    confidence: float
    supporting_event_ids: List[str] = field(default_factory=list)
    supporting_photo_ids: List[str] = field(default_factory=list)
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class UncertaintyDTO:
    uncertainty_id: str
    upstream_ref: Dict[str, str]
    field_name: str
    status: str
    reason: str


@dataclass(slots=True)
class ChangeLogEntryDTO:
    change_id: str
    object_type: str
    object_id: str
    change_type: str
    summary: str
    upstream_ref: Dict[str, str] = field(default_factory=dict)
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TimeScopeDTO:
    raw_text: str
    start_at: str = ""
    end_at: str = ""
    resolution: str = "unresolved"
    confidence: float = 0.0


@dataclass(slots=True)
class AgentMemoryQueryRequestDTO:
    user_id: str
    question: str
    query_id: str
    context_hints: Dict[str, Any] = field(default_factory=dict)
    time_hint: Optional[str] = None
    answer_shape_hint: Optional[str] = None


@dataclass(slots=True)
class OperatorPlanDTO:
    intent: str
    time_scope: TimeScopeDTO
    ordinal: Optional[int] = None
    threshold: Optional[float] = None
    group_by: Optional[str] = None
    metric: Optional[str] = None
    target_concepts: List[str] = field(default_factory=list)
    target_entities: List[str] = field(default_factory=list)
    output_shape: str = "summary"
    fallback_policy: str = "semantic_then_graph"


@dataclass(slots=True)
class QueryPlanDTO:
    subject_binding: str
    target_spec: Dict[str, Any] = field(default_factory=dict)
    operators: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    source_order: List[str] = field(default_factory=list)
    answer_schema: Dict[str, Any] = field(default_factory=dict)
    evidence_requirements: List[str] = field(default_factory=list)


@dataclass(slots=True)
class QueryDSLDTO:
    intent: str
    graph_filters: Dict[str, Any] = field(default_factory=dict)
    time_scope: Dict[str, Any] = field(default_factory=dict)
    ranking_rule: Dict[str, Any] = field(default_factory=dict)
    evidence_fill: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EntityRecallCandidateDTO:
    entity_type: str
    entity_id: str
    score: float
    matched_concept: Optional[str] = None
    source: str = "index"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnswerDTO:
    answer_type: str
    summary: str
    confidence: float
    resolved_entities: List[Dict[str, Any]] = field(default_factory=list)
    resolved_concepts: List[str] = field(default_factory=list)
    time_window: Dict[str, Any] = field(default_factory=dict)
    supporting_events: List[Dict[str, Any]] = field(default_factory=list)
    supporting_facts: List[Dict[str, Any]] = field(default_factory=list)
    supporting_relationships: List[Dict[str, Any]] = field(default_factory=list)
    representative_photo_ids: List[str] = field(default_factory=list)
    evidence_segment_ids: List[str] = field(default_factory=list)
    explanation: str = ""
    uncertainty_flags: List[str] = field(default_factory=list)


@dataclass(slots=True)
class GraphDebugTraceDTO:
    operator_plan: Dict[str, Any]
    recall_candidates: List[Dict[str, Any]]
    dsl: Dict[str, Any]
    executed_cypher: str
    evidence_fill: Dict[str, Any]


@dataclass(slots=True)
class MemoryIngestionEnvelopeDTO:
    scope: ScopeDTO
    photos: List[PhotoFactDTO] = field(default_factory=list)
    bursts: List[BurstDTO] = field(default_factory=list)
    sessions: List[SessionDTO] = field(default_factory=list)
    movements: List[MovementDTO] = field(default_factory=list)
    day_timelines: List[DayTimelineDTO] = field(default_factory=list)
    event_candidates: List[EventCandidateDTO] = field(default_factory=list)
    relationship_hypotheses: List[RelationshipHypothesisDTO] = field(default_factory=list)
    profile_evidence: List[ProfileEvidenceDTO] = field(default_factory=list)
    artifacts: List[ArtifactRefDTO] = field(default_factory=list)
    change_log: List[ChangeLogEntryDTO] = field(default_factory=list)
