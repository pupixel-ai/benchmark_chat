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
class RelationshipHypothesisDTO:
    relationship_id: str
    upstream_ref: Dict[str, str]
    face_person_id: str
    person_uuid: str
    relationship_type: str
    label: str
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)
    reason: str = ""


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
