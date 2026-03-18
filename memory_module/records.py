"""Storage-layer records for the memory module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class PersonIdentityMapRecord:
    user_id: str
    tenant_id: Optional[str]
    face_person_id: str
    person_uuid: str
    source_system: str


@dataclass(slots=True)
class PhotoIdentityMapRecord:
    user_id: str
    tenant_id: Optional[str]
    photo_id: str
    photo_uuid: str
    source_hash: str
    source_system: str


@dataclass(slots=True)
class Neo4jUserNodeRecord:
    user_id: str
    tenant_id: Optional[str]
    labels: List[str] = field(default_factory=lambda: ["User"])
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Neo4jPersonNodeRecord:
    person_uuid: str
    labels: List[str] = field(default_factory=lambda: ["Person"])
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Neo4jPlaceNodeRecord:
    place_uuid: str
    labels: List[str] = field(default_factory=lambda: ["PlaceAnchor"])
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Neo4jSessionNodeRecord:
    session_uuid: str
    labels: List[str] = field(default_factory=lambda: ["Event"])
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Neo4jEventNodeRecord:
    event_uuid: str
    labels: List[str] = field(default_factory=lambda: ["Fact"])
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Neo4jRelationshipHypothesisNodeRecord:
    relationship_uuid: str
    labels: List[str] = field(default_factory=lambda: ["RelationshipHypothesis"])
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Neo4jMoodStateNodeRecord:
    mood_uuid: str
    labels: List[str] = field(default_factory=lambda: ["MoodStateHypothesis"])
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Neo4jPrimaryPersonHypothesisNodeRecord:
    primary_person_hypothesis_uuid: str
    labels: List[str] = field(default_factory=lambda: ["PrimaryPersonHypothesis"])
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Neo4jPeriodHypothesisNodeRecord:
    period_uuid: str
    labels: List[str] = field(default_factory=lambda: ["PeriodHypothesis"])
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Neo4jConceptNodeRecord:
    concept_uuid: str
    labels: List[str] = field(default_factory=lambda: ["Concept"])
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Neo4jRelationshipEdgeRecord:
    edge_id: str
    from_id: str
    to_id: str
    edge_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MilvusSegmentRecord:
    segment_uuid: str
    tenant_id: Optional[str]
    user_id: str
    photo_uuid: Optional[str]
    event_uuid: Optional[str]
    person_uuid: Optional[str]
    session_uuid: Optional[str]
    relationship_uuid: Optional[str]
    concept_uuid: Optional[str]
    segment_type: str
    text: str
    started_at: Optional[str] = None
    ended_at: Optional[str] = None
    place_uuid: Optional[str] = None
    location_hint: str = ""
    sparse_terms: List[str] = field(default_factory=list)
    embedding_source: str = "textual_stub"
    importance_score: float = 0.0
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)


@dataclass(slots=True)
class RedisProfileCoreRecord:
    key: str
    payload: Dict[str, Any]


@dataclass(slots=True)
class RedisProfileRelationshipsRecord:
    key: str
    payload: Dict[str, Any]


@dataclass(slots=True)
class RedisProfileRecentEventsRecord:
    key: str
    payload: Dict[str, Any]


@dataclass(slots=True)
class RedisProfileRecentFactsRecord:
    key: str
    payload: Dict[str, Any]


@dataclass(slots=True)
class RedisProfileMetaRecord:
    key: str
    payload: Dict[str, Any]


@dataclass(slots=True)
class RedisProfileDebugRefsRecord:
    key: str
    payload: Dict[str, Any]
