"""Transparency and read-model views for the memory module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(slots=True)
class FaceStageView:
    total_faces: int
    total_persons: int
    primary_face_person_id: str | None
    failed_images: int


@dataclass(slots=True)
class VLMStageView:
    processed_photos: int
    cached_hits: int
    representative_photo_count: int = 0
    total_input_photos: int = 0
    summaries: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class SequenceStageView:
    burst_count: int
    session_count: int
    movement_count: int
    timeline_count: int
    summaries: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class LLMStageView:
    event_candidate_count: int
    relationship_hypothesis_count: int
    profile_evidence_count: int
    observation_count: int = 0
    claim_count: int = 0
    profile_delta_count: int = 0
    uncertainty_count: int = 0
    slice_count: int = 0
    summaries: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class Neo4jStateView:
    node_counts: Dict[str, int]
    edge_count: int


@dataclass(slots=True)
class GraphNodeView:
    node_id: str
    label: str
    node_type: str
    ring: int
    is_primary: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GraphEdgeView:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    label: str = ""
    confidence: float | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FocusGraphView:
    center_node_id: str
    primary_face_person_id: str | None
    primary_person_uuid: str | None
    node_count: int
    edge_count: int
    nodes: List[GraphNodeView] = field(default_factory=list)
    edges: List[GraphEdgeView] = field(default_factory=list)
    mermaid: str = ""


@dataclass(slots=True)
class MilvusStateView:
    segment_count: int
    segment_type_counts: Dict[str, int]


@dataclass(slots=True)
class RedisStateView:
    profile_version: int
    published_field_count: int
    relationship_count: int
    recent_event_count: int
    recent_timeline_count: int


@dataclass(slots=True)
class ObjectDiffView:
    change_count: int
    changes: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class TraceView:
    trace_id: str
    trace_type: str
    summary: str
    evidence_chain: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class EvidenceChainView:
    target_id: str
    chain: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class PublishDecisionView:
    field_key: str
    status: str
    confidence: float
    reason: str
    evidence_refs: List[Dict[str, str]] = field(default_factory=list)
