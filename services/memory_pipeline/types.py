from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PersonScreening:
    person_id: str
    person_kind: str
    memory_value: str
    screening_refs: List[Dict[str, Any]] = field(default_factory=list)
    block_reasons: List[str] = field(default_factory=list)
    no_contact_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryState:
    photos: List[Any]
    face_db: Dict[str, Any]
    vlm_results: List[Dict[str, Any]]
    screening: Dict[str, PersonScreening] = field(default_factory=dict)
    primary_decision: Dict[str, Any] | None = None
    primary_reflection: Dict[str, Any] | None = None
    events: List[Any] = field(default_factory=list)
    relationships: List[Any] = field(default_factory=list)
    relationship_dossiers: List["RelationshipDossier"] = field(default_factory=list)
    groups: List[Any] = field(default_factory=list)
    profile_context: Dict[str, Any] | None = None


@dataclass
class ProfileState:
    structured_profile: Dict[str, Any]
    resolved_tags: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    resolved_facts_summary: Dict[str, Any] = field(default_factory=dict)
    tool_cache: Dict[str, Any] = field(default_factory=dict)
    field_decisions: List[Dict[str, Any]] = field(default_factory=list)
    llm_batch_debug: List[Dict[str, Any]] = field(default_factory=list)
    consistency_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RelationshipDossier:
    person_id: str
    person_kind: str
    memory_value: str
    photo_count: int
    time_span_days: int
    recent_gap_days: int
    monthly_frequency: float
    scene_profile: Dict[str, Any]
    interaction_signals: List[str]
    shared_events: List[Dict[str, Any]]
    trend_detail: Dict[str, Any]
    co_appearing_persons: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    evidence_refs: List[Dict[str, Any]]
    block_reasons: List[str] = field(default_factory=list)
    retention_decision: str = "review"
    retention_reason: str = ""
    group_eligible: bool = False
    group_block_reason: Optional[str] = None
    group_weight: float = 0.0
    relationship_result: Dict[str, Any] = field(default_factory=dict)
    relationship_reflection: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroupArtifact:
    group_id: str
    members: List[str]
    group_type_candidate: str
    confidence: float
    strong_evidence_refs: List[Dict[str, Any]]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RelationshipTypeSpec:
    type_key: str
    allowed_evidence: List[str]
    strong_evidence: List[str]
    supporting_evidence: List[str]
    blocker_evidence: List[str]
    cot_steps: List[str]
    reflection_questions: List[str]
    downgrade_target: str | None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FieldSpec:
    field_key: str
    risk_level: str
    allowed_sources: List[str]
    strong_evidence: List[str]
    cot_steps: List[str]
    owner_resolution_steps: List[str]
    time_reasoning_steps: List[str]
    counter_evidence_checks: List[str]
    weak_evidence: List[str]
    hard_blocks: List[str]
    owner_checks: List[str]
    time_layer_rule: str
    weak_evidence_caution: List[str]
    reflection_questions: List[str]
    reflection_rounds: int
    requires_social_media: bool = False
    requires_protagonist_face: bool = False
    cot_hint: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FieldBundle:
    field_key: str
    field_spec: FieldSpec
    allowed_refs: Dict[str, List[Dict[str, Any]]]
    supporting_refs: Dict[str, List[Dict[str, Any]]]
    contradicting_refs: Dict[str, List[Dict[str, Any]]]
    gate_result: Dict[str, Any]
    null_preferred: bool
    reflection_questions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["field_spec"] = self.field_spec.to_dict()
        return payload


@dataclass
class FactFieldDecision:
    field_key: str
    gate_result: Dict[str, Any]
    draft: Dict[str, Any]
    reflection_1: Dict[str, Any]
    reflection_2: Dict[str, Any]
    final: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
