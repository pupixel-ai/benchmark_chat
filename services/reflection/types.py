from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List


@dataclass
class ReflectionAssetPaths:
    root_dir: str
    observation_cases_path: str
    case_facts_path: str
    profile_field_gt_path: str
    gt_comparisons_path: str
    profile_field_trace_index_path: str
    profile_field_trace_payload_dir: str
    human_review_records_path: str
    upstream_patterns_path: str
    upstream_experiments_path: str
    upstream_outcomes_path: str
    downstream_audit_patterns_path: str
    downstream_audit_experiments_path: str
    engineering_alerts_path: str
    difficult_cases_path: str
    difficult_case_actions_path: str
    decision_review_items_path: str
    tasks_path: str
    task_actions_path: str
    proposals_path: str
    proposal_actions_path: str
    engineering_change_requests_path: str
    reflection_feedback_path: str
    experiments_dir: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class ObservationCase:
    case_id: str
    user_name: str
    album_id: str
    stage: str
    dimension: str
    entity_type: str
    entity_id: str
    signal_source: str
    first_seen_stage: str
    surfaced_stage: str
    decision_trace: Dict[str, Any] = field(default_factory=dict)
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    tool_usage_summary: Dict[str, Any] = field(default_factory=dict)
    guardrail_trigger: List[str] = field(default_factory=list)
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CaseFact:
    case_id: str
    user_name: str
    album_id: str
    entity_type: str
    entity_id: str
    dimension: str
    signal_source: str
    first_seen_stage: str
    surfaced_stage: str
    routing_result: str
    business_priority: str
    auto_confidence: float
    accuracy_gap_status: str = ""
    resolution_route: str = ""
    badcase_source: str = ""
    badcase_kind: str = ""
    triage_reason: str = ""
    root_cause_family: str = ""
    fix_surface_confidence: float = 0.0
    causality_route: str = ""
    audit_action_type: str = ""
    trace_payload_path: str = ""
    comparison_grade: str = ""
    comparison_score: float = 0.0
    comparison_method: str = ""
    pre_audit_comparison_grade: str = ""
    pre_audit_comparison_score: float = 0.0
    pre_audit_comparison_method: str = ""
    agent_reasoning_summary: str = ""
    why_not_other_surfaces: str = ""
    decision_tree_path: List[str] = field(default_factory=list)
    proposal_id: str = ""
    support_count: int = 0
    decision_trace: Dict[str, Any] = field(default_factory=dict)
    tool_usage_summary: Dict[str, Any] = field(default_factory=dict)
    pre_audit_output: Dict[str, Any] = field(default_factory=dict)
    upstream_output: Dict[str, Any] = field(default_factory=dict)
    gt_payload: Dict[str, Any] = field(default_factory=dict)
    pre_audit_comparison_result: Dict[str, Any] = field(default_factory=dict)
    comparison_result: Dict[str, Any] = field(default_factory=dict)
    downstream_challenge: Dict[str, Any] = field(default_factory=dict)
    downstream_v2: Dict[str, Any] = field(default_factory=dict)
    downstream_judge: Dict[str, Any] = field(default_factory=dict)
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HumanReviewRecord:
    review_id: str
    case_id: str
    user_name: str
    review_type: str
    status: str = "pending"
    reviewer: str = ""
    decision: str = ""
    notes: str = ""
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PatternCluster:
    pattern_id: str
    user_name: str
    lane: str
    business_priority: str
    root_cause_candidates: List[str]
    fix_surface_candidates: List[str]
    support_case_ids: List[str]
    is_direction_clear: bool
    entity_type: str = ""
    dimension: str = ""
    triage_reason: str = ""
    album_id: str = ""
    support_count: int = 0
    summary: str = ""
    fix_surface_confidence: float = 0.0
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "new"
    why_blocked: str = ""
    recommended_option: str = ""
    eligible_for_task: bool = False
    resolution_route: str = ""
    comparison_grade: str = ""
    has_trace_payload: bool = False
    history_pattern_ids: List[str] = field(default_factory=list)
    history_experiment_ids: List[str] = field(default_factory=list)
    history_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionReviewItem:
    task_id: str
    task_type: str
    pattern_id: str
    user_name: str
    lane: str
    priority: str
    summary: str
    detail_url: str
    support_case_ids: List[str]
    options: List[str]
    recommended_option: str
    status: str = "new"
    resolved_option: str = ""
    reviewer_note: str = ""
    reviewed_by: str = ""
    last_action_type: str = ""
    created_at: str = ""
    updated_at: str = ""
    album_id: str = ""
    why_blocked: str = ""
    proposal_id: str = ""
    experiment_id: str = ""
    change_request_id: str = ""
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyExperiment:
    experiment_id: str
    pattern_id: str
    user_name: str
    lane: str
    fix_surface: str
    change_scope: str
    hypothesis: str
    status: str = "proposed"
    field_key: str = ""
    override_bundle_path: str = ""
    experiment_report_path: str = ""
    proposal_status: str = ""
    approval_required: bool = True
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    history_pattern_ids: List[str] = field(default_factory=list)
    history_experiment_ids: List[str] = field(default_factory=list)
    learning_summary: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DifficultCaseRecord:
    case_id: str
    user_name: str
    album_id: str
    dimension: str
    entity_type: str
    summary: str
    detail_url: str
    status: str = "new"
    comparison_grade: str = ""
    comparison_score: float = 0.0
    resolution_route: str = "difficult_case"
    trace_payload_path: str = ""
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentOutcome:
    outcome_id: str
    experiment_id: str
    user_name: str
    status: str
    summary: str
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProposalReviewRecord:
    proposal_id: str
    task_id: str
    experiment_id: str
    pattern_id: str
    user_name: str
    lane: str
    field_key: str
    fix_surface: str
    summary: str
    detail_url: str
    status: str = "pending_review"
    approval_required: bool = True
    recommended_option: str = "approve"
    options: List[str] = field(default_factory=lambda: ["approve", "reject", "need_revision"])
    agent_reasoning_summary: str = ""
    key_evidence_zh: List[str] = field(default_factory=list)
    why_this_surface_zh: str = ""
    why_not_other_surfaces: str = ""
    decision_tree_path: List[str] = field(default_factory=list)
    patch_intent: Dict[str, Any] = field(default_factory=dict)
    patch_preview: Dict[str, Any] = field(default_factory=dict)
    diff_summary: List[str] = field(default_factory=list)
    baseline_metrics: Dict[str, Any] = field(default_factory=dict)
    candidate_metrics: Dict[str, Any] = field(default_factory=dict)
    metric_gain: Dict[str, Any] = field(default_factory=dict)
    execution_path_recommendation: str = ""
    gt_value: Any = None
    current_output: Any = None
    candidate_output: Any = None
    result_delta_summary: str = ""
    override_bundle_path: str = ""
    experiment_report_path: str = ""
    validation_summary: Dict[str, Any] = field(default_factory=dict)
    proposal_status: str = "pending_review"
    resolved_option: str = ""
    reviewer_note: str = ""
    reviewed_by: str = ""
    last_action_type: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EngineeringChangeRequest:
    change_request_id: str
    task_id: str
    proposal_id: str
    experiment_id: str
    pattern_id: str
    user_name: str
    field_key: str
    fix_surface: str
    execution_path: str
    approved_scope: str
    detail_url: str
    change_summary_zh: str
    short_reason_zh: str
    gt_value: Any = None
    current_output: Any = None
    candidate_output: Any = None
    overlay_experiment_summary: str = ""
    key_evidence_zh: List[str] = field(default_factory=list)
    support_case_ids: List[str] = field(default_factory=list)
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)
    patch_preview: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending_review"
    reviewer_note: str = ""
    reviewed_by: str = ""
    last_action_type: str = ""
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EngineeringAlert:
    alert_id: str
    user_name: str
    album_id: str
    alert_type: str
    message: str
    stage: str
    severity: str = "high"
    status: str = "new"
    evidence_refs: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
