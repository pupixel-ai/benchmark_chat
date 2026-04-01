from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


from services.memory_pipeline.profile_fields import FIELD_SPECS
from services.memory_pipeline.rule_asset_loader import load_repo_rule_assets
from .gt import apply_profile_field_gt, auto_generate_pseudo_gt, load_profile_field_gt
from .storage import build_reflection_asset_paths, ensure_reflection_root
from .types import CaseFact, DecisionReviewItem, DifficultCaseRecord, EngineeringAlert, EngineeringChangeRequest, ExperimentOutcome, PatternCluster, ProposalReviewRecord, StrategyExperiment
from .upstream_agent import BadcasePacketAssembler, CoverageProbe, ExperimentPlanner, ProposalBuilder, UpstreamReflectionAgent
from .upstream_triage import UpstreamTriageScorer, recommend_fix_surface, resolve_accuracy_gap_route, retrieve_similar_patterns


UPSTREAM_TASK_OPTIONS = [
    "field_cot",
    "tool_rule",
    "call_policy",
    "engineering_issue",
    "watch_only",
]

DOWNSTREAM_TASK_OPTIONS = [
    "critic_rule",
    "judge_boundary",
    "need_more_evidence",
    "engineering_issue",
    "watch_only",
]

UPSTREAM_EXPERIMENT_SURFACES = {"field_cot", "tool_rule", "call_policy"}

PRIORITY_RANK = {
    "low": 1,
    "medium": 2,
    "high": 3,
}


def build_pattern_clusters(case_facts: Iterable[CaseFact | Dict[str, Any]]) -> Tuple[List[PatternCluster], List[PatternCluster], List[EngineeringAlert]]:
    grouped: Dict[Tuple[str, ...], List[CaseFact]] = defaultdict(list)
    engineering_alerts: Dict[str, EngineeringAlert] = {}

    for raw_fact in case_facts:
        fact = _coerce_case_fact(raw_fact)
        if fact.routing_result == "expected_uncertainty":
            continue
        if fact.routing_result == "engineering_issue":
            alert = _build_engineering_alert(fact)
            engineering_alerts[alert.alert_id] = alert
            continue
        if fact.routing_result not in {"strategy_candidate", "audit_disagreement"}:
            continue

        lane = _resolve_lane(fact)
        if not lane:
            continue
        grouped[_build_cluster_key(fact, lane)].append(fact)

    upstream_patterns: List[PatternCluster] = []
    downstream_patterns: List[PatternCluster] = []
    for cluster_key, facts in grouped.items():
        pattern = _build_pattern_cluster(cluster_key=cluster_key, facts=facts)
        if pattern.lane == "upstream":
            upstream_patterns.append(pattern)
        else:
            downstream_patterns.append(pattern)

    upstream_patterns.sort(key=_pattern_sort_key, reverse=True)
    downstream_patterns.sort(key=_pattern_sort_key, reverse=True)
    alerts = sorted(engineering_alerts.values(), key=lambda item: item.alert_id)
    return upstream_patterns, downstream_patterns, alerts


def build_decision_tasks(pattern_clusters: Iterable[PatternCluster | Dict[str, Any]]) -> List[DecisionReviewItem]:
    now = _utcnow_iso()
    tasks: List[DecisionReviewItem] = []
    for raw_pattern in pattern_clusters:
        pattern = _coerce_pattern_cluster(raw_pattern)
        if (
            pattern.resolution_route != "strategy_fix"
            or not pattern.eligible_for_task
            or pattern.business_priority != "high"
            or pattern.is_direction_clear
        ):
            continue

        task_type = "upstream_decision_task" if pattern.lane == "upstream" else "downstream_decision_task"
        task_id = _stable_id("task", f"{task_type}|{pattern.pattern_id}")
        options = list(UPSTREAM_TASK_OPTIONS if pattern.lane == "upstream" else DOWNSTREAM_TASK_OPTIONS)
        tasks.append(
            DecisionReviewItem(
                task_id=task_id,
                task_type=task_type,
                pattern_id=pattern.pattern_id,
                user_name=pattern.user_name,
                album_id=pattern.album_id,
                lane=pattern.lane,
                priority=pattern.business_priority,
                summary=pattern.summary,
                detail_url=f"/review/task/{task_id}",
                support_case_ids=list(pattern.support_case_ids),
                options=options,
                recommended_option=pattern.recommended_option,
                status="new",
                created_at=now,
                updated_at=now,
                why_blocked=pattern.why_blocked,
                evidence_refs=list(pattern.evidence_refs),
            )
        )
    tasks.sort(key=lambda item: item.task_id)
    return tasks


def build_strategy_experiments(pattern_clusters: Iterable[PatternCluster | Dict[str, Any]]) -> List[StrategyExperiment]:
    experiments: List[StrategyExperiment] = []
    for raw_pattern in pattern_clusters:
        pattern = _coerce_pattern_cluster(raw_pattern)
        allow_single_case_gt_mismatch = _can_single_case_lp3_gt_mismatch_enter_experiment(pattern)
        if (
            pattern.lane != "upstream"
            or pattern.resolution_route != "strategy_fix"
            or (not pattern.eligible_for_task and not allow_single_case_gt_mismatch)
            or pattern.business_priority != "high"
            or not pattern.is_direction_clear
            or pattern.recommended_option not in UPSTREAM_EXPERIMENT_SURFACES
            or pattern.comparison_grade == "partial_match"
        ):
            continue
        history_summary = dict(pattern.history_summary or {})
        if int(history_summary.get("open_recommended_experiment_count") or 0) > 0:
            continue
        experiments.append(
            StrategyExperiment(
                experiment_id=_stable_id("exp", pattern.pattern_id),
                pattern_id=pattern.pattern_id,
                user_name=pattern.user_name,
                lane=pattern.lane,
                fix_surface=pattern.recommended_option,
                change_scope="single_pattern",
                hypothesis=_build_experiment_hypothesis(pattern),
                status="proposed",
                evidence_refs=list(pattern.evidence_refs),
                history_pattern_ids=list(pattern.history_pattern_ids),
                history_experiment_ids=list(pattern.history_experiment_ids),
                learning_summary=history_summary,
                metrics={
                    "support_count": pattern.support_count,
                    "fix_surface_confidence": pattern.fix_surface_confidence,
                    "history_pattern_count": len(pattern.history_pattern_ids),
                    "history_success_count": int(history_summary.get("recommended_option_success_count") or 0),
                    "history_failure_count": int(history_summary.get("recommended_option_failure_count") or 0),
                },
            )
        )
    experiments.sort(key=lambda item: item.experiment_id)
    return experiments


def _can_single_case_lp3_gt_mismatch_enter_experiment(pattern: PatternCluster) -> bool:
    return (
        pattern.lane == "upstream"
        and pattern.entity_type == "profile_field"
        and pattern.business_priority == "high"
        and pattern.resolution_route == "strategy_fix"
        and pattern.comparison_grade == "mismatch"
        and pattern.support_count >= 1
    )


def enrich_patterns_with_learning_history(
    *,
    patterns: Iterable[PatternCluster | Dict[str, Any]],
    historical_patterns: Iterable[PatternCluster | Dict[str, Any]],
    historical_experiments: Iterable[StrategyExperiment | Dict[str, Any]],
    historical_outcomes: Iterable[ExperimentOutcome | Dict[str, Any]],
) -> List[PatternCluster]:
    history_patterns_payloads = [_coerce_pattern_cluster(item).to_dict() for item in historical_patterns]
    history_experiments_payloads = [_coerce_strategy_experiment(item).to_dict() for item in historical_experiments]
    history_outcomes_payloads = [_coerce_experiment_outcome(item).to_dict() for item in historical_outcomes]

    enriched: List[PatternCluster] = []
    for raw_pattern in patterns:
        pattern = _coerce_pattern_cluster(raw_pattern)
        history = _build_pattern_learning_history(
            pattern=pattern,
            historical_patterns=history_patterns_payloads,
            historical_experiments=history_experiments_payloads,
            historical_outcomes=history_outcomes_payloads,
        )
        pattern.history_pattern_ids = list(history["history_pattern_ids"])
        pattern.history_experiment_ids = list(history["history_experiment_ids"])
        pattern.history_summary = dict(history["history_summary"])

        success_count = int(pattern.history_summary.get("recommended_option_success_count") or 0)
        failure_count = int(pattern.history_summary.get("recommended_option_failure_count") or 0)
        if success_count > failure_count and success_count > 0 and pattern.recommended_option in UPSTREAM_EXPERIMENT_SURFACES:
            pattern.fix_surface_confidence = round(min(0.97, pattern.fix_surface_confidence + min(success_count, 2) * 0.05), 4)
        if failure_count >= 2 and success_count == 0 and pattern.recommended_option in UPSTREAM_EXPERIMENT_SURFACES:
            pattern.is_direction_clear = False
            if pattern.eligible_for_task:
                pattern.why_blocked = "historical_failures_need_review"
        enriched.append(pattern)
    return enriched


def persist_reflection_tasks(
    *,
    project_root: str,
    user_name: str,
    upstream_patterns: Iterable[PatternCluster | Dict[str, Any]],
    downstream_patterns: Iterable[PatternCluster | Dict[str, Any]],
    tasks: Iterable[DecisionReviewItem | Dict[str, Any]],
    engineering_alerts: Iterable[EngineeringAlert | Dict[str, Any]],
    difficult_cases: Iterable[DifficultCaseRecord | Dict[str, Any]] | None = None,
    strategy_experiments: Iterable[StrategyExperiment | Dict[str, Any]] | None = None,
    proposals: Iterable[ProposalReviewRecord | Dict[str, Any]] | None = None,
    engineering_change_requests: Iterable[EngineeringChangeRequest | Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    paths = build_reflection_asset_paths(project_root=project_root, user_name=user_name)
    ensure_reflection_root(paths)

    upstream_payloads = [_coerce_pattern_cluster(item).to_dict() for item in upstream_patterns]
    downstream_payloads = [_coerce_pattern_cluster(item).to_dict() for item in downstream_patterns]
    alert_payloads = [_coerce_engineering_alert(item).to_dict() for item in engineering_alerts]
    task_payloads = [_coerce_decision_task(item).to_dict() for item in tasks]
    difficult_case_payloads = [_coerce_difficult_case(item).to_dict() for item in list(difficult_cases or [])]
    experiment_payloads = [_coerce_strategy_experiment(item).to_dict() for item in list(strategy_experiments or [])]
    proposal_payloads = [_coerce_proposal(item).to_dict() for item in list(proposals or [])]
    engineering_change_request_payloads = [_coerce_engineering_change_request(item).to_dict() for item in list(engineering_change_requests or [])]

    merged_upstream_patterns = _merge_records_by_id(
        existing_payloads=_read_json_array(paths.upstream_patterns_path),
        incoming_payloads=upstream_payloads,
        id_key="pattern_id",
    )
    merged_downstream_patterns = _merge_records_by_id(
        existing_payloads=_read_json_array(paths.downstream_audit_patterns_path),
        incoming_payloads=downstream_payloads,
        id_key="pattern_id",
    )
    merged_upstream_experiments = _merge_records_by_id(
        existing_payloads=_read_json_array(paths.upstream_experiments_path),
        incoming_payloads=experiment_payloads,
        id_key="experiment_id",
        preserve_fields={
            "status": {"keep_existing_when_incoming": {"proposed"}},
        },
    )

    _write_json_file(paths.upstream_patterns_path, merged_upstream_patterns)
    _write_json_file(paths.downstream_audit_patterns_path, merged_downstream_patterns)
    _write_json_file(paths.upstream_experiments_path, merged_upstream_experiments)
    _write_jsonl_snapshot(paths.engineering_alerts_path, alert_payloads)
    merged_difficult_cases = _merge_records_by_id(
        existing_payloads=_read_jsonl_records(paths.difficult_cases_path),
        incoming_payloads=difficult_case_payloads,
        id_key="case_id",
    )
    _write_jsonl_snapshot(paths.difficult_cases_path, merged_difficult_cases)

    task_actions_file = Path(paths.task_actions_path)
    task_actions_file.parent.mkdir(parents=True, exist_ok=True)
    task_actions_file.touch(exist_ok=True)
    proposal_actions_file = Path(paths.proposal_actions_path)
    proposal_actions_file.parent.mkdir(parents=True, exist_ok=True)
    proposal_actions_file.touch(exist_ok=True)
    difficult_case_actions_file = Path(paths.difficult_case_actions_path)
    difficult_case_actions_file.parent.mkdir(parents=True, exist_ok=True)
    difficult_case_actions_file.touch(exist_ok=True)
    engineering_change_requests_file = Path(paths.engineering_change_requests_path)
    engineering_change_requests_file.parent.mkdir(parents=True, exist_ok=True)
    engineering_change_requests_file.touch(exist_ok=True)
    reflection_feedback_file = Path(paths.reflection_feedback_path)
    reflection_feedback_file.parent.mkdir(parents=True, exist_ok=True)
    reflection_feedback_file.touch(exist_ok=True)
    merged_proposals = _merge_records_by_id(
        existing_payloads=_read_jsonl_records(paths.proposals_path),
        incoming_payloads=proposal_payloads,
        id_key="proposal_id",
        preserve_fields={
            "status": {"keep_existing_when_incoming": {"pending_review", "new"}},
        },
    )
    _write_jsonl_snapshot(paths.proposals_path, merged_proposals)
    merged_engineering_change_requests = _merge_records_by_id(
        existing_payloads=_read_jsonl_records(paths.engineering_change_requests_path),
        incoming_payloads=engineering_change_request_payloads,
        id_key="change_request_id",
        preserve_fields={
            "status": {"keep_existing_when_incoming": {"pending_review", "new"}},
        },
    )
    _write_jsonl_snapshot(paths.engineering_change_requests_path, merged_engineering_change_requests)

    existing_tasks = {payload.get("task_id"): payload for payload in _read_jsonl_records(paths.tasks_path)}
    for incoming in task_payloads:
        task_id = str(incoming.get("task_id") or "").strip()
        if not task_id:
            continue
        previous = dict(existing_tasks.get(task_id) or {})
        merged = dict(previous)
        merged.update(incoming)
        if previous.get("created_at"):
            merged["created_at"] = previous["created_at"]
        if previous.get("status") and incoming.get("status") == "new":
            merged["status"] = previous["status"]
        existing_tasks[task_id] = merged

    ordered_tasks = sorted(
        existing_tasks.values(),
        key=lambda payload: (_timestamp_rank(payload.get("updated_at")), str(payload.get("task_id") or "")),
        reverse=True,
    )
    _write_jsonl_snapshot(paths.tasks_path, ordered_tasks)

    return {
        "upstream_patterns_path": paths.upstream_patterns_path,
        "downstream_patterns_path": paths.downstream_audit_patterns_path,
        "engineering_alerts_path": paths.engineering_alerts_path,
        "difficult_cases_path": paths.difficult_cases_path,
        "tasks_path": paths.tasks_path,
        "task_actions_path": paths.task_actions_path,
        "upstream_experiments_path": paths.upstream_experiments_path,
        "proposals_path": paths.proposals_path,
        "proposal_actions_path": paths.proposal_actions_path,
        "engineering_change_requests_path": paths.engineering_change_requests_path,
        "reflection_feedback_path": paths.reflection_feedback_path,
        "written_upstream_pattern_count": len(upstream_payloads),
        "written_downstream_pattern_count": len(downstream_payloads),
        "written_engineering_alert_count": len(alert_payloads),
        "written_difficult_case_count": len(difficult_case_payloads),
        "written_task_count": len(task_payloads),
        "written_strategy_experiment_count": len(experiment_payloads),
        "written_proposal_count": len(proposal_payloads),
        "written_engineering_change_request_count": len(engineering_change_request_payloads),
    }


def run_reflection_task_generation(*, project_root: str, user_name: str) -> Dict[str, Any]:
    paths = build_reflection_asset_paths(project_root=project_root, user_name=user_name)
    ensure_reflection_root(paths)

    case_facts = load_case_facts(project_root=project_root, user_name=user_name)
    auto_generate_pseudo_gt(case_facts, paths.profile_field_gt_path)
    gt_records = load_profile_field_gt(paths.profile_field_gt_path)
    case_facts, gt_comparisons = apply_profile_field_gt(case_facts, gt_records)
    case_facts = _run_data_sufficiency_gate(case_facts)
    case_facts = _run_coverage_probe(case_facts, project_root=project_root)
    scorer = UpstreamTriageScorer()
    existing_upstream_patterns = _read_json_array(paths.upstream_patterns_path)
    existing_upstream_experiments = _read_json_array(paths.upstream_experiments_path)
    existing_upstream_outcomes = _read_json_array(paths.upstream_outcomes_path)
    enriched_case_facts = _enrich_upstream_case_facts(case_facts, scorer, existing_upstream_patterns)
    reflected_case_facts = _apply_upstream_agent_reflection(
        project_root=project_root,
        case_facts=enriched_case_facts,
    )

    # 保存旧 case_id → field_key 映射（覆写前），让旧 proposal 引用可追溯
    if paths.case_facts_path.exists():
        _save_case_id_mapping(paths.case_facts_path, user_name, reflected_case_facts)

    _write_jsonl_snapshot(paths.case_facts_path, [fact.to_dict() for fact in reflected_case_facts])
    _write_jsonl_snapshot(paths.gt_comparisons_path, gt_comparisons)

    upstream_patterns, downstream_patterns, engineering_alerts = build_pattern_clusters(reflected_case_facts)
    upstream_patterns = enrich_patterns_with_learning_history(
        patterns=upstream_patterns,
        historical_patterns=existing_upstream_patterns,
        historical_experiments=existing_upstream_experiments,
        historical_outcomes=existing_upstream_outcomes,
    )
    tasks = build_decision_tasks([*upstream_patterns, *downstream_patterns])
    difficult_cases = build_difficult_cases(reflected_case_facts)
    strategy_experiments, proposals, proposal_tasks = _plan_strategy_experiments_with_proposals(
        project_root=project_root,
        patterns=upstream_patterns,
        case_facts=reflected_case_facts,
    )
    tasks.extend(proposal_tasks)
    persisted = persist_reflection_tasks(
        project_root=project_root,
        user_name=user_name,
        upstream_patterns=upstream_patterns,
        downstream_patterns=downstream_patterns,
        tasks=tasks,
        engineering_alerts=engineering_alerts,
        difficult_cases=difficult_cases,
        strategy_experiments=strategy_experiments,
        proposals=proposals,
    )
    persisted["gt_comparisons_path"] = paths.gt_comparisons_path
    persisted["written_gt_comparison_count"] = len(gt_comparisons)
    persisted["case_facts_path"] = paths.case_facts_path
    return persisted




def _apply_upstream_agent_reflection(*, project_root: str, case_facts: Iterable[CaseFact]) -> List[CaseFact]:
    packet_assembler = BadcasePacketAssembler(project_root=project_root)
    reflection_agent = UpstreamReflectionAgent()
    reflected: List[CaseFact] = []
    for fact in case_facts:
        if fact.signal_source != "mainline_profile" or fact.accuracy_gap_status != "open":
            reflected.append(fact)
            continue
        packet = packet_assembler.assemble(fact)
        recall = packet_assembler.history_recall(packet)
        if str(recall.get("resolution_signal") or "").strip() == "already_fixed":
            reflected.append(fact)
            continue
        result = reflection_agent.reflect(packet)
        fact.agent_reasoning_summary = str(result.get("judgment_summary_zh") or "").strip()
        fact.why_not_other_surfaces = str(result.get("why_not_other_surfaces_zh") or "").strip()
        fact.decision_tree_path = list(result.get("decision_tree_path") or [])
        fact.tool_usage_summary = {
            **dict(fact.tool_usage_summary or {}),
            "agent_key_evidence_zh": list(result.get("key_evidence_zh") or []),
            "agent_why_this_surface_zh": str(result.get("why_this_surface_zh") or "").strip(),
        }
        if result.get("status") in {"ok", "needs_review"}:
            fact.root_cause_family = str(result.get("root_cause_family") or fact.root_cause_family or "").strip()
            fact.fix_surface_confidence = float(result.get("confidence") or fact.fix_surface_confidence or 0.0)
            fact.tool_usage_summary = {
                **dict(fact.tool_usage_summary or {}),
                "agent_recommended_fix_surface": str(result.get("recommended_fix_surface") or ""),
                "agent_patch_intent": dict(result.get("patch_intent") or {}),
                "agent_expected_metric_gain": dict(result.get("expected_metric_gain") or {}),
            }
            if result.get("status") == "needs_review":
                fact.resolution_route = "difficult_case"
            else:
                fact.resolution_route = resolve_accuracy_gap_route(fact)
        reflected.append(fact)
    return reflected


def _plan_strategy_experiments_with_proposals(
    *,
    project_root: str,
    patterns: Iterable[PatternCluster | Dict[str, Any]],
    case_facts: Iterable[CaseFact | Dict[str, Any]],
) -> Tuple[List[StrategyExperiment], List[ProposalReviewRecord | Dict[str, Any]], List[DecisionReviewItem]]:
    selected_experiments = build_strategy_experiments(patterns)
    pattern_lookup = {
        pattern.pattern_id: pattern
        for pattern in [_coerce_pattern_cluster(item) for item in patterns]
    }
    case_fact_lookup = {
        fact.case_id: fact
        for fact in [_coerce_case_fact(item) for item in case_facts]
    }
    planner = ExperimentPlanner(project_root=project_root)
    proposal_builder = ProposalBuilder()
    planned_experiments: List[StrategyExperiment] = []
    proposals: List[ProposalReviewRecord | Dict[str, Any]] = []
    proposal_tasks: List[DecisionReviewItem] = []

    for base_experiment in selected_experiments:
        pattern = pattern_lookup.get(base_experiment.pattern_id)
        if pattern is None:
            continue
        support_cases = [
            case_fact_lookup[case_id]
            for case_id in pattern.support_case_ids
            if case_id in case_fact_lookup
        ]
        agent_result = _build_pattern_agent_result(pattern=pattern, case_fact_lookup=case_fact_lookup)
        planned = planner.plan(pattern=pattern, agent_result=agent_result)
        report = planner.execute(
            pattern=pattern,
            experiment=planned,
            agent_result=agent_result,
            support_cases=support_cases,
        )
        planned.status = str(report.get("status") or planned.status).strip() or planned.status
        if report.get("is_significant_improvement"):
            planned.proposal_status = "pending_review"
        elif planned.status in {"need_revision", "failed"}:
            planned.proposal_status = planned.status
        proposal_payload = proposal_builder.build(
            pattern=pattern,
            experiment=planned,
            agent_result=agent_result,
            experiment_report=report,
            support_cases=support_cases,
        )
        if proposal_payload:
            planned.proposal_status = "pending_review"
            proposals.append(proposal_payload["proposal"])
            proposal_tasks.append(proposal_payload["task"])
        planned_experiments.append(planned)
    return planned_experiments, proposals, proposal_tasks


def _build_pattern_agent_result(*, pattern: PatternCluster, case_fact_lookup: Dict[str, CaseFact]) -> Dict[str, Any]:
    support_cases = [
        case_fact_lookup[case_id]
        for case_id in pattern.support_case_ids
        if case_id in case_fact_lookup
    ]
    representative = support_cases[0] if support_cases else None
    fix_surface = str(pattern.recommended_option or "watch_only")
    field_key = str(pattern.dimension or pattern.entity_type or "")
    patch_intent = {
        "field_key": field_key,
    }
    if fix_surface == "field_cot":
        patch_intent["field_spec_overrides"] = {
            field_key: {
                "cot_steps": [f"针对 {field_key} 补一条新的 agent 归纳步骤"],
            }
        }
    elif fix_surface == "tool_rule":
        patch_intent["tool_rules"] = {
            field_key: {
                "max_refs_per_source": 8,
            }
        }
    elif fix_surface == "call_policy":
        patch_intent["call_policies"] = {
            field_key: {
                "append_allowed_sources": ["feature"],
            }
        }
    return {
        "root_cause_family": str((representative.root_cause_family if representative else "") or (pattern.root_cause_candidates[0] if pattern.root_cause_candidates else "watch_only")),
        "recommended_fix_surface": fix_surface,
        "confidence": float((representative.fix_surface_confidence if representative else 0.0) or pattern.fix_surface_confidence or 0.0),
        "reason": str((representative.agent_reasoning_summary if representative else "") or pattern.summary or ""),
        "why_not_other_surfaces": str((representative.why_not_other_surfaces if representative else "") or "当前 pattern 已经形成单一主导改面，先不扩展到其他 surface。"),
        "decision_tree_path": list((representative.decision_tree_path if representative else []) or []),
        "patch_intent": patch_intent,
        "expected_metric_gain": {
            "exact_or_close_delta": max(pattern.support_count, 1),
            "mismatch_delta": -max(pattern.support_count, 1),
        },
    }


def load_case_facts(*, project_root: str, user_name: str) -> List[CaseFact]:
    paths = build_reflection_asset_paths(project_root=project_root, user_name=user_name)
    return [_coerce_case_fact(payload) for payload in _read_jsonl_records(paths.case_facts_path)]


def list_reflection_tasks_payload(*, project_root: str, user_name: str) -> Dict[str, Any]:
    tasks = load_reflection_tasks(project_root=project_root, user_name=user_name)
    return {
        "task_count": len(tasks),
        "tasks": tasks,
    }


def load_reflection_tasks(*, project_root: str, user_name: str) -> List[Dict[str, Any]]:
    paths = build_reflection_asset_paths(project_root=project_root, user_name=user_name)
    tasks = _read_jsonl_records(paths.tasks_path)
    tasks.sort(key=lambda payload: (_timestamp_rank(payload.get("updated_at")), str(payload.get("task_id") or "")), reverse=True)
    return tasks


def load_difficult_cases(*, project_root: str, user_name: str) -> List[Dict[str, Any]]:
    paths = build_reflection_asset_paths(project_root=project_root, user_name=user_name)
    difficult_cases = _read_jsonl_records(paths.difficult_cases_path)
    difficult_cases.sort(
        key=lambda payload: (_timestamp_rank(payload.get("updated_at")), str(payload.get("case_id") or "")),
        reverse=True,
    )
    return difficult_cases


def get_reflection_task_detail_payload(*, project_root: str, user_name: str, task_id: str) -> Dict[str, Any]:
    paths = build_reflection_asset_paths(project_root=project_root, user_name=user_name)
    task = next(
        (payload for payload in _read_jsonl_records(paths.tasks_path) if str(payload.get("task_id") or "") == task_id),
        None,
    )
    if not task:
        raise KeyError(task_id)

    pattern_by_id = {
        str(payload.get("pattern_id") or ""): payload
        for payload in [*_read_json_array(paths.upstream_patterns_path), *_read_json_array(paths.downstream_audit_patterns_path)]
    }
    current_pattern = pattern_by_id.get(str(task.get("pattern_id") or ""), {})
    support_case_ids = [str(case_id).strip() for case_id in list(task.get("support_case_ids") or []) if str(case_id).strip()]
    support_case_lookup = {
        str(payload.get("case_id") or ""): payload
        for payload in _read_jsonl_records(paths.case_facts_path)
    }
    support_cases = [support_case_lookup[case_id] for case_id in support_case_ids if case_id in support_case_lookup]
    trace_lookup = {
        str(payload.get("case_id") or ""): payload
        for payload in _read_jsonl_records(paths.profile_field_trace_index_path)
    }
    evidence_refs = list(task.get("evidence_refs") or current_pattern.get("evidence_refs") or _merge_evidence_refs_from_cases(support_cases))
    stage_trace = [
        {
            "case_id": payload.get("case_id"),
            "first_seen_stage": payload.get("first_seen_stage"),
            "surfaced_stage": payload.get("surfaced_stage"),
            "signal_source": payload.get("signal_source"),
            "triage_reason": payload.get("triage_reason"),
            "business_priority": payload.get("business_priority"),
        }
        for payload in support_cases
    ]
    history_patterns_all = _read_json_array(paths.upstream_patterns_path) if str(task.get("lane") or "") == "upstream" else []
    history_experiments_all = _read_json_array(paths.upstream_experiments_path) if str(task.get("lane") or "") == "upstream" else []
    history_outcomes_all = _read_json_array(paths.upstream_outcomes_path) if str(task.get("lane") or "") == "upstream" else []
    history = _build_pattern_learning_history(
        pattern=_coerce_pattern_cluster(current_pattern) if current_pattern else PatternCluster(
            pattern_id=str(task.get("pattern_id") or ""),
            user_name=user_name,
            lane=str(task.get("lane") or ""),
            business_priority=str(task.get("priority") or ""),
            root_cause_candidates=[],
            fix_surface_candidates=[],
            support_case_ids=support_case_ids,
            is_direction_clear=False,
            summary=str(task.get("summary") or ""),
            recommended_option=str(task.get("recommended_option") or ""),
        ),
        historical_patterns=history_patterns_all,
        historical_experiments=history_experiments_all,
        historical_outcomes=history_outcomes_all,
    )
    history_pattern_lookup = {str(payload.get("pattern_id") or ""): payload for payload in history_patterns_all}
    history_experiment_lookup = {str(payload.get("experiment_id") or ""): payload for payload in history_experiments_all}
    history_outcome_lookup = {str(payload.get("outcome_id") or ""): payload for payload in history_outcomes_all}
    proposal_lookup = {
        str(payload.get("proposal_id") or ""): payload
        for payload in _read_jsonl_records(paths.proposals_path)
    }
    proposal = proposal_lookup.get(str(task.get("proposal_id") or ""), {})
    change_request_lookup = {
        str(payload.get("change_request_id") or ""): payload
        for payload in _read_jsonl_records(paths.engineering_change_requests_path)
    }
    change_request = change_request_lookup.get(str(task.get("change_request_id") or ""), {})
    experiment_report = {}
    experiment_report_path = str(proposal.get("experiment_report_path") or "").strip()
    if experiment_report_path:
        experiment_report = _load_json_object(experiment_report_path)
    return {
        "task": task,
        "pattern": current_pattern,
        "proposal": {
            **proposal,
            "experiment_report": experiment_report,
        } if proposal else {},
        "change_request": change_request,
        "support_cases": support_cases,
        "evidence_refs": evidence_refs,
        "stage_trace": stage_trace,
        "gt_comparisons": [
            {
                "case_id": payload.get("case_id"),
                **dict(payload.get("comparison_result") or {}),
            }
            for payload in support_cases
            if payload.get("comparison_result")
        ],
        "field_trace_summaries": [
            trace_lookup[case_id]
            for case_id in support_case_ids
            if case_id in trace_lookup
        ],
        "route_decisions": [
            {
                "case_id": payload.get("case_id"),
                "resolution_route": payload.get("resolution_route"),
                "accuracy_gap_status": payload.get("accuracy_gap_status"),
                "comparison_grade": payload.get("comparison_grade"),
                "pre_audit_comparison_grade": payload.get("pre_audit_comparison_grade"),
                "causality_route": payload.get("causality_route"),
                "audit_action_type": payload.get("audit_action_type"),
            }
            for payload in support_cases
        ],
        "history_summary": history["history_summary"],
        "history_patterns": [
            history_pattern_lookup[pattern_id]
            for pattern_id in history["history_pattern_ids"]
            if pattern_id in history_pattern_lookup
        ],
        "history_experiments": [
            history_experiment_lookup[experiment_id]
            for experiment_id in history["history_experiment_ids"]
            if experiment_id in history_experiment_lookup
        ],
        "history_outcomes": [
            history_outcome_lookup[outcome_id]
            for outcome_id in history["history_outcome_ids"]
            if outcome_id in history_outcome_lookup
        ],
    }


def get_difficult_case_detail_payload(*, project_root: str, user_name: str, case_id: str) -> Dict[str, Any]:
    paths = build_reflection_asset_paths(project_root=project_root, user_name=user_name)
    case_record = next(
        (payload for payload in _read_jsonl_records(paths.difficult_cases_path) if str(payload.get("case_id") or "") == case_id),
        None,
    )
    if not case_record:
        raise KeyError(case_id)

    case_payload = next(
        (payload for payload in _read_jsonl_records(paths.case_facts_path) if str(payload.get("case_id") or "") == case_id),
        {},
    )
    trace_summary = next(
        (payload for payload in _read_jsonl_records(paths.profile_field_trace_index_path) if str(payload.get("case_id") or "") == case_id),
        {},
    )
    trace_payload_path = str(
        case_record.get("trace_payload_path")
        or case_payload.get("trace_payload_path")
        or trace_summary.get("trace_payload_path")
        or ""
    )
    trace_payload = {}
    if trace_payload_path:
        trace_file = Path(trace_payload_path)
        if trace_file.exists():
            try:
                trace_payload = json.loads(trace_file.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                trace_payload = {}

    comparison_payload = dict(case_payload.get("comparison_result") or {})
    if "output_value" not in comparison_payload:
        comparison_payload["output_value"] = (case_payload.get("upstream_output") or {}).get("value")
    if "gt_value" not in comparison_payload:
        comparison_payload["gt_value"] = (case_payload.get("gt_payload") or {}).get("gt_value")

    return {
        "case": case_record,
        "case_fact": case_payload,
        "gt_comparison": comparison_payload,
        "trace_summary": trace_summary,
        "trace_payload": trace_payload,
        "evidence_refs": list(case_record.get("evidence_refs") or case_payload.get("evidence_refs") or []),
        "difficulty_reason": _build_difficult_case_reason(case_record=case_record, case_payload=case_payload, gt_comparison=comparison_payload),
        "route_decision": {
            "resolution_route": str(case_payload.get("resolution_route") or case_record.get("resolution_route") or ""),
            "accuracy_gap_status": str(case_payload.get("accuracy_gap_status") or ""),
            "comparison_grade": str(case_payload.get("comparison_grade") or case_record.get("comparison_grade") or ""),
            "pre_audit_comparison_grade": str(case_payload.get("pre_audit_comparison_grade") or ""),
            "causality_route": str(case_payload.get("causality_route") or ""),
            "audit_action_type": str(case_payload.get("audit_action_type") or ""),
        },
    }


_MODAL_MISSING_NULL_SIGNALS = (
    "requires_social_media",
    "cleared_by_field_gate",
)


def _is_data_insufficient(fact: CaseFact, group_facts: List[CaseFact]) -> Tuple[bool, str]:
    tool_trace = dict((fact.decision_trace or {}).get("tool_trace") or {})
    evidence_bundle = dict(tool_trace.get("evidence_bundle") or {})

    null_reason = str((fact.upstream_output or {}).get("null_reason") or "")
    if any(sig in null_reason for sig in _MODAL_MISSING_NULL_SIGNALS):
        return True, f"null_reason_modal_missing:{null_reason}"

    total_refs = sum(
        len(v) if isinstance(v, list) else int((v or {}).get("hit_count") or 0)
        for v in evidence_bundle.values()
    )
    if total_refs == 0:
        group_ref_counts = []
        for gf in group_facts:
            gf_bundle = dict((gf.decision_trace or {}).get("tool_trace", {}).get("evidence_bundle") or {})
            group_ref_counts.append(sum(
                len(v) if isinstance(v, list) else int((v or {}).get("hit_count") or 0)
                for v in gf_bundle.values()
            ))
        if any(c > 0 for c in group_ref_counts):
            return False, ""
        return True, "group_level_zero_refs"

    return False, ""


def _run_data_sufficiency_gate(case_facts: Iterable[CaseFact]) -> List[CaseFact]:
    facts = list(case_facts)
    domain_to_facts: Dict[str, List[CaseFact]] = {}
    for fact in facts:
        spec = FIELD_SPECS.get(fact.dimension)
        domain = str(getattr(spec, "domain", None) or "").strip() if spec else ""
        domain_to_facts.setdefault(domain or fact.dimension, []).append(fact)

    gated: List[CaseFact] = []
    for fact in facts:
        if fact.signal_source != "mainline_profile" or fact.accuracy_gap_status != "open":
            gated.append(fact)
            continue
        spec = FIELD_SPECS.get(fact.dimension)
        domain = str(getattr(spec, "domain", None) or "").strip() if spec else fact.dimension
        group_facts = [f for f in domain_to_facts.get(domain or fact.dimension, []) if f.case_id != fact.case_id]
        is_insufficient, reason = _is_data_insufficient(fact, group_facts)
        if is_insufficient:
            fact.routing_result = "difficult_case"
            fact.resolution_route = "difficult_case"
            fact.triage_reason = f"data_insufficient:{reason}"
            fact.tool_usage_summary = {
                **dict(fact.tool_usage_summary or {}),
                "difficult_case_reason": "data_insufficient",
                "data_insufficiency_reason": reason,
            }
        gated.append(fact)
    return gated


def _run_coverage_probe(case_facts: Iterable[CaseFact], *, project_root: str) -> List[CaseFact]:
    probe = CoverageProbe()
    rule_assets = load_repo_rule_assets(project_root=project_root)
    call_policies = dict(rule_assets.get("call_policies") or {})
    tool_rules = dict(rule_assets.get("tool_rules") or {})

    facts = list(case_facts)
    domain_to_facts: Dict[str, List[CaseFact]] = {}
    for fact in facts:
        spec = FIELD_SPECS.get(fact.dimension)
        domain = str(getattr(spec, "domain", None) or "").strip() if spec else ""
        domain_to_facts.setdefault(domain or fact.dimension, []).append(fact)

    probed: List[CaseFact] = []
    for fact in facts:
        if fact.signal_source != "mainline_profile" or fact.accuracy_gap_status != "open":
            probed.append(fact)
            continue
        spec = FIELD_SPECS.get(fact.dimension)
        if spec is None:
            probed.append(fact)
            continue
        allowed_sources = list(getattr(spec, "allowed_sources", []) or [])
        tool_trace = dict((fact.decision_trace or {}).get("tool_trace") or {})
        domain = str(getattr(spec, "domain", None) or "").strip() or fact.dimension
        group_traces = [
            dict((f.decision_trace or {}).get("tool_trace") or {})
            for f in domain_to_facts.get(domain, [])
            if f.case_id != fact.case_id
        ]
        coverage_gap = probe.probe(
            field_key=fact.dimension,
            tool_trace=tool_trace,
            allowed_sources=allowed_sources,
            call_policies=call_policies,
            tool_rules=tool_rules,
            group_tool_traces=group_traces or None,
        )
        if coverage_gap["has_gap"]:
            fact.tool_usage_summary = {
                **dict(fact.tool_usage_summary or {}),
                "coverage_gap": coverage_gap,
            }
        probed.append(fact)
    return probed


def _enrich_upstream_case_facts(
    case_facts: Iterable[CaseFact],
    scorer: UpstreamTriageScorer,
    existing_upstream_patterns: Iterable[Dict[str, Any]],
) -> List[CaseFact]:
    enriched: List[CaseFact] = []
    for fact in case_facts:
        if fact.signal_source != "mainline_profile":
            enriched.append(fact)
            continue
        similar_patterns = retrieve_similar_patterns(fact, existing_upstream_patterns)
        enriched.append(scorer.enrich_case(fact, similar_patterns))
    return enriched


def _build_pattern_cluster(*, cluster_key: Tuple[str, ...], facts: List[CaseFact]) -> PatternCluster:
    first = facts[0]
    lane = cluster_key[0]
    support_case_ids = _dedupe_strings([fact.case_id for fact in facts])
    root_cause_candidates = _dedupe_strings([fact.root_cause_family or fact.triage_reason for fact in facts])
    evidence_refs = _dedupe_evidence_refs(ref for fact in facts for ref in fact.evidence_refs)
    support_count = len(support_case_ids)
    business_priority = max((fact.business_priority for fact in facts), key=lambda item: PRIORITY_RANK.get(item, 0))
    recommendation = recommend_fix_surface(facts)
    evidence_complete = len(evidence_refs) >= 2 and any(_case_has_reason(fact) for fact in facts)
    eligible_for_task = (
        first.resolution_route == "strategy_fix"
        and business_priority == "high"
        and support_count >= 2
        and evidence_complete
    )
    why_blocked = ""
    if first.resolution_route == "difficult_case":
        why_blocked = "difficult_case"
    elif business_priority != "high":
        why_blocked = "low_priority"
    elif support_count < 2:
        why_blocked = "insufficient_support_cases"
    elif not evidence_complete:
        why_blocked = "incomplete_evidence_package"
    elif not recommendation["is_direction_clear"]:
        why_blocked = "direction_not_clear"

    fix_surface_candidates = _ordered_fix_surface_candidates(
        lane=lane,
        recommended_option=recommendation["recommended_option"],
        entity_type=first.entity_type,
    )

    return PatternCluster(
        pattern_id=_stable_id("pattern", "|".join(cluster_key)),
        user_name=first.user_name,
        lane=lane,
        business_priority=business_priority,
        root_cause_candidates=root_cause_candidates,
        fix_surface_candidates=fix_surface_candidates,
        support_case_ids=support_case_ids,
        is_direction_clear=bool(recommendation["is_direction_clear"]),
        entity_type=first.entity_type,
        dimension=first.dimension,
        triage_reason=first.triage_reason,
        album_id=first.album_id,
        support_count=support_count,
        summary=_summarize_pattern(first=first, lane=lane, support_count=support_count),
        fix_surface_confidence=float(recommendation["fix_surface_confidence"]),
        evidence_refs=evidence_refs,
        status="new",
        why_blocked=why_blocked,
        recommended_option=str(recommendation["recommended_option"]),
        eligible_for_task=eligible_for_task,
        resolution_route=first.resolution_route,
        comparison_grade=first.comparison_grade,
        has_trace_payload=bool(first.trace_payload_path),
    )


def _build_cluster_key(fact: CaseFact, lane: str) -> Tuple[str, ...]:
    if lane == "downstream":
        return (
            lane,
            fact.entity_type,
            fact.dimension,
            str((fact.downstream_judge or {}).get("verdict") or ""),
            fact.triage_reason,
        )

    if fact.entity_type == "primary_person":
        return (lane, fact.entity_type, fact.triage_reason)

    if fact.entity_type == "relationship_candidate":
        return (
            lane,
            fact.entity_type,
            str((fact.decision_trace or {}).get("retention_reason") or ""),
            str((fact.decision_trace or {}).get("relationship_type") or ""),
        )

    if fact.entity_type == "profile_field":
        return (
            lane,
            fact.dimension,
            fact.root_cause_family or fact.triage_reason,
            fact.badcase_source or "history_recurrence",
            fact.resolution_route or "unrouted",
            fact.comparison_grade or "no_grade",
            "with_trace" if fact.trace_payload_path else "without_trace",
        )

    return (lane, fact.entity_type, fact.dimension, fact.triage_reason)


def _resolve_lane(fact: CaseFact) -> str:
    if fact.signal_source == "mainline_profile" and fact.causality_route in {"downstream_caused", "downstream_exacerbated"}:
        return "downstream"
    signal_source = fact.signal_source
    if signal_source in {"mainline_primary", "mainline_relationship", "mainline_profile"}:
        return "upstream"
    if signal_source == "downstream_audit":
        return "downstream"
    return ""


def _ordered_fix_surface_candidates(*, lane: str, recommended_option: str, entity_type: str) -> List[str]:
    base = list(_default_fix_surface_candidates(lane=lane, entity_type=entity_type))
    if recommended_option in base:
        base.remove(recommended_option)
        return [recommended_option, *base]
    return base


def _default_fix_surface_candidates(*, lane: str, entity_type: str) -> List[str]:
    if lane == "upstream":
        if entity_type == "profile_field":
            return ["field_cot", "tool_rule", "call_policy"]
        return ["tool_rule", "call_policy", "field_cot"]

    if entity_type == "profile_field":
        return ["critic_rule", "judge_boundary", "need_more_evidence"]
    return ["judge_boundary", "critic_rule", "need_more_evidence"]


def _summarize_pattern(*, first: CaseFact, lane: str, support_count: int) -> str:
    if lane == "upstream":
        if first.entity_type == "profile_field":
            if first.resolution_route == "difficult_case":
                return f"{first.dimension} 字段与 GT 存在 {first.comparison_grade or '复杂'} 偏差，当前归因不稳定，已转疑难 case"
            if first.badcase_source == "empty_output_candidate":
                return f"{first.dimension} 字段在 {support_count} 个 case 中反复输出为空，值得做上游反思"
            if first.badcase_source == "gt_mismatch_candidate":
                return f"{first.dimension} 字段在 {support_count} 个 case 中与 GT 不一致"
            return f"{first.dimension} 字段在 {support_count} 个 case 中形成稳定反思模式"
        if first.entity_type == "primary_person":
            return f"主角判断在 {support_count} 个 case 中持续出现可复用模式"
        if first.entity_type == "relationship_candidate":
            return f"{first.entity_id} 的关系判断在 {support_count} 个 case 中反复出现稳定信号"
        return f"{first.dimension} 在上游链路中形成 {support_count} 个可聚类 case"

    if first.entity_type == "profile_field":
        if first.causality_route in {"downstream_caused", "downstream_exacerbated"}:
            return f"{first.dimension} 字段在下游回流后与 GT 偏离，当前判定为 {first.causality_route}"
        return f"{first.dimension} 在下游裁决中被 {support_count} 个 case 反复挑战"
    if first.entity_type == "protagonist_tag":
        return f"主角裁决在下游被 {support_count} 个 case 反复挑战"
    if first.entity_type == "relationship_tag":
        return f"关系裁决在下游被 {support_count} 个 case 反复挑战"
    return f"{first.dimension} 在下游审计中形成 {support_count} 个稳定挑战样本"


def _build_experiment_hypothesis(pattern: PatternCluster) -> str:
    history_summary = dict(pattern.history_summary or {})
    success_count = int(history_summary.get("recommended_option_success_count") or 0)
    failure_count = int(history_summary.get("recommended_option_failure_count") or 0)
    history_note = ""
    if success_count > 0:
        history_note = f" 历史上同类 pattern 已有 {success_count} 次成功实验。"
    elif failure_count > 0:
        history_note = f" 历史上同类 pattern 已有 {failure_count} 次失败实验，需要更谨慎验证。"
    return (
        f"如果针对 {pattern.dimension} 只修改 {pattern.recommended_option}，"
        f"那么应能减少当前 pattern 中 {pattern.support_count} 个 case 的重复问题。"
        f"{history_note}"
    )


def _build_pattern_learning_history(
    *,
    pattern: PatternCluster,
    historical_patterns: Iterable[Dict[str, Any]],
    historical_experiments: Iterable[Dict[str, Any]],
    historical_outcomes: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    if pattern.lane != "upstream":
        return {
            "history_pattern_ids": [],
            "history_experiment_ids": [],
            "history_outcome_ids": [],
            "history_summary": {
                "similar_pattern_count": 0,
                "recommended_option_success_count": 0,
                "recommended_option_failure_count": 0,
                "open_recommended_experiment_count": 0,
            },
        }

    ranked_patterns: List[Tuple[int, Dict[str, Any]]] = []
    for payload in historical_patterns:
        historical_pattern_id = str(payload.get("pattern_id") or "").strip()
        if not historical_pattern_id or historical_pattern_id == pattern.pattern_id:
            continue
        if str(payload.get("lane") or "") != pattern.lane:
            continue
        score = _historical_pattern_match_score(pattern, payload)
        if score > 0:
            ranked_patterns.append((score, payload))
    ranked_patterns.sort(key=lambda item: (item[0], _timestamp_rank(item[1].get("updated_at"))), reverse=True)
    similar_patterns = [payload for _, payload in ranked_patterns[:5]]
    similar_pattern_ids = [str(payload.get("pattern_id") or "") for payload in similar_patterns]

    relevant_experiments: List[Dict[str, Any]] = []
    for payload in historical_experiments:
        experiment_pattern_id = str(payload.get("pattern_id") or "").strip()
        if experiment_pattern_id in similar_pattern_ids:
            relevant_experiments.append(payload)
    experiment_ids = [str(payload.get("experiment_id") or "") for payload in relevant_experiments if str(payload.get("experiment_id") or "").strip()]
    outcome_by_experiment = _group_outcomes_by_experiment(historical_outcomes)
    relevant_outcomes = [
        outcome
        for experiment_id in experiment_ids
        for outcome in outcome_by_experiment.get(experiment_id, [])
    ]
    outcome_ids = [str(payload.get("outcome_id") or "") for payload in relevant_outcomes if str(payload.get("outcome_id") or "").strip()]

    recommended_success_count = 0
    recommended_failure_count = 0
    open_recommended_experiment_count = 0
    for payload in relevant_experiments:
        if str(payload.get("fix_surface") or "") != pattern.recommended_option:
            continue
        status = str(payload.get("status") or "").strip().lower()
        if status in {"proposed", "running", "pending", "new"}:
            open_recommended_experiment_count += 1
        for outcome in outcome_by_experiment.get(str(payload.get("experiment_id") or ""), []):
            normalized = _normalize_outcome_status(outcome.get("status"))
            if normalized == "success":
                recommended_success_count += 1
            elif normalized == "failed":
                recommended_failure_count += 1

    return {
        "history_pattern_ids": similar_pattern_ids,
        "history_experiment_ids": experiment_ids,
        "history_outcome_ids": outcome_ids,
        "history_summary": {
            "similar_pattern_count": len(similar_pattern_ids),
            "recommended_option_success_count": recommended_success_count,
            "recommended_option_failure_count": recommended_failure_count,
            "open_recommended_experiment_count": open_recommended_experiment_count,
        },
    }


def _historical_pattern_match_score(pattern: PatternCluster, payload: Dict[str, Any]) -> int:
    same_dimension = str(payload.get("dimension") or "") == pattern.dimension and bool(pattern.dimension)
    root_cause_overlap = bool(
        set(str(item).strip() for item in list(payload.get("root_cause_candidates") or []) if str(item).strip())
        & set(pattern.root_cause_candidates)
    )
    if not same_dimension and not root_cause_overlap:
        return 0

    score = 0
    if same_dimension:
        score += 3
    if str(payload.get("entity_type") or "") == pattern.entity_type and pattern.entity_type:
        score += 1
    if str(payload.get("triage_reason") or "") == pattern.triage_reason and pattern.triage_reason:
        score += 1
    if str(payload.get("recommended_option") or "") == pattern.recommended_option and pattern.recommended_option:
        score += 1
    if root_cause_overlap:
        score += 2
    return score


def _group_outcomes_by_experiment(historical_outcomes: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for payload in historical_outcomes:
        experiment_id = str(payload.get("experiment_id") or "").strip()
        if experiment_id:
            grouped[experiment_id].append(payload)
    return grouped


def _normalize_outcome_status(status: Any) -> str:
    text = str(status or "").strip().lower()
    if text in {"success", "succeeded", "improved", "landed"}:
        return "success"
    if text in {"failed", "failure", "regressed", "rejected"}:
        return "failed"
    return text


def _build_engineering_alert(fact: CaseFact) -> EngineeringAlert:
    reason = _extract_case_reason(fact) or fact.triage_reason or "reflection engineering issue"
    return EngineeringAlert(
        alert_id=_stable_id("alert", f"{fact.case_id}|{fact.entity_type}|{fact.triage_reason}"),
        user_name=fact.user_name,
        album_id=fact.album_id,
        alert_type="engineering_issue",
        message=reason,
        stage=fact.surfaced_stage,
        severity="high" if fact.business_priority == "high" else "medium",
        status="new",
        evidence_refs=list(fact.evidence_refs),
    )


def _extract_case_reason(fact: CaseFact) -> str:
    for candidate in (
        (fact.decision_trace or {}).get("reason"),
        (fact.decision_trace or {}).get("reasoning"),
        (fact.downstream_judge or {}).get("reason"),
        (fact.downstream_challenge or {}).get("reason"),
    ):
        value = str(candidate or "").strip()
        if value:
            return value
    return ""


def _case_has_reason(fact: CaseFact) -> bool:
    return bool(_extract_case_reason(fact))


def _merge_evidence_refs_from_cases(support_cases: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _dedupe_evidence_refs(
        ref
        for payload in support_cases
        for ref in list(payload.get("evidence_refs") or [])
    )


def _dedupe_evidence_refs(evidence_refs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for raw_ref in evidence_refs:
        ref = dict(raw_ref or {})
        identity = (
            str(ref.get("source_type") or ""),
            str(ref.get("source_id") or ""),
            str(ref.get("description") or ""),
            tuple(str(item).strip() for item in list(ref.get("feature_names") or []) if str(item).strip()),
        )
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(ref)
    return normalized


def _dedupe_strings(values: Iterable[str]) -> List[str]:
    resolved: List[str] = []
    seen = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        resolved.append(text)
    return resolved


def _read_jsonl_records(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    payloads: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
    return payloads


def _read_json_array(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _load_json_object(path: str) -> Dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _merge_records_by_id(
    *,
    existing_payloads: Iterable[Dict[str, Any]],
    incoming_payloads: Iterable[Dict[str, Any]],
    id_key: str,
    preserve_fields: Dict[str, Dict[str, set[str]]] | None = None,
) -> List[Dict[str, Any]]:
    merged = {
        str(payload.get(id_key) or "").strip(): dict(payload)
        for payload in existing_payloads
        if str(payload.get(id_key) or "").strip()
    }
    preserve_fields = preserve_fields or {}
    for incoming in incoming_payloads:
        record_id = str(incoming.get(id_key) or "").strip()
        if not record_id:
            continue
        previous = dict(merged.get(record_id) or {})
        payload = dict(previous)
        payload.update(incoming)
        for field_name, policy in preserve_fields.items():
            incoming_value = str(incoming.get(field_name) or "").strip()
            existing_value = str(previous.get(field_name) or "").strip()
            keep_existing_when_incoming = set(policy.get("keep_existing_when_incoming") or set())
            if existing_value and incoming_value in keep_existing_when_incoming:
                payload[field_name] = previous[field_name]
        merged[record_id] = payload
    return sorted(merged.values(), key=lambda payload: str(payload.get(id_key) or ""))


def _write_json_file(path: str, payload: List[Dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_case_id_mapping(case_facts_path: str, user_name: str, new_facts: List[CaseFact]) -> None:
    """Save old→new case_id mapping before overwriting case_facts."""
    old_ids: Dict[str, str] = {}
    path = Path(case_facts_path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                old = json.loads(line)
                cid = old.get("case_id", "")
                if cid:
                    old_ids[cid] = old.get("dimension", "")
            except json.JSONDecodeError:
                pass
    if not old_ids:
        return
    new_ids = {f.case_id: f.dimension for f in new_facts}
    mapping = {"old_to_field": old_ids, "new_to_field": new_ids, "updated_at": datetime.now().isoformat()}
    mapping_path = path.parent / f"case_id_mapping_{user_name}.json"
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl_snapshot(path: str, payloads: List[Dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")




def _coerce_case_fact(payload: CaseFact | Dict[str, Any]) -> CaseFact:
    if isinstance(payload, CaseFact):
        return payload
    return CaseFact(**payload)


def _coerce_pattern_cluster(payload: PatternCluster | Dict[str, Any]) -> PatternCluster:
    if isinstance(payload, PatternCluster):
        return payload
    return PatternCluster(**payload)


def _coerce_decision_task(payload: DecisionReviewItem | Dict[str, Any]) -> DecisionReviewItem:
    if isinstance(payload, DecisionReviewItem):
        return payload
    return DecisionReviewItem(**payload)


def _coerce_strategy_experiment(payload: StrategyExperiment | Dict[str, Any]) -> StrategyExperiment:
    if isinstance(payload, StrategyExperiment):
        return payload
    return StrategyExperiment(**payload)


def _coerce_experiment_outcome(payload: ExperimentOutcome | Dict[str, Any]) -> ExperimentOutcome:
    if isinstance(payload, ExperimentOutcome):
        return payload
    return ExperimentOutcome(**payload)


def _coerce_proposal(payload: ProposalReviewRecord | Dict[str, Any]) -> ProposalReviewRecord:
    if isinstance(payload, ProposalReviewRecord):
        return payload
    return ProposalReviewRecord(**payload)


def _coerce_engineering_change_request(payload: EngineeringChangeRequest | Dict[str, Any]) -> EngineeringChangeRequest:
    if isinstance(payload, EngineeringChangeRequest):
        return payload
    return EngineeringChangeRequest(**payload)


def _coerce_engineering_alert(payload: EngineeringAlert | Dict[str, Any]) -> EngineeringAlert:
    if isinstance(payload, EngineeringAlert):
        return payload
    return EngineeringAlert(**payload)


def _coerce_difficult_case(payload: DifficultCaseRecord | Dict[str, Any]) -> DifficultCaseRecord:
    if isinstance(payload, DifficultCaseRecord):
        return payload
    return DifficultCaseRecord(**payload)


def _pattern_sort_key(pattern: PatternCluster) -> Tuple[int, int, float, str]:
    return (
        PRIORITY_RANK.get(pattern.business_priority, 0),
        pattern.support_count,
        pattern.fix_surface_confidence,
        pattern.pattern_id,
    )


def _timestamp_rank(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _stable_id(prefix: str, raw_key: str) -> str:
    digest = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _utcnow_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_difficult_cases(case_facts: Iterable[CaseFact | Dict[str, Any]]) -> List[DifficultCaseRecord]:
    records: List[DifficultCaseRecord] = []
    for raw_fact in case_facts:
        fact = _coerce_case_fact(raw_fact)
        if fact.resolution_route != "difficult_case" or fact.accuracy_gap_status != "open":
            continue
        records.append(
            DifficultCaseRecord(
                case_id=fact.case_id,
                user_name=fact.user_name,
                album_id=fact.album_id,
                dimension=fact.dimension,
                entity_type=fact.entity_type,
                summary=_summarize_difficult_case(fact),
                detail_url=f"/review/difficult-case/{fact.case_id}",
                status="new",
                comparison_grade=fact.comparison_grade,
                comparison_score=fact.comparison_score,
                resolution_route="difficult_case",
                trace_payload_path=fact.trace_payload_path,
                evidence_refs=list(fact.evidence_refs),
            )
        )
    records.sort(key=lambda item: item.case_id)
    return records


def _summarize_difficult_case(fact: CaseFact) -> str:
    if fact.comparison_grade == "partial_match":
        return f"{fact.dimension} 与 GT 仅部分重合，当前根因不稳定"
    if fact.badcase_source == "empty_output_candidate":
        return f"{fact.dimension} 输出为空，但当前还不能稳定归因到策略面"
    return f"{fact.dimension} 与 GT 不一致，当前需要作为疑难 case 继续分析"


def _build_difficult_case_reason(*, case_record: Dict[str, Any], case_payload: Dict[str, Any], gt_comparison: Dict[str, Any]) -> str:
    agent_reasoning = str(case_payload.get("agent_reasoning_summary") or "").strip()
    why_not_other_surfaces = str(case_payload.get("why_not_other_surfaces") or "").strip()
    if agent_reasoning and why_not_other_surfaces:
        return f"{agent_reasoning} {why_not_other_surfaces}"
    if agent_reasoning:
        return agent_reasoning

    comparison_grade = str(gt_comparison.get("grade") or case_payload.get("comparison_grade") or case_record.get("comparison_grade") or "").strip()
    output_value = gt_comparison.get("output_value")
    gt_value = gt_comparison.get("gt_value")
    causality_route = str(case_payload.get("causality_route") or "").strip()
    badcase_source = str(case_payload.get("badcase_source") or "").strip()

    if comparison_grade == "partial_match" and isinstance(output_value, list) and isinstance(gt_value, list):
        normalized_output = {_stringify_case_value(item) for item in output_value if _stringify_case_value(item)}
        normalized_gt = [_stringify_case_value(item) for item in gt_value if _stringify_case_value(item)]
        missing_from_output = [item for item in normalized_gt if item not in normalized_output]
        if missing_from_output:
            return (
                "当前结果只覆盖了 GT 的一部分，缺少："
                + "、".join(missing_from_output)
                + "。暂时无法稳定判断是召回不全、字段归纳口径偏窄，还是 GT 标注粒度与字段定义存在差异。"
            )
        return "当前结果与 GT 只有部分重合，暂时无法稳定判断是证据召回不全，还是字段归纳边界不一致。"

    if badcase_source == "empty_output_candidate":
        return "当前字段输出为空，现有证据还不足以稳定判断是字段 COT、tool 规则还是 tool 调用策略导致。"

    if causality_route in {"downstream_caused", "downstream_exacerbated"}:
        return "当前 case 受到下游裁决影响，暂时无法直接判断问题来自上游策略还是下游回流。"

    return str(case_payload.get("summary") or case_record.get("summary") or "当前归因还不稳定，需要继续人工判断。").strip()


def _stringify_case_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()
