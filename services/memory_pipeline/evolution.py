from __future__ import annotations

import json
import os
import re
import uuid
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from config import PROJECT_ROOT
from services.reflection.engine.field_diagnostics import diagnose_field as engine_diagnose_field
from services.reflection.engine.signal_extractor import extract_signals as engine_extract_signals
from services.reflection.engine.patch_planner import plan_patch as engine_plan_patch
from .rule_asset_loader import apply_repo_rule_patch


EVOLUTION_RELATIVE_DIR = Path("memory") / "evolution"
REFLECTION_RELATIVE_DIR = Path("memory") / "reflection"
FOCUS_FIELD_COUNT_DEFAULT = 3
FIELD_LOOP_NO_SIGNAL_STREAK_THRESHOLD_DEFAULT = 2
FIELD_LOOP_THROTTLE_COOLDOWN_ROUNDS_DEFAULT = 2

_GRADE_PRIORITY = {
    "mismatch": 100.0,
    "missing_prediction": 95.0,
    "partial_match": 70.0,
    "close_match": 10.0,
    "exact_match": 0.0,
    "missing_gt": 0.0,
}

_SEVERITY_PRIORITY = {
    "high": 30.0,
    "medium": 15.0,
    "low": 5.0,
}

_FIELD_RISK_PRIORITY = {
    "long_term_facts.social_identity.education": 18.0,
    "long_term_facts.material.brand_preference": 18.0,
    "short_term_expression.motivation_shift": 20.0,
    "short_term_expression.stress_signal": 16.0,
}

_RESOLVED_GRADES = {"exact_match", "close_match"}


def build_memory_run_trace(
    *,
    run_type: str,
    user_name: str,
    stage_reports: List[Dict[str, Any]] | None,
    downstream_audit_report: Dict[str, Any] | None,
    profile_llm_batch_debug: List[Dict[str, Any]] | None,
    test_issue_log: Dict[str, Any] | None,
    artifacts: Dict[str, Any] | None,
    generated_at: str | None = None,
    completion_hints: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    now_iso = generated_at or datetime.now().isoformat()
    date_str = _derive_date(now_iso)
    normalized_stage_reports = [dict(item or {}) for item in list(stage_reports or [])]
    normalized_downstream = dict(downstream_audit_report or {})
    normalized_batch_debug = [dict(item or {}) for item in list(profile_llm_batch_debug or [])]
    normalized_issue_log = dict(test_issue_log or {})
    normalized_artifacts = dict(artifacts or {})
    hints = dict(completion_hints or {})

    stage_total_count = len(normalized_stage_reports)
    stage_failure_count = sum(
        1
        for item in normalized_stage_reports
        if str(item.get("status") or "").strip().lower() not in {"ok", "success", "passed"}
    )

    metadata = dict(normalized_downstream.get("metadata") or {})
    summary = dict(normalized_downstream.get("summary") or {})
    audit_status = str(metadata.get("audit_status") or "ok")
    verification_passed = audit_status not in {"skipped_init_failure", "failed", "error"}

    deliverables_met = _infer_deliverables_met(normalized_artifacts)
    scope_respected = bool(hints.get("scope_respected", True))
    contract_respected = bool(hints.get("contract_respected", True))
    handoff_ready = bool(normalized_artifacts.get("downstream_audit_report_path") or normalized_downstream)

    completion_score = (
        (0.35 if deliverables_met else 0.0)
        + (0.35 if verification_passed else 0.0)
        + (0.15 if scope_respected else 0.0)
        + (0.10 if contract_respected else 0.0)
        + (0.05 if handoff_ready else 0.0)
    )
    if not scope_respected:
        completion_score = 0.0
    elif not verification_passed:
        completion_score = min(completion_score, 0.59)
    completion_score = round(_clip01(completion_score), 3)

    completion_status = _determine_completion_status(
        deliverables_met=deliverables_met,
        verification_passed=verification_passed,
        scope_respected=scope_respected,
        stage_failure_count=stage_failure_count,
    )

    batch_total_count = len(normalized_batch_debug)
    fallback_batch_count = sum(
        1
        for item in normalized_batch_debug
        if item.get("used_offline_fallback")
        or item.get("fallback_reason")
        or item.get("raw_result_parseable") is False
    )
    fallback_ratio = (
        round(fallback_batch_count / batch_total_count, 3)
        if batch_total_count
        else 0.0
    )

    rejected_count = int(summary.get("rejected_count") or 0)
    audited_count = int(summary.get("total_audited_tags") or 0)
    not_audited_count = int(summary.get("not_audited_count") or 0)
    issue_summary = dict(normalized_issue_log.get("summary") or {})
    issue_count = int(issue_summary.get("issue_count") or 0)
    high_risk_issue_count = int(issue_summary.get("high_risk_issue_count") or 0)

    stage_ok_ratio = (
        1.0 - (stage_failure_count / max(stage_total_count, 1))
        if stage_total_count
        else 1.0
    )
    deterministic_score = (
        0.50 * stage_ok_ratio
        + 0.30 * (1.0 if verification_passed else 0.0)
        + 0.20 * (1.0 - fallback_ratio)
    )
    if audited_count > 0:
        review_score = 1.0 - (rejected_count / max(audited_count, 1))
    else:
        review_score = 0.80 if verification_passed else 0.30
    human_feedback_score = float(hints.get("human_feedback_score", 0.50))
    rerun_count = _count_feedback_reruns(normalized_downstream)
    issue_component = 1.0 - min(1.0, issue_count * 0.10 + high_risk_issue_count * 0.35)
    rerun_component = max(0.0, 1.0 - rerun_count * 0.35)
    fallback_component = 1.0 - fallback_ratio
    implicit_score = (issue_component + rerun_component + fallback_component) / 3.0
    quality_score = round(
        _clip01(
            0.40 * deterministic_score
            + 0.20 * review_score
            + 0.20 * human_feedback_score
            + 0.20 * implicit_score
        ),
        3,
    )

    return {
        "trace_id": f"mrun_{date_str.replace('-', '')}_{uuid.uuid4().hex[:8]}",
        "generated_at": now_iso,
        "date": date_str,
        "user_name": str(user_name or "default"),
        "run_type": str(run_type or "mainline"),
        "completion": {
            "deliverables_met": deliverables_met,
            "verification_passed": verification_passed,
            "scope_respected": scope_respected,
            "contract_respected": contract_respected,
            "handoff_ready": handoff_ready,
            "completion_status": completion_status,
        },
        "scores": {
            "completion_score": completion_score,
            "quality_score": quality_score,
        },
        "metrics": {
            "stage_total_count": stage_total_count,
            "stage_failure_count": stage_failure_count,
            "llm_batch_count": batch_total_count,
            "llm_fallback_batch_count": fallback_batch_count,
            "llm_fallback_ratio": fallback_ratio,
            "downstream_audit_status": audit_status,
            "downstream_rejected_count": rejected_count,
            "downstream_audited_count": audited_count,
            "downstream_not_audited_count": not_audited_count,
            "issue_count": issue_count,
            "high_risk_issue_count": high_risk_issue_count,
            "feedback_rerun_count": rerun_count,
        },
        "artifacts": normalized_artifacts,
        "stage_reports": normalized_stage_reports,
        "downstream_audit_summary": summary,
    }


def persist_memory_run_trace(
    *,
    project_root: str = PROJECT_ROOT,
    output_dir: str,
    trace_payload: Dict[str, Any],
) -> Dict[str, str]:
    root_path = Path(project_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    normalized_payload = deepcopy(dict(trace_payload or {}))
    if not normalized_payload.get("generated_at"):
        normalized_payload["generated_at"] = datetime.now().isoformat()
    normalized_payload["date"] = normalized_payload.get("date") or _derive_date(str(normalized_payload["generated_at"]))
    normalized_payload["user_name"] = str(normalized_payload.get("user_name") or "default")

    trace_json_path = output_path / "memory_pipeline_run_trace.json"
    trace_json_path.write_text(
        json.dumps(normalized_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    ledger_dir = root_path / EVOLUTION_RELATIVE_DIR / "traces" / normalized_payload["user_name"]
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = ledger_dir / f"{normalized_payload['date']}.jsonl"
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(normalized_payload, ensure_ascii=False) + "\n")

    return {
        "trace_id": str(normalized_payload.get("trace_id") or ""),
        "trace_json_path": str(trace_json_path),
        "trace_ledger_path": str(ledger_path),
    }


def load_memory_run_traces(
    *,
    project_root: str = PROJECT_ROOT,
    user_name: str,
    date_str: str,
) -> List[Dict[str, Any]]:
    path = Path(project_root) / EVOLUTION_RELATIVE_DIR / "traces" / user_name / f"{date_str}.jsonl"
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def run_memory_nightly_evaluation(
    *,
    project_root: str = PROJECT_ROOT,
    user_name: str,
    date_str: str,
    top_k_fields: int = FOCUS_FIELD_COUNT_DEFAULT,
) -> Dict[str, Any]:
    traces = load_memory_run_traces(project_root=project_root, user_name=user_name, date_str=date_str)

    reports_dir = Path(project_root) / EVOLUTION_RELATIVE_DIR / "reports" / user_name
    insights_dir = Path(project_root) / EVOLUTION_RELATIVE_DIR / "insights" / user_name
    proposals_dir = Path(project_root) / EVOLUTION_RELATIVE_DIR / "proposals" / user_name
    focus_dir = Path(project_root) / EVOLUTION_RELATIVE_DIR / "focus_fields" / user_name
    field_cycles_dir = Path(project_root) / EVOLUTION_RELATIVE_DIR / "field_cycles" / user_name
    reports_dir.mkdir(parents=True, exist_ok=True)
    insights_dir.mkdir(parents=True, exist_ok=True)
    proposals_dir.mkdir(parents=True, exist_ok=True)
    focus_dir.mkdir(parents=True, exist_ok=True)
    field_cycles_dir.mkdir(parents=True, exist_ok=True)

    summary = _build_nightly_summary(traces)

    skipped_count = summary.get("downstream_audit_skipped_count", 0)
    if skipped_count > 0 and os.getenv("NIGHTLY_REQUIRE_DOWNSTREAM_AUDIT", "").lower() in ("1", "true"):
        raise RuntimeError(
            f"downstream audit was skipped in {skipped_count}/{summary.get('total_runs', 0)} runs "
            f"(audit_status=skipped_init_failure). "
            f"Set NIGHTLY_REQUIRE_DOWNSTREAM_AUDIT=false to allow fallback."
        )

    legacy_insights = _build_nightly_insights(summary=summary, traces=traces)
    legacy_proposals = _build_nightly_proposals(
        date_str=date_str,
        user_name=user_name,
        insights=legacy_insights,
        traces=traces,
    )
    field_loop_payload = _run_gt_field_loop(
        project_root=project_root,
        user_name=user_name,
        date_str=date_str,
        top_k_fields=top_k_fields,
    )
    insights = _merge_nightly_insights(
        field_insights=list(field_loop_payload.get("insights") or []),
        legacy_insights=legacy_insights,
    )
    proposals = list(field_loop_payload.get("proposals") or [])
    if not proposals:
        proposals = legacy_proposals

    report_payload = {
        "date": date_str,
        "user_name": user_name,
        "summary": summary,
        "insights": insights,
        "proposals": proposals,
        "focus_fields": field_loop_payload.get("focus_fields", []),
        "field_cycles": field_loop_payload.get("field_cycles", []),
        "field_loop_summary": field_loop_payload.get("summary", {}),
    }

    report_path = reports_dir / f"{date_str}.json"
    insights_path = insights_dir / f"{date_str}.json"
    proposals_path = proposals_dir / f"{date_str}.json"
    focus_path = focus_dir / f"{date_str}.json"
    field_cycles_path = field_cycles_dir / f"{date_str}.json"
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    insights_path.write_text(json.dumps(insights, ensure_ascii=False, indent=2), encoding="utf-8")
    proposals_path.write_text(json.dumps(proposals, ensure_ascii=False, indent=2), encoding="utf-8")
    focus_path.write_text(
        json.dumps(field_loop_payload.get("focus_fields", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    field_cycles_path.write_text(
        json.dumps(field_loop_payload.get("field_cycles", []), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "report_path": str(report_path),
        "insights_path": str(insights_path),
        "proposals_path": str(proposals_path),
        "focus_fields_path": str(focus_path),
        "field_cycles_path": str(field_cycles_path),
        "total_traces": len(traces),
        "total_insights": len(insights),
        "total_proposals": len(proposals),
        "total_focus_fields": len(field_loop_payload.get("focus_fields", [])),
    }


def run_memory_nightly_user_set_evaluation(
    *,
    project_root: str = PROJECT_ROOT,
    user_names: Iterable[str] | None,
    date_str: str,
    top_k_fields: int = FOCUS_FIELD_COUNT_DEFAULT,
) -> Dict[str, Any]:
    resolved_users = _resolve_user_set(project_root=project_root, user_names=user_names)
    results: List[Dict[str, Any]] = []
    for user_name in resolved_users:
        result = run_memory_nightly_evaluation(
            project_root=project_root,
            user_name=user_name,
            date_str=date_str,
            top_k_fields=top_k_fields,
        )
        results.append(
            {
                "user_name": user_name,
                "report_path": result["report_path"],
                "insights_path": result["insights_path"],
                "proposals_path": result["proposals_path"],
                "focus_fields_path": result.get("focus_fields_path"),
                "field_cycles_path": result.get("field_cycles_path"),
                "total_traces": int(result.get("total_traces") or 0),
                "total_focus_fields": int(result.get("total_focus_fields") or 0),
                "total_proposals": int(result.get("total_proposals") or 0),
            }
        )

    aggregate = {
        "date": date_str,
        "total_users": len(resolved_users),
        "users": results,
        "summary": {
            "total_traces": sum(item["total_traces"] for item in results),
            "total_focus_fields": sum(item["total_focus_fields"] for item in results),
            "total_proposals": sum(item["total_proposals"] for item in results),
        },
    }
    aggregate_dir = Path(project_root) / EVOLUTION_RELATIVE_DIR / "reports" / "_user_set"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = aggregate_dir / f"{date_str}.json"
    aggregate_path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "report_path": str(aggregate_path),
        "date": date_str,
        "total_users": len(resolved_users),
        "users": results,
    }


def apply_memory_evolution_proposal(
    *,
    project_root: str = PROJECT_ROOT,
    proposal: Dict[str, Any],
    actor: str,
) -> Dict[str, Any]:
    normalized_proposal = dict(proposal or {})
    user_name = str(normalized_proposal.get("user_name") or "default")
    proposal_id = str(normalized_proposal.get("proposal_id") or f"proposal_{uuid.uuid4().hex[:8]}")
    patch_preview = dict(normalized_proposal.get("patch_preview") or {})

    if not patch_preview:
        action = {
            "proposal_id": proposal_id,
            "proposal_type": str(normalized_proposal.get("proposal_type") or ""),
            "actor": actor,
            "status": "manual_required",
            "applied_at": datetime.now().isoformat(),
            "reason": "empty_patch_preview",
        }
        proposal_actions_path = _append_proposal_action(
            project_root=project_root,
            user_name=user_name,
            action=action,
        )
        return {
            "status": "manual_required",
            "proposal_actions_path": proposal_actions_path,
            "asset_paths": {},
        }

    asset_paths = apply_repo_rule_patch(
        project_root=project_root,
        patch_preview=patch_preview,
    )
    action = {
        "proposal_id": proposal_id,
        "proposal_type": str(normalized_proposal.get("proposal_type") or ""),
        "actor": actor,
        "status": "applied",
        "applied_at": datetime.now().isoformat(),
        "asset_paths": asset_paths,
        "patch_preview": patch_preview,
    }
    proposal_actions_path = _append_proposal_action(
        project_root=project_root,
        user_name=user_name,
        action=action,
    )
    return {
        "status": "applied",
        "proposal_actions_path": proposal_actions_path,
        "asset_paths": asset_paths,
    }


def _append_proposal_action(*, project_root: str, user_name: str, action: Dict[str, Any]) -> str:
    evolution_dir = Path(project_root) / EVOLUTION_RELATIVE_DIR
    evolution_dir.mkdir(parents=True, exist_ok=True)
    path = evolution_dir / f"proposal_actions_{user_name}.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(action, ensure_ascii=False) + "\n")
    return str(path)


def _build_nightly_summary(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not traces:
        return {
            "total_runs": 0,
            "avg_completion_score": 0.0,
            "avg_quality_score": 0.0,
            "completed_count": 0,
            "concern_count": 0,
            "failed_count": 0,
            "by_run_type": {},
            "avg_fallback_ratio": 0.0,
            "downstream_audit_skipped_count": 0,
            "high_risk_issue_count": 0,
        }

    completion_scores = [float((item.get("scores") or {}).get("completion_score") or 0.0) for item in traces]
    quality_scores = [float((item.get("scores") or {}).get("quality_score") or 0.0) for item in traces]
    fallback_ratios = [float((item.get("metrics") or {}).get("llm_fallback_ratio") or 0.0) for item in traces]
    by_run_type: Dict[str, int] = {}
    completed_count = 0
    concern_count = 0
    failed_count = 0
    downstream_audit_skipped_count = 0
    high_risk_issue_count = 0
    for item in traces:
        run_type = str(item.get("run_type") or "unknown")
        by_run_type[run_type] = by_run_type.get(run_type, 0) + 1

        completion = dict(item.get("completion") or {})
        status = str(completion.get("completion_status") or "")
        if status == "completed":
            completed_count += 1
        elif status == "completed_with_concern":
            concern_count += 1
        else:
            failed_count += 1

        metrics = dict(item.get("metrics") or {})
        if str(metrics.get("downstream_audit_status") or "") == "skipped_init_failure":
            downstream_audit_skipped_count += 1
        high_risk_issue_count += int(metrics.get("high_risk_issue_count") or 0)

    return {
        "total_runs": len(traces),
        "avg_completion_score": round(sum(completion_scores) / len(completion_scores), 3),
        "avg_quality_score": round(sum(quality_scores) / len(quality_scores), 3),
        "completed_count": completed_count,
        "concern_count": concern_count,
        "failed_count": failed_count,
        "by_run_type": by_run_type,
        "avg_fallback_ratio": round(sum(fallback_ratios) / len(fallback_ratios), 3),
        "downstream_audit_skipped_count": downstream_audit_skipped_count,
        "high_risk_issue_count": high_risk_issue_count,
    }


def _build_nightly_insights(*, summary: Dict[str, Any], traces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    insights: List[Dict[str, Any]] = []
    if not traces:
        return insights

    if summary.get("downstream_audit_skipped_count", 0) > 0:
        insights.append(
            {
                "type": "downstream_audit_skipped",
                "severity": "high",
                "detail": f"有 {summary['downstream_audit_skipped_count']} 次 run 跳过了 downstream audit。",
            }
        )

    avg_quality = float(summary.get("avg_quality_score") or 0.0)
    if avg_quality < 0.70:
        insights.append(
            {
                "type": "low_quality_score",
                "severity": "medium",
                "detail": f"平均 quality_score={avg_quality:.3f}，低于 0.70。",
            }
        )

    avg_completion = float(summary.get("avg_completion_score") or 0.0)
    if avg_completion < 0.80:
        insights.append(
            {
                "type": "low_completion_score",
                "severity": "medium",
                "detail": f"平均 completion_score={avg_completion:.3f}，低于 0.80。",
            }
        )

    avg_fallback_ratio = float(summary.get("avg_fallback_ratio") or 0.0)
    if avg_fallback_ratio > 0.20:
        insights.append(
            {
                "type": "high_llm_fallback_ratio",
                "severity": "medium",
                "detail": f"平均 LP3 fallback_ratio={avg_fallback_ratio:.3f}，高于 0.20。",
            }
        )

    if int(summary.get("high_risk_issue_count") or 0) > 0:
        insights.append(
            {
                "type": "high_risk_issues_detected",
                "severity": "high",
                "detail": f"高风险问题累计 {summary['high_risk_issue_count']} 个。",
            }
        )

    if int(summary.get("concern_count") or 0) > 0:
        insights.append(
            {
                "type": "completion_with_concerns",
                "severity": "low",
                "detail": f"{summary['concern_count']} 次 run 以 completed_with_concern 收敛。",
            }
        )

    return insights


def _build_nightly_proposals(
    *,
    date_str: str,
    user_name: str,
    insights: List[Dict[str, Any]],
    traces: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    proposals: List[Dict[str, Any]] = []
    for index, insight in enumerate(insights, start=1):
        insight_type = str(insight.get("type") or "")
        proposal_id = f"proposal_{date_str.replace('-', '')}_{index:03d}"

        if insight_type == "high_llm_fallback_ratio":
            proposals.append(
                {
                    "proposal_id": proposal_id,
                    "date": date_str,
                    "user_name": user_name,
                    "status": "proposal_only",
                    "proposal_type": "tool_rule_patch",
                    "title": "收敛 LP3 证据输入，降低 batch fallback 风险",
                    "reason": insight.get("detail"),
                    "patch_preview": {
                        "tool_rules": {
                            "long_term_facts.material.brand_preference": {"max_refs_per_source": 20},
                            "short_term_facts.recent_interests": {"max_refs_per_source": 20},
                            "long_term_facts.geography.location_anchors": {"max_refs_per_source": 20},
                        }
                    },
                    "approval_required": True,
                }
            )
            continue

        if insight_type == "low_quality_score":
            proposals.append(
                {
                    "proposal_id": proposal_id,
                    "date": date_str,
                    "user_name": user_name,
                    "status": "proposal_only",
                    "proposal_type": "field_spec_patch",
                    "title": "强化高风险字段 null 优先条件",
                    "reason": insight.get("detail"),
                    "patch_preview": {
                        "field_spec_overrides": {
                            "long_term_facts.social_identity.education": {
                                "null_preferred_when": [
                                    "证据仅来自单事件窗口时输出 null",
                                    "主体归属不清时输出 null",
                                ]
                            },
                            "long_term_facts.material.brand_preference": {
                                "null_preferred_when": [
                                    "品牌线索仅出现在单次曝光时输出 null",
                                    "品牌线索集中在 venue context 时输出 null",
                                ]
                            },
                        }
                    },
                    "approval_required": True,
                }
            )
            continue

        if insight_type == "downstream_audit_skipped":
            proposals.append(
                {
                    "proposal_id": proposal_id,
                    "date": date_str,
                    "user_name": user_name,
                    "status": "proposal_only",
                    "proposal_type": "runtime_guardrail",
                    "title": "补齐 downstream runtime 预检与告警",
                    "reason": insight.get("detail"),
                    "patch_preview": {},
                    "approval_required": True,
                }
            )
            continue

        if insight_type == "high_risk_issues_detected":
            proposals.append(
                {
                    "proposal_id": proposal_id,
                    "date": date_str,
                    "user_name": user_name,
                    "status": "proposal_only",
                    "proposal_type": "call_policy_patch",
                    "title": "高风险字段默认开启更保守来源策略",
                    "reason": insight.get("detail"),
                    "patch_preview": {
                        "call_policies": {
                            "long_term_facts.social_identity.education": {
                                "append_allowed_sources": ["relationship"],
                            },
                            "long_term_facts.material.brand_preference": {
                                "append_allowed_sources": ["event"],
                            },
                        }
                    },
                    "approval_required": True,
                }
            )
            continue

        proposals.append(
            {
                "proposal_id": proposal_id,
                "date": date_str,
                "user_name": user_name,
                "status": "proposal_only",
                "proposal_type": "analysis_followup",
                "title": "保持 proposal-first，先人工确认后改规则",
                "reason": insight.get("detail"),
                "patch_preview": {},
                "approval_required": True,
            }
        )

    if not proposals and traces:
        proposals.append(
            {
                "proposal_id": f"proposal_{date_str.replace('-', '')}_001",
                "date": date_str,
                "user_name": user_name,
                "status": "proposal_only",
                "proposal_type": "analysis_followup",
                "title": "样本不足，保持观察",
                "reason": "当前无显著异常，继续累计 trace 并观察趋势。",
                "patch_preview": {},
                "approval_required": True,
            }
        )
    return proposals


def _merge_nightly_insights(
    *,
    field_insights: List[Dict[str, Any]],
    legacy_insights: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for payload in list(field_insights or []) + list(legacy_insights or []):
        item = dict(payload or {})
        key = (str(item.get("type") or ""), str(item.get("field_key") or ""), str(item.get("detail") or ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _run_gt_field_loop(
    *,
    project_root: str,
    user_name: str,
    date_str: str,
    top_k_fields: int,
) -> Dict[str, Any]:
    reflection_dir = Path(project_root) / REFLECTION_RELATIVE_DIR
    comparisons_path = reflection_dir / f"gt_comparisons_{user_name}.jsonl"
    gt_path = reflection_dir / f"profile_field_gt_{user_name}.jsonl"
    comparisons = _load_jsonl_records(comparisons_path)
    gt_records = _load_jsonl_records(gt_path)
    state_path = Path(project_root) / EVOLUTION_RELATIVE_DIR / "field_loop_state" / f"{user_name}.json"
    field_state = _decay_field_loop_cooldowns(_load_field_loop_state(state_path))
    policy = _resolve_field_loop_policy()

    focus_fields, latest_by_field = _select_focus_fields_by_gt(
        comparisons=comparisons,
        gt_records=gt_records,
        top_k_fields=top_k_fields,
        field_state=field_state,
    )
    assets = _load_repo_rule_assets_safe(project_root=project_root)
    field_cycles, field_insights, field_proposals, next_state = _build_gt_field_cycles(
        date_str=date_str,
        user_name=user_name,
        focus_fields=focus_fields,
        latest_by_field=latest_by_field,
        gt_records=gt_records,
        field_state=field_state,
        current_assets=assets,
        policy=policy,
    )
    _persist_field_loop_state(path=state_path, state_payload=next_state)

    summary = {
        "total_focus_fields": len(focus_fields),
        "new_signal_count": sum(1 for item in field_cycles if item.get("new_signal_found")),
        "new_rule_candidate_count": sum(1 for item in field_cycles if item.get("cycle_status") == "new_rule_candidate"),
        "throttled_fields_count": sum(1 for item in field_cycles if item.get("cycle_status") == "throttled"),
        "active_focus_fields_count": sum(1 for item in field_cycles if item.get("cycle_status") != "throttled"),
        "no_signal_streak_threshold": int(policy["no_signal_streak_threshold"]),
        "throttle_cooldown_rounds": int(policy["throttle_cooldown_rounds"]),
    }
    return {
        "focus_fields": focus_fields,
        "field_cycles": field_cycles,
        "insights": field_insights,
        "proposals": field_proposals,
        "summary": summary,
        "inputs": {
            "gt_comparisons_path": str(comparisons_path),
            "profile_field_gt_path": str(gt_path),
        },
    }


def _resolve_user_set(*, project_root: str, user_names: Iterable[str] | None) -> List[str]:
    requested = [str(item or "").strip() for item in list(user_names or []) if str(item or "").strip()]
    if requested:
        return list(dict.fromkeys(requested))
    discovered = _discover_gt_users(project_root=project_root)
    return discovered or ["default"]


def _discover_gt_users(*, project_root: str) -> List[str]:
    users: Dict[str, None] = {}
    reflection_dir = Path(project_root) / REFLECTION_RELATIVE_DIR
    for file_path in reflection_dir.glob("profile_field_gt_*.jsonl"):
        users[file_path.stem.removeprefix("profile_field_gt_")] = None
    for file_path in reflection_dir.glob("gt_comparisons_*.jsonl"):
        users[file_path.stem.removeprefix("gt_comparisons_")] = None
    traces_dir = Path(project_root) / EVOLUTION_RELATIVE_DIR / "traces"
    if traces_dir.exists():
        for child in traces_dir.iterdir():
            if child.is_dir():
                users[child.name] = None
    return sorted(key for key in users if key)


def _load_jsonl_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _select_focus_fields_by_gt(
    *,
    comparisons: List[Dict[str, Any]],
    gt_records: List[Dict[str, Any]],
    top_k_fields: int,
    field_state: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    latest_by_field: Dict[str, Dict[str, Any]] = {}
    scored_by_field: Dict[str, Dict[str, Any]] = {}
    per_field_state = dict(dict(field_state or {}).get("fields") or {})
    total = max(len(comparisons), 1)
    for index, row in enumerate(comparisons, start=1):
        field_key = str(row.get("field_key") or "").strip()
        if not field_key:
            continue
        comparison_result = dict(row.get("comparison_result") or {})
        grade = str(comparison_result.get("grade") or "mismatch")
        score = float(comparison_result.get("score") or 0.0)
        severity = str(comparison_result.get("severity") or "medium").strip().lower()
        issue_score = _score_gt_issue(
            field_key=field_key,
            grade=grade,
            score=score,
            severity=severity,
            recency_weight=(index / total),
        )
        latest_by_field[field_key] = row
        entry = scored_by_field.setdefault(
            field_key,
            {
                "field_key": field_key,
                "issue_score": 0.0,
                "badcase_count": 0,
                "latest_grade": grade,
                "latest_score": score,
                "severity": severity,
                "rationale": "",
            },
        )
        if grade not in _RESOLVED_GRADES:
            entry["badcase_count"] = int(entry["badcase_count"]) + 1
        entry["issue_score"] = float(entry["issue_score"]) + issue_score
        entry["latest_grade"] = grade
        entry["latest_score"] = score
        entry["severity"] = severity

    focus_candidates: List[Dict[str, Any]] = []
    for field_key, entry in scored_by_field.items():
        if int(entry["badcase_count"]) <= 0:
            continue
        candidate = dict(entry)
        candidate["issue_score"] = round(float(candidate["issue_score"]), 3)
        candidate["rationale"] = f"GT对比存在 {candidate['badcase_count']} 条未对齐记录。"
        saved_state = dict(per_field_state.get(field_key) or {})
        cooldown_remaining = max(0, int(saved_state.get("cooldown_remaining") or 0))
        no_new_signal_streak = max(0, int(saved_state.get("no_new_signal_streak") or 0))
        last_status = str(saved_state.get("last_status") or "").strip().lower()
        candidate["throttled"] = cooldown_remaining > 0
        candidate["cooldown_remaining"] = cooldown_remaining
        candidate["no_new_signal_streak"] = no_new_signal_streak
        candidate["recent_gain"] = last_status in {"new_rule_candidate", "new_insight_found"}
        focus_candidates.append(candidate)
    focus_candidates.sort(
        key=lambda item: (float(item.get("issue_score") or 0.0), int(item.get("badcase_count") or 0)),
        reverse=True,
    )

    if len(focus_candidates) < max(top_k_fields, 1):
        existing = {item["field_key"] for item in focus_candidates}
        for gt_row in gt_records:
            field_key = str(gt_row.get("field_key") or "").strip()
            if not field_key or field_key in existing:
                continue
            confidence = _safe_float(gt_row.get("original_confidence"), default=0.6)
            observe_score = round((1.0 - confidence) * 40.0 + _FIELD_RISK_PRIORITY.get(field_key, 0.0), 3)
            saved_state = dict(per_field_state.get(field_key) or {})
            cooldown_remaining = max(0, int(saved_state.get("cooldown_remaining") or 0))
            no_new_signal_streak = max(0, int(saved_state.get("no_new_signal_streak") or 0))
            last_status = str(saved_state.get("last_status") or "").strip().lower()
            focus_candidates.append(
                {
                    "field_key": field_key,
                    "issue_score": observe_score,
                    "badcase_count": 0,
                    "latest_grade": "observe",
                    "latest_score": confidence,
                    "severity": "medium",
                    "rationale": "当前 GT 无明显错误，但属于高风险/低置信字段，纳入循环观察。",
                    "throttled": cooldown_remaining > 0,
                    "cooldown_remaining": cooldown_remaining,
                    "no_new_signal_streak": no_new_signal_streak,
                    "recent_gain": last_status in {"new_rule_candidate", "new_insight_found"},
                }
            )
            existing.add(field_key)

    active_candidates = [item for item in focus_candidates if not item.get("throttled")]
    throttled_candidates = [item for item in focus_candidates if item.get("throttled")]
    active_candidates.sort(
        key=lambda item: (
            int(bool(item.get("recent_gain"))),
            -int(item.get("no_new_signal_streak") or 0),
            float(item.get("issue_score") or 0.0),
            int(item.get("badcase_count") or 0),
        ),
        reverse=True,
    )
    throttled_candidates.sort(
        key=lambda item: (
            -int(item.get("cooldown_remaining") or 0),
            -int(item.get("no_new_signal_streak") or 0),
            int(bool(item.get("recent_gain"))),
            float(item.get("issue_score") or 0.0),
            int(item.get("badcase_count") or 0),
        ),
        reverse=True,
    )

    selected: List[Dict[str, Any]] = []
    limit = max(top_k_fields, 1)
    for candidate in active_candidates:
        if len(selected) >= limit:
            break
        selected.append(candidate)
    if len(selected) < limit:
        for candidate in throttled_candidates:
            if len(selected) >= limit:
                break
            selected.append(candidate)

    trimmed = selected[:limit]
    for rank, item in enumerate(trimmed, start=1):
        item["priority_rank"] = rank
    return trimmed, latest_by_field


def _score_gt_issue(
    *,
    field_key: str,
    grade: str,
    score: float,
    severity: str,
    recency_weight: float,
) -> float:
    grade_weight = _GRADE_PRIORITY.get(grade, 40.0)
    severity_weight = _SEVERITY_PRIORITY.get(severity, 10.0)
    score_penalty = (1.0 - max(0.0, min(1.0, score))) * 35.0
    risk_weight = _FIELD_RISK_PRIORITY.get(field_key, 0.0)
    recency_bonus = max(0.0, min(1.0, recency_weight)) * 8.0
    return grade_weight + severity_weight + score_penalty + risk_weight + recency_bonus


def _build_gt_field_cycles(
    *,
    date_str: str,
    user_name: str,
    focus_fields: List[Dict[str, Any]],
    latest_by_field: Dict[str, Dict[str, Any]],
    gt_records: List[Dict[str, Any]],
    field_state: Dict[str, Any],
    current_assets: Dict[str, Dict[str, Any]],
    policy: Dict[str, int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    gt_by_field = {
        str(item.get("field_key") or "").strip(): dict(item)
        for item in gt_records
        if str(item.get("field_key") or "").strip()
    }
    state_payload = dict(field_state or {})
    per_field_state = dict(state_payload.get("fields") or {})

    cycles: List[Dict[str, Any]] = []
    insights: List[Dict[str, Any]] = []
    proposals: List[Dict[str, Any]] = []
    no_signal_streak_threshold = max(1, int(policy.get("no_signal_streak_threshold") or FIELD_LOOP_NO_SIGNAL_STREAK_THRESHOLD_DEFAULT))
    throttle_cooldown_rounds = max(1, int(policy.get("throttle_cooldown_rounds") or FIELD_LOOP_THROTTLE_COOLDOWN_ROUNDS_DEFAULT))

    for index, focus in enumerate(focus_fields, start=1):
        field_key = str(focus.get("field_key") or "").strip()
        comparison_row = dict(latest_by_field.get(field_key) or {})
        comparison_result = dict(comparison_row.get("comparison_result") or {})
        gt_row = dict(gt_by_field.get(field_key) or {})
        previous_state = dict(per_field_state.get(field_key) or {})
        cycle_index = int(previous_state.get("cycle_count") or 0) + 1
        previous_streak = max(0, int(previous_state.get("no_new_signal_streak") or 0))
        previous_cooldown = max(0, int(previous_state.get("cooldown_remaining") or 0))
        grade = str(comparison_result.get("grade") or focus.get("latest_grade") or "observe")
        output_value = comparison_result.get("output_value")
        gt_value = comparison_result.get("gt_value")
        if gt_value is None:
            gt_value = gt_row.get("gt_value")

        if previous_cooldown > 0:
            cooldown_remaining = previous_cooldown
            cycle_entry = {
                "field_key": field_key,
                "priority_rank": int(focus.get("priority_rank") or index),
                "issue_score": float(focus.get("issue_score") or 0.0),
                "cycle_index": cycle_index,
                "grade": grade,
                "failure_mode": "throttled_waiting",
                "new_signal_found": False,
                "signal_key": "",
                "overlooked_signals": [],
                "cycle_status": "throttled",
                "rationale": str(focus.get("rationale") or ""),
                "no_new_signal_streak": previous_streak,
                "cooldown_remaining": cooldown_remaining,
                "throttle_reason": (
                    f"连续 {previous_streak} 轮无新线索，进入降频窗口 "
                    f"(remaining={cooldown_remaining})。"
                ),
                "inspiration": (
                    f"{field_key} 当前处于降频窗口，先将预算集中到其他仍有增益字段。"
                ),
            }
            cycles.append(cycle_entry)
            insights.append(
                {
                    "type": "field_loop_throttled",
                    "severity": "low",
                    "field_key": field_key,
                    "detail": (
                        f"[{field_key}] cycle={cycle_index} throttled "
                        f"streak={previous_streak} cooldown_remaining={cooldown_remaining}"
                    ),
                }
            )
            per_field_state[field_key] = {
                "cycle_count": cycle_index,
                "last_grade": grade,
                "last_issue_score": float(focus.get("issue_score") or 0.0),
                "last_status": "throttled",
                "last_signal_key": str(previous_state.get("last_signal_key") or ""),
                "seen_signal_keys": _dedupe_non_empty(previous_state.get("seen_signal_keys") or []),
                "no_new_signal_streak": previous_streak,
                "cooldown_remaining": cooldown_remaining,
                "throttled_round_count": int(previous_state.get("throttled_round_count") or 0) + 1,
                "last_updated": datetime.now().isoformat(),
            }
            continue

        diagnosis = engine_diagnose_field(
            field_key=field_key,
            output_value=output_value,
            gt_value=gt_value,
            comparison_grade=grade,
        )
        failure_mode = diagnosis.failure_mode

        seen_signals = [str(item or "").strip() for item in list(previous_state.get("seen_signal_keys") or []) if str(item or "").strip()]
        signal_result = engine_extract_signals(
            field_key=field_key,
            output_value=output_value,
            gt_value=gt_value,
            gt_notes=str(gt_row.get("notes") or ""),
            evidence_summary=str(gt_row.get("evidence_summary") or ""),
            accuracy_note=str(gt_row.get("accuracy_note") or ""),
            seen_signal_keys=seen_signals,
        )
        overlooked_signals = signal_result.signals
        signal_key = signal_result.signal_key
        new_signal_found = signal_result.is_new
        if new_signal_found:
            seen_signals.append(signal_key)

        patch_plan = engine_plan_patch(
            field_key=field_key,
            diagnosis=diagnosis,
            signals=signal_result,
            current_assets=current_assets,
        )
        patch_preview = patch_plan.patch_preview
        proposal_type = patch_plan.proposal_type
        no_new_signal_streak = 0 if new_signal_found else previous_streak + 1
        cooldown_remaining = 0
        if not new_signal_found and no_new_signal_streak >= no_signal_streak_threshold:
            cooldown_remaining = throttle_cooldown_rounds
        cycle_status = "needs_next_cycle"
        if new_signal_found and patch_preview:
            cycle_status = "new_rule_candidate"
        elif new_signal_found:
            cycle_status = "new_insight_found"
        elif grade in _RESOLVED_GRADES or grade == "observe":
            cycle_status = "monitoring"
        if cooldown_remaining > 0:
            cycle_status = "throttle_armed"

        cycle_entry = {
            "field_key": field_key,
            "priority_rank": int(focus.get("priority_rank") or index),
            "issue_score": float(focus.get("issue_score") or 0.0),
            "cycle_index": cycle_index,
            "grade": grade,
            "failure_mode": failure_mode,
            "new_signal_found": new_signal_found,
            "signal_key": signal_key,
            "overlooked_signals": overlooked_signals,
            "cycle_status": cycle_status,
            "rationale": str(focus.get("rationale") or ""),
            "no_new_signal_streak": no_new_signal_streak,
            "cooldown_remaining": cooldown_remaining,
            "throttle_reason": (
                f"该字段连续无新线索达到 {no_new_signal_streak} 轮，下一轮进入降频 {throttle_cooldown_rounds} 轮。"
                if cooldown_remaining > 0
                else ""
            ),
            "inspiration": _build_cycle_inspiration(
                field_key=field_key,
                failure_mode=failure_mode,
                overlooked_signals=overlooked_signals,
            ),
        }
        cycles.append(cycle_entry)
        insights.append(
            {
                "type": "field_loop_focus",
                "severity": "high" if grade not in _RESOLVED_GRADES else "low",
                "field_key": field_key,
                "detail": (
                    f"[{field_key}] cycle={cycle_index} grade={grade} "
                    f"failure_mode={failure_mode} overlooked={', '.join(overlooked_signals[:3]) or 'none'} "
                    f"streak={no_new_signal_streak} cooldown={cooldown_remaining}"
                ),
            }
        )
        if patch_preview and new_signal_found:
            proposals.append(
                {
                    "proposal_id": f"proposal_{date_str.replace('-', '')}_{index:03d}_{_slug_field(field_key)}",
                    "date": date_str,
                    "user_name": user_name,
                    "status": "proposal_only",
                    "proposal_type": proposal_type,
                    "field_key": field_key,
                    "title": f"字段循环修正：{field_key}",
                    "reason": (
                        f"cycle={cycle_index}, grade={grade}, failure_mode={failure_mode}, "
                        f"new_signal={new_signal_found}"
                    ),
                    "overlooked_signals": overlooked_signals,
                    "patch_preview": patch_preview,
                    "approval_required": True,
                }
            )

        per_field_state[field_key] = {
            "cycle_count": cycle_index,
            "last_grade": grade,
            "last_issue_score": float(focus.get("issue_score") or 0.0),
            "last_status": cycle_status,
            "last_signal_key": signal_key,
            "seen_signal_keys": seen_signals,
            "no_new_signal_streak": no_new_signal_streak,
            "cooldown_remaining": cooldown_remaining,
            "throttled_round_count": int(previous_state.get("throttled_round_count") or 0),
            "last_updated": datetime.now().isoformat(),
        }

    state_payload["updated_at"] = datetime.now().isoformat()
    state_payload["fields"] = per_field_state
    return cycles, insights, proposals, state_payload



# --- Old functions removed: _infer_failure_mode, _extract_overlooked_signals,
#     _propose_field_rule_patch, _infer_source_hints, _extract_structured_refs,
#     _tokenize_signal_text, _is_value_empty
# Now provided by services.reflection.engine (diagnose_field, extract_signals, plan_patch)
#


def _build_cycle_inspiration(
    *,
    field_key: str,
    failure_mode: str,
    overlooked_signals: List[str],
) -> str:
    if overlooked_signals:
        return (
            f"{field_key} 当前最值得追的忽视线索是 {overlooked_signals[0]}，"
            f"建议围绕 {failure_mode} 继续做字段级小循环验证。"
        )
    return f"{field_key} 本轮未抽出新线索，建议继续补充该字段反例样本后再循环。"


def _load_field_loop_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _decay_field_loop_cooldowns(state_payload: Dict[str, Any]) -> Dict[str, Any]:
    state = dict(state_payload or {})
    fields = dict(state.get("fields") or {})
    changed = False
    for field_key, raw_field in list(fields.items()):
        field = dict(raw_field or {})
        cooldown_remaining = max(0, int(field.get("cooldown_remaining") or 0))
        if cooldown_remaining <= 0:
            continue
        field["cooldown_remaining"] = max(0, cooldown_remaining - 1)
        fields[field_key] = field
        changed = True
    if changed:
        state["fields"] = fields
        state["cooldown_decayed_at"] = datetime.now().isoformat()
    return state


def _persist_field_loop_state(*, path: Path, state_payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state_payload or {}, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_field_loop_policy() -> Dict[str, int]:
    return {
        "no_signal_streak_threshold": _read_positive_int_env(
            "FIELD_LOOP_NO_SIGNAL_STREAK_THRESHOLD",
            default=FIELD_LOOP_NO_SIGNAL_STREAK_THRESHOLD_DEFAULT,
        ),
        "throttle_cooldown_rounds": _read_positive_int_env(
            "FIELD_LOOP_THROTTLE_COOLDOWN_ROUNDS",
            default=FIELD_LOOP_THROTTLE_COOLDOWN_ROUNDS_DEFAULT,
        ),
    }


def _read_positive_int_env(env_name: str, *, default: int) -> int:
    raw = str(os.environ.get(env_name, "")).strip()
    if not raw:
        return max(1, int(default))
    try:
        value = int(raw)
    except ValueError:
        return max(1, int(default))
    return max(1, value)


def _load_repo_rule_assets_safe(*, project_root: str) -> Dict[str, Dict[str, Any]]:
    try:
        from .rule_asset_loader import load_repo_rule_assets
    except Exception:
        return {"field_spec_overrides": {}, "tool_rules": {}, "call_policies": {}}
    try:
        return load_repo_rule_assets(project_root=project_root)
    except Exception:
        return {"field_spec_overrides": {}, "tool_rules": {}, "call_policies": {}}


def _slug_field(field_key: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", str(field_key or "").strip()).strip("_")
    return normalized[:36] if normalized else "field"


def _dedupe_non_empty(values: Iterable[Any]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _safe_float(value: Any, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if parsed < 0:
        return default
    if parsed > 1:
        return min(parsed / 100.0, 1.0)
    return parsed




def _determine_completion_status(
    *,
    deliverables_met: bool,
    verification_passed: bool,
    scope_respected: bool,
    stage_failure_count: int,
) -> str:
    if not scope_respected:
        return "failed"
    if not deliverables_met:
        return "failed"
    if not verification_passed:
        return "completed_with_concern"
    if stage_failure_count > 0:
        return "completed_with_concern"
    return "completed"


def _infer_deliverables_met(artifacts: Dict[str, Any]) -> bool:
    if not artifacts:
        return False
    preferred_keys = (
        "structured_profile_path",
        "downstream_audit_report_path",
        "profile_fact_decisions_path",
        "relationships_path",
    )
    if any(artifacts.get(key) for key in preferred_keys):
        return True
    return any(value for value in artifacts.values())


def _count_feedback_reruns(downstream_audit_report: Dict[str, Any]) -> int:
    feedback_loop = dict((downstream_audit_report or {}).get("feedback_loop") or {})
    count = 0
    if feedback_loop.get("protagonist_rerun_applied"):
        count += 1
    if feedback_loop.get("relationship_rerun_applied"):
        count += 1
    return count


def _derive_date(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return datetime.now().strftime("%Y-%m-%d")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).strftime("%Y-%m-%d")
    except ValueError:
        return text[:10] if len(text) >= 10 else datetime.now().strftime("%Y-%m-%d")


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
