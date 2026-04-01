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
from services.reflection.engine.patch_verifier import compute_patch_effect
from .rule_asset_loader import apply_repo_rule_patch


EVOLUTION_RELATIVE_DIR = Path("memory") / "evolution"
REFLECTION_RELATIVE_DIR = Path("memory") / "reflection"
FOCUS_FIELD_COUNT_DEFAULT = 3
FIELD_LOOP_NO_SIGNAL_STREAK_THRESHOLD_DEFAULT = 2
FIELD_LOOP_THROTTLE_COOLDOWN_ROUNDS_DEFAULT = 2
FIELD_LOOP_EXHAUSTION_CYCLE_THRESHOLD_DEFAULT = 4

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

_RESOLVED_GRADES = {"exact_match", "close_match", "improved"}


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

    # default=str: 防止 Path 等不可序列化对象导致静默失败
    trace_json_path = output_path / "memory_pipeline_run_trace.json"
    trace_json_path.write_text(
        json.dumps(normalized_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    # 版本化 ledger: {date}_c{N}.jsonl（与 field_cycles 版本号策略一致）
    user_name = normalized_payload["user_name"]
    date_str = normalized_payload["date"]
    ledger_dir = root_path / EVOLUTION_RELATIVE_DIR / "traces" / user_name
    ledger_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(ledger_dir.glob(f"{date_str}_c*.jsonl"))
    next_version = len(existing) + 1
    versioned_path = ledger_dir / f"{date_str}_c{next_version}.jsonl"

    line = json.dumps(normalized_payload, ensure_ascii=False, default=str) + "\n"
    # 写版本化 ledger
    versioned_path.write_text(line, encoding="utf-8")
    # 同时 append 到按日汇总 ledger（evolve 读这个）
    ledger_path = ledger_dir / f"{date_str}.jsonl"
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(line)

    return {
        "trace_id": str(normalized_payload.get("trace_id") or ""),
        "trace_json_path": str(trace_json_path),
        "trace_ledger_path": str(ledger_path),
        "trace_versioned_path": str(versioned_path),
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
    priority_fields: List[str] | None = None,
) -> Dict[str, Any]:
    traces = load_memory_run_traces(project_root=project_root, user_name=user_name, date_str=date_str)

    # 清空旧进度文件，开始新的进度追踪
    _progress_path = Path(project_root) / EVOLUTION_RELATIVE_DIR / f"_progress_{user_name}.jsonl"
    _progress_path.unlink(missing_ok=True)

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
        priority_fields=priority_fields,
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

    # 文件名加版本号（全局递增），防止同一天多轮覆盖
    existing_versions = sorted(proposals_dir.glob(f"{date_str}_c*.json")) if proposals_dir.exists() else []
    next_version = len(existing_versions) + 1
    version_suffix = f"{date_str}_c{next_version}"
    report_path = reports_dir / f"{version_suffix}.json"
    insights_path = insights_dir / f"{version_suffix}.json"
    proposals_path = proposals_dir / f"{version_suffix}.json"
    focus_path = focus_dir / f"{version_suffix}.json"
    field_cycles_path = field_cycles_dir / f"{version_suffix}.json"
    # 同时保留一份 {date}.json 作为最新版本（前端读取用）
    latest_report_path = reports_dir / f"{date_str}.json"
    latest_proposals_path = proposals_dir / f"{date_str}.json"
    latest_field_cycles_path = field_cycles_dir / f"{date_str}.json"
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
    # 写 latest 版本（前端 API 读取用）— 直接覆盖，反映当前轮次的最新状态
    latest_report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # latest proposals: 收集所有 active 字段的最新提案（本轮 + 历史版本中不在本轮焦点的）
    _state_path = Path(project_root) / EVOLUTION_RELATIVE_DIR / "field_loop_state" / f"{user_name}.json"
    _persisted_state = json.loads(_state_path.read_text(encoding="utf-8")) if _state_path.exists() else {}
    _active_statuses = {"needs_next_cycle", "new_rule_candidate", "new_insight_found", "initial_snapshot"}
    _field_states = _persisted_state.get("fields") or {}
    # 本轮提案按 field_key 索引
    current_by_field: Dict[str, Dict[str, Any]] = {}
    for p in proposals:
        fk = p.get("field_key", "")
        if fk and _field_states.get(fk, {}).get("last_status") in _active_statuses:
            current_by_field[fk] = p
    # 从历史版本中补充本轮不在焦点的 active 字段的最新提案
    _proposals_dir = Path(project_root) / EVOLUTION_RELATIVE_DIR / "proposals" / user_name
    if _proposals_dir.exists():
        for vf in sorted(_proposals_dir.glob(f"{date_str}_c*.json"), reverse=True):
            try:
                hist = json.loads(vf.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(hist, list):
                continue
            for hp in hist:
                fk = hp.get("field_key", "")
                if fk and fk not in current_by_field and _field_states.get(fk, {}).get("last_status") in _active_statuses:
                    current_by_field[fk] = hp
    latest_proposals = list(current_by_field.values())
    if latest_proposals:
        latest_proposals_path.write_text(json.dumps(latest_proposals, ensure_ascii=False, indent=2), encoding="utf-8")
    elif latest_proposals_path.exists():
        latest_proposals_path.unlink()
    latest_field_cycles_path.write_text(
        json.dumps(field_loop_payload.get("field_cycles", []), ensure_ascii=False, indent=2), encoding="utf-8",
    )

    # 收敛检测：如果所有活跃字段都已暂停或达标，生成复盘
    _recap_state_path = Path(project_root) / EVOLUTION_RELATIVE_DIR / "field_loop_state" / f"{user_name}.json"
    persisted_state = json.loads(_recap_state_path.read_text(encoding="utf-8")) if _recap_state_path.exists() else {}
    converged = _check_and_generate_recap(
        project_root=project_root,
        user_name=user_name,
        date_str=date_str,
        state_payload=persisted_state,
        field_cycles_all=field_loop_payload.get("field_cycles", []),
    )

    _emit_progress(project_root, user_name, {
        "event": "evolve_complete",
        "total_proposals": len(proposals),
        "total_focus_fields": len(field_loop_payload.get("focus_fields", [])),
        "converged": converged,
    })

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
        "converged": converged,
    }


def _check_and_generate_recap(
    *,
    project_root: str,
    user_name: str,
    date_str: str,
    state_payload: Dict[str, Any],
    field_cycles_all: List[Dict[str, Any]],
) -> bool:
    """检测用户是否收敛，如果收敛则生成复盘报告。

    收敛条件：没有任何字段还在"活跃探索"或"临时降频"中。
    throttle_armed / throttled 视为仍在循环中（暂停但会回来），不算收敛。
    """
    fields = state_payload.get("fields") or {}
    # 活跃 = 正在探索的 + 临时降频的（throttle 是暂停不是结束）
    _NOT_CONVERGED = {
        "needs_next_cycle", "new_rule_candidate", "new_insight_found",
        "initial_snapshot", "throttle_armed", "throttled",
    }
    active_fields = [
        fk for fk, fs in fields.items()
        if fs.get("last_status") in _NOT_CONVERGED and fs.get("cycle_count", 0) > 0
    ]
    if active_fields:
        return False

    # 所有字段收敛 → 生成复盘
    recap_dir = Path(project_root) / EVOLUTION_RELATIVE_DIR / "recaps" / user_name
    recap_dir.mkdir(parents=True, exist_ok=True)
    recap_path = recap_dir / f"{date_str}.json"

    # 收集所有历史轮次的字段数据
    cycles_dir = Path(project_root) / EVOLUTION_RELATIVE_DIR / "field_cycles" / user_name
    all_history: Dict[str, List[Dict[str, Any]]] = {}
    if cycles_dir.exists():
        for f in sorted(cycles_dir.glob("*.json")):
            data = json.loads(f.read_text(encoding="utf-8")) if f.exists() else []
            if not isinstance(data, list):
                continue
            for c in data:
                fk = c.get("field_key", "")
                if fk not in all_history:
                    all_history[fk] = []
                all_history[fk].append(c)

    # 收集审批历史
    approvals: List[Dict[str, Any]] = []
    approval_path = Path(project_root) / EVOLUTION_RELATIVE_DIR / "approval_history" / user_name / "approvals.jsonl"
    if approval_path.exists():
        for line in approval_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    approvals.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    # 构建复盘
    max_cycle = max((fs.get("cycle_count", 0) for fs in fields.values()), default=0)
    reflected_fields = [fk for fk, fs in fields.items() if fs.get("cycle_count", 0) > 0]
    approved_fields = list(set(a.get("field_key", "") for a in approvals if a.get("field_key")))

    field_recaps = []
    for fk in reflected_fields:
        fs = fields.get(fk, {})
        history = all_history.get(fk, [])
        field_recaps.append({
            "field_key": fk,
            "total_cycles": fs.get("cycle_count", 0),
            "final_status": fs.get("last_status", ""),
            "final_grade": fs.get("last_grade", ""),
            "score_history": fs.get("issue_score_history", []),
            "score_trend": fs.get("score_trend", ""),
            "was_approved": fk in approved_fields,
            "cycle_history": [
                {
                    "cycle_index": c.get("cycle_index"),
                    "grade": c.get("grade"),
                    "failure_mode": c.get("failure_mode"),
                    "cycle_status": c.get("cycle_status"),
                    "judgment_summary_zh": c.get("judgment_summary_zh", ""),
                    "proposed_reasoning_direction": c.get("proposed_reasoning_direction", ""),
                }
                for c in history
            ],
        })

    # Token 黑洞检测
    token_summary = _load_token_usage_summary(project_root, user_name)
    token_blackholes = _detect_token_blackholes(user_name, state_payload, token_summary)

    recap = {
        "user_name": user_name,
        "date": date_str,
        "converged": True,
        "total_rounds": max_cycle,
        "total_reflected_fields": len(reflected_fields),
        "total_approved_fields": len(approved_fields),
        "approved_fields": approved_fields,
        "field_recaps": field_recaps,
        "token_blackholes": token_blackholes,
        "generated_at": datetime.now().isoformat(),
    }

    recap_path.write_text(json.dumps(recap, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [recap] 用户 {user_name} 已收敛，复盘已生成: {recap_path}")
    return True


def _run_single_user_evolve(
    project_root: str,
    user_name: str,
    date_str: str,
    top_k_fields: int,
) -> Dict[str, Any]:
    """单用户 evolve 包装函数（供 ProcessPoolExecutor 调用）。"""
    result = run_memory_nightly_evaluation(
        project_root=project_root,
        user_name=user_name,
        date_str=date_str,
        top_k_fields=top_k_fields,
    )
    return {
        "user_name": user_name,
        "status": "ok",
        "report_path": result["report_path"],
        "insights_path": result["insights_path"],
        "proposals_path": result["proposals_path"],
        "focus_fields_path": result.get("focus_fields_path"),
        "field_cycles_path": result.get("field_cycles_path"),
        "total_traces": int(result.get("total_traces") or 0),
        "total_focus_fields": int(result.get("total_focus_fields") or 0),
        "total_proposals": int(result.get("total_proposals") or 0),
    }


def run_memory_nightly_user_set_evaluation(
    *,
    project_root: str = PROJECT_ROOT,
    user_names: Iterable[str] | None,
    date_str: str,
    top_k_fields: int = FOCUS_FIELD_COUNT_DEFAULT,
) -> Dict[str, Any]:
    resolved_users = _resolve_user_set(project_root=project_root, user_names=user_names)
    max_workers = int(os.environ.get("EVOLVE_PARALLEL_WORKERS", "3"))

    results: List[Dict[str, Any]] = []
    failed_users: List[str] = []

    if max_workers > 1 and len(resolved_users) > 1:
        # ── 并行执行 ──
        from concurrent.futures import ProcessPoolExecutor, as_completed
        user_futures: Dict[Any, str] = {}
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            for user_name in resolved_users:
                future = pool.submit(
                    _run_single_user_evolve,
                    project_root, user_name, date_str, top_k_fields,
                )
                user_futures[future] = user_name

            for future in as_completed(user_futures):
                user_name = user_futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    print(f"  [evolve] {user_name} failed: {exc}")
                    failed_users.append(user_name)
                    results.append({
                        "user_name": user_name,
                        "status": "error",
                        "reason": str(exc),
                        "total_traces": 0,
                        "total_focus_fields": 0,
                        "total_proposals": 0,
                    })

        # ── 失败用户重试一次（串行，避免加剧 rate limit）──
        for user_name in failed_users:
            print(f"  [evolve] Retrying {user_name} (serial)...")
            try:
                retry_result = _run_single_user_evolve(
                    project_root, user_name, date_str, top_k_fields,
                )
                # 替换之前的 error 记录
                results = [r for r in results if r.get("user_name") != user_name]
                results.append(retry_result)
                print(f"  [evolve] {user_name} retry succeeded")
            except Exception as exc:
                print(f"  [evolve] {user_name} retry also failed: {exc}")
    else:
        # ── 串行执行（单用户或 workers=1）──
        for user_name in resolved_users:
            try:
                results.append(_run_single_user_evolve(
                    project_root, user_name, date_str, top_k_fields,
                ))
            except Exception as exc:
                print(f"  [evolve] {user_name} failed: {exc}")
                results.append({
                    "user_name": user_name,
                    "status": "error",
                    "reason": str(exc),
                    "total_traces": 0,
                    "total_focus_fields": 0,
                    "total_proposals": 0,
                })

    # ── Aggregate report（并行区域外，无竞争）──
    ok_results = [r for r in results if r.get("status") != "error"]
    aggregate = {
        "date": date_str,
        "total_users": len(resolved_users),
        "succeeded": len(ok_results),
        "failed": len(results) - len(ok_results),
        "users": results,
        "summary": {
            "total_traces": sum(int(r.get("total_traces") or 0) for r in ok_results),
            "total_focus_fields": sum(int(r.get("total_focus_fields") or 0) for r in ok_results),
            "total_proposals": sum(int(r.get("total_proposals") or 0) for r in ok_results),
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
        "succeeded": len(ok_results),
        "failed": len(results) - len(ok_results),
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
                                "weak_evidence_caution": [
                                    "证据仅来自单事件窗口时输出 null",
                                    "主体归属不清时输出 null",
                                ]
                            },
                            "long_term_facts.material.brand_preference": {
                                "weak_evidence_caution": [
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


# ── Reflection agent integration ──

_REFLECTION_AGENT_CACHE = None


def _get_or_create_reflection_agent(project_root: str):
    """Lazy-init the UpstreamReflectionAgent (expensive LLM processor inside)."""
    global _REFLECTION_AGENT_CACHE
    if _REFLECTION_AGENT_CACHE is None:
        try:
            from services.reflection.upstream_agent import UpstreamReflectionAgent
            _REFLECTION_AGENT_CACHE = UpstreamReflectionAgent(project_root=project_root)
        except Exception:
            _REFLECTION_AGENT_CACHE = None
    return _REFLECTION_AGENT_CACHE


def _load_case_facts_by_field(project_root: str, user_name: str) -> Dict[str, Dict[str, Any]]:
    """Load case_facts indexed by dimension (field_key), profile fields only."""
    path = Path(project_root) / REFLECTION_RELATIVE_DIR / f"case_facts_{user_name}.jsonl"
    result: Dict[str, Dict[str, Any]] = {}
    for rec in _load_jsonl_records(path):
        dim = str(rec.get("dimension") or "").strip()
        if not dim:
            continue
        # Only profile fields (skip primary/relationship case_facts)
        src = str(rec.get("signal_source") or "")
        if "profile" not in src and rec.get("entity_type") != "profile_field":
            continue
        result[dim] = rec
    return result


def _try_reflect_field(
    field_key: str,
    case_facts_by_field: Dict[str, Dict[str, Any]],
    project_root: str,
    reviewer_note: str = "",
    evolution_context: Dict[str, Any] | None = None,
    user_name: str = "",
) -> Dict[str, Any] | None:
    """Attempt LLM reflection on a field. Returns reflect result or None on failure."""
    cf = case_facts_by_field.get(field_key)
    if not cf or not cf.get("trace_payload_path"):
        return None
    agent = _get_or_create_reflection_agent(project_root)
    if agent is None:
        return None
    # 标记 user/field 供 token 计量使用
    if hasattr(agent, "llm_processor") and agent.llm_processor:
        agent.llm_processor._usage_user = user_name
        agent.llm_processor._usage_field = field_key
    try:
        from services.reflection.upstream_agent import BadcasePacketAssembler
        assembler = BadcasePacketAssembler(project_root=project_root)
        packet = assembler.assemble(cf)
        # 临时注入审批备注（用完即弃，不写入任何持久化文件）
        if reviewer_note:
            packet["human_reviewer_note"] = reviewer_note
        # 注入自循环上下文（上一轮结果、趋势、改进效果）
        if evolution_context:
            packet["evolution_context"] = evolution_context
        result = agent.reflect(packet)
        if result.get("status") == "failed":
            return None
        # 转录持久化：保存完整 prompt 输入 + reflect 输出
        _save_transcript(project_root, user_name, field_key, packet, result)
        return result
    except Exception as exc:
        print(f"  [reflect] {field_key} failed: {exc}")
        return None


def _save_transcript(project_root: str, user_name: str, field_key: str,
                     packet: Dict[str, Any], result: Dict[str, Any]) -> None:
    """转录持久化：保存 reflect 的输入和输出供事后审计。"""
    try:
        transcript_dir = Path(project_root) / EVOLUTION_RELATIVE_DIR / "transcripts" / user_name
        transcript_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y-%m-%d")
        transcript_path = transcript_dir / f"{date_str}.jsonl"
        entry = {
            "ts": datetime.now().isoformat(),
            "field_key": field_key,
            "input_keys": list(packet.keys()),
            "evolution_context": packet.get("evolution_context"),
            "result_status": result.get("status"),
            "result_confidence": result.get("confidence"),
            "root_cause_family": result.get("root_cause_family"),
            "judgment_summary_zh": result.get("judgment_summary_zh", "")[:200],
            "patch_intent_summary": (result.get("patch_intent") or {}).get("change_summary_zh", "")[:200],
        }
        with open(transcript_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 转录失败不影响主逻辑


def _load_reviewer_notes(project_root: str, user_name: str) -> Dict[str, str]:
    """Load the latest reviewer_note per field_key from proposal_actions.jsonl.

    Returns {field_key: note}. Only the most recent note per field is kept.
    These are ephemeral — injected into reflect() for one round then discarded.
    """
    actions_path = Path(project_root) / REFLECTION_RELATIVE_DIR / "proposal_actions.jsonl"
    if not actions_path.exists():
        return {}
    notes: Dict[str, str] = {}
    for line in actions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        note = str(record.get("reviewer_note") or "").strip()
        if not note:
            continue
        # 从 proposal_id 提取 field_key（需要查 proposals 文件）
        pid = record.get("proposal_id", "")
        # 也检查 approval_history 里的 field_key
        # 简单方案：遍历该用户的 proposals 找 field_key
        for subdir in (Path(project_root) / EVOLUTION_RELATIVE_DIR / "proposals" / user_name).glob("*.json"):
            try:
                proposals = json.loads(subdir.read_text(encoding="utf-8"))
                if not isinstance(proposals, list):
                    continue
                for p in proposals:
                    if p.get("proposal_id", "").startswith(pid[:30]) or pid.startswith(p.get("proposal_id", "")[:30]):
                        fk = p.get("field_key", "")
                        if fk:
                            notes[fk] = note
            except Exception:
                continue
    return notes


def _load_vlm_summaries(*, project_root: str, user_name: str) -> Dict[str, str]:
    """Load photo_id → short summary from vlm_cache.json for evidence search."""
    source_dir = Path(project_root) / "datasets" / user_name / "source"
    for filename in ("vlm_cache.json", "vp1_observations.json"):
        path = source_dir / filename
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        items = data if isinstance(data, list) else data.get("photos", []) if isinstance(data, dict) else []
        index: Dict[str, str] = {}
        for item in items:
            pid = str(item.get("photo_id") or "").strip()
            if not pid:
                continue
            vlm = item.get("vlm_analysis") or {}
            summary = str(vlm.get("summary") or "").strip()
            if not summary:
                scene = vlm.get("scene") or {}
                loc = scene.get("location_detected", "") if isinstance(scene, dict) else ""
                activity = ((vlm.get("event") or {}).get("activity", "")) if isinstance(vlm.get("event"), dict) else ""
                summary = " ".join(filter(None, [loc, activity]))
            if summary:
                index[pid] = summary[:200]
        if index:
            return index
    return {}


def _load_event_summaries(*, project_root: str, user_name: str) -> Dict[str, str]:
    """Load event_id → description from events data for evidence search."""
    source_dir = Path(project_root) / "datasets" / user_name / "source"
    for filename in ("lp1_events.json", "lp1_events_compact.json"):
        path = source_dir / filename
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        index: Dict[str, str] = {}
        for item in data:
            eid = str(item.get("event_id") or "").strip()
            if not eid:
                continue
            title = str(item.get("title") or "").strip()
            narrative = str(item.get("narrative_synthesis") or "").strip()
            desc = narrative[:200] if narrative else title
            if desc:
                index[eid] = desc
        if index:
            return index
    return {}


def _load_token_usage_summary(project_root: str, user_name: str) -> Dict[str, Dict[str, Any]]:
    """从 llm_usage.jsonl 聚合每个 field 的 token 消耗。"""
    log_path = Path(project_root) / EVOLUTION_RELATIVE_DIR / "llm_usage.jsonl"
    if not log_path.exists():
        return {}
    summary: Dict[str, Dict[str, Any]] = {}
    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("user") != user_name:
                continue
            fk = entry.get("field", "")
            if not fk:
                continue
            item = summary.setdefault(fk, {"total_in": 0, "total_out": 0, "call_count": 0, "callers": {}})
            item["total_in"] += int(entry.get("in", 0))
            item["total_out"] += int(entry.get("out", 0))
            item["call_count"] += 1
            caller = entry.get("caller", "unknown")
            item["callers"][caller] = item["callers"].get(caller, 0) + 1
    except Exception:
        pass
    return summary


def _emit_progress(project_root: str, user_name: str, event_data: Dict[str, Any]) -> None:
    """写入进度事件到 _progress_{user}.jsonl（供 SSE 推送）。"""
    try:
        path = Path(project_root) / EVOLUTION_RELATIVE_DIR / f"_progress_{user_name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**event_data, "ts": datetime.now().isoformat()}, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _detect_token_blackholes(
    user_name: str,
    field_state: Dict[str, Any],
    token_summary: Dict[str, Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """检测 token 黑洞字段：消耗远超平均但 grade 未改善。"""
    if not token_summary:
        return []
    # 计算平均 token 消耗
    all_totals = [v["total_in"] + v["total_out"] for v in token_summary.values()]
    if not all_totals:
        return []
    avg_tokens = sum(all_totals) / len(all_totals)
    if avg_tokens == 0:
        return []

    fields = field_state.get("fields", {})
    blackholes = []
    for fk, usage in token_summary.items():
        total = usage["total_in"] + usage["total_out"]
        fs = fields.get(fk, {})
        cycle_count = int(fs.get("cycle_count", 0))
        score_trend = str(fs.get("score_trend", ""))
        last_grade = str(fs.get("last_grade", ""))

        # 判定条件：token > 均值 ×2 且 grade 未改善 且至少 3 轮
        if (total > avg_tokens * 2
                and score_trend in ("stable", "rising", "oscillating")
                and cycle_count >= 3
                and last_grade not in ("exact_match", "close_match", "improved")):
            blackholes.append({
                "field_key": fk,
                "total_tokens": total,
                "avg_tokens": round(avg_tokens),
                "ratio": round(total / avg_tokens, 1),
                "cycle_count": cycle_count,
                "score_trend": score_trend,
                "last_grade": last_grade,
                "call_count": usage["call_count"],
            })
    blackholes.sort(key=lambda x: x["total_tokens"], reverse=True)
    return blackholes


def _run_gt_field_loop(
    *,
    project_root: str,
    user_name: str,
    date_str: str,
    top_k_fields: int,
    priority_fields: List[str] | None = None,
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
        priority_fields=priority_fields,
    )
    assets = _load_repo_rule_assets_safe(project_root=project_root)

    # Load VLM + event summaries for evidence-based signal extraction
    vlm_summaries = _load_vlm_summaries(project_root=project_root, user_name=user_name)
    event_summaries = _load_event_summaries(project_root=project_root, user_name=user_name)

    # Load case_facts for reflect() — indexed by field_key
    case_facts_by_field = _load_case_facts_by_field(project_root, user_name)

    # Load reviewer notes — ephemeral, injected into reflect then discarded
    reviewer_notes = _load_reviewer_notes(project_root, user_name)

    field_cycles, field_insights, field_proposals, next_state = _build_gt_field_cycles(
        date_str=date_str,
        user_name=user_name,
        focus_fields=focus_fields,
        latest_by_field=latest_by_field,
        gt_records=gt_records,
        field_state=field_state,
        current_assets=assets,
        policy=policy,
        vlm_summaries=vlm_summaries,
        event_summaries=event_summaries,
        case_facts_by_field=case_facts_by_field,
        project_root=project_root,
        reviewer_notes=reviewer_notes,
    )
    # 补全非焦点字段的状态（让热力图能看到所有 GT 对比过的字段）
    all_fields_in_state = next_state.get("fields") or {}
    for comp in comparisons:
        fk = str(comp.get("field_key") or "").strip()
        if not fk or fk in all_fields_in_state:
            continue
        cr = comp.get("comparison_result") or {}
        grade = str(cr.get("grade") or "").strip()
        if not grade:
            continue
        all_fields_in_state[fk] = {
            "cycle_count": 0,
            "last_grade": grade,
            "last_issue_score": 0.0,
            "last_status": "monitoring" if grade in ("exact_match", "close_match", "improved") else "not_focused",
            "last_signal_key": "",
            "seen_signal_keys": [],
            "no_new_signal_streak": 0,
            "cooldown_remaining": 0,
            "throttled_round_count": 0,
            "issue_score_history": [],
            "score_trend": "",
            "last_patch_grade": None,
            "last_patch_score": None,
            "reflect_fail_streak": 0,
            "last_updated": datetime.now().isoformat(),
        }
    next_state["fields"] = all_fields_in_state

    # 耗尽字段检测（基于多轮 Reflect Agent LLM 信号累积）— 必须在 persist state 之前，
    # 以便 exhausted 标记写入 field_loop_state，让下轮 focus selection 能过滤掉
    token_summary = _load_token_usage_summary(project_root, user_name)
    exhausted = _persist_exhausted_candidates(
        project_root=project_root,
        user_name=user_name,
        field_state=next_state,
        policy=policy,
        token_summary=token_summary,
    )
    for c in (exhausted or []):
        fk = c.get("field_key", "")
        if fk and fk in next_state.get("fields", {}):
            next_state["fields"][fk]["exhausted"] = True
            next_state["fields"][fk]["exhaustion_reason"] = c.get("exhaustion_reason", "")

    _persist_field_loop_state(path=state_path, state_payload=next_state)

    # 统计全局状态（不只是本轮 focus，包括所有字段）
    all_fields = next_state.get("fields") or {}
    global_throttled = sum(1 for fs in all_fields.values() if fs.get("last_status") in ("throttled", "throttle_armed"))
    global_active = sum(1 for fs in all_fields.values() if fs.get("last_status") in ("needs_next_cycle", "new_rule_candidate", "new_insight_found"))
    global_terminal = sum(1 for fs in all_fields.values() if fs.get("last_status") in ("monitoring", "not_focused") or fs.get("exhausted") or fs.get("human_resolved"))
    all_throttled_no_active = global_active == 0 and global_throttled > 0

    summary = {
        "total_focus_fields": len(focus_fields),
        "new_signal_count": sum(1 for item in field_cycles if item.get("new_signal_found")),
        "new_rule_candidate_count": sum(1 for item in field_cycles if item.get("cycle_status") == "new_rule_candidate"),
        "throttled_fields_count": sum(1 for item in field_cycles if item.get("cycle_status") == "throttled"),
        "active_focus_fields_count": sum(1 for item in field_cycles if item.get("cycle_status") != "throttled"),
        "no_signal_streak_threshold": int(policy["no_signal_streak_threshold"]),
        "throttle_cooldown_rounds": int(policy["throttle_cooldown_rounds"]),
        "global_active": global_active,
        "global_throttled": global_throttled,
        "global_terminal": global_terminal,
        "all_throttled_no_active": all_throttled_no_active,
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
    priority_fields: List[str] | None = None,
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
        # 人工修正的 grade 优先于机器判定
        grade = str(comparison_result.get("grade") or "mismatch")
        if comparison_result.get("human_override"):
            grade = str(comparison_result.get("grade") or grade)
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

    # 已匹配的字段不进入焦点
    _SKIP_GRADES = {"exact_match", "close_match", "improved"}

    focus_candidates: List[Dict[str, Any]] = []
    for field_key, entry in scored_by_field.items():
        if int(entry["badcase_count"]) <= 0:
            continue
        if entry.get("latest_grade") in _SKIP_GRADES:
            continue
        # 人工标记已修正的字段不再进入焦点
        saved = per_field_state.get(field_key) or {}
        if saved.get("human_resolved"):
            continue
        # 耗尽字段不再进入焦点（LLM 反复判定无法改善）
        if saved.get("exhausted"):
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
        candidate["reflect_fail_streak"] = int(saved_state.get("reflect_fail_streak") or 0)
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
            -int(item.get("reflect_fail_streak") or 0),  # reflect 连续失败的排后面
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
    # throttled 字段不占名额 — 它们在 _build_gt_field_cycles 里会 skip reflect，
    # 浪费 top_k 名额。只有在完全没有 active 候选时才补入（防止空转）。
    if not selected:
        for candidate in throttled_candidates:
            if len(selected) >= limit:
                break
            selected.append(candidate)

    trimmed = selected[:limit]

    # priority_fields: rerun 后的字段必须进入焦点（在 top_k 之外额外加入）
    if priority_fields:
        selected_keys = {item["field_key"] for item in trimmed}
        all_candidates_by_key = {item["field_key"]: item for item in focus_candidates}
        for pf in priority_fields:
            if pf in selected_keys:
                continue
            if pf in all_candidates_by_key:
                trimmed.append(all_candidates_by_key[pf])
            elif pf in scored_by_field:
                # 即使被 _SKIP_GRADES 过滤掉了，rerun 后也要进来看新结果
                trimmed.append(scored_by_field[pf])

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
    vlm_summaries: Dict[str, str] | None = None,
    event_summaries: Dict[str, str] | None = None,
    case_facts_by_field: Dict[str, Dict[str, Any]] | None = None,
    project_root: str = "",
    reviewer_notes: Dict[str, str] | None = None,
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
    _consecutive_system_failures = 0  # 连续系统性故障计数（API 超时/异常，非 LLM 无提案）

    # 进度推送：本轮开始
    _emit_progress(project_root, user_name, {
        "event": "cycle_start",
        "fields": [str(f.get("field_key", "")) for f in focus_fields],
        "total_fields": len(focus_fields),
    })

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
        rejected_signals = [str(item or "").strip() for item in list(previous_state.get("rejected_signal_keys") or []) if str(item or "").strip()]
        signal_result = engine_extract_signals(
            field_key=field_key,
            output_value=output_value,
            gt_value=gt_value,
            gt_notes=str(gt_row.get("notes") or ""),
            evidence_summary=str(gt_row.get("evidence_summary") or ""),
            accuracy_note=str(gt_row.get("accuracy_note") or ""),
            seen_signal_keys=seen_signals,
            vlm_summaries=vlm_summaries,
            event_summaries=event_summaries,
            excluded_signal_keys=rejected_signals,
        )
        overlooked_signals = signal_result.signals
        signal_key = signal_result.signal_key
        new_signal_found = signal_result.is_new
        if new_signal_found:
            seen_signals.append(signal_key)

        no_new_signal_streak = 0 if new_signal_found else previous_streak + 1

        # ── 构建新线索摘要（仅当本轮发现新信号时）──
        overlooked_evidence_digest: list[dict[str, str]] = []
        if new_signal_found and overlooked_signals:
            for sig in overlooked_signals[:3]:
                if sig.startswith("evidence_ref:"):
                    ref_id = sig[len("evidence_ref:"):]
                    summary = ""
                    if ref_id.startswith("photo_"):
                        summary = (vlm_summaries or {}).get(ref_id, "")
                    elif ref_id.startswith("EVT_"):
                        summary = (event_summaries or {}).get(ref_id, "")
                    if summary:
                        overlooked_evidence_digest.append({"ref": ref_id, "summary": summary[:120]})

        # ── 计算上轮 patch 效果（提前算，传给 reflect agent）──
        pre_reflect_patch_effect = None
        _prev_patch_grade = previous_state.get("last_patch_grade")
        _prev_patch_score = previous_state.get("last_patch_score")
        if _prev_patch_grade and grade:
            pre_reflect_patch_effect = compute_patch_effect(
                before_grade=_prev_patch_grade,
                before_score=float(_prev_patch_score or 0.0),
                after_grade=grade,
                after_score=float(comparison_result.get("score") or 0.0),
            )

        # ── 构建自循环上下文 ──
        evolution_context = {
            "cycle_index": cycle_index,
            "previous_grade": str(previous_state.get("last_grade") or ""),
            "previous_score": float(previous_state.get("last_issue_score") or 0.0),
            "score_trend": str(previous_state.get("score_trend") or ""),
            "no_new_signal_streak": int(previous_state.get("no_new_signal_streak") or 0),
            "patch_effect": pre_reflect_patch_effect,
            "last_proposed_direction": str(previous_state.get("last_proposed_direction") or ""),
            **({"overlooked_evidence": overlooked_evidence_digest} if overlooked_evidence_digest else {}),
        }

        # ── LLM 反思（唯一的提案来源）──
        _emit_progress(project_root, user_name, {
            "event": "field_start", "field": field_key, "index": index,
            "total": len(focus_fields), "grade": grade,
        })
        field_reviewer_note = (reviewer_notes or {}).get(field_key, "")
        reflect_result = _try_reflect_field(
            field_key, case_facts_by_field or {}, project_root,
            reviewer_note=field_reviewer_note,
            evolution_context=evolution_context if (cycle_index > 1 or overlooked_evidence_digest) else None,
            user_name=user_name,
        )
        # 级别 1: 系统性故障重试（reflect_result is None = 异常/超时，不是"LLM 没有新提案"）
        if reflect_result is None:
            import time as _time
            _time.sleep(3)
            print(f"  [reflect] {field_key} 系统故障，3s 后重试...")
            reflect_result = _try_reflect_field(
                field_key, case_facts_by_field or {}, project_root,
                reviewer_note=field_reviewer_note,
                evolution_context=evolution_context if (cycle_index > 1 or overlooked_evidence_digest) else None,
                user_name=user_name,
            )

        cf = (case_facts_by_field or {}).get(field_key, {})
        original_reasoning = str((cf.get("decision_trace") or {}).get("reasoning") or "")

        has_valid_proposal = (
            reflect_result is not None
            and reflect_result.get("status") in ("ok", "needs_review")
            and reflect_result.get("confidence", 0) >= 0.6
        )

        # reflect fail streak — 连续无法反思就换字段
        prev_reflect_fail_streak = int(previous_state.get("reflect_fail_streak") or 0)
        if has_valid_proposal:
            reflect_fail_streak = 0
        elif reflect_result is None:
            # 系统性故障（重试后仍然失败）
            reflect_fail_streak = prev_reflect_fail_streak + 1
            _consecutive_system_failures += 1
        else:
            # LLM 正常返回但没有合适的提案 — 不算系统故障
            reflect_fail_streak = prev_reflect_fail_streak + 1
            _consecutive_system_failures = 0  # 重置：LLM 能正常工作

        # 级别 3: 连续系统故障中止（3 个字段重试后都返回 None = 系统性问题）
        if _consecutive_system_failures >= 3:
            print(f"  [evolve] 连续 {_consecutive_system_failures} 个字段系统故障（API 超时/异常），中止本轮")
            break

        if has_valid_proposal:
            patch_intent = reflect_result.get("patch_intent") or {}
            # rule_patch 是可执行的规则修改（field_spec_overrides/tool_rules/call_policies）
            rule_patch = patch_intent.get("rule_patch") or {}
            # patch_preview 包含完整信息：可执行 patch + 方向说明
            patch_preview = {
                **patch_intent,
                # 确保 rule_patch 里的三个 key 存在（apply_repo_rule_patch 需要）
                "field_spec_overrides": rule_patch.get("field_spec_overrides") or {},
                "tool_rules": rule_patch.get("tool_rules") or {},
                "call_policies": rule_patch.get("call_policies") or {},
            }
            proposal_type = reflect_result.get("recommended_fix_surface", "field_cot")
        else:
            patch_preview = {}
            proposal_type = ""

        # P2: issue_score 时间序列 + 趋势分类
        current_score = float(focus.get("issue_score") or 0.0)
        previous_history = list(previous_state.get("issue_score_history") or [])
        issue_score_history = (previous_history + [current_score])[-8:]
        score_trend = _classify_score_trend(issue_score_history)

        # P2: 收敛逻辑增强
        cooldown_remaining = 0
        if score_trend == "stable" and no_new_signal_streak >= 1:
            cooldown_remaining = throttle_cooldown_rounds
        elif score_trend == "rising" and new_signal_found:
            no_new_signal_streak = 0
            cooldown_remaining = 0
        elif not new_signal_found and no_new_signal_streak >= no_signal_streak_threshold:
            cooldown_remaining = throttle_cooldown_rounds
        cycle_status = "needs_next_cycle"
        if has_valid_proposal:
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
            "patch_preview": dict(patch_preview) if patch_preview else {},
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
            # reflect 结论
            "judgment_summary_zh": (reflect_result or {}).get("judgment_summary_zh", ""),
            "original_reasoning": original_reasoning,
            "proposed_reasoning_direction": (reflect_result or {}).get("patch_intent", {}).get("change_summary_zh", ""),
            "key_evidence_zh": (reflect_result or {}).get("key_evidence_zh", []),
            "root_cause_family": (reflect_result or {}).get("root_cause_family", failure_mode),
            "why_this_surface_zh": (reflect_result or {}).get("why_this_surface_zh", ""),
            # reflect 原始信号（用于疑难杂症耗尽判定）
            "reflect_status": (reflect_result or {}).get("status", ""),
            "reflect_confidence": _safe_float((reflect_result or {}).get("confidence"), default=0.0),
            # 实际输出值与 GT（供热力图展示）
            "output_value": output_value,
            "gt_value": gt_value,
        }
        cycles.append(cycle_entry)
        _emit_progress(project_root, user_name, {
            "event": "field_done", "field": field_key, "index": index,
            "total": len(focus_fields), "has_proposal": has_valid_proposal,
            "grade": grade, "cycle_status": cycle_status,
        })
        insights.append(
            {
                "type": "field_loop_focus",
                "severity": "high" if grade not in _RESOLVED_GRADES else "low",
                "field_key": field_key,
                "detail": (
                    f"[{field_key}] cycle={cycle_index} grade={grade} "
                    f"failure_mode={failure_mode} reflect={'ok' if has_valid_proposal else 'no'} "
                    f"streak={no_new_signal_streak} cooldown={cooldown_remaining}"
                ),
            }
        )
        # 提案：仅在 reflect 成功时生成
        if has_valid_proposal:
            # missing_gt = 系统有输出但 GT 没标注 → 如果人类认可，这是召回提升
            improvement_type = "recall" if grade == "missing_gt" else "precision"
            proposals.append(
                {
                    "proposal_id": f"proposal_{date_str.replace('-', '')}_c{cycle_index}_{uuid.uuid4().hex[:6]}_{_slug_field(field_key)}",
                    "date": date_str,
                    "user_name": user_name,
                    "status": "proposal_only",
                    "proposal_type": proposal_type,
                    "field_key": field_key,
                    "title": reflect_result.get("judgment_summary_zh", f"字段反思：{field_key}"),
                    "reason": reflect_result.get("why_this_surface_zh", ""),
                    "original_reasoning": original_reasoning,
                    "proposed_reasoning_direction": reflect_result.get("patch_intent", {}).get("change_summary_zh", ""),
                    "judgment_summary_zh": reflect_result.get("judgment_summary_zh", ""),
                    "key_evidence_zh": reflect_result.get("key_evidence_zh", []),
                    "why_this_surface_zh": reflect_result.get("why_this_surface_zh", ""),
                    "root_cause_family": reflect_result.get("root_cause_family", ""),
                    "confidence": reflect_result.get("confidence", 0),
                    "improvement_type": improvement_type,
                    "patch_preview": patch_preview,
                    "approval_required": True,
                }
            )

        # P3: 补丁效果自动验证（复用前面已算的 pre_reflect_patch_effect）
        if pre_reflect_patch_effect:
            cycle_entry["patch_effect"] = pre_reflect_patch_effect

        # P3: 记录补丁基线（当生成了 patch_preview 时）
        new_patch_grade = previous_state.get("last_patch_grade")
        new_patch_score = previous_state.get("last_patch_score")
        if cycle_status == "new_rule_candidate":
            new_patch_grade = grade
            new_patch_score = float(comparison_result.get("score") or 0.0)

        per_field_state[field_key] = {
            "cycle_count": cycle_index,
            "last_grade": grade,
            "last_issue_score": current_score,
            "last_status": cycle_status,
            "last_signal_key": signal_key,
            "seen_signal_keys": seen_signals,
            "no_new_signal_streak": no_new_signal_streak,
            "cooldown_remaining": cooldown_remaining,
            "throttled_round_count": int(previous_state.get("throttled_round_count") or 0),
            "issue_score_history": issue_score_history,
            "score_trend": score_trend,
            "last_patch_grade": new_patch_grade,
            "last_patch_score": new_patch_score,
            "reflect_fail_streak": reflect_fail_streak,
            "last_proposed_direction": (reflect_result or {}).get("patch_intent", {}).get("change_summary_zh", ""),
            "last_updated": datetime.now().isoformat(),
        }

    state_payload["updated_at"] = datetime.now().isoformat()
    state_payload["fields"] = per_field_state
    return cycles, insights, proposals, state_payload



# --- Old functions removed: _infer_failure_mode, _extract_overlooked_signals,
#     _propose_field_rule_patch, _infer_source_hints, _extract_structured_refs,
#     _tokenize_signal_text, _is_value_empty
# Diagnostics and signal extraction via services.reflection.engine.
# Proposals now exclusively via UpstreamReflectionAgent.reflect() (no engine_plan_patch).
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
        "exhaustion_cycle_threshold": _read_positive_int_env(
            "FIELD_LOOP_EXHAUSTION_CYCLE_THRESHOLD",
            default=FIELD_LOOP_EXHAUSTION_CYCLE_THRESHOLD_DEFAULT,
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


def _classify_score_trend(history: List[float]) -> str:
    """P2: 从 issue_score 时间序列分类趋势。"""
    if len(history) < 3:
        return "insufficient"

    recent = history[-3:]

    # stable: max-min < 5.0
    if max(recent) - min(recent) < 5.0:
        return "stable"

    # 方向变化检测
    diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    direction_changes = sum(1 for i in range(len(diffs) - 1) if diffs[i] * diffs[i + 1] < 0)

    # oscillating: 4+ 条中方向变化 >= 2 次
    if len(history) >= 4:
        full_diffs = [history[i + 1] - history[i] for i in range(len(history) - 1)]
        full_changes = sum(1 for i in range(len(full_diffs) - 1) if full_diffs[i] * full_diffs[i + 1] < 0)
        if full_changes >= 2:
            return "oscillating"

    # declining: 单调不增 or 线性回归斜率 < -2.0
    if all(d <= 0 for d in diffs):
        return "declining"

    # rising: 单调不减 or 斜率 > 2.0
    if all(d >= 0 for d in diffs):
        return "rising"

    # 线性回归斜率
    n = len(history)
    x_mean = (n - 1) / 2.0
    y_mean = sum(history) / n
    numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(history))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    slope = numerator / denominator if denominator > 0 else 0.0

    if slope < -2.0:
        return "declining"
    if slope > 2.0:
        return "rising"
    return "stable"


# ── Exhausted field detection (LLM signal-driven) ──


def _collect_field_cycle_history(
    project_root: str, user_name: str, field_key: str,
) -> List[Dict[str, Any]]:
    """Collect all historical cycle entries for a specific field from field_cycles archives."""
    cycles_dir = Path(project_root) / EVOLUTION_RELATIVE_DIR / "field_cycles" / user_name
    if not cycles_dir.exists():
        return []
    entries: List[Dict[str, Any]] = []
    for f in sorted(cycles_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, list):
            continue
        for c in data:
            if c.get("field_key") == field_key:
                entries.append(c)
    return entries


def _is_field_exhausted(
    *,
    field_key: str = "",
    field_state: Dict[str, Any],
    cycle_history: List[Dict[str, Any]],
    policy: Dict[str, int],
    token_summary: Dict[str, Dict[str, Any]] | None = None,
) -> str | None:
    """Check if a field has exhausted all improvement avenues based on LLM signals.

    Returns the exhaustion_reason string if exhausted, None otherwise.
    """
    cycle_count = int(field_state.get("cycle_count") or 0)
    last_grade = str(field_state.get("last_grade") or "")
    threshold = int(policy.get("exhaustion_cycle_threshold") or FIELD_LOOP_EXHAUSTION_CYCLE_THRESHOLD_DEFAULT)

    # 基础门槛：轮次不够不算耗尽
    if cycle_count < threshold:
        return None

    # 评级已达标或 GT 缺失不算耗尽
    if last_grade in _RESOLVED_GRADES or last_grade == "missing_gt":
        return None

    # 需要有历史 cycle 数据才能做 LLM 信号判定
    if not cycle_history:
        return None

    # 从历史 cycle 中提取 LLM 信号
    root_causes = [str(c.get("root_cause_family") or "") for c in cycle_history if c.get("root_cause_family")]
    reflect_statuses = [str(c.get("reflect_status") or "") for c in cycle_history if c.get("reflect_status")]
    confidences = [
        _safe_float(c.get("reflect_confidence"), default=-1.0)
        for c in cycle_history
        if c.get("reflect_confidence") is not None
    ]
    confidences = [c for c in confidences if c >= 0.0]

    total_cycles_with_rc = len(root_causes)

    # LLM 信号条件（四选一）
    # 1. 反复 watch_only：LLM 多次判定"无法归因"
    watch_only_count = sum(1 for rc in root_causes if rc == "watch_only")
    if total_cycles_with_rc >= 2 and watch_only_count / total_cycles_with_rc >= 0.5:
        return "watch_only_dominant"

    # 2. 反复 needs_review：LLM 多次判定"不适合自动改策略"
    needs_review_count = sum(1 for s in reflect_statuses if s == "needs_review")
    if needs_review_count >= 2:
        return "needs_review_repeated"

    # 3. 低 confidence 累积：LLM 反复对自己的判断没信心
    if len(confidences) >= 2 and sum(confidences) / len(confidences) < 0.4:
        return "low_confidence_persistent"

    # 4. Token ROI 衰减：token 消耗高于平均但 score 未改善
    if token_summary and field_key:
        token_info = token_summary.get(field_key, {})
        field_tokens = token_info.get("total_in", 0) + token_info.get("total_out", 0)
        all_totals = [v["total_in"] + v["total_out"] for v in token_summary.values()]
        avg_tokens = sum(all_totals) / len(all_totals) if all_totals else 0
        score_trend = str(field_state.get("score_trend", ""))
        if (avg_tokens > 0
                and field_tokens > avg_tokens * 1.5
                and score_trend in ("stable", "rising")
                and cycle_count >= 4):
            return "token_roi_exhausted"

    return None


def _persist_exhausted_candidates(
    *,
    project_root: str,
    user_name: str,
    field_state: Dict[str, Any],
    policy: Dict[str, int],
    token_summary: Dict[str, Dict[str, Any]] | None = None,
) -> List[Dict[str, Any]]:
    """Detect and persist fields that have exhausted all improvement avenues.

    Checks ALL fields (not just current focus), writes to exhausted_fields/{user}.json.
    Returns the list of exhausted candidates.
    """
    fields = field_state.get("fields") or {}
    candidates: List[Dict[str, Any]] = []

    for fk, fs in fields.items():
        # 快速跳过：轮次不够或已达标
        cycle_count = int(fs.get("cycle_count") or 0)
        if cycle_count < int(policy.get("exhaustion_cycle_threshold") or FIELD_LOOP_EXHAUSTION_CYCLE_THRESHOLD_DEFAULT):
            continue
        last_grade = str(fs.get("last_grade") or "")
        if last_grade in _RESOLVED_GRADES or last_grade == "missing_gt":
            continue

        # 加载该字段的历史 cycle 数据
        cycle_history = _collect_field_cycle_history(project_root, user_name, fk)

        reason = _is_field_exhausted(field_key=fk, field_state=fs, cycle_history=cycle_history, policy=policy, token_summary=token_summary)
        if reason is None:
            continue

        # 从历史 cycle 中提取分类信息
        root_causes = [str(c.get("root_cause_family") or "") for c in cycle_history if c.get("root_cause_family")]
        confidences = [
            _safe_float(c.get("reflect_confidence"), default=-1.0)
            for c in cycle_history
            if c.get("reflect_confidence") is not None
        ]
        confidences = [c for c in confidences if c >= 0.0]
        statuses = [str(c.get("reflect_status") or "") for c in cycle_history if c.get("reflect_status")]

        # 最后一轮有效的 reflect 信息
        last_cycle = cycle_history[-1] if cycle_history else {}

        # 统计最高频 root_cause
        rc_counter: Dict[str, int] = {}
        for rc in root_causes:
            rc_counter[rc] = rc_counter.get(rc, 0) + 1
        dominant_root_cause = max(rc_counter, key=rc_counter.get) if rc_counter else ""

        # 统计最高频 fix_surface
        surfaces = [str(c.get("patch_preview", {}).get("fix_surface") or "") for c in cycle_history]
        surfaces = [s for s in surfaces if s]
        sf_counter: Dict[str, int] = {}
        for sf in surfaces:
            sf_counter[sf] = sf_counter.get(sf, 0) + 1
        dominant_fix_surface = max(sf_counter, key=sf_counter.get) if sf_counter else ""

        candidates.append({
            "field_key": fk,
            "user_name": user_name,
            "cycle_count": cycle_count,
            "last_grade": last_grade,
            # 分类信息（来自 Reflect Agent 累积判断）
            "exhaustion_reason": reason,
            "dominant_root_cause": dominant_root_cause,
            "dominant_fix_surface": dominant_fix_surface,
            "last_judgment_zh": str(last_cycle.get("judgment_summary_zh") or ""),
            "key_evidence_zh": list(last_cycle.get("key_evidence_zh") or []),
            "why_stuck_zh": str(last_cycle.get("why_this_surface_zh") or ""),
            # 轨迹摘要
            "confidence_history": confidences,
            "root_cause_history": root_causes,
            "status_history": statuses,
            "score_trend": str(fs.get("score_trend") or ""),
            "exhausted_at": datetime.now().isoformat(),
        })

    # 持久化（全量覆写）
    output_dir = Path(project_root) / EVOLUTION_RELATIVE_DIR / "exhausted_fields"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{user_name}.json"
    output_path.write_text(
        json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    if candidates:
        print(f"  [exhausted] {user_name}: {len(candidates)} 个字段判定为耗尽 → {output_path}")

    return candidates
