from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from utils import save_json
from utils.output_artifacts import serialize_event, serialize_face_db, serialize_relationship

from .downstream_audit import (
    apply_downstream_profile_backflow,
    apply_downstream_protagonist_backflow,
    apply_downstream_relationship_backflow,
    inspect_profile_agent_runtime_health,
    run_downstream_profile_agent_audit,
)
from .evolution import build_memory_run_trace, persist_memory_run_trace
from .groups import detect_groups
from .person_screening import screen_people
from .precomputed_loader import load_precomputed_memory_state
from .primary_person import analyze_primary_person_with_reflection
from .profile_fields import build_profile_context, generate_structured_profile
from .profile_llm import OpenRouterProfileLLMProcessor
from .relationships import build_relationship_dossiers, infer_relationships_from_dossiers


DEFAULT_PROFILE_MODEL = "google/gemini-3.1-flash-lite-preview"


def run_precomputed_bundle_pipeline(
    *,
    bundle_dir: str | Path,
    output_dir: str | Path | None = None,
    llm_processor: Any | None = None,
    profile_llm_processor: Any | None = None,
    profile_openrouter_key: str | None = None,
    profile_model: str | None = None,
    user_name: str = "default",
) -> Dict[str, Any]:
    bundle_path = Path(bundle_dir)
    if not bundle_path.exists():
        raise FileNotFoundError(f"bundle 不存在: {bundle_path}")

    output_path = Path(output_dir) if output_dir else bundle_path / f"precomputed_pipeline_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path.mkdir(parents=True, exist_ok=True)

    reporter = _StageReporter(output_path)
    state = load_precomputed_memory_state(bundle_path)
    primary_person_id = (state.primary_decision or {}).get("primary_person_id")
    effective_llm_processor = llm_processor or _resolve_llm_processor(primary_person_id)
    effective_profile_llm = profile_llm_processor or _resolve_profile_llm_processor(
        primary_person_id=primary_person_id,
        profile_openrouter_key=profile_openrouter_key,
        profile_model=profile_model,
        fallback_llm_processor=effective_llm_processor,
    )

    reporter.emit(
        stage="load_precomputed_state",
        status="ok",
        summary=f"加载 bundle 完成，复用 {len(state.face_db)} 个人物 / {len(state.vlm_results)} 条 VLM / {len(state.events)} 个 LP1 事件。",
        counts=_counts_from_state(state),
        artifacts={"bundle_dir": str(bundle_path)},
    )

    state.screening = screen_people(state)
    screening_counts = Counter(item.memory_value for item in state.screening.values())
    reporter.emit(
        stage="screen_people",
        status="ok",
        summary=(
            "人物筛查完成："
            f"core={screening_counts.get('core', 0)} / "
            f"candidate={screening_counts.get('candidate', 0)} / "
            f"low_value={screening_counts.get('low_value', 0)} / "
            f"block={screening_counts.get('block', 0)}"
        ),
        counts={
            **_counts_from_state(state),
            "screened_people": len(state.screening),
            "core_people": screening_counts.get("core", 0),
            "candidate_people": screening_counts.get("candidate", 0),
            "blocked_people": screening_counts.get("block", 0),
        },
        artifacts={},
    )

    precomputed_primary = bool(state.primary_decision and state.primary_decision.get("primary_person_id"))
    if precomputed_primary:
        state.primary_reflection = state.primary_reflection or {
            "triggered": False,
            "questions": [],
            "issues": [],
            "action": "keep",
            "decision_source": "precomputed_face_primary_person",
            "primary_signal_trace": {
                "selected_person_id": primary_person_id,
                "selected_mode": (state.primary_decision or {}).get("mode"),
                "candidate_signals": [],
                "llm_decision": {},
            },
        }
        primary_summary = f"复用预计算主角 {primary_person_id}，本阶段不重跑主角判定。"
    else:
        decision, reflection = analyze_primary_person_with_reflection(
            state=state,
            fallback_primary_person_id=primary_person_id,
            llm_processor=effective_llm_processor,
        )
        state.primary_decision = decision.to_dict()
        state.primary_reflection = reflection
        primary_person_id = (state.primary_decision or {}).get("primary_person_id")
        primary_summary = (
            f"主角判定完成：mode={decision.mode}，primary_person_id={decision.primary_person_id or 'None'}，confidence={decision.confidence:.2f}"
        )
    reporter.emit(
        stage="primary_decision",
        status="ok",
        summary=primary_summary,
        counts=_counts_from_state(state),
        artifacts={},
    )

    reporter.emit(
        stage="events_loaded",
        status="ok",
        summary=f"复用 {len(state.events)} 个预计算 LP1 事件，不重跑事件提取。",
        counts=_counts_from_state(state),
        artifacts={},
    )

    relationship_payload = _run_relationship_stage(state=state, llm_processor=effective_llm_processor)
    initial_relationship_archive = _archive_initial_relationship_outputs(output_path=output_path, state=state)
    reporter.emit(
        stage="relationships",
        status="ok",
        summary=(
            f"关系识别完成：dossier={relationship_payload['total_dossiers']}，"
            f"正式关系={len(state.relationships)}。"
        ),
        counts={
            **_counts_from_state(state),
            "total_dossiers": relationship_payload["total_dossiers"],
        },
        artifacts=initial_relationship_archive["artifacts"],
    )

    state.groups = detect_groups(state)
    reporter.emit(
        stage="groups",
        status="ok",
        summary=f"圈层识别完成：group_artifacts={len(state.groups)}。",
        counts=_counts_from_state(state),
        artifacts={},
    )

    profile_result = _run_profile_stage(
        state=state,
        llm_processor=effective_profile_llm,
    )
    pipeline_result = _compose_pipeline_result(state, profile_result)
    pre_audit_archive = _archive_pre_audit_outputs(output_path=output_path, pipeline_result=pipeline_result)
    reporter.emit(
        stage="profile_lp3",
        status="ok",
        summary=(
            f"LP3 画像完成：non_null_fields={_count_non_null_fields(profile_result.get('structured', {}))}，"
            f"field_decisions={len(profile_result.get('field_decisions', []))}，"
            f"profile_model={_describe_profile_model(profile_openrouter_key, profile_model, effective_profile_llm)}。"
        ),
        counts={
            **_counts_from_state(state),
            "non_null_fields": _count_non_null_fields(profile_result.get("structured", {})),
            "field_decisions": len(profile_result.get("field_decisions", [])),
        },
        artifacts={
            "structured_profile_path": str(output_path / "structured_profile.json"),
            **pre_audit_archive["artifacts"],
        },
    )

    audit_album_id = f"{bundle_path.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    downstream_audit_report = _run_downstream_audit_with_fallback(
        album_id=audit_album_id,
        primary_decision=pipeline_result.get("internal_artifacts", {}).get("primary_decision"),
        relationships=pipeline_result.get("relationships", []),
        structured_profile=pipeline_result.get("structured"),
        profile_fact_decisions=pipeline_result.get("internal_artifacts", {}).get("profile_fact_decisions", []),
    )
    audit_rounds = [_audit_round_snapshot("initial", downstream_audit_report)]
    reporter.emit(
        stage="downstream_audit",
        status="ok",
        summary=(
            f"下游审计完成：audited={downstream_audit_report.get('summary', {}).get('total_audited_tags', 0)}，"
            f"rejected={downstream_audit_report.get('summary', {}).get('rejected_count', 0)}，"
            f"not_audited={downstream_audit_report.get('summary', {}).get('not_audited_count', 0)}。"
        ),
        counts={
            **_counts_from_state(state),
            "audited_tags": downstream_audit_report.get("summary", {}).get("total_audited_tags", 0),
            "rejected_tags": downstream_audit_report.get("summary", {}).get("rejected_count", 0),
            "not_audited_tags": downstream_audit_report.get("summary", {}).get("not_audited_count", 0),
        },
        artifacts={"downstream_audit_report_path": str(output_path / "downstream_audit_report.json")},
    )

    updated_primary_decision, protagonist_changed = apply_downstream_protagonist_backflow(
        pipeline_result.get("internal_artifacts", {}).get("primary_decision"),
        downstream_audit_report,
    )
    if protagonist_changed:
        state.primary_decision = updated_primary_decision
        if updated_primary_decision.get("mode") == "photographer_mode":
            state.relationships = []
            state.relationship_dossiers = []
            state.groups = []
            profile_result = _run_profile_stage(
                state=state,
                llm_processor=effective_profile_llm,
            )
        else:
            _run_relationship_stage(state=state, llm_processor=effective_llm_processor)
            state.groups = detect_groups(state)
            profile_result = _run_profile_stage(
                state=state,
                llm_processor=effective_profile_llm,
            )
        pipeline_result = _compose_pipeline_result(state, profile_result)
        downstream_audit_report = _run_downstream_audit_with_fallback(
            album_id=audit_album_id,
            primary_decision=pipeline_result.get("internal_artifacts", {}).get("primary_decision"),
            relationships=pipeline_result.get("relationships", []),
            structured_profile=pipeline_result.get("structured"),
            profile_fact_decisions=pipeline_result.get("internal_artifacts", {}).get("profile_fact_decisions", []),
        )
        audit_rounds.append(_audit_round_snapshot("after_protagonist_rerun", downstream_audit_report))
        reporter.emit(
            stage="after_protagonist_rerun",
            status="ok",
            summary=(
                f"主角回流后已重跑受影响阶段：mode={(state.primary_decision or {}).get('mode')}，"
                f"relationships={len(state.relationships)}，groups={len(state.groups)}。"
            ),
            counts={
                **_counts_from_state(state),
                "audited_tags": downstream_audit_report.get("summary", {}).get("total_audited_tags", 0),
                "rejected_tags": downstream_audit_report.get("summary", {}).get("rejected_count", 0),
            },
            artifacts={},
        )

    updated_relationships, updated_dossiers, relationship_changed = apply_downstream_relationship_backflow(
        pipeline_result.get("relationships", []),
        state.relationship_dossiers,
        downstream_audit_report,
    )
    if relationship_changed:
        state.relationships = updated_relationships
        state.relationship_dossiers = updated_dossiers
        state.groups = detect_groups(state)
        profile_result = _run_profile_stage(
            state=state,
            llm_processor=effective_profile_llm,
        )
        pipeline_result = _compose_pipeline_result(state, profile_result)
        downstream_audit_report = _run_downstream_audit_with_fallback(
            album_id=audit_album_id,
            primary_decision=pipeline_result.get("internal_artifacts", {}).get("primary_decision"),
            relationships=pipeline_result.get("relationships", []),
            structured_profile=pipeline_result.get("structured"),
            profile_fact_decisions=pipeline_result.get("internal_artifacts", {}).get("profile_fact_decisions", []),
        )
        audit_rounds.append(_audit_round_snapshot("after_relationship_rerun", downstream_audit_report))
        reporter.emit(
            stage="after_relationship_rerun",
            status="ok",
            summary=(
                f"关系回流后已重跑受影响阶段：relationships={len(state.relationships)}，groups={len(state.groups)}。"
            ),
            counts={
                **_counts_from_state(state),
                "audited_tags": downstream_audit_report.get("summary", {}).get("total_audited_tags", 0),
                "rejected_tags": downstream_audit_report.get("summary", {}).get("rejected_count", 0),
            },
            artifacts={},
        )

    updated_structured_profile, updated_profile_fact_decisions = apply_downstream_profile_backflow(
        pipeline_result.get("structured"),
        downstream_audit_report,
        field_decisions=pipeline_result.get("internal_artifacts", {}).get("profile_fact_decisions", []),
    )
    pipeline_result["structured"] = updated_structured_profile
    pipeline_result.setdefault("internal_artifacts", {})["profile_fact_decisions"] = updated_profile_fact_decisions
    downstream_audit_report["feedback_loop"] = {
        "protagonist_rerun_applied": protagonist_changed,
        "relationship_rerun_applied": relationship_changed,
        "audit_rounds": audit_rounds,
    }

    events = pipeline_result.get("events", [])
    relationships = pipeline_result.get("relationships", [])
    final_primary_person_id = (pipeline_result.get("internal_artifacts", {}).get("primary_decision") or {}).get("primary_person_id")

    save_json([serialize_event(event) for event in events], str(output_path / "events.json"))
    save_json([serialize_relationship(rel) for rel in relationships], str(output_path / "relationships.json"))
    save_json(pipeline_result.get("structured", {}), str(output_path / "structured_profile.json"))
    save_json(pipeline_result.get("consistency", {}), str(output_path / "consistency.json"))
    save_json(pipeline_result.get("internal_artifacts", {}), str(output_path / "internal_artifacts.json"))
    save_json(
        pipeline_result.get("internal_artifacts", {}).get("profile_llm_batch_debug", []),
        str(output_path / "profile_llm_batch_debug.json"),
    )
    save_json(downstream_audit_report, str(output_path / "downstream_audit_report.json"))
    save_json(_build_state_snapshot(state), str(output_path / "normalized_state_snapshot.json"))
    save_json(_build_mapping_debug(bundle_path, state), str(output_path / "mapping_debug.json"))
    save_json(serialize_face_db(state.face_db), str(output_path / "face_db.json"))
    issue_log = _build_test_issue_log(
        initial_relationship_summary=initial_relationship_archive["summary"],
        final_relationships=relationships,
        downstream_audit_report=downstream_audit_report,
        profile_llm_batch_debug=pipeline_result.get("internal_artifacts", {}).get("profile_llm_batch_debug", []),
    )
    save_json(issue_log, str(output_path / "test_issue_log.json"))

    run_trace_result = {}
    try:
        run_trace_payload = build_memory_run_trace(
            run_type="precomputed_bundle",
            user_name=user_name,
            stage_reports=list(reporter.records),
            downstream_audit_report=downstream_audit_report,
            profile_llm_batch_debug=pipeline_result.get("internal_artifacts", {}).get("profile_llm_batch_debug", []),
            test_issue_log=issue_log,
            artifacts={
                "structured_profile_path": str(output_path / "structured_profile.json"),
                "profile_fact_decisions_path": str(output_path / "profile_fact_decisions.json"),
                "downstream_audit_report_path": str(output_path / "downstream_audit_report.json"),
                "stage_reports_path": str(reporter.path),
                "test_issue_log_path": str(output_path / "test_issue_log.json"),
            },
        )
        run_trace_result = persist_memory_run_trace(
            project_root=str(Path(__file__).resolve().parents[2]),
            output_dir=str(output_path),
            trace_payload=run_trace_payload,
        )
    except Exception as exc:
        print(f"[bundle pipeline][warn] memory run trace 写入失败: {exc}")

    return {
        "output_dir": str(output_path),
        "events_path": str(output_path / "events.json"),
        "relationships_path": str(output_path / "relationships.json"),
        "structured_profile_path": str(output_path / "structured_profile.json"),
        "downstream_audit_report_path": str(output_path / "downstream_audit_report.json"),
        "profile_llm_batch_debug_path": str(output_path / "profile_llm_batch_debug.json"),
        "initial_relationships_before_backflow_path": initial_relationship_archive["artifacts"]["initial_relationships_path"],
        "initial_relationship_dossiers_before_backflow_path": initial_relationship_archive["artifacts"]["initial_relationship_dossiers_path"],
        "initial_relationships_before_backflow_summary_path": initial_relationship_archive["artifacts"]["initial_relationship_summary_path"],
        "pre_audit_snapshot_dir": pre_audit_archive["artifacts"]["pre_audit_snapshot_dir"],
        "pre_audit_summary_path": pre_audit_archive["artifacts"]["pre_audit_summary_path"],
        "pre_audit_primary_decision_path": pre_audit_archive["artifacts"]["pre_audit_primary_decision_path"],
        "pre_audit_relationships_path": pre_audit_archive["artifacts"]["pre_audit_relationships_path"],
        "pre_audit_structured_profile_path": pre_audit_archive["artifacts"]["pre_audit_structured_profile_path"],
        "stage_reports_path": str(reporter.path),
        "test_issue_log_path": str(output_path / "test_issue_log.json"),
        "final_primary_person_id": final_primary_person_id,
        "total_events": len(events),
        "total_relationships": len(relationships),
        "run_trace_path": run_trace_result.get("trace_json_path", str(output_path / "memory_pipeline_run_trace.json")),
        "run_trace_ledger_path": run_trace_result.get("trace_ledger_path", ""),
    }


def _run_relationship_stage(*, state: Any, llm_processor: Any | None) -> Dict[str, Any]:
    dossiers = build_relationship_dossiers(state=state, llm_processor=llm_processor)
    relationships, dossiers = infer_relationships_from_dossiers(
        state=state,
        llm_processor=llm_processor,
        dossiers=dossiers,
    )
    state.relationship_dossiers = dossiers
    state.relationships = relationships
    return {
        "total_dossiers": len(dossiers),
        "total_relationships": len(relationships),
    }


def _run_profile_stage(*, state: Any, llm_processor: Any | None) -> Dict[str, Any]:
    state.profile_context = build_profile_context(state)
    return generate_structured_profile(state, llm_processor=llm_processor)


def _compose_pipeline_result(state: Any, profile_result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "events": list(state.events or []),
        "relationships": list(state.relationships or []),
        "structured": profile_result.get("structured", {}),
        "report": "",
        "debug": {
            "field_decision_count": len(profile_result.get("field_decisions", [])),
            "report_reasoning": {},
        },
        "consistency": profile_result.get("consistency", {}),
        "internal_artifacts": {
            "screening": {person_id: screening.to_dict() for person_id, screening in (state.screening or {}).items()},
            "primary_decision": dict(state.primary_decision or {}),
            "primary_reflection": dict(state.primary_reflection or {}),
            "relationship_dossiers": [dossier.to_dict() for dossier in state.relationship_dossiers],
            "group_artifacts": [group.to_dict() for group in state.groups],
            "profile_fact_decisions": list(profile_result.get("field_decisions", [])),
            "profile_llm_batch_debug": list(profile_result.get("llm_batch_debug", [])),
        },
    }


def _resolve_llm_processor(primary_person_id: str | None) -> Any | None:
    try:
        from services.llm_processor import LLMProcessor

        return LLMProcessor(primary_person_id=primary_person_id)
    except Exception:
        return None


def _resolve_profile_llm_processor(
    *,
    primary_person_id: str | None,
    profile_openrouter_key: str | None,
    profile_model: str | None,
    fallback_llm_processor: Any | None,
) -> Any | None:
    key = str(profile_openrouter_key or "").strip()
    if not key:
        return fallback_llm_processor
    return OpenRouterProfileLLMProcessor(
        api_key=key,
        base_url=None,
        model=(profile_model or DEFAULT_PROFILE_MODEL).strip() or DEFAULT_PROFILE_MODEL,
        primary_person_id=primary_person_id,
    )


def _counts_from_state(state: Any) -> Dict[str, int]:
    return {
        "face_db": len(state.face_db or {}),
        "vlm_results": len(state.vlm_results or []),
        "events": len(state.events or []),
        "relationships": len(state.relationships or []),
        "groups": len(state.groups or []),
    }


def _count_non_null_fields(payload: Any) -> int:
    if isinstance(payload, dict):
        if {"value", "confidence", "evidence", "reasoning"} <= set(payload.keys()):
            return 1 if payload.get("value") is not None else 0
        return sum(_count_non_null_fields(value) for value in payload.values())
    if isinstance(payload, list):
        return sum(_count_non_null_fields(item) for item in payload)
    return 0


def _describe_profile_model(profile_openrouter_key: str | None, profile_model: str | None, llm_processor: Any | None) -> str:
    if profile_openrouter_key:
        return (profile_model or DEFAULT_PROFILE_MODEL).strip() or DEFAULT_PROFILE_MODEL
    model = getattr(llm_processor, "model", None)
    return str(model or "default_profile_llm")


def _archive_initial_relationship_outputs(*, output_path: Path, state: Any) -> Dict[str, Any]:
    relationships_path = output_path / "initial_relationships_before_backflow.json"
    dossiers_path = output_path / "initial_relationship_dossiers_before_backflow.json"
    summary_path = output_path / "initial_relationships_before_backflow_summary.json"
    serialized_relationships = [serialize_relationship(rel) for rel in state.relationships or []]
    serialized_dossiers = [dossier.to_dict() for dossier in state.relationship_dossiers or []]
    summary = {
        "captured_stage": "relationships",
        "captured_at": datetime.now().isoformat(),
        "primary_person_id": (state.primary_decision or {}).get("primary_person_id"),
        "total_dossiers": len(serialized_dossiers),
        "total_relationships": len(serialized_relationships),
        "relationship_person_ids": [item.get("person_id") for item in serialized_relationships],
        "relationship_types": [item.get("relationship_type") for item in serialized_relationships],
        "note": "关系阶段初次判定后立刻归档，记录早于任何 downstream backflow rerun。",
    }
    save_json(serialized_relationships, str(relationships_path))
    save_json(serialized_dossiers, str(dossiers_path))
    save_json(summary, str(summary_path))
    return {
        "summary": summary,
        "artifacts": {
            "initial_relationships_path": str(relationships_path),
            "initial_relationship_dossiers_path": str(dossiers_path),
            "initial_relationship_summary_path": str(summary_path),
        },
    }


def _archive_pre_audit_outputs(*, output_path: Path, pipeline_result: Dict[str, Any]) -> Dict[str, Any]:
    snapshot_dir = output_path / "pre_audit_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    internal_artifacts = dict(pipeline_result.get("internal_artifacts", {}) or {})
    events = [serialize_event(event) for event in pipeline_result.get("events", []) or []]
    relationships = [serialize_relationship(relationship) for relationship in pipeline_result.get("relationships", []) or []]
    screening = dict(internal_artifacts.get("screening", {}) or {})
    primary_decision = dict(internal_artifacts.get("primary_decision", {}) or {})
    primary_reflection = dict(internal_artifacts.get("primary_reflection", {}) or {})
    relationship_dossiers = list(internal_artifacts.get("relationship_dossiers", []) or [])
    group_artifacts = list(internal_artifacts.get("group_artifacts", []) or [])
    profile_fact_decisions = list(internal_artifacts.get("profile_fact_decisions", []) or [])
    profile_llm_batch_debug = list(internal_artifacts.get("profile_llm_batch_debug", []) or [])
    structured_profile = dict(pipeline_result.get("structured", {}) or {})
    consistency = dict(pipeline_result.get("consistency", {}) or {})

    summary = {
        "captured_stage": "pre_downstream_audit",
        "captured_at": datetime.now().isoformat(),
        "primary_person_id": primary_decision.get("primary_person_id"),
        "event_count": len(events),
        "relationship_count": len(relationships),
        "relationship_person_ids": [item.get("person_id") for item in relationships],
        "total_dossiers": len(relationship_dossiers),
        "group_count": len(group_artifacts),
        "field_decisions": len(profile_fact_decisions),
        "non_null_fields": _count_non_null_fields(structured_profile),
        "note": "首轮 LP3 完成后、进入 downstream audit 前立刻归档，记录早于任何 downstream backflow rerun。",
    }

    save_json(summary, str(snapshot_dir / "summary.json"))
    save_json(events, str(snapshot_dir / "events.json"))
    save_json(screening, str(snapshot_dir / "screening.json"))
    save_json(primary_decision, str(snapshot_dir / "primary_decision.json"))
    save_json(primary_reflection, str(snapshot_dir / "primary_reflection.json"))
    save_json(relationships, str(snapshot_dir / "relationships.json"))
    save_json(relationship_dossiers, str(snapshot_dir / "relationship_dossiers.json"))
    save_json(group_artifacts, str(snapshot_dir / "group_artifacts.json"))
    save_json(structured_profile, str(snapshot_dir / "structured_profile.json"))
    save_json(consistency, str(snapshot_dir / "consistency.json"))
    save_json(profile_fact_decisions, str(snapshot_dir / "profile_fact_decisions.json"))
    save_json(profile_llm_batch_debug, str(snapshot_dir / "profile_llm_batch_debug.json"))

    return {
        "summary": summary,
        "artifacts": {
            "pre_audit_snapshot_dir": str(snapshot_dir),
            "pre_audit_summary_path": str(snapshot_dir / "summary.json"),
            "pre_audit_primary_decision_path": str(snapshot_dir / "primary_decision.json"),
            "pre_audit_relationships_path": str(snapshot_dir / "relationships.json"),
            "pre_audit_structured_profile_path": str(snapshot_dir / "structured_profile.json"),
        },
    }


def _run_downstream_audit_with_fallback(
    *,
    album_id: str,
    primary_decision: Dict | None,
    relationships: list | None,
    structured_profile: Dict | None,
    profile_fact_decisions: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    runtime_health = inspect_profile_agent_runtime_health()
    if runtime_health.get("status") != "ok":
        return _build_skipped_downstream_audit_report(
            album_id=album_id,
            primary_decision=primary_decision or {},
            relationships=relationships or [],
            structured_profile=structured_profile or {},
            runtime_health=runtime_health,
            error=RuntimeError(
                f"downstream_runtime_unhealthy:{runtime_health.get('error_code', 'runtime_unhealthy')}"
            ),
        )
    try:
        return run_downstream_profile_agent_audit(
            album_id=album_id,
            primary_decision=primary_decision,
            relationships=relationships or [],
            structured_profile=structured_profile,
            profile_fact_decisions=profile_fact_decisions or [],
            runtime_health=runtime_health,
        )
    except Exception as exc:
        return _build_skipped_downstream_audit_report(
            album_id=album_id,
            primary_decision=primary_decision or {},
            relationships=relationships or [],
            structured_profile=structured_profile or {},
            runtime_health=runtime_health,
            error=exc,
        )


def _build_skipped_downstream_audit_report(
    *,
    album_id: str,
    primary_decision: Dict[str, Any],
    relationships: list,
    structured_profile: Dict[str, Any],
    runtime_health: Dict[str, Any] | None = None,
    error: Exception,
) -> Dict[str, Any]:
    error_type = type(error).__name__
    error_message = str(error)
    protagonist_not_audited = [{"target_id": "primary_decision", "reason": "audit_runtime_failure"}] if primary_decision else []
    relationship_not_audited = []
    for relationship in relationships:
        if isinstance(relationship, dict):
            person_id = relationship.get("person_id")
        else:
            person_id = getattr(relationship, "person_id", None)
        relationship_not_audited.append(
            {
                "person_id": person_id,
                "reason": "audit_runtime_failure",
            }
        )
    total_not_audited = len(protagonist_not_audited) + len(relationship_not_audited)
    return {
        "metadata": {
            "downstream_engine": "profile_agent",
            "audit_mode": "selective_profile_domain_rules_facts_only",
            "audit_status": "skipped_init_failure",
            "audit_error_type": error_type,
            "audit_error_message": error_message,
        },
        "summary": {
            "total_audited_tags": 0,
            "challenged_count": 0,
            "accepted_count": 0,
            "downgraded_count": 0,
            "rejected_count": 0,
            "not_audited_count": total_not_audited,
        },
        "backflow": {
            "album_id": album_id,
            "storage_saved": False,
            "audit_failure": {
                "error_type": error_type,
                "error_message": error_message,
            },
            "protagonist": {
                "official_output_applied": False,
                "merged_output": {"agent_type": "protagonist", "tags": []},
                "actions": [],
            },
            "relationship": {
                "official_output_applied": False,
                "merged_output": {"agent_type": "relationship", "tags": []},
                "actions": [],
            },
            "profile": {
                "official_output_applied": False,
                "merged_output": {"agent_type": "profile", "tags": []},
                "field_actions": [],
            },
        },
        "protagonist": {
            "extractor_output": {},
            "critic_output": {"challenges": []},
            "judge_output": {"decisions": [], "hard_cases": []},
            "audit_flags": [],
            "not_audited": protagonist_not_audited,
        },
        "relationship": {
            "extractor_output": {},
            "critic_output": {"challenges": []},
            "judge_output": {"decisions": [], "hard_cases": []},
            "audit_flags": [],
            "not_audited": relationship_not_audited,
        },
        "profile": {
            "extractor_output": {},
            "critic_output": {"challenges": []},
            "judge_output": {"decisions": [], "hard_cases": []},
            "audit_flags": [],
            "not_audited": [],
        },
    }


def _audit_round_snapshot(stage: str, report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "stage": stage,
        "summary": dict(report.get("summary", {}) or {}),
        "backflow": dict(report.get("backflow", {}) or {}),
    }


def _build_state_snapshot(state: Any) -> Dict[str, Any]:
    return {
        "primary_decision": dict(state.primary_decision or {}),
        "face_db_count": len(state.face_db or {}),
        "vlm_result_count": len(state.vlm_results or []),
        "event_count": len(state.events or []),
        "relationship_count": len(state.relationships or []),
        "group_count": len(state.groups or []),
    }


def _build_mapping_debug(bundle_path: Path, state: Any) -> Dict[str, Any]:
    face_payload = {}
    face_file = bundle_path / "face" / "face_recognition_output.json"
    if face_file.exists():
        face_payload = json.loads(face_file.read_text(encoding="utf-8"))
    bundle_primary_person_id = face_payload.get("primary_person_id") or face_payload.get("face_recognition", {}).get("primary_person_id")
    event_trace_mapping = {
        event.event_id: dict((event.meta_info or {}).get("trace", {}) or {})
        for event in state.events or []
    }
    event_participant_mapping = {
        event.event_id: {
            "participants": list(getattr(event, "participants", []) or []),
            "raw_participants": list((event.meta_info or {}).get("raw_participants", []) or []),
        }
        for event in state.events or []
    }
    return {
        "bundle_primary_person_id": bundle_primary_person_id,
        "canonical_primary_person_id": (state.primary_decision or {}).get("primary_person_id"),
        "event_trace_mapping": event_trace_mapping,
        "event_participant_mapping": event_participant_mapping,
    }


def _build_test_issue_log(
    *,
    initial_relationship_summary: Dict[str, Any],
    final_relationships: List[Dict[str, Any]] | List[Any],
    downstream_audit_report: Dict[str, Any],
    profile_llm_batch_debug: List[Dict[str, Any]] | List[Any],
) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    metadata = dict(downstream_audit_report.get("metadata", {}) or {})
    feedback_loop = dict(downstream_audit_report.get("feedback_loop", {}) or {})
    initial_relationship_count = int(initial_relationship_summary.get("total_relationships") or 0)
    final_relationship_count = len(final_relationships or [])

    if metadata.get("audit_status") == "skipped_init_failure":
        issues.append(
            {
                "code": "DOWNSTREAM_AUDIT_SKIPPED",
                "severity": "medium",
                "summary": "downstream audit 初始化失败，本次 run 使用了 skipped fallback 报告。",
                "details": {
                    "error_type": metadata.get("audit_error_type"),
                    "error_message": metadata.get("audit_error_message"),
                },
            }
        )

    failed_profile_batches = [
        item
        for item in profile_llm_batch_debug or []
        if item.get("used_offline_fallback")
        or item.get("fallback_reason")
        or not item.get("raw_result_parseable", True)
    ]
    if failed_profile_batches:
        issues.append(
            {
                "code": "LP3_BATCH_FALLBACK_DETECTED",
                "severity": "high",
                "summary": (
                    f"LP3 共 {len(profile_llm_batch_debug or [])} 个 batch，其中 "
                    f"{len(failed_profile_batches)} 个发生 fallback 或解析失败。"
                ),
                "details": {
                    "fallback_reasons": Counter(
                        str(item.get("fallback_reason") or "unknown") for item in failed_profile_batches
                    ),
                    "http_status_codes": Counter(
                        str(item.get("http_status_code"))
                        for item in failed_profile_batches
                        if item.get("http_status_code") is not None
                    ),
                },
            }
        )

    protagonist_action = _extract_feedback_action(feedback_loop, "protagonist")
    if feedback_loop.get("protagonist_rerun_applied"):
        issues.append(
            {
                "code": "PROTAGONIST_BACKFLOW_RERUN",
                "severity": "high",
                "summary": "downstream protagonist audit 触发了回流重跑，并改写了上游主角判定。",
                "details": {
                    "value_before": protagonist_action.get("value_before"),
                    "value_after": protagonist_action.get("value_after"),
                    "judge_reason": protagonist_action.get("judge_reason"),
                },
            }
        )

    relationship_action = _extract_feedback_action(feedback_loop, "relationship")
    if feedback_loop.get("relationship_rerun_applied"):
        issues.append(
            {
                "code": "RELATIONSHIP_BACKFLOW_RERUN",
                "severity": "medium",
                "summary": "downstream relationship audit 触发了回流重跑，并改写了正式关系集合。",
                "details": {
                    "value_before": relationship_action.get("value_before"),
                    "value_after": relationship_action.get("value_after"),
                    "judge_reason": relationship_action.get("judge_reason"),
                },
            }
        )

    if initial_relationship_count != final_relationship_count:
        issues.append(
            {
                "code": "INITIAL_RELATIONSHIPS_CHANGED_AFTER_BACKFLOW",
                "severity": "medium",
                "summary": (
                    f"关系阶段初次判定产出 {initial_relationship_count} 条关系，"
                    f"但最终输出只保留了 {final_relationship_count} 条。"
                ),
                "details": {
                    "initial_relationship_count": initial_relationship_count,
                    "final_relationship_count": final_relationship_count,
                    "relationship_person_ids_before_backflow": list(initial_relationship_summary.get("relationship_person_ids", [])),
                },
            }
        )

    return {
        "summary": {
            "issue_count": len(issues),
            "high_risk_issue_count": sum(1 for issue in issues if issue.get("severity") == "high"),
        },
        "issues": issues,
    }


def _extract_feedback_action(feedback_loop: Dict[str, Any], actor_key: str) -> Dict[str, Any]:
    rounds = feedback_loop.get("audit_rounds") or []
    for round_payload in rounds:
        backflow = dict((round_payload or {}).get("backflow", {}) or {})
        actor_payload = dict(backflow.get(actor_key, {}) or {})
        actions = actor_payload.get("actions") or []
        if actions:
            return dict(actions[0] or {})
    return {}


class _StageReporter:
    def __init__(self, output_path: Path) -> None:
        self.path = output_path / "stage_reports.jsonl"
        self.path.write_text("", encoding="utf-8")
        self.records: List[Dict[str, Any]] = []

    def emit(
        self,
        *,
        stage: str,
        status: str,
        summary: str,
        counts: Dict[str, Any],
        artifacts: Dict[str, Any],
    ) -> None:
        payload = {
            "stage": stage,
            "status": status,
            "summary": summary,
            "counts": counts,
            "artifacts": artifacts,
            "timestamp": datetime.now().isoformat(),
        }
        self.records.append(payload)
        counts_preview = ", ".join(f"{key}={value}" for key, value in counts.items())
        print(f"[bundle pipeline][{stage}] {status} | {summary} | {counts_preview}")
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
