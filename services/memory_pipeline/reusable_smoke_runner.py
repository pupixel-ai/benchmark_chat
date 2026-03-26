from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from config import PROFILE_AGENT_ROOT
from utils import save_json
from utils.output_artifacts import (
    build_internal_artifact,
    build_relationships_artifact,
    serialize_event,
    serialize_face_db,
    serialize_relationship,
)

from .downstream_audit import (
    apply_downstream_profile_backflow,
    apply_downstream_protagonist_backflow,
    apply_downstream_relationship_backflow,
    run_downstream_profile_agent_audit,
)
from .orchestrator import (
    rerun_pipeline_from_primary_backflow,
    rerun_pipeline_from_relationship_backflow,
    run_memory_pipeline,
)
from .reusable_smoke_llm import resolve_reusable_smoke_llm_processor
from .reusable_smoke_loader import ReusableSmokeLoadResult, load_reusable_smoke_case


def run_reusable_smoke_pipeline(
    *,
    case_dir: str | Path,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    llm_processor: Any | None = None,
) -> Dict[str, Any]:
    load_result = load_reusable_smoke_case(case_dir)
    effective_run_id = str(run_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_path = Path(output_dir) if output_dir else load_result.case_path / f"temp_smoke_run_{effective_run_id}"
    output_path.mkdir(parents=True, exist_ok=True)

    normalized_state_snapshot = _build_state_snapshot(load_result)
    effective_llm_processor = llm_processor or _resolve_llm_processor(load_result.fallback_primary_person_id)
    if effective_llm_processor is None:
        raise ValueError("未解析到可用的 OpenRouter LLM processor；请检查 .env 或 open router key.md")

    pipeline_result = run_memory_pipeline(
        state=load_result.state,
        llm_processor=effective_llm_processor,
        fallback_primary_person_id=load_result.fallback_primary_person_id,
    )

    audit_album_id = f"reusable_smoke_{load_result.case_path.name}_{effective_run_id}"
    downstream_audit_report = _run_downstream_audit_with_fallback(
        album_id=audit_album_id,
        primary_decision=pipeline_result.get("internal_artifacts", {}).get("primary_decision"),
        relationships=pipeline_result.get("relationships", []),
        structured_profile=pipeline_result.get("structured"),
    )
    audit_rounds = [_audit_round_snapshot("initial", downstream_audit_report)]

    updated_primary_decision, protagonist_changed = apply_downstream_protagonist_backflow(
        pipeline_result.get("internal_artifacts", {}).get("primary_decision"),
        downstream_audit_report,
    )
    if protagonist_changed:
        load_result.state.primary_decision = updated_primary_decision
        if updated_primary_decision.get("mode") == "photographer_mode":
            load_result.state.relationships = []
            load_result.state.relationship_dossiers = []
            load_result.state.groups = []
            pipeline_result = rerun_pipeline_from_relationship_backflow(
                state=load_result.state,
                llm_processor=effective_llm_processor,
            )
            rerun_stage = "after_protagonist_photographer_rerun"
        else:
            pipeline_result = rerun_pipeline_from_primary_backflow(
                state=load_result.state,
                llm_processor=effective_llm_processor,
            )
            rerun_stage = "after_protagonist_rerun"
        downstream_audit_report = _run_downstream_audit_with_fallback(
            album_id=audit_album_id,
            primary_decision=pipeline_result.get("internal_artifacts", {}).get("primary_decision"),
            relationships=pipeline_result.get("relationships", []),
            structured_profile=pipeline_result.get("structured"),
        )
        audit_rounds.append(_audit_round_snapshot(rerun_stage, downstream_audit_report))

    updated_relationships, updated_dossiers, relationship_changed = apply_downstream_relationship_backflow(
        pipeline_result.get("relationships", []),
        load_result.state.relationship_dossiers,
        downstream_audit_report,
    )
    if relationship_changed:
        load_result.state.relationships = updated_relationships
        load_result.state.relationship_dossiers = updated_dossiers
        pipeline_result = rerun_pipeline_from_relationship_backflow(
            state=load_result.state,
            llm_processor=effective_llm_processor,
        )
        downstream_audit_report = _run_downstream_audit_with_fallback(
            album_id=audit_album_id,
            primary_decision=pipeline_result.get("internal_artifacts", {}).get("primary_decision"),
            relationships=pipeline_result.get("relationships", []),
            structured_profile=pipeline_result.get("structured"),
        )
        audit_rounds.append(_audit_round_snapshot("after_relationship_rerun", downstream_audit_report))

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
    comparison_summary, comparison_diff = _build_reference_comparison(
        load_result=load_result,
        new_relationships=relationships,
        new_structured_profile=pipeline_result.get("structured", {}),
    )

    save_json(normalized_state_snapshot, str(output_path / "normalized_state_snapshot.json"))
    save_json(load_result.mapping_debug, str(output_path / "mapping_debug.json"))
    save_json(
        build_relationships_artifact(
            relationships=relationships,
            primary_person_id=final_primary_person_id,
        ),
        str(output_path / "relationships.json"),
    )
    save_json(pipeline_result.get("structured", {}), str(output_path / "structured_profile.json"))
    save_json(
        build_internal_artifact(
            artifact_name="relationship_dossiers",
            payload=pipeline_result.get("internal_artifacts", {}).get("relationship_dossiers", []),
            primary_person_id=final_primary_person_id,
            total_dossiers=len(pipeline_result.get("internal_artifacts", {}).get("relationship_dossiers", [])),
        ),
        str(output_path / "relationship_dossiers.json"),
    )
    save_json(
        build_internal_artifact(
            artifact_name="group_artifacts",
            payload=pipeline_result.get("internal_artifacts", {}).get("group_artifacts", []),
            primary_person_id=final_primary_person_id,
            total_groups=len(pipeline_result.get("internal_artifacts", {}).get("group_artifacts", [])),
        ),
        str(output_path / "group_artifacts.json"),
    )
    save_json(
        build_internal_artifact(
            artifact_name="profile_fact_decisions",
            payload=pipeline_result.get("internal_artifacts", {}).get("profile_fact_decisions", []),
            primary_person_id=final_primary_person_id,
            total_fields=len(pipeline_result.get("internal_artifacts", {}).get("profile_fact_decisions", [])),
        ),
        str(output_path / "profile_fact_decisions.json"),
    )
    save_json(downstream_audit_report, str(output_path / "downstream_audit_report.json"))
    save_json(comparison_summary, str(output_path / "comparison_summary.json"))
    save_json(comparison_diff, str(output_path / "comparison_diff.json"))

    return {
        "output_dir": str(output_path),
        "final_primary_person_id": final_primary_person_id,
        "relationships_path": str(output_path / "relationships.json"),
        "structured_profile_path": str(output_path / "structured_profile.json"),
        "downstream_audit_report_path": str(output_path / "downstream_audit_report.json"),
        "comparison_summary_path": str(output_path / "comparison_summary.json"),
        "comparison_diff_path": str(output_path / "comparison_diff.json"),
        "total_events": len(events),
        "total_relationships": len(relationships),
    }


def _resolve_llm_processor(primary_person_id: str | None) -> Any | None:
    return resolve_reusable_smoke_llm_processor(primary_person_id=primary_person_id)


def _build_state_snapshot(load_result: ReusableSmokeLoadResult) -> Dict[str, Any]:
    state = load_result.state
    return {
        "case_dir": str(load_result.case_path),
        "fallback_primary_person_id": load_result.fallback_primary_person_id,
        "input_contract": {
            "allowed_runtime_inputs": ["face_recognition_output", "vlm_cache", "events.events"],
            "blocked_runtime_inputs": ["events.relationships", "events.face_db", "profile_structured", "face_recognition_state"],
        },
        "counts": {
            "face_db_count": len(state.face_db or {}),
            "vlm_result_count": len(state.vlm_results or []),
            "event_count": len(state.events or []),
            "input_relationship_count": len(state.relationships or []),
        },
        "face_db": serialize_face_db(state.face_db),
        "vlm_results": list(state.vlm_results or []),
        "events": [serialize_event(event) for event in state.events or []],
    }


def _build_reference_comparison(
    *,
    load_result: ReusableSmokeLoadResult,
    new_relationships: Iterable[Any],
    new_structured_profile: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    old_relationships = list(load_result.reference_relationships or [])
    new_relationship_payloads = [serialize_relationship(relationship) for relationship in new_relationships]
    old_profile = _load_optional_json(load_result.reference_profile_path)

    old_profile_fields = _flatten_profile_fields(old_profile)
    new_profile_fields = _flatten_profile_fields(new_structured_profile)

    summary = {
        "reference_only_not_fed": True,
        "sources": {
            "legacy_relationships": f"{load_result.runtime_input_paths['events']}#relationships",
            "legacy_profile_structured": str(load_result.reference_profile_path) if load_result.reference_profile_path else None,
        },
        "old_relationship_count": len(old_relationships),
        "new_relationship_count": len(new_relationship_payloads),
        "old_profile_non_null_field_count": _count_non_null_profile_fields(old_profile_fields),
        "new_profile_non_null_field_count": _count_non_null_profile_fields(new_profile_fields),
    }
    diff = {
        "metadata": {
            "reference_only_not_fed": True,
            "case_dir": str(load_result.case_path),
        },
        "relationships": _diff_relationships(old_relationships, new_relationship_payloads),
        "profile_fields": _diff_profile_fields(old_profile_fields, new_profile_fields),
    }
    return summary, diff


def _load_optional_json(path: Path | None) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _flatten_profile_fields(payload: Any, prefix: str = "") -> Dict[str, Dict[str, Any]]:
    flat: Dict[str, Dict[str, Any]] = {}
    if not isinstance(payload, dict):
        return flat

    if "value" in payload and "confidence" in payload:
        flat[prefix] = {
            "value": payload.get("value"),
            "confidence": payload.get("confidence"),
        }
        return flat

    for key, value in payload.items():
        next_prefix = f"{prefix}.{key}" if prefix else key
        flat.update(_flatten_profile_fields(value, next_prefix))
    return flat


def _count_non_null_profile_fields(flat_profile: Dict[str, Dict[str, Any]]) -> int:
    return sum(1 for item in flat_profile.values() if item.get("value") not in (None, "", []))


def _diff_relationships(
    old_relationships: List[Dict[str, Any]],
    new_relationships: List[Dict[str, Any]],
) -> Dict[str, Any]:
    old_by_person = {str(item.get("person_id") or ""): _normalize_reference_relationship(item) for item in old_relationships}
    new_by_person = {str(item.get("person_id") or ""): _normalize_reference_relationship(item) for item in new_relationships}

    old_only = [old_by_person[person_id] for person_id in sorted(set(old_by_person) - set(new_by_person))]
    new_only = [new_by_person[person_id] for person_id in sorted(set(new_by_person) - set(old_by_person))]
    changed: List[Dict[str, Any]] = []
    unchanged: List[Dict[str, Any]] = []

    for person_id in sorted(set(old_by_person) & set(new_by_person)):
        old_item = old_by_person[person_id]
        new_item = new_by_person[person_id]
        comparable_old = {key: old_item.get(key) for key in ("person_id", "relationship_type", "status")}
        comparable_new = {key: new_item.get(key) for key in ("person_id", "relationship_type", "status")}
        if comparable_old == comparable_new:
            unchanged.append(comparable_new)
            continue
        changed.append({"person_id": person_id, "old": old_item, "new": new_item})

    return {
        "old_only": old_only,
        "new_only": new_only,
        "changed": changed,
        "unchanged": unchanged,
    }


def _normalize_reference_relationship(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "person_id": item.get("person_id"),
        "relationship_type": item.get("relationship_type"),
        "status": item.get("status"),
        "confidence": item.get("confidence"),
        "reasoning": item.get("reasoning") or item.get("reason", ""),
    }


def _diff_profile_fields(
    old_fields: Dict[str, Dict[str, Any]],
    new_fields: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    old_only = []
    new_only = []
    changed = []
    unchanged = []

    for field_key in sorted(set(old_fields) - set(new_fields)):
        old_only.append({"field_key": field_key, "old": old_fields[field_key]})
    for field_key in sorted(set(new_fields) - set(old_fields)):
        new_only.append({"field_key": field_key, "new": new_fields[field_key]})
    for field_key in sorted(set(old_fields) & set(new_fields)):
        old_item = old_fields[field_key]
        new_item = new_fields[field_key]
        if _json_like_equal(old_item, new_item):
            unchanged.append({"field_key": field_key, "value": new_item})
            continue
        changed.append({"field_key": field_key, "old": old_item, "new": new_item})

    return {
        "old_only": old_only,
        "new_only": new_only,
        "changed": changed,
        "unchanged": unchanged,
    }


def _json_like_equal(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True, ensure_ascii=False) == json.dumps(right, sort_keys=True, ensure_ascii=False)


def _audit_round_snapshot(stage: str, report: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "stage": stage,
        "summary": dict(report.get("summary", {}) or {}),
        "backflow": dict(report.get("backflow", {}) or {}),
    }


def _run_downstream_audit_with_fallback(
    *,
    album_id: str,
    primary_decision: Dict[str, Any] | None,
    relationships: list | None,
    structured_profile: Dict[str, Any] | None,
) -> Dict[str, Any]:
    try:
        return run_downstream_profile_agent_audit(
            album_id=album_id,
            primary_decision=primary_decision,
            relationships=relationships or [],
            structured_profile=structured_profile,
        )
    except Exception as exc:
        return _build_skipped_downstream_audit_report(
            album_id=album_id,
            primary_decision=primary_decision or {},
            relationships=relationships or [],
            structured_profile=structured_profile or {},
            error=exc,
        )


def _build_skipped_downstream_audit_report(
    *,
    album_id: str,
    primary_decision: Dict[str, Any],
    relationships: list,
    structured_profile: Dict[str, Any],
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
            "profile_agent_root": PROFILE_AGENT_ROOT,
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
