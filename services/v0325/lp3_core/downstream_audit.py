from __future__ import annotations

from dataclasses import asdict
from copy import deepcopy
import importlib
import importlib.util
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple

from dataclasses import replace

from config import PROFILE_AGENT_ROOT
from .relationships import (
    RELATIONSHIP_TYPE_SPECS,
    _determine_group_eligibility,
    _determine_retention,
)
from .profile_agent_adapter import (
    PROFILE_DIMENSION_FIELD_PATHS,
    RELATIONSHIP_DIMENSION_MAP,
    build_profile_agent_extractor_outputs,
)
from .types import RelationshipDossier, RelationshipRecord


def run_downstream_profile_agent_audit(
    *,
    primary_decision: Dict[str, Any] | None,
    relationships: Iterable[RelationshipRecord | Dict[str, Any]],
    structured_profile: Dict[str, Any] | None,
    album_id: str = "memory_pipeline_downstream_audit",
) -> Dict[str, Any]:
    relationship_list = list(relationships)
    adapter_outputs = build_profile_agent_extractor_outputs(
        primary_decision=primary_decision,
        relationships=relationship_list,
        structured_profile=structured_profile or {},
    )
    audit_outputs, merged_outputs, storage_saved = _run_profile_agent_judges(
        adapter_outputs,
        album_id=album_id,
    )
    return build_downstream_audit_report(
        album_id=album_id,
        primary_decision=primary_decision or {},
        relationships=relationship_list,
        structured_profile=structured_profile or {},
        adapter_outputs=adapter_outputs,
        audit_outputs=audit_outputs,
        merged_outputs=merged_outputs,
        storage_saved=storage_saved,
    )


def build_downstream_audit_report(
    *,
    album_id: str,
    primary_decision: Dict[str, Any],
    relationships: Iterable[RelationshipRecord | Dict[str, Any]],
    structured_profile: Dict[str, Any],
    adapter_outputs: Dict[str, Dict[str, Any]],
    audit_outputs: Dict[str, Dict[str, Any]],
    merged_outputs: Dict[str, Dict[str, Any]],
    storage_saved: bool,
) -> Dict[str, Any]:
    relationship_list = [_coerce_relationship_dict(item) for item in relationships]

    protagonist_flags, protagonist_not_audited = _build_protagonist_flags(
        primary_decision=primary_decision,
        extractor_output=adapter_outputs.get("protagonist", {}),
        audit_output=audit_outputs.get("protagonist", {}),
    )
    relationship_flags, relationship_not_audited = _build_relationship_flags(
        relationships=relationship_list,
        extractor_output=adapter_outputs.get("relationship", {}),
        audit_output=audit_outputs.get("relationship", {}),
    )
    profile_flags, profile_not_audited = _build_profile_flags(
        structured_profile=structured_profile,
        extractor_output=adapter_outputs.get("profile", {}),
        audit_output=audit_outputs.get("profile", {}),
    )

    all_flags = protagonist_flags + relationship_flags + profile_flags
    summary = {
        "total_audited_tags": len(all_flags),
        "challenged_count": sum(1 for flag in all_flags if flag.get("was_challenged")),
        "accepted_count": sum(1 for flag in all_flags if flag.get("audit_status") == "accepted"),
        "downgraded_count": sum(1 for flag in all_flags if flag.get("audit_status") == "downgraded"),
        "rejected_count": sum(1 for flag in all_flags if flag.get("audit_status") == "rejected"),
        "not_audited_count": len(protagonist_not_audited) + len(relationship_not_audited) + len(profile_not_audited),
    }

    return {
        "metadata": {
            "downstream_engine": "profile_agent",
            "audit_mode": "selective_profile_domain_rules_facts_only",
            "profile_agent_root": PROFILE_AGENT_ROOT,
        },
        "summary": summary,
        "backflow": _build_backflow_payload(
            album_id=album_id,
            primary_decision=primary_decision,
            relationships=relationship_list,
            structured_profile=structured_profile,
            adapter_outputs=adapter_outputs,
            audit_outputs=audit_outputs,
            merged_outputs=merged_outputs,
            storage_saved=storage_saved,
        ),
        "protagonist": {
            "extractor_output": adapter_outputs.get("protagonist", {}),
            "critic_output": audit_outputs.get("protagonist", {}).get("critic_output", {"challenges": []}),
            "judge_output": audit_outputs.get("protagonist", {}).get("judge_output", {"decisions": [], "hard_cases": []}),
            "audit_flags": protagonist_flags,
            "not_audited": protagonist_not_audited,
        },
        "relationship": {
            "extractor_output": adapter_outputs.get("relationship", {}),
            "critic_output": audit_outputs.get("relationship", {}).get("critic_output", {"challenges": []}),
            "judge_output": audit_outputs.get("relationship", {}).get("judge_output", {"decisions": [], "hard_cases": []}),
            "audit_flags": relationship_flags,
            "not_audited": relationship_not_audited,
        },
        "profile": {
            "extractor_output": adapter_outputs.get("profile", {}),
            "critic_output": audit_outputs.get("profile", {}).get("critic_output", {"challenges": []}),
            "judge_output": audit_outputs.get("profile", {}).get("judge_output", {"decisions": [], "hard_cases": []}),
            "audit_flags": profile_flags,
            "not_audited": profile_not_audited,
        },
    }


def _run_profile_agent_judges(
    adapter_outputs: Dict[str, Dict[str, Any]],
    *,
    album_id: str = "memory_pipeline_downstream_audit",
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], bool]:
    Storage, CriticAgent, JudgeAgent = _load_profile_agent_runtime()
    storage = Storage()
    critic = CriticAgent(storage)
    judge = JudgeAgent(storage)

    audit_outputs: Dict[str, Dict[str, Any]] = {}
    merged_outputs: Dict[str, Dict[str, Any]] = {}
    for agent_type, extractor_output in adapter_outputs.items():
        if not extractor_output.get("tags"):
            audit_outputs[agent_type] = {
                "critic_output": {"challenges": []},
                "judge_output": {"decisions": [], "hard_cases": []},
            }
            merged_outputs[agent_type] = _empty_merged_output(agent_type)
            continue
        critic_result = critic.run(extractor_output)
        critic_output = asdict(critic_result)
        if critic_result.has_challenges():
            judge_result = judge.run(extractor_output, critic_output, album_id=album_id)
        else:
            judge_result = judge.run_no_challenges(extractor_output)
        audit_outputs[agent_type] = {
            "critic_output": critic_output,
            "judge_output": asdict(judge_result),
        }
        merged_outputs[agent_type] = _merge_judge_decisions(
            extractor_output,
            audit_outputs[agent_type]["judge_output"],
        )

    storage.save_profile(
        album_id,
        merged_outputs.get("protagonist", _empty_merged_output("protagonist")),
        merged_outputs.get("relationship", _empty_merged_output("relationship")),
        merged_outputs.get("profile", _empty_merged_output("profile")),
    )
    return audit_outputs, merged_outputs, True


def _load_profile_agent_runtime():
    profile_agent_root = Path(PROFILE_AGENT_ROOT)
    _ensure_profile_agent_package_loaded(profile_agent_root)
    Storage = importlib.import_module("profile_agent.storage").Storage
    CriticAgent = importlib.import_module("profile_agent.agents.critic").CriticAgent
    JudgeAgent = importlib.import_module("profile_agent.agents.judge").JudgeAgent

    return Storage, CriticAgent, JudgeAgent


def _ensure_profile_agent_package_loaded(profile_agent_root: Path) -> None:
    package_name = "profile_agent"
    init_path = profile_agent_root / "__init__.py"
    if not init_path.exists():
        raise FileNotFoundError(f"未找到下游 profile_agent 包入口: {init_path}")

    existing = sys.modules.get(package_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", "")
        if existing_file and Path(existing_file).resolve() == init_path.resolve():
            return
        for module_name in list(sys.modules):
            if module_name == package_name or module_name.startswith(f"{package_name}."):
                sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(profile_agent_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为下游 profile_agent 构建 importlib spec: {profile_agent_root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)


def _build_protagonist_flags(
    *,
    primary_decision: Dict[str, Any],
    extractor_output: Dict[str, Any],
    audit_output: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tags = list(extractor_output.get("tags", []))
    if not tags:
        reason = "photographer_mode_not_audited" if primary_decision.get("mode") == "photographer_mode" else "no_mapped_protagonist_tag"
        return [], [{"target_id": "primary_decision", "reason": reason}]

    challenges = list((audit_output.get("critic_output") or {}).get("challenges", []))
    decisions = list((audit_output.get("judge_output") or {}).get("decisions", []))
    flags: List[Dict[str, Any]] = []
    for tag in tags:
        dimension = str(tag.get("dimension") or "")
        matched_challenges = _matched_challenges(dimension, challenges)
        decision = _find_decision(dimension, decisions)
        flags.append(
            {
                "target_id": "primary_decision",
                "mapped_dimension": dimension,
                "audit_status": _decision_to_status(decision),
                "was_challenged": bool(matched_challenges),
                "critic_challenges": matched_challenges,
                "judge_verdict": decision.get("verdict") if decision else None,
                "judge_reason": decision.get("reason") if decision else "",
                "value": tag.get("value"),
            }
        )
    return flags, []


def _build_relationship_flags(
    *,
    relationships: List[Dict[str, Any]],
    extractor_output: Dict[str, Any],
    audit_output: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tags = list(extractor_output.get("tags", []))
    challenges = list((audit_output.get("critic_output") or {}).get("challenges", []))
    decisions = list((audit_output.get("judge_output") or {}).get("decisions", []))
    flags: List[Dict[str, Any]] = []
    audited_person_ids = set()
    for tag in tags:
        dimension = str(tag.get("dimension") or "")
        person_id = str(tag.get("value") or "")
        matched_challenges = _matched_challenges(dimension, challenges)
        decision = _find_decision(dimension, decisions)
        source_relationship = next((rel for rel in relationships if str(rel.get("person_id")) == person_id), {})
        audited_person_ids.add(person_id)
        flags.append(
            {
                "person_id": person_id,
                "source_relationship_type": source_relationship.get("relationship_type"),
                "mapped_dimension": dimension,
                "audit_status": _decision_to_status(decision),
                "was_challenged": bool(matched_challenges),
                "critic_challenges": matched_challenges,
                "judge_verdict": decision.get("verdict") if decision else None,
                "judge_reason": decision.get("reason") if decision else "",
            }
        )

    not_audited = []
    for relationship in relationships:
        person_id = str(relationship.get("person_id") or "")
        if person_id in audited_person_ids:
            continue
        not_audited.append(
            {
                "person_id": person_id,
                "relationship_type": relationship.get("relationship_type"),
                "reason": "unsupported_relationship_type_for_downstream_plugin",
            }
        )
    return flags, not_audited


def _build_profile_flags(
    *,
    structured_profile: Dict[str, Any],
    extractor_output: Dict[str, Any],
    audit_output: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tags = list(extractor_output.get("tags", []))
    challenges = list((audit_output.get("critic_output") or {}).get("challenges", []))
    decisions = list((audit_output.get("judge_output") or {}).get("decisions", []))
    audited_field_keys = set()
    flags: List[Dict[str, Any]] = []
    for tag in tags:
        dimension = str(tag.get("dimension") or "")
        field_key = _profile_field_key_for_dimension(dimension, structured_profile, tag.get("value"))
        if field_key:
            audited_field_keys.add(field_key)
        matched_challenges = _matched_challenges(dimension, challenges)
        decision = _find_decision(dimension, decisions)
        flags.append(
            {
                "field_key": field_key,
                "mapped_dimension": dimension,
                "audit_status": _decision_to_status(decision),
                "was_challenged": bool(matched_challenges),
                "critic_challenges": matched_challenges,
                "judge_verdict": decision.get("verdict") if decision else None,
                "judge_reason": decision.get("reason") if decision else "",
                "value": tag.get("value"),
            }
        )

    not_audited: List[Dict[str, Any]] = []
    for field_key, value in _iter_non_null_profile_fields(structured_profile):
        if field_key in audited_field_keys:
            continue
        not_audited.append(
            {
                "field_key": field_key,
                "value": value,
                "reason": "field_not_covered_by_downstream_plugin",
            }
        )
    return flags, not_audited


def _build_backflow_payload(
    *,
    album_id: str,
    primary_decision: Dict[str, Any],
    relationships: List[Dict[str, Any]],
    structured_profile: Dict[str, Any],
    adapter_outputs: Dict[str, Dict[str, Any]],
    audit_outputs: Dict[str, Dict[str, Any]],
    merged_outputs: Dict[str, Dict[str, Any]],
    storage_saved: bool,
) -> Dict[str, Any]:
    return {
        "album_id": album_id,
        "storage_saved": storage_saved,
        "protagonist": {
            "official_output_applied": False,
            "merged_output": merged_outputs.get("protagonist", _empty_merged_output("protagonist")),
            "actions": _build_protagonist_backflow_actions(
                primary_decision=primary_decision,
                adapter_output=adapter_outputs.get("protagonist", {}),
                audit_output=audit_outputs.get("protagonist", {}),
            ),
        },
        "relationship": {
            "official_output_applied": False,
            "merged_output": merged_outputs.get("relationship", _empty_merged_output("relationship")),
            "actions": _build_relationship_backflow_actions(
                relationships=relationships,
                adapter_output=adapter_outputs.get("relationship", {}),
                audit_output=audit_outputs.get("relationship", {}),
            ),
        },
        "profile": {
            "official_output_applied": True,
            "merged_output": merged_outputs.get("profile", _empty_merged_output("profile")),
            "field_actions": _build_profile_backflow_actions(
                structured_profile=structured_profile,
                adapter_output=adapter_outputs.get("profile", {}),
                audit_output=audit_outputs.get("profile", {}),
            ),
        },
    }


def _build_protagonist_backflow_actions(
    *,
    primary_decision: Dict[str, Any],
    adapter_output: Dict[str, Any],
    audit_output: Dict[str, Any],
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    decisions = list((audit_output.get("judge_output") or {}).get("decisions", []))
    tag_by_dimension = {
        str(tag.get("dimension") or ""): tag for tag in list(adapter_output.get("tags", []))
    }
    for decision in decisions:
        dimension = str(decision.get("dimension") or "")
        if not dimension:
            continue
        source_tag = tag_by_dimension.get(dimension, {})
        verdict = str(decision.get("verdict") or "accept")
        actions.append(
            {
                "target_id": "primary_decision",
                "mapped_dimension": dimension,
                "verdict": verdict,
                "judge_reason": decision.get("reason") or "",
                "value_before": source_tag.get("value"),
                "value_after": decision.get("value"),
                "applied_change": _backflow_change_name(verdict, is_profile=False),
                "requires_upstream_rerun": verdict in {"nullify", "downgrade"},
            }
        )
    return actions


def _build_relationship_backflow_actions(
    *,
    relationships: List[Dict[str, Any]],
    adapter_output: Dict[str, Any],
    audit_output: Dict[str, Any],
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    decisions = list((audit_output.get("judge_output") or {}).get("decisions", []))
    tag_by_dimension = {
        str(tag.get("dimension") or ""): tag for tag in list(adapter_output.get("tags", []))
    }
    rel_by_person = {
        str(rel.get("person_id") or ""): rel for rel in relationships
    }
    for decision in decisions:
        dimension = str(decision.get("dimension") or "")
        if not dimension:
            continue
        source_tag = tag_by_dimension.get(dimension, {})
        person_id = str(source_tag.get("value") or "")
        source_relationship = rel_by_person.get(person_id, {})
        verdict = str(decision.get("verdict") or "accept")
        actions.append(
            {
                "person_id": person_id or None,
                "mapped_dimension": dimension,
                "source_relationship_type": source_relationship.get("relationship_type"),
                "verdict": verdict,
                "judge_reason": decision.get("reason") or "",
                "value_before": source_tag.get("value"),
                "value_after": decision.get("value"),
                "applied_change": _backflow_change_name(verdict, is_profile=False),
                "requires_upstream_rerun": verdict in {"nullify", "downgrade"},
            }
        )
    return actions


def _build_profile_backflow_actions(
    *,
    structured_profile: Dict[str, Any],
    adapter_output: Dict[str, Any],
    audit_output: Dict[str, Any],
) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    decisions = list((audit_output.get("judge_output") or {}).get("decisions", []))
    tag_by_dimension = {
        str(tag.get("dimension") or ""): tag for tag in list(adapter_output.get("tags", []))
    }
    for decision in decisions:
        dimension = str(decision.get("dimension") or "")
        field_key = _profile_field_key_for_dimension(dimension, structured_profile, decision.get("value"))
        if not field_key:
            continue
        verdict = str(decision.get("verdict") or "accept")
        source_tag = tag_by_dimension.get(dimension, {})
        if verdict == "accept":
            continue
        actions.append(
            {
                "field_key": field_key,
                "mapped_dimension": dimension,
                "verdict": verdict,
                "judge_reason": decision.get("reason") or "",
                "value_before": source_tag.get("value"),
                "value_after": decision.get("value"),
                "applied_change": _backflow_change_name(verdict, is_profile=True),
            }
        )
    return actions


def apply_downstream_profile_backflow(
    structured_profile: Dict[str, Any] | None,
    downstream_audit_report: Dict[str, Any] | None,
    *,
    field_decisions: List[Dict[str, Any]] | None = None,
) -> Tuple[Dict[str, Any] | None, List[Dict[str, Any]]]:
    if not isinstance(structured_profile, dict):
        return structured_profile, list(field_decisions or [])
    updated_structured_profile = deepcopy(structured_profile)
    profile_backflow = (
        (downstream_audit_report or {}).get("backflow", {}).get("profile", {})
    )
    actions = list(profile_backflow.get("field_actions") or [])
    updated_field_decisions = deepcopy(list(field_decisions or []))
    decision_index = {
        str(item.get("field_key") or ""): item
        for item in updated_field_decisions
        if isinstance(item, dict) and str(item.get("field_key") or "")
    }
    for action in actions:
        field_key = str(action.get("field_key") or "")
        if not field_key:
            continue
        field_object = _get_nested(updated_structured_profile, field_key)
        if not isinstance(field_object, dict):
            continue
        evidence = field_object.get("evidence")
        if not isinstance(evidence, dict):
            evidence = {}
            field_object["evidence"] = evidence
        notes = evidence.setdefault("constraint_notes", [])
        if not isinstance(notes, list):
            notes = [str(notes)]
            evidence["constraint_notes"] = notes

        verdict = str(action.get("verdict") or "accept")
        reason = str(action.get("judge_reason") or "").strip()
        note = f"downstream_judge:{verdict}" + (f":{reason}" if reason else "")
        if note not in notes:
            notes.append(note)

        if verdict == "nullify":
            field_object["value"] = None
            field_object["confidence"] = 0.0
            field_object["reasoning"] = _append_reasoning_note(
                field_object.get("reasoning"),
                f"下游 Judge 否决：{reason}" if reason else "下游 Judge 否决",
            )
        elif verdict == "downgrade":
            field_object["reasoning"] = _append_reasoning_note(
                field_object.get("reasoning"),
                f"downstream Judge 建议降为 short_term：{reason}" if reason else "downstream Judge 建议降为 short_term",
            )
        decision = decision_index.get(field_key)
        if isinstance(decision, dict):
            _apply_profile_backflow_to_decision(decision, field_object, action)
    return updated_structured_profile, updated_field_decisions


def _apply_profile_backflow_to_decision(
    decision: Dict[str, Any],
    updated_final: Dict[str, Any],
    action: Dict[str, Any],
) -> None:
    if "final_before_backflow" not in decision and isinstance(decision.get("final"), dict):
        decision["final_before_backflow"] = deepcopy(decision["final"])
    decision["final"] = deepcopy(updated_final)
    decision["backflow"] = {
        "verdict": str(action.get("verdict") or "accept"),
        "judge_reason": str(action.get("judge_reason") or "").strip(),
        "applied_change": str(action.get("applied_change") or ""),
    }


def apply_downstream_protagonist_backflow(
    primary_decision: Dict[str, Any] | None,
    downstream_audit_report: Dict[str, Any] | None,
) -> Tuple[Dict[str, Any], bool]:
    current = deepcopy(primary_decision or {})
    actions = list(
        ((downstream_audit_report or {}).get("backflow", {}).get("protagonist", {}) or {}).get("actions", [])
    )
    triggered_actions = [
        action for action in actions if str(action.get("verdict") or "accept") in {"nullify", "downgrade"}
    ]
    if not triggered_actions:
        return current, False

    evidence = current.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}
        current["evidence"] = evidence
    notes = evidence.setdefault("constraint_notes", [])
    if not isinstance(notes, list):
        notes = [str(notes)]
        evidence["constraint_notes"] = notes
    original_primary_person_id = current.get("primary_person_id")
    if original_primary_person_id:
        evidence["rejected_primary_person_id"] = original_primary_person_id

    summary_reasons: List[str] = []
    for action in triggered_actions:
        verdict = str(action.get("verdict") or "accept")
        reason = str(action.get("judge_reason") or "").strip()
        note = f"downstream_judge:{verdict}" + (f":{reason}" if reason else "")
        if note not in notes:
            notes.append(note)
        if reason:
            summary_reasons.append(reason)

    current["mode"] = "photographer_mode"
    current["primary_person_id"] = None
    current["confidence"] = 0.0
    reason_text = "；".join(summary_reasons) if summary_reasons else "下游 Judge 未接受当前主角身份"
    current["reasoning"] = _append_reasoning_note(
        current.get("reasoning"),
        f"下游 Judge 触发主角回流：{reason_text}",
    )
    return current, True


def apply_downstream_relationship_backflow(
    relationships: Iterable[RelationshipRecord | Dict[str, Any]],
    dossiers: Iterable[RelationshipDossier],
    downstream_audit_report: Dict[str, Any] | None,
) -> Tuple[List[RelationshipRecord], List[RelationshipDossier], bool]:
    current_relationships = [_coerce_relationship_dataclass(item) for item in relationships]
    current_dossiers = [deepcopy(dossier) for dossier in dossiers]
    actions = list(
        ((downstream_audit_report or {}).get("backflow", {}).get("relationship", {}) or {}).get("actions", [])
    )
    actionable = {
        str(action.get("person_id") or ""): action
        for action in actions
        if str(action.get("verdict") or "accept") in {"nullify", "downgrade"} and str(action.get("person_id") or "")
    }
    if not actionable:
        return current_relationships, current_dossiers, False

    dossier_by_person = {dossier.person_id: dossier for dossier in current_dossiers}
    updated_relationships: List[RelationshipRecord] = []
    changed = False

    for relationship in current_relationships:
        action = actionable.get(relationship.person_id)
        if not action:
            updated_relationships.append(relationship)
            continue
        verdict = str(action.get("verdict") or "accept")
        reason = str(action.get("judge_reason") or "").strip()
        dossier = dossier_by_person.get(relationship.person_id)
        changed = True

        if verdict == "nullify":
            if dossier is not None:
                dossier.retention_decision = "drop"
                dossier.retention_reason = f"downstream_judge_nullify:{reason}" if reason else "downstream_judge_nullify"
                dossier.group_eligible = False
                dossier.group_block_reason = "downstream_judge_nullify"
                dossier.group_weight = 0.0
                dossier.relationship_result = {
                    "relationship_type": None,
                    "status": relationship.status,
                    "confidence": 0.0,
                    "reasoning": _append_reasoning_note(relationship.reasoning, f"下游 Judge 否决：{reason}" if reason else "下游 Judge 否决"),
                }
            continue

        downgrade_target = RELATIONSHIP_TYPE_SPECS.get(relationship.relationship_type)
        new_type = (
            downgrade_target.downgrade_target
            if downgrade_target and downgrade_target.downgrade_target
            else relationship.relationship_type
        )
        updated_evidence = dict(relationship.evidence or {})
        notes = updated_evidence.setdefault("constraint_notes", [])
        if not isinstance(notes, list):
            notes = [str(notes)]
            updated_evidence["constraint_notes"] = notes
        note = f"downstream_judge:downgrade" + (f":{reason}" if reason else "")
        if note not in notes:
            notes.append(note)

        updated_relationship = replace(
            relationship,
            relationship_type=new_type,
            confidence=min(relationship.confidence, 0.7),
            reasoning=_append_reasoning_note(
                relationship.reasoning,
                f"downstream Judge 建议降级为 {new_type}：{reason}" if reason else f"downstream Judge 建议降级为 {new_type}",
            ),
            evidence=updated_evidence,
        )
        if dossier is not None:
            dossier.relationship_result = {
                "relationship_type": updated_relationship.relationship_type,
                "status": updated_relationship.status,
                "confidence": updated_relationship.confidence,
                "reasoning": updated_relationship.reasoning,
            }
            dossier.retention_decision, dossier.retention_reason = _determine_retention(dossier, updated_relationship)
            dossier.group_eligible, dossier.group_block_reason, dossier.group_weight = _determine_group_eligibility(
                dossier,
                updated_relationship,
            )
        updated_relationships.append(updated_relationship)

    return updated_relationships, current_dossiers, changed


def _append_reasoning_note(original: Any, note: str) -> str:
    base = str(original or "").strip()
    if not base:
        return note
    if note in base:
        return base
    return f"{base} {note}"


def _backflow_change_name(verdict: str, *, is_profile: bool) -> str:
    if verdict == "nullify":
        return "nullify_value" if is_profile else "requires_rerun_or_manual_apply"
    if verdict == "downgrade":
        return "annotate_short_term_downgrade" if is_profile else "requires_rerun_or_manual_apply"
    return "no_change"


def _merge_judge_decisions(
    extractor_output: Dict[str, Any],
    judge_output: Dict[str, Any],
) -> Dict[str, Any]:
    result = {
        "agent_type": extractor_output.get("agent_type"),
        "tags": [],
    }
    decision_map = {
        str(decision.get("dimension") or ""): decision
        for decision in list(judge_output.get("decisions") or [])
    }
    for tag in list(extractor_output.get("tags") or []):
        merged_tag = deepcopy(tag)
        dimension = str(tag.get("dimension") or "")
        decision = decision_map.get(dimension)
        if decision:
            verdict = str(decision.get("verdict") or "accept")
            if verdict == "nullify":
                merged_tag["value"] = None
            elif verdict == "downgrade":
                merged_tag["stability"] = "short_term"
        result["tags"].append(merged_tag)
    return result


def _empty_merged_output(agent_type: str) -> Dict[str, Any]:
    return {"agent_type": agent_type, "tags": []}


def _iter_non_null_profile_fields(structured_profile: Dict[str, Any]) -> List[Tuple[str, Any]]:
    results: List[Tuple[str, Any]] = []

    def walk(node: Any, prefix: str) -> None:
        if not isinstance(node, dict):
            return
        if {"value", "confidence", "evidence", "reasoning"}.issubset(node.keys()):
            value = node.get("value")
            if value not in (None, "", []):
                results.append((prefix, value))
            return
        for key, child in node.items():
            child_prefix = f"{prefix}.{key}" if prefix else key
            walk(child, child_prefix)

    walk(structured_profile, "")
    return results


def _profile_field_key_for_dimension(dimension: str, structured_profile: Dict[str, Any], tag_value: Any) -> str | None:
    return PROFILE_DIMENSION_FIELD_PATHS.get(dimension)


def _matched_challenges(dimension: str, challenges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [challenge for challenge in challenges if str(challenge.get("target_tag") or "").startswith(dimension)]


def _find_decision(dimension: str, decisions: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    for decision in decisions:
        if str(decision.get("dimension") or "") == dimension:
            return decision
    return None


def _decision_to_status(decision: Dict[str, Any] | None) -> str:
    if not decision:
        return "accepted"
    verdict = str(decision.get("verdict") or "accept")
    if verdict == "accept":
        return "accepted"
    if verdict == "downgrade":
        return "downgraded"
    if verdict == "nullify":
        return "rejected"
    return "accepted"


def _coerce_relationship_dict(relationship: RelationshipRecord | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(relationship, RelationshipRecord):
        return {
            "person_id": relationship.person_id,
            "relationship_type": relationship.relationship_type,
            "intimacy_score": relationship.intimacy_score,
            "status": relationship.status,
            "confidence": relationship.confidence,
            "reasoning": relationship.reasoning,
            "shared_events": relationship.shared_events,
            "evidence": relationship.evidence,
        }
    return dict(relationship)


def _coerce_relationship_dataclass(relationship: RelationshipRecord | Dict[str, Any]) -> RelationshipRecord:
    if isinstance(relationship, RelationshipRecord):
        return relationship
    payload = dict(relationship)
    return RelationshipRecord(
        person_id=str(payload.get("person_id") or ""),
        relationship_type=str(payload.get("relationship_type") or ""),
        intimacy_score=float(payload.get("intimacy_score", 0.0) or 0.0),
        status=str(payload.get("status") or ""),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        reasoning=str(payload.get("reasoning") or ""),
        shared_events=list(payload.get("shared_events") or []),
        evidence=dict(payload.get("evidence") or {}),
    )


def _get_nested(payload: Dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def _stringify_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item or "").strip())
    return str(value)
