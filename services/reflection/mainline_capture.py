from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .downstream_capture import _append_jsonl, _build_case_id, _load_existing_case_ids
from .storage import build_reflection_asset_paths, ensure_reflection_root
from .triage import route_case_fact
from .types import CaseFact, ObservationCase


def persist_mainline_reflection_assets(
    *,
    internal_artifacts: Dict[str, Any],
    project_root: str,
    user_name: str,
    album_id: str,
) -> Dict[str, Any]:
    paths = build_reflection_asset_paths(project_root=project_root, user_name=user_name)
    ensure_reflection_root(paths)

    observation_cases, case_facts = extract_mainline_reflection_assets(
        internal_artifacts=internal_artifacts,
        user_name=user_name,
        album_id=album_id,
    )

    existing_observation_ids = _load_existing_case_ids(paths.observation_cases_path)
    existing_case_fact_ids = _load_existing_case_ids(paths.case_facts_path)
    written_profile_field_trace_count = 0
    profile_field_trace_index_records, trace_paths_by_case_id = _persist_profile_field_traces(
        internal_artifacts=internal_artifacts,
        paths=paths,
        user_name=user_name,
        album_id=album_id,
        case_facts=case_facts,
    )
    written_profile_field_trace_count = len(profile_field_trace_index_records)
    new_observation_cases = [
        case.to_dict() for case in observation_cases if case.case_id not in existing_observation_ids
    ]
    new_case_facts = [
        _case_fact_with_trace_path(fact, trace_paths_by_case_id).to_dict()
        for fact in case_facts
        if fact.case_id not in existing_case_fact_ids
    ]

    if new_observation_cases:
        _append_jsonl(paths.observation_cases_path, new_observation_cases)
    if new_case_facts:
        _append_jsonl(paths.case_facts_path, new_case_facts)

    return {
        "observation_cases_path": paths.observation_cases_path,
        "case_facts_path": paths.case_facts_path,
        "written_observation_count": len(new_observation_cases),
        "written_case_fact_count": len(new_case_facts),
        "profile_field_trace_index_path": paths.profile_field_trace_index_path,
        "written_profile_field_trace_count": written_profile_field_trace_count,
        "case_ids": [fact["case_id"] for fact in new_case_facts],
    }


def extract_mainline_reflection_assets(
    *,
    internal_artifacts: Dict[str, Any],
    user_name: str,
    album_id: str,
) -> Tuple[List[ObservationCase], List[CaseFact]]:
    observations: List[ObservationCase] = []
    facts: List[CaseFact] = []

    primary_decision = dict(internal_artifacts.get("primary_decision") or {})
    if primary_decision:
        observation, fact = _build_primary_assets(
            primary_decision=primary_decision,
            user_name=user_name,
            album_id=album_id,
        )
        observations.append(observation)
        facts.append(fact)

    for dossier in list(internal_artifacts.get("relationship_dossiers") or []):
        if not isinstance(dossier, dict):
            continue
        observation, fact = _build_relationship_assets(
            dossier=dossier,
            user_name=user_name,
            album_id=album_id,
        )
        observations.append(observation)
        facts.append(fact)

    for field_decision in list(internal_artifacts.get("profile_fact_decisions") or []):
        if not isinstance(field_decision, dict):
            continue
        observation, fact = _build_profile_field_assets(
            field_decision=field_decision,
            user_name=user_name,
            album_id=album_id,
        )
        observations.append(observation)
        facts.append(fact)

    return observations, facts


def _build_primary_assets(
    *,
    primary_decision: Dict[str, Any],
    user_name: str,
    album_id: str,
) -> Tuple[ObservationCase, CaseFact]:
    evidence_refs = _normalize_primary_evidence(primary_decision.get("evidence") or {})
    decision_trace = {
        "mode": primary_decision.get("mode"),
        "reasoning": primary_decision.get("reasoning"),
    }
    case_id = _build_case_id(
        album_id=album_id,
        agent_type="mainline_primary",
        dimension="primary_decision",
        verdict=str(primary_decision.get("mode") or "unknown"),
        reason=str(primary_decision.get("reasoning") or ""),
        evidence_refs=evidence_refs,
    )
    observation = ObservationCase(
        case_id=case_id,
        user_name=user_name,
        album_id=album_id,
        stage="lp1",
        dimension="primary_decision",
        entity_type="primary_person",
        entity_id=str(primary_decision.get("primary_person_id") or primary_decision.get("mode") or "unknown"),
        signal_source="mainline_primary",
        first_seen_stage="lp1",
        surfaced_stage="lp1",
        decision_trace=decision_trace,
        evidence_summary={"evidence_count": len(evidence_refs)},
        tool_usage_summary={"source": "primary_decision"},
        raw_payload=primary_decision,
    )
    fact = CaseFact(
        case_id=case_id,
        user_name=user_name,
        album_id=album_id,
        entity_type="primary_person",
        entity_id=str(primary_decision.get("primary_person_id") or primary_decision.get("mode") or "unknown"),
        dimension="primary_decision",
        signal_source="mainline_primary",
        first_seen_stage="lp1",
        surfaced_stage="lp1",
        routing_result="pending_triage",
        business_priority="high",
        auto_confidence=_safe_float(primary_decision.get("confidence"), default=0.7),
        decision_trace=decision_trace,
        tool_usage_summary={"source": "primary_decision"},
        upstream_output=primary_decision,
        evidence_refs=evidence_refs,
    )
    return observation, route_case_fact(fact)


def _build_relationship_assets(
    *,
    dossier: Dict[str, Any],
    user_name: str,
    album_id: str,
) -> Tuple[ObservationCase, CaseFact]:
    person_id = str(dossier.get("person_id") or "unknown")
    relationship_result = dict(dossier.get("relationship_result") or {})
    evidence_refs = _normalize_relationship_evidence(dossier)
    decision_trace = {
        "retention_decision": dossier.get("retention_decision"),
        "retention_reason": dossier.get("retention_reason"),
        "relationship_type": relationship_result.get("relationship_type"),
    }
    case_id = _build_case_id(
        album_id=album_id,
        agent_type="mainline_relationship",
        dimension=f"relationship:{person_id}",
        verdict=str(dossier.get("retention_decision") or "review"),
        reason=str(dossier.get("retention_reason") or ""),
        evidence_refs=evidence_refs,
    )
    observation = ObservationCase(
        case_id=case_id,
        user_name=user_name,
        album_id=album_id,
        stage="lp2",
        dimension=f"relationship:{person_id}",
        entity_type="relationship_candidate",
        entity_id=person_id,
        signal_source="mainline_relationship",
        first_seen_stage="lp2",
        surfaced_stage="lp2",
        decision_trace=decision_trace,
        evidence_summary={"evidence_count": len(evidence_refs)},
        tool_usage_summary={"source": "relationship_dossier"},
        raw_payload=dossier,
    )
    fact = CaseFact(
        case_id=case_id,
        user_name=user_name,
        album_id=album_id,
        entity_type="relationship_candidate",
        entity_id=person_id,
        dimension=f"relationship:{person_id}",
        signal_source="mainline_relationship",
        first_seen_stage="lp2",
        surfaced_stage="lp2",
        routing_result="pending_triage",
        business_priority="medium",
        auto_confidence=_safe_float(relationship_result.get("confidence"), default=0.6),
        decision_trace=decision_trace,
        tool_usage_summary={"source": "relationship_dossier"},
        upstream_output=relationship_result,
        evidence_refs=evidence_refs,
    )
    return observation, route_case_fact(fact)


def _build_profile_field_assets(
    *,
    field_decision: Dict[str, Any],
    user_name: str,
    album_id: str,
) -> Tuple[ObservationCase, CaseFact]:
    field_key = str(field_decision.get("field_key") or "unknown_field")
    backflow_payload = dict(field_decision.get("backflow") or {})
    final_payload = dict(field_decision.get("final") or {})
    pre_audit_payload = dict(field_decision.get("final_before_backflow") or final_payload)
    tool_usage_summary = _summarize_profile_tool_usage(field_decision.get("tool_trace") or {})
    evidence_refs = _normalize_profile_evidence(final_payload.get("evidence") or {})
    decision_trace = {
        "field_key": field_key,
        "null_reason": field_decision.get("null_reason"),
        "reasoning": final_payload.get("reasoning"),
        "audit_action_type": str(backflow_payload.get("verdict") or ""),
    }
    case_id = _build_case_id(
        album_id=album_id,
        agent_type="mainline_profile",
        dimension=field_key,
        verdict="field_decision",
        reason=str(final_payload.get("reasoning") or ""),
        evidence_refs=evidence_refs,
    )
    observation = ObservationCase(
        case_id=case_id,
        user_name=user_name,
        album_id=album_id,
        stage="lp3",
        dimension=field_key,
        entity_type="profile_field",
        entity_id=field_key,
        signal_source="mainline_profile",
        first_seen_stage="lp3",
        surfaced_stage="downstream_audit" if backflow_payload else "lp3",
        decision_trace=decision_trace,
        evidence_summary={"evidence_count": len(evidence_refs)},
        tool_usage_summary=tool_usage_summary,
        raw_payload=field_decision,
    )
    fact = CaseFact(
        case_id=case_id,
        user_name=user_name,
        album_id=album_id,
        entity_type="profile_field",
        entity_id=field_key,
        dimension=field_key,
        signal_source="mainline_profile",
        first_seen_stage="lp3",
        surfaced_stage="downstream_audit" if backflow_payload else "lp3",
        routing_result="pending_triage",
        business_priority="medium",
        auto_confidence=_safe_float(final_payload.get("confidence"), default=0.6),
        decision_trace=decision_trace,
        tool_usage_summary=tool_usage_summary,
        pre_audit_output=pre_audit_payload,
        upstream_output=final_payload,
        audit_action_type=str(backflow_payload.get("verdict") or ""),
        downstream_judge=(
            {
                "verdict": str(backflow_payload.get("verdict") or ""),
                "reason": str(backflow_payload.get("judge_reason") or "").strip(),
                "agent_type": "profile",
            }
            if backflow_payload
            else {}
        ),
        evidence_refs=evidence_refs,
    )
    return observation, route_case_fact(fact)


def _normalize_primary_evidence(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for event_id in list(evidence.get("event_ids") or []):
        refs.append({"source_type": "event", "source_id": str(event_id), "description": "", "feature_names": []})
    for photo_id in list(evidence.get("photo_ids") or []):
        refs.append({"source_type": "photo", "source_id": str(photo_id), "description": "", "feature_names": []})
    for person_id in list(evidence.get("person_ids") or []):
        refs.append({"source_type": "person", "source_id": str(person_id), "description": "", "feature_names": []})
    if evidence.get("feature_names"):
        refs.append(
            {
                "source_type": "feature",
                "source_id": "primary_features",
                "description": "",
                "feature_names": [str(name) for name in list(evidence.get("feature_names") or []) if str(name)],
            }
        )
    return refs


def _normalize_relationship_evidence(dossier: Dict[str, Any]) -> List[Dict[str, Any]]:
    evidence_refs = list(dossier.get("evidence_refs") or [])
    if evidence_refs:
        return [
            {
                "source_type": str(ref.get("source_type") or "unknown"),
                "source_id": str(ref.get("source_id") or ""),
                "description": str(ref.get("description") or ""),
                "feature_names": [str(name) for name in list(ref.get("feature_names") or []) if str(name)],
            }
            for ref in evidence_refs
            if isinstance(ref, dict)
        ]
    refs: List[Dict[str, Any]] = []
    for event in list(dossier.get("shared_events") or []):
        event_id = str((event or {}).get("event_id") or "").strip()
        if event_id:
            refs.append({"source_type": "event", "source_id": event_id, "description": "", "feature_names": []})
    if dossier.get("interaction_signals"):
        refs.append(
            {
                "source_type": "feature",
                "source_id": "relationship_signals",
                "description": "",
                "feature_names": [str(name) for name in list(dossier.get("interaction_signals") or []) if str(name)],
            }
        )
    return refs


def _normalize_profile_evidence(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for ref in list(evidence.get("supporting_refs") or []):
        if not isinstance(ref, dict):
            continue
        refs.append(
            {
                "source_type": str(ref.get("source_type") or "unknown"),
                "source_id": str(ref.get("source_id") or ""),
                "description": str(ref.get("description") or ""),
                "feature_names": [str(name) for name in list(ref.get("feature_names") or []) if str(name)],
            }
        )
    if evidence.get("feature_names"):
        refs.append(
            {
                "source_type": "feature",
                "source_id": "profile_features",
                "description": str(evidence.get("summary") or ""),
                "feature_names": [str(name) for name in list(evidence.get("feature_names") or []) if str(name)],
            }
        )
    return refs


def _summarize_profile_tool_usage(tool_trace: Dict[str, Any]) -> Dict[str, Any]:
    evidence_bundle = dict(tool_trace.get("evidence_bundle") or {})
    supporting_refs = dict(evidence_bundle.get("supporting_refs") or {})
    flattened = []
    for refs in supporting_refs.values():
        flattened.extend(list(refs or []))
    return {
        "source": "profile_fact_decision",
        "tool_called": bool(tool_trace),
        "retrieval_hit_count": len(flattened),
        "support_count": len(flattened),
        "tool_trace_present": bool(tool_trace),
    }


def _persist_profile_field_traces(
    *,
    internal_artifacts: Dict[str, Any],
    paths,
    user_name: str,
    album_id: str,
    case_facts: List[CaseFact],
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    existing_trace_case_ids = _load_existing_case_ids(paths.profile_field_trace_index_path)
    llm_debug_by_batch = {
        str(payload.get("batch_name") or "").strip(): dict(payload)
        for payload in list(internal_artifacts.get("profile_llm_batch_debug") or [])
        if isinstance(payload, dict) and str(payload.get("batch_name") or "").strip()
    }
    facts_by_field = {
        fact.dimension: fact
        for fact in case_facts
        if fact.signal_source == "mainline_profile" and fact.entity_type == "profile_field"
    }
    index_records: List[Dict[str, Any]] = []
    trace_paths_by_case_id: Dict[str, str] = {}
    for field_decision in list(internal_artifacts.get("profile_fact_decisions") or []):
        if not isinstance(field_decision, dict):
            continue
        field_key = str(field_decision.get("field_key") or "").strip()
        if not field_key or field_key not in facts_by_field:
            continue
        fact = facts_by_field[field_key]
        trace_payload_path = str(Path(paths.profile_field_trace_payload_dir) / f"{fact.case_id}.json")
        trace_paths_by_case_id[fact.case_id] = trace_payload_path

        final_payload = dict(field_decision.get("final") or {})
        final_evidence = dict(final_payload.get("evidence") or {})
        supporting_refs = list(final_evidence.get("supporting_refs") or [])
        contradicting_refs = list(final_evidence.get("contradicting_refs") or [])

        payload = {
            "case_id": fact.case_id,
            "user_name": user_name,
            "album_id": album_id,
            "field_key": field_key,
            "batch_name": str(field_decision.get("batch_name") or "").strip(),
            "field_spec_snapshot": dict(field_decision.get("field_spec_snapshot") or {}),
            "tool_trace": dict(field_decision.get("tool_trace") or {}),
            "draft": dict(field_decision.get("draft") or {}),
            "final_before_backflow": dict(field_decision.get("final_before_backflow") or {}),
            "final": final_payload,
            "backflow": dict(field_decision.get("backflow") or {}),
            "null_reason": field_decision.get("null_reason"),
            "llm_batch_debug": llm_debug_by_batch.get(str(field_decision.get("batch_name") or "").strip(), {}),
            "resolved_facts_summary_at_decision_time": dict(field_decision.get("resolved_facts_summary_at_decision_time") or {}),
        }
        trace_file = Path(trace_payload_path)
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if fact.case_id in existing_trace_case_ids:
            continue
        index_records.append(
            {
                "case_id": fact.case_id,
                "album_id": album_id,
                "field_key": field_key,
                "batch_name": str(field_decision.get("batch_name") or "").strip(),
                "tool_called": bool(field_decision.get("tool_trace")),
                "retrieval_hit_count": int(((fact.tool_usage_summary or {}).get("retrieval_hit_count") or 0)),
                "null_reason": str(field_decision.get("null_reason") or ""),
                "selected_supporting_ref_ids": _collect_ref_ids(supporting_refs),
                "selected_contradicting_ref_ids": _collect_ref_ids(contradicting_refs),
                "trace_payload_path": trace_payload_path,
            }
        )
    if index_records:
        _append_jsonl(paths.profile_field_trace_index_path, index_records)
    return index_records, trace_paths_by_case_id


def _case_fact_with_trace_path(fact: CaseFact, trace_paths_by_case_id: Dict[str, str]) -> CaseFact:
    if fact.case_id in trace_paths_by_case_id:
        fact.trace_payload_path = trace_paths_by_case_id[fact.case_id]
    return fact


def _collect_ref_ids(refs: List[Dict[str, Any]]) -> List[str]:
    resolved: List[str] = []
    seen = set()
    for ref in refs:
        source_id = str(ref.get("source_id") or ref.get("event_id") or ref.get("photo_id") or ref.get("person_id") or "").strip()
        if not source_id or source_id in seen:
            continue
        seen.add(source_id)
        resolved.append(source_id)
    return resolved


def _safe_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
