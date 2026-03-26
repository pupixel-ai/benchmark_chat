from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .storage import build_reflection_asset_paths, ensure_reflection_root
from .triage import route_case_fact
from .types import CaseFact, ObservationCase


def persist_downstream_audit_reflection_assets(
    *,
    downstream_audit_report: Dict[str, Any],
    project_root: str,
    user_name: str,
    album_id: str,
) -> Dict[str, Any]:
    paths = build_reflection_asset_paths(project_root=project_root, user_name=user_name)
    ensure_reflection_root(paths)

    observation_cases, case_facts = extract_downstream_audit_reflection_assets(
        downstream_audit_report=downstream_audit_report,
        user_name=user_name,
        album_id=album_id,
    )

    existing_observation_ids = _load_existing_case_ids(paths.observation_cases_path)
    existing_case_fact_ids = _load_existing_case_ids(paths.case_facts_path)

    new_observation_cases = [
        case.to_dict() for case in observation_cases if case.case_id not in existing_observation_ids
    ]
    new_case_facts = [
        fact.to_dict() for fact in case_facts if fact.case_id not in existing_case_fact_ids
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
        "case_ids": [fact["case_id"] for fact in new_case_facts],
    }


def extract_downstream_audit_reflection_assets(
    *,
    downstream_audit_report: Dict[str, Any],
    user_name: str,
    album_id: str,
) -> Tuple[List[ObservationCase], List[CaseFact]]:
    metadata = downstream_audit_report.get("metadata", {}) if isinstance(downstream_audit_report, dict) else {}
    audit_status = str(metadata.get("audit_status") or "")
    if audit_status == "skipped_init_failure":
        runtime_reason = (
            f"skipped_init_failure: {metadata.get('audit_error_type', 'RuntimeError')}: "
            f"{metadata.get('audit_error_message', '')}"
        ).strip()
        case_id = _build_case_id(
            album_id=album_id,
            agent_type="system",
            dimension="system>audit_runtime_failure",
            verdict="runtime_failure",
            reason=runtime_reason,
            evidence_refs=[],
        )
        observation = ObservationCase(
            case_id=case_id,
            user_name=user_name,
            album_id=album_id,
            stage="downstream_audit",
            dimension="system>audit_runtime_failure",
            entity_type="system_runtime",
            entity_id="profile_agent_runtime",
            signal_source="downstream_audit",
            first_seen_stage="downstream_audit",
            surfaced_stage="downstream_audit",
            decision_trace={"verdict": "runtime_failure", "reason": runtime_reason},
            evidence_summary={"evidence_count": 0},
            tool_usage_summary={"judge_present": False, "extractor_v1_present": False, "extractor_v2_present": False},
            guardrail_trigger=["audit_runtime_failure"],
            raw_payload={"metadata": metadata},
        )
        fact = CaseFact(
            case_id=case_id,
            user_name=user_name,
            album_id=album_id,
            entity_type="system_runtime",
            entity_id="profile_agent_runtime",
            dimension="system>audit_runtime_failure",
            signal_source="downstream_audit",
            first_seen_stage="downstream_audit",
            surfaced_stage="downstream_audit",
            routing_result="engineering_issue",
            business_priority="high",
            auto_confidence=1.0,
            decision_trace={"verdict": "runtime_failure", "reason": runtime_reason},
            tool_usage_summary={"judge_present": False, "extractor_v1_present": False, "extractor_v2_present": False},
            downstream_challenge={"reason": runtime_reason},
            downstream_judge={"verdict": "runtime_failure", "reason": runtime_reason, "agent_type": "system"},
            evidence_refs=[],
        )
        fact = route_case_fact(fact)
        return [observation], [fact]

    observations: List[ObservationCase] = []
    case_facts: List[CaseFact] = []
    for agent_type in ("protagonist", "relationship", "profile"):
        section = downstream_audit_report.get(agent_type, {})
        if not isinstance(section, dict):
            continue
        decisions = list((section.get("judge_output") or {}).get("decisions") or [])
        v1_tags = list((section.get("extractor_output") or {}).get("tags") or [])
        v2_tags = list((section.get("extractor_v2_output") or {}).get("tags") or v1_tags)
        v1_by_dimension = {str(tag.get("dimension") or ""): tag for tag in v1_tags}
        v2_by_dimension = {str(tag.get("dimension") or ""): tag for tag in v2_tags}

        for decision in decisions:
            verdict = str(decision.get("verdict") or "accept")
            if verdict not in {"nullify", "downgrade"}:
                continue
            dimension = str(decision.get("dimension") or "").strip()
            if not dimension:
                continue
            reason = str(decision.get("reason") or "").strip()
            v1_tag = v1_by_dimension.get(dimension, {})
            v2_tag = v2_by_dimension.get(dimension, {})
            evidence_refs = _normalize_evidence_refs(
                list(v2_tag.get("evidence") or v1_tag.get("evidence") or [])
            )
            case_id = _build_case_id(
                album_id=album_id,
                agent_type=agent_type,
                dimension=dimension,
                verdict=verdict,
                reason=reason,
                evidence_refs=evidence_refs,
            )
            entity_type = _resolve_entity_type(agent_type)
            first_seen_stage = _resolve_first_seen_stage(agent_type)
            tool_usage_summary = {
                "judge_present": True,
                "extractor_v1_present": bool(v1_tag),
                "extractor_v2_present": bool(v2_tag),
            }
            decision_trace = {
                "agent_type": agent_type,
                "verdict": verdict,
                "reason": reason,
            }
            observation = ObservationCase(
                case_id=case_id,
                user_name=user_name,
                album_id=album_id,
                stage="downstream_audit",
                dimension=dimension,
                entity_type=entity_type,
                entity_id=dimension,
                signal_source="downstream_audit",
                first_seen_stage=first_seen_stage,
                surfaced_stage="downstream_audit",
                decision_trace=decision_trace,
                evidence_summary=_build_evidence_summary(evidence_refs),
                tool_usage_summary=tool_usage_summary,
                raw_payload={
                    "extractor_v1": v1_tag,
                    "extractor_v2": v2_tag,
                    "judge_decision": decision,
                },
            )
            fact = CaseFact(
                case_id=case_id,
                user_name=user_name,
                album_id=album_id,
                entity_type=entity_type,
                entity_id=dimension,
                dimension=dimension,
                signal_source="downstream_audit",
                first_seen_stage=first_seen_stage,
                surfaced_stage="downstream_audit",
                routing_result="audit_disagreement",
                business_priority=_resolve_business_priority(agent_type),
                auto_confidence=_resolve_auto_confidence(decision, v2_tag, v1_tag),
                decision_trace=decision_trace,
                tool_usage_summary=tool_usage_summary,
                upstream_output={
                    "extractor_v1_value": v1_tag.get("value"),
                    "extractor_v2_value": v2_tag.get("value"),
                },
                downstream_challenge={"reason": reason, "verdict": verdict},
                downstream_v2=v2_tag,
                downstream_judge={"verdict": verdict, "reason": reason, "agent_type": agent_type},
                evidence_refs=evidence_refs,
            )
            fact = route_case_fact(fact)
            observations.append(observation)
            case_facts.append(fact)

    return observations, case_facts


def _resolve_entity_type(agent_type: str) -> str:
    return {
        "protagonist": "protagonist_tag",
        "relationship": "relationship_tag",
        "profile": "profile_field",
    }.get(agent_type, "audit_tag")


def _resolve_first_seen_stage(agent_type: str) -> str:
    return {
        "protagonist": "lp1",
        "relationship": "lp2",
        "profile": "lp3",
    }.get(agent_type, "downstream_audit")


def _resolve_business_priority(agent_type: str) -> str:
    return "high" if agent_type in {"protagonist", "relationship"} else "medium"


def _resolve_auto_confidence(decision: Dict[str, Any], v2_tag: Dict[str, Any], v1_tag: Dict[str, Any]) -> float:
    candidate = decision.get("confidence")
    if candidate is None:
        candidate = v2_tag.get("confidence")
    if candidate is None:
        candidate = v1_tag.get("confidence")
    try:
        return float(candidate)
    except (TypeError, ValueError):
        return 0.7


def _build_evidence_summary(evidence_refs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "evidence_count": len(evidence_refs),
        "event_ids": [ref["source_id"] for ref in evidence_refs if ref.get("source_type") == "event"],
        "photo_ids": [ref["source_id"] for ref in evidence_refs if ref.get("source_type") == "photo"],
        "person_ids": [ref["source_id"] for ref in evidence_refs if ref.get("source_type") == "person"],
        "feature_names": sorted(
            {
                feature_name
                for ref in evidence_refs
                for feature_name in list(ref.get("feature_names") or [])
                if feature_name
            }
        ),
    }


def _normalize_evidence_refs(evidence_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for item in evidence_items:
        if not isinstance(item, dict):
            continue
        source_type = "event" if item.get("event_id") else "photo" if item.get("photo_id") else "person" if item.get("person_id") else "unknown"
        source_id = str(item.get("event_id") or item.get("photo_id") or item.get("person_id") or "").strip()
        payload = {
            "source_type": source_type,
            "source_id": source_id,
            "description": str(item.get("description") or "").strip(),
            "feature_names": [
                str(feature_name).strip()
                for feature_name in list(item.get("feature_names") or [])
                if str(feature_name).strip()
            ],
            "evidence_type": str(item.get("evidence_type") or "direct"),
            "inference_depth": int(item.get("inference_depth") or 1),
        }
        identity = (
            payload["source_type"],
            payload["source_id"],
            payload["description"],
            tuple(payload["feature_names"]),
        )
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(payload)
    return normalized


def _build_case_id(
    *,
    album_id: str,
    agent_type: str,
    dimension: str,
    verdict: str,
    reason: str,
    evidence_refs: List[Dict[str, Any]],
) -> str:
    fingerprint = json.dumps(evidence_refs, ensure_ascii=False, sort_keys=True)
    raw = "|".join([album_id, agent_type, dimension, verdict, reason.strip(), fingerprint])
    return f"rc_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]}"


def _load_existing_case_ids(path: str) -> set[str]:
    file_path = Path(path)
    if not file_path.exists():
        return set()
    case_ids: set[str] = set()
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            case_id = str((payload or {}).get("case_id") or "").strip()
            if case_id:
                case_ids.add(case_id)
    return case_ids


def _append_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
