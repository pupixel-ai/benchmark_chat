from __future__ import annotations

from .types import CaseFact


def route_case_fact(case_fact: CaseFact) -> CaseFact:
    case_fact.support_count = len(list(case_fact.evidence_refs or []))

    if case_fact.entity_type == "system_runtime":
        case_fact.routing_result = "engineering_issue"
        case_fact.business_priority = "high"
        case_fact.triage_reason = "downstream_runtime_failure"
        case_fact.accuracy_gap_status = "open"
        case_fact.resolution_route = "engineering_fix"
        return case_fact

    if case_fact.signal_source == "downstream_audit":
        case_fact.routing_result = "audit_disagreement"
        if case_fact.entity_type in {"primary_person", "protagonist_tag", "relationship_candidate", "relationship_tag"}:
            case_fact.business_priority = "high"
        else:
            case_fact.business_priority = "medium"
        case_fact.triage_reason = "downstream_judge_challenged_existing_output"
        return case_fact

    if case_fact.signal_source == "mainline_primary":
        case_fact.routing_result = "strategy_candidate"
        case_fact.business_priority = "high"
        case_fact.triage_reason = "primary_decision_ready_for_patterning"
        return case_fact

    if case_fact.signal_source == "mainline_relationship":
        if (
            str(case_fact.decision_trace.get("retention_decision") or "") == "suppress"
            and case_fact.support_count == 0
        ):
            case_fact.routing_result = "expected_uncertainty"
            case_fact.business_priority = "low"
            case_fact.triage_reason = "suppressed_relationship_without_supporting_evidence"
            return case_fact
        case_fact.routing_result = "strategy_candidate"
        case_fact.business_priority = "medium"
        case_fact.triage_reason = "relationship_signal_ready_for_patterning"
        return case_fact

    if case_fact.signal_source == "mainline_profile":
        upstream_value = case_fact.upstream_output.get("value")
        if upstream_value in (None, "", []):
            case_fact.routing_result = "strategy_candidate"
            case_fact.business_priority = "high"
            case_fact.triage_reason = "profile_field_missing_value_candidate"
            case_fact.badcase_source = "empty_output_candidate"
            case_fact.badcase_kind = "missing_value"
            case_fact.accuracy_gap_status = "open"
            return case_fact
        case_fact.routing_result = "strategy_candidate"
        case_fact.business_priority = "medium"
        case_fact.triage_reason = "profile_field_ready_for_patterning"
        return case_fact

    case_fact.routing_result = "pending_triage"
    case_fact.business_priority = case_fact.business_priority or "medium"
    case_fact.triage_reason = "manual_review_needed"
    return case_fact
