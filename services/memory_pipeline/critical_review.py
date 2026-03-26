from __future__ import annotations

import json
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Tuple

from .types import CleanupRecord, CriticalReviewTicket, ReviewRecord, ReviewState

NON_REPAIRABLE_REASON_CODES = {
    "silent_by_missing_social_media",
    "hard_policy_block",
    "no_evidence_in_allowed_sources",
    "user_annotation_required",
    "max_rounds_reached",
    "rerun_no_state_change",
}

PRIMARY_REPAIRABLE_REASON_CODES = {
    "primary_ambiguity",
    "split_risk",
    "photographer_mode_conflict",
    "photographed_subject_veto_conflict",
    "selected_low_value_person",
    "selected_non_real_person",
    "primary_under_supported",
}

RELATIONSHIP_REPAIRABLE_REASON_CODES = {
    "screen_or_virtual_context",
    "single_photo_without_strong_signal",
    "low_confidence_low_signal_relationship",
    "over_suppressed_relationship",
    "relationship_type_over_upgrade",
    "group_eligibility_conflict",
}

PROFILE_REPAIRABLE_REASON_CODES = {
    "missing_text_anchor_generator",
    "missing_item_candidate_generator",
    "missing_place_candidate_generator",
    "missing_time_pattern_generator",
    "missing_expression_candidate_generator",
    "value_generator_failed_after_candidates",
    "candidate_noise_brand_tokens",
    "pseudo_place_candidates",
    "recent_window_vs_long_term_conflict_unresolved",
    "expression_conflict_needs_counter_evidence_recheck",
    "trace_cleanup",
}

REASON_REPAIR_MODE: Dict[str, str] = {
    "trace_cleanup": "orchestrator_fixable",
    "screen_or_virtual_context": "orchestrator_fixable",
    "single_photo_without_strong_signal": "orchestrator_fixable",
    "low_confidence_low_signal_relationship": "orchestrator_fixable",
    "over_suppressed_relationship": "orchestrator_fixable",
    "relationship_type_over_upgrade": "orchestrator_fixable",
    "group_eligibility_conflict": "orchestrator_fixable",
    "primary_ambiguity": "orchestrator_fixable",
    "split_risk": "orchestrator_fixable",
    "photographer_mode_conflict": "orchestrator_fixable",
    "photographed_subject_veto_conflict": "orchestrator_fixable",
    "selected_low_value_person": "orchestrator_fixable",
    "selected_non_real_person": "orchestrator_fixable",
    "primary_under_supported": "orchestrator_fixable",
    "missing_text_anchor_generator": "strategy_required",
    "missing_item_candidate_generator": "strategy_required",
    "missing_place_candidate_generator": "strategy_required",
    "missing_time_pattern_generator": "strategy_required",
    "missing_expression_candidate_generator": "strategy_required",
    "value_generator_failed_after_candidates": "strategy_required",
    "candidate_noise_brand_tokens": "strategy_required",
    "pseudo_place_candidates": "strategy_required",
    "recent_window_vs_long_term_conflict_unresolved": "strategy_required",
    "expression_conflict_needs_counter_evidence_recheck": "strategy_required",
    "silent_by_missing_social_media": "non_repairable",
    "hard_policy_block": "non_repairable",
    "no_evidence_in_allowed_sources": "non_repairable",
    "user_annotation_required": "non_repairable",
    "max_rounds_reached": "non_repairable",
    "rerun_no_state_change": "non_repairable",
}

TERMINAL_STATES = {
    ReviewState.TERMINAL_KEEP.value,
    ReviewState.TERMINAL_NULL.value,
    ReviewState.TERMINAL_SUPPRESS.value,
    ReviewState.TERMINAL_NO_CHANGE.value,
    ReviewState.TERMINAL_NON_REPAIRABLE.value,
    ReviewState.TERMINAL_MAX_ROUNDS.value,
    ReviewState.TERMINAL_FAILED.value,
}

TRACE_CLEANUP_FIELD_WHITELIST = {"long_term_facts.identity.gender"}

PSEUDO_PLACE_VALUES = {
    "博物馆",
    "大众化餐饮店",
    "大众化餐饮店/食堂",
    "大学教室或学习场所",
    "抓娃娃机区",
    "办公区",
    "便利店",
    "置身于博物馆",
    "室内环境",
    "户外公共空间或公园",
    "商场电玩中心",
}

BRAND_NOISE_TOKENS = {
    "happy",
    "画面",
    "摄影棚或特定风格",
    "摄影棚",
    "特定风格",
    "下当前面部状态与",
    "有粉色蝴蝶结虚拟",
}

RELAXED_CONFLICT_SCORE_REASON_CODES = {
    "candidate_noise_brand_tokens",
    "pseudo_place_candidates",
    "value_generator_failed_after_candidates",
}

HARD_POLICY_BLOCK_FIELDS = {
    "long_term_expression.morality",
    "long_term_expression.philosophy",
    "long_term_expression.personality_mbti",
    "short_term_expression.mental_state",
    "short_term_expression.motivation_shift",
    "short_term_expression.stress_signal",
    "long_term_facts.material.income_model",
}


def build_critical_review_records(
    state: Any,
    profile_result: Dict[str, Any],
    *,
    round_number: int = 0,
) -> List[ReviewRecord]:
    records: List[ReviewRecord] = []
    review_index = 1

    primary_record = _build_primary_review_record(state, round_number, review_index)
    records.append(primary_record)
    review_index += 1

    for dossier in state.relationship_dossiers or []:
        records.append(_build_relationship_review_record(dossier, round_number, review_index))
        review_index += 1

    for field_decision in profile_result.get("field_decisions", []):
        records.append(_build_profile_review_record(field_decision, round_number, review_index))
        review_index += 1

    return records


def build_critical_rerun_tickets(
    records: List[ReviewRecord] | List[Dict[str, Any]],
    *,
    round_number: int = 1,
    max_rounds: int = 2,
) -> List[CriticalReviewTicket]:
    tickets: List[CriticalReviewTicket] = []
    ticket_index = 1
    for record in records:
        review_status = _record_attr(record, "review_status")
        reparability = _record_attr(record, "reparability")
        if review_status != ReviewState.QUEUED.value:
            continue
        if reparability != "repairable":
            continue
        if round_number > max_rounds:
            continue
        reason_code = str(_record_attr(record, "reason_code") or "")
        scope = str(_record_attr(record, "scope") or "")
        candidate_family = _record_attr(record, "candidate_family")
        target_id = str(_record_attr(record, "target_id") or "")
        review_id = str(_record_attr(record, "review_id") or "")
        evidence_ids = _record_attr(record, "evidence_ids") or {}
        questioned_evidence_ids = _record_attr(record, "questioned_evidence_ids") or []
        evidence_focus_ids = _record_attr(record, "evidence_focus_ids") or []
        suggested_tools = _suggested_tools_for_reason(reason_code, scope, candidate_family)
        rerun_scope = _rerun_scope_for_reason(scope, reason_code)
        downstream_impact = _downstream_impact_for_scope(scope)
        evidence_hash = _stable_evidence_hash(target_id, reason_code, evidence_focus_ids)
        tickets.append(
            CriticalReviewTicket(
                ticket_id=f"CRT_{ticket_index:04d}",
                source_review_id=review_id,
                scope=scope,
                target_id=target_id,
                critical_type=str(_record_attr(record, "critical_type") or ""),
                reason_code=reason_code,
                candidate_family=candidate_family,
                evidence_ids=evidence_ids,
                questioned_evidence_ids=questioned_evidence_ids,
                suggested_tools=suggested_tools,
                rerun_scope=rerun_scope,
                downstream_impact=downstream_impact,
                round=round_number,
                max_rounds=max_rounds,
                evidence_hash=evidence_hash,
                attempt_count=int(_record_attr(record, "attempt_count", 0) or 0),
                value_fingerprint=str(_record_attr(record, "value_fingerprint") or ""),
                evidence_fingerprint=str(_record_attr(record, "evidence_fingerprint") or ""),
                evidence_focus_ids=evidence_focus_ids,
            )
        )
        ticket_index += 1
    return tickets


def annotate_local_reviews(state: Any, profile_result: Dict[str, Any], records: List[ReviewRecord] | List[Dict[str, Any]]) -> None:
    record_map = {
        (str(_record_attr(record, "scope") or ""), str(_record_attr(record, "target_id") or "")): (
            record if isinstance(record, dict) else record.to_dict()
        )
        for record in records
    }
    if state.primary_reflection is not None:
        state.primary_reflection["review"] = record_map.get(("primary", "primary_decision"), {})
    for dossier in state.relationship_dossiers or []:
        dossier.review = record_map.get(("relationship", dossier.person_id), {})
    for field_decision in profile_result.get("field_decisions", []):
        field_decision["review"] = record_map.get(("profile_field", field_decision.get("field_key", "")), {})


def build_critical_review_summary(
    records: List[ReviewRecord] | List[Dict[str, Any]],
    tickets: List[CriticalReviewTicket] | List[Dict[str, Any]],
) -> Dict[str, Any]:
    status_counts: Dict[str, int] = {}
    for record in records:
        review_status = str(_record_attr(record, "review_status") or "")
        status_counts[review_status] = status_counts.get(review_status, 0) + 1
    resolved_count = status_counts.get(ReviewState.RESOLVED.value, 0)
    terminal_count = sum(count for state, count in status_counts.items() if state in TERMINAL_STATES)
    unresolved_count = status_counts.get(ReviewState.QUEUED.value, 0) + status_counts.get(ReviewState.RUNNING.value, 0)
    return {
        "record_count": len(records),
        "ticket_count": len(tickets),
        "status_counts": status_counts,
        "ticket_targets": [str(_record_attr(ticket, "target_id") or "") for ticket in tickets],
        "resolved_count": resolved_count,
        "terminal_count": terminal_count,
        "unresolved_count": unresolved_count,
    }


def transition_review_state(record: Dict[str, Any], next_state: str, reason: str) -> Dict[str, Any]:
    prev_state = str(record.get("review_status", ""))
    legal_transitions = {
        ReviewState.QUEUED.value: {
            ReviewState.RUNNING.value,
            ReviewState.TERMINAL_NON_REPAIRABLE.value,
            ReviewState.TERMINAL_MAX_ROUNDS.value,
            ReviewState.TERMINAL_FAILED.value,
        },
        ReviewState.RUNNING.value: {
            ReviewState.QUEUED.value,
            ReviewState.RESOLVED.value,
            ReviewState.TERMINAL_NO_CHANGE.value,
            ReviewState.TERMINAL_NON_REPAIRABLE.value,
            ReviewState.TERMINAL_MAX_ROUNDS.value,
            ReviewState.TERMINAL_FAILED.value,
        },
    }
    if prev_state in legal_transitions and next_state not in legal_transitions[prev_state]:
        record.setdefault("transition_warnings", []).append(
            {
                "from": prev_state,
                "to": next_state,
                "reason": f"illegal_transition:{reason}",
            }
        )
    record["review_status"] = next_state
    record["terminal"] = next_state in TERMINAL_STATES
    record.setdefault("state_history", []).append({"from": prev_state, "to": next_state, "reason": reason})
    return record


def classify_change(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    value_changed = before.get("value_fingerprint") != after.get("value_fingerprint")
    evidence_changed_only = not value_changed and before.get("evidence_fingerprint") != after.get("evidence_fingerprint")
    unchanged = not value_changed and not evidence_changed_only
    return {
        "value_changed": value_changed,
        "evidence_changed_only": evidence_changed_only,
        "unchanged": unchanged,
    }


def capture_ticket_snapshot(
    *,
    state: Any,
    profile_result: Dict[str, Any],
    scope: str,
    target_id: str,
) -> Dict[str, Any]:
    if scope == "primary":
        decision = state.primary_decision or {}
        evidence = decision.get("evidence", {}) or {}
        focus_ids = _build_evidence_focus_ids([], evidence)
        decision_status = decision.get("mode", "unknown")
        value_snapshot = {
            "mode": decision.get("mode"),
            "primary_person_id": decision.get("primary_person_id"),
        }
        confidence = float(decision.get("confidence", 0.0) or 0.0)
    elif scope == "relationship":
        dossier = next((item for item in state.relationship_dossiers or [] if item.person_id == target_id), None)
        if dossier is None:
            evidence = {}
            focus_ids = []
            decision_status = "missing"
            value_snapshot = None
            confidence = 0.0
        else:
            evidence = _extract_ids_from_evidence_refs(dossier.evidence_refs or [])
            focus_ids = _build_evidence_focus_ids([dossier.retention_reason] if dossier.retention_reason else [], evidence)
            decision_status = dossier.retention_decision
            result = dossier.relationship_result or {}
            value_snapshot = {
                "relationship_type": result.get("relationship_type"),
                "status": result.get("status"),
            }
            confidence = float(result.get("confidence", 0.0) or 0.0)
    else:
        decision = _get_field_decision(profile_result, target_id)
        final = (decision or {}).get("final", {})
        evidence = (final.get("evidence", {}) or {})
        questioned = ((decision or {}).get("review", {}) or {}).get("questioned_evidence_ids", []) or []
        focus_ids = _build_evidence_focus_ids(questioned, evidence)
        decision_status = "non_null" if final.get("value") is not None else "null"
        value_snapshot = final.get("value")
        confidence = float(final.get("confidence", 0.0) or 0.0)

    value_fingerprint = _value_fingerprint(value_snapshot, confidence, decision_status)
    evidence_fingerprint = _evidence_fingerprint(focus_ids)
    return {
        "decision_status": decision_status,
        "value": value_snapshot,
        "confidence": confidence,
        "evidence_focus_ids": focus_ids,
        "value_fingerprint": value_fingerprint,
        "evidence_fingerprint": evidence_fingerprint,
    }


def apply_trace_cleanup(
    *,
    profile_result: Dict[str, Any],
    target_id: str,
) -> Tuple[CleanupRecord | None, bool, str | None]:
    if target_id not in TRACE_CLEANUP_FIELD_WHITELIST:
        return None, False, "trace_cleanup_not_whitelisted"

    decision = _get_field_decision(profile_result, target_id)
    if not decision:
        return None, False, "trace_cleanup_field_missing"

    final = decision.get("final", {})
    before_metadata = {
        "under_generation_reason": final.get("under_generation_reason"),
        "should_retry_generation": final.get("should_retry_generation"),
        "missing_extractor_type": final.get("missing_extractor_type"),
    }
    before_value = final.get("value")
    cleaned_fields: List[str] = []

    for key in ("under_generation_reason", "should_retry_generation", "missing_extractor_type"):
        if key in final:
            cleaned_fields.append(key)
            final.pop(key, None)

    after_metadata = {
        "under_generation_reason": final.get("under_generation_reason"),
        "should_retry_generation": final.get("should_retry_generation"),
        "missing_extractor_type": final.get("missing_extractor_type"),
    }
    after_value = final.get("value")

    cleanup_record = CleanupRecord(
        target_id=target_id,
        cleaned_fields=cleaned_fields,
        before_value=before_value,
        after_value=after_value,
        before_metadata=before_metadata,
        after_metadata=after_metadata,
    )
    changed = bool(cleaned_fields)
    return cleanup_record, changed, None


def reason_to_repair_mode(reason_code: str) -> str:
    if reason_code in REASON_REPAIR_MODE:
        return REASON_REPAIR_MODE[reason_code]
    return "orchestrator_fixable"


def is_unclassified_reason_code(reason_code: str) -> bool:
    return bool(reason_code) and reason_code not in REASON_REPAIR_MODE


def should_attempt_reference_repair(record: Dict[str, Any], ticket: Dict[str, Any]) -> bool:
    repair_mode = str(record.get("repair_mode", "") or "")
    if repair_mode == "orchestrator_fixable":
        return True
    if repair_mode != "strategy_required":
        return False

    evidence_focus_ids = list(record.get("evidence_focus_ids", []) or [])
    questioned_ids = list(record.get("questioned_evidence_ids", []) or [])
    candidate_count = int(record.get("candidate_count", 0) or 0)
    strong_evidence_met = bool(record.get("strong_evidence_met"))
    evidence_ids = dict(record.get("evidence_ids", {}) or {})
    evidence_ref_count = (
        len(evidence_ids.get("event_ids", []) or [])
        + len(evidence_ids.get("photo_ids", []) or [])
        + len(evidence_ids.get("feature_names", []) or [])
    )
    ticket_evidence_focus = list(ticket.get("evidence_focus_ids", []) or [])
    return bool(
        candidate_count > 0
        or strong_evidence_met
        or evidence_ref_count > 0
        or evidence_focus_ids
        or questioned_ids
        or ticket_evidence_focus
    )


def build_conflict_evidence_package(
    *,
    record: Dict[str, Any],
    ticket: Dict[str, Any],
    before: Dict[str, Any],
    after: Dict[str, Any],
) -> Dict[str, Any]:
    questioned_ids = list(dict.fromkeys(
        str(item)
        for item in (
            list(record.get("questioned_evidence_ids", []) or [])
            + list(ticket.get("questioned_evidence_ids", []) or [])
        )
        if item
    ))
    record_focus = list(record.get("evidence_focus_ids", []) or [])
    before_focus = list(before.get("evidence_focus_ids", []) or [])
    after_focus = list(after.get("evidence_focus_ids", []) or [])
    focus_ids = list(dict.fromkeys(str(item) for item in (after_focus or before_focus or record_focus) if item))

    evidence_ids = dict(record.get("evidence_ids", {}) or {})
    evidence_ref_count = (
        len(evidence_ids.get("event_ids", []) or [])
        + len(evidence_ids.get("photo_ids", []) or [])
        + len(evidence_ids.get("person_ids", []) or [])
        + len(evidence_ids.get("group_ids", []) or [])
        + len(evidence_ids.get("feature_names", []) or [])
    )
    candidate_count = int(record.get("candidate_count", 0) or 0)
    strong_evidence_met = bool(record.get("strong_evidence_met"))
    evidence_changed = before.get("evidence_fingerprint") != after.get("evidence_fingerprint")

    score = 0
    if questioned_ids:
        score += 2
    if focus_ids:
        score += 1
    if evidence_ref_count > 0:
        score += 1
    if candidate_count > 0:
        score += 1
    if strong_evidence_met:
        score += 1
    if evidence_changed:
        score += 1

    return {
        "reason_code": str(record.get("reason_code") or ticket.get("reason_code") or ""),
        "questioned_ids": questioned_ids,
        "focus_ids": focus_ids,
        "evidence_ref_count": evidence_ref_count,
        "candidate_count": candidate_count,
        "strong_evidence_met": strong_evidence_met,
        "evidence_changed": evidence_changed,
        "score": score,
    }


def conflict_evidence_is_strong(
    package: Dict[str, Any],
    *,
    reason_code: str,
) -> bool:
    score = int(package.get("score", 0) or 0)
    questioned_ids = list(package.get("questioned_ids", []) or [])
    evidence_ref_count = int(package.get("evidence_ref_count", 0) or 0)
    candidate_count = int(package.get("candidate_count", 0) or 0)
    strong_evidence_met = bool(package.get("strong_evidence_met"))

    threshold = 2 if reason_code in RELAXED_CONFLICT_SCORE_REASON_CODES else 3
    has_support = bool(questioned_ids or evidence_ref_count > 0 or candidate_count > 0 or strong_evidence_met)
    return bool(score >= threshold and has_support)


def apply_profile_conflict_value_correction(
    *,
    profile_result: Dict[str, Any],
    target_id: str,
    reason_code: str,
    conflict_package: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    decision = _get_field_decision(profile_result, target_id)
    if not decision:
        return False, {"reason": "field_decision_not_found"}
    final = decision.get("final", {})
    if not isinstance(final, dict):
        return False, {"reason": "invalid_field_final_payload"}

    before_value = final.get("value")
    before_confidence = float(final.get("confidence", 0.0) or 0.0)
    changed = False
    corrected_value: Any = before_value
    corrected_confidence = before_confidence

    if reason_code == "candidate_noise_brand_tokens":
        corrected_value, changed = _sanitize_brand_values(before_value)
        corrected_confidence = 0.72 if corrected_value is not None else 0.0
    elif reason_code == "pseudo_place_candidates":
        corrected_value, changed = _sanitize_place_anchor_values(before_value)
        corrected_confidence = 0.68 if corrected_value is not None else 0.0
    elif reason_code == "value_generator_failed_after_candidates" and target_id == "long_term_facts.geography.mobility_pattern":
        corrected_value, changed = _derive_mobility_pattern_from_structured(profile_result)
        corrected_confidence = 0.6 if corrected_value is not None else 0.0
    else:
        return False, {"reason": "reason_code_not_supported_for_local_correction"}

    if not changed:
        return False, {"reason": "no_correctable_delta_detected"}

    final["value"] = corrected_value
    final["confidence"] = corrected_confidence
    evidence = final.setdefault("evidence", {})
    constraint_notes = list(evidence.get("constraint_notes", []) or [])
    constraint_notes.append(f"conflict_corrected:{reason_code}")
    constraint_notes.append(f"conflict_evidence_score:{conflict_package.get('score', 0)}")
    evidence["constraint_notes"] = list(dict.fromkeys(str(note) for note in constraint_notes if note))
    reasoning = str(final.get("reasoning", "") or "")
    correction_sentence = "回流裁决检测到强冲突证据包，已按可追溯证据完成本地改值校正。"
    final["reasoning"] = f"{reasoning} {correction_sentence}".strip()
    final.pop("under_generation_reason", None)
    final.pop("should_retry_generation", None)
    final.pop("missing_extractor_type", None)
    reflection_1 = decision.get("reflection_1", {})
    if isinstance(reflection_1, dict):
        reflection_1["under_generation_reason"] = None
        reflection_1["should_retry_generation"] = False
        reflection_1["missing_extractor_type"] = None
    reflection_2 = decision.get("reflection_2", {})
    if isinstance(reflection_2, dict) and reflection_2.get("null_reason"):
        reflection_2["null_reason"] = None

    return True, {
        "reason": "local_conflict_correction_applied",
        "before_value": before_value,
        "after_value": corrected_value,
        "before_confidence": before_confidence,
        "after_confidence": corrected_confidence,
    }


def _build_primary_review_record(state: Any, round_number: int, review_index: int) -> ReviewRecord:
    reflection = state.primary_reflection or {}
    decision = state.primary_decision or {}
    issues = list(reflection.get("issues", []) or [])
    reason_code = _map_primary_reason_code(reflection, decision)
    repairable = bool(reason_code and reason_code in PRIMARY_REPAIRABLE_REASON_CODES)
    review_status = ReviewState.QUEUED.value if repairable else ReviewState.PASS.value
    critical_type = "reflection_conflict" if repairable else "pass"
    evidence_ids = _normalize_evidence_ids(decision.get("evidence", {}))
    questioned = [issue for issue in issues if issue]
    focus_ids = _build_evidence_focus_ids(questioned, evidence_ids)
    confidence = float(decision.get("confidence", 0.0) or 0.0)
    value_snapshot = {
        "mode": decision.get("mode"),
        "primary_person_id": decision.get("primary_person_id"),
    }
    decision_status = decision.get("mode", "unknown")
    if reason_code and is_unclassified_reason_code(reason_code):
        issues.append("unclassified_reason_code")
    return ReviewRecord(
        review_id=f"REV_{review_index:04d}",
        scope="primary",
        target_id="primary_decision",
        current_status=decision_status,
        review_status=review_status,
        reparability="repairable" if repairable else "non_repairable",
        repair_mode=reason_to_repair_mode(reason_code or "no_primary_issue"),
        critical_type=critical_type,
        reason_code=reason_code or "no_primary_issue",
        reason_text="; ".join(issues) if issues else "primary decision passed review",
        candidate_family="primary_signal",
        candidate_count=len((reflection.get("primary_signal_trace", {}) or {}).get("candidate_signals", []) or []),
        strong_evidence_met=decision.get("primary_person_id") is not None,
        must_null=decision.get("mode") == "photographer_mode",
        issues_found=issues,
        evidence_ids=evidence_ids,
        questioned_evidence_ids=questioned,
        evidence_focus_ids=focus_ids,
        decision_status=decision_status,
        value_snapshot=value_snapshot,
        confidence_snapshot=confidence,
        value_fingerprint=_value_fingerprint(value_snapshot, confidence, decision_status),
        evidence_fingerprint=_evidence_fingerprint(focus_ids),
        dependency_effect="rerun_dependents" if repairable else "none",
        round=round_number,
        terminal=not repairable,
    )


def _build_relationship_review_record(dossier: Any, round_number: int, review_index: int) -> ReviewRecord:
    relationship_result = dict(dossier.relationship_result or {})
    issues = list((dossier.relationship_reflection or {}).get("issues", []) or [])
    reason_code, review_status, critical_type, reparability, terminal = _classify_relationship_review(dossier, relationship_result, issues)
    evidence_ids = _extract_ids_from_evidence_refs(dossier.evidence_refs or [])
    questioned = [reason_code] if reason_code else []
    focus_ids = _build_evidence_focus_ids(questioned, evidence_ids)
    confidence = float(relationship_result.get("confidence", 0.0) or 0.0)
    value_snapshot = {
        "relationship_type": relationship_result.get("relationship_type"),
        "status": relationship_result.get("status"),
    }
    decision_status = dossier.retention_decision
    if reason_code and is_unclassified_reason_code(reason_code):
        issues.append("unclassified_reason_code")
    return ReviewRecord(
        review_id=f"REV_{review_index:04d}",
        scope="relationship",
        target_id=dossier.person_id,
        current_status=decision_status,
        review_status=review_status,
        reparability=reparability,
        repair_mode=reason_to_repair_mode(reason_code),
        critical_type=critical_type,
        reason_code=reason_code,
        reason_text=dossier.retention_reason or "; ".join(issues) or "relationship passed review",
        candidate_family="relationship_dossier",
        candidate_count=len(dossier.shared_events or []),
        strong_evidence_met=bool(dossier.photo_count >= 2 or len(dossier.shared_events or []) >= 1),
        must_null=dossier.retention_decision != "keep",
        issues_found=issues,
        evidence_ids=evidence_ids,
        questioned_evidence_ids=questioned,
        evidence_focus_ids=focus_ids,
        decision_status=decision_status,
        value_snapshot=value_snapshot,
        confidence_snapshot=confidence,
        value_fingerprint=_value_fingerprint(value_snapshot, confidence, decision_status),
        evidence_fingerprint=_evidence_fingerprint(focus_ids),
        dependency_effect="rerun_dependents" if review_status == ReviewState.QUEUED.value else "none",
        round=round_number,
        terminal=terminal,
    )


def _build_profile_review_record(field_decision: Dict[str, Any], round_number: int, review_index: int) -> ReviewRecord:
    field_key = field_decision.get("field_key", "")
    gate_result = field_decision.get("gate_result", {}) or {}
    reflection_1 = field_decision.get("reflection_1", {}) or {}
    reflection_2 = field_decision.get("reflection_2", {}) or {}
    final = field_decision.get("final", {}) or {}
    evidence = final.get("evidence", {}) or {}
    issues = list(reflection_1.get("issues_found", []) or [])
    review_status, reparability, critical_type, reason_code, terminal = _classify_profile_review(
        field_key=field_key,
        gate_result=gate_result,
        reflection_1=reflection_1,
        reflection_2=reflection_2,
        final=final,
    )
    evidence_ids = _normalize_evidence_ids(evidence)
    questioned = _questioned_ids_for_profile(field_key, evidence, reason_code)
    focus_ids = _build_evidence_focus_ids(questioned, evidence_ids)
    decision_status = "non_null" if final.get("value") is not None else "null"
    confidence = float(final.get("confidence", 0.0) or 0.0)
    value_snapshot = final.get("value")
    if reason_code and is_unclassified_reason_code(reason_code):
        issues.append("unclassified_reason_code")
    return ReviewRecord(
        review_id=f"REV_{review_index:04d}",
        scope="profile_field",
        target_id=field_key,
        current_status=decision_status,
        review_status=review_status,
        reparability=reparability,
        repair_mode=reason_to_repair_mode(reason_code),
        critical_type=critical_type,
        reason_code=reason_code,
        reason_text=reflection_2.get("null_reason") or reflection_1.get("under_generation_reason") or "profile field reviewed",
        candidate_family=gate_result.get("candidate_family"),
        candidate_count=int(gate_result.get("candidate_count", 0) or 0),
        strong_evidence_met=bool(gate_result.get("strong_evidence_met")),
        must_null=bool(gate_result.get("must_null")),
        issues_found=issues,
        evidence_ids=evidence_ids,
        questioned_evidence_ids=questioned,
        evidence_focus_ids=focus_ids,
        decision_status=decision_status,
        value_snapshot=value_snapshot,
        confidence_snapshot=confidence,
        value_fingerprint=_value_fingerprint(value_snapshot, confidence, decision_status),
        evidence_fingerprint=_evidence_fingerprint(focus_ids),
        dependency_effect=_dependency_effect_for_field(field_key, review_status),
        round=round_number,
        terminal=terminal,
    )


def _classify_relationship_review(
    dossier: Any,
    relationship_result: Dict[str, Any],
    issues: List[str],
) -> tuple[str, str, str, str, bool]:
    confidence = float(relationship_result.get("confidence", 0.0) or 0.0)
    repeated_pattern = bool(
        dossier.photo_count >= 3
        or dossier.time_span_days >= 30
        or len(dossier.shared_events or []) >= 2
        or confidence >= 0.62
    )
    retention_reason = dossier.retention_reason or "relationship_retained"
    if dossier.retention_decision == "suppress":
        if retention_reason == "screen_or_virtual_context":
            if repeated_pattern:
                return "screen_or_virtual_context", ReviewState.QUEUED.value, "over_suppressed_result", "repairable", False
            return "screen_or_virtual_context", ReviewState.TERMINAL_SUPPRESS.value, "over_suppressed_result", "non_repairable", True
        if retention_reason == "single_photo_without_strong_signal":
            if repeated_pattern:
                return "single_photo_without_strong_signal", ReviewState.QUEUED.value, "over_suppressed_result", "repairable", False
            return "single_photo_without_strong_signal", ReviewState.TERMINAL_SUPPRESS.value, "over_suppressed_result", "non_repairable", True
        if retention_reason in {"low_confidence_low_signal_relationship", "over_suppressed_relationship", "group_eligibility_conflict"}:
            return retention_reason, ReviewState.QUEUED.value, "over_suppressed_result", "repairable", False
        return retention_reason or "suppressed_relationship", ReviewState.TERMINAL_SUPPRESS.value, "over_suppressed_result", "non_repairable", True
    if issues:
        return "relationship_type_over_upgrade", ReviewState.QUEUED.value, "reflection_conflict", "repairable", False
    return "relationship_pass", ReviewState.PASS.value, "pass", "non_repairable", False


def _classify_profile_review(
    *,
    field_key: str,
    gate_result: Dict[str, Any],
    reflection_1: Dict[str, Any],
    reflection_2: Dict[str, Any],
    final: Dict[str, Any],
) -> tuple[str, str, str, str, bool]:
    constraint_notes = set(final.get("evidence", {}).get("constraint_notes", []) or [])
    candidate_family = gate_result.get("candidate_family")
    candidate_count = int(gate_result.get("candidate_count", 0) or 0)
    if "silent_by_missing_social_media" in constraint_notes:
        return ReviewState.TERMINAL_NULL.value, "non_repairable", "missing_external_modality", "silent_by_missing_social_media", True
    if field_key in HARD_POLICY_BLOCK_FIELDS and final.get("value") is None:
        return ReviewState.TERMINAL_NULL.value, "non_repairable", "hard_policy_block", "hard_policy_block", True
    if final.get("value") is not None:
        if field_key == "long_term_facts.geography.location_anchors" and _has_pseudo_place_values(final.get("value")):
            return ReviewState.QUEUED.value, "repairable", "suspicious_non_null_value", "pseudo_place_candidates", False
        if field_key == "long_term_facts.material.brand_preference" and _has_brand_noise(final.get("value")):
            return ReviewState.QUEUED.value, "repairable", "suspicious_non_null_value", "candidate_noise_brand_tokens", False
        if field_key == "long_term_facts.geography.cross_border" and final.get("value") is True and not final.get("evidence", {}).get("event_ids"):
            return ReviewState.QUEUED.value, "repairable", "suspicious_non_null_value", "pseudo_place_candidates", False
        if reflection_1.get("under_generation_reason"):
            return ReviewState.QUEUED.value, "repairable", "trace_cleanup", "trace_cleanup", False
        return ReviewState.PASS.value, "non_repairable", "pass", "field_pass", False
    if reflection_1.get("under_generation_reason"):
        return ReviewState.QUEUED.value, "repairable", "under_generated_after_candidates", reflection_1.get("under_generation_reason"), False
    if gate_result.get("must_null") and candidate_count == 0:
        if gate_result.get("strong_evidence_met"):
            return ReviewState.QUEUED.value, "repairable", "gate_blocked", _missing_generator_reason(candidate_family), False
        if _looks_like_missing_generator(field_key, candidate_family):
            return ReviewState.QUEUED.value, "repairable", "gate_blocked", _missing_generator_reason(candidate_family), False
        return ReviewState.TERMINAL_NULL.value, "non_repairable", "gate_blocked", "no_evidence_in_allowed_sources", True
    if reflection_2.get("null_reason") == "null_due_to_expression_conflict_reflection":
        return ReviewState.QUEUED.value, "repairable", "reflection_conflict", "expression_conflict_needs_counter_evidence_recheck", False
    return ReviewState.PASS.value, "non_repairable", "pass", "field_pass", False


def _map_primary_reason_code(reflection: Dict[str, Any], decision: Dict[str, Any]) -> str | None:
    issues = set(reflection.get("issues", []) or [])
    if "primary_ambiguity" in issues or "ambiguous_top_candidates" in issues:
        return "primary_ambiguity"
    if "split_risk" in issues or "split_hypothesis_present" in issues:
        return "split_risk"
    if "photographer_mode_conflict" in issues:
        return "photographer_mode_conflict"
    if "other_photo_candidate_is_likely_a_photographed_subject" in issues:
        return "photographed_subject_veto_conflict"
    if "selected_candidate_low_value" in issues:
        return "selected_low_value_person"
    for issue in issues:
        if "non_real_person" in issue:
            return "selected_non_real_person"
    if decision.get("mode") == "photographer_mode" and not issues:
        return None
    if issues:
        return next(iter(issues))
    return None


def _looks_like_missing_generator(field_key: str, candidate_family: str | None) -> bool:
    if field_key.startswith(("long_term_expression.", "short_term_expression.")):
        return True
    if field_key.startswith("long_term_facts.social_identity."):
        return True
    return candidate_family in {"text_anchor", "place", "item", "time_pattern", "expression"}


def _missing_generator_reason(candidate_family: str | None) -> str:
    return {
        "text_anchor": "missing_text_anchor_generator",
        "item": "missing_item_candidate_generator",
        "place": "missing_place_candidate_generator",
        "time_pattern": "missing_time_pattern_generator",
        "expression": "missing_expression_candidate_generator",
    }.get(candidate_family, "missing_value_generator")


def _normalize_evidence_ids(evidence: Dict[str, Any]) -> Dict[str, List[str]]:
    return {
        "event_ids": [str(item) for item in evidence.get("event_ids", []) or []],
        "photo_ids": [str(item) for item in evidence.get("photo_ids", []) or []],
        "person_ids": [str(item) for item in evidence.get("person_ids", []) or []],
        "group_ids": [str(item) for item in evidence.get("group_ids", []) or []],
        "feature_names": [str(item) for item in evidence.get("feature_names", []) or []],
    }


def _extract_ids_from_evidence_refs(refs: Iterable[Dict[str, Any]]) -> Dict[str, List[str]]:
    evidence_ids = {
        "event_ids": [],
        "photo_ids": [],
        "person_ids": [],
        "group_ids": [],
        "feature_names": [],
    }
    for ref in refs:
        source_type = ref.get("source_type")
        source_id = ref.get("source_id")
        if not source_id:
            continue
        if source_type == "event":
            evidence_ids["event_ids"].append(str(source_id))
        elif source_type == "photo":
            evidence_ids["photo_ids"].append(str(source_id))
        elif source_type == "person":
            evidence_ids["person_ids"].append(str(source_id))
        elif source_type == "group":
            evidence_ids["group_ids"].append(str(source_id))
        elif source_type == "feature":
            evidence_ids["feature_names"].append(str(source_id))
    for key, values in evidence_ids.items():
        evidence_ids[key] = list(dict.fromkeys(values))
    return evidence_ids


def _questioned_ids_for_profile(field_key: str, evidence: Dict[str, Any], reason_code: str) -> List[str]:
    questioned: List[str] = []
    if reason_code in {"pseudo_place_candidates", "missing_place_candidate_generator"}:
        questioned.extend(f"event:{event_id}" for event_id in (evidence.get("event_ids", []) or [])[:2])
        questioned.extend(f"feature:{feature_name}" for feature_name in (evidence.get("feature_names", []) or [])[:2])
    elif reason_code in {"candidate_noise_brand_tokens", "missing_item_candidate_generator"}:
        questioned.extend(f"photo:{photo_id}" for photo_id in (evidence.get("photo_ids", []) or [])[:2])
    elif reason_code.startswith("missing_text_anchor"):
        questioned.extend(f"event:{event_id}" for event_id in (evidence.get("event_ids", []) or [])[:2])
        questioned.extend(f"photo:{photo_id}" for photo_id in (evidence.get("photo_ids", []) or [])[:2])
    elif reason_code.startswith("missing_time_pattern"):
        questioned.extend(f"event:{event_id}" for event_id in (evidence.get("event_ids", []) or [])[:3])
    else:
        questioned.extend(f"feature:{feature_name}" for feature_name in (evidence.get("feature_names", []) or [])[:2])
    return questioned


def _build_evidence_focus_ids(questioned_evidence_ids: List[str], evidence_ids: Dict[str, List[str]]) -> List[str]:
    if questioned_evidence_ids:
        return sorted(list(dict.fromkeys(str(item) for item in questioned_evidence_ids if item)))
    fallback: List[str] = []
    for event_id in evidence_ids.get("event_ids", [])[:3]:
        fallback.append(f"event:{event_id}")
    for photo_id in evidence_ids.get("photo_ids", [])[:3]:
        fallback.append(f"photo:{photo_id}")
    for feature_name in evidence_ids.get("feature_names", [])[:2]:
        fallback.append(f"feature:{feature_name}")
    return sorted(list(dict.fromkeys(fallback)))


def _dependency_effect_for_field(field_key: str, review_status: str) -> str:
    if review_status != ReviewState.QUEUED.value:
        return "none"
    if field_key.startswith(("long_term_facts.", "short_term_facts.")):
        return "rerun_dependents"
    return "none"


def _suggested_tools_for_reason(reason_code: str, scope: str, candidate_family: str | None) -> List[str]:
    if scope == "primary":
        return [
            "collect_primary_signal_bundle",
            "recompute_selfie_anchor_signals",
            "recompute_identity_anchor_signals",
            "recompute_other_photo_subjectness",
            "rerun_primary_reflection",
        ]
    if scope == "relationship":
        tool_map = {
            "screen_or_virtual_context": ["recollect_relationship_evidence", "recompute_mediated_scene_dominance", "rerun_retention"],
            "single_photo_without_strong_signal": ["recollect_relationship_evidence", "expand_shared_events", "rerun_retention"],
            "low_confidence_low_signal_relationship": ["recollect_relationship_evidence", "expand_shared_events", "rerun_relationship_type", "rerun_retention"],
            "over_suppressed_relationship": ["recollect_relationship_evidence", "expand_shared_events", "rerun_relationship_type", "rerun_retention"],
            "relationship_type_over_upgrade": ["recollect_relationship_evidence", "recompute_interaction_signals", "rerun_relationship_type"],
            "group_eligibility_conflict": ["recollect_relationship_evidence", "rerun_retention"],
        }
        return tool_map.get(reason_code, ["recollect_relationship_evidence", "rerun_retention"])
    tool_map = {
        "missing_text_anchor_generator": ["collect_field_evidence", "extract_text_anchor_candidates"],
        "missing_item_candidate_generator": ["collect_field_evidence", "extract_item_candidates", "resolve_ownership_and_subject_binding"],
        "missing_place_candidate_generator": ["collect_field_evidence", "extract_place_candidates", "compute_frequency_and_distribution"],
        "missing_time_pattern_generator": ["collect_field_evidence", "extract_time_pattern_candidates", "compute_frequency_and_distribution"],
        "missing_expression_candidate_generator": ["collect_field_evidence", "extract_expression_candidates", "find_counter_evidence"],
        "value_generator_failed_after_candidates": ["collect_field_evidence", f"retry_{candidate_family or 'field'}_generator"],
        "candidate_noise_brand_tokens": ["collect_field_evidence", "extract_item_candidates", "resolve_ownership_and_subject_binding"],
        "pseudo_place_candidates": ["collect_field_evidence", "extract_place_candidates", "find_counter_evidence"],
        "expression_conflict_needs_counter_evidence_recheck": ["collect_field_evidence", "extract_expression_candidates", "find_counter_evidence"],
        "trace_cleanup": ["apply_trace_cleanup"],
    }
    return tool_map.get(reason_code, ["collect_field_evidence"])


def _rerun_scope_for_reason(scope: str, reason_code: str) -> str:
    if scope == "primary":
        return "primary_only"
    if scope == "relationship":
        return "relationship_only"
    if reason_code == "trace_cleanup":
        return "profile_field_only"
    if reason_code in {
        "missing_text_anchor_generator",
        "missing_item_candidate_generator",
        "missing_place_candidate_generator",
        "missing_time_pattern_generator",
        "value_generator_failed_after_candidates",
    }:
        return "profile_field_with_dependents"
    return "profile_field_only"


def _downstream_impact_for_scope(scope: str) -> str:
    if scope == "primary":
        return "rerun_all_downstream"
    if scope == "relationship":
        return "rerun_profile"
    return "none"


def _stable_evidence_hash(target_id: str, reason_code: str, evidence_focus_ids: List[str]) -> str:
    flattened = [target_id, reason_code]
    flattened.extend(sorted(str(value) for value in (evidence_focus_ids or [])))
    return sha1("|".join(flattened).encode("utf-8")).hexdigest()


def _value_fingerprint(value_snapshot: Any, confidence_snapshot: float, decision_status: str) -> str:
    payload = {
        "value": value_snapshot,
        "confidence": round(float(confidence_snapshot or 0.0), 6),
        "decision_status": decision_status,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha1(encoded.encode("utf-8")).hexdigest()


def _evidence_fingerprint(evidence_focus_ids: List[str]) -> str:
    encoded = json.dumps(sorted(str(item) for item in (evidence_focus_ids or [])), ensure_ascii=False, separators=(",", ":"))
    return sha1(encoded.encode("utf-8")).hexdigest()


def _normalize_scalar_or_list(value: Any) -> List[str]:
    values = value if isinstance(value, list) else [value]
    normalized: List[str] = []
    for item in values:
        text = str(item).strip() if item is not None else ""
        if text:
            normalized.append(text)
    return normalized


def _dedupe_keep_order(values: List[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _sanitize_brand_values(value: Any) -> Tuple[Any, bool]:
    raw_values = _normalize_scalar_or_list(value)
    if not raw_values:
        return value, False
    has_kitty_clue = any("kitty" in item.lower() or "蝴蝶结" in item for item in raw_values)
    cleaned: List[str] = []
    for item in raw_values:
        lowered = item.lower()
        if lowered in BRAND_NOISE_TOKENS:
            continue
        if "当前面部状态" in item:
            continue
        if lowered == "hello" and has_kitty_clue:
            item = "Hello Kitty"
        cleaned.append(item)
    cleaned = _dedupe_keep_order(cleaned)
    if not cleaned:
        return None, value is not None
    if not isinstance(value, list) and len(cleaned) == 1:
        new_value: Any = cleaned[0]
    else:
        new_value = cleaned
    return new_value, new_value != value


def _sanitize_place_anchor_values(value: Any) -> Tuple[Any, bool]:
    raw_values = _normalize_scalar_or_list(value)
    if not raw_values:
        return value, False
    cleaned: List[str] = []
    for item in raw_values:
        if item in PSEUDO_PLACE_VALUES:
            continue
        if item.endswith("社区") and len(item) > 2:
            item = item[:-2]
        cleaned.append(item)
    cleaned = _dedupe_keep_order(cleaned)
    if not cleaned:
        return None, value is not None
    if not isinstance(value, list) and len(cleaned) == 1:
        new_value: Any = cleaned[0]
    else:
        new_value = cleaned
    return new_value, new_value != value


def _derive_mobility_pattern_from_structured(profile_result: Dict[str, Any]) -> Tuple[Any, bool]:
    structured = profile_result.get("structured", {}) or {}
    geography = ((structured.get("long_term_facts", {}) or {}).get("geography", {}) or {})
    location_anchors = (geography.get("location_anchors", {}) or {}).get("value")
    cross_border = (geography.get("cross_border", {}) or {}).get("value")
    anchors = _normalize_scalar_or_list(location_anchors)
    if not anchors:
        return None, False
    if bool(cross_border) and len(anchors) >= 2:
        return "cross_border_multi_anchor", True
    if len(anchors) >= 2:
        return "multi_anchor_city_pattern", True
    return "single_anchor_stable", True


def _has_pseudo_place_values(value: Any) -> bool:
    values = value if isinstance(value, list) else [value]
    lowered = [str(item).strip() for item in values if item is not None]
    return any(item in PSEUDO_PLACE_VALUES for item in lowered)


def _has_brand_noise(value: Any) -> bool:
    values = value if isinstance(value, list) else [value]
    lowered = [str(item).strip().lower() for item in values if item is not None]
    return any(item in BRAND_NOISE_TOKENS for item in lowered)


def _get_field_decision(profile_result: Dict[str, Any], field_key: str) -> Dict[str, Any] | None:
    for decision in profile_result.get("field_decisions", []):
        if isinstance(decision, dict) and decision.get("field_key") == field_key:
            return decision
    return None


def _record_attr(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)
