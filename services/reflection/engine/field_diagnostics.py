"""Unified field-level diagnostics.

Merges:
- Nightly's _infer_failure_mode (5 failure modes)
- Harness's CoverageProbe (4 structural gap types)
- Harness's UpstreamTriageScorer._heuristic_score (root_cause → fix_surface mapping)

All rule-based, zero LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


ALLOWED_ROOT_CAUSE_FAMILIES = (
    "field_reasoning",
    "evidence_packaging",
    "tool_retrieval",
    "tool_selection_policy",
    "orchestration_guardrail",
    "engineering_issue",
    "watch_only",
    "coverage_gap_source",
    "coverage_gap_tool",
)

ROOT_CAUSE_TO_FIX_SURFACE = {
    "field_reasoning": "field_cot",
    "evidence_packaging": "field_cot",
    "tool_retrieval": "tool_rule",
    "tool_selection_policy": "call_policy",
    "orchestration_guardrail": "engineering_issue",
    "engineering_issue": "engineering_issue",
    "watch_only": "watch_only",
    "coverage_gap_source": "call_policy",
    "coverage_gap_tool": "tool_rule",
}


@dataclass
class CoverageGap:
    has_gap: bool = False
    gap_type: str = "none"
    affected_sources: List[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_gap": self.has_gap,
            "gap_type": self.gap_type,
            "affected_sources": list(self.affected_sources),
            "detail": self.detail,
        }


@dataclass
class FieldDiagnosis:
    failure_mode: str  # missing_signal / wrong_value / overclaim / partial_coverage / monitoring / unknown
    root_cause_family: str  # from ALLOWED_ROOT_CAUSE_FAMILIES
    fix_surface: str  # field_cot / tool_rule / call_policy / watch_only / engineering_issue
    confidence: float
    coverage_gap: CoverageGap = field(default_factory=CoverageGap)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "failure_mode": self.failure_mode,
            "root_cause_family": self.root_cause_family,
            "fix_surface": self.fix_surface,
            "confidence": self.confidence,
            "coverage_gap": self.coverage_gap.to_dict(),
        }


def _is_value_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def infer_failure_mode(*, grade: str, output_value: Any, gt_value: Any) -> str:
    """Determine the failure mode from GT comparison grade and values.

    Ported from evolution._infer_failure_mode, unchanged semantics.
    """
    _RESOLVED_GRADES = {"exact_match", "close_match"}
    if grade == "missing_prediction":
        return "missing_signal"
    if grade == "partial_match":
        return "partial_coverage"
    if grade in {"mismatch", "missing_gt"}:
        if _is_value_empty(output_value) and not _is_value_empty(gt_value):
            return "missing_signal"
        if not _is_value_empty(output_value) and _is_value_empty(gt_value):
            return "overclaim"
        return "wrong_value"
    if grade in _RESOLVED_GRADES:
        return "monitoring"
    return "unknown"


def probe_coverage_gap(
    *,
    field_key: str,
    tool_trace: Dict[str, Any],
    allowed_sources: List[str],
    call_policies: Dict[str, Any],
    tool_rules: Dict[str, Any],
    group_tool_traces: List[Dict[str, Any]] | None = None,
) -> CoverageGap:
    """Detect structural gaps in the tool pipeline for a field.

    Ported from upstream_agent.CoverageProbe.probe(), identical logic.
    4 gap types: tool_rule_blocked, source_unconfigured, tool_called_no_hit, index_path_suspect.
    """
    evidence_bundle = dict(tool_trace.get("evidence_bundle") or {})
    tool_called = bool(tool_trace)

    gap_type = "none"
    affected_sources: List[str] = []
    detail = ""

    # Rule 1: tool_rule_blocked
    field_tool_rule = dict((tool_rules or {}).get(field_key) or {})
    if not allowed_sources:
        gap_type = "tool_rule_blocked"
        detail = "allowed_sources is empty"
    elif (
        field_tool_rule.get("max_refs_per_source") is not None
        and int(field_tool_rule["max_refs_per_source"]) == 0
    ):
        gap_type = "tool_rule_blocked"
        detail = f"tool_rules[{field_key}].max_refs_per_source == 0"

    # Rule 2 & 3: source_unconfigured / tool_called_no_hit
    if gap_type == "none" and tool_called:
        field_call_policy = dict((call_policies or {}).get(field_key) or {})
        appended_sources = list(field_call_policy.get("append_allowed_sources") or [])
        effective_sources = list(set(allowed_sources + appended_sources))

        for source in effective_sources:
            source_bundle = evidence_bundle.get(source)
            hit_count = 0
            if isinstance(source_bundle, list):
                hit_count = len(source_bundle)
            elif isinstance(source_bundle, dict):
                hit_count = int(source_bundle.get("hit_count") or 0)

            if hit_count > 0:
                continue

            if source not in appended_sources and field_key not in (call_policies or {}):
                gap_type = "source_unconfigured"
                affected_sources.append(source)
                detail = f"source '{source}' in allowed_sources but call_policy not configured for {field_key}"
                break

            if source in evidence_bundle:
                gap_type = "tool_called_no_hit"
                affected_sources.append(source)
                detail = f"source '{source}' called but hit_count == 0"

    # Rule 4: index_path_suspect
    if gap_type == "none" and group_tool_traces:
        this_total = sum(
            len(v) if isinstance(v, list) else int((v or {}).get("hit_count") or 0)
            for v in evidence_bundle.values()
        )
        if this_total == 0:
            group_totals = []
            for gt in group_tool_traces:
                gb = dict((gt or {}).get("evidence_bundle") or {})
                group_totals.append(
                    sum(
                        len(v) if isinstance(v, list) else int((v or {}).get("hit_count") or 0)
                        for v in gb.values()
                    )
                )
            if any(c > 0 for c in group_totals):
                gap_type = "index_path_suspect"
                detail = f"field {field_key} has 0 refs while same-group fields have evidence"

    return CoverageGap(
        has_gap=gap_type != "none",
        gap_type=gap_type,
        affected_sources=affected_sources,
        detail=detail,
    )


def _heuristic_root_cause(
    *,
    failure_mode: str,
    tool_called: bool,
    evidence_count: int,
    retrieval_hit_count: int,
) -> tuple[str, float]:
    """Map failure mode + evidence signals to root_cause_family + confidence.

    Merged from:
    - upstream_triage._heuristic_score (Harness, uses badcase_source)
    - Nightly failure_mode (simpler, no badcase_source)

    This version uses failure_mode as primary axis (works for both callers).
    """
    if failure_mode in {"wrong_value", "partial_coverage"}:
        if tool_called and evidence_count >= 2:
            return "field_reasoning", 0.9
        if tool_called and retrieval_hit_count == 0:
            return "tool_retrieval", 0.84
        if not tool_called:
            return "tool_selection_policy", 0.85
        return "evidence_packaging", 0.78

    if failure_mode == "missing_signal":
        if tool_called and evidence_count >= 1:
            return "field_reasoning", 0.82
        if tool_called and retrieval_hit_count == 0:
            return "tool_retrieval", 0.86
        if not tool_called:
            return "tool_selection_policy", 0.88
        return "evidence_packaging", 0.76

    if failure_mode == "overclaim":
        if tool_called and evidence_count >= 2:
            return "field_reasoning", 0.85
        return "evidence_packaging", 0.78

    if failure_mode == "monitoring":
        return "watch_only", 0.0

    return "watch_only", 0.45


def diagnose_field(
    *,
    field_key: str,
    output_value: Any,
    gt_value: Any,
    comparison_grade: str,
    tool_trace: Dict[str, Any] | None = None,
    allowed_sources: List[str] | None = None,
    call_policies: Dict[str, Any] | None = None,
    tool_rules: Dict[str, Any] | None = None,
    group_tool_traces: List[Dict[str, Any]] | None = None,
) -> FieldDiagnosis:
    """Unified field diagnosis: failure_mode + coverage_gap + root_cause + fix_surface.

    This is the single entry point for both Harness and Nightly callers.
    """
    failure_mode = infer_failure_mode(
        grade=comparison_grade,
        output_value=output_value,
        gt_value=gt_value,
    )

    coverage_gap = CoverageGap()
    if tool_trace and allowed_sources is not None:
        coverage_gap = probe_coverage_gap(
            field_key=field_key,
            tool_trace=tool_trace or {},
            allowed_sources=allowed_sources or [],
            call_policies=call_policies or {},
            tool_rules=tool_rules or {},
            group_tool_traces=group_tool_traces,
        )

    # Coverage gap takes priority for root_cause (same as Harness TriageScorer logic)
    if coverage_gap.has_gap:
        if coverage_gap.gap_type == "source_unconfigured":
            return FieldDiagnosis(
                failure_mode=failure_mode,
                root_cause_family="coverage_gap_source",
                fix_surface="call_policy",
                confidence=0.92,
                coverage_gap=coverage_gap,
            )
        else:
            return FieldDiagnosis(
                failure_mode=failure_mode,
                root_cause_family="coverage_gap_tool",
                fix_surface="tool_rule",
                confidence=0.90,
                coverage_gap=coverage_gap,
            )

    # Heuristic root cause from failure_mode + evidence signals
    evidence_bundle = dict((tool_trace or {}).get("evidence_bundle") or {})
    evidence_count = sum(
        len(v) if isinstance(v, list) else int((v or {}).get("hit_count") or 0)
        for v in evidence_bundle.values()
    )
    retrieval_hit_count = evidence_count
    tool_called = bool(tool_trace)

    root_cause, confidence = _heuristic_root_cause(
        failure_mode=failure_mode,
        tool_called=tool_called,
        evidence_count=evidence_count,
        retrieval_hit_count=retrieval_hit_count,
    )

    fix_surface = ROOT_CAUSE_TO_FIX_SURFACE.get(root_cause, "watch_only")

    return FieldDiagnosis(
        failure_mode=failure_mode,
        root_cause_family=root_cause,
        fix_surface=fix_surface,
        confidence=confidence,
        coverage_gap=coverage_gap,
    )
