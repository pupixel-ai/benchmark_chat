"""Extract difficult cases from EngineeringCritic LLM analysis results.

Replaces the old rule-based difficult_case_taxonomy.py.
Difficult cases are identified by Critic signals:
- recommendations containing new_tool_needed / architecture_change
- low Critic confidence (< 0.4)
"""

from __future__ import annotations

from typing import Any, Dict, List

from .harness_types import CriticReport, CrossUserPattern

# Recommendation types that indicate the problem exceeds current system capabilities
_BLOCKING_REC_TYPES = {"new_tool_needed", "architecture_change"}


def extract_difficult_cases_from_critics(
    critic_reports: List[CriticReport],
    patterns: List[CrossUserPattern],
) -> List[Dict[str, Any]]:
    """Extract difficult cases from Critic analysis results.

    A pattern becomes a difficult case when the Critic determines that:
    1. It needs a new tool or architecture change (rule changes won't suffice), OR
    2. The Critic itself has low confidence in its analysis (< 0.4)
    """
    pattern_by_id = {p.pattern_id: p for p in patterns}
    cases: List[Dict[str, Any]] = []

    for report in critic_reports:
        if not report.pattern_id:
            continue

        pattern = pattern_by_id.get(report.pattern_id)
        if pattern is None:
            continue

        # Check for blocking recommendations
        blocked_recs = [
            r for r in (report.recommendations or [])
            if str(r.get("type") or "") in _BLOCKING_REC_TYPES
        ]

        # Determine difficulty type
        difficulty_type = _classify_difficulty(report, blocked_recs)
        if difficulty_type is None:
            continue

        # Parse root_cause_depth (can be str or dict from LLM)
        rcd = report.raw_llm_response.get("step2", {}).get("root_cause_depth", {})
        if isinstance(rcd, str):
            root_cause_surface = rcd
            root_cause_deep = ""
        elif isinstance(rcd, dict):
            root_cause_surface = str(rcd.get("surface") or "")
            root_cause_deep = str(rcd.get("deep") or "")
        else:
            root_cause_surface = ""
            root_cause_deep = ""

        cases.append({
            "pattern_id": report.pattern_id,
            "dimension": report.dimension,
            "lane": pattern.lane,
            "affected_users": pattern.affected_users,
            "case_count": pattern.total_case_count,
            # Critic LLM classification
            "difficulty_type": difficulty_type,
            "system_diagnosis": report.system_diagnosis,
            "root_cause_surface": root_cause_surface,
            "root_cause_deep": root_cause_deep,
            "blocked_recommendations": [
                {
                    "type": str(r.get("type") or ""),
                    "description": str(r.get("description") or ""),
                    "expected_impact": str(r.get("expected_impact") or ""),
                }
                for r in blocked_recs
            ],
            "critic_confidence": report.confidence,
        })

    return cases


def _classify_difficulty(
    report: CriticReport,
    blocked_recs: List[Dict[str, Any]],
) -> str | None:
    """Classify the difficulty type based on Critic signals.

    Returns None if the pattern is not a difficult case.
    """
    has_arch_change = any(
        str(r.get("type") or "") == "architecture_change" for r in blocked_recs
    )
    has_new_tool = any(
        str(r.get("type") or "") == "new_tool_needed" for r in blocked_recs
    )

    if has_arch_change:
        return "needs_architecture_change"
    if has_new_tool:
        return "needs_new_tool"
    if report.confidence < 0.4:
        return "critic_uncertain"

    return None
