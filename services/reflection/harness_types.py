"""Cross-user Harness Engineering data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class CrossUserPattern:
    """A pattern that appears across multiple users for the same field/dimension."""

    pattern_id: str
    lane: str = "profile"  # protagonist / relationship / profile
    dimension: str = ""  # field_key
    failure_mode: str = ""  # missing_signal / wrong_value / overclaim / partial_coverage
    root_cause_family: str = ""
    affected_users: List[str] = field(default_factory=list)
    user_coverage: float = 0.0  # affected_users / total_users
    total_case_count: int = 0
    per_user_examples: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    cross_user_consistency: float = 0.0  # how consistent is the failure across users
    root_cause_candidates: List[str] = field(default_factory=list)
    fix_surface_candidates: List[str] = field(default_factory=list)
    avg_confidence: float = 0.0
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "lane": self.lane,
            "dimension": self.dimension,
            "failure_mode": self.failure_mode,
            "root_cause_family": self.root_cause_family,
            "affected_users": self.affected_users,
            "user_coverage": self.user_coverage,
            "total_case_count": self.total_case_count,
            "per_user_examples": {
                u: [_compact_case(c) for c in cases[:2]]
                for u, cases in self.per_user_examples.items()
            },
            "cross_user_consistency": self.cross_user_consistency,
            "root_cause_candidates": self.root_cause_candidates,
            "fix_surface_candidates": self.fix_surface_candidates,
            "avg_confidence": self.avg_confidence,
            "summary": self.summary,
        }


@dataclass
class CriticReport:
    """Deep analysis of a cross-user pattern by EngineeringCritic."""

    pattern_id: str
    dimension: str
    system_diagnosis: str = ""  # Why this keeps failing at the system level
    root_cause_depth: str = ""  # Surface root_cause vs deeper architectural issue
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    referenced_users: List[str] = field(default_factory=list)
    referenced_evolution_context: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    raw_llm_response: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_id": self.pattern_id,
            "dimension": self.dimension,
            "system_diagnosis": self.system_diagnosis,
            "root_cause_depth": self.root_cause_depth,
            "recommendations": self.recommendations,
            "referenced_users": self.referenced_users,
            "referenced_evolution_context": self.referenced_evolution_context,
            "confidence": self.confidence,
        }


@dataclass
class MissingCapability:
    """A capability the system is missing, detected from cross-user patterns."""

    capability_type: str  # new_tool / new_data_source / new_agent_step / rule_redesign
    description: str
    affected_patterns: List[str] = field(default_factory=list)
    affected_users: List[str] = field(default_factory=list)
    affected_fields: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "capability_type": self.capability_type,
            "description": self.description,
            "affected_patterns": self.affected_patterns,
            "affected_users": self.affected_users,
            "affected_fields": self.affected_fields,
            "evidence": self.evidence,
            "confidence": self.confidence,
        }


@dataclass
class HarnessEngineeringReport:
    """Full output of a cross-user harness engineering run."""

    total_users: int = 0
    total_case_facts: int = 0
    cross_user_patterns: List[CrossUserPattern] = field(default_factory=list)
    critic_reports: List[CriticReport] = field(default_factory=list)
    missing_capabilities: List[MissingCapability] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_users": self.total_users,
            "total_case_facts": self.total_case_facts,
            "cross_user_patterns": [p.to_dict() for p in self.cross_user_patterns],
            "critic_reports": [r.to_dict() for r in self.critic_reports],
            "missing_capabilities": [m.to_dict() for m in self.missing_capabilities],
            "summary": self.summary,
        }


def _compact_case(case: Dict[str, Any]) -> Dict[str, Any]:
    """Compact a case_fact for pattern display (drop verbose fields)."""
    return {
        "case_id": case.get("case_id", ""),
        "user_name": case.get("user_name", ""),
        "dimension": case.get("dimension", ""),
        "comparison_grade": (case.get("comparison_result") or {}).get("grade", ""),
        "output_value": (case.get("comparison_result") or {}).get("output_value"),
        "gt_value": (case.get("comparison_result") or case.get("gt_payload") or {}).get("gt_value"),
        "root_cause_family": case.get("root_cause_family", ""),
        "resolution_route": case.get("resolution_route", ""),
        "agent_reasoning_summary": case.get("agent_reasoning_summary", ""),
    }
