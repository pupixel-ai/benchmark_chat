"""Unified patch planning.

Ported from evolution._propose_field_rule_patch + _infer_source_hints.
Generates override_bundle / patch_preview for a field based on diagnosis and signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from services.reflection.engine.field_diagnostics import FieldDiagnosis
from services.reflection.engine.signal_extractor import SignalResult


_SOURCE_HINTS_BY_FIELD_PREFIX = {
    "long_term_facts.identity.": ["vlm_observations"],
    "long_term_facts.social_identity.": ["event", "vlm_observations"],
    "long_term_facts.material.": ["event", "vlm_observations"],
    "long_term_facts.geography.": ["vlm_observations", "event"],
    "long_term_facts.time.": ["event"],
    "long_term_facts.relationships.": ["event", "vlm_observations"],
    "long_term_facts.hobbies.": ["event", "vlm_observations"],
    "long_term_facts.physiology.": ["vlm_observations"],
    "short_term_facts.": ["event", "vlm_observations"],
    "long_term_expression.": ["vlm_observations", "event"],
    "short_term_expression.": ["event", "vlm_observations"],
}


def _dedupe_non_empty(items: List[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        stripped = str(item).strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            result.append(stripped)
    return result


def _infer_source_hints(field_key: str) -> List[str]:
    for prefix, hints in _SOURCE_HINTS_BY_FIELD_PREFIX.items():
        if field_key.startswith(prefix):
            return list(hints)
    return ["event"]


@dataclass
class PatchPlan:
    patch_preview: Dict[str, Any] = field(default_factory=dict)
    override_bundle: Dict[str, Any] = field(default_factory=dict)
    proposal_type: str = "field_cycle_patch"

    @property
    def has_patch(self) -> bool:
        return bool(self.patch_preview)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patch_preview": self.patch_preview,
            "override_bundle": self.override_bundle,
            "proposal_type": self.proposal_type,
        }


def plan_patch(
    *,
    field_key: str,
    diagnosis: FieldDiagnosis,
    signals: SignalResult,
    current_assets: Dict[str, Dict[str, Any]],
) -> PatchPlan:
    """Generate a patch plan based on diagnosis and signals.

    Unified logic from:
    - evolution._propose_field_rule_patch (Nightly)
    - ExperimentPlanner._build_overlay_bundle (Harness)
    """
    patch_preview: Dict[str, Any] = {}
    proposal_type = "field_cycle_patch"
    failure_mode = diagnosis.failure_mode
    fix_surface = diagnosis.fix_surface

    call_policies = dict(current_assets.get("call_policies") or {})
    tool_rules = dict(current_assets.get("tool_rules") or {})
    field_overrides = dict(current_assets.get("field_spec_overrides") or {})

    # 1. call_policy patch: add sources for missing_signal / partial_coverage
    #    Also triggered when engine says fix_surface == "call_policy"
    if failure_mode in {"missing_signal", "partial_coverage"} or fix_surface == "call_policy":
        existing_call_policy = dict(call_policies.get(field_key) or {})
        source_hints = _infer_source_hints(field_key)
        if source_hints:
            current_sources = [
                str(item)
                for item in list(existing_call_policy.get("append_allowed_sources") or [])
                if str(item)
            ]
            next_sources = _dedupe_non_empty(current_sources + source_hints)
            if next_sources != current_sources:
                patch_preview.setdefault("call_policies", {})[field_key] = {
                    "append_allowed_sources": next_sources,
                }
                proposal_type = "call_policy_patch"

    # 2. tool_rule patch: adjust max_refs for wrong_value / overclaim / partial_coverage
    #    Also triggered when engine says fix_surface == "tool_rule"
    if failure_mode in {"wrong_value", "overclaim", "partial_coverage"} or fix_surface == "tool_rule":
        existing_tool_rule = dict(tool_rules.get(field_key) or {})
        current_max_refs = existing_tool_rule.get("max_refs_per_source")
        normalized_max_refs = int(current_max_refs) if isinstance(current_max_refs, int) else 0
        target_max_refs = 14 if failure_mode == "partial_coverage" else 10
        if normalized_max_refs <= 0 or normalized_max_refs > target_max_refs:
            patch_preview.setdefault("tool_rules", {})[field_key] = {
                "max_refs_per_source": target_max_refs,
            }
            if proposal_type == "field_cycle_patch":
                proposal_type = "tool_rule_patch"

    # 3. field_spec patch: add weak_evidence_caution hints from REAL evidence signals only.
    #    gt_token signals are convergence markers, not actionable evidence — skip them.
    real_signals = [s for s in signals.signals if not s.startswith("gt_token:")]
    if real_signals and failure_mode in {"wrong_value", "missing_signal", "partial_coverage"}:
        existing_field_override = dict(field_overrides.get(field_key) or {})
        current_null_pref = [
            str(item)
            for item in list(existing_field_override.get("weak_evidence_caution") or [])
            if str(item)
        ]
        clue_hint = f"字段循环线索：{real_signals[0]}"
        next_null_pref = _dedupe_non_empty(current_null_pref + [clue_hint])
        if next_null_pref != current_null_pref:
            patch_preview.setdefault("field_spec_overrides", {})[field_key] = {
                "weak_evidence_caution": next_null_pref,
            }
            if proposal_type == "field_cycle_patch":
                proposal_type = "field_spec_patch"

    # override_bundle mirrors patch_preview for now (Harness format)
    override_bundle = dict(patch_preview)

    return PatchPlan(
        patch_preview=patch_preview,
        override_bundle=override_bundle,
        proposal_type=proposal_type,
    )
