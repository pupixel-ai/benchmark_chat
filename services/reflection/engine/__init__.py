"""Shared reflection diagnostics engine.

Used by both Harness Engineering (upstream_triage / upstream_agent)
and Nightly Evolution (evolution.py) to provide unified field-level
diagnosis, signal extraction, patch planning, and evaluation.
"""

from services.reflection.engine.field_diagnostics import (
    FieldDiagnosis,
    diagnose_field,
    probe_coverage_gap,
)
from services.reflection.engine.signal_extractor import (
    SignalResult,
    extract_signals,
)
from services.reflection.engine.patch_planner import (
    PatchPlan,
    plan_patch,
)

__all__ = [
    "FieldDiagnosis",
    "diagnose_field",
    "probe_coverage_gap",
    "SignalResult",
    "extract_signals",
    "PatchPlan",
    "plan_patch",
]
