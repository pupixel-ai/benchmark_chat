"""Cross-user Harness Engineering: expert critic over aggregated bad cases.

Entry point: run_harness_engineering()

Architecture:
  1. Load all users' case_facts (from per-user run_reflection_task_generation output)
  2. Build cross-user patterns (same dimension+failure across multiple users)
  3. Run EngineeringCritic (LLM, O(M) calls, M = cross-user pattern count)
  4. Detect missing capabilities (rule-based + LLM)
  5. Output HarnessEngineeringReport
"""

from __future__ import annotations

import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from config import PROJECT_ROOT

from .harness_types import (
    CrossUserPattern,
    CriticReport,
    HarnessEngineeringReport,
    MissingCapability,
)
from .difficult_case_taxonomy import detect_diseases, DifficultDisease


REFLECTION_DIR = Path("memory") / "reflection"
EVOLUTION_DIR = Path("memory") / "evolution"


# ────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────


def run_harness_engineering(
    *,
    project_root: str = PROJECT_ROOT,
    user_names: List[str] | None = None,
) -> HarnessEngineeringReport:
    """Cross-user harness engineering: aggregate → cluster → critic → detect."""

    root = Path(project_root)
    resolved_users = user_names or _discover_users(root)
    if not resolved_users:
        return HarnessEngineeringReport(summary={"error": "no_users_found"})

    # 1. Load all users' case_facts
    all_case_facts: List[Dict[str, Any]] = []
    for user in resolved_users:
        facts = _load_case_facts(root, user)
        for f in facts:
            f["user_name"] = user
        all_case_facts.extend(facts)

    if not all_case_facts:
        return HarnessEngineeringReport(
            total_users=len(resolved_users),
            summary={"error": "no_case_facts"},
        )

    # 2. Cross-user clustering
    patterns = build_cross_user_patterns(all_case_facts, total_users=len(resolved_users))

    # 3. Missing capability detection (rule-based)
    rule_assets = _load_rule_assets_safe(root)
    missing = detect_missing_capabilities(patterns, rule_assets)

    # 4. Detect diseases (named difficult case taxonomy)
    diseases = detect_diseases(
        patterns=[p.to_dict() for p in patterns],
        missing_capabilities=[m.to_dict() for m in missing],
        all_case_facts=all_case_facts,
        total_users=len(resolved_users),
    )

    # 5. Build report
    report = HarnessEngineeringReport(
        total_users=len(resolved_users),
        total_case_facts=len(all_case_facts),
        cross_user_patterns=patterns,
        critic_reports=[],  # Phase 3b
        missing_capabilities=missing,
        summary=_build_summary(patterns, missing, resolved_users, diseases),
    )

    report.summary["diseases"] = [d.to_dict() for d in diseases]

    # 6. Persist
    output_path = root / REFLECTION_DIR / "harness_engineering_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    return report


# ────────────────────────────────────────────────────────────────────
# Cross-user pattern clustering
# ────────────────────────────────────────────────────────────────────


def _classify_lane(case: Dict[str, Any]) -> str:
    """Classify a case into protagonist / relationship / profile lane."""
    entity_type = str(case.get("entity_type") or "").strip()
    dimension = str(case.get("dimension") or "").strip()
    signal_source = str(case.get("signal_source") or "").strip()

    if entity_type == "primary_person" or "primary" in dimension.lower() or signal_source == "mainline_primary":
        return "protagonist"
    if entity_type == "relationship_candidate" or dimension.startswith("relationship:") or signal_source == "mainline_relationship":
        return "relationship"
    if entity_type == "profile_field" or dimension.startswith(("long_term_", "short_term_")) or signal_source == "mainline_profile":
        return "profile"
    # System/audit cases → skip
    if "system" in dimension or "audit" in dimension:
        return ""
    return "profile"  # default to profile


def _normalize_dimension(case: Dict[str, Any]) -> str:
    """Normalize dimension for clustering.

    - Profile fields: keep as-is (e.g. long_term_facts.identity.name)
    - Relationships: aggregate by relationship_type (e.g. "relationship_type:friend")
      instead of per-Person_ID
    - Primary: keep as-is
    """
    dimension = str(case.get("dimension") or "").strip()
    if dimension.startswith("relationship:"):
        # Use relationship_type from upstream_output or decision_trace
        rel_type = (
            str((case.get("upstream_output") or {}).get("relationship_type") or "").strip()
            or str((case.get("decision_trace") or {}).get("relationship_type") or "").strip()
        )
        if rel_type:
            return f"relationship_type:{rel_type}"
        return "relationship_type:unknown"
    return dimension


def _pattern_key(case: Dict[str, Any]) -> str:
    """Cluster key: normalized_dimension + failure_mode + root_cause_family.

    Does NOT include user_name → same problem across users clusters together.
    """
    dimension = _normalize_dimension(case)
    comp = case.get("comparison_result") or {}
    grade = str(comp.get("grade") or case.get("comparison_grade") or "").strip()

    # Infer failure_mode from grade
    output_value = comp.get("output_value")
    gt_value = comp.get("gt_value") or (case.get("gt_payload") or {}).get("gt_value")
    if grade == "missing_prediction":
        failure_mode = "missing_signal"
    elif grade == "partial_match":
        failure_mode = "partial_coverage"
    elif grade == "mismatch":
        if _is_empty(output_value) and not _is_empty(gt_value):
            failure_mode = "missing_signal"
        elif not _is_empty(output_value) and _is_empty(gt_value):
            failure_mode = "overclaim"
        else:
            failure_mode = "wrong_value"
    elif grade in {"exact_match", "close_match"}:
        failure_mode = "ok"
    else:
        failure_mode = "unknown"

    root_cause = str(case.get("root_cause_family") or "unknown").strip()
    return f"{dimension}|{failure_mode}|{root_cause}"


def build_cross_user_patterns(
    all_case_facts: List[Dict[str, Any]],
    *,
    total_users: int,
) -> List[CrossUserPattern]:
    """Cluster case_facts across users by (dimension, failure_mode, root_cause)."""

    # Classify entity_type into lanes: protagonist / relationship / profile
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for case in all_case_facts:
        dimension = str(case.get("dimension") or "").strip()
        if not dimension:
            continue
        lane = _classify_lane(case)
        if not lane:
            continue
        key = f"{lane}|" + _pattern_key(case)
        if "|ok|" in key:
            continue
        groups[key].append(case)

    patterns: List[CrossUserPattern] = []
    for key, cases in groups.items():
        parts = key.split("|")
        # key format: lane|dimension|failure_mode|root_cause
        lane = parts[0] if len(parts) > 0 else "profile"
        dimension = parts[1] if len(parts) > 1 else ""
        failure_mode = parts[2] if len(parts) > 2 else ""
        root_cause = parts[3] if len(parts) > 3 else ""

        users_in_pattern = list(set(str(c.get("user_name", "")) for c in cases))
        per_user: Dict[str, List[Dict]] = defaultdict(list)
        for c in cases:
            per_user[str(c.get("user_name", ""))].append(c)

        # Consistency: do all users have the same root_cause for this dimension?
        root_causes_per_user = [
            str(c.get("root_cause_family", ""))
            for c in cases if str(c.get("root_cause_family", ""))
        ]
        rc_counter = Counter(root_causes_per_user)
        top_rc, top_count = rc_counter.most_common(1)[0] if rc_counter else ("", 0)
        consistency = top_count / len(root_causes_per_user) if root_causes_per_user else 0.0

        all_rcs = list(set(root_causes_per_user))
        all_fix_surfaces = list(set(
            str(c.get("tool_usage_summary", {}).get("recommended_fix_surface", ""))
            for c in cases if str(c.get("tool_usage_summary", {}).get("recommended_fix_surface", ""))
        ))

        confidences = [
            float(c.get("fix_surface_confidence") or 0)
            for c in cases if c.get("fix_surface_confidence")
        ]

        pattern_id = hashlib.sha256(key.encode()).hexdigest()[:12]

        patterns.append(CrossUserPattern(
            pattern_id=f"xup_{pattern_id}",
            lane=lane,
            dimension=dimension,
            failure_mode=failure_mode,
            root_cause_family=root_cause,
            affected_users=sorted(users_in_pattern),
            user_coverage=len(users_in_pattern) / total_users if total_users > 0 else 0.0,
            total_case_count=len(cases),
            per_user_examples=dict(per_user),
            cross_user_consistency=round(consistency, 3),
            root_cause_candidates=all_rcs,
            fix_surface_candidates=all_fix_surfaces,
            avg_confidence=round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
            summary=f"{dimension} 在 {len(users_in_pattern)}/{total_users} 个用户中出现 {failure_mode}，主要根因 {root_cause}",
        ))

    # Sort by user_coverage desc, then total_case_count desc
    patterns.sort(key=lambda p: (-p.user_coverage, -p.total_case_count))
    return patterns


# ────────────────────────────────────────────────────────────────────
# Missing capability detection (rule-based)
# ────────────────────────────────────────────────────────────────────


def detect_missing_capabilities(
    patterns: List[CrossUserPattern],
    rule_assets: Dict[str, Any],
) -> List[MissingCapability]:
    """Detect capabilities the system is missing based on cross-user patterns."""

    missing: List[MissingCapability] = []

    # Detection 1: Field group blanks — same domain empty across >50% users
    # Only check profile fields (not individual relationship persons)
    domain_user_empty: Dict[str, set] = defaultdict(set)
    domain_user_total: Dict[str, set] = defaultdict(set)
    for p in patterns:
        if p.lane != "profile":
            continue
        domain = p.dimension.rsplit(".", 1)[0] if "." in p.dimension else p.dimension
        for u in p.affected_users:
            domain_user_total[domain].add(u)
            if p.failure_mode in ("missing_signal", "unknown"):
                domain_user_empty[domain].add(u)

    for domain, empty_users in domain_user_empty.items():
        total = domain_user_total.get(domain, set())
        if len(total) > 0 and len(empty_users) / len(total) > 0.5:
            missing.append(MissingCapability(
                capability_type="new_data_source",
                description=f"字段域 {domain} 在 {len(empty_users)}/{len(total)} 个用户中全部为空，可能缺少该域的数据源",
                affected_users=sorted(empty_users),
                affected_fields=[p.dimension for p in patterns if p.dimension.startswith(domain)],
                confidence=0.75,
            ))

    # Detection 2: Repeated fix_surface failures
    surface_pattern_count: Dict[str, List[str]] = defaultdict(list)
    for p in patterns:
        for fs in p.fix_surface_candidates:
            surface_pattern_count[fs].append(p.pattern_id)
    for surface, pids in surface_pattern_count.items():
        if len(pids) >= 3 and surface not in ("watch_only", "engineering_issue", ""):
            missing.append(MissingCapability(
                capability_type="rule_redesign",
                description=f"修复面 {surface} 在 {len(pids)} 个 pattern 中被反复推荐，可能该修复路径本身不够",
                affected_patterns=pids,
                confidence=0.65,
            ))

    # Detection 3: Coverage blind spots from CoverageProbe results (profile only)
    coverage_gap_types: Dict[str, List[str]] = defaultdict(list)
    for p in patterns:
        if p.lane != "profile":
            continue
        for user, cases in p.per_user_examples.items():
            for c in cases:
                gap = (c.get("tool_usage_summary") or {}).get("coverage_gap") or {}
                if gap.get("has_gap"):
                    coverage_gap_types[gap.get("gap_type", "")].append(p.dimension)

    for gap_type, fields in coverage_gap_types.items():
        if len(fields) >= 2:
            unique_fields = sorted(set(fields))
            missing.append(MissingCapability(
                capability_type="new_tool" if gap_type == "source_unconfigured" else "rule_redesign",
                description=f"结构性缺口 {gap_type} 影响 {len(unique_fields)} 个字段：{', '.join(unique_fields[:5])}",
                affected_fields=unique_fields,
                confidence=0.70,
            ))

    return missing


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────


def _discover_users(root: Path) -> List[str]:
    """Auto-discover users from case_facts files."""
    users: set[str] = set()
    reflection_dir = root / REFLECTION_DIR
    if reflection_dir.exists():
        for f in reflection_dir.glob("case_facts_*.jsonl"):
            name = f.stem.replace("case_facts_", "")
            if name:
                users.add(name)
    return sorted(users)


def _load_case_facts(root: Path, user_name: str) -> List[Dict[str, Any]]:
    path = root / REFLECTION_DIR / f"case_facts_{user_name}.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def _load_rule_assets_safe(root: Path) -> Dict[str, Any]:
    assets: Dict[str, Any] = {}
    for name in ("call_policies", "tool_rules", "field_spec_overrides"):
        path = root / "services" / "memory_pipeline" / "rule_assets" / f"{name}.json"
        if path.exists():
            try:
                assets[name] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                assets[name] = {}
        else:
            assets[name] = {}
    return assets


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def _build_summary(
    patterns: List[CrossUserPattern],
    missing: List[MissingCapability],
    users: List[str],
    diseases: List[DifficultDisease] | None = None,
) -> Dict[str, Any]:
    multi_user = [p for p in patterns if len(p.affected_users) > 1]
    by_lane: Dict[str, int] = {}
    for p in patterns:
        by_lane[p.lane] = by_lane.get(p.lane, 0) + 1
    disease_by_lane: Dict[str, int] = {}
    for d in (diseases or []):
        disease_by_lane[d.lane] = disease_by_lane.get(d.lane, 0) + 1
    return {
        "total_users": len(users),
        "total_patterns": len(patterns),
        "multi_user_patterns": len(multi_user),
        "single_user_patterns": len(patterns) - len(multi_user),
        "missing_capabilities_count": len(missing),
        "total_diseases": len(diseases or []),
        "diseases_by_lane": disease_by_lane,
        "by_lane": by_lane,
        "top_affected_dimensions": [
            {"dimension": p.dimension, "lane": p.lane, "user_coverage": p.user_coverage, "case_count": p.total_case_count}
            for p in patterns[:10]
        ],
    }
