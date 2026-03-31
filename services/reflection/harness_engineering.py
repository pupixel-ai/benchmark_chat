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
import os
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
from .difficult_case_extraction import extract_difficult_cases_from_critics
from .engineering_critic import EngineeringCritic


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

    # 1. Load all users' data sources
    all_case_facts: List[Dict[str, Any]] = []
    gt_by_user: Dict[str, List[Dict[str, Any]]] = {}
    proposals_by_user: Dict[str, List[Dict[str, Any]]] = {}
    states_by_user: Dict[str, Dict[str, Any]] = {}

    for user in resolved_users:
        facts = _load_case_facts(root, user)
        for f in facts:
            f["user_name"] = user
        all_case_facts.extend(facts)
        gt_by_user[user] = _load_gt_comparisons(root, user)
        proposals_by_user[user] = _load_all_proposals(root, user)
        states_by_user[user] = _load_field_loop_state(root, user)

    actions = _load_proposal_actions(root)
    actions_by_id: Dict[str, Dict[str, Any]] = {}
    for a in actions:
        pid = a.get("proposal_id", "")
        if pid:
            actions_by_id[pid] = a  # last action wins

    # Load human grade overrides (fallback for cases not yet synced to files)
    overrides = _load_gt_grade_overrides(root)

    # If no case_facts AND no gt_comparisons, nothing to analyze
    has_gt = any(len(v) > 0 for v in gt_by_user.values())
    if not all_case_facts and not has_gt:
        return HarnessEngineeringReport(
            total_users=len(resolved_users),
            summary={"error": "no_data"},
        )

    # 2. Cross-user clustering (GT-grounded, human-override-aware)
    patterns = build_cross_user_patterns(
        all_case_facts,
        total_users=len(resolved_users),
        gt_by_user=gt_by_user,
        proposals_by_user=proposals_by_user,
        states_by_user=states_by_user,
        actions_by_id=actions_by_id,
        grade_overrides=overrides,
    )

    # 3. Missing capability detection (rule-based)
    rule_assets = _load_rule_assets_safe(root)
    missing = detect_missing_capabilities(patterns, rule_assets)

    # 4. Run EngineeringCritic on top patterns (O(M) LLM calls) — BEFORE difficult case extraction
    critic_reports = _run_critic_on_patterns(
        patterns=patterns,
        total_users=len(resolved_users),
        rule_assets=rule_assets,
    )

    # 5. Extract difficult cases from Critic results (LLM signal-driven, replaces old detect_diseases)
    difficult_cases = extract_difficult_cases_from_critics(
        critic_reports=critic_reports,
        patterns=patterns,
    )

    # 6. Build report
    report = HarnessEngineeringReport(
        total_users=len(resolved_users),
        total_case_facts=len(all_case_facts),
        cross_user_patterns=patterns,
        critic_reports=critic_reports,
        missing_capabilities=missing,
        summary=_build_summary(patterns, missing, resolved_users, difficult_cases=None),
    )

    report.summary["difficult_cases"] = difficult_cases

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

    root_cause = str(case.get("root_cause_family") or "").strip()
    if not root_cause or root_cause == "unknown":
        # Auto-derive from failure_mode + tool_usage_summary
        tool_summary = case.get("tool_usage_summary") or {}
        tool_called = bool(tool_summary.get("tool_called"))
        evidence_count = int(tool_summary.get("retrieval_hit_count") or 0)
        if failure_mode in ("wrong_value", "partial_coverage"):
            root_cause = "field_reasoning" if tool_called and evidence_count >= 2 else "tool_retrieval"
        elif failure_mode == "missing_signal":
            root_cause = "field_reasoning" if tool_called and evidence_count >= 1 else "tool_selection_policy"
        elif failure_mode == "overclaim":
            root_cause = "field_reasoning"
        else:
            root_cause = "watch_only"
    return f"{dimension}|{failure_mode}|{root_cause}"


def build_cross_user_patterns(
    all_case_facts: List[Dict[str, Any]],
    *,
    total_users: int,
    gt_by_user: Dict[str, List[Dict[str, Any]]] | None = None,
    proposals_by_user: Dict[str, List[Dict[str, Any]]] | None = None,
    states_by_user: Dict[str, Dict[str, Any]] | None = None,
    actions_by_id: Dict[str, Dict[str, Any]] | None = None,
    grade_overrides: Dict[str, str] | None = None,
) -> List[CrossUserPattern]:
    """Cluster across users by field_key. GT-grounded: uses gt_comparisons as
    primary data source, case_facts as supplementary enrichment.
    Human grade overrides take precedence over automatic grades."""

    gt_by_user = gt_by_user or {}
    proposals_by_user = proposals_by_user or {}
    states_by_user = states_by_user or {}
    actions_by_id = actions_by_id or {}
    grade_overrides = grade_overrides or {}

    # ── Phase A: Build gt_index from gt_comparisons (primary) ──
    # Apply human overrides: gt_grade_overrides.json takes precedence
    gt_index: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for user, comps in gt_by_user.items():
        for rec in comps:
            fk = rec.get("field_key", "")
            if not fk or not fk.startswith(("long_term_", "short_term_")):
                continue
            cr = dict(rec.get("comparison_result") or {})
            case_id = rec.get("case_id", "")
            override_key = f"{case_id}_{user}"
            if override_key in grade_overrides:
                cr["original_grade"] = cr.get("grade", "")
                cr["grade"] = grade_overrides[override_key]
                cr["human_override"] = True
            rec = {**rec, "comparison_result": cr}
            gt_index[fk][user] = rec

    # ── Phase B: Build case_facts index for enrichment ──
    cf_index: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for case in all_case_facts:
        dim = str(case.get("dimension") or "").strip()
        user = str(case.get("user_name") or "").strip()
        if dim and user:
            cf_index[dim][user] = case

    # ── Phase C: Build proposals index by field_key ──
    proposals_by_field: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for user, props in proposals_by_user.items():
        for p in props:
            fk = p.get("field_key", "")
            if fk:
                proposals_by_field[fk].append(p)

    # ── Phase D: Cluster by field_key, collect bad-grade users ──
    _BAD_GRADES = {"mismatch", "missing_prediction", "partial_match"}
    patterns: List[CrossUserPattern] = []

    for field_key, user_comps in gt_index.items():
        bad_users: Dict[str, Dict[str, Any]] = {}
        all_users_for_field: List[str] = []
        for user, rec in user_comps.items():
            cr = rec.get("comparison_result") or {}
            grade = cr.get("grade", "")
            all_users_for_field.append(user)
            if grade in _BAD_GRADES:
                bad_users[user] = cr

        if not bad_users:
            continue

        # Derive dominant failure_mode from grades
        grade_counter = Counter(cr.get("grade", "") for cr in bad_users.values())
        dominant_grade = grade_counter.most_common(1)[0][0] if grade_counter else ""

        failure_mode = _grade_to_failure_mode(dominant_grade, bad_users)

        # Get root_cause from case_facts if available
        root_causes: List[str] = []
        per_user_examples: Dict[str, List[Dict]] = defaultdict(list)
        for user in bad_users:
            cf = cf_index.get(field_key, {}).get(user)
            if cf:
                rc = str(cf.get("root_cause_family") or "").strip()
                if rc and rc != "unknown":
                    root_causes.append(rc)
                per_user_examples[user].append(cf)

        rc_counter = Counter(root_causes)
        dominant_rc = rc_counter.most_common(1)[0][0] if rc_counter else "field_reasoning"
        consistency = (rc_counter.most_common(1)[0][1] / len(root_causes)) if root_causes else 0.0

        all_fix_surfaces = list(set(
            str(cf_index.get(field_key, {}).get(u, {}).get("tool_usage_summary", {}).get("recommended_fix_surface", ""))
            for u in bad_users
            if str(cf_index.get(field_key, {}).get(u, {}).get("tool_usage_summary", {}).get("recommended_fix_surface", ""))
        ))

        confidences = [
            float(cf_index.get(field_key, {}).get(u, {}).get("fix_surface_confidence") or 0)
            for u in bad_users
            if cf_index.get(field_key, {}).get(u, {}).get("fix_surface_confidence")
        ]

        key = f"profile|{field_key}|{failure_mode}|{dominant_rc}"
        pattern_id = hashlib.sha256(key.encode()).hexdigest()[:12]

        # ── Enrichment: GT comparisons ──
        gt_comps_enriched: Dict[str, Dict[str, Any]] = {}
        pre_loop_grades: Dict[str, str] = {}
        for user, cr in bad_users.items():
            gt_comps_enriched[user] = {
                "grade": cr.get("grade", ""),
                "score": cr.get("score", 0),
                "output_value": cr.get("output_value"),
                "gt_value": cr.get("gt_value"),
                "human_override": cr.get("human_override", False),
            }
            pre_loop_grades[user] = cr.get("grade", "")

        # ── Enrichment: proposals ──
        field_proposals = proposals_by_field.get(field_key, [])
        proposal_history: List[Dict[str, Any]] = []
        approved_count = 0
        for p in field_proposals:
            pid = p.get("proposal_id", "")
            action = actions_by_id.get(pid, {})
            status = action.get("action", p.get("status", "proposal_only"))
            proposal_history.append({
                "proposal_id": pid,
                "user_name": p.get("user_name", ""),
                "title": p.get("title", ""),
                "root_cause_family": p.get("root_cause_family", ""),
                "confidence": p.get("confidence", 0),
                "status": status,
                "patch_preview": p.get("patch_preview"),
            })
            if status == "approve":
                approved_count += 1

        # ── Enrichment: evolution state ──
        evo_state: Dict[str, Dict[str, Any]] = {}
        for user in bad_users:
            state = (states_by_user.get(user) or {}).get("fields", {}).get(field_key)
            if state:
                evo_state[user] = {
                    "cycle_count": state.get("cycle_count", 0),
                    "last_grade": state.get("last_grade", ""),
                    "last_status": state.get("last_status", ""),
                    "score_trend": state.get("score_trend", ""),
                }

        n_bad = len(bad_users)
        patterns.append(CrossUserPattern(
            pattern_id=f"xup_{pattern_id}",
            lane="profile",
            dimension=field_key,
            failure_mode=failure_mode,
            root_cause_family=dominant_rc,
            affected_users=sorted(bad_users.keys()),
            user_coverage=n_bad / total_users if total_users > 0 else 0.0,
            total_case_count=n_bad,
            per_user_examples=dict(per_user_examples),
            cross_user_consistency=round(consistency, 3),
            root_cause_candidates=list(set(root_causes)) or [dominant_rc],
            fix_surface_candidates=all_fix_surfaces,
            avg_confidence=round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
            avg_lane_quality={},
            summary=_build_pattern_summary(field_key, n_bad, total_users, failure_mode, dominant_rc, approved_count),
            gt_comparisons=gt_comps_enriched,
            pre_loop_grades=pre_loop_grades,
            proposal_history=proposal_history,
            approval_count=approved_count,
            evolution_state=evo_state,
        ))

    # Sort by user_coverage desc, then total_case_count desc
    patterns.sort(key=lambda p: (-p.user_coverage, -p.total_case_count))
    return patterns


def _grade_to_failure_mode(
    dominant_grade: str,
    bad_users: Dict[str, Dict[str, Any]],
) -> str:
    """Derive failure_mode from GT comparison grades."""
    if dominant_grade == "missing_prediction":
        return "missing_signal"
    if dominant_grade == "partial_match":
        return "partial_coverage"
    if dominant_grade == "mismatch":
        # Check if output is empty → missing_signal, or GT empty → overclaim
        empty_out = sum(1 for cr in bad_users.values() if _is_empty(cr.get("output_value")))
        if empty_out > len(bad_users) / 2:
            return "missing_signal"
        empty_gt = sum(1 for cr in bad_users.values() if _is_empty(cr.get("gt_value")))
        if empty_gt > len(bad_users) / 2:
            return "overclaim"
        return "wrong_value"
    return "unknown"


def _build_pattern_summary(
    field_key: str, n_bad: int, total: int,
    failure_mode: str, root_cause: str, approved_count: int,
) -> str:
    parts = [f"{field_key} 在 {n_bad}/{total} 个用户中出现 {failure_mode}"]
    if root_cause and root_cause != "watch_only":
        parts.append(f"主要根因 {root_cause}")
    if approved_count > 0:
        parts.append(f"已批准 {approved_count} 个修复方案")
    return "，".join(parts)


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


MAX_CRITIC_PATTERNS = int(os.environ.get("MAX_CRITIC_PATTERNS", "10"))


def _run_critic_on_patterns(
    *,
    patterns: List[CrossUserPattern],
    total_users: int,
    rule_assets: Dict[str, Any],
) -> List[CriticReport]:
    """Run EngineeringCritic on top-N patterns. Skip if no API key configured."""
    from config import ANTHROPIC_API_KEY, OPENROUTER_API_KEY

    if not ANTHROPIC_API_KEY and not OPENROUTER_API_KEY:
        print("[harness] No LLM API key configured, skipping Critic analysis")
        return []

    critic = EngineeringCritic()
    reports: List[CriticReport] = []

    # Only analyze top patterns (sorted by user_coverage desc already)
    top_patterns = patterns[:MAX_CRITIC_PATTERNS]
    # Skip patterns that are all "unknown" failure_mode with 0 confidence
    top_patterns = [p for p in top_patterns if p.failure_mode != "ok"]

    for i, pattern in enumerate(top_patterns):
        print(f"  [critic {i+1}/{len(top_patterns)}] {pattern.dimension} ({pattern.failure_mode})")
        report = critic.analyze_pattern(
            pattern=pattern.to_dict(),
            total_users=total_users,
            rule_assets=rule_assets,
        )
        reports.append(report)

    return reports


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


def _load_gt_comparisons(root: Path, user_name: str) -> List[Dict[str, Any]]:
    path = root / REFLECTION_DIR / f"gt_comparisons_{user_name}.jsonl"
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


def _load_all_proposals(root: Path, user_name: str) -> List[Dict[str, Any]]:
    """Load all evolution proposals across all cycles for a user."""
    proposals_dir = root / EVOLUTION_DIR / "proposals" / user_name
    if not proposals_dir.exists():
        return []
    all_proposals: List[Dict[str, Any]] = []
    seen_ids: set = set()
    for f in sorted(proposals_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        items = data if isinstance(data, list) else [data]
        for p in items:
            pid = p.get("proposal_id", "")
            if pid and pid not in seen_ids:
                seen_ids.add(pid)
                p.setdefault("user_name", user_name)
                all_proposals.append(p)
    return all_proposals


def _load_field_loop_state(root: Path, user_name: str) -> Dict[str, Any]:
    path = root / EVOLUTION_DIR / "field_loop_state" / f"{user_name}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _load_gt_grade_overrides(root: Path) -> Dict[str, str]:
    """Load human grade overrides: {case_id}_{user_name} → grade."""
    path = root / REFLECTION_DIR / "gt_grade_overrides.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _load_proposal_actions(root: Path) -> List[Dict[str, Any]]:
    path = root / EVOLUTION_DIR / "proposal_actions.jsonl"
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
    difficult_cases: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    multi_user = [p for p in patterns if len(p.affected_users) > 1]
    by_lane: Dict[str, int] = {}
    for p in patterns:
        by_lane[p.lane] = by_lane.get(p.lane, 0) + 1
    dc_by_type: Dict[str, int] = {}
    for dc in (difficult_cases or []):
        dt = dc.get("difficulty_type", "unknown")
        dc_by_type[dt] = dc_by_type.get(dt, 0) + 1
    return {
        "total_users": len(users),
        "total_patterns": len(patterns),
        "multi_user_patterns": len(multi_user),
        "single_user_patterns": len(patterns) - len(multi_user),
        "missing_capabilities_count": len(missing),
        "total_difficult_cases": len(difficult_cases or []),
        "difficult_cases_by_type": dc_by_type,
        "by_lane": by_lane,
        "top_affected_dimensions": [
            {"dimension": p.dimension, "lane": p.lane, "user_coverage": p.user_coverage, "case_count": p.total_case_count}
            for p in patterns[:10]
        ],
    }


# ────────────────────────────────────────────────────────────────────
# Field × User grid (GT-first, no pipeline dependency)
# ────────────────────────────────────────────────────────────────────


def build_field_cross_user_grid(
    *,
    project_root: str,
    field_spec_order: List[str] | None = None,
) -> Dict[str, Any]:
    """Build a field × user grid from GT files directly.

    Works for ALL users that have GT annotations, even those without
    pipeline runs. Returns {users, fields, cells, domain_accuracy}.
    Human grade overrides are applied.
    """
    root = Path(project_root)
    datasets_dir = root / "datasets"

    # Load human overrides
    overrides = _load_gt_grade_overrides(root)

    # Discover users with GT
    users: List[str] = []
    gt_values_by_user: Dict[str, Dict[str, str]] = {}
    if datasets_dir.exists():
        for d in sorted(datasets_dir.iterdir()):
            if not d.is_dir():
                continue
            gt_path = d / "gt" / "profile_field_gt.jsonl"
            if not gt_path.exists():
                continue
            user = d.name
            users.append(user)
            gt_vals: Dict[str, str] = {}
            for line in gt_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fk = rec.get("field_key", "")
                # Only profile fields (skip relationship Person_* etc.)
                if fk and fk.startswith(("long_term_", "short_term_")):
                    gt_vals[fk] = str(rec.get("gt_value", ""))
            gt_values_by_user[user] = gt_vals

    if not users:
        return {"users": [], "fields": [], "cells": [], "domain_accuracy": []}

    # Load gt_comparisons (with human overrides applied)
    comp_index: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for user in users:
        for rec in _load_gt_comparisons(root, user):
            fk = rec.get("field_key", "")
            if not fk:
                continue
            cr = dict(rec.get("comparison_result") or {})
            # Apply override
            case_id = rec.get("case_id", "")
            override_key = f"{case_id}_{user}"
            if override_key in overrides:
                cr["original_grade"] = cr.get("grade", "")
                cr["grade"] = overrides[override_key]
                cr["human_override"] = True
            comp_index[fk][user] = cr

    # Load field_loop_state
    state_index: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for user in users:
        state = _load_field_loop_state(root, user)
        if state:
            state_index[user] = state.get("fields", {})

    # Load proposal counts
    proposal_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for user in users:
        for p in _load_all_proposals(root, user):
            fk = p.get("field_key", "")
            if fk:
                proposal_counts[user][fk] += 1

    # Build fields list
    all_fields: set = set()
    for gt_vals in gt_values_by_user.values():
        all_fields.update(gt_vals.keys())
    if field_spec_order:
        all_fields.update(field_spec_order)
    ordered_fields = sorted(all_fields, key=lambda f: (
        field_spec_order.index(f) if field_spec_order and f in field_spec_order else 999,
        f,
    ))

    # Build cells
    _GOOD = {"exact_match", "close_match", "improved"}
    cells: List[Dict[str, Any]] = []
    # Accumulate per-field accuracy across users
    field_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "good": 0, "bad": 0, "pending": 0})
    for fk in ordered_fields:
        for user in users:
            gt_val = gt_values_by_user.get(user, {}).get(fk)
            comp = comp_index.get(fk, {}).get(user)
            state = (state_index.get(user) or {}).get(fk)
            pcnt = proposal_counts.get(user, {}).get(fk, 0)

            grade = comp.get("grade", "") if comp else ""
            human_override = comp.get("human_override", False) if comp else False

            cell: Dict[str, Any] = {
                "field_key": fk,
                "user_name": user,
                "has_gt": gt_val is not None,
                "gt_value": gt_val,
                "output_value": comp.get("output_value") if comp else None,
                "grade": grade,
                "score": comp.get("score", 0) if comp else 0,
                "human_override": human_override,
                "proposal_count": pcnt,
                "evolution_status": state.get("last_status", "") if state else "",
                "cycle_count": state.get("cycle_count", 0) if state else 0,
            }
            cells.append(cell)

            # Per-field accuracy accumulation
            if gt_val is not None and grade:
                field_stats[fk]["total"] += 1
                if grade in _GOOD:
                    field_stats[fk]["good"] += 1
                else:
                    field_stats[fk]["bad"] += 1
            elif gt_val is not None:
                field_stats[fk]["pending"] += 1

    # Build per-field accuracy table
    field_accuracy: List[Dict[str, Any]] = []
    for fk in ordered_fields:
        s = field_stats.get(fk)
        if not s:
            continue
        total = s["total"]
        good = s["good"]
        accuracy = round(good / total, 3) if total > 0 else 0.0
        field_accuracy.append({
            "field_key": fk,
            "total": total,
            "good": good,
            "bad": s["bad"],
            "pending": s["pending"],
            "accuracy": accuracy,
        })
    field_accuracy.sort(key=lambda d: (d["accuracy"], d["field_key"]))

    return {"users": users, "fields": ordered_fields, "cells": cells, "field_accuracy": field_accuracy}


# ────────────────────────────────────────────────────────────────────
# P4: 关系层 / 主角层结构化质量评分
# ────────────────────────────────────────────────────────────────────


def score_protagonist_quality(case: Dict[str, Any]) -> Dict[str, float]:
    """主角质量评分：evidence_sufficiency + mode_evidence_consistency。"""
    upstream = case.get("upstream_output") or {}
    decision_trace = case.get("decision_trace") or {}
    selfie_count = int(upstream.get("selfie_count") or decision_trace.get("selfie_count") or 0)
    anchor_count = int(upstream.get("identity_anchor_count") or decision_trace.get("identity_anchor_count") or 0)
    mode = str(upstream.get("mode") or decision_trace.get("mode") or "").strip()
    evidence_refs = case.get("evidence_refs") or []
    photo_refs = sum(1 for r in evidence_refs if str(r.get("source_type", "")).startswith("photo"))
    event_refs = sum(1 for r in evidence_refs if str(r.get("source_type", "")).startswith("event"))

    # evidence_sufficiency: 按证据强度分级
    if selfie_count > 0 or anchor_count > 0:
        sufficiency = 0.8 + min(0.2, (selfie_count + anchor_count) * 0.05)
    elif photo_refs > 0:
        sufficiency = min(0.6, photo_refs * 0.2)
    elif event_refs > 0:
        sufficiency = min(0.3, event_refs * 0.1)
    else:
        sufficiency = 0.0

    # mode_evidence_consistency
    if mode == "selfie" and selfie_count > 0:
        consistency = 1.0
    elif mode == "selfie" and selfie_count == 0:
        consistency = 0.2
    elif mode == "photographer_mode" and selfie_count == 0:
        consistency = 0.9
    elif mode == "photographer_mode" and selfie_count > 3:
        consistency = 0.2
    else:
        consistency = 0.5

    return {
        "evidence_sufficiency": round(min(1.0, sufficiency), 3),
        "mode_evidence_consistency": round(consistency, 3),
    }


def score_relationship_quality(case: Dict[str, Any]) -> Dict[str, float]:
    """关系质量评分：evidence_diversity + retention_justification + type_confidence_alignment。"""
    decision_trace = case.get("decision_trace") or {}
    upstream = case.get("upstream_output") or {}

    retention = str(decision_trace.get("retention_decision") or "").strip()
    if retention != "keep":
        return {}  # 只评 keep 的关系

    person_kind = str(decision_trace.get("person_kind") or "uncertain")
    rel_type = str(decision_trace.get("relationship_type") or upstream.get("relationship_type") or "")
    time_span = int(decision_trace.get("time_span_days") or 0)
    photo_count = int(decision_trace.get("photo_count") or 0)
    interaction_signals = list(decision_trace.get("interaction_signals") or [])
    scene_profile = decision_trace.get("scene_profile") or {}
    scenes = list(scene_profile.get("scenes") or [])

    strong_signals = {"kiss", "hug", "holding_hands", "selfie_together"}
    has_strong = bool(set(interaction_signals) & strong_signals)
    shared_events = list(upstream.get("shared_events") or decision_trace.get("shared_events") or [])

    # evidence_diversity: temporal_span_score + scene_diversity_score 取平均
    temporal_score = min(1.0, time_span / 180.0)  # 半年为满分
    scene_score = min(1.0, len(set(scenes)) / 4.0)  # 4 种场景为满分
    evidence_diversity = round((temporal_score + scene_score) / 2.0, 3)

    # retention_justification
    if person_kind not in ("real_person", "primary_contact", "secondary_contact", ""):
        justification = 0.2
    elif not interaction_signals and not shared_events:
        justification = 0.3
    elif len(set(scenes)) >= 2 and has_strong:
        justification = 1.0
    elif len(set(scenes)) >= 2 or has_strong:
        justification = 0.7
    else:
        justification = 0.5

    # type_confidence_alignment
    if rel_type in ("romantic", "family") and has_strong:
        alignment = 1.0
    elif rel_type in ("romantic", "family") and not has_strong and photo_count < 5:
        alignment = 0.3
    elif rel_type in ("acquaintance", "activity_buddy") and photo_count <= 3:
        alignment = 0.9
    elif rel_type in ("acquaintance", "activity_buddy") and has_strong:
        alignment = 0.3
    else:
        alignment = 0.6

    return {
        "evidence_diversity": evidence_diversity,
        "retention_justification": round(justification, 3),
        "type_confidence_alignment": round(alignment, 3),
    }


def _aggregate_lane_quality(cases: List[Dict[str, Any]], lane: str) -> Dict[str, float]:
    """聚合一组 cases 的平均 lane 质量分。"""
    if lane == "protagonist":
        scores = [score_protagonist_quality(c) for c in cases]
    elif lane == "relationship":
        scores = [score_relationship_quality(c) for c in cases]
        scores = [s for s in scores if s]  # 过滤掉非 keep 的空字典
    else:
        return {}

    if not scores:
        return {}

    all_keys = set()
    for s in scores:
        all_keys.update(s.keys())

    avg: Dict[str, float] = {}
    for key in all_keys:
        vals = [s[key] for s in scores if key in s]
        avg[key] = round(sum(vals) / len(vals), 3) if vals else 0.0
    return avg
