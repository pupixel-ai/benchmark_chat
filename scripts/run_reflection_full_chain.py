#!/usr/bin/env python3
"""Full reflection automation chain.

Usage:
    python scripts/run_reflection_full_chain.py [--users user1,user2] [--date 2026-03-28] [--skip-nightly] [--skip-harness]

Chain:
    1. Nightly Evolution per user (GT comparison → field loop → proposals)
    2. Per-user Harness reflection (case_facts → triage → patterns → proposals)
    3. Cross-user Harness Engineering (aggregate → cross-user patterns → diseases)
    4. Print summary

Output:
    - memory/evolution/{user}/*.json  (nightly reports)
    - memory/reflection/*_{user}.jsonl (case_facts, patterns, proposals)
    - memory/reflection/harness_engineering_report.json (cross-user)
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import PROJECT_ROOT as CONFIG_ROOT


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full reflection chain")
    parser.add_argument("--users", type=str, default="", help="Comma-separated user names (empty=auto-discover)")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--top-k-fields", type=int, default=3)
    parser.add_argument("--skip-nightly", action="store_true")
    parser.add_argument("--skip-harness", action="store_true")
    args = parser.parse_args()

    user_names = [u.strip() for u in args.users.split(",") if u.strip()] if args.users else None

    print(f"=== Reflection Full Chain | date={args.date} ===")
    print()

    # ── Step 1: Nightly Evolution ──
    if not args.skip_nightly:
        from services.memory_pipeline.evolution import (
            run_memory_nightly_evaluation,
            run_memory_nightly_user_set_evaluation,
        )

        if user_names and len(user_names) == 1:
            print(f"[1/3] Nightly Evolution: {user_names[0]}")
            result = run_memory_nightly_evaluation(
                project_root=CONFIG_ROOT,
                user_name=user_names[0],
                date_str=args.date,
                top_k_fields=args.top_k_fields,
            )
            print(f"  traces={result.get('total_traces', 0)} insights={result.get('total_insights', 0)} proposals={result.get('total_proposals', 0)}")
        else:
            print(f"[1/3] Nightly Evolution: {'auto-discover' if not user_names else ','.join(user_names)}")
            result = run_memory_nightly_user_set_evaluation(
                project_root=CONFIG_ROOT,
                user_names=user_names or [],
                date_str=args.date,
                top_k_fields=args.top_k_fields,
            )
            print(f"  users={result.get('total_users', 0)}")
        print()
    else:
        print("[1/3] Nightly Evolution: SKIPPED")
        print()

    # ── Step 2: Per-user Harness Reflection ──
    if not args.skip_harness:
        from services.reflection.tasks import run_reflection_task_generation

        # Discover users from case_facts
        reflection_dir = Path(CONFIG_ROOT) / "memory" / "reflection"
        harness_users = user_names or []
        if not harness_users and reflection_dir.exists():
            harness_users = sorted(set(
                f.stem.replace("case_facts_", "")
                for f in reflection_dir.glob("case_facts_*.jsonl")
            ))

        print(f"[2/3] Per-user Harness: {len(harness_users)} users")
        for user in harness_users:
            try:
                result = run_reflection_task_generation(
                    project_root=CONFIG_ROOT,
                    user_name=user,
                )
                counts = {k: v for k, v in result.items() if isinstance(v, int)}
                print(f"  {user}: {counts}")
            except Exception as e:
                print(f"  {user}: ERROR {e}")
        print()

    # ── Step 3: Cross-user Harness Engineering ──
    print("[3/3] Cross-user Harness Engineering")
    from services.reflection.harness_engineering import run_harness_engineering
    report = run_harness_engineering(
        project_root=CONFIG_ROOT,
        user_names=user_names,
    )
    d = report.to_dict()
    print(f"  users={d['total_users']} patterns={len(d['cross_user_patterns'])} missing={len(d['missing_capabilities'])}")
    diseases = d.get("summary", {}).get("diseases", [])
    print(f"  diseases={len(diseases)}")
    by_lane = d.get("summary", {}).get("by_lane", {})
    print(f"  by_lane: {by_lane}")
    print()

    # ── Summary ──
    print("=== Done ===")
    print(f"Reports written to: {CONFIG_ROOT}/memory/")
    print(f"  evolution/reports/  — nightly evaluation reports")
    print(f"  reflection/        — case facts, patterns, proposals")
    print(f"  reflection/harness_engineering_report.json — cross-user analysis")
    print()
    print("Next: Review proposals at http://localhost:5180/proposals")

    return 0


if __name__ == "__main__":
    sys.exit(main())
