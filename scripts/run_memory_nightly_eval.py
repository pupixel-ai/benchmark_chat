#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.memory_pipeline.evolution import run_memory_nightly_evaluation, run_memory_nightly_user_set_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 memory pipeline nightly insight/proposal 评测")
    parser.add_argument(
        "--user-name",
        default="",
        help="单用户名称（读取 memory/evolution/traces/{user}/{date}.jsonl）",
    )
    parser.add_argument(
        "--user-names",
        default="",
        help="多用户集合，逗号分隔；为空时自动发现有 GT 或 trace 的用户",
    )
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="评测日期 YYYY-MM-DD",
    )
    parser.add_argument(
        "--top-k-fields",
        type=int,
        default=3,
        help="每个用户按 GT 选择的高优字段数量",
    )
    args = parser.parse_args()

    user_names = []
    if args.user_names.strip():
        user_names = [item.strip() for item in args.user_names.split(",") if item.strip()]
    elif args.user_name.strip():
        user_names = [args.user_name.strip()]
    elif os.environ.get("MEMORY_USER_NAME", "").strip():
        user_names = [os.environ["MEMORY_USER_NAME"].strip()]

    if len(user_names) <= 1:
        target_user = user_names[0] if user_names else "default"
        result = run_memory_nightly_evaluation(
            project_root=str(ROOT),
            user_name=target_user,
            date_str=args.date,
            top_k_fields=max(1, int(args.top_k_fields)),
        )
        print(f"[nightly eval] user={target_user} date={args.date}")
        print(
            f"[nightly eval] traces={result['total_traces']} "
            f"focus_fields={result.get('total_focus_fields', 0)} "
            f"insights={result['total_insights']} proposals={result['total_proposals']}"
        )
        print(f"[nightly eval] report: {result['report_path']}")
        print(f"[nightly eval] insights: {result['insights_path']}")
        print(f"[nightly eval] proposals: {result['proposals_path']}")
        print(f"[nightly eval] focus_fields: {result.get('focus_fields_path')}")
        print(f"[nightly eval] field_cycles: {result.get('field_cycles_path')}")
        return 0

    aggregate = run_memory_nightly_user_set_evaluation(
        project_root=str(ROOT),
        user_names=user_names,
        date_str=args.date,
        top_k_fields=max(1, int(args.top_k_fields)),
    )
    print(f"[nightly eval] user_set_size={aggregate['total_users']} date={args.date}")
    print(f"[nightly eval] aggregate report: {aggregate['report_path']}")
    for user_result in aggregate["users"]:
        print(
            "[nightly eval] "
            f"user={user_result['user_name']} traces={user_result['total_traces']} "
            f"focus_fields={user_result['total_focus_fields']} proposals={user_result['total_proposals']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
