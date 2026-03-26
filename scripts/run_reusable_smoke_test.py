#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.memory_pipeline.reusable_smoke_runner import run_reusable_smoke_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 reusable user data 的隔离临时 smoke test")
    parser.add_argument("--case-dir", required=True, help="reusable case 目录")
    parser.add_argument("--output-dir", default=None, help="可选输出目录")
    parser.add_argument("--run-id", default=None, help="可选 run id；默认使用时间戳")
    args = parser.parse_args()

    result = run_reusable_smoke_pipeline(
        case_dir=args.case_dir,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print(f"[reusable smoke] 输出目录: {result['output_dir']}")
    print(f"[reusable smoke] 主角: {result.get('final_primary_person_id')}")
    print(f"[reusable smoke] 关系数: {result.get('total_relationships', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
