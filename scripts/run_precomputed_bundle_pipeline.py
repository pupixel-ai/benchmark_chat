#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from services.memory_pipeline.precomputed_bundle_runner import run_precomputed_bundle_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="对 face/vlm/lp1 离线 bundle 运行主角/关系/画像/下游审计临时测试")
    parser.add_argument("--bundle-dir", required=True, help="bundle 根目录")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    parser.add_argument("--profile-openrouter-key", default=None, help="仅 LP3 使用的 OpenRouter key")
    parser.add_argument(
        "--profile-model",
        default="google/gemini-3.1-flash-lite-preview",
        help="仅 LP3 使用的画像模型",
    )
    args = parser.parse_args()

    result = run_precomputed_bundle_pipeline(
        bundle_dir=args.bundle_dir,
        output_dir=args.output_dir,
        profile_openrouter_key=args.profile_openrouter_key,
        profile_model=args.profile_model,
    )

    print(f"[bundle pipeline] 输入: {args.bundle_dir}")
    print(f"[bundle pipeline] 输出: {result['output_dir']}")
    print(f"[bundle pipeline] 主角: {result['final_primary_person_id']}")
    print(f"[bundle pipeline] 事件数: {result['total_events']}")
    print(f"[bundle pipeline] 关系数: {result['total_relationships']}")
    print(f"[bundle pipeline] 画像: {result['structured_profile_path']}")
    print(f"[bundle pipeline] 下游审计: {result['downstream_audit_report_path']}")
    print(f"[bundle pipeline] 阶段汇报: {result['stage_reports_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
