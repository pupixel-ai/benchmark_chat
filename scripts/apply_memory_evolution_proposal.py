#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from services.memory_pipeline.evolution import apply_memory_evolution_proposal


def _load_proposals(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("proposals"), list):
            return [item for item in payload.get("proposals", []) if isinstance(item, dict)]
        return [payload]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description="手动应用 nightly proposal 到 rule_assets（proposal-first 人工确认后执行）")
    parser.add_argument("--proposal-file", required=True, help="proposal JSON 文件路径")
    parser.add_argument("--proposal-id", required=True, help="要应用的 proposal_id")
    parser.add_argument("--actor", default="manual_cli", help="执行人标识")
    args = parser.parse_args()

    proposal_file = Path(args.proposal_file)
    if not proposal_file.exists():
        print(f"[proposal apply] 文件不存在: {proposal_file}")
        return 1

    proposals = _load_proposals(proposal_file)
    target = next((item for item in proposals if str(item.get("proposal_id") or "") == args.proposal_id), None)
    if target is None:
        print(f"[proposal apply] 未找到 proposal_id={args.proposal_id}")
        return 1

    result = apply_memory_evolution_proposal(
        project_root=str(ROOT),
        proposal=target,
        actor=args.actor,
    )
    print(f"[proposal apply] status={result.get('status')}")
    print(f"[proposal apply] actions={result.get('proposal_actions_path')}")
    if result.get("asset_paths"):
        print(f"[proposal apply] assets={json.dumps(result.get('asset_paths'), ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

