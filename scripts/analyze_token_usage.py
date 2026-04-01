#!/usr/bin/env python3
"""离线分析 LLM token 使用情况 — 识别 token 黑洞和无效调用。

用法: python scripts/analyze_token_usage.py [--user USER]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
USAGE_LOG = PROJECT_ROOT / "memory" / "evolution" / "llm_usage.jsonl"
REFLECTION_DIR = PROJECT_ROOT / "memory" / "reflection"


def load_usage() -> list[dict]:
    if not USAGE_LOG.exists():
        print("未找到 llm_usage.jsonl，请先运行 evolve 生成数据。")
        sys.exit(1)
    entries = []
    for line in USAGE_LOG.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def analyze(filter_user: str = ""):
    entries = load_usage()
    if filter_user:
        entries = [e for e in entries if e.get("user") == filter_user]

    if not entries:
        print("无匹配的使用记录。")
        return

    # === 总览 ===
    total_in = sum(e.get("in", 0) for e in entries)
    total_out = sum(e.get("out", 0) for e in entries)
    print(f"\n{'='*60}")
    print(f"Token 使用总览 {'(用户: ' + filter_user + ')' if filter_user else '(全部用户)'}")
    print(f"{'='*60}")
    print(f"总调用次数: {len(entries)}")
    print(f"总 input tokens: {total_in:,}")
    print(f"总 output tokens: {total_out:,}")
    print(f"总 tokens: {total_in + total_out:,}")

    # === 按 caller 分组 ===
    by_caller: dict[str, dict] = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0})
    for e in entries:
        c = by_caller[e.get("caller", "unknown")]
        c["calls"] += 1
        c["in"] += e.get("in", 0)
        c["out"] += e.get("out", 0)

    print(f"\n--- 按环节 ---")
    for caller, stats in sorted(by_caller.items(), key=lambda x: x[1]["in"] + x[1]["out"], reverse=True):
        total = stats["in"] + stats["out"]
        print(f"  {caller:15s}  calls={stats['calls']:3d}  tokens={total:>8,}  (in={stats['in']:,} out={stats['out']:,})")

    # === 按 user 分组 ===
    by_user: dict[str, dict] = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0})
    for e in entries:
        u = by_user[e.get("user", "?")]
        u["calls"] += 1
        u["in"] += e.get("in", 0)
        u["out"] += e.get("out", 0)

    print(f"\n--- 按用户 ---")
    for user, stats in sorted(by_user.items(), key=lambda x: x[1]["in"] + x[1]["out"], reverse=True):
        total = stats["in"] + stats["out"]
        print(f"  {user:20s}  calls={stats['calls']:3d}  tokens={total:>8,}")

    # === 按 field 分组（top 10）===
    by_field: dict[str, dict] = defaultdict(lambda: {"calls": 0, "in": 0, "out": 0, "callers": defaultdict(int)})
    for e in entries:
        fk = e.get("field", "")
        if not fk:
            continue
        f = by_field[fk]
        f["calls"] += 1
        f["in"] += e.get("in", 0)
        f["out"] += e.get("out", 0)
        f["callers"][e.get("caller", "?")] += 1

    print(f"\n--- 按字段 (Top 10) ---")
    sorted_fields = sorted(by_field.items(), key=lambda x: x[1]["in"] + x[1]["out"], reverse=True)
    for fk, stats in sorted_fields[:10]:
        total = stats["in"] + stats["out"]
        callers_str = ", ".join(f"{c}={n}" for c, n in stats["callers"].items())
        print(f"  {fk:50s}  tokens={total:>8,}  calls={stats['calls']:2d}  ({callers_str})")

    # === GT Matcher 重复调用检测 ===
    gt_matcher_fields: dict[str, int] = defaultdict(int)
    for e in entries:
        if e.get("caller") == "gt_matcher" and e.get("field"):
            gt_matcher_fields[e["field"]] += 1

    repeated = {fk: n for fk, n in gt_matcher_fields.items() if n >= 3}
    if repeated:
        print(f"\n--- GT Matcher 重复调用 (>=3 次) ---")
        print(f"  以下字段被 LLM 反复评判，建议在 TOKEN_ALIASES/FIELD_HIERARCHIES 添加 rule：")
        for fk, n in sorted(repeated.items(), key=lambda x: x[1], reverse=True):
            print(f"    {fk}: {n} 次 LLM 调用")


if __name__ == "__main__":
    user = ""
    if "--user" in sys.argv:
        idx = sys.argv.index("--user")
        if idx + 1 < len(sys.argv):
            user = sys.argv[idx + 1]
    analyze(filter_user=user)
