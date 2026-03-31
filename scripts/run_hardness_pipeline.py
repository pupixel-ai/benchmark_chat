#!/usr/bin/env python3
"""
Hardness Engineering 统一入口 — 从 datasets/ 驱动全链路。

三个子命令:
  run      对指定用户首次运行 pipeline → 对比 GT → 捕获 badcase → 写入 runs/
  evolve   对指定用户执行单人多轮优化（夜间自回归）
  harness  跨用户 pattern 聚类 + 策略建议

用法:
  # Step 1: 首次运行单个用户
  python scripts/run_hardness_pipeline.py run --user youruixun

  # Step 1: 首次运行所有用户
  python scripts/run_hardness_pipeline.py run --all

  # Step 2: 单人多轮优化
  python scripts/run_hardness_pipeline.py evolve --user youruixun
  python scripts/run_hardness_pipeline.py evolve --all

  # Step 3: 跨用户 Harness Engineering
  python scripts/run_hardness_pipeline.py harness

  # 一键全链路（run → evolve → harness）
  python scripts/run_hardness_pipeline.py full --all

数据流:
  datasets/{user}/source/  →  run_precomputed_bundle_pipeline()
                           →  persist_mainline_reflection_assets()
                           →  persist_downstream_audit_reflection_assets()
                           →  datasets/{user}/runs/{run_id}/
                           →  memory/reflection/case_facts_{user}.jsonl
  datasets/{user}/gt/      →  (同步到 memory/reflection/)
                           →  apply_profile_field_gt()
  memory/reflection/       →  run_reflection_task_generation()
                           →  run_memory_nightly_evaluation()
                           →  run_harness_engineering()
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(Path(PROJECT_ROOT) / ".env")
except Exception:
    pass

import requests as _requests

from config import PROJECT_ROOT as CONFIG_ROOT, DATASETS_DIR, OPENROUTER_API_KEY, OPENROUTER_BASE_URL

DATASETS_PATH = Path(DATASETS_DIR)
REFLECTION_DIR = Path(CONFIG_ROOT) / "memory" / "reflection"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GT 桥接
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _resolve_album_id(user_name: str, explicit_album_id: str | None = None) -> str:
    """确定 GT 应使用的 album_id：显式传入 > 最新 run_meta > 兜底。"""
    if explicit_album_id:
        return explicit_album_id
    # 从最新 run 的 run_meta.json 读取
    runs_dir = DATASETS_PATH / user_name / "runs"
    if runs_dir.exists():
        run_dirs = sorted(runs_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
        for d in run_dirs:
            meta = d / "run_meta.json"
            if meta.exists():
                try:
                    return json.loads(meta.read_text(encoding="utf-8")).get("album_id", "")
                except Exception:
                    pass
    return f"{user_name}_default"


def sync_gt_to_reflection(user_name: str, album_id: str | None = None) -> bool:
    """将 datasets/{user}/gt/profile_field_gt.jsonl 同步到 memory/reflection/。

    album_id 对齐策略：
    - 显式传入 album_id → 使用它（cmd_run 场景）
    - 未传入 → 从最新 run_meta.json 读取（cmd_evolve 等场景），保持和 case_facts 一致
    """
    gt_source = DATASETS_PATH / user_name / "gt" / "profile_field_gt.jsonl"
    if not gt_source.exists():
        return False

    resolved_id = _resolve_album_id(user_name, album_id)

    REFLECTION_DIR.mkdir(parents=True, exist_ok=True)
    gt_dest = REFLECTION_DIR / f"profile_field_gt_{user_name}.jsonl"

    # 读取并替换 album_id
    lines = gt_source.read_text(encoding="utf-8").splitlines()
    output_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            record["album_id"] = resolved_id
            output_lines.append(json.dumps(record, ensure_ascii=False))
        except json.JSONDecodeError:
            output_lines.append(line)

    gt_dest.write_text("\n".join(output_lines) + "\n", encoding="utf-8")
    print(f"  [gt] 同步 GT: {gt_source.name} → {gt_dest.name} (album_id={resolved_id})")
    return True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 用户发现
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def discover_dataset_users() -> List[str]:
    """从 datasets/ 目录发现所有用户。"""
    if not DATASETS_PATH.exists():
        return []
    return sorted(
        d.name for d in DATASETS_PATH.iterdir()
        if d.is_dir() and (d / "manifest.json").exists()
    )


def resolve_users(args) -> List[str]:
    """根据命令行参数解析用户列表。"""
    if getattr(args, "all", False):
        users = discover_dataset_users()
        if not users:
            print("[error] datasets/ 下没有用户数据集")
            sys.exit(1)
        return users
    if getattr(args, "user", None):
        user = args.user
        if not (DATASETS_PATH / user / "manifest.json").exists():
            print(f"[error] 用户数据集不存在: datasets/{user}/")
            sys.exit(1)
        return [user]
    print("[error] 请指定 --user <name> 或 --all")
    sys.exit(1)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1: run — 首次运行 + GT 对比 + badcase 捕获
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_run(args) -> int:
    from services.memory_pipeline.precomputed_bundle_runner import run_precomputed_bundle_pipeline
    from services.reflection import (
        persist_mainline_reflection_assets,
        persist_downstream_audit_reflection_assets,
    )

    users = resolve_users(args)
    profile_model = args.profile_model
    profile_key = args.profile_openrouter_key or OPENROUTER_API_KEY
    date_str = datetime.now().strftime("%Y-%m-%d")
    results = []

    for user_name in users:
        print(f"\n{'=' * 60}")
        print(f"[run] 用户: {user_name}")
        print(f"{'=' * 60}")

        source_dir = DATASETS_PATH / user_name / "source"
        if not source_dir.exists():
            print(f"  [skip] 缺少 source 目录")
            results.append({"user": user_name, "status": "skip", "reason": "no source"})
            continue

        # ── 1. 生成 run_id，创建 runs 目录 ──
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        album_id = f"{user_name}_{run_id}"
        run_dir = DATASETS_PATH / user_name / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # ── 2. 同步 GT（album_id 对齐，确保 GT 对比能匹配） ──
        has_gt = sync_gt_to_reflection(user_name, album_id=album_id)
        print(f"  [gt] {'有' if has_gt else '无'} GT 数据")

        # ── 3. 运行 precomputed bundle pipeline ──
        print(f"  [pipeline] 运行中...")
        try:
            pipeline_result = run_precomputed_bundle_pipeline(
                bundle_dir=str(source_dir),
                output_dir=str(run_dir),
                profile_openrouter_key=profile_key,
                profile_model=profile_model,
                user_name=user_name,
            )
        except Exception as exc:
            print(f"  [error] pipeline 失败: {exc}")
            results.append({"user": user_name, "status": "error", "reason": str(exc)})
            continue

        print(f"  [pipeline] 完成: 事件={pipeline_result['total_events']} 关系={pipeline_result['total_relationships']}")
        print(f"  [pipeline] 主角={pipeline_result['final_primary_person_id']}")

        # ── 4. 加载 pipeline 输出的 internal_artifacts ──
        internal_artifacts_path = run_dir / "internal_artifacts.json"
        downstream_audit_path = run_dir / "downstream_audit_report.json"

        internal_artifacts = {}
        if internal_artifacts_path.exists():
            internal_artifacts = json.loads(internal_artifacts_path.read_text(encoding="utf-8"))

        downstream_audit_report = {}
        if downstream_audit_path.exists():
            downstream_audit_report = json.loads(downstream_audit_path.read_text(encoding="utf-8"))

        # ── 5. 持久化 mainline badcase → case_facts ──
        print(f"  [capture] 捕获 mainline badcase...")
        try:
            mainline_result = persist_mainline_reflection_assets(
                internal_artifacts=internal_artifacts,
                project_root=CONFIG_ROOT,
                user_name=user_name,
                album_id=album_id,
            )
            print(f"  [capture] 写入 case_facts: {mainline_result.get('written_case_fact_count', 0)} 条")
        except Exception as exc:
            print(f"  [capture] mainline 捕获失败: {exc}")

        # ── 6. 持久化 downstream audit badcase ──
        if downstream_audit_report:
            print(f"  [capture] 捕获 downstream audit badcase...")
            try:
                downstream_result = persist_downstream_audit_reflection_assets(
                    downstream_audit_report=downstream_audit_report,
                    project_root=CONFIG_ROOT,
                    user_name=user_name,
                    album_id=album_id,
                )
                print(f"  [capture] 写入 audit case_facts: {downstream_result.get('written_case_fact_count', 0)} 条")
            except Exception as exc:
                print(f"  [capture] downstream 捕获失败: {exc}")

        # ── 7. GT 对比（直接调 apply_profile_field_gt，不跑 reflection agent） ──
        if has_gt:
            print(f"  [gt] 运行 GT 对比...")
            try:
                from services.reflection.gt import load_profile_field_gt, apply_profile_field_gt
                from services.reflection.types import CaseFact
                from dataclasses import fields as dc_fields

                valid_fields = {f.name for f in dc_fields(CaseFact)}
                case_facts_path = REFLECTION_DIR / f"case_facts_{user_name}.jsonl"
                gt_comp_path = REFLECTION_DIR / f"gt_comparisons_{user_name}.jsonl"

                facts = []
                for line in case_facts_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        d = json.loads(line)
                        facts.append(CaseFact(**{k: v for k, v in d.items() if k in valid_fields}))

                gt_records = load_profile_field_gt(str(REFLECTION_DIR / f"profile_field_gt_{user_name}.jsonl"))
                updated_facts, comparisons = apply_profile_field_gt(facts, gt_records)

                # 写出 gt_comparisons（保护人工校准不被覆盖）
                human_overrides = _load_human_overrides(user_name)
                with open(gt_comp_path, "w", encoding="utf-8") as f:
                    for c in comparisons:
                        cid = c.get("case_id", "")
                        override_grade = human_overrides.get(cid)
                        if override_grade:
                            cr = c.get("comparison_result", {})
                            cr["original_grade"] = cr.get("grade", "")
                            cr["grade"] = override_grade
                            cr["human_override"] = True
                            c["comparison_result"] = cr
                        f.write(json.dumps(c, ensure_ascii=False) + "\n")

                # 更新 case_facts（带 comparison 结果）
                with open(case_facts_path, "w", encoding="utf-8") as f:
                    for fact in updated_facts:
                        f.write(json.dumps(fact.to_dict(), ensure_ascii=False) + "\n")

                from collections import Counter
                grades = Counter(
                    (c.get("comparison_result") or {}).get("grade", "unknown") for c in comparisons
                )
                print(f"  [gt] GT 对比完成: {len(comparisons)} 条")
                for g, n in sorted(grades.items(), key=lambda x: -x[1]):
                    print(f"    {g}: {n}")
            except Exception as exc:
                print(f"  [gt] GT 对比失败: {exc}")
                import traceback; traceback.print_exc()

        # ── 8. 确保 trace ledger 存在（bundle_runner 内部写入可能静默失败） ──
        trace_json = run_dir / "memory_pipeline_run_trace.json"
        trace_ledger_dir = Path(CONFIG_ROOT) / "memory" / "evolution" / "traces" / user_name
        trace_ledger = trace_ledger_dir / f"{date_str}.jsonl"
        if trace_json.exists() and not trace_ledger.exists():
            try:
                from services.memory_pipeline.evolution import persist_memory_run_trace
                trace_payload = json.loads(trace_json.read_text(encoding="utf-8"))
                persist_memory_run_trace(
                    project_root=CONFIG_ROOT,
                    output_dir=str(run_dir),
                    trace_payload=trace_payload,
                )
                print(f"  [trace] ledger 补写成功")
            except Exception as exc:
                print(f"  [trace] ledger 补写失败: {exc}")

        # ── 9. 写 run_meta.json ──
        run_meta = {
            "run_id": run_id,
            "user_name": user_name,
            "album_id": album_id,
            "created_at": datetime.now().isoformat(),
            "date": date_str,
            "profile_model": profile_model,
            "has_gt": has_gt,
            "total_events": pipeline_result.get("total_events", 0),
            "total_relationships": pipeline_result.get("total_relationships", 0),
            "final_primary_person_id": pipeline_result.get("final_primary_person_id"),
        }
        (run_dir / "run_meta.json").write_text(
            json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # ── 10. 发飞书 GT 对比卡片（如果有 GT） ──
        if has_gt:
            try:
                feishu_resp = _requests.post(
                    "http://localhost:8700/api/errors/send-gt-card",
                    json={"user_name": user_name},
                    timeout=15,
                )
                if feishu_resp.ok:
                    print(f"  [feishu] GT 对比卡片已发送")
                else:
                    print(f"  [feishu] 发送失败: {feishu_resp.text[:100]}")
            except Exception as exc:
                print(f"  [feishu] 发送跳过: {exc}")

        # ── 11. 初始化 field_loop_state（让热力图在 GT 对比后就可用） ──
        if has_gt:
            _init_field_loop_state_from_gt(user_name)

        results.append({"user": user_name, "status": "ok", "run_id": run_id})
        print(f"  [done] 输出: datasets/{user_name}/runs/{run_id}/")

    # ── 汇总 ──
    _print_summary("run", results)
    return 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 1.5: reflect — 独立的 reflection agent 分析（慢，单独跑）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_reflect(args) -> int:
    from services.reflection.tasks import run_reflection_task_generation

    users = resolve_users(args)
    results = []

    for user_name in users:
        sync_gt_to_reflection(user_name)
        print(f"\n[reflect] {user_name}: 运行 reflection task generation...")
        try:
            result = run_reflection_task_generation(
                project_root=CONFIG_ROOT,
                user_name=user_name,
            )
            patterns = result.get("upstream_patterns_count", 0)
            proposals = result.get("proposals_count", 0)
            print(f"  patterns: {patterns}, proposals: {proposals}")
            results.append({"user": user_name, "status": "ok"})
        except Exception as exc:
            print(f"  [error] {exc}")
            results.append({"user": user_name, "status": "error", "reason": str(exc)})

    _print_summary("reflect", results)
    return 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 2: evolve — 单人多轮优化
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_human_overrides(user_name: str) -> Dict[str, str]:
    """加载人工校准记录。返回 {case_id: override_grade}。人工评注是唯一基线，不可被覆盖。"""
    overrides_path = REFLECTION_DIR / "gt_grade_overrides.json"
    if not overrides_path.exists():
        return {}
    try:
        all_overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    suffix = f"_{user_name}"
    return {
        k.replace(suffix, ""): v
        for k, v in all_overrides.items()
        if k.endswith(suffix)
    }


def _init_field_loop_state_from_gt(user_name: str) -> None:
    """GT 对比完成后初始化 field_loop_state，让热力图立即可用。"""
    gt_path = REFLECTION_DIR / f"gt_comparisons_{user_name}.jsonl"
    if not gt_path.exists():
        return
    state_path = Path(CONFIG_ROOT) / "memory" / "evolution" / "field_loop_state" / f"{user_name}.json"
    # 如果已有 state 则不覆盖（evolve 可能已经写过更详细的状态）
    if state_path.exists():
        return
    fields: Dict[str, Any] = {}
    for line in gt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        fk = r.get("field_key", "")
        if not fk:
            continue
        cr = r.get("comparison_result", {})
        grade = cr.get("grade", "")
        if not grade:
            continue
        good = grade in ("exact_match", "close_match", "improved")
        fields[fk] = {
            "cycle_count": 0,
            "last_grade": grade,
            "last_issue_score": 0.0,
            "last_status": "monitoring" if good else "not_focused",
            "last_signal_key": "",
            "seen_signal_keys": [],
            "no_new_signal_streak": 0,
            "cooldown_remaining": 0,
            "issue_score_history": [],
            "score_trend": "",
            "reflect_fail_streak": 0,
            "last_updated": datetime.now().isoformat(),
        }
    if fields:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state = {"updated_at": datetime.now().isoformat(), "fields": fields}
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [heatmap] 初始化 field_loop_state: {len(fields)} 字段")


def _notify_evolution_round(user_name: str, date_str: str, result: dict) -> None:
    """发送 evolution 轮次汇报到飞书个人，失败不阻塞 pipeline。"""
    try:
        from services.reflection.feishu import send_evolution_round_card

        # 加载本轮 field_cycles 和 proposals
        evo_dir = Path(CONFIG_ROOT) / "memory" / "evolution"
        field_cycles = []
        proposals = []
        fc_path = evo_dir / "field_cycles" / user_name / f"{date_str}.json"
        if fc_path.exists():
            field_cycles = json.loads(fc_path.read_text(encoding="utf-8"))
        pp_path = evo_dir / "proposals" / user_name / f"{date_str}.json"
        if pp_path.exists():
            proposals = json.loads(pp_path.read_text(encoding="utf-8"))

        # 加载 GT 对比数据
        gt_comparisons: dict[str, dict] = {}
        gt_path = REFLECTION_DIR / f"gt_comparisons_{user_name}.jsonl"
        if gt_path.exists():
            for line in gt_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    fk = rec.get("field_key", "")
                    cr = rec.get("comparison_result", {})
                    if fk and cr:
                        gt_comparisons[fk] = cr
                except json.JSONDecodeError:
                    pass

        # 加载 VLM/event summaries 用于卡片证据描述
        from services.memory_pipeline.evolution import _load_vlm_summaries, _load_event_summaries
        vlm_summaries = _load_vlm_summaries(project_root=CONFIG_ROOT, user_name=user_name)
        event_summaries = _load_event_summaries(project_root=CONFIG_ROOT, user_name=user_name)

        resp = send_evolution_round_card(
            user_name=user_name,
            date_str=date_str,
            result=result,
            field_cycles=field_cycles,
            proposals=proposals,
            gt_comparisons=gt_comparisons,
            vlm_summaries=vlm_summaries,
            event_summaries=event_summaries,
        )
        print(f"  [飞书] 轮次汇报已发送 message_id={resp.get('message_id', '')}")
    except Exception as exc:
        print(f"  [飞书] 汇报发送失败: {exc}")


def cmd_evolve(args) -> int:
    from services.memory_pipeline.evolution import (
        run_memory_nightly_evaluation,
        run_memory_nightly_user_set_evaluation,
    )

    users = resolve_users(args)
    date_str = args.date or datetime.now().strftime("%Y-%m-%d")
    top_k = args.top_k_fields
    skip_gt_check = getattr(args, "skip_gt_check", False)
    results = []

    # 先同步所有用户的 GT
    for user_name in users:
        sync_gt_to_reflection(user_name)

    # GT 校准门槛：首次 evolve 前必须有人工校准记录
    if not skip_gt_check:
        unconfirmed = [u for u in users if not _check_gt_confirmed(u)]
        if unconfirmed:
            print(f"\n⚠️  以下用户的 GT 对比尚未人工校准，evolve 已跳过:")
            for u in unconfirmed:
                print(f"  - {u}")
            print(f"\n请先在平台校准: http://localhost:5180/errors")
            print(f"校准完成后重新运行，或加 --skip-gt-check 跳过")
            return 1

    if len(users) == 1:
        user_name = users[0]
        print(f"\n[evolve] 单人优化: {user_name} | date={date_str} | top_k={top_k}")
        try:
            result = run_memory_nightly_evaluation(
                project_root=CONFIG_ROOT,
                user_name=user_name,
                date_str=date_str,
                top_k_fields=top_k,
            )
            traces = result.get("total_traces", 0)
            insights = result.get("total_insights", 0)
            proposals = result.get("total_proposals", 0)
            focus = result.get("total_focus_fields", 0)
            print(f"  traces={traces} focus_fields={focus} insights={insights} proposals={proposals}")
            print(f"  报告: {result.get('report_path', '')}")
            results.append({"user": user_name, "status": "ok"})
            # 飞书轮次汇报
            _notify_evolution_round(user_name, date_str, result)
        except Exception as exc:
            print(f"  [error] {exc}")
            results.append({"user": user_name, "status": "error", "reason": str(exc)})
    else:
        print(f"\n[evolve] 多用户优化: {len(users)} 用户 | date={date_str}")
        try:
            result = run_memory_nightly_user_set_evaluation(
                project_root=CONFIG_ROOT,
                user_names=users,
                date_str=date_str,
                top_k_fields=top_k,
            )
            print(f"  总用户: {result.get('total_users', 0)}")
            for u in result.get("users", []):
                uname = u.get("user_name", "?")
                print(f"    {uname}: focus={u.get('total_focus_fields', 0)} proposals={u.get('total_proposals', 0)}")
                results.append({"user": uname, "status": "ok"})
                # 飞书轮次汇报（每个用户单独发）
                _notify_evolution_round(uname, date_str, u)
            print(f"  报告: {result.get('report_path', '')}")
        except Exception as exc:
            print(f"  [error] {exc}")
            results.append({"user": "all", "status": "error", "reason": str(exc)})

    _print_summary("evolve", results)
    return 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Step 3: harness — 跨用户聚类 + 策略建议
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_harness(args) -> int:
    from services.reflection.harness_engineering import run_harness_engineering

    # 先同步所有用户的 GT 并运行 per-user reflection
    users = None
    if getattr(args, "all", False):
        users = discover_dataset_users()
    elif getattr(args, "users", None):
        users = [u.strip() for u in args.users.split(",") if u.strip()]

    # 同步 GT
    all_dataset_users = discover_dataset_users()
    for u in all_dataset_users:
        sync_gt_to_reflection(u)

    # Per-user reflection task generation（确保 case_facts 有 GT 对比结果）
    print(f"\n[harness] Step 1/2: Per-user reflection")
    from services.reflection.tasks import run_reflection_task_generation

    reflection_users = users or _discover_reflection_users()
    for user_name in reflection_users:
        try:
            result = run_reflection_task_generation(
                project_root=CONFIG_ROOT,
                user_name=user_name,
            )
            patterns = result.get("upstream_patterns_count", 0)
            proposals = result.get("proposals_count", 0)
            print(f"  {user_name}: patterns={patterns} proposals={proposals}")
        except Exception as exc:
            print(f"  {user_name}: ERROR {exc}")

    # Cross-user harness engineering
    print(f"\n[harness] Step 2/2: Cross-user Harness Engineering")
    report = run_harness_engineering(
        project_root=CONFIG_ROOT,
        user_names=users,
    )
    d = report.to_dict()
    total_patterns = len(d.get("cross_user_patterns", []))
    missing_caps = len(d.get("missing_capabilities", []))
    diseases = len(d.get("summary", {}).get("diseases", []))
    by_lane = d.get("summary", {}).get("by_lane", {})

    print(f"  用户数: {d['total_users']}")
    print(f"  跨用户 patterns: {total_patterns}")
    print(f"  缺失能力: {missing_caps}")
    print(f"  diseases: {diseases}")
    print(f"  by_lane: {by_lane}")
    print(f"  报告: memory/reflection/harness_engineering_report.json")
    return 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# full — 一键全链路
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def cmd_full(args) -> int:
    print("=" * 60)
    print("Hardness Engineering 全链路")
    print("=" * 60)

    print("\n[1/3] run: 首次运行 + GT 对比 + badcase 捕获")
    cmd_run(args)

    print(f"\n{'=' * 60}")
    print("[2/3] evolve: 单人多轮优化")
    cmd_evolve(args)

    print(f"\n{'=' * 60}")
    print("[3/3] harness: 跨用户 pattern 聚类 + 策略建议")
    cmd_harness(args)

    print(f"\n{'=' * 60}")
    print("[done] 全链路完成")
    print(f"  datasets/ — 每用户运行记录")
    print(f"  memory/reflection/ — 反思数据")
    print(f"  memory/evolution/ — 自回归数据")
    return 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# autoloop — 自动循环（run → evolve 多轮 → harness → 等待审批）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _is_user_converged(user_name: str) -> bool:
    """检查用户是否所有活跃字段都已收敛。"""
    state_path = Path(CONFIG_ROOT) / "memory" / "evolution" / "field_loop_state" / f"{user_name}.json"
    if not state_path.exists():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    fields = state.get("fields") or {}
    if not fields:
        return False
    # throttle_armed / throttled 仍在循环中（暂停不等于结束）
    _NOT_CONVERGED = {"needs_next_cycle", "new_rule_candidate", "new_insight_found", "initial_snapshot", "throttle_armed", "throttled"}
    for field_state in fields.values():
        status = str(field_state.get("last_status") or "").strip()
        if status in _NOT_CONVERGED:
            return False
    return True


def _check_gt_confirmed(user_name: str) -> bool:
    """检查该用户的 GT 对比结果是否已被人工确认。

    两种确认方式：
    1. gt_grade_overrides.json 中有该用户的修正记录（key 包含 _{user_name}）
    2. datasets/{user}/gt/human_overrides.jsonl 存在且非空
    3. gt_confirmed_{user}.flag 标记文件存在
    """
    # 方式 1: 全局 overrides 中有该用户的记录
    overrides_path = Path(CONFIG_ROOT) / "memory" / "reflection" / "gt_grade_overrides.json"
    if overrides_path.exists():
        try:
            data = json.loads(overrides_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                user_keys = [k for k in data if k.endswith(f"_{user_name}")]
                if user_keys:
                    return True
        except Exception:
            pass

    # 方式 2: 用户的 human_overrides.jsonl 存在且非空
    user_overrides = Path(CONFIG_ROOT) / "datasets" / user_name / "gt" / "human_overrides.jsonl"
    if user_overrides.exists() and user_overrides.stat().st_size > 10:
        return True

    # 方式 3: 手动标记文件
    confirmed_path = Path(CONFIG_ROOT) / "memory" / "reflection" / f"gt_confirmed_{user_name}.flag"
    return confirmed_path.exists()


def cmd_autoloop(args) -> int:
    max_rounds = getattr(args, "max_rounds", 10)
    skip_gt_check = getattr(args, "skip_gt_check", False)
    users = resolve_users(args)

    print("=" * 60)
    print(f"Autoloop: {len(users)} 用户, 最多 {max_rounds} 轮 evolve")
    print("=" * 60)

    # Step 1: run
    print("\n[1/3] run: 画像 pipeline + GT 对比")
    cmd_run(args)

    # Step 1.5: 等待 GT 对比人工确认
    if not skip_gt_check:
        unconfirmed = [u for u in users if not _check_gt_confirmed(u)]
        if unconfirmed:
            print(f"\n⚠️  以下用户的 GT 对比结果尚未人工校对:")
            for u in unconfirmed:
                print(f"  - {u}")
            print(f"\n请先在平台校对: http://localhost:5180/errors")
            print(f"校对完成后重新运行: python scripts/run_hardness_pipeline.py autoloop --user {','.join(users)} --skip-gt-check")
            print(f"或标记为已确认: python scripts/run_hardness_pipeline.py autoloop --user {','.join(users)} --skip-gt-check")
            return 1

    # Step 2: evolve 循环
    for round_idx in range(1, max_rounds + 1):
        print(f"\n[2/3] evolve 第 {round_idx} 轮")
        args.date = datetime.now().strftime("%Y-%m-%d")
        cmd_evolve(args)

        converged_count = sum(1 for u in users if _is_user_converged(u))
        print(f"  收敛: {converged_count}/{len(users)}")
        if converged_count == len(users):
            print(f"  所有用户已收敛，共 {round_idx} 轮")
            break
    else:
        # 达到 max_rounds 仍未收敛 — 强制复盘
        print(f"\n  达到最大轮次 ({max_rounds})，强制生成复盘")
        date_str_now = datetime.now().strftime("%Y-%m-%d")
        for u in users:
            if not _is_user_converged(u):
                _force_recap_if_needed(u, date_str_now)

    # Step 3: harness
    print("\n[3/3] harness: 跨用户聚类")
    cmd_harness(args)

    print(f"\n{'=' * 60}")
    print("[autoloop] 完成。请审批提案:")
    print(f"  http://localhost:5180/proposals")
    return 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# rerun-fields — approve 后只重跑指定字段
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _find_latest_run_dir(user_name: str) -> Path | None:
    """找到用户最近一次 run 的输出目录。"""
    runs_dir = DATASETS_PATH / user_name / "runs"
    if not runs_dir.exists():
        return None
    run_dirs = sorted(runs_dir.iterdir(), reverse=True)
    return run_dirs[0] if run_dirs else None


def cmd_rerun_fields(args) -> int:
    """规则修改后只重跑指定字段的 LP3 推理 + GT 对比。"""
    from services.memory_pipeline.precomputed_loader import load_precomputed_memory_state
    from services.memory_pipeline.profile_fields import generate_structured_profile, build_profile_context
    from services.memory_pipeline.profile_llm import OpenRouterProfileLLMProcessor

    user_name = args.user
    field_keys = set(f.strip() for f in args.fields.split(",") if f.strip())
    if not field_keys:
        print("[rerun-fields] 未指定字段")
        return 1

    print(f"\n[rerun-fields] 用户: {user_name} | 字段: {field_keys}")

    # 1. 加载上次 run 的 state（复用 VLM/事件/关系）
    source_dir = DATASETS_PATH / user_name / "source"
    if not source_dir.exists():
        print(f"[rerun-fields] source 目录不存在: {source_dir}")
        return 1

    state = load_precomputed_memory_state(str(source_dir))
    state.profile_context = build_profile_context(state)
    print(f"  state 加载完成: events={len(state.events or [])} vlm={len(state.vlm_results or [])}")

    # 2. 构建 LLM processor
    api_key = OPENROUTER_API_KEY
    if not api_key:
        print("[rerun-fields] 无 OpenRouter API key")
        return 1
    llm_processor = OpenRouterProfileLLMProcessor(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        model="deepseek/deepseek-chat-v3-0324",
    )

    # 3. 只重跑指定字段
    print(f"  重跑 {len(field_keys)} 个字段...")
    profile_result = generate_structured_profile(
        state, llm_processor=llm_processor,
        target_field_keys=field_keys,
    )

    # 4. 读取上次 run 的完整 profile，合并新结果
    latest_run = _find_latest_run_dir(user_name)
    if latest_run:
        ia_path = latest_run / "internal_artifacts.json"
        if ia_path.exists():
            ia = json.loads(ia_path.read_text(encoding="utf-8"))
            old_decisions = ia.get("profile_fact_decisions", [])
            new_decisions = profile_result.get("field_decisions", [])
            # 用新结果覆盖对应字段
            new_by_key = {d["field_key"]: d for d in new_decisions}
            merged_decisions = []
            for d in old_decisions:
                if d["field_key"] in new_by_key:
                    merged_decisions.append(new_by_key[d["field_key"]])
                    print(f"  已更新: {d['field_key']} → 新值: {new_by_key[d['field_key']].get('final', {}).get('value', '?')}")
                else:
                    merged_decisions.append(d)
            # 保存更新后的 artifacts
            ia["profile_fact_decisions"] = merged_decisions
            ia_path.write_text(json.dumps(ia, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  artifacts 已更新: {ia_path}")

    # 5. 更新被修改字段的 GT 对比结果
    gt_comps_path = REFLECTION_DIR / f"gt_comparisons_{user_name}.jsonl"
    gt_path = REFLECTION_DIR / f"profile_field_gt_{user_name}.jsonl"
    before_scores: Dict[str, float] = {}

    if gt_comps_path.exists():
        # 读取修改前的分数
        for line in gt_comps_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            fk = r.get("field_key", "")
            if fk in field_keys:
                before_scores[fk] = float(r.get("comparison_result", {}).get("score", 0))

    # 用新的 field_decisions 更新 GT 对比
    new_decisions = profile_result.get("field_decisions", [])
    new_by_key = {d["field_key"]: d for d in new_decisions}

    gt_by_field: Dict[str, Any] = {}
    if gt_path.exists():
        for line in gt_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            fk = r.get("field_key", "")
            if fk:
                gt_by_field[fk] = r

    # 保存 GT 对比快照（rerun 前）
    gt_snapshots_dir = Path(CONFIG_ROOT) / "memory" / "evolution" / "gt_snapshots" / user_name
    gt_snapshots_dir.mkdir(parents=True, exist_ok=True)
    if gt_comps_path.exists():
        import shutil
        snapshot_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        shutil.copy2(gt_comps_path, gt_snapshots_dir / snapshot_name)

    # 重新对比被修改字段
    from services.reflection.gt_matcher import compare_profile_field_values
    if gt_comps_path.exists():
        lines = gt_comps_path.read_text().splitlines()
        updated_lines = []
        for line in lines:
            if not line.strip():
                continue
            r = json.loads(line)
            fk = r.get("field_key", "")
            if fk in new_by_key:
                # 人工校准过的字段不覆盖——人工评注是唯一基线
                cr = r.get("comparison_result", {})
                if cr.get("human_override"):
                    updated_lines.append(json.dumps(r, ensure_ascii=False))
                    print(f"  跳过人工校准字段: {fk} (grade={cr.get('grade')})")
                    continue
                new_val = new_by_key[fk].get("final", {}).get("value")
                gt_rec = gt_by_field.get(fk, {})
                gt_val = gt_rec.get("gt_value")
                new_comp = compare_profile_field_values(field_key=fk, predicted_value=new_val, gt_value=gt_val)
                r["comparison_result"] = new_comp
                r["comparison_result"]["output_value"] = new_val
                r["comparison_result"]["gt_value"] = gt_val
            updated_lines.append(json.dumps(r, ensure_ascii=False))
        gt_comps_path.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
        print(f"  GT 对比已更新: {len(new_by_key)} 个字段")

    # 6. 读取重跑后的新分数和 grade（用于对比）
    after_scores: Dict[str, float] = {}
    after_grades: Dict[str, str] = {}
    before_grades: Dict[str, str] = {}
    if gt_comps_path.exists():
        for line in gt_comps_path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            fk = r.get("field_key", "")
            if fk in field_keys:
                cr = r.get("comparison_result", {})
                after_scores[fk] = float(cr.get("score", 0))
                after_grades[fk] = cr.get("grade", "?")

    # before_grades 从 snapshot 读取
    if gt_snapshots_dir.exists():
        snaps = sorted(gt_snapshots_dir.glob("*.jsonl"), reverse=True)
        if snaps:
            for line in snaps[0].read_text().splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                fk = r.get("field_key", "")
                if fk in field_keys:
                    before_grades[fk] = (r.get("comparison_result") or {}).get("grade", "?")

    GRADE_ZH = {"exact_match": "完全匹配", "close_match": "接近匹配", "partial_match": "部分匹配",
                "mismatch": "不匹配", "missing_prediction": "未召回", "improved": "已优化", "missing_gt": "GT缺失"}

    for fk in field_keys:
        if fk in new_by_key:
            new_val = new_by_key[fk].get("final", {}).get("value", "?")
            gt_val = gt_by_field.get(fk, {}).get("gt_value", "?")
            b_score = before_scores.get(fk, 0)
            a_score = after_scores.get(fk, 0)
            b_grade = before_grades.get(fk, "?")
            a_grade = after_grades.get(fk, "?")
            delta = "+" if a_score > b_score else "-" if a_score < b_score else "="
            print(f"  {fk}: {b_grade}→{a_grade} | score {b_score:.1f}→{a_score:.1f} ({delta})")

    # 7. 先发改进结果飞书通知
    try:
        from services.reflection.feishu import _get_tenant_access_token, _translate_values_for_card
        from services.reflection.labels import load_bilingual_label_rows
        import requests as _rerun_requests
        from config import FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_API_BASE_URL, FEISHU_DEFAULT_RECEIVE_ID, FEISHU_DEFAULT_RECEIVE_ID_TYPE

        label_map = {}
        try:
            for row in load_bilingual_label_rows():
                if row.get("key") and row.get("zh_label"):
                    label_map[row["key"]] = row["zh_label"]
        except Exception:
            pass

        rerun_lines = []
        for fk in field_keys:
            if fk not in new_by_key:
                continue
            new_val = str(new_by_key[fk].get("final", {}).get("value", "—"))[:60]
            gt_rec = gt_by_field.get(fk, {})
            gt_val_str = str(gt_rec.get("gt_value", "—"))[:60]
            zh_name = label_map.get(fk, fk.rsplit(".", 1)[-1])

            # 翻译值
            val_zh = _translate_values_for_card([new_val, gt_val_str])
            new_val_zh = val_zh.get(new_val, new_val)
            gt_val_zh = val_zh.get(gt_val_str, gt_val_str)

            b_grade = GRADE_ZH.get(before_grades.get(fk, ""), before_grades.get(fk, "—"))
            a_grade = GRADE_ZH.get(after_grades.get(fk, ""), after_grades.get(fk, "—"))
            b_score = before_scores.get(fk, 0)
            a_score = after_scores.get(fk, 0)
            improved = a_score < b_score

            rerun_lines.append(
                f"**{zh_name}**\n"
                f"重跑后: `{new_val_zh}`\n"
                f"GT: `{gt_val_zh}`\n"
                f"评分: {b_grade} ({b_score:.0f}) → {a_grade} ({a_score:.0f}) {'✅' if improved else '➡️' if a_score == b_score else '⚠️'}"
            )

        if rerun_lines and FEISHU_APP_ID and FEISHU_APP_SECRET and FEISHU_DEFAULT_RECEIVE_ID:
            # 计算改善/持平/恶化数
            improved_count = sum(1 for fk in field_keys if after_scores.get(fk, 0) < before_scores.get(fk, 0))
            same_count = sum(1 for fk in field_keys if after_scores.get(fk, 0) == before_scores.get(fk, 0))
            worse_count = len(field_keys) - improved_count - same_count

            token = _get_tenant_access_token(app_id=FEISHU_APP_ID, app_secret=FEISHU_APP_SECRET, api_base_url=FEISHU_API_BASE_URL)
            card = {
                "schema": "2.0",
                "config": {"enable_forward": True, "width_mode": "fill"},
                "header": {
                    "title": {"tag": "plain_text", "content": f"规则重跑结果 — {user_name}"},
                    "template": "green" if improved_count > 0 and worse_count == 0 else "wathet",
                    "subtitle": {"tag": "plain_text", "content": f"{len(field_keys)} 个字段 | ✅{improved_count} 改善 ➡️{same_count} 持平 ⚠️{worse_count} 恶化"},
                },
                "body": {"elements": [{"tag": "markdown", "content": "\n\n".join(rerun_lines)}]},
            }
            _rerun_requests.post(
                f"{FEISHU_API_BASE_URL}/open-apis/im/v1/messages?receive_id_type={FEISHU_DEFAULT_RECEIVE_ID_TYPE}",
                json={"receive_id": FEISHU_DEFAULT_RECEIVE_ID, "msg_type": "interactive", "content": json.dumps(card, ensure_ascii=False)},
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
                timeout=15,
            )
            print(f"  [飞书] 重跑结果已发送")
    except Exception as exc:
        print(f"  [飞书] 重跑结果发送失败: {exc}")

    # 8. 自动触发下一轮 evolve（用新分数生成下一轮提案）
    # 版本上限保护：同一天超过 MAX_VERSIONS_PER_DAY 则不再 evolve
    MAX_VERSIONS_PER_DAY = 10
    date_str_now = datetime.now().strftime("%Y-%m-%d")
    proposals_dir = Path(CONFIG_ROOT) / "memory" / "evolution" / "proposals" / user_name
    existing_versions = sorted(proposals_dir.glob(f"{date_str_now}_c*.json")) if proposals_dir.exists() else []
    if len(existing_versions) >= MAX_VERSIONS_PER_DAY:
        print(f"\n  [auto] 已达单日版本上限 ({MAX_VERSIONS_PER_DAY} 轮)，停止 evolve")
        # 强制触发复盘
        _force_recap_if_needed(user_name, date_str_now)
        return 0

    print(f"\n  [auto] 触发下一轮 evolve (当前第 {len(existing_versions) + 1} 轮)...")
    from services.memory_pipeline.evolution import run_memory_nightly_evaluation
    try:
        evolve_result = run_memory_nightly_evaluation(
            project_root=CONFIG_ROOT,
            user_name=user_name,
            date_str=date_str_now,
            top_k_fields=3,
        )
        total_proposals = evolve_result.get("total_proposals", 0)
        converged = evolve_result.get("converged", False)
        print(f"  [auto] evolve 完成: focus={evolve_result.get('total_focus_fields', 0)} proposals={total_proposals} converged={converged}")

        if converged:
            # 已收敛 — 通知用户循环结束
            _notify_evolution_round(user_name, date_str_now, evolve_result)
            print(f"  [auto] 用户 {user_name} 已收敛，循环结束")
        elif total_proposals > 0:
            # 有新提案 — 正常通知
            _notify_evolution_round(user_name, date_str_now, evolve_result)
        else:
            # 无新提案 — 通知用户当前轮次无可执行提案
            print(f"  [auto] 本轮无新提案，通知用户")
            _notify_no_proposals(user_name, date_str_now, evolve_result)
    except Exception as exc:
        print(f"  [auto] evolve 失败: {exc}")

    return 0


def _force_recap_if_needed(user_name: str, date_str: str) -> None:
    """达到版本上限时强制生成复盘（即使还有活跃字段）。"""
    from services.memory_pipeline.evolution import _check_and_generate_recap
    state_path = Path(CONFIG_ROOT) / "memory" / "evolution" / "field_loop_state" / f"{user_name}.json"
    if not state_path.exists():
        return
    state = json.loads(state_path.read_text(encoding="utf-8"))
    cycles_dir = Path(CONFIG_ROOT) / "memory" / "evolution" / "field_cycles" / user_name
    all_cycles = []
    if cycles_dir.exists():
        latest = cycles_dir / f"{date_str}.json"
        if latest.exists():
            all_cycles = json.loads(latest.read_text(encoding="utf-8"))

    # 将剩余活跃字段标记为 throttled（强制收敛）
    _ACTIVE = {"needs_next_cycle", "new_rule_candidate", "new_insight_found", "initial_snapshot"}
    fields = state.get("fields") or {}
    forced = []
    for fk, fs in fields.items():
        if fs.get("last_status") in _ACTIVE:
            fs["last_status"] = "throttled"
            fs["cooldown_remaining"] = 0
            forced.append(fk)
    if forced:
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [recap] 强制收敛 {len(forced)} 个活跃字段: {forced}")

    _check_and_generate_recap(
        project_root=CONFIG_ROOT,
        user_name=user_name,
        date_str=date_str,
        state_payload=state,
        field_cycles_all=all_cycles,
    )


def _notify_no_proposals(user_name: str, date_str: str, evolve_result: dict) -> None:
    """通知用户本轮无新提案。"""
    try:
        from services.reflection.feishu import _get_tenant_access_token
        import requests as _requests
        from config import (
            FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_API_BASE_URL,
            FEISHU_DEFAULT_RECEIVE_ID, FEISHU_DEFAULT_RECEIVE_ID_TYPE,
        )
        if not (FEISHU_APP_ID and FEISHU_APP_SECRET and FEISHU_DEFAULT_RECEIVE_ID):
            return
        token = _get_tenant_access_token(
            app_id=FEISHU_APP_ID, app_secret=FEISHU_APP_SECRET,
            api_base_url=FEISHU_API_BASE_URL,
        )
        focus = evolve_result.get("total_focus_fields", 0)
        converged = evolve_result.get("converged", False)
        card = {
            "schema": "2.0",
            "config": {"enable_forward": True, "width_mode": "fill"},
            "header": {
                "title": {"tag": "plain_text", "content": f"循环暂停 — {user_name}"},
                "template": "grey",
            },
            "body": {"elements": [{
                "tag": "markdown",
                "content": (
                    f"本轮 evolve 未产出新提案（{focus} 个焦点字段均未达到提案置信度阈值）。\n\n"
                    f"循环暂停，等待下次手动触发或新数据输入。\n\n"
                    f"[查看详情](http://localhost:5180/user/{user_name}?date={date_str})"
                ),
            }]},
        }
        _requests.post(
            f"{FEISHU_API_BASE_URL}/open-apis/im/v1/messages?receive_id_type={FEISHU_DEFAULT_RECEIVE_ID_TYPE}",
            json={"receive_id": FEISHU_DEFAULT_RECEIVE_ID, "msg_type": "interactive",
                  "content": json.dumps(card, ensure_ascii=False)},
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
            timeout=15,
        )
        print(f"  [飞书] 暂停通知已发送")
    except Exception as exc:
        print(f"  [飞书] 暂停通知发送失败: {exc}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _discover_reflection_users() -> List[str]:
    """从 memory/reflection/case_facts_*.jsonl 发现用户。"""
    if not REFLECTION_DIR.exists():
        return []
    return sorted(set(
        f.stem.replace("case_facts_", "")
        for f in REFLECTION_DIR.glob("case_facts_*.jsonl")
    ))


def _print_summary(phase: str, results: List[Dict[str, Any]]) -> None:
    ok = sum(1 for r in results if r["status"] == "ok")
    skip = sum(1 for r in results if r["status"] == "skip")
    err = sum(1 for r in results if r["status"] == "error")
    print(f"\n[{phase}] 汇总: {ok} 成功, {skip} 跳过, {err} 失败")
    for r in results:
        if r["status"] == "error":
            print(f"  [error] {r['user']}: {r.get('reason', '')}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI 入口
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hardness Engineering 统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
子命令:
  run      首次运行 pipeline → GT 对比 → 捕获 badcase
  evolve   单人多轮优化（夜间自回归）
  harness  跨用户 pattern 聚类 + 策略建议
  full     一键全链路 (run → evolve → harness)

示例:
  python scripts/run_hardness_pipeline.py run --user youruixun
  python scripts/run_hardness_pipeline.py run --all
  python scripts/run_hardness_pipeline.py evolve --all --date 2026-03-28
  python scripts/run_hardness_pipeline.py harness
  python scripts/run_hardness_pipeline.py full --all
""",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── run ──
    p_run = subparsers.add_parser("run", help="首次运行 + GT 对比")
    p_run.add_argument("--user", type=str, help="用户名")
    p_run.add_argument("--all", action="store_true", help="所有 datasets/ 用户")
    p_run.add_argument(
        "--profile-model",
        default="deepseek/deepseek-chat-v3-0324",
        help="LP3 画像模型",
    )
    p_run.add_argument("--profile-openrouter-key", default=None, help="LP3 OpenRouter key")

    # ── reflect ──
    p_reflect = subparsers.add_parser("reflect", help="反思 agent 分析（独立于 run，较慢）")
    p_reflect.add_argument("--user", type=str, help="用户名")
    p_reflect.add_argument("--all", action="store_true", help="所有 datasets/ 用户")

    # ── evolve ──
    p_evolve = subparsers.add_parser("evolve", help="单人多轮优化")
    p_evolve.add_argument("--user", type=str, help="用户名")
    p_evolve.add_argument("--all", action="store_true", help="所有 datasets/ 用户")
    p_evolve.add_argument("--date", type=str, default=None, help="日期 (默认今天)")
    p_evolve.add_argument("--top-k-fields", type=int, default=3, help="聚焦字段数")
    p_evolve.add_argument("--skip-gt-check", action="store_true", help="跳过 GT 人工校准检查")

    # ── harness ──
    p_harness = subparsers.add_parser("harness", help="跨用户 Harness Engineering")
    p_harness.add_argument("--users", type=str, default=None, help="逗号分隔用户名 (空=自动发现)")
    p_harness.add_argument("--all", action="store_true", help="所有 datasets/ 用户")

    # ── full ──
    p_full = subparsers.add_parser("full", help="一键全链路")
    p_full.add_argument("--user", type=str, help="用户名")
    p_full.add_argument("--all", action="store_true", help="所有 datasets/ 用户")
    p_full.add_argument(
        "--profile-model",
        default="deepseek/deepseek-chat-v3-0324",
        help="LP3 画像模型",
    )
    p_full.add_argument("--profile-openrouter-key", default=None, help="LP3 OpenRouter key")
    p_full.add_argument("--date", type=str, default=None, help="日期 (默认今天)")
    p_full.add_argument("--top-k-fields", type=int, default=3, help="聚焦字段数")

    # ── autoloop ──
    p_auto = subparsers.add_parser("autoloop", help="自动循环 (run → evolve 多轮 → harness → 审批)")
    p_auto.add_argument("--user", type=str, help="用户名")
    p_auto.add_argument("--all", action="store_true", help="所有 datasets/ 用户")
    p_auto.add_argument("--max-rounds", type=int, default=10, help="evolve 最大轮数 (默认 10)")
    p_auto.add_argument("--top-k-fields", type=int, default=3, help="聚焦字段数")
    p_auto.add_argument("--skip-gt-check", action="store_true", help="跳过 GT 人工校对检查")
    p_auto.add_argument(
        "--profile-model",
        default="deepseek/deepseek-chat-v3-0324",
        help="LP3 画像模型",
    )
    p_auto.add_argument("--profile-openrouter-key", default=None, help="LP3 OpenRouter key")
    p_auto.add_argument("--date", type=str, default=None, help="日期 (默认今天)")

    # ── rerun-fields ──
    p_rerun = subparsers.add_parser("rerun-fields", help="规则修改后只重跑指定字段")
    p_rerun.add_argument("--user", type=str, required=True, help="用户名")
    p_rerun.add_argument("--fields", type=str, required=True, help="逗号分隔的字段名")

    args = parser.parse_args()
    handlers = {
        "run": cmd_run,
        "reflect": cmd_reflect,
        "evolve": cmd_evolve,
        "harness": cmd_harness,
        "full": cmd_full,
        "autoloop": cmd_autoloop,
        "rerun-fields": cmd_rerun_fields,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
