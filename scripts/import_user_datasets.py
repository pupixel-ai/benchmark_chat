#!/usr/bin/env python3
"""
将 Desktop/data/output_user_v1 中的用户测试数据导入到项目 datasets/ 目录。

datasets/ 结构:
  {user_name}/
    manifest.json          — 数据集元信息
    source/                — 不可变预计算输入 (face + vlm + lp1)
      face_recognition_output.json
      vlm_cache.json
      lp1_events.json
    gt/                    — GT 真值
      profile_field_gt.jsonl
    runs/                  — 版本化运行记录 (由 pipeline 写入)

用法:
  python scripts/import_user_datasets.py
  python scripts/import_user_datasets.py --dry-run
  python scripts/import_user_datasets.py --users youruixun,lijia
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_SOURCE = Path(os.getenv("DATA_SOURCE", os.path.expanduser("~/Desktop/data/output_user_v1")))
DATASETS_DIR = PROJECT_ROOT / "datasets"

# 用户名映射: 目录名前缀 -> 干净的 user_name
USER_MAP = {
    "01_chenqiantong": "chenqiantong",
    "02_wangzhichao": "wangzhichao",
    "03_guohaoyan": "guohaoyan",
    "04_caizhengsheng": "caizhengsheng",
    "05_chenyan_incomplete": "chenyan",
    "06_zhangweibang": "zhangweibang",
    "07_zhangchengyang": "zhangchengyang",
    "08_chenmeiyi": "chenmeiyi",
    "09_lijia": "lijia",
    "10_youruixun": "youruixun",
    "11_zhengyilang": "zhengyilang",
}

# GT 目录单独存在的情况
GT_OVERRIDE_DIRS = {
    "youruixun": "10_youruixun_gt",
}


def _find_source_file(src_dir: Path, patterns: list[str]) -> Path | None:
    """在 src_dir 中按优先级查找匹配文件。"""
    for pattern in patterns:
        matches = list(src_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def _find_gt_file(src_dir: Path, prefix: str, gt_dir: Path | None) -> Path | None:
    """查找 GT jsonl 文件，优先级: gt_dir > src_dir。"""
    # 优先从 GT override 目录找
    if gt_dir and gt_dir.exists():
        # 优先找 profile_field_gt_*.jsonl (反思系统格式)
        for f in gt_dir.glob("profile_field_gt_*.jsonl"):
            return f
        # 其次找 XX_gt.jsonl
        for f in gt_dir.glob("*_gt.jsonl"):
            return f

    # 从源目录找
    for f in src_dir.glob("*_gt.jsonl"):
        return f
    return None


def import_user(
    dir_prefix: str,
    user_name: str,
    *,
    dry_run: bool = False,
) -> dict:
    """导入单个用户的数据集。返回导入状态摘要。"""
    src_dir = DATA_SOURCE / dir_prefix
    if not src_dir.exists():
        return {"user": user_name, "status": "skip", "reason": f"源目录不存在: {src_dir}"}

    dest = DATASETS_DIR / user_name
    result = {"user": user_name, "status": "ok", "files": {}}

    # ─── source 文件查找 ─────────────────────────────────────────
    face_file = _find_source_file(src_dir, [
        "face_recognition_output.json",
        f"{dir_prefix}_face_recognition_output.json",
        "face/face_recognition_output.json",
    ])
    vlm_file = _find_source_file(src_dir, [
        "vlm_cache.json",
        f"{dir_prefix}_vlm_cache.json",
        "vp1_observations.json",
        "vlm/vp1_observations.json",
    ])
    events_file = _find_source_file(src_dir, [
        "lp1_events_compact.json",
        f"{dir_prefix}_events.json",
        "lp1/lp1_events_compact.json",
    ])

    if not face_file:
        return {"user": user_name, "status": "skip", "reason": "缺少 face_recognition_output"}
    if not events_file:
        return {"user": user_name, "status": "skip", "reason": "缺少 events 文件"}

    # ─── GT 查找 ─────────────────────────────────────────────────
    gt_override = GT_OVERRIDE_DIRS.get(user_name)
    gt_dir = DATA_SOURCE / gt_override if gt_override else None
    gt_file = _find_gt_file(src_dir, dir_prefix, gt_dir)

    # ─── 执行复制 ────────────────────────────────────────────────
    source_dir = dest / "source"
    gt_dest = dest / "gt"
    runs_dir = dest / "runs"

    if dry_run:
        result["files"] = {
            "face": str(face_file) if face_file else None,
            "vlm": str(vlm_file) if vlm_file else None,
            "events": str(events_file) if events_file else None,
            "gt": str(gt_file) if gt_file else None,
        }
        result["status"] = "dry_run"
        return result

    source_dir.mkdir(parents=True, exist_ok=True)
    gt_dest.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    def _copy(src: Path, dst: Path) -> None:
        shutil.copy2(src, dst)
        result["files"][dst.name] = str(src)

    _copy(face_file, source_dir / "face_recognition_output.json")
    if vlm_file:
        _copy(vlm_file, source_dir / "vlm_cache.json")
    _copy(events_file, source_dir / "lp1_events.json")

    if gt_file:
        _copy(gt_file, gt_dest / "profile_field_gt.jsonl")

    # ─── manifest ────────────────────────────────────────────────
    manifest = {
        "user_name": user_name,
        "source_dir_prefix": dir_prefix,
        "imported_at": datetime.now().isoformat(),
        "source_origin": str(DATA_SOURCE),
        "has_vlm": vlm_file is not None,
        "has_gt": gt_file is not None,
        "source_files": {
            "face": face_file.name if face_file else None,
            "vlm": vlm_file.name if vlm_file else None,
            "events": events_file.name if events_file else None,
            "gt": gt_file.name if gt_file else None,
        },
    }
    manifest_path = dest / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    result["files"]["manifest.json"] = "generated"

    return result


def main():
    parser = argparse.ArgumentParser(description="导入用户测试数据到 datasets/")
    parser.add_argument("--dry-run", action="store_true", help="只显示将要执行的操作，不实际复制")
    parser.add_argument("--users", type=str, default="", help="逗号分隔的用户名列表，空=全部")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的数据集")
    args = parser.parse_args()

    if not DATA_SOURCE.exists():
        print(f"[error] 数据源目录不存在: {DATA_SOURCE}")
        return

    filter_users = set(args.users.split(",")) if args.users else set()

    results = []
    for dir_prefix, user_name in sorted(USER_MAP.items()):
        if filter_users and user_name not in filter_users:
            continue

        dest = DATASETS_DIR / user_name
        if dest.exists() and not args.force and not args.dry_run:
            results.append({"user": user_name, "status": "skip", "reason": "已存在（使用 --force 覆盖）"})
            continue

        if dest.exists() and args.force and not args.dry_run:
            shutil.rmtree(dest)

        result = import_user(dir_prefix, user_name, dry_run=args.dry_run)
        results.append(result)

    # ─── 打印结果 ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"数据集导入{'预览' if args.dry_run else '完成'}")
    print(f"{'=' * 60}")
    for r in results:
        status = r["status"]
        user = r["user"]
        if status == "ok":
            files = r.get("files", {})
            print(f"  [ok] {user}: {len(files)} 文件")
        elif status == "dry_run":
            files = r.get("files", {})
            present = [k for k, v in files.items() if v]
            missing = [k for k, v in files.items() if not v]
            print(f"  [dry] {user}: 有 {present}, 缺 {missing}")
        else:
            print(f"  [{status}] {user}: {r.get('reason', '')}")
    print()


if __name__ == "__main__":
    main()
