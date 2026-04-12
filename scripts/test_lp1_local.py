#!/usr/bin/env python3
"""
本地 LP1 事件生成测试脚本
链路: 照片 → 人脸识别 → VLM → VP1 observations → LP1 事件
跳过 LP2 关系提取 / LP3 画像生成

用法:
  python scripts/test_lp1_local.py --photos /path/to/photos
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from config import TASK_VERSION_V0325
from services.image_processor import ImageProcessor
from services.face_recognition import FaceRecognition
from services.vlm_analyzer import VLMAnalyzer
from services.llm_processor import LLMProcessor
from services.v0325.pipeline import V0325PipelineFamily
from utils import save_json


def _dummy_url(path: Path) -> str:
    return path.as_uri()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photos", required=True, help="照片目录")
    parser.add_argument("--max-photos", type=int, default=30, help="最多处理照片数")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output" / "lp1_test"), help="输出目录")
    args = parser.parse_args()

    print(f"\n=== LP1 本地测试 ===")
    print(f"Task version: {TASK_VERSION_V0325}")
    print(f"照片目录: {args.photos}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_dir = output_dir / "task"
    task_dir.mkdir(parents=True, exist_ok=True)

    # ── [1] 加载照片 ─────────────────────────────────────
    print("\n[1/5] 加载照片...")
    image_processor = ImageProcessor()
    photos = image_processor.load_photos(args.photos, max_photos=args.max_photos)
    print(f"  加载 {len(photos)} 张")

    if not photos:
        print("  ❌ 没有找到照片，退出")
        return

    # ── [2] HEIC 转换 ────────────────────────────────────
    print("\n[2/5] HEIC 转换...")
    photos = image_processor.convert_to_jpeg(photos)
    photos = image_processor.dedupe_before_face_recognition(photos)
    print(f"  去重后 {len(photos)} 张")

    # ── [3] 人脸识别 ─────────────────────────────────────
    print("\n[3/5] 人脸识别...")
    face_rec = FaceRecognition()
    for photo in photos:
        face_rec.process_photo(photo)
    face_rec.reorder_protagonist(photos)
    primary_person_id = face_rec.get_primary_person_id()
    face_db = face_rec.get_all_persons()
    face_output = face_rec.get_face_output()
    print(f"  主角: {primary_person_id or '未识别'}  |  人物数: {len(face_db)}")

    # 压缩照片（VLM 用）
    photos = image_processor.preprocess(photos)

    # ── [4] VLM 分析 ─────────────────────────────────────
    print(f"\n[4/5] VLM 分析（{len(photos)} 张）...")
    vlm = VLMAnalyzer(
        cache_path=str(task_dir / "vlm_cache.json"),
        task_version=TASK_VERSION_V0325,
    )
    print(f"  VLM provider={vlm.provider}, model={vlm.model}")

    vlm_ok = vlm_fail = 0
    for i, photo in enumerate(photos, 1):
        try:
            result = vlm.analyze_photo(photo, face_db, primary_person_id)
            if result:
                vlm.add_result(photo, result)
                vlm_ok += 1
            else:
                vlm_fail += 1
        except Exception as exc:
            print(f"  ⚠ [{i}] {photo.filename}: {exc}")
            vlm_fail += 1

    vlm.save_cache()
    print(f"  成功 {vlm_ok} / 失败 {vlm_fail}")

    if not vlm.results:
        print("  ❌ VLM 全部失败，退出")
        return

    # ── [5] LP1 事件生成 ─────────────────────────────────
    print(f"\n[5/5] LP1 事件生成...")
    llm = LLMProcessor(
        task_version=TASK_VERSION_V0325,
        relationship_provider_override="openrouter",  # 跳过 bedrock，测试不需要 LP2
    )
    print(f"  LLM provider={llm.provider}, model={llm.model}")

    from services.v0325.pipeline import V0325PipelineFamily
    from utils import save_json

    # Stub asset store（本地测试不需要 S3/botocore）
    class _LocalAssetStore:
        def upload_file(self, *a, **kw): pass
        def sync_task_directory(self, *a, **kw): pass
        def get_public_url(self, *a, **kw): return ""

    asset_store = _LocalAssetStore()

    family = V0325PipelineFamily(
        task_id="lp1_test",
        task_dir=task_dir,
        user_id=None,
        asset_store=asset_store,
        llm_processor=llm,
        public_url_builder=_dummy_url,
    )

    # 构建 VP1 observations
    observations = family._build_vp1_observations(photos, vlm.results)
    print(f"  VP1 observations: {len(observations)} 条")

    save_json(observations, str(output_dir / "vp1_observations.json"))

    # 初始化 observation_index / photo_order_index（正常跑 run() 时由其设置，直接调用 _run_lp1_batches 时需手动设置）
    family.observation_index = {obs["photo_id"]: obs for obs in observations}
    family.photo_order_index = {obs["photo_id"]: int(obs["sequence_index"]) for obs in observations}

    # 跑 LP1 batches（只跑事件生成，不跑 LP2/LP3）
    t0 = time.perf_counter()
    lp1_result = family._run_lp1_batches(
        observations,
        fallback_primary_person_id=primary_person_id,
    )
    elapsed = time.perf_counter() - t0

    events_raw = lp1_result.get("lp1_events_raw", [])
    events_compact = lp1_result.get("lp1_events", [])

    print(f"\n  ✅ LP1 完成，耗时 {elapsed:.1f}s")
    print(f"  raw events: {len(events_raw)}")
    print(f"  compact events: {len(events_compact)}")

    # 保存结果
    save_json(events_raw, str(output_dir / "lp1_events_raw.json"))
    save_json(events_compact, str(output_dir / "lp1_events_compact.json"))
    save_json(lp1_result, str(output_dir / "lp1_full_result.json"))

    # 打印事件摘要
    print("\n── 事件摘要 ──")
    for evt in events_compact[:20]:
        eid = evt.get("event_id", "?")
        title = evt.get("title") or evt.get("meta_info", {}).get("title", "")
        ts = evt.get("timestamp") or evt.get("meta_info", {}).get("timestamp", "")
        loc = evt.get("location_context") or evt.get("meta_info", {}).get("location_context", "")
        narrative = evt.get("narrative_synthesis", "")[:80]
        print(f"  [{eid}] {title} | {ts} | {loc}")
        if narrative:
            print(f"         {narrative}")

    print(f"\n输出目录: {output_dir}")
    print(f"  vp1_observations.json")
    print(f"  lp1_events_raw.json")
    print(f"  lp1_events_compact.json")


if __name__ == "__main__":
    main()
