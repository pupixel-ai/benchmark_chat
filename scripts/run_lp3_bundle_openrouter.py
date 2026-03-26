#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.memory_pipeline.precomputed_loader import load_precomputed_memory_state
from services.memory_pipeline.profile_fields import build_profile_context, generate_structured_profile
from services.memory_pipeline.profile_llm import OpenRouterProfileLLMProcessor
from utils import save_json


def main() -> int:
    parser = argparse.ArgumentParser(description="使用 OpenRouter/Gemini 对 bundle 运行 LP3-only 画像")
    parser.add_argument("--bundle-dir", required=True, help="bundle 根目录")
    parser.add_argument("--output-dir", default=None, help="输出目录")
    args = parser.parse_args()

    bundle_path = Path(args.bundle_dir)
    if not bundle_path.exists():
        print(f"❌ bundle 不存在: {bundle_path}")
        return 1

    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = bundle_path / f"lp3_openrouter_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path.mkdir(parents=True, exist_ok=True)

    state = load_precomputed_memory_state(bundle_path)
    state.profile_context = build_profile_context(state)
    primary_person_id = (state.primary_decision or {}).get("primary_person_id")
    llm_processor = OpenRouterProfileLLMProcessor(primary_person_id=primary_person_id)
    result = generate_structured_profile(state, llm_processor=llm_processor)

    structured = result.get("structured", {})
    field_decisions = result.get("field_decisions", [])
    consistency = result.get("consistency", {})

    save_json(structured, str(output_path / "structured_profile.json"))
    save_json(field_decisions, str(output_path / "field_decisions.json"))
    save_json(consistency, str(output_path / "consistency.json"))
    save_json(_build_state_snapshot(state), str(output_path / "normalized_state_snapshot.json"))
    save_json(_build_mapping_debug(bundle_path, state), str(output_path / "mapping_debug.json"))

    print(f"[LP3 bundle] 输入: {bundle_path}")
    print(f"[LP3 bundle] 输出: {output_path}")
    print(f"[LP3 bundle] 主角: {primary_person_id}")
    print(f"[LP3 bundle] VLM条数: {len(state.vlm_results)}")
    print(f"[LP3 bundle] 事件条数: {len(state.events)}")
    print(f"[LP3 bundle] 非空字段数: {_count_non_null_fields(structured)}")
    return 0


def _build_state_snapshot(state) -> Dict[str, Any]:
    return {
        "primary_person_id": (state.primary_decision or {}).get("primary_person_id"),
        "face_db_count": len(state.face_db or {}),
        "vlm_result_count": len(state.vlm_results or []),
        "event_count": len(state.events or []),
        "vlm_results": list(state.vlm_results or []),
        "events": [
            {
                "event_id": event.event_id,
                "date": event.date,
                "time_range": event.time_range,
                "title": event.title,
                "participants": list(event.participants or []),
                "location": event.location,
                "description": event.description,
                "photo_count": event.photo_count,
                "meta_info": dict(event.meta_info or {}),
                "objective_fact": dict(event.objective_fact or {}),
            }
            for event in state.events or []
        ],
    }


def _build_mapping_debug(bundle_path: Path, state) -> Dict[str, Any]:
    face_payload = json.loads((bundle_path / "face" / "face_recognition_output.json").read_text(encoding="utf-8"))
    bundle_primary_person_id = face_payload.get("primary_person_id") or face_payload.get("face_recognition", {}).get("primary_person_id")
    return {
        "bundle_primary_person_id": bundle_primary_person_id,
        "canonical_primary_person_id": (state.primary_decision or {}).get("primary_person_id"),
        "event_trace_mapping": {
            event.event_id: dict((event.meta_info or {}).get("trace", {}) or {})
            for event in state.events or []
        },
        "vp1_extra_mapping": {
            item.get("photo_id"): {
                "face_person_ids": list(item.get("face_person_ids", []) or []),
                "media_kind": item.get("media_kind"),
                "is_reference_like": item.get("is_reference_like"),
                "sequence_index": item.get("sequence_index"),
                "ocr_hits": list((item.get("vlm_analysis", {}) or {}).get("ocr_hits", []) or []),
                "brands": list((item.get("vlm_analysis", {}) or {}).get("brands", []) or []),
                "place_candidates": list((item.get("vlm_analysis", {}) or {}).get("place_candidates", []) or []),
            }
            for item in state.vlm_results or []
        },
    }


def _count_non_null_fields(payload: Any) -> int:
    if isinstance(payload, dict):
        if {"value", "confidence", "evidence", "reasoning"} <= set(payload.keys()):
            return 1 if payload.get("value") is not None else 0
        return sum(_count_non_null_fields(value) for value in payload.values())
    if isinstance(payload, list):
        return sum(_count_non_null_fields(item) for item in payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
