#!/usr/bin/env python3
"""
Resume a failed v0323 task from the post-VLM boundary by reusing saved VP1/VLM artifacts.
"""
from __future__ import annotations

import argparse
import copy
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.artifact_store import ArtifactCatalogStore, build_task_asset_manifest
from backend.db import SessionLocal
from backend.models import TaskRecord
from backend.progress_utils import append_terminal_error, append_terminal_info, merge_stage_progress
from backend.task_store import TaskStore, normalize_task_options
from config import TASK_VERSION_V0323
from services.asset_store import TaskAssetStore
from services.llm_processor import LLMProcessor
from services.pipeline_service import MemoryPipelineService
from services.v0323.pipeline import V0323PipelineFamily
from services.vlm_analyzer import VLMAnalyzer
from utils import load_json, save_json


def _load_task_record(task_id: str) -> TaskRecord:
    with SessionLocal() as session:
        record = session.get(TaskRecord, task_id)
        if record is None:
            raise KeyError(f"任务不存在: {task_id}")
        session.expunge(record)
        return record


def _build_progress_callback(task_store: TaskStore, task_id: str, progress_state_holder: Dict[str, Dict[str, Any]]):
    def _callback(stage: str, payload: Dict[str, Any]) -> None:
        progress_state_holder["value"] = merge_stage_progress(progress_state_holder["value"], stage, payload)
        task_store.update_task(
            task_id,
            status="running",
            stage=stage,
            progress=copy.deepcopy(progress_state_holder["value"]),
            error=None,
        )

    return _callback


def _archive_path(path: Path, *, suffix: str) -> None:
    if not path.exists():
        return
    archived = path.with_name(f"{path.name}.{suffix}")
    if archived.exists():
        if archived.is_dir():
            shutil.rmtree(archived)
        else:
            archived.unlink()
    shutil.move(str(path), str(archived))


def _archive_previous_outputs(service: MemoryPipelineService) -> None:
    timestamp = datetime.now().strftime("resume_backup_%Y%m%d%H%M%S")
    _archive_path(service.task_dir / "v0323", suffix=timestamp)
    _archive_path(service.output_dir / "result.json", suffix=timestamp)


def _restore_cached_face_stage(service: MemoryPipelineService, photos: List[Any]) -> Tuple[List[Any], str | None, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    images = service.face_recognition.state.get("images", {}) or {}
    by_photo_id: Dict[str, Dict[str, Any]] = {}
    by_source_hash: Dict[str, Dict[str, Any]] = {}
    for image in images.values():
        photo_id = str(image.get("photo_id") or "").strip()
        source_hash = str(image.get("source_hash") or "").strip()
        if photo_id:
            by_photo_id[photo_id] = image
        if source_hash:
            by_source_hash[source_hash] = image

    primary_person_id = service.face_recognition.get_primary_person_id()
    face_ready_photos: List[Any] = []
    for photo in photos:
        cached = by_photo_id.get(photo.photo_id)
        if cached is None and photo.source_hash:
            cached = by_source_hash.get(photo.source_hash)
        if cached is None:
            continue
        photo.faces = [dict(face) for face in cached.get("faces", [])]
        photo.face_image_hash = cached.get("image_hash")
        photo.source_hash = cached.get("source_hash", photo.source_hash)
        photo.primary_person_id = primary_person_id
        boxed_filename = f"{Path(photo.filename).stem}_boxed.webp"
        boxed_path = Path(service.image_processor.boxed_dir) / boxed_filename
        if boxed_path.exists():
            photo.boxed_path = str(boxed_path)
        face_ready_photos.append(photo)

    face_output = service.face_recognition.get_face_output()
    face_payload = service._build_face_recognition_payload(photos, face_output)
    face_db = service.face_recognition.get_all_persons()
    return face_ready_photos, primary_person_id, face_output, face_payload, face_db


def _hydrate_cached_visual_assets(service: MemoryPipelineService, photos: List[Any]) -> List[Any]:
    missing_compressed: List[Any] = []
    for photo in photos:
        compressed_path = Path(service.image_processor.compress_dir) / f"compressed_{photo.photo_id}.webp"
        if compressed_path.exists():
            photo.compressed_path = str(compressed_path)
        else:
            missing_compressed.append(photo)

        if not getattr(photo, "boxed_path", None):
            boxed_path = Path(service.image_processor.boxed_dir) / f"{Path(photo.filename).stem}_boxed.webp"
            if boxed_path.exists():
                photo.boxed_path = str(boxed_path)

    if missing_compressed:
        service.image_processor.preprocess(missing_compressed)
    return photos


def _load_or_rebuild_observations(
    *,
    service: MemoryPipelineService,
    face_ready_photos: Sequence[Any],
    cached_photo_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    vp1_path = service.task_dir / "v0323" / "vp1_observations.json"
    if vp1_path.exists():
        observations = load_json(str(vp1_path))
        if isinstance(observations, list) and observations:
            return [dict(item) for item in observations if isinstance(item, dict)]

    vlm = VLMAnalyzer(cache_path=str(service.vlm_cache_path), task_version=service.task_version)
    if not vlm.load_cache() or not vlm.results:
        raise RuntimeError("没有找到可复用的 VP1/VLM 缓存，无法从 VLM 边界续跑")

    cached_photo_ids_set = {str(item) for item in list(cached_photo_ids or []) if str(item).strip()}
    aligned_vlm_results = []
    vlm_by_photo_id = {
        str(item.get("photo_id") or ""): item
        for item in vlm.results
        if isinstance(item, dict) and str(item.get("photo_id") or "").strip()
    }
    for photo in face_ready_photos:
        cached = vlm_by_photo_id.get(str(getattr(photo, "photo_id", "") or ""))
        if not cached:
            continue
        aligned_vlm_results.append(cached)
        cached_photo_ids_set.add(photo.photo_id)
    if not aligned_vlm_results:
        raise RuntimeError("VLM 缓存与当前任务照片无法对齐，续跑已终止")

    family = V0323PipelineFamily(
        task_id=service.task_id,
        task_dir=service.task_dir,
        user_id=service.user_id,
        asset_store=service.asset_store,
        llm_processor=LLMProcessor(task_version=service.task_version),
        public_url_builder=service._public_url,
    )
    observations = family._build_vp1_observations(list(face_ready_photos), aligned_vlm_results)
    return [dict(item) for item in observations]


def _build_v0323_result(
    *,
    service: MemoryPipelineService,
    photos: Sequence[Any],
    face_ready_photos: Sequence[Any],
    face_output: Dict[str, Any],
    face_payload: Dict[str, Any],
    primary_person_id: str | None,
    memory: Dict[str, Any],
) -> Dict[str, Any]:
    lp1_events = list(memory.get("lp1_events", []) or [])
    lp2_relationships = list(memory.get("lp2_relationships", []) or [])
    profile_payload = dict(memory.get("lp3_profile", {}) or {})
    profile_markdown = str(profile_payload.get("report_markdown") or "")
    return {
        "task_id": service.task_id,
        "version": service.task_version,
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "task_version": service.task_version,
            "pipeline_family": memory.get("pipeline_family"),
            "total_uploaded": service._count_uploaded_files(),
            "loaded_images": len(list(photos or [])),
            "failed_images": len(service.failed_images),
            "face_processed_images": len(list(face_ready_photos or [])),
            "vlm_processed_images": len(memory.get("vp1_observations", []) or []),
            "total_faces": face_output.get("metrics", {}).get("total_faces", 0),
            "total_persons": face_output.get("metrics", {}).get("total_persons", 0),
            "primary_person_id": primary_person_id,
            "event_count": len(lp1_events),
            "fact_count": len(lp1_events),
            "relationship_count": len(lp2_relationships),
            "observation_count": len(memory.get("vp1_observations", []) or []),
            "claim_count": 0,
            "profile_delta_count": len(dict(profile_payload.get("structured") or {})),
            "profile_version": 1 if profile_payload else 0,
            "profile_generation_mode": memory.get("summary", {}).get("profile_generation_mode"),
        },
        "face_recognition": face_payload,
        "face_report": service._build_face_report(face_payload),
        "failed_images": service.failed_images,
        "warnings": service.warnings,
        "facts": lp1_events,
        "relationships": lp2_relationships,
        "profile_markdown": profile_markdown,
        "memory_contract": None,
        "llm_chunk_artifacts": {},
        "dedupe_report": load_json(str(service.dedupe_report_path)) if service.dedupe_report_path.exists() else {},
        "memory": memory,
        "artifacts": {
            "result_url": service._public_url(service.output_dir / "result.json"),
            "face_output_url": service._public_url(service.cache_dir / "face_recognition_output.json"),
            "vlm_cache_url": service._public_url(service.vlm_cache_path),
            "vlm_failures_url": service._public_url(service.vlm_failures_path) if service.vlm_failures_path.exists() else None,
            "dedupe_report_url": service._public_url(service.dedupe_report_path) if service.dedupe_report_path.exists() else None,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume a v0323 task from VP1/VLM outputs.")
    parser.add_argument("task_id", help="Target task id")
    args = parser.parse_args()

    record = _load_task_record(args.task_id)
    if str(record.version or "").strip() != TASK_VERSION_V0323:
        raise RuntimeError(f"任务 {record.task_id} 不是 v0323，当前版本为 {record.version}")

    task_store = TaskStore()
    asset_store = TaskAssetStore()
    artifact_catalog = ArtifactCatalogStore()
    service = MemoryPipelineService(
        task_id=record.task_id,
        task_dir=record.task_dir,
        asset_store=asset_store,
        user_id=record.user_id,
        task_version=record.version or TASK_VERSION_V0323,
        task_options=normalize_task_options(record.options),
    )
    service._clear_legacy_outputs_for_revision_first_family()

    progress_state_holder: Dict[str, Dict[str, Any]] = {"value": copy.deepcopy(record.progress or {})}
    progress_state_holder["value"] = merge_stage_progress(
        progress_state_holder["value"],
        "resume",
        {
            "message": "从 VLM 边界重跑 v0323 LP1/LP2/LP3",
            "reuse_cached_vlm": True,
            "reuse_cached_face": True,
        },
    )
    progress_callback = _build_progress_callback(task_store, record.task_id, progress_state_holder)
    task_store.update_task(
        record.task_id,
        status="running",
        stage="resume",
        progress=copy.deepcopy(progress_state_holder["value"]),
        error=None,
    )

    try:
        print(f"[resume_v0323] start task={record.task_id}", flush=True)

        service._notify(progress_callback, "loading", {"message": "重建照片元数据（复用缓存）"})
        photos, load_errors = service.image_processor.load_photos_with_errors(str(service.upload_dir))
        service.failed_images = list(service.initial_upload_failures)
        service.failed_images.extend(load_errors)
        photos = service.image_processor.convert_to_jpeg(
            photos,
            normalize_live_photos=bool(service.task_options.get("normalize_live_photos")),
        )
        photos = service._apply_image_policies(photos)
        photos = service.image_processor.dedupe_before_face_recognition(photos)
        dedupe_report = load_json(str(service.dedupe_report_path)) if service.dedupe_report_path.exists() else {}
        if not dedupe_report:
            dedupe_report = getattr(service.image_processor, "last_dedupe_report", {}) or {}
            if dedupe_report:
                save_json(dedupe_report, str(service.dedupe_report_path))
        print(f"[resume_v0323] photos_after_dedupe={len(photos)}", flush=True)

        face_ready_photos, primary_person_id, face_output, face_payload, _face_db = _restore_cached_face_stage(service, photos)
        if not face_ready_photos:
            raise RuntimeError("没有找到可复用的人脸识别缓存，无法从 VLM 边界续跑")
        _hydrate_cached_visual_assets(service, face_ready_photos)
        service._notify(
            progress_callback,
            "face_recognition",
            {
                "message": "复用已完成人脸识别缓存",
                "completed": True,
                "processed": len(face_ready_photos),
                "input_photo_count": len(photos),
                "percent": 100,
                "runtime_seconds": 0.0,
                "face_result_preview": service._build_face_stage_preview(face_payload),
                "face_output_url": service._public_url(service.cache_dir / "face_recognition_output.json"),
            },
        )

        cached_photo_ids = [str(getattr(photo, "photo_id", "") or "") for photo in face_ready_photos if str(getattr(photo, "photo_id", "") or "").strip()]
        observations = _load_or_rebuild_observations(
            service=service,
            face_ready_photos=face_ready_photos,
            cached_photo_ids=cached_photo_ids,
        )
        print(f"[resume_v0323] observations={len(observations)}", flush=True)
        service._notify(
            progress_callback,
            "vlm",
            {
                "message": "复用已完成的 VP1/VLM 结果",
                "photo_count": len(face_ready_photos),
                "processed": len(observations),
                "cached_hits": len(cached_photo_ids),
                "queued": 0,
                "in_flight": 0,
                "completed_count": len(observations),
                "failed_count": 0,
                "retry_count": 0,
                "flush_count": 0,
                "concurrency": 0,
                "avg_latency_seconds": 0.0,
                "percent": 100,
                "runtime_seconds": 0.0,
            },
        )
        _archive_previous_outputs(service)

        family = V0323PipelineFamily(
            task_id=service.task_id,
            task_dir=service.task_dir,
            user_id=service.user_id,
            asset_store=service.asset_store,
            llm_processor=LLMProcessor(task_version=service.task_version),
            public_url_builder=service._public_url,
        )
        memory = family.run_from_observations(
            observations=observations,
            face_output=face_output,
            primary_person_id=primary_person_id,
            cached_photo_ids=cached_photo_ids,
            dedupe_report=dedupe_report,
            progress_callback=lambda stage, payload: service._notify(
                progress_callback,
                "memory" if stage == "v0323" else stage,
                payload,
            ),
        )

        detailed_output = _build_v0323_result(
            service=service,
            photos=photos,
            face_ready_photos=face_ready_photos,
            face_output=face_output,
            face_payload=face_payload,
            primary_person_id=primary_person_id,
            memory=memory,
        )
        for artifact_key, artifact_value in memory.get("artifacts", {}).items():
            if artifact_key.endswith("_url"):
                detailed_output["artifacts"][artifact_key] = artifact_value
        save_json(detailed_output, str(service.output_dir / "result.json"))
        asset_store.sync_task_directory(service.task_id, service.task_dir)

        completed_progress = append_terminal_info(
            progress_state_holder["value"],
            stage="completed",
            message="任务已从 VLM 边界重跑完成",
        )
        task_store.update_task(
            service.task_id,
            status="completed",
            stage="completed",
            result=detailed_output,
            error=None,
            progress=copy.deepcopy(completed_progress),
        )
        manifest = build_task_asset_manifest(service.task_id, service.task_dir, asset_store)
        task_store.set_result_summary(service.task_id, detailed_output.get("summary"), manifest)
        if record.user_id:
            artifact_catalog.replace_task_artifacts(service.task_id, record.user_id, manifest)
        print("[resume_v0323] completed", flush=True)
        return 0
    except Exception as exc:
        failed_progress = append_terminal_error(progress_state_holder["value"], stage="failed", error=str(exc))
        task_store.update_task(
            record.task_id,
            status="failed",
            stage="failed",
            error=str(exc),
            progress=copy.deepcopy(failed_progress),
        )
        print(f"[resume_v0323] failed error={exc}", flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
