#!/usr/bin/env python3
"""
Resume a failed task from the completed VLM boundary by reusing cached face/VLM artifacts
and rerunning only the LLM + memory materialization stages.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.artifact_store import ArtifactCatalogStore, build_task_asset_manifest
from backend.db import SessionLocal
from backend.models import TaskRecord
from backend.task_store import TaskStore
from config import TASK_VERSION_V0317_HEAVY
from services.asset_store import TaskAssetStore
from memory_module import MemoryModuleService
from services.llm_processor import LLMProcessor
from services.pipeline_service import MemoryPipelineService
from services.vlm_analyzer import VLMAnalyzer
from utils import load_json, save_json


def _load_task_record(task_id: str) -> TaskRecord:
    with SessionLocal() as session:
        record = session.get(TaskRecord, task_id)
        if record is None:
            raise KeyError(f"任务不存在: {task_id}")
        session.expunge(record)
        return record


def _build_progress_callback(task_store: TaskStore, task_id: str, progress_state: Dict[str, object]):
    def _callback(stage: str, payload: dict):
        stages = progress_state.setdefault("stages", {})
        if not isinstance(stages, dict):
            stages = {}
            progress_state["stages"] = stages
        stage_payload = copy.deepcopy(stages.get(stage) or {})
        stage_payload.update(copy.deepcopy(payload or {}))
        stage_payload["updated_at"] = datetime.now().isoformat()
        stages[stage] = stage_payload
        progress_state["current_stage"] = stage
        progress_state["updated_at"] = stage_payload["updated_at"]
        task_store.update_task(task_id, status="running", stage=stage, progress=copy.deepcopy(progress_state), error=None)

    return _callback


def _archive_partial_resume_outputs(service: MemoryPipelineService) -> None:
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    candidates = [
        service.llm_chunks_path,
        service.llm_contract_path,
        service.profile_report_path,
        service.output_dir / "result.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        archived = path.with_name(f"{path.stem}.resume_backup_{timestamp}{path.suffix}")
        path.replace(archived)
    memory_output_dir = service.output_dir / "memory"
    if memory_output_dir.exists():
        archived_dir = service.output_dir / f"memory.resume_backup_{timestamp}"
        memory_output_dir.replace(archived_dir)


def _restore_cached_face_stage(service: MemoryPipelineService, photos: List) -> Tuple[List, str | None, Dict, Dict, Dict]:
    images = service.face_recognition.state.get("images", {}) or {}
    by_photo_id: Dict[str, Dict] = {}
    by_source_hash: Dict[str, Dict] = {}
    for image in images.values():
        photo_id = str(image.get("photo_id") or "").strip()
        source_hash = str(image.get("source_hash") or "").strip()
        if photo_id:
            by_photo_id[photo_id] = image
        if source_hash:
            by_source_hash[source_hash] = image

    primary_person_id = service.face_recognition.get_primary_person_id()
    face_ready_photos = []
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


def _hydrate_cached_visual_assets(service: MemoryPipelineService, photos: List) -> List:
    missing_compressed: List = []
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume a failed task from cached VLM outputs.")
    parser.add_argument("task_id", help="Target task id")
    parser.add_argument("--max-photos", type=int, default=None, help="Optional cap when rebuilding photo metadata")
    args = parser.parse_args()

    record = _load_task_record(args.task_id)
    task_store = TaskStore()
    asset_store = TaskAssetStore()
    artifact_catalog = ArtifactCatalogStore()

    service = MemoryPipelineService(
        task_id=record.task_id,
        task_dir=record.task_dir,
        asset_store=asset_store,
        user_id=record.user_id,
        task_version=record.version,
    )

    progress_state: Dict[str, object] = copy.deepcopy(record.progress or {})
    if not isinstance(progress_state.get("stages"), dict):
        progress_state["stages"] = {}
    _archive_partial_resume_outputs(service)
    progress_state["stages"].pop("llm", None)
    progress_state["stages"].pop("memory", None)
    progress_state["current_stage"] = "resume"
    progress_state["updated_at"] = datetime.now().isoformat()
    progress_state["stages"]["resume"] = {
        "message": "从 VLM 边界续跑 LLM 与记忆落位",
        "completed": False,
        "updated_at": progress_state["updated_at"],
    }
    progress_callback = _build_progress_callback(task_store, record.task_id, progress_state)
    task_store.update_task(record.task_id, status="running", stage="resume", progress=copy.deepcopy(progress_state), error=None)

    try:
        print(f"[resume] start task={record.task_id}", flush=True)
        service._notify(progress_callback, "loading", {"message": "重建照片元数据（复用缓存）"})
        photos, load_errors = service.image_processor.load_photos_with_errors(
            str(service.upload_dir),
            max_photos=args.max_photos,
        )
        service.failed_images.extend(load_errors)
        print(f"[resume] loaded_photos={len(photos)} load_errors={len(load_errors)}", flush=True)
        photos = service.image_processor.convert_to_jpeg(photos)
        photos = service._apply_image_policies(photos)
        photos_to_process = service.image_processor.dedupe_before_face_recognition(photos)
        dedupe_report = load_json(str(service.dedupe_report_path)) if service.dedupe_report_path.exists() else {}
        if not dedupe_report:
            dedupe_report = getattr(service.image_processor, "last_dedupe_report", {}) or {}
            if dedupe_report:
                save_json(dedupe_report, str(service.dedupe_report_path))
        print(f"[resume] photos_after_dedupe={len(photos_to_process)}", flush=True)

        face_ready_photos, primary_person_id, face_output, face_payload, face_db = _restore_cached_face_stage(service, photos_to_process)
        if not face_ready_photos:
            raise RuntimeError("没有找到可复用的人脸识别缓存，无法从 VLM 边界续跑")
        print(f"[resume] restored_face_photos={len(face_ready_photos)} primary_person_id={primary_person_id}", flush=True)
        service._notify(
            progress_callback,
            "face_recognition",
            {
                "message": "复用已完成人脸识别缓存",
                "completed": True,
                "processed": len(face_ready_photos),
                "input_photo_count": len(photos_to_process),
                "percent": 100,
                "runtime_seconds": 0.0,
                "face_result_preview": service._build_face_stage_preview(face_payload),
                "face_output_url": service._public_url(service.cache_dir / "face_recognition_output.json"),
            },
        )

        photos_for_vlm = _hydrate_cached_visual_assets(service, face_ready_photos)
        print(f"[resume] hydrated_visual_assets={len(photos_for_vlm)}", flush=True)
        vlm = VLMAnalyzer(cache_path=str(service.vlm_cache_path), task_version=service.task_version)
        if not vlm.load_cache() or not vlm.results:
            raise RuntimeError("没有找到可复用的 VLM 缓存，无法从断点继续")
        vlm_by_photo_id = {
            item.get("photo_id"): item
            for item in vlm.results
            if isinstance(item, dict) and item.get("photo_id")
        }
        cached_photo_ids = set()
        filtered_vlm_results = []
        for photo in photos_for_vlm:
            cached = vlm_by_photo_id.get(photo.photo_id)
            if not cached:
                continue
            photo.vlm_analysis = cached.get("vlm_analysis")
            filtered_vlm_results.append(cached)
            cached_photo_ids.add(photo.photo_id)
        if not filtered_vlm_results:
            raise RuntimeError("VLM 缓存与当前照片元数据无法对齐，续跑已终止")
        vlm.results = filtered_vlm_results
        print(f"[resume] aligned_vlm_results={len(vlm.results)}", flush=True)
        service._notify(
            progress_callback,
            "vlm",
            service._build_vlm_stage_progress(
                vlm.results,
                cached_photo_ids,
                total_input_photos=len(photos_for_vlm),
                runtime_seconds=0.0,
            ),
        )

        facts = []
        relationships = []
        profile_markdown = ""
        memory_contract: Dict[str, object] = {
            "facts": [],
            "observations": [],
            "claims": [],
            "relationship_hypotheses": [],
            "profile_deltas": [],
            "uncertainty": [],
        }
        llm_chunk_artifacts: Dict[str, object] = {}

        llm_started_at = perf_counter()
        service._notify(
            progress_callback,
            "llm",
            {
                "message": "从缓存继续提取事实、关系与画像",
                "processed_slices": 0,
                "slice_count": 0,
                "processed_events": 0,
                "event_count": 0,
                "percent": 0,
                "runtime_seconds": 0.0,
            },
        )
        llm = LLMProcessor(task_version=service.task_version)
        print("[resume] llm_start", flush=True)
        pre_relationship_contract = (
            load_json(str(service.llm_pre_relationship_contract_path))
            if service.llm_pre_relationship_contract_path.exists()
            else {}
        )
        prior_llm_chunk_artifacts = (
            load_json(str(service.llm_chunks_path))
            if service.llm_chunks_path.exists()
            else {}
        )
        try:
            if (
                service.task_version == TASK_VERSION_V0317_HEAVY
                and isinstance(pre_relationship_contract, dict)
                and bool(pre_relationship_contract)
            ):
                print("[resume] llm_resume_from=pre_relationship_contract", flush=True)
                photo_facts = llm._build_photo_fact_buffer(vlm.results)
                memory_contract = copy.deepcopy(pre_relationship_contract)
                memory_contract["relationship_hypotheses"] = llm._run_heavy_relationship_pass(
                    contract=memory_contract,
                    photo_facts=photo_facts,
                    primary_person_id=primary_person_id,
                    progress_callback=lambda payload: service._notify(progress_callback, "llm", payload),
                    llm_started_at=llm_started_at,
                    slice_count=int(prior_llm_chunk_artifacts.get("slice_count") or 0),
                    event_count=int(prior_llm_chunk_artifacts.get("raw_event_count") or 0),
                    processed_events=int(prior_llm_chunk_artifacts.get("raw_event_count") or 0),
                    processed_slices=int(prior_llm_chunk_artifacts.get("slice_count") or 0),
                )
                llm.last_memory_contract = memory_contract
                llm.last_chunk_artifacts = {
                    **dict(prior_llm_chunk_artifacts or {}),
                    **dict(getattr(llm, "last_chunk_artifacts", {}) or {}),
                    "pre_relationship_contract": copy.deepcopy(pre_relationship_contract),
                }
            else:
                memory_contract = llm.extract_memory_contract(
                    vlm.results,
                    face_db,
                    primary_person_id,
                    progress_callback=lambda payload: service._notify(progress_callback, "llm", payload),
                )
        except Exception:
            llm_chunk_artifacts = dict(getattr(llm, "last_chunk_artifacts", {}) or {})
            partial_contract = dict(getattr(llm, "last_memory_contract", {}) or {})
            service._persist_llm_intermediate_outputs(
                llm_chunk_artifacts=llm_chunk_artifacts,
                partial_contract=partial_contract,
            )
            if llm_chunk_artifacts:
                save_json(llm_chunk_artifacts, str(service.llm_chunks_path))
            if partial_contract:
                save_json(partial_contract, str(service.llm_contract_path))
            raise
        llm_chunk_artifacts = dict(getattr(llm, "last_chunk_artifacts", {}) or {})
        service._persist_llm_intermediate_outputs(
            llm_chunk_artifacts=llm_chunk_artifacts,
            partial_contract=memory_contract,
        )
        save_json(memory_contract, str(service.llm_contract_path))
        save_json(llm_chunk_artifacts, str(service.llm_chunks_path))
        facts = llm.facts_from_memory_contract(memory_contract)
        relationships = llm.relationships_from_memory_contract(memory_contract)
        print(
            "[resume] llm_done "
            f"facts={len(facts)} observations={len(memory_contract.get('observations', []))} "
            f"claims={len(memory_contract.get('claims', []))} relationships={len(relationships)}",
            flush=True,
        )
        service._notify(
            progress_callback,
            "llm",
            {
                "message": "LLM 用户画像生成中",
                "substage": "profile_materialization",
                "processed_slices": llm_chunk_artifacts.get("slice_count", 0),
                "slice_count": llm_chunk_artifacts.get("slice_count", 0),
                "processed_events": llm_chunk_artifacts.get("raw_event_count", 0),
                "event_count": llm_chunk_artifacts.get("raw_event_count", 0),
                "percent": 99,
                "provider": llm.provider,
                "model": llm.model,
                "runtime_seconds": round(perf_counter() - llm_started_at, 4),
            },
        )
        profile_markdown = llm.generate_profile(facts, relationships, primary_person_id)
        service._write_profile_report(profile_markdown)
        service._notify(
            progress_callback,
            "llm",
            service._build_llm_stage_progress(
                memory_contract,
                llm_chunk_artifacts,
                profile_markdown=profile_markdown,
                runtime_seconds=perf_counter() - llm_started_at,
            ),
        )

        memory_started_at = perf_counter()
        service._notify(
            progress_callback,
            "memory",
            {
                "message": "构建记忆框架输出",
                "percent": 5,
                "runtime_seconds": 0.0,
            },
        )
        print("[resume] memory_start", flush=True)
        memory_payload = MemoryModuleService(
            task_id=service.task_id,
            task_dir=str(service.task_dir),
            user_id=service.user_id,
            pipeline_version=service.task_version,
            public_url_builder=service._public_url,
        ).materialize(
            photos=photos_for_vlm,
            face_output=face_output,
            vlm_results=vlm.results,
            events=facts,
            relationships=relationships,
            profile_markdown=profile_markdown,
            cached_photo_ids=cached_photo_ids,
            memory_contract=memory_contract,
            dedupe_report=dedupe_report,
            chunk_artifacts=llm_chunk_artifacts,
        )
        print(
            "[resume] memory_done "
            f"events={memory_payload.get('summary', {}).get('event_count', 0)} "
            f"profile_version={memory_payload.get('summary', {}).get('profile_version', 0)}",
            flush=True,
        )
        service._notify(
            progress_callback,
            "memory",
            service._build_memory_stage_progress(memory_payload, runtime_seconds=perf_counter() - memory_started_at),
        )

        detailed_output = {
            "task_id": service.task_id,
            "version": service.task_version,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "task_version": service.task_version,
                "total_uploaded": service._count_uploaded_files(),
                "loaded_images": len(photos),
                "failed_images": len(service.failed_images),
                "face_processed_images": len(face_ready_photos),
                "vlm_processed_images": len(vlm.results),
                "total_faces": face_output.get("metrics", {}).get("total_faces", 0),
                "total_persons": face_output.get("metrics", {}).get("total_persons", 0),
                "primary_person_id": primary_person_id,
                "event_count": memory_payload.get("summary", {}).get("event_count", 0),
                "fact_count": len(facts),
                "relationship_count": len(relationships),
                "observation_count": len(memory_contract.get("observations", [])),
                "claim_count": len(memory_contract.get("claims", [])),
                "profile_delta_count": len(memory_contract.get("profile_deltas", [])),
                "profile_version": memory_payload.get("summary", {}).get("profile_version", 0),
            },
            "face_recognition": face_payload,
            "face_report": service._build_face_report(face_payload),
            "failed_images": service.failed_images,
            "warnings": service.warnings,
            "facts": [service._serialize_event(event) for event in facts],
            "relationships": [service._serialize_relationship(item) for item in relationships],
            "profile_markdown": profile_markdown,
            "memory_contract": memory_contract,
            "llm_chunk_artifacts": llm_chunk_artifacts,
            "dedupe_report": dedupe_report,
            "memory": memory_payload,
            "artifacts": {},
        }

        result_path = service.output_dir / "result.json"
        save_json(detailed_output, str(result_path))
        detailed_output["artifacts"]["result_url"] = service._public_url(result_path)
        detailed_output["artifacts"]["face_output_url"] = service._public_url(service.cache_dir / "face_recognition_output.json")
        detailed_output["artifacts"]["vlm_cache_url"] = service._public_url(service.vlm_cache_path)
        detailed_output["artifacts"]["dedupe_report_url"] = service._public_url(service.dedupe_report_path) if service.dedupe_report_path.exists() else None
        detailed_output["artifacts"]["memory_contract_url"] = service._public_url(service.llm_contract_path) if service.llm_contract_path.exists() else None
        detailed_output["artifacts"]["llm_chunks_url"] = service._public_url(service.llm_chunks_path) if service.llm_chunks_path.exists() else None
        detailed_output["artifacts"]["slice_contracts_url"] = service._public_url(service.llm_slice_contracts_path) if service.llm_slice_contracts_path.exists() else None
        detailed_output["artifacts"]["event_merges_url"] = service._public_url(service.llm_event_merges_path) if service.llm_event_merges_path.exists() else None
        detailed_output["artifacts"]["pre_relationship_contract_url"] = service._public_url(service.llm_pre_relationship_contract_path) if service.llm_pre_relationship_contract_path.exists() else None
        detailed_output["artifacts"]["profile_report_url"] = service._public_url(service.profile_report_path) if service.profile_report_path.exists() else None
        for artifact_key, artifact_value in memory_payload.get("artifacts", {}).items():
            if artifact_key.endswith("_url"):
                detailed_output["artifacts"][artifact_key] = artifact_value
        asset_store.sync_task_directory(service.task_id, service.task_dir)

        task_store.update_task(
            service.task_id,
            status="completed",
            stage="completed",
            result=detailed_output,
            error=None,
            progress=copy.deepcopy(progress_state),
        )
        manifest = build_task_asset_manifest(service.task_id, service.task_dir, asset_store)
        task_store.set_result_summary(service.task_id, detailed_output.get("summary"), manifest)
        if record.user_id:
            artifact_catalog.replace_task_artifacts(service.task_id, record.user_id, manifest)
        print("[resume] completed", flush=True)
        return 0
    except Exception as exc:
        print(f"[resume] failed error={exc}", flush=True)
        task_store.update_task(
            record.task_id,
            status="failed",
            stage="failed",
            error=str(exc),
            progress=copy.deepcopy(progress_state),
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
