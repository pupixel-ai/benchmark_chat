"""
任务级 pipeline 服务。
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from config import (
    DEFAULT_NORMALIZE_LIVE_PHOTOS,
    DEFAULT_TASK_VERSION,
    FACE_MIN_SIZE,
    MAX_UPLOAD_PHOTOS,
    TASK_VERSION_V0315,
    TASK_VERSION_V0317,
    TASK_VERSION_V0317_HEAVY,
    TASK_VERSION_V0321_2,
    TASK_VERSION_V0321_3,
    TASK_VERSION_V0323,
    TASK_VERSION_V0325,
    TASK_VERSION_V0327_DB,
    TASK_VERSION_V0327_DB_QUERY,
    TASK_VERSION_V0327_EXP,
    VLM_CACHE_FLUSH_EVERY_N,
    VLM_CACHE_FLUSH_INTERVAL_SECONDS,
    VLM_ENABLE_PRIORITY_SCHEDULING,
    VLM_MAX_CONCURRENCY,
)
from memory_module import MemoryModuleService
from utils import load_json, save_json

try:
    from services.asset_store import TaskAssetStore
except ModuleNotFoundError:  # pragma: no cover - dependency-light test envs
    TaskAssetStore = None  # type: ignore[assignment]

try:
    from services.face_recognition import FaceRecognition
except ModuleNotFoundError:  # pragma: no cover - dependency-light test envs
    FaceRecognition = None  # type: ignore[assignment]

try:
    from services.image_processor import ImageProcessor
except ModuleNotFoundError:  # pragma: no cover - dependency-light test envs
    ImageProcessor = None  # type: ignore[assignment]

try:
    from services.llm_processor import LLMProcessor
except ModuleNotFoundError:  # pragma: no cover - dependency-light test envs
    LLMProcessor = None  # type: ignore[assignment]

try:
    from services.vlm_analyzer import VLMAnalyzer
except ModuleNotFoundError:  # pragma: no cover - dependency-light test envs
    VLMAnalyzer = None  # type: ignore[assignment]

try:
    from services.v0321_2.pipeline import V03212PipelineFamily
except ModuleNotFoundError:  # pragma: no cover - dependency-light test envs
    V03212PipelineFamily = None  # type: ignore[assignment]

try:
    from services.v0321_3.pipeline import V03213PipelineFamily
except ModuleNotFoundError:  # pragma: no cover - dependency-light test envs
    V03213PipelineFamily = None  # type: ignore[assignment]

try:
    from services.v0323.pipeline import V0323PipelineFamily
except ModuleNotFoundError:  # pragma: no cover - dependency-light test envs
    V0323PipelineFamily = None  # type: ignore[assignment]

try:
    from services.v0325.pipeline import V0325PipelineFamily
except ModuleNotFoundError:  # pragma: no cover - dependency-light test envs
    V0325PipelineFamily = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from backend.face_review_store import FaceReviewStore


class MemoryPipelineService:
    """将现有多模态流程封装为任务级服务。"""

    def __init__(
        self,
        task_id: str,
        task_dir: str,
        asset_store: Optional["TaskAssetStore"] = None,
        user_id: Optional[str] = None,
        face_review_store: Optional["FaceReviewStore"] = None,
        task_version: str = DEFAULT_TASK_VERSION,
        task_options: Optional[Dict[str, object]] = None,
    ):
        self.task_id = task_id
        self.task_dir = Path(task_dir)
        if asset_store is None:
            if TaskAssetStore is None:
                from services.asset_store import TaskAssetStore as _TaskAssetStore
            else:
                _TaskAssetStore = TaskAssetStore
            asset_store = _TaskAssetStore()
        self.asset_store = asset_store
        resolved_task_options = dict(task_options or {})
        self.user_id = user_id or str(resolved_task_options.get("user_id") or "").strip() or None
        if face_review_store is None:
            from backend.face_review_store import FaceReviewStore

            face_review_store = FaceReviewStore()
        self.face_review_store = face_review_store
        self.task_version = task_version
        self.task_options = {
            "normalize_live_photos": bool(resolved_task_options.get("normalize_live_photos", DEFAULT_NORMALIZE_LIVE_PHOTOS)),
        }

        self.upload_dir = self.task_dir / "uploads"
        self.cache_dir = self.task_dir / "cache"
        self.output_dir = self.task_dir / "output"
        self.upload_failures_path = self.task_dir / "upload_failures.json"
        self.vlm_failures_path = self.cache_dir / "vlm_failures.jsonl"

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        image_processor_cls = ImageProcessor
        if image_processor_cls is None:
            from services.image_processor import ImageProcessor as image_processor_cls
        face_recognition_cls = FaceRecognition
        if face_recognition_cls is None:
            from services.face_recognition import FaceRecognition as face_recognition_cls

        self.image_processor = image_processor_cls(cache_dir=str(self.cache_dir))
        self.face_recognition = face_recognition_cls(
            state_path=str(self.cache_dir / "face_recognition_state.json"),
            index_path=str(self.cache_dir / "faces.index"),
            output_path=str(self.cache_dir / "face_recognition_output.json"),
            workspace_dir=str(self.task_dir),
            task_version=self.task_version,
        )
        self.initial_upload_failures = self._load_upload_failures()
        self.failed_images: List[Dict] = list(self.initial_upload_failures)
        self.warnings: List[Dict] = []
        self.vlm_cache_path = self.cache_dir / "vlm_cache.json"
        self.dedupe_report_path = self.cache_dir / "dedupe_report.json"
        self.llm_contract_path = self.output_dir / "memory_contract.json"
        self.llm_chunks_path = self.output_dir / "llm_chunks.json"
        self.llm_slice_contracts_path = self.output_dir / "slice_contracts.jsonl"
        self.llm_event_merges_path = self.output_dir / "event_merges.jsonl"
        self.llm_pre_relationship_contract_path = self.output_dir / "pre_relationship_contract.json"
        self.profile_report_path = self.output_dir / "user_profile_report.md"

    def run(
        self,
        max_photos: int = MAX_UPLOAD_PHOTOS,
        use_cache: bool = False,
        progress_callback: Optional[Callable[[str, Dict], None]] = None,
    ) -> Dict:
        """执行完整 pipeline 并返回前端友好的结果。"""
        self._notify(progress_callback, "loading", {"message": "加载上传图片"})
        photos, load_errors = self.image_processor.load_photos_with_errors(
            str(self.upload_dir),
            max_photos=max_photos,
        )
        self.failed_images.extend(load_errors)

        self._notify(
            progress_callback,
            "converting",
            {
                "message": "转换 HEIC/JPEG",
                "photo_count": len(photos),
                "total": len(photos),
                "processed": 0,
                "percent": 0,
            },
        )
        conversion_total = len(photos)
        last_conversion_emit = {"processed": 0}

        def report_conversion_progress(processed: int, total: int, _photo) -> None:
            if total <= 0:
                return
            should_emit = processed >= total or processed == 1 or processed - last_conversion_emit["processed"] >= 10
            if not should_emit:
                return
            last_conversion_emit["processed"] = processed
            self._notify(
                progress_callback,
                "converting",
                {
                    "message": "转换 HEIC/JPEG",
                    "photo_count": total,
                    "total": total,
                    "processed": processed,
                    "percent": round((processed / total) * 100, 2),
                },
            )

        photos = self.image_processor.convert_to_jpeg(
            photos,
            normalize_live_photos=bool(self.task_options.get("normalize_live_photos")),
            progress_callback=report_conversion_progress,
        )
        if conversion_total == 0:
            self._notify(
                progress_callback,
                "converting",
                {
                    "message": "转换 HEIC/JPEG",
                    "photo_count": 0,
                    "total": 0,
                    "processed": 0,
                    "percent": 100,
                },
            )
        photos_to_process = self._apply_image_policies(photos)
        photos_to_process = self.image_processor.dedupe_before_face_recognition(photos_to_process)
        dedupe_report = getattr(self.image_processor, "last_dedupe_report", {}) or {}
        if dedupe_report:
            save_json(dedupe_report, str(self.dedupe_report_path))

        self._notify(
            progress_callback,
            "face_recognition",
            {
                "message": "进行人脸识别",
                "input_photo_count": len(photos_to_process),
                "original_photo_count": len(photos),
                "total": len(photos_to_process),
                "processed": 0,
                "percent": 0,
                "runtime_seconds": 0.0,
            },
        )
        face_started_at = perf_counter()
        face_ready_photos = self._run_face_recognition(
            photos_to_process,
            progress_callback=progress_callback,
            started_at=face_started_at,
        )

        primary_person_id = None
        if face_ready_photos:
            self.face_recognition.reorder_protagonist(face_ready_photos)
            primary_person_id = self.face_recognition.get_primary_person_id()
            photos_with_faces = [photo for photo in face_ready_photos if photo.faces]
            visualization_total = len(photos_with_faces)
            visualization_started_at = perf_counter()
            rendered_visualizations = 0
            if visualization_total > 0:
                self._notify(
                    progress_callback,
                    "face_visualization",
                    {
                        "message": "生成人脸框预览",
                        "photo_count": visualization_total,
                        "total": visualization_total,
                        "processed": 0,
                        "percent": 0,
                        "runtime_seconds": 0.0,
                    },
                )

            for photo in face_ready_photos:
                photo.primary_person_id = primary_person_id
                if photo.faces:
                    boxed_path = self.image_processor.draw_face_boxes(photo)
                    if boxed_path:
                        photo.boxed_path = boxed_path
                    rendered_visualizations += 1
                    self._notify(
                        progress_callback,
                        "face_visualization",
                        {
                            "message": "生成人脸框预览",
                            "photo_count": visualization_total,
                            "total": visualization_total,
                            "processed": rendered_visualizations,
                            "percent": self._stage_percent(rendered_visualizations, visualization_total),
                            "runtime_seconds": round(perf_counter() - visualization_started_at, 4),
                        },
                    )

        self.face_recognition.save()
        face_output = self.face_recognition.get_face_output()
        face_payload = self._build_face_recognition_payload(photos, face_output)
        face_db = self.face_recognition.get_all_persons()
        self._notify(
            progress_callback,
            "face_recognition",
            {
                "message": "人脸识别完成",
                "completed": True,
                "total": len(photos_to_process),
                "processed": len(photos_to_process),
                "percent": 100,
                "runtime_seconds": round(perf_counter() - face_started_at, 4),
                "face_result_preview": self._build_face_stage_preview(face_payload),
                "face_output_url": self._public_url(self.cache_dir / "face_recognition_output.json"),
            },
        )

        if not self._supports_memory_graph():
            self.warnings.append({
                "stage": "version_gate",
                "message": f"{self.task_version} 当前只保留到人脸识别的原始链路，VLM / LLM / Memory 已跳过",
            })
            detailed_output = {
                "task_id": self.task_id,
                "version": self.task_version,
                "generated_at": datetime.now().isoformat(),
                "summary": {
                    "task_version": self.task_version,
                    "total_uploaded": self._count_uploaded_files(),
                    "loaded_images": len(photos),
                    "failed_images": len(self.failed_images),
                    "face_processed_images": len(face_ready_photos),
                    "vlm_processed_images": 0,
                    "total_faces": face_output.get("metrics", {}).get("total_faces", 0),
                    "total_persons": face_output.get("metrics", {}).get("total_persons", 0),
                    "primary_person_id": primary_person_id,
                    "event_count": 0,
                    "fact_count": 0,
                    "relationship_count": 0,
                    "profile_version": 0,
                },
                "face_recognition": face_payload,
                "face_report": self._build_face_report(face_payload),
                "failed_images": self.failed_images,
                "warnings": self.warnings,
                "facts": [],
                "relationships": [],
                "profile_markdown": "",
                "memory": None,
                "artifacts": {},
            }
            result_path = self.output_dir / "result.json"
            save_json(detailed_output, str(result_path))
            detailed_output["artifacts"]["result_url"] = self._public_url(result_path)
            detailed_output["artifacts"]["face_output_url"] = self._public_url(self.cache_dir / "face_recognition_output.json")
            return detailed_output

        preprocess_total = len(face_ready_photos)
        preprocess_started_at = perf_counter()
        self._notify(
            progress_callback,
            "preprocess",
            {
                "message": "压缩图片",
                "photo_count": preprocess_total,
                "total": preprocess_total,
                "processed": 0,
                "percent": 0 if preprocess_total > 0 else 100,
                "runtime_seconds": 0.0,
            },
        )

        def report_preprocess_progress(processed: int, total: int, _photo: Photo) -> None:
            self._notify(
                progress_callback,
                "preprocess",
                {
                    "message": "压缩图片",
                    "photo_count": total,
                    "total": total,
                    "processed": processed,
                    "percent": self._stage_percent(processed, total),
                    "runtime_seconds": round(perf_counter() - preprocess_started_at, 4),
                },
            )

        photos_for_vlm = self.image_processor.preprocess(
            face_ready_photos,
            progress_callback=report_preprocess_progress,
        )

        vlm_started_at = perf_counter()
        self._notify(
            progress_callback,
            "vlm",
            {
                "message": "进行视觉分析",
                "photo_count": len(photos_for_vlm),
                "processed": 0,
                "queued": len(photos_for_vlm),
                "in_flight": 0,
                "completed_count": 0,
                "failed_count": 0,
                "percent": 0,
                "retry_count": 0,
                "flush_count": 0,
                "concurrency": max(1, VLM_MAX_CONCURRENCY),
                "avg_latency_seconds": 0.0,
                "runtime_seconds": 0.0,
            },
        )
        vlm_analyzer_cls = VLMAnalyzer
        if vlm_analyzer_cls is None:
            from services.vlm_analyzer import VLMAnalyzer as vlm_analyzer_cls
        vlm = vlm_analyzer_cls(cache_path=str(self.vlm_cache_path), task_version=self.task_version)
        cached_photo_ids, failed_vlm_items, vlm_stats = self._run_vlm_stage(
            photos_for_vlm,
            face_db,
            primary_person_id,
            vlm,
            use_cache=use_cache,
            progress_callback=progress_callback,
            started_at=vlm_started_at,
        )
        for item in failed_vlm_items:
            self.warnings.append({
                "stage": "vlm",
                "message": f"{item['filename']} 的 VLM 分析失败，已跳过",
                "image_id": item["photo_id"],
                "error": item["error"],
            })

        self._notify(
            progress_callback,
            "vlm",
            self._build_vlm_stage_progress(
                vlm.results,
                cached_photo_ids,
                total_input_photos=len(photos_for_vlm),
                runtime_seconds=perf_counter() - vlm_started_at,
                failed_items=failed_vlm_items,
                stats=vlm_stats,
            ),
        )

        fallback_primary_person_id = self._resolve_primary_person_id_from_identity_documents(
            photos_for_vlm,
            vlm.results,
            current_primary_person_id=primary_person_id,
        )
        if fallback_primary_person_id and fallback_primary_person_id != primary_person_id:
            primary_person_id = fallback_primary_person_id
            self.face_recognition.primary_person_id = primary_person_id
            self.face_recognition.state["primary_person_id"] = primary_person_id
            for photo in face_ready_photos:
                photo.primary_person_id = primary_person_id
            self.face_recognition.save()
            face_output = self.face_recognition.get_face_output()
            face_payload = self._build_face_recognition_payload(photos, face_output)
            self.warnings.append(
                {
                    "stage": "primary_person_fallback",
                    "message": "未能从人脸频次稳定识别主角，已使用证件照中的单人脸作为主角回填。",
                    "primary_person_id": primary_person_id,
                }
            )

        if self.task_version in {
            TASK_VERSION_V0321_2,
            TASK_VERSION_V0321_3,
            TASK_VERSION_V0323,
            TASK_VERSION_V0325,
            TASK_VERSION_V0327_EXP,
            TASK_VERSION_V0327_DB,
            TASK_VERSION_V0327_DB_QUERY,
        }:
            self._clear_legacy_outputs_for_revision_first_family()
            if self.task_version == TASK_VERSION_V0321_2:
                memory = self._run_v0321_2_family(
                    photos=photos_for_vlm,
                    face_output=face_output,
                    primary_person_id=primary_person_id,
                    cached_photo_ids=cached_photo_ids,
                    dedupe_report=dedupe_report,
                    vlm_results=vlm.results,
                    progress_callback=progress_callback,
                )
            elif self.task_version == TASK_VERSION_V0321_3:
                memory = self._run_v0321_3_family(
                    photos=photos_for_vlm,
                    face_output=face_output,
                    primary_person_id=primary_person_id,
                    cached_photo_ids=cached_photo_ids,
                    dedupe_report=dedupe_report,
                    vlm_results=vlm.results,
                    progress_callback=progress_callback,
                )
            elif self.task_version in {TASK_VERSION_V0325, TASK_VERSION_V0327_EXP, TASK_VERSION_V0327_DB, TASK_VERSION_V0327_DB_QUERY}:
                memory = self._run_v0325_family(
                    photos=photos_for_vlm,
                    face_output=face_output,
                    primary_person_id=primary_person_id,
                    cached_photo_ids=cached_photo_ids,
                    dedupe_report=dedupe_report,
                    vlm_results=vlm.results,
                    progress_callback=progress_callback,
                )
            else:
                memory = self._run_v0323_family(
                    photos=photos_for_vlm,
                    face_output=face_output,
                    primary_person_id=primary_person_id,
                    cached_photo_ids=cached_photo_ids,
                    dedupe_report=dedupe_report,
                    vlm_results=vlm.results,
                    progress_callback=progress_callback,
                )
            if self.task_version in {TASK_VERSION_V0323, TASK_VERSION_V0325, TASK_VERSION_V0327_EXP, TASK_VERSION_V0327_DB, TASK_VERSION_V0327_DB_QUERY}:
                lp1_events = list(memory.get("lp1_events", []) or [])
                lp2_relationships = list(memory.get("lp2_relationships", []) or [])
                profile_payload = dict(memory.get("lp3_profile", {}) or {})
                profile_markdown = str(profile_payload.get("report_markdown") or profile_payload.get("report") or "")
                detailed_output = {
                    "task_id": self.task_id,
                    "version": self.task_version,
                    "generated_at": datetime.now().isoformat(),
                    "summary": {
                        "task_version": self.task_version,
                        "pipeline_family": memory.get("pipeline_family"),
                        "total_uploaded": self._count_uploaded_files(),
                        "loaded_images": len(photos),
                        "failed_images": len(self.failed_images),
                        "face_processed_images": len(face_ready_photos),
                        "vlm_processed_images": len(vlm.results),
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
                    "face_report": self._build_face_report(face_payload),
                    "failed_images": self.failed_images,
                    "warnings": self.warnings,
                    "facts": lp1_events,
                    "relationships": lp2_relationships,
                    "profile_markdown": profile_markdown,
                    "memory_contract": None,
                    "llm_chunk_artifacts": {},
                    "dedupe_report": dedupe_report,
                    "memory": memory,
                    "artifacts": {},
                }

                result_path = self.output_dir / "result.json"
                save_json(detailed_output, str(result_path))
                detailed_output["artifacts"]["result_url"] = self._public_url(result_path)
                detailed_output["artifacts"]["face_output_url"] = self._public_url(self.cache_dir / "face_recognition_output.json")
                detailed_output["artifacts"]["vlm_cache_url"] = self._public_url(self.vlm_cache_path)
                detailed_output["artifacts"]["vlm_failures_url"] = self._public_url(self.vlm_failures_path) if self.vlm_failures_path.exists() else None
                detailed_output["artifacts"]["dedupe_report_url"] = self._public_url(self.dedupe_report_path) if self.dedupe_report_path.exists() else None
                for artifact_key, artifact_value in memory.get("artifacts", {}).items():
                    if artifact_key.endswith("_url"):
                        detailed_output["artifacts"][artifact_key] = artifact_value
                return detailed_output
            delta_profile_revision = dict(memory.get("delta_profile_revision", {}) or {})
            delta_profile_markdown = str(memory.get("delta_profile_markdown") or "")
            full_profile_revision = dict(memory.get("profile_revision", {}) or {})
            full_profile_markdown = str(memory.get("profile_markdown") or "")
            profile_revision = full_profile_revision or delta_profile_revision
            profile_markdown = full_profile_markdown or delta_profile_markdown
            event_revisions = list(memory.get("delta_event_revisions", []) or memory.get("event_revisions", []) or [])
            relationship_revisions = list(
                memory.get("delta_relationship_revisions", []) or memory.get("relationship_revisions", []) or []
            )
            atomic_evidence = list(memory.get("delta_atomic_evidence", []) or memory.get("atomic_evidence", []) or [])
            profile_buckets = dict(delta_profile_revision.get("buckets", {}) or {})

            detailed_output = {
                "task_id": self.task_id,
                "version": self.task_version,
                "generated_at": datetime.now().isoformat(),
                "summary": {
                    "task_version": self.task_version,
                    "pipeline_family": memory.get("pipeline_family"),
                    "total_uploaded": self._count_uploaded_files(),
                    "loaded_images": len(photos),
                    "failed_images": len(self.failed_images),
                    "face_processed_images": len(face_ready_photos),
                    "vlm_processed_images": len(vlm.results),
                    "total_faces": face_output.get("metrics", {}).get("total_faces", 0),
                    "total_persons": face_output.get("metrics", {}).get("total_persons", 0),
                    "primary_person_id": primary_person_id,
                    "event_count": len(event_revisions),
                    "fact_count": len(event_revisions),
                    "relationship_count": len(relationship_revisions),
                    "observation_count": len(atomic_evidence),
                    "claim_count": 0,
                    "profile_delta_count": sum(len(bucket.get("values", [])) for bucket in profile_buckets.values()),
                    "profile_version": int(profile_revision.get("version") or 0),
                    "profile_generation_mode": memory.get("summary", {}).get("profile_generation_mode") or profile_revision.get("generation_mode"),
                },
                "face_recognition": face_payload,
                "face_report": self._build_face_report(face_payload),
                "failed_images": self.failed_images,
                "warnings": self.warnings,
                "facts": event_revisions,
                "relationships": relationship_revisions,
                "profile_markdown": profile_markdown,
                "memory_contract": None,
                "llm_chunk_artifacts": {},
                "dedupe_report": dedupe_report,
                "memory": memory,
                "artifacts": {},
            }

            result_path = self.output_dir / "result.json"
            save_json(detailed_output, str(result_path))
            detailed_output["artifacts"]["result_url"] = self._public_url(result_path)
            detailed_output["artifacts"]["face_output_url"] = self._public_url(self.cache_dir / "face_recognition_output.json")
            detailed_output["artifacts"]["vlm_cache_url"] = self._public_url(self.vlm_cache_path)
            detailed_output["artifacts"]["vlm_failures_url"] = self._public_url(self.vlm_failures_path) if self.vlm_failures_path.exists() else None
            detailed_output["artifacts"]["dedupe_report_url"] = self._public_url(self.dedupe_report_path) if self.dedupe_report_path.exists() else None
            for artifact_key, artifact_value in memory.get("artifacts", {}).items():
                if artifact_key.endswith("_url"):
                    detailed_output["artifacts"][artifact_key] = artifact_value
            return detailed_output

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

        if vlm.results:
            llm_started_at = perf_counter()
            self._notify(
                progress_callback,
                "llm",
                {
                    "message": "提取事实、关系与画像",
                    "substage": "slice_contract",
                    "processed_slices": 0,
                    "slice_count": 0,
                    "processed_events": 0,
                    "event_count": 0,
                    "percent": 0,
                    "runtime_seconds": 0.0,
                },
            )
            llm_processor_cls = LLMProcessor
            if llm_processor_cls is None:
                from services.llm_processor import LLMProcessor as llm_processor_cls
            llm = llm_processor_cls(task_version=self.task_version)
            try:
                memory_contract = llm.extract_memory_contract(
                    vlm.results,
                    face_db,
                    primary_person_id,
                    progress_callback=lambda payload: self._notify(progress_callback, "llm", payload),
                )
            except Exception:
                llm_chunk_artifacts = dict(getattr(llm, "last_chunk_artifacts", {}) or {})
                partial_contract = dict(getattr(llm, "last_memory_contract", {}) or {})
                self._persist_llm_intermediate_outputs(
                    llm_chunk_artifacts=llm_chunk_artifacts,
                    partial_contract=partial_contract,
                )
                if llm_chunk_artifacts:
                    save_json(llm_chunk_artifacts, str(self.llm_chunks_path))
                if partial_contract:
                    save_json(partial_contract, str(self.llm_contract_path))
                raise
            llm_chunk_artifacts = dict(getattr(llm, "last_chunk_artifacts", {}) or {})
            self._persist_llm_intermediate_outputs(
                llm_chunk_artifacts=llm_chunk_artifacts,
                partial_contract=memory_contract,
            )
            save_json(memory_contract, str(self.llm_contract_path))
            save_json(llm_chunk_artifacts, str(self.llm_chunks_path))
            facts = llm.facts_from_memory_contract(memory_contract)
            relationships = llm.relationships_from_memory_contract(memory_contract)
            self._notify(
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
                    "provider": getattr(llm, "provider", None),
                    "model": getattr(llm, "model", None),
                    "runtime_seconds": round(perf_counter() - llm_started_at, 4),
                },
            )
            profile_markdown = llm.generate_profile(facts, relationships, primary_person_id)
            self._write_profile_report(profile_markdown)
            self._notify(
                progress_callback,
                "llm",
                self._build_llm_stage_progress(
                    memory_contract,
                    llm_chunk_artifacts,
                    profile_markdown=profile_markdown,
                    runtime_seconds=perf_counter() - llm_started_at,
                ),
            )
        else:
            self.warnings.append({
                "stage": "llm",
                "message": "VLM 结果为空，LLM 只保留空结果",
            })
            self._notify(
                progress_callback,
                "llm",
                self._build_llm_stage_progress(memory_contract, llm_chunk_artifacts, runtime_seconds=0.0),
            )

        memory_started_at = perf_counter()
        self._notify(
            progress_callback,
            "memory",
            {
                "message": "构建记忆框架输出",
                "percent": 5,
                "runtime_seconds": 0.0,
            },
        )
        memory = MemoryModuleService(
            task_id=self.task_id,
            task_dir=str(self.task_dir),
            user_id=self.user_id,
            pipeline_version=self.task_version,
            public_url_builder=self._public_url,
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
        self._notify(
            progress_callback,
            "memory",
            self._build_memory_stage_progress(memory, runtime_seconds=perf_counter() - memory_started_at),
        )

        detailed_output = {
            "task_id": self.task_id,
            "version": self.task_version,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "task_version": self.task_version,
                "total_uploaded": self._count_uploaded_files(),
                "loaded_images": len(photos),
                "failed_images": len(self.failed_images),
                "face_processed_images": len(face_ready_photos),
                "vlm_processed_images": len(vlm.results),
                "total_faces": face_output.get("metrics", {}).get("total_faces", 0),
                "total_persons": face_output.get("metrics", {}).get("total_persons", 0),
                "primary_person_id": primary_person_id,
                "event_count": memory.get("summary", {}).get("event_count", 0),
                "fact_count": len(facts),
                "relationship_count": len(relationships),
                "observation_count": len(memory_contract.get("observations", [])),
                "claim_count": len(memory_contract.get("claims", [])),
                "profile_delta_count": len(memory_contract.get("profile_deltas", [])),
                "profile_version": memory.get("summary", {}).get("profile_version", 0),
            },
            "face_recognition": face_payload,
            "face_report": self._build_face_report(face_payload),
            "failed_images": self.failed_images,
            "warnings": self.warnings,
            "facts": [self._serialize_event(event) for event in facts],
            "relationships": [self._serialize_relationship(item) for item in relationships],
            "profile_markdown": profile_markdown,
            "memory_contract": memory_contract,
            "llm_chunk_artifacts": llm_chunk_artifacts,
            "dedupe_report": dedupe_report,
            "memory": memory,
            "artifacts": {},
        }

        result_path = self.output_dir / "result.json"
        save_json(detailed_output, str(result_path))
        detailed_output["artifacts"]["result_url"] = self._public_url(result_path)
        detailed_output["artifacts"]["face_output_url"] = self._public_url(self.cache_dir / "face_recognition_output.json")
        detailed_output["artifacts"]["vlm_cache_url"] = self._public_url(self.vlm_cache_path)
        detailed_output["artifacts"]["vlm_failures_url"] = self._public_url(self.vlm_failures_path) if self.vlm_failures_path.exists() else None
        detailed_output["artifacts"]["dedupe_report_url"] = self._public_url(self.dedupe_report_path) if self.dedupe_report_path.exists() else None
        detailed_output["artifacts"]["memory_contract_url"] = self._public_url(self.llm_contract_path) if self.llm_contract_path.exists() else None
        detailed_output["artifacts"]["llm_chunks_url"] = self._public_url(self.llm_chunks_path) if self.llm_chunks_path.exists() else None
        detailed_output["artifacts"]["slice_contracts_url"] = self._public_url(self.llm_slice_contracts_path) if self.llm_slice_contracts_path.exists() else None
        detailed_output["artifacts"]["event_merges_url"] = self._public_url(self.llm_event_merges_path) if self.llm_event_merges_path.exists() else None
        detailed_output["artifacts"]["pre_relationship_contract_url"] = self._public_url(self.llm_pre_relationship_contract_path) if self.llm_pre_relationship_contract_path.exists() else None
        detailed_output["artifacts"]["profile_report_url"] = self._public_url(self.profile_report_path) if self.profile_report_path.exists() else None
        for artifact_key, artifact_value in memory.get("artifacts", {}).items():
            if artifact_key.endswith("_url"):
                detailed_output["artifacts"][artifact_key] = artifact_value
        return detailed_output

    def _run_v0321_2_family(
        self,
        *,
        photos: List,
        face_output: Dict[str, object],
        primary_person_id: Optional[str],
        cached_photo_ids: set[str],
        dedupe_report: Dict[str, object],
        vlm_results: List[Dict[str, object]],
        progress_callback: Optional[Callable[[str, Dict], None]],
    ) -> Dict[str, object]:
        self._notify(
            progress_callback,
            "llm",
            {
                "message": "v0321.2 revision-first 链路启动",
                "pipeline_family": "v0321_2",
                "substage": "event_draft",
                "candidate_count": 0,
                "filtered_count": 0,
                "processed_candidates": 0,
                "percent": None,
                "runtime_seconds": 0.0,
            },
        )
        family_cls = V03212PipelineFamily
        if family_cls is None:
            from services.v0321_2.pipeline import V03212PipelineFamily as family_cls
        llm_processor_cls = LLMProcessor
        if llm_processor_cls is None:
            from services.llm_processor import LLMProcessor as llm_processor_cls
        return family_cls(
            task_id=self.task_id,
            task_dir=self.task_dir,
            user_id=self.user_id,
            asset_store=self.asset_store,
            llm_processor=llm_processor_cls(task_version=self.task_version),
            public_url_builder=self._public_url,
        ).run(
            photos=photos,
            face_output=face_output,
            primary_person_id=primary_person_id,
            vlm_results=vlm_results,
            cached_photo_ids=cached_photo_ids,
            dedupe_report=dedupe_report,
            progress_callback=lambda stage, payload: self._notify(
                progress_callback,
                "memory" if stage == "v0321_2" else stage,
                payload,
            ),
        )

    def _run_v0321_3_family(
        self,
        *,
        photos: List,
        face_output: Dict[str, object],
        primary_person_id: Optional[str],
        cached_photo_ids: set[str],
        dedupe_report: Dict[str, object],
        vlm_results: List[Dict[str, object]],
        progress_callback: Optional[Callable[[str, Dict], None]],
    ) -> Dict[str, object]:
        self._notify(
            progress_callback,
            "llm",
            {
                "message": "v0321.3 revision-first 链路启动",
                "pipeline_family": "v0321_3",
                "substage": "event_draft",
                "candidate_count": 0,
                "filtered_count": 0,
                "processed_candidates": 0,
                "percent": None,
                "runtime_seconds": 0.0,
            },
        )
        family_cls = V03213PipelineFamily
        if family_cls is None:
            from services.v0321_3.pipeline import V03213PipelineFamily as family_cls
        llm_processor_cls = LLMProcessor
        if llm_processor_cls is None:
            from services.llm_processor import LLMProcessor as llm_processor_cls
        return family_cls(
            task_id=self.task_id,
            task_dir=self.task_dir,
            user_id=self.user_id,
            asset_store=self.asset_store,
            llm_processor=llm_processor_cls(task_version=self.task_version),
            public_url_builder=self._public_url,
        ).run(
            photos=photos,
            face_output=face_output,
            primary_person_id=primary_person_id,
            vlm_results=vlm_results,
            cached_photo_ids=cached_photo_ids,
            dedupe_report=dedupe_report,
            progress_callback=lambda stage, payload: self._notify(
                progress_callback,
                "memory" if stage == "v0321_3" else stage,
                payload,
            ),
        )

    def _run_v0323_family(
        self,
        *,
        photos: List,
        face_output: Dict[str, object],
        primary_person_id: Optional[str],
        cached_photo_ids: set[str],
        dedupe_report: Dict[str, object],
        vlm_results: List[Dict[str, object]],
        progress_callback: Optional[Callable[[str, Dict], None]],
    ) -> Dict[str, object]:
        self._notify(
            progress_callback,
            "llm",
            {
                "message": "v0323 LP snapshot 链路启动",
                "pipeline_family": "v0323",
                "substage": "lp1_batch",
                "batch_index": 0,
                "batch_count": 0,
                "event_count": 0,
                "percent": None,
                "runtime_seconds": 0.0,
            },
        )
        family_cls = V0323PipelineFamily
        if family_cls is None:
            from services.v0323.pipeline import V0323PipelineFamily as family_cls
        llm_processor_cls = LLMProcessor
        if llm_processor_cls is None:
            from services.llm_processor import LLMProcessor as llm_processor_cls
        return family_cls(
            task_id=self.task_id,
            task_dir=self.task_dir,
            user_id=self.user_id,
            asset_store=self.asset_store,
            llm_processor=llm_processor_cls(task_version=self.task_version),
            public_url_builder=self._public_url,
        ).run(
            photos=photos,
            face_output=face_output,
            primary_person_id=primary_person_id,
            vlm_results=vlm_results,
            cached_photo_ids=cached_photo_ids,
            dedupe_report=dedupe_report,
            progress_callback=lambda stage, payload: self._notify(
                progress_callback,
                "memory" if stage == "v0323" else stage,
                payload,
            ),
        )

    def _run_v0325_family(
        self,
        *,
        photos: List,
        face_output: Dict[str, object],
        primary_person_id: Optional[str],
        cached_photo_ids: set[str],
        dedupe_report: Dict[str, object],
        vlm_results: List[Dict[str, object]],
        progress_callback: Optional[Callable[[str, Dict], None]],
    ) -> Dict[str, object]:
        self._notify(
            progress_callback,
            "llm",
            {
                "message": "v0325 LP agent 链路启动",
                "pipeline_family": "v0325",
                "substage": "lp1_batch",
                "batch_index": 0,
                "batch_count": 0,
                "event_count": 0,
                "percent": None,
                "runtime_seconds": 0.0,
            },
        )
        family_cls = V0325PipelineFamily
        if family_cls is None:
            from services.v0325.pipeline import V0325PipelineFamily as family_cls
        llm_processor_cls = LLMProcessor
        if llm_processor_cls is None:
            from services.llm_processor import LLMProcessor as llm_processor_cls
        return family_cls(
            task_id=self.task_id,
            task_dir=self.task_dir,
            user_id=self.user_id,
            asset_store=self.asset_store,
            llm_processor=llm_processor_cls(task_version=self.task_version),
            public_url_builder=self._public_url,
        ).run(
            photos=photos,
            face_output=face_output,
            primary_person_id=primary_person_id,
            vlm_results=vlm_results,
            cached_photo_ids=cached_photo_ids,
            dedupe_report=dedupe_report,
            progress_callback=lambda stage, payload: self._notify(
                progress_callback,
                "memory" if stage == "v0325" else stage,
                payload,
            ),
        )

    def _supports_memory_graph(self) -> bool:
        return self.task_version in {
            TASK_VERSION_V0317,
            TASK_VERSION_V0317_HEAVY,
            TASK_VERSION_V0321_2,
            TASK_VERSION_V0321_3,
            TASK_VERSION_V0323,
            TASK_VERSION_V0325,
            TASK_VERSION_V0327_EXP,
            TASK_VERSION_V0327_DB,
            TASK_VERSION_V0327_DB_QUERY,
        }

    def _run_face_recognition(
        self,
        photos: List,
        progress_callback: Optional[Callable[[str, Dict], None]] = None,
        started_at: Optional[float] = None,
    ) -> List:
        successful = []
        total = len(photos)
        stage_started_at = started_at if started_at is not None else perf_counter()
        for index, photo in enumerate(photos, start=1):
            try:
                self.face_recognition.process_photo(photo)
                successful.append(photo)
            except Exception as exc:
                self._record_failure(photo.photo_id, photo.filename, photo.path, "face_recognition", str(exc))
            finally:
                self._notify(
                    progress_callback,
                    "face_recognition",
                    {
                        "message": "进行人脸识别",
                        "processed": index,
                        "input_photo_count": total,
                        "percent": self._stage_percent(index, total),
                        "runtime_seconds": round(perf_counter() - stage_started_at, 4),
                    },
                )
        return successful

    def _run_vlm_stage(
        self,
        photos_for_vlm: List,
        face_db: Dict,
        primary_person_id: Optional[str],
        vlm,
        *,
        use_cache: bool,
        progress_callback: Optional[Callable[[str, Dict], None]],
        started_at: float,
    ) -> tuple[set[str], List[Dict[str, object]], Dict[str, object]]:
        cached_photo_ids: set[str] = set()
        failed_by_index: Dict[int, Dict[str, object]] = {}
        vlm_failure_records: List[Dict[str, object]] = []
        ordered_entries: Dict[int, Dict[str, object]] = {}
        metrics: Dict[str, object] = {
            "queued": 0,
            "in_flight": 0,
            "completed_count": 0,
            "failed_count": 0,
            "retry_count": 0,
            "flush_count": 0,
            "concurrency": max(1, VLM_MAX_CONCURRENCY),
            "latencies": [],
        }
        dirty_successes = 0
        last_flush_at = perf_counter()

        if use_cache:
            vlm.load_cache()
        if self.vlm_failures_path.exists():
            self.vlm_failures_path.unlink()

        indexed_photos = list(enumerate(photos_for_vlm))
        if VLM_ENABLE_PRIORITY_SCHEDULING:
            indexed_photos = self._prioritize_vlm_inputs(indexed_photos)

        pending_jobs: List[tuple[int, object]] = []
        for original_index, photo in indexed_photos:
            cached = vlm.get_result(photo.photo_id) if use_cache else None
            if cached:
                cached_photo_ids.add(photo.photo_id)
                photo.vlm_analysis = cached.get("vlm_analysis")
                ordered_entries[original_index] = dict(cached)
                metrics["completed_count"] = int(metrics["completed_count"]) + 1
                continue
            pending_jobs.append((original_index, photo))

        total_input_photos = len(photos_for_vlm)
        metrics["queued"] = len(pending_jobs)
        self._notify(
            progress_callback,
            "vlm",
            self._build_vlm_progress_update(
                total_input_photos=total_input_photos,
                cached_photo_ids=cached_photo_ids,
                runtime_seconds=perf_counter() - started_at,
                metrics=metrics,
            ),
        )

        if pending_jobs:
            with ThreadPoolExecutor(max_workers=max(1, VLM_MAX_CONCURRENCY)) as executor:
                future_map = {}
                for original_index, photo in pending_jobs:
                    metrics["queued"] = max(0, int(metrics["queued"]) - 1)
                    metrics["in_flight"] = int(metrics["in_flight"]) + 1
                    future = executor.submit(
                        vlm.analyze_photo_with_metadata,
                        photo,
                        face_db,
                        primary_person_id,
                    )
                    future_map[future] = (original_index, photo)
                    self._notify(
                        progress_callback,
                        "vlm",
                        self._build_vlm_progress_update(
                            total_input_photos=total_input_photos,
                            cached_photo_ids=cached_photo_ids,
                            runtime_seconds=perf_counter() - started_at,
                            metrics=metrics,
                        ),
                    )

                for future in as_completed(future_map):
                    original_index, photo = future_map[future]
                    metrics["in_flight"] = max(0, int(metrics["in_flight"]) - 1)
                    result, metadata = future.result()
                    retry_count = int(metadata.get("retry_count") or 0)
                    metrics["retry_count"] = int(metrics["retry_count"]) + retry_count
                    latency = float(metadata.get("runtime_seconds") or 0.0)
                    if latency > 0:
                        latencies = metrics["latencies"]
                        if isinstance(latencies, list):
                            latencies.append(latency)

                    if result:
                        ordered_entries[original_index] = vlm.build_result_entry(photo, result)
                        metrics["completed_count"] = int(metrics["completed_count"]) + 1
                        dirty_successes += 1
                    else:
                        failure_payload = {
                            "photo_id": photo.photo_id,
                            "filename": photo.filename,
                            "error": str(metadata.get("error") or photo.processing_errors.get("vlm") or "未知错误"),
                            "error_type": str(metadata.get("error_type") or ""),
                            "provider": str(metadata.get("provider") or ""),
                            "model": str(metadata.get("model") or ""),
                            "retry_count": int(metadata.get("retry_count") or 0),
                            "runtime_seconds": float(metadata.get("runtime_seconds") or 0.0),
                            "mime_type": str(metadata.get("mime_type") or ""),
                            "prompt_char_count": int(metadata.get("prompt_char_count") or 0),
                            "raw_response_preview": str(metadata.get("raw_response_preview") or ""),
                            "response_status_code": metadata.get("response_status_code"),
                            "parse_error_type": str(metadata.get("parse_error_type") or ""),
                        }
                        failed_by_index[original_index] = failure_payload
                        vlm_failure_records.append(failure_payload)
                        self._append_jsonl(self.vlm_failures_path, failure_payload)
                        metrics["failed_count"] = int(metrics["failed_count"]) + 1

                    now = perf_counter()
                    if self._should_flush_vlm_cache(
                        dirty_successes=dirty_successes,
                        now=now,
                        last_flush_at=last_flush_at,
                    ):
                        self._flush_vlm_results(vlm, ordered_entries)
                        dirty_successes = 0
                        last_flush_at = now
                        metrics["flush_count"] = int(metrics["flush_count"]) + 1

                    self._notify(
                        progress_callback,
                        "vlm",
                        self._build_vlm_progress_update(
                            total_input_photos=total_input_photos,
                            cached_photo_ids=cached_photo_ids,
                            runtime_seconds=perf_counter() - started_at,
                            metrics=metrics,
                        ),
                    )

        if dirty_successes or ordered_entries:
            self._flush_vlm_results(vlm, ordered_entries)
            metrics["flush_count"] = int(metrics["flush_count"]) + 1

        failed_items = [failed_by_index[index] for index in sorted(failed_by_index)]
        return cached_photo_ids, failed_items, self._serialize_vlm_stats(metrics)

    def _resolve_primary_person_id_from_identity_documents(
        self,
        photos_for_vlm: List,
        vlm_results: List[Dict[str, object]],
        *,
        current_primary_person_id: Optional[str],
    ) -> Optional[str]:
        if current_primary_person_id:
            return current_primary_person_id

        results_by_photo = {
            str(item.get("photo_id") or ""): dict(item.get("vlm_analysis", {}) or {})
            for item in vlm_results
            if str(item.get("photo_id") or "")
        }
        candidate_scores: Dict[str, Dict[str, float]] = {}
        for photo in photos_for_vlm:
            analysis = results_by_photo.get(str(getattr(photo, "photo_id", "") or ""))
            if not analysis:
                continue
            score = self._identity_document_signal_score(analysis)
            if score < 1.0:
                continue
            person_ids = []
            for face in list(getattr(photo, "faces", []) or []):
                person_id = str(face.get("person_id") or "")
                if person_id:
                    person_ids.append(person_id)
            unique_person_ids = sorted(set(person_ids))
            if len(unique_person_ids) != 1:
                continue
            person_id = unique_person_ids[0]
            payload = candidate_scores.setdefault(person_id, {"count": 0.0, "best_score": 0.0})
            payload["count"] += 1.0
            payload["best_score"] = max(payload["best_score"], score)

        if not candidate_scores:
            return None

        ranked = sorted(
            candidate_scores.items(),
            key=lambda item: (item[1]["count"], item[1]["best_score"], item[0]),
            reverse=True,
        )
        top_person_id, top_stats = ranked[0]
        if len(ranked) > 1:
            next_stats = ranked[1][1]
            if (
                float(top_stats.get("count") or 0.0) == float(next_stats.get("count") or 0.0)
                and float(top_stats.get("best_score") or 0.0) == float(next_stats.get("best_score") or 0.0)
            ):
                return None
        return top_person_id

    def _identity_document_signal_score(self, analysis: Dict[str, object]) -> float:
        text_pool = []
        for key in ("summary",):
            value = analysis.get(key)
            if value:
                text_pool.append(str(value))
        for bucket in ("details", "key_objects", "ocr_hits", "brands", "place_candidates"):
            text_pool.extend(str(value) for value in list(analysis.get(bucket, []) or []) if value)
        scene = dict(analysis.get("scene", {}) or {})
        event = dict(analysis.get("event", {}) or {})
        text_pool.extend(
            str(value)
            for value in [
                scene.get("location_detected"),
                scene.get("environment_description"),
                event.get("activity"),
                event.get("social_context"),
                event.get("interaction"),
            ]
            if value
        )
        normalized = " ".join(text_pool).lower()
        direct_keywords = [
            "身份证",
            "居民身份证",
            "学生证",
            "学员证",
            "校园卡",
            "证件照",
            "identity card",
            "id card",
            "student id",
            "student card",
            "campus card",
            "identification card",
            "passport",
            "driver license",
            "driving licence",
        ]
        supporting_keywords = [
            "证件",
            "卡片",
            "照片",
            "头像",
            "portrait",
            "document",
            "card",
            "student",
            "school",
            "campus",
            "姓名",
            "学号",
            "身份证号",
        ]
        score = 0.0
        if any(keyword in normalized for keyword in direct_keywords):
            score += 1.0
        support_hits = sum(1 for keyword in supporting_keywords if keyword in normalized)
        if support_hits >= 2:
            score += 0.5
        if re.search(r"\b\d{15,18}[0-9xX]?\b", normalized):
            score += 0.5
        return score

    def _prioritize_vlm_inputs(self, indexed_photos: List[tuple[int, object]]) -> List[tuple[int, object]]:
        def priority_key(item: tuple[int, object]) -> tuple[int, int]:
            original_index, photo = item
            has_location = 1 if getattr(photo, "location", {}).get("name") else 0
            has_faces = 1 if getattr(photo, "faces", []) else 0
            return (-(has_location + has_faces), original_index)

        return sorted(indexed_photos, key=priority_key)

    def _should_flush_vlm_cache(self, *, dirty_successes: int, now: float, last_flush_at: float) -> bool:
        if dirty_successes <= 0:
            return False
        if dirty_successes >= VLM_CACHE_FLUSH_EVERY_N:
            return True
        return (now - last_flush_at) >= VLM_CACHE_FLUSH_INTERVAL_SECONDS

    def _flush_vlm_results(self, vlm, ordered_entries: Dict[int, Dict[str, object]]) -> None:
        ordered_results = [ordered_entries[index] for index in sorted(ordered_entries)]
        vlm.replace_results(ordered_results)
        vlm.save_cache()

    def _serialize_vlm_stats(self, metrics: Dict[str, object]) -> Dict[str, object]:
        latencies = list(metrics.get("latencies", []) or [])
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        return {
            "queued": int(metrics.get("queued") or 0),
            "in_flight": int(metrics.get("in_flight") or 0),
            "completed_count": int(metrics.get("completed_count") or 0),
            "failed_count": int(metrics.get("failed_count") or 0),
            "retry_count": int(metrics.get("retry_count") or 0),
            "flush_count": int(metrics.get("flush_count") or 0),
            "concurrency": int(metrics.get("concurrency") or 0),
            "avg_latency_seconds": round(avg_latency, 4),
            "order_invariant_verified": True,
        }

    def _build_vlm_progress_update(
        self,
        *,
        total_input_photos: int,
        cached_photo_ids: set[str],
        runtime_seconds: float,
        metrics: Dict[str, object],
    ) -> Dict:
        serialized = self._serialize_vlm_stats(metrics)
        completed_count = serialized["completed_count"]
        percent = 0
        if total_input_photos:
            percent = round((completed_count / total_input_photos) * 100)
        return {
            "message": "进行视觉分析",
            "photo_count": total_input_photos,
            "processed": completed_count,
            "cached_hits": len(cached_photo_ids),
            "queued": serialized["queued"],
            "in_flight": serialized["in_flight"],
            "completed_count": completed_count,
            "failed_count": serialized["failed_count"],
            "retry_count": serialized["retry_count"],
            "flush_count": serialized["flush_count"],
            "concurrency": serialized["concurrency"],
            "avg_latency_seconds": serialized["avg_latency_seconds"],
            "percent": percent,
            "runtime_seconds": round(runtime_seconds, 4),
        }

    def _build_face_recognition_payload(self, photos: List, face_output: Dict) -> Dict:
        failure_map = self._group_failures_by_image()
        photo_map = {photo.photo_id: photo for photo in photos}
        image_entries = []

        for image in face_output.get("images", []):
            photo = photo_map.get(image.get("photo_id"))
            boxed_path = getattr(photo, "boxed_path", None) if photo else None
            compressed_path = getattr(photo, "compressed_path", None) if photo else None
            original_path = getattr(photo, "original_path", None) if photo else None
            current_path = getattr(photo, "path", image.get("path")) if photo else image.get("path")

            image_entries.append({
                "image_id": image.get("photo_id"),
                "filename": image.get("filename"),
                "source_hash": image.get("source_hash"),
                "timestamp": image.get("timestamp"),
                "status": "processed",
                "detection_seconds": image.get("detection_seconds", 0.0),
                "embedding_seconds": image.get("embedding_seconds", 0.0),
                "original_image_url": self._public_url(original_path or current_path),
                "display_image_url": self._public_url(boxed_path or compressed_path or current_path),
                "boxed_image_url": self._public_url(boxed_path) if boxed_path else None,
                "compressed_image_url": self._public_url(compressed_path) if compressed_path else None,
                "location": image.get("location"),
                "face_count": len(image.get("faces", [])),
                "faces": [
                    {
                        **face,
                        "image_id": image.get("photo_id"),
                        "source_hash": image.get("source_hash"),
                        "boxed_image_url": self._public_url(boxed_path) if boxed_path else None,
                    }
                    for face in image.get("faces", [])
                ],
                "failures": failure_map.get(image.get("photo_id"), []),
            })

        known_image_ids = {entry["image_id"] for entry in image_entries}
        for photo in photos:
            if photo.photo_id in known_image_ids:
                continue
            image_entries.append({
                "image_id": photo.photo_id,
                "filename": photo.filename,
                "source_hash": photo.source_hash,
                "timestamp": photo.timestamp.isoformat(),
                "status": "abandoned_by_policy" if photo.processing_errors.get("face_policy") else "skipped",
                "detection_seconds": 0.0,
                "embedding_seconds": 0.0,
                "original_image_url": self._public_url(photo.original_path or photo.path),
                "display_image_url": self._public_url(photo.boxed_path or photo.compressed_path or photo.original_path or photo.path),
                "boxed_image_url": self._public_url(photo.boxed_path) if photo.boxed_path else None,
                "compressed_image_url": self._public_url(photo.compressed_path) if photo.compressed_path else None,
                "location": photo.location,
                "face_count": 0,
                "faces": [],
                "failures": failure_map.get(photo.photo_id, []),
            })

        person_groups = self._build_person_groups(
            image_entries,
            photo_map,
            face_output.get("persons", []),
            face_output.get("primary_person_id"),
        )

        return {
            **face_output,
            "images": image_entries,
            "person_groups": person_groups,
            "failed_images": self.failed_images,
        }

    def _build_face_report(self, face_payload: Dict) -> Dict:
        metrics = face_payload.get("metrics", {})
        engine = face_payload.get("engine", {})
        images = face_payload.get("images", [])
        person_groups = face_payload.get("person_groups", [])
        no_face_images = [
            {
                "image_id": image["image_id"],
                "filename": image["filename"],
            }
            for image in face_payload.get("images", [])
            if image.get("face_count", 0) == 0 and image.get("status") != "abandoned_by_policy"
        ]
        failed_items = [
            {
                "image_id": item["image_id"],
                "filename": item["filename"],
                "step": item["step"],
                "error": item["error"],
            }
            for item in self.failed_images
        ]
        ambiguous_faces = 0
        low_quality_faces = 0
        new_person_from_ambiguity = 0
        for image in images:
            for face in image.get("faces", []):
                if face.get("quality_score", 0.0) < 0.40:
                    low_quality_faces += 1
                if face.get("match_decision") in {"gray_match", "new_person_from_ambiguity"}:
                    ambiguous_faces += 1
                if face.get("match_decision") == "new_person_from_ambiguity":
                    new_person_from_ambiguity += 1
        detection_seconds = sum(float(image.get("detection_seconds") or 0.0) for image in images)
        embedding_seconds = sum(float(image.get("embedding_seconds") or 0.0) for image in images)
        total_processing_seconds = detection_seconds + embedding_seconds
        average_image_seconds = total_processing_seconds / len(images) if images else 0.0

        return {
            "status": "completed",
            "generated_at": datetime.now().isoformat(),
            "primary_person_id": face_payload.get("primary_person_id"),
            "total_images": metrics.get("total_images", 0),
            "total_faces": metrics.get("total_faces", 0),
            "total_persons": metrics.get("total_persons", 0),
            "failed_images": len(self.failed_images),
            "ambiguous_faces": ambiguous_faces,
            "low_quality_faces": low_quality_faces,
            "new_person_from_ambiguity": new_person_from_ambiguity,
            "no_face_images": no_face_images,
            "failed_items": failed_items,
            "engine": {
                "model_name": engine.get("model_name"),
                "providers": engine.get("providers", []),
            },
            "timings": {
                "detection_seconds": round(detection_seconds, 6),
                "embedding_seconds": round(embedding_seconds, 6),
                "total_seconds": round(total_processing_seconds, 6),
                "average_image_seconds": round(average_image_seconds, 6),
            },
            "processing": {
                "original_uploads_preserved": True,
                "preview_format": "webp",
                "boxed_format": "webp",
                "recognition_input": (
                    "原始上传文件保持不变；当前任务已启用 Live Photo / HEIC 静态 JPEG 预处理，后续人脸识别、VLM 与下游链路统一使用标准化 JPEG 工作图"
                    if bool(self.task_options.get("normalize_live_photos"))
                    else "原始上传文件保持不变；HEIC 或带方向标签的图片会先生成标准朝向的 JPEG 工作图供识别使用"
                ),
            },
            "precision_enhancements": self._precision_enhancements(),
            "score_guide": {
                "detection_score": "检测分数表示模型对该人脸框本身的置信度，越高越可信",
                "similarity": "相似度表示这张脸与已入库人物特征的接近程度，越高越像同一个人",
            },
            "persons": [
                {
                "person_id": group["person_id"],
                    "is_primary": group.get("is_primary", False),
                    "photo_count": group.get("photo_count", 0),
                    "face_count": group.get("face_count", 0),
                    "avg_score": group.get("avg_score", 0.0),
                    "avg_quality": group.get("avg_quality", 0.0),
                    "high_quality_face_count": group.get("high_quality_face_count", 0),
                }
                for group in person_groups
            ],
        }

    def _precision_enhancements(self) -> List[str]:
        base_items = [
            "原始上传文件原样保留，后续可直接读取完整 EXIF",
            "识别与画框统一使用方向归一化后的工作图，避免展示翻转与坐标错位",
            f"最小人脸尺寸阈值调整为 {FACE_MIN_SIZE}px，提升小脸检出能力",
        ]
        if self.task_version in {
            TASK_VERSION_V0315,
            TASK_VERSION_V0317,
            TASK_VERSION_V0317_HEAVY,
            TASK_VERSION_V0321_2,
            TASK_VERSION_V0321_3,
        }:
            return [
                *base_items,
                "MediaPipe Face Landmarker 为每张脸补充 pose 诊断，MediaPipe 失败时回退到 InsightFace pose",
                "同图人物复用保护已开启，避免多人合影被错误并入同一个 person_id",
            ]
        return base_items

    def _build_person_groups(
        self,
        image_entries: List[Dict],
        photo_map: Dict[str, object],
        persons: List[Dict],
        primary_person_id: Optional[str],
    ) -> List[Dict]:
        stats_map = {person["person_id"]: person for person in persons}
        grouped: Dict[str, Dict] = {}

        for image in image_entries:
            photo = photo_map.get(image["image_id"])
            for face in image["faces"]:
                person_id = face["person_id"]
                group = grouped.setdefault(
                    person_id,
                    {
                        "person_id": person_id,
                        "is_primary": person_id == primary_person_id,
                        "photo_count": stats_map.get(person_id, {}).get("photo_count", 0),
                        "face_count": stats_map.get(person_id, {}).get("face_count", 0),
                        "avg_score": stats_map.get(person_id, {}).get("avg_score", 0.0),
                        "avg_quality": stats_map.get(person_id, {}).get("avg_quality", 0.0),
                        "high_quality_face_count": stats_map.get(person_id, {}).get("high_quality_face_count", 0),
                        "avatar_url": None,
                        "images": [],
                        "_seen_images": set(),
                        "_best_score": -1.0,
                        "_best_photo": None,
                        "_best_face": None,
                    },
                )

                if image["image_id"] not in group["_seen_images"]:
                    group["_seen_images"].add(image["image_id"])
                    group["images"].append({
                        "image_id": image["image_id"],
                        "filename": image["filename"],
                        "timestamp": image.get("timestamp"),
                        "display_image_url": image.get("display_image_url"),
                        "boxed_image_url": image.get("boxed_image_url"),
                        "source_hash": image.get("source_hash"),
                        "face_id": face["face_id"],
                        "score": face["score"],
                        "similarity": face["similarity"],
                        "quality_score": face.get("quality_score", 0.0),
                        "quality_flags": face.get("quality_flags", []),
                        "match_decision": face.get("match_decision"),
                        "match_reason": face.get("match_reason"),
                        "pose_yaw": face.get("pose_yaw"),
                        "pose_pitch": face.get("pose_pitch"),
                        "pose_roll": face.get("pose_roll"),
                        "pose_bucket": face.get("pose_bucket"),
                        "eye_visibility_ratio": face.get("eye_visibility_ratio"),
                        "landmark_detected": face.get("landmark_detected", False),
                        "landmark_source": face.get("landmark_source"),
                    })

                if photo is not None and float(face["score"]) > group["_best_score"]:
                    group["_best_score"] = float(face["score"])
                    group["_best_photo"] = photo
                    group["_best_face"] = face

        person_groups = []
        for person_id, group in grouped.items():
            if group["_best_photo"] is not None and group["_best_face"] is not None:
                avatar_path = self.image_processor.save_face_crop(group["_best_photo"], group["_best_face"])
                if avatar_path:
                    group["avatar_url"] = self._public_url(avatar_path)

            images = sorted(group["images"], key=lambda item: item.get("timestamp") or "")
            person_groups.append({
                "person_id": person_id,
                "is_primary": group["is_primary"],
                "photo_count": group["photo_count"] or len(images),
                "face_count": group["face_count"] or len(images),
                "avg_score": group["avg_score"],
                "avg_quality": group["avg_quality"],
                "high_quality_face_count": group["high_quality_face_count"],
                "avatar_url": group["avatar_url"],
                "images": images,
            })

        person_groups.sort(
            key=lambda item: (
                1 if item["is_primary"] else 0,
                item["photo_count"],
                item["face_count"],
                item["person_id"],
            ),
            reverse=True,
        )
        return person_groups

    def _group_failures_by_image(self) -> Dict[str, List[Dict]]:
        grouped: Dict[str, List[Dict]] = {}
        for failure in self.failed_images:
            grouped.setdefault(failure["image_id"], []).append(failure)
        return grouped

    def _record_failure(self, image_id: str, filename: str, path: str, step: str, error: str):
        self.failed_images.append({
            "image_id": image_id,
            "filename": filename,
            "path": path,
            "step": step,
            "error": error,
        })

    def _count_uploaded_files(self) -> int:
        stored_count = len([path for path in self.upload_dir.iterdir() if path.is_file()]) if self.upload_dir.exists() else 0
        return stored_count + len(self.initial_upload_failures)

    def _load_upload_failures(self) -> List[Dict]:
        payload = {}
        if self.upload_failures_path.exists():
            payload = load_json(str(self.upload_failures_path))
        failures = payload.get("failures", [])
        return failures if isinstance(failures, list) else []

    def _apply_image_policies(self, photos: List) -> List:
        if not self.user_id:
            return photos

        processable = []
        for photo in photos:
            if self.face_review_store.is_image_abandoned(self.user_id, photo.source_hash):
                photo.processing_errors["face_policy"] = "abandoned_by_policy"
                self.warnings.append({
                    "stage": "face_recognition",
                    "message": f"{photo.filename} 已被标记为 abandon，跳过人脸识别",
                })
                continue
            processable.append(photo)
        return processable

    def _notify(self, callback: Optional[Callable[[str, Dict], None]], stage: str, payload: Dict):
        if callback:
            callback(stage, payload)

    def _write_profile_report(self, profile_markdown: str):
        if not profile_markdown:
            return
        self.profile_report_path.write_text(profile_markdown, encoding="utf-8")

    def _serialize_event(self, event) -> Dict:
        return {
            "event_id": event.event_id,
            "date": event.date,
            "time_range": event.time_range,
            "duration": event.duration,
            "title": event.title,
            "type": event.type,
            "participants": list(event.participants or []),
            "location": event.location,
            "description": event.description,
            "photo_count": event.photo_count,
            "confidence": event.confidence,
            "reason": event.reason,
            "narrative": event.narrative,
            "narrative_synthesis": event.narrative_synthesis,
            "meta_info": dict(event.meta_info or {}),
            "objective_fact": dict(event.objective_fact or {}),
            "social_interaction": dict(event.social_interaction or {}),
            "social_dynamics": list(event.social_dynamics or []),
            "original_image_ids": list(event.evidence_photos or []),
            "evidence_photos": list(event.evidence_photos or []),
            "lifestyle_tags": list(event.lifestyle_tags or []),
            "tags": list(event.tags or []),
            "social_slices": list(event.social_slices or []),
            "persona_evidence": dict(event.persona_evidence or {}),
        }

    def _serialize_relationship(self, relationship) -> Dict:
        evidence = dict(relationship.evidence or {})
        return {
            "person_id": relationship.person_id,
            "relationship_type": relationship.relationship_type,
            "label": relationship.label,
            "confidence": relationship.confidence,
            "supporting_fact_ids": list(evidence.get("supporting_fact_ids", evidence.get("supporting_event_ids", [])) or []),
            "supporting_original_image_ids": list(evidence.get("supporting_photo_ids", []) or []),
            "evidence": evidence,
            "reason": relationship.reason,
        }

    def _build_face_stage_preview(self, face_payload: Dict) -> Dict:
        preview = {
            "primary_person_id": face_payload.get("primary_person_id"),
            "metrics": dict(face_payload.get("metrics", {}) or {}),
            "person_groups": list(face_payload.get("person_groups", [])[:8]),
            "images": list(face_payload.get("images", [])[:12]),
            "failed_images": list(face_payload.get("failed_images", [])[:12]),
        }
        return preview

    def _build_vlm_stage_progress(
        self,
        results: List[Dict],
        cached_photo_ids: set[str],
        *,
        total_input_photos: int,
        runtime_seconds: float,
        failed_items: Optional[List[Dict[str, object]]] = None,
        stats: Optional[Dict[str, object]] = None,
    ) -> Dict:
        previews = []
        for item in results[:12]:
            analysis = dict(item.get("vlm_analysis", {}) or {})
            previews.append(
                {
                    "photo_id": item.get("photo_id"),
                    "filename": item.get("filename"),
                    "source_type": item.get("source_type") or dict(item.get("vlm_analysis") or {}).get("source_type"),
                    "original_image_ids": [item.get("photo_id")] if item.get("photo_id") else [],
                    "summary": analysis.get("summary"),
                    "ocr_hits": list(analysis.get("ocr_hits", [])[:10]),
                    "brands": list(analysis.get("brands", [])[:10]),
                    "place_candidates": list(analysis.get("place_candidates", [])[:5]),
                    "route_plan_clues": list(analysis.get("route_plan_clues", [])[:5]),
                    "health_treatment_clues": list(analysis.get("health_treatment_clues", [])[:5]),
                    "object_last_seen_clues": list(analysis.get("object_last_seen_clues", [])[:5]),
                    "uncertainty": list(analysis.get("uncertainty", [])[:5]),
                }
            )
        stats_payload = dict(stats or {})
        return {
            "message": "VLM 识别完成",
            "completed": True,
            "processed": len(results),
            "cached_hits": len(cached_photo_ids),
            "queued": int(stats_payload.get("queued") or 0),
            "in_flight": int(stats_payload.get("in_flight") or 0),
            "completed_count": int(stats_payload.get("completed_count") or len(results)),
            "failed_count": int(stats_payload.get("failed_count") or len(failed_items or [])),
            "retry_count": int(stats_payload.get("retry_count") or 0),
            "flush_count": int(stats_payload.get("flush_count") or 0),
            "concurrency": int(stats_payload.get("concurrency") or max(1, VLM_MAX_CONCURRENCY)),
            "avg_latency_seconds": float(stats_payload.get("avg_latency_seconds") or 0.0),
            "percent": 100 if not failed_items else round((len(results) / total_input_photos) * 100) if total_input_photos else 0,
            "runtime_seconds": round(runtime_seconds, 4),
            "total_input_photos": total_input_photos,
            "failed_items": list(failed_items or [])[:20],
            "order_invariant_verified": bool(stats_payload.get("order_invariant_verified", True)),
            "vlm_results_preview": previews,
            "vlm_cache_url": self._public_url(self.vlm_cache_path),
            "vlm_failures_url": self._public_url(self.vlm_failures_path) if self.vlm_failures_path.exists() else None,
        }

    def _build_llm_stage_progress(
        self,
        memory_contract: Dict[str, object],
        llm_chunk_artifacts: Dict[str, object],
        *,
        profile_markdown: str = "",
        runtime_seconds: float,
    ) -> Dict:
        contract_preview = {
            "facts": list(memory_contract.get("facts", [])[:12]),
            "observations": list(memory_contract.get("observations", [])[:12]),
            "claims": list(memory_contract.get("claims", [])[:12]),
            "relationship_hypotheses": list(memory_contract.get("relationship_hypotheses", [])[:12]),
            "profile_deltas": list(memory_contract.get("profile_deltas", [])[:12]),
            "uncertainty": list(memory_contract.get("uncertainty", [])[:12]),
        }
        return {
            "message": "LLM 改写完成",
            "completed": True,
            "substage": "completed",
            "percent": 100,
            "runtime_seconds": round(runtime_seconds, 4),
            "memory_contract_preview": contract_preview,
            "profile_markdown_preview": profile_markdown[:4000] if profile_markdown else "",
            "memory_contract_url": self._public_url(self.llm_contract_path) if self.llm_contract_path.exists() else None,
            "llm_chunks_url": self._public_url(self.llm_chunks_path) if self.llm_chunks_path.exists() else None,
            "slice_contracts_url": self._public_url(self.llm_slice_contracts_path) if self.llm_slice_contracts_path.exists() else None,
            "event_merges_url": self._public_url(self.llm_event_merges_path) if self.llm_event_merges_path.exists() else None,
            "pre_relationship_contract_url": self._public_url(self.llm_pre_relationship_contract_path) if self.llm_pre_relationship_contract_path.exists() else None,
            "llm_chunk_artifacts_preview": dict(llm_chunk_artifacts or {}),
        }

    def _persist_llm_intermediate_outputs(
        self,
        *,
        llm_chunk_artifacts: Dict[str, object],
        partial_contract: Dict[str, object],
    ) -> None:
        slice_records = list(llm_chunk_artifacts.get("slice_contract_records", []) or [])
        if slice_records:
            self._write_jsonl(self.llm_slice_contracts_path, slice_records)
        event_merges = list(llm_chunk_artifacts.get("event_merge_records", []) or [])
        if event_merges:
            self._write_jsonl(self.llm_event_merges_path, event_merges)
        pre_relationship_contract = llm_chunk_artifacts.get("pre_relationship_contract")
        if isinstance(pre_relationship_contract, dict) and pre_relationship_contract:
            save_json(pre_relationship_contract, str(self.llm_pre_relationship_contract_path))
        elif partial_contract:
            save_json(partial_contract, str(self.llm_pre_relationship_contract_path))

    def _clear_legacy_outputs_for_revision_first_family(self) -> None:
        legacy_paths = [
            self.llm_contract_path,
            self.llm_chunks_path,
            self.llm_slice_contracts_path,
            self.llm_event_merges_path,
            self.llm_pre_relationship_contract_path,
            self.profile_report_path,
            self.output_dir / "memory",
        ]
        for path in legacy_paths:
            if not path.exists():
                continue
            if path.is_dir():
                for child in sorted(path.rglob("*"), reverse=True):
                    if child.is_file():
                        child.unlink(missing_ok=True)
                    elif child.is_dir():
                        child.rmdir()
                path.rmdir()
            else:
                path.unlink(missing_ok=True)

    def _write_jsonl(self, path: Path, records: List[Dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for item in records:
                handle.write(f"{json.dumps(item, ensure_ascii=False)}\n")

    def _append_jsonl(self, path: Path, record: Dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{json.dumps(record, ensure_ascii=False)}\n")

    def _build_memory_stage_progress(self, memory: Dict[str, object], *, runtime_seconds: float) -> Dict:
        storage = dict(memory.get("storage", {}) or {})
        neo4j_storage = dict(storage.get("neo4j", {}) or {})
        neo4j_nodes = dict(neo4j_storage.get("nodes", {}) or {})
        neo4j_preview = {
            "nodes": {
                key: list(value[:8]) if isinstance(value, list) else value
                for key, value in neo4j_nodes.items()
            },
            "edges": list((neo4j_storage.get("edges", []) or [])[:20]),
            "focus_graph": neo4j_storage.get("focus_graph"),
        }
        return {
            "message": "记忆框架落位完成",
            "completed": True,
            "percent": 100,
            "runtime_seconds": round(runtime_seconds, 4),
            "redis_preview": dict(storage.get("redis", {}) or {}),
            "neo4j_preview": neo4j_preview,
            "memory_transparency_preview": dict(memory.get("transparency", {}) or {}),
            "artifacts": dict(memory.get("artifacts", {}) or {}),
        }

    def _stage_percent(self, processed: int, total: int) -> int:
        if total <= 0:
            return 0
        return int(round(min(100.0, (max(processed, 0) / total) * 100.0)))

    def _public_url(self, file_path: Optional[Path | str]) -> Optional[str]:
        if not file_path:
            return None

        path = Path(file_path)
        try:
            relative = path.relative_to(self.task_dir)
        except ValueError:
            return None

        return self.asset_store.asset_url(self.task_id, relative.as_posix())
