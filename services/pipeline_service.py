"""
任务级 pipeline 服务。
"""
from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from config import LLM_MODEL, MAX_UPLOAD_PHOTOS, RUNS_URL_PREFIX, VLM_MODEL
from models import Event, Relationship
from services.face_recognition import FaceRecognition
from services.image_processor import ImageProcessor
from services.llm_processor import LLMProcessor
from services.vlm_analyzer import VLMAnalyzer
from utils import save_json


class MemoryPipelineService:
    """将现有多模态流程封装为任务级服务。"""

    def __init__(self, task_id: str, task_dir: str, public_runs_prefix: str = RUNS_URL_PREFIX):
        self.task_id = task_id
        self.task_dir = Path(task_dir)
        self.public_runs_prefix = public_runs_prefix.rstrip("/")

        self.upload_dir = self.task_dir / "uploads"
        self.cache_dir = self.task_dir / "cache"
        self.output_dir = self.task_dir / "output"

        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.image_processor = ImageProcessor(cache_dir=str(self.cache_dir))
        self.face_recognition = FaceRecognition(
            state_path=str(self.cache_dir / "face_recognition_state.json"),
            index_path=str(self.cache_dir / "faces.index"),
            output_path=str(self.cache_dir / "face_recognition_output.json"),
            workspace_dir=str(self.task_dir),
        )
        self.vlm = VLMAnalyzer(cache_path=str(self.cache_dir / "vlm_results.json"))
        self.failed_images: List[Dict] = []
        self.warnings: List[Dict] = []

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
            {"message": "转换 HEIC/JPEG", "photo_count": len(photos)},
        )
        photos = self.image_processor.convert_to_jpeg(photos)

        self._notify(progress_callback, "face_recognition", {"message": "进行人脸识别"})
        face_ready_photos = self._run_face_recognition(photos)

        primary_person_id = None
        if face_ready_photos:
            self.face_recognition.reorder_protagonist(face_ready_photos)
            primary_person_id = self.face_recognition.get_primary_person_id()
            for photo in face_ready_photos:
                photo.primary_person_id = primary_person_id
                if photo.faces:
                    boxed_path = self.image_processor.draw_face_boxes(photo)
                    if boxed_path:
                        photo.boxed_path = boxed_path

        self._notify(progress_callback, "preprocess", {"message": "压缩图片供 VLM 使用"})
        face_ready_photos = self.image_processor.preprocess(face_ready_photos)

        vlm_candidates = []
        for photo in face_ready_photos:
            if photo.boxed_path or photo.compressed_path:
                vlm_candidates.append(photo)
            else:
                self._record_failure(photo.photo_id, photo.filename, photo.path, "preprocess", "未生成可用于后续分析的图片")

        face_db = self.face_recognition.get_all_persons()
        self._notify(progress_callback, "vlm", {"message": "执行 VLM 分析"})
        self._run_vlm(vlm_candidates, face_db, primary_person_id, use_cache)

        events: List[Event] = []
        relationships: List[Relationship] = []
        profile_markdown = None

        if self.vlm.results:
            self._notify(progress_callback, "llm", {"message": "执行 LLM 推理"})
            try:
                llm = LLMProcessor()
                events = llm.extract_events(self.vlm.results, primary_person_id)
                relationships = llm.infer_relationships(self.vlm.results, face_db, primary_person_id)
                profile_markdown = llm.generate_profile(events, relationships, primary_person_id)
            except Exception as exc:
                self.warnings.append({
                    "stage": "llm",
                    "message": str(exc),
                })
        else:
            self.warnings.append({
                "stage": "vlm",
                "message": "VLM 结果为空，已跳过 LLM 阶段",
            })

        self.face_recognition.save()
        face_output = self.face_recognition.get_face_output()

        detailed_output = {
            "task_id": self.task_id,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_uploaded": self._count_uploaded_files(),
                "loaded_images": len(photos),
                "failed_images": len(self.failed_images),
                "face_processed_images": len(face_ready_photos),
                "vlm_processed_images": len(self.vlm.results),
                "total_faces": face_output.get("metrics", {}).get("total_faces", 0),
                "total_persons": face_output.get("metrics", {}).get("total_persons", 0),
                "primary_person_id": primary_person_id,
            },
            "models": {
                "vlm": VLM_MODEL,
                "llm": LLM_MODEL,
            },
            "face_recognition": self._build_face_recognition_payload(photos, face_output),
            "failed_images": self.failed_images,
            "warnings": self.warnings,
            "events": [self._serialize_event(event) for event in events],
            "relationships": [self._serialize_relationship(rel) for rel in relationships],
            "profile_markdown": profile_markdown or "",
            "artifacts": {},
        }

        if profile_markdown:
            profile_path = self.output_dir / "user_profile_report.md"
            profile_path.write_text(profile_markdown, encoding="utf-8")
            detailed_output["artifacts"]["profile_markdown_url"] = self._public_url(profile_path)

        result_path = self.output_dir / "result.json"
        save_json(detailed_output, str(result_path))
        detailed_output["artifacts"]["result_url"] = self._public_url(result_path)
        detailed_output["artifacts"]["face_output_url"] = self._public_url(self.cache_dir / "face_recognition_output.json")

        return detailed_output

    def _run_face_recognition(self, photos: List) -> List:
        successful = []
        for photo in photos:
            try:
                self.face_recognition.process_photo(photo)
                successful.append(photo)
            except Exception as exc:
                self._record_failure(photo.photo_id, photo.filename, photo.path, "face_recognition", str(exc))
        return successful

    def _run_vlm(self, photos: List, face_db: Dict, primary_person_id: Optional[str], use_cache: bool):
        if use_cache and self.vlm.load_cache():
            return

        self.vlm.results = []
        for photo in photos:
            try:
                result = self.vlm.analyze_photo(photo, face_db, primary_person_id)
            except Exception as exc:
                self._record_failure(photo.photo_id, photo.filename, photo.path, "vlm", str(exc))
                continue

            if result:
                self.vlm.add_result(photo, result)
            else:
                self._record_failure(
                    photo.photo_id,
                    photo.filename,
                    photo.path,
                    "vlm",
                    photo.processing_errors.get("vlm", "VLM 分析返回空结果"),
                )

        self.vlm.save_cache()

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
                "timestamp": image.get("timestamp"),
                "status": "processed",
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
                        "boxed_image_url": self._public_url(boxed_path) if boxed_path else None,
                    }
                    for face in image.get("faces", [])
                ],
                "failures": failure_map.get(image.get("photo_id"), []),
            })

        return {
            **face_output,
            "images": image_entries,
            "failed_images": self.failed_images,
        }

    def _group_failures_by_image(self) -> Dict[str, List[Dict]]:
        grouped: Dict[str, List[Dict]] = {}
        for failure in self.failed_images:
            grouped.setdefault(failure["image_id"], []).append(failure)
        return grouped

    def _serialize_event(self, event: Event) -> Dict:
        return {
            "event_id": event.event_id,
            "date": event.date,
            "time_range": event.time_range,
            "duration": event.duration,
            "title": event.title,
            "type": event.type,
            "participants": event.participants,
            "location": event.location,
            "description": event.description,
            "photo_count": event.photo_count,
            "confidence": event.confidence,
            "reason": event.reason,
            "narrative": event.narrative,
            "narrative_synthesis": event.narrative_synthesis,
            "meta_info": event.meta_info,
            "objective_fact": event.objective_fact,
            "social_interaction": event.social_interaction,
            "social_dynamics": event.social_dynamics,
            "evidence_photos": event.evidence_photos,
            "lifestyle_tags": event.lifestyle_tags,
            "tags": event.tags,
            "social_slices": event.social_slices,
            "persona_evidence": event.persona_evidence,
        }

    def _serialize_relationship(self, relationship: Relationship) -> Dict:
        return {
            "person_id": relationship.person_id,
            "relationship_type": relationship.relationship_type,
            "label": relationship.label,
            "confidence": relationship.confidence,
            "evidence": relationship.evidence,
            "reason": relationship.reason,
        }

    def _record_failure(self, image_id: str, filename: str, path: str, step: str, error: str):
        self.failed_images.append({
            "image_id": image_id,
            "filename": filename,
            "path": path,
            "step": step,
            "error": error,
        })

    def _count_uploaded_files(self) -> int:
        return len([path for path in self.upload_dir.iterdir() if path.is_file()]) if self.upload_dir.exists() else 0

    def _notify(self, callback: Optional[Callable[[str, Dict], None]], stage: str, payload: Dict):
        if callback:
            callback(stage, payload)

    def _public_url(self, file_path: Optional[Path | str]) -> Optional[str]:
        if not file_path:
            return None

        path = Path(file_path)
        if not path.exists():
            return None

        try:
            relative = path.relative_to(self.task_dir.parent)
        except ValueError:
            return None

        return f"{self.public_runs_prefix}/{relative.as_posix()}"
