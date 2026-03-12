"""
任务级 pipeline 服务。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from config import FACE_MIN_SIZE, MAX_UPLOAD_PHOTOS, RUNS_URL_PREFIX
from services.face_recognition import FaceRecognition
from services.image_processor import ImageProcessor
from utils import load_json, save_json


class MemoryPipelineService:
    """将现有多模态流程封装为任务级服务。"""

    def __init__(self, task_id: str, task_dir: str, public_runs_prefix: str = RUNS_URL_PREFIX):
        self.task_id = task_id
        self.task_dir = Path(task_dir)
        self.public_runs_prefix = public_runs_prefix.rstrip("/")

        self.upload_dir = self.task_dir / "uploads"
        self.cache_dir = self.task_dir / "cache"
        self.output_dir = self.task_dir / "output"
        self.upload_failures_path = self.task_dir / "upload_failures.json"

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
        self.initial_upload_failures = self._load_upload_failures()
        self.failed_images: List[Dict] = list(self.initial_upload_failures)
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

        self.face_recognition.save()
        face_output = self.face_recognition.get_face_output()
        face_payload = self._build_face_recognition_payload(photos, face_output)

        detailed_output = {
            "task_id": self.task_id,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_uploaded": self._count_uploaded_files(),
                "loaded_images": len(photos),
                "failed_images": len(self.failed_images),
                "face_processed_images": len(face_ready_photos),
                "vlm_processed_images": 0,
                "total_faces": face_output.get("metrics", {}).get("total_faces", 0),
                "total_persons": face_output.get("metrics", {}).get("total_persons", 0),
                "primary_person_id": primary_person_id,
            },
            "face_recognition": face_payload,
            "face_report": self._build_face_report(face_payload),
            "failed_images": self.failed_images,
            "warnings": self.warnings,
            "events": [],
            "relationships": [],
            "profile_markdown": "",
            "artifacts": {},
        }

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
                        "boxed_image_url": self._public_url(boxed_path) if boxed_path else None,
                    }
                    for face in image.get("faces", [])
                ],
                "failures": failure_map.get(image.get("photo_id"), []),
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
            if image.get("face_count", 0) == 0
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
                "recognition_input": "原始上传文件保持不变；HEIC 或带方向标签的图片会先生成标准朝向的 JPEG 工作图供识别使用",
            },
            "precision_enhancements": [
                "原始上传文件原样保留，后续可直接读取完整 EXIF",
                "识别与画框统一使用方向归一化后的工作图，避免展示翻转与坐标错位",
                f"最小人脸尺寸阈值调整为 {FACE_MIN_SIZE}px，提升小脸检出能力",
            ],
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
                }
                for group in person_groups
            ],
        }

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
                        "face_id": face["face_id"],
                        "score": face["score"],
                        "similarity": face["similarity"],
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
