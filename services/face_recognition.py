"""
人脸识别模块（基于本地 face-recognition 项目）
"""
import os
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    FACE_DET_THRESHOLD,
    FACE_INDEX_PATH,
    FACE_MAX_SIDE,
    FACE_MIN_SIZE,
    FACE_MODEL_NAME,
    FACE_OUTPUT_PATH,
    FACE_PROVIDERS,
    FACE_RECOGNITION_SRC_PATH,
    FACE_SIM_THRESHOLD,
    FACE_STATE_PATH,
)
from models import Person, Photo
from utils import load_json, save_json


def _load_face_recognition_modules():
    src_path = os.path.abspath(FACE_RECOGNITION_SRC_PATH)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    try:
        from face_recognition.config import PipelineConfig
        from face_recognition.engine import FaceEngine
        from face_recognition.image_io import load_image, resize_image
        from face_recognition.index_store import SimilarityIndexStore
    except Exception as exc:
        raise RuntimeError(
            f"无法加载本地 face-recognition 项目，请检查路径: {src_path}"
        ) from exc

    return PipelineConfig, FaceEngine, SimilarityIndexStore, load_image, resize_image


class FaceRecognition:
    """使用本地 face-recognition 引擎的人脸识别器"""

    def __init__(self):
        PipelineConfig, FaceEngine, SimilarityIndexStore, load_image, resize_image = (
            _load_face_recognition_modules()
        )

        self.config = PipelineConfig.from_args(
            input_dir=os.getcwd(),
            db_path=FACE_STATE_PATH,
            index_path=FACE_INDEX_PATH,
            max_side=FACE_MAX_SIDE,
            det_threshold=FACE_DET_THRESHOLD,
            sim_threshold=FACE_SIM_THRESHOLD,
            providers=FACE_PROVIDERS,
            model_name=FACE_MODEL_NAME,
        )
        self.engine = FaceEngine(self.config)
        self.index_store = SimilarityIndexStore(Path(FACE_INDEX_PATH))
        self.load_image = load_image
        self.resize_image = resize_image

        self.state = self._load_state()
        self.faiss_person_map = {
            int(key): value for key, value in self.state.get("faiss_person_map", {}).items()
        }
        self.persons = self._restore_persons()
        self.photo_results: Dict[str, Dict] = {}
        self.primary_person_id = self.state.get("primary_person_id")

        if self.index_store.committed_count != len(self.faiss_person_map):
            raise RuntimeError(
                "faces.index 与 face_recognition_state.json 不一致，请清理缓存后重试"
            )

    def _load_state(self) -> Dict:
        data = load_json(FACE_STATE_PATH)
        if data:
            return data

        return {
            "version": 1,
            "next_person_number": 1,
            "primary_person_id": None,
            "faiss_person_map": {},
            "persons": {},
            "images": {},
            "engine": {
                "src_path": os.path.abspath(FACE_RECOGNITION_SRC_PATH),
                "model_name": FACE_MODEL_NAME,
                "providers": list(FACE_PROVIDERS),
                "max_side": FACE_MAX_SIDE,
                "det_threshold": FACE_DET_THRESHOLD,
                "sim_threshold": FACE_SIM_THRESHOLD,
                "index_path": FACE_INDEX_PATH,
            },
        }

    def _restore_persons(self) -> Dict[str, Person]:
        restored = {}
        for person_id, person_data in self.state.get("persons", {}).items():
            restored[person_id] = Person(
                person_id=person_id,
                name=person_data.get("label", ""),
                features=[],
                photo_count=person_data.get("photo_count", 0),
                first_seen=self._parse_datetime(person_data.get("first_seen")),
                last_seen=self._parse_datetime(person_data.get("last_seen")),
                avg_confidence=person_data.get("avg_score", 0.0),
            )
        return restored

    def _parse_datetime(self, value: Optional[str]):
        if not value:
            return None
        try:
            from datetime import datetime

            return datetime.fromisoformat(value)
        except Exception:
            return None

    def _next_person_id(self) -> str:
        current = int(self.state.get("next_person_number", 1))
        self.state["next_person_number"] = current + 1
        return f"Person_{current:03d}"

    def _ensure_person_state(self, person_id: str, face_id: str) -> Dict:
        people_state = self.state.setdefault("persons", {})
        if person_id not in people_state:
            people_state[person_id] = {
                "person_id": person_id,
                "representative_face_id": face_id,
                "photo_count": 0,
                "face_count": 0,
                "first_seen": None,
                "last_seen": None,
                "avg_score": 0.0,
                "avg_similarity": 0.0,
                "sample_photo_ids": [],
                "label": "",
            }
        return people_state[person_id]

    def process_photo(self, photo: Photo) -> List[Dict]:
        """
        处理单张照片，输出 face-recognition 原生风格结果

        Returns:
            [
                {
                    "face_id": "...",
                    "person_id": "Person_001",
                    "score": 0.98,
                    "similarity": 0.87,
                    "faiss_id": 12,
                    "bbox": [x1, y1, x2, y2],
                    "bbox_xywh": {"x": 10, "y": 20, "w": 30, "h": 40},
                    "kps": [...]
                }
            ]
        """
        image_hash = self._hash_file(photo.path)
        cached_image = self.state.get("images", {}).get(image_hash)
        if cached_image:
            photo.faces = [dict(face) for face in cached_image.get("faces", [])]
            photo.face_image_hash = image_hash
            self.photo_results[photo.photo_id] = cached_image
            return photo.faces

        raw_image = self.load_image(Path(photo.path))
        resized_image = self.resize_image(raw_image, self.config.max_side)
        engine_result = self.engine.detect_and_embed(resized_image.pixels)

        scale_x = raw_image.width / resized_image.width
        scale_y = raw_image.height / resized_image.height

        pending_embeddings = []
        pending_person_map = {}
        faces_output = []
        seen_persons_in_photo = set()

        for detected_face in engine_result.faces:
            bbox = self._scale_bbox(
                detected_face.bbox,
                scale_x,
                scale_y,
                raw_image.width,
                raw_image.height,
            )
            min_dim = min(bbox["bbox_xywh"]["w"], bbox["bbox_xywh"]["h"])
            if min_dim < FACE_MIN_SIZE:
                continue

            search = self.index_store.search(
                detected_face.embedding,
                pending_embeddings=pending_embeddings,
            )
            matched_person_id = None
            if search.faiss_id is not None:
                matched_person_id = self.faiss_person_map.get(search.faiss_id)
                if matched_person_id is None:
                    matched_person_id = pending_person_map.get(search.faiss_id)

            similarity = float(search.score) if search.score is not None else 0.0
            if (
                matched_person_id is not None
                and search.score is not None
                and search.score > self.config.sim_threshold
            ):
                person_id = matched_person_id
            else:
                person_id = self._next_person_id()

            faiss_id = self.index_store.committed_count + len(pending_embeddings)
            pending_embeddings.append(detected_face.embedding)
            pending_person_map[faiss_id] = person_id
            self.faiss_person_map[faiss_id] = person_id

            face_id = str(uuid.uuid4())
            face_output = {
                "face_id": face_id,
                "person_id": person_id,
                "score": float(detected_face.score),
                "similarity": similarity,
                "faiss_id": faiss_id,
                "bbox": bbox["bbox"],
                "bbox_xywh": bbox["bbox_xywh"],
                "kps": detected_face.kps,
            }
            faces_output.append(face_output)

            person_state = self._ensure_person_state(person_id, face_id)
            person_state["face_count"] += 1
            person_state["avg_score"] = self._rolling_average(
                person_state["avg_score"],
                person_state["face_count"],
                float(detected_face.score),
            )
            person_state["avg_similarity"] = self._rolling_average(
                person_state["avg_similarity"],
                person_state["face_count"],
                similarity,
            )
            if person_id not in seen_persons_in_photo:
                seen_persons_in_photo.add(person_id)
                person_state["photo_count"] += 1
                person_state["first_seen"] = self._min_datetime(
                    person_state.get("first_seen"),
                    photo.timestamp.isoformat(),
                )
                person_state["last_seen"] = self._max_datetime(
                    person_state.get("last_seen"),
                    photo.timestamp.isoformat(),
                )
                if photo.photo_id not in person_state["sample_photo_ids"] and len(
                    person_state["sample_photo_ids"]
                ) < 10:
                    person_state["sample_photo_ids"].append(photo.photo_id)

        if pending_embeddings:
            self.index_store.persist_pending(pending_embeddings)

        image_output = {
            "image_hash": image_hash,
            "photo_id": photo.photo_id,
            "filename": photo.filename,
            "path": photo.path,
            "timestamp": photo.timestamp.isoformat(),
            "location": photo.location,
            "width": raw_image.width,
            "height": raw_image.height,
            "faces": faces_output,
            "detection_seconds": engine_result.detection_seconds,
            "embedding_seconds": engine_result.embedding_seconds,
        }
        self.state.setdefault("images", {})[image_hash] = image_output
        self.photo_results[photo.photo_id] = image_output
        photo.face_image_hash = image_hash
        photo.faces = [dict(face) for face in faces_output]

        self._sync_person_cache()
        return photo.faces

    def _sync_person_cache(self):
        primary_person_id = self._compute_primary_person_id()
        self.primary_person_id = primary_person_id
        self.state["primary_person_id"] = primary_person_id

        for person_id, person_state in self.state.get("persons", {}).items():
            label = "Primary" if person_id == primary_person_id else "Person"
            person_state["label"] = label
            self.persons[person_id] = Person(
                person_id=person_id,
                name=label,
                features=[],
                photo_count=person_state.get("photo_count", 0),
                first_seen=self._parse_datetime(person_state.get("first_seen")),
                last_seen=self._parse_datetime(person_state.get("last_seen")),
                avg_confidence=person_state.get("avg_score", 0.0),
            )

    def _compute_primary_person_id(self) -> Optional[str]:
        if self.photo_results:
            counts = {}
            for image in self.photo_results.values():
                seen_in_photo = set()
                for face in image.get("faces", []):
                    person_id = face["person_id"]
                    stats = counts.setdefault(
                        person_id,
                        {"photo_count": 0, "face_count": 0, "first_seen": image.get("timestamp", "")},
                    )
                    stats["face_count"] += 1
                    if person_id not in seen_in_photo:
                        seen_in_photo.add(person_id)
                        stats["photo_count"] += 1
                        first_seen = image.get("timestamp", "")
                        if first_seen and (not stats["first_seen"] or first_seen < stats["first_seen"]):
                            stats["first_seen"] = first_seen

            if counts:
                ranked = sorted(
                    counts.items(),
                    key=lambda item: (
                        item[1]["photo_count"],
                        item[1]["face_count"],
                        item[1]["first_seen"],
                    ),
                    reverse=True,
                )
                return ranked[0][0]

        people_state = self.state.get("persons", {})
        if not people_state:
            return None

        ranked = sorted(
            people_state.values(),
            key=lambda item: (
                item.get("photo_count", 0),
                item.get("face_count", 0),
                item.get("first_seen") or "",
            ),
            reverse=True,
        )
        return ranked[0]["person_id"]

    def reorder_protagonist(self, photos: list) -> dict:
        """
        保留 face-recognition 的原生 Person_### 编号，仅计算主人物。
        """
        self._sync_person_cache()
        self._save_state()
        return {}

    def _save_state(self):
        self.state["faiss_person_map"] = {
            str(key): value for key, value in sorted(self.faiss_person_map.items())
        }
        self.state["engine"]["providers"] = list(self.engine.applied_providers())
        save_json(self.state, FACE_STATE_PATH)
        save_json(self.get_face_output(), FACE_OUTPUT_PATH)

    def save(self):
        """保存当前识别状态"""
        self._save_state()

    def get_person(self, person_id: str) -> Optional[Person]:
        return self.persons.get(person_id)

    def get_all_persons(self) -> Dict[str, Person]:
        return self.persons

    def get_primary_person_id(self) -> Optional[str]:
        return self.primary_person_id

    def get_face_output(self) -> Dict:
        images = list(self.photo_results.values()) if self.photo_results else []
        if not images:
            images = list(self.state.get("images", {}).values())
        images.sort(key=lambda item: item.get("timestamp", ""))

        current_person_ids = {
            face["person_id"]
            for image in images
            for face in image.get("faces", [])
        }
        persons = [
            person
            for person in self.state.get("persons", {}).values()
            if person.get("person_id") in current_person_ids
        ]
        persons.sort(key=lambda item: item.get("person_id", ""))

        total_faces = sum(len(item.get("faces", [])) for item in images)
        return {
            "engine": {
                **self.state.get("engine", {}),
                "providers": list(self.engine.applied_providers()),
            },
            "primary_person_id": self.primary_person_id,
            "metrics": {
                "total_images": len(images),
                "total_faces": total_faces,
                "total_persons": len(persons),
            },
            "cache_metrics": {
                "total_images": len(self.state.get("images", {})),
                "total_persons": len(self.state.get("persons", {})),
                "indexed_faces": len(self.faiss_person_map),
            },
            "persons": persons,
            "images": images,
        }

    def _hash_file(self, image_path: str) -> str:
        from hashlib import sha256

        digest = sha256()
        with open(image_path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _scale_bbox(
        self,
        bbox: List[float],
        scale_x: float,
        scale_y: float,
        width: int,
        height: int,
    ) -> Dict:
        x1, y1, x2, y2 = bbox[:4]
        scaled_x1 = max(0, min(width, int(round(x1 * scale_x))))
        scaled_y1 = max(0, min(height, int(round(y1 * scale_y))))
        scaled_x2 = max(0, min(width, int(round(x2 * scale_x))))
        scaled_y2 = max(0, min(height, int(round(y2 * scale_y))))

        return {
            "bbox": [scaled_x1, scaled_y1, scaled_x2, scaled_y2],
            "bbox_xywh": {
                "x": scaled_x1,
                "y": scaled_y1,
                "w": max(0, scaled_x2 - scaled_x1),
                "h": max(0, scaled_y2 - scaled_y1),
            },
        }

    def _rolling_average(self, current_avg: float, total_count: int, new_value: float) -> float:
        if total_count <= 1:
            return float(new_value)
        previous_count = total_count - 1
        return (current_avg * previous_count + float(new_value)) / total_count

    def _min_datetime(self, current: Optional[str], new_value: str) -> str:
        if not current or new_value < current:
            return new_value
        return current

    def _max_datetime(self, current: Optional[str], new_value: str) -> str:
        if not current or new_value > current:
            return new_value
        return current
