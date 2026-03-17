"""
人脸识别模块（基于本地 face-recognition 项目）
"""
import os
import sys
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from config import (
    DEFAULT_TASK_VERSION,
    FACE_DET_THRESHOLD,
    FACE_INDEX_PATH,
    FACE_MAX_SIDE,
    FACE_MATCH_HIGH_QUALITY_THRESHOLD,
    FACE_MATCH_MARGIN_THRESHOLD,
    FACE_MATCH_MIN_QUALITY_GRAY_ZONE,
    FACE_MATCH_TOP_K,
    FACE_MATCH_WEAK_DELTA,
    FACE_MIN_SIZE,
    FACE_MODEL_NAME,
    FACE_OUTPUT_PATH,
    FACE_PROVIDERS,
    FACE_RECOGNITION_SRC_PATH,
    FACE_SAME_PHOTO_MATCH_THRESHOLD,
    FACE_SIM_THRESHOLD,
    FACE_STATE_PATH,
    TASK_VERSION_V0315,
)
from models import Person, Photo
from services.face_precision import (
    aggregate_candidate_matches,
    compute_face_quality,
    decide_cluster_merge,
    decide_match,
    filter_same_photo_candidates,
    load_strong_threshold,
)
from services.face_landmarks import FaceLandmarkAnalyzer
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

    def __init__(
        self,
        state_path: str = FACE_STATE_PATH,
        index_path: str = FACE_INDEX_PATH,
        output_path: str = FACE_OUTPUT_PATH,
        workspace_dir: Optional[str] = None,
        task_version: str = DEFAULT_TASK_VERSION,
    ):
        PipelineConfig, FaceEngine, SimilarityIndexStore, load_image, resize_image = (
            _load_face_recognition_modules()
        )
        self.state_path = state_path
        self.index_path = index_path
        self.output_path = output_path
        self.workspace_dir = workspace_dir or os.getcwd()
        self.task_version = task_version
        self.pose_evidence_enabled = self.task_version == TASK_VERSION_V0315
        self.same_photo_guard_enabled = self.task_version == TASK_VERSION_V0315

        self.config = PipelineConfig.from_args(
            input_dir=self.workspace_dir,
            db_path=self.state_path,
            index_path=self.index_path,
            max_side=FACE_MAX_SIDE,
            det_threshold=FACE_DET_THRESHOLD,
            sim_threshold=FACE_SIM_THRESHOLD,
            providers=FACE_PROVIDERS,
            model_name=FACE_MODEL_NAME,
        )
        self.engine = FaceEngine(self.config)
        self.index_store = SimilarityIndexStore(Path(self.index_path))
        self.load_image = load_image
        self.resize_image = resize_image
        self.strong_threshold = load_strong_threshold(FACE_SIM_THRESHOLD)
        self.weak_threshold = self.strong_threshold - FACE_MATCH_WEAK_DELTA
        self.margin_threshold = FACE_MATCH_MARGIN_THRESHOLD
        self.min_quality_for_gray_zone = FACE_MATCH_MIN_QUALITY_GRAY_ZONE
        self.high_quality_threshold = FACE_MATCH_HIGH_QUALITY_THRESHOLD
        self.match_top_k = FACE_MATCH_TOP_K
        self.same_photo_match_threshold = FACE_SAME_PHOTO_MATCH_THRESHOLD
        self.landmark_analyzer = FaceLandmarkAnalyzer(enabled=self.pose_evidence_enabled)

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
        data = load_json(self.state_path)
        if data:
            return data

        return {
            "version": 2,
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
                "strong_threshold": self.strong_threshold,
                "weak_threshold": self.weak_threshold,
                "margin_threshold": self.margin_threshold,
                "match_top_k": self.match_top_k,
                "task_version": self.task_version,
                "landmark_source": (
                    "mediapipe-first-with-insightface-fallback"
                    if self.pose_evidence_enabled
                    else "disabled_for_v0312"
                ),
                "index_path": self.index_path,
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
                avg_quality=person_data.get("avg_quality", 0.0),
                high_quality_face_count=person_data.get("high_quality_face_count", 0),
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
                "avg_quality": 0.0,
                "high_quality_face_count": 0,
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
            photo.source_hash = cached_image.get("source_hash", photo.source_hash)
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

            quality = compute_face_quality(raw_image.pixels, bbox["bbox_xywh"])
            if self.pose_evidence_enabled:
                pose = self.landmark_analyzer.analyze_face(
                    raw_image.pixels,
                    bbox["bbox_xywh"],
                    insight_pose=detected_face.insight_pose,
                    insight_landmark_2d_106=detected_face.insight_landmark_2d_106,
                    insight_landmark_3d_68=detected_face.insight_landmark_3d_68,
                )
            else:
                pose = self.landmark_analyzer.analyze_face(raw_image.pixels, bbox["bbox_xywh"])
            if pose.get("pose_bucket") in {"left_profile", "right_profile"}:
                quality["quality_flags"] = sorted(
                    set([*quality["quality_flags"], "profile_face"])
                )
            search_matches = self.index_store.search_many(
                detected_face.embedding,
                pending_embeddings=pending_embeddings,
                top_k=self.match_top_k,
            )
            candidate_summaries = aggregate_candidate_matches(
                search_matches,
                self.faiss_person_map,
                pending_person_map,
            )
            if self.same_photo_guard_enabled:
                candidate_summaries = filter_same_photo_candidates(
                    candidate_summaries,
                    seen_persons_in_photo,
                    same_photo_match_threshold=self.same_photo_match_threshold,
                )
            match_decision = decide_match(
                candidate_summaries,
                float(quality["quality_score"]),
                strong_threshold=self.strong_threshold,
                weak_threshold=self.weak_threshold,
                margin_threshold=self.margin_threshold,
                min_quality_for_gray_zone=self.min_quality_for_gray_zone,
                pose_bucket=str(pose.get("pose_bucket") or "unknown"),
                pose_yaw=float(pose["pose_yaw"]) if pose.get("pose_yaw") is not None else None,
            )

            person_id = match_decision.get("person_id")
            if not person_id:
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
                "similarity": float(match_decision.get("best_similarity") or 0.0),
                "faiss_id": faiss_id,
                "bbox": bbox["bbox"],
                "bbox_xywh": bbox["bbox_xywh"],
                "kps": detected_face.kps,
                "quality_score": float(quality["quality_score"]),
                "quality_flags": list(quality["quality_flags"]),
                "match_decision": str(match_decision["decision"]),
                "match_reason": str(match_decision["reason"]),
                "pose_yaw": pose.get("pose_yaw"),
                "pose_pitch": pose.get("pose_pitch"),
                "pose_roll": pose.get("pose_roll"),
                "pose_bucket": pose.get("pose_bucket"),
                "eye_visibility_ratio": pose.get("eye_visibility_ratio"),
                "landmark_detected": pose.get("landmark_detected"),
                "landmark_source": pose.get("landmark_source"),
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
                float(match_decision.get("best_similarity") or 0.0),
            )
            person_state["avg_quality"] = self._rolling_average(
                person_state["avg_quality"],
                person_state["face_count"],
                float(quality["quality_score"]),
            )
            if float(quality["quality_score"]) >= self.high_quality_threshold:
                person_state["high_quality_face_count"] += 1
            pose_counts = person_state.setdefault("pose_counts", {})
            pose_bucket = str(pose.get("pose_bucket") or "unknown")
            pose_counts[pose_bucket] = int(pose_counts.get(pose_bucket, 0)) + 1
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
            "source_hash": photo.source_hash,
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
                avg_quality=person_state.get("avg_quality", 0.0),
                high_quality_face_count=person_state.get("high_quality_face_count", 0),
            )

    def _select_confident_primary_person_id(self, stats_by_person: Dict[str, Dict]) -> Optional[str]:
        if not stats_by_person:
            return None

        ranked = sorted(
            stats_by_person.items(),
            key=lambda item: (
                int(item[1].get("photo_count", 0)),
                int(item[1].get("face_count", 0)),
                item[1].get("first_seen") or "",
            ),
            reverse=True,
        )
        top_person_id, top_stats = ranked[0]
        top_photo_count = int(top_stats.get("photo_count", 0))

        # 主角推断需要稳定锚点；只出现 1 次或出现次数平票时都视为不可靠。
        if top_photo_count < 2:
            return None

        tied_top_people = [
            person_id
            for person_id, stats in ranked
            if int(stats.get("photo_count", 0)) == top_photo_count
        ]
        if len(tied_top_people) > 1:
            return None

        return top_person_id

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

            primary_person_id = self._select_confident_primary_person_id(counts)
            if primary_person_id:
                return primary_person_id

        people_state = self.state.get("persons", {})
        if not people_state:
            return None

        primary_person_id = self._select_confident_primary_person_id(
            {
                person_id: {
                    "photo_count": person_state.get("photo_count", 0),
                    "face_count": person_state.get("face_count", 0),
                    "first_seen": person_state.get("first_seen") or "",
                }
                for person_id, person_state in people_state.items()
            }
        )
        return primary_person_id

    def reorder_protagonist(self, photos: list) -> dict:
        """
        保留 face-recognition 的原生 Person_### 编号，仅计算主人物。
        """
        merge_summaries: List[Dict[str, object]] = []
        if self.task_version == TASK_VERSION_V0315:
            merge_summaries = self._run_second_pass_cluster_merge()
            if merge_summaries:
                self._refresh_photo_faces(photos)
        self._sync_person_cache()
        if merge_summaries:
            self.state["cluster_merges"] = merge_summaries
        self._save_state()
        return {"cluster_merges": merge_summaries}

    def _save_state(self):
        self.state["faiss_person_map"] = {
            str(key): value for key, value in sorted(self.faiss_person_map.items())
        }
        self.state["engine"]["providers"] = list(self.engine.applied_providers())
        self.state["engine"]["index_path"] = self.index_path
        self.state["engine"]["strong_threshold"] = self.strong_threshold
        self.state["engine"]["weak_threshold"] = self.weak_threshold
        self.state["engine"]["margin_threshold"] = self.margin_threshold
        self.state["engine"]["match_top_k"] = self.match_top_k
        save_json(self.state, self.state_path)
        save_json(self.get_face_output(), self.output_path)

    def _run_second_pass_cluster_merge(self) -> List[Dict[str, object]]:
        merge_summaries: List[Dict[str, object]] = []

        while True:
            candidate = self._best_cluster_merge_candidate()
            if candidate is None:
                break

            target_person_id, source_person_id = self._select_merge_target(
                candidate.left_person_id,
                candidate.right_person_id,
            )
            merge_reason = (
                f"{candidate.reason}; source={source_person_id}; target={target_person_id}"
            )
            self._merge_person_cluster(source_person_id, target_person_id, merge_reason)
            merge_summaries.append(
                {
                    "source_person_id": source_person_id,
                    "target_person_id": target_person_id,
                    "decision": candidate.decision,
                    "reason": merge_reason,
                    "best_similarity": round(candidate.best_similarity, 6),
                    "top_two_mean": round(candidate.top_two_mean, 6),
                    "support_count": candidate.support_count,
                    "profile_bridge": candidate.profile_bridge,
                }
            )

        if merge_summaries:
            self._rebuild_person_state_from_images()
        return merge_summaries

    def _best_cluster_merge_candidate(self):
        person_faces: Dict[str, List[Dict]] = defaultdict(list)
        for image in self.state.get("images", {}).values():
            image_id = image.get("photo_id")
            for face in image.get("faces", []):
                person_faces[str(face.get("person_id"))].append(
                    {
                        **face,
                        "image_id": image_id,
                    }
                )

        person_ids = sorted(person_id for person_id in person_faces.keys() if person_id)
        best_candidate = None

        for left_index, left_person_id in enumerate(person_ids):
            for right_person_id in person_ids[left_index + 1 :]:
                candidate = decide_cluster_merge(
                    left_person_id,
                    person_faces[left_person_id],
                    right_person_id,
                    person_faces[right_person_id],
                    embedding_lookup=self.index_store.vector_at,
                    strong_threshold=self.strong_threshold,
                    high_quality_threshold=self.high_quality_threshold,
                )
                if candidate is None:
                    continue
                if best_candidate is None or (
                    candidate.best_similarity,
                    candidate.top_two_mean,
                    candidate.support_count,
                ) > (
                    best_candidate.best_similarity,
                    best_candidate.top_two_mean,
                    best_candidate.support_count,
                ):
                    best_candidate = candidate

        return best_candidate

    def _select_merge_target(self, left_person_id: str, right_person_id: str) -> tuple[str, str]:
        persons_state = self.state.get("persons", {})
        left_state = persons_state.get(left_person_id, {})
        right_state = persons_state.get(right_person_id, {})
        left_rank = (
            int(left_state.get("photo_count", 0)),
            int(left_state.get("face_count", 0)),
            -self._person_number(left_person_id),
        )
        right_rank = (
            int(right_state.get("photo_count", 0)),
            int(right_state.get("face_count", 0)),
            -self._person_number(right_person_id),
        )
        if left_rank >= right_rank:
            return left_person_id, right_person_id
        return right_person_id, left_person_id

    def _merge_person_cluster(self, source_person_id: str, target_person_id: str, merge_reason: str) -> None:
        if source_person_id == target_person_id:
            return

        for image in self.state.get("images", {}).values():
            for face in image.get("faces", []):
                if face.get("person_id") != source_person_id:
                    continue
                face["person_id"] = target_person_id
                face["cluster_merge_reason"] = merge_reason
                if face.get("match_decision") == "new_person_from_ambiguity":
                    face["match_decision"] = "cluster_merge_match"
                    face["match_reason"] = merge_reason

        for faiss_id, person_id in list(self.faiss_person_map.items()):
            if person_id == source_person_id:
                self.faiss_person_map[faiss_id] = target_person_id

        source_state = self.state.get("persons", {}).pop(source_person_id, None)
        target_state = self.state.get("persons", {}).get(target_person_id)
        if source_state and target_state and not target_state.get("representative_face_id"):
            target_state["representative_face_id"] = source_state.get("representative_face_id")

    def _rebuild_person_state_from_images(self) -> None:
        rebuilt: Dict[str, Dict] = {}

        for image in self.state.get("images", {}).values():
            image_id = image.get("photo_id")
            timestamp = image.get("timestamp")
            seen_in_photo = set()
            for face in image.get("faces", []):
                person_id = str(face.get("person_id") or "")
                if not person_id:
                    continue
                person_state = rebuilt.setdefault(
                    person_id,
                    {
                        "person_id": person_id,
                        "representative_face_id": face.get("face_id"),
                        "photo_count": 0,
                        "face_count": 0,
                        "first_seen": None,
                        "last_seen": None,
                        "avg_score": 0.0,
                        "avg_similarity": 0.0,
                        "avg_quality": 0.0,
                        "high_quality_face_count": 0,
                        "sample_photo_ids": [],
                        "pose_counts": {},
                        "label": "",
                        "_best_score": -1.0,
                    },
                )
                person_state["face_count"] += 1
                person_state["avg_score"] = self._rolling_average(
                    person_state["avg_score"],
                    person_state["face_count"],
                    float(face.get("score") or 0.0),
                )
                person_state["avg_similarity"] = self._rolling_average(
                    person_state["avg_similarity"],
                    person_state["face_count"],
                    float(face.get("similarity") or 0.0),
                )
                person_state["avg_quality"] = self._rolling_average(
                    person_state["avg_quality"],
                    person_state["face_count"],
                    float(face.get("quality_score") or 0.0),
                )
                if float(face.get("quality_score") or 0.0) >= self.high_quality_threshold:
                    person_state["high_quality_face_count"] += 1
                pose_bucket = str(face.get("pose_bucket") or "unknown")
                pose_counts = person_state.setdefault("pose_counts", {})
                pose_counts[pose_bucket] = int(pose_counts.get(pose_bucket, 0)) + 1
                if float(face.get("score") or 0.0) > float(person_state.get("_best_score", -1.0)):
                    person_state["_best_score"] = float(face.get("score") or 0.0)
                    person_state["representative_face_id"] = face.get("face_id")

                if person_id not in seen_in_photo:
                    seen_in_photo.add(person_id)
                    person_state["photo_count"] += 1
                    if timestamp:
                        person_state["first_seen"] = self._min_datetime(
                            person_state.get("first_seen"),
                            timestamp,
                        )
                        person_state["last_seen"] = self._max_datetime(
                            person_state.get("last_seen"),
                            timestamp,
                        )
                    if image_id and image_id not in person_state["sample_photo_ids"] and len(person_state["sample_photo_ids"]) < 10:
                        person_state["sample_photo_ids"].append(image_id)

        for person_state in rebuilt.values():
            person_state.pop("_best_score", None)

        self.state["persons"] = rebuilt

    def _refresh_photo_faces(self, photos: List[Photo]) -> None:
        for photo in photos:
            cached = self.photo_results.get(photo.photo_id)
            if cached:
                photo.faces = [dict(face) for face in cached.get("faces", [])]

    def _person_number(self, person_id: str) -> int:
        try:
            return int(str(person_id).split("_")[-1])
        except (TypeError, ValueError):
            return 10**9

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
