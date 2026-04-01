"""
MediaPipe Face Landmarker 支持。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, Optional, Sequence, Union
import warnings

import numpy as np
try:
    import requests
except ModuleNotFoundError:  # pragma: no cover - test env may not install requests
    requests = None

from config import (
    FACE_LANDMARK_MODEL_PATH,
    FACE_LANDMARK_MODEL_URL,
    FACE_LANDMARKS_ENABLED,
    FACE_POSE_PROFILE_YAW_THRESHOLD,
)

try:
    import mediapipe as mp  # type: ignore
    from mediapipe.tasks.python import BaseOptions  # type: ignore
    from mediapipe.tasks.python.vision import FaceLandmarker  # type: ignore
    from mediapipe.tasks.python.vision import FaceLandmarkerOptions  # type: ignore
    from mediapipe.tasks.python.vision import RunningMode  # type: ignore
except Exception:  # pragma: no cover - mediapipe is expected in runtime
    mp = None
    BaseOptions = None
    FaceLandmarker = None
    FaceLandmarkerOptions = None
    RunningMode = None


LEFT_EYE_INDEXES = (33, 133)
RIGHT_EYE_INDEXES = (362, 263)
NOSE_TIP_INDEX = 1


@dataclass(frozen=True)
class FacePose:
    pose_yaw: Optional[float]
    pose_pitch: Optional[float]
    pose_roll: Optional[float]
    pose_bucket: str
    left_eye_width: Optional[float]
    right_eye_width: Optional[float]
    eye_visibility_ratio: Optional[float]
    landmark_detected: bool
    landmark_source: Optional[str]

    def as_dict(self) -> Dict[str, object]:
        return {
            "pose_yaw": self.pose_yaw,
            "pose_pitch": self.pose_pitch,
            "pose_roll": self.pose_roll,
            "pose_bucket": self.pose_bucket,
            "left_eye_width": self.left_eye_width,
            "right_eye_width": self.right_eye_width,
            "eye_visibility_ratio": self.eye_visibility_ratio,
            "landmark_detected": self.landmark_detected,
            "landmark_source": self.landmark_source,
        }


class FaceLandmarkAnalyzer:
    def __init__(
        self,
        *,
        enabled: bool = FACE_LANDMARKS_ENABLED,
        model_path: str = FACE_LANDMARK_MODEL_PATH,
        model_url: str = FACE_LANDMARK_MODEL_URL,
    ) -> None:
        self.enabled = enabled
        self.model_path = Path(model_path)
        self.model_url = model_url
        self._lock = Lock()
        self._landmarker = None
        self._landmarker_init_failed = False
        self._landmarker_init_error: str | None = None

    def analyze_face(
        self,
        image_pixels: object,
        bbox_xywh: Dict[str, int],
        *,
        insight_pose: Sequence[float] | None = None,
        insight_landmark_2d_106: Sequence[Sequence[float]] | None = None,
        insight_landmark_3d_68: Sequence[Sequence[float]] | None = None,
    ) -> Dict[str, object]:
        fallback_pose = build_insightface_fallback_pose(
            insight_pose,
            insight_landmark_2d_106=insight_landmark_2d_106,
            insight_landmark_3d_68=insight_landmark_3d_68,
        )
        if not self.enabled or mp is None:
            return (fallback_pose or self._default_pose()).as_dict()

        pixels = np.asarray(image_pixels)
        crop = self._crop_face_region(pixels, bbox_xywh)
        if crop.size == 0:
            return (fallback_pose or self._default_pose()).as_dict()

        landmarker = self._get_landmarker()
        if landmarker is None:
            return (fallback_pose or self._default_pose()).as_dict()

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(crop.astype("uint8")),
        )
        try:
            result = landmarker.detect(mp_image)
        except Exception:
            return (fallback_pose or self._default_pose()).as_dict()

        if not getattr(result, "face_landmarks", None):
            return (fallback_pose or self._default_pose()).as_dict()

        landmarks = result.face_landmarks[0]
        rotation = None
        if getattr(result, "facial_transformation_matrixes", None):
            rotation = np.asarray(result.facial_transformation_matrixes[0], dtype="float32")

        yaw, pitch, roll = estimate_pose_from_landmarks(landmarks, rotation)
        left_eye_width = landmark_distance(landmarks, LEFT_EYE_INDEXES)
        right_eye_width = landmark_distance(landmarks, RIGHT_EYE_INDEXES)
        eye_visibility_ratio = compute_eye_visibility_ratio(left_eye_width, right_eye_width)

        pose = FacePose(
            pose_yaw=yaw,
            pose_pitch=pitch,
            pose_roll=roll,
            pose_bucket=classify_pose_bucket(yaw),
            left_eye_width=round(left_eye_width, 6),
            right_eye_width=round(right_eye_width, 6),
            eye_visibility_ratio=round(eye_visibility_ratio, 6),
            landmark_detected=True,
            landmark_source="mediapipe-face-landmarker",
        )
        return pose.as_dict()

    def _get_landmarker(self):
        if self._landmarker is not None:
            return self._landmarker
        if self._landmarker_init_failed:
            return None
        if FaceLandmarker is None or FaceLandmarkerOptions is None or BaseOptions is None:
            return None

        with self._lock:
            if self._landmarker is not None:
                return self._landmarker
            if self._landmarker_init_failed:
                return None
            ensure_model_downloaded(self.model_path, self.model_url)
            model_bytes = self.model_path.read_bytes()
            options = FaceLandmarkerOptions(
                # Prefer in-memory model bytes so MediaPipe does not rely on mmap.
                base_options=BaseOptions(model_asset_buffer=model_bytes),
                running_mode=RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.35,
                min_face_presence_confidence=0.35,
                min_tracking_confidence=0.35,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,
            )
            try:
                self._landmarker = FaceLandmarker.create_from_options(options)
            except Exception as exc:
                self._landmarker_init_failed = True
                self._landmarker_init_error = str(exc)
                warnings.warn(
                    f"MediaPipe FaceLandmarker 初始化失败，已降级为 InsightFace pose fallback: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return None
            return self._landmarker

    def _crop_face_region(self, image: np.ndarray, bbox_xywh: Dict[str, int]) -> np.ndarray:
        height, width = image.shape[:2]
        x = max(0, int(bbox_xywh.get("x", 0)))
        y = max(0, int(bbox_xywh.get("y", 0)))
        w = max(0, int(bbox_xywh.get("w", 0)))
        h = max(0, int(bbox_xywh.get("h", 0)))
        if w == 0 or h == 0:
            return np.zeros((0, 0, 3), dtype=image.dtype)

        pad_x = max(12, int(round(w * 0.35)))
        pad_y = max(12, int(round(h * 0.35)))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_y)
        return image[y1:y2, x1:x2]

    def _default_pose(self) -> FacePose:
        return FacePose(
            pose_yaw=None,
            pose_pitch=None,
            pose_roll=None,
            pose_bucket="unknown",
            left_eye_width=None,
            right_eye_width=None,
            eye_visibility_ratio=None,
            landmark_detected=False,
            landmark_source=None,
        )


def ensure_model_downloaded(model_path: Path, model_url: str) -> Path:
    if model_path.exists():
        return model_path

    if requests is None:
        raise RuntimeError("requests 未安装，无法下载 MediaPipe face landmark 模型")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(model_url, timeout=60)
    response.raise_for_status()
    tmp_path = model_path.with_suffix(model_path.suffix + ".tmp")
    tmp_path.write_bytes(response.content)
    tmp_path.replace(model_path)
    return model_path


def build_insightface_fallback_pose(
    insight_pose: Sequence[float] | None,
    *,
    insight_landmark_2d_106: Sequence[Sequence[float]] | None = None,
    insight_landmark_3d_68: Sequence[Sequence[float]] | None = None,
) -> FacePose | None:
    if insight_pose is None:
        return None

    pose = np.asarray(insight_pose, dtype="float32").reshape(-1)
    if pose.size < 3 or not np.all(np.isfinite(pose[:3])):
        return None

    pitch = float(pose[0])
    yaw = float(pose[1])
    roll = float(pose[2])

    landmark_detected = True
    return FacePose(
        pose_yaw=round(yaw, 6),
        pose_pitch=round(pitch, 6),
        pose_roll=round(roll, 6),
        pose_bucket=classify_pose_bucket(yaw),
        left_eye_width=None,
        right_eye_width=None,
        eye_visibility_ratio=None,
        landmark_detected=landmark_detected,
        landmark_source="insightface-pose-fallback",
    )


def classify_pose_bucket(yaw: float | None, threshold: float = FACE_POSE_PROFILE_YAW_THRESHOLD) -> str:
    if yaw is None:
        return "unknown"
    if abs(yaw) < threshold:
        return "frontal"
    return "left_profile" if yaw < 0 else "right_profile"


def landmark_distance(landmarks: Sequence[object], indexes: Iterable[int]) -> float:
    indexes_list = list(indexes)
    if len(indexes_list) < 2:
        return 0.0
    first_index, second_index = indexes_list[0], indexes_list[1]
    if max(first_index, second_index) >= len(landmarks):
        return 0.0
    first = landmarks[first_index]
    second = landmarks[second_index]
    return math.hypot(float(first.x) - float(second.x), float(first.y) - float(second.y))


def compute_eye_visibility_ratio(left_eye_width: float, right_eye_width: float) -> float:
    eye_max = max(left_eye_width, right_eye_width, 1e-6)
    eye_min = min(left_eye_width, right_eye_width)
    return float(eye_min / eye_max)


def estimate_pose_from_landmarks(
    landmarks: Sequence[object],
    transformation_matrix: np.ndarray | None = None,
) -> tuple[float, float, float]:
    if transformation_matrix is not None and transformation_matrix.shape[0] >= 3 and transformation_matrix.shape[1] >= 3:
        rotation = transformation_matrix[:3, :3]
        yaw, pitch, roll = euler_from_rotation_matrix(rotation)
        if all(math.isfinite(value) for value in (yaw, pitch, roll)):
            return yaw, pitch, roll

    if len(landmarks) <= max(max(LEFT_EYE_INDEXES), max(RIGHT_EYE_INDEXES), NOSE_TIP_INDEX):
        return 0.0, 0.0, 0.0

    left_eye = midpoint(landmarks[LEFT_EYE_INDEXES[0]], landmarks[LEFT_EYE_INDEXES[1]])
    right_eye = midpoint(landmarks[RIGHT_EYE_INDEXES[0]], landmarks[RIGHT_EYE_INDEXES[1]])
    nose = landmarks[NOSE_TIP_INDEX]

    interocular = max(abs(right_eye[0] - left_eye[0]), 1e-6)
    center_x = (left_eye[0] + right_eye[0]) / 2.0
    yaw = ((float(nose.x) - center_x) / interocular) * 45.0
    roll = math.degrees(math.atan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
    return yaw, 0.0, roll


def euler_from_rotation_matrix(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(float(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2))
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(float(rotation_matrix[2, 1]), float(rotation_matrix[2, 2]))
        yaw = math.atan2(float(-rotation_matrix[2, 0]), sy)
        roll = math.atan2(float(rotation_matrix[1, 0]), float(rotation_matrix[0, 0]))
    else:
        pitch = math.atan2(float(-rotation_matrix[1, 2]), float(rotation_matrix[1, 1]))
        yaw = math.atan2(float(-rotation_matrix[2, 0]), sy)
        roll = 0.0

    return (
        math.degrees(yaw),
        math.degrees(pitch),
        math.degrees(roll),
    )


def midpoint(first: object, second: object) -> tuple[float, float]:
    try:
        return (
            (float(first.x) + float(second.x)) / 2.0,
            (float(first.y) + float(second.y)) / 2.0,
        )
    except (AttributeError, TypeError, ValueError):
        return (0.0, 0.0)
