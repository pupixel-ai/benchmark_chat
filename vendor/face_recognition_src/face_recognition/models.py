from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

Embedding = Sequence[float]
Keypoints = Sequence[Sequence[float]]
PoseVector = Sequence[float]
Landmark2D = Sequence[Sequence[float]]
Landmark3D = Sequence[Sequence[float]]


@dataclass(frozen=True)
class DetectedFace:
    bbox: Sequence[float]
    score: float
    embedding: Embedding
    kps: Optional[Keypoints] = None
    insight_pose: Optional[PoseVector] = None
    insight_landmark_2d_106: Optional[Landmark2D] = None
    insight_landmark_3d_68: Optional[Landmark3D] = None


@dataclass(frozen=True)
class EngineResult:
    faces: List[DetectedFace]
    detection_seconds: float = 0.0
    embedding_seconds: float = 0.0


@dataclass(frozen=True)
class LoadedImage:
    pixels: object
    width: int
    height: int


@dataclass(frozen=True)
class ImageRecord:
    image_hash: str
    file_path: str
    processed_at: str
    width: int
    height: int


@dataclass(frozen=True)
class PersonRecord:
    person_id: str
    created_at: str
    representative_face: str


@dataclass(frozen=True)
class FaceRecord:
    face_id: str
    image_hash: str
    person_id: str
    bbox_json: str
    faiss_id: int
    embedding: Embedding


@dataclass(frozen=True)
class FailedImageRecord:
    run_id: str
    file_path: str
    stage: str
    error_code: str
    error_message: str
    failed_at: str
    retryable: bool
    retry_count: int = 0
    image_hash: Optional[str] = None
    file_size: Optional[int] = None
    file_mtime_ns: Optional[int] = None


@dataclass
class PendingBatch:
    images: List[ImageRecord] = field(default_factory=list)
    persons: List[PersonRecord] = field(default_factory=list)
    faces: List[FaceRecord] = field(default_factory=list)
    faiss_person_map: Dict[int, str] = field(default_factory=dict)
    image_hashes: set[str] = field(default_factory=set)

    def clear(self) -> None:
        self.images.clear()
        self.persons.clear()
        self.faces.clear()
        self.faiss_person_map.clear()
        self.image_hashes.clear()

    @property
    def image_count(self) -> int:
        return len(self.images)
