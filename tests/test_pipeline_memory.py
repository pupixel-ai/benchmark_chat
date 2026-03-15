from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models import Event, Person, Photo, Relationship


class FakeAssetStore:
    def asset_url(self, task_id: str, relative_path: str) -> str:
        return f"/api/assets/{task_id}/{relative_path}"

    def sync_task_directory(self, task_id: str, task_dir: str | Path) -> None:
        return None


class FakeFaceReviewStore:
    def is_image_abandoned(self, user_id: str | None, source_hash: str | None) -> bool:
        return False


class FakeImageProcessor:
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.face_dir = self.cache_dir / "face_crops"
        self.face_dir.mkdir(parents=True, exist_ok=True)
        self.boxed_dir = self.cache_dir / "boxed_images"
        self.boxed_dir.mkdir(parents=True, exist_ok=True)

    def load_photos_with_errors(self, photo_dir: str, max_photos: int | None = None):
        uploads = Path(photo_dir)
        photos = []
        for index, file_path in enumerate(sorted(uploads.iterdir()), start=1):
            photos.append(
                Photo(
                    photo_id=f"photo_{index:03d}",
                    filename=file_path.name,
                    path=str(file_path),
                    timestamp=self._timestamp_for(index),
                    location=self._location_for(index),
                    source_hash=f"hash-photo-{index}",
                    original_path=str(file_path),
                )
            )
        return photos[: max_photos or len(photos)], []

    def convert_to_jpeg(self, photos):
        return photos

    def preprocess(self, photos):
        for photo in photos:
            compressed_path = self.cache_dir / f"compressed_{photo.photo_id}.webp"
            compressed_path.write_bytes(b"compressed")
            photo.compressed_path = str(compressed_path)
        return photos

    def draw_face_boxes(self, photo):
        boxed_path = self.boxed_dir / f"boxed_{photo.photo_id}.webp"
        boxed_path.write_bytes(b"boxed")
        return str(boxed_path)

    def save_face_crop(self, photo, face):
        crop_path = self.face_dir / f"{face['face_id']}.webp"
        crop_path.write_bytes(b"face")
        return str(crop_path)

    def _timestamp_for(self, index: int):
        from datetime import datetime

        if index == 1:
            return datetime(2026, 3, 15, 9, 0)
        if index == 2:
            return datetime(2026, 3, 15, 9, 5)
        return datetime(2026, 3, 15, 14, 0)

    def _location_for(self, index: int):
        if index < 3:
            return {"name": "Home", "lat": 31.23, "lng": 121.47}
        return {"name": "Cafe", "lat": 31.22, "lng": 121.48}


class FakeFaceRecognition:
    def __init__(self, **kwargs):
        self.output_path = kwargs["output_path"]
        self._images = []
        self._persons = {
            "Person_001": Person(person_id="Person_001", name="Primary", photo_count=3),
            "Person_002": Person(person_id="Person_002", name="Friend", photo_count=2),
        }

    def process_photo(self, photo: Photo):
        faces = [
            {
                "face_id": f"face-{photo.photo_id}-1",
                "person_id": "Person_001",
                "score": 0.99,
                "similarity": 0.88,
                "faiss_id": 1,
                "bbox": [10, 20, 60, 70],
                "bbox_xywh": {"x": 10, "y": 20, "w": 50, "h": 50},
                "quality_score": 0.92,
                "quality_flags": ["clear_face"],
                "match_decision": "strong_match",
                "match_reason": "stub",
                "pose_yaw": 0.0,
                "pose_pitch": 0.0,
                "pose_roll": 0.0,
                "pose_bucket": "frontal",
                "eye_visibility_ratio": 1.0,
                "landmark_detected": True,
                "landmark_source": "stub",
            }
        ]
        if photo.photo_id != "photo_003":
            faces.append(
                {
                    "face_id": f"face-{photo.photo_id}-2",
                    "person_id": "Person_002",
                    "score": 0.97,
                    "similarity": 0.82,
                    "faiss_id": 2,
                    "bbox": [90, 22, 140, 72],
                    "bbox_xywh": {"x": 90, "y": 22, "w": 50, "h": 50},
                    "quality_score": 0.87,
                    "quality_flags": ["clear_face"],
                    "match_decision": "strong_match",
                    "match_reason": "stub",
                    "pose_yaw": 0.0,
                    "pose_pitch": 0.0,
                    "pose_roll": 0.0,
                    "pose_bucket": "frontal",
                    "eye_visibility_ratio": 1.0,
                    "landmark_detected": True,
                    "landmark_source": "stub",
                }
            )

        photo.faces = faces
        self._images.append(
            {
                "image_hash": f"img-{photo.photo_id}",
                "photo_id": photo.photo_id,
                "filename": photo.filename,
                "path": photo.path,
                "source_hash": photo.source_hash,
                "timestamp": photo.timestamp.isoformat(),
                "location": photo.location,
                "width": 100,
                "height": 100,
                "faces": faces,
                "detection_seconds": 0.1,
                "embedding_seconds": 0.05,
            }
        )
        return faces

    def reorder_protagonist(self, photos):
        return {"cluster_merges": []}

    def get_primary_person_id(self):
        return "Person_001"

    def save(self):
        return None

    def get_all_persons(self):
        return self._persons

    def get_face_output(self):
        return {
            "primary_person_id": "Person_001",
            "metrics": {
                "total_images": len(self._images),
                "total_faces": sum(len(image["faces"]) for image in self._images),
                "total_persons": len(self._persons),
            },
            "images": list(self._images),
            "persons": [
                {
                    "person_id": person.person_id,
                    "photo_count": person.photo_count,
                    "face_count": person.photo_count,
                    "avg_score": 0.98,
                    "avg_quality": 0.9,
                    "high_quality_face_count": person.photo_count,
                }
                for person in self._persons.values()
            ],
            "engine": {"model_name": "stub", "providers": ["stub"]},
        }


class FakeVLMAnalyzer:
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        self.results = []

    def load_cache(self):
        return False

    def get_result(self, photo_id: str):
        return None

    def analyze_photo(self, photo, face_db, primary_person_id):
        return {
            "summary": f"summary for {photo.photo_id}",
            "people": [
                {"person_id": "Person_001", "appearance": "casual", "clothing": "hoodie", "interaction": "present"}
            ],
            "scene": {"location_detected": photo.location.get("name"), "environment_description": "stub scene"},
            "event": {"activity": "meal" if photo.photo_id != "photo_003" else "meeting", "social_context": "friend", "interaction": "talking", "mood": "warm"},
            "details": ["cup"],
            "key_objects": ["table"],
        }

    def add_result(self, photo, result, persist: bool = False):
        photo.vlm_analysis = result
        self.results.append(
            {
                "photo_id": photo.photo_id,
                "filename": photo.filename,
                "timestamp": photo.timestamp.isoformat(),
                "location": photo.location,
                "face_person_ids": [face["person_id"] for face in photo.faces],
                "vlm_analysis": result,
            }
        )


class FakeLLMProcessor:
    def extract_events(self, vlm_results, primary_person_id):
        return [
            Event(
                event_id="EVT_001",
                date="2026-03-15",
                time_range="09:00 - 09:30",
                duration="0.5小时",
                title="Breakfast Session",
                type="用餐",
                participants=["Person_001", "Person_002"],
                location="Home",
                description="Breakfast at home",
                photo_count=2,
                confidence=0.8,
                reason="stub",
                narrative_synthesis="Breakfast at home with a friend.",
                tags=["breakfast", "home"],
                persona_evidence={"behavioral": ["hosts breakfast"], "aesthetic": ["simple table"], "socioeconomic": ["stable home life"]},
                evidence_photos=["photo_001", "photo_002"],
            )
        ]

    def infer_relationships(self, vlm_results, face_db, primary_person_id):
        return [
            Relationship(
                person_id="Person_002",
                relationship_type="friend",
                label="朋友",
                confidence=0.75,
                evidence={
                    "photo_count": 2,
                    "time_span": "1天",
                    "scenes": ["Home"],
                    "interaction_behavior": ["talking"],
                    "sample_scenes": [{"timestamp": "2026-03-15T09:00:00", "scene": "Home", "summary": "Breakfast", "activity": "meal"}],
                },
                reason="stub",
            )
        ]

    def generate_profile(self, events, relationships, primary_person_id):
        return "# Profile\n\n- enjoys hosting friends at home"


class PipelineMemoryTests(unittest.TestCase):
    def test_task_pipeline_runs_to_memory_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir)
            uploads_dir = task_dir / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            for name in ("a.jpg", "b.jpg", "c.jpg"):
                (uploads_dir / name).write_bytes(b"stub")

            with patch("services.pipeline_service.ImageProcessor", FakeImageProcessor), patch(
                "services.pipeline_service.FaceRecognition", FakeFaceRecognition
            ), patch("services.pipeline_service.VLMAnalyzer", FakeVLMAnalyzer), patch(
                "services.pipeline_service.LLMProcessor", FakeLLMProcessor
            ):
                from services.pipeline_service import MemoryPipelineService

                service = MemoryPipelineService(
                    task_id="task_pipeline_memory",
                    task_dir=str(task_dir),
                    asset_store=FakeAssetStore(),
                    user_id="user_pipeline",
                    face_review_store=FakeFaceReviewStore(),
                )
                result = service.run(max_photos=3, use_cache=False)

            self.assertEqual(result["summary"]["vlm_processed_images"], 3)
            self.assertEqual(result["summary"]["event_count"], 1)
            self.assertEqual(result["summary"]["relationship_count"], 1)
            self.assertIn("memory", result)
            self.assertEqual(result["memory"]["summary"]["session_count"], 2)
            self.assertGreater(result["memory"]["summary"]["profile_field_count"], 0)
            self.assertTrue(result["profile_markdown"].startswith("# Profile"))
            self.assertTrue((task_dir / "output" / "result.json").exists())
            self.assertTrue((task_dir / "output" / "memory" / "memory_envelope.json").exists())
            self.assertTrue((task_dir / "output" / "memory" / "memory_storage.json").exists())


if __name__ == "__main__":
    unittest.main()
