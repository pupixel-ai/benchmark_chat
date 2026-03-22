from __future__ import annotations

import tempfile
import time
import unittest
import json
import re
from pathlib import Path
from unittest.mock import patch

import services.pipeline_service
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

    def dedupe_before_face_recognition(self, photos):
        self.last_dedupe_report = {
            "total_images": len(photos),
            "retained_images": len(photos),
            "exact_duplicates_removed": 0,
            "near_duplicates_removed": 0,
            "burst_group_count": len(photos),
            "burst_groups": [
                {"representative_photo_id": photo.photo_id, "photo_ids": [photo.photo_id]}
                for photo in photos
            ],
            "representative_photo_ids": [photo.photo_id for photo in photos],
            "duplicate_backrefs": {},
        }
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
        from datetime import datetime, timezone

        if index == 1:
            return datetime(2026, 3, 15, 9, 0)
        if index == 2:
            return datetime(2026, 3, 15, 9, 5)
        return datetime(2026, 3, 15, 14, 0)

    def _location_for(self, index: int):
        if index < 3:
            return {"name": "Home", "lat": 31.23, "lng": 121.47}
        return {"name": "Cafe", "lat": 31.22, "lng": 121.48}


class UniqueIdFakeImageProcessor(FakeImageProcessor):
    def load_photos_with_errors(self, photo_dir: str, max_photos: int | None = None):
        uploads = Path(photo_dir)
        photos = []
        for index, file_path in enumerate(sorted(uploads.iterdir()), start=1):
            photos.append(
                Photo(
                    photo_id=file_path.stem,
                    filename=file_path.name,
                    path=str(file_path),
                    timestamp=self._timestamp_for(index),
                    location=self._location_for(index),
                    source_hash=f"hash-{file_path.stem}",
                    original_path=str(file_path),
                )
            )
        return photos[: max_photos or len(photos)], []


class ScaleReplayFakeImageProcessor(FakeImageProcessor):
    def load_photos_with_errors(self, photo_dir: str, max_photos: int | None = None):
        from datetime import datetime, timedelta

        uploads = Path(photo_dir)
        photos = []
        for file_path in sorted(uploads.iterdir()):
            stem = file_path.stem.lower()
            if stem.startswith("camera_event"):
                _, event_idx_text, photo_idx_text = stem.split("_")
                event_idx = int(event_idx_text.replace("event", ""))
                photo_idx = int(photo_idx_text.replace("p", ""))
                base_time = datetime(2026, 1, 1, 8, 0) + timedelta(minutes=40 * event_idx)
                timestamp = base_time + timedelta(minutes=2 * photo_idx)
                location = {"name": f"Place_{event_idx:03d}", "lat": 31.0 + event_idx / 1000.0, "lng": 121.0 + event_idx / 1000.0}
            else:
                ref_idx = int(re.sub(r"\D+", "", stem) or "0")
                timestamp = datetime(2026, 2, 1, 9, 0) + timedelta(minutes=ref_idx)
                location = {"name": "ReferenceLibrary", "lat": 31.23, "lng": 121.47}
            photos.append(
                Photo(
                    photo_id=file_path.stem,
                    filename=file_path.name,
                    path=str(file_path),
                    timestamp=timestamp,
                    location=location,
                    source_hash=f"hash-{file_path.stem}",
                    original_path=str(file_path),
                )
            )
        return photos[: max_photos or len(photos)], []


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
    def __init__(self, cache_path: str, task_version: str = ""):
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

    def analyze_photo_with_metadata(self, photo, face_db, primary_person_id):
        result = self.analyze_photo(photo, face_db, primary_person_id)
        return result, {"retry_count": 0, "runtime_seconds": 0.01}

    def build_result_entry(self, photo, result):
        return {
            "photo_id": photo.photo_id,
            "filename": photo.filename,
            "timestamp": photo.timestamp.isoformat(),
            "location": photo.location,
            "face_person_ids": [face["person_id"] for face in photo.faces],
            "vlm_analysis": result,
        }

    def replace_results(self, ordered_results):
        self.results = list(ordered_results)

    def save_cache(self):
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.cache_path).write_text(
            json.dumps(
                {
                    "metadata": {"schema_version": 3, "face_id_scheme": "Person_###"},
                    "photos": self.results,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def add_result(self, photo, result, persist: bool = False):
        photo.vlm_analysis = result
        self.results.append(self.build_result_entry(photo, result))


class FakeV03212VLMAnalyzer(FakeVLMAnalyzer):
    def analyze_photo(self, photo, face_db, primary_person_id):
        filename = (photo.filename or "").lower()
        if "reference" in filename or "ai" in filename:
            return {
                "summary": "fashion outfit style inspiration board",
                "people": [],
                "scene": {"location_detected": "", "environment_description": "reference media"},
                "event": {"activity": "", "social_context": "", "interaction": "", "mood": ""},
                "details": ["outfit reference"],
                "key_objects": ["lookbook"],
                "uncertainty": ["reference-only"],
            }
        return {
            "summary": f"home meal and coffee for {photo.photo_id}",
            "people": [
                {"person_id": "Person_001", "appearance": "casual", "clothing": "hoodie", "interaction": "present"}
            ],
            "scene": {"location_detected": photo.location.get("name"), "environment_description": "home scene"},
            "event": {"activity": "meal", "social_context": "friend", "interaction": "talking", "mood": "warm"},
            "details": ["cup", "breakfast"],
            "key_objects": ["table", "coffee"],
            "brands": ["local cafe"] if photo.photo_id == "photo_002" else [],
        }


class FakeV03213VLMAnalyzer(FakeV03212VLMAnalyzer):
    pass


class ScaleReplayFakeVLMAnalyzer(FakeVLMAnalyzer):
    def analyze_photo(self, photo, face_db, primary_person_id):
        filename = (photo.filename or "").lower()
        if filename.startswith("ref_") or "reference" in filename or "ai_" in filename:
            return {
                "summary": "fashion outfit style inspiration board",
                "people": [],
                "scene": {"location_detected": "", "environment_description": "reference media"},
                "event": {"activity": "", "social_context": "", "interaction": "", "mood": ""},
                "details": ["outfit reference", "editorial style"],
                "key_objects": ["lookbook", "mood board"],
                "uncertainty": ["reference-only"],
            }
        return {
            "summary": f"meal and coffee meetup for {photo.photo_id}",
            "people": [
                {"person_id": "Person_001", "appearance": "casual", "clothing": "hoodie", "interaction": "present"},
                {"person_id": "Person_002", "appearance": "casual", "clothing": "jacket", "interaction": "present"},
            ],
            "scene": {"location_detected": photo.location.get("name"), "environment_description": "cafe scene"},
            "event": {"activity": "meal", "social_context": "friend", "interaction": "talking", "mood": "warm"},
            "details": ["cup", "breakfast", "menu"],
            "key_objects": ["table", "coffee"],
            "brands": [f"brand-{photo.location.get('name')}"],
            "place_candidates": [photo.location.get("name")],
        }


class SlowOutOfOrderFakeVLMAnalyzer(FakeVLMAnalyzer):
    def analyze_photo_with_metadata(self, photo, face_db, primary_person_id):
        delays = {
            "photo_001": 0.03,
            "photo_002": 0.01,
            "photo_003": 0.02,
        }
        time.sleep(delays.get(photo.photo_id, 0.0))
        result = self.analyze_photo(photo, face_db, primary_person_id)
        return result, {"retry_count": 0, "runtime_seconds": delays.get(photo.photo_id, 0.0) + 0.01}


class FakeLLMProcessor:
    def __init__(self, task_version: str = ""):
        self.task_version = task_version

    def _call_json_prompt(self, prompt: str):
        return None

    def _call_markdown_prompt(self, prompt: str):
        return "# Profile\n\n- stub profile from profile input pack"

    def extract_memory_contract(self, vlm_results, face_db, primary_person_id, progress_callback=None):
        self.last_chunk_artifacts = {
            "photo_fact_count": len(vlm_results),
            "raw_event_count": 2,
            "slice_count": 2,
            "slices": [
                {"slice_id": "slice_0001", "photo_count": 2, "rare_clue_count": 1},
                {"slice_id": "slice_0002", "photo_count": 1, "rare_clue_count": 0},
            ],
            "slice_contract_records": [
                {"slice_id": "slice_0001", "raw_event_id": "raw_event_001", "photo_ids": ["photo_001"], "contract": {"facts": []}},
                {"slice_id": "slice_0002", "raw_event_id": "raw_event_002", "photo_ids": ["photo_002"], "contract": {"facts": []}},
            ],
            "event_merge_records": [
                {"raw_event_id": "raw_event_001", "photo_ids": ["photo_001", "photo_002"], "contract": {"facts": []}},
            ],
            "pre_relationship_contract": {
                "facts": [],
                "observations": [],
                "claims": [],
                "relationship_hypotheses": [],
                "profile_deltas": [],
                "uncertainty": [],
            },
        }
        if progress_callback:
            progress_callback({"processed_slices": 2, "slice_count": 2, "processed_events": 2, "event_count": 2, "percent": 100})
        return {
            "facts": [
                {
                    "fact_id": "EVT_001",
                    "title": "Breakfast Session",
                    "coarse_event_type": "用餐",
                    "event_facets": ["breakfast", "home"],
                    "alternative_type_candidates": [],
                    "started_at": "2026-03-15T09:00:00",
                    "ended_at": "2026-03-15T09:30:00",
                    "location": "Home",
                    "participant_person_ids": ["Person_001", "Person_002"],
                    "photo_ids": ["photo_001", "photo_002"],
                    "event_id": "session_001",
                    "description": "Breakfast at home",
                    "narrative_synthesis": "Breakfast at home with a friend.",
                    "confidence": 0.8,
                    "reason": "stub",
                    "persona_evidence": {"behavioral": ["hosts breakfast"], "aesthetic": ["simple table"], "socioeconomic": ["stable home life"]},
                }
            ],
            "observations": [
                {
                    "observation_id": "obs_001",
                    "category": "scene",
                    "field_key": "meal_context",
                    "field_value": "home breakfast",
                    "confidence": 0.8,
                    "photo_ids": ["photo_001", "photo_002"],
                    "fact_id": "EVT_001",
                    "event_id": "session_001",
                    "person_ids": ["Person_001", "Person_002"],
                    "evidence_refs": [{"ref_type": "photo", "ref_id": "photo_001"}],
                }
            ],
            "claims": [
                {
                    "claim_id": "claim_001",
                    "claim_type": "location",
                    "subject": "EVT_001",
                    "predicate": "happened_at",
                    "object": "Home",
                    "confidence": 0.8,
                    "photo_ids": ["photo_001", "photo_002"],
                    "fact_id": "EVT_001",
                    "event_id": "session_001",
                    "evidence_refs": [{"ref_type": "photo", "ref_id": "photo_001"}],
                }
            ],
            "relationship_hypotheses": [
                {
                    "person_id": "Person_002",
                    "relationship_type": "friend",
                    "label": "朋友",
                    "confidence": 0.75,
                    "reason_summary": "stub",
                    "reason": "stub",
                    "evidence": {
                        "photo_count": 2,
                        "time_span": "1天",
                        "scenes": ["Home"],
                        "interaction_behavior": ["talking"],
                        "sample_scenes": [{"timestamp": "2026-03-15T09:00:00", "scene": "Home", "summary": "Breakfast", "activity": "meal"}],
                    },
                }
            ],
            "profile_deltas": [
                {
                    "profile_key": "preference_profile",
                    "field_key": "hosting",
                    "field_value": "hosts breakfast at home",
                    "summary": "enjoys hosting friends at home",
                    "confidence": 0.72,
                    "supporting_event_ids": ["EVT_001"],
                    "supporting_photo_ids": ["photo_001", "photo_002"],
                    "evidence_refs": [{"ref_type": "event", "ref_id": "EVT_001"}],
                }
            ],
            "uncertainty": [],
        }

    def facts_from_memory_contract(self, memory_contract):
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

    def relationships_from_memory_contract(self, memory_contract):
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

    def profile_markdown_from_memory_contract(self, memory_contract, primary_person_id):
        return "# Profile\n\n- enjoys hosting friends at home"

    def generate_profile(self, facts, relationships, primary_person_id):
        return self.profile_markdown_from_memory_contract({}, primary_person_id)


class CaptureMarkdownLLMProcessor(FakeLLMProcessor):
    last_markdown_prompt = ""

    def _call_json_prompt(self, prompt: str):
        return None

    def _call_markdown_prompt(self, prompt: str):
        type(self).last_markdown_prompt = prompt
        return "# Profile\n\n- stub profile"


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
                    task_version="v0317",
                )
                result = service.run(max_photos=3, use_cache=False)

            self.assertEqual(result["summary"]["vlm_processed_images"], 3)
            self.assertEqual(result["summary"]["event_count"], 2)
            self.assertEqual(result["summary"]["relationship_count"], 1)
            self.assertEqual(result["summary"]["observation_count"], 1)
            self.assertEqual(result["summary"]["claim_count"], 1)
            self.assertEqual(result["summary"]["profile_delta_count"], 1)
            self.assertIn("memory", result)
            self.assertEqual(result["memory"]["summary"]["event_count"], 2)
            self.assertEqual(result["memory"]["summary"]["fact_count"], 1)
            self.assertGreater(result["memory"]["summary"]["profile_field_count"], 0)
            self.assertTrue(result["profile_markdown"].startswith("# Profile"))
            self.assertEqual(result["dedupe_report"]["retained_images"], 3)
            self.assertEqual(result["llm_chunk_artifacts"]["slice_count"], 2)
            self.assertTrue((task_dir / "output" / "result.json").exists())
            self.assertTrue((task_dir / "output" / "memory" / "memory_envelope.json").exists())
            self.assertTrue((task_dir / "output" / "memory" / "memory_storage.json").exists())
            self.assertTrue((task_dir / "output" / "memory_contract.json").exists())
            self.assertTrue((task_dir / "output" / "llm_chunks.json").exists())
            self.assertTrue((task_dir / "output" / "slice_contracts.jsonl").exists())
            self.assertTrue((task_dir / "output" / "event_merges.jsonl").exists())
            self.assertTrue((task_dir / "output" / "pre_relationship_contract.json").exists())

    def test_vlm_parallel_execution_preserves_result_and_cache_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir)
            uploads_dir = task_dir / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            for name in ("a.jpg", "b.jpg", "c.jpg"):
                (uploads_dir / name).write_bytes(b"stub")

            progress_events = []

            with patch("services.pipeline_service.ImageProcessor", FakeImageProcessor), patch(
                "services.pipeline_service.FaceRecognition", FakeFaceRecognition
            ), patch("services.pipeline_service.VLMAnalyzer", SlowOutOfOrderFakeVLMAnalyzer), patch(
                "services.pipeline_service.LLMProcessor", FakeLLMProcessor
            ), patch("services.pipeline_service.VLM_MAX_CONCURRENCY", 3), patch(
                "services.pipeline_service.VLM_CACHE_FLUSH_EVERY_N", 1
            ), patch(
                "services.pipeline_service.VLM_CACHE_FLUSH_INTERVAL_SECONDS", 1
            ):
                from services.pipeline_service import MemoryPipelineService

                service = MemoryPipelineService(
                    task_id="task_pipeline_vlm_order",
                    task_dir=str(task_dir),
                    asset_store=FakeAssetStore(),
                    user_id="user_pipeline",
                    face_review_store=FakeFaceReviewStore(),
                    task_version="v0317",
                )
                result = service.run(
                    max_photos=3,
                    use_cache=False,
                    progress_callback=lambda stage, payload: progress_events.append((stage, payload)),
                )

            result_photo_ids = [item["photo_id"] for item in result["memory"]["transparency"]["vlm_stage"]["summaries"]]
            self.assertEqual(result_photo_ids[:3], ["photo_001", "photo_002", "photo_003"])

            cache_payload = json.loads((task_dir / "cache" / "vlm_cache.json").read_text(encoding="utf-8"))
            cache_photo_ids = [item["photo_id"] for item in cache_payload["photos"]]
            self.assertEqual(cache_photo_ids, ["photo_001", "photo_002", "photo_003"])

            final_vlm_payload = [payload for stage, payload in progress_events if stage == "vlm"][-1]
            self.assertTrue(final_vlm_payload["order_invariant_verified"])
            self.assertEqual(final_vlm_payload["concurrency"], 3)
            self.assertGreaterEqual(final_vlm_payload["flush_count"], 1)
            self.assertIn("queued", final_vlm_payload)
            self.assertIn("in_flight", final_vlm_payload)
            self.assertIn("avg_latency_seconds", final_vlm_payload)

    def test_pre_v0317_pipeline_stops_after_face_recognition(self) -> None:
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
                    task_id="task_pipeline_face_only",
                    task_dir=str(task_dir),
                    asset_store=FakeAssetStore(),
                    user_id="user_pipeline",
                    face_review_store=FakeFaceReviewStore(),
                    task_version="v0315",
                )
                result = service.run(max_photos=3, use_cache=False)

            self.assertEqual(result["summary"]["vlm_processed_images"], 0)
            self.assertEqual(result["summary"]["event_count"], 0)
            self.assertEqual(result["summary"]["relationship_count"], 0)
            self.assertIsNone(result["memory"])
            self.assertEqual(result["facts"], [])
            self.assertEqual(result["relationships"], [])
            self.assertEqual(result["profile_markdown"], "")
            self.assertTrue(any(item["stage"] == "version_gate" for item in result["warnings"]))
            self.assertTrue((task_dir / "output" / "result.json").exists())
            self.assertFalse((task_dir / "output" / "memory").exists())

    def test_v0321_2_pipeline_uses_independent_revision_family(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir)
            uploads_dir = task_dir / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            for name in ("camera_a.jpg", "camera_b.jpg", "saved_ai_reference.png"):
                (uploads_dir / name).write_bytes(b"stub")

            with patch("services.pipeline_service.ImageProcessor", FakeImageProcessor), patch(
                "services.pipeline_service.FaceRecognition", FakeFaceRecognition
            ), patch("services.pipeline_service.VLMAnalyzer", FakeV03212VLMAnalyzer), patch(
                "services.pipeline_service.LLMProcessor", FakeLLMProcessor
            ):
                from services.pipeline_service import MemoryPipelineService

                progress_events = []
                service = MemoryPipelineService(
                    task_id="task_pipeline_v0321_2",
                    task_dir=str(task_dir),
                    asset_store=FakeAssetStore(),
                    user_id="user_pipeline",
                    face_review_store=FakeFaceReviewStore(),
                    task_version="v0321.2",
                )
                result = service.run(
                    max_photos=3,
                    use_cache=False,
                    progress_callback=lambda stage, payload: progress_events.append((stage, dict(payload))),
                )

            self.assertEqual(result["version"], "v0321.2")
            self.assertEqual(result["memory"]["pipeline_family"], "v0321_2")
            self.assertEqual(result["summary"]["event_count"], 1)
            self.assertEqual(result["summary"]["relationship_count"], 1)
            self.assertEqual(result["memory"]["summary"]["reference_media_signal_count"], 1)
            self.assertEqual(len(result["memory"]["event_revisions"]), 1)
            self.assertEqual(len(result["memory"]["relationship_revisions"]), 1)
            event_photo_ids = result["memory"]["event_revisions"][0]["original_photo_ids"]
            self.assertEqual(event_photo_ids, ["hash-photo-1", "hash-photo-2"])
            self.assertIn("aesthetic_preference", result["memory"]["profile_revision"]["buckets"])
            self.assertTrue(result["profile_markdown"].startswith("# Profile"))
            self.assertGreater(len(result["memory"]["transparency"]["vlm_stage"]["summaries"]), 0)
            self.assertEqual(result["memory"]["transparency"]["llm_stage"]["fact_count"], 1)
            self.assertEqual(
                result["memory"]["storage"]["redis"]["profile_current"]["profile_revision_id"],
                result["memory"]["profile_revision"]["profile_revision_id"],
            )
            llm_events = [payload for stage, payload in progress_events if stage == "llm"]
            self.assertTrue(llm_events)
            event_draft_events = [payload for payload in llm_events if payload.get("substage") == "event_draft"]
            self.assertTrue(event_draft_events)
            self.assertEqual(event_draft_events[0]["processed_candidates"], 0)
            self.assertGreater(event_draft_events[-1]["processed_candidates"], 0)
            self.assertEqual(
                event_draft_events[-1]["processed_candidates"],
                event_draft_events[-1]["candidate_count"],
            )
            self.assertEqual(llm_events[-1]["substage"], "completed")
            self.assertIn("event_revisions", llm_events[-1]["memory_contract_preview"])
            self.assertTrue(llm_events[-1]["profile_markdown_preview"].startswith("# Profile"))
            self.assertTrue((task_dir / "v0321_2" / "state.db").exists())
            self.assertTrue((task_dir / "v0321_2" / "reference_media.json").exists())
            self.assertTrue((task_dir / "v0321_2" / "external_publish_report.json").exists())
            self.assertFalse((task_dir / "output" / "memory_contract.json").exists())
            self.assertEqual(result["memory_contract"], None)

    def test_v0321_2_bootstraps_prior_family_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_task_dir = root / "task_one"
            second_task_dir = root / "task_two"
            first_uploads = first_task_dir / "uploads"
            second_uploads = second_task_dir / "uploads"
            first_uploads.mkdir(parents=True, exist_ok=True)
            second_uploads.mkdir(parents=True, exist_ok=True)

            for name in ("alpha.jpg", "beta.jpg", "style_reference.png"):
                (first_uploads / name).write_bytes(b"stub")
            for name in ("gamma.jpg", "style_reference_2.png"):
                (second_uploads / name).write_bytes(b"stub")

            with patch("services.pipeline_service.ImageProcessor", UniqueIdFakeImageProcessor), patch(
                "services.pipeline_service.FaceRecognition", FakeFaceRecognition
            ), patch("services.pipeline_service.VLMAnalyzer", FakeV03212VLMAnalyzer), patch(
                "services.pipeline_service.LLMProcessor", FakeLLMProcessor
            ):
                from services.pipeline_service import MemoryPipelineService

                first = MemoryPipelineService(
                    task_id="task_one",
                    task_dir=str(first_task_dir),
                    asset_store=FakeAssetStore(),
                    user_id="user_pipeline",
                    face_review_store=FakeFaceReviewStore(),
                    task_version="v0321.2",
                ).run(max_photos=3, use_cache=False)

                second = MemoryPipelineService(
                    task_id="task_two",
                    task_dir=str(second_task_dir),
                    asset_store=FakeAssetStore(),
                    user_id="user_pipeline",
                    face_review_store=FakeFaceReviewStore(),
                    task_version="v0321.2",
                ).run(max_photos=2, use_cache=False)

            self.assertEqual(first["summary"]["event_count"], 1)
            self.assertEqual(second["summary"]["event_count"], 1)
            self.assertTrue(second["memory"]["summary"]["bootstrap_applied"])
            self.assertEqual(second["memory"]["summary"]["bootstrap_source_task_id"], "task_one")
            merged_photo_ids = second["memory"]["event_revisions"][0]["original_photo_ids"]
            self.assertEqual(merged_photo_ids, ["hash-alpha", "hash-beta", "hash-gamma"])
            self.assertEqual(len(second["memory"]["delta_event_revisions"]), 1)
            self.assertEqual(
                second["memory"]["delta_profile_revision"]["original_photo_ids"],
                ["hash-style_reference_2"],
            )
            self.assertEqual(
                second["profile_markdown"],
                second["memory"]["delta_profile_markdown"],
            )
            self.assertEqual(second["memory"]["summary"]["reference_media_signal_count"], 2)
            self.assertEqual(
                second["memory"]["profile_revision"]["original_photo_ids"],
                ["hash-style_reference", "hash-style_reference_2"],
            )

    def test_v0321_3_pipeline_emits_profile_input_pack_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir)
            uploads_dir = task_dir / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            for name in ("camera_a.jpg", "camera_b.jpg", "saved_ai_reference.png"):
                (uploads_dir / name).write_bytes(b"stub")

            with patch("services.pipeline_service.ImageProcessor", FakeImageProcessor), patch(
                "services.pipeline_service.FaceRecognition", FakeFaceRecognition
            ), patch("services.pipeline_service.VLMAnalyzer", FakeV03213VLMAnalyzer), patch(
                "services.pipeline_service.LLMProcessor", FakeLLMProcessor
            ):
                from services.pipeline_service import MemoryPipelineService

                result = MemoryPipelineService(
                    task_id="task_pipeline_v0321_3",
                    task_dir=str(task_dir),
                    asset_store=FakeAssetStore(),
                    user_id="user_pipeline",
                    face_review_store=FakeFaceReviewStore(),
                    task_version="v0321.3",
                ).run(max_photos=3, use_cache=False)

            self.assertEqual(result["version"], "v0321.3")
            self.assertEqual(result["memory"]["pipeline_family"], "v0321_3")
            self.assertIn("profile_input_pack_partial", result["memory"])
            self.assertIn("profile_input_pack", result["memory"])
            partial_pack = result["memory"]["profile_input_pack_partial"]
            final_pack = result["memory"]["profile_input_pack"]
            self.assertEqual(partial_pack["pipeline_family"], "v0321_3")
            self.assertIn("baseline_rhythm", partial_pack)
            self.assertIn("place_patterns", partial_pack)
            self.assertIn("activity_patterns", partial_pack)
            self.assertIn("event_grounded_signals", partial_pack)
            self.assertIn("reference_media_weak_signals", partial_pack)
            self.assertIn("social_patterns", final_pack)
            self.assertIn("change_points", final_pack)
            self.assertIn("key_relationship_refs", final_pack)
            self.assertEqual(result["memory"]["profile_revision"]["generation_mode"], "profile_input_pack_llm")
            self.assertEqual(result["memory"]["summary"]["profile_generation_mode"], "profile_input_pack_llm")
            self.assertEqual(result["summary"]["profile_generation_mode"], "profile_input_pack_llm")
            weak_signal_labels = [
                item["label"]
                for item in final_pack["reference_media_weak_signals"]["aesthetic_hints"]
            ]
            self.assertTrue(weak_signal_labels)
            self.assertTrue((task_dir / "v0321_3" / "profile_input_pack_partial.json").exists())
            self.assertTrue((task_dir / "v0321_3" / "profile_input_pack.json").exists())
            self.assertIn(
                "profile_input_pack_preview",
                result["memory"]["transparency"]["llm_stage"],
            )
            self.assertEqual(
                result["memory"]["transparency"]["llm_stage"]["profile_generation_mode"],
                "profile_input_pack_llm",
            )

    def test_v0321_3_bootstraps_prior_family_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_task_dir = root / "task_three_one"
            second_task_dir = root / "task_three_two"
            first_uploads = first_task_dir / "uploads"
            second_uploads = second_task_dir / "uploads"
            first_uploads.mkdir(parents=True, exist_ok=True)
            second_uploads.mkdir(parents=True, exist_ok=True)

            for name in ("alpha.jpg", "beta.jpg", "style_reference.png"):
                (first_uploads / name).write_bytes(b"stub")
            for name in ("gamma.jpg", "style_reference_2.png"):
                (second_uploads / name).write_bytes(b"stub")

            with patch("services.pipeline_service.ImageProcessor", UniqueIdFakeImageProcessor), patch(
                "services.pipeline_service.FaceRecognition", FakeFaceRecognition
            ), patch("services.pipeline_service.VLMAnalyzer", FakeV03213VLMAnalyzer), patch(
                "services.pipeline_service.LLMProcessor", FakeLLMProcessor
            ):
                from services.pipeline_service import MemoryPipelineService

                MemoryPipelineService(
                    task_id="task_three_one",
                    task_dir=str(first_task_dir),
                    asset_store=FakeAssetStore(),
                    user_id="user_pipeline",
                    face_review_store=FakeFaceReviewStore(),
                    task_version="v0321.3",
                ).run(max_photos=3, use_cache=False)

                second = MemoryPipelineService(
                    task_id="task_three_two",
                    task_dir=str(second_task_dir),
                    asset_store=FakeAssetStore(),
                    user_id="user_pipeline",
                    face_review_store=FakeFaceReviewStore(),
                    task_version="v0321.3",
                ).run(max_photos=2, use_cache=False)

            self.assertTrue(second["memory"]["summary"]["bootstrap_applied"])
            self.assertEqual(second["memory"]["summary"]["bootstrap_source_task_id"], "task_three_one")
            self.assertEqual(second["memory"]["pipeline_family"], "v0321_3")
            self.assertIn("profile_input_pack", second["memory"])
            self.assertGreater(
                len(second["memory"]["profile_input_pack"]["reference_media_weak_signals"]["aesthetic_hints"]),
                0,
            )

    def test_v0321_3_profile_prompt_uses_input_pack_not_raw_revision_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir)
            uploads_dir = task_dir / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)
            for name in ("camera_a.jpg", "camera_b.jpg", "saved_ai_reference.png"):
                (uploads_dir / name).write_bytes(b"stub")

            CaptureMarkdownLLMProcessor.last_markdown_prompt = ""

            with patch("services.pipeline_service.ImageProcessor", FakeImageProcessor), patch(
                "services.pipeline_service.FaceRecognition", FakeFaceRecognition
            ), patch("services.pipeline_service.VLMAnalyzer", FakeV03213VLMAnalyzer), patch(
                "services.pipeline_service.LLMProcessor", CaptureMarkdownLLMProcessor
            ):
                from services.pipeline_service import MemoryPipelineService

                MemoryPipelineService(
                    task_id="task_pipeline_v0321_3_prompt",
                    task_dir=str(task_dir),
                    asset_store=FakeAssetStore(),
                    user_id="user_pipeline",
                    face_review_store=FakeFaceReviewStore(),
                    task_version="v0321.3",
                ).run(max_photos=3, use_cache=False)

            prompt = CaptureMarkdownLLMProcessor.last_markdown_prompt
            self.assertIn("PROFILE_INPUT_PACK=", prompt)
            self.assertIn("KEY_EVENT_CONTEXT=", prompt)
            self.assertIn("KEY_RELATIONSHIP_CONTEXT=", prompt)
            self.assertNotIn("EVENT_REVISIONS=", prompt)
            self.assertNotIn("RELATIONSHIP_REVISIONS=", prompt)
            self.assertNotIn("REFERENCE_MEDIA_WEAK_SIGNALS=", prompt)

    def test_v0321_2_bootstraps_prior_family_state_from_task_record_result(self) -> None:
        from datetime import datetime, timezone
        import shutil

        from sqlalchemy import delete

        from backend.db import SessionLocal
        from backend.models import TaskRecord

        task_ids = ["task_db_one", "task_db_two"]
        user_id = "user_pipeline_db_bootstrap"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                first_task_dir = root / "task_db_one"
                second_task_dir = root / "task_db_two"
                first_uploads = first_task_dir / "uploads"
                second_uploads = second_task_dir / "uploads"
                first_uploads.mkdir(parents=True, exist_ok=True)
                second_uploads.mkdir(parents=True, exist_ok=True)

                for name in ("alpha.jpg", "beta.jpg", "style_reference.png"):
                    (first_uploads / name).write_bytes(b"stub")
                for name in ("gamma.jpg", "style_reference_2.png"):
                    (second_uploads / name).write_bytes(b"stub")

                with patch("services.pipeline_service.ImageProcessor", UniqueIdFakeImageProcessor), patch(
                    "services.pipeline_service.FaceRecognition", FakeFaceRecognition
                ), patch("services.pipeline_service.VLMAnalyzer", FakeV03212VLMAnalyzer), patch(
                    "services.pipeline_service.LLMProcessor", FakeLLMProcessor
                ):
                    from services.pipeline_service import MemoryPipelineService

                    first = MemoryPipelineService(
                        task_id="task_db_one",
                        task_dir=str(first_task_dir),
                        asset_store=FakeAssetStore(),
                        user_id=user_id,
                        face_review_store=FakeFaceReviewStore(),
                        task_version="v0321.2",
                    ).run(max_photos=3, use_cache=False)

                    now = datetime.now(timezone.utc)
                    with SessionLocal() as session:
                        session.execute(delete(TaskRecord).where(TaskRecord.task_id.in_(task_ids)))
                        session.add(
                            TaskRecord(
                                task_id="task_db_one",
                                user_id=user_id,
                                version="v0321.2",
                                status="completed",
                                stage="completed",
                                upload_count=3,
                                task_dir=str(first_task_dir),
                                progress=None,
                                uploads=[],
                                result={"memory": first["memory"]},
                                result_summary=first.get("summary"),
                                asset_manifest=None,
                                error=None,
                                worker_instance_id=None,
                                worker_private_ip=None,
                                worker_status=None,
                                delete_state=None,
                                created_at=now,
                                updated_at=now,
                                expires_at=None,
                                deleted_at=None,
                                last_worker_sync_at=None,
                            )
                        )
                        session.commit()

                    shutil.rmtree(first_task_dir / "v0321_2", ignore_errors=True)

                    second = MemoryPipelineService(
                        task_id="task_db_two",
                        task_dir=str(second_task_dir),
                        asset_store=FakeAssetStore(),
                        user_id=user_id,
                        face_review_store=FakeFaceReviewStore(),
                        task_version="v0321.2",
                    ).run(max_photos=2, use_cache=False)

                self.assertTrue(second["memory"]["summary"]["bootstrap_applied"])
                self.assertEqual(second["memory"]["summary"]["bootstrap_source"], "db")
                self.assertEqual(second["memory"]["summary"]["bootstrap_source_task_id"], "task_db_one")
                self.assertGreater(second["memory"]["summary"]["bootstrap_prior_person_appearance_count"], 0)
                self.assertEqual(
                    second["memory"]["event_revisions"][0]["original_photo_ids"],
                    ["hash-alpha", "hash-beta", "hash-gamma"],
                )
                self.assertEqual(len(second["memory"]["delta_event_revisions"]), 1)
                self.assertEqual(
                    second["memory"]["delta_profile_revision"]["original_photo_ids"],
                    ["hash-style_reference_2"],
                )
                self.assertEqual(
                    second["profile_markdown"],
                    second["memory"]["delta_profile_markdown"],
                )
                self.assertEqual(second["memory"]["summary"]["reference_media_signal_count"], 2)
                self.assertCountEqual(
                    second["memory"]["profile_revision"]["original_photo_ids"],
                    ["hash-style_reference", "hash-style_reference_2"],
                )
        finally:
            with SessionLocal() as session:
                session.execute(delete(TaskRecord).where(TaskRecord.task_id.in_(task_ids)))
                session.commit()

    def test_v0321_2_scale_replay_handles_1000_images_without_oversegmentation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir)
            uploads_dir = task_dir / "uploads"
            uploads_dir.mkdir(parents=True, exist_ok=True)

            for event_idx in range(120):
                for photo_idx in range(5):
                    (uploads_dir / f"camera_event{event_idx:03d}_p{photo_idx:02d}.jpg").write_bytes(b"stub")
            for ref_idx in range(400):
                (uploads_dir / f"style_reference_{ref_idx:03d}.png").write_bytes(b"stub")

            with patch("services.pipeline_service.ImageProcessor", ScaleReplayFakeImageProcessor), patch(
                "services.pipeline_service.FaceRecognition", FakeFaceRecognition
            ), patch("services.pipeline_service.VLMAnalyzer", ScaleReplayFakeVLMAnalyzer), patch(
                "services.pipeline_service.LLMProcessor", FakeLLMProcessor
            ):
                from services.pipeline_service import MemoryPipelineService

                service = MemoryPipelineService(
                    task_id="task_pipeline_v0321_2_scale",
                    task_dir=str(task_dir),
                    asset_store=FakeAssetStore(),
                    user_id="user_pipeline_scale",
                    face_review_store=FakeFaceReviewStore(),
                    task_version="v0321.2",
                )
                result = service.run(max_photos=1000, use_cache=False)

            memory = result["memory"]
            self.assertEqual(result["version"], "v0321.2")
            self.assertEqual(memory["pipeline_family"], "v0321_2")
            self.assertEqual(result["summary"]["event_count"], 120)
            self.assertEqual(memory["summary"]["event_window_count"], 120)
            self.assertFalse(memory["summary"]["over_segmentation_anomaly"])
            self.assertEqual(memory["summary"]["reference_media_signal_count"], 400)
            self.assertEqual(len(memory["event_revisions"]), 120)
            self.assertEqual(len(memory["reference_media_signals"]), 400)
            self.assertIn("aesthetic_preference", memory["profile_revision"]["buckets"])
            self.assertEqual(len(memory["profile_revision"]["original_photo_ids"]), 400)
            self.assertEqual(
                len(memory["event_revisions"][0]["original_photo_ids"]),
                5,
            )
            self.assertTrue((task_dir / "v0321_2" / "state.db").exists())

    def test_v0321_2_segmentation_uses_people_and_vlm_signals_conservatively(self) -> None:
        from datetime import datetime, timedelta

        from services.v0321_2.pipeline import V03212PipelineFamily

        with tempfile.TemporaryDirectory() as tmpdir:
            family = V03212PipelineFamily(
                task_id="task_segmentation",
                task_dir=tmpdir,
                user_id="user_segmentation",
                asset_store=FakeAssetStore(),
                llm_processor=FakeLLMProcessor(task_version="v0321.2"),
            )
            started_at = datetime(2026, 3, 22, 10, 0)
            assets = [
                {
                    "asset_id": "asset_1",
                    "photo_id": "photo_1",
                    "timestamp": started_at.isoformat(),
                    "place_key": "unknown_a",
                },
                {
                    "asset_id": "asset_2",
                    "photo_id": "photo_2",
                    "timestamp": (started_at + timedelta(minutes=6)).isoformat(),
                    "place_key": "unknown_b",
                },
                {
                    "asset_id": "asset_3",
                    "photo_id": "photo_3",
                    "timestamp": (started_at + timedelta(hours=2)).isoformat(),
                    "place_key": "far_place",
                },
            ]
            observations_by_photo = {
                "photo_1": {
                    "place_candidates": ["Cafe_A"],
                    "activity_hint": "meal",
                    "scene_hint": "cafe interior",
                },
                "photo_2": {
                    "place_candidates": ["Cafe_A"],
                    "activity_hint": "meal",
                    "scene_hint": "cafe interior",
                },
                "photo_3": {
                    "place_candidates": ["Office_B"],
                    "activity_hint": "meeting",
                    "scene_hint": "meeting room",
                },
            }
            appearances = [
                {"photo_id": "photo_1", "person_id": "Person_001", "appearance_mode": "live_presence"},
                {"photo_id": "photo_2", "person_id": "Person_001", "appearance_mode": "live_presence"},
                {"photo_id": "photo_3", "person_id": "Person_002", "appearance_mode": "live_presence"},
            ]

            bursts = family._build_bursts(
                assets,
                observations_by_photo=observations_by_photo,
                appearances=appearances,
            )
            boundaries = family._score_boundaries(bursts)

            self.assertEqual(len(bursts), 2)
            self.assertEqual(bursts[0]["photo_ids"], ["photo_1", "photo_2"])
            self.assertEqual(len(boundaries), 1)
            self.assertEqual(boundaries[0]["decision"], "split")
            self.assertFalse(boundaries[0]["place_overlap"])
            self.assertFalse(boundaries[0]["live_overlap"])

    def test_v0321_2_frontier_keeps_stale_event_open_when_continuity_is_strong(self) -> None:
        from services.v0321_2.pipeline import V03212PipelineFamily

        with tempfile.TemporaryDirectory() as tmpdir:
            family = V03212PipelineFamily(
                task_id="task_frontier",
                task_dir=tmpdir,
                user_id="user_frontier",
                asset_store=FakeAssetStore(),
                llm_processor=FakeLLMProcessor(task_version="v0321.2"),
            )
            stale_candidate = {
                "event_root_id": "event_root_old",
                "event_revision_id": "event_rev_old",
                "ended_at": "2026-03-22T09:00:00",
                "participant_person_ids": ["Person_001"],
                "place_refs": ["Cafe_A"],
                "title": "meal @ Cafe_A",
                "event_summary": "meal with close friend",
            }
            current_draft = {
                "started_at": "2026-03-22T12:30:00",
                "participant_person_ids": ["Person_001"],
                "place_refs": ["Cafe_A"],
                "title": "meal @ Cafe_A",
                "event_summary": "meal with close friend and continued chat",
            }
            distant_draft = {
                "started_at": "2026-03-22T12:30:00",
                "participant_person_ids": ["Person_009"],
                "place_refs": ["Office_B"],
                "title": "meeting @ Office_B",
                "event_summary": "work meeting",
            }

            self.assertTrue(family._should_keep_frontier_open(stale_candidate, current_draft))
            self.assertFalse(family._should_keep_frontier_open(stale_candidate, distant_draft))


if __name__ == "__main__":
    unittest.main()
