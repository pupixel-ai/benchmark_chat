from __future__ import annotations

import io
import shutil
import unittest
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import delete

from backend.app import app, task_store
from backend.db import SessionLocal
from backend.models import (
    FaceRecognitionImagePolicyRecord,
    FaceReviewRecord,
    SessionRecord,
    TaskRecord,
    UserRecord,
)


class TaskApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.username = f"codex_{uuid.uuid4().hex[:10]}"
        self.password = "passw0rd!"
        self.task_ids: list[str] = []

        response = self.client.post(
            "/api/auth/register",
            json={"username": self.username, "password": self.password},
        )
        self.assertEqual(response.status_code, 200)
        self.user_id = response.json()["user"]["user_id"]

    def tearDown(self) -> None:
        for task_id in self.task_ids:
            shutil.rmtree(task_store.task_dir(task_id), ignore_errors=True)

        with SessionLocal() as session:
            session.execute(delete(FaceReviewRecord).where(FaceReviewRecord.user_id == self.user_id))
            session.execute(
                delete(FaceRecognitionImagePolicyRecord).where(
                    FaceRecognitionImagePolicyRecord.user_id == self.user_id
                )
            )
            session.execute(delete(SessionRecord).where(SessionRecord.user_id == self.user_id))
            session.execute(delete(TaskRecord).where(TaskRecord.user_id == self.user_id))
            session.execute(delete(UserRecord).where(UserRecord.user_id == self.user_id))
            session.commit()

    def test_upload_batches_and_start_task(self) -> None:
        create_response = self.client.post("/api/tasks")
        self.assertEqual(create_response.status_code, 200)
        task_id = create_response.json()["task_id"]
        self.task_ids.append(task_id)
        self.assertEqual(create_response.json()["version"], "v0315")

        batch_response = self.client.post(
            f"/api/tasks/{task_id}/upload-batches",
            files=[
                ("files", ("one.png", self._image_bytes("red"), "image/png")),
                ("files", ("two.png", self._image_bytes("blue"), "image/png")),
            ],
        )
        self.assertEqual(batch_response.status_code, 200)
        self.assertEqual(batch_response.json()["status"], "uploading")
        self.assertEqual(batch_response.json()["upload_count"], 2)
        self.assertEqual(batch_response.json()["version"], "v0315")

        task = task_store.get_task(task_id, user_id=self.user_id)
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task["version"], "v0315")
        self.assertEqual(len(task.get("uploads") or []), 2)
        self.assertTrue((task_store.task_dir(task_id) / "uploads").exists())
        self.assertTrue(all(upload.get("source_hash") for upload in task["uploads"]))

        with patch("backend.app._run_pipeline_task") as run_pipeline:
            start_response = self.client.post(
                f"/api/tasks/{task_id}/start",
                json={"max_photos": 2, "use_cache": False},
            )

        self.assertEqual(start_response.status_code, 200)
        self.assertEqual(start_response.json()["status"], "queued")
        self.assertEqual(start_response.json()["version"], "v0315")
        run_pipeline.assert_called_once_with(task_id, self.user_id, 2, False, "v0315")

    def test_create_task_accepts_explicit_version_and_rejects_invalid_values(self) -> None:
        create_response = self.client.post("/api/tasks", json={"version": "v0312"})
        self.assertEqual(create_response.status_code, 200)
        payload = create_response.json()
        self.task_ids.append(payload["task_id"])
        self.assertEqual(payload["version"], "v0312")

        task = task_store.get_task(payload["task_id"], user_id=self.user_id)
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task["version"], "v0312")

        invalid_response = self.client.post("/api/tasks", json={"version": "v9999"})
        self.assertEqual(invalid_response.status_code, 400)
        self.assertIn("不支持的任务版本", invalid_response.json()["detail"])

    def test_health_reports_available_task_versions(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["app_version"], "v0315")
        self.assertEqual(payload["default_task_version"], "v0315")
        self.assertEqual(payload["available_task_versions"], ["v0312", "v0315"])

    def test_legacy_task_without_version_is_serialized_as_v0312(self) -> None:
        task_id = uuid.uuid4().hex
        self.task_ids.append(task_id)
        task_dir = task_store.task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        with SessionLocal() as session:
            session.add(
                TaskRecord(
                    task_id=task_id,
                    user_id=self.user_id,
                    version=None,
                    status="completed",
                    stage="completed",
                    upload_count=0,
                    task_dir=str(task_dir),
                    progress=None,
                    uploads=[],
                    result=None,
                    result_summary=None,
                    asset_manifest=None,
                    error=None,
                    worker_instance_id=None,
                    worker_private_ip=None,
                    worker_status=None,
                    delete_state=None,
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    expires_at=None,
                    deleted_at=None,
                    last_worker_sync_at=None,
                )
            )
            session.commit()

        task = task_store.get_task(task_id, user_id=self.user_id)
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task["version"], "v0312")

    def test_review_and_policy_are_merged_into_task_detail(self) -> None:
        create_response = self.client.post("/api/tasks")
        self.assertEqual(create_response.status_code, 200)
        task_id = create_response.json()["task_id"]
        self.task_ids.append(task_id)

        task_store.append_uploads(
            task_id,
            [
                {
                    "image_id": "photo_001",
                    "filename": "sample.png",
                    "stored_filename": "001_sample.png",
                    "path": "uploads/001_sample.png",
                    "url": None,
                    "preview_url": None,
                    "content_type": "image/png",
                    "width": 100,
                    "height": 100,
                    "source_hash": "hash-001",
                }
            ],
            status="completed",
            stage="completed",
        )
        task_store.update_task(task_id, result=self._synthetic_result(task_id), status="completed", stage="completed")

        review_response = self.client.put(
            f"/api/tasks/{task_id}/faces/face_001/review",
            json={"is_inaccurate": True, "comment_text": "Needs regrouping"},
        )
        self.assertEqual(review_response.status_code, 200)

        policy_response = self.client.put(
            f"/api/tasks/{task_id}/images/photo_001/face-policy",
            json={"is_abandoned": True},
        )
        self.assertEqual(policy_response.status_code, 200)

        task_response = self.client.get(f"/api/tasks/{task_id}")
        self.assertEqual(task_response.status_code, 200)
        payload = task_response.json()
        image = payload["result"]["face_recognition"]["images"][0]
        face = image["faces"][0]
        grouped_image = payload["result"]["face_recognition"]["person_groups"][0]["images"][0]

        self.assertTrue(image["is_abandoned"])
        self.assertTrue(face["is_inaccurate"])
        self.assertEqual(face["comment_text"], "Needs regrouping")
        self.assertTrue(grouped_image["is_inaccurate"])
        self.assertTrue(grouped_image["is_abandoned"])
        self.assertEqual(grouped_image["comment_text"], "Needs regrouping")

        reviews_response = self.client.get(f"/api/tasks/{task_id}/reviews")
        self.assertEqual(reviews_response.status_code, 200)
        feedback = reviews_response.json()
        self.assertIn("face_001", feedback["reviews"])
        self.assertIn("hash-001", feedback["policies"])

    def test_memory_query_endpoint_returns_agent_answer(self) -> None:
        create_response = self.client.post("/api/tasks")
        self.assertEqual(create_response.status_code, 200)
        task_id = create_response.json()["task_id"]
        self.task_ids.append(task_id)

        task_store.update_task(task_id, result=self._synthetic_result(task_id), status="completed", stage="completed")

        response = self.client.post(
            f"/api/tasks/{task_id}/memory/query",
            json={"question": "我过去3个月去过的演唱会"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["answer"]["answer_type"], "event_search")
        self.assertIn("concert", payload["answer"]["resolved_concepts"])
        self.assertGreaterEqual(len(payload["answer"]["supporting_events"]), 1)

    def _image_bytes(self, color: str) -> bytes:
        image = Image.new("RGB", (48, 48), color=color)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _synthetic_result(self, task_id: str) -> dict:
        timestamp = datetime.utcnow().isoformat()
        return {
            "task_id": task_id,
            "generated_at": timestamp,
            "summary": {
                "total_uploaded": 1,
                "loaded_images": 1,
                "failed_images": 0,
                "face_processed_images": 1,
                "vlm_processed_images": 0,
                "total_faces": 1,
                "total_persons": 1,
                "primary_person_id": "person_001",
            },
            "face_recognition": {
                "primary_person_id": "person_001",
                "metrics": {
                    "total_images": 1,
                    "total_faces": 1,
                    "total_persons": 1,
                },
                "engine": {
                    "model_name": "test",
                    "providers": ["CPUExecutionProvider"],
                },
                "persons": [
                    {
                        "person_id": "person_001",
                        "photo_count": 1,
                        "face_count": 1,
                        "avg_score": 0.92,
                        "avg_quality": 0.81,
                        "high_quality_face_count": 1,
                    }
                ],
                "images": [
                    {
                        "image_id": "photo_001",
                        "filename": "sample.png",
                        "source_hash": "hash-001",
                        "timestamp": timestamp,
                        "status": "processed",
                        "detection_seconds": 0.1,
                        "embedding_seconds": 0.1,
                        "original_image_url": None,
                        "display_image_url": None,
                        "boxed_image_url": None,
                        "compressed_image_url": None,
                        "location": None,
                        "face_count": 1,
                        "faces": [
                            {
                                "face_id": "face_001",
                                "person_id": "person_001",
                                "score": 0.92,
                                "similarity": 0.87,
                                "faiss_id": 1,
                                "bbox": [0, 0, 10, 10],
                                "image_id": "photo_001",
                                "source_hash": "hash-001",
                                "quality_score": 0.81,
                                "quality_flags": [],
                                "match_decision": "strong_match",
                                "match_reason": "test",
                            }
                        ],
                        "failures": [],
                    }
                ],
                "person_groups": [
                    {
                        "person_id": "person_001",
                        "is_primary": True,
                        "photo_count": 1,
                        "face_count": 1,
                        "avg_score": 0.92,
                        "avg_quality": 0.81,
                        "high_quality_face_count": 1,
                        "avatar_url": None,
                        "images": [
                            {
                                "image_id": "photo_001",
                                "filename": "sample.png",
                                "timestamp": timestamp,
                                "display_image_url": None,
                                "boxed_image_url": None,
                                "source_hash": "hash-001",
                                "face_id": "face_001",
                                "score": 0.92,
                                "similarity": 0.87,
                                "quality_score": 0.81,
                                "quality_flags": [],
                                "match_decision": "strong_match",
                                "match_reason": "test",
                            }
                        ],
                    }
                ],
                "failed_images": [],
            },
            "face_report": {
                "status": "completed",
                "generated_at": timestamp,
                "primary_person_id": "person_001",
                "total_images": 1,
                "total_faces": 1,
                "total_persons": 1,
                "failed_images": 0,
                "ambiguous_faces": 0,
                "low_quality_faces": 0,
                "new_person_from_ambiguity": 0,
                "failed_items": [],
                "engine": {"model_name": "test", "providers": ["CPUExecutionProvider"]},
                "timings": {
                    "detection_seconds": 0.1,
                    "embedding_seconds": 0.1,
                    "total_seconds": 0.2,
                    "average_image_seconds": 0.2,
                },
                "processing": {
                    "original_uploads_preserved": True,
                    "preview_format": "webp",
                    "boxed_format": "webp",
                    "recognition_input": "test",
                },
                "precision_enhancements": ["test"],
                "score_guide": {"detection_score": "test", "similarity": "test"},
                "no_face_images": [],
                "persons": [
                    {
                        "person_id": "person_001",
                        "is_primary": True,
                        "photo_count": 1,
                        "face_count": 1,
                        "avg_score": 0.92,
                        "avg_quality": 0.81,
                        "high_quality_face_count": 1,
                    }
                ],
            },
            "failed_images": [],
            "warnings": [],
            "events": [],
            "memory": {
                "envelope": {"scope": {"user_id": self.user_id}},
                "storage": {
                    "neo4j": {
                        "nodes": {
                            "user": [{"user_id": self.user_id, "labels": ["User"], "properties": {"profile_version": 1}}],
                            "persons": [],
                            "places": [],
                            "sessions": [
                                {
                                    "session_uuid": "session_001",
                                    "labels": ["Session"],
                                    "properties": {
                                        "user_id": self.user_id,
                                        "started_at": "2026-02-15T20:00:00",
                                        "ended_at": "2026-02-15T22:00:00",
                                        "participant_count": 2,
                                        "representative_photo_ids": ["photo_001"],
                                    },
                                }
                            ],
                            "timelines": [],
                            "events": [
                                {
                                    "event_uuid": "event_001",
                                    "labels": ["Event"],
                                    "properties": {
                                        "user_id": self.user_id,
                                        "title": "Live Concert Night",
                                        "normalized_event_type": "concert",
                                        "event_subtype": "concert",
                                        "started_at": "2026-02-15T20:00:00",
                                        "ended_at": "2026-02-15T22:00:00",
                                        "confidence": 0.91,
                                        "representative_photo_ids": ["photo_001"],
                                    },
                                }
                            ],
                            "relationship_hypotheses": [],
                            "mood_states": [],
                            "primary_person_hypotheses": [],
                            "period_hypotheses": [
                                {
                                    "period_uuid": "period_recent",
                                    "labels": ["PeriodHypothesis"],
                                    "properties": {
                                        "user_id": self.user_id,
                                        "period_type": "recent_period",
                                        "label": "最近",
                                        "window_start": "2026-01-01T00:00:00",
                                        "window_end": "2026-03-10T00:00:00",
                                        "confidence": 0.9,
                                    },
                                }
                            ],
                            "concepts": [
                                {
                                    "concept_uuid": "concept_concert",
                                    "labels": ["Concept"],
                                    "properties": {
                                        "canonical_name": "concert",
                                        "aliases": ["演唱会", "concert"],
                                        "concept_type": "event",
                                        "search_text": "concert 演唱会",
                                    },
                                }
                            ],
                        },
                        "edges": [
                            {
                                "edge_id": "edge_event_concept",
                                "from_id": "event_001",
                                "to_id": "concept_concert",
                                "edge_type": "HAS_CONCEPT",
                                "properties": {},
                            },
                            {
                                "edge_id": "edge_event_session",
                                "from_id": "event_001",
                                "to_id": "session_001",
                                "edge_type": "DERIVED_FROM_SESSION",
                                "properties": {},
                            },
                        ],
                    },
                    "milvus": {"segments": []},
                    "redis": {},
                },
            },
        }


if __name__ == "__main__":
    unittest.main()
