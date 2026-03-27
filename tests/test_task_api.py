from __future__ import annotations

import io
import json
import shutil
import threading
import time
import unittest
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import delete

from backend.app import _task_upload_lock, app, memory_db_sync, task_store
from backend.db import SessionLocal
from backend.task_store import normalize_task_options
from config import APP_VERSION, AVAILABLE_TASK_VERSIONS, DEFAULT_NORMALIZE_LIVE_PHOTOS, DEFAULT_TASK_VERSION
from backend.models import (
    ArtifactRecord,
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
            memory_db_sync.delete_task_snapshot(task_id, self.user_id, delete_task_record=True)

        with SessionLocal() as session:
            session.execute(delete(ArtifactRecord).where(ArtifactRecord.user_id == self.user_id))
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
        self.assertEqual(create_response.json()["version"], DEFAULT_TASK_VERSION)
        self.assertEqual(
            create_response.json()["options"]["normalize_live_photos"],
            DEFAULT_NORMALIZE_LIVE_PHOTOS,
        )
        self.assertEqual(create_response.json()["options"]["creation_source"], "manual")

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
        self.assertEqual(batch_response.json()["version"], DEFAULT_TASK_VERSION)

        task = task_store.get_task(task_id, user_id=self.user_id)
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task["version"], DEFAULT_TASK_VERSION)
        self.assertEqual(task["options"]["normalize_live_photos"], DEFAULT_NORMALIZE_LIVE_PHOTOS)
        self.assertEqual(task["options"]["creation_source"], "manual")
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
        self.assertEqual(start_response.json()["version"], DEFAULT_TASK_VERSION)
        self.assertEqual(
            start_response.json()["options"]["normalize_live_photos"],
            DEFAULT_NORMALIZE_LIVE_PHOTOS,
        )
        self.assertEqual(start_response.json()["options"]["creation_source"], "manual")
        run_pipeline.assert_called_once_with(
            task_id,
            self.user_id,
            2,
            False,
            DEFAULT_TASK_VERSION,
            {
                "normalize_live_photos": DEFAULT_NORMALIZE_LIVE_PHOTOS,
                "creation_source": "manual",
                "expected_upload_count": None,
                "requested_max_photos": None,
                "auto_start_on_upload_complete": False,
            },
        )

    def test_ingest_endpoint_creates_uploads_and_starts_task(self) -> None:
        with patch("backend.app._run_pipeline_task") as run_pipeline:
            response = self.client.post(
                "/api/tasks/ingest",
                data={
                    "version": "v0323",
                    "max_photos": "2",
                    "use_cache": "false",
                    "normalize_live_photos": "true",
                },
                files=[
                    ("files", ("one.png", self._image_bytes("red"), "image/png")),
                    ("files", ("two.png", self._image_bytes("blue"), "image/png")),
                ],
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        task_id = payload["task_id"]
        self.task_ids.append(task_id)
        self.assertEqual(payload["status"], "queued")
        self.assertEqual(payload["stage"], "queued")
        self.assertEqual(payload["version"], "v0323")
        self.assertEqual(payload["accepted_count"], 2)
        self.assertEqual(payload["failed_count"], 0)
        self.assertEqual(payload["upload_count"], 2)
        self.assertEqual(payload["max_photos"], 2)
        self.assertTrue(payload["options"]["normalize_live_photos"])
        self.assertEqual(payload["options"]["creation_source"], "api")

        task = task_store.get_task(task_id, user_id=self.user_id)
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task["version"], "v0323")
        self.assertEqual(task["status"], "queued")
        self.assertEqual(task["stage"], "queued")
        self.assertEqual(task["upload_count"], 2)
        self.assertEqual(task["options"]["creation_source"], "api")
        self.assertEqual(len(task.get("uploads") or []), 2)
        self.assertTrue((task_store.task_dir(task_id) / "uploads").exists())

        run_pipeline.assert_called_once_with(
            task_id,
            self.user_id,
            2,
            False,
            "v0323",
            {
                "normalize_live_photos": True,
                "creation_source": "api",
                "expected_upload_count": 2,
                "requested_max_photos": 2,
                "auto_start_on_upload_complete": False,
            },
        )

    def test_ingest_endpoint_rejects_invalid_version(self) -> None:
        response = self.client.post(
            "/api/tasks/ingest",
            data={"version": "v9999"},
            files=[
                ("files", ("one.png", self._image_bytes("red"), "image/png")),
            ],
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("不支持的任务版本", response.json()["detail"])

    def test_create_task_accepts_explicit_version_and_rejects_invalid_values(self) -> None:
        create_response = self.client.post(
            "/api/tasks",
            json={"version": "v0312", "normalize_live_photos": False, "creation_source": "api"},
        )
        self.assertEqual(create_response.status_code, 200)
        payload = create_response.json()
        self.task_ids.append(payload["task_id"])
        self.assertEqual(payload["version"], "v0312")
        self.assertFalse(payload["options"]["normalize_live_photos"])
        self.assertEqual(payload["options"]["creation_source"], "api")

        task = task_store.get_task(payload["task_id"], user_id=self.user_id)
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task["version"], "v0312")
        self.assertFalse(task["options"]["normalize_live_photos"])
        self.assertEqual(task["options"]["creation_source"], "api")

        invalid_response = self.client.post("/api/tasks", json={"version": "v9999"})
        self.assertEqual(invalid_response.status_code, 400)
        self.assertIn("不支持的任务版本", invalid_response.json()["detail"])

    def test_manual_task_options_fall_back_to_requested_max_photos_for_auto_start(self) -> None:
        options = normalize_task_options(
            {
                "creation_source": "manual",
                "requested_max_photos": 769,
            }
        )

        self.assertEqual(options["expected_upload_count"], 769)
        self.assertEqual(options["requested_max_photos"], 769)
        self.assertTrue(options["auto_start_on_upload_complete"])

    def test_upload_batches_accepts_livp_container(self) -> None:
        create_response = self.client.post("/api/tasks")
        self.assertEqual(create_response.status_code, 200)
        task_id = create_response.json()["task_id"]
        self.task_ids.append(task_id)

        batch_response = self.client.post(
            f"/api/tasks/{task_id}/upload-batches",
            files=[
                ("files", ("sample.livp", self._livp_bytes("purple"), "application/octet-stream")),
            ],
        )
        self.assertEqual(batch_response.status_code, 200)
        self.assertEqual(batch_response.json()["failed_count"], 0)
        self.assertEqual(batch_response.json()["upload_count"], 1)

        task = task_store.get_task(task_id, user_id=self.user_id)
        self.assertIsNotNone(task)
        assert task is not None
        uploads = task.get("uploads") or []
        self.assertEqual(len(uploads), 1)
        self.assertTrue(bool(uploads[0].get("is_live_photo_candidate")))
        self.assertEqual(uploads[0].get("stored_filename"), "001_sample.livp")
        self.assertTrue((task_store.task_dir(task_id) / "uploads" / "001_sample.livp").exists())

    def test_upload_batches_auto_start_after_expected_upload_count(self) -> None:
        create_response = self.client.post(
            "/api/tasks",
            json={
                "version": "v0323",
                "normalize_live_photos": True,
                "expected_upload_count": 2,
                "requested_max_photos": 2,
                "auto_start_on_upload_complete": True,
            },
        )
        self.assertEqual(create_response.status_code, 200)
        task_id = create_response.json()["task_id"]
        self.task_ids.append(task_id)

        with patch("backend.app._run_pipeline_task") as run_pipeline:
            batch_response = self.client.post(
                f"/api/tasks/{task_id}/upload-batches",
                files=[
                    ("files", ("one.png", self._image_bytes("red"), "image/png")),
                    ("files", ("two.png", self._image_bytes("blue"), "image/png")),
                ],
            )

        self.assertEqual(batch_response.status_code, 200)
        payload = batch_response.json()
        self.assertTrue(payload["auto_started"])
        self.assertEqual(payload["status"], "queued")
        self.assertEqual(payload["stage"], "queued")

        task = task_store.get_task(task_id, user_id=self.user_id)
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task["status"], "queued")
        self.assertEqual(task["stage"], "queued")
        self.assertEqual(task["upload_count"], 2)
        self.assertEqual(task["options"]["creation_source"], "manual")
        self.assertEqual(task["options"]["expected_upload_count"], 2)
        self.assertEqual(task["options"]["requested_max_photos"], 2)
        self.assertTrue(task["options"]["auto_start_on_upload_complete"])

        run_pipeline.assert_called_once_with(
            task_id,
            self.user_id,
            2,
            False,
            "v0323",
            {
                "normalize_live_photos": True,
                "creation_source": "manual",
                "expected_upload_count": 2,
                "requested_max_photos": 2,
                "auto_start_on_upload_complete": True,
            },
        )

    def test_create_task_defaults_manual_auto_start_when_expected_count_is_known(self) -> None:
        create_response = self.client.post(
            "/api/tasks",
            json={
                "version": "v0323",
                "normalize_live_photos": True,
                "expected_upload_count": 769,
                "requested_max_photos": 769,
            },
        )
        self.assertEqual(create_response.status_code, 200)
        payload = create_response.json()
        self.task_ids.append(payload["task_id"])
        self.assertEqual(payload["options"]["creation_source"], "manual")
        self.assertEqual(payload["options"]["expected_upload_count"], 769)
        self.assertEqual(payload["options"]["requested_max_photos"], 769)
        self.assertTrue(payload["options"]["auto_start_on_upload_complete"])

    def test_upload_batches_can_backfill_auto_start_metadata_for_legacy_manual_task(self) -> None:
        create_response = self.client.post("/api/tasks", json={"version": "v0323"})
        self.assertEqual(create_response.status_code, 200)
        task_id = create_response.json()["task_id"]
        self.task_ids.append(task_id)

        with patch("backend.app._run_pipeline_task") as run_pipeline:
            batch_response = self.client.post(
                f"/api/tasks/{task_id}/upload-batches",
                data={
                    "creation_source": "manual",
                    "expected_upload_count": "2",
                    "requested_max_photos": "2",
                    "auto_start_on_upload_complete": "true",
                },
                files=[
                    ("files", ("one.png", self._image_bytes("red"), "image/png")),
                    ("files", ("two.png", self._image_bytes("blue"), "image/png")),
                ],
            )

        self.assertEqual(batch_response.status_code, 200)
        payload = batch_response.json()
        self.assertTrue(payload["auto_started"])
        self.assertEqual(payload["status"], "queued")
        task = task_store.get_task(task_id, user_id=self.user_id)
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task["options"]["creation_source"], "manual")
        self.assertEqual(task["options"]["expected_upload_count"], 2)
        self.assertEqual(task["options"]["requested_max_photos"], 2)
        self.assertTrue(task["options"]["auto_start_on_upload_complete"])
        run_pipeline.assert_called_once()

    def test_task_upload_lock_serializes_same_task_requests(self) -> None:
        create_response = self.client.post("/api/tasks")
        self.assertEqual(create_response.status_code, 200)
        task_id = create_response.json()["task_id"]
        self.task_ids.append(task_id)

        release_first = threading.Event()
        second_acquired = threading.Event()
        order: list[str] = []

        def first_worker() -> None:
            with _task_upload_lock(task_id):
                order.append("first_acquired")
                release_first.wait(timeout=2)
                order.append("first_releasing")

        def second_worker() -> None:
            with _task_upload_lock(task_id):
                order.append("second_acquired")
                second_acquired.set()
                order.append("second_releasing")

        first = threading.Thread(target=first_worker)
        second = threading.Thread(target=second_worker)
        first.start()
        time.sleep(0.1)
        second.start()
        time.sleep(0.2)

        self.assertFalse(second_acquired.is_set())

        release_first.set()
        first.join(timeout=2)
        second.join(timeout=2)

        self.assertTrue(second_acquired.is_set())
        self.assertEqual(
            order,
            ["first_acquired", "first_releasing", "second_acquired", "second_releasing"],
        )

    def test_health_reports_available_task_versions(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["app_version"], APP_VERSION)
        self.assertEqual(payload["default_task_version"], DEFAULT_TASK_VERSION)
        self.assertEqual(payload["available_task_versions"], list(AVAILABLE_TASK_VERSIONS))

    def test_memory_steps_endpoint_returns_v0323_step_payload_for_failed_task(self) -> None:
        create_response = self.client.post("/api/tasks", json={"version": "v0323"})
        self.assertEqual(create_response.status_code, 200)
        task_id = create_response.json()["task_id"]
        self.task_ids.append(task_id)

        family_dir = task_store.task_dir(task_id) / "v0323"
        family_dir.mkdir(parents=True, exist_ok=True)
        (family_dir / "lp1_events_compact.json").write_text(
            json.dumps([{"event_id": "EVT_0001", "title": "Breakfast"}], ensure_ascii=False),
            encoding="utf-8",
        )
        (family_dir / "lp1_batch_outputs.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "batch_id": "BATCH_0001",
                            "attempt": 1,
                            "max_attempts": 2,
                            "prompt_kind": "primary",
                            "parse_status": "raw_parse_failed",
                            "response_char_count": 12345,
                            "error": "malformed json",
                        },
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        {
                            "batch_id": "BATCH_0001",
                            "attempt": 2,
                            "max_attempts": 2,
                            "prompt_kind": "retry",
                            "parse_status": "retry_ok",
                            "response_char_count": 6789,
                            "error": None,
                        },
                        ensure_ascii=False,
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (family_dir / "lp2_relationships.jsonl").write_text(
            json.dumps({"person_id": "Person_002", "relationship_type": "friend"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (family_dir / "llm_failures.jsonl").write_text(
            json.dumps({"step": "lp2_relationship", "person_id": "Person_003", "error": "timeout"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        task_store.update_task(
            task_id,
            status="failed",
            stage="failed",
            progress={
                "current_stage": "failed",
                "stages": {
                    "memory": {
                        "substage": "lp2_relationship",
                        "processed_candidates": 1,
                        "candidate_count": 2,
                        "relationship_count": 1,
                        "person_id": "Person_002",
                        "current_person_id": "Person_003",
                        "current_candidate_index": 2,
                        "call_started_at": "2026-03-24T08:30:33",
                        "call_timeout_seconds": 180,
                    }
                },
            },
        )

        response = self.client.get(f"/api/tasks/{task_id}/memory/steps")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["pipeline_family"], "v0323")
        self.assertEqual(payload["steps"]["lp1"]["status"], "completed")
        self.assertEqual(payload["steps"]["lp1"]["summary"]["attempt_count"], 2)
        self.assertEqual(payload["steps"]["lp1"]["summary"]["retry_count"], 1)
        self.assertEqual(payload["steps"]["lp1"]["summary"]["last_parse_status"], "retry_ok")
        self.assertEqual(payload["steps"]["lp1"]["attempts"][1]["prompt_kind"], "retry")
        self.assertEqual(payload["steps"]["lp2"]["status"], "failed")
        self.assertEqual(payload["steps"]["lp2"]["summary"]["current_person_id"], "Person_003")
        self.assertEqual(payload["steps"]["lp2"]["data"][0]["person_id"], "Person_002")
        self.assertEqual(payload["steps"]["lp2"]["failures"][0]["person_id"], "Person_003")

    def test_memory_steps_endpoint_returns_v0325_step_payload_for_failed_task(self) -> None:
        create_response = self.client.post("/api/tasks", json={"version": "v0325"})
        self.assertEqual(create_response.status_code, 200)
        task_id = create_response.json()["task_id"]
        self.task_ids.append(task_id)

        family_dir = task_store.task_dir(task_id) / "v0325"
        family_dir.mkdir(parents=True, exist_ok=True)
        (family_dir / "lp1_events_compact.json").write_text(
            json.dumps([{"event_id": "EVT_0001", "title": "Breakfast"}], ensure_ascii=False),
            encoding="utf-8",
        )
        (family_dir / "lp1_batch_outputs.jsonl").write_text(
            json.dumps(
                {
                    "batch_id": "BATCH_0001",
                    "attempt": 1,
                    "max_attempts": 1,
                    "prompt_kind": "primary",
                    "parse_status": "ok",
                    "response_char_count": 4096,
                    "contract_version": "v0325.lp1.output_window.v1",
                    "convert_finish_reason": "stop",
                    "error": None,
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (family_dir / "lp1_salvage_report.json").write_text(
            json.dumps(
                {
                    "summary": {
                        "salvage_detected": True,
                        "salvaged_event_count": 2,
                        "contract_version": "v0325.lp1.output_window.v1",
                    },
                    "batches": [
                        {
                            "batch_id": "BATCH_0001",
                            "attempt": 1,
                            "salvage_status": "detected",
                            "salvaged_event_count": 2,
                            "provider_finish_reason": "length",
                        }
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (family_dir / "lp1_salvaged_events.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"debug_only": True, "event_id": "TEMP_EVT_001"}, ensure_ascii=False),
                    json.dumps({"debug_only": True, "event_id": "TEMP_EVT_002"}, ensure_ascii=False),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (family_dir / "lp2_relationships.json").write_text(
            json.dumps(
                [
                    {
                        "person_id": "Person_002",
                        "relationship_type": "close_friend",
                        "confidence": 0.86,
                        "evidence": {"photo_ids": ["photo_001"], "event_ids": ["EVT_0001"]},
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (family_dir / "lp3_profile.json").write_text(
            json.dumps({"report_markdown": "# Profile\n\n- stable social circle"}, ensure_ascii=False),
            encoding="utf-8",
        )
        (family_dir / "raw_upstream_manifest.json").write_text(
            json.dumps({"attachments": [{"attachment_key": "raw_face_output"}]}, ensure_ascii=False),
            encoding="utf-8",
        )
        (family_dir / "raw_upstream_index.json").write_text(
            json.dumps({"photo": {"photo_001": {}}}, ensure_ascii=False),
            encoding="utf-8",
        )
        (family_dir / "llm_failures.jsonl").write_text(
            json.dumps({"step": "lp3_profile", "field_key": "long_term_facts.identity.role", "error": "timeout"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        task_store.update_task(
            task_id,
            status="failed",
            stage="failed",
            progress={
                "current_stage": "failed",
                "stages": {
                    "memory": {
                        "substage": "lp3_profile",
                        "processed_candidates": 1,
                        "candidate_count": 1,
                        "relationship_count": 1,
                        "provider": "openrouter",
                        "model": "google/gemini-3.1-pro-preview",
                    }
                },
            },
        )

        response = self.client.get(f"/api/tasks/{task_id}/memory/steps")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["pipeline_family"], "v0325")
        self.assertEqual(payload["steps"]["lp1"]["status"], "completed")
        self.assertEqual(payload["steps"]["lp1"]["summary"]["salvage_status"], "detected")
        self.assertEqual(payload["steps"]["lp1"]["summary"]["salvaged_event_count"], 2)
        self.assertEqual(payload["steps"]["lp1"]["summary"]["provider_finish_reason"], "stop")
        self.assertEqual(payload["steps"]["lp1"]["summary"]["contract_version"], "v0325.lp1.output_window.v1")
        self.assertTrue(payload["steps"]["lp1"]["artifacts"]["lp1_salvage_report"]["exists"])
        self.assertTrue(payload["steps"]["lp1"]["artifacts"]["lp1_salvaged_events"]["exists"])
        self.assertEqual(payload["steps"]["lp2"]["data"][0]["person_id"], "Person_002")
        self.assertEqual(payload["steps"]["lp3"]["status"], "failed")
        self.assertTrue(payload["steps"]["lp3"]["artifacts"]["raw_upstream_manifest"]["exists"])
        self.assertTrue(payload["steps"]["lp3"]["artifacts"]["raw_upstream_index"]["exists"])
        self.assertEqual(payload["steps"]["lp3"]["failures"][0]["field_key"], "long_term_facts.identity.role")

    def test_completed_v0323_task_exposes_analysis_bundle_download(self) -> None:
        create_response = self.client.post("/api/tasks", json={"version": "v0323"})
        self.assertEqual(create_response.status_code, 200)
        task_id = create_response.json()["task_id"]
        self.task_ids.append(task_id)

        task_dir = task_store.task_dir(task_id)
        (task_dir / "cache" / "boxed_images").mkdir(parents=True, exist_ok=True)
        (task_dir / "cache" / "face_crops").mkdir(parents=True, exist_ok=True)
        (task_dir / "v0323").mkdir(parents=True, exist_ok=True)

        (task_dir / "cache" / "face_recognition_output.json").write_text("{}", encoding="utf-8")
        (task_dir / "cache" / "face_recognition_state.json").write_text("{}", encoding="utf-8")
        (task_dir / "cache" / "boxed_images" / "photo_001_boxed.jpg").write_bytes(b"boxed")
        (task_dir / "cache" / "face_crops" / "face_001.jpg").write_bytes(b"crop")
        (task_dir / "v0323" / "vp1_observations.json").write_text("[]", encoding="utf-8")
        (task_dir / "v0323" / "lp1_events_compact.json").write_text("[]", encoding="utf-8")
        (task_dir / "v0323" / "lp1_batch_outputs.jsonl").write_text("", encoding="utf-8")

        task_store.update_task(task_id, status="completed", stage="completed")

        task_response = self.client.get(f"/api/tasks/{task_id}")
        self.assertEqual(task_response.status_code, 200)
        payload = task_response.json()
        bundle = payload["downloads"]["analysis_bundle"]
        self.assertTrue(bundle["url"].endswith("/downloads/analysis-bundle"))

        download_response = self.client.get(bundle["url"])
        self.assertEqual(download_response.status_code, 200)
        self.assertEqual(download_response.headers["content-type"], "application/zip")

        with zipfile.ZipFile(io.BytesIO(download_response.content)) as archive:
            members = set(archive.namelist())
            self.assertIn("bundle_manifest.json", members)
            self.assertIn("face/face_recognition_output.json", members)
            self.assertIn("face/boxed_images/photo_001_boxed.jpg", members)
            self.assertIn("face/face_crops/face_001.jpg", members)
            self.assertIn("vlm/vp1_observations.json", members)
            self.assertIn("lp1/lp1_events_compact.json", members)

    def test_completed_v0325_task_exposes_analysis_bundle_download_with_raw_artifacts(self) -> None:
        create_response = self.client.post("/api/tasks", json={"version": "v0325"})
        self.assertEqual(create_response.status_code, 200)
        task_id = create_response.json()["task_id"]
        self.task_ids.append(task_id)

        task_dir = task_store.task_dir(task_id)
        (task_dir / "cache" / "boxed_images").mkdir(parents=True, exist_ok=True)
        (task_dir / "cache" / "face_crops").mkdir(parents=True, exist_ok=True)
        (task_dir / "v0325").mkdir(parents=True, exist_ok=True)

        (task_dir / "cache" / "face_recognition_output.json").write_text("{}", encoding="utf-8")
        (task_dir / "cache" / "face_recognition_state.json").write_text("{}", encoding="utf-8")
        (task_dir / "cache" / "dedupe_report.json").write_text("{}", encoding="utf-8")
        (task_dir / "cache" / "boxed_images" / "photo_001_boxed.jpg").write_bytes(b"boxed")
        (task_dir / "cache" / "face_crops" / "face_001.jpg").write_bytes(b"crop")
        (task_dir / "v0325" / "vp1_observations.json").write_text("[]", encoding="utf-8")
        (task_dir / "v0325" / "lp1_events_compact.json").write_text("[]", encoding="utf-8")
        (task_dir / "v0325" / "lp1_batch_outputs.jsonl").write_text("", encoding="utf-8")
        (task_dir / "v0325" / "raw_upstream_manifest.json").write_text("{}", encoding="utf-8")
        (task_dir / "v0325" / "raw_upstream_index.json").write_text("{}", encoding="utf-8")

        task_store.update_task(task_id, status="completed", stage="completed")

        task_response = self.client.get(f"/api/tasks/{task_id}")
        self.assertEqual(task_response.status_code, 200)
        payload = task_response.json()
        bundle = payload["downloads"]["analysis_bundle"]
        self.assertIn("raw", bundle["categories"])

        download_response = self.client.get(bundle["url"])
        self.assertEqual(download_response.status_code, 200)

        with zipfile.ZipFile(io.BytesIO(download_response.content)) as archive:
            members = set(archive.namelist())
            self.assertIn("bundle_manifest.json", members)
            self.assertIn("vlm/vp1_observations.json", members)
            self.assertIn("lp1/lp1_events_compact.json", members)
            self.assertIn("raw/raw_upstream_manifest.json", members)
            self.assertIn("raw/raw_upstream_index.json", members)

    def test_legacy_task_without_version_is_serialized_as_default_version(self) -> None:
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
        self.assertEqual(task["version"], DEFAULT_TASK_VERSION)

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

    def test_task_detail_strips_bootstrap_fields_from_client_payload(self) -> None:
        create_response = self.client.post("/api/tasks", json={"version": "v0321.3"})
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
                    "url": "/assets/uploads/001_sample.png",
                    "preview_url": None,
                    "content_type": "image/png",
                    "width": 100,
                    "height": 100,
                    "source_hash": "hash-001",
                    "timestamp": "2026-03-20T20:00:00",
                }
            ],
            status="completed",
            stage="completed",
        )
        result = self._synthetic_revision_first_result(task_id)
        result["memory"]["summary"] = {
            **(result["memory"].get("summary") or {}),
            "bootstrap_applied": True,
            "bootstrap_source": "db",
            "bootstrap_source_task_id": "task_old",
            "bootstrap_prior_event_revision_count": 57,
        }
        task_store.update_task(
            task_id,
            result=result,
            result_summary={
                "event_count": 1,
                "bootstrap_applied": True,
                "bootstrap_source_task_id": "task_old",
            },
            status="completed",
            stage="completed",
        )

        response = self.client.get(f"/api/tasks/{task_id}")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        memory_summary = ((payload.get("result") or {}).get("memory") or {}).get("summary") or {}
        self.assertNotIn("bootstrap_applied", memory_summary)
        self.assertNotIn("bootstrap_source", memory_summary)
        self.assertNotIn("bootstrap_source_task_id", memory_summary)
        self.assertNotIn("bootstrap_prior_event_revision_count", memory_summary)
        self.assertNotIn("bootstrap_applied", payload.get("result_summary") or {})
        self.assertNotIn("bootstrap_source_task_id", payload.get("result_summary") or {})

    def test_memory_query_endpoint_returns_agent_answer(self) -> None:
        create_response = self.client.post("/api/tasks", json={"version": "v0321.3"})
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
                    "url": "/assets/uploads/001_sample.png",
                    "preview_url": None,
                    "content_type": "image/png",
                    "width": 100,
                    "height": 100,
                    "source_hash": "hash-001",
                    "timestamp": "2026-03-20T20:00:00",
                }
            ],
            status="completed",
            stage="completed",
        )
        task_store.update_task(
            task_id,
            result=self._synthetic_revision_first_result(task_id),
            status="completed",
            stage="completed",
        )

        response = self.client.post(
            f"/api/tasks/{task_id}/memory/query",
            json={"question": "我过去3个月去过的演唱会"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query_plan"]["plan_type"], "hybrid")
        self.assertEqual(payload["answer"]["answer_type"], "event_search")
        self.assertEqual(payload["answer"]["original_photo_ids"], ["hash-001"])
        self.assertGreaterEqual(len(payload["supporting_units"]), 1)
        self.assertGreaterEqual(len(payload["supporting_evidence"]), 1)

        overview_response = self.client.post(
            f"/api/tasks/{task_id}/memory/query",
            json={"question": "请总结这个任务里的主要事件、人物关系和用户画像"},
        )
        self.assertEqual(overview_response.status_code, 200)
        overview_payload = overview_response.json()
        self.assertEqual(overview_payload["query_plan"]["plan_type"], "hybrid")
        self.assertEqual(overview_payload["answer"]["answer_type"], "task_overview")
        self.assertGreaterEqual(len(overview_payload["supporting_units"]), 1)
        self.assertGreaterEqual(len(overview_payload["supporting_graph_entities"]), 1)
        self.assertGreaterEqual(len(overview_payload["supporting_evidence"]), 1)
        self.assertNotIn("推理草稿箱", overview_payload["answer"]["summary"])
        self.assertNotIn("1. 时空锚点确认", overview_payload["answer"]["summary"])

    def test_memory_core_endpoint_returns_events_relationships_profile_with_photos(self) -> None:
        create_response = self.client.post("/api/tasks", json={"version": "v0321.3"})
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
                    "url": "/assets/uploads/001_sample.png",
                    "preview_url": None,
                    "content_type": "image/png",
                    "width": 100,
                    "height": 100,
                    "source_hash": "hash-001",
                    "timestamp": "2026-03-20T20:00:00",
                }
            ],
            status="completed",
            stage="completed",
        )
        task_store.update_task(
            task_id,
            result=self._synthetic_revision_first_result(task_id),
            status="completed",
            stage="completed",
        )

        response = self.client.get(f"/api/tasks/{task_id}/memory/core")
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(len(payload["events"]), 1)
        self.assertEqual(len(payload["relationships"]), 1)
        self.assertEqual(len(payload["vlm"]), 1)
        self.assertEqual(set(payload.keys()), {"vlm", "events", "relationships", "profile"})

        event = payload["events"][0]
        self.assertEqual(event["llm_summary"], "Live concert with a close friend.")
        self.assertEqual(event["photo_ids"], ["photo_001"])
        self.assertEqual(event["person_ids"], ["Person_001", "Person_002"])
        self.assertEqual(len(event["vlm"]), 1)

        relationship = payload["relationships"][0]
        self.assertEqual(relationship["person_id"], "Person_002")
        self.assertEqual(relationship["photo_ids"], ["photo_001"])

        profile = payload["profile"]
        self.assertTrue(profile["report_markdown"].startswith("# Profile"))

        vlm = payload["vlm"][0]
        self.assertEqual(vlm["photo_id"], "photo_001")
        self.assertEqual(vlm["person_ids"], ["Person_001"])

    def test_memory_core_endpoint_uses_task_scoped_delta_not_bootstrap_snapshot(self) -> None:
        create_response = self.client.post("/api/tasks", json={"version": "v0321.3"})
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
                    "url": "/assets/uploads/001_sample.png",
                    "preview_url": None,
                    "content_type": "image/png",
                    "width": 100,
                    "height": 100,
                    "source_hash": "hash-001",
                    "timestamp": "2026-03-20T20:00:00",
                }
            ],
            status="completed",
            stage="completed",
        )
        result = self._synthetic_revision_first_result(task_id)
        result["memory"]["event_revisions"].append(
            {
                "event_root_id": "event_root_old",
                "event_revision_id": "event_rev_old",
                "revision": 1,
                "title": "Historical Event",
                "event_summary": "Should not appear in task-scoped full recall.",
                "started_at": "2026-02-01T20:00:00",
                "ended_at": "2026-02-01T22:00:00",
                "participant_person_ids": ["Person_999"],
                "depicted_person_ids": ["Person_999"],
                "place_refs": ["Beijing"],
                "original_photo_ids": ["hash-old"],
                "confidence": 0.9,
                "status": "active",
                "sealed_state": "sealed",
                "atomic_evidence": [],
            }
        )
        result["memory"]["relationship_revisions"].append(
            {
                "relationship_root_id": "rel_root_old",
                "relationship_revision_id": "rel_rev_old",
                "target_person_id": "Person_999",
                "relationship_type": "friend",
                "label": "historical friend",
                "confidence": 0.5,
                "supporting_event_ids": ["event_rev_old"],
            }
        )
        task_store.update_task(task_id, result=result, status="completed", stage="completed")

        response = self.client.get(f"/api/tasks/{task_id}/memory/core")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["events"]), 1)
        self.assertEqual(len(payload["relationships"]), 1)
        self.assertEqual([item["llm_summary"] for item in payload["events"]], ["Live concert with a close friend."])
        self.assertEqual([item["person_id"] for item in payload["relationships"]], ["Person_002"])

    def test_artifact_catalog_endpoint_returns_uploaded_files(self) -> None:
        create_response = self.client.post("/api/tasks")
        self.assertEqual(create_response.status_code, 200)
        task_id = create_response.json()["task_id"]
        self.task_ids.append(task_id)

        batch_response = self.client.post(
            f"/api/tasks/{task_id}/upload-batches",
            files=[("files", ("one.png", self._image_bytes("green"), "image/png"))],
        )
        self.assertEqual(batch_response.status_code, 200)

        response = self.client.get(f"/api/tasks/{task_id}/artifacts")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertGreaterEqual(payload["artifact_count"], 1)
        artifact = next(item for item in payload["artifacts"] if item["relative_path"] == "uploads/001_one.png")
        self.assertEqual(artifact["relative_path"], "uploads/001_one.png")
        self.assertEqual(artifact["stage"], "uploads")
        self.assertEqual(artifact["storage_backend"], "local")
        self.assertTrue(artifact["asset_url"].endswith("/uploads/001_one.png"))

    def test_user_memory_faces_endpoint_returns_stable_urls_and_assets(self) -> None:
        create_response = self.client.post("/api/tasks", json={"version": "v0327-exp"})
        self.assertEqual(create_response.status_code, 200)
        task_id = create_response.json()["task_id"]
        self.task_ids.append(task_id)

        self._append_uploaded_sample(task_id, color="purple")
        task_store.update_task(
            task_id,
            result=self._synthetic_v0327_memory_result(task_id),
            status="completed",
            stage="completed",
        )

        response = self.client.get(f"/api/users/{self.user_id}/memory/faces", params={"task_id": task_id})
        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertEqual(payload["query"]["task_id"], task_id)
        self.assertEqual(len(payload["tasks"]), 1)
        task_payload = payload["tasks"][0]
        self.assertEqual(task_payload["pipeline_version"], 327)
        self.assertEqual(task_payload["pipeline_channel"], "exp")
        self.assertEqual(task_payload["stage_versions"]["face"], 327)
        self.assertEqual(len(task_payload["identities"]), 2)
        self.assertEqual(len(task_payload["faces"]), 2)
        self.assertEqual(len(task_payload["photos"]), 1)

        face = task_payload["faces"][0]
        photo = task_payload["photos"][0]
        self.assertTrue(face["urls"]["crop"].startswith("/api/assets/faces/"))
        self.assertTrue(face["urls"]["boxed_full"].startswith("/api/assets/photos/"))
        self.assertTrue(photo["urls"]["raw"].startswith("/api/assets/photos/"))
        self.assertTrue(photo["urls"]["display"].startswith("/api/assets/photos/"))

        raw_response = self.client.get(photo["urls"]["raw"])
        self.assertEqual(raw_response.status_code, 200)
        self.assertEqual(raw_response.headers["content-type"], "image/png")

        crop_response = self.client.get(face["urls"]["crop"])
        self.assertEqual(crop_response.status_code, 200)
        self.assertEqual(crop_response.headers["content-type"], "image/webp")

    def test_user_memory_endpoints_return_task_scoped_structured_payloads(self) -> None:
        create_response = self.client.post("/api/tasks", json={"version": "v0327-exp"})
        self.assertEqual(create_response.status_code, 200)
        task_id = create_response.json()["task_id"]
        self.task_ids.append(task_id)

        self._append_uploaded_sample(task_id, color="orange")
        task_store.update_task(
            task_id,
            result=self._synthetic_v0327_memory_result(task_id),
            status="completed",
            stage="completed",
        )

        events_response = self.client.get(
            f"/api/users/{self.user_id}/memory/events",
            params={"task_id": task_id, "include_raw": "true"},
        )
        self.assertEqual(events_response.status_code, 200)
        event_payload = events_response.json()["tasks"][0]
        self.assertEqual(event_payload["events"][0]["event_id"], "EVT_001")
        self.assertEqual(event_payload["events"][0]["participant_person_ids"], ["Person_001", "Person_002"])
        self.assertEqual(len(event_payload["raw_events"]), 1)

        vlm_response = self.client.get(
            f"/api/users/{self.user_id}/memory/vlm",
            params={"task_id": task_id, "include_raw": "true"},
        )
        self.assertEqual(vlm_response.status_code, 200)
        vlm_payload = vlm_response.json()["tasks"][0]
        self.assertEqual(vlm_payload["vlm_observations"][0]["summary"], "Two friends enjoying a live concert.")
        self.assertIn("raw", vlm_payload["vlm_observations"][0])

        relationship_response = self.client.get(
            f"/api/users/{self.user_id}/memory/relationships",
            params={"task_id": task_id, "include_raw": "true"},
        )
        self.assertEqual(relationship_response.status_code, 200)
        relationship_payload = relationship_response.json()["tasks"][0]
        self.assertEqual(relationship_payload["relationships"][0]["person_id"], "Person_002")
        self.assertEqual(relationship_payload["relationships"][0]["shared_events"][0]["event_id"], "EVT_001")
        self.assertEqual(len(relationship_payload["relationship_dossiers"]), 1)

        profile_response = self.client.get(
            f"/api/users/{self.user_id}/memory/profiles",
            params={"task_id": task_id, "include_raw": "true"},
        )
        self.assertEqual(profile_response.status_code, 200)
        profile_payload = profile_response.json()["tasks"][0]
        self.assertEqual(profile_payload["profiles"][0]["primary_person_id"], "Person_001")
        self.assertEqual(
            profile_payload["profiles"][0]["structured"]["long_term_facts"]["identity"]["name"]["value"],
            "Vigar",
        )
        self.assertEqual(len(profile_payload["profiles"][0]["field_decisions"]), 1)

        photos_response = self.client.get(f"/api/users/{self.user_id}/memory/photos", params={"task_id": task_id})
        self.assertEqual(photos_response.status_code, 200)
        photos_payload = photos_response.json()["tasks"][0]
        self.assertEqual(len(photos_payload["photos"]), 1)
        self.assertNotIn("events", photos_payload)
        self.assertTrue(photos_payload["photos"][0]["urls"]["compressed"].startswith("/api/assets/photos/"))

    def test_user_memory_dataset_and_version_filters_follow_latest_and_all_rules(self) -> None:
        task_shared_old = self.client.post("/api/tasks", json={"version": "v0325"}).json()["task_id"]
        self.task_ids.append(task_shared_old)
        self._append_uploaded_sample(task_shared_old, source_hash="hash-shared", color="red")
        task_store.update_task(
            task_shared_old,
            result=self._synthetic_v0327_memory_result(task_shared_old),
            status="completed",
            stage="completed",
        )

        time.sleep(0.02)

        task_shared_new = self.client.post("/api/tasks", json={"version": "v0327-exp"}).json()["task_id"]
        self.task_ids.append(task_shared_new)
        self._append_uploaded_sample(task_shared_new, source_hash="hash-shared", color="blue")
        task_store.update_task(
            task_shared_new,
            result=self._synthetic_v0327_memory_result(task_shared_new),
            status="completed",
            stage="completed",
        )

        time.sleep(0.02)

        task_latest_dataset = self.client.post("/api/tasks", json={"version": "v0325"}).json()["task_id"]
        self.task_ids.append(task_latest_dataset)
        self._append_uploaded_sample(task_latest_dataset, source_hash="hash-latest", color="green")
        task_store.update_task(
            task_latest_dataset,
            result=self._synthetic_v0327_memory_result(task_latest_dataset),
            status="completed",
            stage="completed",
        )

        datasets_response = self.client.get(f"/api/users/{self.user_id}/datasets")
        self.assertEqual(datasets_response.status_code, 200)
        datasets = datasets_response.json()["datasets"]
        self.assertEqual(len(datasets), 2)
        latest_dataset_id = next(item["dataset_id"] for item in datasets if item["latest_task_id"] == task_latest_dataset)
        shared_dataset_id = next(item["dataset_id"] for item in datasets if item["latest_task_id"] == task_shared_new)

        default_bundle = self.client.get(f"/api/users/{self.user_id}/memory/bundle")
        self.assertEqual(default_bundle.status_code, 200)
        self.assertEqual(default_bundle.json()["tasks"][0]["task_id"], task_latest_dataset)

        shared_all = self.client.get(
            f"/api/users/{self.user_id}/memory/bundle",
            params={"dataset_id": shared_dataset_id, "all": "true"},
        )
        self.assertEqual(shared_all.status_code, 200)
        shared_task_ids = [item["task_id"] for item in shared_all.json()["tasks"]]
        self.assertEqual(shared_task_ids, [task_shared_new, task_shared_old])

        filtered_tasks = self.client.get(
            f"/api/users/{self.user_id}/tasks",
            params={"pipeline_version": 327, "pipeline_channel": "exp", "scope": "user"},
        )
        self.assertEqual(filtered_tasks.status_code, 200)
        self.assertEqual(filtered_tasks.json()["tasks"][0]["task_id"], task_shared_new)

        updated_after = datetime.now().isoformat()
        time.sleep(0.02)

        task_user_scope_new = self.client.post("/api/tasks", json={"version": "v0327-exp"}).json()["task_id"]
        self.task_ids.append(task_user_scope_new)
        self._append_uploaded_sample(task_user_scope_new, source_hash="hash-user-scope", color="yellow")
        task_store.update_task(
            task_user_scope_new,
            result=self._synthetic_v0327_memory_result(task_user_scope_new),
            status="completed",
            stage="completed",
        )

        updated_tasks = self.client.get(
            f"/api/users/{self.user_id}/tasks",
            params={"scope": "user", "all": "true", "updated_after": updated_after},
        )
        self.assertEqual(updated_tasks.status_code, 200)
        self.assertEqual([item["task_id"] for item in updated_tasks.json()["tasks"]], [task_user_scope_new])

        versions_response = self.client.get(f"/api/users/{self.user_id}/versions")
        self.assertEqual(versions_response.status_code, 200)
        versions_payload = versions_response.json()
        self.assertEqual(versions_payload["pipeline_versions"], [325, 327])
        self.assertEqual(versions_payload["pipeline_channels"], ["exp"])
        self.assertIn(latest_dataset_id, [item["dataset_id"] for item in datasets])

    def test_user_memory_tasks_reject_invalid_updated_after(self) -> None:
        response = self.client.get(
            f"/api/users/{self.user_id}/tasks",
            params={"updated_after": "not-a-time"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("updated_after", response.json()["detail"])

    def test_delete_task_removes_db_backed_memory_snapshot(self) -> None:
        create_response = self.client.post("/api/tasks", json={"version": "v0327-exp"})
        self.assertEqual(create_response.status_code, 200)
        task_id = create_response.json()["task_id"]
        self.task_ids.append(task_id)

        self._append_uploaded_sample(task_id, color="black")
        task_store.update_task(
            task_id,
            result=self._synthetic_v0327_memory_result(task_id),
            status="completed",
            stage="completed",
        )

        warm_response = self.client.get(f"/api/users/{self.user_id}/memory/bundle", params={"task_id": task_id})
        self.assertEqual(warm_response.status_code, 200)
        self.assertEqual(len(warm_response.json()["tasks"]), 1)

        delete_response = self.client.delete(f"/api/tasks/{task_id}")
        self.assertEqual(delete_response.status_code, 200)
        self.assertEqual(delete_response.json()["status"], "deleted")

        lookup_response = self.client.get(f"/api/users/{self.user_id}/memory/bundle", params={"task_id": task_id})
        self.assertEqual(lookup_response.status_code, 200)
        self.assertEqual(lookup_response.json()["tasks"], [])

    def _image_bytes(self, color: str) -> bytes:
        image = Image.new("RGB", (48, 48), color=color)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _append_uploaded_sample(
        self,
        task_id: str,
        *,
        image_id: str = "photo_001",
        filename: str = "sample.png",
        stored_filename: str = "001_sample.png",
        source_hash: str = "hash-001",
        color: str = "red",
    ) -> None:
        uploads_dir = task_store.task_dir(task_id) / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        local_path = uploads_dir / stored_filename
        Image.new("RGB", (64, 64), color=color).save(local_path, format="PNG")
        task_store.append_uploads(
            task_id,
            [
                {
                    "image_id": image_id,
                    "filename": filename,
                    "stored_filename": stored_filename,
                    "path": f"uploads/{stored_filename}",
                    "url": f"/api/assets/{task_id}/uploads/{stored_filename}",
                    "preview_url": None,
                    "content_type": "image/png",
                    "width": 64,
                    "height": 64,
                    "source_hash": source_hash,
                    "timestamp": "2026-03-20T20:00:00",
                }
            ],
            status="completed",
            stage="completed",
        )

    def _livp_bytes(self, color: str) -> bytes:
        image_payload = self._image_bytes(color)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, mode="w") as archive:
            archive.writestr("cover.jpg", image_payload)
            archive.writestr("motion.mp4", b"fake-video")
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
            "facts": [],
            "memory": {
                "envelope": {"scope": {"user_id": self.user_id}},
                "storage": {
                    "neo4j": {
                        "nodes": {
                            "user": [{"user_id": self.user_id, "labels": ["User"], "properties": {"profile_version": 1}}],
                            "persons": [],
                            "places": [],
                            "events": [
                                {
                                    "session_uuid": "session_001",
                                    "labels": ["Event"],
                                    "properties": {
                                        "user_id": self.user_id,
                                        "event_id": "event_001",
                                        "started_at": "2026-02-15T20:00:00",
                                        "ended_at": "2026-02-15T22:00:00",
                                        "participant_count": 2,
                                        "representative_photo_ids": ["photo_001"],
                                    },
                                }
                            ],
                            "facts": [
                                {
                                    "event_uuid": "event_001",
                                    "labels": ["Fact"],
                                    "properties": {
                                        "user_id": self.user_id,
                                        "fact_id": "EVT_001",
                                        "title": "Live Concert Night",
                                        "coarse_event_type": "concert",
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
                                "edge_type": "DERIVED_FROM_EVENT",
                                "properties": {},
                            },
                        ],
                    },
                    "milvus": {"segments": []},
                    "redis": {},
                },
            },
        }

    def _synthetic_revision_first_result(self, task_id: str) -> dict:
        timestamp = datetime.utcnow().isoformat()
        return {
            "task_id": task_id,
            "generated_at": timestamp,
            "summary": {
                "primary_person_id": "Person_001",
            },
            "face_recognition": {
                "primary_person_id": "Person_001",
                "images": [
                    {
                        "image_id": "photo_001",
                        "filename": "sample.png",
                        "source_hash": "hash-001",
                        "timestamp": "2026-03-20T20:00:00",
                        "original_image_url": "/assets/uploads/001_sample.png",
                        "display_image_url": "/assets/uploads/001_sample.png",
                        "boxed_image_url": None,
                        "compressed_image_url": None,
                        "location": None,
                        "faces": [],
                    }
                ],
                "person_groups": [
                    {
                        "person_id": "Person_001",
                        "is_primary": True,
                        "photo_count": 1,
                        "images": [
                            {
                                "image_id": "photo_001",
                                "filename": "sample.png",
                                "timestamp": "2026-03-20T20:00:00",
                                "source_hash": "hash-001",
                            }
                        ],
                    }
                ],
            },
            "memory": {
                "pipeline_family": "v0321_3",
                "event_revisions": [
                    {
                        "event_root_id": "event_root_001",
                        "event_revision_id": "event_rev_001",
                        "revision": 1,
                        "title": "Concert Night",
                        "event_summary": "Live concert with a close friend.",
                        "started_at": "2026-03-20T20:00:00",
                        "ended_at": "2026-03-20T22:00:00",
                        "participant_person_ids": ["Person_001", "Person_002"],
                        "depicted_person_ids": ["Person_001", "Person_002"],
                        "place_refs": ["Shanghai"],
                        "original_photo_ids": ["hash-001"],
                        "confidence": 0.9,
                        "status": "active",
                        "sealed_state": "sealed",
                        "atomic_evidence": [
                            {
                                "evidence_id": "evidence_001",
                                "root_event_revision_id": "event_rev_001",
                                "evidence_type": "brand",
                                "value_or_text": "MOET",
                                "provenance": "brand",
                                "original_photo_ids": ["hash-001"],
                                "confidence": 0.8,
                            }
                        ],
                    }
                ],
                "atomic_evidence": [
                    {
                        "evidence_id": "evidence_001",
                        "root_event_revision_id": "event_rev_001",
                        "evidence_type": "brand",
                        "value_or_text": "MOET",
                        "provenance": "brand",
                        "original_photo_ids": ["hash-001"],
                        "confidence": 0.8,
                    }
                ],
                "relationship_revisions": [
                    {
                        "relationship_root_id": "rel_root_001",
                        "relationship_revision_id": "rel_rev_001",
                        "target_person_id": "Person_002",
                        "relationship_type": "friend",
                        "label": "close friend",
                        "confidence": 0.85,
                        "supporting_event_ids": ["event_rev_001"],
                    }
                ],
                "profile_revision": {
                    "profile_revision_id": "profile_rev_001",
                    "primary_person_id": "Person_001",
                    "scope": "cumulative",
                    "generation_mode": "profile_input_pack_llm",
                    "original_photo_ids": ["hash-001"],
                },
                "profile_markdown": "# Profile\n\nConcert-heavy social life.",
                "delta_event_revisions": [
                    {
                        "event_root_id": "event_root_001",
                        "event_revision_id": "event_rev_001",
                        "revision": 1,
                        "title": "Concert Night",
                        "event_summary": "Live concert with a close friend.",
                        "started_at": "2026-03-20T20:00:00",
                        "ended_at": "2026-03-20T22:00:00",
                        "participant_person_ids": ["Person_001", "Person_002"],
                        "depicted_person_ids": ["Person_001", "Person_002"],
                        "place_refs": ["Shanghai"],
                        "original_photo_ids": ["hash-001"],
                        "confidence": 0.9,
                        "status": "active",
                        "sealed_state": "sealed",
                        "atomic_evidence": [
                            {
                                "evidence_id": "evidence_001",
                                "root_event_revision_id": "event_rev_001",
                                "evidence_type": "brand",
                                "value_or_text": "MOET",
                                "provenance": "brand",
                                "original_photo_ids": ["hash-001"],
                                "confidence": 0.8,
                            }
                        ],
                    }
                ],
                "delta_atomic_evidence": [
                    {
                        "evidence_id": "evidence_001",
                        "root_event_revision_id": "event_rev_001",
                        "evidence_type": "brand",
                        "value_or_text": "MOET",
                        "provenance": "brand",
                        "original_photo_ids": ["hash-001"],
                        "confidence": 0.8,
                    }
                ],
                "delta_relationship_revisions": [
                    {
                        "relationship_root_id": "rel_root_001",
                        "relationship_revision_id": "rel_rev_001",
                        "target_person_id": "Person_002",
                        "relationship_type": "friend",
                        "label": "close friend",
                        "confidence": 0.85,
                        "supporting_event_ids": ["event_rev_001"],
                    }
                ],
                "delta_profile_revision": {
                    "profile_revision_id": "profile_rev_001",
                    "primary_person_id": "Person_001",
                    "scope": "delta",
                    "generation_mode": "profile_input_pack_llm",
                    "original_photo_ids": ["hash-001"],
                },
                "delta_profile_markdown": "# Profile\n\nConcert-heavy social life.",
                "profile_input_pack": {
                    "profile_input_pack_id": "profile_pack_001",
                    "time_range": {"start": "2026-03-20T20:00:00", "end": "2026-03-20T22:00:00"},
                    "baseline_rhythm": {"dominant_activity_window": "evening"},
                    "place_patterns": {"top_place_refs": [{"place_ref": "Shanghai", "count": 1}]},
                    "activity_patterns": {"top_activities": [{"activity_type": "concert", "count": 1}]},
                    "identity_signals": {
                        "role_hints": [
                            {
                                "label": "夜间社交活跃",
                                "confidence": 0.6,
                                "evidence_level": "event_grounded",
                                "supporting_event_ids": ["event_rev_001"],
                                "supporting_signal_ids": [],
                            }
                        ]
                    },
                    "lifestyle_consumption_signals": {
                        "diet_hints": [],
                    },
                    "event_grounded_signals": {
                        "interest_signals": [
                            {"label": "concert", "count": 1, "supporting_event_ids": ["event_rev_001"]}
                        ]
                    },
                    "reference_media_weak_signals": {},
                    "social_patterns": {
                        "top_relationships": [
                            {
                                "relationship_revision_id": "rel_rev_001",
                                "target_person_id": "Person_002",
                                "relationship_type": "friend",
                                "confidence": 0.85,
                            }
                        ],
                        "relationship_summary": {"close_relationship_count": 1},
                        "social_style_hints": {"one_on_one_bias": 0.7},
                    },
                    "change_points": [],
                    "key_event_refs": [{"event_revision_id": "event_rev_001", "title": "Concert Night"}],
                    "key_relationship_refs": [{"relationship_revision_id": "rel_rev_001"}],
                    "evidence_guardrails": {"forbidden_direct_inference_from_reference_media": ["真实到访"]},
                },
                "delta_profile_input_pack": {
                    "profile_input_pack_id": "profile_pack_001_delta",
                    "time_range": {"start": "2026-03-20T20:00:00", "end": "2026-03-20T22:00:00"},
                    "baseline_rhythm": {"dominant_activity_window": "evening"},
                    "place_patterns": {"top_place_refs": [{"place_ref": "Shanghai", "count": 1}]},
                    "activity_patterns": {"top_activities": [{"activity_type": "concert", "count": 1}]},
                    "identity_signals": {
                        "role_hints": [
                            {
                                "label": "夜间社交活跃",
                                "confidence": 0.6,
                                "evidence_level": "event_grounded",
                                "supporting_event_ids": ["event_rev_001"],
                                "supporting_signal_ids": [],
                            }
                        ]
                    },
                    "lifestyle_consumption_signals": {"diet_hints": []},
                    "event_grounded_signals": {
                        "interest_signals": [
                            {"label": "concert", "count": 1, "supporting_event_ids": ["event_rev_001"]}
                        ]
                    },
                    "reference_media_weak_signals": {},
                    "social_patterns": {
                        "top_relationships": [
                            {
                                "relationship_revision_id": "rel_rev_001",
                                "target_person_id": "Person_002",
                                "relationship_type": "friend",
                                "confidence": 0.85,
                            }
                        ],
                        "relationship_summary": {"close_relationship_count": 1},
                        "social_style_hints": {"one_on_one_bias": 0.7},
                    },
                    "change_points": [],
                    "key_event_refs": [{"event_revision_id": "event_rev_001", "title": "Concert Night"}],
                    "key_relationship_refs": [{"relationship_revision_id": "rel_rev_001"}],
                    "evidence_guardrails": {"forbidden_direct_inference_from_reference_media": ["真实到访"]},
                },
                "vlm_observations": [
                    {
                        "photo_id": "photo_001",
                        "original_photo_ids": ["hash-001"],
                        "summary": "A concert scene.",
                        "scene_hint": "indoor venue",
                        "activity_hint": "concert",
                        "social_hint": "with friend",
                        "ocr_hits": [],
                        "brands": ["MOET"],
                        "place_candidates": ["Shanghai"],
                        "object_clues": ["stage"],
                        "embedded_media_signals": [],
                        "person_ids": ["Person_001"],
                    }
                ],
                "delta_profile_truth_v1": {
                    "profile_truth_id": "profile_rev_001:truth",
                    "profile_revision_id": "profile_rev_001",
                    "original_photo_ids": ["hash-001"],
                },
                "profile_truth_v1": {
                    "profile_truth_id": "profile_rev_001:truth",
                    "profile_revision_id": "profile_rev_001",
                    "original_photo_ids": ["hash-001"],
                },
                "envelope": {
                    "scope": {"user_id": self.user_id},
                },
            },
        }

    def _synthetic_v0327_memory_result(self, task_id: str) -> dict:
        raw_upload_url = f"/api/assets/{task_id}/uploads/001_sample.png"
        return {
            "task_id": task_id,
            "generated_at": "2026-03-20T20:00:00",
            "summary": {
                "primary_person_id": "Person_001",
            },
            "face_recognition": {
                "primary_person_id": "Person_001",
                "images": [
                    {
                        "image_id": "photo_001",
                        "filename": "sample.png",
                        "source_hash": "hash-001",
                        "timestamp": "2026-03-20T20:00:00",
                        "original_image_url": raw_upload_url,
                        "display_image_url": raw_upload_url,
                        "boxed_image_url": raw_upload_url,
                        "compressed_image_url": raw_upload_url,
                        "location": {"city": "Shanghai"},
                        "width": 64,
                        "height": 64,
                        "faces": [
                            {
                                "face_id": "face_001",
                                "person_id": "Person_001",
                                "score": 0.95,
                                "similarity": 0.92,
                                "faiss_id": 1,
                                "bbox": [0, 0, 32, 32],
                                "bbox_xywh": {"x": 0, "y": 0, "w": 32, "h": 32},
                                "quality_score": 0.88,
                                "match_decision": "strong_match",
                                "match_reason": "primary",
                            },
                            {
                                "face_id": "face_002",
                                "person_id": "Person_002",
                                "score": 0.91,
                                "similarity": 0.89,
                                "faiss_id": 2,
                                "bbox": [32, 0, 32, 32],
                                "bbox_xywh": {"x": 32, "y": 0, "w": 32, "h": 32},
                                "quality_score": 0.84,
                                "match_decision": "strong_match",
                                "match_reason": "friend",
                            },
                        ],
                    }
                ],
                "person_groups": [
                    {
                        "person_id": "Person_001",
                        "is_primary": True,
                        "photo_count": 1,
                        "face_count": 1,
                        "avg_score": 0.95,
                        "avg_quality": 0.88,
                        "high_quality_face_count": 1,
                        "images": [{"image_id": "photo_001", "filename": "sample.png"}],
                    },
                    {
                        "person_id": "Person_002",
                        "is_primary": False,
                        "photo_count": 1,
                        "face_count": 1,
                        "avg_score": 0.91,
                        "avg_quality": 0.84,
                        "high_quality_face_count": 1,
                        "images": [{"image_id": "photo_001", "filename": "sample.png"}],
                    },
                ],
            },
            "memory": {
                "vp1_observations": [
                    {
                        "photo_id": "photo_001",
                        "filename": "sample.png",
                        "timestamp": "2026-03-20T20:00:00",
                        "vlm_analysis": {
                            "summary": "Two friends enjoying a live concert.",
                            "people": [
                                {"person_id": "Person_001", "name": "primary", "confidence": 0.93},
                                {"person_id": "Person_002", "name": "friend", "confidence": 0.88},
                            ],
                            "relations": [
                                {
                                    "source_person_id": "Person_001",
                                    "target_person_id": "Person_002",
                                    "relationship": "friend",
                                    "confidence": 0.81,
                                }
                            ],
                            "scene": {"location": "Shanghai", "environment": "indoor concert venue"},
                            "event": {"activity": "concert"},
                            "details": [{"text": "Holding drinks near the stage"}],
                            "key_objects": ["stage", "drink"],
                            "ocr_hits": ["LIVE"],
                            "brands": ["MOET"],
                            "place_candidates": ["Shanghai"],
                        },
                    }
                ],
                "lp1_events": [
                    {
                        "event_id": "EVT_001",
                        "title": "Concert Night",
                        "date": "2026-03-20",
                        "time_range": {"start": "20:00:00", "end": "22:00:00"},
                        "duration": "2h",
                        "type": "concert",
                        "location": {"city": "Shanghai"},
                        "description": "An evening concert with a close friend.",
                        "photo_count": 1,
                        "confidence": 0.9,
                        "reason": "Time and scene aligned across the dataset.",
                        "narrative": "They attended a concert together.",
                        "narrative_synthesis": "A night out at a live concert with a close friend.",
                        "participants": ["Person_001", "Person_002"],
                        "photo_ids": ["photo_001"],
                        "tags": ["music", "nightlife"],
                        "social_dynamics": {"mood": "close"},
                        "persona_evidence": [{"signal": "social"}],
                    }
                ],
                "lp1_events_raw": [
                    {
                        "source_temp_event_id": "tmp_evt_001",
                        "supporting_photo_ids": ["photo_001"],
                    }
                ],
                "lp2_relationships": [
                    {
                        "relationship_id": "REL_001",
                        "person_id": "Person_002",
                        "relationship_type": "close_friend",
                        "intimacy_score": 0.83,
                        "status": "active",
                        "confidence": 0.87,
                        "reasoning": "Repeated one-on-one nightlife events.",
                        "evidence": {"photo_ids": ["photo_001"], "event_ids": ["EVT_001"]},
                        "shared_events": [
                            {
                                "event_id": "EVT_001",
                                "date": "2026-03-20",
                                "narrative": "Concert Night",
                            }
                        ],
                    }
                ],
                "lp3_profile": {
                    "primary_person_id": "Person_001",
                    "structured": {
                        "long_term_facts": {
                            "identity": {
                                "name": {
                                    "value": "Vigar",
                                    "confidence": 0.88,
                                    "reasoning": "Repeated naming signal in profile synthesis.",
                                    "evidence": {"photo_ids": ["photo_001"]},
                                }
                            }
                        }
                    },
                    "report": "# Profile\n\nConcert-heavy social life.",
                    "summary": "Concert-heavy social life.",
                    "consistency": {"summary": {"status": "ok"}},
                    "debug": {"trace": ["lp3"]},
                    "internal_artifacts": {
                        "primary_decision": {"primary_person_id": "Person_001"},
                        "relationship_dossiers": [
                            {
                                "person_id": "Person_002",
                                "memory_value": "keep",
                                "photo_count": 1,
                            }
                        ],
                        "group_artifacts": [
                            {
                                "group_id": "GROUP_001",
                                "members": ["Person_001", "Person_002"],
                                "group_type_candidate": "friends",
                                "confidence": 0.8,
                                "reason": "Shared nightlife event",
                            }
                        ],
                        "profile_fact_decisions": [
                            {
                                "field_key": "long_term_facts.identity.name",
                                "draft": {
                                    "value": "Vigar",
                                    "confidence": 0.8,
                                    "reasoning": "draft reasoning",
                                    "evidence": {"photo_ids": ["photo_001"]},
                                },
                                "final": {
                                    "value": "Vigar",
                                    "confidence": 0.88,
                                    "reasoning": "final reasoning",
                                    "evidence": {"photo_ids": ["photo_001"]},
                                },
                            }
                        ],
                    },
                },
            },
        }


if __name__ == "__main__":
    unittest.main()
