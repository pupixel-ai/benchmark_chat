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
from sqlalchemy import delete, select

from backend.app import _run_pipeline_task, _task_upload_lock, app, task_store
from backend.db import SessionLocal, ensure_schema
from backend.task_event_outbox import build_terminal_dedupe_key
from backend.task_store import normalize_task_options
from config import (
    APP_VERSION,
    AVAILABLE_TASK_VERSIONS,
    DEFAULT_NORMALIZE_LIVE_PHOTOS,
    DEFAULT_TASK_VERSION,
    KAFKA_SURVEY_IMPORT_TOPIC,
    KAFKA_TERMINAL_TOPIC,
)
from backend.models import (
    ArtifactRecord,
    FaceRecognitionImagePolicyRecord,
    FaceReviewRecord,
    SessionRecord,
    TaskEventOutboxRecord,
    TaskRecord,
    SubjectUserRecord,
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
        self._original_outbox_enabled = task_store.outbox_store.enabled

    def tearDown(self) -> None:
        for task_id in self.task_ids:
            shutil.rmtree(task_store.task_dir(task_id), ignore_errors=True)

        task_store.outbox_store.enabled = self._original_outbox_enabled
        with SessionLocal() as session:
            session.execute(delete(ArtifactRecord).where(ArtifactRecord.user_id == self.user_id))
            session.execute(delete(FaceReviewRecord).where(FaceReviewRecord.user_id == self.user_id))
            session.execute(
                delete(FaceRecognitionImagePolicyRecord).where(
                    FaceRecognitionImagePolicyRecord.user_id == self.user_id
                )
            )
            if self.task_ids:
                session.execute(delete(TaskEventOutboxRecord).where(TaskEventOutboxRecord.task_id.in_(self.task_ids)))
            session.execute(delete(SessionRecord).where(SessionRecord.user_id == self.user_id))
            session.execute(delete(TaskRecord).where(TaskRecord.operator_user_id == self.user_id))
            session.execute(delete(TaskRecord).where(TaskRecord.user_id == self.user_id))
            session.execute(delete(SubjectUserRecord))
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
                "survey_username": self.username,
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
                "survey_username": self.username,
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

    def test_create_task_defaults_v0327_db_query_subject_to_current_user(self) -> None:
        create_response = self.client.post("/api/tasks", json={"version": "v0327-db-query"})
        self.assertEqual(create_response.status_code, 200)
        payload = create_response.json()
        self.task_ids.append(payload["task_id"])
        self.assertEqual(payload["user_id"], self.user_id)
        self.assertEqual(payload["operator_user_id"], self.user_id)
        self.assertEqual(payload["options"]["survey_username"], self.username)
        self.assertNotIn("subject_user_id", payload["options"])
        self.assertNotIn("operator_user_id", payload["options"])

        task = task_store.get_task(payload["task_id"], user_id=self.user_id)
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task["user_id"], self.user_id)
        self.assertEqual(task["operator_user_id"], self.user_id)
        self.assertEqual(task["options"]["survey_username"], self.username)
        self.assertNotIn("subject_user_id", task["options"])
        self.assertNotIn("operator_user_id", task["options"])

    def test_create_task_maps_v0327_db_query_to_subject_user_and_operator(self) -> None:
        create_response = self.client.post(
            "/api/tasks",
            json={
                "version": "v0327-db-query",
                "creation_source": "directory",
                "survey_username": "alice",
                "expected_upload_count": 3,
                "requested_max_photos": 3,
            },
        )
        self.assertEqual(create_response.status_code, 200)
        payload = create_response.json()
        self.task_ids.append(payload["task_id"])
        self.assertEqual(payload["options"]["creation_source"], "directory")
        self.assertEqual(payload["options"]["survey_username"], "alice")
        self.assertNotEqual(payload["user_id"], self.user_id)
        self.assertEqual(payload["operator_user_id"], self.user_id)

        task = task_store.get_task(payload["task_id"], user_id=self.user_id)
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task["user_id"], payload["user_id"])
        self.assertEqual(task["operator_user_id"], self.user_id)
        self.assertEqual(task["options"]["survey_username"], "alice")
        self.assertEqual(task["options"]["creation_source"], "directory")
        self.assertNotIn("subject_user_id", task["options"])
        self.assertNotIn("operator_user_id", task["options"])

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

    def test_subject_scoped_task_listing_uses_photo_owner_user_id(self) -> None:
        create_response = self.client.post(
            "/api/tasks",
            json={
                "version": "v0327-db-query",
                "creation_source": "directory",
                "survey_username": "alice",
            },
        )
        self.assertEqual(create_response.status_code, 200)
        payload = create_response.json()
        self.task_ids.append(payload["task_id"])

        response = self.client.get(f"/api/users/{payload['user_id']}/tasks")
        self.assertEqual(response.status_code, 200)
        task_ids = [item["task_id"] for item in response.json()["tasks"]]
        self.assertIn(payload["task_id"], task_ids)

    def test_create_task_accepts_explicit_user_id_for_admin_upload(self) -> None:
        owner_user_id = "subject_jennie_demo"
        survey_username = "jennie"
        create_response = self.client.post(
            "/api/tasks",
            json={
                "version": "v0327-db-query",
                "user_id": owner_user_id,
                "creation_source": "admin_console",
                "survey_username": survey_username,
            },
        )
        self.assertEqual(create_response.status_code, 200)
        payload = create_response.json()
        self.task_ids.append(payload["task_id"])
        self.assertEqual(payload["user_id"], owner_user_id)
        self.assertEqual(payload["operator_user_id"], self.user_id)
        self.assertEqual(payload["options"]["survey_username"], survey_username)

        task = task_store.get_task(payload["task_id"], user_id=self.user_id)
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task["user_id"], owner_user_id)
        self.assertEqual(task["operator_user_id"], self.user_id)
        self.assertEqual(task["options"]["survey_username"], survey_username)
        self.assertNotIn("subject_user_id", task["options"])
        self.assertNotIn("operator_user_id", task["options"])

        response = self.client.get(f"/api/users/{owner_user_id}/tasks")
        self.assertEqual(response.status_code, 200)
        task_ids = [item["task_id"] for item in response.json()["tasks"]]
        self.assertIn(payload["task_id"], task_ids)

    def test_ensure_schema_backfills_operator_user_id_from_legacy_user_id(self) -> None:
        task_id = uuid.uuid4().hex
        self.task_ids.append(task_id)
        task_dir = task_store.task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()

        with SessionLocal() as session:
            session.add(
                TaskRecord(
                    task_id=task_id,
                    user_id=self.user_id,
                    operator_user_id=None,
                    version="v0327-db-query",
                    status="draft",
                    stage="draft",
                    upload_count=0,
                    task_dir=str(task_dir),
                    progress=None,
                    uploads=None,
                    options={"survey_username": self.username},
                    result=None,
                    result_summary=None,
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

        ensure_schema()

        migrated = task_store.get_task(task_id, user_id=self.user_id)
        self.assertIsNotNone(migrated)
        assert migrated is not None
        self.assertEqual(migrated["user_id"], self.user_id)
        self.assertEqual(migrated["operator_user_id"], self.user_id)
        self.assertNotIn("subject_user_id", migrated["options"])

    def test_ensure_schema_promotes_legacy_subject_user_id_without_losing_operator(self) -> None:
        task_id = uuid.uuid4().hex
        self.task_ids.append(task_id)
        task_dir = task_store.task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        owner_user_id = f"subject_{uuid.uuid4().hex[:8]}"

        with SessionLocal() as session:
            session.add(
                TaskRecord(
                    task_id=task_id,
                    user_id=self.user_id,
                    operator_user_id=None,
                    version="v0327-db-query",
                    status="draft",
                    stage="draft",
                    upload_count=0,
                    task_dir=str(task_dir),
                    progress=None,
                    uploads=None,
                    options={"survey_username": "alice", "subject_user_id": owner_user_id},
                    result=None,
                    result_summary=None,
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

        ensure_schema()

        migrated = task_store.get_task(task_id, user_id=self.user_id)
        self.assertIsNotNone(migrated)
        assert migrated is not None
        self.assertEqual(migrated["user_id"], owner_user_id)
        self.assertEqual(migrated["operator_user_id"], self.user_id)
        self.assertNotIn("subject_user_id", migrated["options"])
        self.assertEqual(migrated["options"]["survey_username"], "alice")

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
                "survey_username": self.username,
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

    def test_memory_steps_endpoint_accepts_v0327_db_query_tasks(self) -> None:
        create_response = self.client.post(
            "/api/tasks",
            json={"version": "v0327-db-query", "creation_source": "directory", "survey_username": "alice"},
        )
        self.assertEqual(create_response.status_code, 200)
        task_id = create_response.json()["task_id"]
        self.task_ids.append(task_id)

        family_dir = task_store.task_dir(task_id) / "v0325"
        family_dir.mkdir(parents=True, exist_ok=True)
        (family_dir / "lp1_events_compact.json").write_text(
            json.dumps([{"event_id": "EVT_0001", "title": "Query Event"}], ensure_ascii=False),
            encoding="utf-8",
        )
        (family_dir / "lp2_relationships.json").write_text("[]", encoding="utf-8")
        (family_dir / "lp3_profile.json").write_text("{}", encoding="utf-8")

        task_store.update_task(
            task_id,
            result={
                "memory": {
                    "pipeline_family": "v0325",
                    "lp1_events": [],
                    "lp2_relationships": [],
                    "lp3_profile": {},
                }
            },
            status="completed",
            stage="completed",
        )

        response = self.client.get(f"/api/tasks/{task_id}/memory/steps")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["pipeline_family"], "v0325")

    def test_local_pipeline_terminal_finalize_enqueues_outbox_for_v0327_db_query(self) -> None:
        task_store.outbox_store.enabled = True
        create_response = self.client.post(
            "/api/tasks",
            json={"version": "v0327-db-query", "creation_source": "directory", "survey_username": "alice"},
        )
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
                    "width": 48,
                    "height": 48,
                    "source_hash": "hash-001",
                }
            ],
            status="draft",
            stage="draft",
        )

        class _FakePipelineService:
            def __init__(self, **_: object) -> None:
                pass

            def run(self, *, max_photos: int, use_cache: bool, progress_callback):
                del max_photos, use_cache
                progress_callback("memory", {"substage": "lp1"})
                return self_result

        self_result = self._synthetic_result(task_id)
        with patch("backend.app.MemoryPipelineService", _FakePipelineService), patch(
            "backend.app.build_task_asset_manifest",
            return_value={"artifact_count": 1, "files": [], "named_urls": {"report_url": "/asset/report.json"}},
        ):
            _run_pipeline_task(
                task_id,
                self.user_id,
                1,
                False,
                "v0327-db-query",
                normalize_task_options({"normalize_live_photos": True}),
            )

        task = task_store.get_task(task_id, user_id=self.user_id)
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task["status"], "completed")
        self.assertEqual(task["result_summary"]["primary_person_id"], "person_001")
        self.assertIsNone(task["result"])

        outbox = task_store.get_outbox_record(
            build_terminal_dedupe_key(task_id, "task.completed")
        )
        self.assertIsNotNone(outbox)
        assert outbox is not None
        self.assertEqual(outbox["payload_json"]["task_version"], "v0327-db-query")
        self.assertEqual(outbox["payload_json"]["event_type"], "task.completed")

    def test_internal_worker_terminal_callback_is_idempotent_for_v0327_db_query(self) -> None:
        task_store.outbox_store.enabled = True
        create_response = self.client.post(
            "/api/tasks",
            json={"version": "v0327-db-query", "creation_source": "directory", "survey_username": "alice"},
        )
        self.assertEqual(create_response.status_code, 200)
        task_id = create_response.json()["task_id"]
        self.task_ids.append(task_id)

        payload = {
            "status": "completed",
            "stage": "completed",
            "result": self._synthetic_result(task_id),
            "result_summary": {"primary_person_id": "person_001"},
            "asset_manifest": {"artifact_count": 0, "files": [], "named_urls": {}},
            "version": "v0327-db-query",
        }

        with patch("backend.app.WORKER_SHARED_TOKEN", "worker-secret"):
            first = self.client.post(
                f"/internal/tasks/{task_id}/terminal-update",
                json=payload,
                headers={"Authorization": "Bearer worker-secret"},
            )
            second = self.client.post(
                f"/internal/tasks/{task_id}/terminal-update",
                json=payload,
                headers={"Authorization": "Bearer worker-secret"},
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        with SessionLocal() as session:
            rows = session.execute(
                select(TaskEventOutboxRecord).where(TaskEventOutboxRecord.task_id == task_id)
            ).scalars().all()
        self.assertEqual(len(rows), 1)

    def test_upload_batches_preserve_survey_username_for_directory_tasks(self) -> None:
        create_response = self.client.post(
            "/api/tasks",
            json={
                "version": "v0327-db-query",
                "creation_source": "directory",
                "survey_username": "alice",
                "expected_upload_count": 2,
                "requested_max_photos": 2,
            },
        )
        self.assertEqual(create_response.status_code, 200)
        task_id = create_response.json()["task_id"]
        self.task_ids.append(task_id)

        batch_response = self.client.post(
            f"/api/tasks/{task_id}/upload-batches",
            data={
                "creation_source": "directory",
                "survey_username": "alice",
                "expected_upload_count": "2",
                "requested_max_photos": "2",
                "auto_start_on_upload_complete": "false",
            },
            files=[
                ("files", ("one.png", self._image_bytes("red"), "image/png")),
            ],
        )
        self.assertEqual(batch_response.status_code, 200)

        task = task_store.get_task(task_id, user_id=self.user_id)
        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task["options"]["creation_source"], "directory")
        self.assertEqual(task["options"]["survey_username"], "alice")

    def test_internal_worker_terminal_callback_enqueues_full_and_survey_events_for_v0327_db_query(self) -> None:
        task_store.outbox_store.enabled = True
        create_response = self.client.post(
            "/api/tasks",
            json={
                "version": "v0327-db-query",
                "creation_source": "directory",
                "survey_username": "alice",
            },
        )
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
            status="draft",
            stage="draft",
        )

        result = self._synthetic_revision_first_result(task_id)
        result["face_recognition"]["images"][0]["boxed_image_url"] = "https://cdn.example.com/photo_001_boxed.webp"
        result["memory"]["delta_relationship_revisions"][0]["supporting_photo_ids"] = ["hash-001"]
        result["memory"]["relationship_revisions"][0]["supporting_photo_ids"] = ["hash-001"]
        result["memory"]["lp3_profile"] = {
            "report_markdown": "# Profile\n\nConcert-heavy social life.",
            "structured": {
                "long_term_facts": {
                    "identity": {
                        "name": {
                            "value": "Alice",
                            "confidence": 0.91,
                        }
                    }
                }
            },
        }

        payload = {
            "status": "completed",
            "stage": "completed",
            "result": result,
            "result_summary": {"primary_person_id": "Person_001"},
            "asset_manifest": {"artifact_count": 0, "files": [], "named_urls": {}},
            "version": "v0327-db-query",
            "options": {
                "creation_source": "directory",
                "survey_username": "alice",
                "normalize_live_photos": True,
            },
        }

        with patch("backend.app.WORKER_SHARED_TOKEN", "worker-secret"):
            response = self.client.post(
                f"/internal/tasks/{task_id}/terminal-update",
                json=payload,
                headers={"Authorization": "Bearer worker-secret"},
            )

        self.assertEqual(response.status_code, 200)
        with SessionLocal() as session:
            rows = session.execute(
                select(TaskEventOutboxRecord).where(TaskEventOutboxRecord.task_id == task_id)
            ).scalars().all()

        self.assertEqual(len(rows), 2)
        rows_by_type = {row.event_type: row for row in rows}
        self.assertEqual(rows_by_type["task.completed"].topic, KAFKA_TERMINAL_TOPIC)
        self.assertEqual(rows_by_type["survey.import.ready"].topic, KAFKA_SURVEY_IMPORT_TOPIC)
        self.assertEqual(rows_by_type["survey.import.ready"].payload_json["username"], "alice")
        self.assertEqual(
            rows_by_type["survey.import.ready"].payload_json["payload"]["profile"]["structured_profile"]["long_term_facts"]["identity"]["name"]["value"],
            "Alice",
        )
        self.assertNotIn("photos", rows_by_type["survey.import.ready"].payload_json["payload"])
        self.assertEqual(
            rows_by_type["survey.import.ready"].payload_json["payload"]["relationships"][0]["boxed_image_url"],
            "https://cdn.example.com/photo_001_boxed.webp",
        )

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

        ensure_schema()
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

    def _image_bytes(self, color: str) -> bytes:
        image = Image.new("RGB", (48, 48), color=color)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

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
                "vp1_observations": [],
                "lp1_events": [],
                "lp2_relationships": [],
                "lp3_profile": {},
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


if __name__ == "__main__":
    unittest.main()
