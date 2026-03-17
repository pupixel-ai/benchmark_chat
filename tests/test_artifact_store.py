from __future__ import annotations

import shutil
import tempfile
import unittest
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import delete

from backend.artifact_store import ArtifactCatalogStore, build_task_asset_manifest
from backend.db import SessionLocal
from backend.models import ArtifactRecord, TaskRecord, UserRecord
from backend.task_store import TaskStore
from services.asset_store import TaskAssetStore


class ArtifactStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = ArtifactCatalogStore()
        self.task_store = TaskStore()
        self.user_id = f"user_{uuid.uuid4().hex[:8]}"
        self.task_id = f"task_{uuid.uuid4().hex[:8]}"
        self.task_dir = self.task_store.task_dir(self.task_id)
        self.task_dir.mkdir(parents=True, exist_ok=True)
        (self.task_dir / "output").mkdir(parents=True, exist_ok=True)
        (self.task_dir / "output" / "result.json").write_text('{"ok": true}', encoding="utf-8")

        now = datetime.utcnow()
        with SessionLocal() as session:
            session.add(
                UserRecord(
                    user_id=self.user_id,
                    username=f"artifact_{uuid.uuid4().hex[:8]}",
                    password_hash="hash",
                    created_at=now,
                    updated_at=now,
                )
            )
            session.add(
                TaskRecord(
                    task_id=self.task_id,
                    user_id=self.user_id,
                    version="v0317",
                    status="completed",
                    stage="completed",
                    upload_count=0,
                    task_dir=str(self.task_dir),
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
                    created_at=now,
                    updated_at=now,
                    expires_at=None,
                    deleted_at=None,
                    last_worker_sync_at=None,
                )
            )
            session.commit()

    def tearDown(self) -> None:
        shutil.rmtree(self.task_dir, ignore_errors=True)
        with SessionLocal() as session:
            session.execute(delete(ArtifactRecord).where(ArtifactRecord.task_id == self.task_id))
            session.execute(delete(TaskRecord).where(TaskRecord.task_id == self.task_id))
            session.execute(delete(UserRecord).where(UserRecord.user_id == self.user_id))
            session.commit()

    def test_build_manifest_and_replace_artifacts(self) -> None:
        manifest = build_task_asset_manifest(self.task_id, self.task_dir, TaskAssetStore())
        rows = self.catalog.replace_task_artifacts(self.task_id, self.user_id, manifest)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["relative_path"], "output/result.json")
        self.assertEqual(rows[0]["stage"], "output")
        self.assertEqual(rows[0]["storage_backend"], "local")
        self.assertTrue(rows[0]["sha256"])

    def test_task_asset_store_enables_aws_bucket_without_explicit_keys(self) -> None:
        with patch("services.asset_store.OBJECT_STORAGE_BUCKET", "memory-artifacts"):
            store = TaskAssetStore()
            self.assertTrue(store.enabled)

        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir)
            (test_dir / "cache").mkdir()
            (test_dir / "cache" / "x.txt").write_text("hello", encoding="utf-8")
            manifest = build_task_asset_manifest("task_aws", test_dir, store)

        self.assertEqual(manifest["files"][0]["storage_backend"], "s3")


if __name__ == "__main__":
    unittest.main()
