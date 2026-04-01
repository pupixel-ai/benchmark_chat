from __future__ import annotations

import shutil
import threading
import unittest
import uuid
from datetime import datetime, timedelta

from sqlalchemy import delete, select

from backend.db import SessionLocal
from backend.models import TaskEventOutboxRecord, TaskRecord, UserRecord
from backend.task_event_outbox import TaskEventOutboxStore
from backend.task_store import TaskStore


class TaskEventOutboxStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task_store = TaskStore()
        self.outbox_store = TaskEventOutboxStore()
        self.user_id = uuid.uuid4().hex
        self.task_id = uuid.uuid4().hex
        self.task_dir = self.task_store.task_dir(self.task_id)
        self.task_dir.mkdir(parents=True, exist_ok=True)

        with SessionLocal() as session:
            session.add(
                UserRecord(
                    user_id=self.user_id,
                    username=f"outbox_{self.user_id[:8]}",
                    password_hash="hash",
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                )
            )
            session.add(
                TaskRecord(
                    task_id=self.task_id,
                    user_id=self.user_id,
                    version="v0327-db-query",
                    status="queued",
                    stage="queued",
                    upload_count=0,
                    task_dir=str(self.task_dir),
                    progress=None,
                    uploads=[],
                    options={},
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

    def tearDown(self) -> None:
        shutil.rmtree(self.task_dir, ignore_errors=True)
        with SessionLocal() as session:
            session.execute(delete(TaskEventOutboxRecord).where(TaskEventOutboxRecord.task_id == self.task_id))
            session.execute(delete(TaskRecord).where(TaskRecord.task_id == self.task_id))
            session.execute(delete(UserRecord).where(UserRecord.user_id == self.user_id))
            session.commit()

    def test_enqueue_terminal_event_deduplicates_by_dedupe_key(self) -> None:
        payload = {"event_id": uuid.uuid4().hex, "event_type": "task.completed", "task_id": self.task_id}
        with SessionLocal() as session:
            first = self.outbox_store.enqueue_terminal_event(
                session,
                task_id=self.task_id,
                event_type="task.completed",
                payload_json=payload,
            )
            second = self.outbox_store.enqueue_terminal_event(
                session,
                task_id=self.task_id,
                event_type="task.completed",
                payload_json={"event_id": uuid.uuid4().hex, "event_type": "task.completed", "task_id": self.task_id},
            )
            first_id = first.outbox_id
            second_id = second.outbox_id
            session.commit()

        self.assertEqual(first_id, second_id)
        with SessionLocal() as session:
            rows = session.execute(
                select(TaskEventOutboxRecord).where(TaskEventOutboxRecord.task_id == self.task_id)
            ).scalars().all()
        self.assertEqual(len(rows), 1)

    def test_claim_batch_only_allows_one_consumer_to_take_the_same_row(self) -> None:
        with SessionLocal() as session:
            self.outbox_store.enqueue_terminal_event(
                session,
                task_id=self.task_id,
                event_type="task.completed",
                payload_json={"event_id": uuid.uuid4().hex, "event_type": "task.completed", "task_id": self.task_id},
            )
            session.commit()

        barrier = threading.Barrier(3)
        claimed: list[str] = []
        claimed_lock = threading.Lock()

        def worker() -> None:
            barrier.wait()
            items = self.outbox_store.claim_batch(batch_size=1, locked_by=uuid.uuid4().hex)
            with claimed_lock:
                claimed.extend(item.outbox_id for item in items)

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join(timeout=2)

        self.assertEqual(len(claimed), 1)

    def test_claim_batch_reclaims_stale_publishing_rows(self) -> None:
        with SessionLocal() as session:
            record = self.outbox_store.enqueue_terminal_event(
                session,
                task_id=self.task_id,
                event_type="task.completed",
                payload_json={"event_id": uuid.uuid4().hex, "event_type": "task.completed", "task_id": self.task_id},
            )
            record.status = "publishing"
            record.locked_at = datetime.now() - timedelta(seconds=self.outbox_store.lock_seconds + 5)
            record.locked_by = "stale-publisher"
            session.add(record)
            session.commit()
            outbox_id = record.outbox_id

        items = self.outbox_store.claim_batch(batch_size=1, locked_by="fresh-publisher")

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].outbox_id, outbox_id)
        with SessionLocal() as session:
            refreshed = session.get(TaskEventOutboxRecord, outbox_id)
            assert refreshed is not None
            self.assertEqual(refreshed.status, "publishing")
            self.assertEqual(refreshed.locked_by, "fresh-publisher")
