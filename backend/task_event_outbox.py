"""
Persistence helpers for terminal-event outbox delivery.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session

from backend.db import SessionLocal
from backend.models import TaskEventOutboxRecord
from config import (
    KAFKA_ENABLED,
    KAFKA_PUBLISHER_LOCK_SECONDS,
    KAFKA_PUBLISHER_RETRY_SECONDS,
    KAFKA_TERMINAL_TOPIC,
)

STATUS_PENDING = "pending"
STATUS_PUBLISHING = "publishing"
STATUS_PUBLISHED = "published"
STATUS_RETRY = "retry"


def build_terminal_dedupe_key(task_id: str, event_type: str) -> str:
    return f"task:{task_id}:terminal:{event_type}"


@dataclass
class TaskEventOutboxItem:
    outbox_id: str
    event_id: str
    topic: str
    event_type: str
    task_id: str
    event_key: str
    dedupe_key: str
    payload_json: dict
    status: str
    attempt_count: int
    available_at: str
    locked_at: Optional[str]
    locked_by: Optional[str]
    published_at: Optional[str]
    last_error: Optional[str]
    created_at: str


class TaskEventOutboxStore:
    def __init__(self) -> None:
        self.enabled = KAFKA_ENABLED
        self.topic = KAFKA_TERMINAL_TOPIC
        self.retry_seconds = max(1, KAFKA_PUBLISHER_RETRY_SECONDS)
        self.lock_seconds = max(5, KAFKA_PUBLISHER_LOCK_SECONDS)

    def enqueue_terminal_event(
        self,
        session: Session,
        *,
        task_id: str,
        event_type: str,
        payload_json: dict,
        topic: str | None = None,
    ) -> TaskEventOutboxRecord:
        dedupe_key = build_terminal_dedupe_key(task_id, event_type)
        for pending in session.new:
            if isinstance(pending, TaskEventOutboxRecord) and pending.dedupe_key == dedupe_key:
                return pending
        existing = session.execute(
            select(TaskEventOutboxRecord).where(TaskEventOutboxRecord.dedupe_key == dedupe_key)
        ).scalar_one_or_none()
        if existing is not None:
            return existing

        now = datetime.now()
        record = TaskEventOutboxRecord(
            outbox_id=uuid.uuid4().hex,
            event_id=str(payload_json.get("event_id") or uuid.uuid4().hex),
            topic=(topic or self.topic).strip() or self.topic,
            event_type=event_type,
            task_id=task_id,
            event_key=task_id,
            dedupe_key=dedupe_key,
            payload_json=payload_json,
            status=STATUS_PENDING,
            attempt_count=0,
            available_at=now,
            locked_at=None,
            locked_by=None,
            published_at=None,
            last_error=None,
            created_at=now,
        )
        session.add(record)
        return record

    def claim_batch(self, *, batch_size: int, locked_by: str) -> List[TaskEventOutboxItem]:
        now = datetime.now()
        stale_before = now - timedelta(seconds=self.lock_seconds)
        claimed_ids: List[str] = []
        reclaimable_lock = or_(
            TaskEventOutboxRecord.locked_at.is_(None),
            TaskEventOutboxRecord.locked_at <= stale_before,
        )
        claimable_condition = or_(
            and_(
                TaskEventOutboxRecord.status.in_((STATUS_PENDING, STATUS_RETRY)),
                TaskEventOutboxRecord.available_at <= now,
                reclaimable_lock,
            ),
            and_(
                TaskEventOutboxRecord.status == STATUS_PUBLISHING,
                reclaimable_lock,
            ),
        )

        with SessionLocal() as session:
            candidate_ids = [
                row[0]
                for row in session.execute(
                    select(TaskEventOutboxRecord.outbox_id)
                    .where(claimable_condition)
                    .order_by(TaskEventOutboxRecord.created_at.asc())
                    .limit(max(1, batch_size))
                ).all()
            ]

            for outbox_id in candidate_ids:
                result = session.execute(
                    update(TaskEventOutboxRecord)
                    .where(
                        TaskEventOutboxRecord.outbox_id == outbox_id,
                        claimable_condition,
                    )
                    .values(
                        status=STATUS_PUBLISHING,
                        locked_at=now,
                        locked_by=locked_by,
                    )
                )
                if result.rowcount:
                    claimed_ids.append(outbox_id)
            session.commit()

        if not claimed_ids:
            return []

        with SessionLocal() as session:
            records = session.execute(
                select(TaskEventOutboxRecord)
                .where(TaskEventOutboxRecord.outbox_id.in_(claimed_ids))
                .order_by(TaskEventOutboxRecord.created_at.asc())
            ).scalars().all()
            return [self._serialize(record) for record in records]

    def mark_published(self, outbox_id: str) -> None:
        with SessionLocal() as session:
            record = session.get(TaskEventOutboxRecord, outbox_id)
            if record is None:
                return
            record.status = STATUS_PUBLISHED
            record.attempt_count = int(record.attempt_count or 0) + 1
            record.published_at = datetime.now()
            record.locked_at = None
            record.locked_by = None
            record.last_error = None
            session.add(record)
            session.commit()

    def mark_retry(self, outbox_id: str, *, error: str) -> None:
        with SessionLocal() as session:
            record = session.get(TaskEventOutboxRecord, outbox_id)
            if record is None:
                return
            record.status = STATUS_RETRY
            record.attempt_count = int(record.attempt_count or 0) + 1
            record.available_at = datetime.now() + timedelta(seconds=self.retry_seconds)
            record.locked_at = None
            record.locked_by = None
            record.last_error = error[:4000]
            session.add(record)
            session.commit()

    def _serialize(self, record: TaskEventOutboxRecord) -> TaskEventOutboxItem:
        return TaskEventOutboxItem(
            outbox_id=record.outbox_id,
            event_id=record.event_id,
            topic=record.topic,
            event_type=record.event_type,
            task_id=record.task_id,
            event_key=record.event_key,
            dedupe_key=record.dedupe_key,
            payload_json=dict(record.payload_json or {}),
            status=record.status,
            attempt_count=int(record.attempt_count or 0),
            available_at=record.available_at.isoformat(),
            locked_at=record.locked_at.isoformat() if record.locked_at else None,
            locked_by=record.locked_by,
            published_at=record.published_at.isoformat() if record.published_at else None,
            last_error=record.last_error,
            created_at=record.created_at.isoformat(),
        )
