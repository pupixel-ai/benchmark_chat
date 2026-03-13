"""
SQL-backed task state store.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import desc, select

from backend.db import Base, SessionLocal, engine, ensure_schema
from backend.models import TaskRecord
from config import TASKS_DIR


class TaskStore:
    """使用 SQL 数据库保存任务状态。"""

    def __init__(self, tasks_root: str = TASKS_DIR):
        self.tasks_root = Path(tasks_root)
        self.tasks_root.mkdir(parents=True, exist_ok=True)
        ensure_schema()
        Base.metadata.create_all(bind=engine)

    def task_dir(self, task_id: str) -> Path:
        return self.tasks_root / task_id

    def create_task(self, task_id: str, upload_count: int, user_id: str) -> Dict:
        task_dir = self.task_dir(task_id)
        task_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()

        record = TaskRecord(
            task_id=task_id,
            user_id=user_id,
            status="queued",
            stage="queued",
            upload_count=upload_count,
            task_dir=str(task_dir),
            progress=None,
            uploads=None,
            result=None,
            error=None,
            created_at=now,
            updated_at=now,
        )

        with SessionLocal() as session:
            session.add(record)
            session.commit()
            session.refresh(record)

        return self._serialize(record)

    def get_task(self, task_id: str, user_id: str) -> Optional[Dict]:
        with SessionLocal() as session:
            record = session.execute(
                select(TaskRecord).where(
                    TaskRecord.task_id == task_id,
                    TaskRecord.user_id == user_id,
                )
            ).scalar_one_or_none()
            if record is None:
                return None
            return self._serialize(record)

    def list_tasks(self, user_id: str, limit: int = 20) -> List[Dict]:
        with SessionLocal() as session:
            stmt = (
                select(TaskRecord)
                .where(TaskRecord.user_id == user_id)
                .order_by(desc(TaskRecord.created_at))
                .limit(limit)
            )
            records = session.execute(stmt).scalars().all()
            return [self._serialize(record) for record in records]

    def update_task(self, task_id: str, **updates) -> Dict:
        with SessionLocal() as session:
            record = session.get(TaskRecord, task_id)
            if record is None:
                raise KeyError(f"任务不存在: {task_id}")

            for key, value in updates.items():
                setattr(record, key, value)

            record.updated_at = datetime.now()
            session.add(record)
            session.commit()
            session.refresh(record)
            return self._serialize(record)

    def _serialize(self, record: TaskRecord) -> Dict:
        return {
            "task_id": record.task_id,
            "user_id": record.user_id,
            "status": record.status,
            "stage": record.stage,
            "upload_count": record.upload_count,
            "task_dir": record.task_dir,
            "progress": record.progress,
            "uploads": record.uploads,
            "result": record.result,
            "error": record.error,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        }
