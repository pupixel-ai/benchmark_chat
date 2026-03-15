"""
SQL-backed task state store.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import delete, desc, or_, select

from backend.db import Base, SessionLocal, engine, ensure_schema
from backend.models import TaskRecord
from config import TASKS_DIR, TASK_VERSION_V0312


class TaskStore:
    """使用 SQL 数据库保存任务状态。"""

    def __init__(self, tasks_root: str = TASKS_DIR):
        self.tasks_root = Path(tasks_root)
        self.tasks_root.mkdir(parents=True, exist_ok=True)
        ensure_schema()
        Base.metadata.create_all(bind=engine)

    def task_dir(self, task_id: str) -> Path:
        return self.tasks_root / task_id

    def create_task(
        self,
        task_id: str,
        upload_count: int,
        user_id: str,
        version: str,
        provision_local_dir: bool = True,
        status: str = "queued",
        stage: str = "queued",
    ) -> Dict:
        task_dir = self.task_dir(task_id)
        if provision_local_dir:
            task_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()

        record = TaskRecord(
            task_id=task_id,
            user_id=user_id,
            version=version,
            status=status,
            stage=stage,
            upload_count=upload_count,
            task_dir=str(task_dir),
            progress=None,
            uploads=None,
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
            return [self._serialize(record, include_details=False) for record in records]

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

    def append_uploads(self, task_id: str, uploads: List[Dict], *, status: str = "draft", stage: str = "uploading") -> Dict:
        with SessionLocal() as session:
            record = session.get(TaskRecord, task_id)
            if record is None:
                raise KeyError(f"任务不存在: {task_id}")

            current_uploads = list(record.uploads or [])
            current_uploads.extend(uploads)
            record.uploads = current_uploads
            record.upload_count = len(current_uploads)
            record.status = status
            record.stage = stage
            record.updated_at = datetime.now()
            session.add(record)
            session.commit()
            session.refresh(record)
            return self._serialize(record)

    def delete_task(self, task_id: str, user_id: str) -> bool:
        with SessionLocal() as session:
            result = session.execute(
                delete(TaskRecord).where(
                    TaskRecord.task_id == task_id,
                    TaskRecord.user_id == user_id,
                )
            )
            session.commit()
            return bool(result.rowcount)

    def attach_worker(
        self,
        task_id: str,
        instance_id: str,
        private_ip: str | None,
        expires_at: datetime | None,
        worker_status: str = "launching",
    ) -> Dict:
        return self.update_task(
            task_id,
            worker_instance_id=instance_id,
            worker_private_ip=private_ip,
            worker_status=worker_status,
            expires_at=expires_at,
            delete_state="active",
            last_worker_sync_at=datetime.now(),
        )

    def update_worker_state(
        self,
        task_id: str,
        worker_status: str | None = None,
        last_worker_sync_at: datetime | None = None,
        **updates,
    ) -> Dict:
        if worker_status is not None:
            updates["worker_status"] = worker_status
        updates["last_worker_sync_at"] = last_worker_sync_at or datetime.now()
        return self.update_task(task_id, **updates)

    def set_result_summary(self, task_id: str, result_summary: dict | None, asset_manifest: dict | None) -> Dict:
        return self.update_task(task_id, result_summary=result_summary, asset_manifest=asset_manifest)

    def mark_delete_requested(self, task_id: str, user_id: str) -> Optional[Dict]:
        task = self.get_task(task_id, user_id=user_id)
        if task is None:
            return None
        return self.update_task(task_id, delete_state="requested")

    def mark_deleted(self, task_id: str, user_id: str, deleted_at: datetime | None = None) -> Optional[Dict]:
        task = self.get_task(task_id, user_id=user_id)
        if task is None:
            return None
        return self.update_task(
            task_id,
            delete_state="deleted",
            deleted_at=deleted_at or datetime.now(),
            worker_status="terminated",
        )

    def clear_sensitive_payloads(self, task_id: str) -> Dict:
        return self.update_task(
            task_id,
            uploads=None,
            result=None,
            progress=None,
            error=None,
        )

    def list_deletion_candidates(self, limit: int = 100) -> List[Dict]:
        with SessionLocal() as session:
            stmt = (
                select(TaskRecord)
                .where(TaskRecord.delete_state == "requested")
                .order_by(desc(TaskRecord.updated_at))
                .limit(limit)
            )
            records = session.execute(stmt).scalars().all()
            return [self._serialize(record) for record in records]

    def list_expired_tasks(self, before: datetime, limit: int = 100) -> List[Dict]:
        with SessionLocal() as session:
            stmt = (
                select(TaskRecord)
                .where(
                    TaskRecord.expires_at.is_not(None),
                    TaskRecord.expires_at <= before,
                    or_(TaskRecord.delete_state.is_(None), TaskRecord.delete_state != "deleted"),
                )
                .order_by(desc(TaskRecord.expires_at))
                .limit(limit)
            )
            records = session.execute(stmt).scalars().all()
            return [self._serialize(record) for record in records]

    def _serialize(self, record: TaskRecord, include_details: bool = True) -> Dict:
        return {
            "task_id": record.task_id,
            "user_id": record.user_id,
            "version": record.version or TASK_VERSION_V0312,
            "status": record.status,
            "stage": record.stage,
            "upload_count": record.upload_count,
            "task_dir": record.task_dir,
            "progress": record.progress if include_details else None,
            "uploads": record.uploads if include_details else None,
            "result": record.result if include_details else None,
            "result_summary": record.result_summary,
            "asset_manifest": record.asset_manifest,
            "error": record.error,
            "worker_instance_id": record.worker_instance_id,
            "worker_private_ip": record.worker_private_ip,
            "worker_status": record.worker_status,
            "delete_state": record.delete_state,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
            "expires_at": record.expires_at.isoformat() if record.expires_at else None,
            "deleted_at": record.deleted_at.isoformat() if record.deleted_at else None,
            "last_worker_sync_at": record.last_worker_sync_at.isoformat() if record.last_worker_sync_at else None,
        }
