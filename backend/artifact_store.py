"""
Artifact catalog helpers backed by the application database.
"""
from __future__ import annotations

import hashlib
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import List

from sqlalchemy import delete, select

from backend.db import SessionLocal
from backend.models import ArtifactRecord
from services.asset_store import TaskAssetStore


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_task_asset_manifest(task_id: str, task_dir: str | Path, asset_store: TaskAssetStore) -> dict:
    root = Path(task_dir)
    generated_at = datetime.utcnow().isoformat()
    if not root.exists():
        return {
            "generated_at": generated_at,
            "storage_backend": "s3" if asset_store.enabled else "local",
            "files": [],
        }

    files: List[dict] = []
    storage_backend = "s3" if asset_store.enabled else "local"

    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():
            continue
        relative_path = file_path.relative_to(root).as_posix()
        content_type, _ = mimetypes.guess_type(relative_path)
        files.append(
            {
                "path": relative_path,
                "relative_path": relative_path,
                "stage": relative_path.split("/", 1)[0] if "/" in relative_path else relative_path,
                "size": file_path.stat().st_size,
                "size_bytes": file_path.stat().st_size,
                "sha256": _hash_file(file_path),
                "content_type": content_type or "application/octet-stream",
                "storage_backend": storage_backend,
                "object_key": asset_store.object_key(task_id, relative_path) if asset_store.enabled else None,
                "asset_url": asset_store.asset_url(task_id, relative_path),
                "metadata": {
                    "local_path": str(file_path),
                    "generated_at": generated_at,
                },
            }
        )

    return {
        "generated_at": generated_at,
        "storage_backend": storage_backend,
        "files": files,
    }


class ArtifactCatalogStore:
    """Stores per-task artifact metadata for preview and AWS-backed retrieval."""

    def replace_task_artifacts(self, task_id: str, user_id: str, manifest: dict | None) -> List[dict]:
        files = manifest.get("files", []) if isinstance(manifest, dict) else []
        now = datetime.utcnow()

        with SessionLocal() as session:
            session.execute(delete(ArtifactRecord).where(ArtifactRecord.task_id == task_id))
            for item in files:
                relative_path = str(item.get("relative_path") or item.get("path") or "").strip()
                if not relative_path:
                    continue
                artifact_id = self._artifact_id(task_id, relative_path)
                session.add(
                    ArtifactRecord(
                        artifact_id=artifact_id,
                        task_id=task_id,
                        user_id=user_id,
                        relative_path=relative_path,
                        stage=str(item.get("stage") or self._infer_stage(relative_path)),
                        content_type=str(item.get("content_type") or "application/octet-stream"),
                        size_bytes=int(item.get("size_bytes") or item.get("size") or 0),
                        sha256=str(item.get("sha256") or ""),
                        storage_backend=str(item.get("storage_backend") or "local"),
                        object_key=self._optional_str(item.get("object_key")),
                        asset_url=self._optional_str(item.get("asset_url")),
                        metadata_json=item.get("metadata") if isinstance(item.get("metadata"), dict) else None,
                        created_at=now,
                        updated_at=now,
                    )
                )
            session.commit()

        return self.list_task_artifacts(task_id, user_id)

    def list_task_artifacts(self, task_id: str, user_id: str) -> List[dict]:
        del user_id
        with SessionLocal() as session:
            records = session.execute(
                select(ArtifactRecord)
                .where(ArtifactRecord.task_id == task_id)
                .order_by(ArtifactRecord.relative_path.asc())
            ).scalars().all()
            return [self._serialize(record) for record in records]

    def get_task_artifact(self, task_id: str, user_id: str, relative_path: str) -> Optional[dict]:
        del user_id
        with SessionLocal() as session:
            record = session.execute(
                select(ArtifactRecord).where(
                    ArtifactRecord.task_id == task_id,
                    ArtifactRecord.relative_path == relative_path,
                )
            ).scalar_one_or_none()
            return self._serialize(record) if record is not None else None

    def delete_task_artifacts(self, task_id: str, user_id: str | None = None) -> int:
        with SessionLocal() as session:
            stmt = delete(ArtifactRecord).where(ArtifactRecord.task_id == task_id)
            if user_id is not None:
                stmt = stmt.where(ArtifactRecord.user_id == user_id)
            result = session.execute(stmt)
            session.commit()
            return int(result.rowcount or 0)

    def _artifact_id(self, task_id: str, relative_path: str) -> str:
        return f"{task_id}:{relative_path}"

    def _infer_stage(self, relative_path: str) -> str:
        return relative_path.split("/", 1)[0] if "/" in relative_path else relative_path

    def _optional_str(self, value: object) -> str | None:
        text = str(value or "").strip()
        return text or None

    def _serialize(self, record: ArtifactRecord) -> dict:
        return {
            "artifact_id": record.artifact_id,
            "task_id": record.task_id,
            "user_id": record.user_id,
            "relative_path": record.relative_path,
            "stage": record.stage,
            "content_type": record.content_type,
            "size_bytes": record.size_bytes,
            "sha256": record.sha256,
            "storage_backend": record.storage_backend,
            "object_key": record.object_key,
            "asset_url": record.asset_url,
            "metadata": record.metadata_json,
            "created_at": record.created_at.isoformat(),
            "updated_at": record.updated_at.isoformat(),
        }
