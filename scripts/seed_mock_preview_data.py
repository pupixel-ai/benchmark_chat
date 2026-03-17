#!/usr/bin/env python3
"""Seed local preview DB with reproducible mock tasks for frontend configuration review."""

from __future__ import annotations

import json
import mimetypes
import shutil
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import delete, select

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.auth import _hash_password
from backend.db import Base, SessionLocal, engine, ensure_schema
from backend.models import TaskRecord, UserRecord
from config import TASKS_DIR


MOCK_USERNAME = "mock_preview_v0317"
MOCK_PASSWORD = "memory0317demo"
MOCK_USER_ID = "mock_preview_v0317_user"

SOURCE_TASK_ID = "fresh_full_e2e_20260317_01"
SOURCE_TASK_DIR = Path(TASKS_DIR) / SOURCE_TASK_ID

MOCK_TASKS = [
    {
        "task_id": "mock_v0317_full_20260317",
        "version": "v0317",
        "offset_hours": 0,
        "mode": "full",
    },
    {
        "task_id": "mock_v0315_faceonly_20260317",
        "version": "v0315",
        "offset_hours": 1,
        "mode": "face_only",
    },
    {
        "task_id": "mock_v0312_faceonly_20260317",
        "version": "v0312",
        "offset_hours": 2,
        "mode": "face_only",
    },
]


def _load_source_result() -> dict:
    result_path = SOURCE_TASK_DIR / "output" / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"缺少 source result: {result_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))


def _replace_task_id_references(value, source_task_id: str, target_task_id: str):
    if isinstance(value, dict):
        return {key: _replace_task_id_references(item, source_task_id, target_task_id) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_task_id_references(item, source_task_id, target_task_id) for item in value]
    if isinstance(value, str):
        return value.replace(f"/api/assets/{source_task_id}/", f"/api/assets/{target_task_id}/")
    return value


def _build_uploads(task_id: str, task_dir: Path, result_payload: dict) -> list[dict]:
    image_entries = (result_payload.get("face_recognition") or {}).get("images") or []
    image_id_by_filename = {
        str(item.get("filename") or ""): str(item.get("image_id") or "")
        for item in image_entries
        if item.get("filename") and item.get("image_id")
    }
    uploads = []
    for file_path in sorted((task_dir / "uploads").iterdir()):
        if not file_path.is_file():
            continue
        content_type, _ = mimetypes.guess_type(file_path.name)
        uploads.append(
            {
                "image_id": image_id_by_filename.get(file_path.name) or file_path.stem,
                "filename": file_path.name,
                "stored_filename": file_path.name,
                "path": f"uploads/{file_path.name}",
                "url": f"/api/assets/{task_id}/uploads/{file_path.name}",
                "preview_url": None,
                "width": None,
                "height": None,
                "content_type": content_type or "application/octet-stream",
                "source_hash": None,
            }
        )
    return uploads


def _prepare_result(source_result: dict, source_task_id: str, target_task_id: str, version: str, mode: str) -> dict:
    payload = _replace_task_id_references(deepcopy(source_result), source_task_id, target_task_id)
    payload["task_id"] = target_task_id
    payload["version"] = version
    payload.setdefault("summary", {})
    payload["summary"]["task_version"] = version

    if mode == "face_only":
        payload["summary"]["vlm_processed_images"] = 0
        payload["summary"]["event_count"] = 0
        payload["summary"]["relationship_count"] = 0
        payload["summary"]["session_count"] = 0
        payload["summary"]["profile_version"] = 0
        warnings = list(payload.get("warnings") or [])
        warnings.append(
            {
                "stage": "version_gate",
                "message": f"{version} 当前只保留到人脸识别的原始链路，VLM / LLM / Memory 已跳过",
            }
        )
        payload["warnings"] = warnings
        payload["events"] = []
        payload["relationships"] = []
        payload["profile_markdown"] = ""
        payload["memory"] = None

    return payload


def _copy_task_dir(source_dir: Path, target_dir: Path, include_memory: bool) -> None:
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    if not include_memory:
        memory_dir = target_dir / "output" / "memory"
        if memory_dir.exists():
            shutil.rmtree(memory_dir)
        profile_report = target_dir / "output" / "user_profile_report.md"
        if profile_report.exists():
            profile_report.unlink()
        vlm_cache = target_dir / "cache" / "vlm_cache.json"
        if vlm_cache.exists():
            vlm_cache.unlink()


def main() -> None:
    ensure_schema()
    Base.metadata.create_all(bind=engine)
    source_result = _load_source_result()
    now = datetime.now()

    with SessionLocal() as session:
        user = session.execute(select(UserRecord).where(UserRecord.username == MOCK_USERNAME)).scalar_one_or_none()
        if user is None:
            user = UserRecord(
                user_id=MOCK_USER_ID,
                username=MOCK_USERNAME,
                password_hash=_hash_password(MOCK_PASSWORD),
                created_at=now,
                updated_at=now,
            )
            session.add(user)
            session.commit()
        else:
            user.user_id = MOCK_USER_ID
            user.password_hash = _hash_password(MOCK_PASSWORD)
            user.updated_at = now
            session.add(user)
            session.commit()

    with SessionLocal() as session:
        session.execute(delete(TaskRecord).where(TaskRecord.task_id.in_([item["task_id"] for item in MOCK_TASKS])))
        session.commit()

    for item in MOCK_TASKS:
        target_task_id = item["task_id"]
        version = item["version"]
        mode = item["mode"]
        include_memory = mode == "full"
        task_dir = Path(TASKS_DIR) / target_task_id
        _copy_task_dir(SOURCE_TASK_DIR, task_dir, include_memory=include_memory)

        result_payload = _prepare_result(source_result, SOURCE_TASK_ID, target_task_id, version, mode)
        result_path = task_dir / "output" / "result.json"
        result_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        uploads = _build_uploads(target_task_id, task_dir, result_payload)
        created_at = now - timedelta(hours=item["offset_hours"])
        record = TaskRecord(
            task_id=target_task_id,
            user_id=MOCK_USER_ID,
            version=version,
            status="completed",
            stage="completed",
            upload_count=len(uploads),
            task_dir=str(task_dir),
            progress=None,
            uploads=uploads,
            result=result_payload,
            result_summary=result_payload.get("summary"),
            asset_manifest=None,
            error=None,
            worker_instance_id=None,
            worker_private_ip=None,
            worker_status=None,
            delete_state=None,
            created_at=created_at,
            updated_at=created_at,
            expires_at=None,
            deleted_at=None,
            last_worker_sync_at=None,
        )
        with SessionLocal() as session:
            session.add(record)
            session.commit()

    print("Seeded mock preview data")
    print(f"username={MOCK_USERNAME}")
    print(f"password={MOCK_PASSWORD}")
    for item in MOCK_TASKS:
        print(f"{item['task_id']} -> {item['version']} ({item['mode']})")


if __name__ == "__main__":
    main()
