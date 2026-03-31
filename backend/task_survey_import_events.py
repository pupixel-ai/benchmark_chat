"""
Build lightweight survey-import Kafka payloads for v0327-db-query tasks.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from backend.memory_full_retrieval import build_task_survey_import_payload
from config import APP_VERSION, TASK_VERSION_V0327_DB_QUERY


def build_survey_import_event_payload(task: dict) -> Optional[dict]:
    if str(task.get("status") or "").strip() != "completed":
        return None
    if str(task.get("version") or "").strip() != TASK_VERSION_V0327_DB_QUERY:
        return None

    options = dict(task.get("options") or {})
    username = str(options.get("survey_username") or "").strip()
    if not username:
        return None

    payload = build_task_survey_import_payload(task)
    report_markdown = str(((payload.get("profile") or {}).get("report_markdown")) or "").strip()
    structured_profile = (payload.get("profile") or {}).get("structured_profile") or {}
    relationships = list(payload.get("relationships") or [])
    if not report_markdown and not structured_profile and not relationships:
        return None

    return {
        "event_id": uuid.uuid4().hex,
        "event_type": "survey.import.ready",
        "schema_version": "v1",
        "occurred_at": _resolve_occurred_at(task),
        "task_id": str(task.get("task_id") or "").strip(),
        "task_version": TASK_VERSION_V0327_DB_QUERY,
        "username": username,
        "producer": {
            "service": "memory-engineering",
            "version": APP_VERSION,
        },
        "payload": payload,
    }


def _resolve_occurred_at(task: dict) -> str:
    raw = task.get("updated_at") or task.get("created_at")
    if isinstance(raw, datetime):
        return raw.isoformat()
    text = str(raw or "").strip()
    return text or datetime.now().isoformat()
