"""
Build survey-import Kafka payloads for v0327-db-query tasks.

The enhanced (v2) payload carries events, VLM observations and face data
alongside the original profile + relationships so that downstream consumers
(e.g. Me_Reflection) can write source files without a second API call.
Heavy sections are automatically dropped when the serialised payload exceeds
the configured Kafka message size limit; the ``snapshot_mode`` field tells
the consumer whether it needs to pull the remaining data via the bundle API.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from backend.memory_full_retrieval import (
    build_task_survey_import_payload,
    build_task_survey_import_payload_v2,
)
from config import APP_VERSION, KAFKA_MESSAGE_MAX_BYTES, TASK_VERSION_V0327_DB_QUERY


def build_survey_import_event_payload(task: dict) -> Optional[dict]:
    if str(task.get("status") or "").strip() != "completed":
        return None
    if str(task.get("version") or "").strip() != TASK_VERSION_V0327_DB_QUERY:
        return None

    options = dict(task.get("options") or {})
    username = str(options.get("survey_username") or "").strip()
    if not username:
        return None

    task_id = str(task.get("task_id") or "").strip()

    # Build the enhanced payload with adaptive size control.
    # Reserve ~20 % of the Kafka limit for the event envelope.
    payload_budget = int(KAFKA_MESSAGE_MAX_BYTES * 0.8) if KAFKA_MESSAGE_MAX_BYTES > 0 else 0
    try:
        payload = build_task_survey_import_payload_v2(task, max_bytes=payload_budget)
    except Exception:
        # Fall back to the lightweight v1 payload on any build error.
        payload = build_task_survey_import_payload(task)

    report_markdown = str(((payload.get("profile") or {}).get("report_markdown")) or "").strip()
    structured_profile = (payload.get("profile") or {}).get("structured_profile") or {}
    relationships = list(payload.get("relationships") or [])
    if not report_markdown and not structured_profile and not relationships:
        return None

    payload["api_bundle_url"] = f"/api/tasks/{task_id}/memory/bundle"

    return {
        "event_id": uuid.uuid4().hex,
        "event_type": "survey.import.ready",
        "schema_version": "v2",
        "occurred_at": _resolve_occurred_at(task),
        "task_id": task_id,
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
