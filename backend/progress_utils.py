"""
Helpers for task progress snapshots, task logs, and error surfacing.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List

MAX_PROGRESS_LOGS = 300


def _iso_now() -> str:
    return datetime.now().isoformat()


def _coerce_number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = float(value)
        except ValueError:
            return None
        return int(parsed)
    return None


def _progress_counters(payload: Dict[str, Any]) -> tuple[int | None, int | None]:
    processed = (
        _coerce_number(payload.get("processed"))
        or _coerce_number(payload.get("processed_slices"))
        or _coerce_number(payload.get("processed_events"))
        or _coerce_number(payload.get("processed_candidates"))
        or _coerce_number(payload.get("completed_count"))
    )
    total = (
        _coerce_number(payload.get("photo_count"))
        or _coerce_number(payload.get("slice_count"))
        or _coerce_number(payload.get("event_count"))
        or _coerce_number(payload.get("filtered_count"))
        or _coerce_number(payload.get("candidate_count"))
        or _coerce_number(payload.get("total_input_photos"))
    )
    return processed, total


def _build_log_entry(stage: str, payload: Dict[str, Any], *, level: str) -> Dict[str, Any]:
    processed, total = _progress_counters(payload)
    entry: Dict[str, Any] = {
        "timestamp": _iso_now(),
        "level": level,
        "stage": stage,
        "substage": payload.get("substage"),
        "message": payload.get("message") or stage,
        "percent": _coerce_number(payload.get("percent")),
        "processed": processed,
        "total": total,
        "provider": payload.get("provider"),
        "model": payload.get("model"),
    }
    current_person_id = payload.get("current_person_id")
    if isinstance(current_person_id, str) and current_person_id.strip():
        entry["current_person_id"] = current_person_id.strip()
    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        entry["error"] = error.strip()
    return entry


def _log_signature(entry: Dict[str, Any]) -> tuple[Any, ...]:
    return (
        entry.get("level"),
        entry.get("stage"),
        entry.get("substage"),
        entry.get("message"),
        entry.get("percent"),
        entry.get("processed"),
        entry.get("total"),
        entry.get("provider"),
        entry.get("model"),
        entry.get("current_person_id"),
        entry.get("error"),
    )


def append_progress_log(existing: Dict[str, Any] | None, stage: str, payload: Dict[str, Any], *, level: str | None = None) -> Dict[str, Any]:
    base = deepcopy(existing or {})
    logs = list(base.get("logs") or [])
    resolved_level = level or ("error" if payload.get("error") else "info")
    entry = _build_log_entry(stage, payload, level=resolved_level)
    if not logs or _log_signature(logs[-1]) != _log_signature(entry):
        logs.append(entry)
    base["logs"] = logs[-MAX_PROGRESS_LOGS:]
    return base


def merge_stage_progress(existing: Dict[str, Any] | None, stage: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    base = append_progress_log(existing, stage, payload)
    merged_stages = dict(base.get("stages") or {})
    current_stage_payload = merged_stages.get(stage)
    if isinstance(current_stage_payload, dict):
        next_stage_payload = {**current_stage_payload, **deepcopy(payload)}
    else:
        next_stage_payload = deepcopy(payload)
    next_stage_payload["updated_at"] = _iso_now()
    merged_stages[stage] = next_stage_payload
    base["stages"] = merged_stages
    base["current_stage"] = stage
    base["updated_at"] = next_stage_payload["updated_at"]
    return base


def append_terminal_error(existing: Dict[str, Any] | None, *, stage: str, error: str, substage: str | None = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "message": "任务执行失败",
        "error": error,
        "substage": substage,
    }
    base = append_progress_log(existing, stage, payload, level="error")
    base["current_stage"] = stage
    base["updated_at"] = _iso_now()
    return base


def append_terminal_info(existing: Dict[str, Any] | None, *, stage: str, message: str) -> Dict[str, Any]:
    base = append_progress_log(existing, stage, {"message": message}, level="info")
    base["current_stage"] = stage
    base["updated_at"] = _iso_now()
    return base

