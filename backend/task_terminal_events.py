"""
Build canonical terminal-event payloads for Kafka delivery.
"""
from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime
from typing import Any, Callable, Iterable, Optional

from backend.face_review_store import FaceReviewStore
from backend.memory_full_retrieval import build_task_memory_core_payload
from backend.memory_step_retrieval import build_task_memory_steps_payload
from config import (
    APP_VERSION,
    KAFKA_MESSAGE_MAX_BYTES,
    TASK_VERSION_V0323,
    TASK_VERSION_V0325,
    TASK_VERSION_V0327_DB,
    TASK_VERSION_V0327_DB_QUERY,
    TASK_VERSION_V0327_EXP,
)

SUPPORTED_STEPS_VERSIONS = {
    TASK_VERSION_V0323,
    TASK_VERSION_V0325,
    TASK_VERSION_V0327_EXP,
    TASK_VERSION_V0327_DB,
    TASK_VERSION_V0327_DB_QUERY,
}


def _task_source_hashes(task: dict) -> set[str]:
    hashes: set[str] = set()
    uploads = task.get("uploads") or []
    if isinstance(uploads, list):
        for upload in uploads:
            if not isinstance(upload, dict):
                continue
            source_hash = upload.get("source_hash")
            if source_hash:
                hashes.add(str(source_hash))
    result = task.get("result") or {}
    face_payload = result.get("face_recognition") if isinstance(result, dict) else {}
    images = face_payload.get("images") if isinstance(face_payload, dict) else []
    if isinstance(images, list):
        for image in images:
            if not isinstance(image, dict):
                continue
            source_hash = image.get("source_hash")
            if source_hash:
                hashes.add(str(source_hash))
    return hashes


def build_terminal_event_payload(
    task: dict,
    *,
    face_review_store: FaceReviewStore,
    asset_url_builder: Callable[[str, str], str],
    message_max_bytes: int = KAFKA_MESSAGE_MAX_BYTES,
) -> dict:
    status = str(task.get("status") or "").strip()
    event_type = "task.completed" if status == "completed" else "task.failed"
    event = {
        "event_id": uuid.uuid4().hex,
        "event_type": event_type,
        "schema_version": "v1",
        "occurred_at": _resolve_occurred_at(task),
        "task_id": str(task.get("task_id") or "").strip(),
        "task_version": str(task.get("version") or "").strip(),
        "snapshot_mode": "full",
        "producer": {
            "service": "memory-engineering",
            "version": APP_VERSION,
        },
        "payload": _build_business_payload(
            task,
            face_review_store=face_review_store,
            asset_url_builder=asset_url_builder,
        ),
    }
    if _payload_size(event) <= message_max_bytes:
        return event

    reduced = _build_reduced_payload(event)
    if _payload_size(reduced) <= message_max_bytes:
        return reduced

    return _build_minimal_reduced_payload(reduced)


def build_fallback_terminal_event_payload(task: dict, *, reason: str) -> dict:
    status = str(task.get("status") or "").strip()
    event_type = "task.completed" if status == "completed" else "task.failed"
    return {
        "event_id": uuid.uuid4().hex,
        "event_type": event_type,
        "schema_version": "v1",
        "occurred_at": _resolve_occurred_at(task),
        "task_id": str(task.get("task_id") or "").strip(),
        "task_version": str(task.get("version") or "").strip(),
        "snapshot_mode": "reduced",
        "producer": {
            "service": "memory-engineering",
            "version": APP_VERSION,
        },
        "payload": {
            "task": _build_task_payload(task),
            "summary": copy.deepcopy(task.get("result_summary")),
            "memory_core": None,
            "steps": None,
            "reviews": {"review_count": 0, "policy_count": 0},
            "artifacts": _build_artifact_payload(task),
            "failure": _build_failure_payload(task),
            "payload_build_error": reason,
        },
    }


def _resolve_occurred_at(task: dict) -> str:
    raw = task.get("updated_at") or task.get("created_at")
    if isinstance(raw, datetime):
        return raw.isoformat()
    text = str(raw or "").strip()
    return text or datetime.now().isoformat()


def _build_business_payload(
    task: dict,
    *,
    face_review_store: FaceReviewStore,
    asset_url_builder: Callable[[str, str], str],
) -> dict:
    return {
        "task": _build_task_payload(task),
        "summary": copy.deepcopy(task.get("result_summary")),
        "memory_core": _build_memory_core_payload(task),
        "steps": _build_steps_payload(task, asset_url_builder=asset_url_builder),
        "reviews": _build_reviews_payload(task, face_review_store=face_review_store),
        "artifacts": _build_artifact_payload(task),
        "failure": _build_failure_payload(task),
    }


def _build_task_payload(task: dict) -> dict:
    return {
        "task_id": task.get("task_id"),
        "user_id": task.get("user_id"),
        "version": task.get("version"),
        "status": task.get("status"),
        "stage": task.get("stage"),
        "upload_count": task.get("upload_count"),
        "worker_status": task.get("worker_status"),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
    }


def _build_memory_core_payload(task: dict) -> Optional[dict]:
    try:
        return build_task_memory_core_payload(task)
    except Exception:
        return None


def _build_steps_payload(task: dict, *, asset_url_builder: Callable[[str, str], str]) -> Optional[dict]:
    version = str(task.get("version") or "").strip()
    if version not in SUPPORTED_STEPS_VERSIONS:
        return None
    try:
        return build_task_memory_steps_payload(task, asset_url_builder=asset_url_builder)
    except Exception:
        return None


def _build_reviews_payload(task: dict, *, face_review_store: FaceReviewStore) -> dict:
    user_id = str(task.get("user_id") or "").strip()
    if not user_id:
        return {"reviews": {}, "policies": {}}
    return face_review_store.get_task_feedback(
        task_id=str(task.get("task_id") or "").strip(),
        user_id=user_id,
        source_hashes=_task_source_hashes(task),
    )


def _build_artifact_payload(task: dict) -> dict:
    manifest = dict(task.get("asset_manifest") or {})
    files = list(manifest.get("files", []) or [])
    result_artifacts = dict(((task.get("result") or {}).get("artifacts") or {}))
    named_urls = {
        key: value
        for key, value in result_artifacts.items()
        if isinstance(key, str) and key.endswith("_url") and value
    }
    return {
        "artifact_count": len(files),
        "files": files,
        "named_urls": named_urls,
    }


def _build_failure_payload(task: dict) -> Optional[dict]:
    error = str(task.get("error") or "").strip()
    if not error and str(task.get("status") or "").strip() != "failed":
        return None
    return {
        "status": task.get("status"),
        "stage": task.get("stage"),
        "error_message": error or None,
    }


def _build_reduced_payload(event: dict) -> dict:
    reduced = copy.deepcopy(event)
    reduced["snapshot_mode"] = "reduced"
    payload = reduced["payload"]
    payload["artifacts"] = _reduce_artifacts(payload.get("artifacts"))
    payload["steps"] = _reduce_steps(payload.get("steps"))
    return reduced


def _build_minimal_reduced_payload(event: dict) -> dict:
    minimal = copy.deepcopy(event)
    minimal["snapshot_mode"] = "reduced"
    payload = minimal["payload"]
    payload["reviews"] = _reduce_reviews(payload.get("reviews"))
    payload["steps"] = _minimal_steps(payload.get("steps"))
    return minimal


def _reduce_artifacts(artifacts: Any) -> Any:
    if not isinstance(artifacts, dict):
        return artifacts
    files = list(artifacts.get("files", []) or [])
    return {
        "artifact_count": len(files),
        "named_urls": dict(artifacts.get("named_urls") or {}),
        "file_samples": files[:20],
    }


def _reduce_steps(steps: Any) -> Any:
    if not isinstance(steps, dict):
        return steps
    reduced = {key: value for key, value in steps.items() if key != "steps"}
    reduced_steps = {}
    for step_name, step in dict(steps.get("steps") or {}).items():
        if not isinstance(step, dict):
            reduced_steps[step_name] = step
            continue
        item = {
            "status": step.get("status"),
            "substage": step.get("substage"),
            "updated_at": step.get("updated_at"),
            "summary": copy.deepcopy(step.get("summary")),
            "artifacts": copy.deepcopy(step.get("artifacts")),
        }
        if "data" in step:
            item["data_count"] = _count_container(step.get("data"))
        if "attempts" in step:
            item["attempt_count"] = _count_container(step.get("attempts"))
        if "failures" in step:
            item["failure_count"] = _count_container(step.get("failures"))
        reduced_steps[step_name] = item
    reduced["steps"] = reduced_steps
    return reduced


def _minimal_steps(steps: Any) -> Any:
    if not isinstance(steps, dict):
        return steps
    minimized = {key: value for key, value in steps.items() if key != "steps"}
    minimized["steps"] = {
        step_name: {
            "status": step.get("status"),
            "summary": copy.deepcopy(step.get("summary")),
        }
        for step_name, step in dict(steps.get("steps") or {}).items()
        if isinstance(step, dict)
    }
    return minimized


def _reduce_reviews(reviews: Any) -> Any:
    if not isinstance(reviews, dict):
        return reviews
    return {
        "review_count": len(dict(reviews.get("reviews") or {})),
        "policy_count": len(dict(reviews.get("policies") or {})),
    }


def _count_container(value: Any) -> int:
    if isinstance(value, dict):
        return len(value)
    if isinstance(value, list):
        return len(value)
    return 0


def _payload_size(value: dict) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _task_source_hashes(task: dict) -> set[str]:
    hashes = set()
    for upload in _iter_uploads(task.get("uploads")):
        source_hash = upload.get("source_hash")
        if source_hash:
            hashes.add(str(source_hash))
    result = task.get("result") or {}
    face_payload = result.get("face_recognition") or {}
    for image in list(face_payload.get("images", []) or []):
        if not isinstance(image, dict):
            continue
        source_hash = image.get("source_hash")
        if source_hash:
            hashes.add(str(source_hash))
    return hashes


def _iter_uploads(uploads: Any) -> Iterable[dict]:
    if not isinstance(uploads, list):
        return ()
    return (item for item in uploads if isinstance(item, dict))
