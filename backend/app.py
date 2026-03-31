"""
FastAPI backend entrypoint.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime
import fcntl
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, Cookie, Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.artifact_store import ArtifactCatalogStore, build_task_asset_manifest
from backend.auth import (
    AUTH_SESSION_COOKIE_NAME,
    authenticate_response,
    get_current_user,
    login_user,
    logout_current_session,
    register_user,
)
from backend.db import SessionLocal
from backend.face_review_store import FaceReviewStore
from backend.memory_full_retrieval import build_task_memory_core_payload
from backend.memory_step_retrieval import build_task_memory_steps_payload
from backend.models import TaskRecord
from backend.progress_utils import append_terminal_error, append_terminal_info, merge_stage_progress
from backend.query_v1 import QueryEngineV1
from backend.subject_users import ensure_subject_user, get_subject_user, resolve_subject_identity
from backend.task_completion import ensure_completion_outputs
from backend.task_download_bundle import build_task_analysis_bundle, describe_task_downloads
from backend.task_memory_views import (
    build_task_memory_bundle_response,
    build_task_memory_events_response,
    build_task_memory_faces_response,
    build_task_memory_profiles_response,
    build_task_memory_relationships_response,
    build_task_memory_vlm_response,
)
from backend.task_survey_import_events import build_survey_import_event_payload
from backend.task_terminal_events import build_fallback_terminal_event_payload, build_terminal_event_payload
from backend.task_store import TaskStore, normalize_task_options
from backend.upload_utils import (
    UPLOAD_FAILURES_FILENAME,
    is_live_photo_candidate,
    save_upload_original_streamed,
    stored_upload_filename,
    task_asset_path,
    write_upload_failures,
)
from backend.worker_client import WorkerClient
from backend.worker_manager import WorkerManager
from config import (
    ALLOW_SELF_REGISTRATION,
    APP_VERSION,
    APP_ROLE,
    ASSET_URL_PREFIX,
    AVAILABLE_TASK_VERSIONS,
    BACKEND_HOST,
    BACKEND_PORT,
    BACKEND_RELOAD,
    CORS_ALLOW_ORIGINS,
    DEFAULT_NORMALIZE_LIVE_PHOTOS,
    DEFAULT_TASK_VERSION,
    FRONTEND_ORIGIN,
    HIGH_SECURITY_MODE,
    MAX_UPLOAD_PHOTOS,
    MEMORY_QUERY_V1_ENABLED,
    MEMORY_QUERY_V1_SHADOW_COMPARE,
    KAFKA_SURVEY_IMPORT_TOPIC,
    TASK_VERSION_V0325,
    TASK_VERSION_V0327_DB,
    TASK_VERSION_V0327_DB_QUERY,
    TASK_VERSION_V0327_EXP,
    TASKS_DIR,
    WORKER_SHARED_TOKEN,
    normalize_task_version,
)
from memory_module import MemoryQueryService
from services.asset_store import TaskAssetStore
from services.pipeline_service import MemoryPipelineService
from utils import load_json


app = FastAPI(title="Memory Engineering API", version="1.0.0")
logger = logging.getLogger(__name__)
task_store = TaskStore()
asset_store = TaskAssetStore()
artifact_catalog = ArtifactCatalogStore()
worker_manager = WorkerManager()
worker_client = WorkerClient()
face_review_store = FaceReviewStore()

UPLOAD_BATCH_MAX_FILES = 50
TASK_RESULT_RELATIVE_PATH = "output/result.json"
UPLOAD_BATCH_MAX_BYTES = 64 * 1024 * 1024
_UNSET = object()

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ALLOW_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

Path(TASKS_DIR).mkdir(parents=True, exist_ok=True)


class AuthPayload(BaseModel):
    username: str
    password: str


class TaskStartPayload(BaseModel):
    max_photos: Optional[int] = None
    use_cache: bool = False
    normalize_live_photos: bool = DEFAULT_NORMALIZE_LIVE_PHOTOS


class TaskCreatePayload(BaseModel):
    user_id: Optional[str] = None
    version: Optional[str] = None
    normalize_live_photos: bool = DEFAULT_NORMALIZE_LIVE_PHOTOS
    creation_source: Optional[str] = None
    expected_upload_count: Optional[int] = None
    requested_max_photos: Optional[int] = None
    auto_start_on_upload_complete: Optional[bool] = None
    survey_username: Optional[str] = None


class FaceReviewPayload(BaseModel):
    is_inaccurate: Optional[bool] = None
    comment_text: Optional[str] = None


class ImagePolicyPayload(BaseModel):
    is_abandoned: bool


class MemoryQueryPayload(BaseModel):
    question: str
    context_hints: Optional[Dict[str, str]] = None
    time_hint: Optional[str] = None
    answer_shape_hint: Optional[str] = None


class WorkerTerminalUpdatePayload(BaseModel):
    status: str
    stage: str
    progress: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    result_summary: Optional[Dict[str, Any]] = None
    asset_manifest: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    worker_status: Optional[str] = None
    version: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


def _apply_no_store_headers(response: Response) -> Response:
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def _require_internal_worker_token(authorization: str | None = Header(default=None)) -> None:
    if not WORKER_SHARED_TOKEN:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="worker token 未配置")
    expected = f"Bearer {WORKER_SHARED_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的内部访问凭证")


def _load_task_json_artifact(task: dict, relative_path: str) -> Optional[dict]:
    task_dir_value = str(task.get("task_dir") or "").strip()
    if task_dir_value:
        local_path = Path(task_dir_value) / relative_path
        if local_path.exists():
            try:
                payload = load_json(str(local_path))
            except Exception:
                logger.exception("Failed to load task artifact from local file task_id=%s path=%s", task.get("task_id"), local_path)
                return None
            return payload if isinstance(payload, dict) else None

    task_id = str(task.get("task_id") or "").strip()
    if not task_id or not asset_store.enabled:
        return None

    try:
        if not asset_store.has_object(task_id, relative_path):
            return None
        body, _ = asset_store.read_bytes(task_id, relative_path)
        payload = json.loads(body.decode("utf-8"))
    except Exception:
        logger.exception("Failed to load task artifact from object storage task_id=%s path=%s", task_id, relative_path)
        return None
    return payload if isinstance(payload, dict) else None


def _hydrate_task_payloads(task: dict) -> dict:
    hydrated = copy.deepcopy(task)
    artifact_result = _load_task_json_artifact(hydrated, TASK_RESULT_RELATIVE_PATH)
    if isinstance(artifact_result, dict):
        hydrated["result"] = artifact_result
    return hydrated


def _remote_worker_mode_enabled() -> bool:
    return APP_ROLE == "control-plane" and worker_manager.enabled and worker_client.enabled


def _task_uploads(task: dict) -> List[dict]:
    uploads = task.get("uploads") or []
    return uploads if isinstance(uploads, list) else []


def _task_source_hashes(task: dict) -> set[str]:
    hashes = set()
    for upload in _task_uploads(task):
        source_hash = upload.get("source_hash")
        if source_hash:
            hashes.add(source_hash)
    result = task.get("result") or {}
    face_payload = result.get("face_recognition") or {}
    for image in face_payload.get("images", []):
        source_hash = image.get("source_hash")
        if source_hash:
            hashes.add(source_hash)
    return hashes


def _get_task_by_id(task_id: str) -> Optional[dict]:
    with SessionLocal() as session:
        record = session.get(TaskRecord, task_id)
        if record is None:
            return None
        return task_store._serialize(record)


def _value_or_unset(payload: dict, key: str):
    return payload[key] if key in payload else _UNSET


def _build_task_asset_manifest_safely(task_id: str, task_dir: Path) -> Optional[dict]:
    try:
        return build_task_asset_manifest(task_id, task_dir, asset_store)
    except Exception:
        logger.exception("Failed to build asset manifest for task_id=%s", task_id)
        return None


def _merge_task_snapshot(task: dict, **updates) -> dict:
    snapshot = copy.deepcopy(task)
    for key, value in updates.items():
        if value is _UNSET:
            continue
        snapshot[key] = value
    snapshot["updated_at"] = datetime.now().isoformat()
    return snapshot


def _normalize_survey_username(value: str | None) -> str | None:
    username = str(value or "").strip()
    if not username:
        return None
    if "/" in username or "\\" in username:
        raise HTTPException(status_code=400, detail="survey_username 不能包含路径分隔符")
    return username


def _resolve_subject_ref(
    current_user: dict,
    *,
    subject_user_id: str | None = None,
    survey_username: str | None = None,
) -> dict:
    normalized_subject_user_id = str(subject_user_id or "").strip()
    if normalized_subject_user_id:
        subject = get_subject_user(normalized_subject_user_id)
        if subject is None:
            subject = ensure_subject_user(normalized_subject_user_id)
        return {
            "user_id": normalized_subject_user_id,
            "operator_user_id": None if normalized_subject_user_id == current_user["user_id"] else current_user["user_id"],
            "subject_username": subject.get("username"),
            "subject_source": subject.get("source"),
        }
    return resolve_subject_identity(
        operator_user_id=current_user["user_id"],
        operator_username=current_user.get("username"),
        subject_username=survey_username,
    )
def _build_outbox_events(task: dict) -> list[dict]:
    if not task_store.outbox_store.enabled:
        return []
    hydrated = _hydrate_task_payloads(copy.deepcopy(task))
    events: list[dict] = []
    try:
        terminal_event = build_terminal_event_payload(
            hydrated,
            face_review_store=face_review_store,
            asset_url_builder=lambda resolved_task_id, relative_path: asset_store.asset_url(resolved_task_id, relative_path),
        )
    except Exception as exc:
        logger.exception("Failed to build full terminal event payload for task_id=%s", task.get("task_id"))
        terminal_event = build_fallback_terminal_event_payload(hydrated, reason=str(exc))
    events.append({"event": terminal_event, "topic": None})

    try:
        survey_event = build_survey_import_event_payload(hydrated)
    except Exception:
        logger.exception("Failed to build survey import event payload for task_id=%s", task.get("task_id"))
        survey_event = None
    if survey_event is not None:
        events.append({"event": survey_event, "topic": KAFKA_SURVEY_IMPORT_TOPIC})
    return events


def _finalize_task_terminal_state(
    task_id: str,
    user_id: Optional[str],
    *,
    status: str,
    stage: str,
    progress: dict | None,
    result: Any = _UNSET,
    event_result: Any = _UNSET,
    error: Any = _UNSET,
    result_summary: Any = _UNSET,
    asset_manifest: Any = _UNSET,
    event_asset_manifest: Any = _UNSET,
    worker_status: Any = _UNSET,
    last_worker_sync_at: Any = _UNSET,
    version: Any = _UNSET,
    options: Any = _UNSET,
) -> dict:
    stored_task = _get_task_by_id(task_id)
    if stored_task is None:
        raise KeyError(f"任务不存在: {task_id}")

    snapshot_result = event_result if event_result is not _UNSET else result
    snapshot_asset_manifest = asset_manifest if asset_manifest is not _UNSET else event_asset_manifest
    snapshot = _merge_task_snapshot(
        stored_task,
        status=status,
        stage=stage,
        progress=progress,
        result=snapshot_result,
        error=error,
        result_summary=result_summary,
        asset_manifest=snapshot_asset_manifest,
        worker_status=worker_status,
        version=version,
        options=options,
    )
    outbox_events = _build_outbox_events(snapshot)
    finalize_kwargs = {
        "status": status,
        "stage": stage,
        "progress": progress,
        "outbox_events": outbox_events,
    }
    if result is not _UNSET:
        finalize_kwargs["result"] = result
    if error is not _UNSET:
        finalize_kwargs["error"] = error
    if result_summary is not _UNSET:
        finalize_kwargs["result_summary"] = result_summary
    if asset_manifest is not _UNSET:
        finalize_kwargs["asset_manifest"] = asset_manifest
    if worker_status is not _UNSET:
        finalize_kwargs["worker_status"] = worker_status
    if last_worker_sync_at is not _UNSET:
        finalize_kwargs["last_worker_sync_at"] = last_worker_sync_at
    if version is not _UNSET:
        finalize_kwargs["version"] = version
    if options is not _UNSET:
        finalize_kwargs["options"] = options

    finalized = task_store.finalize_task(task_id, **finalize_kwargs)
    if user_id and asset_manifest is not _UNSET and isinstance(asset_manifest, dict):
        artifact_catalog.replace_task_artifacts(task_id, user_id, asset_manifest)
    return finalized


def _existing_upload_failures(task_dir: Path) -> List[dict]:
    payload = load_json(str(task_dir / UPLOAD_FAILURES_FILENAME))
    failures = payload.get("failures", [])
    return failures if isinstance(failures, list) else []


def _write_merged_upload_failures(task_dir: Path, failures: List[dict]) -> None:
    merged_failures = _existing_upload_failures(task_dir)
    merged_failures.extend(failures)
    write_upload_failures(task_dir, merged_failures)
    if asset_store.enabled:
        failures_path = task_dir / UPLOAD_FAILURES_FILENAME
        if failures_path.exists():
            asset_store.upload_file(task_dir.name, UPLOAD_FAILURES_FILENAME, failures_path)


def _save_upload_batch(task_id: str, task_dir: Path, files: List[UploadFile], start_index: int) -> tuple[List[dict], List[dict], int]:
    uploads_dir = task_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    saved_files: List[dict] = []
    upload_failures: List[dict] = []
    total_bytes = 0

    for offset, upload in enumerate(files):
        index = start_index + offset
        image_id = f"photo_{index:03d}"
        stored_name = stored_upload_filename(upload.filename or "", index)
        destination = uploads_dir / stored_name
        try:
            image_info = save_upload_original_streamed(upload, destination)
            total_bytes += int(destination.stat().st_size)
            original_asset_path = task_asset_path("uploads", stored_name)
            if asset_store.enabled:
                asset_store.upload_file(task_id, original_asset_path, destination)
            saved_files.append(
                {
                    "image_id": image_id,
                    "filename": upload.filename or stored_name,
                    "stored_filename": stored_name,
                    "path": original_asset_path,
                    "url": asset_store.asset_url(task_id, original_asset_path),
                    "preview_url": None,
                    "is_live_photo_candidate": is_live_photo_candidate(upload.filename or stored_name, image_info.get("content_type")),
                    **image_info,
                }
            )
        except Exception as exc:
            destination.unlink(missing_ok=True)
            upload_failures.append(
                {
                    "image_id": image_id,
                    "filename": upload.filename or stored_name,
                    "path": str(destination),
                    "step": "upload",
                    "error": f"保存原始上传文件失败: {exc}",
                }
            )

    return saved_files, upload_failures, total_bytes


def _find_face(task: dict, face_id: str) -> Optional[Dict]:
    result = task.get("result") or {}
    face_payload = result.get("face_recognition") or {}
    for image in face_payload.get("images", []):
        for face in image.get("faces", []):
            if face.get("face_id") == face_id:
                return {
                    "face": face,
                    "image": image,
                    "person_id": face.get("person_id"),
                    "image_id": image.get("image_id"),
                    "source_hash": face.get("source_hash") or image.get("source_hash"),
                }
    return None


def _find_image(task: dict, image_id: str) -> Optional[Dict]:
    result = task.get("result") or {}
    face_payload = result.get("face_recognition") or {}
    for image in face_payload.get("images", []):
        if image.get("image_id") == image_id:
            return image
    for upload in _task_uploads(task):
        if upload.get("image_id") == image_id:
            return upload
    return None


def _merge_feedback(task: dict, user_id: str) -> dict:
    hydrated = copy.deepcopy(task)
    result = hydrated.get("result")
    if not isinstance(result, dict):
        return hydrated

    face_payload = result.get("face_recognition")
    if not isinstance(face_payload, dict):
        return hydrated

    feedback = face_review_store.get_task_feedback(
        task_id=hydrated["task_id"],
        user_id=user_id,
        source_hashes=_task_source_hashes(hydrated),
    )
    reviews = feedback.get("reviews", {})
    policies = feedback.get("policies", {})

    images = face_payload.setdefault("images", [])
    known_image_ids = {image.get("image_id") for image in images}
    for upload in _task_uploads(hydrated):
        image_id = upload.get("image_id")
        if not image_id or image_id in known_image_ids:
            continue
        policy = policies.get(upload.get("source_hash") or "")
        images.append(
            {
                "image_id": image_id,
                "filename": upload.get("filename"),
                "source_hash": upload.get("source_hash"),
                "timestamp": None,
                "status": "abandoned_by_policy" if policy and policy.get("is_abandoned") else "skipped",
                "detection_seconds": 0.0,
                "embedding_seconds": 0.0,
                "original_image_url": upload.get("url"),
                "display_image_url": upload.get("preview_url") or upload.get("url"),
                "boxed_image_url": None,
                "compressed_image_url": None,
                "location": None,
                "face_count": 0,
                "faces": [],
                "failures": [],
                "is_abandoned": bool(policy and policy.get("is_abandoned")),
            }
        )

    for image in images:
        policy = policies.get(image.get("source_hash") or "")
        image["is_abandoned"] = bool(policy and policy.get("is_abandoned"))
        for face in image.get("faces", []):
            review = reviews.get(face.get("face_id"), {})
            face["is_inaccurate"] = bool(review.get("is_inaccurate"))
            face["comment_text"] = review.get("comment_text", "")

    for group in face_payload.get("person_groups", []):
        for image in group.get("images", []):
            review = reviews.get(image.get("face_id"), {})
            policy = policies.get(image.get("source_hash") or "")
            image["is_inaccurate"] = bool(review.get("is_inaccurate"))
            image["comment_text"] = review.get("comment_text", "")
            image["is_abandoned"] = bool(policy and policy.get("is_abandoned"))

    return hydrated


def _hydrate_task(task: dict, user_id: str) -> dict:
    synced = _hydrate_task_payloads(_sync_task_from_worker(task, user_id))
    return _merge_feedback(synced, user_id)


def _strip_bootstrap_fields(payload):
    if isinstance(payload, dict):
        cleaned = {}
        for key, value in payload.items():
            if isinstance(key, str) and (
                key.startswith("bootstrap_")
                or key == "bootstrap_source"
                or key == "bootstrap_source_task_id"
            ):
                continue
            cleaned[key] = _strip_bootstrap_fields(value)
        return cleaned
    if isinstance(payload, list):
        return [_strip_bootstrap_fields(item) for item in payload]
    return payload


def _strip_heavy_memory_snapshot_fields(memory: dict) -> dict:
    stripped = copy.deepcopy(memory)
    for key in (
        "vp1_observations",
        "lp1_events",
        "lp1_batches",
        "lp2_relationships",
        "lp3_profile",
    ):
        stripped.pop(key, None)
    return stripped


def _sanitize_task_for_client(task: dict) -> dict:
    sanitized = copy.deepcopy(task)
    downloads = describe_task_downloads(sanitized)
    if downloads:
        sanitized["downloads"] = downloads
    if isinstance(sanitized.get("result_summary"), dict):
        sanitized["result_summary"] = _strip_bootstrap_fields(sanitized["result_summary"])
    result = sanitized.get("result")
    if isinstance(result, dict):
        memory = result.get("memory")
        if isinstance(memory, dict):
            result["memory"] = _strip_heavy_memory_snapshot_fields(_strip_bootstrap_fields(memory))
    return sanitized


def _sync_task_from_worker(task: dict, user_id: str) -> dict:
    if not _remote_worker_mode_enabled():
        return task
    if not task.get("worker_instance_id") or task.get("delete_state") == "requested":
        return task

    instance = worker_manager.describe_worker(task["worker_instance_id"])
    if instance is None:
        if task.get("worker_status") != "terminated":
            task_store.update_worker_state(task["task_id"], worker_status="terminated")
        refreshed = task_store.get_task(task["task_id"], user_id=user_id)
        return refreshed or task

    worker_updates = {}
    if instance.private_ip and instance.private_ip != task.get("worker_private_ip"):
        worker_updates["worker_private_ip"] = instance.private_ip
    if instance.expires_at:
        worker_updates["expires_at"] = instance.expires_at
    if worker_updates or instance.state != task.get("worker_status"):
        task_store.update_worker_state(
            task["task_id"],
            worker_status=instance.state,
            **worker_updates,
        )
        refreshed = task_store.get_task(task["task_id"], user_id=user_id)
        task = refreshed or task

    private_ip = instance.private_ip or task.get("worker_private_ip")
    if instance.state != "running" or not private_ip:
        return task

    try:
        remote_payload = worker_client.fetch_status(private_ip, task["task_id"])
    except Exception:
        return task

    terminal_status = str(remote_payload.get("status") or "").strip()
    if terminal_status in {"completed", "failed"}:
        persisted_asset_manifest = (
            _value_or_unset(remote_payload, "asset_manifest") if terminal_status == "completed" else _UNSET
        )
        event_asset_manifest = (
            _value_or_unset(remote_payload, "asset_manifest") if terminal_status != "completed" else _UNSET
        )
        refreshed = _finalize_task_terminal_state(
            task["task_id"],
            user_id,
            status=terminal_status,
            stage=str(remote_payload.get("stage") or terminal_status).strip() or terminal_status,
            progress=remote_payload.get("progress"),
            result=_value_or_unset(remote_payload, "result"),
            error=_value_or_unset(remote_payload, "error"),
            result_summary=_value_or_unset(remote_payload, "result_summary"),
            asset_manifest=persisted_asset_manifest,
            event_asset_manifest=event_asset_manifest,
            worker_status=remote_payload.get("worker_status", instance.state),
            last_worker_sync_at=datetime.now(),
            version=_value_or_unset(remote_payload, "version"),
            options=_value_or_unset(remote_payload, "options"),
        )
        return refreshed or task

    updates = {}
    for key in ("status", "stage", "progress", "result", "error", "version", "options"):
        if key in remote_payload:
            updates[key] = remote_payload[key]
    if "uploads" in remote_payload and not task.get("uploads"):
        updates["uploads"] = remote_payload["uploads"]
    task_store.update_worker_state(
        task["task_id"],
        worker_status=remote_payload.get("worker_status", instance.state),
        **updates,
    )
    if "result_summary" in remote_payload or "asset_manifest" in remote_payload:
        task_store.set_result_summary(
            task["task_id"],
            remote_payload.get("result_summary"),
            remote_payload.get("asset_manifest"),
        )
        if remote_payload.get("asset_manifest"):
            artifact_catalog.replace_task_artifacts(
                task["task_id"],
                user_id,
                remote_payload["asset_manifest"],
            )
    refreshed = task_store.get_task(task["task_id"], user_id=user_id)
    return refreshed or task


def _remote_upload_entries(task: dict, user_id: str) -> tuple[List[dict], int]:
    entries: List[dict] = []
    skipped = 0
    task_dir = task_store.task_dir(task["task_id"])
    for upload in _task_uploads(task):
        source_hash = upload.get("source_hash")
        if face_review_store.is_image_abandoned(user_id, source_hash):
            skipped += 1
            continue
        relative_path = upload.get("path")
        if not relative_path:
            continue
        local_path = task_dir / relative_path
        if not local_path.exists():
            raise FileNotFoundError(f"任务文件不存在: {local_path}")
        entries.append(
            {
                "filename": os.path.basename(upload.get("filename") or local_path.name),
                "content_type": upload.get("content_type") or "application/octet-stream",
                "local_path": local_path,
                "size": int(local_path.stat().st_size),
            }
        )
    return entries, skipped


def _chunk_remote_upload_entries(entries: List[dict]) -> List[List[dict]]:
    batches: List[List[dict]] = []
    current: List[dict] = []
    current_bytes = 0
    for entry in entries:
        next_size = current_bytes + int(entry["size"])
        if current and (len(current) >= UPLOAD_BATCH_MAX_FILES or next_size > UPLOAD_BATCH_MAX_BYTES):
            batches.append(current)
            current = []
            current_bytes = 0
        current.append(entry)
        current_bytes += int(entry["size"])
    if current:
        batches.append(current)
    return batches


def _run_pipeline_task(
    task_id: str,
    user_id: Optional[str],
    max_photos: int,
    use_cache: bool,
    task_version: str,
    task_options: dict | None,
):
    task_dir = task_store.task_dir(task_id)
    progress_state: Dict[str, object] = {
        "current_stage": "starting",
        "updated_at": datetime.now().isoformat(),
        "stages": {
            "starting": {
                "message": "准备启动推理任务",
                "updated_at": datetime.now().isoformat(),
            }
        },
    }
    progress_state = append_terminal_info(
        progress_state,
        stage="starting",
        message="准备启动推理任务",
    )

    def progress_callback(stage: str, payload: dict):
        nonlocal progress_state
        progress_state = merge_stage_progress(progress_state, stage, payload or {})
        task_store.update_task(task_id, status="running", stage=stage, progress=copy.deepcopy(progress_state))

    try:
        task_store.update_task(
            task_id,
            status="running",
            stage="starting",
            progress=copy.deepcopy(progress_state),
            result=None,
            result_summary=None,
            error=None,
        )
        service = MemoryPipelineService(
            task_id=task_id,
            task_dir=str(task_dir),
            asset_store=asset_store,
            user_id=user_id,
            face_review_store=face_review_store,
            task_version=task_version,
            task_options=task_options,
        )
        result = service.run(
            max_photos=max_photos,
            use_cache=use_cache,
            progress_callback=progress_callback,
        )
        ensure_completion_outputs(task_version, result)
        manifest = build_task_asset_manifest(task_id, task_dir, asset_store)
        _finalize_task_terminal_state(
            task_id,
            user_id,
            status="completed",
            stage="completed",
            progress=copy.deepcopy(
                append_terminal_info(
                    progress_state,
                    stage="completed",
                    message="任务执行完成",
                )
            ),
            error=None,
            result_summary=result.get("summary"),
            asset_manifest=manifest,
            version=task_version,
            options=task_options,
            event_result=result,
        )
    except Exception as exc:
        logger.exception("Pipeline task failed for task_id=%s version=%s", task_id, task_version)
        failed_progress = append_terminal_error(
            progress_state,
            stage="failed",
            error=str(exc),
        )
        _finalize_task_terminal_state(
            task_id,
            user_id,
            status="failed",
            stage="failed",
            progress=copy.deepcopy(failed_progress),
            error=str(exc),
            event_asset_manifest=_build_task_asset_manifest_safely(task_id, task_dir),
            version=task_version,
            options=task_options,
        )


def _resolve_task_version_and_options(
    version: str | None,
    normalize_live_photos: bool = DEFAULT_NORMALIZE_LIVE_PHOTOS,
) -> tuple[str, dict]:
    try:
        task_version = normalize_task_version(version)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    task_options = normalize_task_options({"normalize_live_photos": normalize_live_photos})
    return task_version, task_options


def _create_task_record(
    current_user: dict,
    version: str | None,
    normalize_live_photos: bool,
    *,
    creation_source: str,
    subject_user_id: str | None = None,
    expected_upload_count: int | None = None,
    requested_max_photos: int | None = None,
    auto_start_on_upload_complete: bool | None = None,
    survey_username: str | None = None,
) -> dict:
    task_version, task_options = _resolve_task_version_and_options(version, normalize_live_photos)
    subject_ref = _resolve_subject_ref(
        current_user,
        subject_user_id=subject_user_id,
        survey_username=_normalize_survey_username(survey_username),
    )
    normalized_survey_username = _normalize_survey_username(subject_ref.get("subject_username"))
    if task_version == TASK_VERSION_V0327_DB_QUERY and not normalized_survey_username:
        raise HTTPException(status_code=400, detail="v0327-db-query 任务必须提供 user_id 或 survey_username")
    option_payload = {
        "normalize_live_photos": bool(task_options.get("normalize_live_photos", normalize_live_photos)),
        "creation_source": creation_source,
        "expected_upload_count": expected_upload_count,
        "requested_max_photos": requested_max_photos,
        "survey_username": normalized_survey_username,
        "subject_user_id": str(subject_ref.get("user_id") or ""),
        "operator_user_id": str(subject_ref.get("operator_user_id") or "").strip() or None,
    }
    if auto_start_on_upload_complete is not None:
        option_payload["auto_start_on_upload_complete"] = auto_start_on_upload_complete
    task_options = normalize_task_options(option_payload)
    task_id = uuid.uuid4().hex
    task_store.create_task(
        task_id,
        upload_count=0,
        user_id=str(subject_ref.get("user_id") or ""),
        operator_user_id=str(subject_ref.get("operator_user_id") or "").strip() or None,
        version=task_version,
        options=task_options,
        status="draft",
        stage="draft",
    )
    return {
        "task_id": task_id,
        "user_id": str(subject_ref.get("user_id") or ""),
        "operator_user_id": str(subject_ref.get("operator_user_id") or "").strip() or None,
        "version": task_version,
        "options": task_options,
        "status": "draft",
        "stage": "draft",
        "upload_count": 0,
        "task_url": f"/api/tasks/{task_id}",
    }


def _save_upload_collection(task_id: str, task_dir: Path, files: List[UploadFile]) -> tuple[List[dict], List[dict]]:
    saved_files: List[dict] = []
    upload_failures: List[dict] = []
    start_index = 1
    for offset in range(0, len(files), UPLOAD_BATCH_MAX_FILES):
        batch = files[offset: offset + UPLOAD_BATCH_MAX_FILES]
        batch_saved, batch_failures, _ = _save_upload_batch(task_id, task_dir, batch, start_index)
        saved_files.extend(batch_saved)
        upload_failures.extend(batch_failures)
        start_index += len(batch)
    return saved_files, upload_failures


def _merge_upload_task_options(
    current_options: dict | None,
    *,
    creation_source: str | None = None,
    expected_upload_count: int | None = None,
    requested_max_photos: int | None = None,
    auto_start_on_upload_complete: bool | None = None,
    survey_username: str | None = None,
) -> dict:
    merged = dict(current_options or {})
    if creation_source:
        merged["creation_source"] = creation_source
    if expected_upload_count is not None:
        merged["expected_upload_count"] = expected_upload_count
    if requested_max_photos is not None:
        merged["requested_max_photos"] = requested_max_photos
    normalized_survey_username = _normalize_survey_username(survey_username)
    if normalized_survey_username is not None:
        merged["survey_username"] = normalized_survey_username

    effective_creation_source = str(merged.get("creation_source") or "manual").strip().lower()
    effective_expected_upload_count = merged.get("expected_upload_count")
    if auto_start_on_upload_complete is not None:
        merged["auto_start_on_upload_complete"] = auto_start_on_upload_complete
    elif (
        effective_creation_source in {"manual", "directory"}
        and effective_expected_upload_count is not None
        and not bool(merged.get("auto_start_on_upload_complete"))
    ):
        merged["auto_start_on_upload_complete"] = True

    return normalize_task_options(merged)


@contextmanager
def _task_upload_lock(task_id: str):
    task_dir = task_store.task_dir(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    lock_path = task_dir / ".upload-batches.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield task_dir
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _maybe_auto_start_after_upload(
    task_id: str,
    user_id: str,
    updated_task: dict,
    *,
    task_dir: Path,
    background_tasks: BackgroundTasks,
) -> dict | None:
    options = normalize_task_options(updated_task.get("options"))
    expected_upload_count = options.get("expected_upload_count")
    should_auto_start = bool(options.get("auto_start_on_upload_complete"))
    if not should_auto_start or not expected_upload_count:
        return None
    successful_upload_count = int(updated_task.get("upload_count") or 0)
    failed_upload_count = len(_existing_upload_failures(task_dir))
    attempted_upload_count = successful_upload_count + failed_upload_count
    if attempted_upload_count < int(expected_upload_count):
        return None
    if successful_upload_count <= 0:
        task_store.update_task(task_id, status="failed", stage="failed", error="照片全部上传失败，无法启动任务")
        raise HTTPException(status_code=400, detail="照片全部上传失败，无法启动任务")

    return _start_task_impl(
        task_id,
        user_id,
        max_photos=options.get("requested_max_photos"),
        use_cache=False,
        normalize_live_photos=bool(options.get("normalize_live_photos", DEFAULT_NORMALIZE_LIVE_PHOTOS)),
        background_tasks=background_tasks,
    )


def _start_task_impl(
    task_id: str,
    operator_user_id: str,
    *,
    max_photos: Optional[int],
    use_cache: bool,
    normalize_live_photos: bool,
    background_tasks: BackgroundTasks,
) -> dict:
    task = task_store.get_task(task_id, user_id=operator_user_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task["status"] in {"queued", "running", "completed"}:
        raise HTTPException(status_code=409, detail="当前任务已开始或已完成")

    uploads = _task_uploads(task)
    if not uploads:
        raise HTTPException(status_code=400, detail="请先上传图片，再开始处理")

    requested_max = max_photos or len(uploads)
    if requested_max < 1 or requested_max > MAX_UPLOAD_PHOTOS:
        raise HTTPException(status_code=400, detail=f"max_photos 必须在 1 到 {MAX_UPLOAD_PHOTOS} 之间")
    effective_max_photos = min(requested_max, len(uploads), MAX_UPLOAD_PHOTOS)
    task_version = task.get("version") or DEFAULT_TASK_VERSION
    task_options = normalize_task_options(
        {
            **(task.get("options") or {}),
            "normalize_live_photos": normalize_live_photos,
        }
    )
    task_store.update_task(task_id, options=task_options)

    if _remote_worker_mode_enabled():
        launched_worker = None
        try:
            active_entries, skipped_uploads = _remote_upload_entries(task, operator_user_id)
            if not active_entries:
                raise HTTPException(status_code=400, detail="当前任务没有可处理的图片，可能都已被标记为 abandon")

            launched_worker = worker_manager.launch_worker(task_id)
            ready_worker = worker_manager.wait_until_ready(launched_worker.instance_id)
            ready_worker.expires_at = launched_worker.expires_at
            task_store.attach_worker(
                task_id,
                ready_worker.instance_id,
                ready_worker.private_ip,
                ready_worker.expires_at,
                worker_status=ready_worker.state,
            )
            if not ready_worker.private_ip:
                raise RuntimeError("worker 未返回私网地址")

            worker_client.wait_for_health(ready_worker.private_ip)
            for batch in _chunk_remote_upload_entries(active_entries):
                worker_client.upload_batch(
                    ready_worker.private_ip,
                    task_id,
                    [
                        {
                            "filename": item["filename"],
                            "content_type": item["content_type"],
                            "payload": item["local_path"].read_bytes(),
                        }
                        for item in batch
                    ],
                    version=task_version,
                )
            start_payload = worker_client.start_task(
                ready_worker.private_ip,
                task_id,
                max_photos=effective_max_photos,
                use_cache=use_cache,
                version=task_version,
                options=task_options,
            )
            task_store.update_worker_state(
                task_id,
                worker_status=start_payload.get("worker_status", ready_worker.state),
                version=start_payload.get("version", task_version),
                options=start_payload.get("options", task_options),
                status=start_payload.get("status", "queued"),
                stage=start_payload.get("stage", "queued"),
                progress=start_payload.get("progress"),
                error=None,
            )
            return {
                "task_id": task_id,
                "status": start_payload.get("status", "queued"),
                "stage": start_payload.get("stage", "queued"),
                "version": start_payload.get("version", task_version),
                "options": start_payload.get("options", task_options),
                "upload_count": len(uploads),
                "skipped_uploads": skipped_uploads,
                "max_photos": effective_max_photos,
                "task_url": f"/api/tasks/{task_id}",
            }
        except HTTPException as exc:
            if launched_worker is not None:
                worker_manager.terminate_worker(launched_worker.instance_id)
            task_store.update_task(task_id, status="failed", stage="failed", error=str(exc.detail))
            raise
        except Exception as exc:
            if launched_worker is not None:
                worker_manager.terminate_worker(launched_worker.instance_id)
            task_store.update_task(task_id, status="failed", stage="failed", error=f"创建 worker 任务失败: {exc}")
            raise HTTPException(status_code=502, detail=f"创建 worker 任务失败: {exc}") from exc

    task_store.update_task(task_id, status="queued", stage="queued", progress=None, error=None)
    background_tasks.add_task(
        _run_pipeline_task,
        task_id,
        str(task.get("user_id") or "").strip() or operator_user_id,
        effective_max_photos,
        use_cache,
        task_version,
        task_options,
    )
    return {
        "task_id": task_id,
        "version": task_version,
        "options": task_options,
        "status": "queued",
        "stage": "queued",
        "upload_count": len(uploads),
        "max_photos": effective_max_photos,
        "task_url": f"/api/tasks/{task_id}",
    }


@app.get("/api/health")
def healthcheck(response: Response):
    _apply_no_store_headers(response)
    return {
        "status": "ok",
        "app_version": APP_VERSION,
        "app_role": APP_ROLE,
        "frontend_origin": FRONTEND_ORIGIN,
        "max_upload_photos": MAX_UPLOAD_PHOTOS,
        "default_task_version": DEFAULT_TASK_VERSION,
        "available_task_versions": list(AVAILABLE_TASK_VERSIONS),
        "self_registration_enabled": ALLOW_SELF_REGISTRATION,
        "high_security_mode": HIGH_SECURITY_MODE,
        "asset_url_prefix": ASSET_URL_PREFIX,
        "object_storage_enabled": asset_store.enabled,
        "object_storage_bucket": asset_store.bucket or None,
        "worker_orchestration_enabled": _remote_worker_mode_enabled(),
    }


@app.post("/internal/tasks/{task_id}/terminal-update")
def receive_worker_terminal_update(
    task_id: str,
    payload: WorkerTerminalUpdatePayload,
    response: Response,
    _: None = Depends(_require_internal_worker_token),
):
    _apply_no_store_headers(response)
    terminal_status = str(payload.status or "").strip()
    if terminal_status not in {"completed", "failed"}:
        raise HTTPException(status_code=400, detail="仅支持 terminal 状态回调")

    stored_task = _get_task_by_id(task_id)
    if stored_task is None:
        raise HTTPException(status_code=404, detail="任务不存在")

    payload_dict = payload.model_dump(exclude_unset=True) if hasattr(payload, "model_dump") else payload.dict(exclude_unset=True)
    persisted_asset_manifest = (
        _value_or_unset(payload_dict, "asset_manifest") if terminal_status == "completed" else _UNSET
    )
    event_asset_manifest = (
        _value_or_unset(payload_dict, "asset_manifest") if terminal_status != "completed" else _UNSET
    )
    finalized = _finalize_task_terminal_state(
        task_id,
        stored_task.get("user_id"),
        status=terminal_status,
        stage=str(payload.stage or terminal_status).strip() or terminal_status,
        progress=payload.progress,
        result=_value_or_unset(payload_dict, "result"),
        error=_value_or_unset(payload_dict, "error"),
        result_summary=_value_or_unset(payload_dict, "result_summary"),
        asset_manifest=persisted_asset_manifest,
        event_asset_manifest=event_asset_manifest,
        worker_status=_value_or_unset(payload_dict, "worker_status"),
        last_worker_sync_at=datetime.now(),
        version=_value_or_unset(payload_dict, "version"),
        options=_value_or_unset(payload_dict, "options"),
    )
    return {
        "status": "ok",
        "task_id": task_id,
        "task_status": finalized.get("status"),
        "event_enqueued": bool(task_store.outbox_store.enabled),
    }


@app.post("/api/auth/register")
def register(payload: AuthPayload, response: Response):
    _apply_no_store_headers(response)
    if not ALLOW_SELF_REGISTRATION:
        raise HTTPException(status_code=403, detail="注册已关闭")
    user = register_user(payload.username, payload.password)
    _, session_token = login_user(payload.username, payload.password)
    return authenticate_response(response, user, session_token)


@app.post("/api/auth/login")
def login(payload: AuthPayload, response: Response):
    _apply_no_store_headers(response)
    user, session_token = login_user(payload.username, payload.password)
    return authenticate_response(response, user, session_token)


@app.get("/api/auth/me")
def current_user(response: Response, user: dict = Depends(get_current_user)):
    _apply_no_store_headers(response)
    return {"user": user}


@app.post("/api/auth/logout")
def logout(
    response: Response,
    session_token: str | None = Cookie(default=None, alias=AUTH_SESSION_COOKIE_NAME),
    user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    del user
    logout_current_session(session_token, response)
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Memory Engineering API",
        "role": APP_ROLE,
        "healthcheck": "/api/health",
        "tasks_endpoint": "/api/tasks",
        "photo_collection_ingest_endpoint": "/api/tasks/ingest",
        "assets_prefix": ASSET_URL_PREFIX,
        "auth_me": "/api/auth/me",
    }


@app.post("/api/tasks")
def create_task(
    response: Response,
    payload: TaskCreatePayload | None = None,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    task_payload = _create_task_record(
        current_user,
        payload.version if payload else None,
        payload.normalize_live_photos if payload else DEFAULT_NORMALIZE_LIVE_PHOTOS,
        creation_source=(payload.creation_source if payload and payload.creation_source else "manual"),
        subject_user_id=payload.user_id if payload else None,
        expected_upload_count=payload.expected_upload_count if payload else None,
        requested_max_photos=payload.requested_max_photos if payload else None,
        auto_start_on_upload_complete=payload.auto_start_on_upload_complete if payload else None,
        survey_username=payload.survey_username if payload else None,
    )
    return task_payload


@app.post("/api/tasks/ingest")
async def ingest_photo_collection(
    response: Response,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    user_id: Optional[str] = Form(default=None),
    version: Optional[str] = Form(default=None),
    max_photos: Optional[int] = Form(default=None),
    use_cache: bool = Form(default=False),
    normalize_live_photos: bool = Form(default=DEFAULT_NORMALIZE_LIVE_PHOTOS),
    survey_username: Optional[str] = Form(default=None),
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    if not files:
        raise HTTPException(status_code=400, detail="至少上传一张图片")
    if len(files) > MAX_UPLOAD_PHOTOS:
        raise HTTPException(status_code=400, detail=f"单次照片集最多上传 {MAX_UPLOAD_PHOTOS} 张图片")
    if max_photos is not None and (max_photos < 1 or max_photos > MAX_UPLOAD_PHOTOS):
        raise HTTPException(status_code=400, detail=f"max_photos 必须在 1 到 {MAX_UPLOAD_PHOTOS} 之间")

    task_payload = _create_task_record(
        current_user,
        version,
        normalize_live_photos,
        creation_source="api",
        subject_user_id=user_id,
        expected_upload_count=len(files),
        requested_max_photos=max_photos,
        auto_start_on_upload_complete=False,
        survey_username=survey_username,
    )
    task_id = task_payload["task_id"]
    task_dir = task_store.task_dir(task_id)

    try:
        saved_files, upload_failures = _save_upload_collection(task_id, task_dir, files)
    except Exception as exc:
        task_store.update_task(task_id, status="failed", stage="failed", error=f"照片集接收失败: {exc}")
        raise HTTPException(status_code=500, detail=f"照片集接收失败: {exc}") from exc
    finally:
        for upload in files:
            await upload.close()

    _write_merged_upload_failures(task_dir, upload_failures)
    if not saved_files:
        task_store.update_task(task_id, status="failed", stage="failed", error="照片集接收失败，没有成功保存任何图片")
        raise HTTPException(status_code=400, detail="照片集接收失败，没有成功保存任何图片")

    updated_task = task_store.append_uploads(task_id, saved_files, status="uploading", stage="uploading")
    manifest = build_task_asset_manifest(task_id, task_dir, asset_store)
    task_store.update_task(task_id, asset_manifest=manifest)
    artifact_catalog.replace_task_artifacts(task_id, current_user["user_id"], manifest)

    start_payload = _start_task_impl(
        task_id,
        current_user["user_id"],
        max_photos=max_photos,
        use_cache=use_cache,
        normalize_live_photos=normalize_live_photos,
        background_tasks=background_tasks,
    )
    return {
        **start_payload,
        "accepted_count": len(saved_files),
        "failed_count": len(upload_failures),
        "upload_count": updated_task["upload_count"],
    }


@app.post("/api/tasks/{task_id}/upload-batches")
async def upload_task_batch(
    task_id: str,
    background_tasks: BackgroundTasks,
    response: Response,
    files: List[UploadFile] = File(...),
    expected_upload_count: Optional[int] = Form(default=None),
    requested_max_photos: Optional[int] = Form(default=None),
    creation_source: Optional[str] = Form(default=None),
    auto_start_on_upload_complete: Optional[bool] = Form(default=None),
    survey_username: Optional[str] = Form(default=None),
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    if not files:
        raise HTTPException(status_code=400, detail="至少上传一张图片")
    if len(files) > UPLOAD_BATCH_MAX_FILES:
        raise HTTPException(status_code=400, detail=f"单批最多上传 {UPLOAD_BATCH_MAX_FILES} 张图片")

    try:
        with _task_upload_lock(task_id) as task_dir:
            task = task_store.get_task(task_id, user_id=current_user["user_id"])
            if not task:
                raise HTTPException(status_code=404, detail="任务不存在")
            if task["status"] in {"queued", "running", "completed"}:
                raise HTTPException(status_code=409, detail="当前任务已开始处理，不能继续追加上传")

            merged_options = _merge_upload_task_options(
                task.get("options"),
                creation_source=creation_source,
                expected_upload_count=expected_upload_count,
                requested_max_photos=requested_max_photos,
                auto_start_on_upload_complete=auto_start_on_upload_complete,
                survey_username=survey_username,
            )
            if merged_options != normalize_task_options(task.get("options")):
                task = task_store.update_task(task_id, options=merged_options)

            start_index = len(_task_uploads(task)) + 1
            saved_files, upload_failures, _ = _save_upload_batch(task_id, task_dir, files, start_index)

            _write_merged_upload_failures(task_dir, upload_failures)
            updated_task = task_store.append_uploads(task_id, saved_files, status="uploading", stage="uploading")
            manifest = build_task_asset_manifest(task_id, task_dir, asset_store)
            task_store.update_task(task_id, asset_manifest=manifest)
            artifact_catalog.replace_task_artifacts(task_id, current_user["user_id"], manifest)
            auto_start_payload = _maybe_auto_start_after_upload(
                task_id,
                current_user["user_id"],
                updated_task,
                task_dir=task_dir,
                background_tasks=background_tasks,
            )
    finally:
        for upload in files:
            await upload.close()

    return {
        "task_id": task_id,
        "version": (auto_start_payload or updated_task)["version"],
        "status": (auto_start_payload or updated_task)["status"],
        "stage": (auto_start_payload or updated_task)["stage"],
        "batch_count": len(saved_files),
        "failed_count": len(upload_failures),
        "upload_count": updated_task["upload_count"],
        "auto_started": bool(auto_start_payload),
        "task_url": f"/api/tasks/{task_id}",
    }


@app.post("/api/tasks/{task_id}/start")
def start_task(
    task_id: str,
    payload: TaskStartPayload,
    background_tasks: BackgroundTasks,
    response: Response,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    with _task_upload_lock(task_id):
        return _start_task_impl(
            task_id,
            current_user["user_id"],
            max_photos=payload.max_photos,
            use_cache=payload.use_cache,
            normalize_live_photos=payload.normalize_live_photos,
            background_tasks=background_tasks,
        )


@app.get("/api/tasks")
def list_tasks(response: Response, limit: int = 20, current_user: dict = Depends(get_current_user)):
    _apply_no_store_headers(response)
    safe_limit = max(1, min(limit, 100))
    return {
        "tasks": task_store.list_tasks(user_id=current_user["user_id"], limit=safe_limit),
    }


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str, response: Response, current_user: dict = Depends(get_current_user)):
    _apply_no_store_headers(response)
    task = task_store.get_task(task_id, user_id=current_user["user_id"])
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _sanitize_task_for_client(_hydrate_task(task, current_user["user_id"]))


@app.get("/api/tasks/{task_id}/downloads/analysis-bundle")
def download_task_analysis_bundle(task_id: str, current_user: dict = Depends(get_current_user)):
    task = task_store.get_task(task_id, user_id=current_user["user_id"])
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    hydrated = _sync_task_from_worker(task, current_user["user_id"])
    downloads = describe_task_downloads(hydrated)
    bundle = (downloads or {}).get("analysis_bundle")
    if not bundle:
        raise HTTPException(status_code=404, detail="当前任务没有可下载的 Face/VLM/LP1 打包结果")

    try:
        archive_path = build_task_analysis_bundle(hydrated)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    response = FileResponse(
        archive_path,
        media_type="application/zip",
        filename=str(bundle.get("filename") or f"{task_id[:12]}-face-vlm-lp1-bundle.zip"),
    )
    return _apply_no_store_headers(response)


@app.post("/api/tasks/{task_id}/memory/query")
def query_task_memory(
    task_id: str,
    payload: MemoryQueryPayload,
    response: Response,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    task = task_store.get_task(task_id, user_id=current_user["user_id"])
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    hydrated = _hydrate_task(task, current_user["user_id"])
    subject_user_id = str(hydrated.get("user_id") or "").strip() or current_user["user_id"]
    memory_payload = (hydrated.get("result") or {}).get("memory")
    if MEMORY_QUERY_V1_ENABLED and str(hydrated.get("version") or "").strip() in {
        TASK_VERSION_V0325,
        TASK_VERSION_V0327_EXP,
        TASK_VERSION_V0327_DB,
        TASK_VERSION_V0327_DB_QUERY,
    }:
        engine = QueryEngineV1()
        try:
            answer = engine.answer_task(
                hydrated,
                payload.question,
                user_id=subject_user_id,
                context_hints=payload.context_hints or {},
                time_hint=payload.time_hint,
                answer_shape_hint=payload.answer_shape_hint,
            )
        except Exception:
            logger.exception("query_v1 engine failed for task_id=%s", task_id)
        else:
            if MEMORY_QUERY_V1_SHADOW_COMPARE and isinstance(memory_payload, dict):
                try:
                    legacy = MemoryQueryService().answer(
                        memory_payload,
                        payload.question,
                        user_id=subject_user_id,
                        context_hints=payload.context_hints or {},
                        time_hint=payload.time_hint,
                        answer_shape_hint=payload.answer_shape_hint,
                    )
                except Exception:
                    logger.exception("legacy shadow compare failed for task_id=%s", task_id)
                else:
                    answer["shadow_compare"] = {
                        "legacy_plan_type": ((legacy.get("query_plan") or {}).get("plan_type")),
                        "legacy_answer_type": ((legacy.get("answer") or {}).get("answer_type")),
                        "legacy_supporting_unit_count": len(list(legacy.get("supporting_units") or [])),
                        "legacy_supporting_evidence_count": len(list(legacy.get("supporting_evidence") or [])),
                    }
            return answer
    if not isinstance(memory_payload, dict):
        raise HTTPException(status_code=404, detail="当前任务没有 memory 输出")
    return MemoryQueryService().answer(
        memory_payload,
        payload.question,
        user_id=subject_user_id,
        context_hints=payload.context_hints or {},
        time_hint=payload.time_hint,
        answer_shape_hint=payload.answer_shape_hint,
    )


def _get_hydrated_task_or_404(task_id: str, current_user: dict) -> dict:
    task = task_store.get_task(task_id, user_id=current_user["user_id"])
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _hydrate_task(task, current_user["user_id"])


def _get_hydrated_subject_task_or_404(subject_user_id: str, task_id: str, current_user: dict) -> dict:
    task = task_store.get_task_for_subject(task_id, subject_user_id=subject_user_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _hydrate_task_payloads(_sync_task_from_worker(task, current_user["user_id"]))


@app.get("/api/users/{user_id}/tasks")
def list_subject_tasks(
    user_id: str,
    response: Response,
    limit: int = 20,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    del current_user
    safe_limit = max(1, min(limit, 100))
    return {
        "user_id": user_id,
        "tasks": task_store.list_tasks_for_subject(subject_user_id=user_id, limit=safe_limit),
    }


@app.get("/api/users/{user_id}/tasks/{task_id}")
def get_subject_task(
    user_id: str,
    task_id: str,
    response: Response,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    task = task_store.get_task_for_subject(task_id, subject_user_id=user_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return _sanitize_task_for_client(_hydrate_task_payloads(_sync_task_from_worker(task, current_user["user_id"])))


@app.post("/api/users/{user_id}/tasks/{task_id}/memory/query")
def query_subject_task_memory(
    user_id: str,
    task_id: str,
    payload: MemoryQueryPayload,
    response: Response,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    task = _get_hydrated_subject_task_or_404(user_id, task_id, current_user)
    subject_user_id = str(task.get("user_id") or "").strip() or user_id
    memory_payload = (task.get("result") or {}).get("memory")
    if MEMORY_QUERY_V1_ENABLED and str(task.get("version") or "").strip() in {
        TASK_VERSION_V0325,
        TASK_VERSION_V0327_EXP,
        TASK_VERSION_V0327_DB,
        TASK_VERSION_V0327_DB_QUERY,
    }:
        engine = QueryEngineV1()
        answer = engine.answer_task(
            task,
            payload.question,
            user_id=subject_user_id,
            context_hints=payload.context_hints or {},
            time_hint=payload.time_hint,
            answer_shape_hint=payload.answer_shape_hint,
        )
        return answer
    if not isinstance(memory_payload, dict):
        raise HTTPException(status_code=404, detail="当前任务没有 memory 输出")
    return MemoryQueryService().answer(
        memory_payload,
        payload.question,
        user_id=subject_user_id,
        context_hints=payload.context_hints or {},
        time_hint=payload.time_hint,
        answer_shape_hint=payload.answer_shape_hint,
    )


@app.get("/api/users/{user_id}/tasks/{task_id}/memory/faces")
def get_subject_memory_faces(
    user_id: str,
    task_id: str,
    response: Response,
    include_artifacts: bool = False,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    task = _get_hydrated_subject_task_or_404(user_id, task_id, current_user)
    return build_task_memory_faces_response(task, include_artifacts=include_artifacts)


@app.get("/api/users/{user_id}/tasks/{task_id}/memory/events")
def get_subject_memory_events(
    user_id: str,
    task_id: str,
    response: Response,
    include_raw: bool = False,
    include_artifacts: bool = False,
    include_traces: bool = False,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    task = _get_hydrated_subject_task_or_404(user_id, task_id, current_user)
    return build_task_memory_events_response(
        task,
        include_raw=include_raw,
        include_artifacts=include_artifacts,
        include_traces=include_traces,
    )


@app.get("/api/users/{user_id}/tasks/{task_id}/memory/vlm")
def get_subject_memory_vlm(
    user_id: str,
    task_id: str,
    response: Response,
    include_raw: bool = False,
    include_artifacts: bool = False,
    include_traces: bool = False,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    task = _get_hydrated_subject_task_or_404(user_id, task_id, current_user)
    return build_task_memory_vlm_response(
        task,
        include_raw=include_raw,
        include_artifacts=include_artifacts,
        include_traces=include_traces,
    )


@app.get("/api/users/{user_id}/tasks/{task_id}/memory/profiles")
def get_subject_memory_profiles(
    user_id: str,
    task_id: str,
    response: Response,
    include_raw: bool = False,
    include_artifacts: bool = False,
    include_traces: bool = False,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    task = _get_hydrated_subject_task_or_404(user_id, task_id, current_user)
    return build_task_memory_profiles_response(
        task,
        include_raw=include_raw,
        include_artifacts=include_artifacts,
        include_traces=include_traces,
    )


@app.get("/api/users/{user_id}/tasks/{task_id}/memory/relationships")
def get_subject_memory_relationships(
    user_id: str,
    task_id: str,
    response: Response,
    include_raw: bool = False,
    include_artifacts: bool = False,
    include_traces: bool = False,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    task = _get_hydrated_subject_task_or_404(user_id, task_id, current_user)
    return build_task_memory_relationships_response(
        task,
        include_raw=include_raw,
        include_artifacts=include_artifacts,
        include_traces=include_traces,
    )


@app.get("/api/users/{user_id}/tasks/{task_id}/memory/bundle")
def get_subject_memory_bundle(
    user_id: str,
    task_id: str,
    response: Response,
    include_raw: bool = False,
    include_artifacts: bool = False,
    include_traces: bool = False,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    task = _get_hydrated_subject_task_or_404(user_id, task_id, current_user)
    return build_task_memory_bundle_response(
        task,
        include_raw=include_raw,
        include_artifacts=include_artifacts,
        include_traces=include_traces,
    )


@app.get("/api/tasks/{task_id}/memory/faces")
def get_task_memory_faces(
    task_id: str,
    response: Response,
    include_artifacts: bool = False,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    task = _get_hydrated_task_or_404(task_id, current_user)
    try:
        return build_task_memory_faces_response(task, include_artifacts=include_artifacts)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/tasks/{task_id}/memory/events")
def get_task_memory_events(
    task_id: str,
    response: Response,
    include_raw: bool = False,
    include_artifacts: bool = False,
    include_traces: bool = False,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    task = _get_hydrated_task_or_404(task_id, current_user)
    try:
        return build_task_memory_events_response(
            task,
            include_raw=include_raw,
            include_artifacts=include_artifacts,
            include_traces=include_traces,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/tasks/{task_id}/memory/vlm")
def get_task_memory_vlm(
    task_id: str,
    response: Response,
    include_raw: bool = False,
    include_artifacts: bool = False,
    include_traces: bool = False,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    task = _get_hydrated_task_or_404(task_id, current_user)
    try:
        return build_task_memory_vlm_response(
            task,
            include_raw=include_raw,
            include_artifacts=include_artifacts,
            include_traces=include_traces,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/tasks/{task_id}/memory/profiles")
def get_task_memory_profiles(
    task_id: str,
    response: Response,
    include_raw: bool = False,
    include_artifacts: bool = False,
    include_traces: bool = False,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    task = _get_hydrated_task_or_404(task_id, current_user)
    try:
        return build_task_memory_profiles_response(
            task,
            include_raw=include_raw,
            include_artifacts=include_artifacts,
            include_traces=include_traces,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/tasks/{task_id}/memory/relationships")
def get_task_memory_relationships(
    task_id: str,
    response: Response,
    include_raw: bool = False,
    include_artifacts: bool = False,
    include_traces: bool = False,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    task = _get_hydrated_task_or_404(task_id, current_user)
    try:
        return build_task_memory_relationships_response(
            task,
            include_raw=include_raw,
            include_artifacts=include_artifacts,
            include_traces=include_traces,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/tasks/{task_id}/memory/bundle")
def get_task_memory_bundle(
    task_id: str,
    response: Response,
    include_raw: bool = False,
    include_artifacts: bool = False,
    include_traces: bool = False,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    task = _get_hydrated_task_or_404(task_id, current_user)
    try:
        return build_task_memory_bundle_response(
            task,
            include_raw=include_raw,
            include_artifacts=include_artifacts,
            include_traces=include_traces,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/tasks/{task_id}/memory/core")
def get_task_memory_core(
    task_id: str,
    response: Response,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    task = task_store.get_task(task_id, user_id=current_user["user_id"])
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    hydrated = _hydrate_task(task, current_user["user_id"])
    try:
        return build_task_memory_core_payload(hydrated)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/tasks/{task_id}/memory/steps")
def get_task_memory_steps(
    task_id: str,
    response: Response,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    task = task_store.get_task(task_id, user_id=current_user["user_id"])
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    hydrated = _hydrate_task_payloads(_sync_task_from_worker(task, current_user["user_id"]))
    if str(hydrated.get("version") or "").strip() not in {
        "v0323",
        TASK_VERSION_V0325,
        TASK_VERSION_V0327_EXP,
        TASK_VERSION_V0327_DB,
        TASK_VERSION_V0327_DB_QUERY,
    }:
        raise HTTPException(status_code=404, detail="当前任务没有 LP steps 输出")
    return build_task_memory_steps_payload(
        hydrated,
        asset_url_builder=lambda resolved_task_id, relative_path: asset_store.asset_url(resolved_task_id, relative_path),
    )


@app.get("/api/tasks/{task_id}/reviews")
def get_task_reviews(task_id: str, response: Response, current_user: dict = Depends(get_current_user)):
    _apply_no_store_headers(response)
    task = task_store.get_task(task_id, user_id=current_user["user_id"])
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    hydrated = _hydrate_task_payloads(_sync_task_from_worker(task, current_user["user_id"]))
    return face_review_store.get_task_feedback(
        task_id=task_id,
        user_id=current_user["user_id"],
        source_hashes=_task_source_hashes(hydrated),
    )


@app.put("/api/tasks/{task_id}/faces/{face_id}/review")
def update_face_review(
    task_id: str,
    face_id: str,
    payload: FaceReviewPayload,
    response: Response,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    stored_task = task_store.get_task(task_id, user_id=current_user["user_id"])
    if not stored_task:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = _hydrate_task(stored_task, current_user["user_id"])
    face_ref = _find_face(task, face_id)
    if not face_ref:
        raise HTTPException(status_code=404, detail="人脸不存在")
    review = face_review_store.upsert_face_review(
        user_id=current_user["user_id"],
        task_id=task_id,
        face_id=face_id,
        image_id=face_ref["image_id"],
        person_id=face_ref["person_id"],
        source_hash=face_ref["source_hash"],
        is_inaccurate=payload.is_inaccurate,
        comment_text=payload.comment_text,
    )
    return {"review": review}


@app.put("/api/tasks/{task_id}/images/{image_id}/face-policy")
def update_image_policy(
    task_id: str,
    image_id: str,
    payload: ImagePolicyPayload,
    response: Response,
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
    stored_task = task_store.get_task(task_id, user_id=current_user["user_id"])
    if not stored_task:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = _hydrate_task(stored_task, current_user["user_id"])
    image = _find_image(task, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="图片不存在")
    source_hash = image.get("source_hash")
    if not source_hash:
        raise HTTPException(status_code=400, detail="当前图片缺少 source_hash，无法设置 abandon")
    policy = face_review_store.upsert_image_policy(
        user_id=current_user["user_id"],
        source_hash=source_hash,
        is_abandoned=payload.is_abandoned,
        last_task_id=task_id,
        last_image_id=image_id,
    )
    return {"policy": policy}


@app.delete("/api/tasks/{task_id}")
def delete_task(task_id: str, response: Response, current_user: dict = Depends(get_current_user)):
    _apply_no_store_headers(response)
    task = task_store.get_task(task_id, user_id=current_user["user_id"])
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if task["status"] not in {"completed", "failed"}:
        raise HTTPException(status_code=409, detail="任务处理中，暂不支持删除")

    task_dir = task_store.task_dir(task_id)

    if task.get("worker_instance_id"):
        task_store.mark_delete_requested(task_id, user_id=current_user["user_id"])
        private_ip = task.get("worker_private_ip")
        if private_ip and worker_client.enabled:
            try:
                worker_client.request_delete(private_ip, task_id)
            except Exception:
                pass
        worker_manager.terminate_worker(task["worker_instance_id"])
        task_store.mark_deleted(task_id, user_id=current_user["user_id"])

    try:
        asset_store.delete_task_assets(task_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"删除对象存储文件失败: {exc}") from exc

    try:
        shutil.rmtree(task_dir, ignore_errors=False)
    except FileNotFoundError:
        pass
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"删除本地任务文件失败: {exc}") from exc

    artifact_catalog.delete_task_artifacts(task_id, user_id=current_user["user_id"])
    deleted = task_store.delete_task(task_id, user_id=current_user["user_id"])
    if not deleted:
        raise HTTPException(status_code=500, detail="删除任务记录失败")

    return {"status": "deleted", "task_id": task_id}


@app.get(f"{ASSET_URL_PREFIX}" + "/{task_id}/{asset_path:path}")
def get_asset(task_id: str, asset_path: str, current_user: dict = Depends(get_current_user)):
    task = task_store.get_task(task_id, user_id=current_user["user_id"])
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    task = _sync_task_from_worker(task, current_user["user_id"])
    private_ip = task.get("worker_private_ip")
    if task.get("worker_instance_id") and private_ip and worker_client.enabled:
        try:
            payload, content_type = worker_client.fetch_asset(private_ip, task_id, asset_path)
            response = Response(content=payload, media_type=content_type)
            return _apply_no_store_headers(response)
        except Exception:
            pass

    task_dir = task_store.task_dir(task_id)
    if asset_store.enabled:
        try:
            payload, content_type = asset_store.read_bytes(task_id, asset_path)
            response = Response(content=payload, media_type=content_type)
            return _apply_no_store_headers(response)
        except Exception:
            pass

    local_path = asset_store.local_asset_path(task_dir, asset_path)
    if local_path and local_path.exists():
        response = FileResponse(local_path)
        return _apply_no_store_headers(response)

    raise HTTPException(status_code=404, detail="资产不存在")


@app.get("/api/tasks/{task_id}/artifacts")
def list_task_artifacts(task_id: str, response: Response, current_user: dict = Depends(get_current_user)):
    _apply_no_store_headers(response)
    task = task_store.get_task(task_id, user_id=current_user["user_id"])
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    task = _sync_task_from_worker(task, current_user["user_id"])
    artifacts = artifact_catalog.list_task_artifacts(task_id, current_user["user_id"])
    return {
        "task_id": task_id,
        "version": task.get("version"),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=BACKEND_RELOAD)
