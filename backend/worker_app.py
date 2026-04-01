"""
Private worker app used for per-task processing.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.artifact_store import build_task_asset_manifest
from backend.control_plane_client import ControlPlaneClient
from backend.progress_utils import append_terminal_error, append_terminal_info, merge_stage_progress
from backend.task_completion import ensure_completion_outputs
from backend.task_store import normalize_task_options
from backend.upload_utils import (
    UPLOAD_FAILURES_FILENAME,
    is_live_photo_candidate,
    save_upload_original_streamed,
    stored_upload_filename,
    task_asset_path,
    write_upload_failures,
)
from config import (
    ASSET_URL_PREFIX,
    DEFAULT_NORMALIZE_LIVE_PHOTOS,
    DEFAULT_TASK_VERSION,
    MAX_UPLOAD_PHOTOS,
    WORKER_SHARED_TOKEN,
    WORKER_TASK_ROOT,
    normalize_task_version,
)
from services.asset_store import TaskAssetStore
from services.pipeline_service import MemoryPipelineService
from utils import load_json, save_json


app = FastAPI(title="Memory Engineering Worker", version="1.0.0")
logger = logging.getLogger(__name__)
asset_store = TaskAssetStore()
control_plane_client = ControlPlaneClient()
worker_root = Path(WORKER_TASK_ROOT)
STATUS_FILENAME = "worker_status.json"
UPLOAD_BATCH_MAX_FILES = 50


class TaskStartPayload(BaseModel):
    max_photos: int = MAX_UPLOAD_PHOTOS
    use_cache: bool = False
    version: str = DEFAULT_TASK_VERSION
    user_id: str | None = None
    normalize_live_photos: bool = DEFAULT_NORMALIZE_LIVE_PHOTOS
    options: dict | None = None


def _require_internal_token(authorization: str | None = Header(default=None)) -> None:
    if not WORKER_SHARED_TOKEN:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="worker token 未配置")
    expected = f"Bearer {WORKER_SHARED_TOKEN}"
    if authorization != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的内部访问凭证")


def _apply_no_store_headers(response: Response) -> Response:
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def _task_dir(task_id: str) -> Path:
    worker_root.mkdir(parents=True, exist_ok=True)
    return worker_root / task_id


def _status_path(task_id: str) -> Path:
    return _task_dir(task_id) / STATUS_FILENAME


def _read_status(task_id: str) -> dict:
    return load_json(str(_status_path(task_id)))


def _write_status(task_id: str, payload: dict) -> dict:
    task_dir = _task_dir(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = datetime.utcnow().isoformat()
    save_json(payload, str(_status_path(task_id)))
    return payload


def _update_status(task_id: str, **updates) -> dict:
    current = _read_status(task_id)
    current.update(updates)
    return _write_status(task_id, current)


def _build_task_asset_manifest_safely(task_id: str, task_dir: Path) -> dict | None:
    try:
        return build_task_asset_manifest(task_id, task_dir, asset_store)
    except Exception:
        logger.exception("Failed to build worker asset manifest for task_id=%s", task_id)
        return None


def _notify_control_plane_terminal_update(task_id: str, payload: dict) -> None:
    if not control_plane_client.enabled:
        return
    callback_payload = {
        "status": payload.get("status"),
        "stage": payload.get("stage"),
        "progress": payload.get("progress"),
        "result": payload.get("result"),
        "result_summary": payload.get("result_summary"),
        "asset_manifest": payload.get("asset_manifest"),
        "error": payload.get("error"),
        "worker_status": payload.get("worker_status"),
        "version": payload.get("version"),
        "options": payload.get("options"),
    }
    try:
        control_plane_client.publish_terminal_update(task_id, callback_payload)
    except Exception:
        logger.exception("Failed to notify control-plane terminal update for task_id=%s", task_id)


def _run_pipeline_task(
    task_id: str,
    user_id: str | None,
    max_photos: int,
    use_cache: bool,
    task_version: str,
    task_options: dict | None,
) -> None:
    task_dir = _task_dir(task_id)

    def progress_callback(stage: str, payload: dict) -> None:
        current = _read_status(task_id)
        merged_progress = merge_stage_progress(
            current.get("progress") if isinstance(current, dict) else None,
            stage,
            payload or {},
        )
        _update_status(task_id, status="running", stage=stage, progress=merged_progress, worker_status="running")

    try:
        _update_status(
            task_id,
            status="running",
            stage="starting",
            progress=append_terminal_info(None, stage="starting", message="准备启动推理任务"),
            error=None,
            worker_status="running",
            version=task_version,
        )
        result = MemoryPipelineService(
            task_id=task_id,
            task_dir=str(task_dir),
            asset_store=asset_store,
            user_id=user_id,
            task_version=task_version,
            task_options=task_options,
        ).run(
            max_photos=max_photos,
            use_cache=use_cache,
            progress_callback=progress_callback,
        )
        ensure_completion_outputs(task_version, result)
        terminal_payload = _update_status(
            task_id,
            status="completed",
            stage="completed",
            progress=append_terminal_info(
                _read_status(task_id).get("progress"),
                stage="completed",
                message="任务执行完成",
            ),
            result=result,
            result_summary=result.get("summary", {}),
            asset_manifest=build_task_asset_manifest(task_id, task_dir, asset_store),
            error=None,
            worker_status="running",
            version=task_version,
        )
        _notify_control_plane_terminal_update(task_id, terminal_payload)
    except Exception as exc:
        logger.exception("Worker pipeline task failed for task_id=%s version=%s", task_id, task_version)
        terminal_payload = _update_status(
            task_id,
            status="failed",
            stage="failed",
            progress=append_terminal_error(
                _read_status(task_id).get("progress"),
                stage="failed",
                error=str(exc),
            ),
            error=str(exc),
            asset_manifest=_build_task_asset_manifest_safely(task_id, task_dir),
            worker_status="running",
        )
        _notify_control_plane_terminal_update(task_id, terminal_payload)


@app.get("/internal/health")
def healthcheck(response: Response, _: None = Depends(_require_internal_token)):
    _apply_no_store_headers(response)
    return {
        "status": "ok",
        "role": "worker",
        "task_root": str(worker_root),
        "internal_asset_prefix": ASSET_URL_PREFIX,
    }


def _existing_upload_failures(task_dir: Path) -> list[dict]:
    payload = load_json(str(task_dir / UPLOAD_FAILURES_FILENAME))
    failures = payload.get("failures", [])
    return failures if isinstance(failures, list) else []


@app.post("/internal/tasks/{task_id}/upload-batches")
async def upload_task_batch(
    task_id: str,
    response: Response,
    version: str = Form(DEFAULT_TASK_VERSION),
    files: List[UploadFile] = File(...),
    _: None = Depends(_require_internal_token),
):
    _apply_no_store_headers(response)
    try:
        task_version = normalize_task_version(version)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not files:
        raise HTTPException(status_code=400, detail="至少上传一张图片")
    if len(files) > UPLOAD_BATCH_MAX_FILES:
        raise HTTPException(status_code=400, detail=f"单批最多上传 {UPLOAD_BATCH_MAX_FILES} 张图片")

    current = _read_status(task_id)
    if current.get("status") in {"queued", "running", "completed"}:
        raise HTTPException(status_code=409, detail="worker 任务已开始处理，不能继续追加上传")
    if current and current.get("version") and current.get("version") != task_version:
        raise HTTPException(status_code=409, detail="worker 任务版本已锁定，不能修改")
    task_dir = _task_dir(task_id)
    uploads_dir = task_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    upload_failures = []
    start_index = len(current.get("uploads") or []) + 1
    for offset, upload in enumerate(files):
        index = start_index + offset
        stored_name = stored_upload_filename(upload.filename or "", index)
        destination = uploads_dir / stored_name
        try:
            image_info = save_upload_original_streamed(upload, destination)
            original_asset_path = task_asset_path("uploads", stored_name)
            saved_files.append(
                {
                    "image_id": f"photo_{index:03d}",
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
                    "image_id": f"photo_{index:03d}",
                    "filename": upload.filename or stored_name,
                    "path": str(destination),
                    "step": "upload",
                    "error": f"保存原始上传文件失败: {exc}",
                }
            )
        finally:
            await upload.close()

    merged_failures = _existing_upload_failures(task_dir)
    merged_failures.extend(upload_failures)
    write_upload_failures(task_dir, merged_failures)

    uploads = list(current.get("uploads") or [])
    uploads.extend(saved_files)
    initial_status = current or {
        "task_id": task_id,
        "status": "draft",
        "stage": "draft",
        "upload_count": 0,
        "uploads": [],
        "progress": None,
        "result": None,
        "result_summary": None,
        "asset_manifest": None,
        "error": None,
        "worker_status": "running",
        "created_at": datetime.utcnow().isoformat(),
        "version": task_version,
        "options": {"normalize_live_photos": DEFAULT_NORMALIZE_LIVE_PHOTOS},
    }
    initial_status.update(
        {
            "status": "uploading",
            "stage": "uploading",
            "upload_count": len(uploads),
            "uploads": uploads,
            "error": None,
            "version": task_version,
        }
    )
    _write_status(task_id, initial_status)
    _update_status(task_id, asset_manifest=build_task_asset_manifest(task_id, task_dir, asset_store))
    return {
        "task_id": task_id,
        "status": "uploading",
        "stage": "uploading",
        "version": task_version,
        "upload_count": len(uploads),
        "uploads": uploads,
        "worker_status": "running",
    }


@app.post("/internal/tasks/{task_id}/start")
def start_task(
    task_id: str,
    payload: TaskStartPayload,
    background_tasks: BackgroundTasks,
    response: Response,
    _: None = Depends(_require_internal_token),
):
    _apply_no_store_headers(response)
    current = _read_status(task_id)
    if not current:
        raise HTTPException(status_code=404, detail="worker 任务不存在")
    uploads = current.get("uploads") or []
    if not uploads:
        raise HTTPException(status_code=400, detail="请先上传图片，再开始处理")
    if current.get("status") in {"queued", "running", "completed"}:
        raise HTTPException(status_code=409, detail="worker 任务已开始或已完成")
    try:
        task_version = normalize_task_version(payload.version)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if current.get("version") and current.get("version") != task_version:
        raise HTTPException(status_code=409, detail="worker 任务版本已锁定，不能修改")
    task_options = normalize_task_options(
        {
            **dict(payload.options or {}),
            "normalize_live_photos": bool(payload.normalize_live_photos),
        }
    )

    max_photos = min(max(1, int(payload.max_photos)), len(uploads), MAX_UPLOAD_PHOTOS)
    _update_status(
        task_id,
        status="queued",
        stage="queued",
        progress=None,
        error=None,
        worker_status="running",
        version=task_version,
        options=task_options,
    )
    background_tasks.add_task(
        _run_pipeline_task,
        task_id,
        payload.user_id,
        max_photos,
        payload.use_cache,
        task_version,
        task_options,
    )
    return {
        "task_id": task_id,
        "status": "queued",
        "stage": "queued",
        "version": task_version,
        "options": task_options,
        "upload_count": len(uploads),
        "worker_status": "running",
    }


@app.get("/internal/tasks/{task_id}/status")
def get_task_status(task_id: str, response: Response, _: None = Depends(_require_internal_token)):
    _apply_no_store_headers(response)
    payload = _read_status(task_id)
    if not payload:
        raise HTTPException(status_code=404, detail="worker 任务不存在")
    return payload


@app.get("/internal/tasks/{task_id}/assets/{asset_path:path}")
def get_task_asset(task_id: str, asset_path: str, _: None = Depends(_require_internal_token)):
    task_dir = _task_dir(task_id)
    local_path = asset_store.local_asset_path(task_dir, asset_path)
    if local_path and local_path.exists():
        response = FileResponse(local_path)
        return _apply_no_store_headers(response)
    raise HTTPException(status_code=404, detail="worker 资产不存在")


@app.delete("/internal/tasks/{task_id}")
def delete_task(task_id: str, response: Response, _: None = Depends(_require_internal_token)):
    _apply_no_store_headers(response)
    task_dir = _task_dir(task_id)
    if not task_dir.exists():
        return {"status": "deleted", "task_id": task_id}
    shutil.rmtree(task_dir, ignore_errors=True)
    return {"status": "deleted", "task_id": task_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.worker_app:app", host="0.0.0.0", port=WORKER_INTERNAL_PORT, reload=False)
