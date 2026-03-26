"""
Private worker app used for per-task processing.
"""
from __future__ import annotations

import json
import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import List

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.artifact_store import build_task_asset_manifest
from backend.progress_utils import append_terminal_error, append_terminal_info, merge_stage_progress
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
worker_root = Path(WORKER_TASK_ROOT)
STATUS_FILENAME = "worker_status.json"
UPLOAD_BATCH_MAX_FILES = 50
STATUS_FLUSH_INTERVAL_SECONDS = 1.0
STATUS_MIN_PERCENT_DELTA = 1


class TaskStartPayload(BaseModel):
    max_photos: int = MAX_UPLOAD_PHOTOS
    use_cache: bool = False
    version: str = DEFAULT_TASK_VERSION
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
    status_path = _status_path(task_id)
    tmp_path = status_path.with_suffix(f"{status_path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    tmp_path.replace(status_path)
    return payload


def _update_status(task_id: str, **updates) -> dict:
    current = _read_status(task_id)
    current.update(updates)
    return _write_status(task_id, current)


def _progress_substage(progress: dict | None, stage: str) -> str:
    stages = dict((progress or {}).get("stages") or {})
    stage_payload = dict(stages.get(stage) or {})
    return str(stage_payload.get("substage") or "").strip()


def _progress_percent(progress: dict | None, stage: str) -> int | None:
    stages = dict((progress or {}).get("stages") or {})
    stage_payload = dict(stages.get(stage) or {})
    value = stage_payload.get("percent")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _run_pipeline_task(task_id: str, max_photos: int, use_cache: bool, task_version: str, task_options: dict | None) -> None:
    task_dir = _task_dir(task_id)
    status_snapshot = _read_status(task_id)
    last_flush_at = monotonic()
    last_flushed_stage = str(status_snapshot.get("stage") or "").strip()
    last_flushed_substage = _progress_substage(status_snapshot.get("progress"), last_flushed_stage)
    last_flushed_percent = _progress_percent(status_snapshot.get("progress"), last_flushed_stage)

    def flush_status(*, force: bool = False) -> dict | None:
        nonlocal last_flush_at, last_flushed_stage, last_flushed_substage, last_flushed_percent
        stage = str(status_snapshot.get("stage") or "").strip()
        progress = status_snapshot.get("progress") if isinstance(status_snapshot.get("progress"), dict) else {}
        substage = _progress_substage(progress, stage)
        percent = _progress_percent(progress, stage)
        percent_advanced = (
            percent is not None
            and (
                last_flushed_percent is None
                or percent >= (last_flushed_percent + STATUS_MIN_PERCENT_DELTA)
            )
        )
        should_flush = force or stage != last_flushed_stage or substage != last_flushed_substage or percent_advanced
        if not should_flush and (monotonic() - last_flush_at) < STATUS_FLUSH_INTERVAL_SECONDS:
            return None
        payload = _write_status(task_id, dict(status_snapshot))
        last_flush_at = monotonic()
        last_flushed_stage = stage
        last_flushed_substage = substage
        last_flushed_percent = percent
        return payload

    def progress_callback(stage: str, payload: dict) -> None:
        merged_progress = merge_stage_progress(
            status_snapshot.get("progress") if isinstance(status_snapshot, dict) else None,
            stage,
            payload or {},
        )
        status_snapshot.update(status="running", stage=stage, progress=merged_progress, worker_status="running")
        flush_status()

    try:
        status_snapshot.update(
            status="running",
            stage="starting",
            progress=append_terminal_info(None, stage="starting", message="准备启动推理任务"),
            error=None,
            worker_status="running",
            version=task_version,
            options=task_options,
        )
        flush_status(force=True)
        result = MemoryPipelineService(
            task_id=task_id,
            task_dir=str(task_dir),
            asset_store=asset_store,
            task_version=task_version,
            task_options=task_options,
        ).run(
            max_photos=max_photos,
            use_cache=use_cache,
            progress_callback=progress_callback,
        )
        status_snapshot.update(
            status="completed",
            stage="completed",
            progress=append_terminal_info(
                status_snapshot.get("progress") if isinstance(status_snapshot.get("progress"), dict) else None,
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
        flush_status(force=True)
    except Exception as exc:
        logger.exception("Worker pipeline task failed for task_id=%s version=%s", task_id, task_version)
        status_snapshot.update(
            status="failed",
            stage="failed",
            progress=append_terminal_error(
                status_snapshot.get("progress") if isinstance(status_snapshot.get("progress"), dict) else None,
                stage="failed",
                error=str(exc),
            ),
            error=str(exc),
            worker_status="running",
        )
        flush_status(force=True)


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


@app.post("/internal/tasks/{task_id}/checkpoint-archive")
async def upload_checkpoint_archive(
    task_id: str,
    response: Response,
    archive: UploadFile = File(...),
    _: None = Depends(_require_internal_token),
):
    _apply_no_store_headers(response)
    if archive.filename is None:
        raise HTTPException(status_code=400, detail="checkpoint archive 文件缺失")
    task_dir = _task_dir(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    archive_path = task_dir / "_resume_checkpoint.zip"
    try:
        with archive_path.open("wb") as handle:
            while True:
                chunk = await archive.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        with zipfile.ZipFile(archive_path, "r") as bundle:
            for member in bundle.infolist():
                relative_name = str(member.filename or "").strip()
                if not relative_name or relative_name.endswith("/"):
                    continue
                target = (task_dir / relative_name).resolve()
                if task_dir.resolve() not in target.parents and target != task_dir.resolve():
                    raise HTTPException(status_code=400, detail="checkpoint archive 包含非法路径")
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member, "r") as source, target.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
    finally:
        try:
            archive_path.unlink()
        except Exception:
            pass
        await archive.close()
    return {"task_id": task_id, "status": "ok"}


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
    task_options = dict(payload.options or {})
    task_options["normalize_live_photos"] = bool(task_options.get("normalize_live_photos", payload.normalize_live_photos))

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
    background_tasks.add_task(_run_pipeline_task, task_id, max_photos, payload.use_cache, task_version, task_options)
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
