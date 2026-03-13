"""
Private worker app used for per-task processing.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import List

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, Response, UploadFile, status
from fastapi.responses import FileResponse

from backend.upload_utils import (
    UPLOAD_FAILURES_FILENAME,
    preview_filename,
    save_preview_as_webp,
    save_upload_original,
    stored_upload_filename,
    task_asset_path,
    write_upload_failures,
)
from config import ASSET_URL_PREFIX, MAX_UPLOAD_PHOTOS, WORKER_INTERNAL_PORT, WORKER_SHARED_TOKEN, WORKER_TASK_ROOT
from services.asset_store import TaskAssetStore
from services.pipeline_service import MemoryPipelineService
from utils import load_json, save_json


app = FastAPI(title="Memory Engineering Worker", version="1.0.0")
asset_store = TaskAssetStore()
worker_root = Path(WORKER_TASK_ROOT)
STATUS_FILENAME = "worker_status.json"


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


def _build_asset_manifest(task_dir: Path) -> dict:
    files = []
    for file_path in sorted(task_dir.rglob("*")):
        if file_path.is_file():
            files.append({"path": file_path.relative_to(task_dir).as_posix(), "size": file_path.stat().st_size})
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "files": files,
    }


def _run_pipeline_task(task_id: str, max_photos: int, use_cache: bool) -> None:
    task_dir = _task_dir(task_id)

    def progress_callback(stage: str, payload: dict) -> None:
        _update_status(task_id, status="running", stage=stage, progress=payload, worker_status="running")

    try:
        _update_status(task_id, status="running", stage="starting", progress=None, error=None, worker_status="running")
        result = MemoryPipelineService(task_id=task_id, task_dir=str(task_dir), asset_store=asset_store).run(
            max_photos=max_photos,
            use_cache=use_cache,
            progress_callback=progress_callback,
        )
        _update_status(
            task_id,
            status="completed",
            stage="completed",
            progress=None,
            result=result,
            result_summary=result.get("summary", {}),
            asset_manifest=_build_asset_manifest(task_dir),
            error=None,
            worker_status="running",
        )
    except Exception as exc:
        _update_status(
            task_id,
            status="failed",
            stage="failed",
            error=str(exc),
            worker_status="running",
        )


@app.get("/internal/health")
def healthcheck(response: Response, _: None = Depends(_require_internal_token)):
    _apply_no_store_headers(response)
    return {
        "status": "ok",
        "role": "worker",
        "task_root": str(worker_root),
        "internal_asset_prefix": ASSET_URL_PREFIX,
    }


@app.post("/internal/tasks/{task_id}/ingest")
async def ingest_task(
    task_id: str,
    background_tasks: BackgroundTasks,
    response: Response,
    files: List[UploadFile] = File(...),
    max_photos: int = Form(MAX_UPLOAD_PHOTOS),
    use_cache: bool = Form(False),
    _: None = Depends(_require_internal_token),
):
    _apply_no_store_headers(response)
    if not files:
        raise HTTPException(status_code=400, detail="至少上传一张图片")
    if len(files) > MAX_UPLOAD_PHOTOS:
        raise HTTPException(status_code=400, detail=f"单次最多上传 {MAX_UPLOAD_PHOTOS} 张图片")

    task_dir = _task_dir(task_id)
    if task_dir.exists():
        raise HTTPException(status_code=409, detail="worker 上已存在同名任务")

    uploads_dir = task_dir / "uploads"
    previews_dir = task_dir / "previews"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    upload_failures = []
    for index, upload in enumerate(files, start=1):
        stored_name = stored_upload_filename(upload.filename or "", index)
        destination = uploads_dir / stored_name
        preview_name = preview_filename(upload.filename or "", index)
        preview_destination = previews_dir / preview_name
        try:
            payload, image_info = save_upload_original(upload, destination)
            original_asset_path = task_asset_path("uploads", stored_name)

            preview_url = None
            try:
                save_preview_as_webp(payload, preview_destination)
                preview_asset_path = task_asset_path("previews", preview_name)
                preview_url = asset_store.asset_url(task_id, preview_asset_path)
            except Exception:
                preview_destination.unlink(missing_ok=True)

            saved_files.append(
                {
                    "filename": upload.filename or stored_name,
                    "stored_filename": stored_name,
                    "path": original_asset_path,
                    "url": asset_store.asset_url(task_id, original_asset_path),
                    "preview_url": preview_url,
                    **image_info,
                }
            )
        except Exception as exc:
            destination.unlink(missing_ok=True)
            preview_destination.unlink(missing_ok=True)
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

    write_upload_failures(task_dir, upload_failures)
    initial_status = {
        "task_id": task_id,
        "status": "queued",
        "stage": "queued",
        "upload_count": len(saved_files),
        "uploads": saved_files,
        "progress": None,
        "result": None,
        "result_summary": None,
        "asset_manifest": None,
        "error": None,
        "worker_status": "running",
        "created_at": datetime.utcnow().isoformat(),
    }
    _write_status(task_id, initial_status)
    _update_status(task_id, asset_manifest=_build_asset_manifest(task_dir))

    background_tasks.add_task(_run_pipeline_task, task_id, min(max_photos, len(saved_files)), use_cache)

    return {
        "task_id": task_id,
        "status": "queued",
        "stage": "queued",
        "upload_count": len(saved_files),
        "uploads": saved_files,
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
