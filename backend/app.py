"""
FastAPI backend entrypoint.
"""
from __future__ import annotations

import io
import os
import uuid
from pathlib import Path
from typing import List

from fastapi import BackgroundTasks, Cookie, Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

from backend.auth import (
    AUTH_SESSION_COOKIE_NAME,
    authenticate_response,
    get_current_user,
    login_user,
    logout_current_session,
    register_user,
)
from backend.task_store import TaskStore
from config import (
    ASSET_URL_PREFIX,
    BACKEND_HOST,
    BACKEND_PORT,
    BACKEND_RELOAD,
    CORS_ALLOW_ORIGINS,
    FRONTEND_ORIGIN,
    MAX_UPLOAD_PHOTOS,
    TASKS_DIR,
)
from services.asset_store import TaskAssetStore
from services.pipeline_service import MemoryPipelineService
from utils import save_json


register_heif_opener()
UPLOAD_FAILURES_FILENAME = "upload_failures.json"


app = FastAPI(title="Memory Engineering API", version="1.0.0")
task_store = TaskStore()
asset_store = TaskAssetStore()

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ALLOW_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path(TASKS_DIR).mkdir(parents=True, exist_ok=True)

ORIENTATION_TAG = 274


class AuthPayload(BaseModel):
    username: str
    password: str


def _safe_filename(filename: str, fallback: str) -> str:
    basename = os.path.basename(filename or fallback)
    basename = basename.replace("/", "_").replace("\\", "_")
    return basename or fallback


def _stored_upload_filename(filename: str, index: int) -> str:
    safe_name = _safe_filename(filename, f"upload_{index:03d}")
    stem = Path(safe_name).stem or f"upload_{index:03d}"
    suffix = Path(safe_name).suffix.lower() or ".bin"
    return f"{index:03d}_{stem}{suffix}"


def _preview_filename(filename: str, index: int) -> str:
    safe_name = _safe_filename(filename, f"upload_{index:03d}")
    stem = Path(safe_name).stem or f"upload_{index:03d}"
    return f"{index:03d}_{stem}_preview.webp"


def _task_asset_path(directory: str, filename: str) -> str:
    return f"{directory}/{filename}"


def _normalized_exif_bytes(image: Image.Image) -> bytes | None:
    try:
        exif = image.getexif()
    except Exception:
        return None

    if not exif:
        return None

    if ORIENTATION_TAG in exif:
        exif[ORIENTATION_TAG] = 1
    return exif.tobytes()


def _save_upload_original(upload: UploadFile, destination: Path) -> tuple[bytes, dict]:
    upload.file.seek(0)
    payload = upload.file.read()
    if not payload:
        raise ValueError("文件为空")

    destination.write_bytes(payload)
    image_info = {
        "content_type": upload.content_type or "application/octet-stream",
        "width": None,
        "height": None,
    }

    with Image.open(io.BytesIO(payload)) as image:
        image_info["width"], image_info["height"] = image.size
        image_info["content_type"] = upload.content_type or Image.MIME.get(image.format, image_info["content_type"])

    return payload, image_info


def _save_preview_as_webp(payload: bytes, destination: Path) -> dict:
    with Image.open(io.BytesIO(payload)) as image:
        normalized = ImageOps.exif_transpose(image)
        exif = _normalized_exif_bytes(normalized)
        working = normalized.convert("RGBA") if "A" in normalized.getbands() else normalized.convert("RGB")
        save_kwargs = {
            "format": "WEBP",
            "quality": 90,
            "method": 6,
        }
        if exif:
            save_kwargs["exif"] = exif
        working.save(destination, **save_kwargs)
        width, height = working.size

    return {
        "width": width,
        "height": height,
        "content_type": "image/webp",
    }


def _write_upload_failures(task_dir: Path, failures: list[dict]) -> None:
    save_json({"failures": failures}, str(task_dir / UPLOAD_FAILURES_FILENAME))


def _run_pipeline_task(task_id: str, max_photos: int, use_cache: bool):
    task_dir = task_store.task_dir(task_id)

    def progress_callback(stage: str, payload: dict):
        task_store.update_task(task_id, status="running", stage=stage, progress=payload)

    try:
        task_store.update_task(task_id, status="running", stage="starting", error=None)
        service = MemoryPipelineService(
            task_id=task_id,
            task_dir=str(task_dir),
            asset_store=asset_store,
        )
        result = service.run(
            max_photos=max_photos,
            use_cache=use_cache,
            progress_callback=progress_callback,
        )
        task_store.update_task(
            task_id,
            status="completed",
            stage="completed",
            result=result,
            error=None,
        )
    except Exception as exc:
        task_store.update_task(
            task_id,
            status="failed",
            stage="failed",
            error=str(exc),
        )


@app.get("/api/health")
def healthcheck():
    return {
        "status": "ok",
        "frontend_origin": FRONTEND_ORIGIN,
        "max_upload_photos": MAX_UPLOAD_PHOTOS,
        "asset_url_prefix": ASSET_URL_PREFIX,
        "object_storage_enabled": asset_store.enabled,
        "object_storage_bucket": asset_store.bucket or None,
    }


@app.post("/api/auth/register")
def register(payload: AuthPayload, response: Response):
    user = register_user(payload.username, payload.password)
    _, session_token = login_user(payload.username, payload.password)
    return authenticate_response(response, user, session_token)


@app.post("/api/auth/login")
def login(payload: AuthPayload, response: Response):
    user, session_token = login_user(payload.username, payload.password)
    return authenticate_response(response, user, session_token)


@app.get("/api/auth/me")
def current_user(user: dict = Depends(get_current_user)):
    return {"user": user}


@app.post("/api/auth/logout")
def logout(
    response: Response,
    session_token: str | None = Cookie(default=None, alias=AUTH_SESSION_COOKIE_NAME),
    user: dict = Depends(get_current_user),
):
    del user
    logout_current_session(session_token, response)
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Memory Engineering API",
        "healthcheck": "/api/health",
        "tasks_endpoint": "/api/tasks",
        "assets_prefix": ASSET_URL_PREFIX,
        "auth_me": "/api/auth/me",
    }


@app.post("/api/tasks")
async def create_task(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    max_photos: int = Form(MAX_UPLOAD_PHOTOS),
    use_cache: bool = Form(False),
    current_user: dict = Depends(get_current_user),
):
    if not files:
        raise HTTPException(status_code=400, detail="至少上传一张图片")

    if len(files) > MAX_UPLOAD_PHOTOS:
        raise HTTPException(
            status_code=400,
            detail=f"单次最多上传 {MAX_UPLOAD_PHOTOS} 张图片",
        )

    if max_photos < 1 or max_photos > MAX_UPLOAD_PHOTOS:
        raise HTTPException(
            status_code=400,
            detail=f"max_photos 必须在 1 到 {MAX_UPLOAD_PHOTOS} 之间",
        )

    task_id = uuid.uuid4().hex
    task_dir = task_store.task_dir(task_id)
    uploads_dir = task_dir / "uploads"
    previews_dir = task_dir / "previews"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    upload_failures = []
    for index, upload in enumerate(files, start=1):
        stored_name = _stored_upload_filename(upload.filename or "", index)
        destination = uploads_dir / stored_name
        preview_name = _preview_filename(upload.filename or "", index)
        preview_destination = previews_dir / preview_name
        try:
            payload, image_info = _save_upload_original(upload, destination)
            original_asset_path = _task_asset_path("uploads", stored_name)
            if asset_store.enabled:
                asset_store.upload_file(task_id, original_asset_path, destination)

            preview_url = None
            try:
                _save_preview_as_webp(payload, preview_destination)
                preview_asset_path = _task_asset_path("previews", preview_name)
                if asset_store.enabled:
                    asset_store.upload_file(task_id, preview_asset_path, preview_destination)
                preview_url = asset_store.asset_url(task_id, preview_asset_path)
            except Exception:
                preview_destination.unlink(missing_ok=True)
            saved_files.append({
                "filename": upload.filename or stored_name,
                "stored_filename": stored_name,
                "path": original_asset_path,
                "url": asset_store.asset_url(task_id, original_asset_path),
                "preview_url": preview_url,
                **image_info,
            })
        except Exception as exc:
            if destination.exists():
                destination.unlink(missing_ok=True)
            preview_destination.unlink(missing_ok=True)
            upload_failures.append({
                "image_id": f"photo_{index:03d}",
                "filename": upload.filename or stored_name,
                "path": str(destination),
                "step": "upload",
                "error": f"保存原始上传文件失败: {exc}",
            })
        finally:
            await upload.close()

    _write_upload_failures(task_dir, upload_failures)
    if asset_store.enabled:
        upload_failures_path = task_dir / UPLOAD_FAILURES_FILENAME
        if upload_failures_path.exists():
            asset_store.upload_file(task_id, UPLOAD_FAILURES_FILENAME, upload_failures_path)

    task_payload = task_store.create_task(task_id, upload_count=len(files), user_id=current_user["user_id"])
    task_payload["uploads"] = saved_files
    task_store.update_task(task_id, uploads=saved_files)

    background_tasks.add_task(_run_pipeline_task, task_id, min(max_photos, len(saved_files)), use_cache)

    return {
        "task_id": task_id,
        "status": "queued",
        "upload_count": len(saved_files),
        "max_photos": min(max_photos, len(saved_files)),
        "task_url": f"/api/tasks/{task_id}",
    }


@app.get("/api/tasks")
def list_tasks(limit: int = 20, current_user: dict = Depends(get_current_user)):
    safe_limit = max(1, min(limit, 100))
    return {
        "tasks": task_store.list_tasks(user_id=current_user["user_id"], limit=safe_limit),
    }


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str, current_user: dict = Depends(get_current_user)):
    task = task_store.get_task(task_id, user_id=current_user["user_id"])
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


@app.get(f"{ASSET_URL_PREFIX}" + "/{task_id}/{asset_path:path}")
def get_asset(task_id: str, asset_path: str, current_user: dict = Depends(get_current_user)):
    task = task_store.get_task(task_id, user_id=current_user["user_id"])
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    task_dir = task_store.task_dir(task_id)
    if asset_store.enabled:
        try:
            payload, content_type = asset_store.read_bytes(task_id, asset_path)
            return Response(content=payload, media_type=content_type)
        except Exception:
            pass

    local_path = asset_store.local_asset_path(task_dir, asset_path)
    if local_path and local_path.exists():
        return FileResponse(local_path)

    raise HTTPException(status_code=404, detail="资产不存在")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=BACKEND_RELOAD)
