"""
FastAPI backend entrypoint.
"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import BackgroundTasks, Cookie, Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.auth import (
    AUTH_SESSION_COOKIE_NAME,
    authenticate_response,
    get_current_user,
    login_user,
    logout_current_session,
    register_user,
)
from backend.task_store import TaskStore
from backend.upload_utils import (
    UPLOAD_FAILURES_FILENAME,
    preview_filename,
    save_preview_as_webp,
    save_upload_original,
    stored_upload_filename,
    task_asset_path,
    write_upload_failures,
)
from backend.worker_client import WorkerClient
from backend.worker_manager import WorkerManager
from config import (
    ALLOW_SELF_REGISTRATION,
    APP_ROLE,
    ASSET_URL_PREFIX,
    BACKEND_HOST,
    BACKEND_PORT,
    BACKEND_RELOAD,
    CORS_ALLOW_ORIGINS,
    FRONTEND_ORIGIN,
    HIGH_SECURITY_MODE,
    MAX_UPLOAD_PHOTOS,
    TASKS_DIR,
)
from services.asset_store import TaskAssetStore
from services.pipeline_service import MemoryPipelineService
from utils import save_json


app = FastAPI(title="Memory Engineering API", version="1.0.0")
task_store = TaskStore()
asset_store = TaskAssetStore()
worker_manager = WorkerManager()
worker_client = WorkerClient()

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(CORS_ALLOW_ORIGINS),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path(TASKS_DIR).mkdir(parents=True, exist_ok=True)


class AuthPayload(BaseModel):
    username: str
    password: str


def _apply_no_store_headers(response: Response) -> Response:
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def _remote_worker_mode_enabled() -> bool:
    return APP_ROLE == "control-plane" and worker_manager.enabled and worker_client.enabled


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

    updates = {}
    for key in ("status", "stage", "progress", "uploads", "result", "error"):
        if key in remote_payload:
            updates[key] = remote_payload[key]
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
    refreshed = task_store.get_task(task["task_id"], user_id=user_id)
    return refreshed or task


async def _create_remote_task(
    files: List[UploadFile],
    max_photos: int,
    use_cache: bool,
    current_user: dict,
) -> dict:
    upload_payloads = []
    for upload in files:
        payload = await upload.read()
        if not payload:
            raise HTTPException(status_code=400, detail=f"文件为空: {upload.filename or '未命名文件'}")
        upload_payloads.append(
            {
                "filename": os.path.basename(upload.filename or "upload.bin"),
                "content_type": upload.content_type or "application/octet-stream",
                "payload": payload,
            }
        )

    task_id = uuid.uuid4().hex
    task_store.create_task(
        task_id,
        upload_count=len(upload_payloads),
        user_id=current_user["user_id"],
        provision_local_dir=False,
    )

    launched_worker = None
    try:
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
        ingest_payload = worker_client.ingest_uploads(
            ready_worker.private_ip,
            task_id,
            upload_payloads,
            min(max_photos, len(upload_payloads)),
            use_cache,
        )

        task_store.update_worker_state(
            task_id,
            worker_status=ingest_payload.get("worker_status", ready_worker.state),
            status=ingest_payload.get("status", "queued"),
            stage=ingest_payload.get("stage", "queued"),
            uploads=ingest_payload.get("uploads"),
            progress=ingest_payload.get("progress"),
        )

        return {
            "task_id": task_id,
            "status": ingest_payload.get("status", "queued"),
            "upload_count": len(upload_payloads),
            "max_photos": min(max_photos, len(upload_payloads)),
            "task_url": f"/api/tasks/{task_id}",
            "worker_status": ingest_payload.get("worker_status", ready_worker.state),
        }
    except HTTPException:
        if launched_worker is not None:
            worker_manager.terminate_worker(launched_worker.instance_id)
        task_store.delete_task(task_id, user_id=current_user["user_id"])
        raise
    except Exception as exc:
        if launched_worker is not None:
            worker_manager.terminate_worker(launched_worker.instance_id)
        task_store.delete_task(task_id, user_id=current_user["user_id"])
        raise HTTPException(status_code=502, detail=f"创建 worker 任务失败: {exc}") from exc
    finally:
        for upload in files:
            await upload.close()


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
def healthcheck(response: Response):
    _apply_no_store_headers(response)
    return {
        "status": "ok",
        "app_role": APP_ROLE,
        "frontend_origin": FRONTEND_ORIGIN,
        "max_upload_photos": MAX_UPLOAD_PHOTOS,
        "self_registration_enabled": ALLOW_SELF_REGISTRATION,
        "high_security_mode": HIGH_SECURITY_MODE,
        "asset_url_prefix": ASSET_URL_PREFIX,
        "object_storage_enabled": asset_store.enabled,
        "object_storage_bucket": asset_store.bucket or None,
        "worker_orchestration_enabled": _remote_worker_mode_enabled(),
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
        "assets_prefix": ASSET_URL_PREFIX,
        "auth_me": "/api/auth/me",
    }


@app.post("/api/tasks")
async def create_task(
    background_tasks: BackgroundTasks,
    response: Response,
    files: List[UploadFile] = File(...),
    max_photos: int = Form(MAX_UPLOAD_PHOTOS),
    use_cache: bool = Form(False),
    current_user: dict = Depends(get_current_user),
):
    _apply_no_store_headers(response)
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

    if _remote_worker_mode_enabled():
        return await _create_remote_task(files, max_photos, use_cache, current_user)

    task_id = uuid.uuid4().hex
    task_dir = task_store.task_dir(task_id)
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
            if asset_store.enabled:
                asset_store.upload_file(task_id, original_asset_path, destination)

            preview_url = None
            try:
                save_preview_as_webp(payload, preview_destination)
                preview_asset_path = task_asset_path("previews", preview_name)
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

    write_upload_failures(task_dir, upload_failures)
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
    return _sync_task_from_worker(task, current_user["user_id"])


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=BACKEND_RELOAD)
