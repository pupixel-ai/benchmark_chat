"""
FastAPI backend entrypoint.
"""
from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.task_store import TaskStore
from config import (
    BACKEND_HOST,
    BACKEND_PORT,
    FRONTEND_ORIGIN,
    MAX_UPLOAD_PHOTOS,
    RUNS_URL_PREFIX,
    TASKS_DIR,
)
from services.pipeline_service import MemoryPipelineService


app = FastAPI(title="Memory Engineering API", version="1.0.0")
task_store = TaskStore()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://127.0.0.1:3000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Path(TASKS_DIR).mkdir(parents=True, exist_ok=True)
app.mount(RUNS_URL_PREFIX, StaticFiles(directory=TASKS_DIR), name="runs")


def _safe_filename(filename: str, fallback: str) -> str:
    basename = os.path.basename(filename or fallback)
    basename = basename.replace("/", "_").replace("\\", "_")
    return basename or fallback


def _run_pipeline_task(task_id: str, max_photos: int, use_cache: bool):
    task_dir = task_store.task_dir(task_id)

    def progress_callback(stage: str, payload: dict):
        task_store.update_task(task_id, status="running", stage=stage, progress=payload)

    try:
        task_store.update_task(task_id, status="running", stage="starting", error=None)
        service = MemoryPipelineService(task_id=task_id, task_dir=str(task_dir))
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
        "runs_url_prefix": RUNS_URL_PREFIX,
    }


@app.post("/api/tasks")
async def create_task(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    max_photos: int = Form(MAX_UPLOAD_PHOTOS),
    use_cache: bool = Form(False),
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
    uploads_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    for index, upload in enumerate(files, start=1):
        safe_name = _safe_filename(upload.filename or "", f"upload_{index:03d}.bin")
        stored_name = f"{index:03d}_{safe_name}"
        destination = uploads_dir / stored_name
        with destination.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        saved_files.append({
            "filename": upload.filename or stored_name,
            "stored_filename": stored_name,
            "path": str(destination),
        })
        await upload.close()

    task_payload = task_store.create_task(task_id, upload_count=len(saved_files))
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
def list_tasks(limit: int = 20):
    safe_limit = max(1, min(limit, 100))
    return {
        "tasks": task_store.list_tasks(limit=safe_limit),
    }


@app.get("/api/tasks/{task_id}")
def get_task(task_id: str):
    task = task_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)
