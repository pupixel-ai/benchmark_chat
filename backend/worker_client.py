"""
HTTP client used by the control-plane to talk to private worker instances.
"""
from __future__ import annotations

import time

import requests

from config import WORKER_BOOT_TIMEOUT_SECONDS, WORKER_INTERNAL_PORT, WORKER_SHARED_TOKEN


class WorkerClient:
    def __init__(self) -> None:
        self.enabled = bool(WORKER_SHARED_TOKEN and WORKER_INTERNAL_PORT > 0)
        self.port = WORKER_INTERNAL_PORT
        self.shared_token = WORKER_SHARED_TOKEN

    def wait_for_health(self, private_ip: str, timeout_seconds: int = WORKER_BOOT_TIMEOUT_SECONDS) -> dict:
        deadline = time.monotonic() + timeout_seconds
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                response = requests.get(
                    f"{self._base_url(private_ip)}/internal/health",
                    headers=self._headers(),
                    timeout=(3, 5),
                )
                self._raise_for_status(response, "worker 健康检查失败")
                return response.json()
            except Exception as exc:
                last_error = exc
                time.sleep(3)
        raise TimeoutError(f"等待 worker 健康检查通过超时: {last_error}")

    def ingest_uploads(self, private_ip: str, task_id: str, files: list[dict], max_photos: int, use_cache: bool) -> dict:
        multipart = [
            (
                "files",
                (
                    item["filename"],
                    item["payload"],
                    item.get("content_type") or "application/octet-stream",
                ),
            )
            for item in files
        ]
        response = requests.post(
            f"{self._base_url(private_ip)}/internal/tasks/{task_id}/ingest",
            headers=self._headers(),
            data={
                "max_photos": str(max_photos),
                "use_cache": "true" if use_cache else "false",
            },
            files=multipart,
            timeout=(10, 120),
        )
        self._raise_for_status(response, "上传文件到 worker 失败")
        return response.json()

    def fetch_status(self, private_ip: str, task_id: str) -> dict:
        response = requests.get(
            f"{self._base_url(private_ip)}/internal/tasks/{task_id}/status",
            headers=self._headers(),
            timeout=(5, 15),
        )
        self._raise_for_status(response, "读取 worker 状态失败")
        return response.json()

    def fetch_asset(self, private_ip: str, task_id: str, asset_path: str) -> tuple[bytes, str]:
        response = requests.get(
            f"{self._base_url(private_ip)}/internal/tasks/{task_id}/assets/{asset_path}",
            headers=self._headers(),
            timeout=(5, 60),
        )
        self._raise_for_status(response, "读取 worker 资产失败")
        return response.content, response.headers.get("Content-Type", "application/octet-stream")

    def request_delete(self, private_ip: str, task_id: str) -> dict:
        response = requests.delete(
            f"{self._base_url(private_ip)}/internal/tasks/{task_id}",
            headers=self._headers(),
            timeout=(5, 30),
        )
        self._raise_for_status(response, "删除 worker 任务失败")
        return response.json()

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.shared_token}"}

    def _base_url(self, private_ip: str) -> str:
        return f"http://{private_ip}:{self.port}"

    def _raise_for_status(self, response: requests.Response, fallback_message: str) -> None:
        if response.ok:
            return
        detail = None
        try:
            payload = response.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            detail = payload.get("detail")
        raise RuntimeError(detail or fallback_message)
