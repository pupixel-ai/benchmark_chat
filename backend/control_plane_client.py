"""
HTTP client used by workers to push terminal status back to the control-plane.
"""
from __future__ import annotations

import requests

from config import CONTROL_PLANE_INTERNAL_ORIGIN, WORKER_SHARED_TOKEN


class ControlPlaneClient:
    def __init__(self) -> None:
        self.origin = CONTROL_PLANE_INTERNAL_ORIGIN.rstrip("/")
        self.shared_token = WORKER_SHARED_TOKEN
        self.enabled = bool(self.origin and self.shared_token)

    def publish_terminal_update(self, task_id: str, payload: dict) -> dict:
        if not self.enabled:
            raise RuntimeError("control-plane terminal callback 未启用")
        response = requests.post(
            f"{self.origin}/internal/tasks/{task_id}/terminal-update",
            headers={"Authorization": f"Bearer {self.shared_token}"},
            json=payload,
            timeout=(5, 30),
        )
        self._raise_for_status(response, "回调 control-plane 终态失败")
        return response.json()

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
