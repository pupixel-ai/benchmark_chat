from __future__ import annotations

from fastapi import HTTPException, Response

from config import PROJECT_ROOT
from services.reflection import (
    get_difficult_case_detail_payload,
    get_reflection_task_detail_payload,
    list_reflection_tasks_payload,
)


def _apply_no_store_headers(response: Response) -> Response:
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def list_reflection_tasks(response: Response, current_user: dict) -> dict:
    _apply_no_store_headers(response)
    return list_reflection_tasks_payload(
        project_root=PROJECT_ROOT,
        user_name=str(current_user.get("username") or "").strip(),
    )


def get_reflection_task_detail(task_id: str, response: Response, current_user: dict) -> dict:
    _apply_no_store_headers(response)
    try:
        return get_reflection_task_detail_payload(
            project_root=PROJECT_ROOT,
            user_name=str(current_user.get("username") or "").strip(),
            task_id=task_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="反思任务不存在") from exc


def get_difficult_case_detail(case_id: str, response: Response, current_user: dict) -> dict:
    _apply_no_store_headers(response)
    try:
        return get_difficult_case_detail_payload(
            project_root=PROJECT_ROOT,
            user_name=str(current_user.get("username") or "").strip(),
            case_id=case_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="疑难 case 不存在") from exc
