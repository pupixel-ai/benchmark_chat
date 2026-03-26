from __future__ import annotations

from fastapi import HTTPException, Response

from config import PROJECT_ROOT
from services.reflection.feishu import (
    build_difficult_case_alert_card_preview,
    build_reflection_task_card_preview,
    handle_feishu_callback,
    send_difficult_case_alert_for_case,
    send_reflection_task_card_for_task,
)


def _apply_no_store_headers(response: Response) -> Response:
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def preview_reflection_task_card(task_id: str, response: Response, current_user: dict) -> dict:
    _apply_no_store_headers(response)
    try:
        return build_reflection_task_card_preview(
            project_root=PROJECT_ROOT,
            user_name=str(current_user.get("username") or "").strip(),
            task_id=task_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="反思任务不存在") from exc


def preview_difficult_case_alert_card(case_id: str, response: Response, current_user: dict) -> dict:
    _apply_no_store_headers(response)
    try:
        return build_difficult_case_alert_card_preview(
            project_root=PROJECT_ROOT,
            user_name=str(current_user.get("username") or "").strip(),
            case_id=case_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="疑难 case 不存在") from exc


def submit_mock_feishu_action(task_id: str, payload: dict, response: Response, current_user: dict) -> dict:
    _apply_no_store_headers(response)
    try:
        return handle_feishu_callback(
            project_root=PROJECT_ROOT,
            payload={
                "operator": {
                    "open_id": str(current_user.get("username") or "").strip(),
                    "user_id": str(current_user.get("user_id") or "").strip(),
                },
                "action": {
                    "value": {
                        "task_id": task_id,
                        "user_name": str(current_user.get("username") or "").strip(),
                        "action_type": str(payload.get("action_type") or "defer").strip(),
                        "candidate_id": str(payload.get("candidate_id") or "").strip(),
                    },
                    "form_value": {
                        "reviewer_note": str(payload.get("reviewer_note") or "").strip(),
                    },
                },
                "source": "feishu_mock",
            },
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="反思任务不存在") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def send_reflection_task_card(task_id: str, payload: dict, response: Response, current_user: dict) -> dict:
    _apply_no_store_headers(response)
    try:
        return send_reflection_task_card_for_task(
            project_root=PROJECT_ROOT,
            user_name=str(current_user.get("username") or "").strip(),
            task_id=task_id,
            receive_id=str(payload.get("receive_id") or "").strip(),
            receive_id_type=str(payload.get("receive_id_type") or "open_id").strip() or "open_id",
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="反思任务不存在") from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def send_difficult_case_alert(case_id: str, payload: dict, response: Response, current_user: dict) -> dict:
    _apply_no_store_headers(response)
    try:
        return send_difficult_case_alert_for_case(
            project_root=PROJECT_ROOT,
            user_name=str(current_user.get("username") or "").strip(),
            case_id=case_id,
            receive_id=str(payload.get("receive_id") or "").strip(),
            receive_id_type=str(payload.get("receive_id_type") or "open_id").strip() or "open_id",
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="疑难 case 不存在") from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def handle_feishu_callback_event(payload: dict, response: Response) -> dict:
    _apply_no_store_headers(response)
    try:
        return handle_feishu_callback(project_root=PROJECT_ROOT, payload=payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="反思任务不存在") from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
