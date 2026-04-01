from __future__ import annotations

import importlib
import logging
from datetime import UTC, datetime
from typing import Any, Dict

from config import (
    FEISHU_APP_ID,
    FEISHU_APP_SECRET,
    FEISHU_CALLBACK_ENCRYPT_KEY,
    FEISHU_CALLBACK_VERIFICATION_TOKEN,
    FEISHU_LONG_CONNECTION_LOG_LEVEL,
    PROJECT_ROOT,
)

from .feishu import handle_feishu_callback

logger = logging.getLogger(__name__)


def normalize_long_connection_card_action_event(event: Any) -> Dict[str, Any]:
    payload = _as_dict(_get_value(event, "event"))
    operator = _as_dict(payload.get("operator"))
    action = _as_dict(payload.get("action"))
    context = _as_dict(payload.get("context"))

    return {
        "token": str(payload.get("token") or "").strip(),
        "operator": {
            "open_id": str(operator.get("open_id") or "").strip(),
            "user_id": str(operator.get("user_id") or "").strip(),
            "union_id": str(operator.get("union_id") or "").strip(),
            "tenant_key": str(operator.get("tenant_key") or "").strip(),
        },
        "action": {
            "value": dict(_as_dict(action.get("value"))),
            "form_value": dict(_as_dict(action.get("form_value"))),
            "tag": str(action.get("tag") or "").strip(),
            "option": str(action.get("option") or "").strip(),
            "name": str(action.get("name") or "").strip(),
            "input_value": str(action.get("input_value") or "").strip(),
        },
        "context": {
            "url": str(context.get("url") or "").strip(),
            "preview_token": str(context.get("preview_token") or "").strip(),
            "open_message_id": str(context.get("open_message_id") or "").strip(),
            "open_chat_id": str(context.get("open_chat_id") or "").strip(),
        },
        "delivery_type": str(payload.get("delivery_type") or "").strip(),
        "source": "feishu_long_connection",
        "submitted_at": _utcnow_iso(),
    }


def process_long_connection_card_action_event(*, project_root: str = PROJECT_ROOT, event: Any) -> Dict[str, Any]:
    normalized = normalize_long_connection_card_action_event(event)
    try:
        result = handle_feishu_callback(project_root=project_root, payload=normalized)
    except Exception as exc:
        logger.exception("Feishu long connection action failed: %s", exc)
        return {
            "toast": {
                "type": "danger",
                "content": f"卡片动作处理失败：{exc}",
            }
        }

    action = dict(result.get("action") or {})
    task = dict(result.get("task") or {})
    return {
        "toast": {
            "type": "info",
            "content": _build_success_toast(task=task, action=action),
        }
    }


def build_feishu_long_connection_event_handler(
    *,
    project_root: str = PROJECT_ROOT,
    encrypt_key: str = FEISHU_CALLBACK_ENCRYPT_KEY,
    verification_token: str = FEISHU_CALLBACK_VERIFICATION_TOKEN,
    log_level: str = FEISHU_LONG_CONNECTION_LOG_LEVEL,
) -> Any:
    sdk = _load_lark_oapi_sdk()
    response_class = _load_card_action_trigger_response_class()
    sdk_log_level = getattr(sdk.LogLevel, str(log_level or "INFO").upper(), sdk.LogLevel.INFO)

    def _on_card_action(event: Any) -> Any:
        payload = process_long_connection_card_action_event(project_root=project_root, event=event)
        return response_class(payload)

    return (
        sdk.EventDispatcherHandler.builder(encrypt_key, verification_token, sdk_log_level)
        .register_p2_card_action_trigger(_on_card_action)
        .build()
    )


def run_feishu_long_connection(
    *,
    project_root: str = PROJECT_ROOT,
    app_id: str = FEISHU_APP_ID,
    app_secret: str = FEISHU_APP_SECRET,
    encrypt_key: str = FEISHU_CALLBACK_ENCRYPT_KEY,
    verification_token: str = FEISHU_CALLBACK_VERIFICATION_TOKEN,
    log_level: str = FEISHU_LONG_CONNECTION_LOG_LEVEL,
) -> None:
    if not app_id or not app_secret:
        raise RuntimeError("feishu_credentials_missing")

    sdk = _load_lark_oapi_sdk()
    sdk_log_level = getattr(sdk.LogLevel, str(log_level or "INFO").upper(), sdk.LogLevel.INFO)
    event_handler = build_feishu_long_connection_event_handler(
        project_root=project_root,
        encrypt_key=encrypt_key,
        verification_token=verification_token,
        log_level=log_level,
    )
    client = sdk.ws.Client(
        app_id,
        app_secret,
        event_handler=event_handler,
        log_level=sdk_log_level,
    )
    logger.info("Starting Feishu long connection client")
    client.start()


def _load_lark_oapi_sdk() -> Any:
    try:
        return importlib.import_module("lark_oapi")
    except ModuleNotFoundError as exc:
        raise RuntimeError("feishu_long_connection_sdk_missing") from exc


def _load_card_action_trigger_response_class() -> Any:
    try:
        module = importlib.import_module("lark_oapi.event.callback.model.p2_card_action_trigger")
    except ModuleNotFoundError as exc:
        raise RuntimeError("feishu_long_connection_sdk_missing") from exc
    return getattr(module, "P2CardActionTriggerResponse")


def _get_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return {}


def _build_success_toast(*, task: Dict[str, Any], action: Dict[str, Any]) -> str:
    action_type = str(action.get("action_type") or "").strip()
    candidate_id = str(action.get("candidate_id") or "").strip()
    task_id = str(task.get("task_id") or "").strip() or "unknown_task"
    if action_type == "adopt_candidate" and candidate_id:
        return f"已记录 {task_id}，采用方案：{candidate_id}"
    if action_type == "custom_override":
        return f"已记录 {task_id} 的自定义处理意见"
    return f"已记录 {task_id}，暂不决策"


def _utcnow_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
