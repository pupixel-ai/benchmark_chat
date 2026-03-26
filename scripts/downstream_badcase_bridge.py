#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict


def _pre_parse_user_name() -> None:
    for index, arg in enumerate(sys.argv):
        if arg == "--user-name" and index + 1 < len(sys.argv):
            os.environ["MEMORY_USER_NAME"] = sys.argv[index + 1]
            return


_pre_parse_user_name()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import (  # noqa: E402
    DOWNSTREAM_BADCASE_QUEUE_PATH,
    DOWNSTREAM_NOTIFICATION_STATUS_PATH,
    FEISHU_WEBHOOK_URL,
    PROFILE_AGENT_ROOT,
    REVIEW_BASE_URL,
)
from services.memory_pipeline.feedback_cases import (  # noqa: E402
    load_pending_feedback_cases,
    mark_feedback_cases_processed,
    mirror_pending_cases_to_downstream,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="主工程 badcase -> 下游审批桥接")
    parser.add_argument("command", choices=["propose", "apply"])
    parser.add_argument("--user-name", type=str, default=os.environ.get("MEMORY_USER_NAME", "default"))
    args = parser.parse_args()
    os.environ["MEMORY_USER_NAME"] = args.user_name

    try:
        if args.command == "propose":
            result = run_propose()
        else:
            result = run_apply()
    except Exception as exc:
        result = {
            "bridge_command": args.command,
            "state": "failed",
            "error": str(exc),
            "notification_status": {
                "command": args.command,
                "state": "failed",
                "error": str(exc),
                "notification_failures": [],
            },
        }
        _save_notification_status(result)
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_propose() -> Dict[str, Any]:
    pending_cases = load_pending_feedback_cases(DOWNSTREAM_BADCASE_QUEUE_PATH)
    mirror_path, mirrored_count = mirror_pending_cases_to_downstream(
        pending_cases=pending_cases,
        profile_agent_root=PROFILE_AGENT_ROOT,
    )

    Storage, RuleEvolver, downstream_config, notify_module = _load_downstream_runtime()
    storage = Storage()
    evolver = RuleEvolver(storage)
    propose_result = evolver.propose(human_corrections=[])

    notification_status = {
        "command": "propose",
        "state": "not_triggered",
        "error": "",
        "notification_failures": [],
    }
    webhook_url = FEISHU_WEBHOOK_URL or getattr(downstream_config, "FEISHU_WEBHOOK_URL", "")
    review_base_url = REVIEW_BASE_URL or getattr(downstream_config, "REVIEW_BASE_URL", "http://localhost:8080")
    proposals_file = Path(getattr(downstream_config, "PROPOSALS_FILE"))
    if propose_result.get("proposed", 0) > 0 and webhook_url:
        proposals = []
        if proposals_file.exists():
            with proposals_file.open("r", encoding="utf-8") as handle:
                proposals = json.load(handle)
        try:
            notify_module.send_proposals_notification(
                webhook_url=webhook_url,
                review_url=f"{review_base_url}/review",
                proposals=proposals,
                hard_cases=pending_cases,
                batch_info={"album_count": "?", "hard_cases": propose_result.get("systematic", 0)},
            )
            notification_status["state"] = "sent"
        except Exception as exc:  # pragma: no cover - 通知失败不阻断
            notification_status["state"] = "failed"
            notification_status["error"] = str(exc)
            notification_status["notification_failures"].append(
                {"stage": "propose", "error": str(exc)}
            )
    elif propose_result.get("proposed", 0) > 0:
        notification_status["state"] = "skipped_no_webhook"

    status_payload = {
        "bridge_command": "propose",
        "queue_path": DOWNSTREAM_BADCASE_QUEUE_PATH,
        "pending_count": len(pending_cases),
        "mirror_path": mirror_path,
        "mirrored_count": mirrored_count,
        "propose_result": propose_result,
        "notification_status": notification_status,
    }
    _save_notification_status(status_payload)
    return status_payload


def run_apply() -> Dict[str, Any]:
    pending_before = load_pending_feedback_cases(DOWNSTREAM_BADCASE_QUEUE_PATH)
    mirror_pending_cases_to_downstream(
        pending_cases=pending_before,
        profile_agent_root=PROFILE_AGENT_ROOT,
    )

    Storage, RuleEvolver, downstream_config, notify_module = _load_downstream_runtime()
    storage = Storage()
    evolver = RuleEvolver(storage)
    apply_result = evolver.apply_approved()

    should_mark_processed = (
        "error" not in apply_result
        and (
            apply_result.get("applied", 0) > 0
            or apply_result.get("rejected", 0) > 0
            or apply_result.get("pending", 0) == 0
        )
    )
    processed_summary = (
        mark_feedback_cases_processed(queue_path=DOWNSTREAM_BADCASE_QUEUE_PATH)
        if should_mark_processed
        else {"updated": 0, "total": len(pending_before)}
    )
    pending_after = load_pending_feedback_cases(DOWNSTREAM_BADCASE_QUEUE_PATH)
    mirror_path, mirrored_count = mirror_pending_cases_to_downstream(
        pending_cases=pending_after,
        profile_agent_root=PROFILE_AGENT_ROOT,
    )

    notification_status = {
        "command": "apply",
        "state": "not_triggered",
        "error": "",
        "notification_failures": [],
    }
    webhook_url = FEISHU_WEBHOOK_URL or getattr(downstream_config, "FEISHU_WEBHOOK_URL", "")
    if webhook_url:
        try:
            notify_module.send_apply_result(
                webhook_url=webhook_url,
                applied=apply_result.get("applied", 0),
                rejected=apply_result.get("rejected", 0),
                version=evolver.version,
            )
            notification_status["state"] = "sent"
        except Exception as exc:  # pragma: no cover - 通知失败不阻断
            notification_status["state"] = "failed"
            notification_status["error"] = str(exc)
            notification_status["notification_failures"].append(
                {"stage": "apply", "error": str(exc)}
            )
    else:
        notification_status["state"] = "skipped_no_webhook"

    status_payload = {
        "bridge_command": "apply",
        "queue_path": DOWNSTREAM_BADCASE_QUEUE_PATH,
        "pending_before_count": len(pending_before),
        "pending_after_count": len(pending_after),
        "mirror_path": mirror_path,
        "mirrored_count": mirrored_count,
        "apply_result": apply_result,
        "processed_summary": processed_summary,
        "notification_status": notification_status,
    }
    _save_notification_status(status_payload)
    return status_payload


def _save_notification_status(payload: Dict[str, Any]) -> None:
    path = Path(DOWNSTREAM_NOTIFICATION_STATUS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _load_downstream_runtime():
    profile_agent_root = Path(PROFILE_AGENT_ROOT)
    init_path = profile_agent_root / "__init__.py"
    if not init_path.exists():
        raise FileNotFoundError(f"未找到下游 profile_agent 包入口: {init_path}")

    package_name = "profile_agent"
    existing = sys.modules.get(package_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", "")
        if existing_file and Path(existing_file).resolve() != init_path.resolve():
            for module_name in list(sys.modules):
                if module_name == package_name or module_name.startswith(f"{package_name}."):
                    sys.modules.pop(module_name, None)

    if package_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            package_name,
            init_path,
            submodule_search_locations=[str(profile_agent_root)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载 profile_agent: {profile_agent_root}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[package_name] = module
        spec.loader.exec_module(module)

    Storage = importlib.import_module("profile_agent.storage").Storage
    RuleEvolver = importlib.import_module("profile_agent.agents.rule_evolver").RuleEvolver
    downstream_config = importlib.import_module("profile_agent.config")
    notify_module = importlib.import_module("profile_agent.feishu.notify")
    return Storage, RuleEvolver, downstream_config, notify_module


if __name__ == "__main__":
    main()
