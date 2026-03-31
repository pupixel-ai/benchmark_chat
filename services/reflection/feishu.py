from __future__ import annotations

import json
import os
import re
import uuid

# 飞书 API 走直连，不经过系统代理
os.environ.setdefault("NO_PROXY", "open.feishu.cn")
os.environ.setdefault("no_proxy", "open.feishu.cn")
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests

from config import (
    FEISHU_APPROVAL_RECEIVE_ID,
    FEISHU_APPROVAL_RECEIVE_ID_TYPE,
    FEISHU_API_BASE_URL,
    FEISHU_APP_ID,
    FEISHU_APP_SECRET,
    FEISHU_CALLBACK_VERIFICATION_TOKEN,
    FEISHU_DEFAULT_RECEIVE_ID,
    FEISHU_DEFAULT_RECEIVE_ID_TYPE,
    PROJECT_ROOT,
    REVIEW_BASE_URL,
)

from .labels import load_bilingual_label_rows, lookup_bilingual_label, resolve_profile_field_hint
from .storage import build_reflection_asset_paths, ensure_reflection_root
from .tasks import get_difficult_case_detail_payload, get_reflection_task_detail_payload
from .upstream_agent import MemoryEngineerAgent, MutationExecutor


def _build_card_shell(
    *,
    title: str,
    subtitle: str,
    template: str,
    summary: str,
    elements: List[Dict[str, Any]],
    text_tags: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    header: Dict[str, Any] = {
        "title": {"tag": "plain_text", "content": title},
        "template": template,
        "padding": "12px 8px 12px 8px",
    }
    if subtitle:
        header["subtitle"] = {"tag": "plain_text", "content": subtitle}
    if text_tags:
        header["text_tag_list"] = text_tags[:3]
    return {
        "schema": "2.0",
        "config": {
            "enable_forward": True,
            "update_multi": True,
            "width_mode": "fill",
            "summary": {
                "content": _truncate_card_text(_plain_language_card_text(summary or title), limit=80),
            },
        },
        "header": header,
        "body": {
            "padding": "12px 12px 12px 12px",
            "vertical_spacing": "12px",
            "elements": elements,
        },
    }


def _build_header_tag(content: str, *, color: str = "neutral") -> Dict[str, Any]:
    return {
        "tag": "text_tag",
        "text": {"tag": "plain_text", "content": content},
        "color": color,
    }


def _build_markdown_block(
    content: str,
    *,
    text_size: str = "normal",
    margin: str = "0",
) -> Dict[str, Any]:
    return {
        "tag": "markdown",
        "content": content,
        "text_size": text_size,
        "margin": margin,
    }


def _build_button_component(
    *,
    text: str,
    button_type: str = "default",
    callback_value: Dict[str, Any] | None = None,
    url: str | None = None,
    confirm_title: str | None = None,
    confirm_text: str | None = None,
) -> Dict[str, Any]:
    behaviors: List[Dict[str, Any]] = []
    if callback_value is not None:
        behaviors.append({"type": "callback", "value": callback_value})
    if url:
        behaviors.append({"type": "open_url", "default_url": url})
    button: Dict[str, Any] = {
        "tag": "button",
        "text": {"tag": "plain_text", "content": text},
        "type": button_type,
        "size": "medium",
        "width": "fill",
        "behaviors": behaviors,
        "margin": "0",
    }
    if confirm_title and confirm_text:
        button["confirm"] = {
            "title": {"tag": "plain_text", "content": confirm_title},
            "text": {"tag": "plain_text", "content": confirm_text},
        }
    return button


def build_reflection_task_card(task_detail_payload: Dict[str, Any], *, review_base_url: str = REVIEW_BASE_URL) -> Dict[str, Any]:
    task = dict(task_detail_payload.get("task") or {})
    proposal = dict(task_detail_payload.get("proposal") or {})
    change_request = dict(task_detail_payload.get("change_request") or {})
    pattern = dict(task_detail_payload.get("pattern") or {})
    evidence_refs = list(task_detail_payload.get("evidence_refs") or [])
    support_cases = list(task_detail_payload.get("support_cases") or [])
    history_summary = dict(task_detail_payload.get("history_summary") or {})
    review_url = _absolute_review_url(review_base_url, str(task.get("detail_url") or ""))
    field_label = _resolve_task_field_label(task=task, pattern=pattern, proposal=proposal, support_cases=support_cases)

    if str(task.get("task_type") or "") == "proposal_review":
        return _build_proposal_review_card(
            task=task,
            proposal=proposal,
            review_url=review_url,
            evidence_refs=evidence_refs,
            support_cases=support_cases,
        )
    if str(task.get("task_type") or "") == "engineering_execute_review":
        return _build_engineering_execute_review_card(
            task=task,
            change_request=change_request,
            proposal=proposal,
            review_url=review_url,
            evidence_refs=evidence_refs,
            support_cases=support_cases,
        )

    evidence_lines = _build_evidence_snippet_lines(evidence_refs=evidence_refs, support_cases=support_cases)

    actions = [
        _build_candidate_button(task, option)
        for option in list(task.get("options") or [])
    ]
    actions.extend(
        [
            _build_button_component(
                text="暂不决策",
                button_type="default",
                callback_value=_build_callback_value(task, action_type="defer"),
            ),
        ]
    )
    candidate_lines = []
    for option in list(task.get("options") or []):
        label = _humanize_option(option)
        marker = "（系统更推荐）" if option == str(task.get("recommended_option") or "") else ""
        candidate_lines.append(f"- {label}{marker}")

    summary_lines = [
        f"- 画像标签：{field_label or '未识别字段标签'}",
        f"- 这次想解决：{_plain_language_card_text(str(task.get('summary') or pattern.get('summary') or '暂无摘要'))}",
        f"- 系统现在更建议：{_humanize_option(str(task.get('recommended_option') or 'watch_only'))}",
    ]
    if evidence_lines:
        summary_lines.append("- 关键证据：")
        summary_lines.extend(f"  - {line}" for line in evidence_lines)
    if candidate_lines:
        summary_lines.append("- 你可以怎么选：")
        summary_lines.extend(f"  {line}" for line in candidate_lines)
    summary_lines.extend(
        [
            f"- 同类问题以前出现过：{int(history_summary.get('similar_pattern_count') or 0)} 次",
            f"- 当前建议这条路以前成功过：{int(history_summary.get('recommended_option_success_count') or 0)} 次",
        ]
    )

    elements = [
        _build_markdown_block("\n".join(summary_lines), text_size="normal"),
        *actions,
    ]

    return _build_card_shell(
        title="请帮我看一下这条问题",
        subtitle=field_label or "",
        template="blue",
        summary=str(task.get("summary") or pattern.get("summary") or "反思任务待处理"),
        text_tags=[],
        elements=elements,
    )


def build_difficult_case_alert_card(difficult_case_payload: Dict[str, Any], *, review_base_url: str = REVIEW_BASE_URL) -> Dict[str, Any]:
    case_payload = dict(difficult_case_payload.get("case") or {})
    comparison = dict(difficult_case_payload.get("gt_comparison") or {})
    evidence_refs = list(difficult_case_payload.get("evidence_refs") or [])
    review_url = _absolute_review_url(review_base_url, str(case_payload.get("detail_url") or ""))
    field_label = _profile_field_card_label(str(case_payload.get("dimension") or ""))
    comparison_grade = str(comparison.get("grade") or case_payload.get("comparison_grade") or "unknown").strip()
    difficulty_reason = _localize_card_text(
        str(
            difficult_case_payload.get("difficulty_reason")
            or case_payload.get("summary")
            or "当前归因还不稳定，需要进一步人工判断。"
        )
    )

    evidence_lines = []
    for index, ref in enumerate(evidence_refs[:3], start=1):
        summary = str(ref.get("description") or ref.get("summary") or "未提供证据摘要").strip()
        ref_id = str(ref.get("source_id") or ref.get("ref") or "").strip()
        evidence_lines.append(f"{index}. {summary}" + (f"（{ref_id}）" if ref_id else ""))
    if not evidence_lines:
        evidence_lines.append("1. 当前没有聚合到可展示的证据摘要")

    comparison_lines = [
        f"- GT 标注: {_render_case_value_for_card(comparison.get('gt_value'))}",
        f"- 当前识别: {_render_case_value_for_card(comparison.get('output_value'))}",
    ]
    causality_route = str((difficult_case_payload.get("route_decision") or {}).get("causality_route") or case_payload.get("causality_route") or "").strip()
    if causality_route:
        comparison_lines.append(f"- 当前归因: {_humanize_label('causality_route', causality_route)}")

    summary_lines = [
        f"- 画像标签：{field_label or '未识别字段标签'}",
        f"- 这条问题：{_plain_language_card_text(str(case_payload.get('summary') or '暂无摘要'))}",
        *comparison_lines,
        f"- 为什么现在还难下结论：{difficulty_reason}",
    ]
    if evidence_lines:
        summary_lines.append("- 关键线索：")
        summary_lines.extend(f"  - {line}" for line in evidence_lines)
    elements = [_build_markdown_block("\n".join(summary_lines))]
    return _build_card_shell(
        title="这条疑难问题请看一下",
        subtitle=field_label or "",
        template="orange",
        summary=str(case_payload.get("summary") or "疑难 case 告警"),
        text_tags=[],
        elements=elements,
    )


def build_difficult_case_alert_card_preview(*, project_root: str, user_name: str, case_id: str) -> Dict[str, Any]:
    detail_payload = get_difficult_case_detail_payload(project_root=project_root, user_name=user_name, case_id=case_id)
    return {
        "case_id": case_id,
        "card": build_difficult_case_alert_card(detail_payload),
        "detail": detail_payload,
    }


def send_reflection_task_card(
    *,
    task_detail_payload: Dict[str, Any],
    receive_id: str,
    receive_id_type: str = "open_id",
    review_base_url: str = REVIEW_BASE_URL,
    app_id: str = FEISHU_APP_ID,
    app_secret: str = FEISHU_APP_SECRET,
    api_base_url: str = FEISHU_API_BASE_URL,
) -> Dict[str, Any]:
    if not app_id or not app_secret:
        raise RuntimeError("feishu_credentials_missing")
    if not receive_id:
        raise ValueError("receive_id is required")

    tenant_access_token = _get_tenant_access_token(
        app_id=app_id,
        app_secret=app_secret,
        api_base_url=api_base_url,
    )
    card = build_reflection_task_card(task_detail_payload, review_base_url=review_base_url)
    response = requests.post(
        f"{api_base_url.rstrip('/')}/open-apis/im/v1/messages?receive_id_type={receive_id_type}",
        json={
            "receive_id": receive_id,
            "msg_type": "interactive",
            "content": json.dumps(card, ensure_ascii=False),
        },
        headers={
            "Authorization": f"Bearer {tenant_access_token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "message_id": str((payload.get("data") or {}).get("message_id") or ""),
        "raw_response": payload,
        "card": card,
    }


def send_difficult_case_alert(
    *,
    difficult_case_payload: Dict[str, Any],
    receive_id: str,
    receive_id_type: str = "open_id",
    review_base_url: str = REVIEW_BASE_URL,
    app_id: str = FEISHU_APP_ID,
    app_secret: str = FEISHU_APP_SECRET,
    api_base_url: str = FEISHU_API_BASE_URL,
) -> Dict[str, Any]:
    if not app_id or not app_secret:
        raise RuntimeError("feishu_credentials_missing")
    if not receive_id:
        raise ValueError("receive_id is required")

    tenant_access_token = _get_tenant_access_token(
        app_id=app_id,
        app_secret=app_secret,
        api_base_url=api_base_url,
    )
    card = build_difficult_case_alert_card(difficult_case_payload, review_base_url=review_base_url)
    response = requests.post(
        f"{api_base_url.rstrip('/')}/open-apis/im/v1/messages?receive_id_type={receive_id_type}",
        json={
            "receive_id": receive_id,
            "msg_type": "interactive",
            "content": json.dumps(card, ensure_ascii=False),
        },
        headers={
            "Authorization": f"Bearer {tenant_access_token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "message_id": str((payload.get("data") or {}).get("message_id") or ""),
        "raw_response": payload,
        "card": card,
    }


def send_reflection_task_card_for_task(
    *,
    project_root: str = PROJECT_ROOT,
    user_name: str,
    task_id: str,
    receive_id: str,
    receive_id_type: str = "open_id",
    review_base_url: str = REVIEW_BASE_URL,
    app_id: str = FEISHU_APP_ID,
    app_secret: str = FEISHU_APP_SECRET,
    api_base_url: str = FEISHU_API_BASE_URL,
) -> Dict[str, Any]:
    detail_payload = get_reflection_task_detail_payload(project_root=project_root, user_name=user_name, task_id=task_id)
    send_result = send_reflection_task_card(
        task_detail_payload=detail_payload,
        receive_id=receive_id,
        receive_id_type=receive_id_type,
        review_base_url=review_base_url,
        app_id=app_id,
        app_secret=app_secret,
        api_base_url=api_base_url,
    )
    task = mark_task_feishu_sent(
        project_root=project_root,
        user_name=user_name,
        task_id=task_id,
        message_id=str(send_result.get("message_id") or ""),
    )
    return {
        "task": task,
        "message_id": str(send_result.get("message_id") or ""),
        "card": send_result["card"],
        "raw_response": send_result["raw_response"],
    }


def send_difficult_case_alert_for_case(
    *,
    project_root: str = PROJECT_ROOT,
    user_name: str,
    case_id: str,
    receive_id: str,
    receive_id_type: str = "open_id",
    review_base_url: str = REVIEW_BASE_URL,
    app_id: str = FEISHU_APP_ID,
    app_secret: str = FEISHU_APP_SECRET,
    api_base_url: str = FEISHU_API_BASE_URL,
) -> Dict[str, Any]:
    detail_payload = get_difficult_case_detail_payload(project_root=project_root, user_name=user_name, case_id=case_id)
    send_result = send_difficult_case_alert(
        difficult_case_payload=detail_payload,
        receive_id=receive_id,
        receive_id_type=receive_id_type,
        review_base_url=review_base_url,
        app_id=app_id,
        app_secret=app_secret,
        api_base_url=api_base_url,
    )
    case_payload = mark_difficult_case_feishu_sent(
        project_root=project_root,
        user_name=user_name,
        case_id=case_id,
        message_id=str(send_result.get("message_id") or ""),
    )
    return {
        "case": case_payload,
        "message_id": str(send_result.get("message_id") or ""),
        "card": send_result["card"],
        "raw_response": send_result["raw_response"],
    }


def handle_feishu_callback(*, project_root: str = PROJECT_ROOT, payload: Dict[str, Any]) -> Dict[str, Any]:
    challenge = str(payload.get("challenge") or "").strip()
    if challenge:
        return {"challenge": challenge}

    _validate_callback_token(payload)
    normalized = _normalize_callback_payload(payload)
    result = apply_reflection_task_action(
        project_root=project_root,
        user_name=str(normalized["user_name"]),
        task_id=str(normalized["task_id"]),
        action_payload=normalized,
    )
    return {
        "status": "ok",
        "task": result["task"],
        "action": result["action"],
    }


def apply_reflection_task_action(
    *,
    project_root: str,
    user_name: str,
    task_id: str,
    action_payload: Dict[str, Any],
) -> Dict[str, Any]:
    paths = build_reflection_asset_paths(project_root=project_root, user_name=user_name)
    ensure_reflection_root(paths)
    _touch_action_and_feedback_files(paths)
    tasks = _read_jsonl_records(paths.tasks_path)
    updated_task: Dict[str, Any] | None = None
    action_record = _build_action_record(task_id=task_id, user_name=user_name, action_payload=action_payload)
    _require_reviewer_note_if_needed(action_record)
    proposals = _read_jsonl_records(paths.proposals_path)
    change_requests = _read_jsonl_records(paths.engineering_change_requests_path)
    proposal_lookup = {
        str(payload.get("proposal_id") or "").strip(): payload
        for payload in proposals
        if str(payload.get("proposal_id") or "").strip()
    }
    proposal_payload: Dict[str, Any] | None = proposal_lookup.get(str(action_payload.get("proposal_id") or "").strip())
    change_request_lookup = {
        str(payload.get("change_request_id") or "").strip(): payload
        for payload in change_requests
        if str(payload.get("change_request_id") or "").strip()
    }
    change_request_payload: Dict[str, Any] | None = change_request_lookup.get(str(action_payload.get("change_request_id") or "").strip())
    engineering_task_payload: Dict[str, Any] | None = None
    engineering_send_result: Dict[str, Any] | None = None

    for payload in tasks:
        if str(payload.get("task_id") or "") != task_id:
            continue
        payload["updated_at"] = action_record["submitted_at"]
        payload["reviewer_note"] = action_record["reviewer_note"]
        payload["reviewed_by"] = action_record["operator_open_id"] or action_record["operator_user_id"]
        payload["last_action_type"] = action_record["action_type"]
        payload["feishu_status"] = "acted"
        if action_record["action_type"] == "adopt_candidate":
            payload["status"] = "approved"
            payload["resolved_option"] = action_record["candidate_id"]
        elif action_record["action_type"] == "proposal_approve":
            payload["status"] = "approved"
            payload["resolved_option"] = "approve"
        elif action_record["action_type"] == "proposal_reject":
            payload["status"] = "rejected"
            payload["resolved_option"] = "reject"
        elif action_record["action_type"] == "proposal_need_revision":
            payload["status"] = "need_revision"
            payload["resolved_option"] = "need_revision"
        elif action_record["action_type"] == "engineering_execute_approve":
            payload["status"] = "approved"
            payload["resolved_option"] = "approve"
        elif action_record["action_type"] == "engineering_execute_reject":
            payload["status"] = "rejected"
            payload["resolved_option"] = "reject"
        elif action_record["action_type"] == "engineering_execute_need_revision":
            payload["status"] = "need_revision"
            payload["resolved_option"] = "need_revision"
        elif action_record["action_type"] == "custom_override":
            payload["status"] = "custom_override"
            payload["resolved_option"] = ""
        else:
            payload["status"] = "pending"
            payload["resolved_option"] = ""
        updated_task = dict(payload)
        break

    if updated_task is None:
        raise KeyError(task_id)

    mutation_result: Dict[str, Any] | None = None
    if proposal_payload is not None:
        proposal_payload["updated_at"] = action_record["submitted_at"]
        proposal_payload["reviewer_note"] = action_record["reviewer_note"]
        proposal_payload["reviewed_by"] = action_record["operator_open_id"] or action_record["operator_user_id"]
        proposal_payload["last_action_type"] = action_record["action_type"]
        proposal_payload["feishu_status"] = "acted"
        if action_record["action_type"] == "proposal_approve":
            proposal_payload["status"] = "approved_for_engineering"
            proposal_payload["proposal_status"] = "approved_for_engineering"
            proposal_payload["resolved_option"] = "approve"
            engineer = MemoryEngineerAgent()
            change_request_payload = engineer.build_change_request(
                proposal=proposal_payload,
                reviewer_note=action_record["reviewer_note"],
            )
            change_request_payload["support_case_ids"] = list(updated_task.get("support_case_ids") or [])
            change_request_payload["evidence_refs"] = list(updated_task.get("evidence_refs") or [])
            engineering_task = engineer.build_execute_task(change_request=change_request_payload)
            engineering_task_payload = engineering_task.to_dict()
            change_requests = [
                payload
                for payload in change_requests
                if str(payload.get("change_request_id") or "").strip() != str(change_request_payload.get("change_request_id") or "").strip()
            ]
            change_requests.append(change_request_payload)
            tasks = [
                payload
                for payload in tasks
                if str(payload.get("task_id") or "").strip() != str(engineering_task_payload.get("task_id") or "").strip()
            ]
            tasks.append(engineering_task_payload)
            _append_reflection_feedback(
                paths=paths,
                action_record=action_record,
                task_payload=updated_task,
                proposal_payload=proposal_payload,
            )
        elif action_record["action_type"] == "proposal_reject":
            proposal_payload["status"] = "rejected"
            proposal_payload["proposal_status"] = "rejected"
            proposal_payload["resolved_option"] = "reject"
            _append_reflection_feedback(
                paths=paths,
                action_record=action_record,
                task_payload=updated_task,
                proposal_payload=proposal_payload,
            )
        elif action_record["action_type"] == "proposal_need_revision":
            proposal_payload["status"] = "need_revision"
            proposal_payload["proposal_status"] = "need_revision"
            proposal_payload["resolved_option"] = "need_revision"
            _append_reflection_feedback(
                paths=paths,
                action_record=action_record,
                task_payload=updated_task,
                proposal_payload=proposal_payload,
            )
        _write_jsonl_records(paths.proposals_path, proposals)

    if change_request_payload is not None:
        change_request_payload["updated_at"] = action_record["submitted_at"]
        change_request_payload["reviewer_note"] = action_record["reviewer_note"]
        change_request_payload["reviewed_by"] = action_record["operator_open_id"] or action_record["operator_user_id"]
        change_request_payload["last_action_type"] = action_record["action_type"]
        change_request_payload["feishu_status"] = "acted"
        if action_record["action_type"] == "engineering_execute_approve":
            change_request_payload["status"] = "approved"
            mutation_result = MutationExecutor(project_root=project_root).execute_change_request(change_request=change_request_payload)
            if str(mutation_result.get("status") or "") == "applied":
                change_request_payload["status"] = "applied"
                if updated_task is not None:
                    updated_task["status"] = "applied"
                for payload in tasks:
                    if str(payload.get("task_id") or "").strip() == task_id:
                        payload["status"] = "applied"
                        payload["resolved_option"] = "approve"
                        break
            _append_reflection_feedback(
                paths=paths,
                action_record=action_record,
                task_payload=updated_task,
                change_request_payload=change_request_payload,
            )
        elif action_record["action_type"] == "engineering_execute_reject":
            change_request_payload["status"] = "rejected"
            _append_reflection_feedback(
                paths=paths,
                action_record=action_record,
                task_payload=updated_task,
                change_request_payload=change_request_payload,
            )
        elif action_record["action_type"] == "engineering_execute_need_revision":
            change_request_payload["status"] = "need_revision"
            _append_reflection_feedback(
                paths=paths,
                action_record=action_record,
                task_payload=updated_task,
                change_request_payload=change_request_payload,
            )
        _write_jsonl_records(paths.engineering_change_requests_path, change_requests)

    _write_jsonl_records(paths.tasks_path, tasks)
    actions = _read_jsonl_records(paths.task_actions_path)
    actions.append(action_record)
    _write_jsonl_records(paths.task_actions_path, actions)
    if proposal_payload is not None:
        proposal_actions = _read_jsonl_records(paths.proposal_actions_path)
        proposal_actions.append(action_record)
        _write_jsonl_records(paths.proposal_actions_path, proposal_actions)
    if engineering_task_payload is not None:
        receive_id = str(FEISHU_APPROVAL_RECEIVE_ID or "").strip()
        receive_id_type = str(FEISHU_APPROVAL_RECEIVE_ID_TYPE or "chat_id").strip() or "chat_id"
        if not receive_id:
            receive_id = str(action_record.get("operator_open_id") or "").strip()
            receive_id_type = "open_id"
        if not receive_id:
            receive_id = str(action_record.get("operator_user_id") or "").strip()
            receive_id_type = "user_id"
        if receive_id:
            try:
                engineering_send_result = send_reflection_task_card_for_task(
                    project_root=project_root,
                    user_name=user_name,
                    task_id=str(engineering_task_payload.get("task_id") or ""),
                    receive_id=receive_id,
                    receive_id_type=receive_id_type,
                )
            except Exception as exc:
                engineering_send_result = {
                    "status": "failed",
                    "error": str(exc),
                    "task_id": str(engineering_task_payload.get("task_id") or ""),
                }
    return {
        "task": updated_task,
        "action": action_record,
        "proposal": proposal_payload,
        "change_request": change_request_payload,
        "engineering_task": engineering_task_payload,
        "engineering_send_result": engineering_send_result or {},
        "mutation_result": mutation_result or {},
    }


def build_reflection_task_card_preview(*, project_root: str, user_name: str, task_id: str) -> Dict[str, Any]:
    detail_payload = get_reflection_task_detail_payload(project_root=project_root, user_name=user_name, task_id=task_id)
    return {
        "task_id": task_id,
        "card": build_reflection_task_card(detail_payload),
        "detail": detail_payload,
    }


def mark_task_feishu_sent(*, project_root: str, user_name: str, task_id: str, message_id: str) -> Dict[str, Any]:
    paths = build_reflection_asset_paths(project_root=project_root, user_name=user_name)
    ensure_reflection_root(paths)
    tasks = _read_jsonl_records(paths.tasks_path)
    updated_task: Dict[str, Any] | None = None
    for payload in tasks:
        if str(payload.get("task_id") or "") != task_id:
            continue
        payload["feishu_status"] = "sent"
        payload["feishu_message_id"] = message_id
        payload["feishu_last_sent_at"] = _utcnow_iso()
        updated_task = dict(payload)
        break
    if updated_task is None:
        raise KeyError(task_id)
    _write_jsonl_records(paths.tasks_path, tasks)
    return updated_task


def mark_difficult_case_feishu_sent(*, project_root: str, user_name: str, case_id: str, message_id: str) -> Dict[str, Any]:
    paths = build_reflection_asset_paths(project_root=project_root, user_name=user_name)
    ensure_reflection_root(paths)
    difficult_cases = _read_jsonl_records(paths.difficult_cases_path)
    updated_case: Dict[str, Any] | None = None
    for payload in difficult_cases:
        if str(payload.get("case_id") or "") != case_id:
            continue
        payload["feishu_status"] = "sent"
        payload["feishu_message_id"] = message_id
        payload["feishu_last_sent_at"] = _utcnow_iso()
        updated_case = dict(payload)
        break
    if updated_case is None:
        raise KeyError(case_id)
    _write_jsonl_records(paths.difficult_cases_path, difficult_cases)
    return updated_case


def _translate_values_for_card(values: List[str]) -> Dict[str, str]:
    """批量翻译英文值为中文，用于飞书卡片展示。Google Translate 免费，带缓存。"""
    import re as _re
    result: Dict[str, str] = {}
    to_translate: List[str] = []
    for v in values:
        if not v or v == "—" or v == "None":
            result[v] = v
        elif len(_re.findall(r"[\u4e00-\u9fff]", v)) > len(v) * 0.3:
            result[v] = v  # 已是中文
        elif not _re.search(r"[a-zA-Z_]", v):
            result[v] = v  # 非英文
        else:
            to_translate.append(v)
    if to_translate:
        try:
            from deep_translator import GoogleTranslator
            translator = GoogleTranslator(source="en", target="zh-CN")
            for v in to_translate:
                try:
                    translated = translator.translate(v.replace("_", " ").strip())
                    result[v] = translated or v
                except Exception:
                    result[v] = v
        except ImportError:
            for v in to_translate:
                result[v] = v
    return result


def send_evolution_round_card(
    *,
    user_name: str,
    date_str: str,
    result: Dict[str, Any],
    field_cycles: List[Dict[str, Any]] | None = None,
    proposals: List[Dict[str, Any]] | None = None,
    gt_comparisons: Dict[str, Dict[str, Any]] | None = None,
    vlm_summaries: Dict[str, str] | None = None,
    event_summaries: Dict[str, str] | None = None,
    receive_id: str = "",
    receive_id_type: str = "",
    review_base_url: str = REVIEW_BASE_URL,
    app_id: str = FEISHU_APP_ID,
    app_secret: str = FEISHU_APP_SECRET,
    api_base_url: str = FEISHU_API_BASE_URL,
) -> Dict[str, Any]:
    """发送 evolution 单轮汇报卡片到飞书个人。

    gt_comparisons: {field_key: {output_value, gt_value, grade, score}} 用于展示对比详情。
    vlm_summaries: {photo_id: summary} 用于展示照片证据描述。
    event_summaries: {event_id: summary} 用于展示事件证据描述。
    """
    if not receive_id:
        receive_id = str(FEISHU_DEFAULT_RECEIVE_ID or "").strip()
    if not receive_id_type:
        receive_id_type = str(FEISHU_DEFAULT_RECEIVE_ID_TYPE or "open_id").strip()
    if not app_id or not app_secret or not receive_id:
        raise RuntimeError("feishu_credentials_or_receive_id_missing")

    gt_map = gt_comparisons or {}
    total_focus = result.get("total_focus_fields", 0)
    total_insights = result.get("total_insights", 0)
    total_proposals = result.get("total_proposals", 0)

    # 字段名中文映射
    label_map: Dict[str, str] = {}
    try:
        for row in load_bilingual_label_rows():
            if row.get("key") and row.get("zh_label"):
                label_map[row["key"]] = row["zh_label"]
    except Exception:
        pass

    def zh_field(fk: str) -> str:
        return label_map.get(fk, fk.rsplit(".", 1)[-1])

    GRADE_ZH = {
        "mismatch": "不匹配", "missing_prediction": "缺失", "partial_match": "部分匹配",
        "close_match": "接近匹配", "exact_match": "完全匹配", "improved": "优化",
    }
    FAILURE_ZH = {
        "missing_signal": "信号缺失", "wrong_value": "值错误", "overclaim": "过度推断",
        "partial_coverage": "覆盖不全", "null_value": "空值", "unknown": "未分类",
    }
    STATUS_ZH = {
        "new_rule_candidate": "💡 可提规则", "throttle_armed": "⏸️ 即将暂停",
        "throttled": "⏹️ 已暂停", "needs_next_cycle": "🔄 需下轮",
        "initial_snapshot": "📸 初始快照",
    }

    # 确定最大轮次号
    max_cycle = max((fc.get("cycle_index", 1) for fc in (field_cycles or [{}])), default=1)

    # --- 批量翻译 GT 值和输出值 ---
    values_to_translate: List[str] = []
    for fk in [fc.get("field_key", "") for fc in (field_cycles or [])]:
        gi = gt_map.get(fk, {})
        for v in [str(gi.get("output_value", "")), str(gi.get("gt_value", ""))]:
            if v and v not in values_to_translate:
                values_to_translate.append(v[:80])
    _val_zh = _translate_values_for_card(values_to_translate)

    def zh_val(v: str) -> str:
        return _val_zh.get(v, v)

    # --- 构建卡片 elements ---
    elements: List[Dict[str, Any]] = [
        {"tag": "markdown", "content": (
            f"**用户**: {user_name}　|　**日期**: {date_str}　|　**第 {max_cycle} 轮**\n"
            f"**关注字段**: {total_focus}　|　**洞察**: {total_insights}　|　**提案**: {total_proposals}"
        )},
    ]

    # 每个字段的详细迭代卡片
    for fc in (field_cycles or [])[:10]:
        fk = fc.get("field_key", "")
        cycle_idx = fc.get("cycle_index", 1)
        status = fc.get("cycle_status", "")
        grade = fc.get("grade", "")
        fm = fc.get("failure_mode", "")
        score = fc.get("issue_score", 0)
        prev_score = fc.get("prev_issue_score", 0)

        status_text = STATUS_ZH.get(status, f"▪️ {status}")
        grade_text = GRADE_ZH.get(grade, grade)
        fm_text = FAILURE_ZH.get(fm, fm)

        # 分数变化
        score_text = f"分数: {score:.1f}"
        if prev_score and prev_score != score:
            delta = score - prev_score
            arrow = "↓" if delta < 0 else "↑"
            score_text += f" ({arrow}{abs(delta):.1f})"

        # GT 对比
        gt_info = gt_map.get(fk, {})
        output_val = zh_val(str(gt_info.get("output_value", "—"))[:80])
        gt_val = zh_val(str(gt_info.get("gt_value", "—"))[:80])

        # reflect 结论
        judgment = fc.get("judgment_summary_zh", "")
        original_reasoning = fc.get("original_reasoning", "")
        proposed_direction = fc.get("proposed_reasoning_direction", "")
        key_evidence = fc.get("key_evidence_zh") or []

        lines = (
            f"{status_text}　**{zh_field(fk)}**　第 {cycle_idx} 轮　{grade_text} / {fm_text}\n"
            f"📊 {score_text}　|　系统: `{output_val}`　→　GT: `{gt_val}`"
        )
        if judgment:
            lines += f"\n{judgment}"
        else:
            lines += "\n*本轮未生成反思结论*"

        elements.append({"tag": "markdown", "content": lines})

    # 提案详情（reflect 驱动，展示 reasoning 对比）
    ROOT_CAUSE_ZH_MAP = {
        "field_reasoning": "COT 推理", "evidence_packaging": "证据组装",
        "tool_retrieval": "工具检索", "tool_selection_policy": "工具选择策略",
        "coverage_gap_source": "来源缺口", "coverage_gap_tool": "工具缺口",
        "engineering_issue": "工程问题", "watch_only": "持续观察",
    }
    if proposals:
        elements.append({"tag": "markdown", "content": "---\n**修改提案（待审批）**"})
        for p in (proposals or [])[:5]:
            fk = p.get("field_key", "")
            gt_info = gt_map.get(fk, {})
            rc = p.get("root_cause_family", "")
            conf = p.get("confidence", 0)
            orig = p.get("original_reasoning", "")
            proposed = p.get("proposed_reasoning_direction", "")
            why = p.get("why_this_surface_zh", "")
            key_ev = p.get("key_evidence_zh") or []

            imp_type = p.get("improvement_type", "precision")
            imp_label = "📈 召回提升（GT 未标注，系统发现了新信号）" if imp_type == "recall" else "🎯 准确率优化"
            content = f"📋 **{zh_field(fk)}**　{imp_label}\n"
            content += f"当前输出: `{zh_val(str(gt_info.get('output_value', '—'))[:60])}`\n"
            content += f"GT 标准: `{zh_val(str(gt_info.get('gt_value', '—'))[:60])}`\n"
            content += f"**根因**: {ROOT_CAUSE_ZH_MAP.get(rc, rc)}（置信度 {conf:.0%}）\n"
            if orig:
                content += f"\n**之前的判断逻辑**:\n{orig[:200]}\n"
            if proposed:
                content += f"\n**应改为**:\n{proposed[:200]}\n"
            if why:
                content += f"\n**为什么**: {why[:150]}"
            if key_ev:
                content += "\n\n**关键证据**:\n" + "\n".join(f"- {e[:100]}" for e in key_ev[:3])

            # 展示具体的规则修改
            patch = p.get("patch_preview") or {}
            rule_lines: List[str] = []
            fso = patch.get("field_spec_overrides") or {}
            for field, spec in fso.items():
                if isinstance(spec, dict) and spec.get("weak_evidence_caution"):
                    conditions = spec["weak_evidence_caution"]
                    if isinstance(conditions, list):
                        rule_lines.append(f"**{zh_field(field)}** — 以下情况证据较弱，需谨慎判断:")
                        for c in conditions[:3]:
                            rule_lines.append(f"  · 当{c}时")
            tr = patch.get("tool_rules") or {}
            for field, rule in tr.items():
                if isinstance(rule, dict):
                    parts = []
                    for k, v in rule.items():
                        parts.append(f"{k}: {v}")
                    if parts:
                        rule_lines.append(f"**证据规则修改** ({zh_field(field)}): {', '.join(parts)}")
            cp = patch.get("call_policies") or {}
            for field, policy in cp.items():
                if isinstance(policy, dict) and policy.get("append_allowed_sources"):
                    sources = policy["append_allowed_sources"]
                    rule_lines.append(f"**数据来源修改** ({zh_field(field)}): 新增 {', '.join(sources)}")

            if rule_lines:
                content += "\n\n**审批通过后生效的规则**:\n" + "\n".join(rule_lines)

            elements.append({"tag": "markdown", "content": content})
    else:
        elements.append({"tag": "markdown", "content": "*本轮未发现可提交的修改提案*"})

    # 链接
    links: List[str] = []
    if total_proposals > 0:
        links.append(f"[👉 去审批提案]({review_base_url}/proposals?user_name={user_name})")
    links.append(f"[📊 查看详情]({review_base_url}/user/{user_name})")
    elements.append({"tag": "markdown", "content": "　".join(links)})

    # 颜色：有提案用蓝色，全部收敛用绿色
    template = "green" if all(
        fc.get("cycle_status") in ("throttled", "throttle_armed") for fc in (field_cycles or [])
    ) else "blue"

    card = {
        "schema": "2.0",
        "config": {"enable_forward": True, "width_mode": "fill"},
        "header": {
            "title": {"tag": "plain_text", "content": f"Evolution 第 {max_cycle} 轮 — {user_name}"},
            "template": template,
            "subtitle": {"tag": "plain_text", "content": f"{date_str} | {total_focus} 字段 | {total_proposals} 提案"},
        },
        "body": {"elements": elements},
    }

    tenant_access_token = _get_tenant_access_token(
        app_id=app_id, app_secret=app_secret, api_base_url=api_base_url,
    )
    response = requests.post(
        f"{api_base_url.rstrip('/')}/open-apis/im/v1/messages?receive_id_type={receive_id_type}",
        json={
            "receive_id": receive_id,
            "msg_type": "interactive",
            "content": json.dumps(card, ensure_ascii=False),
        },
        headers={
            "Authorization": f"Bearer {tenant_access_token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    return {
        "message_id": str((payload.get("data") or {}).get("message_id") or ""),
        "raw_response": payload,
    }


def _get_tenant_access_token(*, app_id: str, app_secret: str, api_base_url: str) -> str:
    response = requests.post(
        f"{api_base_url.rstrip('/')}/open-apis/auth/v3/tenant_access_token/internal",
        json={
            "app_id": app_id,
            "app_secret": app_secret,
        },
        headers={"Content-Type": "application/json; charset=utf-8"},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    token = str(payload.get("tenant_access_token") or "").strip()
    if not token:
        raise RuntimeError("feishu_tenant_access_token_missing")
    return token


def _build_candidate_button(task: Dict[str, Any], option: str) -> Dict[str, Any]:
    return _build_button_component(
        text=f"采用 {_humanize_option(option)}",
        button_type="primary_filled" if option == str(task.get("recommended_option") or "") else "default",
        callback_value=_build_callback_value(task, action_type="adopt_candidate", candidate_id=option),
    )


def _build_callback_value(task: Dict[str, Any], *, action_type: str, candidate_id: str = "") -> Dict[str, Any]:
    payload = {
        "task_id": str(task.get("task_id") or ""),
        "user_name": str(task.get("user_name") or ""),
        "pattern_id": str(task.get("pattern_id") or ""),
        "action_type": action_type,
    }
    if task.get("proposal_id"):
        payload["proposal_id"] = str(task.get("proposal_id") or "")
    if task.get("experiment_id"):
        payload["experiment_id"] = str(task.get("experiment_id") or "")
    if task.get("change_request_id"):
        payload["change_request_id"] = str(task.get("change_request_id") or "")
    if candidate_id:
        payload["candidate_id"] = candidate_id
    return payload


def _normalize_callback_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    action = dict(payload.get("action") or {})
    value = dict(action.get("value") or payload.get("value") or {})
    form_value = dict(action.get("form_value") or payload.get("form_value") or {})
    operator = dict(payload.get("operator") or {})
    submitted_at = str(payload.get("submitted_at") or "").strip() or _utcnow_iso()
    reviewer_note = (
        str(form_value.get("reviewer_note") or "").strip()
        or str(payload.get("reviewer_note") or "").strip()
    )
    return {
        "task_id": str(value.get("task_id") or payload.get("task_id") or "").strip(),
        "user_name": str(value.get("user_name") or payload.get("user_name") or "").strip(),
        "pattern_id": str(value.get("pattern_id") or payload.get("pattern_id") or "").strip(),
        "proposal_id": str(value.get("proposal_id") or payload.get("proposal_id") or "").strip(),
        "experiment_id": str(value.get("experiment_id") or payload.get("experiment_id") or "").strip(),
        "change_request_id": str(value.get("change_request_id") or payload.get("change_request_id") or "").strip(),
        "action_type": str(value.get("action_type") or payload.get("action_type") or "defer").strip(),
        "candidate_id": str(value.get("candidate_id") or payload.get("candidate_id") or "").strip(),
        "reviewer_note": reviewer_note,
        "operator_open_id": str(operator.get("open_id") or payload.get("operator_open_id") or "").strip(),
        "operator_user_id": str(operator.get("user_id") or payload.get("operator_user_id") or "").strip(),
        "source": str(payload.get("source") or "feishu_card").strip(),
        "submitted_at": submitted_at,
        "raw_payload": payload,
    }


def _validate_callback_token(payload: Dict[str, Any]) -> None:
    expected = FEISHU_CALLBACK_VERIFICATION_TOKEN
    if not expected:
        return
    candidates = [
        str(payload.get("token") or "").strip(),
        str((payload.get("header") or {}).get("token") or "").strip(),
    ]
    if expected not in candidates:
        raise PermissionError("invalid_feishu_callback_token")


def _absolute_review_url(review_base_url: str, detail_url: str) -> str:
    base = review_base_url.rstrip("/")
    path = detail_url if detail_url.startswith("/") else f"/{detail_url}"
    return f"{base}{path}"


def _humanize_option(option: str) -> str:
    return _humanize_label("task_option", option, default=option or "unknown")


def _task_type_label(task_type: str) -> str:
    return _humanize_label("task_type", task_type, default="反思任务")


def _build_proposal_review_card(
    *,
    task: Dict[str, Any],
    proposal: Dict[str, Any],
    review_url: str,
    evidence_refs: List[Dict[str, Any]],
    support_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    diff_summary = list(proposal.get("diff_summary") or [])
    if not diff_summary:
        diff_summary = ["当前没有生成可展示的修改摘要"]
    field_label = _resolve_task_field_label(task=task, pattern={}, proposal=proposal, support_cases=[])
    localized_summary = _plain_language_card_text(str(task.get("summary") or proposal.get("summary") or "暂无摘要"))
    localized_reasoning = _plain_language_card_text(str(proposal.get("agent_reasoning_summary") or "暂无"))
    localized_agent_key_evidence = [
        _plain_language_card_text(str(item))
        for item in list(proposal.get("key_evidence_zh") or [])
        if str(item).strip()
    ]
    localized_why_not = _plain_language_card_text(str(proposal.get("why_not_other_surfaces") or "暂无"))
    localized_diff_summary = [_plain_language_card_text(str(line)) for line in diff_summary]
    evidence_lines = _build_evidence_snippet_lines(evidence_refs=evidence_refs, support_cases=support_cases)
    actions = [
        _build_button_component(
            text="批准并执行",
            button_type="primary_filled",
            callback_value=_build_callback_value(task, action_type="proposal_approve"),
            confirm_title="确认推进这次改码提案？",
            confirm_text="批准后会进入工程执行审批。",
        ),
        _build_button_component(
            text="驳回",
            button_type="danger",
            callback_value=_build_callback_value(task, action_type="proposal_reject"),
        ),
        _build_button_component(
            text="需要修订",
            button_type="default",
            callback_value=_build_callback_value(task, action_type="proposal_need_revision"),
        ),
    ]
    summary_lines = [
        f"- 画像标签：{field_label or '未识别字段标签'}",
        f"- 这次准备解决：{localized_summary}",
        f"- 我为什么建议这样改：{localized_reasoning}",
    ]
    if localized_agent_key_evidence:
        summary_lines.append("- 关键线索：")
        summary_lines.extend(f"  - {line}" for line in localized_agent_key_evidence)
    if evidence_lines:
        summary_lines.append("- 证据摘录：")
        summary_lines.extend(f"  - {line}" for line in evidence_lines)
    summary_lines.append(f"- 先不改别的，因为：{localized_why_not}")
    if localized_diff_summary:
        summary_lines.append("- 准备怎么改：")
        summary_lines.extend(f"  - {line}" for line in localized_diff_summary)
    elements = [
        _build_markdown_block("\n".join(summary_lines)),
        *actions,
    ]
    return _build_card_shell(
        title="请确认这次改动",
        subtitle=field_label or "",
        template="blue",
        summary=str(task.get("summary") or proposal.get("summary") or "改码提案待确认"),
        text_tags=[],
        elements=elements,
    )


def _build_engineering_execute_review_card(
    *,
    task: Dict[str, Any],
    change_request: Dict[str, Any],
    proposal: Dict[str, Any],
    review_url: str,
    evidence_refs: List[Dict[str, Any]],
    support_cases: List[Dict[str, Any]],
) -> Dict[str, Any]:
    summary = _plain_language_card_text(str(change_request.get("change_summary_zh") or task.get("summary") or "暂无改动说明"))
    short_reason = _plain_language_card_text(str(change_request.get("short_reason_zh") or "暂无理由"))
    evidence_lines = _build_evidence_snippet_lines(
        preferred_lines=list(change_request.get("key_evidence_zh") or proposal.get("key_evidence_zh") or []),
        evidence_refs=list(change_request.get("evidence_refs") or evidence_refs),
        support_cases=support_cases,
    )
    actions = [
        _build_button_component(
            text="批准并执行",
            button_type="primary_filled",
            callback_value=_build_callback_value(task, action_type="engineering_execute_approve"),
            confirm_title="确认执行这次改动？",
            confirm_text="批准后会开始正式改码并跑验证。",
        ),
        _build_button_component(
            text="驳回",
            button_type="danger",
            callback_value=_build_callback_value(task, action_type="engineering_execute_reject"),
        ),
        _build_button_component(
            text="需要修订",
            button_type="default",
            callback_value=_build_callback_value(task, action_type="engineering_execute_need_revision"),
        ),
    ]
    summary_lines = [
        f"- 准备怎么改：{summary}",
        f"- 为什么改：{short_reason}",
    ]
    if evidence_lines:
        summary_lines.append("- 关键线索：")
        summary_lines.extend(f"  - {line}" for line in evidence_lines)
    elements = [
        _build_markdown_block("\n".join(summary_lines)),
        *actions,
    ]
    field_label = _profile_field_card_label(str(change_request.get("field_key") or ""))
    return _build_card_shell(
        title="请确认是否执行",
        subtitle=field_label or "",
        template="blue",
        summary=str(change_request.get("change_summary_zh") or task.get("summary") or "工程执行审批待确认"),
        text_tags=[],
        elements=elements,
    )


def _build_action_record(*, task_id: str, user_name: str, action_payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "action_id": uuid.uuid4().hex,
        "task_id": task_id,
        "user_name": user_name,
        "pattern_id": str(action_payload.get("pattern_id") or "").strip(),
        "proposal_id": str(action_payload.get("proposal_id") or "").strip(),
        "experiment_id": str(action_payload.get("experiment_id") or "").strip(),
        "change_request_id": str(action_payload.get("change_request_id") or "").strip(),
        "action_type": str(action_payload.get("action_type") or "defer").strip(),
        "candidate_id": str(action_payload.get("candidate_id") or "").strip(),
        "reviewer_note": str(action_payload.get("reviewer_note") or "").strip(),
        "operator_open_id": str(action_payload.get("operator_open_id") or "").strip(),
        "operator_user_id": str(action_payload.get("operator_user_id") or "").strip(),
        "source": str(action_payload.get("source") or "feishu_card").strip(),
        "submitted_at": str(action_payload.get("submitted_at") or _utcnow_iso()).strip(),
    }


def _build_evidence_snippet_lines(
    *,
    preferred_lines: List[str] | None = None,
    evidence_refs: List[Dict[str, Any]],
    support_cases: List[Dict[str, Any]],
) -> List[str]:
    lines: List[str] = []
    for line in list(preferred_lines or [])[:3]:
        normalized = _truncate_card_text(_plain_language_card_text(str(line).strip()))
        if normalized:
            lines.append(normalized)
    if lines:
        return lines

    for case_payload in support_cases[:3]:
        tool_usage_summary = dict(case_payload.get("tool_usage_summary") or {})
        for line in list(tool_usage_summary.get("agent_key_evidence_zh") or [])[:3]:
            normalized = _truncate_card_text(_plain_language_card_text(str(line).strip()))
            if normalized:
                lines.append(normalized)
        if lines:
            return lines

    for ref in evidence_refs[:3]:
        raw_summary = str(ref.get("description") or ref.get("summary") or "").strip()
        if _looks_english_heavy(raw_summary):
            continue
        summary = _truncate_card_text(_plain_language_card_text(raw_summary))
        ref_id = str(ref.get("source_id") or ref.get("ref") or "").strip()
        if not summary:
            continue
        lines.append(summary + (f"（{ref_id}）" if ref_id else ""))
    if lines:
        return lines

    for case_payload in support_cases[:3]:
        decision_trace = dict(case_payload.get("decision_trace") or {})
        reasoning = str(
            decision_trace.get("reason")
            or decision_trace.get("reasoning")
            or ((case_payload.get("upstream_output") or {}).get("reasoning") if isinstance(case_payload.get("upstream_output"), dict) else "")
            or ""
        ).strip()
        if not reasoning:
            continue
        if _looks_english_heavy(reasoning):
            continue
        lines.append(_truncate_card_text(_plain_language_card_text(reasoning)))
    if lines:
        return lines
    return []


def _truncate_card_text(text: str, limit: int = 72) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _read_jsonl_records(path: str) -> List[Dict[str, Any]]:
    file_path = Path(path)
    if not file_path.exists():
        return []
    payloads: List[Dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                payloads.append(payload)
    return payloads


def _write_jsonl_records(path: str, payloads: Iterable[Dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _upsert_jsonl_record(*, path: str, payload: Dict[str, Any], id_key: str) -> None:
    records = _read_jsonl_records(path)
    record_id = str(payload.get(id_key) or "").strip()
    if not record_id:
        return
    merged: List[Dict[str, Any]] = []
    replaced = False
    for existing in records:
        if str(existing.get(id_key) or "").strip() == record_id:
            merged.append(dict(payload))
            replaced = True
        else:
            merged.append(existing)
    if not replaced:
        merged.append(dict(payload))
    _write_jsonl_records(path, merged)


def _touch_action_and_feedback_files(paths: Any) -> None:
    for path in (
        paths.task_actions_path,
        paths.proposal_actions_path,
        paths.engineering_change_requests_path,
        paths.reflection_feedback_path,
    ):
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch(exist_ok=True)


def _require_reviewer_note_if_needed(action_record: Dict[str, Any]) -> None:
    if action_record["action_type"] not in {
        "proposal_reject",
        "proposal_need_revision",
        "engineering_execute_reject",
        "engineering_execute_need_revision",
    }:
        return
    if not str(action_record.get("reviewer_note") or "").strip():
        raise ValueError("reviewer_note_required")


def _append_reflection_feedback(
    *,
    paths: Any,
    action_record: Dict[str, Any],
    task_payload: Dict[str, Any] | None,
    proposal_payload: Dict[str, Any] | None = None,
    change_request_payload: Dict[str, Any] | None = None,
) -> None:
    feedback = {
        "feedback_id": uuid.uuid4().hex,
        "task_id": str((task_payload or {}).get("task_id") or action_record.get("task_id") or ""),
        "proposal_id": str((proposal_payload or {}).get("proposal_id") or action_record.get("proposal_id") or ""),
        "change_request_id": str((change_request_payload or {}).get("change_request_id") or action_record.get("change_request_id") or ""),
        "field_key": str((proposal_payload or {}).get("field_key") or (change_request_payload or {}).get("field_key") or ""),
        "recommended_fix_surface": str((proposal_payload or {}).get("fix_surface") or (change_request_payload or {}).get("fix_surface") or ""),
        "human_action": str(action_record.get("action_type") or ""),
        "reviewer_note": str(action_record.get("reviewer_note") or ""),
        "submitted_at": str(action_record.get("submitted_at") or _utcnow_iso()),
        "reflection_summary_snapshot": str((proposal_payload or {}).get("agent_reasoning_summary") or ""),
        "experiment_summary_snapshot": str((proposal_payload or {}).get("result_delta_summary") or (change_request_payload or {}).get("overlay_experiment_summary") or ""),
        "change_summary_snapshot": str((change_request_payload or {}).get("change_summary_zh") or ""),
    }
    feedback_records = _read_jsonl_records(paths.reflection_feedback_path)
    feedback_records.append(feedback)
    _write_jsonl_records(paths.reflection_feedback_path, feedback_records)


def _utcnow_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _humanize_label(category: str, key: str, *, default: str = "") -> str:
    normalized_key = str(key or "").strip()
    if not normalized_key:
        return default
    return lookup_bilingual_label(category, normalized_key, default=default or normalized_key)


def _resolve_task_field_label(
    *,
    task: Dict[str, Any],
    pattern: Dict[str, Any],
    proposal: Dict[str, Any],
    support_cases: List[Dict[str, Any]],
) -> str:
    field_key = _resolve_task_field_key(task=task, pattern=pattern, proposal=proposal, support_cases=support_cases)
    if not field_key:
        return ""
    return _profile_field_card_label(field_key)


def _resolve_task_field_key(
    *,
    task: Dict[str, Any],
    pattern: Dict[str, Any],
    proposal: Dict[str, Any],
    support_cases: List[Dict[str, Any]],
) -> str:
    candidates = [
        str(proposal.get("field_key") or "").strip(),
        str(pattern.get("dimension") or "").strip(),
        str(task.get("dimension") or "").strip(),
    ]
    for case_payload in support_cases:
        candidates.append(str((case_payload or {}).get("dimension") or "").strip())
    candidates.extend(
        _extract_profile_field_hints(
            [
                str(task.get("summary") or "").strip(),
                str(pattern.get("summary") or "").strip(),
                str(proposal.get("summary") or "").strip(),
                *[str(line).strip() for line in list(proposal.get("diff_summary") or [])],
            ]
        )
    )
    for candidate in candidates:
        resolved = resolve_profile_field_hint(candidate)
        if resolved:
            return resolved
        if candidate and "." in candidate:
            return candidate
    return ""


def _extract_profile_field_hints(texts: List[str]) -> List[str]:
    hints: List[str] = []
    for text in texts:
        for token in text.replace("/", " ").replace("：", " ").replace(":", " ").replace("，", " ").replace(",", " ").split():
            normalized = token.strip().strip("()[]{}")
            if not normalized:
                continue
            if normalized.isidentifier() or "." in normalized:
                hints.append(normalized)
    return hints


def _profile_field_card_label(field_key: str) -> str:
    resolved_key = resolve_profile_field_hint(str(field_key or "").strip())
    if not resolved_key:
        return str(field_key or "").strip()
    return lookup_bilingual_label("profile_field", resolved_key, default=resolved_key)


def _localize_card_text(text: str) -> str:
    localized = str(text or "")
    if not localized:
        return localized

    profile_replacements = []
    leaf_replacements = {}
    for row in load_bilingual_label_rows():
        category = str(row.get("category") or "").strip()
        key = str(row.get("key") or "").strip()
        zh_label = str(row.get("zh_label") or "").strip()
        if not key or not zh_label:
            continue
        if category == "profile_field":
            profile_replacements.append((key, zh_label))
            leaf = key.split(".")[-1].strip()
            if leaf and resolve_profile_field_hint(leaf) == key:
                leaf_replacements[leaf] = zh_label
        elif category in {"system_term", "comparison_grade", "comparison_method", "causality_route", "metric_value"}:
            localized = _replace_exact_term(localized, key, zh_label)

    for key, zh_label in sorted(profile_replacements, key=lambda item: len(item[0]), reverse=True):
        localized = localized.replace(key, zh_label)
    for leaf, zh_label in sorted(leaf_replacements.items(), key=lambda item: len(item[0]), reverse=True):
        localized = _replace_exact_term(localized, leaf, zh_label)
    return localized


def _plain_language_card_text(text: str) -> str:
    localized = _localize_card_text(text)
    plain_terms = [
        ("字段级 COT", "判断口径"),
        ("field_cot", "判断口径"),
        ("tool 规则", "找证据规则"),
        ("tool_rule", "找证据规则"),
        ("tool 调用策略", "调用时机"),
        ("call_policy", "调用时机"),
        ("tool 调用轨迹", "取证过程"),
        ("tool_trace", "取证过程"),
        ("trace", "过程记录"),
        ("badcase", "这个问题"),
        ("difficult_case", "疑难问题"),
    ]
    simplified = localized
    for source, target in plain_terms:
        simplified = _replace_exact_term(simplified, source, target)
    simplified = simplified.replace("字段级 COT", "判断口径")
    simplified = simplified.replace("tool 规则", "找证据规则")
    simplified = simplified.replace("tool 调用策略", "调用时机")
    return simplified


def _replace_exact_term(text: str, source: str, target: str) -> str:
    normalized_source = str(source or "").strip()
    if not normalized_source:
        return text
    if re.fullmatch(r"[A-Za-z0-9_./:-]+", normalized_source):
        pattern = rf"(?<![A-Za-z0-9_./:-]){re.escape(normalized_source)}(?![A-Za-z0-9_./:-])"
        return re.sub(pattern, target, text)
    return text.replace(normalized_source, target)


def _looks_english_heavy(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    ascii_letters = sum(1 for ch in normalized if ("a" <= ch.lower() <= "z"))
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", normalized))
    return ascii_letters >= 8 and ascii_letters > cjk_chars * 2


def _format_metric_group(group_label: str, metrics: Dict[str, Any]) -> str:
    if not metrics:
        return f"- {group_label}: 暂无"
    lines = [f"- {group_label}:"]
    for key, value in metrics.items():
        key_label = _humanize_label("metric_key", str(key), default=str(key))
        value_text = _format_metric_value(str(key), value)
        lines.append(f"  - {key_label}: {value_text}")
    return "\n".join(lines)


def _format_metric_value(metric_key: str, value: Any) -> str:
    if metric_key == "comparison_grade":
        return _humanize_label("comparison_grade", str(value), default=str(value))
    if metric_key == "field_key":
        return _profile_field_card_label(str(value))
    if metric_key == "validation_mode":
        return _humanize_label("metric_value", str(value), default=str(value))
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return str(value)
    return _localize_card_text(str(value))


def _render_case_value_for_card(value: Any) -> str:
    if value in (None, "", []):
        return "空值 / 未识别"
    if isinstance(value, list):
        items = [_localize_card_text(str(item).strip()) for item in value if str(item).strip()]
        return "、".join(items) if items else "空值 / 未识别"
    if isinstance(value, dict):
        return _localize_card_text(json.dumps(value, ensure_ascii=False))
    return _localize_card_text(str(value))
