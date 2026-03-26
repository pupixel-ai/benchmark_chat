from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import Response

from services.reflection import build_reflection_asset_paths


def _walk_card_components(components):
    for component in components or []:
        if not isinstance(component, dict):
            continue
        yield component
        for child in component.get("elements", []) or []:
            yield from _walk_card_components([child])
        for column in component.get("columns", []) or []:
            if isinstance(column, dict):
                yield from _walk_card_components(column.get("elements", []) or [])


class ReflectionFeishuLogicTests(unittest.TestCase):
    def test_build_reflection_task_card_contains_candidate_buttons_without_review_link(self) -> None:
        from services.reflection.feishu import build_reflection_task_card

        payload = {
            "task": {
                "task_id": "task_001",
                "task_type": "upstream_decision_task",
                "pattern_id": "pattern_001",
                "user_name": "vigar",
                "summary": "education 字段反复出现稳定校园证据",
                "recommended_option": "field_cot",
                "options": ["field_cot", "tool_rule", "call_policy"],
                "detail_url": "/review/task/task_001",
                "priority": "high",
                "created_at": "2026-03-26T12:00:00Z",
            },
            "pattern": {
                "pattern_id": "pattern_001",
                "summary": "education 字段反复出现稳定校园证据",
                "recommended_option": "field_cot",
            },
            "support_cases": [
                {
                    "case_id": "case_001",
                    "dimension": "long_term_facts.social_identity.education",
                    "decision_trace": {"reason": "校园课堂主线稳定"},
                },
                {
                    "case_id": "case_002",
                    "dimension": "long_term_facts.social_identity.education",
                    "decision_trace": {"reason": "校园场景再次出现"},
                },
            ],
            "evidence_refs": [
                {"source_type": "event", "source_id": "EVT_001", "description": "课堂"},
                {"source_type": "event", "source_id": "EVT_002", "description": "校园"},
            ],
            "history_summary": {
                "recommended_option_success_count": 1,
                "recommended_option_failure_count": 0,
                "open_recommended_experiment_count": 0,
            },
        }

        card = build_reflection_task_card(payload, review_base_url="http://localhost:3000")

        self.assertEqual(card["schema"], "2.0")
        self.assertIn("body", card)
        self.assertEqual(card["header"]["title"]["content"], "请帮我看一下这条问题")
        self.assertEqual(card["header"]["subtitle"]["content"], "社会身份：教育背景")
        self.assertNotIn("text_tag_list", card["header"])
        button_values = [
            component["behaviors"][0]["value"]
            for component in _walk_card_components(card["body"]["elements"])
            if component.get("tag") == "button"
            and component.get("behaviors")
            and component["behaviors"][0].get("type") == "callback"
            and "value" in component["behaviors"][0]
        ]
        self.assertTrue(any(value["action_type"] == "adopt_candidate" and value["candidate_id"] == "field_cot" for value in button_values))
        self.assertFalse(
            any(
                any(behavior.get("type") == "open_url" for behavior in component.get("behaviors", []))
                for component in _walk_card_components(card["body"]["elements"])
                if component.get("tag") == "button"
            )
        )

    def test_handle_feishu_callback_updates_task_and_appends_action(self) -> None:
        from services.reflection.feishu import handle_feishu_callback

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_reflection_asset_paths(project_root=tmpdir, user_name="vigar")
            Path(paths.root_dir).mkdir(parents=True, exist_ok=True)
            with open(paths.tasks_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "task_id": "task_001",
                            "task_type": "upstream_decision_task",
                            "pattern_id": "pattern_001",
                            "user_name": "vigar",
                            "lane": "upstream",
                            "priority": "high",
                            "summary": "education 字段反复出现稳定校园证据",
                            "detail_url": "/review/task/task_001",
                            "support_case_ids": ["case_001", "case_002"],
                            "options": ["field_cot", "tool_rule", "call_policy", "engineering_issue", "watch_only"],
                            "recommended_option": "field_cot",
                            "status": "new",
                            "feishu_status": "sent",
                            "created_at": "2026-03-26T12:00:00Z",
                            "updated_at": "2026-03-26T12:00:00Z",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            result = handle_feishu_callback(
                project_root=tmpdir,
                payload={
                    "operator": {"open_id": "ou_test"},
                    "action": {
                        "value": {
                            "task_id": "task_001",
                            "user_name": "vigar",
                            "action_type": "adopt_candidate",
                            "candidate_id": "field_cot",
                        },
                        "form_value": {"reviewer_note": "先按 field_cot 跑"},
                    },
                },
            )

            self.assertEqual(result["status"], "ok")
            with open(paths.tasks_path, "r", encoding="utf-8") as handle:
                task = json.loads(handle.readline())
            actions = [json.loads(line) for line in Path(paths.task_actions_path).read_text(encoding="utf-8").splitlines() if line.strip()]

            self.assertEqual(task["status"], "approved")
            self.assertEqual(task["resolved_option"], "field_cot")
            self.assertEqual(task["reviewer_note"], "先按 field_cot 跑")
            self.assertEqual(task["feishu_status"], "acted")
            self.assertEqual(len(actions), 1)
            self.assertEqual(actions[0]["action_type"], "adopt_candidate")

    def test_handle_feishu_callback_on_proposal_review_marks_proposal_approved_for_engineering(self) -> None:
        from services.reflection.feishu import handle_feishu_callback

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_reflection_asset_paths(project_root=tmpdir, user_name="vigar")
            Path(paths.root_dir).mkdir(parents=True, exist_ok=True)
            Path(paths.tasks_path).write_text(
                json.dumps(
                    {
                        "task_id": "task_proposal_001",
                        "task_type": "proposal_review",
                        "pattern_id": "pattern_001",
                        "user_name": "vigar",
                        "lane": "upstream",
                        "priority": "high",
                        "summary": "education 提案等待审批",
                        "detail_url": "/review/task/task_proposal_001",
                        "support_case_ids": ["case_001"],
                        "options": ["approve", "reject", "need_revision"],
                        "recommended_option": "approve",
                        "status": "new",
                        "proposal_id": "proposal_001",
                        "experiment_id": "exp_001",
                        "feishu_status": "sent",
                        "created_at": "2026-03-26T12:00:00Z",
                        "updated_at": "2026-03-26T12:00:00Z",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            Path(paths.proposals_path).write_text(
                json.dumps(
                    {
                        "proposal_id": "proposal_001",
                        "task_id": "task_proposal_001",
                        "experiment_id": "exp_001",
                        "pattern_id": "pattern_001",
                        "user_name": "vigar",
                        "lane": "upstream",
                        "field_key": "long_term_facts.social_identity.education",
                        "fix_surface": "field_cot",
                        "summary": "education 提案等待审批",
                        "detail_url": "/review/task/task_proposal_001",
                        "status": "pending_review",
                        "approval_required": True,
                        "patch_preview": {
                            "field_spec_overrides": {
                                "long_term_facts.social_identity.education": {
                                    "cot_steps": ["新增 education 专项 COT 步骤"],
                                }
                            }
                        },
                        "created_at": "2026-03-26T12:00:00Z",
                        "updated_at": "2026-03-26T12:00:00Z",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("services.reflection.feishu.MutationExecutor.execute_change_request") as execute_mutation:
                result = handle_feishu_callback(
                    project_root=tmpdir,
                    payload={
                        "operator": {"open_id": "ou_test"},
                        "action": {
                            "value": {
                                "task_id": "task_proposal_001",
                                "user_name": "vigar",
                                "proposal_id": "proposal_001",
                                "experiment_id": "exp_001",
                                "action_type": "proposal_approve",
                            },
                            "form_value": {"reviewer_note": "可以执行"},
                        },
                    },
                )

            proposals = [json.loads(line) for line in Path(paths.proposals_path).read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(result["status"], "ok")
        self.assertEqual(proposals[0]["status"], "approved_for_engineering")
        execute_mutation.assert_not_called()

    def test_handle_feishu_callback_on_proposal_approve_creates_engineering_execute_review_task(self) -> None:
        from services.reflection.feishu import handle_feishu_callback

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_reflection_asset_paths(project_root=tmpdir, user_name="vigar")
            Path(paths.root_dir).mkdir(parents=True, exist_ok=True)
            Path(paths.tasks_path).write_text(
                json.dumps(
                    {
                        "task_id": "task_proposal_path_a_001",
                        "task_type": "proposal_review",
                        "pattern_id": "pattern_001",
                        "user_name": "vigar",
                        "lane": "upstream",
                        "priority": "high",
                        "summary": "临时人设提案等待审批",
                        "detail_url": "/review/task/task_proposal_path_a_001",
                        "support_case_ids": ["case_001"],
                        "options": ["approve", "reject", "need_revision"],
                        "recommended_option": "approve",
                        "status": "new",
                        "proposal_id": "proposal_path_a_001",
                        "experiment_id": "exp_001",
                        "feishu_status": "sent",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            Path(paths.proposals_path).write_text(
                json.dumps(
                    {
                        "proposal_id": "proposal_path_a_001",
                        "task_id": "task_proposal_path_a_001",
                        "experiment_id": "exp_001",
                        "pattern_id": "pattern_001",
                        "user_name": "vigar",
                        "lane": "upstream",
                        "field_key": "short_term_expression.motivation_shift",
                        "fix_surface": "field_cot",
                        "summary": "临时人设提案等待审批",
                        "detail_url": "/review/task/task_proposal_path_a_001",
                        "status": "pending_review",
                        "execution_path_recommendation": "overlay_direct_apply",
                        "patch_preview": {
                            "field_spec_overrides": {
                                "short_term_expression.motivation_shift": {
                                    "cot_steps": ["优先判断同步/一致状态。"]
                                }
                            }
                        },
                        "gt_value": "同步性",
                        "current_output": "由兴趣导向向高端体验消费偏移",
                        "candidate_output": "同步性",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("services.reflection.feishu.MutationExecutor.execute_change_request") as execute_mutation:
                result = handle_feishu_callback(
                    project_root=tmpdir,
                    payload={
                        "operator": {"open_id": "ou_test"},
                        "action": {
                            "value": {
                                "task_id": "task_proposal_path_a_001",
                                "user_name": "vigar",
                                "proposal_id": "proposal_path_a_001",
                                "experiment_id": "exp_001",
                                "action_type": "proposal_approve",
                            },
                            "form_value": {"reviewer_note": "分析结论成立，进入工程执行审批"},
                        },
                    },
                )

            tasks = [json.loads(line) for line in Path(paths.tasks_path).read_text(encoding="utf-8").splitlines() if line.strip()]
            change_requests = [json.loads(line) for line in Path(paths.engineering_change_requests_path).read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(result["status"], "ok")
        self.assertFalse(execute_mutation.called)
        self.assertTrue(any(task["task_type"] == "engineering_execute_review" for task in tasks))
        self.assertEqual(change_requests[0]["execution_path"], "overlay_direct_apply")

    def test_handle_feishu_callback_on_proposal_approve_immediately_sends_engineering_execute_review_task(self) -> None:
        from services.reflection.feishu import handle_feishu_callback

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_reflection_asset_paths(project_root=tmpdir, user_name="vigar")
            Path(paths.root_dir).mkdir(parents=True, exist_ok=True)
            Path(paths.tasks_path).write_text(
                json.dumps(
                    {
                        "task_id": "task_proposal_send_001",
                        "task_type": "proposal_review",
                        "pattern_id": "pattern_001",
                        "user_name": "vigar",
                        "lane": "upstream",
                        "priority": "high",
                        "summary": "临时人设提案等待审批",
                        "detail_url": "/review/task/task_proposal_send_001",
                        "support_case_ids": ["case_001"],
                        "options": ["approve", "reject", "need_revision"],
                        "recommended_option": "approve",
                        "status": "new",
                        "proposal_id": "proposal_send_001",
                        "experiment_id": "exp_001",
                        "feishu_status": "sent",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            Path(paths.proposals_path).write_text(
                json.dumps(
                    {
                        "proposal_id": "proposal_send_001",
                        "task_id": "task_proposal_send_001",
                        "experiment_id": "exp_001",
                        "pattern_id": "pattern_001",
                        "user_name": "vigar",
                        "lane": "upstream",
                        "field_key": "short_term_expression.motivation_shift",
                        "fix_surface": "field_cot",
                        "summary": "临时人设提案等待审批",
                        "detail_url": "/review/task/task_proposal_send_001",
                        "status": "pending_review",
                        "execution_path_recommendation": "overlay_direct_apply",
                        "patch_preview": {
                            "field_spec_overrides": {
                                "short_term_expression.motivation_shift": {
                                    "cot_steps": ["优先判断同步/一致状态。"]
                                }
                            }
                        },
                        "gt_value": "同步性",
                        "current_output": "由兴趣导向向高端体验消费偏移",
                        "candidate_output": "同步性",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("services.reflection.feishu.FEISHU_APPROVAL_RECEIVE_ID", "oc_approval_group_001"), patch(
                "services.reflection.feishu.FEISHU_APPROVAL_RECEIVE_ID_TYPE", "chat_id"
            ), patch("services.reflection.feishu.send_reflection_task_card_for_task") as send_task_card:
                send_task_card.return_value = {
                    "task": {
                        "task_id": "task_engineering_send_001",
                        "task_type": "engineering_execute_review",
                        "feishu_status": "sent",
                    },
                    "message_id": "om_engineering_001",
                }
                result = handle_feishu_callback(
                    project_root=tmpdir,
                    payload={
                        "operator": {"open_id": "ou_test_push"},
                        "action": {
                            "value": {
                                "task_id": "task_proposal_send_001",
                                "user_name": "vigar",
                                "proposal_id": "proposal_send_001",
                                "experiment_id": "exp_001",
                                "action_type": "proposal_approve",
                            },
                            "form_value": {"reviewer_note": "直接进第二张工程执行卡"},
                        },
                    },
                )

        self.assertEqual(result["status"], "ok")
        send_task_card.assert_called_once()
        self.assertEqual(send_task_card.call_args.kwargs["receive_id"], "oc_approval_group_001")
        self.assertEqual(send_task_card.call_args.kwargs["receive_id_type"], "chat_id")

    def test_handle_feishu_callback_requires_reviewer_note_for_reject_and_need_revision(self) -> None:
        from services.reflection.feishu import handle_feishu_callback

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_reflection_asset_paths(project_root=tmpdir, user_name="vigar")
            Path(paths.root_dir).mkdir(parents=True, exist_ok=True)
            Path(paths.tasks_path).write_text(
                json.dumps(
                    {
                        "task_id": "task_proposal_note_001",
                        "task_type": "proposal_review",
                        "pattern_id": "pattern_001",
                        "user_name": "vigar",
                        "summary": "需要审批",
                        "detail_url": "/review/task/task_proposal_note_001",
                        "status": "new",
                        "proposal_id": "proposal_note_001",
                        "feishu_status": "sent",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            Path(paths.proposals_path).write_text(
                json.dumps(
                    {
                        "proposal_id": "proposal_note_001",
                        "task_id": "task_proposal_note_001",
                        "user_name": "vigar",
                        "field_key": "short_term_expression.motivation_shift",
                        "fix_surface": "field_cot",
                        "summary": "需要审批",
                        "detail_url": "/review/task/task_proposal_note_001",
                        "status": "pending_review",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                handle_feishu_callback(
                    project_root=tmpdir,
                    payload={
                        "operator": {"open_id": "ou_test"},
                        "action": {
                            "value": {
                                "task_id": "task_proposal_note_001",
                                "user_name": "vigar",
                                "proposal_id": "proposal_note_001",
                                "action_type": "proposal_reject",
                            },
                            "form_value": {"reviewer_note": ""},
                        },
                    },
                )

    def test_handle_feishu_callback_on_engineering_execute_review_invokes_mutation_executor(self) -> None:
        from services.reflection.feishu import handle_feishu_callback

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_reflection_asset_paths(project_root=tmpdir, user_name="vigar")
            Path(paths.root_dir).mkdir(parents=True, exist_ok=True)
            Path(paths.tasks_path).write_text(
                json.dumps(
                    {
                        "task_id": "task_engineer_001",
                        "task_type": "engineering_execute_review",
                        "pattern_id": "pattern_001",
                        "user_name": "vigar",
                        "summary": "执行审批",
                        "detail_url": "/review/task/task_engineer_001",
                        "status": "new",
                        "change_request_id": "change_request_001",
                        "feishu_status": "sent",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            Path(paths.engineering_change_requests_path).write_text(
                json.dumps(
                    {
                        "change_request_id": "change_request_001",
                        "task_id": "task_engineer_001",
                        "proposal_id": "proposal_001",
                        "user_name": "vigar",
                        "field_key": "short_term_expression.motivation_shift",
                        "fix_surface": "field_cot",
                        "execution_path": "overlay_direct_apply",
                        "change_summary_zh": "把临时人设字段的 COT 收窄到同步/一致状态。",
                        "short_reason_zh": "因为 GT 要的是同步性。",
                        "patch_preview": {
                            "field_spec_overrides": {
                                "short_term_expression.motivation_shift": {
                                    "cot_steps": ["优先判断同步/一致状态。"]
                                }
                            }
                        },
                        "status": "pending_review",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("services.reflection.feishu.MutationExecutor.execute_change_request") as execute_mutation:
                execute_mutation.return_value = {"status": "applied"}
                result = handle_feishu_callback(
                    project_root=tmpdir,
                    payload={
                        "operator": {"open_id": "ou_test"},
                        "action": {
                            "value": {
                                "task_id": "task_engineer_001",
                                "user_name": "vigar",
                                "change_request_id": "change_request_001",
                                "action_type": "engineering_execute_approve",
                            },
                            "form_value": {"reviewer_note": "可以执行"},
                        },
                    },
                )

        self.assertEqual(result["status"], "ok")
        execute_mutation.assert_called_once()

    def test_send_reflection_task_card_uses_internal_app_token_and_interactive_message(self) -> None:
        from services.reflection.feishu import send_reflection_task_card

        detail_payload = {
            "task": {
                "task_id": "task_001",
                "task_type": "upstream_decision_task",
                "pattern_id": "pattern_001",
                "user_name": "vigar",
                "summary": "education 字段反复出现稳定校园证据",
                "recommended_option": "field_cot",
                "options": ["field_cot", "tool_rule", "call_policy"],
                "detail_url": "/review/task/task_001",
                "priority": "high",
                "created_at": "2026-03-26T12:00:00Z",
            },
            "pattern": {},
            "support_cases": [],
            "evidence_refs": [],
            "history_summary": {},
        }

        calls: list[dict] = []

        class _FakeResponse:
            def __init__(self, payload: dict) -> None:
                self.payload = payload

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return self.payload

        def fake_post(url: str, json=None, headers=None, timeout=None):
            calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
            if "tenant_access_token" in url:
                return _FakeResponse({"tenant_access_token": "tenant_token_001"})
            return _FakeResponse({"data": {"message_id": "om_001"}})

        with patch("services.reflection.feishu.requests.post", side_effect=fake_post):
            result = send_reflection_task_card(
                task_detail_payload=detail_payload,
                receive_id="ou_receiver",
                receive_id_type="open_id",
                review_base_url="http://localhost:3000",
                app_id="cli_test",
                app_secret="secret_test",
            )

        self.assertEqual(result["message_id"], "om_001")
        self.assertEqual(len(calls), 2)
        self.assertIn("tenant_access_token/internal", calls[0]["url"])
        self.assertIn("messages?receive_id_type=open_id", calls[1]["url"])
        self.assertEqual(calls[1]["json"]["msg_type"], "interactive")
        self.assertEqual(calls[1]["json"]["receive_id"], "ou_receiver")

    def test_build_difficult_case_alert_card_shows_gt_and_prediction_and_why_hard(self) -> None:
        from services.reflection.feishu import build_difficult_case_alert_card

        payload = {
            "case": {
                "case_id": "case_difficult_001",
                "summary": "兴趣字段与 GT 部分重叠，当前根因不稳定",
                "detail_url": "/review/difficult-case/case_difficult_001",
                "comparison_grade": "partial_match",
                "resolution_route": "difficult_case",
                "dimension": "long_term_facts.social_identity.education",
            },
            "gt_comparison": {
                "grade": "partial_match",
                "score": 0.66,
                "method": "rule_set_overlap",
                "output_value": ["音乐", "游戏"],
                "gt_value": ["音乐", "电影", "游戏"],
            },
            "difficulty_reason": "当前结果只覆盖了 GT 的一部分，暂时无法稳定判断是召回不全，还是字段归纳口径偏窄。",
            "evidence_refs": [
                {"source_type": "event", "source_id": "EVT_003", "description": "音乐"},
                {"source_type": "event", "source_id": "EVT_004", "description": "游戏"},
            ],
        }

        card = build_difficult_case_alert_card(payload, review_base_url="http://localhost:3000")
        card_text = json.dumps(card, ensure_ascii=False)

        self.assertEqual(card["header"]["title"]["content"], "这条疑难问题请看一下")
        self.assertEqual(card["header"]["subtitle"]["content"], "社会身份：教育背景")
        self.assertIn("社会身份：教育背景", card_text)
        self.assertIn("GT 标注", card_text)
        self.assertIn("当前识别", card_text)
        self.assertIn("电影", card_text)
        self.assertIn("为什么现在还难下结论", card_text)
        self.assertIn("暂时无法稳定判断", card_text)
        self.assertNotIn("匹配等级", card_text)
        self.assertNotIn("匹配分数", card_text)
        self.assertNotIn("对比方法", card_text)
        self.assertNotIn("字段 Trace 概览", card_text)
        self.assertEqual(card["schema"], "2.0")
        self.assertFalse(
            any(
                any(behavior.get("type") == "open_url" for behavior in component.get("behaviors", []))
                for component in _walk_card_components(card["body"]["elements"])
                if component.get("tag") == "button"
            )
        )

    def test_build_proposal_review_card_hides_experiment_metrics_and_keeps_approval_buttons(self) -> None:
        from services.reflection.feishu import build_reflection_task_card

        payload = {
            "task": {
                "task_id": "task_proposal_001",
                "task_type": "proposal_review",
                "pattern_id": "pattern_001",
                "user_name": "vigar",
                "summary": "education 字段实验显著改善，等待你确认是否落正式规则",
                "recommended_option": "approve",
                "options": ["approve", "reject", "need_revision"],
                "detail_url": "/review/task/task_proposal_001",
                "priority": "high",
                "created_at": "2026-03-26T12:00:00Z",
            },
            "pattern": {
                "pattern_id": "pattern_001",
                "summary": "education 字段反复误判",
            },
            "proposal": {
                "proposal_id": "proposal_001",
                "status": "pending_review",
                "agent_reasoning_summary": "tool 已命中且证据充分，问题在字段归纳",
                "why_not_other_surfaces": "没有明显召回缺失，也没有调用策略缺口",
                "diff_summary": ["education 字段新增 campus-to-degree 归纳步骤"],
                "experiment_report": {
                    "baseline_metrics": {"exact_or_close_count": 0, "mismatch_count": 2},
                    "candidate_metrics": {"exact_or_close_count": 2, "mismatch_count": 0},
                },
            },
            "support_cases": [],
            "evidence_refs": [
                {"source_id": "EVT_001", "description": "校园课堂与同学持续互动，讨论课题和作业。"},
                {"source_id": "EVT_002", "description": "在图书馆共同自习，节奏高度同步。"},
            ],
            "history_summary": {},
        }

        card = build_reflection_task_card(payload, review_base_url="http://localhost:3000")
        card_text = json.dumps(card, ensure_ascii=False)

        actions = [component for component in _walk_card_components(card["body"]["elements"]) if component.get("tag") == "button"]
        texts = {action.get("text", {}).get("content") for action in actions}

        self.assertEqual(card["header"]["title"]["content"], "请确认这次改动")
        self.assertEqual(card["header"]["subtitle"]["content"], "社会身份：教育背景")
        self.assertEqual(card["schema"], "2.0")
        self.assertIn("社会身份：教育背景", card_text)
        self.assertIn("我为什么建议这样改", card_text)
        self.assertIn("先不改别的，因为", card_text)
        self.assertIn("准备怎么改", card_text)
        self.assertNotIn("Agent 判断依据", card_text)
        self.assertNotIn("为什么不是其他改面", card_text)
        self.assertNotIn("具体修改内容", card_text)
        self.assertNotIn("实验结果", card_text)
        self.assertIn("证据摘录", card_text)
        self.assertIn("校园课堂与同学持续互动", card_text)
        self.assertIn("批准并执行", texts)
        self.assertIn("驳回", texts)
        self.assertIn("需要修订", texts)
        self.assertNotIn('"tag": "action"', card_text)

    def test_build_proposal_review_card_localizes_raw_field_keys_and_metric_names(self) -> None:
        from services.reflection.feishu import build_reflection_task_card

        payload = {
            "task": {
                "task_id": "task_proposal_002",
                "task_type": "proposal_review",
                "pattern_id": "pattern_002",
                "user_name": "vigar",
                "summary": "真实 badcase 快测提案：修 short_term_expression.motivation_shift 的字段级 COT",
                "recommended_option": "approve",
                "options": ["approve", "reject", "need_revision"],
                "detail_url": "/review/task/task_proposal_002",
                "priority": "high",
                "created_at": "2026-03-26T12:00:00Z",
            },
            "proposal": {
                "proposal_id": "proposal_002",
                "field_key": "short_term_expression.motivation_shift",
                "status": "pending_review",
                "agent_reasoning_summary": "tool_trace 命中充分，但输出把同步性误判成 motivation_shift。",
                "key_evidence_zh": [
                    "tool_trace 里已经拿到和同伴同步行动的线索。",
                    "最终输出把 motivation_shift 外推出了消费倾向变化。",
                ],
                "why_not_other_surfaces": "不是 tool_rule，也不是 call_policy。",
                "diff_summary": [
                    "仅修改 short_term_expression.motivation_shift 的字段级 COT，不动 tool_rule / call_policy。"
                ],
                "experiment_report": {
                    "baseline_metrics": {
                        "comparison_grade": "mismatch",
                        "mismatch_count": 1,
                        "exact_or_close_count": 0,
                        "field_key": "short_term_expression.motivation_shift",
                    },
                    "candidate_metrics": {
                        "validation_mode": "dry_run_fast_test",
                        "expected_exact_or_close_count": 1,
                        "expected_mismatch_count": 0,
                    },
                },
            },
            "pattern": {},
            "support_cases": [],
            "evidence_refs": [],
            "history_summary": {},
        }

        card = build_reflection_task_card(payload, review_base_url="http://localhost:3000")
        card_text = json.dumps(card, ensure_ascii=False)

        self.assertEqual(card["schema"], "2.0")
        self.assertIn("临时人设", card_text)
        self.assertIn("判断口径", card_text)
        self.assertIn("找证据规则", card_text)
        self.assertIn("调用时机", card_text)
        self.assertIn("关键线索", card_text)
        self.assertIn("取证过程 里已经拿到和同伴同步行动的线索", card_text)
        self.assertNotIn("字段级 COT", card_text)
        self.assertNotIn("tool 规则", card_text)
        self.assertNotIn("tool 调用策略", card_text)
        self.assertNotIn("tool_trace", card_text)
        self.assertNotIn("motivation_shift", card_text)
        self.assertNotIn("匹配等级", card_text)
        self.assertNotIn("不匹配数", card_text)
        self.assertNotIn("精确/接近匹配数", card_text)
        self.assertNotIn("验证模式", card_text)
        self.assertNotIn("short_term_expression.motivation_shift", card_text)

    def test_build_engineering_execute_review_card_shows_evidence_snippets_when_review_page_unavailable(self) -> None:
        from services.reflection.feishu import build_reflection_task_card

        payload = {
            "task": {
                "task_id": "task_engineering_001",
                "task_type": "engineering_execute_review",
                "pattern_id": "pattern_002",
                "user_name": "vigar",
                "summary": "更新临时人设字段 COT",
                "recommended_option": "approve",
                "options": ["approve", "reject", "need_revision"],
                "detail_url": "/review/task/task_engineering_001",
                "priority": "high",
                "created_at": "2026-03-26T12:00:00Z",
                "support_case_ids": ["case_001"],
                "evidence_refs": [
                    {
                        "source_id": "EVT_045",
                        "description": "The protagonist browses skincare products at various cosmetic counters and stores, comparing prices and promotions.",
                    },
                    {
                        "source_id": "EVT_046",
                        "description": "The protagonist completes online purchases for skincare products on Taobao and JD.com.",
                    },
                ],
            },
            "change_request": {
                "change_request_id": "change_request_001",
                "field_key": "short_term_expression.motivation_shift",
                "change_summary_zh": "更新临时人设字段的 field_cot，增加同步性优先判断。",
                "short_reason_zh": "因为 GT 要的是同步性。",
                "key_evidence_zh": [
                    "关键线索集中在护肤消费这一条线上，没有出现明确的前后动机转折。",
                    "现有证据更像持续的同类消费，不足以支撑“动机偏移”这种结论。",
                ],
                "evidence_refs": [
                    {
                        "source_id": "EVT_045",
                        "description": "The protagonist browses skincare products at various cosmetic counters and stores, comparing prices and promotions.",
                    },
                    {
                        "source_id": "EVT_046",
                        "description": "The protagonist completes online purchases for skincare products on Taobao and JD.com.",
                    },
                ],
            },
            "support_cases": [
                {
                    "case_id": "case_001",
                    "decision_trace": {
                        "reason": "多次出现与同伴同步行动的线索。"
                    },
                }
            ],
            "evidence_refs": [
                {
                    "source_id": "EVT_045",
                    "description": "The protagonist browses skincare products at various cosmetic counters and stores, comparing prices and promotions.",
                },
                {
                    "source_id": "EVT_046",
                    "description": "The protagonist completes online purchases for skincare products on Taobao and JD.com.",
                },
            ],
        }

        card = build_reflection_task_card(payload, review_base_url="http://localhost:3000")
        card_text = json.dumps(card, ensure_ascii=False)

        self.assertEqual(card["schema"], "2.0")
        self.assertEqual(card["header"]["title"]["content"], "请确认是否执行")
        self.assertEqual(card["header"]["subtitle"]["content"], "临时人设")
        self.assertIn("关键线索", card_text)
        self.assertIn("关键线索集中在护肤消费这一条线上", card_text)
        self.assertIn("现有证据更像持续的同类消费", card_text)
        self.assertNotIn("The protagonist", card_text)


class ReflectionFeishuApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.username = "reflection_feishu_vigar"
        self._write_reflection_fixture()

    def tearDown(self) -> None:
        paths = build_reflection_asset_paths(project_root=Path(__file__).resolve().parents[1], user_name=self.username)
        for path in [
            paths.case_facts_path,
            paths.upstream_patterns_path,
            paths.downstream_audit_patterns_path,
            paths.upstream_experiments_path,
            paths.upstream_outcomes_path,
            paths.tasks_path,
            paths.task_actions_path,
        ]:
            Path(path).unlink(missing_ok=True)

    def test_card_preview_endpoint_returns_feishu_card(self) -> None:
        from backend.feishu_api import preview_reflection_task_card

        payload = preview_reflection_task_card(
            task_id="task_001",
            response=Response(),
            current_user={"username": self.username, "user_id": "user_001"},
        )
        self.assertIn("card", payload)
        self.assertEqual(payload["task_id"], "task_001")

    def test_mock_action_endpoint_updates_task_status(self) -> None:
        from backend.feishu_api import submit_mock_feishu_action

        payload = submit_mock_feishu_action(
            task_id="task_001",
            payload={
                "action_type": "adopt_candidate",
                "candidate_id": "field_cot",
                "reviewer_note": "先按推荐项试一轮",
            },
            response=Response(),
            current_user={"username": self.username, "user_id": "user_001"},
        )

        self.assertEqual(payload["task"]["status"], "approved")
        self.assertEqual(payload["task"]["resolved_option"], "field_cot")

    def test_callback_endpoint_echoes_challenge_without_auth(self) -> None:
        from backend.feishu_api import handle_feishu_callback_event

        payload = handle_feishu_callback_event(
            payload={"challenge": "challenge_token_001"},
            response=Response(),
        )

        self.assertEqual(payload, {"challenge": "challenge_token_001"})

    def _write_reflection_fixture(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        paths = build_reflection_asset_paths(project_root=project_root, user_name=self.username)
        Path(paths.root_dir).mkdir(parents=True, exist_ok=True)
        Path(paths.case_facts_path).write_text(
            json.dumps(
                {
                    "case_id": "case_001",
                    "user_name": self.username,
                    "album_id": "album_001",
                    "entity_type": "profile_field",
                    "entity_id": "long_term_facts.social_identity.education",
                    "dimension": "long_term_facts.social_identity.education",
                    "signal_source": "mainline_profile",
                    "first_seen_stage": "lp3",
                    "surfaced_stage": "lp3",
                    "routing_result": "strategy_candidate",
                    "business_priority": "high",
                    "auto_confidence": 0.81,
                    "triage_reason": "profile_field_ready_for_patterning",
                    "decision_trace": {"reason": "校园课堂主线稳定"},
                    "evidence_refs": [
                        {"source_type": "event", "source_id": "EVT_001", "description": "课堂", "feature_names": ["campus_scene"]},
                    ],
                    "upstream_output": {"value": "college_student"},
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        Path(paths.upstream_patterns_path).write_text(
            json.dumps(
                [
                    {
                        "pattern_id": "pattern_001",
                        "user_name": self.username,
                        "lane": "upstream",
                        "business_priority": "high",
                        "root_cause_candidates": ["field_reasoning"],
                        "fix_surface_candidates": ["field_cot", "tool_rule", "call_policy"],
                        "support_case_ids": ["case_001"],
                        "is_direction_clear": False,
                        "support_count": 1,
                        "summary": "education 字段需要确认修正面",
                        "evidence_refs": [
                            {"source_type": "event", "source_id": "EVT_001"},
                        ],
                        "recommended_option": "field_cot",
                        "eligible_for_task": True,
                        "dimension": "long_term_facts.social_identity.education",
                        "entity_type": "profile_field",
                        "triage_reason": "profile_field_ready_for_patterning",
                    }
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        Path(paths.downstream_audit_patterns_path).write_text("[]", encoding="utf-8")
        with open(paths.tasks_path, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "task_id": "task_001",
                        "task_type": "upstream_decision_task",
                        "pattern_id": "pattern_001",
                        "user_name": self.username,
                        "album_id": "album_001",
                        "lane": "upstream",
                        "priority": "high",
                        "summary": "education 字段需要确认修正面",
                        "detail_url": "/review/task/task_001",
                        "support_case_ids": ["case_001"],
                        "options": ["field_cot", "tool_rule", "call_policy", "engineering_issue", "watch_only"],
                        "recommended_option": "field_cot",
                        "status": "new",
                        "feishu_status": "not_triggered",
                        "created_at": "2026-03-26T12:00:00Z",
                        "updated_at": "2026-03-26T12:00:00Z",
                        "evidence_refs": [
                            {"source_type": "event", "source_id": "EVT_001"},
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


class ReflectionFeishuLongConnectionTests(unittest.TestCase):
    class _Object:
        def __init__(self, **kwargs) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def _build_card_action_event(self):
        return self._Object(
            event=self._Object(
                operator=self._Object(
                    open_id="ou_long_connection",
                    user_id="user_long_connection",
                ),
                token="verification_token_001",
                action=self._Object(
                    value={
                        "task_id": "task_001",
                        "user_name": "vigar",
                        "pattern_id": "pattern_001",
                        "action_type": "adopt_candidate",
                        "candidate_id": "field_cot",
                    },
                    form_value={"reviewer_note": "长链接回调测试"},
                ),
                delivery_type="stream",
                context=self._Object(
                    open_message_id="om_001",
                    open_chat_id="oc_001",
                    url="https://open.feishu.cn/card/case/task_001",
                    preview_token="preview_001",
                ),
            ),
        )

    def test_normalize_long_connection_card_action_event_maps_to_existing_callback_shape(self) -> None:
        from services.reflection.feishu_long_connection import normalize_long_connection_card_action_event

        normalized = normalize_long_connection_card_action_event(self._build_card_action_event())

        self.assertEqual(normalized["token"], "verification_token_001")
        self.assertEqual(normalized["source"], "feishu_long_connection")
        self.assertEqual(normalized["operator"]["open_id"], "ou_long_connection")
        self.assertEqual(normalized["operator"]["user_id"], "user_long_connection")
        self.assertEqual(normalized["action"]["value"]["task_id"], "task_001")
        self.assertEqual(normalized["action"]["value"]["candidate_id"], "field_cot")
        self.assertEqual(normalized["action"]["form_value"]["reviewer_note"], "长链接回调测试")
        self.assertEqual(normalized["context"]["open_message_id"], "om_001")
        self.assertEqual(normalized["delivery_type"], "stream")

    def test_process_long_connection_card_action_event_reuses_existing_callback_logic(self) -> None:
        from services.reflection.feishu_long_connection import process_long_connection_card_action_event

        event = self._build_card_action_event()

        with patch("services.reflection.feishu_long_connection.handle_feishu_callback") as handle_callback:
            handle_callback.return_value = {
                "status": "ok",
                "task": {"task_id": "task_001", "status": "approved"},
                "action": {"action_type": "adopt_candidate", "candidate_id": "field_cot"},
            }

            result = process_long_connection_card_action_event(project_root="/tmp/reflection", event=event)

        handle_callback.assert_called_once()
        callback_payload = handle_callback.call_args.kwargs["payload"]
        self.assertEqual(callback_payload["source"], "feishu_long_connection")
        self.assertEqual(callback_payload["token"], "verification_token_001")
        self.assertEqual(result["toast"]["type"], "info")
        self.assertIn("field_cot", result["toast"]["content"])

    def test_run_feishu_long_connection_raises_clear_error_when_sdk_missing(self) -> None:
        from services.reflection.feishu_long_connection import run_feishu_long_connection

        with patch("services.reflection.feishu_long_connection._load_lark_oapi_sdk", side_effect=RuntimeError("feishu_long_connection_sdk_missing")):
            with self.assertRaisesRegex(RuntimeError, "feishu_long_connection_sdk_missing"):
                run_feishu_long_connection(
                    project_root="/tmp/reflection",
                    app_id="cli_test",
                    app_secret="secret_test",
                )

    def test_long_connection_script_can_render_help_from_repo_root(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        script_path = project_root / "scripts" / "run_feishu_long_connection.py"

        completed = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Run Feishu long connection callback worker.", completed.stdout)
