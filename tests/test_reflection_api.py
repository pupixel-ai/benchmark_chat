from __future__ import annotations

import json
import unittest
from pathlib import Path

from fastapi import HTTPException, Response

from backend.reflection_api import get_difficult_case_detail, get_reflection_task_detail, list_reflection_tasks
from config import PROJECT_ROOT
from services.reflection import build_reflection_asset_paths


class ReflectionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.username = "reflection_api_vigar"
        self.other_username = "reflection_api_other"
        self.paths = build_reflection_asset_paths(project_root=PROJECT_ROOT, user_name=self.username)
        self.other_paths = build_reflection_asset_paths(project_root=PROJECT_ROOT, user_name=self.other_username)
        Path(self.paths.root_dir).mkdir(parents=True, exist_ok=True)
        self._write_fixture_payloads()

    def tearDown(self) -> None:
        for path in [
            self.paths.observation_cases_path,
            self.paths.case_facts_path,
            self.paths.upstream_patterns_path,
            self.paths.upstream_experiments_path,
            self.paths.upstream_outcomes_path,
            self.paths.downstream_audit_patterns_path,
            self.paths.engineering_alerts_path,
            self.paths.difficult_cases_path,
            self.paths.difficult_case_actions_path,
            self.paths.profile_field_trace_index_path,
            self.paths.tasks_path,
            self.paths.task_actions_path,
            self.other_paths.tasks_path,
            self.other_paths.task_actions_path,
        ]:
            Path(path).unlink(missing_ok=True)

    def test_list_reflection_tasks_returns_only_current_users_tasks(self) -> None:
        payload = list_reflection_tasks(
            response=Response(),
            current_user={"username": self.username, "user_id": "user_001"},
        )

        self.assertEqual(payload["task_count"], 1)
        self.assertEqual(payload["tasks"][0]["task_id"], "task_001")
        self.assertEqual(payload["tasks"][0]["user_name"], self.username)

    def test_get_reflection_task_detail_returns_pattern_cases_and_stage_trace(self) -> None:
        payload = get_reflection_task_detail(
            task_id="task_001",
            response=Response(),
            current_user={"username": self.username, "user_id": "user_001"},
        )

        self.assertEqual(payload["task"]["task_id"], "task_001")
        self.assertEqual(payload["pattern"]["pattern_id"], "pattern_001")
        self.assertEqual(len(payload["support_cases"]), 2)
        self.assertEqual(len(payload["evidence_refs"]), 2)
        self.assertEqual(payload["stage_trace"][0]["first_seen_stage"], "lp3")
        self.assertEqual(payload["history_summary"]["recommended_option_success_count"], 1)
        self.assertEqual(payload["history_patterns"][0]["pattern_id"], "pattern_history_001")
        self.assertEqual(payload["history_experiments"][0]["experiment_id"], "exp_history_001")
        self.assertEqual(payload["history_outcomes"][0]["outcome_id"], "outcome_history_001")

    def test_get_reflection_task_detail_returns_404_for_other_user(self) -> None:
        with self.assertRaises(HTTPException) as ctx:
            get_reflection_task_detail(
                task_id="task_001",
                response=Response(),
                current_user={"username": self.other_username, "user_id": "user_002"},
            )

        self.assertEqual(ctx.exception.status_code, 404)

    def test_get_difficult_case_detail_returns_gt_values_and_difficulty_reason(self) -> None:
        payload = get_difficult_case_detail(
            case_id="case_difficult_001",
            response=Response(),
            current_user={"username": self.username, "user_id": "user_001"},
        )

        self.assertEqual(payload["case"]["case_id"], "case_difficult_001")
        self.assertEqual(payload["case"]["resolution_route"], "difficult_case")
        self.assertEqual(payload["case"]["comparison_grade"], "partial_match")
        self.assertEqual(payload["gt_comparison"]["output_value"], ["音乐", "游戏"])
        self.assertEqual(payload["gt_comparison"]["gt_value"], ["音乐", "电影", "游戏"])
        self.assertEqual(payload["gt_comparison"]["grade"], "partial_match")
        self.assertIn("只覆盖了 GT 的一部分", payload["difficulty_reason"])

    def _write_fixture_payloads(self) -> None:
        case_facts = [
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
                "support_count": 1,
                "decision_trace": {"reason": "校园课堂主线稳定"},
                "evidence_refs": [
                    {"source_type": "event", "source_id": "EVT_001", "description": "课堂", "feature_names": ["campus_scene"]},
                ],
                "upstream_output": {"value": "college_student"},
            },
            {
                "case_id": "case_002",
                "user_name": self.username,
                "album_id": "album_002",
                "entity_type": "profile_field",
                "entity_id": "long_term_facts.social_identity.education",
                "dimension": "long_term_facts.social_identity.education",
                "signal_source": "mainline_profile",
                "first_seen_stage": "lp3",
                "surfaced_stage": "lp3",
                "routing_result": "strategy_candidate",
                "business_priority": "high",
                "auto_confidence": 0.78,
                "triage_reason": "profile_field_ready_for_patterning",
                "support_count": 1,
                "decision_trace": {"reason": "校园场景再次出现"},
                "evidence_refs": [
                    {"source_type": "event", "source_id": "EVT_002", "description": "校园", "feature_names": ["campus_scene"]},
                ],
                "upstream_output": {"value": "college_student"},
            },
        ]
        with open(self.paths.case_facts_path, "w", encoding="utf-8") as handle:
            for record in case_facts:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.write(
                json.dumps(
                    {
                        "case_id": "case_difficult_001",
                        "user_name": self.username,
                        "album_id": "album_003",
                        "entity_type": "profile_field",
                        "entity_id": "long_term_facts.hobbies.interests",
                        "dimension": "long_term_facts.hobbies.interests",
                        "signal_source": "mainline_profile",
                        "first_seen_stage": "lp3",
                        "surfaced_stage": "lp3",
                        "routing_result": "strategy_candidate",
                        "business_priority": "medium",
                        "auto_confidence": 0.58,
                        "triage_reason": "profile_field_ready_for_patterning",
                        "support_count": 2,
                        "decision_trace": {"reason": "只命中了部分兴趣"},
                        "evidence_refs": [
                            {"source_type": "event", "source_id": "EVT_003", "description": "音乐", "feature_names": ["music"]},
                            {"source_type": "event", "source_id": "EVT_004", "description": "游戏", "feature_names": ["game"]},
                        ],
                        "upstream_output": {"value": ["音乐", "游戏"]},
                        "gt_payload": {"gt_value": ["音乐", "电影", "游戏"]},
                        "comparison_result": {"grade": "partial_match", "score": 0.66, "method": "rule_set_overlap"},
                        "comparison_grade": "partial_match",
                        "comparison_score": 0.66,
                        "comparison_method": "rule_set_overlap",
                        "accuracy_gap_status": "open",
                        "resolution_route": "difficult_case",
                        "agent_reasoning_summary": "当前结果只覆盖了 GT 的一部分，暂时无法稳定判断是召回不全还是字段归纳口径偏窄。",
                        "trace_payload_path": str(Path(self.paths.profile_field_trace_payload_dir) / "case_difficult_001.json"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

        upstream_patterns = [
            {
                "pattern_id": "pattern_001",
                "user_name": self.username,
                "lane": "upstream",
                "business_priority": "high",
                "root_cause_candidates": ["profile_field_ready_for_patterning"],
                "fix_surface_candidates": ["field_cot", "tool_rule", "call_policy"],
                "support_case_ids": ["case_001", "case_002"],
                "is_direction_clear": False,
                "status": "new",
                "why_blocked": "",
                "recommended_option": "field_cot",
                "support_count": 2,
                "summary": "education 字段反复出现稳定校园证据",
                "dimension": "long_term_facts.social_identity.education",
                "triage_reason": "profile_field_ready_for_patterning",
                "entity_type": "profile_field",
                "evidence_refs": [
                    {"source_type": "event", "source_id": "EVT_001"},
                    {"source_type": "event", "source_id": "EVT_002"},
                ],
                "eligible_for_task": True,
            },
            {
                "pattern_id": "pattern_history_001",
                "user_name": self.username,
                "lane": "upstream",
                "business_priority": "high",
                "root_cause_candidates": ["field_reasoning"],
                "fix_surface_candidates": ["field_cot", "tool_rule", "call_policy"],
                "support_case_ids": ["history_case_001", "history_case_002"],
                "is_direction_clear": True,
                "status": "closed",
                "why_blocked": "",
                "recommended_option": "field_cot",
                "support_count": 2,
                "summary": "历史上的 education 误判",
                "dimension": "long_term_facts.social_identity.education",
                "triage_reason": "profile_field_ready_for_patterning",
                "entity_type": "profile_field",
                "evidence_refs": [
                    {"source_type": "event", "source_id": "EVT_H1"},
                    {"source_type": "event", "source_id": "EVT_H2"},
                ],
                "eligible_for_task": True,
            }
        ]
        Path(self.paths.upstream_patterns_path).write_text(json.dumps(upstream_patterns, ensure_ascii=False, indent=2), encoding="utf-8")
        Path(self.paths.upstream_experiments_path).write_text(
            json.dumps(
                [
                    {
                        "experiment_id": "exp_history_001",
                        "pattern_id": "pattern_history_001",
                        "user_name": self.username,
                        "lane": "upstream",
                        "fix_surface": "field_cot",
                        "change_scope": "single_pattern",
                        "hypothesis": "历史上 education 改 field_cot 有效果",
                        "status": "completed",
                    }
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        Path(self.paths.upstream_outcomes_path).write_text(
            json.dumps(
                [
                    {
                        "outcome_id": "outcome_history_001",
                        "experiment_id": "exp_history_001",
                        "user_name": self.username,
                        "status": "success",
                        "summary": "education 字段显著改善",
                    }
                ],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        Path(self.paths.downstream_audit_patterns_path).write_text("[]", encoding="utf-8")
        Path(self.paths.difficult_cases_path).write_text(
            json.dumps(
                {
                    "case_id": "case_difficult_001",
                    "user_name": self.username,
                    "album_id": "album_003",
                    "dimension": "long_term_facts.hobbies.interests",
                    "entity_type": "profile_field",
                    "summary": "兴趣字段与 GT 部分重叠，当前根因不稳定",
                    "detail_url": "/review/difficult-case/case_difficult_001",
                    "status": "new",
                    "comparison_grade": "partial_match",
                    "comparison_score": 0.66,
                    "resolution_route": "difficult_case",
                    "trace_payload_path": str(Path(self.paths.profile_field_trace_payload_dir) / "case_difficult_001.json"),
                    "evidence_refs": [
                        {"source_type": "event", "source_id": "EVT_003"},
                        {"source_type": "event", "source_id": "EVT_004"},
                    ],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        Path(self.paths.profile_field_trace_index_path).write_text(
            json.dumps(
                {
                    "case_id": "case_difficult_001",
                    "album_id": "album_003",
                    "field_key": "long_term_facts.hobbies.interests",
                    "batch_name": "Semantic Expression::batch_1",
                    "tool_called": True,
                    "retrieval_hit_count": 2,
                    "null_reason": "",
                    "selected_supporting_ref_ids": ["EVT_003", "EVT_004"],
                    "selected_contradicting_ref_ids": [],
                    "trace_payload_path": str(Path(self.paths.profile_field_trace_payload_dir) / "case_difficult_001.json"),
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        Path(self.paths.profile_field_trace_payload_dir).mkdir(parents=True, exist_ok=True)
        Path(self.paths.profile_field_trace_payload_dir, "case_difficult_001.json").write_text(
            json.dumps(
                {
                    "case_id": "case_difficult_001",
                    "batch_name": "Semantic Expression::batch_1",
                    "tool_trace": {"stats_bundle": {"support_count": 2}},
                    "draft": {"value": ["音乐", "游戏"]},
                    "final": {"value": ["音乐", "游戏"], "reasoning": "只命中了部分兴趣"},
                    "llm_batch_debug": [{"raw_response_preview": "{}"}],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        tasks = [
            {
                "task_id": "task_001",
                "task_type": "upstream_decision_task",
                "pattern_id": "pattern_001",
                "user_name": self.username,
                "album_id": "album_002",
                "lane": "upstream",
                "priority": "high",
                "summary": "education 字段反复出现稳定校园证据",
                "detail_url": "/review/task/task_001",
                "support_case_ids": ["case_001", "case_002"],
                "options": ["field_cot", "tool_rule", "call_policy", "engineering_issue", "watch_only"],
                "recommended_option": "field_cot",
                "status": "new",
                "feishu_status": "not_triggered",
                "created_at": "2026-03-26T12:00:00",
                "updated_at": "2026-03-26T12:00:00",
                "why_blocked": "",
                "evidence_refs": [
                    {"source_type": "event", "source_id": "EVT_001"},
                    {"source_type": "event", "source_id": "EVT_002"},
                ],
            }
        ]
        with open(self.paths.tasks_path, "w", encoding="utf-8") as handle:
            for record in tasks:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
