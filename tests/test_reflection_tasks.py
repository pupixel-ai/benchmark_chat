from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch


class ReflectionTaskPipelineTests(unittest.TestCase):
    def test_build_pattern_clusters_uses_root_cause_family_to_recommend_fix_surface(self) -> None:
        from services.reflection import CaseFact
        from services.reflection.tasks import build_pattern_clusters

        case_facts = [
            CaseFact(
                case_id="case_tool_policy_001",
                user_name="vigar",
                album_id="album_001",
                entity_type="profile_field",
                entity_id="long_term_facts.material.brand_preference",
                dimension="long_term_facts.material.brand_preference",
                signal_source="mainline_profile",
                first_seen_stage="lp3",
                surfaced_stage="lp3",
                routing_result="strategy_candidate",
                business_priority="high",
                auto_confidence=0.64,
                triage_reason="profile_field_ready_for_patterning",
                root_cause_family="tool_selection_policy",
                resolution_route="strategy_fix",
                fix_surface_confidence=0.87,
                support_count=1,
                decision_trace={"reason": "字段没有触发正确工具链"},
                evidence_refs=[
                    {"source_type": "event", "source_id": "EVT_001", "description": "品牌线索存在", "feature_names": ["brand_item"]},
                    {"source_type": "event", "source_id": "EVT_002", "description": "重复品牌线索", "feature_names": ["brand_item"]},
                ],
                upstream_output={"value": None},
                badcase_source="empty_output_candidate",
                badcase_kind="missing_value",
            ),
            CaseFact(
                case_id="case_tool_policy_002",
                user_name="vigar",
                album_id="album_002",
                entity_type="profile_field",
                entity_id="long_term_facts.material.brand_preference",
                dimension="long_term_facts.material.brand_preference",
                signal_source="mainline_profile",
                first_seen_stage="lp3",
                surfaced_stage="lp3",
                routing_result="strategy_candidate",
                business_priority="high",
                auto_confidence=0.67,
                triage_reason="profile_field_ready_for_patterning",
                root_cause_family="tool_selection_policy",
                resolution_route="strategy_fix",
                fix_surface_confidence=0.84,
                support_count=1,
                decision_trace={"reason": "同字段再次未触发正确工具"},
                evidence_refs=[
                    {"source_type": "event", "source_id": "EVT_003", "description": "品牌线索存在", "feature_names": ["brand_item"]},
                    {"source_type": "event", "source_id": "EVT_004", "description": "重复品牌线索", "feature_names": ["brand_item"]},
                ],
                upstream_output={"value": None},
                badcase_source="empty_output_candidate",
                badcase_kind="missing_value",
            ),
        ]

        upstream_patterns, _, _ = build_pattern_clusters(case_facts)

        self.assertEqual(len(upstream_patterns), 1)
        self.assertEqual(upstream_patterns[0].recommended_option, "call_policy")
        self.assertTrue(upstream_patterns[0].is_direction_clear)
        self.assertGreaterEqual(upstream_patterns[0].fix_surface_confidence, 0.8)

    def test_build_pattern_clusters_groups_supported_cases_and_emits_engineering_alerts(self) -> None:
        from services.reflection import CaseFact
        from services.reflection.tasks import build_pattern_clusters

        case_facts = [
            CaseFact(
                case_id="case_001",
                user_name="vigar",
                album_id="album_001",
                entity_type="profile_field",
                entity_id="long_term_facts.social_identity.education",
                dimension="long_term_facts.social_identity.education",
                signal_source="mainline_profile",
                first_seen_stage="lp3",
                surfaced_stage="lp3",
                routing_result="strategy_candidate",
                business_priority="high",
                auto_confidence=0.82,
                triage_reason="profile_field_ready_for_patterning",
                support_count=1,
                resolution_route="strategy_fix",
                decision_trace={"reason": "校园课堂主线稳定"},
                evidence_refs=[
                    {"source_type": "event", "source_id": "EVT_001", "description": "课堂", "feature_names": ["campus_scene"]},
                ],
                upstream_output={"value": "college_student"},
            ),
            CaseFact(
                case_id="case_002",
                user_name="vigar",
                album_id="album_002",
                entity_type="profile_field",
                entity_id="long_term_facts.social_identity.education",
                dimension="long_term_facts.social_identity.education",
                signal_source="mainline_profile",
                first_seen_stage="lp3",
                surfaced_stage="lp3",
                routing_result="strategy_candidate",
                business_priority="high",
                auto_confidence=0.79,
                triage_reason="profile_field_ready_for_patterning",
                support_count=1,
                resolution_route="strategy_fix",
                decision_trace={"reason": "校园场景再次出现"},
                evidence_refs=[
                    {"source_type": "event", "source_id": "EVT_002", "description": "校园", "feature_names": ["campus_scene"]},
                ],
                upstream_output={"value": "college_student"},
            ),
            CaseFact(
                case_id="case_003",
                user_name="vigar",
                album_id="album_003",
                entity_type="system_runtime",
                entity_id="profile_agent_runtime",
                dimension="system>audit_runtime_failure",
                signal_source="downstream_audit",
                first_seen_stage="downstream_audit",
                surfaced_stage="downstream_audit",
                routing_result="engineering_issue",
                business_priority="high",
                auto_confidence=1.0,
                triage_reason="downstream_runtime_failure",
                decision_trace={"reason": "profile agent unavailable"},
                evidence_refs=[],
            ),
            CaseFact(
                case_id="case_004",
                user_name="vigar",
                album_id="album_004",
                entity_type="profile_field",
                entity_id="short_term_expression.current_mood",
                dimension="short_term_expression.current_mood",
                signal_source="mainline_profile",
                first_seen_stage="lp3",
                surfaced_stage="lp3",
                routing_result="expected_uncertainty",
                business_priority="low",
                auto_confidence=0.18,
                triage_reason="null_field_without_supporting_evidence",
                decision_trace={},
                evidence_refs=[],
                upstream_output={"value": None},
            ),
        ]

        upstream_patterns, downstream_patterns, engineering_alerts = build_pattern_clusters(case_facts)

        self.assertEqual(len(upstream_patterns), 1)
        self.assertEqual(len(downstream_patterns), 0)
        self.assertEqual(len(engineering_alerts), 1)
        self.assertEqual(upstream_patterns[0].support_count, 2)
        self.assertEqual(upstream_patterns[0].lane, "upstream")
        self.assertTrue(upstream_patterns[0].eligible_for_task)
        self.assertEqual(upstream_patterns[0].recommended_option, "field_cot")
        self.assertEqual(engineering_alerts[0].alert_type, "engineering_issue")

    def test_build_decision_tasks_only_emits_eligible_high_priority_patterns(self) -> None:
        from services.reflection import PatternCluster
        from services.reflection.tasks import build_decision_tasks

        patterns = [
            PatternCluster(
                pattern_id="pattern_001",
                user_name="vigar",
                lane="upstream",
                business_priority="high",
                root_cause_candidates=["profile_field_ready_for_patterning"],
                fix_surface_candidates=["field_cot", "tool_rule", "call_policy"],
                support_case_ids=["case_001", "case_002"],
                is_direction_clear=False,
                support_count=2,
                summary="education 字段反复出现稳定校园证据",
                evidence_refs=[
                    {"source_type": "event", "source_id": "EVT_001"},
                    {"source_type": "event", "source_id": "EVT_002"},
                ],
                recommended_option="field_cot",
                eligible_for_task=True,
                resolution_route="strategy_fix",
            ),
            PatternCluster(
                pattern_id="pattern_002",
                user_name="vigar",
                lane="downstream",
                business_priority="medium",
                root_cause_candidates=["downstream_judge_challenged_existing_output"],
                fix_surface_candidates=["critic_rule", "judge_boundary"],
                support_case_ids=["case_010", "case_011"],
                is_direction_clear=False,
                support_count=2,
                summary="中优先级 pattern",
                evidence_refs=[
                    {"source_type": "event", "source_id": "EVT_010"},
                    {"source_type": "event", "source_id": "EVT_011"},
                ],
                recommended_option="critic_rule",
                eligible_for_task=False,
                why_blocked="low_priority",
            ),
        ]

        tasks = build_decision_tasks(patterns)

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_type, "upstream_decision_task")
        self.assertEqual(tasks[0].recommended_option, "field_cot")
        self.assertEqual(
            tasks[0].options,
            ["field_cot", "tool_rule", "call_policy", "engineering_issue", "watch_only"],
        )
        self.assertEqual(tasks[0].detail_url, f"/review/task/{tasks[0].task_id}")

    def test_persist_reflection_tasks_upserts_existing_open_task(self) -> None:
        from services.reflection import DecisionReviewItem, PatternCluster
        from services.reflection.tasks import persist_reflection_tasks

        with tempfile.TemporaryDirectory() as tmpdir:
            patterns = [
                PatternCluster(
                    pattern_id="pattern_001",
                    user_name="vigar",
                    lane="upstream",
                    business_priority="high",
                    root_cause_candidates=["profile_field_ready_for_patterning"],
                    fix_surface_candidates=["field_cot", "tool_rule", "call_policy"],
                    support_case_ids=["case_001", "case_002"],
                    is_direction_clear=False,
                    support_count=2,
                    summary="first summary",
                    evidence_refs=[
                        {"source_type": "event", "source_id": "EVT_001"},
                        {"source_type": "event", "source_id": "EVT_002"},
                    ],
                    recommended_option="field_cot",
                    eligible_for_task=True,
                )
            ]
            tasks = [
                DecisionReviewItem(
                    task_id="task_001",
                    task_type="upstream_decision_task",
                    pattern_id="pattern_001",
                    user_name="vigar",
                    lane="upstream",
                    priority="high",
                    summary="first summary",
                    detail_url="/review/task/task_001",
                    support_case_ids=["case_001", "case_002"],
                    options=["field_cot", "tool_rule", "call_policy", "engineering_issue", "watch_only"],
                    recommended_option="field_cot",
                    status="new",
                    feishu_status="not_triggered",
                    created_at="2026-03-26T10:00:00",
                    updated_at="2026-03-26T10:00:00",
                )
            ]

            first = persist_reflection_tasks(
                project_root=tmpdir,
                user_name="vigar",
                upstream_patterns=patterns,
                downstream_patterns=[],
                tasks=tasks,
                engineering_alerts=[],
            )
            time.sleep(0.01)
            tasks[0].summary = "updated summary"
            tasks[0].support_case_ids = ["case_001", "case_002", "case_003"]
            second = persist_reflection_tasks(
                project_root=tmpdir,
                user_name="vigar",
                upstream_patterns=patterns,
                downstream_patterns=[],
                tasks=tasks,
                engineering_alerts=[],
            )

            with open(first["tasks_path"], "r", encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle if line.strip()]

            self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["summary"], "updated summary")
        self.assertEqual(records[0]["support_case_ids"], ["case_001", "case_002", "case_003"])
        self.assertEqual(second["written_task_count"], 1)

    def test_run_reflection_task_generation_marks_gt_mismatch_and_writes_comparisons(self) -> None:
        from services.reflection import build_reflection_asset_paths
        from services.reflection.tasks import run_reflection_task_generation

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_reflection_asset_paths(project_root=tmpdir, user_name="vigar")
            Path(paths.root_dir).mkdir(parents=True, exist_ok=True)
            with open(paths.case_facts_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "case_id": "case_gt_001",
                            "user_name": "vigar",
                            "album_id": "album_010",
                            "entity_type": "profile_field",
                            "entity_id": "long_term_facts.social_identity.education",
                            "dimension": "long_term_facts.social_identity.education",
                            "signal_source": "mainline_profile",
                            "first_seen_stage": "lp3",
                            "surfaced_stage": "lp3",
                            "routing_result": "strategy_candidate",
                            "business_priority": "medium",
                            "auto_confidence": 0.81,
                            "triage_reason": "profile_field_ready_for_patterning",
                            "decision_trace": {"reason": "校园主线成立"},
                            "tool_usage_summary": {
                                "tool_called": True,
                                "retrieval_hit_count": 2,
                            },
                            "upstream_output": {"value": "college_student"},
                            "evidence_refs": [
                                {"source_type": "event", "source_id": "EVT_010", "description": "课堂", "feature_names": ["campus_scene"]},
                                {"source_type": "event", "source_id": "EVT_011", "description": "校园", "feature_names": ["campus_scene"]},
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            with open(paths.profile_field_gt_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "album_id": "album_010",
                            "field_key": "long_term_facts.social_identity.education",
                            "gt_value": "master_student",
                            "labeler": "vigar",
                            "notes": "人工确认是研究生阶段",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            result = run_reflection_task_generation(project_root=tmpdir, user_name="vigar")

            with open(paths.case_facts_path, "r", encoding="utf-8") as handle:
                case_fact = json.loads(handle.readline())
            with open(paths.gt_comparisons_path, "r", encoding="utf-8") as handle:
                comparison = json.loads(handle.readline())

            self.assertEqual(case_fact["badcase_source"], "gt_mismatch_candidate")
            self.assertEqual(case_fact["badcase_kind"], "wrong_value")
            self.assertEqual(case_fact["accuracy_gap_status"], "open")
            self.assertEqual(case_fact["comparison_grade"], "mismatch")
            self.assertEqual(case_fact["comparison_result"]["grade"], "mismatch")
            self.assertEqual(case_fact["gt_payload"]["gt_value"], "master_student")
            self.assertEqual(comparison["case_id"], "case_gt_001")
            self.assertEqual(comparison["comparison_result"]["severity"], "high")
            self.assertEqual(result["written_gt_comparison_count"], 1)

    def test_run_reflection_task_generation_marks_close_match_as_resolved(self) -> None:
        from services.reflection import build_reflection_asset_paths
        from services.reflection.tasks import run_reflection_task_generation

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_reflection_asset_paths(project_root=tmpdir, user_name="vigar")
            Path(paths.root_dir).mkdir(parents=True, exist_ok=True)
            with open(paths.case_facts_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "case_id": "case_gt_close_001",
                            "user_name": "vigar",
                            "album_id": "album_close_010",
                            "entity_type": "profile_field",
                            "entity_id": "long_term_facts.social_identity.education",
                            "dimension": "long_term_facts.social_identity.education",
                            "signal_source": "mainline_profile",
                            "first_seen_stage": "lp3",
                            "surfaced_stage": "lp3",
                            "routing_result": "strategy_candidate",
                            "business_priority": "medium",
                            "auto_confidence": 0.81,
                            "triage_reason": "profile_field_ready_for_patterning",
                            "decision_trace": {"reason": "校园主线成立"},
                            "tool_usage_summary": {
                                "tool_called": True,
                                "retrieval_hit_count": 2,
                            },
                            "upstream_output": {"value": "college_student"},
                            "evidence_refs": [
                                {"source_type": "event", "source_id": "EVT_010", "description": "课堂", "feature_names": ["campus_scene"]},
                                {"source_type": "event", "source_id": "EVT_011", "description": "校园", "feature_names": ["campus_scene"]},
                            ],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            with open(paths.profile_field_gt_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "album_id": "album_close_010",
                            "field_key": "long_term_facts.social_identity.education",
                            "gt_value": "student",
                            "labeler": "vigar",
                            "notes": "更宽泛的学生标签",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            run_reflection_task_generation(project_root=tmpdir, user_name="vigar")

            with open(paths.case_facts_path, "r", encoding="utf-8") as handle:
                case_fact = json.loads(handle.readline())

            self.assertEqual(case_fact["comparison_grade"], "close_match")
            self.assertEqual(case_fact["accuracy_gap_status"], "resolved")
            self.assertEqual(case_fact["routing_result"], "resolved_ok")

    def test_run_reflection_task_generation_writes_difficult_case_for_partial_match(self) -> None:
        from services.reflection import build_reflection_asset_paths
        from services.reflection.tasks import run_reflection_task_generation

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_reflection_asset_paths(project_root=tmpdir, user_name="vigar")
            Path(paths.root_dir).mkdir(parents=True, exist_ok=True)
            with open(paths.case_facts_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "case_id": "case_gt_partial_001",
                            "user_name": "vigar",
                            "album_id": "album_partial_010",
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
                            "decision_trace": {"reason": "近期兴趣有音乐和游戏"},
                            "tool_usage_summary": {
                                "tool_called": True,
                                "retrieval_hit_count": 2,
                            },
                            "upstream_output": {"value": ["音乐", "游戏"]},
                            "evidence_refs": [
                                {"source_type": "event", "source_id": "EVT_020", "description": "音乐活动", "feature_names": ["music"]},
                                {"source_type": "event", "source_id": "EVT_021", "description": "游戏内容", "feature_names": ["game"]},
                            ],
                            "trace_payload_path": "/tmp/profile_trace_case_gt_partial_001.json",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            with open(paths.profile_field_gt_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "album_id": "album_partial_010",
                            "field_key": "long_term_facts.hobbies.interests",
                            "gt_value": ["音乐", "电影", "游戏"],
                            "labeler": "vigar",
                            "notes": "人工标注是三项兴趣",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            result = run_reflection_task_generation(project_root=tmpdir, user_name="vigar")

            with open(paths.case_facts_path, "r", encoding="utf-8") as handle:
                case_fact = json.loads(handle.readline())
            with open(paths.difficult_cases_path, "r", encoding="utf-8") as handle:
                difficult_case = json.loads(handle.readline())

            self.assertEqual(case_fact["comparison_grade"], "partial_match")
            self.assertEqual(case_fact["accuracy_gap_status"], "open")
            self.assertEqual(case_fact["resolution_route"], "difficult_case")
            self.assertEqual(difficult_case["case_id"], "case_gt_partial_001")
            self.assertEqual(difficult_case["detail_url"], "/review/difficult-case/case_gt_partial_001")
            self.assertEqual(result["written_difficult_case_count"], 1)

    def test_run_reflection_task_generation_marks_downstream_caused_when_backflow_breaks_gt_alignment(self) -> None:
        from services.reflection import build_reflection_asset_paths
        from services.reflection.tasks import run_reflection_task_generation

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_reflection_asset_paths(project_root=tmpdir, user_name="vigar")
            Path(paths.root_dir).mkdir(parents=True, exist_ok=True)
            with open(paths.case_facts_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "case_id": "case_gt_downstream_001",
                            "user_name": "vigar",
                            "album_id": "album_downstream_010",
                            "entity_type": "profile_field",
                            "entity_id": "long_term_facts.social_identity.education",
                            "dimension": "long_term_facts.social_identity.education",
                            "signal_source": "mainline_profile",
                            "first_seen_stage": "lp3",
                            "surfaced_stage": "lp3",
                            "routing_result": "strategy_candidate",
                            "business_priority": "high",
                            "auto_confidence": 0.82,
                            "triage_reason": "profile_field_missing_value_candidate",
                            "decision_trace": {"reason": "回流后被下游否决"},
                            "tool_usage_summary": {
                                "tool_called": True,
                                "retrieval_hit_count": 2,
                            },
                            "upstream_output": {"value": None},
                            "pre_audit_output": {"value": "college_student"},
                            "audit_action_type": "nullify",
                            "downstream_judge": {
                                "verdict": "nullify",
                                "reason": "证据不足",
                                "agent_type": "profile",
                            },
                            "evidence_refs": [
                                {"source_type": "event", "source_id": "EVT_030", "description": "课堂", "feature_names": ["campus_scene"]},
                                {"source_type": "event", "source_id": "EVT_031", "description": "校园", "feature_names": ["campus_scene"]},
                            ],
                            "trace_payload_path": "/tmp/profile_trace_case_gt_downstream_001.json",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            with open(paths.profile_field_gt_path, "w", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "album_id": "album_downstream_010",
                            "field_key": "long_term_facts.social_identity.education",
                            "gt_value": "student",
                            "labeler": "vigar",
                            "notes": "宽泛学生标签",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            result = run_reflection_task_generation(project_root=tmpdir, user_name="vigar")

            with open(paths.case_facts_path, "r", encoding="utf-8") as handle:
                case_fact = json.loads(handle.readline())
            with open(paths.difficult_cases_path, "r", encoding="utf-8") as handle:
                difficult_case = json.loads(handle.readline())

            self.assertEqual(case_fact["pre_audit_comparison_grade"], "close_match")
            self.assertEqual(case_fact["comparison_grade"], "missing_prediction")
            self.assertEqual(case_fact["causality_route"], "downstream_caused")
            self.assertEqual(case_fact["routing_result"], "audit_disagreement")
            self.assertEqual(case_fact["resolution_route"], "difficult_case")
            self.assertEqual(difficult_case["case_id"], "case_gt_downstream_001")
            self.assertEqual(result["written_difficult_case_count"], 1)

    def test_build_strategy_experiments_only_emits_clear_single_variable_upstream_patterns(self) -> None:
        from services.reflection import PatternCluster
        from services.reflection.tasks import build_strategy_experiments

        patterns = [
            PatternCluster(
                pattern_id="pattern_clear_001",
                user_name="vigar",
                lane="upstream",
                business_priority="high",
                root_cause_candidates=["field_reasoning"],
                fix_surface_candidates=["field_cot", "tool_rule", "call_policy"],
                support_case_ids=["case_001", "case_002"],
                is_direction_clear=True,
                support_count=2,
                summary="education 字段稳定误判",
                evidence_refs=[
                    {"source_type": "event", "source_id": "EVT_001"},
                    {"source_type": "event", "source_id": "EVT_002"},
                ],
                recommended_option="field_cot",
                fix_surface_confidence=0.92,
                eligible_for_task=True,
                resolution_route="strategy_fix",
            ),
            PatternCluster(
                pattern_id="pattern_pending_002",
                user_name="vigar",
                lane="upstream",
                business_priority="high",
                root_cause_candidates=["tool_selection_policy"],
                fix_surface_candidates=["call_policy", "tool_rule", "field_cot"],
                support_case_ids=["case_003", "case_004"],
                is_direction_clear=False,
                support_count=2,
                summary="brand_preference 方向未定",
                evidence_refs=[
                    {"source_type": "event", "source_id": "EVT_003"},
                    {"source_type": "event", "source_id": "EVT_004"},
                ],
                recommended_option="call_policy",
                fix_surface_confidence=0.56,
                eligible_for_task=True,
                resolution_route="strategy_fix",
            ),
        ]

        experiments = build_strategy_experiments(patterns)

        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0].fix_surface, "field_cot")
        self.assertEqual(experiments[0].change_scope, "single_pattern")

    def test_build_strategy_experiments_allows_single_case_lp3_gt_mismatch_high(self) -> None:
        from services.reflection import PatternCluster
        from services.reflection.tasks import build_strategy_experiments

        patterns = [
            PatternCluster(
                pattern_id="pattern_single_mismatch_001",
                user_name="vigar",
                lane="upstream",
                business_priority="high",
                root_cause_candidates=["field_reasoning"],
                fix_surface_candidates=["field_cot", "tool_rule", "call_policy"],
                support_case_ids=["case_single_001"],
                is_direction_clear=True,
                support_count=1,
                summary="motivation_shift 单 case GT mismatch",
                evidence_refs=[
                    {"source_type": "event", "source_id": "EVT_101"},
                    {"source_type": "event", "source_id": "EVT_102"},
                ],
                recommended_option="field_cot",
                fix_surface_confidence=0.91,
                eligible_for_task=False,
                resolution_route="strategy_fix",
                entity_type="profile_field",
                dimension="short_term_expression.motivation_shift",
                comparison_grade="mismatch",
            )
        ]

        experiments = build_strategy_experiments(patterns)

        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0].pattern_id, "pattern_single_mismatch_001")
        self.assertEqual(experiments[0].fix_surface, "field_cot")

    def test_enrich_patterns_with_learning_history_uses_success_and_failure_memory(self) -> None:
        from services.reflection import ExperimentOutcome, PatternCluster, StrategyExperiment
        from services.reflection.tasks import enrich_patterns_with_learning_history

        patterns = [
            PatternCluster(
                pattern_id="pattern_current_success",
                user_name="vigar",
                lane="upstream",
                business_priority="high",
                root_cause_candidates=["field_reasoning"],
                fix_surface_candidates=["field_cot", "tool_rule", "call_policy"],
                support_case_ids=["case_001", "case_002"],
                is_direction_clear=True,
                support_count=2,
                summary="education 字段再次出现稳定误判",
                evidence_refs=[
                    {"source_type": "event", "source_id": "EVT_001"},
                    {"source_type": "event", "source_id": "EVT_002"},
                ],
                recommended_option="field_cot",
                fix_surface_confidence=0.82,
                eligible_for_task=True,
                resolution_route="strategy_fix",
                dimension="long_term_facts.social_identity.education",
                triage_reason="profile_field_ready_for_patterning",
                entity_type="profile_field",
            ),
            PatternCluster(
                pattern_id="pattern_current_failure",
                user_name="vigar",
                lane="upstream",
                business_priority="high",
                root_cause_candidates=["tool_selection_policy"],
                fix_surface_candidates=["call_policy", "tool_rule", "field_cot"],
                support_case_ids=["case_003", "case_004"],
                is_direction_clear=True,
                support_count=2,
                summary="brand_preference 字段继续未触发工具",
                evidence_refs=[
                    {"source_type": "event", "source_id": "EVT_003"},
                    {"source_type": "event", "source_id": "EVT_004"},
                ],
                recommended_option="call_policy",
                fix_surface_confidence=0.83,
                eligible_for_task=True,
                resolution_route="strategy_fix",
                dimension="long_term_facts.material.brand_preference",
                triage_reason="profile_field_ready_for_patterning",
                entity_type="profile_field",
            ),
        ]

        history_patterns = [
            {
                "pattern_id": "pattern_history_success",
                "user_name": "vigar",
                "lane": "upstream",
                "business_priority": "high",
                "root_cause_candidates": ["field_reasoning"],
                "fix_surface_candidates": ["field_cot", "tool_rule", "call_policy"],
                "support_case_ids": ["old_case_001", "old_case_002"],
                "is_direction_clear": True,
                "support_count": 2,
                "summary": "旧的 education 误判",
                "evidence_refs": [],
                "recommended_option": "field_cot",
                "eligible_for_task": True,
                "dimension": "long_term_facts.social_identity.education",
                "triage_reason": "profile_field_ready_for_patterning",
                "entity_type": "profile_field",
            },
            {
                "pattern_id": "pattern_history_failure",
                "user_name": "vigar",
                "lane": "upstream",
                "business_priority": "high",
                "root_cause_candidates": ["tool_selection_policy"],
                "fix_surface_candidates": ["call_policy", "tool_rule", "field_cot"],
                "support_case_ids": ["old_case_003", "old_case_004"],
                "is_direction_clear": True,
                "support_count": 2,
                "summary": "旧的 brand_preference 缺失",
                "evidence_refs": [],
                "recommended_option": "call_policy",
                "eligible_for_task": True,
                "dimension": "long_term_facts.material.brand_preference",
                "triage_reason": "profile_field_ready_for_patterning",
                "entity_type": "profile_field",
            },
        ]
        history_experiments = [
            StrategyExperiment(
                experiment_id="exp_success",
                pattern_id="pattern_history_success",
                user_name="vigar",
                lane="upstream",
                fix_surface="field_cot",
                change_scope="single_pattern",
                hypothesis="教育字段改 field_cot",
                status="completed",
            ),
            StrategyExperiment(
                experiment_id="exp_failure_one",
                pattern_id="pattern_history_failure",
                user_name="vigar",
                lane="upstream",
                fix_surface="call_policy",
                change_scope="single_pattern",
                hypothesis="品牌字段改 call_policy",
                status="completed",
            ),
            StrategyExperiment(
                experiment_id="exp_failure_two",
                pattern_id="pattern_history_failure",
                user_name="vigar",
                lane="upstream",
                fix_surface="call_policy",
                change_scope="single_pattern",
                hypothesis="品牌字段再次改 call_policy",
                status="completed",
            ),
        ]
        history_outcomes = [
            ExperimentOutcome(
                outcome_id="outcome_success",
                experiment_id="exp_success",
                user_name="vigar",
                status="success",
                summary="education 字段明显改善",
            ),
            ExperimentOutcome(
                outcome_id="outcome_failure_one",
                experiment_id="exp_failure_one",
                user_name="vigar",
                status="failed",
                summary="call_policy 调整后没有改善",
            ),
            ExperimentOutcome(
                outcome_id="outcome_failure_two",
                experiment_id="exp_failure_two",
                user_name="vigar",
                status="failed",
                summary="再次失败",
            ),
        ]

        enriched = enrich_patterns_with_learning_history(
            patterns=patterns,
            historical_patterns=history_patterns,
            historical_experiments=history_experiments,
            historical_outcomes=history_outcomes,
        )

        self.assertEqual(len(enriched[0].history_pattern_ids), 1)
        self.assertGreater(enriched[0].fix_surface_confidence, 0.82)
        self.assertTrue(enriched[0].is_direction_clear)
        self.assertEqual(enriched[0].history_summary["recommended_option_success_count"], 1)
        self.assertEqual(enriched[1].history_summary["recommended_option_failure_count"], 2)
        self.assertFalse(enriched[1].is_direction_clear)
        self.assertEqual(enriched[1].why_blocked, "historical_failures_need_review")

    def test_build_strategy_experiments_skips_duplicate_open_history_and_carries_learning_summary(self) -> None:
        from services.reflection import PatternCluster, StrategyExperiment
        from services.reflection.tasks import build_strategy_experiments

        patterns = [
            PatternCluster(
                pattern_id="pattern_new_001",
                user_name="vigar",
                lane="upstream",
                business_priority="high",
                root_cause_candidates=["field_reasoning"],
                fix_surface_candidates=["field_cot", "tool_rule", "call_policy"],
                support_case_ids=["case_001", "case_002"],
                is_direction_clear=True,
                support_count=2,
                summary="education 字段稳定误判",
                evidence_refs=[
                    {"source_type": "event", "source_id": "EVT_001"},
                    {"source_type": "event", "source_id": "EVT_002"},
                ],
                recommended_option="field_cot",
                fix_surface_confidence=0.92,
                eligible_for_task=True,
                resolution_route="strategy_fix",
                history_pattern_ids=["pattern_old_001"],
                history_experiment_ids=["exp_done_001"],
                history_summary={
                    "recommended_option_success_count": 1,
                    "recommended_option_failure_count": 0,
                    "open_recommended_experiment_count": 0,
                },
            ),
            PatternCluster(
                pattern_id="pattern_dup_002",
                user_name="vigar",
                lane="upstream",
                business_priority="high",
                root_cause_candidates=["tool_selection_policy"],
                fix_surface_candidates=["call_policy", "tool_rule", "field_cot"],
                support_case_ids=["case_003", "case_004"],
                is_direction_clear=True,
                support_count=2,
                summary="brand_preference 继续未触发工具",
                evidence_refs=[
                    {"source_type": "event", "source_id": "EVT_003"},
                    {"source_type": "event", "source_id": "EVT_004"},
                ],
                recommended_option="call_policy",
                fix_surface_confidence=0.91,
                eligible_for_task=True,
                resolution_route="strategy_fix",
                history_pattern_ids=["pattern_old_002"],
                history_experiment_ids=["exp_open_002"],
                history_summary={
                    "recommended_option_success_count": 0,
                    "recommended_option_failure_count": 0,
                    "open_recommended_experiment_count": 1,
                },
            ),
        ]

        experiments = build_strategy_experiments(patterns)

        self.assertEqual(len(experiments), 1)
        self.assertEqual(experiments[0].pattern_id, "pattern_new_001")
        self.assertEqual(experiments[0].history_pattern_ids, ["pattern_old_001"])
        self.assertEqual(experiments[0].metrics["history_success_count"], 1)
        self.assertEqual(experiments[0].metrics["history_failure_count"], 0)

    def test_persist_reflection_tasks_keeps_historical_patterns_and_experiments(self) -> None:
        from services.reflection import PatternCluster, StrategyExperiment, build_reflection_asset_paths
        from services.reflection.tasks import persist_reflection_tasks

        with tempfile.TemporaryDirectory() as tmpdir:
            first_patterns = [
                PatternCluster(
                    pattern_id="pattern_first",
                    user_name="vigar",
                    lane="upstream",
                    business_priority="high",
                    root_cause_candidates=["field_reasoning"],
                    fix_surface_candidates=["field_cot", "tool_rule", "call_policy"],
                    support_case_ids=["case_001", "case_002"],
                    is_direction_clear=True,
                    support_count=2,
                    summary="first pattern",
                    evidence_refs=[],
                    recommended_option="field_cot",
                    eligible_for_task=True,
                    resolution_route="strategy_fix",
                )
            ]
            first_experiments = [
                StrategyExperiment(
                    experiment_id="exp_first",
                    pattern_id="pattern_first",
                    user_name="vigar",
                    lane="upstream",
                    fix_surface="field_cot",
                    change_scope="single_pattern",
                    hypothesis="first hypothesis",
                    status="proposed",
                )
            ]
            persist_reflection_tasks(
                project_root=tmpdir,
                user_name="vigar",
                upstream_patterns=first_patterns,
                downstream_patterns=[],
                tasks=[],
                engineering_alerts=[],
                strategy_experiments=first_experiments,
            )

            second_patterns = [
                PatternCluster(
                    pattern_id="pattern_second",
                    user_name="vigar",
                    lane="upstream",
                    business_priority="high",
                    root_cause_candidates=["tool_selection_policy"],
                    fix_surface_candidates=["call_policy", "tool_rule", "field_cot"],
                    support_case_ids=["case_003", "case_004"],
                    is_direction_clear=False,
                    support_count=2,
                    summary="second pattern",
                    evidence_refs=[],
                    recommended_option="call_policy",
                    eligible_for_task=True,
                    resolution_route="strategy_fix",
                )
            ]
            persist_reflection_tasks(
                project_root=tmpdir,
                user_name="vigar",
                upstream_patterns=second_patterns,
                downstream_patterns=[],
                tasks=[],
                engineering_alerts=[],
                strategy_experiments=[],
            )

            paths = build_reflection_asset_paths(project_root=tmpdir, user_name="vigar")
            upstream_patterns_payload = json.loads(Path(paths.upstream_patterns_path).read_text(encoding="utf-8"))
            upstream_experiments_payload = json.loads(Path(paths.upstream_experiments_path).read_text(encoding="utf-8"))

            self.assertEqual({item["pattern_id"] for item in upstream_patterns_payload}, {"pattern_first", "pattern_second"})
            self.assertEqual({item["experiment_id"] for item in upstream_experiments_payload}, {"exp_first"})

    def test_persist_reflection_tasks_writes_proposals_and_proposal_review_tasks(self) -> None:
        from services.reflection import build_reflection_asset_paths
        from services.reflection.tasks import persist_reflection_tasks

        with tempfile.TemporaryDirectory() as tmpdir:
            payload = persist_reflection_tasks(
                project_root=tmpdir,
                user_name="vigar",
                upstream_patterns=[],
                downstream_patterns=[],
                tasks=[
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
                        "feishu_status": "not_triggered",
                        "proposal_id": "proposal_001",
                        "experiment_id": "exp_001",
                        "created_at": "2026-03-26T12:00:00Z",
                        "updated_at": "2026-03-26T12:00:00Z",
                    }
                ],
                engineering_alerts=[],
                proposals=[
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
                        "recommended_option": "approve",
                        "options": ["approve", "reject", "need_revision"],
                        "agent_reasoning_summary": "字段归纳口径有偏差",
                        "why_not_other_surfaces": "没有明显工具召回问题",
                        "diff_summary": ["新增 education 专项 COT 步骤"],
                        "baseline_metrics": {"exact_or_close_count": 0},
                        "candidate_metrics": {"exact_or_close_count": 2},
                        "metric_gain": {"exact_or_close_delta": 2},
                        "override_bundle_path": "/tmp/overlay.json",
                        "experiment_report_path": "/tmp/report.json",
                        "proposal_status": "pending_review",
                        "created_at": "2026-03-26T12:00:00Z",
                        "updated_at": "2026-03-26T12:00:00Z",
                    }
                ],
            )

            paths = build_reflection_asset_paths(project_root=tmpdir, user_name="vigar")
            proposals = [json.loads(line) for line in Path(paths.proposals_path).read_text(encoding="utf-8").splitlines() if line.strip()]
            tasks = [json.loads(line) for line in Path(paths.tasks_path).read_text(encoding="utf-8").splitlines() if line.strip()]

        self.assertEqual(payload["written_proposal_count"], 1)
        self.assertEqual(proposals[0]["proposal_id"], "proposal_001")
        self.assertEqual(tasks[0]["task_type"], "proposal_review")

    def test_dispatch_strict_reflection_notifications_sends_new_tasks_and_difficult_cases_once(self) -> None:
        from services.reflection import build_reflection_asset_paths
        from services.reflection.tasks import dispatch_strict_reflection_notifications

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_reflection_asset_paths(project_root=tmpdir, user_name="vigar")
            Path(paths.root_dir).mkdir(parents=True, exist_ok=True)
            Path(paths.tasks_path).write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "task_id": "task_notify_001",
                                "task_type": "proposal_review",
                                "pattern_id": "pattern_001",
                                "user_name": "vigar",
                                "detail_url": "/review/task/task_notify_001",
                                "status": "new",
                                "feishu_status": "not_triggered",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "task_id": "task_skip_001",
                                "task_type": "upstream_decision_task",
                                "pattern_id": "pattern_002",
                                "user_name": "vigar",
                                "detail_url": "/review/task/task_skip_001",
                                "status": "approved",
                                "feishu_status": "acted",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            Path(paths.difficult_cases_path).write_text(
                json.dumps(
                    {
                        "case_id": "case_difficult_001",
                        "user_name": "vigar",
                        "detail_url": "/review/difficult-case/case_difficult_001",
                        "resolution_route": "difficult_case",
                        "accuracy_gap_status": "open",
                        "feishu_status": "not_triggered",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("services.reflection.tasks.FEISHU_APPROVAL_RECEIVE_ID", "oc_approval_group_001"), patch(
                "services.reflection.tasks.FEISHU_APPROVAL_RECEIVE_ID_TYPE", "chat_id"
            ), patch(
                "services.reflection.tasks.FEISHU_DIFFICULT_CASE_RECEIVE_ID", "oc_difficult_group_001"
            ), patch(
                "services.reflection.tasks.FEISHU_DIFFICULT_CASE_RECEIVE_ID_TYPE", "chat_id"
            ), patch("services.reflection.feishu.send_reflection_task_card_for_task") as send_task, patch(
                "services.reflection.feishu.send_difficult_case_alert_for_case"
            ) as send_case:
                send_task.return_value = {"message_id": "om_task"}
                send_case.return_value = {"message_id": "om_case"}
                result = dispatch_strict_reflection_notifications(
                    project_root=tmpdir,
                    user_name="vigar",
                )

        self.assertEqual(result["sent_task_count"], 1)
        self.assertEqual(result["sent_difficult_case_count"], 1)
        send_task.assert_called_once()
        send_case.assert_called_once()
        self.assertEqual(send_task.call_args.kwargs["receive_id"], "oc_approval_group_001")
        self.assertEqual(send_task.call_args.kwargs["receive_id_type"], "chat_id")
        self.assertEqual(send_case.call_args.kwargs["receive_id"], "oc_difficult_group_001")
        self.assertEqual(send_case.call_args.kwargs["receive_id_type"], "chat_id")
