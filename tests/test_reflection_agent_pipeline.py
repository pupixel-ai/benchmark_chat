from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class ReflectionAgentPipelineTests(unittest.TestCase):
    def test_badcase_packet_assembler_collects_trace_and_history_payload(self) -> None:
        from services.reflection import build_reflection_asset_paths
        from services.reflection.types import CaseFact
        from services.reflection.upstream_agent import BadcasePacketAssembler

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_reflection_asset_paths(project_root=tmpdir, user_name="vigar")
            Path(paths.root_dir).mkdir(parents=True, exist_ok=True)
            Path(paths.upstream_patterns_path).write_text(
                json.dumps(
                    [
                        {
                            "pattern_id": "pattern_history_001",
                            "dimension": "long_term_facts.social_identity.education",
                            "root_cause_candidates": ["field_reasoning"],
                            "recommended_option": "field_cot",
                        }
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            Path(paths.upstream_experiments_path).write_text(
                json.dumps(
                    [
                        {
                            "experiment_id": "exp_history_001",
                            "pattern_id": "pattern_history_001",
                            "fix_surface": "field_cot",
                            "status": "completed",
                        }
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            Path(paths.upstream_outcomes_path).write_text(
                json.dumps(
                    [
                        {
                            "outcome_id": "outcome_history_001",
                            "experiment_id": "exp_history_001",
                            "status": "success",
                        }
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            trace_payload_path = Path(paths.profile_field_trace_payload_dir) / "case_001.json"
            trace_payload_path.parent.mkdir(parents=True, exist_ok=True)
            trace_payload_path.write_text(
                json.dumps(
                    {
                        "comparison_result": {"grade": "mismatch"},
                        "final_before_backflow": {"value": "college_student"},
                        "final": {"value": "master_student"},
                        "tool_trace": {"stats_bundle": {"support_count": 2}},
                        "llm_batch_debug": [{"raw_response_preview": "{}"}],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            case_fact = CaseFact(
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
                auto_confidence=0.81,
                accuracy_gap_status="open",
                resolution_route="strategy_fix",
                comparison_grade="mismatch",
                trace_payload_path=str(trace_payload_path),
                comparison_result={"grade": "mismatch"},
                evidence_refs=[{"source_type": "event", "source_id": "EVT_001"}],
            )

            packet = BadcasePacketAssembler(project_root=tmpdir).assemble(case_fact)

        self.assertEqual(packet["case_fact"]["case_id"], "case_001")
        self.assertEqual(packet["trace_payload"]["final_before_backflow"]["value"], "college_student")
        self.assertEqual(packet["history_patterns"][0]["pattern_id"], "pattern_history_001")
        self.assertEqual(packet["history_experiments"][0]["experiment_id"], "exp_history_001")
        self.assertEqual(packet["history_outcomes"][0]["outcome_id"], "outcome_history_001")

    def test_badcase_packet_assembler_normalizes_dict_shaped_llm_batch_debug(self) -> None:
        from services.reflection import build_reflection_asset_paths
        from services.reflection.types import CaseFact
        from services.reflection.upstream_agent import BadcasePacketAssembler

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_reflection_asset_paths(project_root=tmpdir, user_name="vigar")
            Path(paths.root_dir).mkdir(parents=True, exist_ok=True)
            for file_path in (
                paths.upstream_patterns_path,
                paths.upstream_experiments_path,
                paths.upstream_outcomes_path,
            ):
                Path(file_path).write_text("[]", encoding="utf-8")

            trace_payload_path = Path(paths.profile_field_trace_payload_dir) / "case_dict_debug_001.json"
            trace_payload_path.parent.mkdir(parents=True, exist_ok=True)
            trace_payload_path.write_text(
                json.dumps(
                    {
                        "llm_batch_debug": {
                            "batch_name": "Semantic Expression::batch_2",
                            "status": "ok",
                            "raw_response_preview": "preview",
                        }
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            case_fact = CaseFact(
                case_id="case_dict_debug_001",
                user_name="vigar",
                album_id="album_001",
                entity_type="profile_field",
                entity_id="short_term_expression.motivation_shift",
                dimension="short_term_expression.motivation_shift",
                signal_source="mainline_profile",
                first_seen_stage="lp3",
                surfaced_stage="lp3",
                routing_result="strategy_candidate",
                business_priority="high",
                auto_confidence=0.7,
                accuracy_gap_status="open",
                resolution_route="strategy_fix",
                comparison_grade="mismatch",
                trace_payload_path=str(trace_payload_path),
            )

            packet = BadcasePacketAssembler(project_root=tmpdir).assemble(case_fact)

        self.assertIsInstance(packet["llm_batch_debug"], list)
        self.assertEqual(packet["llm_batch_debug"][0]["batch_name"], "Semantic Expression::batch_2")

    def test_badcase_packet_assembler_trace_diagnose_returns_compact_diagnostic_card(self) -> None:
        from services.reflection import build_reflection_asset_paths
        from services.reflection.types import CaseFact
        from services.reflection.upstream_agent import BadcasePacketAssembler

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_reflection_asset_paths(project_root=tmpdir, user_name="vigar")
            Path(paths.root_dir).mkdir(parents=True, exist_ok=True)
            trace_payload_path = Path(paths.profile_field_trace_payload_dir) / "case_diag_001.json"
            trace_payload_path.parent.mkdir(parents=True, exist_ok=True)
            trace_payload_path.write_text(
                json.dumps(
                    {
                        "final_before_backflow": {
                            "value": "college_student",
                            "reasoning": "连续校园和课堂证据",
                        },
                        "final": {
                            "value": "college_student",
                            "reasoning": "连续校园和课堂证据",
                            "evidence": {
                                "supporting_refs": [
                                    {"source_id": "EVT_001", "description": "校园课堂"},
                                    {"source_id": "EVT_002", "description": "毕业服"},
                                ],
                                "contradicting_refs": [
                                    {"source_id": "EVT_003", "description": "家庭聚餐"},
                                ],
                            },
                        },
                        "tool_trace": {
                            "evidence_bundle": {
                                "supporting_refs": {
                                    "event": [
                                        {"source_id": "EVT_001", "description": "校园课堂"},
                                        {"source_id": "EVT_002", "description": "毕业服"},
                                    ],
                                },
                                "contradicting_refs": {
                                    "event": [
                                        {"source_id": "EVT_003", "description": "家庭聚餐"},
                                    ],
                                },
                            }
                        },
                        "llm_batch_debug": {
                            "batch_name": "Identity Facts::batch_1",
                            "raw_response_preview": '{"value":"college_student"}',
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            case_fact = CaseFact(
                case_id="case_diag_001",
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
                auto_confidence=0.81,
                accuracy_gap_status="open",
                resolution_route="strategy_fix",
                comparison_grade="mismatch",
                comparison_result={"grade": "mismatch", "output_value": "college_student", "gt_value": "student"},
                trace_payload_path=str(trace_payload_path),
            )

            packet = BadcasePacketAssembler(project_root=tmpdir).assemble(case_fact)
            diagnostic = BadcasePacketAssembler(project_root=tmpdir).trace_diagnose(packet)

        self.assertEqual(diagnostic["field_key"], "long_term_facts.social_identity.education")
        self.assertEqual(diagnostic["tool_called"], True)
        self.assertEqual(diagnostic["retrieval_hit_count"], 2)
        self.assertEqual(len(diagnostic["top_supporting_refs"]), 2)
        self.assertIn("hits_exist_but_semantic_mapping_suspect", diagnostic["diagnostic_flags"])

    def test_badcase_packet_assembler_history_recall_returns_top_relevant_history(self) -> None:
        from services.reflection import build_reflection_asset_paths
        from services.reflection.types import CaseFact
        from services.reflection.upstream_agent import BadcasePacketAssembler

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_reflection_asset_paths(project_root=tmpdir, user_name="vigar")
            Path(paths.root_dir).mkdir(parents=True, exist_ok=True)
            Path(paths.upstream_patterns_path).write_text(
                json.dumps(
                    [
                        {
                            "pattern_id": "pattern_edu_success",
                            "dimension": "long_term_facts.social_identity.education",
                            "root_cause_candidates": ["field_reasoning"],
                            "recommended_option": "field_cot",
                            "summary": "education 字段曾通过 COT 修好",
                        },
                        {
                            "pattern_id": "pattern_edu_fail",
                            "dimension": "long_term_facts.social_identity.education",
                            "root_cause_candidates": ["tool_retrieval"],
                            "recommended_option": "tool_rule",
                            "summary": "education 字段曾尝试改 tool_rule",
                        },
                        {
                            "pattern_id": "pattern_other",
                            "dimension": "short_term_expression.motivation_shift",
                            "root_cause_candidates": ["field_reasoning"],
                            "recommended_option": "field_cot",
                            "summary": "别的字段",
                        },
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            Path(paths.upstream_experiments_path).write_text(
                json.dumps(
                    [
                        {"experiment_id": "exp_ok", "pattern_id": "pattern_edu_success", "fix_surface": "field_cot", "status": "completed"},
                        {"experiment_id": "exp_bad", "pattern_id": "pattern_edu_fail", "fix_surface": "tool_rule", "status": "completed"},
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            Path(paths.upstream_outcomes_path).write_text(
                json.dumps(
                    [
                        {"outcome_id": "out_ok", "experiment_id": "exp_ok", "status": "success"},
                        {"outcome_id": "out_bad", "experiment_id": "exp_bad", "status": "failed"},
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            case_fact = CaseFact(
                case_id="case_history_001",
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
                auto_confidence=0.81,
                accuracy_gap_status="open",
                resolution_route="strategy_fix",
                comparison_grade="mismatch",
            )

            packet = BadcasePacketAssembler(project_root=tmpdir).assemble(case_fact)
            history = BadcasePacketAssembler(project_root=tmpdir).history_recall(
                packet,
                root_cause_candidates=["field_reasoning"],
                fix_surface_candidates=["field_cot", "tool_rule"],
            )

        self.assertEqual(history["similar_pattern_count"], 2)
        self.assertEqual(history["same_field_successful_surfaces"], ["field_cot"])
        self.assertEqual(history["same_field_failed_surfaces"], ["tool_rule"])
        self.assertEqual(history["recommended_surface_prior"], "field_cot")
        self.assertEqual(len(history["history_patterns"]), 2)

    def test_badcase_packet_assembler_history_recall_includes_human_feedback_summary(self) -> None:
        from services.reflection import build_reflection_asset_paths
        from services.reflection.types import CaseFact
        from services.reflection.upstream_agent import BadcasePacketAssembler

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = build_reflection_asset_paths(project_root=tmpdir, user_name="vigar")
            Path(paths.root_dir).mkdir(parents=True, exist_ok=True)
            Path(paths.reflection_feedback_path).write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "feedback_id": "feedback_approve_001",
                                "field_key": "short_term_expression.motivation_shift",
                                "recommended_fix_surface": "field_cot",
                                "human_action": "proposal_approve",
                                "reviewer_note": "这个方向对，继续推进。",
                            },
                            ensure_ascii=False,
                        ),
                        json.dumps(
                            {
                                "feedback_id": "feedback_revision_001",
                                "field_key": "short_term_expression.motivation_shift",
                                "recommended_fix_surface": "field_cot",
                                "human_action": "proposal_need_revision",
                                "reviewer_note": "方向基本对，但表达还不够准确。",
                            },
                            ensure_ascii=False,
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            case_fact = CaseFact(
                case_id="case_feedback_001",
                user_name="vigar",
                album_id="album_001",
                entity_type="profile_field",
                entity_id="short_term_expression.motivation_shift",
                dimension="short_term_expression.motivation_shift",
                signal_source="mainline_profile",
                first_seen_stage="lp3",
                surfaced_stage="lp3",
                routing_result="strategy_candidate",
                business_priority="high",
                auto_confidence=0.81,
                accuracy_gap_status="open",
                resolution_route="strategy_fix",
                comparison_grade="mismatch",
            )

            packet = BadcasePacketAssembler(project_root=tmpdir).assemble(case_fact)
            history = BadcasePacketAssembler(project_root=tmpdir).history_recall(
                packet,
                root_cause_candidates=["field_reasoning"],
                fix_surface_candidates=["field_cot"],
            )

        self.assertEqual(history["feedback_summary"]["approve_count"], 1)
        self.assertEqual(history["feedback_summary"]["need_revision_count"], 1)
        self.assertIn("这个方向对", history["recent_feedback_notes"][0]["reviewer_note"])

    def test_upstream_reflection_agent_uses_trace_and_history_tools_when_model_requests_them(self) -> None:
        from services.reflection.upstream_agent import UpstreamReflectionAgent

        class FakeLLM:
            def __init__(self) -> None:
                self.prompts: list[str] = []
                self.calls = 0

            def _call_llm_via_official_api(self, prompt: str, **_: object) -> dict:
                self.prompts.append(prompt)
                self.calls += 1
                if self.calls == 1:
                    return {
                        "tool_requests": [
                            {"tool_name": "trace_diagnose"},
                            {
                                "tool_name": "history_recall",
                                "arguments": {
                                    "root_cause_candidates": ["field_reasoning"],
                                    "fix_surface_candidates": ["field_cot", "tool_rule"],
                                },
                            },
                        ]
                    }
                return {
                    "root_cause_family": "field_reasoning",
                    "recommended_fix_surface": "field_cot",
                    "confidence": 0.92,
                    "judgment_summary_zh": "这条 badcase 更像字段归纳问题，建议先改 field_cot。",
                    "key_evidence_zh": [
                        "GT 要的是学生，当前输出已经更像学历层级归纳偏差。",
                        "trace 显示关键校园证据已经命中，不像是没取到证据。",
                    ],
                    "why_this_surface_zh": "因为问题主要发生在字段语义归纳，而不是证据召回。",
                    "why_not_other_surfaces_zh": "历史上 field_cot 成功，tool_rule 曾失败。",
                    "patch_intent": {
                        "field_key": "long_term_facts.social_identity.education",
                        "fix_surface": "field_cot",
                        "change_summary_zh": "收窄 education 的字段归纳口径，更优先输出学生层级。",
                    },
                }

        packet = {
            "case_fact": {
                "case_id": "case_001",
                "user_name": "vigar",
                "album_id": "album_001",
                "entity_type": "profile_field",
                "dimension": "long_term_facts.social_identity.education",
                "comparison_grade": "mismatch",
                "accuracy_gap_status": "open",
                "causality_route": "",
                "root_cause_family": "field_reasoning",
                "fix_surface_confidence": 0.8,
                "decision_trace": {"reason": "校园证据稳定"},
            },
            "comparison_result": {"grade": "mismatch", "output_value": "college_student", "gt_value": "student"},
            "pre_audit_comparison_result": {"grade": "mismatch"},
            "final_before_backflow": {"value": "college_student"},
            "final_after_backflow": {"value": "college_student"},
            "null_reason": "",
            "tool_trace": {
                "evidence_bundle": {
                    "supporting_refs": {"event": [{"source_id": "EVT_001", "description": "校园课堂"}]},
                }
            },
            "llm_batch_debug": {"batch_name": "Identity Facts::batch_1", "raw_response_preview": '{"value":"college_student"}'},
            "history_patterns": [
                {
                    "pattern_id": "pattern_edu_success",
                    "dimension": "long_term_facts.social_identity.education",
                    "root_cause_candidates": ["field_reasoning"],
                    "recommended_option": "field_cot",
                    "summary": "education 字段曾通过 COT 修好",
                }
            ],
            "history_experiments": [
                {"experiment_id": "exp_ok", "pattern_id": "pattern_edu_success", "fix_surface": "field_cot", "status": "completed"},
            ],
            "history_outcomes": [
                {"outcome_id": "out_ok", "experiment_id": "exp_ok", "status": "success"},
            ],
        }

        llm = FakeLLM()
        result = UpstreamReflectionAgent(llm_processor=llm).reflect(packet)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["recommended_fix_surface"], "field_cot")
        self.assertIn("字段归纳问题", result["judgment_summary_zh"])
        self.assertEqual(llm.calls, 2)
        self.assertIn('"trace_diagnosis"', llm.prompts[1])
        self.assertIn('"history_recall"', llm.prompts[1])

    def test_upstream_reflection_agent_normalizes_string_patch_intent_and_metric_gain(self) -> None:
        from services.reflection.upstream_agent import UpstreamReflectionAgent

        class FakeLLM:
            def _call_llm_via_official_api(self, prompt: str, **_: object) -> dict:
                return {
                    "root_cause_family": "field_reasoning",
                    "recommended_fix_surface": "field_cot",
                    "confidence": 0.85,
                    "judgment_summary_zh": "这条 badcase 更像字段语义归纳问题。",
                    "key_evidence_zh": [
                        "GT 要的是同步性，当前输出语义方向跑偏。",
                        "没有迹象表明关键证据缺失。",
                    ],
                    "why_this_surface_zh": "因为该字段已经拿到了关键线索，但归纳口径偏了。",
                    "why_not_other_surfaces_zh": "不是 tool 问题，也不是策略调用问题。",
                    "patch_intent": "把 motivation_shift 的字段级 COT 调整为优先判断同步性。",
                }

        packet = {
            "case_fact": {
                "case_id": "case_001",
                "dimension": "short_term_expression.motivation_shift",
                "entity_id": "short_term_expression.motivation_shift",
                "comparison_grade": "mismatch",
                "accuracy_gap_status": "open",
                "causality_route": "",
                "root_cause_family": "field_reasoning",
                "fix_surface_confidence": 0.8,
                "decision_trace": {"reason": "GT 是同步性"},
            },
            "comparison_result": {"grade": "mismatch"},
            "pre_audit_comparison_result": {"grade": "mismatch"},
            "final_before_backflow": {},
            "final_after_backflow": {},
            "tool_trace": {},
            "llm_batch_debug": [],
            "history_patterns": [],
            "history_experiments": [],
            "history_outcomes": [],
        }

        result = UpstreamReflectionAgent(llm_processor=FakeLLM()).reflect(packet)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["patch_intent"]["field_key"], "short_term_expression.motivation_shift")
        self.assertEqual(result["patch_intent"]["fix_surface"], "field_cot")
        self.assertIn("同步性", result["patch_intent"]["change_summary_zh"])
        self.assertEqual(result["judgment_summary_zh"], "这条 badcase 更像字段语义归纳问题。")

    def test_proposal_builder_emits_proposal_review_when_experiment_improves(self) -> None:
        from services.reflection.types import PatternCluster, StrategyExperiment
        from services.reflection.upstream_agent import ProposalBuilder

        pattern = PatternCluster(
            pattern_id="pattern_001",
            user_name="vigar",
            lane="upstream",
            business_priority="high",
            root_cause_candidates=["field_reasoning"],
            fix_surface_candidates=["field_cot", "tool_rule", "call_policy"],
            support_case_ids=["case_001", "case_002"],
            is_direction_clear=True,
            support_count=2,
            summary="education 字段反复误判",
            recommended_option="field_cot",
            eligible_for_task=True,
            resolution_route="strategy_fix",
            dimension="long_term_facts.social_identity.education",
            entity_type="profile_field",
        )
        experiment = StrategyExperiment(
            experiment_id="exp_001",
            pattern_id="pattern_001",
            user_name="vigar",
            lane="upstream",
            fix_surface="field_cot",
            change_scope="single_field_single_surface",
            hypothesis="补 education 字段 COT 后会减少 GT mismatch",
            status="completed",
            override_bundle_path="/tmp/overlay.json",
            experiment_report_path="/tmp/report.json",
            proposal_status="ready_for_review",
            approval_required=True,
        )
        agent_result = {
            "root_cause_family": "field_reasoning",
            "recommended_fix_surface": "field_cot",
            "confidence": 0.91,
            "reason": "tool 已命中且证据充分，主要问题出在字段归纳口径",
            "why_not_other_surfaces": "没有看到明显的工具召回缺失，也没有调用策略缺口",
            "decision_tree_path": [
                "accuracy_gap_open",
                "mismatch",
                "field_reasoning",
                "field_cot",
            ],
            "patch_intent": {
                "field_key": "long_term_facts.social_identity.education",
                "field_spec_overrides": {
                    "long_term_facts.social_identity.education": {
                        "cot_steps": ["新增 education 专项归纳步骤"],
                    }
                },
            },
            "expected_metric_gain": {
                "exact_or_close_delta": 1,
            },
        }
        experiment_report = {
            "status": "completed",
            "baseline_metrics": {"exact_or_close_count": 0, "mismatch_count": 2},
            "candidate_metrics": {"exact_or_close_count": 2, "mismatch_count": 0},
            "metric_gain": {"exact_or_close_delta": 2, "mismatch_delta": -2},
            "is_significant_improvement": True,
            "patch_preview": {
                "field_spec_overrides": {
                    "long_term_facts.social_identity.education": {
                        "cot_steps": ["新增 education 专项归纳步骤"],
                    }
                }
            },
            "diff_summary": ["education 字段新增一条 campus-to-degree 归纳步骤"],
        }

        proposal = ProposalBuilder().build(
            pattern=pattern,
            experiment=experiment,
            agent_result=agent_result,
            experiment_report=experiment_report,
        )

        self.assertIsNotNone(proposal)
        self.assertEqual(proposal["proposal"]["status"], "pending_review")
        self.assertEqual(proposal["task"].task_type, "proposal_review")
        self.assertEqual(proposal["task"].recommended_option, "approve")
        self.assertIn("campus-to-degree", proposal["proposal"]["diff_summary"][0])

    def test_proposal_builder_marks_overlay_direct_apply_when_current_badcase_reaches_close_match(self) -> None:
        from services.reflection.types import CaseFact, PatternCluster, StrategyExperiment
        from services.reflection.upstream_agent import ProposalBuilder

        pattern = PatternCluster(
            pattern_id="pattern_overlay_001",
            user_name="vigar",
            lane="upstream",
            business_priority="high",
            root_cause_candidates=["field_reasoning"],
            fix_surface_candidates=["field_cot"],
            support_case_ids=["case_gt_001"],
            is_direction_clear=True,
            support_count=1,
            summary="临时人设字段需要修正",
            recommended_option="field_cot",
            eligible_for_task=True,
            resolution_route="strategy_fix",
            dimension="short_term_expression.motivation_shift",
            entity_type="profile_field",
        )
        support_case = CaseFact(
            case_id="case_gt_001",
            user_name="vigar",
            album_id="album_001",
            entity_type="profile_field",
            entity_id="short_term_expression.motivation_shift",
            dimension="short_term_expression.motivation_shift",
            signal_source="mainline_profile",
            first_seen_stage="lp3",
            surfaced_stage="lp3",
            routing_result="strategy_candidate",
            business_priority="high",
            auto_confidence=0.88,
            comparison_grade="mismatch",
            accuracy_gap_status="open",
            resolution_route="strategy_fix",
            gt_payload={"gt_value": "同步性"},
            upstream_output={"value": "由兴趣导向向高端体验消费偏移"},
        )
        experiment = StrategyExperiment(
            experiment_id="exp_overlay_001",
            pattern_id="pattern_overlay_001",
            user_name="vigar",
            lane="upstream",
            fix_surface="field_cot",
            change_scope="single_field_single_surface",
            hypothesis="收窄临时人设字段归纳口径",
            status="completed",
            override_bundle_path="/tmp/overlay.json",
            experiment_report_path="/tmp/report.json",
            proposal_status="ready_for_review",
            approval_required=True,
        )
        proposal = ProposalBuilder().build(
            pattern=pattern,
            experiment=experiment,
            agent_result={
                "reason": "当前字段把状态同步误判成消费倾向迁移。",
                "why_not_other_surfaces": "关键线索已经存在，不像 tool 召回问题。",
                "decision_tree_path": ["accuracy_gap_open", "mismatch", "field_reasoning", "field_cot"],
                "patch_intent": {
                    "field_key": "short_term_expression.motivation_shift",
                    "field_spec_overrides": {
                        "short_term_expression.motivation_shift": {
                            "cot_steps": ["把临时人设的判断口径收窄到同步/一致状态。"]
                        }
                    },
                },
            },
            experiment_report={
                "status": "completed",
                "candidate_metrics": {"comparison_grade": "close_match"},
                "current_case_result": {
                    "gt_value": "同步性",
                    "current_output": "由兴趣导向向高端体验消费偏移",
                    "candidate_output": "同步性",
                    "candidate_comparison_grade": "close_match",
                },
                "patch_preview": {
                    "field_spec_overrides": {
                        "short_term_expression.motivation_shift": {
                            "cot_steps": ["把临时人设的判断口径收窄到同步/一致状态。"]
                        }
                    }
                },
                "diff_summary": ["临时人设字段新增“同步/一致性”优先判断步骤"],
            },
            support_cases=[support_case],
        )

        self.assertEqual(proposal["proposal"]["execution_path_recommendation"], "overlay_direct_apply")
        self.assertEqual(proposal["proposal"]["gt_value"], "同步性")
        self.assertEqual(proposal["proposal"]["current_output"], "由兴趣导向向高端体验消费偏移")
        self.assertEqual(proposal["proposal"]["candidate_output"], "同步性")

    def test_memory_engineer_agent_builds_direct_and_rewrite_change_requests(self) -> None:
        from services.reflection.upstream_agent import MemoryEngineerAgent

        agent = MemoryEngineerAgent(provider="disabled")
        direct = agent.build_change_request(
            proposal={
                "proposal_id": "proposal_direct_001",
                "experiment_id": "exp_direct_001",
                "pattern_id": "pattern_direct_001",
                "user_name": "vigar",
                "field_key": "short_term_expression.motivation_shift",
                "fix_surface": "field_cot",
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
            reviewer_note="",
        )
        rewrite = agent.build_change_request(
            proposal={
                "proposal_id": "proposal_rewrite_001",
                "experiment_id": "exp_rewrite_001",
                "pattern_id": "pattern_rewrite_001",
                "user_name": "vigar",
                "field_key": "short_term_expression.motivation_shift",
                "fix_surface": "field_cot",
                "execution_path_recommendation": "engineer_rewrite_apply",
                "patch_intent": {"field_key": "short_term_expression.motivation_shift"},
                "gt_value": "同步性",
                "current_output": "由兴趣导向向高端体验消费偏移",
            },
            reviewer_note="把口径收窄到同步/一致状态，不要外推成消费倾向变化。",
        )

        self.assertEqual(direct["execution_path"], "overlay_direct_apply")
        self.assertEqual(
            direct["patch_preview"]["field_spec_overrides"]["short_term_expression.motivation_shift"]["cot_steps"][0],
            "优先判断同步/一致状态。",
        )
        self.assertEqual(rewrite["execution_path"], "engineer_rewrite_apply")
        self.assertIn("同步/一致状态", rewrite["change_summary_zh"])
        self.assertIn("同步性", rewrite["short_reason_zh"])

    def test_memory_engineer_agent_uses_llm_prompt_for_feishu_friendly_summary(self) -> None:
        from services.reflection.upstream_agent import MemoryEngineerAgent

        class FakeLLM:
            def __init__(self) -> None:
                self.prompts: list[str] = []

            def _call_llm_via_official_api(self, prompt: str, **_: object) -> dict:
                self.prompts.append(prompt)
                return {
                    "execution_path": "engineer_rewrite_apply",
                    "change_summary_zh": "把 short_term_expression.motivation_shift 的 field_cot 收窄到同步/一致状态判断。",
                    "short_reason_zh": "因为 GT 要的是同步性，当前输出语义方向跑偏。",
                    "target_files": ["rule_assets/field_specs.overrides.json"],
                    "patch_plan": {
                        "field_key": "short_term_expression.motivation_shift",
                        "fix_surface": "field_cot",
                        "change_intent_zh": "收窄字段归纳口径。",
                    },
                }

        llm = FakeLLM()
        change_request = MemoryEngineerAgent(llm_processor=llm).build_change_request(
            proposal={
                "proposal_id": "proposal_llm_001",
                "experiment_id": "exp_llm_001",
                "pattern_id": "pattern_llm_001",
                "user_name": "vigar",
                "field_key": "short_term_expression.motivation_shift",
                "fix_surface": "field_cot",
                "execution_path_recommendation": "engineer_rewrite_apply",
                "patch_intent": {"field_key": "short_term_expression.motivation_shift"},
                "gt_value": "同步性",
                "current_output": "由兴趣导向向高端体验消费偏移",
            },
            reviewer_note="不要外推成消费倾向迁移。",
        )

        self.assertEqual(change_request["execution_path"], "engineer_rewrite_apply")
        self.assertIn("同步/一致状态判断", change_request["change_summary_zh"])
        self.assertIn("同步性", change_request["short_reason_zh"])
        self.assertEqual(len(llm.prompts), 1)
        self.assertIn("你的输出会直接进入飞书审批", llm.prompts[0])
        self.assertIn("一句改动说明", llm.prompts[0])
        self.assertIn("通俗易懂的人话", llm.prompts[0])
        self.assertIn("不要解释术语", llm.prompts[0])

    def test_reflection_prompt_uses_compact_packet_summary_for_large_trace_payload(self) -> None:
        from services.reflection.upstream_agent import UpstreamReflectionAgent

        large_refs = [
            {
                "source_type": "event",
                "source_id": f"EVT_{index:04d}",
                "description": "x" * 400,
            }
            for index in range(1200)
        ]
        packet = {
            "case_fact": {
                "case_id": "case_big_001",
                "dimension": "short_term_expression.motivation_shift",
                "comparison_grade": "mismatch",
            },
            "comparison_result": {
                "grade": "mismatch",
                "score": 0.0,
                "output_value": "由兴趣导向向高端体验消费偏移",
                "gt_value": "同步性",
            },
            "pre_audit_comparison_result": {
                "grade": "mismatch",
                "score": 0.0,
            },
            "final_before_backflow": {
                "value": "由兴趣导向向高端体验消费偏移",
                "reasoning": "早期兴趣活动 + 后期护肤消费",
                "evidence": {"supporting_refs": large_refs},
            },
            "final_after_backflow": {
                "value": "由兴趣导向向高端体验消费偏移",
                "reasoning": "早期兴趣活动 + 后期护肤消费",
                "evidence": {"supporting_refs": large_refs},
            },
            "tool_trace": {
                "evidence_bundle": {
                    "supporting_refs": {
                        "events": large_refs,
                    }
                }
            },
            "llm_batch_debug": [
                {
                    "batch_name": "short_term_expression",
                    "raw_response_preview": "y" * 5000,
                }
            ],
            "history_patterns": [{"pattern_id": f"pattern_{i}", "summary": "z" * 300} for i in range(20)],
            "history_experiments": [{"experiment_id": f"exp_{i}", "status": "completed"} for i in range(20)],
            "history_outcomes": [{"outcome_id": f"out_{i}", "status": "success"} for i in range(20)],
        }

        prompt = UpstreamReflectionAgent(llm_processor=None)._build_prompt(packet)

        self.assertLess(len(prompt), 50000)
        self.assertIn('"supporting_ref_count"', prompt)
        self.assertIn('"history_pattern_count"', prompt)
        self.assertNotIn("EVT_1199", prompt)
        self.assertNotIn("y" * 1000, prompt)

    def test_reflection_prompt_accepts_dict_shaped_llm_batch_debug(self) -> None:
        from services.reflection.upstream_agent import UpstreamReflectionAgent

        packet = {
            "case_fact": {
                "case_id": "case_dict_debug_001",
                "dimension": "short_term_expression.motivation_shift",
                "comparison_grade": "mismatch",
            },
            "comparison_result": {"grade": "mismatch"},
            "pre_audit_comparison_result": {"grade": "mismatch"},
            "final_before_backflow": {},
            "final_after_backflow": {},
            "tool_trace": {},
            "llm_batch_debug": {
                "batch_name": "Semantic Expression::batch_2",
                "status": "ok",
                "raw_response_preview": "abc" * 100,
            },
        }

        prompt = UpstreamReflectionAgent(llm_processor=None)._build_prompt(packet)

        self.assertIn("Semantic Expression::batch_2", prompt)
        self.assertIn('"llm_batch_debug_summary"', prompt)

    def test_reflection_prompt_requires_simplified_chinese_output(self) -> None:
        from services.reflection.upstream_agent import UpstreamReflectionAgent

        packet = {
            "case_fact": {
                "case_id": "case_lang_001",
                "dimension": "short_term_expression.motivation_shift",
                "comparison_grade": "mismatch",
            },
            "comparison_result": {"grade": "mismatch"},
            "pre_audit_comparison_result": {"grade": "mismatch"},
            "final_before_backflow": {},
            "final_after_backflow": {},
            "tool_trace": {},
            "llm_batch_debug": [],
        }

        prompt = UpstreamReflectionAgent(llm_processor=None)._build_prompt(packet)

        self.assertIn("简体中文", prompt)
        self.assertIn("你是一个擅长找线索、分析严谨的反思 Agent", prompt)
        self.assertIn("你的背景目标", prompt)
        self.assertIn("你定位 badcase 时，必须按这个顺序思考", prompt)
        self.assertIn("什么时候用 tool", prompt)
        self.assertIn("judgment_summary_zh", prompt)
        self.assertIn("key_evidence_zh", prompt)
        self.assertIn("why_this_surface_zh", prompt)
        self.assertIn("why_not_other_surfaces_zh", prompt)
        self.assertIn("通俗易懂的人话", prompt)
        self.assertIn("不要解释术语", prompt)

    def test_mutation_executor_refuses_unapproved_proposal(self) -> None:
        from services.memory_pipeline.rule_asset_loader import ensure_rule_asset_files
        from services.reflection.upstream_agent import MutationExecutor

        with tempfile.TemporaryDirectory() as tmpdir:
            ensure_rule_asset_files(project_root=tmpdir)
            result = MutationExecutor(project_root=tmpdir).execute(
                proposal={
                    "proposal_id": "proposal_001",
                    "user_name": "vigar",
                    "status": "pending_review",
                    "fix_surface": "field_cot",
                    "patch_preview": {
                        "field_spec_overrides": {
                            "long_term_facts.social_identity.education": {
                                "cot_steps": ["未审批前不应落库"],
                            }
                        }
                    },
                }
            )
            rule_asset_path = Path(tmpdir) / "services" / "memory_pipeline" / "rule_assets" / "field_specs.overrides.json"
            written_payload = json.loads(rule_asset_path.read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "blocked_unapproved")
        self.assertEqual(written_payload, {})

    def test_upstream_reflection_agent_prefers_reflection_specific_openrouter_key(self) -> None:
        import services.reflection.upstream_agent as upstream_agent

        with patch.object(upstream_agent, "REFLECTION_AGENT_OPENROUTER_API_KEY", "sk-test-reflection"), \
             patch.object(upstream_agent, "OPENROUTER_API_KEY", "sk-test-default"), \
             patch.object(upstream_agent, "OpenRouterProfileLLMProcessor") as mock_processor:
            upstream_agent.UpstreamReflectionAgent(
                llm_processor=None,
                provider="openrouter",
                model="google/gemini-3.1-flash-lite-preview",
            )

        self.assertEqual(mock_processor.call_args.kwargs["api_key"], "sk-test-reflection")
