from __future__ import annotations

import json
import tempfile
import unittest


class ReflectionSkeletonTests(unittest.TestCase):
    def test_build_reflection_asset_paths_are_user_scoped(self) -> None:
        from services.reflection import build_reflection_asset_paths

        paths = build_reflection_asset_paths(
            project_root="/tmp/memory-engineering",
            user_name="vigar",
        )

        self.assertEqual(paths.root_dir, "/tmp/memory-engineering/memory/reflection")
        self.assertEqual(
            paths.observation_cases_path,
            "/tmp/memory-engineering/memory/reflection/observation_cases_vigar.jsonl",
        )
        self.assertEqual(paths.case_facts_path, "/tmp/memory-engineering/memory/reflection/case_facts_vigar.jsonl")
        self.assertEqual(
            paths.decision_review_items_path,
            "/tmp/memory-engineering/memory/reflection/decision_review_items_vigar.json",
        )
        self.assertEqual(
            paths.tasks_path,
            "/tmp/memory-engineering/memory/reflection/tasks_vigar.jsonl",
        )
        self.assertEqual(
            paths.task_actions_path,
            "/tmp/memory-engineering/memory/reflection/task_actions_vigar.jsonl",
        )
        self.assertEqual(
            paths.profile_field_gt_path,
            "/tmp/memory-engineering/memory/reflection/profile_field_gt_vigar.jsonl",
        )
        self.assertEqual(
            paths.gt_comparisons_path,
            "/tmp/memory-engineering/memory/reflection/gt_comparisons_vigar.jsonl",
        )
        self.assertEqual(
            paths.profile_field_trace_index_path,
            "/tmp/memory-engineering/memory/reflection/profile_field_trace_index_vigar.jsonl",
        )
        self.assertEqual(
            paths.profile_field_trace_payload_dir,
            "/tmp/memory-engineering/memory/reflection/profile_field_trace_payloads/vigar",
        )
        self.assertEqual(
            paths.difficult_cases_path,
            "/tmp/memory-engineering/memory/reflection/difficult_cases_vigar.jsonl",
        )
        self.assertEqual(
            paths.difficult_case_actions_path,
            "/tmp/memory-engineering/memory/reflection/difficult_case_actions_vigar.jsonl",
        )
        self.assertEqual(
            paths.engineering_change_requests_path,
            "/tmp/memory-engineering/memory/reflection/engineering_change_requests_vigar.jsonl",
        )
        self.assertEqual(
            paths.reflection_feedback_path,
            "/tmp/memory-engineering/memory/reflection/reflection_feedback_vigar.jsonl",
        )
        self.assertEqual(
            paths.downstream_audit_experiments_path,
            "/tmp/memory-engineering/memory/reflection/downstream_audit_experiments_vigar.json",
        )

    def test_case_fact_serialization_preserves_shared_fact_fields(self) -> None:
        from services.reflection import CaseFact

        fact = CaseFact(
            case_id="case_001",
            user_name="vigar",
            album_id="album_001",
            entity_type="profile_field",
            entity_id="long_term_facts.social_identity.education",
            dimension="profile>education",
            signal_source="lp3",
            first_seen_stage="lp3",
            surfaced_stage="downstream_audit",
            routing_result="strategy_candidate",
            business_priority="high",
            auto_confidence=0.72,
            decision_trace={"judge": "downgrade"},
            tool_usage_summary={"tools_called": ["profile_tool"]},
            upstream_output={"value": "college_student"},
            downstream_challenge={"reason": "evidence weak"},
        )

        payload = fact.to_dict()

        self.assertEqual(payload["case_id"], "case_001")
        self.assertEqual(payload["routing_result"], "strategy_candidate")
        self.assertEqual(payload["business_priority"], "high")
        self.assertEqual(payload["support_count"], 0)
        self.assertEqual(payload["accuracy_gap_status"], "")
        self.assertEqual(payload["resolution_route"], "")
        self.assertEqual(payload["trace_payload_path"], "")
        self.assertEqual(payload["causality_route"], "")
        self.assertEqual(payload["pre_audit_comparison_grade"], "")
        self.assertEqual(payload["decision_trace"]["judge"], "downgrade")
        self.assertEqual(payload["tool_usage_summary"]["tools_called"], ["profile_tool"])

    def test_pattern_cluster_defaults_to_new_status_with_fix_candidates(self) -> None:
        from services.reflection import PatternCluster

        cluster = PatternCluster(
            pattern_id="pattern_001",
            user_name="vigar",
            lane="upstream",
            business_priority="high",
            root_cause_candidates=["field_reasoning"],
            fix_surface_candidates=["field_cot", "tool_rule"],
            support_case_ids=["case_001", "case_002"],
            is_direction_clear=False,
        )

        payload = cluster.to_dict()

        self.assertEqual(payload["status"], "new")
        self.assertFalse(payload["is_direction_clear"])
        self.assertEqual(payload["support_case_ids"], ["case_001", "case_002"])
        self.assertEqual(payload["fix_surface_candidates"], ["field_cot", "tool_rule"])

    def test_persist_downstream_audit_reflection_assets_writes_user_scoped_jsonl(self) -> None:
        from services.reflection import persist_downstream_audit_reflection_assets

        downstream_audit_report = {
            "profile": {
                "extractor_output": {
                    "tags": [
                        {
                            "dimension": "long_term_facts.social_identity.education",
                            "value": "college_student",
                            "evidence": [
                                {
                                    "event_id": "EVT_001",
                                    "photo_id": "PHOTO_001",
                                    "person_id": "Person_001",
                                    "feature_names": ["campus_scene"],
                                    "description": "校园课堂主线",
                                    "evidence_type": "direct",
                                }
                            ],
                        }
                    ]
                },
                "extractor_v2_output": {
                    "tags": [
                        {
                            "dimension": "long_term_facts.social_identity.education",
                            "value": None,
                            "evidence": [
                                {
                                    "event_id": "EVT_001",
                                    "photo_id": "PHOTO_001",
                                    "person_id": "Person_001",
                                    "feature_names": ["campus_scene"],
                                    "description": "校园课堂主线",
                                    "evidence_type": "direct",
                                }
                            ],
                        }
                    ]
                },
                "judge_output": {
                    "decisions": [
                        {
                            "dimension": "long_term_facts.social_identity.education",
                            "verdict": "nullify",
                            "reason": "evidence weak",
                        }
                    ]
                },
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = persist_downstream_audit_reflection_assets(
                downstream_audit_report=downstream_audit_report,
                project_root=tmpdir,
                user_name="vigar",
                album_id="album_001",
            )

            self.assertEqual(result["written_observation_count"], 1)
            self.assertEqual(result["written_case_fact_count"], 1)

            with open(result["observation_cases_path"], "r", encoding="utf-8") as handle:
                observation_payload = json.loads(handle.readline())
            with open(result["case_facts_path"], "r", encoding="utf-8") as handle:
                case_fact_payload = json.loads(handle.readline())

            self.assertEqual(observation_payload["user_name"], "vigar")
            self.assertEqual(observation_payload["entity_type"], "profile_field")
            self.assertEqual(observation_payload["dimension"], "long_term_facts.social_identity.education")
            self.assertEqual(case_fact_payload["routing_result"], "audit_disagreement")
            self.assertEqual(case_fact_payload["business_priority"], "medium")
            self.assertEqual(case_fact_payload["triage_reason"], "downstream_judge_challenged_existing_output")
            self.assertEqual(case_fact_payload["support_count"], 1)
            self.assertEqual(case_fact_payload["downstream_judge"]["verdict"], "nullify")

    def test_persist_downstream_audit_reflection_assets_dedupes_repeat_cases(self) -> None:
        from services.reflection import persist_downstream_audit_reflection_assets

        downstream_audit_report = {
            "metadata": {
                "audit_status": "skipped_init_failure",
                "audit_error_type": "RuntimeError",
                "audit_error_message": "profile agent unavailable",
            }
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            first = persist_downstream_audit_reflection_assets(
                downstream_audit_report=downstream_audit_report,
                project_root=tmpdir,
                user_name="vigar",
                album_id="album_001",
            )
            second = persist_downstream_audit_reflection_assets(
                downstream_audit_report=downstream_audit_report,
                project_root=tmpdir,
                user_name="vigar",
                album_id="album_001",
            )

            self.assertEqual(first["written_case_fact_count"], 1)
            self.assertEqual(second["written_case_fact_count"], 0)

    def test_persist_mainline_reflection_assets_writes_lp1_lp2_lp3_cases(self) -> None:
        from services.reflection import persist_mainline_reflection_assets

        internal_artifacts = {
            "primary_decision": {
                "mode": "person_id",
                "primary_person_id": "Person_001",
                "confidence": 0.93,
                "evidence": {
                    "photo_ids": ["PHOTO_001"],
                    "event_ids": ["EVT_001"],
                    "person_ids": ["Person_001"],
                    "feature_names": ["selfie"],
                },
                "reasoning": "主角稳定出现",
            },
            "relationship_dossiers": [
                {
                    "person_id": "Person_002",
                    "retention_decision": "keep",
                    "retention_reason": "multi_scene_strong_signal",
                    "relationship_result": {
                        "relationship_type": "close_friend",
                        "confidence": 0.74,
                    },
                    "evidence_refs": [
                        {
                            "source_type": "event",
                            "source_id": "EVT_002",
                            "description": "多场景稳定同框",
                            "feature_names": ["private_scene", "selfie_together"],
                        }
                    ],
                }
            ],
            "profile_fact_decisions": [
                {
                    "field_key": "long_term_facts.social_identity.education",
                    "domain_name": "Foundation & Social Identity",
                    "batch_name": "Foundation & Social Identity::batch_1",
                    "field_spec_snapshot": {
                        "field_key": "long_term_facts.social_identity.education",
                        "risk_level": "P0",
                    },
                    "tool_trace": {
                        "evidence_bundle": {
                            "supporting_refs": {
                                "events": [
                                    {
                                        "source_type": "event",
                                        "source_id": "EVT_003",
                                        "description": "校园课堂主线",
                                        "feature_names": ["campus_scene"],
                                    }
                                ]
                            },
                            "ref_index": {},
                            "allowed_refs": {
                                "events": [],
                                "relationships": [],
                                "vlm_observations": [],
                                "group_artifacts": [],
                                "feature_refs": [],
                            },
                            "field_key": "long_term_facts.social_identity.education",
                        },
                        "stats_bundle": {"support_count": 1, "event_count": 1},
                        "ownership_bundle": {"owner": "primary_person"},
                        "counter_bundle": {"contradicting_refs": []},
                    },
                    "draft": {
                        "value": "college_student",
                        "confidence": 0.66,
                    },
                    "final": {
                        "value": "college_student",
                        "confidence": 0.71,
                        "evidence": {
                            "summary": "profile_agent:education",
                            "supporting_refs": [
                                {
                                    "source_type": "event",
                                    "source_id": "EVT_003",
                                    "description": "校园课堂主线",
                                    "feature_names": ["campus_scene"],
                                }
                            ],
                        },
                        "reasoning": "校园与课堂连续出现",
                    },
                    "null_reason": None,
                }
            ],
            "profile_llm_batch_debug": [
                {
                    "batch_name": "Foundation & Social Identity::batch_1",
                    "field_keys": ["long_term_facts.social_identity.education"],
                    "raw_response_preview": "{\"fields\": {\"long_term_facts.social_identity.education\": {\"value\": \"college_student\"}}}",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = persist_mainline_reflection_assets(
                internal_artifacts=internal_artifacts,
                project_root=tmpdir,
                user_name="vigar",
                album_id="album_001",
            )

            self.assertEqual(result["written_observation_count"], 3)
            self.assertEqual(result["written_case_fact_count"], 3)

            with open(result["case_facts_path"], "r", encoding="utf-8") as handle:
                case_facts = [json.loads(line) for line in handle if line.strip()]

            self.assertEqual([fact["first_seen_stage"] for fact in case_facts], ["lp1", "lp2", "lp3"])
            self.assertEqual(case_facts[0]["entity_type"], "primary_person")
            self.assertEqual(case_facts[1]["entity_type"], "relationship_candidate")
            self.assertEqual(case_facts[2]["entity_type"], "profile_field")
            self.assertEqual(case_facts[0]["routing_result"], "strategy_candidate")
            self.assertEqual(case_facts[1]["routing_result"], "strategy_candidate")
            self.assertEqual(case_facts[2]["routing_result"], "strategy_candidate")
            self.assertEqual(case_facts[0]["triage_reason"], "primary_decision_ready_for_patterning")
            self.assertEqual(case_facts[1]["triage_reason"], "relationship_signal_ready_for_patterning")
            self.assertEqual(case_facts[2]["triage_reason"], "profile_field_ready_for_patterning")
            self.assertTrue(result["profile_field_trace_index_path"].endswith("profile_field_trace_index_vigar.jsonl"))
            self.assertEqual(result["written_profile_field_trace_count"], 1)
            self.assertTrue(case_facts[2]["trace_payload_path"].endswith("album_001-mainline_profile-long_term_facts.social_identity.education-field_decision-profile_agent_%7Ceducation.json".replace("%7C", "|")) is False)
            self.assertTrue(case_facts[2]["trace_payload_path"])
            with open(result["profile_field_trace_index_path"], "r", encoding="utf-8") as handle:
                trace_index = json.loads(handle.readline())
            self.assertEqual(trace_index["field_key"], "long_term_facts.social_identity.education")
            self.assertEqual(trace_index["batch_name"], "Foundation & Social Identity::batch_1")
            self.assertTrue(trace_index["tool_called"])
            with open(trace_index["trace_payload_path"], "r", encoding="utf-8") as handle:
                trace_payload = json.load(handle)
            self.assertIn("tool_trace", trace_payload)
            self.assertIn("llm_batch_debug", trace_payload)
            self.assertEqual(trace_payload["final"]["value"], "college_student")

    def test_route_case_fact_promotes_low_evidence_null_field_into_empty_output_candidate(self) -> None:
        from services.reflection import CaseFact, route_case_fact

        fact = CaseFact(
            case_id="case_weak_001",
            user_name="vigar",
            album_id="album_001",
            entity_type="profile_field",
            entity_id="short_term_expression.current_mood",
            dimension="short_term_expression.current_mood",
            signal_source="mainline_profile",
            first_seen_stage="lp3",
            surfaced_stage="lp3",
            routing_result="pending_triage",
            business_priority="medium",
            auto_confidence=0.18,
            upstream_output={"value": None},
            evidence_refs=[],
        )

        routed = route_case_fact(fact)

        self.assertEqual(routed.routing_result, "strategy_candidate")
        self.assertEqual(routed.business_priority, "high")
        self.assertEqual(routed.support_count, 0)
        self.assertEqual(routed.triage_reason, "profile_field_missing_value_candidate")
        self.assertEqual(routed.badcase_source, "empty_output_candidate")
        self.assertEqual(routed.badcase_kind, "missing_value")

    def test_persist_mainline_reflection_assets_keeps_pre_audit_profile_snapshot_for_backflow_field(self) -> None:
        from services.reflection import persist_mainline_reflection_assets

        internal_artifacts = {
            "profile_fact_decisions": [
                {
                    "field_key": "long_term_facts.social_identity.education",
                    "batch_name": "Foundation & Social Identity::batch_2",
                    "field_spec_snapshot": {
                        "field_key": "long_term_facts.social_identity.education",
                    },
                    "tool_trace": {
                        "evidence_bundle": {
                            "supporting_refs": {
                                "events": [
                                    {
                                        "source_type": "event",
                                        "source_id": "EVT_101",
                                        "description": "校园场景",
                                        "feature_names": ["campus_scene"],
                                    }
                                ]
                            }
                        }
                    },
                    "draft": {"value": "college_student", "confidence": 0.64},
                    "final_before_backflow": {
                        "value": "college_student",
                        "confidence": 0.72,
                        "evidence": {
                            "supporting_refs": [
                                {
                                    "source_type": "event",
                                    "source_id": "EVT_101",
                                    "description": "校园场景",
                                    "feature_names": ["campus_scene"],
                                }
                            ]
                        },
                        "reasoning": "回流前判断是大学生",
                    },
                    "final": {
                        "value": None,
                        "confidence": 0.0,
                        "evidence": {
                            "supporting_refs": [
                                {
                                    "source_type": "event",
                                    "source_id": "EVT_101",
                                    "description": "校园场景",
                                    "feature_names": ["campus_scene"],
                                }
                            ],
                            "constraint_notes": ["downstream_judge:nullify:证据不足"],
                        },
                        "reasoning": "下游 Judge 否决",
                    },
                    "backflow": {
                        "verdict": "nullify",
                        "judge_reason": "证据不足",
                        "applied_change": "nullify_field",
                    },
                    "null_reason": "downstream_judge_nullify",
                }
            ],
            "profile_llm_batch_debug": [
                {
                    "batch_name": "Foundation & Social Identity::batch_2",
                    "field_keys": ["long_term_facts.social_identity.education"],
                    "raw_response_preview": "{\"fields\": {\"long_term_facts.social_identity.education\": {\"value\": \"college_student\"}}}",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            result = persist_mainline_reflection_assets(
                internal_artifacts=internal_artifacts,
                project_root=tmpdir,
                user_name="vigar",
                album_id="album_backflow_001",
            )

            with open(result["case_facts_path"], "r", encoding="utf-8") as handle:
                case_fact = json.loads(handle.readline())
            with open(result["profile_field_trace_index_path"], "r", encoding="utf-8") as handle:
                trace_index = json.loads(handle.readline())
            with open(trace_index["trace_payload_path"], "r", encoding="utf-8") as handle:
                trace_payload = json.load(handle)

            self.assertEqual(case_fact["pre_audit_output"]["value"], "college_student")
            self.assertEqual(case_fact["upstream_output"]["value"], None)
            self.assertEqual(case_fact["audit_action_type"], "nullify")
            self.assertEqual(case_fact["downstream_judge"]["verdict"], "nullify")
            self.assertEqual(trace_payload["final_before_backflow"]["value"], "college_student")
            self.assertEqual(trace_payload["backflow"]["verdict"], "nullify")
