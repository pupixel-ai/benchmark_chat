from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


class PrecomputedBundleLoaderNormalizationTests(unittest.TestCase):
    def test_load_precomputed_memory_state_normalizes_bundle_participants_to_canonical_ids(self) -> None:
        from services.memory_pipeline.precomputed_loader import load_precomputed_memory_state

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            (base / "face").mkdir()
            (base / "vlm").mkdir()
            (base / "lp1").mkdir()

            (base / "face" / "face_recognition_output.json").write_text(
                json.dumps(
                    {
                        "primary_person_id": "Person_011",
                        "persons": [
                            {
                                "person_id": "Person_011",
                                "photo_count": 5,
                                "first_seen": "2026-03-01T10:00:00",
                                "last_seen": "2026-03-02T10:00:00",
                            },
                            {
                                "person_id": "Person_001",
                                "photo_count": 3,
                                "first_seen": "2026-03-01T10:00:00",
                                "last_seen": "2026-03-02T10:00:00",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (base / "vlm" / "vp1_observations.json").write_text(
                json.dumps(
                    [
                        {
                            "photo_id": "photo_001",
                            "timestamp": "2026-03-01T10:00:00",
                            "face_person_ids": ["Person_011", "Person_001"],
                            "vlm_analysis": {
                                "summary": "【主角】和 Person_001 一起出行。",
                                "people": [
                                    {"person_id": "Person_011", "contact_type": "no_contact"},
                                    {"person_id": "Person_001", "contact_type": "standing_near"},
                                ],
                                "relations": [],
                                "scene": {"location_detected": "车内"},
                                "event": {"activity": "出行"},
                                "details": [],
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (base / "lp1" / "lp1_events_compact.json").write_text(
                json.dumps(
                    [
                        {
                            "event_id": "EVT_0001",
                            "batch_id": "BATCH_0001",
                            "anchor_photo_id": "photo_001",
                            "supporting_photo_ids": ["photo_001"],
                            "started_at": "2026-03-01T10:00:00",
                            "ended_at": "2026-03-01T11:00:00",
                            "title": "出行",
                            "narrative_synthesis": "【主角】和同行驾驶员一起开车出门。",
                            "participant_person_ids": ["【主角】", "同行驾驶员(Person_001)", "游戏队友"],
                            "depicted_person_ids": ["Person_011", "Person_001"],
                            "place_refs": ["深圳"],
                            "social_dynamics": [
                                {
                                    "target_id": "【主角】",
                                    "interaction_type": "同行",
                                    "social_clue": "一起出发",
                                }
                            ],
                            "persona_evidence": {"behavioral": []},
                            "tags": ["#出行"],
                            "confidence": 0.9,
                            "reason": "test",
                            "meta_info": {"title": "出行", "location_context": "深圳", "photo_count": 1},
                            "objective_fact": {
                                "scene_description": "一起出门",
                                "participants": ["【主角】", "同行驾驶员(Person_001)", "游戏队友"],
                            },
                            "source_temp_event_id": "TEMP_EVT_001",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            state = load_precomputed_memory_state(base)

        event = state.events[0]
        self.assertEqual(event.participants, ["Person_011", "Person_001"])
        self.assertEqual(event.objective_fact["participants"], ["Person_011", "Person_001"])
        self.assertEqual(
            event.meta_info["raw_participants"],
            ["【主角】", "同行驾驶员(Person_001)", "游戏队友"],
        )
        self.assertEqual(
            event.objective_fact["raw_participants"],
            ["【主角】", "同行驾驶员(Person_001)", "游戏队友"],
        )
        self.assertEqual(event.social_dynamics[0]["target_id"], "Person_011")
        self.assertEqual(
            event.meta_info["trace"]["depicted_person_ids"],
            ["Person_011", "Person_001"],
        )


class PrecomputedBundlePipelineRunnerTests(unittest.TestCase):
    def test_run_precomputed_bundle_pipeline_writes_core_outputs(self) -> None:
        from services.memory_pipeline.precomputed_bundle_runner import run_precomputed_bundle_pipeline

        class StubLLMProcessor:
            def __init__(self) -> None:
                self.primary_person_id = "Person_011"

            def _collect_relationship_evidence(self, person_id, vlm_results, events=None):
                if person_id != "Person_001":
                    return {
                        "photo_count": 0,
                        "time_span_days": 0,
                        "recent_gap_days": 0,
                        "scenes": [],
                        "private_scene_ratio": 0.0,
                        "dominant_scene_ratio": 0.0,
                        "interaction_behavior": [],
                        "with_user_only": True,
                        "contact_types": [],
                        "rela_events": [],
                        "monthly_frequency": 0.0,
                        "trend_detail": {},
                        "co_appearing_persons": [],
                        "anomalies": [],
                    }
                return {
                    "photo_count": 4,
                    "time_span_days": 12,
                    "recent_gap_days": 1,
                    "scenes": ["车内", "餐厅"],
                    "private_scene_ratio": 0.3,
                    "dominant_scene_ratio": 0.5,
                    "interaction_behavior": ["聊天"],
                    "with_user_only": True,
                    "contact_types": ["standing_near"],
                    "rela_events": [
                        {
                            "event_id": "EVT_0001",
                            "date": "2026-03-01",
                            "title": "出行",
                            "location": "深圳",
                            "description": "一起出门",
                            "participants": ["Person_011", "Person_001"],
                            "narrative_synthesis": "一起出门",
                            "social_dynamics": [],
                        }
                    ],
                    "monthly_frequency": 2.5,
                    "trend_detail": {"direction": "up"},
                    "co_appearing_persons": [],
                    "anomalies": [],
                }

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir) / "bundle"
            output_dir = Path(temp_dir) / "out"
            (base / "face").mkdir(parents=True)
            (base / "vlm").mkdir()
            (base / "lp1").mkdir()

            (base / "face" / "face_recognition_output.json").write_text(
                json.dumps(
                    {
                        "primary_person_id": "Person_011",
                        "persons": [
                            {"person_id": "Person_011", "photo_count": 5},
                            {"person_id": "Person_001", "photo_count": 4},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (base / "vlm" / "vp1_observations.json").write_text(
                json.dumps(
                    [
                        {
                            "photo_id": "photo_001",
                            "timestamp": "2026-03-01T10:00:00",
                            "face_person_ids": ["Person_011", "Person_001"],
                            "vlm_analysis": {
                                "summary": "【主角】和 Person_001 在车内聊天。",
                                "people": [
                                    {"person_id": "Person_011", "contact_type": "no_contact"},
                                    {"person_id": "Person_001", "contact_type": "standing_near"},
                                ],
                                "relations": [],
                                "scene": {"location_detected": "车内"},
                                "event": {"activity": "出行"},
                                "details": [],
                                "ocr_hits": [],
                                "brands": [],
                                "place_candidates": ["深圳"],
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (base / "lp1" / "lp1_events_compact.json").write_text(
                json.dumps(
                    [
                        {
                            "event_id": "EVT_0001",
                            "batch_id": "BATCH_0001",
                            "anchor_photo_id": "photo_001",
                            "supporting_photo_ids": ["photo_001"],
                            "started_at": "2026-03-01T10:00:00",
                            "ended_at": "2026-03-01T11:00:00",
                            "title": "出行",
                            "narrative_synthesis": "【主角】和同行驾驶员一起出门。",
                            "participant_person_ids": ["【主角】", "同行驾驶员(Person_001)"],
                            "depicted_person_ids": ["Person_011", "Person_001"],
                            "place_refs": ["深圳"],
                            "social_dynamics": [],
                            "persona_evidence": {"behavioral": []},
                            "tags": ["#出行"],
                            "confidence": 0.9,
                            "reason": "test",
                            "meta_info": {"title": "出行", "location_context": "深圳", "photo_count": 1},
                            "objective_fact": {"scene_description": "一起出门", "participants": ["【主角】", "同行驾驶员(Person_001)"]},
                            "source_temp_event_id": "TEMP_EVT_001",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with patch(
                "services.memory_pipeline.precomputed_bundle_runner.run_downstream_profile_agent_audit",
                side_effect=RuntimeError("profile_agent unavailable"),
            ):
                result = run_precomputed_bundle_pipeline(
                    bundle_dir=base,
                    output_dir=output_dir,
                    llm_processor=StubLLMProcessor(),
                )

            self.assertTrue((output_dir / "relationships.json").exists())
            self.assertTrue((output_dir / "structured_profile.json").exists())
            self.assertTrue((output_dir / "downstream_audit_report.json").exists())
            self.assertTrue((output_dir / "profile_llm_batch_debug.json").exists())
            self.assertTrue((output_dir / "initial_relationships_before_backflow.json").exists())
            self.assertTrue((output_dir / "initial_relationship_dossiers_before_backflow.json").exists())
            self.assertTrue((output_dir / "initial_relationships_before_backflow_summary.json").exists())
            self.assertTrue((output_dir / "pre_audit_snapshot").exists())
            self.assertTrue((output_dir / "pre_audit_snapshot" / "summary.json").exists())
            self.assertTrue((output_dir / "pre_audit_snapshot" / "primary_decision.json").exists())
            self.assertTrue((output_dir / "pre_audit_snapshot" / "relationships.json").exists())
            self.assertTrue((output_dir / "pre_audit_snapshot" / "structured_profile.json").exists())
            self.assertTrue((output_dir / "pre_audit_snapshot" / "profile_llm_batch_debug.json").exists())
            self.assertTrue((output_dir / "test_issue_log.json").exists())
            self.assertEqual(result["final_primary_person_id"], "Person_011")

            relationships = json.loads((output_dir / "relationships.json").read_text(encoding="utf-8"))
            self.assertEqual(len(relationships), 1)
            self.assertEqual(relationships[0]["person_id"], "Person_001")

            initial_relationships = json.loads(
                (output_dir / "initial_relationships_before_backflow.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(initial_relationships), 1)
            self.assertEqual(initial_relationships[0]["person_id"], "Person_001")

            initial_summary = json.loads(
                (output_dir / "initial_relationships_before_backflow_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(initial_summary["total_relationships"], 1)
            self.assertEqual(initial_summary["relationship_person_ids"], ["Person_001"])

            pre_audit_summary = json.loads(
                (output_dir / "pre_audit_snapshot" / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(pre_audit_summary["primary_person_id"], "Person_011")
            self.assertEqual(pre_audit_summary["relationship_count"], 1)
            self.assertEqual(pre_audit_summary["field_decisions"], 50)

            pre_audit_primary_decision = json.loads(
                (output_dir / "pre_audit_snapshot" / "primary_decision.json").read_text(encoding="utf-8")
            )
            self.assertEqual(pre_audit_primary_decision["primary_person_id"], "Person_011")

            pre_audit_relationships = json.loads(
                (output_dir / "pre_audit_snapshot" / "relationships.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(pre_audit_relationships), 1)
            self.assertEqual(pre_audit_relationships[0]["person_id"], "Person_001")

            issue_log = json.loads((output_dir / "test_issue_log.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(issue_log["summary"]["issue_count"], 1)
            self.assertIn(
                "DOWNSTREAM_AUDIT_SKIPPED",
                [item["code"] for item in issue_log["issues"]],
            )

            mapping_debug = json.loads((output_dir / "mapping_debug.json").read_text(encoding="utf-8"))
            self.assertEqual(mapping_debug["canonical_primary_person_id"], "Person_011")

            stage_reports_path = output_dir / "stage_reports.jsonl"
            self.assertTrue(stage_reports_path.exists())
            stage_reports = [
                json.loads(line)
                for line in stage_reports_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(
                [item["stage"] for item in stage_reports],
                [
                    "load_precomputed_state",
                    "screen_people",
                    "primary_decision",
                    "events_loaded",
                    "relationships",
                    "groups",
                    "profile_lp3",
                    "downstream_audit",
                ],
            )
            self.assertEqual(stage_reports[-1]["counts"]["relationships"], 1)
            self.assertEqual(
                stage_reports[4]["artifacts"]["initial_relationships_path"],
                str(output_dir / "initial_relationships_before_backflow.json"),
            )
            self.assertEqual(
                stage_reports[6]["artifacts"]["pre_audit_snapshot_dir"],
                str(output_dir / "pre_audit_snapshot"),
            )
            self.assertEqual(result["stage_reports_path"], str(stage_reports_path))
            self.assertEqual(
                result["profile_llm_batch_debug_path"],
                str(output_dir / "profile_llm_batch_debug.json"),
            )
            self.assertEqual(
                result["initial_relationships_before_backflow_path"],
                str(output_dir / "initial_relationships_before_backflow.json"),
            )
            self.assertEqual(
                result["test_issue_log_path"],
                str(output_dir / "test_issue_log.json"),
            )
            self.assertEqual(
                result["pre_audit_snapshot_dir"],
                str(output_dir / "pre_audit_snapshot"),
            )
            self.assertEqual(
                result["pre_audit_structured_profile_path"],
                str(output_dir / "pre_audit_snapshot" / "structured_profile.json"),
            )

    def test_run_precomputed_bundle_pipeline_preserves_initial_relationship_archive_when_protagonist_backflow_clears_final_output(
        self,
    ) -> None:
        from services.memory_pipeline.precomputed_bundle_runner import run_precomputed_bundle_pipeline

        class StubLLMProcessor:
            def __init__(self) -> None:
                self.primary_person_id = "Person_011"

            def _collect_relationship_evidence(self, person_id, vlm_results, events=None):
                if person_id != "Person_001":
                    return {
                        "photo_count": 0,
                        "time_span_days": 0,
                        "recent_gap_days": 0,
                        "scenes": [],
                        "private_scene_ratio": 0.0,
                        "dominant_scene_ratio": 0.0,
                        "interaction_behavior": [],
                        "with_user_only": True,
                        "contact_types": [],
                        "rela_events": [],
                        "monthly_frequency": 0.0,
                        "trend_detail": {},
                        "co_appearing_persons": [],
                        "anomalies": [],
                    }
                return {
                    "photo_count": 4,
                    "time_span_days": 12,
                    "recent_gap_days": 1,
                    "scenes": ["车内", "餐厅"],
                    "private_scene_ratio": 0.3,
                    "dominant_scene_ratio": 0.5,
                    "interaction_behavior": ["聊天"],
                    "with_user_only": True,
                    "contact_types": ["standing_near"],
                    "rela_events": [
                        {
                            "event_id": "EVT_0001",
                            "date": "2026-03-01",
                            "title": "出行",
                            "location": "深圳",
                            "description": "一起出门",
                            "participants": ["Person_011", "Person_001"],
                            "narrative_synthesis": "一起出门",
                            "social_dynamics": [],
                        }
                    ],
                    "monthly_frequency": 2.5,
                    "trend_detail": {"direction": "up"},
                    "co_appearing_persons": [],
                    "anomalies": [],
                }

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir) / "bundle"
            output_dir = Path(temp_dir) / "out"
            (base / "face").mkdir(parents=True)
            (base / "vlm").mkdir()
            (base / "lp1").mkdir()

            (base / "face" / "face_recognition_output.json").write_text(
                json.dumps(
                    {
                        "primary_person_id": "Person_011",
                        "persons": [
                            {"person_id": "Person_011", "photo_count": 5},
                            {"person_id": "Person_001", "photo_count": 4},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (base / "vlm" / "vp1_observations.json").write_text(
                json.dumps(
                    [
                        {
                            "photo_id": "photo_001",
                            "timestamp": "2026-03-01T10:00:00",
                            "face_person_ids": ["Person_011", "Person_001"],
                            "vlm_analysis": {
                                "summary": "【主角】和 Person_001 在车内聊天。",
                                "people": [
                                    {"person_id": "Person_011", "contact_type": "no_contact"},
                                    {"person_id": "Person_001", "contact_type": "standing_near"},
                                ],
                                "relations": [],
                                "scene": {"location_detected": "车内"},
                                "event": {"activity": "出行"},
                                "details": [],
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (base / "lp1" / "lp1_events_compact.json").write_text(
                json.dumps(
                    [
                        {
                            "event_id": "EVT_0001",
                            "batch_id": "BATCH_0001",
                            "anchor_photo_id": "photo_001",
                            "supporting_photo_ids": ["photo_001"],
                            "started_at": "2026-03-01T10:00:00",
                            "ended_at": "2026-03-01T11:00:00",
                            "title": "出行",
                            "narrative_synthesis": "【主角】和同行驾驶员一起出门。",
                            "participant_person_ids": ["【主角】", "同行驾驶员(Person_001)"],
                            "depicted_person_ids": ["Person_011", "Person_001"],
                            "place_refs": ["深圳"],
                            "social_dynamics": [],
                            "persona_evidence": {"behavioral": []},
                            "tags": ["#出行"],
                            "confidence": 0.9,
                            "reason": "test",
                            "meta_info": {"title": "出行", "location_context": "深圳", "photo_count": 1},
                            "objective_fact": {"scene_description": "一起出门", "participants": ["【主角】", "同行驾驶员(Person_001)"]},
                            "source_temp_event_id": "TEMP_EVT_001",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            minimal_audit_report = {
                "metadata": {
                    "downstream_engine": "profile_agent",
                    "audit_mode": "selective_profile_domain_rules_facts_only",
                },
                "summary": {
                    "total_audited_tags": 1,
                    "challenged_count": 1,
                    "accepted_count": 0,
                    "downgraded_count": 0,
                    "rejected_count": 1,
                    "not_audited_count": 0,
                },
                "backflow": {
                    "album_id": "bundle_test",
                    "storage_saved": True,
                    "protagonist": {"official_output_applied": False, "merged_output": {"agent_type": "protagonist", "tags": []}, "actions": []},
                    "relationship": {"official_output_applied": False, "merged_output": {"agent_type": "relationship", "tags": []}, "actions": []},
                    "profile": {"official_output_applied": True, "merged_output": {"agent_type": "profile", "tags": []}, "field_actions": []},
                },
                "protagonist": {"extractor_output": {}, "critic_output": {"challenges": []}, "judge_output": {"decisions": [], "hard_cases": []}, "audit_flags": [], "not_audited": []},
                "relationship": {"extractor_output": {}, "critic_output": {"challenges": []}, "judge_output": {"decisions": [], "hard_cases": []}, "audit_flags": [], "not_audited": []},
                "profile": {"extractor_output": {"agent_type": "profile", "tags": []}, "critic_output": {"challenges": []}, "judge_output": {"decisions": [], "hard_cases": []}, "audit_flags": [], "not_audited": []},
            }

            with patch(
                "services.memory_pipeline.precomputed_bundle_runner.generate_structured_profile",
                return_value={"structured": {}, "field_decisions": [], "consistency": {}, "llm_batch_debug": []},
            ), patch(
                "services.memory_pipeline.precomputed_bundle_runner.run_downstream_profile_agent_audit",
                return_value=minimal_audit_report,
            ), patch(
                "services.memory_pipeline.precomputed_bundle_runner.apply_downstream_protagonist_backflow",
                return_value=(
                    {"mode": "photographer_mode", "primary_person_id": None, "confidence": 0.0},
                    True,
                ),
            ), patch(
                "services.memory_pipeline.precomputed_bundle_runner.apply_downstream_relationship_backflow",
                side_effect=lambda relationships, dossiers, report: (relationships, dossiers, False),
            ), patch(
                "services.memory_pipeline.precomputed_bundle_runner.apply_downstream_profile_backflow",
                side_effect=lambda structured, report, field_decisions=None: (structured, field_decisions or []),
            ):
                result = run_precomputed_bundle_pipeline(
                    bundle_dir=base,
                    output_dir=output_dir,
                    llm_processor=StubLLMProcessor(),
                )

            final_relationships = json.loads((output_dir / "relationships.json").read_text(encoding="utf-8"))
            self.assertEqual(final_relationships, [])

            initial_relationships = json.loads(
                (output_dir / "initial_relationships_before_backflow.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(initial_relationships), 1)
            self.assertEqual(initial_relationships[0]["person_id"], "Person_001")

            pre_audit_primary_decision = json.loads(
                (output_dir / "pre_audit_snapshot" / "primary_decision.json").read_text(encoding="utf-8")
            )
            self.assertEqual(pre_audit_primary_decision["primary_person_id"], "Person_011")

            pre_audit_relationships = json.loads(
                (output_dir / "pre_audit_snapshot" / "relationships.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(pre_audit_relationships), 1)
            self.assertEqual(pre_audit_relationships[0]["person_id"], "Person_001")

            issue_log = json.loads((output_dir / "test_issue_log.json").read_text(encoding="utf-8"))
            issue_codes = [item["code"] for item in issue_log["issues"]]
            self.assertIn("PROTAGONIST_BACKFLOW_RERUN", issue_codes)
            self.assertIn("INITIAL_RELATIONSHIPS_CHANGED_AFTER_BACKFLOW", issue_codes)
            self.assertIsNone(result["final_primary_person_id"])

    def test_run_precomputed_bundle_pipeline_uses_openrouter_only_for_lp3_when_key_is_provided(self) -> None:
        from services.memory_pipeline.precomputed_bundle_runner import run_precomputed_bundle_pipeline

        class StubLLMProcessor:
            def __init__(self) -> None:
                self.primary_person_id = "Person_011"

            def _collect_relationship_evidence(self, person_id, vlm_results, events=None):
                return {
                    "photo_count": 0,
                    "time_span_days": 0,
                    "recent_gap_days": 0,
                    "scenes": [],
                    "private_scene_ratio": 0.0,
                    "dominant_scene_ratio": 0.0,
                    "interaction_behavior": [],
                    "with_user_only": True,
                    "contact_types": [],
                    "rela_events": [],
                    "monthly_frequency": 0.0,
                    "trend_detail": {},
                    "co_appearing_persons": [],
                    "anomalies": [],
                }

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir) / "bundle"
            output_dir = Path(temp_dir) / "out"
            (base / "face").mkdir(parents=True)
            (base / "vlm").mkdir()
            (base / "lp1").mkdir()

            (base / "face" / "face_recognition_output.json").write_text(
                json.dumps(
                    {
                        "primary_person_id": "Person_011",
                        "persons": [
                            {"person_id": "Person_011", "photo_count": 5},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (base / "vlm" / "vp1_observations.json").write_text(
                json.dumps(
                    [
                        {
                            "photo_id": "photo_001",
                            "timestamp": "2026-03-01T10:00:00",
                            "face_person_ids": ["Person_011"],
                            "vlm_analysis": {
                                "summary": "【主角】独自出行。",
                                "people": [{"person_id": "Person_011", "contact_type": "no_contact"}],
                                "relations": [],
                                "scene": {"location_detected": "车内"},
                                "event": {"activity": "出行"},
                                "details": [],
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (base / "lp1" / "lp1_events_compact.json").write_text(
                json.dumps(
                    [
                        {
                            "event_id": "EVT_0001",
                            "batch_id": "BATCH_0001",
                            "anchor_photo_id": "photo_001",
                            "supporting_photo_ids": ["photo_001"],
                            "started_at": "2026-03-01T10:00:00",
                            "ended_at": "2026-03-01T11:00:00",
                            "title": "独自出行",
                            "narrative_synthesis": "【主角】独自出门。",
                            "participant_person_ids": ["【主角】"],
                            "depicted_person_ids": ["Person_011"],
                            "place_refs": ["深圳"],
                            "social_dynamics": [],
                            "persona_evidence": {"behavioral": []},
                            "tags": ["#出行"],
                            "confidence": 0.9,
                            "reason": "test",
                            "meta_info": {"title": "独自出行", "location_context": "深圳", "photo_count": 1},
                            "objective_fact": {"scene_description": "独自出门", "participants": ["【主角】"]},
                            "source_temp_event_id": "TEMP_EVT_001",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            fake_profile_processor = object()
            with patch(
                "services.memory_pipeline.precomputed_bundle_runner.OpenRouterProfileLLMProcessor",
                return_value=fake_profile_processor,
            ) as openrouter_ctor, patch(
                "services.memory_pipeline.precomputed_bundle_runner.generate_structured_profile",
                return_value={"structured": {}, "field_decisions": [], "consistency": {}},
            ) as generate_profile, patch(
                "services.memory_pipeline.precomputed_bundle_runner.run_downstream_profile_agent_audit",
                side_effect=RuntimeError("profile_agent unavailable"),
            ):
                run_precomputed_bundle_pipeline(
                    bundle_dir=base,
                    output_dir=output_dir,
                    llm_processor=StubLLMProcessor(),
                    profile_openrouter_key="sk-test-profile",
                    profile_model="google/gemini-3.1-flash-lite-preview",
                )

        openrouter_ctor.assert_called_once_with(
            api_key="sk-test-profile",
            base_url=None,
            model="google/gemini-3.1-flash-lite-preview",
            primary_person_id="Person_011",
        )
        self.assertIs(generate_profile.call_args.kwargs["llm_processor"], fake_profile_processor)
