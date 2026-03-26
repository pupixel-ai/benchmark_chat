from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from models import Event, Relationship
import main
from utils.output_artifacts import (
    build_artifacts_manifest,
    build_final_output_payload,
    build_profile_debug_artifact,
    build_relationships_artifact,
    save_markdown_report,
)


class OutputArtifactsTests(unittest.TestCase):
    def test_build_relationships_artifact_keeps_decision_trace(self) -> None:
        relationships = [
            Relationship(
                person_id="Person_002",
                relationship_type="friend",
                intimacy_score=0.42,
                status="stable",
                confidence=0.77,
                reasoning="LLM override accepted",
                shared_events=[{"event_id": "EVT_001", "date": "2026-03-01", "narrative": "一起吃饭"}],
                evidence={
                    "photo_count": 4,
                    "decision_trace": {
                        "candidate_types": ["classmate_colleague", "acquaintance"],
                        "code_status_suggestion": "growing",
                        "code_confidence_baseline": 0.6,
                        "llm_relationship_type": "friend",
                        "llm_status": "stable",
                        "llm_confidence": 0.9,
                        "final_relationship_type": "friend",
                        "final_status": "stable",
                        "final_confidence": 0.78,
                        "applied_vetoes": [],
                    },
                },
            )
        ]

        payload = build_relationships_artifact(
            relationships=relationships,
            primary_person_id="Person_001",
            generated_at="2026-03-22T10:00:00",
        )

        self.assertEqual(payload["metadata"]["primary_person_id"], "Person_001")
        self.assertEqual(payload["metadata"]["total_relationships"], 1)
        self.assertEqual(
            payload["relationships"][0]["evidence"]["decision_trace"]["final_relationship_type"],
            "friend",
        )

    def test_build_profile_debug_artifact_preserves_debug_and_consistency_without_structured_profile(self) -> None:
        payload = build_profile_debug_artifact(
            profile_result={
                "debug": {"report_reasoning": {"step": "ok"}},
                "consistency": {"summary": {"issue_count": 1}, "issues": [{"code": "X"}]},
            },
            primary_person_id="Person_001",
            total_events=3,
            total_relationships=2,
            generated_at="2026-03-22T10:00:00",
        )

        self.assertEqual(payload["metadata"]["total_events"], 3)
        self.assertEqual(payload["debug"]["report_reasoning"]["step"], "ok")
        self.assertEqual(payload["consistency"]["summary"]["issue_count"], 1)

    def test_build_final_output_payload_includes_artifacts_manifest(self) -> None:
        events = [
            Event(
                event_id="EVT_001",
                date="2026-03-01",
                time_range="10:00 - 11:00",
                duration="1小时",
                title="一起喝咖啡",
                type="社交",
                participants=["Person_001", "Person_002"],
                location="咖啡馆",
                description="一起聊天",
                photo_count=2,
                confidence=0.8,
                reason="",
            )
        ]
        relationships = [
            Relationship(
                person_id="Person_002",
                relationship_type="friend",
                intimacy_score=0.42,
                status="stable",
                confidence=0.77,
                reasoning="",
                evidence={"decision_trace": {"applied_vetoes": []}},
            )
        ]
        artifacts = build_artifacts_manifest(
            dedupe_report_path="/tmp/dedupe.json",
            face_state_path="/tmp/face_state.json",
            face_output_path="/tmp/face_output.json",
            vlm_cache_path="/tmp/vlm_cache.json",
            relationships_path="/tmp/relationships.json",
            structured_profile_path="/tmp/profile_structured.json",
            profile_report_path="/tmp/profile.md",
            profile_debug_path="/tmp/profile_debug.json",
            detailed_report_path="/tmp/detailed.md",
        )

        payload = build_final_output_payload(
            events=events,
            relationships=relationships,
            face_db={},
            artifacts=artifacts,
            models={"vlm": "gemini-2.0-flash", "llm": "gemini-2.5-flash", "face": "InsightFace/buffalo_l"},
            generated_at="2026-03-22T10:00:00",
        )

        self.assertEqual(payload["metadata"]["total_events"], 1)
        self.assertEqual(payload["artifacts"]["relationships_path"], "/tmp/relationships.json")
        self.assertEqual(payload["relationships"][0]["status"], "stable")

    def test_build_final_output_payload_preserves_full_lp1_event_fields(self) -> None:
        events = [
            Event(
                event_id="EVT_001",
                date="2026-03-01",
                time_range="10:00 - 11:00",
                duration="1小时",
                title="一起喝咖啡",
                type="社交",
                participants=["Person_001", "Person_002"],
                location="咖啡馆",
                description="一起聊天",
                photo_count=2,
                confidence=0.8,
                reason="事件闭环完整",
                narrative="旧字段叙事",
                narrative_synthesis="主角和朋友在咖啡馆聊天",
                meta_info={
                    "title": "咖啡馆聊天",
                    "timestamp": "2026-03-01 10:00 - 11:00",
                    "location_context": "咖啡馆",
                    "photo_count": 2,
                },
                objective_fact={
                    "scene_description": "咖啡馆桌面与饮品",
                    "participants": ["Person_001", "Person_002"],
                },
                social_dynamics=[
                    {
                        "target_id": "Person_002",
                        "interaction_type": "共同进餐",
                        "social_clue": "面对面交谈",
                        "relation_hypothesis": "friend",
                        "confidence": 0.8,
                    }
                ],
                tags=["#coffee", "#friend"],
                persona_evidence={"behavioral": ["social"], "aesthetic": [], "socioeconomic": []},
            )
        ]

        payload = build_final_output_payload(
            events=events,
            relationships=[],
            face_db={},
            artifacts={},
            models={"vlm": "gemini-2.0-flash", "llm": "gemini-2.5-flash", "face": "InsightFace/buffalo_l"},
            generated_at="2026-03-22T10:00:00",
        )

        event_payload = payload["events"][0]
        self.assertEqual(event_payload["narrative_synthesis"], "主角和朋友在咖啡馆聊天")
        self.assertEqual(event_payload["meta_info"]["location_context"], "咖啡馆")
        self.assertEqual(event_payload["objective_fact"]["scene_description"], "咖啡馆桌面与饮品")
        self.assertEqual(event_payload["social_dynamics"][0]["target_id"], "Person_002")
        self.assertEqual(event_payload["tags"], ["#coffee", "#friend"])

    def test_save_markdown_report_contains_core_sections(self) -> None:
        result = {
            "metadata": {
                "generated_at": "2026-03-22T10:00:00",
                "version": "2.0",
            },
            "summary": {
                "total_events": 1,
                "total_relationships": 1,
                "primary_person_id": "Person_001",
            },
            "artifacts": {
                "relationships_path": "/tmp/relationships.json",
                "profile_debug_path": "/tmp/profile_debug.json",
            },
            "relationships": [
                {
                    "person_id": "Person_002",
                    "relationship_type": "friend",
                    "status": "stable",
                    "confidence": 0.77,
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "report.md"
            save_markdown_report(result, str(output_path))
            content = output_path.read_text(encoding="utf-8")

        self.assertIn("记忆工程 v2.0 输出摘要", content)
        self.assertIn("主角 person_id: Person_001", content)
        self.assertIn("relationships_path", content)
        self.assertIn("Person_002", content)

    def test_run_memory_pipeline_uses_single_helper_and_returns_internal_paths(self) -> None:
        llm = object()
        fake_state = SimpleNamespace(
            primary_decision=None,
            relationship_dossiers=[],
            relationships=[],
            groups=[],
            profile_context=None,
        )
        profile_result = {
            "events": ["event"],
            "relationships": [],
            "structured": {"long_term_facts": {}},
            "report": "",
            "debug": {},
            "consistency": {},
            "internal_artifacts": {
                "relationship_dossiers": [{"person_id": "Person_002"}],
                "group_artifacts": [{"group_id": "GRP_001"}],
                "profile_fact_decisions": [{"field_key": "x"}],
            },
        }

        with (
            patch.object(main, "build_memory_state", return_value=fake_state) as build_state_mock,
            patch.object(main, "run_memory_pipeline", return_value=profile_result) as run_pipeline_mock,
            patch.object(main, "run_downstream_profile_agent_audit", return_value={"summary": {}, "backflow": {}}),
            patch.object(main, "save_internal_artifact", side_effect=["/tmp/dossiers.json", "/tmp/groups.json", "/tmp/facts.json", "/tmp/audit.json"]),
        ):
            result = main.run_memory_pipeline_entry(
                llm=llm,
                photos=[],
                face_db={"Person_001": {}},
                vlm_results=[],
                primary_person_id="Person_001",
            )

        build_state_mock.assert_called_once()
        run_pipeline_mock.assert_called_once()
        run_kwargs = run_pipeline_mock.call_args.kwargs
        self.assertEqual(run_kwargs["state"], fake_state)
        self.assertEqual(run_kwargs["llm_processor"], llm)
        self.assertEqual(run_kwargs["fallback_primary_person_id"], "Person_001")
        self.assertFalse(run_kwargs.get("debug_review_audit", False))
        self.assertEqual(result["relationship_dossiers_path"], "/tmp/dossiers.json")
        self.assertEqual(result["group_artifacts_path"], "/tmp/groups.json")
        self.assertEqual(result["profile_fact_decisions_path"], "/tmp/facts.json")

    def test_run_memory_pipeline_entry_applies_profile_backflow_before_returning(self) -> None:
        llm = object()
        fake_state = SimpleNamespace(
            primary_decision=None,
            relationship_dossiers=[],
            relationships=[],
            groups=[],
            profile_context=None,
        )
        profile_result = {
            "events": ["event"],
            "relationships": [],
            "structured": {
                "long_term_facts": {
                    "social_identity": {
                        "education": {
                            "value": "college_student",
                            "confidence": 0.84,
                            "evidence": {"event_ids": ["EVT_001"]},
                            "reasoning": "跨事件校园学习场景支持 education。",
                        },
                        "career_phase": {
                            "value": "student",
                            "confidence": 0.72,
                            "evidence": {"event_ids": ["EVT_002"]},
                            "reasoning": "当前仍以校园主线为主。",
                        },
                    }
                }
            },
            "report": "",
            "debug": {},
            "consistency": {},
            "internal_artifacts": {
                "relationship_dossiers": [],
                "group_artifacts": [],
                "profile_fact_decisions": [
                    {
                        "field_key": "long_term_facts.social_identity.education",
                        "draft": {"value": "college_student"},
                        "final": {
                            "value": "college_student",
                            "confidence": 0.84,
                            "evidence": {"event_ids": ["EVT_001"]},
                            "reasoning": "跨事件校园学习场景支持 education。",
                        },
                        "tool_trace": {},
                        "null_reason": None,
                    },
                    {
                        "field_key": "long_term_facts.social_identity.career_phase",
                        "draft": {"value": "student"},
                        "final": {
                            "value": "student",
                            "confidence": 0.72,
                            "evidence": {"event_ids": ["EVT_002"]},
                            "reasoning": "当前仍以校园主线为主。",
                        },
                        "tool_trace": {},
                        "null_reason": None,
                    },
                ],
                "primary_decision": {"mode": "person_id", "primary_person_id": "Person_001"},
            },
        }
        downstream_report = {
            "summary": {},
            "backflow": {
                "profile": {
                    "field_actions": [
                        {
                            "field_key": "long_term_facts.social_identity.education",
                            "verdict": "nullify",
                            "judge_reason": "证据不满足审计规则",
                            "applied_change": "nullify_value",
                        },
                        {
                            "field_key": "long_term_facts.social_identity.career_phase",
                            "verdict": "downgrade",
                            "judge_reason": "更适合作为短期状态保留",
                            "applied_change": "annotate_short_term_downgrade",
                        },
                    ]
                }
            },
        }

        with (
            patch.object(main, "build_memory_state", return_value=fake_state),
            patch.object(main, "run_memory_pipeline", return_value=profile_result),
            patch.object(main, "run_downstream_profile_agent_audit", return_value=downstream_report),
            patch.object(main, "save_internal_artifact", side_effect=["/tmp/dossiers.json", "/tmp/groups.json", "/tmp/facts.json", "/tmp/audit.json"]),
        ):
            result = main.run_memory_pipeline_entry(
                llm=llm,
                photos=[],
                face_db={"Person_001": {}},
                vlm_results=[],
                primary_person_id="Person_001",
            )

        education = result["profile_result"]["structured"]["long_term_facts"]["social_identity"]["education"]
        career_phase = result["profile_result"]["structured"]["long_term_facts"]["social_identity"]["career_phase"]

        self.assertIsNone(education["value"])
        self.assertEqual(education["confidence"], 0.0)
        self.assertIn("下游 Judge 否决", education["reasoning"])
        self.assertIn("downstream_judge:nullify", education["evidence"]["constraint_notes"][0])

        self.assertEqual(career_phase["value"], "student")
        self.assertIn("downstream Judge 建议降为 short_term", career_phase["reasoning"])
        self.assertIn("downstream_judge:downgrade", career_phase["evidence"]["constraint_notes"][0])
        decisions = result["profile_result"]["internal_artifacts"]["profile_fact_decisions"]
        education_decision = next(item for item in decisions if item["field_key"] == "long_term_facts.social_identity.education")
        career_phase_decision = next(item for item in decisions if item["field_key"] == "long_term_facts.social_identity.career_phase")

        self.assertIsNone(education_decision["final"]["value"])
        self.assertEqual(education_decision["final_before_backflow"]["value"], "college_student")
        self.assertEqual(education_decision["backflow"]["verdict"], "nullify")
        self.assertIn("downstream Judge 建议降为 short_term", career_phase_decision["final"]["reasoning"])
        self.assertEqual(career_phase_decision["backflow"]["verdict"], "downgrade")

    def test_run_memory_pipeline_entry_switches_to_photographer_mode_without_rerunning_relationship_inference(self) -> None:
        llm = object()
        fake_state = SimpleNamespace(
            primary_decision=None,
            relationship_dossiers=[],
            relationships=[],
            groups=[],
            profile_context=None,
        )
        initial_profile_result = {
            "events": ["event"],
            "relationships": [
                Relationship(
                    person_id="Person_002",
                    relationship_type="bestie",
                    intimacy_score=0.82,
                    status="stable",
                    confidence=0.86,
                    reasoning="多场景稳定互动。",
                    evidence={"constraint_notes": []},
                )
            ],
            "structured": {
                "long_term_facts": {
                    "social_identity": {
                        "education": {
                            "value": "college_student",
                            "confidence": 0.84,
                            "evidence": {"event_ids": ["EVT_001"]},
                            "reasoning": "跨事件校园学习场景支持 education。",
                        }
                    }
                }
            },
            "report": "",
            "debug": {},
            "consistency": {},
            "internal_artifacts": {
                "relationship_dossiers": [],
                "group_artifacts": [],
                "profile_fact_decisions": [],
                "primary_decision": {
                    "mode": "person_id",
                    "primary_person_id": "Person_001",
                    "confidence": 0.91,
                    "evidence": {"photo_ids": ["PHOTO_001"]},
                    "reasoning": "自拍与身份锚点都领先。",
                },
            },
        }
        after_primary_rerun = {
            "events": ["event"],
            "relationships": [],
            "structured": initial_profile_result["structured"],
            "report": "",
            "debug": {},
            "consistency": {},
            "internal_artifacts": {
                "relationship_dossiers": [],
                "group_artifacts": [],
                "profile_fact_decisions": [],
                "primary_decision": {
                    "mode": "photographer_mode",
                    "primary_person_id": None,
                    "confidence": 0.0,
                    "evidence": {"constraint_notes": ["downstream_judge:nullify:证据不足"]},
                    "reasoning": "下游 Judge 否决主角身份。",
                },
            },
        }
        initial_audit = {
            "summary": {},
            "backflow": {
                "protagonist": {
                    "actions": [
                        {
                            "mapped_dimension": "主角>身份确认",
                            "verdict": "nullify",
                            "judge_reason": "证据不足",
                        }
                    ]
                },
                "relationship": {"actions": []},
                "profile": {"field_actions": []},
            },
        }
        final_audit = {
            "summary": {},
            "backflow": {
                "protagonist": {"actions": []},
                "relationship": {"actions": []},
                "profile": {
                    "field_actions": [
                        {
                            "field_key": "long_term_facts.social_identity.education",
                            "verdict": "nullify",
                            "judge_reason": "教育证据不足",
                            "applied_change": "nullify_value",
                        }
                    ]
                },
            },
        }

        with (
            patch.object(main, "build_memory_state", return_value=fake_state),
            patch.object(main, "run_memory_pipeline", return_value=initial_profile_result),
            patch.object(
                main,
                "run_downstream_profile_agent_audit",
                side_effect=[initial_audit, final_audit],
            ) as audit_mock,
            patch.object(main, "rerun_pipeline_from_primary_backflow", return_value=after_primary_rerun) as rerun_primary_mock,
            patch.object(main, "rerun_pipeline_from_relationship_backflow", return_value=after_primary_rerun) as rerun_relationship_mock,
            patch.object(main, "save_internal_artifact", side_effect=["/tmp/dossiers.json", "/tmp/groups.json", "/tmp/facts.json", "/tmp/audit.json"]),
        ):
            result = main.run_memory_pipeline_entry(
                llm=llm,
                photos=[],
                face_db={"Person_001": {}},
                vlm_results=[],
                primary_person_id="Person_001",
            )

        self.assertEqual(audit_mock.call_count, 2)
        rerun_primary_mock.assert_not_called()
        rerun_relationship_mock.assert_called_once()
        self.assertEqual(fake_state.relationships, [])
        self.assertEqual(fake_state.relationship_dossiers, [])
        self.assertEqual(fake_state.groups, [])
        self.assertEqual(result["relationships"], [])
        education = result["profile_result"]["structured"]["long_term_facts"]["social_identity"]["education"]
        self.assertIsNone(education["value"])

    def test_run_memory_pipeline_entry_falls_back_when_downstream_audit_runtime_fails(self) -> None:
        llm = object()
        fake_state = SimpleNamespace(
            primary_decision=None,
            relationship_dossiers=[],
            relationships=[],
            groups=[],
            profile_context=None,
        )
        profile_result = {
            "events": ["event"],
            "relationships": [],
            "structured": {"long_term_facts": {}},
            "report": "",
            "debug": {},
            "consistency": {},
            "internal_artifacts": {
                "relationship_dossiers": [],
                "group_artifacts": [],
                "profile_fact_decisions": [],
                "primary_decision": {
                    "mode": "person_id",
                    "primary_person_id": "Person_001",
                    "confidence": 0.91,
                    "evidence": {"photo_ids": ["PHOTO_001"]},
                    "reasoning": "自拍与身份锚点都领先。",
                },
            },
        }
        saved_payloads = {}

        def _capture_artifact(*, artifact_name, payload, path, **metadata):
            saved_payloads[artifact_name] = payload
            return path

        with (
            patch.object(main, "build_memory_state", return_value=fake_state),
            patch.object(main, "run_memory_pipeline", return_value=profile_result),
            patch.object(main, "run_downstream_profile_agent_audit", side_effect=RuntimeError("judge init failed")),
            patch.object(main, "save_internal_artifact", side_effect=_capture_artifact),
        ):
            result = main.run_memory_pipeline_entry(
                llm=llm,
                photos=[],
                face_db={"Person_001": {}},
                vlm_results=[],
                primary_person_id="Person_001",
            )

        audit_payload = saved_payloads["downstream_audit_report"]
        self.assertEqual(audit_payload["metadata"]["audit_status"], "skipped_init_failure")
        self.assertEqual(audit_payload["metadata"]["audit_error_type"], "RuntimeError")
        self.assertFalse(audit_payload["backflow"]["storage_saved"])
        self.assertEqual(result["final_primary_person_id"], "Person_001")

    def test_main_module_no_longer_exposes_legacy_or_flagged_dual_entrypoints(self) -> None:
        self.assertFalse(hasattr(main, "run_classic_llm_pipeline"))
        self.assertFalse(hasattr(main, "run_multi_agent_llm_pipeline"))


if __name__ == "__main__":
    unittest.main()
