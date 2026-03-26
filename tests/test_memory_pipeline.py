from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import json
from unittest.mock import patch

from models import Event, Relationship
from utils.output_artifacts import serialize_relationship


class MemoryPipelineArchitectureTests(unittest.TestCase):
    def test_primary_decision_only_exposes_final_fields(self) -> None:
        from services.memory_pipeline.primary_person import PrimaryDecision

        decision = PrimaryDecision(
            mode="person_id",
            primary_person_id="Person_001",
            confidence=0.92,
            evidence={
                "photo_ids": ["PHOTO_001"],
                "event_ids": [],
                "person_ids": ["Person_001"],
                "group_ids": [],
                "feature_names": ["explicit_selfie_count"],
                "supporting_refs": [],
                "contradicting_refs": [],
            },
            reasoning="Person_001 在自拍证据上领先，判为主角。",
        )

        self.assertEqual(
            decision.to_dict(),
            {
                "mode": "person_id",
                "primary_person_id": "Person_001",
                "confidence": 0.92,
                "evidence": {
                    "photo_ids": ["PHOTO_001"],
                    "event_ids": [],
                    "person_ids": ["Person_001"],
                    "group_ids": [],
                    "feature_names": ["explicit_selfie_count"],
                    "supporting_refs": [],
                    "contradicting_refs": [],
                },
                "reasoning": "Person_001 在自拍证据上领先，判为主角。",
            },
        )
        self.assertNotIn("alias_person_ids", decision.to_dict())

    def test_relationship_dossiers_include_non_block_people_without_threshold_gate(self) -> None:
        from services.memory_pipeline.relationships import build_relationship_dossiers
        from services.memory_pipeline.types import MemoryState, PersonScreening

        class StubLLMProcessor:
            def extract_events(self, vlm_results):
                return []

            def _collect_relationship_evidence(self, person_id, vlm_results, events=None):
                return {
                    "photo_count": 1,
                    "time_span": "1天",
                    "time_span_days": 1,
                    "recent_gap_days": 0,
                    "scenes": ["咖啡馆"],
                    "interaction_behavior": ["聊天"],
                    "contact_types": ["no_contact"],
                    "rela_events": [
                        {
                            "event_id": "EVT_001",
                            "date": "2026-03-01",
                            "title": "咖啡馆聊天",
                            "location": "咖啡馆",
                            "photo_count": 1,
                            "description": "聊天",
                            "participants": ["Person_001", person_id],
                            "narrative_synthesis": "一起喝咖啡",
                            "social_dynamics": [],
                        }
                    ],
                    "monthly_frequency": 1.0,
                    "trend_detail": {"direction": "flat"},
                    "co_appearing_persons": [],
                    "anomalies": [],
                    "private_scene_ratio": 0.0,
                    "dominant_scene_ratio": 1.0,
                    "with_user_only": True,
                    "sample_scenes": [],
                    "weekend_frequency": "低",
                }

        state = MemoryState(
            photos=[],
            face_db={
                "Person_001": {"photo_count": 10, "first_seen": datetime(2026, 3, 1), "last_seen": datetime(2026, 3, 5)},
                "Person_002": {"photo_count": 1, "first_seen": datetime(2026, 3, 1), "last_seen": datetime(2026, 3, 1)},
            },
            vlm_results=[],
            screening={
                "Person_001": PersonScreening(person_id="Person_001", person_kind="real_person", memory_value="core"),
                "Person_002": PersonScreening(person_id="Person_002", person_kind="real_person", memory_value="candidate"),
            },
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
            events=[],
        )

        dossiers = build_relationship_dossiers(state=state, llm_processor=StubLLMProcessor())

        self.assertEqual(len(dossiers), 1)
        self.assertEqual(dossiers[0].person_id, "Person_002")
        self.assertEqual(dossiers[0].photo_count, 1)

    def test_primary_reflection_switches_to_photographer_mode_when_top_candidates_are_ambiguous(self) -> None:
        from services.memory_pipeline.primary_person import analyze_primary_person_with_reflection
        from services.memory_pipeline.types import MemoryState, PersonScreening

        state = MemoryState(
            photos=[],
            face_db={
                "Person_001": {"photo_count": 3},
                "Person_002": {"photo_count": 3},
            },
            vlm_results=[],
            screening={
                "Person_001": PersonScreening(person_id="Person_001", person_kind="real_person", memory_value="candidate"),
                "Person_002": PersonScreening(person_id="Person_002", person_kind="real_person", memory_value="candidate"),
            },
        )

        decision, reflection = analyze_primary_person_with_reflection(state)

        self.assertEqual(decision.mode, "photographer_mode")
        self.assertIsNone(decision.primary_person_id)
        self.assertTrue(reflection["triggered"])
        self.assertIn("ambiguous_top_candidates", reflection["issues"])

    def test_primary_agent_vetoes_other_photo_candidate_when_user_is_primarily_shooting_them(self) -> None:
        from services.memory_pipeline.primary_person import analyze_primary_person_with_reflection
        from services.memory_pipeline.types import MemoryState, PersonScreening

        vlm_results = []
        for idx in range(6):
            vlm_results.append(
                {
                    "photo_id": f"PHOTO_{idx+1:03d}",
                    "timestamp": f"2026-03-{idx+1:02d}T10:00:00",
                    "vlm_analysis": {
                        "summary": "【主角】作为拍摄者记录 Person_002 的人像展示照",
                        "people": [{"person_id": "Person_002"}],
                        "scene": {"location_detected": "户外街区"},
                        "event": {"activity": "人像拍摄"},
                    },
                }
            )

        state = MemoryState(
            photos=[],
            face_db={
                "Person_001": {"photo_count": 1},
                "Person_002": {"photo_count": 6},
            },
            vlm_results=vlm_results,
            screening={
                "Person_001": PersonScreening(person_id="Person_001", person_kind="real_person", memory_value="candidate"),
                "Person_002": PersonScreening(person_id="Person_002", person_kind="real_person", memory_value="candidate"),
            },
        )

        decision, reflection = analyze_primary_person_with_reflection(state)

        self.assertEqual(decision.mode, "photographer_mode")
        self.assertIsNone(decision.primary_person_id)
        self.assertIn("other_photo_candidate_is_likely_a_photographed_subject", reflection["issues"])
        self.assertEqual(decision.evidence["person_ids"], [])
        self.assertIn("拍别人", decision.reasoning)

    def test_group_detector_ignores_low_confidence_relationships(self) -> None:
        from services.memory_pipeline.groups import detect_groups
        from services.memory_pipeline.types import MemoryState

        relationships = [
            Relationship(
                person_id="Person_002",
                relationship_type="friend",
                intimacy_score=0.6,
                status="stable",
                confidence=0.88,
                reasoning="",
                shared_events=[{"event_id": "EVT_001", "date": "2026-03-01", "narrative": "formal night"}],
                evidence={"photo_count": 5},
            ),
            Relationship(
                person_id="Person_003",
                relationship_type="friend",
                intimacy_score=0.58,
                status="stable",
                confidence=0.49,
                reasoning="",
                shared_events=[{"event_id": "EVT_001", "date": "2026-03-01", "narrative": "formal night"}],
                evidence={"photo_count": 5},
            ),
        ]
        events = [
            Event(
                event_id="EVT_001",
                date="2026-03-01",
                time_range="19:00 - 22:00",
                duration="3小时",
                title="Formal Night",
                type="社交",
                participants=["Person_001", "Person_002", "Person_003"],
                location="sorority house",
                description="group photo in front of Greek letters",
                photo_count=4,
                confidence=0.88,
                reason="",
                narrative_synthesis="sorority formal night",
            )
        ]
        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[],
            relationships=relationships,
            events=events,
        )

        groups = detect_groups(state)

        self.assertEqual(groups, [])

    def test_relationship_type_reflection_downgrades_close_friend_without_strong_signal(self) -> None:
        from services.memory_pipeline.relationships import build_relationship_dossiers, infer_relationships_from_dossiers
        from services.memory_pipeline.types import MemoryState, PersonScreening

        class StubLLMProcessor:
            def _collect_relationship_evidence(self, person_id, vlm_results, events=None):
                return {
                    "photo_count": 2,
                    "time_span": "45天",
                    "time_span_days": 45,
                    "recent_gap_days": 4,
                    "scenes": ["咖啡馆"],
                    "private_scene_ratio": 0.0,
                    "dominant_scene_ratio": 1.0,
                    "interaction_behavior": ["聊天"],
                    "weekend_frequency": "低",
                    "with_user_only": False,
                    "sample_scenes": [],
                    "contact_types": ["no_contact"],
                    "rela_events": [
                        {
                            "event_id": "EVT_001",
                            "date": "2026-03-01",
                            "title": "咖啡馆碰面",
                            "location": "咖啡馆",
                            "photo_count": 2,
                            "description": "两次同场景同框",
                            "participants": ["Person_001", person_id],
                            "narrative_synthesis": "一起在咖啡馆",
                            "social_dynamics": [],
                        }
                    ],
                    "monthly_frequency": 1.2,
                    "trend_detail": {"direction": "flat"},
                    "co_appearing_persons": [],
                    "anomalies": [],
                }

            def _infer_relationship(self, person_id, evidence, vlm_results, face_db):
                return Relationship(
                    person_id=person_id,
                    relationship_type="close_friend",
                    intimacy_score=0.58,
                    status="stable",
                    confidence=0.81,
                    reasoning="llm judged close friend from sparse evidence",
                    shared_events=[{"event_id": "EVT_001", "date": "2026-03-01", "narrative": "一起在咖啡馆"}],
                    evidence={"photo_count": 2, "decision_trace": {"final_relationship_type": "close_friend"}},
                )

        state = MemoryState(
            photos=[],
            face_db={"Person_001": {"photo_count": 12}, "Person_002": {"photo_count": 2}},
            vlm_results=[],
            screening={
                "Person_001": PersonScreening(person_id="Person_001", person_kind="real_person", memory_value="core"),
                "Person_002": PersonScreening(person_id="Person_002", person_kind="real_person", memory_value="candidate"),
            },
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.95, "evidence": {}, "reasoning": "test"},
            events=[],
        )

        dossiers = build_relationship_dossiers(state=state, llm_processor=StubLLMProcessor())
        relationships, updated_dossiers = infer_relationships_from_dossiers(
            state=state,
            llm_processor=StubLLMProcessor(),
            dossiers=dossiers,
        )

        self.assertEqual(len(relationships), 1)
        self.assertEqual(relationships[0].relationship_type, "friend")
        self.assertTrue(updated_dossiers[0].relationship_reflection["triggered"])
        self.assertIn("close_friend_without_strong_signal", updated_dossiers[0].relationship_reflection["issues"])
        self.assertIn("event_ids", relationships[0].evidence)
        self.assertIn("reasoning", relationships[0].__dict__)

    def test_field_judge_clears_p0_without_strong_evidence_and_keeps_traceable_refs(self) -> None:
        from services.memory_pipeline.profile_fields import FieldJudge, build_profile_context
        from services.memory_pipeline.types import MemoryState

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[
                {
                    "photo_id": "PHOTO_001",
                    "timestamp": "2026-03-01T10:00:00",
                    "vlm_analysis": {
                        "summary": "路过校园",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "校园"},
                        "event": {"activity": "散步"},
                    },
                }
            ],
            events=[],
            relationships=[],
            groups=[],
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
        )

        context = build_profile_context(state)
        judge = FieldJudge()
        decision = judge.judge_field("long_term_facts.social_identity.education", context)

        self.assertEqual(decision.final["value"], None)
        self.assertEqual(decision.final["confidence"], 0.0)
        self.assertEqual(decision.final["evidence"]["events"], [])
        self.assertEqual(decision.final["evidence"]["relationships"], [])
        self.assertEqual(
            decision.final["evidence"]["vlm_observations"][0]["photo_id"],
            "PHOTO_001",
        )
        self.assertIn("reasoning", decision.final)

    def test_field_spec_carries_complete_field_cot_configuration(self) -> None:
        from services.memory_pipeline.profile_fields import FIELD_SPECS

        spec = FIELD_SPECS["long_term_facts.social_identity.education"]

        self.assertTrue(spec.cot_steps)
        self.assertTrue(spec.owner_resolution_steps)
        self.assertTrue(spec.time_reasoning_steps)
        self.assertTrue(spec.counter_evidence_checks)
        self.assertIn("先看跨事件校园/课堂主线", spec.cot_steps[0])

    def test_profile_field_tools_are_split_into_four_layers(self) -> None:
        from services.memory_pipeline.profile_fields import (
            LONG_TERM_EXPRESSION_FIELD_SPECS,
            LONG_TERM_FACT_FIELD_SPECS,
            SHORT_TERM_EXPRESSION_FIELD_SPECS,
            SHORT_TERM_FACT_FIELD_SPECS,
            LongTermExpressionJudge,
            LongTermFactsJudge,
            ShortTermExpressionJudge,
            ShortTermFactsJudge,
        )

        self.assertTrue(LONG_TERM_FACT_FIELD_SPECS)
        self.assertTrue(SHORT_TERM_FACT_FIELD_SPECS)
        self.assertTrue(LONG_TERM_EXPRESSION_FIELD_SPECS)
        self.assertTrue(SHORT_TERM_EXPRESSION_FIELD_SPECS)
        self.assertIn("long_term_expression.personality_mbti", LONG_TERM_EXPRESSION_FIELD_SPECS)
        self.assertIn("short_term_expression.current_mood", SHORT_TERM_EXPRESSION_FIELD_SPECS)
        self.assertIsInstance(LongTermFactsJudge(), LongTermFactsJudge)
        self.assertIsInstance(ShortTermFactsJudge(), ShortTermFactsJudge)
        self.assertIsInstance(LongTermExpressionJudge(), LongTermExpressionJudge)
        self.assertIsInstance(ShortTermExpressionJudge(), ShortTermExpressionJudge)

    def test_field_judge_includes_field_cot_in_llm_prompt(self) -> None:
        from services.memory_pipeline.profile_fields import FieldJudge, build_profile_context
        from services.memory_pipeline.types import MemoryState

        class CaptureLLMProcessor:
            def __init__(self) -> None:
                self.prompts = []

            def _call_llm_via_official_api(self, prompt, response_mime_type="application/json"):
                self.prompts.append(prompt)
                return {"value": "本科在读", "confidence": 0.71}

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[
                {
                    "photo_id": "PHOTO_001",
                    "timestamp": "2026-03-01T10:00:00",
                    "vlm_analysis": {
                        "summary": "主角在校园教室内上课",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "校园教室"},
                        "event": {"activity": "上课"},
                    },
                },
                {
                    "photo_id": "PHOTO_002",
                    "timestamp": "2026-03-03T11:00:00",
                    "vlm_analysis": {
                        "summary": "主角在 campus classroom 做作业",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "campus"},
                        "event": {"activity": "study"},
                    },
                },
            ],
            events=[
                Event(
                    event_id="EVT_001",
                    date="2026-03-01",
                    time_range="10:00 - 11:00",
                    duration="1小时",
                    title="课堂学习",
                    type="学习",
                    participants=["Person_001"],
                    location="校园教室",
                    description="在教室上课",
                    photo_count=1,
                    confidence=0.8,
                    reason="",
                    narrative_synthesis="主角在校园教室上课",
                ),
                Event(
                    event_id="EVT_002",
                    date="2026-03-03",
                    time_range="11:00 - 12:00",
                    duration="1小时",
                    title="课后学习",
                    type="学习",
                    participants=["Person_001"],
                    location="campus classroom",
                    description="在教室做作业",
                    photo_count=1,
                    confidence=0.82,
                    reason="",
                    narrative_synthesis="主角在 campus classroom 做作业",
                ),
            ],
            relationships=[
                Relationship(
                    person_id="Person_002",
                    relationship_type="friend",
                    intimacy_score=0.46,
                    status="stable",
                    confidence=0.73,
                    reasoning="与主角有稳定社交互动",
                )
            ],
            groups=[],
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
        )

        context = build_profile_context(state)
        context["social_media_available"] = True
        llm_processor = CaptureLLMProcessor()
        judge = FieldJudge()

        decision = judge.judge_field(
            "long_term_facts.social_identity.education",
            context,
            llm_processor=llm_processor,
        )

        self.assertEqual(decision.final["value"], "本科在读")
        self.assertEqual(len(llm_processor.prompts), 1)
        self.assertIn("字段级COT", llm_processor.prompts[0])
        self.assertIn("先看跨事件校园/课堂主线", llm_processor.prompts[0])
        self.assertIn("主体归属检查", llm_processor.prompts[0])
        self.assertIn("时间层判断", llm_processor.prompts[0])
        self.assertIn("反证检查", llm_processor.prompts[0])

    def test_expression_layer_judges_generate_expression_fields_with_same_output_shape(self) -> None:
        from services.memory_pipeline.profile_fields import (
            LongTermExpressionJudge,
            ShortTermExpressionJudge,
            build_profile_context,
        )
        from services.memory_pipeline.types import MemoryState

        class CaptureLLMProcessor:
            def __init__(self) -> None:
                self.prompts = []

            def _call_llm_via_official_api(self, prompt, response_mime_type="application/json"):
                self.prompts.append(prompt)
                if "long_term_expression.personality_mbti" in prompt:
                    return {"value": "ENFP", "confidence": 0.41}
                if "short_term_expression.current_mood" in prompt:
                    return {"value": "neutral", "confidence": 0.52}
                return {"value": None, "confidence": 0.0}

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[
                {
                    "photo_id": "PHOTO_101",
                    "timestamp": "2026-03-01T21:00:00",
                    "vlm_analysis": {
                        "summary": "主角和朋友在夜间轻松聚会，表情自然放松",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "公寓"},
                        "event": {"activity": "聚会"},
                    },
                }
            ],
            events=[
                Event(
                    event_id="EVT_101",
                    date="2026-03-01",
                    time_range="21:00 - 22:00",
                    duration="1小时",
                    title="夜间聚会",
                    type="社交",
                    participants=["Person_001", "Person_002"],
                    location="公寓",
                    description="轻松聚会",
                    photo_count=2,
                    confidence=0.84,
                    reason="",
                    narrative_synthesis="主角和朋友在夜间轻松聚会",
                )
            ],
            relationships=[
                Relationship(
                    person_id="Person_002",
                    relationship_type="friend",
                    intimacy_score=0.46,
                    status="stable",
                    confidence=0.73,
                    reasoning="与主角有稳定社交互动",
                )
            ],
            groups=[],
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
        )

        context = build_profile_context(state)
        context["social_media_available"] = True
        llm_processor = CaptureLLMProcessor()

        long_term_decision = LongTermExpressionJudge().judge_field(
            "long_term_expression.personality_mbti",
            context,
            llm_processor=llm_processor,
        )
        short_term_decision = ShortTermExpressionJudge().judge_field(
            "short_term_expression.current_mood",
            context,
            llm_processor=llm_processor,
        )

        self.assertEqual(long_term_decision.final["value"], "ENFP")
        self.assertEqual(short_term_decision.final["value"], "neutral")
        self.assertIn("evidence", long_term_decision.final)
        self.assertIn("reasoning", short_term_decision.final)
        self.assertEqual(short_term_decision.final["evidence"]["photo_ids"], ["PHOTO_101"])

    def test_generate_structured_profile_runs_facts_before_expression_and_expression_reads_resolved_facts(self) -> None:
        from services.memory_pipeline.profile_fields import generate_structured_profile
        from services.memory_pipeline.types import MemoryState

        class CaptureLLMProcessor:
            def __init__(self) -> None:
                self.prompts = []

            def _call_llm_via_official_api(self, prompt, response_mime_type="application/json"):
                self.prompts.append(prompt)
                if "long_term_expression.attitude_style" in prompt:
                    return {"value": "casual_polished", "confidence": 0.56}
                if "long_term_facts.social_identity.education" in prompt:
                    return {"value": "college_student", "confidence": 0.77}
                return {"value": None, "confidence": 0.0}

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[
                {
                    "photo_id": "PHOTO_001",
                    "timestamp": "2026-03-01T10:00:00",
                    "vlm_analysis": {
                        "summary": "主角在校园教室内上课，穿着简洁休闲",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "campus classroom"},
                        "event": {"activity": "study"},
                    },
                },
                {
                    "photo_id": "PHOTO_002",
                    "timestamp": "2026-03-05T10:00:00",
                    "vlm_analysis": {
                        "summary": "主角再次在校园学习，穿搭风格稳定",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "campus"},
                        "event": {"activity": "study"},
                    },
                },
            ],
            events=[
                Event(
                    event_id="EVT_001",
                    date="2026-03-01",
                    time_range="10:00 - 11:00",
                    duration="1小时",
                    title="校园上课",
                    type="学习",
                    participants=["Person_001"],
                    location="campus classroom",
                    description="上课",
                    photo_count=1,
                    confidence=0.8,
                    reason="",
                    narrative_synthesis="主角在校园上课",
                ),
                Event(
                    event_id="EVT_002",
                    date="2026-03-05",
                    time_range="10:00 - 11:00",
                    duration="1小时",
                    title="校园学习",
                    type="学习",
                    participants=["Person_001"],
                    location="campus",
                    description="学习",
                    photo_count=1,
                    confidence=0.8,
                    reason="",
                    narrative_synthesis="主角在校园学习",
                ),
            ],
            relationships=[],
            groups=[],
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
        )

        llm_processor = CaptureLLMProcessor()
        result = generate_structured_profile(state, llm_processor=llm_processor)
        prompts = result["field_decisions"]

        education = result["structured"]["long_term_facts"]["social_identity"]["education"]
        attitude_style = result["structured"]["long_term_expression"]["attitude_style"]

        self.assertEqual(education["value"], "college_student")
        self.assertEqual(attitude_style["value"], "casual_polished")
        self.assertTrue(any(decision["field_key"] == "long_term_facts.social_identity.education" for decision in prompts))
        self.assertTrue(any(decision["field_key"] == "long_term_expression.attitude_style" for decision in prompts))
        expression_prompts = [prompt for prompt in llm_processor.prompts if "long_term_expression.attitude_style" in prompt]
        self.assertEqual(len(expression_prompts), 1)
        self.assertIn("已定稿facts", expression_prompts[0])
        self.assertIn("college_student", expression_prompts[0])

    def test_long_term_facts_reflection_rejects_burst_non_daily_event_inference(self) -> None:
        from services.memory_pipeline.profile_fields import LongTermFactsJudge, build_profile_context
        from services.memory_pipeline.types import MemoryState

        class CaptureLLMProcessor:
            def _call_llm_via_official_api(self, prompt, response_mime_type="application/json"):
                return {"value": "car_enthusiast", "confidence": 0.63}

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[
                {
                    "photo_id": f"PHOTO_{idx:03d}",
                    "timestamp": f"2026-03-02T1{idx%10}:00:00",
                    "vlm_analysis": {
                        "summary": "主角在车展展厅大量拍摄豪车与品牌展台",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "car expo hall"},
                        "event": {"activity": "车展参观"},
                    },
                }
                for idx in range(1, 6)
            ],
            events=[
                Event(
                    event_id="EVT_900",
                    date="2026-03-02",
                    time_range="10:00 - 16:00",
                    duration="6小时",
                    title="车展参观",
                    type="展会",
                    participants=["Person_001"],
                    location="car expo hall",
                    description="集中拍了很多豪车和品牌展台",
                    photo_count=5,
                    confidence=0.9,
                    reason="",
                    narrative_synthesis="主角在车展集中拍摄豪车",
                )
            ],
            relationships=[],
            groups=[],
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
        )

        context = build_profile_context(state)
        decision = LongTermFactsJudge().judge_field(
            "long_term_facts.material.brand_preference",
            context,
            llm_processor=CaptureLLMProcessor(),
        )

        self.assertIsNone(decision.final["value"])
        self.assertIn("非日常", decision.final["reasoning"])

    def test_expression_judge_silently_nulls_fields_that_require_social_media(self) -> None:
        from services.memory_pipeline.profile_fields import LongTermExpressionJudge, build_profile_context
        from services.memory_pipeline.types import MemoryState

        class ShouldNotBeCalledLLMProcessor:
            def _call_llm_via_official_api(self, prompt, response_mime_type="application/json"):
                raise AssertionError("LLM should not be called for silent expression fields without social media")

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[],
            events=[],
            relationships=[],
            groups=[],
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
        )

        context = build_profile_context(state)
        decision = LongTermExpressionJudge().judge_field(
            "long_term_expression.personality_mbti",
            context,
            llm_processor=ShouldNotBeCalledLLMProcessor(),
        )

        self.assertIsNone(decision.final["value"])
        self.assertIn("silent_by_missing_social_media", decision.final["evidence"]["constraint_notes"])
        self.assertIn("缺少社媒", decision.final["reasoning"])

    def test_expression_conflict_reflection_nulls_when_events_contradict_fact_led_expression(self) -> None:
        from services.memory_pipeline.profile_fields import LongTermExpressionJudge

        class CaptureLLMProcessor:
            def _call_llm_via_official_api(self, prompt, response_mime_type="application/json"):
                return {"value": "ENFP", "confidence": 0.58}

        context = {
            "primary_person_id": "Person_001",
            "events": [
                Event(
                    event_id="EVT_301",
                    date="2026-03-01",
                    time_range="09:00 - 18:00",
                    duration="9小时",
                    title="安静独处日",
                    type="独处",
                    participants=["Person_001"],
                    location="home",
                    description="长时间独处阅读和整理",
                    photo_count=3,
                    confidence=0.82,
                    reason="",
                    narrative_synthesis="主角安静独处",
                )
            ],
            "relationships": [
                Relationship(
                    person_id="Person_002",
                    relationship_type="friend",
                    intimacy_score=0.42,
                    status="stable",
                    confidence=0.74,
                    reasoning="近期有稳定同框社交",
                )
            ],
            "groups": [],
            "vlm_observations": [
                {
                    "photo_id": "PHOTO_301",
                    "summary": "主角在家安静看书",
                    "location": "home",
                    "activity": "reading alone",
                    "people": ["Person_001"],
                }
            ],
            "social_media_available": True,
            "resolved_facts": {
                "long_term_facts": {
                    "hobbies": {
                        "solo_vs_social": {
                            "value": "social",
                            "confidence": 0.8,
                            "evidence": {"feature_names": ["close_circle_size"]},
                            "reasoning": "先验上更偏社交",
                        }
                    }
                }
            },
        }

        decision = LongTermExpressionJudge().judge_field(
            "long_term_expression.personality_mbti",
            context,
            llm_processor=CaptureLLMProcessor(),
        )

        self.assertIsNone(decision.final["value"])
        self.assertIn("null_due_to_expression_conflict_reflection", decision.final["evidence"]["constraint_notes"])
        self.assertIn("未通过表达层反思", decision.final["reasoning"])

    def test_brand_preference_uses_item_level_paths_for_owned_and_fandom_candidates(self) -> None:
        from services.memory_pipeline.profile_fields import LongTermFactsJudge, build_profile_context
        from services.memory_pipeline.types import MemoryState

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[
                {
                    "photo_id": "PHOTO_001",
                    "timestamp": "2026-03-01T10:00:00",
                    "vlm_analysis": {
                        "summary": "主角卧室里摆着 Hello Kitty 地垫和 Hello Kitty 手机壳",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "home"},
                        "event": {"activity": "居家记录"},
                    },
                },
                {
                    "photo_id": "PHOTO_002",
                    "timestamp": "2026-03-05T10:00:00",
                    "vlm_analysis": {
                        "summary": "主角背着 Pinko 包自拍",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "campus"},
                        "event": {"activity": "自拍"},
                    },
                },
                {
                    "photo_id": "PHOTO_003",
                    "timestamp": "2026-03-06T10:00:00",
                    "vlm_analysis": {
                        "summary": "主角桌上摆着佰草集面霜和护肤用品",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "home"},
                        "event": {"activity": "护肤"},
                    },
                },
                {
                    "photo_id": "PHOTO_004",
                    "timestamp": "2026-03-08T20:00:00",
                    "vlm_analysis": {
                        "summary": "主角展示 EXO 应援棒和专辑收藏",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "home"},
                        "event": {"activity": "追星收藏"},
                    },
                },
                {
                    "photo_id": "PHOTO_005",
                    "timestamp": "2026-03-09T20:00:00",
                    "vlm_analysis": {
                        "summary": "商店货架上的 NIKE 广告海报和陈列商品",
                        "people": [],
                        "scene": {"location_detected": "mall"},
                        "event": {"activity": "逛街"},
                    },
                },
            ],
            events=[],
            relationships=[],
            groups=[],
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
        )

        decision = LongTermFactsJudge().judge_field(
            "long_term_facts.material.brand_preference",
            build_profile_context(state),
        )

        self.assertCountEqual(
            decision.final["value"],
            ["Hello Kitty", "Pinko", "佰草集", "EXO"],
        )
        self.assertNotIn("NIKE", decision.final["value"])

    def test_location_anchors_keep_named_real_places_and_drop_generic_scene_labels(self) -> None:
        from services.memory_pipeline.profile_fields import LongTermFactsJudge

        events = [
            Event(
                event_id="EVT_001",
                date="2026-03-01",
                time_range="10:00 - 11:00",
                duration="1小时",
                title="天坛游览",
                type="旅行",
                participants=["Person_001"],
                location="北京天坛公园",
                description="旅行",
                photo_count=2,
                confidence=0.8,
                reason="",
            ),
            Event(
                event_id="EVT_002",
                date="2026-03-02",
                time_range="10:00 - 11:00",
                duration="1小时",
                title="北京市记录",
                type="生活",
                participants=["Person_001"],
                location="北京市",
                description="城市记录",
                photo_count=1,
                confidence=0.8,
                reason="",
            ),
            Event(
                event_id="EVT_003",
                date="2026-03-03",
                time_range="10:00 - 11:00",
                duration="1小时",
                title="社区生活",
                type="生活",
                participants=["Person_001"],
                location="中山首府",
                description="小区生活",
                photo_count=1,
                confidence=0.8,
                reason="",
            ),
            Event(
                event_id="EVT_004",
                date="2026-03-04",
                time_range="10:00 - 11:00",
                duration="1小时",
                title="泛场景",
                type="其他",
                participants=["Person_001"],
                location="户外公共空间或公园",
                description="泛地点",
                photo_count=1,
                confidence=0.8,
                reason="",
            ),
            Event(
                event_id="EVT_005",
                date="2026-03-05",
                time_range="10:00 - 11:00",
                duration="1小时",
                title="泛室内",
                type="其他",
                participants=["Person_001"],
                location="室内环境",
                description="泛地点",
                photo_count=1,
                confidence=0.8,
                reason="",
            ),
            Event(
                event_id="EVT_006",
                date="2026-03-06",
                time_range="10:00 - 11:00",
                duration="1小时",
                title="泛场景重复",
                type="其他",
                participants=["Person_001"],
                location="户外公共空间或公园",
                description="泛地点",
                photo_count=1,
                confidence=0.8,
                reason="",
            ),
            Event(
                event_id="EVT_007",
                date="2026-03-07",
                time_range="10:00 - 11:00",
                duration="1小时",
                title="泛场景重复",
                type="其他",
                participants=["Person_001"],
                location="户外公共空间或公园",
                description="泛地点",
                photo_count=1,
                confidence=0.8,
                reason="",
            ),
        ]

        context = {
            "primary_person_id": "Person_001",
            "events": events,
            "relationships": [],
            "groups": [],
            "vlm_observations": [],
            "relationship_dossiers": [],
            "social_media_available": False,
            "resolved_facts": {},
        }
        decision = LongTermFactsJudge().judge_field("long_term_facts.geography.location_anchors", context)
        anchors = decision.final["value"] or []

        self.assertIn("北京天坛公园", anchors)
        self.assertIn("北京市", anchors)
        self.assertIn("中山首府", anchors)
        self.assertNotIn("户外公共空间或公园", anchors)
        self.assertNotIn("室内环境", anchors)

    def test_under_generation_reflection_marks_strong_evidence_null_outputs(self) -> None:
        from services.memory_pipeline.profile_fields import LongTermFactsJudge

        class NullReturningLLMProcessor:
            def _call_llm_via_official_api(self, prompt, response_mime_type="application/json"):
                return {"value": None, "confidence": 0.0}

        context = {
            "primary_person_id": "Person_001",
            "events": [],
            "relationships": [],
            "groups": [],
            "vlm_observations": [
                {
                    "photo_id": "PHOTO_001",
                    "summary": "主角多次自拍，外观线索稳定",
                    "location": "home",
                    "activity": "自拍",
                    "people": ["Person_001"],
                },
                {
                    "photo_id": "PHOTO_002",
                    "summary": "主角自拍，女性外观稳定",
                    "location": "campus",
                    "activity": "自拍",
                    "people": ["Person_001"],
                },
            ],
            "social_media_available": False,
            "resolved_facts": {},
        }

        decision = LongTermFactsJudge().judge_field(
            "long_term_facts.identity.gender",
            context,
            llm_processor=NullReturningLLMProcessor(),
        )

        self.assertEqual(decision.final["value"], "female")
        self.assertIn("under_generated_with_strong_evidence", decision.reflection_1["issues_found"])
        self.assertEqual(decision.reflection_1["under_generation_reason"], "value_generator_failed_after_candidates")
        self.assertTrue(decision.reflection_1["should_retry_generation"])
        self.assertEqual(decision.reflection_1["missing_extractor_type"], "missing_text_anchor_generator")

    def test_pets_field_requires_repeated_home_and_care_signals(self) -> None:
        from services.memory_pipeline.profile_fields import LongTermFactsJudge, build_profile_context
        from services.memory_pipeline.types import MemoryState

        sparse_state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[
                {
                    "photo_id": "PHOTO_001",
                    "timestamp": "2026-03-01T10:00:00",
                    "vlm_analysis": {
                        "summary": "街边看到一只猫",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "street"},
                        "event": {"activity": "路过"},
                    },
                }
            ],
            events=[],
            relationships=[],
            groups=[],
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
        )

        repeated_state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[
                {
                    "photo_id": "PHOTO_011",
                    "timestamp": "2026-03-01T21:00:00",
                    "vlm_analysis": {
                        "summary": "主角在卧室给猫喂食，旁边有猫粮和食盆",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "home"},
                        "event": {"activity": "喂猫"},
                    },
                },
                {
                    "photo_id": "PHOTO_012",
                    "timestamp": "2026-03-03T21:00:00",
                    "vlm_analysis": {
                        "summary": "主角清理猫砂盆，黑猫在家里活动",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "home"},
                        "event": {"activity": "清理猫砂"},
                    },
                },
                {
                    "photo_id": "PHOTO_013",
                    "timestamp": "2026-03-05T21:00:00",
                    "vlm_analysis": {
                        "summary": "主角抱着猫在卧室休息",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "卧室"},
                        "event": {"activity": "居家陪伴"},
                    },
                },
            ],
            events=[],
            relationships=[],
            groups=[],
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
        )

        sparse = LongTermFactsJudge().judge_field(
            "long_term_facts.relationships.pets",
            build_profile_context(sparse_state),
        )
        repeated = LongTermFactsJudge().judge_field(
            "long_term_facts.relationships.pets",
            build_profile_context(repeated_state),
        )

        self.assertIsNone(sparse.final["value"])
        self.assertEqual(repeated.final["value"], "cat")

    def test_recent_fields_use_item_level_candidate_extraction(self) -> None:
        from services.memory_pipeline.profile_fields import ShortTermFactsJudge, build_profile_context
        from services.memory_pipeline.types import MemoryState

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[
                {
                    "photo_id": "PHOTO_021",
                    "timestamp": "2026-03-10T20:00:00",
                    "vlm_analysis": {
                        "summary": "主角在桌面摆出塔罗牌进行占卜",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "home"},
                        "event": {"activity": "塔罗占卜"},
                    },
                },
                {
                    "photo_id": "PHOTO_022",
                    "timestamp": "2026-03-11T20:00:00",
                    "vlm_analysis": {
                        "summary": "主角使用 AI 工具咨询网络暴力和隐私泄露问题",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "home"},
                        "event": {"activity": "AI咨询"},
                    },
                },
            ],
            events=[
                Event(
                    event_id="EVT_021",
                    date="2026-03-10",
                    time_range="20:00 - 20:30",
                    duration="30分钟",
                    title="塔罗牌占卜",
                    type="自我探索",
                    participants=["Person_001"],
                    location="home",
                    description="主角反复摆放塔罗牌进行占卜",
                    photo_count=1,
                    confidence=0.8,
                    reason="",
                    narrative_synthesis="主角进行塔罗占卜",
                ),
                Event(
                    event_id="EVT_022",
                    date="2026-03-11",
                    time_range="21:00 - 21:30",
                    duration="30分钟",
                    title="AI 咨询网络安全",
                    type="咨询",
                    participants=["Person_001"],
                    location="home",
                    description="主角使用 AI 工具咨询网暴与隐私泄露问题",
                    photo_count=1,
                    confidence=0.8,
                    reason="",
                    narrative_synthesis="主角咨询网络安全问题",
                ),
            ],
            relationships=[],
            groups=[],
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
        )

        context = build_profile_context(state)
        habits = ShortTermFactsJudge().judge_field("short_term_facts.recent_habits", context)
        interests = ShortTermFactsJudge().judge_field("short_term_facts.recent_interests", context)

        self.assertEqual(habits.final["value"], ["tarot_reading", "ai_tool_usage"])
        self.assertEqual(interests.final["value"], ["network_safety"])

    def test_location_anchors_accept_international_named_places_from_meta_and_narrative(self) -> None:
        from services.memory_pipeline.profile_fields import LongTermFactsJudge, build_profile_context
        from services.memory_pipeline.types import MemoryState

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[
                {
                    "photo_id": "PHOTO_031",
                    "timestamp": "2026-03-01T10:00:00",
                    "vlm_analysis": {
                        "summary": "主角在 Berlin 街头散步",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "室内环境"},
                        "event": {"activity": "walk"},
                    },
                }
            ],
            events=[
                Event(
                    event_id="EVT_031",
                    date="2026-03-01",
                    time_range="10:00 - 11:00",
                    duration="1小时",
                    title="海外校园打卡",
                    type="生活",
                    participants=["Person_001"],
                    location="户外公共空间或公园",
                    description="主角在 Los Angeles 校园周边活动",
                    photo_count=1,
                    confidence=0.8,
                    reason="",
                    narrative_synthesis="主角在 Los Angeles 停留",
                    meta_info={"location_context": "Los Angeles"},
                    objective_fact={"scene_description": "University of Southern California campus area"},
                )
            ],
            relationships=[],
            groups=[],
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
        )

        decision = LongTermFactsJudge().judge_field(
            "long_term_facts.geography.location_anchors",
            build_profile_context(state),
        )

        self.assertIn("Los Angeles", decision.final["value"])
        self.assertIn("Berlin", decision.final["value"])
        self.assertNotIn("户外公共空间或公园", decision.final["value"])
        self.assertNotIn("室内环境", decision.final["value"])

    def test_location_anchors_do_not_create_anchors_from_generic_scene_words_in_low_trust_fields(self) -> None:
        from services.memory_pipeline.profile_fields import LongTermFactsJudge, build_profile_context
        from services.memory_pipeline.types import MemoryState

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[
                {
                    "photo_id": "PHOTO_034",
                    "timestamp": "2026-03-10T10:00:00",
                    "vlm_analysis": {
                        "summary": "主角在博物馆里参观",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "室内环境"},
                        "event": {"activity": "参观"},
                    },
                }
            ],
            events=[
                Event(
                    event_id="EVT_034",
                    date="2026-03-10",
                    time_range="10:00 - 11:00",
                    duration="1小时",
                    title="周末出行",
                    type="生活",
                    participants=["Person_001"],
                    location="户外公共空间或公园",
                    description="主角在北京市和朋友碰面，随后进入博物馆参观",
                    photo_count=1,
                    confidence=0.82,
                    reason="",
                    narrative_synthesis="主角在抓娃娃机区和大众化餐饮店短暂停留后回到北京市活动",
                    meta_info={"location_context": "大学教室或学习场所"},
                    objective_fact={"scene_description": "博物馆内部与抓娃娃机区"},
                )
            ],
            relationships=[],
            groups=[],
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
        )

        decision = LongTermFactsJudge().judge_field(
            "long_term_facts.geography.location_anchors",
            build_profile_context(state),
        )

        self.assertEqual(decision.final["value"], ["北京市"])
        self.assertNotIn("博物馆", decision.final["value"])
        self.assertNotIn("抓娃娃机区", decision.final["value"])
        self.assertNotIn("大众化餐饮店", decision.final["value"])
        self.assertNotIn("大学教室或学习场所", decision.final["value"])

    def test_frequent_activities_use_raw_event_and_vlm_clues_not_event_type_top3(self) -> None:
        from services.memory_pipeline.profile_fields import LongTermFactsJudge, build_profile_context
        from services.memory_pipeline.types import MemoryState

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[
                {
                    "photo_id": "PHOTO_041",
                    "timestamp": "2026-03-10T20:00:00",
                    "vlm_analysis": {
                        "summary": "主角在桌面摆出塔罗牌进行占卜",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "home"},
                        "event": {"activity": "塔罗占卜"},
                    },
                },
                {
                    "photo_id": "PHOTO_042",
                    "timestamp": "2026-03-11T20:00:00",
                    "vlm_analysis": {
                        "summary": "主角打开 AI 工具咨询隐私泄露和网络暴力问题",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "home"},
                        "event": {"activity": "AI 咨询"},
                    },
                },
                {
                    "photo_id": "PHOTO_043",
                    "timestamp": "2026-03-12T20:00:00",
                    "vlm_analysis": {
                        "summary": "主角再次摆放塔罗牌进行占卜",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "home"},
                        "event": {"activity": "塔罗占卜"},
                    },
                },
            ],
            events=[
                Event(
                    event_id="EVT_041",
                    date="2026-03-10",
                    time_range="20:00 - 20:30",
                    duration="30分钟",
                    title="塔罗占卜",
                    type="生活",
                    participants=["Person_001"],
                    location="home",
                    description="主角摆放塔罗牌进行占卜",
                    photo_count=1,
                    confidence=0.8,
                    reason="",
                    narrative_synthesis="主角进行塔罗占卜",
                    objective_fact={"scene_description": "tarot cards on desk"},
                ),
                Event(
                    event_id="EVT_042",
                    date="2026-03-11",
                    time_range="21:00 - 21:30",
                    duration="30分钟",
                    title="AI 咨询",
                    type="生活",
                    participants=["Person_001"],
                    location="home",
                    description="主角用 AI 工具咨询网络安全问题",
                    photo_count=1,
                    confidence=0.8,
                    reason="",
                    narrative_synthesis="主角咨询 AI 工具",
                    objective_fact={"scene_description": "AI assistant on laptop"},
                ),
                Event(
                    event_id="EVT_043",
                    date="2026-03-12",
                    time_range="21:00 - 21:30",
                    duration="30分钟",
                    title="再次塔罗占卜",
                    type="生活",
                    participants=["Person_001"],
                    location="home",
                    description="主角再次进行塔罗占卜",
                    photo_count=1,
                    confidence=0.8,
                    reason="",
                    narrative_synthesis="主角再次塔罗占卜",
                    objective_fact={"scene_description": "tarot reading at home"},
                ),
            ],
            relationships=[],
            groups=[],
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
        )

        decision = LongTermFactsJudge().judge_field(
            "long_term_facts.hobbies.frequent_activities",
            build_profile_context(state),
        )

        self.assertEqual(decision.final["value"], ["tarot_reading", "ai_tool_usage"])
        self.assertNotIn("生活", decision.final["value"])

    def test_text_anchor_candidates_restore_identity_and_social_identity_fields(self) -> None:
        from services.memory_pipeline.profile_fields import LongTermFactsJudge, build_profile_context
        from services.memory_pipeline.types import MemoryState

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[
                {
                    "photo_id": "PHOTO_051",
                    "timestamp": "2026-03-20T12:00:00",
                    "vlm_analysis": {
                        "summary": "主角的军事理论末考论文上写着姓名陈美伊、建筑艺术系，内容为中文",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "campus classroom"},
                        "event": {"activity": "study"},
                    },
                },
                {
                    "photo_id": "PHOTO_052",
                    "timestamp": "2026-03-22T12:00:00",
                    "vlm_analysis": {
                        "summary": "主角在教室里准备课程作业，建筑艺术系学生，中文文本清晰可见",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "campus"},
                        "event": {"activity": "study"},
                    },
                },
            ],
            events=[
                Event(
                    event_id="EVT_051",
                    date="2026-03-20",
                    time_range="12:00 - 13:00",
                    duration="1小时",
                    title="军事理论末考论文",
                    type="学习",
                    participants=["Person_001"],
                    location="campus classroom",
                    description="姓名：陈美伊；建筑艺术系；课程论文",
                    photo_count=1,
                    confidence=0.8,
                    reason="",
                    narrative_synthesis="主角提交军事理论论文",
                    objective_fact={"scene_description": "建筑艺术系学生课程论文，中文文本明显"},
                ),
                Event(
                    event_id="EVT_052",
                    date="2026-03-22",
                    time_range="12:00 - 13:00",
                    duration="1小时",
                    title="课程作业准备",
                    type="学习",
                    participants=["Person_001"],
                    location="campus",
                    description="主角作为学生准备建筑与艺术相关课程作业",
                    photo_count=1,
                    confidence=0.8,
                    reason="",
                    narrative_synthesis="主角准备建筑艺术相关作业",
                    objective_fact={"scene_description": "student preparing architecture art coursework"},
                ),
            ],
            relationships=[],
            groups=[],
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
        )

        context = build_profile_context(state)
        judge = LongTermFactsJudge()

        self.assertEqual(judge.judge_field("long_term_facts.identity.name", context).final["value"], "陈美伊")
        self.assertEqual(judge.judge_field("long_term_facts.identity.role", context).final["value"], "student")
        self.assertEqual(judge.judge_field("long_term_facts.social_identity.education", context).final["value"], "higher_education")
        self.assertEqual(judge.judge_field("long_term_facts.social_identity.career", context).final["value"], "architecture_art_student")
        self.assertEqual(judge.judge_field("long_term_facts.social_identity.language_culture", context).final["value"], "mandarin")

    def test_single_formal_document_anchor_can_restore_education_and_career(self) -> None:
        from services.memory_pipeline.profile_fields import LongTermFactsJudge, build_profile_context
        from services.memory_pipeline.types import MemoryState

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[
                {
                    "photo_id": "PHOTO_058",
                    "timestamp": "2026-03-20T12:00:00",
                    "vlm_analysis": {
                        "summary": "主角课程论文清晰可见：姓名陈美伊，学号202603，建筑艺术系，全文为中文",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "campus classroom"},
                        "event": {"activity": "study"},
                    },
                }
            ],
            events=[
                Event(
                    event_id="EVT_058",
                    date="2026-03-20",
                    time_range="12:00 - 13:00",
                    duration="1小时",
                    title="课程论文",
                    type="学习",
                    participants=["Person_001"],
                    location="campus classroom",
                    description="姓名：陈美伊；学号：202603；建筑艺术系；课程论文",
                    photo_count=1,
                    confidence=0.83,
                    reason="",
                    narrative_synthesis="主角展示建筑艺术系课程论文",
                    objective_fact={"scene_description": "formal academic document for architecture art coursework"},
                )
            ],
            relationships=[],
            groups=[],
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
        )

        context = build_profile_context(state)
        judge = LongTermFactsJudge()

        self.assertEqual(judge.judge_field("long_term_facts.social_identity.education", context).final["value"], "higher_education")
        self.assertEqual(judge.judge_field("long_term_facts.social_identity.career", context).final["value"], "architecture_art_student")

    def test_under_generation_retry_can_recover_gender_from_text_anchor(self) -> None:
        from services.memory_pipeline.profile_fields import LongTermFactsJudge

        class NullReturningLLMProcessor:
            def _call_llm_via_official_api(self, prompt, response_mime_type="application/json"):
                return {"value": None, "confidence": 0.0}

        context = {
            "primary_person_id": "Person_001",
            "events": [],
            "relationships": [],
            "groups": [],
            "vlm_observations": [
                {
                    "photo_id": "PHOTO_061",
                    "summary": "主角女生自拍，宿舍镜子前自拍",
                    "location": "home",
                    "activity": "自拍",
                    "people": ["Person_001"],
                },
                {
                    "photo_id": "PHOTO_062",
                    "summary": "主角是女生，在校园自拍，女性外观稳定",
                    "location": "campus",
                    "activity": "自拍",
                    "people": ["Person_001"],
                },
            ],
            "social_media_available": False,
            "resolved_facts": {},
        }

        decision = LongTermFactsJudge().judge_field(
            "long_term_facts.identity.gender",
            context,
            llm_processor=NullReturningLLMProcessor(),
        )

        self.assertEqual(decision.final["value"], "female")
        self.assertTrue(decision.reflection_1["should_retry_generation"])
        self.assertEqual(decision.reflection_1["under_generation_reason"], "value_generator_failed_after_candidates")
        self.assertEqual(decision.reflection_1["missing_extractor_type"], "missing_text_anchor_generator")

    def test_recent_habits_filters_baseline_selfie_noise(self) -> None:
        from services.memory_pipeline.profile_fields import ShortTermFactsJudge, build_profile_context
        from services.memory_pipeline.types import MemoryState

        vlm_results = []
        for idx in range(6):
            vlm_results.append(
                {
                    "photo_id": f"PHOTO_S_{idx:03d}",
                    "timestamp": f"2026-03-0{idx+1}T10:00:00",
                    "vlm_analysis": {
                        "summary": "主角镜前自拍",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "home"},
                        "event": {"activity": "自拍"},
                    },
                }
            )
        vlm_results.extend(
            [
                {
                    "photo_id": "PHOTO_091",
                    "timestamp": "2026-03-20T20:00:00",
                    "vlm_analysis": {
                        "summary": "主角摆放塔罗牌进行占卜",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "home"},
                        "event": {"activity": "塔罗占卜"},
                    },
                },
                {
                    "photo_id": "PHOTO_092",
                    "timestamp": "2026-03-21T20:00:00",
                    "vlm_analysis": {
                        "summary": "主角再次使用 AI 工具咨询隐私泄露和网暴问题",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "home"},
                        "event": {"activity": "AI咨询"},
                    },
                },
            ]
        )

        events = [
            Event(
                event_id="EVT_091",
                date="2026-03-20",
                time_range="20:00 - 20:30",
                duration="30分钟",
                title="塔罗占卜",
                type="生活",
                participants=["Person_001"],
                location="home",
                description="主角进行塔罗占卜",
                photo_count=1,
                confidence=0.8,
                reason="",
                narrative_synthesis="主角摆放塔罗牌占卜",
            ),
            Event(
                event_id="EVT_092",
                date="2026-03-21",
                time_range="21:00 - 21:30",
                duration="30分钟",
                title="AI 咨询",
                type="生活",
                participants=["Person_001"],
                location="home",
                description="主角使用 AI 工具咨询隐私泄露和网暴问题",
                photo_count=1,
                confidence=0.8,
                reason="",
                narrative_synthesis="主角进行 AI 咨询",
            ),
        ]

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=vlm_results,
            events=events,
            relationships=[],
            groups=[],
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
        )
        decision = ShortTermFactsJudge().judge_field("short_term_facts.recent_habits", build_profile_context(state))

        self.assertIn("tarot_reading", decision.final["value"])
        self.assertIn("ai_tool_usage", decision.final["value"])
        self.assertNotIn("selfie", decision.final["value"])

    def test_recent_interests_prioritize_recent_text_topics_over_long_term_visual_interests(self) -> None:
        from services.memory_pipeline.profile_fields import ShortTermFactsJudge, build_profile_context
        from services.memory_pipeline.types import MemoryState

        vlm_results = [
            {
                "photo_id": f"PHOTO_F_{idx:03d}",
                "timestamp": f"2026-03-0{idx+1}T10:00:00",
                "vlm_analysis": {
                    "summary": "主角展示穿搭和美甲自拍",
                    "people": [{"person_id": "Person_001"}],
                    "scene": {"location_detected": "home"},
                    "event": {"activity": "自拍"},
                },
            }
            for idx in range(5)
        ]
        vlm_results.extend(
            [
                {
                    "photo_id": "PHOTO_101",
                    "timestamp": "2026-03-22T20:00:00",
                    "vlm_analysis": {
                        "summary": "主角使用 AI 工具咨询网暴和隐私泄露",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "home"},
                        "event": {"activity": "AI咨询"},
                    },
                },
                {
                    "photo_id": "PHOTO_102",
                    "timestamp": "2026-03-23T20:00:00",
                    "vlm_analysis": {
                        "summary": "主角再次查看关于网络安全和隐私泄露的内容",
                        "people": [{"person_id": "Person_001"}],
                        "scene": {"location_detected": "home"},
                        "event": {"activity": "信息搜索"},
                    },
                },
            ]
        )
        events = [
            Event(
                event_id="EVT_101",
                date="2026-03-22",
                time_range="20:00 - 20:30",
                duration="30分钟",
                title="AI 咨询网络安全",
                type="咨询",
                participants=["Person_001"],
                location="home",
                description="主角反复咨询网暴、隐私泄露和网络安全",
                photo_count=1,
                confidence=0.8,
                reason="",
                narrative_synthesis="主角咨询网络安全问题",
            ),
            Event(
                event_id="EVT_102",
                date="2026-03-23",
                time_range="20:00 - 20:30",
                duration="30分钟",
                title="继续研究隐私问题",
                type="咨询",
                participants=["Person_001"],
                location="home",
                description="主角继续阅读网暴和隐私泄露相关内容",
                photo_count=1,
                confidence=0.8,
                reason="",
                narrative_synthesis="主角继续研究隐私问题",
            ),
        ]

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=vlm_results,
            events=events,
            relationships=[],
            groups=[],
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
        )
        decision = ShortTermFactsJudge().judge_field("short_term_facts.recent_interests", build_profile_context(state))

        self.assertEqual(decision.final["value"], ["network_safety"])

    def test_expression_fields_require_real_candidates_not_generic_support_count(self) -> None:
        from services.memory_pipeline.profile_fields import LongTermExpressionJudge

        context = {
            "primary_person_id": "Person_001",
            "events": [
                Event(
                    event_id="EVT_120",
                    date="2026-03-21",
                    time_range="10:00 - 11:00",
                    duration="1小时",
                    title="普通记录",
                    type="生活",
                    participants=["Person_001"],
                    location="home",
                    description="主角在室内普通活动",
                    photo_count=1,
                    confidence=0.8,
                    reason="",
                    narrative_synthesis="主角在室内普通活动",
                )
            ],
            "relationships": [],
            "groups": [],
            "vlm_observations": [
                {
                    "photo_id": "PHOTO_120",
                    "summary": "主角在家里普通记录生活",
                    "location": "home",
                    "activity": "生活记录",
                    "people": ["Person_001"],
                }
            ],
            "social_media_available": True,
            "resolved_facts": {"long_term_facts": {}, "short_term_facts": {}},
        }

        decision = LongTermExpressionJudge().judge_field("long_term_expression.attitude_style", context)

        self.assertFalse(decision.gate_result["strong_evidence_met"])
        self.assertIsNone(decision.reflection_1["under_generation_reason"])

    def test_close_circle_size_override_can_use_high_quality_dossiers(self) -> None:
        from services.memory_pipeline.profile_fields import LongTermFactsJudge
        from services.memory_pipeline.types import RelationshipDossier

        dossier = RelationshipDossier(
            person_id="Person_002",
            person_kind="real_person",
            memory_value="candidate",
            photo_count=8,
            time_span_days=80,
            recent_gap_days=2,
            monthly_frequency=3.5,
            scene_profile={"scenes": ["宿舍", "咖啡馆"], "private_scene_ratio": 0.3, "dominant_scene_ratio": 0.6, "with_user_only": True},
            interaction_signals=["selfie_together"],
            shared_events=[{"event_id": "EVT_A"}, {"event_id": "EVT_B"}],
            trend_detail={"direction": "up"},
            co_appearing_persons=[],
            anomalies=[],
            evidence_refs=[],
            retention_decision="suppress",
            retention_reason="screen_or_virtual_context",
            relationship_result={"relationship_type": "close_friend", "confidence": 0.72},
        )

        decision = LongTermFactsJudge().judge_field(
            "long_term_facts.relationships.close_circle_size",
            {
                "primary_person_id": "Person_001",
                "events": [],
                "relationships": [],
                "groups": [],
                "vlm_observations": [],
                "relationship_dossiers": [dossier],
                "social_media_available": False,
                "resolved_facts": {},
            },
        )

        self.assertEqual(decision.final["value"], 1)

    def test_screen_or_virtual_context_requires_dominant_mediated_ratio(self) -> None:
        from services.memory_pipeline.relationships import _determine_retention
        from services.memory_pipeline.types import RelationshipDossier

        dossier = RelationshipDossier(
            person_id="Person_002",
            person_kind="real_person",
            memory_value="candidate",
            photo_count=6,
            time_span_days=40,
            recent_gap_days=3,
            monthly_frequency=4.2,
            scene_profile={
                "scenes": ["wechat screen", "宿舍", "咖啡馆", "校园"],
                "private_scene_ratio": 0.5,
                "dominant_scene_ratio": 0.4,
                "with_user_only": True,
            },
            interaction_signals=["selfie_together", "聊天"],
            shared_events=[
                {"event_id": "EVT_071"},
                {"event_id": "EVT_072"},
            ],
            trend_detail={"direction": "up"},
            co_appearing_persons=[],
            anomalies=[],
            evidence_refs=[],
        )

        relationship = Relationship(
            person_id="Person_002",
            relationship_type="close_friend",
            intimacy_score=0.73,
            status="growing",
            confidence=0.81,
            reasoning="test",
            shared_events=[{"event_id": "EVT_071"}, {"event_id": "EVT_072"}],
            evidence={},
        )

        retention, reason = _determine_retention(dossier, relationship)

        self.assertEqual(retention, "keep")
        self.assertEqual(reason, "relationship_retained")

    def test_extract_events_from_state_calls_llm_once_with_full_album(self) -> None:
        from services.memory_pipeline.events import extract_events_from_state
        from services.memory_pipeline.types import MemoryState

        call_sizes = []

        class StubLLMProcessor:
            def extract_events(self, vlm_results):
                call_sizes.append(len(vlm_results))
                return [
                    Event(
                        event_id="TEMP_A",
                        date="2026-03-01",
                        time_range="10:00 - 11:00",
                        duration="1小时",
                        title="上午咖啡",
                        type="社交",
                        participants=["Person_001", "Person_002"],
                        location="咖啡馆",
                        description="上午见面",
                        photo_count=2,
                        confidence=0.8,
                        reason="stub",
                    ),
                    Event(
                        event_id="TEMP_B",
                        date="2026-03-01",
                        time_range="20:00 - 21:00",
                        duration="1小时",
                        title="晚间回家",
                        type="生活",
                        participants=["Person_001", "Person_003"],
                        location="公寓",
                        description="晚间回家",
                        photo_count=1,
                        confidence=0.7,
                        reason="stub",
                    ),
                ]

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[
                {
                    "photo_id": "PHOTO_001",
                    "timestamp": "2026-03-01T10:00:00",
                    "location": {"name": "上海"},
                    "vlm_analysis": {
                        "people": [{"person_id": "Person_001"}, {"person_id": "Person_002"}],
                        "scene": {"location_detected": "咖啡馆"},
                    },
                },
                {
                    "photo_id": "PHOTO_002",
                    "timestamp": "2026-03-01T11:00:00",
                    "location": {"name": "上海"},
                    "vlm_analysis": {
                        "people": [{"person_id": "Person_001"}, {"person_id": "Person_002"}],
                        "scene": {"location_detected": "咖啡馆"},
                    },
                },
                {
                    "photo_id": "PHOTO_003",
                    "timestamp": "2026-03-01T20:00:00",
                    "location": {"name": "上海"},
                    "vlm_analysis": {
                        "people": [{"person_id": "Person_001"}, {"person_id": "Person_003"}],
                        "scene": {"location_detected": "公寓"},
                    },
                },
            ],
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
        )

        events = extract_events_from_state(state, StubLLMProcessor())

        self.assertEqual(call_sizes, [3])
        self.assertEqual([event.event_id for event in events], ["EVT_001", "EVT_002"])

    def test_orchestrator_keeps_internal_group_artifacts_out_of_relationship_schema(self) -> None:
        from services.memory_pipeline.orchestrator import run_memory_pipeline
        from services.memory_pipeline.types import MemoryState, PersonScreening

        class StubLLMProcessor:
            def __init__(self):
                self.primary_person_id = "Person_001"

            def extract_events(self, vlm_results):
                return [
                    Event(
                        event_id="EVT_001",
                        date="2026-03-01",
                        time_range="19:00 - 22:00",
                        duration="3小时",
                        title="Formal Night",
                        type="社交",
                        participants=["Person_001", "Person_002", "Person_003"],
                        location="sorority house",
                        description="group photo in front of Greek letters",
                        photo_count=4,
                        confidence=0.88,
                        reason="",
                        narrative_synthesis="sorority formal night",
                    )
                ]

            def _collect_relationship_evidence(self, person_id, vlm_results, events=None):
                return {
                    "photo_count": 4,
                    "time_span": "30天",
                    "time_span_days": 30,
                    "recent_gap_days": 1,
                    "scenes": ["sorority house"],
                    "private_scene_ratio": 0.0,
                    "dominant_scene_ratio": 1.0,
                    "interaction_behavior": ["合影"],
                    "weekend_frequency": "高",
                    "with_user_only": False,
                    "sample_scenes": [],
                    "contact_types": ["selfie_together"],
                    "rela_events": [
                        {
                            "event_id": "EVT_001",
                            "date": "2026-03-01",
                            "title": "Formal Night",
                            "location": "sorority house",
                            "photo_count": 4,
                            "description": "formal",
                            "participants": ["Person_001", "Person_002", "Person_003"],
                            "narrative_synthesis": "sorority formal night",
                            "social_dynamics": [],
                        }
                    ],
                    "monthly_frequency": 4.0,
                    "trend_detail": {"direction": "flat"},
                    "co_appearing_persons": [{"person_id": "Person_003" if person_id == "Person_002" else "Person_002", "co_count": 4, "co_ratio": 1.0}],
                    "anomalies": [],
                }

            def _infer_relationship(self, person_id, evidence, vlm_results, face_db):
                return Relationship(
                    person_id=person_id,
                    relationship_type="friend",
                    intimacy_score=0.72,
                    status="stable",
                    confidence=0.86,
                    reasoning="stable social relationship",
                    shared_events=[{"event_id": "EVT_001", "date": "2026-03-01", "narrative": "sorority formal night"}],
                    evidence={"decision_trace": {"final_relationship_type": "friend"}},
                )

        state = MemoryState(
            photos=[],
            face_db={
                "Person_001": {"photo_count": 10},
                "Person_002": {"photo_count": 4},
                "Person_003": {"photo_count": 4},
            },
            vlm_results=[
                {
                    "photo_id": "PHOTO_001",
                    "timestamp": "2026-03-01T19:00:00",
                    "location": {"name": "sorority house"},
                    "vlm_analysis": {
                        "summary": "group photo at sorority formal",
                        "people": [{"person_id": "Person_001"}, {"person_id": "Person_002"}, {"person_id": "Person_003"}],
                        "scene": {"location_detected": "sorority house"},
                        "event": {"activity": "formal night", "social_context": "club"},
                    },
                }
            ],
            screening={
                "Person_001": PersonScreening(person_id="Person_001", person_kind="real_person", memory_value="core"),
                "Person_002": PersonScreening(person_id="Person_002", person_kind="real_person", memory_value="candidate"),
                "Person_003": PersonScreening(person_id="Person_003", person_kind="real_person", memory_value="candidate"),
            },
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
        )

        result = run_memory_pipeline(state=state, llm_processor=StubLLMProcessor())

        self.assertEqual(result["relationships"][0].person_id, "Person_002")
        self.assertGreaterEqual(len(result["internal_artifacts"]["group_artifacts"]), 1)
        serialized = serialize_relationship(result["relationships"][0])
        self.assertNotIn("group_eligible", serialized)
        self.assertNotIn("group_block_reason", serialized)
        self.assertNotIn("group_weight", serialized)

    def test_orchestrator_populates_primary_reflection_even_with_preseeded_primary_decision(self) -> None:
        from services.memory_pipeline.orchestrator import run_memory_pipeline
        from services.memory_pipeline.types import MemoryState, PersonScreening

        class StubLLMProcessor:
            def __init__(self):
                self.primary_person_id = "Person_001"

            def extract_events(self, vlm_results):
                return []

            def _collect_relationship_evidence(self, person_id, vlm_results, events=None):
                return {
                    "photo_count": 1,
                    "time_span": "1天",
                    "time_span_days": 1,
                    "recent_gap_days": 0,
                    "scenes": ["校园"],
                    "private_scene_ratio": 0.0,
                    "dominant_scene_ratio": 1.0,
                    "interaction_behavior": [],
                    "weekend_frequency": "低",
                    "with_user_only": False,
                    "sample_scenes": [],
                    "contact_types": [],
                    "rela_events": [],
                    "monthly_frequency": 1.0,
                    "trend_detail": {"direction": "flat"},
                    "co_appearing_persons": [],
                    "anomalies": [],
                }

            def _infer_relationship(self, person_id, evidence, vlm_results, face_db):
                return Relationship(
                    person_id=person_id,
                    relationship_type="acquaintance",
                    intimacy_score=0.2,
                    status="new",
                    confidence=0.4,
                    reasoning="fallback",
                    shared_events=[],
                    evidence={"decision_trace": {}},
                )

        state = MemoryState(
            photos=[],
            face_db={"Person_001": {"photo_count": 5}, "Person_002": {"photo_count": 1}},
            vlm_results=[],
            screening={
                "Person_001": PersonScreening(person_id="Person_001", person_kind="real_person", memory_value="core"),
                "Person_002": PersonScreening(person_id="Person_002", person_kind="real_person", memory_value="candidate"),
            },
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "preseeded"},
            events=[],
        )

        result = run_memory_pipeline(state=state, llm_processor=StubLLMProcessor())

        self.assertIn("primary_reflection", result["internal_artifacts"])
        self.assertIn("triggered", result["internal_artifacts"]["primary_reflection"])

    def test_orchestrator_keeps_precomputed_events_without_reextracting(self) -> None:
        from services.memory_pipeline.orchestrator import run_memory_pipeline
        from services.memory_pipeline.types import MemoryState, PersonScreening

        class StubLLMProcessor:
            def __init__(self):
                self.primary_person_id = "Person_001"

            def extract_events(self, vlm_results):
                raise AssertionError("should not re-extract events when state.events already exists")

            def _collect_relationship_evidence(self, person_id, vlm_results, events=None):
                return {
                    "photo_count": 1,
                    "time_span": "1天",
                    "time_span_days": 1,
                    "recent_gap_days": 0,
                    "scenes": ["校园"],
                    "private_scene_ratio": 0.0,
                    "dominant_scene_ratio": 1.0,
                    "interaction_behavior": [],
                    "weekend_frequency": "低",
                    "with_user_only": False,
                    "sample_scenes": [],
                    "contact_types": [],
                    "rela_events": [{"event_id": "EVT_900", "date": "2026-03-01", "title": "预计算事件", "location": "校园", "photo_count": 1, "description": "", "participants": ["Person_001", person_id], "narrative_synthesis": "", "social_dynamics": []}],
                    "monthly_frequency": 1.0,
                    "trend_detail": {"direction": "flat"},
                    "co_appearing_persons": [],
                    "anomalies": [],
                }

            def _infer_relationship(self, person_id, evidence, vlm_results, face_db):
                return Relationship(
                    person_id=person_id,
                    relationship_type="acquaintance",
                    intimacy_score=0.2,
                    status="stable",
                    confidence=0.5,
                    reasoning="fallback",
                    shared_events=[{"event_id": "EVT_900", "date": "2026-03-01", "narrative": "预计算事件"}],
                    evidence={"decision_trace": {}},
                )

        precomputed_events = [
            Event(
                event_id="EVT_900",
                date="2026-03-01",
                time_range="12:00 - 13:00",
                duration="1小时",
                title="预计算事件",
                type="社交",
                participants=["Person_001", "Person_002"],
                location="校园",
                description="预先装载",
                photo_count=1,
                confidence=0.8,
                reason="",
            )
        ]

        state = MemoryState(
            photos=[],
            face_db={"Person_001": {"photo_count": 5}, "Person_002": {"photo_count": 1}},
            vlm_results=[],
            screening={
                "Person_001": PersonScreening(person_id="Person_001", person_kind="real_person", memory_value="core"),
                "Person_002": PersonScreening(person_id="Person_002", person_kind="real_person", memory_value="candidate"),
            },
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.9, "evidence": {}, "reasoning": "test"},
            events=precomputed_events,
        )

        result = run_memory_pipeline(state=state, llm_processor=StubLLMProcessor())

        self.assertEqual(result["events"][0].event_id, "EVT_900")

    def test_orchestrator_filters_low_evidence_relationships_from_official_output(self) -> None:
        from services.memory_pipeline.orchestrator import run_memory_pipeline
        from services.memory_pipeline.types import MemoryState, PersonScreening

        class StubLLMProcessor:
            def __init__(self):
                self.primary_person_id = "Person_001"

            def extract_events(self, vlm_results):
                return []

            def _collect_relationship_evidence(self, person_id, vlm_results, events=None):
                return {
                    "photo_count": 1,
                    "time_span": "1天",
                    "time_span_days": 1,
                    "recent_gap_days": 0,
                    "scenes": ["海滨沙滩"],
                    "private_scene_ratio": 0.0,
                    "dominant_scene_ratio": 1.0,
                    "interaction_behavior": ["no_contact"],
                    "weekend_frequency": "低",
                    "with_user_only": False,
                    "sample_scenes": [],
                    "contact_types": ["no_contact"],
                    "rela_events": [
                        {
                            "event_id": "EVT_001",
                            "date": "2026-03-01",
                            "title": "海边合影",
                            "location": "海滨沙滩",
                            "photo_count": 1,
                            "description": "单次同框",
                            "participants": ["Person_001", person_id],
                            "narrative_synthesis": "海边偶遇合影",
                            "social_dynamics": [],
                        }
                    ],
                    "monthly_frequency": 1.0,
                    "trend_detail": {"direction": "flat"},
                    "co_appearing_persons": [],
                    "anomalies": [],
                }

            def _infer_relationship(self, person_id, evidence, vlm_results, face_db):
                return Relationship(
                    person_id=person_id,
                    relationship_type="activity_buddy",
                    intimacy_score=0.22,
                    status="new",
                    confidence=0.46,
                    reasoning="low-signal fallback",
                    shared_events=[{"event_id": "EVT_001", "date": "2026-03-01", "narrative": "海边偶遇合影"}],
                    evidence={"photo_count": 1, "decision_trace": {"final_relationship_type": "activity_buddy"}},
                )

        state = MemoryState(
            photos=[],
            face_db={"Person_001": {"photo_count": 12}, "Person_002": {"photo_count": 1}},
            vlm_results=[],
            screening={
                "Person_001": PersonScreening(person_id="Person_001", person_kind="real_person", memory_value="core"),
                "Person_002": PersonScreening(person_id="Person_002", person_kind="real_person", memory_value="candidate"),
            },
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.95, "evidence": {}, "reasoning": "test"},
        )

        result = run_memory_pipeline(state=state, llm_processor=StubLLMProcessor())

        self.assertEqual(result["relationships"], [])
        self.assertEqual(result["internal_artifacts"]["relationship_dossiers"][0]["retention_decision"], "suppress")

    def test_multi_agent_relationship_status_corrects_short_term_repeated_connection(self) -> None:
        from services.memory_pipeline.relationships import build_relationship_dossiers, infer_relationships_from_dossiers
        from services.memory_pipeline.types import MemoryState, PersonScreening

        class StubLLMProcessor:
            def _collect_relationship_evidence(self, person_id, vlm_results, events=None):
                return {
                    "photo_count": 4,
                    "time_span": "21天",
                    "time_span_days": 21,
                    "recent_gap_days": 2,
                    "scenes": ["咖啡馆", "宿舍"],
                    "private_scene_ratio": 0.5,
                    "dominant_scene_ratio": 0.5,
                    "interaction_behavior": ["聊天", "自拍"],
                    "weekend_frequency": "高",
                    "with_user_only": True,
                    "sample_scenes": [],
                    "contact_types": ["selfie_together"],
                    "rela_events": [
                        {
                            "event_id": "EVT_001",
                            "date": "2026-03-01",
                            "title": "咖啡馆聊天",
                            "location": "咖啡馆",
                            "photo_count": 2,
                            "description": "多次单独见面",
                            "participants": ["Person_001", person_id],
                            "narrative_synthesis": "一起喝咖啡",
                            "social_dynamics": [],
                        },
                        {
                            "event_id": "EVT_002",
                            "date": "2026-03-20",
                            "title": "宿舍自拍",
                            "location": "宿舍",
                            "photo_count": 2,
                            "description": "宿舍自拍",
                            "participants": ["Person_001", person_id],
                            "narrative_synthesis": "宿舍一起自拍",
                            "social_dynamics": [],
                        },
                    ],
                    "monthly_frequency": 4.0,
                    "trend_detail": {"direction": "up"},
                    "co_appearing_persons": [],
                    "anomalies": [],
                }

            def _infer_relationship(self, person_id, evidence, vlm_results, face_db):
                return Relationship(
                    person_id=person_id,
                    relationship_type="close_friend",
                    intimacy_score=0.73,
                    status="new",
                    confidence=0.79,
                    reasoning="llm stayed conservative on short window",
                    shared_events=[
                        {"event_id": "EVT_001", "date": "2026-03-01", "narrative": "一起喝咖啡"},
                        {"event_id": "EVT_002", "date": "2026-03-20", "narrative": "宿舍一起自拍"},
                    ],
                    evidence={"photo_count": 4, "decision_trace": {"final_status": "new"}},
                )

        state = MemoryState(
            photos=[],
            face_db={"Person_001": {"photo_count": 12}, "Person_002": {"photo_count": 4}},
            vlm_results=[],
            screening={
                "Person_001": PersonScreening(person_id="Person_001", person_kind="real_person", memory_value="core"),
                "Person_002": PersonScreening(person_id="Person_002", person_kind="real_person", memory_value="candidate"),
            },
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.95, "evidence": {}, "reasoning": "test"},
            events=[],
        )

        dossiers = build_relationship_dossiers(state=state, llm_processor=StubLLMProcessor())
        relationships, _ = infer_relationships_from_dossiers(
            state=state,
            llm_processor=StubLLMProcessor(),
            dossiers=dossiers,
        )

        self.assertEqual(len(relationships), 1)
        self.assertEqual(relationships[0].status, "growing")

    def test_single_private_scene_without_contact_signal_is_not_strong_evidence(self) -> None:
        from services.memory_pipeline.orchestrator import run_memory_pipeline
        from services.memory_pipeline.types import MemoryState, PersonScreening

        class StubLLMProcessor:
            def __init__(self):
                self.primary_person_id = "Person_001"

            def extract_events(self, vlm_results):
                return []

            def _collect_relationship_evidence(self, person_id, vlm_results, events=None):
                return {
                    "photo_count": 1,
                    "time_span": "1天",
                    "time_span_days": 1,
                    "recent_gap_days": 0,
                    "scenes": ["室内试衣间或卧室"],
                    "private_scene_ratio": 1.0,
                    "dominant_scene_ratio": 1.0,
                    "interaction_behavior": ["Person_015 holding 手机", "Person_015 standing_in_front_of 镜子"],
                    "weekend_frequency": "低",
                    "with_user_only": True,
                    "sample_scenes": [],
                    "contact_types": ["no_contact"],
                    "rela_events": [],
                    "monthly_frequency": 1.0,
                    "trend_detail": {"direction": "flat"},
                    "co_appearing_persons": [],
                    "anomalies": [],
                }

            def _infer_relationship(self, person_id, evidence, vlm_results, face_db):
                return Relationship(
                    person_id=person_id,
                    relationship_type="activity_buddy",
                    intimacy_score=0.26,
                    status="new",
                    confidence=0.50,
                    reasoning="mirror scene fallback",
                    shared_events=[],
                    evidence={"photo_count": 1, "decision_trace": {"final_relationship_type": "activity_buddy"}},
                )

        state = MemoryState(
            photos=[],
            face_db={"Person_001": {"photo_count": 12}, "Person_015": {"photo_count": 1}},
            vlm_results=[],
            screening={
                "Person_001": PersonScreening(person_id="Person_001", person_kind="real_person", memory_value="core"),
                "Person_015": PersonScreening(person_id="Person_015", person_kind="real_person", memory_value="candidate"),
            },
            primary_decision={"mode": "person_id", "primary_person_id": "Person_001", "confidence": 0.95, "evidence": {}, "reasoning": "test"},
        )

        result = run_memory_pipeline(state=state, llm_processor=StubLLMProcessor())

        self.assertEqual(result["relationships"], [])
        self.assertEqual(result["internal_artifacts"]["relationship_dossiers"][0]["retention_reason"], "single_photo_without_strong_signal")

    def test_structured_profile_field_output_contains_reasoning_and_normalized_evidence_ids(self) -> None:
        from services.memory_pipeline.profile_fields import generate_structured_profile
        from services.memory_pipeline.types import MemoryState

        relationship = Relationship(
            person_id="Person_002",
            relationship_type="romantic",
            intimacy_score=0.9,
            status="stable",
            confidence=0.88,
            reasoning="Repeated private events support romantic relationship.",
            shared_events=[
                {"event_id": "EVT_011", "date": "2026-03-01", "narrative": "一起在宿舍"},
            ],
            evidence={
                "photo_ids": ["PHOTO_043"],
                "event_ids": ["EVT_011"],
                "person_ids": ["Person_002"],
                "group_ids": [],
                "feature_names": ["relationship_type:romantic"],
                "supporting_refs": [],
                "contradicting_refs": [],
            },
        )
        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[],
            events=[],
            relationships=[relationship],
            groups=[],
            primary_decision={
                "mode": "person_id",
                "primary_person_id": "Person_001",
                "confidence": 0.95,
                "evidence": {
                    "photo_ids": ["PHOTO_001"],
                    "event_ids": [],
                    "person_ids": ["Person_001"],
                    "group_ids": [],
                    "feature_names": ["explicit_selfie_count"],
                    "supporting_refs": [],
                    "contradicting_refs": [],
                },
                "reasoning": "主角自拍信号稳定。",
            },
        )

        result = generate_structured_profile(state)
        tag = result["structured"]["long_term_facts"]["relationships"]["intimate_partner"]

        self.assertEqual(tag["value"], "Person_002")
        self.assertIn("reasoning", tag)
        self.assertIn("evidence", tag)
        self.assertEqual(tag["evidence"]["event_ids"], ["EVT_011"])
        self.assertEqual(tag["evidence"]["photo_ids"], ["PHOTO_043"])
        self.assertIn("EVT_011", tag["reasoning"])

    def test_precomputed_loader_maps_face_vlm_and_events_into_memory_state(self) -> None:
        from services.memory_pipeline.precomputed_loader import load_precomputed_memory_state

        with TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            (base / "face_recognition.json").write_text(
                json.dumps(
                    {
                        "primary_person_id": "Person_002",
                        "face_recognition": {
                            "primary_person_id": "Person_002",
                            "persons": [
                                {
                                    "person_id": "Person_002",
                                    "photo_count": 3,
                                    "first_seen": "2026-03-01T10:00:00",
                                    "last_seen": "2026-03-03T10:00:00",
                                    "avg_score": 0.9,
                                    "avg_quality": 0.8,
                                    "label": "Person",
                                }
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            (base / "vlm_results.json").write_text(
                json.dumps(
                    {
                        "vlm_results": [
                            {
                                "photo_id": "photo_001",
                                "timestamp": "2026-03-01T10:00:00",
                                "vlm_analysis": {"summary": "测试", "people": [], "scene": {}, "event": {}},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (base / "final_events.json").write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "title": "测试事件",
                                "started_at": "2026-03-01T10:00:00",
                                "ended_at": "2026-03-01T11:00:00",
                                "participant_person_ids": ["Person_002"],
                                "depicted_person_ids": [],
                                "place_refs": ["校园"],
                                "original_photo_ids": ["photo_001"],
                                "boundary_reason": "seed",
                                "confidence": 0.8,
                                "atomic_evidence": [{"value_or_text": "测试证据"}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            state = load_precomputed_memory_state(base)

        self.assertEqual(state.primary_decision["primary_person_id"], "Person_002")
        self.assertEqual(len(state.vlm_results), 1)
        self.assertEqual(len(state.events), 1)
        self.assertEqual(state.events[0].title, "测试事件")

    def test_critical_review_records_distinguish_repairable_and_terminal_results(self) -> None:
        from services.memory_pipeline.critical_review import (
            apply_profile_conflict_value_correction,
            build_conflict_evidence_package,
            build_critical_review_records,
            build_critical_rerun_tickets,
            classify_change,
            conflict_evidence_is_strong,
            should_attempt_reference_repair,
            transition_review_state,
        )
        from services.memory_pipeline.types import MemoryState, RelationshipDossier

        state = MemoryState(
            photos=[],
            face_db={},
            vlm_results=[],
            primary_decision={
                "mode": "person_id",
                "primary_person_id": "Person_001",
                "confidence": 0.92,
                "evidence": {"photo_ids": ["PHOTO_001"], "event_ids": [], "person_ids": ["Person_001"], "group_ids": [], "feature_names": ["selfie"]},
                "reasoning": "test",
            },
            primary_reflection={"triggered": False, "issues": [], "primary_signal_trace": {"candidate_signals": [{"person_id": "Person_001"}]}},
            relationship_dossiers=[
                RelationshipDossier(
                    person_id="Person_003",
                    person_kind="real_person",
                    memory_value="candidate",
                    photo_count=4,
                    time_span_days=45,
                    recent_gap_days=2,
                    monthly_frequency=2.0,
                    scene_profile={"scenes": ["screen", "campus"], "private_scene_ratio": 0.0, "dominant_scene_ratio": 0.6, "with_user_only": False},
                    interaction_signals=["chat"],
                    shared_events=[{"event_id": "EVT_001"}, {"event_id": "EVT_002"}],
                    trend_detail={},
                    co_appearing_persons=[],
                    anomalies=[],
                    evidence_refs=[
                        {"source_type": "photo", "source_id": "PHOTO_010"},
                        {"source_type": "event", "source_id": "EVT_001"},
                    ],
                    retention_decision="suppress",
                    retention_reason="screen_or_virtual_context",
                    relationship_result={"relationship_type": "friend", "status": "stable", "confidence": 0.71, "reasoning": "test"},
                    relationship_reflection={"issues": ["low_signal_relationship_pending_retention"]},
                )
            ],
        )
        profile_result = {
            "structured": {},
            "field_decisions": [
                {
                    "field_key": "long_term_facts.geography.mobility_pattern",
                    "gate_result": {
                        "candidate_family": "place",
                        "candidate_count": 2,
                        "strong_evidence_met": True,
                        "must_null": False,
                    },
                    "draft": {"value": None},
                    "reflection_1": {
                        "issues_found": ["under_generated_with_strong_evidence"],
                        "under_generation_reason": "value_generator_failed_after_candidates",
                        "should_retry_generation": True,
                        "missing_extractor_type": "missing_place_candidate_generator",
                    },
                    "reflection_2": {"decision": "keep", "null_reason": None},
                    "final": {
                        "value": None,
                        "evidence": {
                            "event_ids": ["EVT_001"],
                            "photo_ids": ["PHOTO_010"],
                            "person_ids": [],
                            "group_ids": [],
                            "feature_names": ["location_anchors"],
                            "constraint_notes": [],
                        },
                    },
                },
                {
                    "field_key": "long_term_expression.morality",
                    "gate_result": {
                        "candidate_family": "expression",
                        "candidate_count": 0,
                        "strong_evidence_met": False,
                        "must_null": True,
                    },
                    "draft": {"value": None},
                    "reflection_1": {"issues_found": [], "under_generation_reason": None, "should_retry_generation": False, "missing_extractor_type": None},
                    "reflection_2": {"decision": "keep", "null_reason": None},
                    "final": {
                        "value": None,
                        "evidence": {
                            "event_ids": [],
                            "photo_ids": [],
                            "person_ids": [],
                            "group_ids": [],
                            "feature_names": [],
                            "constraint_notes": ["silent_by_missing_social_media"],
                        },
                    },
                },
            ],
        }

        records = build_critical_review_records(state, profile_result, round_number=0)
        tickets = build_critical_rerun_tickets(records, round_number=1, max_rounds=2)
        record_map = {(record.scope, record.target_id): record for record in records}

        self.assertEqual(record_map[("primary", "primary_decision")].review_status, "pass")
        self.assertEqual(record_map[("relationship", "Person_003")].review_status, "queued")
        self.assertEqual(record_map[("relationship", "Person_003")].reparability, "repairable")
        self.assertEqual(record_map[("profile_field", "long_term_facts.geography.mobility_pattern")].review_status, "queued")
        self.assertEqual(record_map[("profile_field", "long_term_expression.morality")].review_status, "terminal_null")
        self.assertEqual(record_map[("profile_field", "long_term_expression.morality")].reparability, "non_repairable")
        ticket_targets = {ticket.target_id for ticket in tickets}
        self.assertIn("Person_003", ticket_targets)
        self.assertIn("long_term_facts.geography.mobility_pattern", ticket_targets)
        self.assertNotIn("long_term_expression.morality", ticket_targets)

        moved = transition_review_state(
            {"review_status": "running", "terminal": False},
            "queued",
            "evidence_changed_only_retry",
        )
        self.assertEqual(moved["review_status"], "queued")
        self.assertFalse(moved["terminal"])

        delta = classify_change(
            {
                "value_fingerprint": "value_a",
                "evidence_fingerprint": "evidence_a",
            },
            {
                "value_fingerprint": "value_a",
                "evidence_fingerprint": "evidence_b",
            },
        )
        self.assertFalse(delta["value_changed"])
        self.assertTrue(delta["evidence_changed_only"])
        self.assertFalse(delta["unchanged"])

        reference_ticket = {
            "scope": "profile_field",
            "target_id": "long_term_facts.geography.mobility_pattern",
            "evidence_focus_ids": ["event:EVT_001"],
        }
        self.assertTrue(
            should_attempt_reference_repair(
                record_map[("profile_field", "long_term_facts.geography.mobility_pattern")].to_dict(),
                reference_ticket,
            )
        )
        self.assertFalse(
            should_attempt_reference_repair(
                record_map[("profile_field", "long_term_expression.morality")].to_dict(),
                {"scope": "profile_field", "target_id": "long_term_expression.morality"},
            )
        )

        conflict_package = build_conflict_evidence_package(
            record=record_map[("profile_field", "long_term_facts.geography.mobility_pattern")].to_dict(),
            ticket=reference_ticket,
            before={
                "evidence_focus_ids": ["event:EVT_001"],
                "evidence_fingerprint": "before_evidence",
            },
            after={
                "evidence_focus_ids": ["event:EVT_001", "event:EVT_002"],
                "evidence_fingerprint": "after_evidence",
            },
        )
        self.assertTrue(
            conflict_evidence_is_strong(
                conflict_package,
                reason_code="value_generator_failed_after_candidates",
            )
        )

        correction_profile = {
            "structured": {
                "long_term_facts": {
                    "geography": {
                        "location_anchors": {"value": ["中山首府社区", "办公区"]},
                        "cross_border": {"value": False},
                    }
                }
            },
            "field_decisions": [
                {
                    "field_key": "long_term_facts.material.brand_preference",
                    "final": {
                        "value": ["Hello", "下当前面部状态与", "有粉色蝴蝶结虚拟"],
                        "confidence": 0.78,
                        "evidence": {"constraint_notes": []},
                        "reasoning": "旧结果",
                    },
                }
            ],
        }
        corrected, details = apply_profile_conflict_value_correction(
            profile_result=correction_profile,
            target_id="long_term_facts.material.brand_preference",
            reason_code="candidate_noise_brand_tokens",
            conflict_package={"score": 3},
        )
        self.assertTrue(corrected)
        self.assertIn("Hello Kitty", details["after_value"])
        final_payload = correction_profile["field_decisions"][0]["final"]
        self.assertIn("conflict_corrected:candidate_noise_brand_tokens", final_payload["evidence"]["constraint_notes"])
        self.assertIn("回流裁决检测到强冲突证据包", final_payload["reasoning"])

    def test_orchestrator_reruns_downstream_when_primary_review_changes_primary_person(self) -> None:
        from services.memory_pipeline.orchestrator import run_memory_pipeline
        from services.memory_pipeline.primary_person import PrimaryDecision
        from services.memory_pipeline.types import MemoryState, PersonScreening

        class StubLLMProcessor:
            def __init__(self) -> None:
                self.relationship_evidence_calls = 0

            def extract_events(self, vlm_results):
                return []

            def _collect_relationship_evidence(self, person_id, vlm_results, events=None):
                self.relationship_evidence_calls += 1
                return {
                    "photo_count": 2,
                    "time_span_days": 15,
                    "recent_gap_days": 1,
                    "monthly_frequency": 1.2,
                    "scenes": ["campus"],
                    "private_scene_ratio": 0.0,
                    "dominant_scene_ratio": 1.0,
                    "with_user_only": False,
                    "interaction_behavior": ["chat"],
                    "contact_types": ["no_contact"],
                    "rela_events": [{"event_id": "EVT_001", "title": "shared", "participants": ["Person_001", person_id]}],
                    "trend_detail": {},
                    "co_appearing_persons": [],
                    "anomalies": [],
                }

            def _infer_relationship(self, person_id, evidence, vlm_results, face_db):
                return Relationship(
                    person_id=person_id,
                    relationship_type="friend",
                    intimacy_score=0.55,
                    status="stable",
                    confidence=0.72,
                    reasoning="test",
                    shared_events=[{"event_id": "EVT_001"}],
                    evidence={"photo_count": evidence.get("photo_count", 0)},
                )

        state = MemoryState(
            photos=[],
            face_db={"Person_001": {"photo_count": 4}, "Person_002": {"photo_count": 5}, "Person_003": {"photo_count": 2}},
            vlm_results=[],
            screening={
                "Person_001": PersonScreening(person_id="Person_001", person_kind="real_person", memory_value="candidate"),
                "Person_002": PersonScreening(person_id="Person_002", person_kind="real_person", memory_value="candidate"),
                "Person_003": PersonScreening(person_id="Person_003", person_kind="real_person", memory_value="candidate"),
            },
        )
        llm_processor = StubLLMProcessor()

        with patch("services.memory_pipeline.orchestrator.analyze_primary_person_with_reflection") as mocked_primary:
            mocked_primary.side_effect = [
                (
                    PrimaryDecision(
                        mode="person_id",
                        primary_person_id="Person_001",
                        confidence=0.64,
                        evidence={"photo_ids": ["PHOTO_001"], "event_ids": [], "person_ids": ["Person_001"], "group_ids": [], "feature_names": ["other_photo"]},
                        reasoning="first pick",
                    ),
                    {
                        "triggered": True,
                        "issues": ["primary_ambiguity"],
                        "action": "switch_to_photographer_mode",
                        "primary_signal_trace": {"candidate_signals": [{"person_id": "Person_001"}, {"person_id": "Person_002"}]},
                    },
                ),
                (
                    PrimaryDecision(
                        mode="person_id",
                        primary_person_id="Person_002",
                        confidence=0.91,
                        evidence={"photo_ids": ["PHOTO_002"], "event_ids": [], "person_ids": ["Person_002"], "group_ids": [], "feature_names": ["selfie"]},
                        reasoning="second pick",
                    ),
                    {
                        "triggered": False,
                        "issues": [],
                        "action": "keep",
                        "primary_signal_trace": {"candidate_signals": [{"person_id": "Person_002"}]},
                    },
                ),
            ]
            result = run_memory_pipeline(state=state, llm_processor=llm_processor, debug_review_audit=True)

        self.assertEqual(result["internal_artifacts"]["primary_decision"]["primary_person_id"], "Person_002")
        self.assertGreaterEqual(llm_processor.relationship_evidence_calls, 4)
        self.assertTrue(result["internal_artifacts"]["critical_review_rounds"])

    def test_orchestrator_reruns_relationship_ticket_and_restores_keep_result(self) -> None:
        from services.memory_pipeline.orchestrator import run_memory_pipeline
        from services.memory_pipeline.types import MemoryState, PersonScreening

        class StubLLMProcessor:
            def __init__(self) -> None:
                self.evidence_calls = 0

            def extract_events(self, vlm_results):
                return []

            def _collect_relationship_evidence(self, person_id, vlm_results, events=None):
                self.evidence_calls += 1
                if self.evidence_calls <= 2:
                    return {
                        "photo_count": 4,
                        "time_span_days": 60,
                        "recent_gap_days": 2,
                        "monthly_frequency": 2.5,
                        "scenes": ["screen", "app screen", "virtual display"],
                        "private_scene_ratio": 0.0,
                        "dominant_scene_ratio": 0.8,
                        "with_user_only": False,
                        "interaction_behavior": ["screen chat"],
                        "contact_types": ["no_contact"],
                        "rela_events": [{"event_id": "EVT_001", "title": "online interaction"}, {"event_id": "EVT_002", "title": "campus meet"}],
                        "trend_detail": {},
                        "co_appearing_persons": [],
                        "anomalies": [],
                    }
                return {
                    "photo_count": 4,
                    "time_span_days": 60,
                    "recent_gap_days": 2,
                    "monthly_frequency": 2.5,
                    "scenes": ["campus", "cafe", "dorm"],
                    "private_scene_ratio": 0.3,
                    "dominant_scene_ratio": 0.4,
                    "with_user_only": True,
                    "interaction_behavior": ["chat", "selfie_together"],
                    "contact_types": ["selfie_together"],
                    "rela_events": [{"event_id": "EVT_001", "title": "cafe"}, {"event_id": "EVT_002", "title": "dorm"}],
                    "trend_detail": {"direction": "up"},
                    "co_appearing_persons": [],
                    "anomalies": [],
                }

            def _infer_relationship(self, person_id, evidence, vlm_results, face_db):
                return Relationship(
                    person_id=person_id,
                    relationship_type="close_friend",
                    intimacy_score=0.71,
                    status="growing",
                    confidence=0.78,
                    reasoning="test",
                    shared_events=[{"event_id": "EVT_001"}, {"event_id": "EVT_002"}],
                    evidence={"photo_count": evidence.get("photo_count", 0)},
                )

        state = MemoryState(
            photos=[],
            face_db={"Person_001": {"photo_count": 6}, "Person_002": {"photo_count": 4}},
            vlm_results=[],
            screening={
                "Person_001": PersonScreening(person_id="Person_001", person_kind="real_person", memory_value="candidate"),
                "Person_002": PersonScreening(person_id="Person_002", person_kind="real_person", memory_value="candidate"),
            },
            primary_decision={
                "mode": "person_id",
                "primary_person_id": "Person_001",
                "confidence": 0.9,
                "evidence": {"photo_ids": ["PHOTO_001"], "event_ids": [], "person_ids": ["Person_001"], "group_ids": [], "feature_names": ["selfie"]},
                "reasoning": "test",
            },
            primary_reflection={
                "triggered": False,
                "issues": [],
                "action": "keep",
                "primary_signal_trace": {"candidate_signals": [{"person_id": "Person_001"}]},
            },
        )
        llm_processor = StubLLMProcessor()

        result = run_memory_pipeline(state=state, llm_processor=llm_processor, debug_review_audit=True)

        round_zero = result["internal_artifacts"]["critical_review_rounds"][0]
        self.assertIn("Person_002", round_zero["summary"]["ticket_targets"])
        self.assertEqual(len(result["relationships"]), 1)
        self.assertEqual(result["relationships"][0].person_id, "Person_002")
        dossiers = result["internal_artifacts"]["relationship_dossiers"]
        self.assertEqual(dossiers[0]["retention_decision"], "keep")

    def test_orchestrator_hides_critical_audit_by_default(self) -> None:
        from services.memory_pipeline.orchestrator import run_memory_pipeline
        from services.memory_pipeline.types import MemoryState, PersonScreening

        class StubLLMProcessor:
            def extract_events(self, vlm_results):
                return []

            def _collect_relationship_evidence(self, person_id, vlm_results, events=None):
                return {
                    "photo_count": 0,
                    "time_span_days": 0,
                    "recent_gap_days": 0,
                    "monthly_frequency": 0.0,
                    "scenes": [],
                    "private_scene_ratio": 0.0,
                    "dominant_scene_ratio": 0.0,
                    "with_user_only": False,
                    "interaction_behavior": [],
                    "contact_types": [],
                    "rela_events": [],
                    "trend_detail": {},
                    "co_appearing_persons": [],
                    "anomalies": [],
                }

            def _infer_relationship(self, person_id, evidence, vlm_results, face_db):
                return Relationship(
                    person_id=person_id,
                    relationship_type="acquaintance",
                    intimacy_score=0.2,
                    status="stable",
                    confidence=0.4,
                    reasoning="stub",
                    shared_events=[],
                    evidence={},
                )

        state = MemoryState(
            photos=[],
            face_db={"Person_001": {"photo_count": 3}},
            vlm_results=[],
            screening={
                "Person_001": PersonScreening(person_id="Person_001", person_kind="real_person", memory_value="candidate"),
            },
            primary_decision={
                "mode": "person_id",
                "primary_person_id": "Person_001",
                "confidence": 0.9,
                "evidence": {"photo_ids": ["PHOTO_001"], "event_ids": [], "person_ids": ["Person_001"], "group_ids": [], "feature_names": ["selfie"]},
                "reasoning": "ok",
            },
            primary_reflection={
                "triggered": False,
                "issues": [],
                "action": "keep",
                "primary_signal_trace": {"candidate_signals": [{"person_id": "Person_001"}]},
            },
        )
        result = run_memory_pipeline(state=state, llm_processor=StubLLMProcessor())
        self.assertNotIn("critical_review_records", result["internal_artifacts"])
        self.assertNotIn("critical_rerun_tickets", result["internal_artifacts"])
        self.assertNotIn("critical_review_rounds", result["internal_artifacts"])


if __name__ == "__main__":
    unittest.main()
