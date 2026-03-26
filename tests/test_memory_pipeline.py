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

if __name__ == "__main__":
    unittest.main()
