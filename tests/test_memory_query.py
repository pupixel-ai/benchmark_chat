from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from memory_module import MemoryModuleService, MemoryQueryService
from models import Event, Photo, Relationship


class MemoryQueryTests(unittest.TestCase):
    def test_query_service_answers_revision_first_memory_with_new_structure(self) -> None:
        query_service = MemoryQueryService(now=datetime(2026, 3, 23, 12, 0))
        memory_payload = {
            "memory": {
                "pipeline_family": "v0317",
                "envelope": {"scope": {"user_id": "user_001"}},
                "event_revisions": [
                    {
                        "event_root_id": "event_root_001",
                        "event_revision_id": "event_rev_001",
                        "revision": 1,
                        "title": "Concert Night",
                        "event_summary": "Live concert with a close friend.",
                        "started_at": "2026-03-20T20:00:00",
                        "ended_at": "2026-03-20T22:00:00",
                        "participant_person_ids": ["Person_001", "Person_002"],
                        "depicted_person_ids": ["Person_001", "Person_002"],
                        "place_refs": ["Shanghai"],
                        "original_photo_ids": ["hash-001"],
                        "confidence": 0.9,
                        "status": "active",
                        "sealed_state": "sealed",
                        "atomic_evidence": [
                            {
                                "evidence_id": "evidence_001",
                                "root_event_revision_id": "event_rev_001",
                                "evidence_type": "brand",
                                "value_or_text": "MOET",
                                "provenance": "brand",
                                "original_photo_ids": ["hash-001"],
                                "confidence": 0.8,
                            }
                        ],
                    }
                ],
                "atomic_evidence": [
                    {
                        "evidence_id": "evidence_001",
                        "root_event_revision_id": "event_rev_001",
                        "evidence_type": "brand",
                        "value_or_text": "MOET",
                        "provenance": "brand",
                        "original_photo_ids": ["hash-001"],
                        "confidence": 0.8,
                    }
                ],
                "relationship_revisions": [
                    {
                        "relationship_root_id": "rel_root_001",
                        "relationship_revision_id": "rel_rev_001",
                        "target_person_id": "Person_002",
                        "relationship_type": "friend",
                        "label": "close friend",
                        "confidence": 0.85,
                        "supporting_event_ids": ["event_rev_001"],
                    }
                ],
                "profile_revision": {
                    "profile_revision_id": "profile_rev_001",
                    "primary_person_id": "Person_001",
                    "scope": "cumulative",
                    "generation_mode": "profile_input_pack_llm",
                    "original_photo_ids": ["hash-001"],
                },
                "profile_markdown": "# Profile\n\nConcert-heavy social life.",
                "profile_input_pack": {
                    "profile_input_pack_id": "profile_pack_001",
                    "time_range": {"start": "2026-03-20T20:00:00", "end": "2026-03-20T22:00:00"},
                    "baseline_rhythm": {"dominant_activity_window": "evening"},
                    "place_patterns": {"top_place_refs": [{"place_ref": "Shanghai", "count": 1}]},
                    "activity_patterns": {"top_activities": [{"activity_type": "concert", "count": 1}]},
                    "identity_signals": {},
                    "lifestyle_consumption_signals": {},
                    "event_grounded_signals": {
                        "interest_signals": [
                            {"label": "concert", "count": 1, "supporting_event_ids": ["event_rev_001"]}
                        ]
                    },
                    "reference_media_weak_signals": {},
                    "social_patterns": {
                        "top_relationships": [
                            {
                                "relationship_revision_id": "rel_rev_001",
                                "target_person_id": "Person_002",
                                "relationship_type": "friend",
                                "confidence": 0.85,
                            }
                        ],
                        "relationship_summary": {"close_relationship_count": 1},
                        "social_style_hints": {"one_on_one_bias": 0.7},
                    },
                    "change_points": [],
                    "key_event_refs": [{"event_revision_id": "event_rev_001", "title": "Concert Night"}],
                    "key_relationship_refs": [{"relationship_revision_id": "rel_rev_001"}],
                    "evidence_guardrails": {},
                },
            }
        }

        event_response = query_service.answer(memory_payload, "我过去3个月去过的演唱会")
        self.assertEqual(event_response["query_plan"]["plan_type"], "hybrid")
        self.assertEqual(event_response["answer"]["answer_type"], "event_search")
        self.assertEqual(event_response["answer"]["original_photo_ids"], ["hash-001"])
        self.assertGreaterEqual(len(event_response["supporting_units"]), 1)
        self.assertGreaterEqual(len(event_response["supporting_evidence"]), 1)

        food_response = query_service.answer(memory_payload, "我最近吃过什么")
        self.assertEqual(food_response["answer"]["answer_type"], "event_search")
        self.assertGreaterEqual(len(food_response["supporting_units"]), 1)
        self.assertEqual(food_response["answer"]["original_photo_ids"], ["hash-001"])

        profile_response = query_service.answer(memory_payload, "给我用户画像")
        self.assertEqual(profile_response["query_plan"]["plan_type"], "graph_first_exact")
        self.assertEqual(profile_response["answer"]["answer_type"], "profile_lookup")
        self.assertIn("profile_truth", profile_response["answer"])
        self.assertIn("report_markdown", profile_response["answer"])

        overview_response = query_service.answer(memory_payload, "请总结这个任务里的主要事件、人物关系和用户画像")
        self.assertEqual(overview_response["query_plan"]["plan_type"], "hybrid")
        self.assertEqual(overview_response["answer"]["answer_type"], "task_overview")
        self.assertGreaterEqual(len(overview_response["supporting_units"]), 1)
        self.assertGreaterEqual(len(overview_response["supporting_graph_entities"]), 1)
        self.assertGreaterEqual(len(overview_response["supporting_evidence"]), 1)
        self.assertIn("主要事件", overview_response["answer"]["summary"])
        self.assertNotIn("推理草稿箱", overview_response["answer"]["summary"])
        self.assertNotIn("1. 时空锚点确认", overview_response["answer"]["summary"])

    def test_query_service_answers_time_relationship_and_mood_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir)
            memory_payload = self._materialized_memory(task_dir)
            query_service = MemoryQueryService(now=datetime(2026, 3, 17, 12, 0))

            concert_response = query_service.answer(memory_payload, "我过去3个月去过的演唱会")
            concert_answer = concert_response["answer"]
            self.assertEqual(concert_answer["answer_type"], "event_search")
            self.assertIn("concert", concert_answer["resolved_concepts"])
            self.assertGreaterEqual(len(concert_response["supporting_units"]), 1)
            self.assertIn("吴嘉轩相关演出活动", concert_answer["summary"])

            recent_live_answer = query_service.answer(memory_payload, "最近去过哪些 live")["answer"]
            self.assertEqual(recent_live_answer["answer_type"], "event_search")
            self.assertIn("concert", recent_live_answer["resolved_concepts"])

            recent_show_answer = query_service.answer(memory_payload, "最近去过哪些 live/show")["answer"]
            self.assertEqual(recent_show_answer["answer_type"], "event_search")
            self.assertIn("concert", recent_show_answer["resolved_concepts"])

            relationship_response = query_service.answer(memory_payload, "帮我探索一下用户和父亲的关系")
            relationship_answer = relationship_response["answer"]
            self.assertEqual(relationship_answer["answer_type"], "relationship_explore")
            self.assertGreaterEqual(len(relationship_response["supporting_graph_entities"]), 1)
            self.assertEqual(relationship_response["supporting_graph_entities"][0]["target_face_person_id"], "Person_002")

            mood_answer = query_service.answer(memory_payload, "帮我找到用户最近的心情")["answer"]
            self.assertEqual(mood_answer["answer_type"], "mood_lookup")
            self.assertIn("happy_mood", mood_answer["resolved_concepts"])

    def test_query_service_merges_duplicate_live_events_and_falls_back_for_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir)
            memory_payload = self._materialized_memory(
                task_dir,
                events_override=[
                    Event(
                        event_id="EVT_CONCERT_1",
                        date="2026-02-15",
                        time_range="20:00 - 22:00",
                        duration="2小时",
                        title="Live Concert Night",
                        type="演唱会",
                        participants=["Person_001", "Person_002"],
                        location="Stadium",
                        description="A live concert attended together.",
                        photo_count=1,
                        confidence=0.91,
                        reason="concert evidence",
                        narrative_synthesis="Attended a live concert together at night.",
                        tags=["concert", "music"],
                        persona_evidence={"behavioral": ["likes live music"]},
                        evidence_photos=["photo_001"],
                    ),
                    Event(
                        event_id="EVT_CONCERT_2",
                        date="2026-02-15",
                        time_range="20:01 - 22:00",
                        duration="1小时59分",
                        title="Street Activity",
                        type="其他",
                        participants=["Person_001", "Person_002"],
                        location="Stadium",
                        description="A duplicate live activity view.",
                        photo_count=1,
                        confidence=0.75,
                        reason="duplicate evidence",
                        narrative_synthesis="Captured more stage content.",
                        tags=["music"],
                        persona_evidence={"behavioral": ["likes live music"]},
                        evidence_photos=["photo_002"],
                    ),
                ],
                relationships_override=[
                    Relationship(
                        person_id="Person_002",
                        relationship_type="friend",
                        label="朋友",
                        confidence=0.73,
                        evidence={
                            "photo_count": 3,
                            "time_span": "1个月",
                            "scenes": ["Home", "Stadium"],
                            "interaction_behavior": ["conflict over plans", "tense conversation"],
                            "sample_scenes": [
                                {
                                    "timestamp": "2026-03-10T19:00:00",
                                    "scene": "Home",
                                    "summary": "Tense dinner conversation",
                                    "activity": "meal",
                                }
                            ],
                        },
                        reason="Recent conflict and tension in repeated private scenes",
                    )
                ],
            )
            query_service = MemoryQueryService(now=datetime(2026, 3, 17, 12, 0))

            concert_response = query_service.answer(memory_payload, "我过去3个月去过的演唱会")
            concert_answer = concert_response["answer"]
            self.assertEqual(len(concert_response["supporting_units"]), 1)
            self.assertEqual(concert_response["supporting_units"][0]["title"], "吴嘉轩相关演出活动")

            conflict_response = query_service.answer(memory_payload, "请帮我寻找一下用户最近的几次冲突")
            conflict_answer = conflict_response["answer"]
            self.assertEqual(conflict_answer["answer_type"], "event_search")
            self.assertIn("derived_from_relationship_fallback", conflict_answer["uncertainty_flags"])
            self.assertGreaterEqual(len(conflict_response["supporting_graph_entities"]), 1)
            self.assertGreaterEqual(len(conflict_response["supporting_evidence"]), 1)

    def test_query_service_routes_profile_and_evidence_queries_by_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir)
            memory_payload = self._materialized_memory(task_dir)
            profile_core = memory_payload["storage"]["redis"]["profile_core"]
            first_event = memory_payload["storage"]["neo4j"]["nodes"]["events"][0]
            first_fact = memory_payload["storage"]["neo4j"]["nodes"]["facts"][0]
            profile_core["profiles"]["preference_profile"] = {
                "favorite_drink": {
                    "value": "iced latte",
                    "summary": "prefers iced latte in cafe and work breaks",
                    "confidence": 0.87,
                    "supporting_event_ids": [first_event["properties"]["event_id"]],
                    "supporting_fact_ids": [first_fact["properties"]["fact_id"]],
                    "supporting_photo_ids": ["photo_001"],
                    "evidence_refs": [{"ref_type": "photo", "ref_id": "photo_001"}],
                    "updated_at": "2026-03-17T00:00:00",
                }
            }
            memory_payload["storage"]["milvus"]["segments"].append(
                {
                    "segment_uuid": "seg_drink_001",
                    "segment_type": "brand_observation",
                    "text": "favorite_drink iced latte from cafe receipt",
                    "event_uuid": first_fact["event_uuid"],
                    "session_uuid": first_event["session_uuid"],
                    "relationship_uuid": None,
                    "started_at": "2026-02-15T20:00:00",
                    "ended_at": "2026-02-15T22:00:00",
                    "location_hint": "Cafe",
                }
            )
            query_service = MemoryQueryService(now=datetime(2026, 3, 17, 12, 0))

            profile_response = query_service.answer(memory_payload, "我最喜欢喝什么饮料")
            self.assertEqual(profile_response["query_plan"]["plan_type"], "graph_first_exact")
            self.assertEqual(profile_response["answer"]["answer_type"], "profile_lookup")
            self.assertIn("original_photo_ids", profile_response["answer"])

            evidence_response = query_service.answer(memory_payload, "这个饮料是什么产品")
            self.assertEqual(evidence_response["query_plan"]["plan_type"], "milvus_first_fuzzy")
            self.assertEqual(evidence_response["answer"]["answer_type"], "evidence_lookup")
            self.assertGreaterEqual(len(evidence_response["supporting_evidence"]), 1)

    def _materialized_memory(
        self,
        task_dir: Path,
        *,
        events_override: list[Event] | None = None,
        relationships_override: list[Relationship] | None = None,
    ) -> dict:
        uploads_dir = task_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        def make_photo(
            photo_id: str,
            filename: str,
            timestamp: datetime,
            location: dict,
            summary: str,
            activity: str,
        ) -> Photo:
            original_path = uploads_dir / filename
            original_path.write_bytes(b"demo")
            compressed_path = uploads_dir / f"compressed_{filename}.webp"
            compressed_path.write_bytes(b"demo")
            boxed_path = uploads_dir / f"boxed_{filename}.webp"
            boxed_path.write_bytes(b"demo")
            return Photo(
                photo_id=photo_id,
                filename=filename,
                path=str(original_path),
                timestamp=timestamp,
                location=location,
                source_hash=f"hash-{photo_id}",
                original_path=str(original_path),
                compressed_path=str(compressed_path),
                boxed_path=str(boxed_path),
                primary_person_id="Person_001",
                faces=[
                    {
                        "face_id": f"{photo_id}-face-1",
                        "person_id": "Person_001",
                        "score": 0.99,
                        "similarity": 0.91,
                        "bbox_xywh": {"x": 10, "y": 20, "w": 60, "h": 60},
                        "quality_score": 0.92,
                        "quality_flags": ["clear_face"],
                    },
                    {
                        "face_id": f"{photo_id}-face-2",
                        "person_id": "Person_002",
                        "score": 0.97,
                        "similarity": 0.86,
                        "bbox_xywh": {"x": 100, "y": 20, "w": 60, "h": 60},
                        "quality_score": 0.88,
                        "quality_flags": ["clear_face"],
                    },
                ],
                vlm_analysis={
                    "summary": summary,
                    "scene": {"location_detected": location.get("name"), "environment_description": summary},
                    "event": {"activity": activity, "social_context": "family", "interaction": "warm", "mood": "warm"},
                    "details": ["NEXT", "Rapeter吴嘉轩"] if "concert" in summary.lower() else ["family dinner"],
                    "key_objects": ["stage", "lights"] if "concert" in summary.lower() else ["table", "dinner"],
                    "people": [
                        {
                            "person_id": "Person_001",
                            "appearance": "casual",
                            "clothing": "jacket",
                            "interaction": "present",
                        },
                        {
                            "person_id": "Person_002",
                            "appearance": "middle-aged",
                            "clothing": "dark jacket",
                            "interaction": "standing together",
                        },
                    ],
                },
            )

        photos = [
            make_photo(
                "photo_001",
                "concert_1.jpg",
                datetime(2026, 2, 15, 20, 0),
                {"name": "Stadium", "lat": 31.2304, "lng": 121.4737},
                "happy concert night at the stadium",
                "concert",
            ),
            make_photo(
                "photo_002",
                "concert_2.jpg",
                datetime(2026, 2, 15, 20, 5),
                {"name": "Stadium", "lat": 31.2304, "lng": 121.4737},
                "happy concert night at the stadium",
                "concert",
            ),
            make_photo(
                "photo_003",
                "dinner.jpg",
                datetime(2026, 3, 10, 19, 0),
                {"name": "Home", "lat": 31.2204, "lng": 121.4637},
                "warm family dinner at home",
                "dinner",
            ),
        ]

        events = events_override or [
            Event(
                event_id="EVT_CONCERT",
                date="2026-02-15",
                time_range="20:00 - 22:00",
                duration="2小时",
                title="Live Concert Night",
                type="演唱会",
                participants=["Person_001", "Person_002"],
                location="Stadium",
                description="A live concert attended together.",
                photo_count=2,
                confidence=0.91,
                reason="concert evidence",
                narrative_synthesis="Attended a live concert together at night.",
                tags=["concert", "music"],
                persona_evidence={"behavioral": ["likes live music"]},
                evidence_photos=["photo_001", "photo_002"],
            )
        ]

        relationships = relationships_override or [
            Relationship(
                person_id="Person_002",
                relationship_type="father",
                label="父亲",
                confidence=0.9,
                evidence={
                    "photo_count": 3,
                    "time_span": "1个月",
                    "scenes": ["Home", "Stadium"],
                    "interaction_behavior": ["family dinner", "supportive concert outing"],
                    "sample_scenes": [
                        {
                            "timestamp": "2026-03-10T19:00:00",
                            "scene": "Home",
                            "summary": "Family dinner",
                            "activity": "meal",
                        }
                    ],
                },
                reason="Repeated family-like scenes with explicit father label",
            )
        ]

        face_output = {
            "primary_person_id": "Person_001",
            "metrics": {"total_faces": 6, "total_persons": 2},
            "failed_images": [],
            "persons": [
                {"person_id": "Person_001"},
                {"person_id": "Person_002"},
            ],
        }

        service = MemoryModuleService(
            task_id="task_query_memory",
            task_dir=str(task_dir),
            user_id="user_query",
            pipeline_version="vnext-test",
        )
        with patch(
            "memory_module.service.MemoryStoragePublisher.publish",
            return_value={
                "enabled": False,
                "redis": {"status": "skipped", "reason": "test"},
                "neo4j": {"status": "skipped", "reason": "test"},
                "milvus": {"status": "skipped", "reason": "test"},
            },
        ):
            return service.materialize(
                photos=photos,
                face_output=face_output,
                vlm_results=[
                    {"photo_id": photo.photo_id, "vlm_analysis": photo.vlm_analysis}
                    for photo in photos
                ],
                events=events,
                relationships=relationships,
                profile_markdown="# profile",
            )


if __name__ == "__main__":
    unittest.main()
