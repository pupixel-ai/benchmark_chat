from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from memory_module import MemoryModuleService
from memory_module.dto import EventCandidateDTO, PhotoFactDTO, SessionDTO, VLMPhotoObservationDTO
from memory_module.ontology import collect_concepts
from models import Event, Photo, Relationship


class MemoryModuleTests(unittest.TestCase):
    def test_collect_concepts_returns_multiple_matches_from_same_text(self) -> None:
        concepts = collect_concepts(["深夜聚会后又去拍了演唱会海报"])

        self.assertIn("concert", concepts)
        self.assertIn("leisure", concepts)

    def test_refine_event_candidate_uses_stage_poster_and_artist_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = MemoryModuleService(
                task_id="task_refine_event",
                task_dir=tmpdir,
                user_id="user_refine_event",
                pipeline_version="v0315",
            )
            event = EventCandidateDTO(
                event_id="EVT_FESTIVAL",
                event_uuid="evt-festival",
                upstream_ref={"object_type": "event_candidate", "object_id": "EVT_FESTIVAL"},
                title="深夜街头聚会与活动记录",
                event_type="其他",
                time_range="01:25 - 01:26",
                started_at="2026-03-16T01:25:00",
                ended_at="2026-03-16T01:26:00",
                location="复合场景",
                photo_ids=["photo_001"],
                session_ids=["session_001"],
                description="凌晨街头与户外舞台活动",
                narrative_synthesis="拍摄带有艺人名字的舞台宣传海报",
            )
            photo = PhotoFactDTO(
                photo_id="photo_001",
                photo_uuid="photo-uuid-001",
                upstream_ref={"object_type": "photo", "object_id": "photo_001"},
                filename="festival.jpg",
                source_hash="hash-photo-001",
                captured_at_original="2026-03-16T01:25:00",
                captured_at_utc="2026-03-16T01:25:00",
                timezone_guess=None,
                time_confidence=1.0,
                location={"name": "户外舞台"},
                primary_face_person_id=None,
                vlm_observation=VLMPhotoObservationDTO(
                    summary="【拍摄者】正在拍摄舞台上的宣传海报，海报上显示‘NEXT’和‘Rapeter吴嘉轩’字样",
                    scene={
                        "environment_description": "户外舞台区域，背景有树木和舞台钢结构",
                        "location_detected": "户外舞台（可能为演唱会或活动场地）",
                    },
                    event={"activity": "拍摄舞台宣传海报", "mood": "平静"},
                    details=["NEXT", "Rapeter吴嘉轩"],
                    key_objects=["宣传海报", "舞台钢结构", "舞台灯光"],
                ),
            )
            session = SessionDTO(
                session_id="session_001",
                session_uuid="session-uuid-001",
                upstream_ref={"object_type": "session", "object_id": "session_001"},
                day_key="2026-03-16",
                photo_ids=["photo_001"],
                photo_uuids=["photo-uuid-001"],
                burst_ids=["burst_001"],
                started_at="2026-03-16T01:25:00",
                ended_at="2026-03-16T01:26:00",
                duration_seconds=60,
                location_hint={"name": "户外舞台"},
                activity_hints=["音乐节海报拍摄"],
                summary_hint="带有艺人名字的舞台宣传海报",
            )

            service._refine_event_candidates([event], [photo], [session])

            self.assertEqual(event.event_type, "music_festival_performance")
            self.assertEqual(event.title, "吴嘉轩相关演出活动")
            self.assertIn("concert", event.tags)

    def test_merge_music_events_within_same_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            service = MemoryModuleService(
                task_id="task_merge_event",
                task_dir=tmpdir,
                user_id="user_merge_event",
                pipeline_version="v0315",
            )
            left = EventCandidateDTO(
                event_id="EVT_001",
                event_uuid="evt-001",
                upstream_ref={"object_type": "event_candidate", "object_id": "EVT_001"},
                title="吴嘉轩相关演出活动",
                event_type="music_festival_performance",
                time_range="01:25 - 01:26",
                started_at="2026-03-16T01:25:00",
                ended_at="2026-03-16T01:26:00",
                location="户外舞台",
                participant_face_person_ids=["Person_003"],
                participant_person_uuids=["person-003"],
                photo_ids=["photo_001"],
                photo_uuids=["photo-uuid-001"],
                session_ids=["session_001"],
                session_uuids=["session-uuid-001"],
                description="舞台海报",
                narrative_synthesis="拍摄艺人宣传海报",
                tags=["concert"],
                confidence=0.72,
                evidence_refs=[{"ref_type": "photo", "ref_id": "photo_001"}],
            )
            right = EventCandidateDTO(
                event_id="EVT_002",
                event_uuid="evt-002",
                upstream_ref={"object_type": "event_candidate", "object_id": "EVT_002"},
                title="相关演出活动记录",
                event_type="concert",
                time_range="01:26 - 01:27",
                started_at="2026-03-16T01:26:00",
                ended_at="2026-03-16T01:27:00",
                location="户外舞台",
                participant_face_person_ids=["Person_003"],
                participant_person_uuids=["person-003"],
                photo_ids=["photo_002"],
                photo_uuids=["photo-uuid-002"],
                session_ids=["session_001"],
                session_uuids=["session-uuid-001"],
                description="舞台区域",
                narrative_synthesis="继续拍摄现场宣传物料",
                tags=["music_festival_performance"],
                confidence=0.81,
                evidence_refs=[{"ref_type": "photo", "ref_id": "photo_002"}],
            )

            merged = service._merge_event_candidates([left, right])

            self.assertEqual(len(merged), 1)
            self.assertEqual(merged[0].title, "吴嘉轩相关演出活动")
            self.assertEqual(merged[0].event_type, "music_festival_performance")
            self.assertEqual(merged[0].photo_ids, ["photo_001", "photo_002"])

    def test_materialize_builds_sequences_and_profile_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir)
            photos = self._sample_photos(task_dir)
            service = MemoryModuleService(
                task_id="task_alpha",
                task_dir=str(task_dir),
                user_id="user_alpha",
                pipeline_version="v0315",
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
                result = service.materialize(
                    photos=photos,
                    face_output=self._face_output(),
                    vlm_results=self._vlm_results(photos),
                    events=self._events(),
                    relationships=self._relationships(),
                    profile_markdown="# profile\n\n- likes brunch",
                    cached_photo_ids={"photo_001"},
                )

            self.assertEqual(result["summary"]["session_count"], 2)
            self.assertEqual(result["summary"]["timeline_count"], 1)
            self.assertEqual(result["summary"]["relationship_count"], 2)
            self.assertGreater(result["summary"]["profile_field_count"], 0)

            profile_fields = result["storage"]["redis"]["profile_core"]["fields"]
            self.assertIn("interests", profile_fields)
            self.assertIn("behavioral", profile_fields)
            self.assertIn("brunch", [value.lower() for value in profile_fields["interests"]["values"]])
            self.assertGreater(len(profile_fields["interests"]["evidence_refs"]), 0)
            self.assertEqual(profile_fields["interests"]["profile_version"], 1)
            self.assertTrue(result["storage"]["redis"]["profile_core"]["key"].startswith("profile:"))

            self.assertEqual(result["transparency"]["vlm_stage"]["cached_hits"], 1)
            self.assertGreater(len(result["transparency"]["traces"]), 0)
            focus_graph = result["transparency"]["focus_graph"]
            self.assertEqual(focus_graph["primary_face_person_id"], "Person_001")
            self.assertGreaterEqual(focus_graph["node_count"], 6)
            self.assertIn("主用户", focus_graph["mermaid"])
            primary_node = next(
                node for node in focus_graph["nodes"] if node["node_type"] == "primary_person"
            )
            self.assertTrue(primary_node["is_primary"])
            self.assertEqual(focus_graph["center_node_id"], primary_node["node_id"])

            relationship_edges = [
                edge for edge in result["storage"]["neo4j"]["edges"]
                if edge["edge_type"] == "HAS_RELATIONSHIP"
            ]
            self.assertEqual(len(relationship_edges), 2)
            self.assertTrue(all(edge["from_id"] == primary_node["node_id"] for edge in relationship_edges))

            person_nodes = result["storage"]["neo4j"]["nodes"]["persons"]
            primary_person_record = next(
                node for node in person_nodes if node["properties"]["face_person_id"] == "Person_001"
            )
            self.assertEqual(primary_person_record["properties"]["is_primary_candidate"], True)
            self.assertNotIn("photos", result["storage"]["neo4j"]["nodes"])
            relationship_nodes = result["storage"]["neo4j"]["nodes"]["relationship_hypotheses"]
            self.assertEqual(len(relationship_nodes), 2)
            self.assertTrue(all(node["properties"]["status"] in {"active", "rejected"} for node in relationship_nodes))
            self.assertIn("concepts", result["storage"]["neo4j"]["nodes"])
            self.assertGreater(len(result["storage"]["neo4j"]["nodes"]["concepts"]), 0)
            self.assertIn("mood_states", result["storage"]["neo4j"]["nodes"])
            self.assertIn("period_hypotheses", result["storage"]["neo4j"]["nodes"])

            self.assertTrue(Path(result["artifacts"]["envelope_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["storage_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["focus_graph_path"]).exists())
            self.assertTrue(Path(result["artifacts"]["focus_graph_mermaid_path"]).exists())
            self.assertEqual(result["summary"]["external_sinks_published"], 0)
            self.assertEqual(result["external_publish"]["redis"]["status"], "skipped")

    def test_person_uuid_is_stable_for_same_user_and_face_person_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir_one, tempfile.TemporaryDirectory() as tmpdir_two:
            photos_one = self._sample_photos(Path(tmpdir_one))
            photos_two = self._sample_photos(Path(tmpdir_two))

            service_one = MemoryModuleService(
                task_id="task_one",
                task_dir=tmpdir_one,
                user_id="shared_user",
                pipeline_version="v0315",
            )
            service_two = MemoryModuleService(
                task_id="task_two",
                task_dir=tmpdir_two,
                user_id="shared_user",
                pipeline_version="v0315",
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
                result_one = service_one.materialize(
                    photos=photos_one,
                    face_output=self._face_output(),
                    vlm_results=self._vlm_results(photos_one),
                    events=self._events(),
                    relationships=self._relationships(),
                    profile_markdown="",
                )
                result_two = service_two.materialize(
                    photos=photos_two,
                    face_output=self._face_output(),
                    vlm_results=self._vlm_results(photos_two),
                    events=self._events(),
                    relationships=self._relationships(),
                    profile_markdown="",
                )

            person_map_one = {
                item["face_person_id"]: item["person_uuid"]
                for item in result_one["storage"]["identity_maps"]["persons"]
            }
            person_map_two = {
                item["face_person_id"]: item["person_uuid"]
                for item in result_two["storage"]["identity_maps"]["persons"]
            }

            self.assertEqual(person_map_one["Person_002"], person_map_two["Person_002"])
            self.assertEqual(person_map_one["Person_003"], person_map_two["Person_003"])

    def test_materialize_without_primary_person_centers_graph_on_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir)
            photos = self._sample_photos(task_dir)
            for photo in photos:
                photo.primary_person_id = None

            service = MemoryModuleService(
                task_id="task_beta",
                task_dir=str(task_dir),
                user_id="user_beta",
                pipeline_version="v0315",
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
                result = service.materialize(
                    photos=photos,
                    face_output={**self._face_output(), "primary_person_id": None},
                    vlm_results=self._vlm_results(photos),
                    events=self._events(),
                    relationships=[],
                    profile_markdown="",
                    cached_photo_ids=set(),
                )

            focus_graph = result["transparency"]["focus_graph"]
            self.assertEqual(focus_graph["center_node_id"], "user_beta")
            self.assertIsNone(focus_graph["primary_face_person_id"])
            self.assertEqual(focus_graph["nodes"][0]["node_type"], "user_account")

            observed_event_edges = [
                edge for edge in result["storage"]["neo4j"]["edges"]
                if edge["edge_type"] == "OBSERVED_EVENT"
            ]
            self.assertGreaterEqual(len(observed_event_edges), 1)
            self.assertTrue(all(edge["from_id"] == "user_beta" for edge in observed_event_edges))
            self.assertEqual(len(result["storage"]["neo4j"]["nodes"]["primary_person_hypotheses"]), 0)

    def _sample_photos(self, task_dir: Path) -> list[Photo]:
        uploads_dir = task_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)

        photo_specs = [
            ("photo_001", "breakfast_1.jpg", datetime(2026, 3, 15, 10, 0), {"name": "Home", "lat": 31.2304, "lng": 121.4737}),
            ("photo_002", "breakfast_2.jpg", datetime(2026, 3, 15, 10, 1), {"name": "Home", "lat": 31.2304, "lng": 121.4737}),
            ("photo_003", "meeting.jpg", datetime(2026, 3, 15, 15, 30), {"name": "Cafe", "lat": 31.2243, "lng": 121.4768}),
        ]
        photos = []
        for photo_id, filename, timestamp, location in photo_specs:
            source_path = uploads_dir / filename
            source_path.write_bytes(b"demo")
            compressed_path = uploads_dir / f"compressed_{filename}.webp"
            compressed_path.write_bytes(b"demo")
            boxed_path = uploads_dir / f"boxed_{filename}.webp"
            boxed_path.write_bytes(b"demo")
            photos.append(
                Photo(
                    photo_id=photo_id,
                    filename=filename,
                    path=str(source_path),
                    timestamp=timestamp,
                    location=location,
                    source_hash=f"hash-{photo_id}",
                    original_path=str(source_path),
                    compressed_path=str(compressed_path),
                    boxed_path=str(boxed_path),
                    primary_person_id="Person_001",
                    faces=self._faces_for_photo(photo_id),
                    vlm_analysis=self._vlm_analysis_for_photo(photo_id),
                )
            )
        return photos

    def _faces_for_photo(self, photo_id: str) -> list[dict]:
        faces = {
            "photo_001": [
                {
                    "face_id": "face_001",
                    "person_id": "Person_001",
                    "score": 0.99,
                    "similarity": 0.88,
                    "bbox_xywh": {"x": 10, "y": 20, "w": 60, "h": 60},
                    "quality_score": 0.91,
                    "quality_flags": ["clear_face"],
                }
            ],
            "photo_002": [
                {
                    "face_id": "face_002",
                    "person_id": "Person_001",
                    "score": 0.98,
                    "similarity": 0.87,
                    "bbox_xywh": {"x": 12, "y": 18, "w": 58, "h": 58},
                    "quality_score": 0.89,
                    "quality_flags": ["clear_face"],
                },
                {
                    "face_id": "face_003",
                    "person_id": "Person_002",
                    "score": 0.96,
                    "similarity": 0.82,
                    "bbox_xywh": {"x": 100, "y": 22, "w": 55, "h": 55},
                    "quality_score": 0.86,
                    "quality_flags": ["clear_face"],
                },
            ],
            "photo_003": [
                {
                    "face_id": "face_004",
                    "person_id": "Person_001",
                    "score": 0.97,
                    "similarity": 0.86,
                    "bbox_xywh": {"x": 16, "y": 26, "w": 58, "h": 58},
                    "quality_score": 0.90,
                    "quality_flags": ["clear_face"],
                },
                {
                    "face_id": "face_005",
                    "person_id": "Person_003",
                    "score": 0.95,
                    "similarity": 0.80,
                    "bbox_xywh": {"x": 108, "y": 28, "w": 54, "h": 54},
                    "quality_score": 0.85,
                    "quality_flags": ["clear_face"],
                },
            ],
        }
        return faces[photo_id]

    def _vlm_analysis_for_photo(self, photo_id: str) -> dict:
        analyses = {
            "photo_001": {
                "summary": "Morning brunch at home.",
                "people": [{"person_id": "Person_001", "appearance": "casual", "clothing": "hoodie", "interaction": "eating"}],
                "scene": {"location_detected": "Home kitchen", "environment_description": "table and coffee"},
                "event": {"activity": "brunch", "social_context": "solo", "interaction": "eating", "mood": "calm"},
                "details": ["coffee", "toast"],
                "key_objects": ["coffee", "plate"],
            },
            "photo_002": {
                "summary": "Two people having brunch at home.",
                "people": [
                    {"person_id": "Person_001", "appearance": "casual", "clothing": "hoodie", "interaction": "hosting"},
                    {"person_id": "Person_002", "appearance": "smiling", "clothing": "sweater", "interaction": "sharing meal"},
                ],
                "scene": {"location_detected": "Home dining room", "environment_description": "brunch table"},
                "event": {"activity": "brunch", "social_context": "friend", "interaction": "chatting", "mood": "warm"},
                "details": ["croissant", "fruit"],
                "key_objects": ["croissant", "coffee"],
            },
            "photo_003": {
                "summary": "Afternoon cafe meeting.",
                "people": [
                    {"person_id": "Person_001", "appearance": "focused", "clothing": "coat", "interaction": "talking"},
                    {"person_id": "Person_003", "appearance": "formal", "clothing": "shirt", "interaction": "discussing"},
                ],
                "scene": {"location_detected": "Cafe", "environment_description": "cafe table and laptop"},
                "event": {"activity": "meeting", "social_context": "colleague", "interaction": "discussion", "mood": "focused"},
                "details": ["laptop", "notebook"],
                "key_objects": ["laptop", "coffee"],
            },
        }
        return analyses[photo_id]

    def _vlm_results(self, photos: list[Photo]) -> list[dict]:
        return [
            {
                "photo_id": photo.photo_id,
                "filename": photo.filename,
                "timestamp": photo.timestamp.isoformat(),
                "location": photo.location,
                "face_person_ids": [face["person_id"] for face in photo.faces],
                "vlm_analysis": photo.vlm_analysis,
            }
            for photo in photos
        ]

    def _events(self) -> list[Event]:
        return [
            Event(
                event_id="EVT_001",
                date="2026-03-15",
                time_range="10:00 - 10:30",
                duration="0.5小时",
                title="Brunch At Home",
                type="用餐",
                participants=["Person_001", "Person_002"],
                location="Home",
                description="Shared brunch at home.",
                photo_count=2,
                confidence=0.84,
                reason="same time and same place",
                narrative_synthesis="A warm brunch moment at home.",
                tags=["brunch", "home"],
                persona_evidence={"behavioral": ["hosts friends at home"], "aesthetic": ["prefers cozy table setup"], "socioeconomic": ["comfortable domestic routine"]},
                evidence_photos=["photo_001", "photo_002"],
            ),
            Event(
                event_id="EVT_002",
                date="2026-03-15",
                time_range="15:30 - 16:30",
                duration="1小时",
                title="Cafe Meeting",
                type="工作",
                participants=["Person_001", "Person_003"],
                location="Cafe",
                description="Discussion with a colleague at a cafe.",
                photo_count=1,
                confidence=0.78,
                reason="cafe laptop meeting",
                narrative_synthesis="An afternoon working session in a cafe.",
                tags=["work", "cafe"],
                persona_evidence={"behavioral": ["mixes work and cafe spaces"], "aesthetic": ["likes clean work surfaces"], "socioeconomic": ["mobile knowledge work"]},
                evidence_photos=["photo_003"],
            ),
        ]

    def _relationships(self) -> list[Relationship]:
        return [
            Relationship(
                person_id="Person_002",
                relationship_type="friend",
                label="朋友",
                confidence=0.71,
                evidence={
                    "photo_count": 1,
                    "time_span": "1天",
                    "scenes": ["Home dining room"],
                    "interaction_behavior": ["chatting"],
                    "sample_scenes": [{"timestamp": "2026-03-15T10:01:00", "scene": "Home dining room", "summary": "Brunch", "activity": "brunch"}],
                },
                reason="appears in intimate brunch scene",
            ),
            Relationship(
                person_id="Person_003",
                relationship_type="colleague",
                label="同事",
                confidence=0.76,
                evidence={
                    "photo_count": 1,
                    "time_span": "1天",
                    "scenes": ["Cafe"],
                    "interaction_behavior": ["discussion"],
                    "sample_scenes": [{"timestamp": "2026-03-15T15:30:00", "scene": "Cafe", "summary": "Meeting", "activity": "meeting"}],
                },
                reason="appears in work-like meeting scene",
            ),
        ]

    def _face_output(self) -> dict:
        return {
            "primary_person_id": "Person_001",
            "metrics": {
                "total_faces": 5,
                "total_persons": 3,
                "total_images": 3,
            },
            "persons": [
                {"person_id": "Person_001"},
                {"person_id": "Person_002"},
                {"person_id": "Person_003"},
            ],
            "failed_images": [],
        }


if __name__ == "__main__":
    unittest.main()
