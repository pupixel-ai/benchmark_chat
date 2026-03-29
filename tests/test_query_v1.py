from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import delete

from backend.app import app, task_store
from backend.db import SessionLocal
from backend.models import SessionRecord, TaskRecord, UserRecord
from backend.query_v1 import QueryEngineV1, QueryStore, materialize_v0325_to_query_store
from backend.query_v1.models import (
    MemoryEvidenceRecord,
    MemoryEventPersonRecord,
    MemoryEventPhotoRecord,
    MemoryEventPlaceRecord,
    MemoryEventRecord,
    MemoryGroupMemberRecord,
    MemoryGroupRecord,
    MemoryMaterializationRecord,
    MemoryPhotoRecord,
    MemoryProfileFactRecord,
    MemoryRelationshipRecord,
    MemoryRelationshipSupportRecord,
)
from backend.query_v1.planner import StructuredQueryPlanner


class _FakeEmbedder:
    KEYWORDS = (
        "演唱会",
        "concert",
        "live",
        "关系",
        "最好",
        "friend",
        "亲密",
        "dinner",
        "晚饭",
        "Person_002",
        "Person_003",
    )

    def embed_query(self, text: str) -> list[float]:
        lowered = str(text or "").lower()
        return [float(lowered.count(keyword.lower())) for keyword in self.KEYWORDS]


class QueryV1Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        self.username = f"queryv1_{uuid.uuid4().hex[:10]}"
        self.password = "passw0rd!"
        self.task_ids: list[str] = []

        response = self.client.post(
            "/api/auth/register",
            json={"username": self.username, "password": self.password},
        )
        self.assertEqual(response.status_code, 200)
        self.user_id = response.json()["user"]["user_id"]
        self.query_store = QueryStore()

    def tearDown(self) -> None:
        for task_id in self.task_ids:
            shutil.rmtree(task_store.task_dir(task_id), ignore_errors=True)

        with SessionLocal() as session:
            for model in (
                MemoryRelationshipSupportRecord,
                MemoryGroupMemberRecord,
                MemoryProfileFactRecord,
                MemoryEvidenceRecord,
                MemoryEventPlaceRecord,
                MemoryEventPersonRecord,
                MemoryEventPhotoRecord,
                MemoryGroupRecord,
                MemoryRelationshipRecord,
                MemoryPhotoRecord,
                MemoryEventRecord,
                MemoryMaterializationRecord,
            ):
                session.execute(delete(model).where(model.user_id == self.user_id))
            session.execute(delete(SessionRecord).where(SessionRecord.user_id == self.user_id))
            session.execute(delete(TaskRecord).where(TaskRecord.user_id == self.user_id))
            session.execute(delete(UserRecord).where(UserRecord.user_id == self.user_id))
            session.commit()

    def test_materializer_persists_relationship_metrics_and_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task = self._synthetic_task(task_id=f"task_{uuid.uuid4().hex[:8]}", task_dir=Path(tmpdir), version="v0327-db-query")

            with patch("backend.query_v1.materializer.MilvusQueryIndexer.publish", return_value={"status": "skipped"}), patch(
                "backend.query_v1.materializer.Neo4jQueryIndexer.publish",
                return_value={"status": "skipped"},
            ):
                materialization = materialize_v0325_to_query_store(task, self.user_id, self.query_store)

            self.assertEqual(materialization["schema_version"], "query_v1")
            bundle = self.query_store.fetch_scope(user_id=self.user_id, source_task_id=task["task_id"])
            relationships = {item["person_id"]: item for item in bundle["relationships"]}
            rel_support = bundle["relationship_support"]
            events = {item["event_id"]: item for item in bundle["events"]}

            self.assertIn("Person_002", relationships)
            self.assertEqual(events["EVT_CONCERT_001"]["cover_photo_id"], "photo_001")
            self.assertAlmostEqual(float(relationships["Person_002"]["intimacy_score"]), 0.93, places=2)
            self.assertAlmostEqual(float(relationships["Person_002"]["monthly_frequency"]), 3.5, places=2)
            self.assertEqual(int(relationships["Person_002"]["recent_gap_days"]), 4)
            self.assertEqual(int(relationships["Person_002"]["shared_event_count"]), 1)
            self.assertTrue(any(item["relationship_id"] == "REL_P002" and item["event_id"] == "EVT_CONCERT_001" for item in rel_support))
            self.assertTrue(any(item["relationship_id"] == "REL_P002" and item["photo_id"] == "photo_001" for item in rel_support))

    def test_query_engine_returns_relationship_rank_with_photo_backed_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task = self._synthetic_task(task_id=f"task_{uuid.uuid4().hex[:8]}", task_dir=Path(tmpdir), version="v0327-db-query")
            planner = StructuredQueryPlanner(now=datetime(2026, 3, 29, 12, 0, 0))
            planner._llm_plan = lambda *args, **kwargs: None  # type: ignore[method-assign]
            engine = QueryEngineV1(store=self.query_store, planner=planner, embedder=_FakeEmbedder(), now=datetime(2026, 3, 29, 12, 0, 0))

            with patch("backend.query_v1.materializer.MilvusQueryIndexer.publish", return_value={"status": "skipped"}), patch(
                "backend.query_v1.materializer.Neo4jQueryIndexer.publish",
                return_value={"status": "skipped"},
            ):
                payload = engine.answer_task(task, "主角和谁关系最好", user_id=self.user_id)

            self.assertEqual(payload["query_plan"]["engine"], "query_v1")
            self.assertEqual(payload["query_plan"]["plan_type"], "graph_first_exact")
            self.assertEqual(payload["answer"]["answer_type"], "relationship_rank_query")
            self.assertEqual(payload["graph_support"][0]["target_person_id"], "Person_002")
            self.assertGreaterEqual(len(payload["matched_events"]), 1)
            self.assertEqual(payload["matched_events"][0]["event_id"], "EVT_CONCERT_001")
            self.assertIn("photo_001", payload["answer"]["original_photo_ids"])
            self.assertGreaterEqual(len(payload["supporting_photos"]), 1)

    def test_memory_query_endpoint_routes_v0327_db_task_to_query_v1(self) -> None:
        create_response = self.client.post("/api/tasks", json={"version": "v0327-db-query"})
        self.assertEqual(create_response.status_code, 200)
        task_id = create_response.json()["task_id"]
        self.task_ids.append(task_id)

        task_store.append_uploads(
            task_id,
            [
                {
                    "image_id": "photo_001",
                    "filename": "concert.png",
                    "stored_filename": "001_concert.png",
                    "path": "uploads/001_concert.png",
                    "url": "/assets/uploads/001_concert.png",
                    "preview_url": None,
                    "content_type": "image/png",
                    "width": 1200,
                    "height": 900,
                    "source_hash": "hash-photo-001",
                    "timestamp": "2026-03-01T20:00:00",
                },
                {
                    "image_id": "photo_002",
                    "filename": "dinner.png",
                    "stored_filename": "002_dinner.png",
                    "path": "uploads/002_dinner.png",
                    "url": "/assets/uploads/002_dinner.png",
                    "preview_url": None,
                    "content_type": "image/png",
                    "width": 1200,
                    "height": 900,
                    "source_hash": "hash-photo-002",
                    "timestamp": "2026-03-05T19:30:00",
                },
            ],
            status="completed",
            stage="completed",
        )
        family_dir = task_store.task_dir(task_id) / "v0325"
        family_dir.mkdir(parents=True, exist_ok=True)
        payloads = self._task_payloads()
        (family_dir / "relationship_dossiers.json").write_text(json.dumps(payloads["relationship_dossiers"], ensure_ascii=False), encoding="utf-8")
        (family_dir / "group_artifacts.json").write_text(json.dumps(payloads["group_artifacts"], ensure_ascii=False), encoding="utf-8")
        (family_dir / "structured_profile.json").write_text(json.dumps(payloads["structured_profile"], ensure_ascii=False), encoding="utf-8")
        (family_dir / "profile_fact_decisions.json").write_text(json.dumps(payloads["profile_fact_decisions"], ensure_ascii=False), encoding="utf-8")

        task_store.update_task(
            task_id,
            result={
                "face_recognition": {
                    "primary_person_id": "Person_001",
                    "images": [
                        {"image_id": "photo_001", "source_hash": "hash-photo-001", "timestamp": "2026-03-01T20:00:00"},
                        {"image_id": "photo_002", "source_hash": "hash-photo-002", "timestamp": "2026-03-05T19:30:00"},
                    ],
                },
                "memory": {
                    "pipeline_family": "v0325",
                    "vp1_observations": payloads["vp1_observations"],
                    "lp1_events": payloads["lp1_events"],
                    "lp2_relationships": payloads["lp2_relationships"],
                    "lp3_profile": {
                        "relationship_dossiers": payloads["relationship_dossiers"],
                        "group_artifacts": payloads["group_artifacts"],
                        "profile_fact_decisions": payloads["profile_fact_decisions"],
                        "structured": payloads["structured_profile"],
                    },
                },
            },
            status="completed",
            stage="completed",
        )

        with patch("backend.query_v1.planner.StructuredQueryPlanner._llm_plan", return_value=None), patch(
            "backend.query_v1.materializer.MilvusQueryIndexer.publish",
            return_value={"status": "skipped"},
        ), patch(
            "backend.query_v1.materializer.Neo4jQueryIndexer.publish",
            return_value={"status": "skipped"},
        ), patch("backend.app.MEMORY_QUERY_V1_ENABLED", True):
            response = self.client.post(
                f"/api/tasks/{task_id}/memory/query",
                json={"question": "主角和谁关系最好"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query_plan"]["engine"], "query_v1")
        self.assertEqual(payload["answer"]["answer_type"], "relationship_rank_query")
        self.assertEqual(payload["graph_support"][0]["target_person_id"], "Person_002")
        self.assertGreaterEqual(len(payload["matched_events"]), 1)
        self.assertEqual(payload["matched_events"][0]["event_id"], "EVT_CONCERT_001")
        self.assertIn("photo_001", payload["answer"]["original_photo_ids"])

    def _synthetic_task(self, *, task_id: str, task_dir: Path, version: str) -> dict:
        task_dir.mkdir(parents=True, exist_ok=True)
        family_dir = task_dir / "v0325"
        family_dir.mkdir(parents=True, exist_ok=True)
        payloads = self._task_payloads()
        (family_dir / "relationship_dossiers.json").write_text(json.dumps(payloads["relationship_dossiers"], ensure_ascii=False), encoding="utf-8")
        (family_dir / "group_artifacts.json").write_text(json.dumps(payloads["group_artifacts"], ensure_ascii=False), encoding="utf-8")
        (family_dir / "structured_profile.json").write_text(json.dumps(payloads["structured_profile"], ensure_ascii=False), encoding="utf-8")
        (family_dir / "profile_fact_decisions.json").write_text(json.dumps(payloads["profile_fact_decisions"], ensure_ascii=False), encoding="utf-8")
        return {
            "task_id": task_id,
            "task_dir": str(task_dir),
            "version": version,
            "updated_at": "2026-03-29T12:00:00",
            "uploads": [
                {
                    "image_id": "photo_001",
                    "path": "uploads/001_concert.png",
                    "url": "/assets/uploads/001_concert.png",
                    "timestamp": "2026-03-01T20:00:00",
                    "content_type": "image/png",
                    "width": 1200,
                    "height": 900,
                    "source_hash": "hash-photo-001",
                },
                {
                    "image_id": "photo_002",
                    "path": "uploads/002_dinner.png",
                    "url": "/assets/uploads/002_dinner.png",
                    "timestamp": "2026-03-05T19:30:00",
                    "content_type": "image/png",
                    "width": 1200,
                    "height": 900,
                    "source_hash": "hash-photo-002",
                },
            ],
            "result": {
                "face_recognition": {
                    "primary_person_id": "Person_001",
                    "images": [
                        {"image_id": "photo_001", "source_hash": "hash-photo-001", "timestamp": "2026-03-01T20:00:00"},
                        {"image_id": "photo_002", "source_hash": "hash-photo-002", "timestamp": "2026-03-05T19:30:00"},
                    ],
                },
                "memory": {
                    "pipeline_family": "v0325",
                    "vp1_observations": payloads["vp1_observations"],
                    "lp1_events": payloads["lp1_events"],
                    "lp2_relationships": payloads["lp2_relationships"],
                    "lp3_profile": {
                        "relationship_dossiers": payloads["relationship_dossiers"],
                        "group_artifacts": payloads["group_artifacts"],
                        "profile_fact_decisions": payloads["profile_fact_decisions"],
                        "structured": payloads["structured_profile"],
                    },
                },
            },
        }

    def _task_payloads(self) -> dict:
        return {
            "lp1_events": [
                {
                    "event_id": "EVT_CONCERT_001",
                    "anchor_photo_id": "photo_001",
                    "supporting_photo_ids": ["photo_001"],
                    "started_at": "2026-03-01T20:00:00",
                    "ended_at": "2026-03-01T22:30:00",
                    "title": "Live Concert Night",
                    "narrative_synthesis": "主角和朋友一起去了演唱会 concert live。",
                    "participant_person_ids": ["Person_001", "Person_002"],
                    "depicted_person_ids": ["Person_001", "Person_002"],
                    "place_refs": ["Shanghai Stadium"],
                    "tags": ["concert", "live", "music", "演唱会"],
                    "objective_fact": {"scene_description": "concert stage with bright lights"},
                    "persona_evidence": {"social": ["和亲密朋友一起听 live"]},
                    "confidence": 0.92,
                },
                {
                    "event_id": "EVT_DINNER_001",
                    "anchor_photo_id": "photo_002",
                    "supporting_photo_ids": ["photo_002"],
                    "started_at": "2026-03-05T19:30:00",
                    "ended_at": "2026-03-05T21:00:00",
                    "title": "Team Dinner",
                    "narrative_synthesis": "和同事一起吃晚饭 dinner。",
                    "participant_person_ids": ["Person_001", "Person_003"],
                    "depicted_person_ids": ["Person_001", "Person_003"],
                    "place_refs": ["Office Bistro"],
                    "tags": ["dinner", "coworker"],
                    "objective_fact": {"scene_description": "casual dinner with coworkers"},
                    "persona_evidence": {"social": ["和同事一起聚餐"]},
                    "confidence": 0.71,
                },
            ],
            "vp1_observations": [
                {
                    "photo_id": "photo_001",
                    "confidence": 0.88,
                    "vlm_analysis": {
                        "summary": "concert live performance with a close friend",
                        "scene": {"location_detected": "Shanghai Stadium"},
                        "event": {"activity": "concert", "interaction": "watching live together"},
                        "details": ["stage lights", "music show"],
                        "key_objects": ["microphone", "stage"],
                    },
                    "ocr_hits": ["LIVE", "MUSIC"],
                },
                {
                    "photo_id": "photo_002",
                    "confidence": 0.77,
                    "vlm_analysis": {
                        "summary": "coworker dinner after work",
                        "scene": {"location_detected": "Office Bistro"},
                        "event": {"activity": "dinner", "interaction": "team gathering"},
                        "details": ["plates", "restaurant"],
                        "key_objects": ["table", "food"],
                    },
                },
            ],
            "lp2_relationships": [
                {
                    "relationship_id": "REL_P002",
                    "person_id": "Person_002",
                    "relationship_type": "friend",
                    "status": "active",
                    "confidence": 0.91,
                    "intimacy_score": 0.93,
                    "reasoning": "长期稳定的高频同游演唱会朋友。",
                    "supporting_event_ids": ["EVT_CONCERT_001"],
                    "supporting_photo_ids": ["photo_001"],
                    "shared_events": [{"event_id": "EVT_CONCERT_001"}],
                    "evidence": {"photo_ids": ["photo_001"]},
                },
                {
                    "relationship_id": "REL_P003",
                    "person_id": "Person_003",
                    "relationship_type": "coworker",
                    "status": "active",
                    "confidence": 0.62,
                    "intimacy_score": 0.31,
                    "reasoning": "主要是同事饭局。",
                    "supporting_event_ids": ["EVT_DINNER_001"],
                    "supporting_photo_ids": ["photo_002"],
                    "shared_events": [{"event_id": "EVT_DINNER_001"}],
                    "evidence": {"photo_ids": ["photo_002"]},
                },
            ],
            "relationship_dossiers": [
                {
                    "person_id": "Person_002",
                    "photo_count": 4,
                    "time_span_days": 90,
                    "recent_gap_days": 4,
                    "monthly_frequency": 3.5,
                    "scene_profile": {"private_scene_ratio": 0.7},
                    "interaction_signals": ["shared concerts", "repeated one-on-one outings"],
                    "trend_detail": {"direction": "stable_up"},
                },
                {
                    "person_id": "Person_003",
                    "photo_count": 2,
                    "time_span_days": 45,
                    "recent_gap_days": 18,
                    "monthly_frequency": 1.2,
                    "scene_profile": {"private_scene_ratio": 0.1},
                    "interaction_signals": ["team dinners"],
                    "trend_detail": {"direction": "stable"},
                },
            ],
            "group_artifacts": [
                {
                    "group_id": "GROUP_001",
                    "members": ["Person_001", "Person_002"],
                    "group_type_candidate": "close_circle",
                    "confidence": 0.81,
                    "strong_evidence_refs": [{"event_id": "EVT_CONCERT_001"}],
                    "reason": "长期一对一演唱会活动",
                }
            ],
            "structured_profile": {
                "long_term_facts": {
                    "interests": {
                        "live_music": {
                            "value": True,
                            "confidence": 0.9,
                        }
                    }
                }
            },
            "profile_fact_decisions": [
                {
                    "field_key": "long_term_facts.interests.live_music",
                    "final": {"value": True, "confidence": 0.9},
                }
            ],
        }
