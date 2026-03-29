from __future__ import annotations

import json
import shutil
import tempfile
import unittest
import uuid
from contextlib import contextmanager
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
from backend.query_v1.writer import AnswerWriterLLM


class _FakeEmbedder:
    KEYWORDS = (
        "演唱会",
        "concert",
        "live",
        "展览",
        "exhibition",
        "猫",
        "pet",
        "关系",
        "romantic",
        "工作",
        "学生",
        "Person_002",
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

    def test_materializer_persists_relationship_metrics_and_profile_facts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task = self._synthetic_task(task_id=f"task_{uuid.uuid4().hex[:8]}", task_dir=Path(tmpdir), version="v0327-db-query")

            with self._materialization_patches():
                materialization = materialize_v0325_to_query_store(task, self.user_id, self.query_store)

            self.assertEqual(materialization["schema_version"], "query_v1")
            bundle = self.query_store.fetch_scope(user_id=self.user_id, source_task_id=task["task_id"])
            relationships = {item["person_id"]: item for item in bundle["relationships"]}
            profile_facts = {item["field_key"]: item for item in bundle["profile_facts"]}

            self.assertIn("Person_002", relationships)
            self.assertAlmostEqual(float(relationships["Person_002"]["intimacy_score"]), 0.93, places=2)
            self.assertAlmostEqual(float(relationships["Person_002"]["monthly_frequency"]), 3.5, places=2)
            self.assertEqual(int(relationships["Person_002"]["recent_gap_days"]), 4)
            self.assertIn("long_term_facts.relationships.pets", profile_facts)
            self.assertEqual(profile_facts["long_term_facts.relationships.pets"]["source_level"], "decision")

    def test_query_engine_fact_first_returns_profile_fact_and_photo_backed_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task = self._synthetic_task(task_id=f"task_{uuid.uuid4().hex[:8]}", task_dir=Path(tmpdir), version="v0327-db-query")
            engine = self._engine()

            with self._materialization_patches(), patch.object(StructuredQueryPlanner, "_llm_route", return_value=None), patch.object(
                AnswerWriterLLM,
                "_llm_summary",
                return_value=None,
            ):
                payload = engine.answer_task(task, "主角养什么宠物？", user_id=self.user_id)

            clause = payload["query_plan"]["clauses"][0]
            self.assertEqual(clause["route"], "fact-first")
            self.assertEqual(payload["answer"]["answer_type"], "fact_lookup_query")
            self.assertEqual(payload["answer"]["supporting_facts"][0]["field_key"], "long_term_facts.relationships.pets")
            self.assertEqual(payload["answer"]["supporting_facts"][0]["value"], ["猫"])
            self.assertGreaterEqual(payload["answer"]["matched_event_count"], 1)
            self.assertEqual(payload["matched_events"][0]["event_id"], "EVT_HOME_001")
            self.assertGreaterEqual(len(payload["answer"]["supporting_photos"]), 1)
            self.assertEqual(payload["answer"]["supporting_photos"][0]["photo_url"], "/assets/uploads/004_cat.png")
            self.assertEqual(payload["matched_events"][0]["cover_photo_url"], "/assets/uploads/004_cat.png")
            self.assertEqual(payload["matched_events"][0]["photos"][0]["photo_url"], "/assets/uploads/004_cat.png")

    def test_query_engine_relationship_rank_returns_photo_backed_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task = self._synthetic_task(task_id=f"task_{uuid.uuid4().hex[:8]}", task_dir=Path(tmpdir), version="v0327-db-query")
            engine = self._engine()

            with self._materialization_patches(), patch.object(StructuredQueryPlanner, "_llm_route", return_value=None), patch.object(
                AnswerWriterLLM,
                "_llm_summary",
                return_value=None,
            ):
                payload = engine.answer_task(task, "主角和谁关系最好？", user_id=self.user_id)

            self.assertEqual(payload["answer"]["answer_type"], "relationship_rank_query")
            self.assertEqual(payload["query_plan"]["plan_type"], "relationship_first")
            self.assertEqual(payload["answer"]["supporting_relationships"][0]["target_person_id"], "Person_002")
            self.assertGreaterEqual(len(payload["matched_events"]), 1)
            self.assertIn(payload["matched_events"][0]["event_id"], {"EVT_CONCERT_001", "EVT_EXHIBITION_001"})
            self.assertIn("photo_001", payload["answer"]["original_photo_ids"])
            self.assertTrue(any(str(item.get("photo_url") or "").startswith("/assets/uploads/") for item in payload["answer"]["supporting_photos"]))

    def test_query_engine_hybrid_judgement_returns_contradicted_for_already_working(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task = self._synthetic_task(task_id=f"task_{uuid.uuid4().hex[:8]}", task_dir=Path(tmpdir), version="v0327-db-query")
            engine = self._engine()

            with self._materialization_patches(), patch.object(StructuredQueryPlanner, "_llm_route", return_value=None), patch.object(
                AnswerWriterLLM,
                "_llm_summary",
                return_value=None,
            ):
                payload = engine.answer_task(
                    task,
                    "如果只看可审计证据，能不能判断主角已经工作？如果不能，请说明为什么。",
                    user_id=self.user_id,
                )

            self.assertEqual(payload["query_plan"]["composition_mode"], "judge_then_explain")
            self.assertEqual(payload["answer"]["answer_type"], "composite_query")
            self.assertEqual(payload["answer"]["judgement_status"], "contradicted")
            self.assertTrue(any(item["route"] == "hybrid-judgement" for item in payload["query_plan"]["clauses"]))

    def test_query_engine_multi_clause_preserves_all_matched_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task = self._synthetic_task(task_id=f"task_{uuid.uuid4().hex[:8]}", task_dir=Path(tmpdir), version="v0327-db-query")
            engine = self._engine()

            with self._materialization_patches(), patch.object(StructuredQueryPlanner, "_llm_route", return_value=None), patch.object(
                AnswerWriterLLM,
                "_llm_summary",
                return_value=None,
            ):
                payload = engine.answer_task(
                    task,
                    "主角的伴侣是谁，主角养什么宠物，地点锚点在哪里？",
                    user_id=self.user_id,
                )

            self.assertEqual(payload["query_plan"]["composition_mode"], "parallel")
            self.assertEqual(len(payload["answer"]["clause_results"]), 3)
            self.assertGreaterEqual(payload["answer"]["matched_event_count"], 2)
            self.assertGreaterEqual(len(payload["matched_events"]), 2)
            routes = {item["route"] for item in payload["query_plan"]["clauses"]}
            self.assertIn("relationship-first", routes)
            self.assertIn("fact-first", routes)

    def test_query_engine_date_query_normalizes_time_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task = self._synthetic_task(task_id=f"task_{uuid.uuid4().hex[:8]}", task_dir=Path(tmpdir), version="v0327-db-query")
            engine = self._engine()

            with self._materialization_patches(), patch.object(StructuredQueryPlanner, "_llm_route", return_value=None), patch.object(
                AnswerWriterLLM,
                "_llm_summary",
                return_value=None,
            ):
                payload = engine.answer_task(task, "2026年1月18日主角做了哪些事情？", user_id=self.user_id)

            clause = payload["query_plan"]["clauses"][0]
            self.assertEqual(clause["route"], "event-first")
            self.assertEqual(clause["time_windows"][0]["label"], "date_2026-01-18")
            self.assertGreaterEqual(len(payload["matched_events"]), 1)
            self.assertEqual(payload["matched_events"][0]["event_id"], "EVT_EXHIBITION_001")

    def test_memory_query_endpoint_routes_v0327_db_task_to_query_v1_with_new_fields(self) -> None:
        create_response = self.client.post("/api/tasks", json={"version": "v0327-db-query"})
        self.assertEqual(create_response.status_code, 200)
        task_id = create_response.json()["task_id"]
        self.task_ids.append(task_id)

        payloads = self._task_payloads()
        uploads = [
            {
                "image_id": "photo_001",
                "filename": "concert.png",
                "stored_filename": "001_concert.png",
                "path": "uploads/001_concert.png",
                "url": "/assets/uploads/001_concert.png",
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
                "content_type": "image/png",
                "width": 1200,
                "height": 900,
                "source_hash": "hash-photo-002",
                "timestamp": "2026-03-05T19:30:00",
            },
            {
                "image_id": "photo_003",
                "filename": "exhibition.png",
                "stored_filename": "003_exhibition.png",
                "path": "uploads/003_exhibition.png",
                "url": "/assets/uploads/003_exhibition.png",
                "content_type": "image/png",
                "width": 1200,
                "height": 900,
                "source_hash": "hash-photo-003",
                "timestamp": "2026-01-18T15:00:00",
            },
            {
                "image_id": "photo_004",
                "filename": "cat.png",
                "stored_filename": "004_cat.png",
                "path": "uploads/004_cat.png",
                "url": "/assets/uploads/004_cat.png",
                "content_type": "image/png",
                "width": 1200,
                "height": 900,
                "source_hash": "hash-photo-004",
                "timestamp": "2026-01-10T18:30:00",
            },
        ]
        task_store.append_uploads(task_id, uploads, status="completed", stage="completed")
        family_dir = task_store.task_dir(task_id) / "v0325"
        family_dir.mkdir(parents=True, exist_ok=True)
        for filename, payload in (
            ("relationship_dossiers.json", payloads["relationship_dossiers"]),
            ("group_artifacts.json", payloads["group_artifacts"]),
            ("structured_profile.json", payloads["structured_profile"]),
            ("profile_fact_decisions.json", payloads["profile_fact_decisions"]),
        ):
            (family_dir / filename).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

        task_store.update_task(
            task_id,
            result={
                "face_recognition": {
                    "primary_person_id": "Person_001",
                    "images": [
                        {"image_id": "photo_001", "source_hash": "hash-photo-001", "timestamp": "2026-03-01T20:00:00"},
                        {"image_id": "photo_002", "source_hash": "hash-photo-002", "timestamp": "2026-03-05T19:30:00"},
                        {"image_id": "photo_003", "source_hash": "hash-photo-003", "timestamp": "2026-01-18T15:00:00"},
                        {"image_id": "photo_004", "source_hash": "hash-photo-004", "timestamp": "2026-01-10T18:30:00"},
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

        with self._materialization_patches(), patch.object(StructuredQueryPlanner, "_llm_route", return_value=None), patch.object(
            AnswerWriterLLM,
            "_llm_summary",
            return_value=None,
        ), patch("backend.app.MEMORY_QUERY_V1_ENABLED", True):
            response = self.client.post(
                f"/api/tasks/{task_id}/memory/query",
                json={"question": "主角养什么宠物？"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query_plan"]["engine"], "query_v1")
        self.assertEqual(payload["query_plan"]["router_version"], "route_plan_v1")
        self.assertEqual(payload["answer"]["answer_type"], "fact_lookup_query")
        self.assertIn("supporting_facts", payload["answer"])
        self.assertIn("clause_results", payload["answer"])
        self.assertGreaterEqual(payload["answer"]["matched_event_count"], 1)
        self.assertEqual(payload["answer"]["supporting_facts"][0]["field_key"], "long_term_facts.relationships.pets")
        self.assertEqual(payload["answer"]["supporting_photos"][0]["photo_url"], "/assets/uploads/004_cat.png")
        self.assertEqual(payload["matched_events"][0]["photos"][0]["photo_url"], "/assets/uploads/004_cat.png")
        self.assertIn("supporting_photo_urls", payload["answer"]["clause_results"][0])

    def _engine(self) -> QueryEngineV1:
        planner = StructuredQueryPlanner(now=datetime(2026, 3, 29, 12, 0, 0))
        writer = AnswerWriterLLM()
        return QueryEngineV1(
            store=self.query_store,
            planner=planner,
            writer=writer,
            embedder=_FakeEmbedder(),
            now=datetime(2026, 3, 29, 12, 0, 0),
        )

    @contextmanager
    def _materialization_patches(self):
        with patch("backend.query_v1.materializer.MilvusQueryIndexer.publish", return_value={"status": "skipped"}), patch(
            "backend.query_v1.materializer.Neo4jQueryIndexer.publish",
            return_value={"status": "skipped"},
        ):
            yield

    def _synthetic_task(self, *, task_id: str, task_dir: Path, version: str) -> dict:
        task_dir.mkdir(parents=True, exist_ok=True)
        family_dir = task_dir / "v0325"
        family_dir.mkdir(parents=True, exist_ok=True)
        payloads = self._task_payloads()
        for filename, payload in (
            ("relationship_dossiers.json", payloads["relationship_dossiers"]),
            ("group_artifacts.json", payloads["group_artifacts"]),
            ("structured_profile.json", payloads["structured_profile"]),
            ("profile_fact_decisions.json", payloads["profile_fact_decisions"]),
        ):
            (family_dir / filename).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
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
                {
                    "image_id": "photo_003",
                    "path": "uploads/003_exhibition.png",
                    "url": "/assets/uploads/003_exhibition.png",
                    "timestamp": "2026-01-18T15:00:00",
                    "content_type": "image/png",
                    "width": 1200,
                    "height": 900,
                    "source_hash": "hash-photo-003",
                },
                {
                    "image_id": "photo_004",
                    "path": "uploads/004_cat.png",
                    "url": "/assets/uploads/004_cat.png",
                    "timestamp": "2026-01-10T18:30:00",
                    "content_type": "image/png",
                    "width": 1200,
                    "height": 900,
                    "source_hash": "hash-photo-004",
                },
            ],
            "result": {
                "face_recognition": {
                    "primary_person_id": "Person_001",
                    "images": [
                        {"image_id": "photo_001", "source_hash": "hash-photo-001", "timestamp": "2026-03-01T20:00:00"},
                        {"image_id": "photo_002", "source_hash": "hash-photo-002", "timestamp": "2026-03-05T19:30:00"},
                        {"image_id": "photo_003", "source_hash": "hash-photo-003", "timestamp": "2026-01-18T15:00:00"},
                        {"image_id": "photo_004", "source_hash": "hash-photo-004", "timestamp": "2026-01-10T18:30:00"},
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
                    "narrative_synthesis": "主角和伴侣一起去了演唱会 concert live。",
                    "participant_person_ids": ["Person_001", "Person_002"],
                    "depicted_person_ids": ["Person_001", "Person_002"],
                    "place_refs": ["Shanghai Stadium"],
                    "tags": ["concert", "live", "music", "演唱会"],
                    "objective_fact": {"scene_description": "concert stage with bright lights"},
                    "persona_evidence": {"social": ["和伴侣一起听 live"]},
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
                {
                    "event_id": "EVT_EXHIBITION_001",
                    "anchor_photo_id": "photo_003",
                    "supporting_photo_ids": ["photo_003"],
                    "started_at": "2026-01-18T15:00:00",
                    "ended_at": "2026-01-18T18:00:00",
                    "title": "Art Exhibition Afternoon",
                    "narrative_synthesis": "主角和伴侣一起去看了艺术展 exhibition。",
                    "participant_person_ids": ["Person_001", "Person_002"],
                    "depicted_person_ids": ["Person_001", "Person_002"],
                    "place_refs": ["Smile Park"],
                    "tags": ["art", "exhibition", "展览"],
                    "objective_fact": {"scene_description": "colorful pop art exhibition hall"},
                    "persona_evidence": {"social": ["和伴侣一起看展"]},
                    "confidence": 0.89,
                },
                {
                    "event_id": "EVT_HOME_001",
                    "anchor_photo_id": "photo_004",
                    "supporting_photo_ids": ["photo_004"],
                    "started_at": "2026-01-10T18:30:00",
                    "ended_at": "2026-01-10T19:30:00",
                    "title": "Cat at Home",
                    "narrative_synthesis": "主角在家抱着猫拍照。",
                    "participant_person_ids": ["Person_001"],
                    "depicted_person_ids": ["Person_001"],
                    "place_refs": ["Shanghai Baoshan Home"],
                    "tags": ["cat", "home", "pet"],
                    "objective_fact": {"scene_description": "home scene with a cat on the sofa"},
                    "persona_evidence": {"life": ["和猫一起在家"]},
                    "confidence": 0.83,
                },
            ],
            "vp1_observations": [
                {
                    "photo_id": "photo_001",
                    "confidence": 0.88,
                    "vlm_analysis": {
                        "summary": "concert live performance with a romantic partner",
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
                {
                    "photo_id": "photo_003",
                    "confidence": 0.82,
                    "vlm_analysis": {
                        "summary": "art exhibition with a romantic partner",
                        "scene": {"location_detected": "Smile Park"},
                        "event": {"activity": "exhibition", "interaction": "watching artworks together"},
                        "details": ["art wall", "gallery lighting"],
                        "key_objects": ["painting", "installation"],
                    },
                    "ocr_hits": ["ART", "SMILE PARK"],
                },
                {
                    "photo_id": "photo_004",
                    "confidence": 0.79,
                    "vlm_analysis": {
                        "summary": "pet cat resting on the sofa at home",
                        "scene": {"location_detected": "Shanghai Baoshan Home"},
                        "event": {"activity": "resting at home", "interaction": "holding a pet cat"},
                        "details": ["orange cat", "living room"],
                        "key_objects": ["cat", "sofa"],
                    },
                },
            ],
            "lp2_relationships": [
                {
                    "relationship_id": "REL_P002",
                    "person_id": "Person_002",
                    "relationship_type": "romantic",
                    "status": "growing",
                    "confidence": 0.95,
                    "intimacy_score": 0.93,
                    "reasoning": "长期稳定的演唱会和看展伴侣线索。",
                    "supporting_event_ids": ["EVT_CONCERT_001", "EVT_EXHIBITION_001"],
                    "supporting_photo_ids": ["photo_001", "photo_003"],
                    "shared_events": [{"event_id": "EVT_CONCERT_001"}, {"event_id": "EVT_EXHIBITION_001"}],
                    "evidence": {"photo_ids": ["photo_001", "photo_003"]},
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
                    "interaction_signals": ["shared concerts", "repeated exhibition dates"],
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
                    "strong_evidence_refs": [{"event_id": "EVT_CONCERT_001"}, {"event_id": "EVT_EXHIBITION_001"}],
                    "reason": "长期一对一演唱会和看展活动",
                }
            ],
            "structured_profile": {
                "long_term_facts": {
                    "interests": {
                        "live_music": {"value": True, "confidence": 0.9},
                        "art_exhibitions": {"value": True, "confidence": 0.86},
                    },
                    "relationships": {
                        "pets": {"value": ["猫"], "confidence": 0.88},
                        "close_circle_size": {"value": 3, "confidence": 0.83},
                    },
                    "geography": {
                        "location_anchors": {"value": ["上海宝山区"], "confidence": 0.84},
                    },
                },
                "social_identity": {
                    "career_phase": {"value": "学生阶段，尚未正式步入职场", "confidence": 0.87},
                },
            },
            "profile_fact_decisions": [
                {
                    "field_key": "long_term_facts.relationships.pets",
                    "final": {
                        "value": ["猫"],
                        "confidence": 0.88,
                        "evidence": {"event_ids": ["EVT_HOME_001"], "photo_ids": ["photo_004"]},
                    },
                },
                {
                    "field_key": "long_term_facts.geography.location_anchors",
                    "final": {
                        "value": ["上海宝山区"],
                        "confidence": 0.84,
                        "evidence": {"event_ids": ["EVT_HOME_001"], "photo_ids": ["photo_004"]},
                    },
                },
                {
                    "field_key": "long_term_facts.relationships.close_circle_size",
                    "final": {
                        "value": 3,
                        "confidence": 0.83,
                        "evidence": {"event_ids": ["EVT_CONCERT_001", "EVT_EXHIBITION_001"], "photo_ids": ["photo_001", "photo_003"]},
                    },
                },
                {
                    "field_key": "social_identity.career_phase",
                    "final": {
                        "value": "学生阶段，尚未正式步入职场",
                        "confidence": 0.87,
                        "evidence": {"event_ids": ["EVT_EXHIBITION_001"], "photo_ids": ["photo_003"]},
                    },
                },
                {
                    "field_key": "long_term_facts.interests.live_music",
                    "final": {
                        "value": True,
                        "confidence": 0.9,
                        "evidence": {"event_ids": ["EVT_CONCERT_001"], "photo_ids": ["photo_001"]},
                    },
                },
            ],
        }


if __name__ == "__main__":
    unittest.main()
