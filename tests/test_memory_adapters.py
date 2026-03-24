from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memory_module.adapters import MemoryStoragePublisher, MilvusStorageAdapter, Neo4jStorageAdapter
from memory_module.embeddings import EmbeddingProvider


class MemoryAdapterTests(unittest.TestCase):
    def test_external_publish_skips_when_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = {
                "redis": {"profile_core": {"key": "profile:user:core", "fields": {}}},
                "neo4j": {"nodes": {}, "edges": []},
                "milvus": {"segments": []},
            }
            with patch("memory_module.adapters.MEMORY_EXTERNAL_SINKS_ENABLED", False):
                report = MemoryStoragePublisher(task_dir=tmpdir).publish(storage, user_id="user_alpha")

            self.assertFalse(report["enabled"])
            self.assertEqual(report["redis"]["status"], "skipped")
            self.assertEqual(report["neo4j"]["status"], "skipped")
            self.assertEqual(report["milvus"]["status"], "skipped")
            self.assertTrue(Path(report["report_path"]).exists())

    def test_embedding_provider_falls_back_to_deterministic_stub(self) -> None:
        provider = EmbeddingProvider.from_config(dim=8)
        first, source_first, model_first = provider.embed_text("hello world")
        second, source_second, model_second = provider.embed_text("hello world")
        third, _, _ = provider.embed_text("another text")

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertEqual(len(first), 8)
        self.assertAlmostEqual(sum(value * value for value in first), 1.0, places=3)
        self.assertEqual(source_first, "textual_stub")
        self.assertEqual(source_second, "textual_stub")
        self.assertEqual(model_first, "textual_stub_v1")
        self.assertEqual(model_second, "textual_stub_v1")

    def test_milvus_adapter_publishes_to_local_db_uri(self) -> None:
        try:
            import milvus_lite  # noqa: F401
        except Exception:
            self.skipTest("milvus-lite is not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = MilvusStorageAdapter(user_id="user_alpha")
            db_path = Path(tmpdir) / "memory_milvus.db"
            payload = {
                "segments": [
                    {
                        "segment_uuid": "seg-1",
                        "photo_uuid": "photo-1",
                        "event_uuid": "",
                        "person_uuid": "",
                        "session_uuid": "",
                        "segment_type": "scene",
                        "text": "sunny park walk",
                        "sparse_terms": ["sunny", "park", "walk"],
                        "embedding_source": "textual_stub",
                        "importance_score": 0.8,
                        "evidence_refs": [{"photo_id": "photo-1"}],
                    }
                ]
            }

            with (
                patch("memory_module.adapters.MEMORY_MILVUS_URI", str(db_path)),
                patch("memory_module.adapters.MEMORY_MILVUS_COLLECTION", "memory_segments_test"),
                patch("memory_module.adapters.MEMORY_MILVUS_VECTOR_DIM", 8),
            ):
                report = adapter.publish(payload)

            self.assertEqual(report["status"], "published")
            self.assertEqual(report["mode"], "local-db")
            self.assertEqual(report["record_count"], 1)
            self.assertEqual(report["collections"][0]["collection"], "memory_segments_test")
            self.assertTrue(db_path.exists())

    def test_milvus_adapter_publishes_units_and_evidence_to_dual_collections(self) -> None:
        try:
            import milvus_lite  # noqa: F401
        except Exception:
            self.skipTest("milvus-lite is not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter = MilvusStorageAdapter(user_id="user_alpha")
            db_path = Path(tmpdir) / "memory_dual.db"
            payload = {
                "memory_units_v2": [
                    {
                        "unit_id": "unit-1",
                        "pipeline_family": "v0321_3",
                        "source_type": "event_revision",
                        "event_root_id": "event-root-1",
                        "event_revision_id": "event-rev-1",
                        "title": "Concert Night",
                        "summary": "Live concert with friends",
                        "retrieval_text": "Concert Night\nLive concert with friends",
                        "started_at": "2026-03-20T20:00:00",
                        "ended_at": "2026-03-20T22:00:00",
                        "original_photo_ids": ["orig-1"],
                        "participant_person_ids": ["Person_001"],
                        "place_refs": ["Shanghai"],
                        "evidence_ids": ["ev-1"],
                        "confidence": 0.88,
                    }
                ],
                "memory_evidence_v2": [
                    {
                        "evidence_id": "ev-1",
                        "pipeline_family": "v0321_3",
                        "source_type": "atomic_evidence",
                        "parent_unit_id": "event-rev-1",
                        "event_root_id": "event-root-1",
                        "event_revision_id": "event-rev-1",
                        "event_title": "Concert Night",
                        "evidence_type": "brand",
                        "value_or_text": "MOET",
                        "provenance": "brand",
                        "retrieval_text": "brand: MOET",
                        "original_photo_ids": ["orig-1"],
                        "participant_person_ids": ["Person_001"],
                        "place_refs": ["Shanghai"],
                        "confidence": 0.77,
                    }
                ],
            }

            with (
                patch("memory_module.adapters.MEMORY_MILVUS_URI", str(db_path)),
                patch("memory_module.adapters.MEMORY_MILVUS_UNITS_COLLECTION", "memory_units_v2_test"),
                patch("memory_module.adapters.MEMORY_MILVUS_EVIDENCE_COLLECTION", "memory_evidence_v2_test"),
                patch("memory_module.adapters.MEMORY_MILVUS_VECTOR_DIM", 8),
            ):
                report = adapter.publish(payload)

            self.assertEqual(report["status"], "published")
            self.assertEqual(report["record_count"], 2)
            self.assertEqual(len(report["collections"]), 2)
            collection_names = {item["collection"] for item in report["collections"]}
            self.assertEqual(collection_names, {"memory_units_v2_test", "memory_evidence_v2_test"})
            self.assertTrue(db_path.exists())

    def test_neo4j_adapter_flattens_nested_properties(self) -> None:
        adapter = Neo4jStorageAdapter()

        props = adapter._sanitize_properties(
            {
                "photo_id": "photo_001",
                "location": {},
                "location_hint": {"name": "unknown", "geo": {"lat": 39.9, "lng": 116.3}},
                "tags": ["music", "festival"],
                "evidence_refs": [{"photo_id": "photo_001"}],
                "captured_at": "2026-03-16T01:25:46.093583",
            }
        )

        self.assertEqual(props["photo_id"], "photo_001")
        self.assertEqual(props["location_hint_name"], "unknown")
        self.assertEqual(props["location_hint_geo_lat"], 39.9)
        self.assertEqual(props["location_hint_geo_lng"], 116.3)
        self.assertEqual(props["tags"], ["music", "festival"])
        self.assertIn("evidence_refs_json", props)
        self.assertNotIn("location", props)


if __name__ == "__main__":
    unittest.main()
