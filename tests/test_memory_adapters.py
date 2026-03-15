from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memory_module.adapters import MemoryStoragePublisher, MilvusStorageAdapter


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

    def test_milvus_adapter_produces_deterministic_stub_vectors(self) -> None:
        adapter = MilvusStorageAdapter(user_id="user_alpha")
        first = adapter._deterministic_vector("hello world", 8)
        second = adapter._deterministic_vector("hello world", 8)
        third = adapter._deterministic_vector("another text", 8)

        self.assertEqual(first, second)
        self.assertNotEqual(first, third)
        self.assertEqual(len(first), 8)
        self.assertAlmostEqual(sum(value * value for value in first), 1.0, places=3)

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
            self.assertEqual(report["segment_count"], 1)
            self.assertTrue(db_path.exists())


if __name__ == "__main__":
    unittest.main()
