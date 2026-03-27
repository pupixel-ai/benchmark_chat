from __future__ import annotations

import unittest
import uuid
from datetime import datetime

from sqlalchemy import delete

from backend.db import SessionLocal
from backend.memory_models import FaceEmbeddingRecord
from backend import models as _legacy_models  # noqa: F401


class MemoryModelsTests(unittest.TestCase):
    def test_face_embedding_vector_type_round_trips_in_sqlite(self) -> None:
        face_id = f"test_face_{uuid.uuid4().hex[:12]}"
        task_id = f"test_task_{uuid.uuid4().hex[:12]}"
        user_id = f"test_user_{uuid.uuid4().hex[:12]}"
        record_id = uuid.uuid4().hex
        now = datetime.now()
        embedding = [0.1, -0.2, 0.3]

        with SessionLocal() as session:
            session.add(
                FaceEmbeddingRecord(
                    id=record_id,
                    face_id=face_id,
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=None,
                    faiss_id=7,
                    embedding=embedding,
                    embedding_dim=len(embedding),
                    embedding_model="Facenet512",
                    embedding_version=327,
                    embedding_hash="hash",
                    source_backend="sqlite_test",
                    created_at=now,
                    updated_at=now,
                )
            )
            session.commit()

            saved = session.get(FaceEmbeddingRecord, record_id)
            self.assertIsNotNone(saved)
            assert saved is not None
            self.assertEqual(saved.embedding_dim, len(embedding))
            self.assertEqual(saved.embedding, embedding)

            session.execute(delete(FaceEmbeddingRecord).where(FaceEmbeddingRecord.id == record_id))
            session.commit()


if __name__ == "__main__":
    unittest.main()
