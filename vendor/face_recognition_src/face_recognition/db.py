from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .models import FaceRecord, FailedImageRecord, ImageRecord, PersonRecord
from .utils import blob_to_embedding, embedding_to_blob, ensure_parent_dir


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS images (
    image_hash TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    processed_at TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS persons (
    person_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    representative_face TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS faces (
    face_id TEXT PRIMARY KEY,
    image_hash TEXT NOT NULL,
    person_id TEXT NOT NULL,
    bbox TEXT NOT NULL,
    faiss_id INTEGER NOT NULL UNIQUE,
    FOREIGN KEY (image_hash) REFERENCES images(image_hash),
    FOREIGN KEY (person_id) REFERENCES persons(person_id)
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    face_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL,
    FOREIGN KEY (face_id) REFERENCES faces(face_id)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    dataset_name TEXT NOT NULL,
    input_dir TEXT NOT NULL,
    db_path TEXT NOT NULL,
    index_path TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    requested_providers TEXT NOT NULL,
    applied_providers TEXT,
    preflight_validate INTEGER NOT NULL,
    resume_enabled INTEGER NOT NULL,
    batch_size INTEGER NOT NULL,
    batch_retry_limit INTEGER NOT NULL,
    batch_retry_backoff_seconds REAL NOT NULL,
    scanned_images INTEGER NOT NULL DEFAULT 0,
    imported_images INTEGER NOT NULL DEFAULT 0,
    duplicate_images INTEGER NOT NULL DEFAULT 0,
    failed_files INTEGER NOT NULL DEFAULT 0,
    bad_images INTEGER NOT NULL DEFAULT 0,
    skipped_failed_images INTEGER NOT NULL DEFAULT 0,
    no_face_images INTEGER NOT NULL DEFAULT 0,
    detected_faces INTEGER NOT NULL DEFAULT 0,
    new_persons INTEGER NOT NULL DEFAULT 0,
    batch_retries INTEGER NOT NULL DEFAULT 0,
    total_seconds REAL NOT NULL DEFAULT 0,
    average_image_seconds REAL NOT NULL DEFAULT 0,
    cumulative_images INTEGER NOT NULL DEFAULT 0,
    cumulative_faces INTEGER NOT NULL DEFAULT 0,
    cumulative_persons INTEGER NOT NULL DEFAULT 0,
    cumulative_no_face_images INTEGER NOT NULL DEFAULT 0,
    cumulative_failed_images INTEGER NOT NULL DEFAULT 0,
    artifact_text_path TEXT,
    artifact_json_path TEXT
);

CREATE TABLE IF NOT EXISTS failed_images (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    image_hash TEXT,
    stage TEXT NOT NULL,
    error_code TEXT NOT NULL,
    error_message TEXT NOT NULL,
    retryable INTEGER NOT NULL DEFAULT 0,
    retry_count INTEGER NOT NULL DEFAULT 0,
    file_size INTEGER,
    file_mtime_ns INTEGER,
    failed_at TEXT NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0,
    resolved_at TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_faces_image_hash ON faces(image_hash);
CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces(person_id);
CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
CREATE INDEX IF NOT EXISTS idx_failed_images_run_id ON failed_images(run_id);
CREATE INDEX IF NOT EXISTS idx_failed_images_file_path ON failed_images(file_path);
"""


class MetadataStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        ensure_parent_dir(self.db_path)
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.executescript(SCHEMA)
        self._next_person_number = self._load_next_person_number()

    def close(self) -> None:
        self.connection.close()

    def image_exists(self, image_hash: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM images WHERE image_hash = ?",
            (image_hash,),
        ).fetchone()
        return row is not None

    def has_existing_state(self) -> bool:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM images").fetchone()
        return bool(row and int(row["count"]) > 0)

    def next_person_id(self) -> str:
        person_id = "Person_{:03d}".format(self._next_person_number)
        self._next_person_number += 1
        return person_id

    def start_run(self, report: object) -> None:
        self.connection.execute(
            """
            INSERT INTO runs (
                run_id,
                dataset_name,
                input_dir,
                db_path,
                index_path,
                status,
                started_at,
                requested_providers,
                preflight_validate,
                resume_enabled,
                batch_size,
                batch_retry_limit,
                batch_retry_backoff_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report.run_id,
                report.dataset_name,
                str(report.config.input_dir),
                str(report.config.db_path),
                str(report.config.index_path),
                report.run_status,
                report.started_at.isoformat(),
                json.dumps(list(report.config.providers)),
                int(report.config.preflight_validate),
                int(report.config.resume),
                report.config.batch_size,
                report.config.batch_retry_limit,
                report.config.batch_retry_backoff_seconds,
            ),
        )
        self.connection.commit()

    def complete_run(self, report: object) -> None:
        text_path = report.artifact_paths.get("text")
        json_path = report.artifact_paths.get("json")
        self.connection.execute(
            """
            UPDATE runs
            SET status = ?,
                completed_at = ?,
                applied_providers = ?,
                scanned_images = ?,
                imported_images = ?,
                duplicate_images = ?,
                failed_files = ?,
                bad_images = ?,
                skipped_failed_images = ?,
                no_face_images = ?,
                detected_faces = ?,
                new_persons = ?,
                batch_retries = ?,
                total_seconds = ?,
                average_image_seconds = ?,
                cumulative_images = ?,
                cumulative_faces = ?,
                cumulative_persons = ?,
                cumulative_no_face_images = ?,
                cumulative_failed_images = ?,
                artifact_text_path = ?,
                artifact_json_path = ?
            WHERE run_id = ?
            """,
            (
                report.run_status,
                report.completed_at.isoformat() if report.completed_at else None,
                json.dumps(list(report.applied_providers)),
                report.scanned_images,
                report.imported_images,
                report.duplicate_images,
                report.failed_files,
                report.bad_images,
                report.skipped_failed_images,
                report.no_face_images,
                report.detected_faces,
                report.new_persons,
                report.batch_retries,
                report.total_seconds,
                report.average_image_seconds,
                int(report.cumulative.get("images", 0)),
                int(report.cumulative.get("faces", 0)),
                int(report.cumulative.get("persons", 0)),
                int(report.cumulative.get("no_face_images", 0)),
                int(report.cumulative.get("failed_images", 0)),
                text_path,
                json_path,
                report.run_id,
            ),
        )
        self.connection.commit()

    def record_failed_image(self, record: FailedImageRecord) -> None:
        self.connection.execute(
            """
            INSERT INTO failed_images (
                run_id,
                file_path,
                image_hash,
                stage,
                error_code,
                error_message,
                retryable,
                retry_count,
                file_size,
                file_mtime_ns,
                failed_at,
                resolved,
                resolved_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)
            """,
            (
                record.run_id,
                record.file_path,
                record.image_hash,
                record.stage,
                record.error_code,
                record.error_message,
                int(record.retryable),
                record.retry_count,
                record.file_size,
                record.file_mtime_ns,
                record.failed_at,
            ),
        )
        self.connection.commit()

    def resolve_failed_image(self, file_path: str) -> None:
        resolved_at = datetime.now(timezone.utc).isoformat()
        self.connection.execute(
            """
            UPDATE failed_images
            SET resolved = 1,
                resolved_at = ?
            WHERE file_path = ?
              AND resolved = 0
            """,
            (resolved_at, file_path),
        )
        self.connection.commit()

    def is_known_non_retryable_failure(
        self,
        file_path: str,
        file_size: Optional[int],
        file_mtime_ns: Optional[int],
    ) -> bool:
        row = self.connection.execute(
            """
            SELECT 1
            FROM failed_images
            WHERE file_path = ?
              AND retryable = 0
              AND resolved = 0
              AND COALESCE(file_size, -1) = COALESCE(?, -1)
              AND COALESCE(file_mtime_ns, -1) = COALESCE(?, -1)
            LIMIT 1
            """,
            (file_path, file_size, file_mtime_ns),
        ).fetchone()
        return row is not None

    def load_faiss_person_map(self) -> Dict[int, str]:
        rows = self.connection.execute(
            "SELECT faiss_id, person_id FROM faces ORDER BY faiss_id ASC"
        ).fetchall()
        return {int(row["faiss_id"]): str(row["person_id"]) for row in rows}

    def count_embeddings(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS count FROM face_embeddings").fetchone()
        return int(row["count"]) if row is not None else 0

    def cumulative_metrics(self) -> Dict[str, int]:
        images = self.connection.execute("SELECT COUNT(*) AS count FROM images").fetchone()
        faces = self.connection.execute("SELECT COUNT(*) AS count FROM faces").fetchone()
        persons = self.connection.execute("SELECT COUNT(*) AS count FROM persons").fetchone()
        no_face_images = self.connection.execute(
            """
            SELECT COUNT(*) AS count
            FROM images
            WHERE image_hash NOT IN (SELECT DISTINCT image_hash FROM faces)
            """
        ).fetchone()
        failed_images = self.connection.execute(
            "SELECT COUNT(*) AS count FROM failed_images WHERE resolved = 0"
        ).fetchone()
        return {
            "images": int(images["count"]) if images is not None else 0,
            "faces": int(faces["count"]) if faces is not None else 0,
            "persons": int(persons["count"]) if persons is not None else 0,
            "no_face_images": int(no_face_images["count"]) if no_face_images is not None else 0,
            "failed_images": int(failed_images["count"]) if failed_images is not None else 0,
        }

    def iter_embeddings_ordered(self) -> Iterable[Tuple[int, Sequence[float]]]:
        rows = self.connection.execute(
            """
            SELECT faces.faiss_id AS faiss_id, face_embeddings.embedding AS embedding, face_embeddings.dim AS dim
            FROM faces
            JOIN face_embeddings ON face_embeddings.face_id = faces.face_id
            ORDER BY faces.faiss_id ASC
            """
        ).fetchall()

        for expected, row in enumerate(rows):
            faiss_id = int(row["faiss_id"])
            if faiss_id != expected:
                raise ValueError(
                    "faiss_id sequence is not contiguous at position {}".format(expected)
                )
            embedding = blob_to_embedding(row["embedding"])
            if len(embedding) != int(row["dim"]):
                raise ValueError("embedding dimension mismatch for faiss_id {}".format(faiss_id))
            yield faiss_id, embedding

    def persist_batch(
        self,
        images: Sequence[ImageRecord],
        persons: Sequence[PersonRecord],
        faces: Sequence[FaceRecord],
    ) -> None:
        if not images and not persons and not faces:
            return

        cursor = self.connection.cursor()
        cursor.execute("BEGIN")
        try:
            cursor.executemany(
                """
                INSERT INTO images (image_hash, file_path, processed_at, width, height)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.image_hash,
                        item.file_path,
                        item.processed_at,
                        item.width,
                        item.height,
                    )
                    for item in images
                ],
            )
            cursor.executemany(
                """
                INSERT INTO persons (person_id, created_at, representative_face)
                VALUES (?, ?, ?)
                """,
                [
                    (item.person_id, item.created_at, item.representative_face)
                    for item in persons
                ],
            )
            cursor.executemany(
                """
                INSERT INTO faces (face_id, image_hash, person_id, bbox, faiss_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        item.face_id,
                        item.image_hash,
                        item.person_id,
                        item.bbox_json,
                        item.faiss_id,
                    )
                    for item in faces
                ],
            )
            cursor.executemany(
                """
                INSERT INTO face_embeddings (face_id, embedding, dim)
                VALUES (?, ?, ?)
                """,
                [
                    (
                        item.face_id,
                        sqlite3.Binary(embedding_to_blob(item.embedding)),
                        len(item.embedding),
                    )
                    for item in faces
                ],
            )
        except Exception:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def _load_next_person_number(self) -> int:
        row = self.connection.execute(
            """
            SELECT MAX(CAST(SUBSTR(person_id, 8) AS INTEGER)) AS max_id
            FROM persons
            WHERE person_id LIKE 'Person_%'
            """
        ).fetchone()
        max_id = row["max_id"] if row is not None else None
        return int(max_id or 0) + 1
