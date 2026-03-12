from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

from .config import PipelineConfig
from .db import MetadataStore
from .engine import FaceEngine
from .image_io import load_image, resize_image
from .index_store import SimilarityIndexStore
from .models import FaceRecord, FailedImageRecord, ImageRecord, PendingBatch, PersonRecord
from .report import RunReport
from .utils import compute_sha256, iter_image_files


class PipelineRunner:
    def __init__(
        self,
        config: PipelineConfig,
        engine: Optional[FaceEngine] = None,
        metadata_store: Optional[MetadataStore] = None,
        index_store: Optional[SimilarityIndexStore] = None,
        image_loader: Optional[Callable[[Path], object]] = None,
        hash_file: Optional[Callable[[Path], str]] = None,
    ) -> None:
        self.config = config
        self.engine = engine or FaceEngine(config)
        self.metadata_store = metadata_store or MetadataStore(config.db_path)
        self.index_store = index_store or SimilarityIndexStore(config.index_path)
        self.image_loader = image_loader or load_image
        self.hash_file = hash_file or compute_sha256

    def close(self) -> None:
        self.metadata_store.close()

    def run_once(self) -> RunReport:
        report = RunReport(config=self.config)
        pending = PendingBatch()
        faiss_person_map: Dict[int, str] = {}

        try:
            self.metadata_store.start_run(report)
            if not self.config.resume and self.metadata_store.has_existing_state():
                raise RuntimeError(
                    "resume is disabled but existing state was found at {}".format(
                        self.config.db_path
                    )
                )
            self.index_store.ensure_consistent(self.metadata_store)
            faiss_person_map = self.metadata_store.load_faiss_person_map()

            for image_path in iter_image_files(self.config.input_dir, self.config.allowed_extensions):
                self._process_path(image_path, pending, faiss_person_map, report)
                if pending.image_count >= self.config.batch_size:
                    self._flush_pending(pending, report)
            self._flush_pending(pending, report)
        except Exception as exc:
            report.mark_failed()
            report.add_error(
                str(self.config.input_dir),
                "fatal error: {}".format(exc),
                stage="run",
                error_code=type(exc).__name__,
                retryable=False,
            )
        else:
            report.mark_completed()
        finally:
            report.applied_providers = list(self._applied_providers())
            report.set_cumulative(self.metadata_store.cumulative_metrics())
            report.write_artifacts()
            self.metadata_store.complete_run(report)
            self.close()
        return report

    def _process_path(
        self,
        image_path: Path,
        pending: PendingBatch,
        faiss_person_map: Dict[int, str],
        report: RunReport,
    ) -> None:
        report.scanned_images += 1
        resolved_path = str(image_path.resolve())

        try:
            stat_info = image_path.stat()
        except Exception as exc:
            self._record_failed_file(
                report=report,
                image_path=image_path,
                stage="filesystem",
                error_code=type(exc).__name__,
                message="file stat failed: {}".format(exc),
                retryable=True,
            )
            return

        if self.metadata_store.is_known_non_retryable_failure(
            resolved_path,
            stat_info.st_size,
            stat_info.st_mtime_ns,
        ):
            report.skipped_failed_images += 1
            return

        try:
            image_hash = self.hash_file(image_path)
        except Exception as exc:
            self._record_failed_file(
                report=report,
                image_path=image_path,
                stage="hashing",
                error_code=type(exc).__name__,
                message="hashing failed: {}".format(exc),
                retryable=True,
                file_size=stat_info.st_size,
                file_mtime_ns=stat_info.st_mtime_ns,
            )
            return

        if self.metadata_store.image_exists(image_hash):
            report.duplicate_images += 1
            self.metadata_store.resolve_failed_image(resolved_path)
            return
        if image_hash in pending.image_hashes:
            report.duplicate_images += 1
            return

        if self.config.preflight_validate and stat_info.st_size == 0:
            self._record_failed_file(
                report=report,
                image_path=image_path,
                stage="preflight",
                error_code="empty_file",
                message="preflight validation failed: empty file",
                retryable=False,
                image_hash=image_hash,
                file_size=stat_info.st_size,
                file_mtime_ns=stat_info.st_mtime_ns,
            )
            report.bad_images += 1
            return

        read_start = time.perf_counter()
        try:
            raw_image = self.image_loader(image_path)
        except Exception as exc:
            report.add_timing("image_read_seconds", time.perf_counter() - read_start)
            report.bad_images += 1
            self._record_failed_file(
                report=report,
                image_path=image_path,
                stage="image_decode",
                error_code="decode_failed",
                message="image decode failed: {}".format(exc),
                retryable=False,
                image_hash=image_hash,
                file_size=stat_info.st_size,
                file_mtime_ns=stat_info.st_mtime_ns,
            )
            return
        report.add_timing("image_read_seconds", time.perf_counter() - read_start)
        self.metadata_store.resolve_failed_image(resolved_path)

        processed_image = resize_image(raw_image, self.config.max_side)
        engine_result = self.engine.detect_and_embed(processed_image.pixels)
        self._stage_image_record(image_hash, image_path, raw_image, pending)
        report.imported_images += 1
        report.add_timing("detection_seconds", engine_result.detection_seconds)
        report.add_timing("embedding_seconds", engine_result.embedding_seconds)

        if not engine_result.faces:
            report.no_face_images += 1
            return

        for face in engine_result.faces:
            cluster_start = time.perf_counter()
            search = self.index_store.search(
                face.embedding,
                pending_embeddings=[record.embedding for record in pending.faces],
            )
            report.add_timing("clustering_seconds", time.perf_counter() - cluster_start)
            matched_person_id = (
                faiss_person_map.get(search.faiss_id) if search.faiss_id is not None else None
            )
            if search.score is not None and search.score > self.config.sim_threshold and matched_person_id:
                person_id = matched_person_id
            else:
                person_id = self.metadata_store.next_person_id()
                report.new_persons += 1
                representative_face_id = str(uuid.uuid4())
                pending.persons.append(
                    PersonRecord(
                        person_id=person_id,
                        created_at=_utc_now(),
                        representative_face=representative_face_id,
                    )
                )
                face_id = representative_face_id
                self._stage_face(
                    face_id=face_id,
                    image_hash=image_hash,
                    person_id=person_id,
                    face=face,
                    pending=pending,
                    faiss_person_map=faiss_person_map,
                    report=report,
                )
                continue

            face_id = str(uuid.uuid4())
            self._stage_face(
                face_id=face_id,
                image_hash=image_hash,
                person_id=person_id,
                face=face,
                pending=pending,
                faiss_person_map=faiss_person_map,
                report=report,
            )

    def _stage_image_record(
        self,
        image_hash: str,
        image_path: Path,
        loaded_image: object,
        pending: PendingBatch,
    ) -> None:
        pending.images.append(
            ImageRecord(
                image_hash=image_hash,
                file_path=str(image_path.resolve()),
                processed_at=_utc_now(),
                width=int(loaded_image.width),
                height=int(loaded_image.height),
            )
        )
        pending.image_hashes.add(image_hash)

    def _stage_face(
        self,
        face_id: str,
        image_hash: str,
        person_id: str,
        face: object,
        pending: PendingBatch,
        faiss_person_map: Dict[int, str],
        report: RunReport,
    ) -> None:
        faiss_id = self.index_store.committed_count + len(pending.faces)
        pending.faces.append(
            FaceRecord(
                face_id=face_id,
                image_hash=image_hash,
                person_id=person_id,
                bbox_json=json.dumps(list(face.bbox)),
                faiss_id=faiss_id,
                embedding=face.embedding,
            )
        )
        pending.faiss_person_map[faiss_id] = person_id
        faiss_person_map[faiss_id] = person_id
        report.detected_faces += 1

    def _flush_pending(self, pending: PendingBatch, report: RunReport) -> None:
        if pending.image_count == 0:
            return

        attempt = 0
        while True:
            persist_start = time.perf_counter()
            try:
                self.metadata_store.persist_batch(pending.images, pending.persons, pending.faces)
            except Exception as exc:
                report.add_timing("persistence_seconds", time.perf_counter() - persist_start)
                if attempt >= self.config.batch_retry_limit:
                    report.add_error(
                        str(self.config.db_path),
                        "database flush failed: {}".format(exc),
                        stage="db_flush",
                        error_code=type(exc).__name__,
                        retryable=False,
                    )
                    raise
                attempt += 1
                report.batch_retries += 1
                if self.config.batch_retry_backoff_seconds > 0.0:
                    time.sleep(self.config.batch_retry_backoff_seconds * attempt)
                continue
            break

        try:
            self.index_store.persist_pending([record.embedding for record in pending.faces])
        except Exception as exc:
            try:
                self.index_store.rebuild_from_store(self.metadata_store)
            except Exception as rebuild_exc:
                report.add_error(
                    str(self.config.index_path),
                    "index rebuild failed: {}".format(rebuild_exc),
                    stage="index_rebuild",
                    error_code=type(rebuild_exc).__name__,
                    retryable=False,
                )
                raise
            report.add_error(
                str(self.config.index_path),
                "index save recovered via rebuild: {}".format(exc),
                stage="index_persist",
                error_code=type(exc).__name__,
                retryable=True,
            )
        finally:
            report.add_timing("persistence_seconds", time.perf_counter() - persist_start)

        pending.clear()

    def _applied_providers(self) -> tuple[str, ...]:
        providers = getattr(self.engine, "applied_providers", None)
        if callable(providers):
            value = providers()
            if value:
                return tuple(str(item) for item in value)
        if providers:
            return tuple(str(item) for item in providers)
        return tuple(str(item) for item in self.config.providers)

    def _record_failed_file(
        self,
        report: RunReport,
        image_path: Path,
        stage: str,
        error_code: str,
        message: str,
        retryable: bool,
        image_hash: Optional[str] = None,
        file_size: Optional[int] = None,
        file_mtime_ns: Optional[int] = None,
    ) -> None:
        report.failed_files += 1
        report.add_error(
            str(image_path),
            message,
            stage=stage,
            error_code=error_code,
            retryable=retryable,
        )
        self.metadata_store.record_failed_image(
            FailedImageRecord(
                run_id=report.run_id,
                file_path=str(image_path.resolve()),
                image_hash=image_hash,
                stage=stage,
                error_code=error_code,
                error_message=message,
                failed_at=_utc_now(),
                retryable=retryable,
                file_size=file_size,
                file_mtime_ns=file_mtime_ns,
            )
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
