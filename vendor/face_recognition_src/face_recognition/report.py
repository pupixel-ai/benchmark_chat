from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .config import PipelineConfig


@dataclass
class RunReport:
    config: PipelineConfig
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    scanned_images: int = 0
    imported_images: int = 0
    duplicate_images: int = 0
    failed_files: int = 0
    bad_images: int = 0
    skipped_failed_images: int = 0
    no_face_images: int = 0
    detected_faces: int = 0
    new_persons: int = 0
    batch_retries: int = 0
    timings: Dict[str, float] = field(
        default_factory=lambda: {
            "image_read_seconds": 0.0,
            "detection_seconds": 0.0,
            "embedding_seconds": 0.0,
            "clustering_seconds": 0.0,
            "persistence_seconds": 0.0,
        }
    )
    errors: List[Dict[str, str]] = field(default_factory=list)
    artifact_paths: Dict[str, str] = field(default_factory=dict)
    cumulative: Dict[str, int] = field(default_factory=dict)
    applied_providers: List[str] = field(default_factory=list)
    run_status: str = "running"

    def add_timing(self, key: str, seconds: float) -> None:
        self.timings[key] = self.timings.get(key, 0.0) + max(0.0, float(seconds))

    def add_error(
        self,
        file_path: str,
        reason: str,
        *,
        stage: Optional[str] = None,
        error_code: Optional[str] = None,
        retryable: Optional[bool] = None,
    ) -> None:
        entry = {"file_path": file_path, "reason": reason}
        if stage is not None:
            entry["stage"] = stage
        if error_code is not None:
            entry["error_code"] = error_code
        if retryable is not None:
            entry["retryable"] = str(bool(retryable)).lower()
        self.errors.append(entry)

    @property
    def dataset_name(self) -> str:
        return self.config.input_dir.name

    def set_cumulative(self, metrics: Dict[str, int]) -> None:
        self.cumulative = dict(metrics)

    def mark_completed(self) -> None:
        self.completed_at = datetime.now(timezone.utc)
        self.run_status = "success" if not self.errors else "completed_with_errors"

    def mark_failed(self) -> None:
        self.completed_at = datetime.now(timezone.utc)
        self.run_status = "failed"

    @property
    def total_seconds(self) -> float:
        end_time = self.completed_at or datetime.now(timezone.utc)
        return max(0.0, (end_time - self.started_at).total_seconds())

    @property
    def average_image_seconds(self) -> float:
        if self.scanned_images == 0:
            return 0.0
        return self.total_seconds / float(self.scanned_images)

    def summary_text(self) -> str:
        lines = [
            "Face recognition import summary",
            "run_id: {}".format(self.run_id),
            "dataset_name: {}".format(self.dataset_name),
            "status: {}".format(self.run_status),
            "input_dir: {}".format(self.config.input_dir),
            "db_path: {}".format(self.config.db_path),
            "index_path: {}".format(self.config.index_path),
            "scanned_images: {}".format(self.scanned_images),
            "imported_images: {}".format(self.imported_images),
            "duplicate_images: {}".format(self.duplicate_images),
            "bad_images: {}".format(self.bad_images),
            "failed_files: {}".format(self.failed_files),
            "skipped_failed_images: {}".format(self.skipped_failed_images),
            "no_face_images: {}".format(self.no_face_images),
            "detected_faces: {}".format(self.detected_faces),
            "new_persons: {}".format(self.new_persons),
            "batch_retries: {}".format(self.batch_retries),
            "total_seconds: {:.6f}".format(self.total_seconds),
            "average_image_seconds: {:.6f}".format(self.average_image_seconds),
        ]
        if self.applied_providers:
            lines.append("applied_providers: {}".format(", ".join(self.applied_providers)))
        for key in sorted(self.timings):
            lines.append("{}: {:.6f}".format(key, self.timings[key]))
        if self.cumulative:
            lines.append("cumulative:")
            for key, value in sorted(self.cumulative.items()):
                lines.append("- {}: {}".format(key, value))
        if self.artifact_paths:
            lines.append("artifacts:")
            for name, path in sorted(self.artifact_paths.items()):
                lines.append("- {}: {}".format(name, path))

        if self.errors:
            lines.append("errors:")
            for item in self.errors:
                lines.append("- {} :: {}".format(item["file_path"], item["reason"]))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.run_status,
            "run_id": self.run_id,
            "dataset_name": self.dataset_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "input_dir": str(self.config.input_dir),
            "db_path": str(self.config.db_path),
            "index_path": str(self.config.index_path),
            "log_dir": str(self.config.log_dir),
            "parameters": {
                "batch_size": self.config.batch_size,
                "batch_retry_limit": self.config.batch_retry_limit,
                "batch_retry_backoff_seconds": self.config.batch_retry_backoff_seconds,
                "max_side": self.config.max_side,
                "det_threshold": self.config.det_threshold,
                "sim_threshold": self.config.sim_threshold,
                "preflight_validate": self.config.preflight_validate,
                "resume": self.config.resume,
                "providers": list(self.config.providers),
            },
            "scanned_images": self.scanned_images,
            "imported_images": self.imported_images,
            "duplicate_images": self.duplicate_images,
            "failed_files": self.failed_files,
            "bad_images": self.bad_images,
            "skipped_failed_images": self.skipped_failed_images,
            "no_face_images": self.no_face_images,
            "detected_faces": self.detected_faces,
            "new_persons": self.new_persons,
            "batch_retries": self.batch_retries,
            "total_seconds": self.total_seconds,
            "average_image_seconds": self.average_image_seconds,
            "applied_providers": list(self.applied_providers),
            "timings": dict(self.timings),
            "cumulative": dict(self.cumulative),
            "artifacts": dict(self.artifact_paths),
            "errors": list(self.errors),
        }

    def write_artifacts(self) -> Dict[str, Path]:
        if self.completed_at is None:
            self.mark_completed()

        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = self.completed_at.astimezone().strftime("%Y%m%d-%H%M%S-%f")
        text_path = self.config.log_dir / "import-{}.log".format(stamp)
        json_path = self.config.log_dir / "import-{}.json".format(stamp)
        self.artifact_paths = {"text": str(text_path), "json": str(json_path)}

        text_path.write_text(self.summary_text() + "\n", encoding="utf-8")
        json_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {"text": text_path, "json": json_path}
