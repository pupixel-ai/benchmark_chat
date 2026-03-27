"""
Database models.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from backend.db import Base


class UserRecord(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class SessionRecord(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)


class TaskRecord(Base):
    __tablename__ = "tasks"

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.user_id"), nullable=True, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    dataset_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    version: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    pipeline_version: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    pipeline_channel: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    face_version: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    vlm_version: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    lp1_version: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    lp2_version: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    lp3_version: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    judge_version: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    upload_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    task_dir: Mapped[str] = mapped_column(String(512), nullable=False)
    progress: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    uploads: Mapped[list | None] = mapped_column(JSON, nullable=True)
    options: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_summary: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    asset_manifest: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    worker_instance_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    worker_private_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    worker_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    delete_state: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_worker_sync_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ArtifactRecord(Base):
    __tablename__ = "artifacts"
    __table_args__ = (UniqueConstraint("task_id", "relative_path", name="uq_artifact_task_path"),)

    artifact_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    relative_path: Mapped[str] = mapped_column(String(512), nullable=False)
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    content_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    storage_backend: Mapped[str] = mapped_column(String(32), nullable=False, default="local", index=True)
    object_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    asset_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class FaceReviewRecord(Base):
    __tablename__ = "face_reviews"
    __table_args__ = (UniqueConstraint("user_id", "task_id", "face_id", name="uq_face_review_user_task_face"),)

    review_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    face_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    image_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    person_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source_hash: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    is_inaccurate: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    comment_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class FaceRecognitionImagePolicyRecord(Base):
    __tablename__ = "face_recognition_image_policies"
    __table_args__ = (UniqueConstraint("user_id", "source_hash", name="uq_face_policy_user_hash"),)

    policy_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    source_hash: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    is_abandoned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    last_task_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    last_image_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
