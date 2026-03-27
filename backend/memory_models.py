"""
Normalized memory-domain tables for the v0327-db retrieval layer.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import UserDefinedType

from backend.db import Base


class Vector512(UserDefinedType):
    cache_ok = True

    def get_col_spec(self, **kw) -> str:
        return "vector(512)"


class DatasetRecord(Base):
    __tablename__ = "datasets"
    __table_args__ = (UniqueConstraint("user_id", "dataset_fingerprint", name="uq_dataset_user_fingerprint"),)

    dataset_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    dataset_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    photo_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    source_hashes_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    first_task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.task_id"), nullable=True, index=True)
    latest_task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.task_id"), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class TaskStageRecord(Base):
    __tablename__ = "task_stage_records"
    __table_args__ = (UniqueConstraint("task_id", "stage_name", name="uq_task_stage_name"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    stage_name: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    stage_version: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    stage_channel: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    summary_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    raw_payload_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    normalized_payload_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    artifact_manifest_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class TaskPhotoItemRecord(Base):
    __tablename__ = "task_photo_items"
    __table_args__ = (UniqueConstraint("task_id", "photo_id", name="uq_task_photo_item"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    photo_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    source_photo_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    upload_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    batch_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    upload_status: Mapped[str] = mapped_column(String(32), nullable=False, default="uploaded")
    source_hash: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class BinaryAssetRecord(Base):
    __tablename__ = "binary_assets"
    __table_args__ = (UniqueConstraint("task_id", "relative_path", name="uq_binary_asset_task_path"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    asset_kind: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    relative_path: Mapped[str] = mapped_column(String(512), nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    storage_backend: Mapped[str | None] = mapped_column(String(32), nullable=True)
    object_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    asset_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class PhotoRecord(Base):
    __tablename__ = "photos"

    photo_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    source_photo_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    original_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    stored_filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    source_hash: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    mime_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    taken_at: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    location_json: Mapped[dict | str | None] = mapped_column(JSON, nullable=True)
    exif_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    raw_relative_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    display_relative_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    boxed_relative_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    compressed_relative_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    raw_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    display_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    boxed_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    compressed_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class PhotoExifRecord(Base):
    __tablename__ = "photo_exif"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    photo_id: Mapped[str] = mapped_column(ForeignKey("photos.photo_id"), nullable=False, unique=True, index=True)
    raw_exif_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    normalized_exif_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    captured_at: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    gps_lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    gps_lng: Mapped[float | None] = mapped_column(Float, nullable=True)
    camera_make: Mapped[str | None] = mapped_column(String(128), nullable=True)
    camera_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class PhotoAssetRecord(Base):
    __tablename__ = "photo_assets"
    __table_args__ = (UniqueConstraint("photo_id", "variant_type", name="uq_photo_asset_variant"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    photo_id: Mapped[str] = mapped_column(ForeignKey("photos.photo_id"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    variant_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    relative_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    asset_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class PersonRecord(Base):
    __tablename__ = "persons"
    __table_args__ = (UniqueConstraint("task_id", "person_id", name="uq_person_task_person"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    person_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    canonical_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_primary_person: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    photo_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    face_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_quality: Mapped[float | None] = mapped_column(Float, nullable=True)
    high_quality_face_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avatar_relative_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    avatar_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class PersonRevisionRecord(Base):
    __tablename__ = "person_revisions"
    __table_args__ = (UniqueConstraint("task_id", "person_id", "revision_no", name="uq_person_revision"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    person_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    revision_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    snapshot_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class FaceObservationRecord(Base):
    __tablename__ = "face_observations"

    face_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    person_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    photo_id: Mapped[str] = mapped_column(ForeignKey("photos.photo_id"), nullable=False, index=True)
    source_photo_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source_hash: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    faiss_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    bbox_json: Mapped[list | dict | None] = mapped_column(JSON, nullable=True)
    bbox_xywh_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    kps_json: Mapped[list | dict | None] = mapped_column(JSON, nullable=True)
    match_decision: Mapped[str | None] = mapped_column(String(64), nullable=True)
    match_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    pose_yaw: Mapped[float | None] = mapped_column(Float, nullable=True)
    pose_pitch: Mapped[float | None] = mapped_column(Float, nullable=True)
    pose_roll: Mapped[float | None] = mapped_column(Float, nullable=True)
    pose_bucket: Mapped[str | None] = mapped_column(String(64), nullable=True)
    eye_visibility_ratio: Mapped[float | None] = mapped_column(Float, nullable=True)
    crop_relative_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    crop_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    boxed_relative_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    boxed_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    face_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class FaceEmbeddingRecord(Base):
    __tablename__ = "face_embeddings"
    __table_args__ = (UniqueConstraint("face_id", name="uq_face_embedding_face"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    face_id: Mapped[str] = mapped_column(ForeignKey("face_observations.face_id"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    faiss_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    embedding: Mapped[object | None] = mapped_column(Vector512(), nullable=True)
    embedding_dim: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    source_backend: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class PersonFaceLinkRecord(Base):
    __tablename__ = "person_face_links"
    __table_args__ = (UniqueConstraint("task_id", "person_id", "face_id", name="uq_person_face_link"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    person_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    face_id: Mapped[str] = mapped_column(ForeignKey("face_observations.face_id"), nullable=False, index=True)
    photo_id: Mapped[str] = mapped_column(ForeignKey("photos.photo_id"), nullable=False, index=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class VLMObservationRevisionRecord(Base):
    __tablename__ = "vlm_observation_revisions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    photo_id: Mapped[str] = mapped_column(ForeignKey("photos.photo_id"), nullable=False, index=True)
    source_photo_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    people_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    relations_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    scene_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    event_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    details_json: Mapped[list | dict | None] = mapped_column(JSON, nullable=True)
    clues_json: Mapped[list | dict | None] = mapped_column(JSON, nullable=True)
    raw_payload_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class VLMObservationPersonRecord(Base):
    __tablename__ = "vlm_observation_people"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    observation_id: Mapped[str] = mapped_column(ForeignKey("vlm_observation_revisions.id"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    photo_id: Mapped[str] = mapped_column(ForeignKey("photos.photo_id"), nullable=False, index=True)
    person_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)
    person_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    raw_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class VLMObservationRelationRecord(Base):
    __tablename__ = "vlm_observation_relations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    observation_id: Mapped[str] = mapped_column(ForeignKey("vlm_observation_revisions.id"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    photo_id: Mapped[str] = mapped_column(ForeignKey("photos.photo_id"), nullable=False, index=True)
    raw_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class VLMObservationClueRecord(Base):
    __tablename__ = "vlm_observation_clues"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    observation_id: Mapped[str] = mapped_column(ForeignKey("vlm_observation_revisions.id"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    photo_id: Mapped[str] = mapped_column(ForeignKey("photos.photo_id"), nullable=False, index=True)
    clue_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    clue_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class EventRootRecord(Base):
    __tablename__ = "event_roots"
    __table_args__ = (UniqueConstraint("task_id", "event_id", name="uq_event_root_task_event"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    current_revision_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class EventRevisionRecord(Base):
    __tablename__ = "event_revisions"
    __table_args__ = (UniqueConstraint("task_id", "event_id", "revision_no", name="uq_event_revision_task_event"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    revision_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    date: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    time_range: Mapped[dict | list | str | None] = mapped_column(JSON, nullable=True)
    duration: Mapped[str | int | None] = mapped_column(JSON, nullable=True)
    type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    location: Mapped[str | dict | None] = mapped_column(JSON, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    photo_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    narrative: Mapped[str | None] = mapped_column(Text, nullable=True)
    narrative_synthesis: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    social_dynamics_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    persona_evidence_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    raw_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class EventParticipantRecord(Base):
    __tablename__ = "event_participants"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_revision_id: Mapped[str] = mapped_column(ForeignKey("event_revisions.id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    person_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    participant_role: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class EventPhotoLinkRecord(Base):
    __tablename__ = "event_photo_links"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_revision_id: Mapped[str] = mapped_column(ForeignKey("event_revisions.id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    photo_id: Mapped[str] = mapped_column(ForeignKey("photos.photo_id"), nullable=False, index=True)
    source_photo_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    evidence_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class EventDetailUnitRecord(Base):
    __tablename__ = "event_detail_units"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    event_revision_id: Mapped[str] = mapped_column(ForeignKey("event_revisions.id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    detail_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    detail_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalized_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_refs_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    sort_key: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class RelationshipRootRecord(Base):
    __tablename__ = "relationship_roots"
    __table_args__ = (UniqueConstraint("task_id", "relationship_id", name="uq_relationship_root_task_rel"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    relationship_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    person_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    current_revision_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class RelationshipDossierRevisionRecord(Base):
    __tablename__ = "relationship_dossier_revisions"
    __table_args__ = (UniqueConstraint("task_id", "relationship_id", "revision_no", name="uq_relationship_dossier_revision"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    relationship_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    revision_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    person_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    dossier_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class RelationshipRevisionRecord(Base):
    __tablename__ = "relationship_revisions"
    __table_args__ = (UniqueConstraint("task_id", "relationship_id", "revision_no", name="uq_relationship_revision"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    relationship_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    revision_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    person_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    relationship_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    intimacy_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    raw_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class RelationshipSharedEventRecord(Base):
    __tablename__ = "relationship_shared_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    relationship_revision_id: Mapped[str] = mapped_column(ForeignKey("relationship_revisions.id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    event_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    date_snapshot: Mapped[str | None] = mapped_column(String(64), nullable=True)
    narrative_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class GroupRootRecord(Base):
    __tablename__ = "group_roots"
    __table_args__ = (UniqueConstraint("task_id", "group_id", name="uq_group_root_task_group"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    group_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    current_revision_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class GroupRevisionRecord(Base):
    __tablename__ = "group_revisions"
    __table_args__ = (UniqueConstraint("task_id", "group_id", "revision_no", name="uq_group_revision"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    group_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    revision_no: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    group_type_candidate: Mapped[str | None] = mapped_column(String(128), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    strong_evidence_refs_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    raw_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class GroupMemberRecord(Base):
    __tablename__ = "group_members"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    group_revision_id: Mapped[str] = mapped_column(ForeignKey("group_revisions.id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    person_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ProfileContextRevisionRecord(Base):
    __tablename__ = "profile_context_revisions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, unique=True, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    primary_person_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    payload_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ProfileRevisionRecord(Base):
    __tablename__ = "profile_revisions"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    profile_revision_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    primary_person_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    structured_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    report_markdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    consistency_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    debug_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    internal_artifacts_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    field_decisions_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ProfileFieldValueRecord(Base):
    __tablename__ = "profile_field_values"
    __table_args__ = (UniqueConstraint("task_id", "field_key", name="uq_profile_field_task"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    profile_revision_id: Mapped[str] = mapped_column(ForeignKey("profile_revisions.id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    field_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    domain_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    batch_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    value_json: Mapped[dict | list | str | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    traceable_evidence_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    null_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ProfileFactDecisionRecord(Base):
    __tablename__ = "profile_fact_decisions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    profile_revision_id: Mapped[str] = mapped_column(ForeignKey("profile_revisions.id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    field_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    payload_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class JudgeDecisionRevisionRecord(Base):
    __tablename__ = "judge_decision_revisions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    judge_type: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    target_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    decision_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    reasoning_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ConsistencyCheckRevisionRecord(Base):
    __tablename__ = "consistency_check_revisions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    profile_revision_id: Mapped[str] = mapped_column(ForeignKey("profile_revisions.id"), nullable=False, index=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.task_id"), nullable=False, index=True)
    check_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    details_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class GroundTruthRevisionRecord(Base):
    __tablename__ = "ground_truth_revisions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    gt_revision_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class GroundTruthAssertionRecord(Base):
    __tablename__ = "ground_truth_assertions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    gt_revision_id: Mapped[str] = mapped_column(ForeignKey("ground_truth_revisions.gt_revision_id"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    subject_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    subject_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    field_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    operation: Mapped[str | None] = mapped_column(String(64), nullable=True)
    value_json: Mapped[dict | list | str | None] = mapped_column(JSON, nullable=True)
    evidence_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AgentRunRecord(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.task_id"), nullable=True, index=True)
    agent_kind: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    agent_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    prompt_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AgentMessageRecord(Base):
    __tablename__ = "agent_messages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"), nullable=False, index=True)
    role: Mapped[str | None] = mapped_column(String(32), nullable=True)
    seq_no: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    content_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AgentToolCallRecord(Base):
    __tablename__ = "agent_tool_calls"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"), nullable=False, index=True)
    tool_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    seq_no: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    args_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AgentTraceEventRecord(Base):
    __tablename__ = "agent_trace_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"), nullable=False, index=True)
    event_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    seq_no: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    visible_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class AgentOutputRecord(Base):
    __tablename__ = "agent_outputs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    agent_run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id"), nullable=False, index=True)
    output_kind: Mapped[str | None] = mapped_column(String(128), nullable=True)
    target_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    normalized_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_json: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ObjectRegistryRecord(Base):
    __tablename__ = "object_registry"
    __table_args__ = (UniqueConstraint("task_id", "object_type", "semantic_id", name="uq_object_registry_task_semantic"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.task_id"), nullable=True, index=True)
    dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True, index=True)
    object_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    semantic_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    table_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    parent_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    root_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class ObjectLinkRecord(Base):
    __tablename__ = "object_links"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), nullable=False, index=True)
    from_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    relation_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    to_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class UserHeadRecord(Base):
    __tablename__ = "user_heads"

    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), primary_key=True)
    active_dataset_id: Mapped[int | None] = mapped_column(ForeignKey("datasets.dataset_id"), nullable=True)
    active_task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.task_id"), nullable=True)
    active_profile_revision_id: Mapped[str | None] = mapped_column(ForeignKey("profile_revisions.id"), nullable=True)
    active_gt_revision_id: Mapped[str | None] = mapped_column(ForeignKey("ground_truth_revisions.gt_revision_id"), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
