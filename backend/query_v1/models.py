"""Canonical query-store SQLAlchemy models for query v1."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from backend.db import Base


class MemoryMaterializationRecord(Base):
    __tablename__ = "memory_materializations"

    materialization_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_task_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    pipeline_family: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source_updated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    milvus_status: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    neo4j_status: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class MemoryEventRecord(Base):
    __tablename__ = "memory_events"

    event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_task_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    pipeline_family: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    materialization_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    title: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    start_ts: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    end_ts: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    cover_photo_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    photo_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    event_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class MemoryEventPhotoRecord(Base):
    __tablename__ = "memory_event_photos"
    __table_args__ = (UniqueConstraint("event_id", "photo_id", name="uq_memory_event_photo"),)

    row_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_task_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    pipeline_family: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    materialization_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    event_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    photo_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    is_cover: Mapped[bool] = mapped_column(default=False, nullable=False, index=True)
    support_strength: Mapped[float | None] = mapped_column(Float, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class MemoryEventPersonRecord(Base):
    __tablename__ = "memory_event_people"
    __table_args__ = (UniqueConstraint("event_id", "person_id", "role", name="uq_memory_event_person_role"),)

    row_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_task_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    pipeline_family: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    materialization_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    event_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    person_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class MemoryEventPlaceRecord(Base):
    __tablename__ = "memory_event_places"
    __table_args__ = (UniqueConstraint("event_id", "place_ref", name="uq_memory_event_place"),)

    row_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_task_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    pipeline_family: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    materialization_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    event_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    place_ref: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    normalized_place: Mapped[str | None] = mapped_column(String(512), nullable=True, index=True)
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class MemoryEvidenceRecord(Base):
    __tablename__ = "memory_evidence"

    evidence_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_task_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    pipeline_family: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    materialization_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    event_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    source_stage: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    evidence_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    photo_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    normalized_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    numeric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    numeric_unit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    evidence_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class MemoryRelationshipRecord(Base):
    __tablename__ = "memory_relationships"

    relationship_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_task_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    pipeline_family: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    materialization_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    person_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    relationship_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    intimacy_score: Mapped[float | None] = mapped_column(Float, nullable=True, index=True)
    photo_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    shared_event_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    monthly_frequency: Mapped[float | None] = mapped_column(Float, nullable=True)
    recent_gap_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    relationship_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class MemoryRelationshipSupportRecord(Base):
    __tablename__ = "memory_relationship_support"

    row_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_task_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    pipeline_family: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    materialization_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    relationship_id: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    event_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    photo_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    support_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    support_strength: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class MemoryGroupRecord(Base):
    __tablename__ = "memory_groups"

    group_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_task_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    pipeline_family: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    materialization_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    group_type: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    group_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class MemoryGroupMemberRecord(Base):
    __tablename__ = "memory_group_members"
    __table_args__ = (UniqueConstraint("group_id", "person_id", name="uq_memory_group_member"),)

    row_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_task_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    pipeline_family: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    materialization_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    group_id: Mapped[str] = mapped_column(String(160), nullable=False, index=True)
    person_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class MemoryProfileFactRecord(Base):
    __tablename__ = "memory_profile_facts"

    fact_id: Mapped[str] = mapped_column(String(160), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_task_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    pipeline_family: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    materialization_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    field_key: Mapped[str] = mapped_column(String(256), nullable=False, index=True)
    value_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_level: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    is_queryable_hint: Mapped[bool] = mapped_column(default=False, nullable=False, index=True)
    fact_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class MemoryPhotoRecord(Base):
    __tablename__ = "memory_photos"

    photo_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_task_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    pipeline_family: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    materialization_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    object_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    asset_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    captured_at: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    content_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    photo_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
