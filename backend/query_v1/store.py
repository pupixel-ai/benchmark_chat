"""SQL-backed canonical store for query v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Type

from sqlalchemy import delete, desc, select

from backend.db import Base, SessionLocal, engine
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


@dataclass(slots=True)
class MaterializedBundle:
    materialization: Dict[str, Any]
    photos: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)
    event_photos: List[Dict[str, Any]] = field(default_factory=list)
    event_people: List[Dict[str, Any]] = field(default_factory=list)
    event_places: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    relationship_support: List[Dict[str, Any]] = field(default_factory=list)
    groups: List[Dict[str, Any]] = field(default_factory=list)
    group_members: List[Dict[str, Any]] = field(default_factory=list)
    profile_facts: List[Dict[str, Any]] = field(default_factory=list)


def _model_to_dict(record: Any) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for column in record.__table__.columns:
        value = getattr(record, column.name)
        if isinstance(value, datetime):
            data[column.name] = value.isoformat()
        else:
            data[column.name] = value
    return data


class QueryStore:
    """Persistence layer for canonical query-store records."""

    _SCOPE_MODELS: Sequence[Type[Any]] = (
        MemoryEventPhotoRecord,
        MemoryEventPersonRecord,
        MemoryEventPlaceRecord,
        MemoryEvidenceRecord,
        MemoryRelationshipSupportRecord,
        MemoryGroupMemberRecord,
        MemoryProfileFactRecord,
        MemoryGroupRecord,
        MemoryRelationshipRecord,
        MemoryPhotoRecord,
        MemoryEventRecord,
        MemoryMaterializationRecord,
    )

    def __init__(self) -> None:
        Base.metadata.create_all(bind=engine)

    def latest_materialization(self, *, user_id: str, source_task_id: str, schema_version: str) -> Optional[Dict[str, Any]]:
        with SessionLocal() as session:
            stmt = (
                select(MemoryMaterializationRecord)
                .where(
                    MemoryMaterializationRecord.user_id == user_id,
                    MemoryMaterializationRecord.source_task_id == source_task_id,
                    MemoryMaterializationRecord.schema_version == schema_version,
                )
                .order_by(desc(MemoryMaterializationRecord.created_at))
                .limit(1)
            )
            record = session.execute(stmt).scalar_one_or_none()
            return _model_to_dict(record) if record is not None else None

    def replace_scope(self, bundle: MaterializedBundle) -> Dict[str, Any]:
        materialization = dict(bundle.materialization)
        user_id = str(materialization["user_id"])
        source_task_id = str(materialization["source_task_id"])
        now = datetime.now()
        with SessionLocal() as session:
            for model in self._SCOPE_MODELS:
                session.execute(
                    delete(model).where(
                        model.user_id == user_id,
                        model.source_task_id == source_task_id,
                    )
                )
            session.add(MemoryMaterializationRecord(**self._normalize_row(materialization, now)))
            self._bulk_add(session, MemoryPhotoRecord, bundle.photos, now)
            self._bulk_add(session, MemoryEventRecord, bundle.events, now)
            self._bulk_add(session, MemoryEventPhotoRecord, bundle.event_photos, now)
            self._bulk_add(session, MemoryEventPersonRecord, bundle.event_people, now)
            self._bulk_add(session, MemoryEventPlaceRecord, bundle.event_places, now)
            self._bulk_add(session, MemoryEvidenceRecord, bundle.evidence, now)
            self._bulk_add(session, MemoryRelationshipRecord, bundle.relationships, now)
            self._bulk_add(session, MemoryRelationshipSupportRecord, bundle.relationship_support, now)
            self._bulk_add(session, MemoryGroupRecord, bundle.groups, now)
            self._bulk_add(session, MemoryGroupMemberRecord, bundle.group_members, now)
            self._bulk_add(session, MemoryProfileFactRecord, bundle.profile_facts, now)
            session.commit()
        return materialization

    def update_materialization_status(
        self,
        *,
        materialization_id: str,
        status: str,
        milvus_status: Dict[str, Any] | None,
        neo4j_status: Dict[str, Any] | None,
        error_summary: str | None,
    ) -> Optional[Dict[str, Any]]:
        with SessionLocal() as session:
            record = session.get(MemoryMaterializationRecord, materialization_id)
            if record is None:
                return None
            record.status = status
            record.milvus_status = dict(milvus_status) if isinstance(milvus_status, dict) else milvus_status
            record.neo4j_status = dict(neo4j_status) if isinstance(neo4j_status, dict) else neo4j_status
            record.error_summary = error_summary
            record.updated_at = datetime.now()
            session.add(record)
            session.commit()
            session.refresh(record)
            return _model_to_dict(record)

    def fetch_scope(self, *, user_id: str, source_task_id: str) -> Dict[str, Any]:
        with SessionLocal() as session:
            return {
                "materialization": self.latest_materialization(
                    user_id=user_id,
                    source_task_id=source_task_id,
                    schema_version="query_v1",
                ),
                "photos": self._list_model(session, MemoryPhotoRecord, user_id, source_task_id),
                "events": self._list_model(session, MemoryEventRecord, user_id, source_task_id),
                "event_photos": self._list_model(session, MemoryEventPhotoRecord, user_id, source_task_id),
                "event_people": self._list_model(session, MemoryEventPersonRecord, user_id, source_task_id),
                "event_places": self._list_model(session, MemoryEventPlaceRecord, user_id, source_task_id),
                "evidence": self._list_model(session, MemoryEvidenceRecord, user_id, source_task_id),
                "relationships": self._list_model(session, MemoryRelationshipRecord, user_id, source_task_id),
                "relationship_support": self._list_model(session, MemoryRelationshipSupportRecord, user_id, source_task_id),
                "groups": self._list_model(session, MemoryGroupRecord, user_id, source_task_id),
                "group_members": self._list_model(session, MemoryGroupMemberRecord, user_id, source_task_id),
                "profile_facts": self._list_model(session, MemoryProfileFactRecord, user_id, source_task_id),
            }

    def _bulk_add(self, session: Any, model: Type[Any], rows: Sequence[Dict[str, Any]], now: datetime) -> None:
        for row in rows:
            session.add(model(**self._normalize_row(row, now)))

    def _normalize_row(self, row: Dict[str, Any], now: datetime) -> Dict[str, Any]:
        payload = dict(row)
        payload.setdefault("created_at", now)
        payload.setdefault("updated_at", now)
        return payload

    def _list_model(self, session: Any, model: Type[Any], user_id: str, source_task_id: str) -> List[Dict[str, Any]]:
        stmt = select(model).where(model.user_id == user_id, model.source_task_id == source_task_id)
        return [_model_to_dict(item) for item in session.execute(stmt).scalars().all()]
