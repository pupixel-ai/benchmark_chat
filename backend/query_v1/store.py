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
    MemoryFaceRecord,
    MemoryGroupMemberRecord,
    MemoryGroupRecord,
    MemoryMaterializationRecord,
    MemoryPersonRecord,
    MemoryPhotoRecord,
    MemoryProfileFactRecord,
    MemoryRelationshipRecord,
    MemoryRelationshipSupportRecord,
)


@dataclass(slots=True)
class MaterializedBundle:
    materialization: Dict[str, Any]
    photos: List[Dict[str, Any]] = field(default_factory=list)
    persons: List[Dict[str, Any]] = field(default_factory=list)
    faces: List[Dict[str, Any]] = field(default_factory=list)
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


def _publicize_value(value: Any, *, source_task_id: str) -> Any:
    if isinstance(value, str):
        prefix = f"{source_task_id}:"
        if value.startswith(prefix):
            return value[len(prefix):]
        return value
    if isinstance(value, list):
        return [_publicize_value(item, source_task_id=source_task_id) for item in value]
    if isinstance(value, dict):
        return {
            key: (
                item
                if key == "source_task_id"
                else _publicize_value(item, source_task_id=source_task_id)
            )
            for key, item in value.items()
        }
    return value


def _model_to_dict(record: Any, *, source_task_id: str | None = None) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    for column in record.__table__.columns:
        value = getattr(record, column.name)
        if isinstance(value, datetime):
            data[column.name] = value.isoformat()
        else:
            data[column.name] = value
    if source_task_id:
        data = _publicize_value(data, source_task_id=source_task_id)
    return data


class QueryStore:
    """Persistence layer for canonical query-store records."""

    _schema_ready: bool = False

    _MODEL_BY_SCOPE_KEY: Dict[str, Type[Any]] = {
        "faces": MemoryFaceRecord,
        "persons": MemoryPersonRecord,
        "event_photos": MemoryEventPhotoRecord,
        "event_people": MemoryEventPersonRecord,
        "event_places": MemoryEventPlaceRecord,
        "evidence": MemoryEvidenceRecord,
        "relationship_support": MemoryRelationshipSupportRecord,
        "group_members": MemoryGroupMemberRecord,
        "profile_facts": MemoryProfileFactRecord,
        "groups": MemoryGroupRecord,
        "relationships": MemoryRelationshipRecord,
        "photos": MemoryPhotoRecord,
        "events": MemoryEventRecord,
    }

    _SCOPE_MODELS: Sequence[Type[Any]] = (
        MemoryFaceRecord,
        MemoryPersonRecord,
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
        pass

    def ensure_schema(self) -> None:
        if self.__class__._schema_ready:
            return
        Base.metadata.create_all(bind=engine)
        self.__class__._schema_ready = True

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
        self.ensure_schema()
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
            self._bulk_add(session, MemoryPersonRecord, bundle.persons, now)
            self._bulk_add(session, MemoryFaceRecord, bundle.faces, now)
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
        return self._fetch_scope_subset(
            user_id=user_id,
            source_task_id=source_task_id,
            keys=tuple(self._MODEL_BY_SCOPE_KEY.keys()),
        )

    def fetch_faces_scope(self, *, user_id: str, source_task_id: str) -> Dict[str, Any]:
        return self._fetch_scope_subset(
            user_id=user_id,
            source_task_id=source_task_id,
            keys=("photos", "persons", "faces"),
        )

    def fetch_events_scope(self, *, user_id: str, source_task_id: str) -> Dict[str, Any]:
        return self._fetch_scope_subset(
            user_id=user_id,
            source_task_id=source_task_id,
            keys=("photos", "events", "event_photos", "event_people", "event_places"),
        )

    def fetch_vlm_scope(self, *, user_id: str, source_task_id: str) -> Dict[str, Any]:
        return self._fetch_scope_subset(
            user_id=user_id,
            source_task_id=source_task_id,
            keys=("photos", "evidence", "event_people"),
        )

    def fetch_profiles_scope(self, *, user_id: str, source_task_id: str) -> Dict[str, Any]:
        return self._fetch_scope_subset(
            user_id=user_id,
            source_task_id=source_task_id,
            keys=("profile_facts", "relationships"),
        )

    def fetch_relationships_scope(self, *, user_id: str, source_task_id: str) -> Dict[str, Any]:
        return self._fetch_scope_subset(
            user_id=user_id,
            source_task_id=source_task_id,
            keys=("photos", "events", "relationships", "relationship_support"),
        )

    def fetch_bundle_scope(self, *, user_id: str, source_task_id: str) -> Dict[str, Any]:
        with SessionLocal() as session:
            return self._fetch_scope_subset(
                user_id=user_id,
                source_task_id=source_task_id,
                keys=tuple(self._MODEL_BY_SCOPE_KEY.keys()),
                session=session,
            )

    def _bulk_add(self, session: Any, model: Type[Any], rows: Sequence[Dict[str, Any]], now: datetime) -> None:
        for row in rows:
            session.add(model(**self._normalize_row(row, now)))

    def _fetch_scope_subset(
        self,
        *,
        user_id: str,
        source_task_id: str,
        keys: Sequence[str],
        session: Any | None = None,
    ) -> Dict[str, Any]:
        if session is not None:
            return self._fetch_scope_subset_with_session(
                session=session,
                user_id=user_id,
                source_task_id=source_task_id,
                keys=keys,
            )
        with SessionLocal() as owned_session:
            return self._fetch_scope_subset_with_session(
                session=owned_session,
                user_id=user_id,
                source_task_id=source_task_id,
                keys=keys,
            )

    def _fetch_scope_subset_with_session(
        self,
        *,
        session: Any,
        user_id: str,
        source_task_id: str,
        keys: Sequence[str],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "materialization": self.latest_materialization(
                user_id=user_id,
                source_task_id=source_task_id,
                schema_version="query_v1",
            )
        }
        for key in keys:
            model = self._MODEL_BY_SCOPE_KEY[key]
            payload[key] = self._list_model(session, model, user_id, source_task_id)
        return payload

    def _normalize_row(self, row: Dict[str, Any], now: datetime) -> Dict[str, Any]:
        payload = dict(row)
        payload.setdefault("created_at", now)
        payload.setdefault("updated_at", now)
        return payload

    def _list_model(self, session: Any, model: Type[Any], user_id: str, source_task_id: str) -> List[Dict[str, Any]]:
        stmt = select(model).where(model.user_id == user_id, model.source_task_id == source_task_id)
        return [_model_to_dict(item, source_task_id=source_task_id) for item in session.execute(stmt).scalars().all()]
