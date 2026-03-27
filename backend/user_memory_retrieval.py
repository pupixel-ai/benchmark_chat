"""
DB-backed retrieval services for user-scoped memory APIs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy import desc, func, select

from backend.artifact_store import ArtifactCatalogStore
from backend.db import SessionLocal
from backend.memory_db_sync import MemoryDBSyncService
from backend.memory_models import (
    DatasetRecord,
    EventParticipantRecord,
    EventPhotoLinkRecord,
    EventRevisionRecord,
    FaceObservationRecord,
    PersonFaceLinkRecord,
    PersonRecord,
    PhotoRecord,
    ProfileRevisionRecord,
    RelationshipDossierRevisionRecord,
    RelationshipRevisionRecord,
    RelationshipSharedEventRecord,
    TaskStageRecord,
    VLMObservationRevisionRecord,
)
from backend.models import TaskRecord
from backend.version_utils import build_stage_version_matrix, parse_numeric_version


@dataclass
class RetrievalFilters:
    task_id: Optional[str] = None
    dataset_id: Optional[int] = None
    all: bool = False
    scope: str = "dataset"
    pipeline_version: Optional[int] = None
    pipeline_channel: Optional[str] = None
    face_version: Optional[int] = None
    vlm_version: Optional[int] = None
    lp1_version: Optional[int] = None
    lp2_version: Optional[int] = None
    lp3_version: Optional[int] = None
    judge_version: Optional[int] = None
    updated_after: Optional[str] = None
    cursor: Optional[str] = None
    limit: int = 20
    include_raw: bool = False
    include_artifacts: bool = False
    include_traces: bool = False


class UserMemoryRetrievalService:
    def __init__(self):
        self.sync_service = MemoryDBSyncService()
        self.artifact_store = ArtifactCatalogStore()

    def list_datasets(self, user_id: str) -> dict:
        self._backfill_task_metadata(user_id)
        with SessionLocal() as session:
            datasets = session.execute(
                select(DatasetRecord)
                .where(DatasetRecord.user_id == user_id)
                .order_by(desc(DatasetRecord.updated_at), desc(DatasetRecord.dataset_id))
            ).scalars().all()
            items = []
            for dataset in datasets:
                latest_task = None
                if dataset.latest_task_id:
                    latest_task = session.get(TaskRecord, dataset.latest_task_id)
                items.append(
                    {
                        "dataset_id": dataset.dataset_id,
                        "photo_count": dataset.photo_count,
                        "source_hash_count": len(dataset.source_hashes_json or []),
                        "first_task_id": dataset.first_task_id,
                        "latest_task_id": dataset.latest_task_id,
                        "latest_task_version": latest_task.version if latest_task is not None else None,
                        "updated_at": dataset.updated_at.isoformat(),
                        "created_at": dataset.created_at.isoformat(),
                    }
                )
            return {"user_id": user_id, "datasets": items}

    def list_tasks(self, user_id: str, filters: RetrievalFilters) -> dict:
        tasks = self._select_tasks(user_id, filters, ensure_mirror=False)
        return {
            "query": self._query_payload(user_id, filters),
            "tasks": [self._serialize_task_header(task) for task in tasks],
            "next_cursor": self._next_cursor(filters, tasks),
        }

    def list_versions(self, user_id: str) -> dict:
        self._backfill_task_metadata(user_id)
        with SessionLocal() as session:
            rows = session.execute(
                select(
                    TaskRecord.pipeline_version,
                    TaskRecord.pipeline_channel,
                    TaskRecord.face_version,
                    TaskRecord.vlm_version,
                    TaskRecord.lp1_version,
                    TaskRecord.lp2_version,
                    TaskRecord.lp3_version,
                    TaskRecord.judge_version,
                ).where(TaskRecord.user_id == user_id)
            ).all()
        return {
            "pipeline_versions": sorted({row.pipeline_version for row in rows if row.pipeline_version is not None}),
            "pipeline_channels": sorted({row.pipeline_channel for row in rows if row.pipeline_channel}),
            "face_versions": sorted({row.face_version for row in rows if row.face_version is not None}),
            "vlm_versions": sorted({row.vlm_version for row in rows if row.vlm_version is not None}),
            "lp1_versions": sorted({row.lp1_version for row in rows if row.lp1_version is not None}),
            "lp2_versions": sorted({row.lp2_version for row in rows if row.lp2_version is not None}),
            "lp3_versions": sorted({row.lp3_version for row in rows if row.lp3_version is not None}),
            "judge_versions": sorted({row.judge_version for row in rows if row.judge_version is not None}),
        }

    def build_faces(self, user_id: str, filters: RetrievalFilters) -> dict:
        return self._build_payload(user_id, filters, include_faces=True)

    def build_events(self, user_id: str, filters: RetrievalFilters) -> dict:
        return self._build_payload(user_id, filters, include_events=True)

    def build_vlm(self, user_id: str, filters: RetrievalFilters) -> dict:
        return self._build_payload(user_id, filters, include_vlm=True)

    def build_profiles(self, user_id: str, filters: RetrievalFilters) -> dict:
        return self._build_payload(user_id, filters, include_profiles=True)

    def build_relationships(self, user_id: str, filters: RetrievalFilters) -> dict:
        return self._build_payload(user_id, filters, include_relationships=True)

    def build_photos(self, user_id: str, filters: RetrievalFilters) -> dict:
        return self._build_payload(user_id, filters, include_photos_only=True)

    def build_bundle(self, user_id: str, filters: RetrievalFilters) -> dict:
        return self._build_payload(
            user_id,
            filters,
            include_faces=True,
            include_events=True,
            include_vlm=True,
            include_profiles=True,
            include_relationships=True,
            include_bundle=True,
        )

    def _build_payload(
        self,
        user_id: str,
        filters: RetrievalFilters,
        *,
        include_faces: bool = False,
        include_events: bool = False,
        include_vlm: bool = False,
        include_profiles: bool = False,
        include_relationships: bool = False,
        include_photos_only: bool = False,
        include_bundle: bool = False,
    ) -> dict:
        tasks = self._select_tasks(user_id, filters, ensure_mirror=True)
        payload_tasks = []
        for task in tasks:
            payload_tasks.append(
                self._build_task_payload(
                    task,
                    include_faces=include_faces,
                    include_events=include_events,
                    include_vlm=include_vlm,
                    include_profiles=include_profiles,
                    include_relationships=include_relationships,
                    include_photos_only=include_photos_only,
                    include_bundle=include_bundle,
                    include_raw=filters.include_raw,
                    include_artifacts=filters.include_artifacts,
                    include_traces=filters.include_traces,
                )
            )
        return {
            "query": self._query_payload(user_id, filters),
            "tasks": payload_tasks,
            "next_cursor": self._next_cursor(filters, tasks),
        }

    def _select_tasks(self, user_id: str, filters: RetrievalFilters, *, ensure_mirror: bool) -> List[TaskRecord]:
        self._backfill_task_metadata(user_id)
        with SessionLocal() as session:
            stmt = select(TaskRecord).where(TaskRecord.user_id == user_id)
            if filters.task_id:
                stmt = stmt.where(TaskRecord.task_id == filters.task_id)
            else:
                dataset_id = filters.dataset_id
                if filters.scope != "user":
                    if dataset_id is None:
                        latest_dataset_id = session.execute(
                            select(DatasetRecord.dataset_id)
                            .where(DatasetRecord.user_id == user_id)
                            .order_by(desc(DatasetRecord.updated_at), desc(DatasetRecord.dataset_id))
                            .limit(1)
                        ).scalar_one_or_none()
                        dataset_id = latest_dataset_id
                    if dataset_id is not None:
                        stmt = stmt.where(TaskRecord.dataset_id == dataset_id)
            if filters.pipeline_version is not None:
                stmt = stmt.where(TaskRecord.pipeline_version == filters.pipeline_version)
            if filters.pipeline_channel:
                stmt = stmt.where(func.lower(TaskRecord.pipeline_channel) == filters.pipeline_channel.lower())
            for field_name, value in (
                ("face_version", filters.face_version),
                ("vlm_version", filters.vlm_version),
                ("lp1_version", filters.lp1_version),
                ("lp2_version", filters.lp2_version),
                ("lp3_version", filters.lp3_version),
                ("judge_version", filters.judge_version),
            ):
                if value is not None:
                    stmt = stmt.where(getattr(TaskRecord, field_name) == value)
            if filters.updated_after:
                stmt = stmt.where(TaskRecord.updated_at >= filters.updated_after)
            stmt = stmt.order_by(desc(TaskRecord.created_at), desc(TaskRecord.task_id))
            offset = int(filters.cursor or "0") if str(filters.cursor or "").isdigit() else 0
            if filters.all:
                stmt = stmt.offset(offset).limit(max(1, min(filters.limit, 100)))
            else:
                stmt = stmt.limit(1)
            tasks = session.execute(stmt).scalars().all()

        if ensure_mirror:
            for task in tasks:
                self.sync_service.sync_task_snapshot(self._task_record_payload(task))
            with SessionLocal() as session:
                refreshed = [
                    session.get(TaskRecord, task.task_id)
                    for task in tasks
                ]
                return [task for task in refreshed if task is not None]
        return tasks

    def _build_task_payload(
        self,
        task: TaskRecord,
        *,
        include_faces: bool,
        include_events: bool,
        include_vlm: bool,
        include_profiles: bool,
        include_relationships: bool,
        include_photos_only: bool,
        include_bundle: bool,
        include_raw: bool,
        include_artifacts: bool,
        include_traces: bool,
    ) -> dict:
        with SessionLocal() as session:
            photo_rows = session.execute(
                select(PhotoRecord).where(PhotoRecord.task_id == task.task_id).order_by(PhotoRecord.source_photo_id.asc())
            ).scalars().all()
            photos = [self._serialize_photo(row) for row in photo_rows]

            payload = self._serialize_task_header(task)
            payload["photos"] = photos

            if include_photos_only:
                if include_artifacts:
                    payload["artifacts"] = self.artifact_store.list_task_artifacts(task.task_id, task.user_id or "")
                return payload

            person_rows = session.execute(
                select(PersonRecord).where(PersonRecord.task_id == task.task_id).order_by(PersonRecord.person_id.asc())
            ).scalars().all()
            person_face_links = session.execute(
                select(PersonFaceLinkRecord).where(PersonFaceLinkRecord.task_id == task.task_id)
            ).scalars().all()
            faces_rows = session.execute(
                select(FaceObservationRecord).where(FaceObservationRecord.task_id == task.task_id).order_by(FaceObservationRecord.face_id.asc())
            ).scalars().all()
            face_by_id = {row.face_id: row for row in faces_rows}
            photo_ids_by_person: dict[str, set[str]] = {}
            face_ids_by_person: dict[str, set[str]] = {}
            for link in person_face_links:
                face_ids_by_person.setdefault(link.person_id, set()).add(link.face_id)
                photo_ids_by_person.setdefault(link.person_id, set()).add(link.photo_id)
            persons = [
                self._serialize_person(
                    row,
                    face_ids=sorted(face_ids_by_person.get(row.person_id, set())),
                    photo_ids=sorted(photo_ids_by_person.get(row.person_id, set())),
                )
                for row in person_rows
            ]
            payload["persons"] = persons

            if include_faces or include_bundle:
                payload["faces"] = [self._serialize_face(row) for row in faces_rows]
                payload["identities"] = [
                    {
                        "person_id": row.person_id,
                        "canonical_name": row.canonical_name,
                        "representative_face_id": sorted(face_ids_by_person.get(row.person_id, set()))[0]
                        if face_ids_by_person.get(row.person_id)
                        else None,
                        "face_ids": sorted(face_ids_by_person.get(row.person_id, set())),
                        "photo_ids": sorted(photo_ids_by_person.get(row.person_id, set())),
                        "face_count": row.face_count,
                        "photo_count": row.photo_count,
                        "is_primary_person": row.is_primary_person,
                    }
                    for row in person_rows
                ]

            if include_vlm or include_bundle:
                vlm_rows = session.execute(
                    select(VLMObservationRevisionRecord)
                    .where(VLMObservationRevisionRecord.task_id == task.task_id)
                    .order_by(VLMObservationRevisionRecord.source_photo_id.asc())
                ).scalars().all()
                payload["vlm_observations"] = [
                    self._serialize_vlm(row, include_raw=include_raw)
                    for row in vlm_rows
                ]

            if include_events or include_bundle or include_relationships:
                event_rows = session.execute(
                    select(EventRevisionRecord)
                    .where(EventRevisionRecord.task_id == task.task_id)
                    .order_by(EventRevisionRecord.date.asc(), EventRevisionRecord.event_id.asc())
                ).scalars().all()
                participant_rows = session.execute(
                    select(EventParticipantRecord).where(EventParticipantRecord.task_id == task.task_id)
                ).scalars().all()
                photo_link_rows = session.execute(
                    select(EventPhotoLinkRecord).where(EventPhotoLinkRecord.task_id == task.task_id)
                ).scalars().all()
                participant_map: dict[str, list[str]] = {}
                for row in participant_rows:
                    participant_map.setdefault(row.event_revision_id, []).append(row.person_id)
                photo_link_map: dict[str, list[str]] = {}
                for row in photo_link_rows:
                    photo_link_map.setdefault(row.event_revision_id, []).append(row.photo_id)
                events = [
                    self._serialize_event(
                        row,
                        participant_person_ids=sorted(participant_map.get(row.id, [])),
                        photo_ids=sorted(photo_link_map.get(row.id, [])),
                    )
                    for row in event_rows
                ]
                payload["events"] = events
                if include_raw:
                    lp1_stage = session.execute(
                        select(TaskStageRecord).where(
                            TaskStageRecord.task_id == task.task_id,
                            TaskStageRecord.stage_name == "lp1",
                        )
                    ).scalar_one_or_none()
                    payload["raw_events"] = list(((lp1_stage.raw_payload_json or {}).get("lp1_events_raw") or [])) if lp1_stage else []

            if include_relationships or include_bundle:
                relationship_rows = session.execute(
                    select(RelationshipRevisionRecord)
                    .where(RelationshipRevisionRecord.task_id == task.task_id)
                    .order_by(RelationshipRevisionRecord.person_id.asc())
                ).scalars().all()
                shared_event_rows = session.execute(
                    select(RelationshipSharedEventRecord).where(RelationshipSharedEventRecord.task_id == task.task_id)
                ).scalars().all()
                dossier_rows = session.execute(
                    select(RelationshipDossierRevisionRecord).where(RelationshipDossierRevisionRecord.task_id == task.task_id)
                ).scalars().all()
                shared_map: dict[str, list[dict]] = {}
                for row in shared_event_rows:
                    shared_map.setdefault(row.relationship_revision_id, []).append(
                        {
                            "event_id": row.event_id,
                            "date": row.date_snapshot,
                            "narrative": row.narrative_snapshot,
                        }
                    )
                payload["relationships"] = [
                    self._serialize_relationship(row, shared_events=shared_map.get(row.id, []))
                    for row in relationship_rows
                ]
                if include_raw:
                    payload["relationship_dossiers"] = [copy.deepcopy(row.dossier_json or {}) for row in dossier_rows]

            if include_profiles or include_bundle:
                profile_rows = session.execute(
                    select(ProfileRevisionRecord).where(ProfileRevisionRecord.task_id == task.task_id)
                ).scalars().all()
                payload["profiles"] = [self._serialize_profile(row, include_raw=include_raw) for row in profile_rows]

            if include_artifacts:
                payload["artifacts"] = self.artifact_store.list_task_artifacts(task.task_id, task.user_id or "")
            if include_traces:
                payload["agent_traces"] = []
            return payload

    def _backfill_task_metadata(self, user_id: str) -> None:
        with SessionLocal() as session:
            tasks = session.execute(select(TaskRecord).where(TaskRecord.user_id == user_id)).scalars().all()
            dirty = False
            for task in tasks:
                if task.pipeline_version is None or task.pipeline_channel is None and task.version:
                    numeric_version, channel = parse_numeric_version(task.version)
                    task.pipeline_version = numeric_version
                    task.pipeline_channel = channel
                    stage_matrix = build_stage_version_matrix(task.version, task.result if isinstance(task.result, dict) else None)
                    task.face_version = stage_matrix.get("face_version")
                    task.vlm_version = stage_matrix.get("vlm_version")
                    task.lp1_version = stage_matrix.get("lp1_version")
                    task.lp2_version = stage_matrix.get("lp2_version")
                    task.lp3_version = stage_matrix.get("lp3_version")
                    task.judge_version = stage_matrix.get("judge_version")
                    dirty = True
                if task.dataset_id is None and task.uploads:
                    self.sync_service.ensure_dataset_for_task(self._task_record_payload(task))
            if dirty:
                session.commit()

    def _task_record_payload(self, task: TaskRecord) -> dict:
        return {
            "task_id": task.task_id,
            "user_id": task.user_id,
            "dataset_id": task.dataset_id,
            "dataset_fingerprint": task.dataset_fingerprint,
            "version": task.version,
            "status": task.status,
            "stage": task.stage,
            "task_dir": task.task_dir,
            "uploads": copy.deepcopy(task.uploads) if isinstance(task.uploads, list) else [],
            "result": copy.deepcopy(task.result) if isinstance(task.result, dict) else None,
            "asset_manifest": copy.deepcopy(task.asset_manifest) if isinstance(task.asset_manifest, dict) else None,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
        }

    def _query_payload(self, user_id: str, filters: RetrievalFilters) -> dict:
        return {
            "user_id": user_id,
            "dataset_id": filters.dataset_id,
            "task_id": filters.task_id,
            "all": filters.all,
            "scope": filters.scope,
            "filters": {
                "pipeline_version": filters.pipeline_version,
                "pipeline_channel": filters.pipeline_channel,
                "face_version": filters.face_version,
                "vlm_version": filters.vlm_version,
                "lp1_version": filters.lp1_version,
                "lp2_version": filters.lp2_version,
                "lp3_version": filters.lp3_version,
                "judge_version": filters.judge_version,
            },
        }

    def _next_cursor(self, filters: RetrievalFilters, tasks: List[TaskRecord]) -> Optional[str]:
        if not filters.all or len(tasks) < max(1, min(filters.limit, 100)):
            return None
        offset = int(filters.cursor or "0") if str(filters.cursor or "").isdigit() else 0
        return str(offset + len(tasks))

    def _serialize_task_header(self, task: TaskRecord) -> dict:
        return {
            "task_id": task.task_id,
            "dataset_id": task.dataset_id,
            "version": task.version,
            "pipeline_version": task.pipeline_version,
            "pipeline_channel": task.pipeline_channel,
            "status": task.status,
            "stage": task.stage,
            "stage_versions": {
                "face": task.face_version,
                "vlm": task.vlm_version,
                "lp1": task.lp1_version,
                "lp2": task.lp2_version,
                "lp3": task.lp3_version,
                "judge": task.judge_version,
            },
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat(),
        }

    def _serialize_photo(self, row: PhotoRecord) -> dict:
        return {
            "photo_id": row.photo_id,
            "source_photo_id": row.source_photo_id,
            "dataset_id": row.dataset_id,
            "original_filename": row.original_filename,
            "source_hash": row.source_hash,
            "taken_at": row.taken_at,
            "location": copy.deepcopy(row.location_json),
            "width": row.width,
            "height": row.height,
            "mime_type": row.mime_type,
            "exif": copy.deepcopy(row.exif_json),
            "urls": {
                "raw": row.raw_url,
                "display": row.display_url or row.raw_url,
                "boxed_full": row.boxed_url or row.display_url or row.raw_url,
                "compressed": row.compressed_url or row.display_url or row.raw_url,
            },
        }

    def _serialize_person(self, row: PersonRecord, *, face_ids: List[str], photo_ids: List[str]) -> dict:
        return {
            "person_id": row.person_id,
            "canonical_name": row.canonical_name,
            "is_primary_person": row.is_primary_person,
            "face_ids": face_ids,
            "photo_ids": photo_ids,
            "face_count": row.face_count,
            "photo_count": row.photo_count,
            "avg_score": row.avg_score,
            "avg_quality": row.avg_quality,
            "high_quality_face_count": row.high_quality_face_count,
            "avatar_url": row.avatar_url,
        }

    def _serialize_face(self, row: FaceObservationRecord) -> dict:
        return {
            "face_id": row.face_id,
            "person_id": row.person_id,
            "photo_id": row.photo_id,
            "source_photo_id": row.source_photo_id,
            "bbox": copy.deepcopy(row.bbox_json),
            "bbox_xywh": copy.deepcopy(row.bbox_xywh_json),
            "score": row.score,
            "similarity": row.similarity,
            "quality_score": row.quality_score,
            "match_decision": row.match_decision,
            "match_reason": row.match_reason,
            "urls": {
                "crop": row.crop_url,
                "boxed_full": row.boxed_url,
            },
        }

    def _serialize_vlm(self, row: VLMObservationRevisionRecord, *, include_raw: bool) -> dict:
        payload = {
            "vlm_observation_id": row.id,
            "photo_id": row.photo_id,
            "source_photo_id": row.source_photo_id,
            "summary": row.summary,
            "people": copy.deepcopy(row.people_json),
            "relations": copy.deepcopy(row.relations_json),
            "scene": copy.deepcopy(row.scene_json),
            "event": copy.deepcopy(row.event_json),
            "details": copy.deepcopy(row.details_json),
            "clues": copy.deepcopy(row.clues_json),
        }
        if include_raw:
            payload["raw"] = copy.deepcopy(row.raw_payload_json)
        return payload

    def _serialize_event(self, row: EventRevisionRecord, *, participant_person_ids: List[str], photo_ids: List[str]) -> dict:
        return {
            "event_id": row.event_id,
            "title": row.title,
            "date": row.date,
            "time_range": copy.deepcopy(row.time_range),
            "duration": copy.deepcopy(row.duration),
            "type": row.type,
            "location": copy.deepcopy(row.location),
            "description": row.description,
            "photo_count": row.photo_count,
            "confidence": row.confidence,
            "reason": row.reason,
            "narrative": row.narrative,
            "narrative_synthesis": row.narrative_synthesis,
            "participant_person_ids": participant_person_ids,
            "photo_ids": photo_ids,
            "tags": copy.deepcopy(row.tags_json),
            "social_dynamics": copy.deepcopy(row.social_dynamics_json),
            "persona_evidence": copy.deepcopy(row.persona_evidence_json),
        }

    def _serialize_relationship(self, row: RelationshipRevisionRecord, *, shared_events: List[dict]) -> dict:
        return {
            "relationship_id": row.relationship_id,
            "person_id": row.person_id,
            "relationship_type": row.relationship_type,
            "intimacy_score": row.intimacy_score,
            "status": row.status,
            "confidence": row.confidence,
            "reasoning": row.reasoning,
            "evidence": copy.deepcopy(row.evidence_json),
            "shared_events": shared_events,
        }

    def _serialize_profile(self, row: ProfileRevisionRecord, *, include_raw: bool) -> dict:
        payload = {
            "profile_revision_id": row.profile_revision_id,
            "primary_person_id": row.primary_person_id,
            "structured": copy.deepcopy(row.structured_json),
            "report": row.report_markdown,
            "summary": row.summary,
            "consistency": copy.deepcopy(row.consistency_json),
        }
        if include_raw:
            payload["debug"] = copy.deepcopy(row.debug_json)
            payload["field_decisions"] = copy.deepcopy(row.field_decisions_json)
            payload["internal_artifacts"] = copy.deepcopy(row.internal_artifacts_json)
        return payload
