"""
Mirror task-directory outputs into normalized database tables.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from PIL import Image, ImageOps
from sqlalchemy import delete, select

from backend.db import SessionLocal
from backend.memory_models import (
    AgentMessageRecord,
    AgentOutputRecord,
    AgentRunRecord,
    AgentToolCallRecord,
    AgentTraceEventRecord,
    BinaryAssetRecord,
    ConsistencyCheckRevisionRecord,
    DatasetRecord,
    EventDetailUnitRecord,
    EventParticipantRecord,
    EventPhotoLinkRecord,
    EventRevisionRecord,
    EventRootRecord,
    FaceEmbeddingRecord,
    FaceObservationRecord,
    GroupMemberRecord,
    GroupRevisionRecord,
    GroupRootRecord,
    ObjectLinkRecord,
    ObjectRegistryRecord,
    PersonFaceLinkRecord,
    PersonRecord,
    PersonRevisionRecord,
    PhotoAssetRecord,
    PhotoExifRecord,
    PhotoRecord,
    ProfileContextRevisionRecord,
    ProfileFactDecisionRecord,
    ProfileFieldValueRecord,
    ProfileRevisionRecord,
    RelationshipDossierRevisionRecord,
    RelationshipRevisionRecord,
    RelationshipRootRecord,
    RelationshipSharedEventRecord,
    TaskPhotoItemRecord,
    TaskStageRecord,
    UserHeadRecord,
    VLMObservationClueRecord,
    VLMObservationPersonRecord,
    VLMObservationRelationRecord,
    VLMObservationRevisionRecord,
)
from backend.models import ArtifactRecord, TaskRecord
from backend.version_utils import build_stage_version_matrix, parse_numeric_version
from config import ASSET_URL_PREFIX
from services.asset_store import TaskAssetStore


PHOTO_VARIANTS = ("raw", "display", "boxed", "compressed")
STAGE_NAMES = ("ingest", "face", "vlm", "lp1", "lp2", "lp3")


def _now() -> datetime:
    return datetime.now()


def _new_id() -> str:
    return uuid.uuid4().hex


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _stringify_exif_text(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            value = str(value)
    text = str(value).strip()
    return text or None


def _canonical_photo_id(task_id: str, source_photo_id: str) -> str:
    return f"{task_id}:{source_photo_id}"


def _canonical_face_id(task_id: str, source_face_id: str) -> str:
    return f"{task_id}:{source_face_id}"


def _canonical_relationship_id(task_id: str, person_id: str) -> str:
    return f"{task_id}:{person_id}"


def _canonical_profile_revision_id(task_id: str) -> str:
    return f"{task_id}:profile"


def _photo_variant_url(photo_id: str, variant: str) -> str:
    return f"{ASSET_URL_PREFIX}/photos/{photo_id}/{variant}"


def _face_crop_url(face_id: str) -> str:
    return f"{ASSET_URL_PREFIX}/faces/{face_id}/crop"


def _json_clone(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return copy.deepcopy(value)
    return value


def _task_asset_relative_path(asset_url: str | None, task_id: str) -> str | None:
    if not asset_url:
        return None
    prefix = f"{ASSET_URL_PREFIX}/{task_id}/"
    if not asset_url.startswith(prefix):
        return None
    return asset_url[len(prefix):]


def _extract_exif(local_path: Path) -> tuple[dict | None, dict | None]:
    if not local_path.exists():
        return None, None
    try:
        with Image.open(local_path) as image:
            exif = image.getexif()
            if not exif:
                return None, None
            raw_payload = {str(key): exif.get(key) for key in exif.keys()}
            normalized = {
                "width": image.width,
                "height": image.height,
                "orientation": exif.get(274),
                "datetime": _stringify_exif_text(exif.get(306)),
                "datetime_original": _stringify_exif_text(exif.get(36867)),
                "camera_make": _stringify_exif_text(exif.get(271)),
                "camera_model": _stringify_exif_text(exif.get(272)),
            }
            return raw_payload, normalized
    except Exception:
        return None, None


def _flatten_profile_fields(node: Any, prefix: str = "") -> Iterable[tuple[str, dict]]:
    if not isinstance(node, dict):
        return
    tag_like = {"value", "confidence", "reasoning", "evidence"}
    if prefix and tag_like.intersection(node.keys()):
        yield prefix, node
    for key, value in node.items():
        if not isinstance(key, str):
            continue
        child_prefix = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            yield from _flatten_profile_fields(value, child_prefix)


def _artifact_manifest_for_stage(manifest: dict | None, prefixes: Iterable[str]) -> dict:
    files = list((manifest or {}).get("files", []) or [])
    allowed = []
    prefixes = tuple(prefixes)
    for item in files:
        relative_path = str(item.get("relative_path") or item.get("path") or "")
        if relative_path.startswith(prefixes):
            allowed.append(copy.deepcopy(item))
    return {"files": allowed}


def _stage_artifact_prefixes(stage_name: str) -> tuple[str, ...]:
    if stage_name == "ingest":
        return ("uploads/",)
    if stage_name == "face":
        return ("cache/",)
    if stage_name in {"vlm", "lp1", "lp2", "lp3"}:
        return ("v0325/", "output/")
    return tuple()


def _coerce_location(value: Any) -> Any:
    if value in (None, "", []):
        return None
    return value


class MemoryDBSyncService:
    def __init__(self, asset_store: Optional[TaskAssetStore] = None):
        self.asset_store = asset_store or TaskAssetStore()

    def _delete_existing_task_rows(self, session, task_id: str) -> None:
        photo_ids = [
            row[0]
            for row in session.execute(
                select(PhotoRecord.photo_id).where(PhotoRecord.task_id == task_id)
            ).all()
        ]
        profile_ids = [
            row[0]
            for row in session.execute(
                select(ProfileRevisionRecord.id).where(ProfileRevisionRecord.task_id == task_id)
            ).all()
        ]
        agent_run_ids = [
            row[0]
            for row in session.execute(
                select(AgentRunRecord.id).where(AgentRunRecord.task_id == task_id)
            ).all()
        ]
        registry_ids = [
            row[0]
            for row in session.execute(
                select(ObjectRegistryRecord.id).where(ObjectRegistryRecord.task_id == task_id)
            ).all()
        ]
        if registry_ids:
            session.execute(
                delete(ObjectLinkRecord).where(
                    ObjectLinkRecord.from_id.in_(registry_ids) | ObjectLinkRecord.to_id.in_(registry_ids)
                )
            )
        if photo_ids:
            session.execute(delete(PhotoExifRecord).where(PhotoExifRecord.photo_id.in_(photo_ids)))
        if profile_ids:
            session.execute(
                delete(ConsistencyCheckRevisionRecord).where(
                    ConsistencyCheckRevisionRecord.profile_revision_id.in_(profile_ids)
                )
            )
            session.execute(
                delete(ProfileFactDecisionRecord).where(
                    ProfileFactDecisionRecord.profile_revision_id.in_(profile_ids)
                )
            )
            session.execute(
                delete(ProfileFieldValueRecord).where(
                    ProfileFieldValueRecord.profile_revision_id.in_(profile_ids)
                )
            )
        if agent_run_ids:
            session.execute(delete(AgentOutputRecord).where(AgentOutputRecord.agent_run_id.in_(agent_run_ids)))
            session.execute(
                delete(AgentTraceEventRecord).where(AgentTraceEventRecord.agent_run_id.in_(agent_run_ids))
            )
            session.execute(
                delete(AgentToolCallRecord).where(AgentToolCallRecord.agent_run_id.in_(agent_run_ids))
            )
            session.execute(delete(AgentMessageRecord).where(AgentMessageRecord.agent_run_id.in_(agent_run_ids)))
            session.execute(delete(AgentRunRecord).where(AgentRunRecord.id.in_(agent_run_ids)))
        for model in (
            ObjectRegistryRecord,
            ProfileRevisionRecord,
            ProfileContextRevisionRecord,
            GroupMemberRecord,
            GroupRevisionRecord,
            GroupRootRecord,
            RelationshipSharedEventRecord,
            RelationshipRevisionRecord,
            RelationshipDossierRevisionRecord,
            RelationshipRootRecord,
            EventDetailUnitRecord,
            EventPhotoLinkRecord,
            EventParticipantRecord,
            EventRevisionRecord,
            EventRootRecord,
            VLMObservationClueRecord,
            VLMObservationRelationRecord,
            VLMObservationPersonRecord,
            VLMObservationRevisionRecord,
            FaceEmbeddingRecord,
            PersonFaceLinkRecord,
            FaceObservationRecord,
            PersonRevisionRecord,
            PersonRecord,
            PhotoAssetRecord,
            TaskPhotoItemRecord,
            PhotoRecord,
            BinaryAssetRecord,
            TaskStageRecord,
        ):
            session.execute(delete(model).where(model.task_id == task_id))  # type: ignore[arg-type]

    def delete_task_snapshot(self, task_id: str, user_id: str, *, delete_task_record: bool = False) -> bool:
        with SessionLocal() as session:
            task_record = session.get(TaskRecord, task_id)
            if task_record is None or task_record.user_id != user_id:
                return False

            dataset_id = task_record.dataset_id
            self._delete_existing_task_rows(session, task_id)

            if delete_task_record:
                session.execute(
                    delete(TaskRecord).where(
                        TaskRecord.task_id == task_id,
                        TaskRecord.user_id == user_id,
                    )
                )

            if dataset_id is not None:
                remaining_tasks = session.execute(
                    select(TaskRecord)
                    .where(
                        TaskRecord.user_id == user_id,
                        TaskRecord.dataset_id == dataset_id,
                    )
                    .order_by(TaskRecord.created_at.asc(), TaskRecord.task_id.asc())
                ).scalars().all()
                dataset = session.get(DatasetRecord, dataset_id)
                if remaining_tasks:
                    if dataset is not None:
                        dataset.first_task_id = remaining_tasks[0].task_id
                        dataset.latest_task_id = remaining_tasks[-1].task_id
                        dataset.updated_at = _now()
                        session.add(dataset)
                elif dataset is not None:
                    session.delete(dataset)

            user_head = session.get(UserHeadRecord, user_id)
            if user_head is not None:
                latest_task = session.execute(
                    select(TaskRecord)
                    .where(TaskRecord.user_id == user_id)
                    .order_by(TaskRecord.created_at.desc(), TaskRecord.task_id.desc())
                    .limit(1)
                ).scalar_one_or_none()
                user_head.active_task_id = latest_task.task_id if latest_task is not None else None
                user_head.active_dataset_id = latest_task.dataset_id if latest_task is not None else None
                if latest_task is not None:
                    latest_profile = session.execute(
                        select(ProfileRevisionRecord.id)
                        .where(ProfileRevisionRecord.task_id == latest_task.task_id)
                        .order_by(ProfileRevisionRecord.updated_at.desc(), ProfileRevisionRecord.id.desc())
                        .limit(1)
                    ).scalar_one_or_none()
                    user_head.active_profile_revision_id = latest_profile
                else:
                    user_head.active_profile_revision_id = None
                session.add(user_head)

            session.commit()
            return True

    def ensure_dataset_for_task(self, task: dict) -> int | None:
        uploads = list(task.get("uploads") or [])
        user_id = str(task.get("user_id") or "").strip()
        task_id = str(task.get("task_id") or "").strip()
        if not user_id or not task_id or not uploads:
            return None
        source_hashes = sorted({str(item.get("source_hash") or "").strip() for item in uploads if item.get("source_hash")})
        if not source_hashes:
            return None
        fingerprint = hashlib.sha256("\n".join(source_hashes).encode("utf-8")).hexdigest()
        now = _now()

        with SessionLocal() as session:
            record = session.execute(
                select(DatasetRecord).where(
                    DatasetRecord.user_id == user_id,
                    DatasetRecord.dataset_fingerprint == fingerprint,
                )
            ).scalar_one_or_none()
            if record is None:
                record = DatasetRecord(
                    dataset_id=None,  # type: ignore[arg-type]
                    user_id=user_id,
                    dataset_fingerprint=fingerprint,
                    photo_count=len(uploads),
                    source_hashes_json=source_hashes,
                    first_task_id=task_id,
                    latest_task_id=task_id,
                    created_at=now,
                    updated_at=now,
                )
                session.add(record)
                session.flush()
            else:
                record.photo_count = max(int(record.photo_count or 0), len(uploads))
                record.latest_task_id = task_id
                record.source_hashes_json = source_hashes
                record.updated_at = now
                session.add(record)

            task_record = session.get(TaskRecord, task_id)
            if task_record is not None:
                task_record.dataset_id = record.dataset_id
                task_record.dataset_fingerprint = fingerprint
                session.add(task_record)

            user_head = session.get(UserHeadRecord, user_id)
            if user_head is None:
                user_head = UserHeadRecord(
                    user_id=user_id,
                    active_dataset_id=record.dataset_id,
                    active_task_id=task_id,
                    active_profile_revision_id=None,
                    active_gt_revision_id=None,
                    updated_at=now,
                )
            else:
                user_head.active_dataset_id = record.dataset_id
                user_head.active_task_id = task_id
                user_head.updated_at = now
            session.add(user_head)
            session.commit()
            return record.dataset_id

    def sync_task_snapshot(
        self,
        task: dict,
        *,
        face_embedding_lookup: Optional[Callable[[int], Optional[Iterable[float]]]] = None,
    ) -> None:
        user_id = str(task.get("user_id") or "").strip()
        task_id = str(task.get("task_id") or "").strip()
        if not user_id or not task_id:
            return

        dataset_id = self.ensure_dataset_for_task(task)
        result = task.get("result") or {}
        memory = result.get("memory") if isinstance(result.get("memory"), dict) else {}
        face_payload = result.get("face_recognition") if isinstance(result.get("face_recognition"), dict) else {}
        manifest = task.get("asset_manifest") if isinstance(task.get("asset_manifest"), dict) else {}
        uploads = list(task.get("uploads") or [])
        task_dir = Path(str(task.get("task_dir") or ""))
        version_text = str(task.get("version") or "").strip()
        pipeline_version, pipeline_channel = parse_numeric_version(version_text)
        stage_versions = build_stage_version_matrix(version_text, result if isinstance(result, dict) else None)
        now = _now()

        photo_rows: list[PhotoRecord] = []
        photo_exif_rows: list[PhotoExifRecord] = []
        photo_asset_rows: list[PhotoAssetRecord] = []
        task_photo_item_rows: list[TaskPhotoItemRecord] = []
        binary_asset_rows: list[BinaryAssetRecord] = []
        person_rows: list[PersonRecord] = []
        person_revision_rows: list[PersonRevisionRecord] = []
        face_rows: list[FaceObservationRecord] = []
        face_embedding_rows: list[FaceEmbeddingRecord] = []
        person_face_link_rows: list[PersonFaceLinkRecord] = []
        vlm_rows: list[VLMObservationRevisionRecord] = []
        vlm_people_rows: list[VLMObservationPersonRecord] = []
        vlm_relation_rows: list[VLMObservationRelationRecord] = []
        vlm_clue_rows: list[VLMObservationClueRecord] = []
        event_root_rows: list[EventRootRecord] = []
        event_revision_rows: list[EventRevisionRecord] = []
        event_participant_rows: list[EventParticipantRecord] = []
        event_photo_rows: list[EventPhotoLinkRecord] = []
        event_detail_rows: list[EventDetailUnitRecord] = []
        relationship_root_rows: list[RelationshipRootRecord] = []
        relationship_dossier_rows: list[RelationshipDossierRevisionRecord] = []
        relationship_revision_rows: list[RelationshipRevisionRecord] = []
        relationship_shared_event_rows: list[RelationshipSharedEventRecord] = []
        group_root_rows: list[GroupRootRecord] = []
        group_revision_rows: list[GroupRevisionRecord] = []
        group_member_rows: list[GroupMemberRecord] = []
        profile_context_rows: list[ProfileContextRevisionRecord] = []
        profile_revision_rows: list[ProfileRevisionRecord] = []
        profile_field_rows: list[ProfileFieldValueRecord] = []
        profile_fact_rows: list[ProfileFactDecisionRecord] = []
        consistency_rows: list[ConsistencyCheckRevisionRecord] = []
        stage_rows: list[TaskStageRecord] = []
        registry_rows: list[ObjectRegistryRecord] = []
        link_rows: list[ObjectLinkRecord] = []

        image_map = {
            str(item.get("image_id") or item.get("photo_id") or ""): item
            for item in list(face_payload.get("images", []) or [])
            if isinstance(item, dict)
        }

        for file_item in list(manifest.get("files", []) or []):
            relative_path = str(file_item.get("relative_path") or file_item.get("path") or "").strip()
            if not relative_path:
                continue
            binary_asset_rows.append(
                BinaryAssetRecord(
                    id=_new_id(),
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    asset_kind=str(file_item.get("stage") or relative_path.split("/", 1)[0] or "artifact"),
                    relative_path=relative_path,
                    mime_type=str(file_item.get("content_type") or "") or None,
                    size_bytes=int(file_item.get("size_bytes") or file_item.get("size") or 0),
                    sha256=str(file_item.get("sha256") or "") or None,
                    storage_backend=str(file_item.get("storage_backend") or "") or None,
                    object_key=str(file_item.get("object_key") or "") or None,
                    asset_url=str(file_item.get("asset_url") or "") or None,
                    metadata_json=_json_clone(file_item.get("metadata")),
                    created_at=now,
                    updated_at=now,
                )
            )

        photo_id_map: dict[str, str] = {}
        for index, upload in enumerate(uploads, start=1):
            source_photo_id = str(upload.get("image_id") or f"photo_{index:03d}").strip()
            photo_id = _canonical_photo_id(task_id, source_photo_id)
            photo_id_map[source_photo_id] = photo_id
            image_payload = image_map.get(source_photo_id, {})
            raw_relative_path = str(upload.get("path") or "").strip() or _task_asset_relative_path(
                image_payload.get("original_image_url"),
                task_id,
            )
            display_relative_path = _task_asset_relative_path(image_payload.get("display_image_url"), task_id)
            boxed_relative_path = _task_asset_relative_path(image_payload.get("boxed_image_url"), task_id)
            compressed_relative_path = _task_asset_relative_path(image_payload.get("compressed_image_url"), task_id)
            raw_local_path = (task_dir / raw_relative_path) if raw_relative_path else None
            raw_exif, normalized_exif = _extract_exif(raw_local_path) if raw_local_path else (None, None)
            taken_at = (
                _stringify_exif_text((normalized_exif or {}).get("datetime_original"))
                or _stringify_exif_text((normalized_exif or {}).get("datetime"))
                or _stringify_exif_text(image_payload.get("timestamp"))
            )

            photo_rows.append(
                PhotoRecord(
                    photo_id=photo_id,
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    source_photo_id=source_photo_id,
                    original_filename=str(upload.get("filename") or "") or None,
                    stored_filename=str(upload.get("stored_filename") or "") or None,
                    source_hash=str(upload.get("source_hash") or "") or None,
                    mime_type=str(upload.get("content_type") or "") or None,
                    width=_safe_int(upload.get("width") or image_payload.get("width")),
                    height=_safe_int(upload.get("height") or image_payload.get("height")),
                    taken_at=taken_at,
                    location_json=_coerce_location(image_payload.get("location")),
                    exif_json=normalized_exif,
                    raw_relative_path=raw_relative_path or None,
                    display_relative_path=display_relative_path or None,
                    boxed_relative_path=boxed_relative_path or None,
                    compressed_relative_path=compressed_relative_path or None,
                    raw_url=_photo_variant_url(photo_id, "raw"),
                    display_url=_photo_variant_url(photo_id, "display"),
                    boxed_url=_photo_variant_url(photo_id, "boxed"),
                    compressed_url=_photo_variant_url(photo_id, "compressed"),
                    metadata_json={"upload": _json_clone(upload), "face_image": _json_clone(image_payload)},
                    created_at=now,
                    updated_at=now,
                )
            )
            photo_exif_rows.append(
                PhotoExifRecord(
                    id=_new_id(),
                    photo_id=photo_id,
                    raw_exif_json=raw_exif,
                    normalized_exif_json=normalized_exif,
                    captured_at=(normalized_exif or {}).get("datetime_original") or (normalized_exif or {}).get("datetime"),
                    gps_lat=None,
                    gps_lng=None,
                    camera_make=(normalized_exif or {}).get("camera_make"),
                    camera_model=(normalized_exif or {}).get("camera_model"),
                    created_at=now,
                    updated_at=now,
                )
            )
            for variant, relative_path in (
                ("raw", raw_relative_path),
                ("display", display_relative_path),
                ("boxed", boxed_relative_path),
                ("compressed", compressed_relative_path),
            ):
                if not relative_path and variant != "display":
                    continue
                photo_asset_rows.append(
                    PhotoAssetRecord(
                        id=_new_id(),
                        photo_id=photo_id,
                        user_id=user_id,
                        task_id=task_id,
                        dataset_id=dataset_id,
                        variant_type=variant,
                        relative_path=relative_path or raw_relative_path,
                        asset_url=_photo_variant_url(photo_id, variant),
                        metadata_json=None,
                        created_at=now,
                        updated_at=now,
                    )
                )
            task_photo_item_rows.append(
                TaskPhotoItemRecord(
                    id=_new_id(),
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    photo_id=photo_id,
                    source_photo_id=source_photo_id,
                    upload_order=index,
                    batch_no=None,
                    upload_status="uploaded",
                    source_hash=str(upload.get("source_hash") or "") or None,
                    metadata_json={"filename": upload.get("filename"), "stored_filename": upload.get("stored_filename")},
                    created_at=now,
                    updated_at=now,
                )
            )
            registry_rows.append(
                ObjectRegistryRecord(
                    id=_new_id(),
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    object_type="photo",
                    semantic_id=photo_id,
                    table_name="photos",
                    parent_id=None,
                    root_id=None,
                    metadata_json={"source_photo_id": source_photo_id},
                    created_at=now,
                )
            )

        person_groups = list(face_payload.get("person_groups", []) or [])
        primary_person_id = str(face_payload.get("primary_person_id") or "").strip() or None
        person_registry: dict[str, str] = {}
        for group in person_groups:
            person_id = str(group.get("person_id") or "").strip()
            if not person_id:
                continue
            registry_id = _new_id()
            person_registry[person_id] = registry_id
            person_rows.append(
                PersonRecord(
                    id=_new_id(),
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    person_id=person_id,
                    canonical_name=person_id,
                    is_primary_person=bool(group.get("is_primary") or person_id == primary_person_id),
                    photo_count=int(group.get("photo_count") or len(group.get("images") or [])),
                    face_count=int(group.get("face_count") or len(group.get("images") or [])),
                    avg_score=_safe_float(group.get("avg_score")),
                    avg_quality=_safe_float(group.get("avg_quality")),
                    high_quality_face_count=int(group.get("high_quality_face_count") or 0),
                    avatar_relative_path=_task_asset_relative_path(group.get("avatar_url"), task_id),
                    avatar_url=str(group.get("avatar_url") or "") or None,
                    metadata_json=_json_clone(group),
                    created_at=now,
                    updated_at=now,
                )
            )
            person_revision_rows.append(
                PersonRevisionRecord(
                    id=_new_id(),
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    person_id=person_id,
                    revision_no=1,
                    snapshot_json=_json_clone(group),
                    created_at=now,
                    updated_at=now,
                )
            )
            registry_rows.append(
                ObjectRegistryRecord(
                    id=registry_id,
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    object_type="person",
                    semantic_id=person_id,
                    table_name="persons",
                    parent_id=None,
                    root_id=None,
                    metadata_json={"is_primary_person": bool(group.get("is_primary") or person_id == primary_person_id)},
                    created_at=now,
                )
            )

        for image in list(face_payload.get("images", []) or []):
            source_photo_id = str(image.get("image_id") or image.get("photo_id") or "").strip()
            canonical_photo_id = photo_id_map.get(source_photo_id)
            if not canonical_photo_id:
                continue
            for face in list(image.get("faces", []) or []):
                source_face_id = str(face.get("face_id") or "").strip()
                if not source_face_id:
                    continue
                face_id = _canonical_face_id(task_id, source_face_id)
                person_id = str(face.get("person_id") or "").strip() or None
                face_rows.append(
                    FaceObservationRecord(
                        face_id=face_id,
                        user_id=user_id,
                        task_id=task_id,
                        dataset_id=dataset_id,
                        person_id=person_id,
                        photo_id=canonical_photo_id,
                        source_photo_id=source_photo_id,
                        source_hash=str(image.get("source_hash") or "") or None,
                        score=_safe_float(face.get("score")),
                        similarity=_safe_float(face.get("similarity")),
                        quality_score=_safe_float(face.get("quality_score")),
                        faiss_id=_safe_int(face.get("faiss_id")),
                        bbox_json=_json_clone(face.get("bbox")),
                        bbox_xywh_json=_json_clone(face.get("bbox_xywh")),
                        kps_json=_json_clone(face.get("kps")),
                        match_decision=str(face.get("match_decision") or "") or None,
                        match_reason=str(face.get("match_reason") or "") or None,
                        pose_yaw=_safe_float(face.get("pose_yaw")),
                        pose_pitch=_safe_float(face.get("pose_pitch")),
                        pose_roll=_safe_float(face.get("pose_roll")),
                        pose_bucket=str(face.get("pose_bucket") or "") or None,
                        eye_visibility_ratio=_safe_float(face.get("eye_visibility_ratio")),
                        crop_relative_path=None,
                        crop_url=_face_crop_url(face_id),
                        boxed_relative_path=_task_asset_relative_path(face.get("boxed_image_url") or image.get("boxed_image_url"), task_id),
                        boxed_url=_photo_variant_url(canonical_photo_id, "boxed"),
                        face_json={"source_face_id": source_face_id, **(_json_clone(face) or {})},
                        created_at=now,
                        updated_at=now,
                    )
                )
                person_face_link_rows.append(
                    PersonFaceLinkRecord(
                        id=_new_id(),
                        user_id=user_id,
                        task_id=task_id,
                        dataset_id=dataset_id,
                        person_id=person_id or "",
                        face_id=face_id,
                        photo_id=canonical_photo_id,
                        confidence=_safe_float(face.get("score")),
                        created_at=now,
                        updated_at=now,
                    )
                )
                registry_id = _new_id()
                registry_rows.append(
                    ObjectRegistryRecord(
                        id=registry_id,
                        user_id=user_id,
                        task_id=task_id,
                        dataset_id=dataset_id,
                        object_type="face",
                        semantic_id=face_id,
                        table_name="face_observations",
                        parent_id=person_registry.get(person_id or ""),
                        root_id=person_registry.get(person_id or ""),
                        metadata_json={"photo_id": canonical_photo_id},
                        created_at=now,
                    )
                )
                if person_id and person_id in person_registry:
                    link_rows.append(
                        ObjectLinkRecord(
                            id=_new_id(),
                            user_id=user_id,
                            from_id=person_registry[person_id],
                            relation_type="has_face",
                            to_id=registry_id,
                            weight=_safe_float(face.get("score")),
                            metadata_json={"task_id": task_id},
                            created_at=now,
                        )
                    )
                faiss_id = _safe_int(face.get("faiss_id"))
                vector = list(face_embedding_lookup(faiss_id) or []) if face_embedding_lookup and faiss_id is not None else []
                embedding_hash = None
                if vector:
                    embedding_hash = hashlib.sha256(
                        ",".join(f"{float(value):.8f}" for value in vector).encode("utf-8")
                    ).hexdigest()
                face_embedding_rows.append(
                    FaceEmbeddingRecord(
                        id=_new_id(),
                        face_id=face_id,
                        user_id=user_id,
                        task_id=task_id,
                        dataset_id=dataset_id,
                        faiss_id=faiss_id,
                        embedding=vector or None,
                        embedding_dim=len(vector) if vector else 512 if faiss_id is not None else None,
                        embedding_model="Facenet512" if faiss_id is not None else None,
                        embedding_version=stage_versions.get("face_version"),
                        embedding_hash=embedding_hash,
                        source_backend="faiss_local" if faiss_id is not None else None,
                        created_at=now,
                        updated_at=now,
                    )
                )

        observations = list(memory.get("vp1_observations", []) or []) if isinstance(memory, dict) else []
        for item in observations:
            if not isinstance(item, dict):
                continue
            source_photo_id = str(item.get("photo_id") or "").strip()
            canonical_photo_id = photo_id_map.get(source_photo_id)
            if not canonical_photo_id:
                continue
            observation_id = f"{task_id}:{source_photo_id}"
            analysis = item.get("vlm_analysis") if isinstance(item.get("vlm_analysis"), dict) else {}
            clues: list[dict] = []
            for clue_type, value in (
                ("key_object", analysis.get("key_objects")),
                ("ocr_hit", analysis.get("ocr_hits")),
                ("brand", analysis.get("brands")),
                ("place_candidate", analysis.get("place_candidates")),
                ("route_plan", analysis.get("route_plan_clues")),
                ("transport", analysis.get("transport_clues")),
                ("health_treatment", analysis.get("health_treatment_clues")),
                ("last_seen_object", analysis.get("object_last_seen_clues")),
            ):
                for clue in list(value or []):
                    if isinstance(clue, (dict, str)):
                        clues.append({"type": clue_type, "value": clue})
            vlm_rows.append(
                VLMObservationRevisionRecord(
                    id=observation_id,
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    photo_id=canonical_photo_id,
                    source_photo_id=source_photo_id,
                    summary=str(analysis.get("summary") or "") or None,
                    people_json=_json_clone(analysis.get("people")),
                    relations_json=_json_clone(analysis.get("relations")),
                    scene_json=_json_clone(analysis.get("scene")),
                    event_json=_json_clone(analysis.get("event")),
                    details_json=_json_clone(analysis.get("details")),
                    clues_json=_json_clone(clues),
                    raw_payload_json=_json_clone(item),
                    created_at=now,
                    updated_at=now,
                )
            )
            registry_rows.append(
                ObjectRegistryRecord(
                    id=_new_id(),
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    object_type="vlm_observation",
                    semantic_id=observation_id,
                    table_name="vlm_observation_revisions",
                    parent_id=None,
                    root_id=None,
                    metadata_json={"photo_id": canonical_photo_id},
                    created_at=now,
                )
            )
            for person_item in list(analysis.get("people", []) or []):
                vlm_people_rows.append(
                    VLMObservationPersonRecord(
                        id=_new_id(),
                        observation_id=observation_id,
                        user_id=user_id,
                        task_id=task_id,
                        photo_id=canonical_photo_id,
                        person_ref=str(person_item.get("name") or person_item.get("person_id") or "") if isinstance(person_item, dict) else None,
                        person_id=str(person_item.get("person_id") or "") if isinstance(person_item, dict) and person_item.get("person_id") else None,
                        raw_json=_json_clone(person_item if isinstance(person_item, dict) else {"value": person_item}),
                        created_at=now,
                    )
                )
            for relation_item in list(analysis.get("relations", []) or []):
                vlm_relation_rows.append(
                    VLMObservationRelationRecord(
                        id=_new_id(),
                        observation_id=observation_id,
                        user_id=user_id,
                        task_id=task_id,
                        photo_id=canonical_photo_id,
                        raw_json=_json_clone(relation_item if isinstance(relation_item, dict) else {"value": relation_item}),
                        created_at=now,
                    )
                )
            for clue in clues:
                vlm_clue_rows.append(
                    VLMObservationClueRecord(
                        id=_new_id(),
                        observation_id=observation_id,
                        user_id=user_id,
                        task_id=task_id,
                        photo_id=canonical_photo_id,
                        clue_type=str(clue.get("type") or "") or None,
                        clue_text=json.dumps(clue.get("value"), ensure_ascii=False) if not isinstance(clue.get("value"), str) else clue.get("value"),
                        raw_json=_json_clone(clue),
                        created_at=now,
                    )
                )

        lp1_events = list(memory.get("lp1_events", []) or []) if isinstance(memory, dict) else []
        for event in lp1_events:
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("event_id") or "").strip()
            if not event_id:
                continue
            event_root_id = _new_id()
            event_revision_id = _new_id()
            event_root_rows.append(
                EventRootRecord(
                    id=event_root_id,
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    event_id=event_id,
                    current_revision_no=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            event_revision_rows.append(
                EventRevisionRecord(
                    id=event_revision_id,
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    event_id=event_id,
                    revision_no=1,
                    title=str(event.get("title") or "") or None,
                    date=str(event.get("date") or "") or None,
                    time_range=_json_clone(event.get("time_range")),
                    duration=_json_clone(event.get("duration")),
                    type=str(event.get("type") or "") or None,
                    location=_json_clone(event.get("location")),
                    description=str(event.get("description") or "") or None,
                    photo_count=_safe_int(event.get("photo_count")),
                    confidence=_safe_float(event.get("confidence")),
                    reason=str(event.get("reason") or "") or None,
                    narrative=str(event.get("narrative") or "") or None,
                    narrative_synthesis=str(event.get("narrative_synthesis") or "") or None,
                    tags_json=_json_clone(event.get("tags")),
                    social_dynamics_json=_json_clone(event.get("social_dynamics")),
                    persona_evidence_json=_json_clone(event.get("persona_evidence")),
                    raw_json=_json_clone(event),
                    created_at=now,
                    updated_at=now,
                )
            )
            registry_rows.append(
                ObjectRegistryRecord(
                    id=event_root_id,
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    object_type="event",
                    semantic_id=event_id,
                    table_name="event_revisions",
                    parent_id=None,
                    root_id=event_root_id,
                    metadata_json={"revision_no": 1},
                    created_at=now,
                )
            )
            participants = list(event.get("participants", []) or event.get("participant_person_ids", []) or [])
            for participant in participants:
                if isinstance(participant, dict):
                    participant_id = str(participant.get("person_id") or participant.get("id") or "").strip()
                    participant_role = str(participant.get("role") or "") or None
                    confidence = _safe_float(participant.get("confidence"))
                else:
                    participant_id = str(participant or "").strip()
                    participant_role = None
                    confidence = None
                if not participant_id:
                    continue
                event_participant_rows.append(
                    EventParticipantRecord(
                        id=_new_id(),
                        event_revision_id=event_revision_id,
                        task_id=task_id,
                        person_id=participant_id,
                        participant_role=participant_role,
                        confidence=confidence,
                        created_at=now,
                    )
                )
            photo_ids = list(event.get("photo_ids", []) or event.get("supporting_photo_ids", []) or [])
            for order, source_photo_id in enumerate(photo_ids):
                canonical_photo_id = photo_id_map.get(str(source_photo_id))
                if not canonical_photo_id:
                    continue
                event_photo_rows.append(
                    EventPhotoLinkRecord(
                        id=_new_id(),
                        event_revision_id=event_revision_id,
                        task_id=task_id,
                        photo_id=canonical_photo_id,
                        source_photo_id=str(source_photo_id),
                        evidence_type="supporting",
                        sort_order=order,
                        created_at=now,
                    )
                )
            detail_fields = (
                ("objective_fact", event.get("objective_fact")),
                ("social_interaction", event.get("social_interaction")),
                ("social_dynamics", event.get("social_dynamics")),
                ("persona_evidence", event.get("persona_evidence")),
                ("lifestyle_tags", event.get("lifestyle_tags")),
                ("social_slices", event.get("social_slices")),
            )
            sort_key = 0
            for detail_type, value in detail_fields:
                if value in (None, "", [], {}):
                    continue
                if isinstance(value, list):
                    items = value
                else:
                    items = [value]
                for item in items:
                    event_detail_rows.append(
                        EventDetailUnitRecord(
                            id=_new_id(),
                            event_revision_id=event_revision_id,
                            task_id=task_id,
                            detail_type=detail_type,
                            detail_text=item if isinstance(item, str) else json.dumps(item, ensure_ascii=False),
                            normalized_text=item if isinstance(item, str) else None,
                            source_refs_json=None,
                            confidence=None,
                            sort_key=sort_key,
                            created_at=now,
                        )
                    )
                    sort_key += 1

        profile_payload = dict(memory.get("lp3_profile", {}) or {}) if isinstance(memory, dict) else {}
        relationship_items = list(memory.get("lp2_relationships", []) or []) if isinstance(memory, dict) else []
        relationship_dossiers = list(((profile_payload.get("internal_artifacts") or {}).get("relationship_dossiers") or []) or [])
        dossier_map = {
            str(item.get("person_id") or ""): item
            for item in relationship_dossiers
            if isinstance(item, dict)
        }
        for item in relationship_items:
            if not isinstance(item, dict):
                continue
            person_id = str(item.get("person_id") or "").strip()
            if not person_id:
                continue
            relationship_id = str(item.get("relationship_id") or _canonical_relationship_id(task_id, person_id))
            relationship_root_id = _new_id()
            relationship_revision_id = _new_id()
            relationship_root_rows.append(
                RelationshipRootRecord(
                    id=relationship_root_id,
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    relationship_id=relationship_id,
                    person_id=person_id,
                    current_revision_no=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            relationship_revision_rows.append(
                RelationshipRevisionRecord(
                    id=relationship_revision_id,
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    relationship_id=relationship_id,
                    revision_no=1,
                    person_id=person_id,
                    relationship_type=str(item.get("relationship_type") or "") or None,
                    intimacy_score=_safe_float(item.get("intimacy_score")),
                    status=str(item.get("status") or "") or None,
                    confidence=_safe_float(item.get("confidence")),
                    reasoning=str(item.get("reasoning") or item.get("reason") or "") or None,
                    evidence_json=_json_clone(item.get("evidence")),
                    raw_json=_json_clone(item),
                    created_at=now,
                    updated_at=now,
                )
            )
            registry_rows.append(
                ObjectRegistryRecord(
                    id=relationship_root_id,
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    object_type="relationship",
                    semantic_id=relationship_id,
                    table_name="relationship_revisions",
                    parent_id=person_registry.get(person_id),
                    root_id=relationship_root_id,
                    metadata_json={"person_id": person_id},
                    created_at=now,
                )
            )
            dossier = dossier_map.get(person_id)
            if dossier:
                relationship_dossier_rows.append(
                    RelationshipDossierRevisionRecord(
                        id=_new_id(),
                        user_id=user_id,
                        task_id=task_id,
                        dataset_id=dataset_id,
                        relationship_id=relationship_id,
                        revision_no=1,
                        person_id=person_id,
                        dossier_json=_json_clone(dossier),
                        created_at=now,
                        updated_at=now,
                    )
                )
            for shared_event in list(item.get("shared_events", []) or []):
                if isinstance(shared_event, dict):
                    shared_event_id = str(shared_event.get("event_id") or "").strip() or None
                    date_snapshot = str(shared_event.get("date") or "") or None
                    narrative_snapshot = str(shared_event.get("narrative") or "") or None
                else:
                    shared_event_id = str(shared_event or "").strip() or None
                    date_snapshot = None
                    narrative_snapshot = None
                relationship_shared_event_rows.append(
                    RelationshipSharedEventRecord(
                        id=_new_id(),
                        relationship_revision_id=relationship_revision_id,
                        task_id=task_id,
                        event_id=shared_event_id,
                        date_snapshot=date_snapshot,
                        narrative_snapshot=narrative_snapshot,
                        created_at=now,
                    )
                )

        group_items = list(((profile_payload.get("internal_artifacts") or {}).get("group_artifacts") or []) or [])
        for item in group_items:
            if not isinstance(item, dict):
                continue
            group_id = str(item.get("group_id") or f"{task_id}:group:{len(group_root_rows)+1}").strip()
            group_root_id = _new_id()
            group_revision_id = _new_id()
            members = list(item.get("members", []) or [])
            group_root_rows.append(
                GroupRootRecord(
                    id=group_root_id,
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    group_id=group_id,
                    current_revision_no=1,
                    created_at=now,
                    updated_at=now,
                )
            )
            group_revision_rows.append(
                GroupRevisionRecord(
                    id=group_revision_id,
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    group_id=group_id,
                    revision_no=1,
                    group_type_candidate=str(item.get("group_type_candidate") or "") or None,
                    confidence=_safe_float(item.get("confidence")),
                    reason=str(item.get("reason") or "") or None,
                    strong_evidence_refs_json=_json_clone(item.get("strong_evidence_refs")),
                    raw_json=_json_clone(item),
                    created_at=now,
                    updated_at=now,
                )
            )
            for member in members:
                group_member_rows.append(
                    GroupMemberRecord(
                        id=_new_id(),
                        group_revision_id=group_revision_id,
                        task_id=task_id,
                        person_id=str(member or "").strip(),
                        weight=None,
                        created_at=now,
                    )
                )

        if profile_payload:
            profile_revision_id = _canonical_profile_revision_id(task_id)
            primary_person = str(
                profile_payload.get("primary_person_id")
                or ((profile_payload.get("internal_artifacts") or {}).get("primary_decision") or {}).get("primary_person_id")
                or primary_person_id
                or ""
            ).strip() or None
            profile_context_rows.append(
                ProfileContextRevisionRecord(
                    id=_new_id(),
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    primary_person_id=primary_person,
                    payload_json={
                        "events": [item.get("event_id") for item in lp1_events if isinstance(item, dict)],
                        "relationships": [item.get("person_id") for item in relationship_items if isinstance(item, dict)],
                        "groups": [item.get("group_id") for item in group_items if isinstance(item, dict)],
                        "vlm_observations": [f"{task_id}:{item.get('photo_id')}" for item in observations if isinstance(item, dict)],
                    },
                    created_at=now,
                    updated_at=now,
                )
            )
            profile_revision_rows.append(
                ProfileRevisionRecord(
                    id=profile_revision_id,
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    profile_revision_id=profile_revision_id,
                    primary_person_id=primary_person,
                    structured_json=_json_clone(profile_payload.get("structured")),
                    report_markdown=str(profile_payload.get("report_markdown") or profile_payload.get("report") or "") or None,
                    summary=str(profile_payload.get("summary") or "") or None,
                    consistency_json=_json_clone(profile_payload.get("consistency")),
                    debug_json=_json_clone(profile_payload.get("debug")),
                    internal_artifacts_json=_json_clone(profile_payload.get("internal_artifacts")),
                    field_decisions_json=_json_clone(((profile_payload.get("internal_artifacts") or {}).get("profile_fact_decisions"))),
                    created_at=now,
                    updated_at=now,
                )
            )
            registry_rows.append(
                ObjectRegistryRecord(
                    id=_new_id(),
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    object_type="profile",
                    semantic_id=profile_revision_id,
                    table_name="profile_revisions",
                    parent_id=person_registry.get(primary_person or ""),
                    root_id=None,
                    metadata_json={"primary_person_id": primary_person},
                    created_at=now,
                )
            )
            structured = profile_payload.get("structured")
            for field_key, tag in _flatten_profile_fields(structured):
                if not isinstance(tag, dict):
                    continue
                profile_field_rows.append(
                    ProfileFieldValueRecord(
                        id=_new_id(),
                        profile_revision_id=profile_revision_id,
                        task_id=task_id,
                        field_key=field_key,
                        domain_name=field_key.split(".", 1)[0] if "." in field_key else field_key,
                        batch_name=None,
                        value_json=_json_clone(tag.get("value")),
                        confidence=_safe_float(tag.get("confidence")),
                        reasoning=str(tag.get("reasoning") or "") or None,
                        traceable_evidence_json=_json_clone(tag.get("evidence")),
                        null_reason=str(tag.get("null_reason") or "") or None,
                        created_at=now,
                        updated_at=now,
                    )
                )
            for decision in list(((profile_payload.get("internal_artifacts") or {}).get("profile_fact_decisions") or []) or []):
                if not isinstance(decision, dict):
                    continue
                profile_fact_rows.append(
                    ProfileFactDecisionRecord(
                        id=_new_id(),
                        profile_revision_id=profile_revision_id,
                        task_id=task_id,
                        field_key=str(decision.get("field_key") or "") or "",
                        payload_json=_json_clone(decision),
                        created_at=now,
                        updated_at=now,
                    )
                )
            consistency_rows.append(
                ConsistencyCheckRevisionRecord(
                    id=_new_id(),
                    profile_revision_id=profile_revision_id,
                    task_id=task_id,
                    check_name="lp3_profile_consistency",
                    status=str(((profile_payload.get("consistency") or {}).get("summary") or {}).get("status") or "available"),
                    details_json=_json_clone(profile_payload.get("consistency")),
                    created_at=now,
                    updated_at=now,
                )
            )

        stage_payloads: dict[str, dict] = {
            "ingest": {
                "uploads": _json_clone(uploads),
                "asset_manifest": _json_clone(manifest),
            },
            "face": _json_clone(face_payload) if face_payload else {},
            "vlm": {"vp1_observations": _json_clone(observations)},
            "lp1": {
                "lp1_events": _json_clone(memory.get("lp1_events")),
                "lp1_events_raw": _json_clone(memory.get("lp1_events_raw")),
                "lp1_batches": _json_clone(memory.get("lp1_batches")),
                "lp1_event_continuation_log": _json_clone(memory.get("lp1_event_continuation_log")),
            } if isinstance(memory, dict) else {},
            "lp2": {
                "lp2_relationships": _json_clone(relationship_items),
                "relationship_dossiers": _json_clone(relationship_dossiers),
            },
            "lp3": _json_clone(profile_payload) if profile_payload else {},
        }

        for stage_name in STAGE_NAMES:
            payload = stage_payloads.get(stage_name) or {}
            if not payload:
                continue
            stage_rows.append(
                TaskStageRecord(
                    id=_new_id(),
                    user_id=user_id,
                    task_id=task_id,
                    dataset_id=dataset_id,
                    stage_name=stage_name,
                    stage_version=(
                        pipeline_version
                        if stage_name == "ingest"
                        else stage_versions.get(
                            {
                                "face": "face_version",
                                "vlm": "vlm_version",
                                "lp1": "lp1_version",
                                "lp2": "lp2_version",
                                "lp3": "lp3_version",
                            }.get(stage_name, "pipeline_version")
                        )
                    ),
                    stage_channel=pipeline_channel if stage_name == "ingest" else None,
                    status="completed" if payload else "missing",
                    summary_json={
                        "task_id": task_id,
                        "stage_name": stage_name,
                    },
                    raw_payload_json=_json_clone(payload),
                    normalized_payload_json=_json_clone(payload),
                    artifact_manifest_json=_artifact_manifest_for_stage(
                        manifest,
                        _stage_artifact_prefixes(stage_name),
                    ),
                    created_at=now,
                    updated_at=now,
                )
            )

        with SessionLocal() as session:
            task_record = session.get(TaskRecord, task_id)
            if task_record is not None:
                task_record.dataset_id = dataset_id
                task_record.dataset_fingerprint = str(task.get("dataset_fingerprint") or task_record.dataset_fingerprint or "")
                task_record.pipeline_version = pipeline_version
                task_record.pipeline_channel = pipeline_channel
                task_record.face_version = stage_versions.get("face_version")
                task_record.vlm_version = stage_versions.get("vlm_version")
                task_record.lp1_version = stage_versions.get("lp1_version")
                task_record.lp2_version = stage_versions.get("lp2_version")
                task_record.lp3_version = stage_versions.get("lp3_version")
                task_record.judge_version = stage_versions.get("judge_version")
                session.add(task_record)

            self._delete_existing_task_rows(session, task_id)

            for row in (
                binary_asset_rows
                + photo_rows
                + photo_exif_rows
                + photo_asset_rows
                + task_photo_item_rows
                + person_rows
                + person_revision_rows
                + face_rows
                + face_embedding_rows
                + person_face_link_rows
                + vlm_rows
                + vlm_people_rows
                + vlm_relation_rows
                + vlm_clue_rows
                + event_root_rows
                + event_revision_rows
                + event_participant_rows
                + event_photo_rows
                + event_detail_rows
                + relationship_root_rows
                + relationship_dossier_rows
                + relationship_revision_rows
                + relationship_shared_event_rows
                + group_root_rows
                + group_revision_rows
                + group_member_rows
                + profile_context_rows
                + profile_revision_rows
                + profile_field_rows
                + profile_fact_rows
                + consistency_rows
                + stage_rows
                + registry_rows
                + link_rows
            ):
                session.add(row)

            user_head = session.get(UserHeadRecord, user_id)
            if user_head is None:
                user_head = UserHeadRecord(
                    user_id=user_id,
                    active_dataset_id=dataset_id,
                    active_task_id=task_id,
                    active_profile_revision_id=profile_revision_rows[0].id if profile_revision_rows else None,
                    active_gt_revision_id=None,
                    updated_at=now,
                )
            else:
                user_head.active_dataset_id = dataset_id
                user_head.active_task_id = task_id
                user_head.active_profile_revision_id = profile_revision_rows[0].id if profile_revision_rows else user_head.active_profile_revision_id
                user_head.updated_at = now
            session.add(user_head)
            session.commit()

    def crop_face_bytes(self, user_id: str, face_id: str) -> tuple[bytes, str] | None:
        with SessionLocal() as session:
            face = session.get(FaceObservationRecord, face_id)
            if face is None or face.user_id != user_id:
                return None
            photo = session.get(PhotoRecord, face.photo_id)
            if photo is None:
                return None
            task_record = session.get(TaskRecord, photo.task_id)
            task_dir_value = str(task_record.task_dir) if task_record is not None else ""
            raw_relative_path = photo.raw_relative_path
            bbox = copy.deepcopy(face.bbox_xywh_json) if isinstance(face.bbox_xywh_json, dict) else None
        if not raw_relative_path:
            return None
        task_dir = Path(task_dir_value)
        local_path = task_dir / raw_relative_path
        if not local_path.exists():
            return None
        if not bbox:
            return None
        try:
            with Image.open(local_path) as image:
                normalized = ImageOps.exif_transpose(image)
                x = max(0, int(bbox.get("x") or 0))
                y = max(0, int(bbox.get("y") or 0))
                w = max(1, int(bbox.get("w") or 1))
                h = max(1, int(bbox.get("h") or 1))
                crop = normalized.crop((x, y, x + w, y + h))
                buffer = io.BytesIO()
                crop.save(buffer, format="WEBP", quality=90, method=6)
                return buffer.getvalue(), "image/webp"
        except Exception:
            return None
