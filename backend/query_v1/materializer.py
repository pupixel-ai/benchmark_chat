"""On-demand materialization for v0325 / v0327-db query v1."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from backend.query_v1.indexers import MilvusQueryIndexer, Neo4jQueryIndexer
from backend.query_v1.store import MaterializedBundle, QueryStore
from config import TASK_VERSION_V0325, TASK_VERSION_V0327_DB, TASK_VERSION_V0327_DB_QUERY, TASK_VERSION_V0327_EXP
from utils import load_json


SCHEMA_VERSION = "query_v1"
SUPPORTED_QUERY_V1_VERSIONS = {TASK_VERSION_V0325, TASK_VERSION_V0327_EXP, TASK_VERSION_V0327_DB, TASK_VERSION_V0327_DB_QUERY}


def materialize_v0325_to_query_store(task: Dict[str, Any], user_id: str, store: Optional[QueryStore] = None) -> Dict[str, Any]:
    """Materialize a v0325/v0327 memory snapshot into canonical query-store rows."""

    query_store = store or QueryStore()
    version = str(task.get("version") or "").strip()
    if version not in SUPPORTED_QUERY_V1_VERSIONS:
        raise ValueError(f"query v1 materializer does not support version={version}")

    source_updated_at = _parse_dt(task.get("updated_at"))
    latest = query_store.latest_materialization(
        user_id=user_id,
        source_task_id=str(task.get("task_id") or ""),
        schema_version=SCHEMA_VERSION,
    )
    if latest and latest.get("source_updated_at") == (source_updated_at.isoformat() if source_updated_at else None):
        return latest

    pipeline_family = "v0325"
    task_id = str(task.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("task_id is required for query v1 materialization")

    memory = dict(((task.get("result") or {}).get("memory")) or {})
    profile = dict(memory.get("lp3_profile") or {})
    face_result = dict((task.get("result") or {}).get("face_recognition") or {})
    family_dir = Path(str(task.get("task_dir") or "")).expanduser() / pipeline_family
    relationship_dossiers = _load_named_payload(profile.get("relationship_dossiers"), family_dir / "relationship_dossiers.json")
    group_artifacts = _load_named_payload(profile.get("group_artifacts"), family_dir / "group_artifacts.json")
    profile_fact_decisions = _load_named_payload(
        profile.get("profile_fact_decisions") or profile.get("field_decisions"),
        family_dir / "profile_fact_decisions.json",
    )
    structured_profile = dict(profile.get("structured") or _load_json_dict(family_dir / "structured_profile.json") or {})

    materialization_id = _stable_id("mat", task_id, task.get("updated_at") or "", SCHEMA_VERSION)
    photo_rows = _build_photo_rows(task, user_id=user_id, task_id=task_id, pipeline_family=pipeline_family, materialization_id=materialization_id)
    photo_by_id = {row["photo_id"]: row for row in photo_rows}

    lp1_events = list(memory.get("lp1_events", []) or _load_json_list(family_dir / "lp1_events_compact.json"))
    lp2_relationships = list(memory.get("lp2_relationships", []) or _load_json_list(family_dir / "lp2_relationships.json"))
    vp1_observations = list(memory.get("vp1_observations", []) or _load_json_list(family_dir / "vp1_observations.json"))

    event_rows, event_photo_rows, event_people_rows, event_place_rows = _build_event_rows(
        lp1_events=lp1_events,
        user_id=user_id,
        task_id=task_id,
        pipeline_family=pipeline_family,
        materialization_id=materialization_id,
        photo_by_id=photo_by_id,
    )
    event_by_id = {row["event_id"]: row for row in event_rows}
    photo_to_events = _photo_to_events(event_photo_rows)
    relationship_rows, relationship_support_rows = _build_relationship_rows(
        lp2_relationships=lp2_relationships,
        relationship_dossiers=relationship_dossiers,
        user_id=user_id,
        task_id=task_id,
        pipeline_family=pipeline_family,
        materialization_id=materialization_id,
    )
    evidence_rows = _build_evidence_rows(
        vp1_observations=vp1_observations,
        lp1_events=lp1_events,
        relationships=relationship_rows,
        relationship_support_rows=relationship_support_rows,
        photo_to_events=photo_to_events,
        user_id=user_id,
        task_id=task_id,
        pipeline_family=pipeline_family,
        materialization_id=materialization_id,
    )
    group_rows, group_member_rows = _build_group_rows(
        groups=group_artifacts,
        user_id=user_id,
        task_id=task_id,
        pipeline_family=pipeline_family,
        materialization_id=materialization_id,
    )
    profile_fact_rows = _build_profile_fact_rows(
        structured_profile=structured_profile,
        profile_fact_decisions=profile_fact_decisions,
        user_id=user_id,
        task_id=task_id,
        pipeline_family=pipeline_family,
        materialization_id=materialization_id,
    )

    event_views = _build_event_views(
        events=event_rows,
        event_photos=event_photo_rows,
        event_people=event_people_rows,
        event_places=event_place_rows,
        evidence_rows=evidence_rows,
        relationships=relationship_rows,
        relationship_support=relationship_support_rows,
    )
    evidence_docs = _build_evidence_docs(evidence_rows, event_people_rows, event_places_rows=event_place_rows)
    graph_payload = _build_graph_payload(
        primary_person_id=str(face_result.get("primary_person_id") or "") or None,
        events=event_rows,
        event_people=event_people_rows,
        event_places=event_place_rows,
        relationships=relationship_rows,
        relationship_support=relationship_support_rows,
        groups=group_rows,
        group_members=group_member_rows,
    )

    initial_materialization = {
        "materialization_id": materialization_id,
        "user_id": user_id,
        "source_task_id": task_id,
        "pipeline_family": pipeline_family,
        "schema_version": SCHEMA_VERSION,
        "status": "materialized",
        "source_updated_at": source_updated_at,
        "milvus_status": {"status": "pending"},
        "neo4j_status": {"status": "pending"},
        "error_summary": None,
    }
    query_store.replace_scope(
        MaterializedBundle(
            materialization=initial_materialization,
            photos=photo_rows,
            events=event_rows,
            event_photos=event_photo_rows,
            event_people=event_people_rows,
            event_places=event_place_rows,
            evidence=evidence_rows,
            relationships=relationship_rows,
            relationship_support=relationship_support_rows,
            groups=group_rows,
            group_members=group_member_rows,
            profile_facts=profile_fact_rows,
        )
    )

    milvus_status = MilvusQueryIndexer().publish(event_views=event_views, evidence_docs=evidence_docs)
    neo4j_status = Neo4jQueryIndexer().publish(graph_payload)
    status = "materialized"
    error_summary = None
    if str(milvus_status.get("status")) == "failed" or str(neo4j_status.get("status")) == "failed":
        status = "partial_failure"
        reasons = [
            item.get("reason")
            for item in (milvus_status, neo4j_status)
            if isinstance(item, dict) and item.get("status") == "failed" and item.get("reason")
        ]
        error_summary = "; ".join(str(item) for item in reasons if item)

    updated = query_store.update_materialization_status(
        materialization_id=materialization_id,
        status=status,
        milvus_status=milvus_status,
        neo4j_status=neo4j_status,
        error_summary=error_summary,
    )
    return updated or initial_materialization


def _load_named_payload(inline_payload: Any, fallback_path: Path) -> List[Dict[str, Any]]:
    if isinstance(inline_payload, list):
        return [dict(item) for item in inline_payload if isinstance(item, dict)]
    return _load_json_list(fallback_path)


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = load_json(str(path))
    except Exception:
        return []
    if isinstance(payload, dict):
        if "relationships" in payload and isinstance(payload["relationships"], list):
            return [dict(item) for item in payload["relationships"] if isinstance(item, dict)]
        if "group_artifacts" in payload and isinstance(payload["group_artifacts"], list):
            return [dict(item) for item in payload["group_artifacts"] if isinstance(item, dict)]
        return []
    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict)]
    return []


def _load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = load_json(str(path))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _parse_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _stable_id(prefix: str, *parts: Any) -> str:
    joined = "||".join(str(part or "") for part in parts)
    return f"{prefix}_{hashlib.sha1(joined.encode('utf-8')).hexdigest()[:24]}"


def _build_photo_rows(
    task: Dict[str, Any],
    *,
    user_id: str,
    task_id: str,
    pipeline_family: str,
    materialization_id: str,
) -> List[Dict[str, Any]]:
    uploads = list(task.get("uploads") or [])
    images = list((((task.get("result") or {}).get("face_recognition") or {}).get("images")) or [])
    by_photo_id: Dict[str, Dict[str, Any]] = {}
    uploads_by_image_id = {
        str(item.get("image_id") or ""): item
        for item in uploads
        if isinstance(item, dict) and str(item.get("image_id") or "").strip()
    }
    uploads_by_hash = {
        str(item.get("source_hash") or ""): item
        for item in uploads
        if isinstance(item, dict) and str(item.get("source_hash") or "").strip()
    }
    for upload in uploads:
        if not isinstance(upload, dict):
            continue
        photo_id = str(upload.get("image_id") or upload.get("source_hash") or "").strip()
        if not photo_id:
            continue
        by_photo_id[photo_id] = {
            "photo_id": photo_id,
            "user_id": user_id,
            "source_task_id": task_id,
            "pipeline_family": pipeline_family,
            "materialization_id": materialization_id,
            "object_key": upload.get("path"),
            "asset_url": upload.get("url") or upload.get("preview_url"),
            "captured_at": upload.get("timestamp"),
            "content_type": upload.get("content_type"),
            "width": upload.get("width"),
            "height": upload.get("height"),
            "photo_payload": copy.deepcopy(upload),
        }
    for image in images:
        if not isinstance(image, dict):
            continue
        photo_id = str(image.get("image_id") or image.get("source_hash") or "").strip()
        if not photo_id:
            continue
        upload = uploads_by_image_id.get(str(image.get("image_id") or "")) or uploads_by_hash.get(str(image.get("source_hash") or "")) or {}
        payload = by_photo_id.setdefault(
            photo_id,
            {
                "photo_id": photo_id,
                "user_id": user_id,
                "source_task_id": task_id,
                "pipeline_family": pipeline_family,
                "materialization_id": materialization_id,
                "object_key": upload.get("path"),
                "asset_url": upload.get("url") or upload.get("preview_url") or image.get("original_image_url"),
                "captured_at": image.get("timestamp") or upload.get("timestamp"),
                "content_type": upload.get("content_type"),
                "width": image.get("width") or upload.get("width"),
                "height": image.get("height") or upload.get("height"),
                "photo_payload": copy.deepcopy(image),
            },
        )
        if isinstance(payload.get("photo_payload"), dict):
            payload["photo_payload"] = {**payload["photo_payload"], **copy.deepcopy(image)}
    return list(by_photo_id.values())


def _build_event_rows(
    *,
    lp1_events: Sequence[Dict[str, Any]],
    user_id: str,
    task_id: str,
    pipeline_family: str,
    materialization_id: str,
    photo_by_id: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    events: List[Dict[str, Any]] = []
    event_photos: List[Dict[str, Any]] = []
    event_people: List[Dict[str, Any]] = []
    event_places: List[Dict[str, Any]] = []
    for event in list(lp1_events or []):
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("event_id") or "").strip()
        if not event_id:
            continue
        cover_photo_id = str(event.get("anchor_photo_id") or "").strip() or None
        supporting_photo_ids = _unique(
            list(event.get("supporting_photo_ids", []) or [])
            or list(event.get("evidence_photos", []) or [])
            or ([cover_photo_id] if cover_photo_id else [])
        )
        if cover_photo_id and cover_photo_id not in supporting_photo_ids:
            supporting_photo_ids = [cover_photo_id, *supporting_photo_ids]
        if not cover_photo_id and supporting_photo_ids:
            cover_photo_id = supporting_photo_ids[0]
        events.append(
            {
                "event_id": event_id,
                "user_id": user_id,
                "source_task_id": task_id,
                "pipeline_family": pipeline_family,
                "materialization_id": materialization_id,
                "title": str(event.get("title") or ""),
                "summary": str(event.get("narrative_synthesis") or event.get("description") or event.get("title") or ""),
                "start_ts": str(event.get("started_at") or event.get("date") or ""),
                "end_ts": str(event.get("ended_at") or ""),
                "confidence": float(event.get("confidence") or 0.0),
                "cover_photo_id": cover_photo_id,
                "photo_count": len(supporting_photo_ids),
                "status": "active",
                "event_payload": copy.deepcopy(event),
            }
        )
        for index, photo_id in enumerate(supporting_photo_ids):
            normalized_photo_id = str(photo_id or "").strip()
            if not normalized_photo_id:
                continue
            event_photos.append(
                {
                    "row_id": _stable_id("evp", event_id, normalized_photo_id),
                    "user_id": user_id,
                    "source_task_id": task_id,
                    "pipeline_family": pipeline_family,
                    "materialization_id": materialization_id,
                    "event_id": event_id,
                    "photo_id": normalized_photo_id,
                    "is_cover": normalized_photo_id == cover_photo_id,
                    "support_strength": 1.0 if normalized_photo_id == cover_photo_id else 0.8,
                    "sort_order": index,
                }
            )
            photo_by_id.setdefault(
                normalized_photo_id,
                {
                    "photo_id": normalized_photo_id,
                    "user_id": user_id,
                    "source_task_id": task_id,
                    "pipeline_family": pipeline_family,
                    "materialization_id": materialization_id,
                    "object_key": None,
                    "asset_url": None,
                    "captured_at": None,
                    "content_type": None,
                    "width": None,
                    "height": None,
                    "photo_payload": {"photo_id": normalized_photo_id},
                },
            )
        participant_ids = _unique(list(event.get("participant_person_ids", []) or []) + list(event.get("participants", []) or []))
        depicted_ids = _unique(list(event.get("depicted_person_ids", []) or []))
        for person_id in participant_ids:
            event_people.append(
                {
                    "row_id": _stable_id("evperson", event_id, person_id, "participant"),
                    "user_id": user_id,
                    "source_task_id": task_id,
                    "pipeline_family": pipeline_family,
                    "materialization_id": materialization_id,
                    "event_id": event_id,
                    "person_id": person_id,
                    "role": "participant",
                    "weight": 1.0,
                }
            )
        for person_id in depicted_ids:
            if person_id in participant_ids:
                continue
            event_people.append(
                {
                    "row_id": _stable_id("evperson", event_id, person_id, "depicted"),
                    "user_id": user_id,
                    "source_task_id": task_id,
                    "pipeline_family": pipeline_family,
                    "materialization_id": materialization_id,
                    "event_id": event_id,
                    "person_id": person_id,
                    "role": "depicted",
                    "weight": 0.7,
                }
            )
        for place_ref in _unique(list(event.get("place_refs", []) or []) + [event.get("location")]):
            normalized = str(place_ref or "").strip()
            if not normalized:
                continue
            event_places.append(
                {
                    "row_id": _stable_id("evplace", event_id, normalized),
                    "user_id": user_id,
                    "source_task_id": task_id,
                    "pipeline_family": pipeline_family,
                    "materialization_id": materialization_id,
                    "event_id": event_id,
                    "place_ref": normalized,
                    "normalized_place": normalized.lower(),
                    "weight": 1.0,
                }
            )
    return events, event_photos, event_people, event_places


def _photo_to_events(event_photo_rows: Sequence[Dict[str, Any]]) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for row in list(event_photo_rows or []):
        photo_id = str(row.get("photo_id") or "").strip()
        event_id = str(row.get("event_id") or "").strip()
        if not photo_id or not event_id:
            continue
        mapping.setdefault(photo_id, []).append(event_id)
    return mapping


def _build_relationship_rows(
    *,
    lp2_relationships: Sequence[Dict[str, Any]],
    relationship_dossiers: Sequence[Dict[str, Any]],
    user_id: str,
    task_id: str,
    pipeline_family: str,
    materialization_id: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dossiers_by_person = {
        str(item.get("person_id") or "").strip(): dict(item)
        for item in list(relationship_dossiers or [])
        if isinstance(item, dict) and str(item.get("person_id") or "").strip()
    }
    relationships: List[Dict[str, Any]] = []
    supports: List[Dict[str, Any]] = []
    for relationship in list(lp2_relationships or []):
        if not isinstance(relationship, dict):
            continue
        person_id = str(relationship.get("person_id") or "").strip()
        if not person_id:
            continue
        relationship_id = str(relationship.get("relationship_id") or f"REL_{person_id}")
        dossier = dossiers_by_person.get(person_id, {})
        supporting_event_ids = _unique(list(relationship.get("supporting_event_ids", []) or []) + [item.get("event_id") for item in list(relationship.get("shared_events", []) or []) if isinstance(item, dict)])
        supporting_photo_ids = _unique(
            list(relationship.get("supporting_photo_ids", []) or [])
            or list(((relationship.get("evidence") or {}).get("photo_ids")) or [])
        )
        photo_count = int(dossier.get("photo_count") or len(supporting_photo_ids) or 0)
        shared_event_count = len(supporting_event_ids)
        monthly_frequency = _safe_float(dossier.get("monthly_frequency"))
        recent_gap_days = _safe_int(dossier.get("recent_gap_days"))
        relationships.append(
            {
                "relationship_id": relationship_id,
                "user_id": user_id,
                "source_task_id": task_id,
                "pipeline_family": pipeline_family,
                "materialization_id": materialization_id,
                "person_id": person_id,
                "relationship_type": str(relationship.get("relationship_type") or ""),
                "status": str(relationship.get("status") or ""),
                "confidence": _safe_float(relationship.get("confidence")),
                "intimacy_score": _safe_float(relationship.get("intimacy_score")),
                "photo_count": photo_count,
                "shared_event_count": shared_event_count,
                "monthly_frequency": monthly_frequency,
                "recent_gap_days": recent_gap_days,
                "reasoning": str(relationship.get("reasoning") or relationship.get("reason") or ""),
                "relationship_payload": {
                    **copy.deepcopy(relationship),
                    "relationship_dossier": copy.deepcopy(dossier) if dossier else None,
                },
            }
        )
        for event_id in supporting_event_ids:
            supports.append(
                {
                    "row_id": _stable_id("relsupport", relationship_id, event_id, "event"),
                    "user_id": user_id,
                    "source_task_id": task_id,
                    "pipeline_family": pipeline_family,
                    "materialization_id": materialization_id,
                    "relationship_id": relationship_id,
                    "event_id": str(event_id),
                    "photo_id": None,
                    "support_type": "event",
                    "support_strength": 1.0,
                }
            )
        for photo_id in supporting_photo_ids:
            supports.append(
                {
                    "row_id": _stable_id("relsupport", relationship_id, photo_id, "photo"),
                    "user_id": user_id,
                    "source_task_id": task_id,
                    "pipeline_family": pipeline_family,
                    "materialization_id": materialization_id,
                    "relationship_id": relationship_id,
                    "event_id": None,
                    "photo_id": str(photo_id),
                    "support_type": "photo",
                    "support_strength": 0.8,
                }
            )
    return relationships, supports


def _build_evidence_rows(
    *,
    vp1_observations: Sequence[Dict[str, Any]],
    lp1_events: Sequence[Dict[str, Any]],
    relationships: Sequence[Dict[str, Any]],
    relationship_support_rows: Sequence[Dict[str, Any]],
    photo_to_events: Dict[str, List[str]],
    user_id: str,
    task_id: str,
    pipeline_family: str,
    materialization_id: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for event in list(lp1_events or []):
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("event_id") or "").strip()
        if not event_id:
            continue
        cover_photo_id = str(event.get("anchor_photo_id") or ((event.get("supporting_photo_ids") or [None])[0] or "")).strip() or None
        tags = [str(item).strip() for item in list(event.get("tags", []) or []) if str(item).strip()]
        if tags:
            rows.append(
                _evidence_row(
                    evidence_id=_stable_id("evidence", event_id, "tags"),
                    user_id=user_id,
                    task_id=task_id,
                    pipeline_family=pipeline_family,
                    materialization_id=materialization_id,
                    event_id=event_id,
                    source_stage="lp1",
                    evidence_type="object",
                    photo_id=cover_photo_id,
                    text=", ".join(tags),
                    normalized_value=", ".join(tags).lower(),
                    confidence=event.get("confidence"),
                    payload={"tags": tags},
                )
            )
        objective_fact = dict(event.get("objective_fact") or {})
        scene_description = str(objective_fact.get("scene_description") or event.get("description") or "").strip()
        if scene_description:
            rows.append(
                _evidence_row(
                    evidence_id=_stable_id("evidence", event_id, "scene"),
                    user_id=user_id,
                    task_id=task_id,
                    pipeline_family=pipeline_family,
                    materialization_id=materialization_id,
                    event_id=event_id,
                    source_stage="lp1",
                    evidence_type="scene",
                    photo_id=cover_photo_id,
                    text=scene_description,
                    normalized_value=scene_description.lower(),
                    confidence=event.get("confidence"),
                    payload={"objective_fact": objective_fact},
                )
            )
        persona_evidence = dict(event.get("persona_evidence") or {})
        for bucket, values in persona_evidence.items():
            for index, value in enumerate(list(values or []), start=1):
                text = str(value or "").strip()
                if not text:
                    continue
                rows.append(
                    _evidence_row(
                        evidence_id=_stable_id("evidence", event_id, "persona", bucket, index),
                        user_id=user_id,
                        task_id=task_id,
                        pipeline_family=pipeline_family,
                        materialization_id=materialization_id,
                        event_id=event_id,
                        source_stage="lp1",
                        evidence_type="mood_style",
                        photo_id=cover_photo_id,
                        text=text,
                        normalized_value=text.lower(),
                        confidence=event.get("confidence"),
                        payload={"bucket": bucket},
                    )
                )
    for observation in list(vp1_observations or []):
        if not isinstance(observation, dict):
            continue
        photo_id = str(observation.get("photo_id") or "").strip()
        event_ids = photo_to_events.get(photo_id, [])
        if not photo_id or not event_ids:
            continue
        analysis = dict(observation.get("vlm_analysis") or observation)
        summary = str(analysis.get("summary") or "").strip()
        scene = dict(analysis.get("scene") or {})
        event_info = dict(analysis.get("event") or {})
        details = [str(item).strip() for item in list(analysis.get("details", []) or []) if str(item).strip()]
        key_objects = [str(item).strip() for item in list(analysis.get("key_objects", []) or []) if str(item).strip()]
        ocr_hits = [str(item).strip() for item in list(observation.get("ocr_hits", []) or analysis.get("ocr_hits", []) or []) if str(item).strip()]
        for event_id in event_ids:
            if summary:
                rows.append(
                    _evidence_row(
                        evidence_id=_stable_id("evidence", event_id, photo_id, "summary"),
                        user_id=user_id,
                        task_id=task_id,
                        pipeline_family=pipeline_family,
                        materialization_id=materialization_id,
                        event_id=event_id,
                        source_stage="vp1",
                        evidence_type="activity",
                        photo_id=photo_id,
                        text=summary,
                        normalized_value=summary.lower(),
                        confidence=observation.get("confidence") or 0.7,
                        payload={"analysis": analysis},
                    )
                )
            for value, evidence_type in (
                (scene.get("location_detected") or scene.get("environment_description"), "scene"),
                (event_info.get("activity"), "activity"),
                (event_info.get("interaction"), "interaction"),
            ):
                text = str(value or "").strip()
                if not text:
                    continue
                rows.append(
                    _evidence_row(
                        evidence_id=_stable_id("evidence", event_id, photo_id, evidence_type, text),
                        user_id=user_id,
                        task_id=task_id,
                        pipeline_family=pipeline_family,
                        materialization_id=materialization_id,
                        event_id=event_id,
                        source_stage="vp1",
                        evidence_type=evidence_type,
                        photo_id=photo_id,
                        text=text,
                        normalized_value=text.lower(),
                        confidence=observation.get("confidence") or 0.65,
                        payload={"photo_id": photo_id},
                    )
                )
            for index, value in enumerate(details[:12], start=1):
                rows.append(
                    _evidence_row(
                        evidence_id=_stable_id("evidence", event_id, photo_id, "detail", index),
                        user_id=user_id,
                        task_id=task_id,
                        pipeline_family=pipeline_family,
                        materialization_id=materialization_id,
                        event_id=event_id,
                        source_stage="vp1",
                        evidence_type="object",
                        photo_id=photo_id,
                        text=value,
                        normalized_value=value.lower(),
                        confidence=observation.get("confidence") or 0.55,
                        payload={"photo_id": photo_id},
                    )
                )
            for index, value in enumerate(key_objects[:12], start=1):
                rows.append(
                    _evidence_row(
                        evidence_id=_stable_id("evidence", event_id, photo_id, "key_object", index),
                        user_id=user_id,
                        task_id=task_id,
                        pipeline_family=pipeline_family,
                        materialization_id=materialization_id,
                        event_id=event_id,
                        source_stage="vp1",
                        evidence_type="object",
                        photo_id=photo_id,
                        text=value,
                        normalized_value=value.lower(),
                        confidence=observation.get("confidence") or 0.55,
                        payload={"photo_id": photo_id},
                    )
                )
            for index, value in enumerate(ocr_hits[:12], start=1):
                rows.append(
                    _evidence_row(
                        evidence_id=_stable_id("evidence", event_id, photo_id, "ocr", index),
                        user_id=user_id,
                        task_id=task_id,
                        pipeline_family=pipeline_family,
                        materialization_id=materialization_id,
                        event_id=event_id,
                        source_stage="vp1",
                        evidence_type="ocr",
                        photo_id=photo_id,
                        text=value,
                        normalized_value=value.lower(),
                        confidence=observation.get("confidence") or 0.5,
                        payload={"photo_id": photo_id},
                    )
                )
    support_by_relationship: Dict[str, List[str]] = {}
    for row in list(relationship_support_rows or []):
        if str(row.get("support_type") or "") != "event":
            continue
        relationship_id = str(row.get("relationship_id") or "").strip()
        event_id = str(row.get("event_id") or "").strip()
        if relationship_id and event_id:
            support_by_relationship.setdefault(relationship_id, []).append(event_id)
    for relationship in list(relationships or []):
        relationship_id = str(relationship.get("relationship_id") or "").strip()
        reasoning = str(relationship.get("reasoning") or "").strip()
        if not relationship_id or not reasoning:
            continue
        for event_id in support_by_relationship.get(relationship_id, []) or []:
            rows.append(
                _evidence_row(
                    evidence_id=_stable_id("evidence", relationship_id, event_id, "relationship_signal"),
                    user_id=user_id,
                    task_id=task_id,
                    pipeline_family=pipeline_family,
                    materialization_id=materialization_id,
                    event_id=event_id,
                    source_stage="lp2",
                    evidence_type="relationship_signal",
                    photo_id=None,
                    text=reasoning,
                    normalized_value=reasoning.lower(),
                    confidence=relationship.get("confidence"),
                    numeric_value=relationship.get("intimacy_score"),
                    numeric_unit="intimacy_score",
                    payload={
                        "relationship_id": relationship_id,
                        "person_id": relationship.get("person_id"),
                        "relationship_type": relationship.get("relationship_type"),
                    },
                )
            )
    return rows


def _evidence_row(
    *,
    evidence_id: str,
    user_id: str,
    task_id: str,
    pipeline_family: str,
    materialization_id: str,
    event_id: str,
    source_stage: str,
    evidence_type: str,
    photo_id: str | None,
    text: str,
    normalized_value: str,
    confidence: Any,
    payload: Dict[str, Any],
    numeric_value: Any = None,
    numeric_unit: str | None = None,
) -> Dict[str, Any]:
    return {
        "evidence_id": evidence_id,
        "user_id": user_id,
        "source_task_id": task_id,
        "pipeline_family": pipeline_family,
        "materialization_id": materialization_id,
        "event_id": event_id,
        "source_stage": source_stage,
        "evidence_type": evidence_type,
        "photo_id": photo_id,
        "text": text,
        "normalized_value": normalized_value,
        "numeric_value": _safe_float(numeric_value),
        "numeric_unit": numeric_unit,
        "confidence": _safe_float(confidence),
        "evidence_payload": copy.deepcopy(payload),
    }


def _build_group_rows(
    *,
    groups: Sequence[Dict[str, Any]],
    user_id: str,
    task_id: str,
    pipeline_family: str,
    materialization_id: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    group_rows: List[Dict[str, Any]] = []
    member_rows: List[Dict[str, Any]] = []
    for group in list(groups or []):
        if not isinstance(group, dict):
            continue
        group_id = str(group.get("group_id") or "").strip()
        if not group_id:
            continue
        group_rows.append(
            {
                "group_id": group_id,
                "user_id": user_id,
                "source_task_id": task_id,
                "pipeline_family": pipeline_family,
                "materialization_id": materialization_id,
                "group_type": str(group.get("group_type_candidate") or ""),
                "confidence": _safe_float(group.get("confidence")),
                "reason": str(group.get("reason") or ""),
                "group_payload": copy.deepcopy(group),
            }
        )
        for member in _unique(list(group.get("members", []) or [])):
            member_rows.append(
                {
                    "row_id": _stable_id("groupmember", group_id, member),
                    "user_id": user_id,
                    "source_task_id": task_id,
                    "pipeline_family": pipeline_family,
                    "materialization_id": materialization_id,
                    "group_id": group_id,
                    "person_id": str(member),
                }
            )
    return group_rows, member_rows


def _build_profile_fact_rows(
    *,
    structured_profile: Dict[str, Any],
    profile_fact_decisions: Sequence[Dict[str, Any]],
    user_id: str,
    task_id: str,
    pipeline_family: str,
    materialization_id: str,
) -> List[Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for field_key, value, confidence, payload in _flatten_structured_profile(structured_profile):
        fact_id = _stable_id("profilefact", task_id, field_key, "structured")
        rows[field_key] = {
            "fact_id": fact_id,
            "user_id": user_id,
            "source_task_id": task_id,
            "pipeline_family": pipeline_family,
            "materialization_id": materialization_id,
            "field_key": field_key,
            "value_json": {"value": value},
            "confidence": confidence,
            "source_level": "structured",
            "is_queryable_hint": value not in (None, "", [], {}),
            "fact_payload": copy.deepcopy(payload),
        }
    for decision in list(profile_fact_decisions or []):
        if not isinstance(decision, dict):
            continue
        field_key = str(decision.get("field_key") or "").strip()
        if not field_key:
            continue
        final = dict(decision.get("final") or {})
        value = final.get("value")
        rows[field_key] = {
            "fact_id": _stable_id("profilefact", task_id, field_key, "decision"),
            "user_id": user_id,
            "source_task_id": task_id,
            "pipeline_family": pipeline_family,
            "materialization_id": materialization_id,
            "field_key": field_key,
            "value_json": copy.deepcopy(final),
            "confidence": _safe_float(final.get("confidence")),
            "source_level": "decision",
            "is_queryable_hint": value not in (None, "", [], {}),
            "fact_payload": copy.deepcopy(decision),
        }
    return list(rows.values())


def _flatten_structured_profile(payload: Dict[str, Any], prefix: str = "") -> Iterable[Tuple[str, Any, float | None, Dict[str, Any]]]:
    for key, value in dict(payload or {}).items():
        field_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict) and "value" in value:
            yield field_key, value.get("value"), _safe_float(value.get("confidence")), value
            continue
        if isinstance(value, dict):
            yield from _flatten_structured_profile(value, field_key)


def _build_event_views(
    *,
    events: Sequence[Dict[str, Any]],
    event_photos: Sequence[Dict[str, Any]],
    event_people: Sequence[Dict[str, Any]],
    event_places: Sequence[Dict[str, Any]],
    evidence_rows: Sequence[Dict[str, Any]],
    relationships: Sequence[Dict[str, Any]],
    relationship_support: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    photos_by_event: Dict[str, List[str]] = {}
    for row in list(event_photos or []):
        photos_by_event.setdefault(str(row.get("event_id") or ""), []).append(str(row.get("photo_id") or ""))
    people_by_event: Dict[str, List[str]] = {}
    for row in list(event_people or []):
        people_by_event.setdefault(str(row.get("event_id") or ""), []).append(str(row.get("person_id") or ""))
    places_by_event: Dict[str, List[str]] = {}
    for row in list(event_places or []):
        places_by_event.setdefault(str(row.get("event_id") or ""), []).append(str(row.get("place_ref") or ""))
    evidence_by_event: Dict[str, List[Dict[str, Any]]] = {}
    for row in list(evidence_rows or []):
        evidence_by_event.setdefault(str(row.get("event_id") or ""), []).append(dict(row))
    rels_by_event: Dict[str, List[Dict[str, Any]]] = {}
    rel_lookup = {str(item.get("relationship_id") or ""): dict(item) for item in list(relationships or [])}
    for row in list(relationship_support or []):
        if str(row.get("support_type") or "") != "event":
            continue
        event_id = str(row.get("event_id") or "")
        relationship_id = str(row.get("relationship_id") or "")
        if event_id and relationship_id in rel_lookup:
            rels_by_event.setdefault(event_id, []).append(rel_lookup[relationship_id])
    docs: List[Dict[str, Any]] = []
    for event in list(events or []):
        event_id = str(event.get("event_id") or "")
        if not event_id:
            continue
        title = str(event.get("title") or "")
        summary = str(event.get("summary") or "")
        persons = _unique(people_by_event.get(event_id, []))
        places = _unique(places_by_event.get(event_id, []))
        event_evidence = evidence_by_event.get(event_id, [])
        tags = _unique(
            [
                str((item.get("evidence_payload") or {}).get("bucket") or item.get("evidence_type") or "")
                for item in event_evidence
                if isinstance(item, dict)
            ]
        )
        relationship_text = "; ".join(
            f"{item.get('person_id')} {item.get('relationship_type')} intimacy={item.get('intimacy_score')}"
            for item in rels_by_event.get(event_id, [])[:6]
        )
        grouped_text = {
            "summary": f"{title}\n{summary}".strip(),
            "people_relation": " ".join([title, " ".join(persons), relationship_text]).strip(),
            "time_place": " ".join(
                [
                    str(event.get("start_ts") or ""),
                    str(event.get("end_ts") or ""),
                    " ".join(places),
                ]
            ).strip(),
            "activity_scene": " ".join(
                [
                    str(item.get("text") or "")
                    for item in event_evidence
                    if str(item.get("evidence_type") or "") in {"scene", "activity", "interaction"}
                ][:12]
            ).strip(),
            "object_ocr": " ".join(
                [
                    str(item.get("text") or "")
                    for item in event_evidence
                    if str(item.get("evidence_type") or "") in {"object", "ocr", "brand"}
                ][:12]
            ).strip(),
            "mood_style": " ".join(
                [
                    str(item.get("text") or "")
                    for item in event_evidence
                    if str(item.get("evidence_type") or "") in {"mood_style", "relationship_signal"}
                ][:12]
            ).strip(),
        }
        for view_type, retrieval_text in grouped_text.items():
            if not retrieval_text:
                continue
            docs.append(
                {
                    "doc_id": f"{event_id}:{view_type}",
                    "user_id": event.get("user_id"),
                    "event_id": event_id,
                    "source_task_id": event.get("source_task_id"),
                    "view_type": view_type,
                    "retrieval_text": retrieval_text,
                    "start_ts": event.get("start_ts"),
                    "end_ts": event.get("end_ts"),
                    "person_ids": persons,
                    "place_refs": places,
                    "tag_keys": tags,
                    "cover_photo_id": event.get("cover_photo_id"),
                    "supporting_photo_ids": _unique(photos_by_event.get(event_id, [])),
                    "confidence": event.get("confidence"),
                }
            )
    return docs


def _build_evidence_docs(
    evidence_rows: Sequence[Dict[str, Any]],
    event_people: Sequence[Dict[str, Any]],
    *,
    event_places_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    people_by_event: Dict[str, List[str]] = {}
    for row in list(event_people or []):
        people_by_event.setdefault(str(row.get("event_id") or ""), []).append(str(row.get("person_id") or ""))
    places_by_event: Dict[str, List[str]] = {}
    for row in list(event_places_rows or []):
        places_by_event.setdefault(str(row.get("event_id") or ""), []).append(str(row.get("place_ref") or ""))
    docs: List[Dict[str, Any]] = []
    for row in list(evidence_rows or []):
        retrieval_text = str(row.get("text") or row.get("normalized_value") or "").strip()
        if not retrieval_text:
            continue
        event_id = str(row.get("event_id") or "")
        docs.append(
            {
                "evidence_id": str(row.get("evidence_id") or ""),
                "user_id": row.get("user_id"),
                "event_id": event_id,
                "photo_id": row.get("photo_id"),
                "source_task_id": row.get("source_task_id"),
                "evidence_type": row.get("evidence_type"),
                "retrieval_text": retrieval_text,
                "normalized_value": row.get("normalized_value"),
                "numeric_value": row.get("numeric_value"),
                "numeric_unit": row.get("numeric_unit"),
                "person_ids": _unique(people_by_event.get(event_id, [])),
                "place_refs": _unique(places_by_event.get(event_id, [])),
                "confidence": row.get("confidence"),
            }
        )
    return docs


def _build_graph_payload(
    *,
    primary_person_id: str | None,
    events: Sequence[Dict[str, Any]],
    event_people: Sequence[Dict[str, Any]],
    event_places: Sequence[Dict[str, Any]],
    relationships: Sequence[Dict[str, Any]],
    relationship_support: Sequence[Dict[str, Any]],
    groups: Sequence[Dict[str, Any]],
    group_members: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    persons: Dict[str, Dict[str, Any]] = {}
    places: Dict[str, Dict[str, Any]] = {}
    if primary_person_id:
        persons.setdefault(primary_person_id, {"person_id": primary_person_id, "is_primary": True})
    for row in list(event_people or []):
        person_id = str(row.get("person_id") or "")
        if person_id:
            persons.setdefault(person_id, {"person_id": person_id})
    for row in list(event_places or []):
        place_ref = str(row.get("place_ref") or "")
        if place_ref:
            places.setdefault(place_ref, {"place_ref": place_ref})
    edges: List[Dict[str, Any]] = []
    for row in list(event_people or []):
        person_id = str(row.get("person_id") or "")
        event_id = str(row.get("event_id") or "")
        if person_id and event_id:
            edges.append(
                {
                    "from_id": person_id,
                    "to_id": event_id,
                    "edge_type": "PARTICIPATES_IN",
                    "properties": {"role": row.get("role"), "weight": row.get("weight")},
                }
            )
    for row in list(event_places or []):
        event_id = str(row.get("event_id") or "")
        place_ref = str(row.get("place_ref") or "")
        if event_id and place_ref:
            edges.append(
                {
                    "from_id": event_id,
                    "to_id": place_ref,
                    "edge_type": "AT",
                    "properties": {"weight": row.get("weight")},
                }
            )
    for relationship in list(relationships or []):
        relationship_id = str(relationship.get("relationship_id") or "")
        person_id = str(relationship.get("person_id") or "")
        if relationship_id and person_id and primary_person_id:
            persons.setdefault(person_id, {"person_id": person_id})
            edges.append(
                {
                    "from_id": primary_person_id,
                    "to_id": relationship_id,
                    "edge_type": "HAS_RELATIONSHIP",
                    "properties": {"confidence": relationship.get("confidence")},
                }
            )
            edges.append(
                {
                    "from_id": relationship_id,
                    "to_id": person_id,
                    "edge_type": "TARGETS",
                    "properties": {"relationship_type": relationship.get("relationship_type")},
                }
            )
    for row in list(relationship_support or []):
        relationship_id = str(row.get("relationship_id") or "")
        event_id = str(row.get("event_id") or "")
        if relationship_id and event_id:
            edges.append(
                {
                    "from_id": relationship_id,
                    "to_id": event_id,
                    "edge_type": "SUPPORTED_BY",
                    "properties": {"support_type": row.get("support_type"), "support_strength": row.get("support_strength")},
                }
            )
    for group in list(groups or []):
        group_id = str(group.get("group_id") or "")
        if not group_id:
            continue
        for member in list(group_members or []):
            if str(member.get("group_id") or "") != group_id:
                continue
            person_id = str(member.get("person_id") or "")
            persons.setdefault(person_id, {"person_id": person_id})
            edges.append(
                {
                    "from_id": group_id,
                    "to_id": person_id,
                    "edge_type": "HAS_MEMBER",
                    "properties": {},
                }
            )
        strong_refs = list((group.get("group_payload") or {}).get("strong_evidence_refs", []) or [])
        for ref in strong_refs:
            event_id = str((ref or {}).get("event_id") or "")
            if event_id:
                edges.append(
                    {
                        "from_id": group_id,
                        "to_id": event_id,
                        "edge_type": "SUPPORTED_BY",
                        "properties": {"reason": group.get("reason")},
                    }
                )
    return {
        "nodes": {
            "persons": list(persons.values()),
            "events": [dict(item) for item in events],
            "relationships": [dict(item) for item in relationships],
            "groups": [dict(item) for item in groups],
            "places": list(places.values()),
        },
        "edges": edges,
    }


def _unique(values: Iterable[Any]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except Exception:
        return None
