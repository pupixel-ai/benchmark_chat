from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence
from urllib.parse import unquote, urlparse

from config import ASSET_URL_PREFIX, FRONTEND_ORIGIN
from services.v0325.profile_compaction import compact_lp3_profile
from services.v0321_3.retrieval_shadow import build_profile_truth_v1
from services.asset_store import TaskAssetStore

_survey_asset_store = TaskAssetStore()


def _unique(values: Iterable[Any]) -> List[Any]:
    seen = set()
    result: List[Any] = []
    for value in values:
        if value is None:
            continue
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(value)
    return result


def _public_asset_url(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered.startswith(("http://", "https://", "data:", "blob:", "//")):
        return text
    if text.startswith("/"):
        origin = FRONTEND_ORIGIN.rstrip("/")
        return f"{origin}{text}" if origin else text
    return text


def _extract_task_asset_relative_path(task_id: str, value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None

    prefixes = (
        f"{ASSET_URL_PREFIX}/{task_id}/",
        f"/api/assets/{task_id}/",
    )
    parsed = urlparse(text)
    path = parsed.path or text
    for prefix in prefixes:
        if path.startswith(prefix):
            return unquote(path[len(prefix) :]).strip() or None
    if "://" not in text and not text.startswith("/"):
        try:
            return _survey_asset_store.sanitize_relative_path(text)
        except Exception:
            return None
    return None


def _survey_accessible_asset_url(task_id: str, *candidates: Any) -> str | None:
    for candidate in candidates:
        text = str(candidate or "").strip()
        if not text:
            continue
        if text.lower().startswith(("http://", "https://", "data:", "blob:", "//")):
            if _extract_task_asset_relative_path(task_id, text):
                continue
            public_url = _public_asset_url(text)
            if public_url:
                return public_url
    for candidate in candidates:
        relative_path = _extract_task_asset_relative_path(task_id, candidate)
        if not relative_path:
            continue
        presigned = _survey_asset_store.presigned_get_url(task_id, relative_path, expires_in=7 * 24 * 3600)
        if presigned:
            return presigned
        public_url = _public_asset_url(_survey_asset_store.asset_url(task_id, relative_path))
        if public_url:
            return public_url
    for candidate in candidates:
        public_url = _public_asset_url(candidate)
        if public_url:
            return public_url
    return None


def _best_display_photo_url(photo: dict | None) -> str | None:
    if not isinstance(photo, dict):
        return None
    return _public_asset_url(
        photo.get("boxed_image_url")
        or photo.get("display_image_url")
        or photo.get("original_image_url")
        or photo.get("asset_url")
    )


def _coalesce_photo_ids(event: dict) -> List[Any]:
    if not isinstance(event, dict):
        return []
    supporting_photo_ids = list(event.get("supporting_photo_ids", []) or [])
    if supporting_photo_ids:
        return supporting_photo_ids
    evidence_photos = list(event.get("evidence_photos", []) or [])
    if evidence_photos:
        return evidence_photos
    trace = dict((event.get("meta_info") or {}).get("trace") or {})
    return list(trace.get("supporting_photo_ids", []) or [])


def _coalesce_person_ids(event: dict) -> List[Any]:
    if not isinstance(event, dict):
        return []
    participants = list(event.get("participants", []) or [])
    if participants:
        return participants
    return list(event.get("participant_person_ids", []) or []) + list(event.get("depicted_person_ids", []) or [])


def _index_asset_manifest(asset_manifest: dict | None) -> Dict[str, dict]:
    if not isinstance(asset_manifest, dict):
        return {}
    files = asset_manifest.get("files") or []
    indexed: Dict[str, dict] = {}
    for item in files:
        if not isinstance(item, dict):
            continue
        relative_path = str(item.get("relative_path") or item.get("path") or "").strip()
        if not relative_path:
            continue
        indexed[relative_path] = item
    return indexed


def _build_photo_catalog(task: dict) -> tuple[Dict[str, dict], List[dict]]:
    result = task.get("result") or {}
    face_payload = result.get("face_recognition") or {}
    images = face_payload.get("images") or []
    uploads = task.get("uploads") or []
    manifest_by_path = _index_asset_manifest(task.get("asset_manifest"))

    by_key: Dict[str, dict] = {}
    ordered: List[dict] = []

    def _register(key: str, payload: dict, *, append: bool) -> None:
        normalized_key = str(key or "").strip()
        if not normalized_key:
            return
        if normalized_key not in by_key:
            if append:
                snapshot = copy.deepcopy(payload)
                by_key[normalized_key] = snapshot
                ordered.append(snapshot)
            else:
                by_key[normalized_key] = payload

    uploads_by_source_hash = {
        str(item.get("source_hash") or "").strip(): item
        for item in uploads
        if isinstance(item, dict) and str(item.get("source_hash") or "").strip()
    }
    uploads_by_image_id = {
        str(item.get("image_id") or "").strip(): item
        for item in uploads
        if isinstance(item, dict) and str(item.get("image_id") or "").strip()
    }

    for upload in uploads:
        if not isinstance(upload, dict):
            continue
        upload_path = str(upload.get("path") or "").strip()
        manifest_item = manifest_by_path.get(upload_path, {})
        photo = {
            "original_photo_id": str(upload.get("source_hash") or upload.get("image_id") or "").strip(),
            "image_id": upload.get("image_id"),
            "source_hash": upload.get("source_hash"),
            "filename": upload.get("filename"),
            "stored_filename": upload.get("stored_filename"),
            "timestamp": upload.get("timestamp"),
            "location": upload.get("location"),
            "path": upload_path or None,
            "asset_url": _public_asset_url(upload.get("url") or manifest_item.get("asset_url")),
            "original_image_url": _public_asset_url(upload.get("url") or manifest_item.get("asset_url")),
            "display_image_url": _public_asset_url(upload.get("preview_url") or upload.get("url") or manifest_item.get("asset_url")),
            "boxed_image_url": None,
            "compressed_image_url": None,
            "content_type": upload.get("content_type"),
            "width": upload.get("width"),
            "height": upload.get("height"),
        }
        _register(photo["original_photo_id"], photo, append=True)
        if photo.get("image_id"):
            _register(str(photo["image_id"]), by_key[str(photo["original_photo_id"])], append=False)

    for image in images:
        if not isinstance(image, dict):
            continue
        source_hash = str(image.get("source_hash") or "").strip()
        image_id = str(image.get("image_id") or "").strip()
        upload = uploads_by_source_hash.get(source_hash) or uploads_by_image_id.get(image_id) or {}
        upload_path = str(upload.get("path") or "").strip()
        manifest_item = manifest_by_path.get(upload_path, {})
        photo = {
            "original_photo_id": source_hash or image_id,
            "image_id": image.get("image_id") or upload.get("image_id"),
            "source_hash": source_hash or upload.get("source_hash"),
            "filename": image.get("filename") or upload.get("filename"),
            "stored_filename": upload.get("stored_filename"),
            "timestamp": image.get("timestamp") or upload.get("timestamp"),
            "location": image.get("location") or upload.get("location"),
            "path": upload_path or None,
            "asset_url": _public_asset_url(upload.get("url") or manifest_item.get("asset_url") or image.get("original_image_url")),
            "original_image_url": _public_asset_url(image.get("original_image_url") or upload.get("url") or manifest_item.get("asset_url")),
            "display_image_url": _public_asset_url(image.get("display_image_url") or upload.get("preview_url") or upload.get("url") or manifest_item.get("asset_url")),
            "boxed_image_url": _public_asset_url(image.get("boxed_image_url")),
            "compressed_image_url": _public_asset_url(image.get("compressed_image_url")),
            "content_type": upload.get("content_type"),
            "width": image.get("width") or upload.get("width"),
            "height": image.get("height") or upload.get("height"),
        }
        canonical = by_key.get(photo["original_photo_id"])
        if canonical is not None:
            for field_name, field_value in photo.items():
                if field_value in (None, ""):
                    continue
                canonical[field_name] = field_value
        else:
            _register(photo["original_photo_id"], photo, append=True)
            canonical = by_key.get(photo["original_photo_id"]) or photo
        if photo.get("image_id"):
            _register(str(photo["image_id"]), canonical, append=False)

    return by_key, ordered


def _resolve_original_photos(original_photo_ids: Sequence[Any], photo_catalog: Dict[str, dict]) -> List[dict]:
    resolved: List[dict] = []
    for original_photo_id in _unique(original_photo_ids):
        key = str(original_photo_id or "").strip()
        if not key:
            continue
        photo = photo_catalog.get(key)
        if photo is None:
            resolved.append(
                {
                    "original_photo_id": key,
                    "image_id": None,
                    "source_hash": key,
                    "filename": None,
                    "stored_filename": None,
                    "timestamp": None,
                    "location": None,
                    "path": None,
                    "asset_url": None,
                    "original_image_url": None,
                    "display_image_url": None,
                    "boxed_image_url": None,
                    "compressed_image_url": None,
                    "content_type": None,
                    "width": None,
                    "height": None,
                }
            )
            continue
        resolved.append(copy.deepcopy(photo))
    return resolved


def _resolve_task_photo_ids(original_photo_ids: Sequence[Any], photo_catalog: Dict[str, dict]) -> List[str]:
    resolved: List[str] = []
    for original_photo_id in _unique(original_photo_ids):
        key = str(original_photo_id or "").strip()
        if not key:
            continue
        photo = photo_catalog.get(key) or {}
        image_id = str(photo.get("image_id") or "").strip()
        resolved.append(image_id or key)
    return _unique(resolved)


def _build_events(memory_payload: dict, photo_catalog: Dict[str, dict], vlm_items: Sequence[dict]) -> List[dict]:
    if memory_payload.get("pipeline_family") == "v0323" or memory_payload.get("lp1_events"):
        return _build_lp_snapshot_events(memory_payload, photo_catalog, vlm_items)
    events: List[dict] = []
    event_source = list(memory_payload.get("delta_event_revisions", []) or [])
    if not event_source:
        event_source = list(memory_payload.get("event_revisions", []) or [])
    photo_to_vlm: Dict[str, List[dict]] = {}
    for item in vlm_items:
        if not isinstance(item, dict):
            continue
        normalized = str(item.get("photo_id") or "").strip()
        if normalized:
            photo_to_vlm.setdefault(normalized, []).append(copy.deepcopy(item))
        for photo_id in list(item.get("photo_ids", []) or []):
            normalized = str(photo_id or "").strip()
            if not normalized:
                continue
            photo_to_vlm.setdefault(normalized, []).append(copy.deepcopy(item))
    for event in event_source:
        original_photo_ids = list(event.get("original_photo_ids", []) or [])
        photo_ids = _resolve_task_photo_ids(original_photo_ids, photo_catalog)
        event_vlm = []
        seen_vlm = set()
        for photo_id in photo_ids:
            normalized = str(photo_id or "").strip()
            for item in photo_to_vlm.get(normalized, []):
                identity = (
                    str(item.get("photo_id") or ""),
                    tuple(item.get("person_ids", []) or []),
                )
                if identity in seen_vlm:
                    continue
                seen_vlm.add(identity)
                event_vlm.append(item)
        events.append(
            {
                "llm_summary": event.get("event_summary") or event.get("title"),
                "person_ids": _unique(
                    list(event.get("participant_person_ids", []) or [])
                    + list(event.get("depicted_person_ids", []) or [])
                ),
                "photo_ids": photo_ids,
                "vlm": event_vlm,
            }
        )
    return events


def _build_relationships(memory_payload: dict, events: Sequence[dict], photo_catalog: Dict[str, dict]) -> List[dict]:
    if memory_payload.get("pipeline_family") == "v0323" or memory_payload.get("lp2_relationships"):
        return _build_lp_snapshot_relationships(memory_payload, events, photo_catalog)
    relationships: List[dict] = []
    relationship_source = list(memory_payload.get("delta_relationship_revisions", []) or [])
    if not relationship_source:
        relationship_source = list(memory_payload.get("relationship_revisions", []) or [])
    for relationship in relationship_source:
        supporting_photo_ids = _resolve_task_photo_ids(
            list(relationship.get("supporting_photo_ids", []) or []),
            photo_catalog,
        )
        if not supporting_photo_ids:
            event_photo_ids = _unique(
                photo_id
                for event in events
                for photo_id in list(event.get("photo_ids", []) or [])
                if relationship.get("target_person_id") in list(event.get("person_ids", []) or [])
            )
            supporting_photo_ids = event_photo_ids
        relationships.append(
            {
                "person_id": relationship.get("target_person_id"),
                "photo_ids": supporting_photo_ids,
            }
        )
    return relationships


def _build_profile(memory_payload: dict, photo_catalog: Dict[str, dict], *, user_id: str, pipeline_family: str) -> dict:
    if memory_payload.get("pipeline_family") == "v0323" or memory_payload.get("lp3_profile"):
        lp3_profile = compact_lp3_profile(memory_payload.get("lp3_profile") or {})
        structured = copy.deepcopy(lp3_profile.get("structured") or {})
        report = str(lp3_profile.get("report_markdown") or lp3_profile.get("report") or "")
        return {
            "summary": str(structured.get("summary") or lp3_profile.get("summary") or ""),
            "report_markdown": report,
            "structured": structured,
        }
    profile_revision = copy.deepcopy(memory_payload.get("delta_profile_revision") or memory_payload.get("profile_revision") or {})
    profile_markdown = str(memory_payload.get("delta_profile_markdown") or memory_payload.get("profile_markdown") or "")
    profile_input_pack = copy.deepcopy(memory_payload.get("delta_profile_input_pack") or memory_payload.get("profile_input_pack") or {})
    if not profile_input_pack:
        profile_input_pack = copy.deepcopy(memory_payload.get("profile_input_pack_partial") or {})
    relationship_revisions = list(memory_payload.get("delta_relationship_revisions", []) or memory_payload.get("relationship_revisions", []) or [])
    profile_truth = copy.deepcopy(memory_payload.get("delta_profile_truth_v1") or memory_payload.get("profile_truth_v1") or {})
    if not profile_truth and profile_revision and profile_input_pack:
        profile_truth = build_profile_truth_v1(
            user_id=user_id,
            pipeline_family=pipeline_family,
            profile_revision=profile_revision,
            profile_input_pack=profile_input_pack,
            relationship_revisions=relationship_revisions,
            profile_markdown=profile_markdown,
        )
    return {
        "report_markdown": profile_markdown,
    }


def _build_survey_events(task_id: str, memory_payload: dict, photo_catalog: Dict[str, dict]) -> List[dict]:
    events: List[dict] = []
    if memory_payload.get("pipeline_family") == "v0323" or memory_payload.get("lp1_events"):
        event_source = list(memory_payload.get("lp1_events", []) or [])
        is_lp_snapshot = True
    else:
        event_source = list(memory_payload.get("delta_event_revisions", []) or [])
        if not event_source:
            event_source = list(memory_payload.get("event_revisions", []) or [])
        is_lp_snapshot = False

    for event in event_source:
        if not isinstance(event, dict):
            continue
        original_photo_ids = (
            _coalesce_photo_ids(event)
            if is_lp_snapshot
            else list(event.get("original_photo_ids", []) or []) or _coalesce_photo_ids(event)
        )
        resolved_photos = _resolve_original_photos(original_photo_ids, photo_catalog)
        photo_ids = _resolve_task_photo_ids(original_photo_ids, photo_catalog)
        person_ids = (
            _unique(_coalesce_person_ids(event))
            if is_lp_snapshot
            else _unique(
                list(event.get("participant_person_ids", []) or [])
                + list(event.get("depicted_person_ids", []) or [])
            )
        )
        place_refs = _unique(list(event.get("place_refs", []) or []))
        summary = str(
            event.get("event_summary")
            or event.get("narrative_synthesis")
            or event.get("narrative")
            or event.get("title")
            or ""
        ).strip()
        scene = str(event.get("title") or (place_refs[0] if place_refs else "") or "相关事件").strip()
        activity = str(
            event.get("coarse_event_type")
            or event.get("event_type")
            or event.get("activity")
            or event.get("title")
            or summary
            or "相关活动"
        ).strip()
        if place_refs:
            social_context = f"地点：{', '.join(str(item) for item in place_refs[:2])}"
        elif person_ids:
            social_context = f"涉及 {', '.join(str(item) for item in person_ids[:3])}"
        else:
            social_context = ""
        events.append(
            {
                "timestamp": str(event.get("started_at") or event.get("timestamp") or event.get("date") or "").strip(),
                "scene": scene,
                "summary": summary,
                "activity": activity,
                "social_context": social_context,
                "person_ids": person_ids,
                "photo_ids": photo_ids,
                "photo_boxed_urls": _unique(
                    _survey_accessible_asset_url(
                        task_id,
                        photo.get("boxed_image_url"),
                        photo.get("display_image_url"),
                        photo.get("original_image_url"),
                        photo.get("asset_url"),
                        photo.get("path"),
                    )
                    for photo in resolved_photos
                    if _survey_accessible_asset_url(
                        task_id,
                        photo.get("boxed_image_url"),
                        photo.get("display_image_url"),
                        photo.get("original_image_url"),
                        photo.get("asset_url"),
                        photo.get("path"),
                    )
                ),
            }
        )
    return events


def _build_survey_relationships(
    task_id: str,
    memory_payload: dict,
    survey_events: Sequence[dict],
    photo_catalog: Dict[str, dict],
) -> List[dict]:
    if memory_payload.get("pipeline_family") == "v0323" or memory_payload.get("lp2_relationships"):
        relationship_source = list(memory_payload.get("lp2_relationships", []) or [])
    else:
        relationship_source = list(memory_payload.get("delta_relationship_revisions", []) or [])
        if not relationship_source:
            relationship_source = list(memory_payload.get("relationship_revisions", []) or [])

    relationships: List[dict] = []
    for relationship in relationship_source:
        if not isinstance(relationship, dict):
            continue
        person_id = str(relationship.get("person_id") or relationship.get("target_person_id") or "").strip()
        if not person_id:
            continue
        evidence = relationship.get("evidence") if isinstance(relationship.get("evidence"), dict) else {}
        original_photo_ids = list(relationship.get("supporting_photo_ids", []) or [])
        if not original_photo_ids and evidence:
            original_photo_ids = list(evidence.get("photo_ids", []) or [])
        resolved_supporting_photos = _resolve_original_photos(original_photo_ids, photo_catalog)
        supporting_photo_ids = _resolve_task_photo_ids(original_photo_ids, photo_catalog)
        related_events = [
            item
            for item in survey_events
            if person_id in list(item.get("person_ids", []) or [])
            or bool(set(supporting_photo_ids) & set(item.get("photo_ids", []) or []))
        ]
        if not supporting_photo_ids:
            supporting_photo_ids = _unique(
                photo_id
                for item in related_events
                for photo_id in list(item.get("photo_ids", []) or [])
            )

        boxed_image_url = next(
            (
                str(
                    _survey_accessible_asset_url(
                        task_id,
                        photo.get("boxed_image_url"),
                        photo.get("display_image_url"),
                        photo.get("original_image_url"),
                        photo.get("asset_url"),
                        photo.get("path"),
                    )
                    or ""
                ).strip()
                for photo in resolved_supporting_photos
                if str(
                    _survey_accessible_asset_url(
                        task_id,
                        photo.get("boxed_image_url"),
                        photo.get("display_image_url"),
                        photo.get("original_image_url"),
                        photo.get("asset_url"),
                        photo.get("path"),
                    )
                    or ""
                ).strip()
            ),
            "",
        )
        if not boxed_image_url:
            boxed_image_url = next(
                (
                    str(url or "").strip()
                    for item in related_events
                    for url in list(item.get("photo_boxed_urls", []) or [])
                    if str(url or "").strip()
                ),
                "",
            )

        relationships.append(
            {
                "person_id": person_id,
                "photo_count": len(supporting_photo_ids),
                "photo_ids_sample": supporting_photo_ids[:5],
                "boxed_image_url": boxed_image_url or None,
                "sample_scenes": [
                    {
                        "timestamp": str(item.get("timestamp") or ""),
                        "scene": str(item.get("scene") or ""),
                        "summary": str(item.get("summary") or ""),
                        "activity": str(item.get("activity") or ""),
                        "social_context": str(item.get("social_context") or ""),
                    }
                    for item in related_events[:5]
                ],
            }
        )
    return relationships


def _build_task_vlm(task: dict, photo_catalog: Dict[str, dict]) -> List[dict]:
    result = task.get("result") or {}
    memory_payload = result.get("memory") or {}
    if memory_payload.get("pipeline_family") == "v0323" or memory_payload.get("vp1_observations"):
        vlm_items: List[dict] = []
        for item in list(memory_payload.get("vp1_observations", []) or []):
            if not isinstance(item, dict):
                continue
            vlm_items.append(
                {
                    "photo_id": str(item.get("photo_id") or "").strip(),
                    "person_ids": _unique(item.get("face_person_ids", []) or []),
                }
            )
        return vlm_items
    face_payload = result.get("face_recognition") or {}
    images = list(face_payload.get("images", []) or [])
    image_person_ids: Dict[str, List[str]] = {}
    for image in images:
        if not isinstance(image, dict):
            continue
        image_id = str(image.get("image_id") or "").strip()
        if not image_id:
            continue
        person_ids = _unique(
            face.get("person_id")
            for face in list(image.get("faces", []) or [])
            if isinstance(face, dict)
        )
        image_person_ids[image_id] = person_ids

    observation_source = list(memory_payload.get("vlm_observations", []) or [])
    if not observation_source:
        task_dir = Path(str(task.get("task_dir") or "")).expanduser()
        cache_payload = task_dir / "cache" / "vlm_cache.json"
        if cache_payload.exists():
            try:
                cache = json.loads(cache_payload.read_text())
            except Exception:
                cache = {}
            photos = list((cache.get("photos") or [])) if isinstance(cache, dict) else []
            for item in photos:
                if not isinstance(item, dict):
                    continue
                photo_id = str(item.get("photo_id") or "").strip()
                analysis = dict(item.get("vlm_analysis") or {})
                event = dict(analysis.get("event") or {})
                scene = dict(analysis.get("scene") or {})
                place_candidates = []
                for candidate in list(analysis.get("place_candidates") or []):
                    if isinstance(candidate, dict):
                        place_candidates.append(candidate)
                    elif candidate is not None:
                        place_candidates.append(str(candidate))
                observation_source.append(
                    {
                        "photo_id": photo_id,
                        "person_ids": list(item.get("face_person_ids") or []),
                    }
                )
    if not observation_source:
        transparency = ((memory_payload.get("transparency") or {}).get("vlm_stage") or {})
        observation_source = list(transparency.get("summaries", []) or [])

    vlm_items: List[dict] = []
    for item in observation_source:
        if not isinstance(item, dict):
            continue
        photo_id = str(item.get("photo_id") or "").strip()
        if not photo_id and item.get("original_photo_ids"):
            original_photo_ids = list(item.get("original_photo_ids", []) or [])
            photo_ids = _resolve_task_photo_ids(original_photo_ids, photo_catalog)
            photo_id = str(photo_ids[0] or "").strip() if photo_ids else ""
        vlm_items.append(
            {
                "photo_id": photo_id,
                "person_ids": list(item.get("person_ids", []) or image_person_ids.get(photo_id, [])),
            }
        )
    return vlm_items


def _build_lp_snapshot_events(memory_payload: dict, photo_catalog: Dict[str, dict], vlm_items: Sequence[dict]) -> List[dict]:
    events: List[dict] = []
    photo_to_vlm: Dict[str, List[dict]] = {}
    for item in vlm_items:
        if not isinstance(item, dict):
            continue
        photo_id = str(item.get("photo_id") or "").strip()
        if photo_id:
            photo_to_vlm.setdefault(photo_id, []).append(copy.deepcopy(item))
    for event in list(memory_payload.get("lp1_events", []) or []):
        if not isinstance(event, dict):
            continue
        photo_ids = _resolve_task_photo_ids(_coalesce_photo_ids(event), photo_catalog)
        event_vlm = []
        seen_vlm = set()
        for photo_id in photo_ids:
            for item in photo_to_vlm.get(photo_id, []):
                identity = (str(item.get("photo_id") or ""), tuple(item.get("person_ids", []) or []))
                if identity in seen_vlm:
                    continue
                seen_vlm.add(identity)
                event_vlm.append(item)
        events.append(
            {
                "llm_summary": event.get("narrative_synthesis") or event.get("narrative") or event.get("title"),
                "person_ids": _unique(_coalesce_person_ids(event)),
                "photo_ids": photo_ids,
                "vlm": event_vlm,
            }
        )
    return events


def _build_lp_snapshot_relationships(memory_payload: dict, events: Sequence[dict], photo_catalog: Dict[str, dict]) -> List[dict]:
    relationships: List[dict] = []
    for relationship in list(memory_payload.get("lp2_relationships", []) or []):
        if not isinstance(relationship, dict):
            continue
        evidence = copy.deepcopy(relationship.get("evidence") or {})
        supporting_photo_ids = _resolve_task_photo_ids(
            list(relationship.get("supporting_photo_ids", []) or [])
            or list(evidence.get("photo_ids", []) or []),
            photo_catalog,
        )
        if not supporting_photo_ids:
            supporting_photo_ids = _unique(
                photo_id
                for event in events
                for photo_id in list(event.get("photo_ids", []) or [])
                if relationship.get("person_id") in list(event.get("person_ids", []) or [])
            )
        if isinstance(evidence, dict):
            evidence["photo_ids"] = supporting_photo_ids
        relationships.append(
            {
                "person_id": relationship.get("person_id"),
                "photo_ids": supporting_photo_ids,
                "relationship_type": relationship.get("relationship_type"),
                "status": relationship.get("status"),
                "confidence": relationship.get("confidence"),
                "intimacy_score": relationship.get("intimacy_score"),
                "reasoning": relationship.get("reasoning") or relationship.get("reason"),
                "shared_events": copy.deepcopy(list(relationship.get("shared_events", []) or [])),
                "evidence": evidence,
            }
        )
    return relationships


def build_task_memory_core_payload(task: dict) -> dict:
    result = task.get("result") or {}
    memory_payload = result.get("memory")
    if not isinstance(memory_payload, dict):
        raise KeyError("当前任务没有 memory 输出")

    photo_catalog, _ = _build_photo_catalog(task)
    user_id = str(task.get("user_id") or "")
    pipeline_family = str(memory_payload.get("pipeline_family") or "")
    vlm_items = _build_task_vlm(task, photo_catalog)
    events = _build_events(memory_payload, photo_catalog, vlm_items)
    relationships = _build_relationships(memory_payload, events, photo_catalog)
    profile = _build_profile(memory_payload, photo_catalog, user_id=user_id, pipeline_family=pipeline_family)

    return {
        "vlm": vlm_items,
        "events": events,
        "relationships": relationships,
        "profile": profile,
    }


def build_task_survey_import_payload(task: dict) -> dict:
    result = task.get("result") or {}
    memory_payload = result.get("memory")
    if not isinstance(memory_payload, dict):
        raise KeyError("当前任务没有 memory 输出")

    task_id = str(task.get("task_id") or "").strip()
    photo_catalog, _ = _build_photo_catalog(task)
    user_id = str(task.get("user_id") or "")
    pipeline_family = str(memory_payload.get("pipeline_family") or "")
    profile = _build_profile(memory_payload, photo_catalog, user_id=user_id, pipeline_family=pipeline_family)
    survey_events = _build_survey_events(task_id, memory_payload, photo_catalog)
    relationships = _build_survey_relationships(task_id, memory_payload, survey_events, photo_catalog)
    return {
        "profile": {
            "report_markdown": str(profile.get("report_markdown") or ""),
            "structured_profile": copy.deepcopy(profile.get("structured") or {}),
        },
        "relationships": relationships,
    }
