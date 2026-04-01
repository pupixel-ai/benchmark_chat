from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from backend.memory_full_retrieval import _build_photo_catalog
from backend.query_v1 import QueryStore
from backend.query_v1.materializer import SUPPORTED_QUERY_V1_VERSIONS


_QUERY_STORE = QueryStore()


def _unique(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _stage_versions(version: str | None) -> Dict[str, Optional[int]]:
    normalized = str(version or "").strip().lower()
    digits = "".join(ch for ch in normalized if ch.isdigit())
    numeric = int(digits) if digits else None
    return {
        "face": numeric,
        "vlm": numeric,
        "lp1": numeric,
        "lp2": numeric,
        "lp3": numeric,
        "judge": numeric,
    }


def _query_shell(
    task_id: str,
    *,
    include_raw: bool = False,
    include_artifacts: bool = False,
    include_traces: bool = False,
) -> dict:
    return {
        "task_id": task_id,
        "include_raw": include_raw,
        "include_artifacts": include_artifacts,
        "include_traces": include_traces,
    }


def _empty_scope() -> dict:
    return {
        "materialization": None,
        "photos": [],
        "persons": [],
        "faces": [],
        "events": [],
        "event_photos": [],
        "event_people": [],
        "event_places": [],
        "evidence": [],
        "relationships": [],
        "relationship_support": [],
        "groups": [],
        "group_members": [],
        "profile_facts": [],
    }


def _task_shell(task: dict) -> dict:
    version = str(task.get("version") or "").strip() or None
    return {
        "task_id": task.get("task_id"),
        "version": version,
        "pipeline_version": _stage_versions(version)["face"],
        "pipeline_channel": version.split("-", 1)[1] if "-" in version else None,
        "status": task.get("status"),
        "stage": task.get("stage"),
        "stage_versions": _stage_versions(version),
        "created_at": task.get("created_at"),
        "updated_at": task.get("updated_at"),
    }


def _first_nonempty(*values: Any) -> Optional[str]:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return None


def _normalized_photo_id(task_id: str, photo: dict) -> str:
    source = str(photo.get("image_id") or photo.get("original_photo_id") or photo.get("source_hash") or "").strip()
    return f"{task_id}:{source}" if source else f"{task_id}:photo"


def _serialize_photo(task_id: str, photo: dict) -> dict:
    raw_url = str(photo.get("original_image_url") or photo.get("asset_url") or "").strip() or None
    display_url = str(photo.get("display_image_url") or raw_url or "").strip() or None
    boxed_url = str(photo.get("boxed_image_url") or display_url or raw_url or "").strip() or None
    compressed_url = str(photo.get("compressed_image_url") or display_url or raw_url or "").strip() or None
    return {
        "photo_id": _normalized_photo_id(task_id, photo),
        "source_photo_id": photo.get("image_id") or photo.get("original_photo_id") or photo.get("source_hash"),
        "original_filename": photo.get("filename"),
        "source_hash": photo.get("source_hash"),
        "taken_at": photo.get("timestamp"),
        "location": copy.deepcopy(photo.get("location")),
        "width": photo.get("width"),
        "height": photo.get("height"),
        "mime_type": photo.get("content_type"),
        "photo_url": display_url,
        "urls": {
            "raw": raw_url,
            "display": display_url,
            "boxed_full": boxed_url,
            "compressed": compressed_url,
        },
    }


def _looks_displayable_url(url: Optional[str]) -> bool:
    normalized = str(url or "").strip().lower()
    if not normalized:
        return False
    if normalized.endswith(".livp"):
        return False
    return True


def _photo_maps(task: dict) -> tuple[List[dict], Dict[str, dict]]:
    task_id = str(task.get("task_id") or "").strip()
    catalog, ordered = _build_photo_catalog(task)
    face_images = list((((task.get("result") or {}).get("face_recognition") or {}).get("images") or []))
    face_overrides: Dict[str, dict] = {}
    for image in face_images:
        if not isinstance(image, dict):
            continue
        override = {
            "original_image_url": image.get("original_image_url"),
            "display_image_url": image.get("display_image_url"),
            "boxed_image_url": image.get("boxed_image_url"),
            "content_type": image.get("content_type"),
            "width": image.get("width"),
            "height": image.get("height"),
            "filename": image.get("filename"),
            "timestamp": image.get("timestamp"),
        }
        for key in (image.get("image_id"), image.get("source_hash")):
            normalized = str(key or "").strip()
            if normalized:
                face_overrides[normalized] = override
    photos: List[dict] = []
    raw_to_serialized: Dict[str, dict] = {}
    for item in ordered:
        override = None
        for key in (item.get("image_id"), item.get("original_photo_id"), item.get("source_hash")):
            normalized = str(key or "").strip()
            if normalized and normalized in face_overrides:
                override = face_overrides[normalized]
                break
        merged = copy.deepcopy(item)
        if override:
            if override.get("original_image_url"):
                merged["original_image_url"] = override.get("original_image_url")
            if override.get("display_image_url") and (
                not _looks_displayable_url(merged.get("display_image_url"))
                or not _looks_displayable_url(merged.get("original_image_url"))
            ):
                merged["display_image_url"] = override.get("display_image_url")
            if override.get("boxed_image_url"):
                merged["boxed_image_url"] = override.get("boxed_image_url")
            if override.get("content_type"):
                merged["content_type"] = override.get("content_type")
            if override.get("width") is not None:
                merged["width"] = override.get("width")
            if override.get("height") is not None:
                merged["height"] = override.get("height")
            if override.get("filename") and not merged.get("filename"):
                merged["filename"] = override.get("filename")
            if override.get("timestamp") and not merged.get("timestamp"):
                merged["timestamp"] = override.get("timestamp")
        serialized = _serialize_photo(task_id, merged)
        photos.append(serialized)
        for key in (
            merged.get("original_photo_id"),
            merged.get("image_id"),
            merged.get("source_hash"),
        ):
            normalized = str(key or "").strip()
            if normalized:
                raw_to_serialized[normalized] = serialized
    return photos, raw_to_serialized


def _query_store_scope(task: dict) -> dict:
    return _query_store_subset(task, "bundle")


def _query_store_subset(task: dict, scope_kind: str) -> dict:
    task_id = str(task.get("task_id") or "").strip()
    user_id = str(task.get("user_id") or "").strip()
    version = str(task.get("version") or "").strip()
    if not task_id or not user_id or version not in SUPPORTED_QUERY_V1_VERSIONS:
        return _empty_scope()
    fetchers = {
        "faces": _QUERY_STORE.fetch_faces_scope,
        "events": _QUERY_STORE.fetch_events_scope,
        "vlm": _QUERY_STORE.fetch_vlm_scope,
        "profiles": _QUERY_STORE.fetch_profiles_scope,
        "relationships": _QUERY_STORE.fetch_relationships_scope,
        "bundle": _QUERY_STORE.fetch_bundle_scope,
    }
    fetcher = fetchers.get(scope_kind, _QUERY_STORE.fetch_bundle_scope)
    return fetcher(user_id=user_id, source_task_id=task_id)


def _serialize_query_store_photo(task_id: str, row: dict, fallback: Optional[dict]) -> dict:
    payload = copy.deepcopy(row.get("photo_payload") or {})
    raw_photo_id = str(
        payload.get("image_id")
        or payload.get("original_photo_id")
        or payload.get("source_hash")
        or row.get("photo_id")
        or ""
    ).strip()
    fallback_urls = copy.deepcopy((fallback or {}).get("urls") or {})
    raw_url = _first_nonempty(
        payload.get("original_image_url"),
        row.get("asset_url"),
        fallback_urls.get("raw"),
        (fallback or {}).get("photo_url"),
    )
    display_candidates = [
        payload.get("display_image_url"),
        fallback_urls.get("display"),
        payload.get("boxed_image_url"),
        row.get("asset_url"),
        raw_url,
    ]
    display_url = next((candidate for candidate in display_candidates if _looks_displayable_url(candidate)), None) or _first_nonempty(*display_candidates)
    boxed_url = _first_nonempty(payload.get("boxed_image_url"), fallback_urls.get("boxed_full"), display_url, raw_url)
    compressed_url = _first_nonempty(payload.get("compressed_image_url"), fallback_urls.get("compressed"), display_url, raw_url)
    normalized_photo_id = str((fallback or {}).get("photo_id") or f"{task_id}:{raw_photo_id or 'photo'}").strip()
    return {
        "photo_id": normalized_photo_id,
        "source_photo_id": raw_photo_id or (fallback or {}).get("source_photo_id"),
        "original_filename": payload.get("filename") or (fallback or {}).get("original_filename"),
        "source_hash": payload.get("source_hash") or (fallback or {}).get("source_hash"),
        "taken_at": payload.get("timestamp") or row.get("captured_at") or (fallback or {}).get("taken_at"),
        "location": copy.deepcopy(payload.get("location") or (fallback or {}).get("location")),
        "width": payload.get("width") or row.get("width") or (fallback or {}).get("width"),
        "height": payload.get("height") or row.get("height") or (fallback or {}).get("height"),
        "mime_type": payload.get("content_type") or row.get("content_type") or (fallback or {}).get("mime_type"),
        "photo_url": display_url,
        "urls": {
            "raw": raw_url,
            "display": display_url,
            "boxed_full": boxed_url,
            "compressed": compressed_url,
        },
    }


def _canonical_photo_assets(task: dict, scope: dict) -> tuple[List[dict], Dict[str, dict]]:
    task_id = str(task.get("task_id") or "").strip()
    serialized: List[dict] = []
    by_raw_id: Dict[str, dict] = {}
    seen_raw_ids: set[str] = set()
    for row in list(scope.get("photos") or []):
        raw_photo_id = str(row.get("photo_id") or "").strip()
        if not raw_photo_id or raw_photo_id in seen_raw_ids:
            continue
        seen_raw_ids.add(raw_photo_id)
        photo = _serialize_query_store_photo(task_id, row, None)
        serialized.append(photo)
        by_raw_id[raw_photo_id] = photo
    return serialized, by_raw_id


def _rows_by_key(rows: Sequence[dict], key: str) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in list(rows or []):
        value = str(row.get(key) or "").strip()
        if value:
            grouped[value].append(row)
    return grouped


def _build_events_from_scope(scope: dict, photo_map: Dict[str, dict]) -> List[dict]:
    event_photos_by_event = _rows_by_key(scope.get("event_photos") or [], "event_id")
    event_people_by_event = _rows_by_key(scope.get("event_people") or [], "event_id")
    events: List[dict] = []
    for row in sorted(list(scope.get("events") or []), key=lambda item: (str(item.get("start_ts") or ""), str(item.get("event_id") or ""))):
        event_id = str(row.get("event_id") or "").strip()
        if not event_id:
            continue
        photo_rows = sorted(event_photos_by_event.get(event_id, []), key=lambda item: int(item.get("sort_order") or 0))
        raw_photo_ids = [str(item.get("photo_id") or "").strip() for item in photo_rows if str(item.get("photo_id") or "").strip()]
        photos = [copy.deepcopy(photo_map[photo_id]) for photo_id in raw_photo_ids if photo_id in photo_map]
        photo_ids, photo_urls = _event_photo_links(raw_photo_ids, photo_map)
        person_ids = _unique(str(item.get("person_id") or "").strip() for item in event_people_by_event.get(event_id, []))
        payload = copy.deepcopy(row.get("event_payload") or {})
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("event_id", event_id)
        payload.setdefault("title", row.get("title"))
        if row.get("summary"):
            payload.setdefault("llm_summary", row.get("summary"))
            payload.setdefault("narrative_synthesis", row.get("summary"))
        if row.get("start_ts"):
            payload.setdefault("date", str(row.get("start_ts") or "")[:10])
            payload.setdefault("started_at", row.get("start_ts"))
        if row.get("end_ts"):
            payload.setdefault("ended_at", row.get("end_ts"))
        payload["person_ids"] = person_ids
        payload["photo_ids"] = photo_ids
        payload["photo_urls"] = photo_urls
        payload["photos"] = photos
        payload["cover_photo_id"] = cover_photo_id = next((str(item.get("photo_id") or "").strip() for item in photo_rows if item.get("is_cover")), "") or (photo_ids[0] if photo_ids else None)
        payload["cover_photo_url"] = next((str(item.get("photo_url") or "").strip() for item in photos if str(item.get("photo_id") or "").strip() == cover_photo_id), None) or (photo_urls[0] if photo_urls else None)
        payload.setdefault(
            "vlm",
            [
                {
                    "photo_id": raw_photo_id,
                    "person_ids": person_ids,
                }
                for raw_photo_id in raw_photo_ids
            ],
        )
        events.append(payload)
    return events


def _build_vlm_from_scope(scope: dict, photo_map: Dict[str, dict]) -> List[dict]:
    event_people_by_event = _rows_by_key(scope.get("event_people") or [], "event_id")
    evidence_by_photo = _rows_by_key(
        [row for row in list(scope.get("evidence") or []) if str(row.get("source_stage") or "") == "vp1" and str(row.get("photo_id") or "").strip()],
        "photo_id",
    )
    items: List[dict] = []
    for raw_photo_id in sorted(evidence_by_photo):
        photo = photo_map.get(raw_photo_id) or {}
        rows = evidence_by_photo[raw_photo_id]
        person_ids = _unique(
            str(person.get("person_id") or "").strip()
            for row in rows
            for person in event_people_by_event.get(str(row.get("event_id") or "").strip(), [])
        )
        analysis = None
        for row in rows:
            payload = row.get("evidence_payload") or {}
            candidate = payload.get("analysis") if isinstance(payload, dict) else None
            if isinstance(candidate, dict) and candidate:
                analysis = copy.deepcopy(candidate)
                break
        item = analysis if isinstance(analysis, dict) else {}
        item["photo_id"] = photo.get("photo_id") or raw_photo_id
        item["source_photo_id"] = photo.get("source_photo_id") or raw_photo_id
        item["person_ids"] = person_ids
        item["photo_url"] = photo.get("photo_url")
        item["urls"] = copy.deepcopy(photo.get("urls") or {})
        if "summary" not in item:
            summary = next((str(row.get("text") or "").strip() for row in rows if str(row.get("evidence_type") or "") == "activity" and str(row.get("text") or "").strip()), "")
            if summary:
                item["summary"] = summary
        item.setdefault(
            "ocr_hits",
            [str(row.get("text") or "").strip() for row in rows if str(row.get("evidence_type") or "") == "ocr" and str(row.get("text") or "").strip()],
        )
        items.append(item)
    return items


def _minimal_event_ref(row: Optional[dict]) -> Optional[dict]:
    if not isinstance(row, dict):
        return None
    event_id = str(row.get("event_id") or "").strip()
    if not event_id:
        return None
    narrative = str((row.get("event_payload") or {}).get("narrative_synthesis") or row.get("summary") or row.get("title") or "").strip()
    return {
        "event_id": event_id,
        "date": str(row.get("start_ts") or "")[:10] or None,
        "narrative": narrative or None,
    }


def _build_relationships_from_scope(scope: dict, photo_map: Dict[str, dict]) -> List[dict]:
    support_by_relationship = _rows_by_key(scope.get("relationship_support") or [], "relationship_id")
    event_by_id = {
        str(item.get("event_id") or "").strip(): item
        for item in list(scope.get("events") or [])
        if str(item.get("event_id") or "").strip()
    }
    relationships: List[dict] = []
    for row in sorted(list(scope.get("relationships") or []), key=lambda item: (-float(item.get("intimacy_score") or 0.0), str(item.get("person_id") or ""))):
        relationship_id = str(row.get("relationship_id") or "").strip()
        if not relationship_id:
            continue
        supports = support_by_relationship.get(relationship_id, [])
        raw_photo_ids = [str(item.get("photo_id") or "").strip() for item in supports if str(item.get("support_type") or "") == "photo" and str(item.get("photo_id") or "").strip()]
        event_ids = [str(item.get("event_id") or "").strip() for item in supports if str(item.get("support_type") or "") == "event" and str(item.get("event_id") or "").strip()]
        photo_ids, photo_urls = _event_photo_links(raw_photo_ids, photo_map)
        payload = copy.deepcopy(row.get("relationship_payload") or {})
        if not isinstance(payload, dict):
            payload = {}
        payload["person_id"] = row.get("person_id")
        payload["relationship_type"] = row.get("relationship_type")
        payload["status"] = row.get("status")
        payload["confidence"] = row.get("confidence")
        payload["intimacy_score"] = row.get("intimacy_score")
        payload["reasoning"] = row.get("reasoning")
        payload["photo_ids"] = photo_ids
        payload["photo_urls"] = photo_urls
        if not payload.get("shared_events"):
            payload["shared_events"] = [event for event in (_minimal_event_ref(event_by_id.get(event_id)) for event_id in event_ids) if event]
        evidence = copy.deepcopy(payload.get("evidence") or {})
        if not isinstance(evidence, dict):
            evidence = {}
        evidence.setdefault("photo_ids", raw_photo_ids)
        evidence.setdefault("event_ids", event_ids)
        evidence.setdefault("person_ids", [row.get("person_id")])
        payload["evidence"] = evidence
        relationships.append(payload)
    return relationships


def _set_nested_field(target: dict, field_key: str, payload: dict) -> None:
    parts = [part for part in str(field_key or "").split(".") if part]
    if not parts:
        return
    cursor = target
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[parts[-1]] = payload


def _rebuild_structured_profile(profile_facts: Sequence[dict]) -> dict:
    structured: dict = {}
    for row in list(profile_facts or []):
        field_key = str(row.get("field_key") or "").strip()
        if not field_key:
            continue
        payload = copy.deepcopy(row.get("fact_payload") or {})
        if not isinstance(payload, dict):
            payload = {}
        payload.setdefault("value", copy.deepcopy(row.get("value_json")))
        payload.setdefault("confidence", row.get("confidence"))
        _set_nested_field(structured, field_key, payload)
    return structured


def _profile_fact_value(profile_facts: Sequence[dict], field_key: str) -> Any:
    for row in list(profile_facts or []):
        if str(row.get("field_key") or "").strip() == field_key:
            return copy.deepcopy(row.get("value_json"))
    return None


def _profile_summary(scope: dict) -> str:
    profile_facts = list(scope.get("profile_facts") or [])
    relationships = list(scope.get("relationships") or [])
    companion = _profile_fact_value(profile_facts, "long_term_facts.relationships.companion")
    close_circle_size = _profile_fact_value(profile_facts, "long_term_facts.relationships.close_circle_size")
    location_anchors = _profile_fact_value(profile_facts, "long_term_facts.geography.location_anchors")
    interests = _profile_fact_value(profile_facts, "long_term_facts.interests.stable_interests")
    parts: List[str] = [f"共识别 {len(relationships)} 条保留关系"]
    if companion:
        parts.append(f"伴侣线索指向 {companion}")
    if close_circle_size not in (None, "", []):
        parts.append(f"亲密圈大小 {close_circle_size}")
    if location_anchors:
        if isinstance(location_anchors, list):
            parts.append("地点锚点 " + ", ".join(str(item) for item in location_anchors[:3]))
        else:
            parts.append(f"地点锚点 {location_anchors}")
    if interests:
        if isinstance(interests, list):
            parts.append("长期兴趣包括 " + ", ".join(str(item) for item in interests[:5]))
        else:
            parts.append(f"长期兴趣包括 {interests}")
    return "；".join(parts)


def _profile_report_markdown(scope: dict) -> str:
    profile_facts = list(scope.get("profile_facts") or [])
    relationships = list(scope.get("relationships") or [])
    companion = _profile_fact_value(profile_facts, "long_term_facts.relationships.companion")
    close_circle_size = _profile_fact_value(profile_facts, "long_term_facts.relationships.close_circle_size")
    location_anchors = _profile_fact_value(profile_facts, "long_term_facts.geography.location_anchors")
    interests = _profile_fact_value(profile_facts, "long_term_facts.interests.stable_interests")
    lines = [
        "# LP3 画像简报",
        "",
        f"- 保留关系数: {len(relationships)}",
    ]
    if companion:
        lines.append(f"- 伴侣: {companion}")
    if close_circle_size not in (None, "", []):
        lines.append(f"- 亲密圈大小: {close_circle_size}")
    if location_anchors:
        if isinstance(location_anchors, list):
            lines.append(f"- 地点锚点: {', '.join(str(item) for item in location_anchors[:3])}")
        else:
            lines.append(f"- 地点锚点: {location_anchors}")
    if interests:
        if isinstance(interests, list):
            lines.append(f"- 长期兴趣: {', '.join(str(item) for item in interests[:5])}")
        else:
            lines.append(f"- 长期兴趣: {interests}")
    if relationships:
        lines.extend(["", "## 关系摘要", ""])
        for row in sorted(relationships, key=lambda item: (-float(item.get("intimacy_score") or 0.0), str(item.get("person_id") or "")))[:8]:
            lines.append(
                f"- {row.get('person_id')}: {row.get('relationship_type') or 'unknown'} / {row.get('status') or 'unknown'} ({float(row.get('confidence') or 0.0):.2f})"
            )
    return "\n".join(lines)


def _build_profiles_from_scope(scope: dict) -> List[dict]:
    profile_facts = list(scope.get("profile_facts") or [])
    if not profile_facts:
        return []
    return [
        {
            "summary": _profile_summary(scope),
            "report_markdown": _profile_report_markdown(scope),
            "structured": _rebuild_structured_profile(profile_facts),
        }
    ]


def _build_faces_payload_from_scope(task: dict, scope: dict, photos: Sequence[dict], photo_map: Dict[str, dict]) -> dict:
    person_rows = list(scope.get("persons") or [])
    face_rows = list(scope.get("faces") or [])
    if not person_rows and not face_rows:
        return {"photos": list(photos or []), "persons": [], "faces": []}

    task_id = str(task.get("task_id") or "").strip()
    persons: List[dict] = []
    for row in sorted(person_rows, key=lambda item: str(item.get("person_id") or "")):
        persons.append(
            {
                "person_id": row.get("person_id"),
                "canonical_name": row.get("canonical_name") or row.get("person_id"),
                "is_primary_person": bool(row.get("is_primary_person")),
                "face_count": int(row.get("face_count") or 0),
                "photo_count": int(row.get("photo_count") or 0),
                "avg_score": row.get("avg_score"),
                "avg_quality": row.get("avg_quality"),
                "high_quality_face_count": int(row.get("high_quality_face_count") or 0),
                "avatar_url": row.get("avatar_url"),
            }
        )

    faces: List[dict] = []
    for row in sorted(face_rows, key=lambda item: (str(item.get("photo_id") or ""), str(item.get("face_id") or ""))):
        raw_photo_id = str(row.get("photo_id") or row.get("source_photo_id") or "").strip()
        photo = photo_map.get(raw_photo_id) or {}
        normalized_photo_id = str((photo or {}).get("photo_id") or f"{task_id}:{raw_photo_id or 'photo'}").strip()
        payload = copy.deepcopy(row.get("face_payload") or {})
        face_urls = {
            "crop": ((payload.get("urls") or {}).get("crop") if isinstance(payload.get("urls"), dict) else None),
            "boxed_full": (
                ((photo or {}).get("urls") or {}).get("boxed_full")
                or payload.get("boxed_image_url")
                or (photo or {}).get("photo_url")
            ),
        }
        faces.append(
            {
                "face_id": row.get("face_id"),
                "person_id": row.get("person_id"),
                "photo_id": normalized_photo_id,
                "source_photo_id": row.get("source_photo_id") or raw_photo_id,
                "bbox": copy.deepcopy(row.get("bbox")),
                "bbox_xywh": copy.deepcopy(row.get("bbox_xywh")),
                "score": row.get("score"),
                "similarity": row.get("similarity"),
                "quality_score": row.get("quality_score"),
                "match_decision": row.get("match_decision"),
                "match_reason": row.get("match_reason"),
                "is_inaccurate": bool(row.get("is_inaccurate")),
                "comment_text": row.get("comment_text"),
                "urls": face_urls,
            }
        )

    return {
        "photos": list(photos or []),
        "persons": persons,
        "faces": faces,
    }


def _build_faces_payload_from_snapshot(task: dict) -> dict:
    result = task.get("result") or {}
    face_payload = result.get("face_recognition")
    if not isinstance(face_payload, dict):
        raise KeyError("当前任务没有 face 输出")

    task_id = str(task.get("task_id") or "").strip()
    photos, raw_to_photo = _photo_maps(task)
    photo_by_source = {
        str(item.get("source_photo_id") or "").strip(): item
        for item in photos
        if str(item.get("source_photo_id") or "").strip()
    }

    person_groups = {
        str(item.get("person_id") or "").strip(): item
        for item in list(face_payload.get("person_groups") or [])
        if isinstance(item, dict) and str(item.get("person_id") or "").strip()
    }
    primary_person_id = str((task.get("result_summary") or {}).get("primary_person_id") or "").strip()

    faces: List[dict] = []
    persons: Dict[str, dict] = {}

    for image in list(face_payload.get("images") or []):
        if not isinstance(image, dict):
            continue
        raw_photo_key = str(image.get("image_id") or image.get("source_hash") or "").strip()
        photo = raw_to_photo.get(raw_photo_key) or photo_by_source.get(raw_photo_key) or {}
        normalized_photo_id = str((photo or {}).get("photo_id") or f"{task_id}:{raw_photo_key or 'photo'}").strip()
        boxed_full_url = (
            ((photo or {}).get("urls") or {}).get("boxed_full")
            or image.get("boxed_image_url")
            or ((photo or {}).get("photo_url"))
        )
        for index, face in enumerate(list(image.get("faces") or []), 1):
            if not isinstance(face, dict):
                continue
            face_id = str(face.get("face_id") or f"{normalized_photo_id}:face_{index:03d}").strip()
            person_id = str(face.get("person_id") or "unknown").strip()
            face_urls = {
                "crop": None,
                "boxed_full": boxed_full_url,
            }
            payload = {
                "face_id": face_id,
                "person_id": person_id,
                "photo_id": normalized_photo_id,
                "source_photo_id": image.get("image_id") or image.get("source_hash"),
                "bbox": copy.deepcopy(face.get("bbox")),
                "bbox_xywh": copy.deepcopy(face.get("bbox_xywh")),
                "score": face.get("score"),
                "similarity": face.get("similarity"),
                "quality_score": face.get("quality_score"),
                "match_decision": face.get("match_decision"),
                "match_reason": face.get("match_reason"),
                "is_inaccurate": bool(face.get("is_inaccurate")),
                "comment_text": face.get("comment_text"),
                "urls": face_urls,
            }
            faces.append(payload)

            bucket = persons.setdefault(
                person_id,
                {
                    "person_id": person_id,
                    "canonical_name": person_id,
                    "is_primary_person": person_id == primary_person_id,
                    "face_ids": [],
                    "photo_ids": [],
                    "face_count": 0,
                    "photo_count": 0,
                    "avg_score": None,
                    "avg_quality": None,
                    "high_quality_face_count": 0,
                    "avatar_url": None,
                    "_scores": [],
                    "_qualities": [],
                },
            )
            bucket["face_ids"].append(face_id)
            bucket["photo_ids"].append(normalized_photo_id)
            if face.get("score") is not None:
                bucket["_scores"].append(float(face.get("score") or 0))
            if face.get("quality_score") is not None:
                quality = float(face.get("quality_score") or 0)
                bucket["_qualities"].append(quality)
                if quality >= 0.8:
                    bucket["high_quality_face_count"] += 1

    serialized_persons: List[dict] = []
    for person_id in sorted(persons):
        bucket = persons[person_id]
        group = person_groups.get(person_id, {})
        scores = list(bucket.pop("_scores", []))
        qualities = list(bucket.pop("_qualities", []))
        bucket["face_ids"] = _unique(bucket.get("face_ids") or [])
        bucket["photo_ids"] = _unique(bucket.get("photo_ids") or [])
        bucket["face_count"] = int(group.get("face_count") or len(bucket["face_ids"]))
        bucket["photo_count"] = int(group.get("photo_count") or len(bucket["photo_ids"]))
        bucket["avg_score"] = round(sum(scores) / len(scores), 4) if scores else group.get("avg_score")
        bucket["avg_quality"] = round(sum(qualities) / len(qualities), 4) if qualities else group.get("avg_quality")
        bucket["avatar_url"] = group.get("avatar_url")
        serialized_persons.append(bucket)

    return {
        "photos": photos,
        "persons": serialized_persons,
        "faces": faces,
    }


def _person_summaries_from_faces(faces_payload: dict) -> List[dict]:
    return [
        {
            "person_id": item.get("person_id"),
            "canonical_name": item.get("canonical_name"),
            "is_primary_person": item.get("is_primary_person"),
        }
        for item in list(faces_payload.get("persons") or [])
    ]


def _event_photo_links(raw_ids: Sequence[str], photo_map: Dict[str, dict]) -> tuple[List[str], List[str]]:
    photo_rows: List[dict] = []
    for raw_id in list(raw_ids or []):
        serialized = photo_map.get(str(raw_id or "").strip())
        if serialized:
            photo_rows.append(serialized)
    return (
        _unique(str(item.get("photo_id") or "") for item in photo_rows),
        _unique(str(item.get("photo_url") or "") for item in photo_rows),
    )


def _build_faces_views(task: dict) -> dict:
    scope = _query_store_subset(task, "faces")
    photos, photo_map = _canonical_photo_assets(task, scope)
    faces_payload = _build_faces_payload_from_scope(task, scope, photos, photo_map)
    return {
        "scope": scope,
        "faces_payload": faces_payload,
    }


def _build_events_views(task: dict) -> dict:
    scope = _query_store_subset(task, "events")
    _, photo_map = _canonical_photo_assets(task, scope)
    return {
        "scope": scope,
        "events": _build_events_from_scope(scope, photo_map),
    }


def _build_vlm_views(task: dict) -> dict:
    scope = _query_store_subset(task, "vlm")
    photos, photo_map = _canonical_photo_assets(task, scope)
    return {
        "scope": scope,
        "photos": photos,
        "vlm_items": _build_vlm_from_scope(scope, photo_map),
    }


def _build_profiles_views(task: dict) -> dict:
    scope = _query_store_subset(task, "profiles")
    return {
        "scope": scope,
        "profiles": _build_profiles_from_scope(scope),
    }


def _build_relationship_views(task: dict) -> dict:
    scope = _query_store_subset(task, "relationships")
    photos, photo_map = _canonical_photo_assets(task, scope)
    return {
        "scope": scope,
        "photos": photos,
        "relationships": _build_relationships_from_scope(scope, photo_map),
    }


def _build_bundle_views(task: dict) -> dict:
    scope = _query_store_subset(task, "bundle")
    photos, photo_map = _canonical_photo_assets(task, scope)
    faces_payload = _build_faces_payload_from_scope(task, scope, photos, photo_map)
    return {
        "scope": scope,
        "photos": photos,
        "faces_payload": faces_payload,
        "person_summaries": _person_summaries_from_faces(faces_payload),
        "events": _build_events_from_scope(scope, photo_map),
        "vlm_items": _build_vlm_from_scope(scope, photo_map),
        "profiles": _build_profiles_from_scope(scope),
        "relationships": _build_relationships_from_scope(scope, photo_map),
    }


def build_task_memory_faces_response(task: dict, *, include_artifacts: bool = False) -> dict:
    views = _build_faces_views(task)
    payload = views["faces_payload"]
    response = {
        "query": _query_shell(str(task.get("task_id") or "")),
        "task": _task_shell(task),
        "data": payload,
    }
    if include_artifacts:
        response["artifacts"] = copy.deepcopy((task.get("asset_manifest") or {}).get("files") or [])
    return response


def build_task_memory_events_response(
    task: dict,
    *,
    include_raw: bool = False,
    include_artifacts: bool = False,
    include_traces: bool = False,
) -> dict:
    views = _build_events_views(task)
    result = {
        "query": _query_shell(str(task.get("task_id") or ""), include_raw=include_raw, include_artifacts=include_artifacts, include_traces=include_traces),
        "task": _task_shell(task),
        "data": {
            "events": views["events"],
        },
    }
    if include_raw:
        result["data"]["raw_events"] = [
            copy.deepcopy(item.get("event_payload") or {})
            for item in list((views["scope"] or {}).get("events") or [])
        ]
    if include_artifacts:
        result["artifacts"] = copy.deepcopy((task.get("asset_manifest") or {}).get("files") or [])
    return result


def build_task_memory_vlm_response(
    task: dict,
    *,
    include_raw: bool = False,
    include_artifacts: bool = False,
    include_traces: bool = False,
) -> dict:
    views = _build_vlm_views(task)
    result = {
        "query": _query_shell(str(task.get("task_id") or ""), include_raw=include_raw, include_artifacts=include_artifacts, include_traces=include_traces),
        "task": _task_shell(task),
        "data": {
            "photos": views["photos"],
            "vlm_items": views["vlm_items"],
        },
    }
    if include_raw:
        result["data"]["raw_vlm"] = [
            copy.deepcopy(item.get("evidence_payload") or {})
            for item in list((views["scope"] or {}).get("evidence") or [])
            if str(item.get("source_stage") or "") == "vp1"
        ]
    if include_artifacts:
        result["artifacts"] = copy.deepcopy((task.get("asset_manifest") or {}).get("files") or [])
    return result


def build_task_memory_profiles_response(
    task: dict,
    *,
    include_raw: bool = False,
    include_artifacts: bool = False,
    include_traces: bool = False,
) -> dict:
    views = _build_profiles_views(task)
    result = {
        "query": _query_shell(str(task.get("task_id") or ""), include_raw=include_raw, include_artifacts=include_artifacts, include_traces=include_traces),
        "task": _task_shell(task),
        "data": {
            "profiles": views["profiles"],
        },
    }
    if include_raw:
        result["data"]["raw_profiles"] = {
            "profile_facts": [copy.deepcopy(item) for item in list((views["scope"] or {}).get("profile_facts") or [])]
        }
    if include_artifacts:
        result["artifacts"] = copy.deepcopy((task.get("asset_manifest") or {}).get("files") or [])
    return result


def build_task_memory_relationships_response(
    task: dict,
    *,
    include_raw: bool = False,
    include_artifacts: bool = False,
    include_traces: bool = False,
) -> dict:
    views = _build_relationship_views(task)
    result = {
        "query": _query_shell(str(task.get("task_id") or ""), include_raw=include_raw, include_artifacts=include_artifacts, include_traces=include_traces),
        "task": _task_shell(task),
        "data": {
            "photos": views["photos"],
            "relationships": views["relationships"],
        },
    }
    if include_raw:
        result["data"]["raw_relationships"] = [
            copy.deepcopy(item.get("relationship_payload") or {})
            for item in list((views["scope"] or {}).get("relationships") or [])
        ]
    if include_artifacts:
        result["artifacts"] = copy.deepcopy((task.get("asset_manifest") or {}).get("files") or [])
    return result


def build_task_memory_bundle_response(
    task: dict,
    *,
    include_raw: bool = False,
    include_artifacts: bool = False,
    include_traces: bool = False,
) -> dict:
    views = _build_bundle_views(task)

    bundle = {
        "query": _query_shell(str(task.get("task_id") or ""), include_raw=include_raw, include_artifacts=include_artifacts, include_traces=include_traces),
        "task": _task_shell(task),
        "data": {
            "photos": views["photos"],
            "persons": views["faces_payload"].get("persons") or [],
            "faces": views["faces_payload"].get("faces") or [],
            "events": views["events"],
            "vlm_items": views["vlm_items"],
            "profiles": views["profiles"],
            "relationships": views["relationships"],
        },
    }
    if include_raw:
        bundle["data"]["raw_events"] = [
            copy.deepcopy(item.get("event_payload") or {})
            for item in list((views["scope"] or {}).get("events") or [])
        ]
        bundle["data"]["raw_vlm"] = [
            copy.deepcopy(item.get("evidence_payload") or {})
            for item in list((views["scope"] or {}).get("evidence") or [])
            if str(item.get("source_stage") or "") == "vp1"
        ]
        bundle["data"]["raw_profiles"] = {
            "profile_facts": [copy.deepcopy(item) for item in list((views["scope"] or {}).get("profile_facts") or [])]
        }
        bundle["data"]["raw_relationships"] = [
            copy.deepcopy(item.get("relationship_payload") or {})
            for item in list((views["scope"] or {}).get("relationships") or [])
        ]
    if include_artifacts:
        bundle["artifacts"] = copy.deepcopy((task.get("asset_manifest") or {}).get("files") or [])
    return bundle
