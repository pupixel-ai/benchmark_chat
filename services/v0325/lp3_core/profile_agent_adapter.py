from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .evidence_utils import extract_ids_from_refs
from .types import RelationshipRecord


RELATIONSHIP_DIMENSION_MAP = {
    "romantic": "关系>伴侣",
    "family": "关系>亲属",
    "bestie": "关系>密友",
    "close_friend": "关系>密友",
}

PROFILE_FIELD_DIMENSION_MAP = {
    "long_term_facts.social_identity.education": "社会身份>教育背景",
    "long_term_facts.social_identity.career": "社会身份>职业/专业",
    "long_term_facts.social_identity.career_phase": "社会身份>职业阶段",
    "long_term_facts.material.asset_level": "物质>资产",
    "long_term_facts.geography.location_anchors": "地理>地点锚点",
    "long_term_facts.geography.mobility_pattern": "出行>mobility_pattern",
    "long_term_facts.relationships.parenting": "家庭状况>parenting",
    "long_term_facts.relationships.pets": "物质>宠物归属",
    "long_term_facts.hobbies.frequent_activities": "长期爱好>活动",
}

PROFILE_DIMENSION_FIELD_PATHS = {
    dimension: field_key for field_key, dimension in PROFILE_FIELD_DIMENSION_MAP.items()
}


def build_profile_agent_extractor_outputs(
    *,
    primary_decision: Dict[str, Any] | None,
    relationships: Iterable[RelationshipRecord | Dict[str, Any]],
    structured_profile: Dict[str, Any] | None,
) -> Dict[str, Dict[str, Any]]:
    relationship_list = list(relationships)
    protagonist_tags = _build_protagonist_tags(primary_decision or {})
    relationship_tags = _build_relationship_tags(relationship_list)
    profile_tags = _build_profile_tags(structured_profile or {})

    return {
        "protagonist": {
            "agent_type": "protagonist",
            "tags": protagonist_tags,
            "reasoning_trace": str(primary_decision.get("reasoning") or "") if primary_decision else "",
        },
        "relationship": {
            "agent_type": "relationship",
            "tags": relationship_tags,
            "reasoning_trace": "\n".join(
                _coerce_relationship_dict(relationship).get("reasoning", "")
                for relationship in relationship_list
                if _coerce_relationship_dict(relationship).get("relationship_type") in RELATIONSHIP_DIMENSION_MAP
            ).strip(),
        },
        "profile": {
            "agent_type": "profile",
            "tags": profile_tags,
            "reasoning_trace": "\n".join(
                f"[{tag['dimension']}] {tag.get('reasoning', '')}".strip()
                for tag in profile_tags
                if tag.get("reasoning")
            ).strip(),
        },
    }


def _build_protagonist_tags(primary_decision: Dict[str, Any]) -> List[Dict[str, Any]]:
    if primary_decision.get("mode") != "person_id":
        return []
    value = primary_decision.get("primary_person_id")
    if not value:
        return []
    reasoning = str(primary_decision.get("reasoning") or "")
    evidence = _build_downstream_evidence(primary_decision.get("evidence") or {}, reasoning)
    if not evidence:
        return []
    return [
        _tag_payload(
            dimension="主角>身份确认",
            value=value,
            confidence=_scale_confidence(primary_decision.get("confidence")),
            stability="long_term",
            evidence=evidence,
            extraction_gap=_derive_extraction_gap(primary_decision.get("evidence") or {}, primary_decision.get("confidence"), reasoning),
            reasoning=reasoning,
        )
    ]


def _build_relationship_tags(relationships: List[RelationshipRecord | Dict[str, Any]]) -> List[Dict[str, Any]]:
    tags: List[Dict[str, Any]] = []
    for relationship in relationships:
        rel = _coerce_relationship_dict(relationship)
        dimension = RELATIONSHIP_DIMENSION_MAP.get(str(rel.get("relationship_type") or ""))
        if not dimension:
            continue
        reasoning = str(rel.get("reasoning") or "")
        evidence_payload = dict(rel.get("evidence") or {})
        shared_event_ids = [event.get("event_id") for event in rel.get("shared_events", []) if event.get("event_id")]
        if shared_event_ids:
            merged_event_ids = list(dict.fromkeys(shared_event_ids + list(evidence_payload.get("event_ids", []) or [])))
            evidence_payload["event_ids"] = merged_event_ids
        evidence = _build_downstream_evidence(evidence_payload, reasoning)
        if not evidence:
            continue
        tags.append(
            _tag_payload(
                dimension=dimension,
                value=str(rel.get("person_id") or ""),
                confidence=_scale_confidence(rel.get("confidence")),
                stability=_relationship_stability(str(rel.get("status") or "")),
                evidence=evidence,
                extraction_gap=_derive_extraction_gap(evidence_payload, rel.get("confidence"), reasoning),
                reasoning=reasoning,
            )
        )
    return tags


def _build_profile_tags(structured_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    tags: List[Dict[str, Any]] = []
    for field_key, dimension in PROFILE_FIELD_DIMENSION_MAP.items():
        tag_object = _get_nested(structured_profile, field_key)
        tag = _profile_tag_from_object(field_key, dimension, tag_object)
        if tag:
            tags.append(tag)
    return tags


def _profile_tag_from_object(field_key: str, dimension: str, tag_object: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if not isinstance(tag_object, dict):
        return None
    value = tag_object.get("value")
    if value in (None, "", []):
        return None
    reasoning = str(tag_object.get("reasoning") or "")
    evidence_payload = dict(tag_object.get("evidence") or {})
    evidence = _build_downstream_evidence(evidence_payload, reasoning)
    if not evidence:
        return None
    return _tag_payload(
        dimension=dimension,
        value=_value_to_string(value),
        confidence=_scale_confidence(tag_object.get("confidence")),
        stability=_stability_for_field_key(field_key),
        evidence=evidence,
        extraction_gap=_derive_extraction_gap(evidence_payload, tag_object.get("confidence"), reasoning),
        reasoning=reasoning,
    )


def _tag_payload(
    *,
    dimension: str,
    value: str,
    confidence: int,
    stability: str,
    evidence: List[Dict[str, Any]],
    extraction_gap: str,
    reasoning: str,
) -> Dict[str, Any]:
    return {
        "dimension": dimension,
        "value": value,
        "confidence": confidence,
        "stability": stability,
        "evidence": evidence,
        "extraction_gap": extraction_gap,
        "reasoning": reasoning,
    }


def _build_downstream_evidence(main_evidence: Dict[str, Any], fallback_description: str) -> List[Dict[str, Any]]:
    supporting_refs = list(main_evidence.get("supporting_refs") or [])
    if supporting_refs:
        evidence_items = [_convert_ref_to_downstream_evidence(ref, fallback_description) for ref in supporting_refs]
        return [item for item in evidence_items if _has_any_anchor(item)]
    synthesized = _synthesize_from_top_level(main_evidence, fallback_description)
    return [item for item in synthesized if _has_any_anchor(item)]


def _convert_ref_to_downstream_evidence(ref: Dict[str, Any], fallback_description: str) -> Dict[str, Any]:
    ids = extract_ids_from_refs([ref])
    event_id = ids["event_ids"][0] if ids["event_ids"] else None
    photo_id = ids["photo_ids"][0] if ids["photo_ids"] else None
    person_id = ids["person_ids"][0] if ids["person_ids"] else None
    feature_names = ids["feature_names"]
    evidence_type = _classify_evidence_type(event_id, photo_id, feature_names)
    inference_depth = _infer_depth(event_id, photo_id, person_id, feature_names)
    description = str(ref.get("signal") or ref.get("description") or fallback_description or "主工程输出证据")
    return {
        "event_id": event_id,
        "photo_id": photo_id,
        "person_id": person_id,
        "feature_names": feature_names,
        "description": description,
        "evidence_type": evidence_type,
        "inference_depth": inference_depth,
    }


def _synthesize_from_top_level(main_evidence: Dict[str, Any], fallback_description: str) -> List[Dict[str, Any]]:
    event_ids = list(main_evidence.get("event_ids") or [])
    photo_ids = list(main_evidence.get("photo_ids") or [])
    person_ids = list(main_evidence.get("person_ids") or [])
    feature_names = list(main_evidence.get("feature_names") or [])
    max_len = max(len(event_ids), len(photo_ids), len(person_ids), 1)
    evidence_items: List[Dict[str, Any]] = []
    for index in range(max_len):
        event_id = event_ids[index] if index < len(event_ids) else None
        photo_id = photo_ids[index] if index < len(photo_ids) else None
        person_id = person_ids[index] if index < len(person_ids) else None
        item_feature_names = feature_names if index == 0 else []
        evidence_items.append(
            {
                "event_id": event_id,
                "photo_id": photo_id,
                "person_id": person_id,
                "feature_names": item_feature_names,
                "description": fallback_description or "主工程输出证据",
                "evidence_type": _classify_evidence_type(event_id, photo_id, item_feature_names),
                "inference_depth": _infer_depth(event_id, photo_id, person_id, item_feature_names),
            }
        )
    return evidence_items


def _classify_evidence_type(event_id: str | None, photo_id: str | None, feature_names: List[str]) -> str:
    if event_id or photo_id:
        return "direct"
    if feature_names:
        return "inferred"
    return "inferred"


def _infer_depth(
    event_id: str | None,
    photo_id: str | None,
    person_id: str | None,
    feature_names: List[str],
) -> int:
    if event_id or photo_id:
        return 1
    if person_id and not feature_names:
        return 2
    if feature_names:
        return 3
    return 2


def _has_any_anchor(evidence: Dict[str, Any]) -> bool:
    return bool(
        evidence.get("event_id")
        or evidence.get("photo_id")
        or evidence.get("person_id")
        or evidence.get("feature_names")
    )


def _coerce_relationship_dict(relationship: RelationshipRecord | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(relationship, RelationshipRecord):
        return {
            "person_id": relationship.person_id,
            "relationship_type": relationship.relationship_type,
            "intimacy_score": relationship.intimacy_score,
            "status": relationship.status,
            "confidence": relationship.confidence,
            "reasoning": relationship.reasoning,
            "shared_events": relationship.shared_events,
            "evidence": relationship.evidence,
        }
    return dict(relationship)


def _scale_confidence(confidence: Any) -> int:
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        return 0
    if value <= 1:
        value *= 100
    return max(0, min(100, round(value)))


def _relationship_stability(status: str) -> str:
    return "short_term" if status in {"new", "growing"} else "long_term"


def _stability_for_field_key(field_key: str) -> str:
    return "short_term" if field_key.startswith("short_term_") else "long_term"


def _derive_extraction_gap(evidence: Dict[str, Any], confidence: Any, reasoning: str) -> str:
    constraint_notes = list(evidence.get("constraint_notes") or [])
    if constraint_notes:
        return "；".join(str(note) for note in constraint_notes if note)
    scaled = _scale_confidence(confidence)
    if scaled and scaled < 70:
        return f"当前标签置信度为 {scaled}，仍需更多稳定证据。"
    if reasoning and "不足" in reasoning:
        return reasoning
    return ""


def _value_to_string(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item or "").strip())
    return str(value)


def _get_nested(payload: Dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def _iter_mapped_profile_tag_objects(structured_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    tag_objects: List[Dict[str, Any]] = []
    for field_path in PROFILE_FIELD_DIMENSION_MAP:
        tag_object = _get_nested(structured_profile, field_path)
        if isinstance(tag_object, dict) and tag_object.get("value") not in (None, "", []):
            tag_objects.append(tag_object)
    return tag_objects
