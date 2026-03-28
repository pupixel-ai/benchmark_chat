from __future__ import annotations

from typing import Any, Dict, Iterable, List

from models import Relationship

from .evidence_utils import extract_ids_from_refs


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


def _build_cross_layer_summary(
    agent_type: str,
    primary_decision: Dict[str, Any] | None,
    relationship_list: List[Dict[str, Any] | Relationship],
    structured_profile: Dict[str, Any] | None,
) -> str:
    """为指定 agent_type 生成其他两层的摘要，注入 reasoning_trace。"""
    pd = primary_decision or {}
    sp = structured_profile or {}
    parts: List[str] = []

    def _protagonist_summary() -> str:
        mode = pd.get("mode", "unknown")
        pid = pd.get("primary_person_id", "unknown")
        conf = pd.get("confidence", 0)
        ev = pd.get("evidence") or {}
        selfie = int(ev.get("selfie_count", 0) or 0)
        anchor = int(ev.get("identity_anchor_count", 0) or 0)
        return f"主角: {pid} (mode={mode}, confidence={conf:.2f}, selfie={selfie}, anchor={anchor})"

    def _relationship_summary() -> str:
        rel_parts: List[str] = []
        for rel in relationship_list:
            r = _coerce_relationship_dict(rel)
            rtype = r.get("relationship_type")
            if rtype in RELATIONSHIP_DIMENSION_MAP:
                pid = r.get("person_id", "?")
                conf = r.get("confidence", 0)
                rel_parts.append(f"{pid}({rtype},conf={conf})")
        if not rel_parts:
            return "关系层: 无核心关系"
        return f"关系层: {', '.join(rel_parts[:5])}，共 {len(relationship_list)} 段关系"

    def _profile_summary() -> str:
        fields: List[str] = []
        for field_key in ("long_term_facts.identity.role", "long_term_facts.social_identity.education",
                          "long_term_facts.geography.location_anchors", "long_term_facts.hobbies.interests"):
            obj = _get_nested_value(sp, field_key)
            if isinstance(obj, dict) and obj.get("value") is not None:
                fields.append(f"{field_key.split('.')[-1]}={obj['value']}")
        return f"画像层: {', '.join(fields)}" if fields else "画像层: 无已确认字段"

    if agent_type == "protagonist":
        parts.append(_relationship_summary())
        parts.append(_profile_summary())
    elif agent_type == "relationship":
        parts.append(_protagonist_summary())
        parts.append(_profile_summary())
    elif agent_type == "profile":
        parts.append(_protagonist_summary())
        parts.append(_relationship_summary())

    return "\n[跨层上下文]\n" + "\n".join(f"- {p}" for p in parts) if parts else ""


def _get_nested_value(d: Dict[str, Any], dotted_key: str) -> Any:
    current: Any = d
    for key in dotted_key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def build_profile_agent_extractor_outputs(
    *,
    primary_decision: Dict[str, Any] | None,
    relationships: Iterable[Relationship | Dict[str, Any]],
    structured_profile: Dict[str, Any] | None,
) -> Dict[str, Dict[str, Any]]:
    relationship_list = list(relationships)
    protagonist_tags = _build_protagonist_tags(primary_decision or {})
    relationship_tags = _build_relationship_tags(relationship_list)
    profile_tags = _build_profile_tags(structured_profile or {})

    protagonist_trace = str(primary_decision.get("reasoning") or "") if primary_decision else ""
    relationship_trace = "\n".join(
        _coerce_relationship_dict(relationship).get("reasoning", "")
        for relationship in relationship_list
        if _coerce_relationship_dict(relationship).get("relationship_type") in RELATIONSHIP_DIMENSION_MAP
    ).strip()
    profile_trace = "\n".join(
        f"[{tag['dimension']}] {tag.get('reasoning', '')}".strip()
        for tag in profile_tags
        if tag.get("reasoning")
    ).strip()

    protagonist_cross = _build_cross_layer_summary("protagonist", primary_decision, relationship_list, structured_profile)
    relationship_cross = _build_cross_layer_summary("relationship", primary_decision, relationship_list, structured_profile)
    profile_cross = _build_cross_layer_summary("profile", primary_decision, relationship_list, structured_profile)

    return {
        "protagonist": {
            "agent_type": "protagonist",
            "tags": protagonist_tags,
            "reasoning_trace": f"{protagonist_trace}{protagonist_cross}" if protagonist_cross else protagonist_trace,
        },
        "relationship": {
            "agent_type": "relationship",
            "tags": relationship_tags,
            "reasoning_trace": f"{relationship_trace}{relationship_cross}" if relationship_cross else relationship_trace,
        },
        "profile": {
            "agent_type": "profile",
            "tags": profile_tags,
            "reasoning_trace": f"{profile_trace}{profile_cross}" if profile_cross else profile_trace,
        },
    }


def _build_protagonist_tags(primary_decision: Dict[str, Any]) -> List[Dict[str, Any]]:
    if primary_decision.get("mode") != "person_id":
        return []
    value = primary_decision.get("primary_person_id")
    if not value:
        return []
    reasoning = str(primary_decision.get("reasoning") or "")
    raw_evidence = primary_decision.get("evidence") or {}
    evidence = _build_downstream_evidence(raw_evidence, reasoning)
    if not evidence:
        return []
    stats_parts = []
    selfie_count = int(raw_evidence.get("selfie_count", 0) or 0)
    anchor_count = int(raw_evidence.get("identity_anchor_count", 0) or 0)
    photo_count = int(raw_evidence.get("photo_count", 0) or 0)
    label_count = int(raw_evidence.get("protagonist_label_count", 0) or 0)
    if selfie_count:
        stats_parts.append(f"自拍 {selfie_count} 次")
    if anchor_count:
        stats_parts.append(f"身份锚点 {anchor_count} 个")
    if photo_count:
        stats_parts.append(f"出现在 {photo_count} 张照片中")
    if label_count:
        stats_parts.append(f"VLM 主角标记 {label_count} 次")
    if stats_parts:
        evidence.append(
            _normalize_evidence_item(
                {
                    "description": f"主角识别统计: {', '.join(stats_parts)}",
                    "evidence_type": "direct",
                    "inference_depth": 1,
                    "feature_names": ["selfie_count", "identity_anchor_count", "photo_count", "protagonist_label_count"],
                }
            )
        )
        reasoning = f"{reasoning}\n[统计] {', '.join(stats_parts)}" if reasoning else f"[统计] {', '.join(stats_parts)}"
    return [
        _tag_payload(
            dimension="主角>身份确认",
            value=value,
            confidence=_scale_confidence(primary_decision.get("confidence")),
            stability="long_term",
            evidence=evidence,
            extraction_gap=_derive_extraction_gap(raw_evidence, primary_decision.get("confidence"), reasoning),
            reasoning=reasoning,
        )
    ]


def _build_relationship_tags(relationships: List[Relationship | Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def _coerce_relationship_dict(relationship: Relationship | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(relationship, Relationship):
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
        selfie_count = int(evidence.get("selfie_count", 0) or 0)
        anchor_count = int(evidence.get("identity_anchor_count", 0) or 0)
        basis_parts = []
        if selfie_count:
            basis_parts.append(f"自拍 {selfie_count} 次")
        if anchor_count:
            basis_parts.append(f"身份锚点 {anchor_count} 个")
        basis = f"，基于 {', '.join(basis_parts)}" if basis_parts else ""
        return f"当前标签置信度为 {scaled}{basis}。"
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
