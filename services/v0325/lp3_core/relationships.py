from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Tuple

from .evidence_utils import build_evidence_payload, extract_ids_from_refs
from .types import MemoryState, RelationshipDossier, RelationshipRecord, RelationshipTypeSpec


STRONG_CONTACT_SIGNALS = {
    "selfie_together",
    "shoulder_lean",
    "hug",
    "kiss",
    "holding_hands",
    "arm_in_arm",
}
EMERGING_INTIMATE_KEYWORDS = ("独处", "单独", "亲密", "搂", "拥抱", "kiss", "1v1")
MEDIATED_SIGNAL_KEYWORDS = ("screen", "屏幕", "displayed_in", "appearing_in", "数字虚拟", "虚拟", "app", "报告")
INTIMATE_SCENE_KEYWORDS = ("家", "home", "卧室", "bedroom", "宿舍", "dorm", "酒店", "hotel")
FUNCTIONAL_SCENE_KEYWORDS = ("课堂", "教室", "校园", "办公室", "工位", "实验室", "meeting", "office", "class")
RELATIONSHIP_TYPES = {
    "family",
    "romantic",
    "bestie",
    "close_friend",
    "friend",
    "classmate_colleague",
    "activity_buddy",
    "acquaintance",
}
LOW_SIGNAL_RELATIONSHIP_TYPES = {"activity_buddy", "acquaintance", "classmate_colleague"}


RELATIONSHIP_TYPE_SPECS: Dict[str, RelationshipTypeSpec] = {
    "family": RelationshipTypeSpec(
        type_key="family",
        allowed_evidence=["event", "scene", "interaction", "time"],
        strong_evidence=["家庭场景", "照护行为", "代际结构"],
        supporting_evidence=["跨时段家庭叙事", "居家场景重复"],
        blocker_evidence=["活动现场儿童", "公共场景随机同框", "服务性角色"],
        cot_steps=["先确认家庭结构或照护闭环", "再排除活动/路人误判"],
        reflection_questions=["是否更像 close_friend 或 acquaintance"],
        downgrade_target="close_friend",
    ),
    "romantic": RelationshipTypeSpec(
        type_key="romantic",
        allowed_evidence=["event", "scene", "interaction", "time"],
        strong_evidence=["亲密接触", "稳定 1v1 私密场景"],
        supporting_evidence=["跨月高频共现", "周末或夜间重复互动"],
        blocker_evidence=["群体活动主导", "单次高光事件", "弱互动"],
        cot_steps=["先看亲密硬信号", "再排除单次/活动偏置"],
        reflection_questions=["移除弱证据后是否还能支撑 romantic"],
        downgrade_target="close_friend",
    ),
    "bestie": RelationshipTypeSpec(
        type_key="bestie",
        allowed_evidence=["event", "scene", "interaction", "time"],
        strong_evidence=["高频", "多场景", "强互动"],
        supporting_evidence=["周末/夜间稳定互动", "重复 1v1"],
        blocker_evidence=["功能性场景主导"],
        cot_steps=["先看高频跨场景", "再排除同学同事功能关系"],
        reflection_questions=["是否仅 close_friend"],
        downgrade_target="close_friend",
    ),
    "close_friend": RelationshipTypeSpec(
        type_key="close_friend",
        allowed_evidence=["event", "scene", "interaction", "time"],
        strong_evidence=["中高频重复互动", "多场景共现"],
        supporting_evidence=["with_user_only", "private_scene_ratio"],
        blocker_evidence=["单次同框", "无互动", "单场景"],
        cot_steps=["先看重复同框与互动", "再排除 activity_buddy/classmate"],
        reflection_questions=["是否应降到 friend"],
        downgrade_target="friend",
    ),
    "friend": RelationshipTypeSpec(
        type_key="friend",
        allowed_evidence=["event", "scene", "interaction", "time"],
        strong_evidence=["重复出现", "轻中度互动"],
        supporting_evidence=["shared_events >= 2"],
        blocker_evidence=["纯群体偶遇"],
        cot_steps=["先确认重复接触", "再确认不足以升级 close_friend"],
        reflection_questions=["是否仅 acquaintance"],
        downgrade_target="acquaintance",
    ),
    "classmate_colleague": RelationshipTypeSpec(
        type_key="classmate_colleague",
        allowed_evidence=["event", "scene", "time"],
        strong_evidence=["功能场景主导"],
        supporting_evidence=["工作日白天重复出现"],
        blocker_evidence=["私密生活场景占比高"],
        cot_steps=["先判功能场景占比", "再确认非私密关系主线"],
        reflection_questions=["是否更像 friend"],
        downgrade_target="friend",
    ),
    "activity_buddy": RelationshipTypeSpec(
        type_key="activity_buddy",
        allowed_evidence=["event", "scene", "time"],
        strong_evidence=["单一活动主线反复出现"],
        supporting_evidence=["活动场景重复"],
        blocker_evidence=["跨生活场景稳定共现"],
        cot_steps=["先看活动主线重复", "再排除私生活主线"],
        reflection_questions=["是否 friend/acquaintance"],
        downgrade_target="friend",
    ),
    "acquaintance": RelationshipTypeSpec(
        type_key="acquaintance",
        allowed_evidence=["event", "scene", "time"],
        strong_evidence=["低频", "群体主导", "排他性弱"],
        supporting_evidence=["shared_events 少", "互动弱"],
        blocker_evidence=["1v1私密场景", "跨场景稳定同框"],
        cot_steps=["先确认低频弱互动", "再确认不满足更高标签"],
        reflection_questions=["是否可升级为 friend"],
        downgrade_target=None,
    ),
}


def build_relationship_dossiers(state: MemoryState, llm_processor: Any) -> List[RelationshipDossier]:
    primary_person_id = (state.primary_decision or {}).get("primary_person_id")
    dossiers: List[RelationshipDossier] = []
    for person_id in state.face_db.keys():
        if person_id == primary_person_id:
            continue
        screening = (state.screening or {}).get(person_id)
        if screening and screening.memory_value == "block":
            continue
        evidence = _safe_collect_relationship_evidence(
            person_id=person_id,
            state=state,
            llm_processor=llm_processor,
        )
        evidence_refs = _build_evidence_refs(person_id, evidence, state.vlm_results)
        person_kind = screening.person_kind if screening else "uncertain"
        memory_value = screening.memory_value if screening else "candidate"
        dossiers.append(
            RelationshipDossier(
                person_id=person_id,
                person_kind=person_kind,
                memory_value=memory_value,
                photo_count=int(evidence.get("photo_count", 0) or 0),
                time_span_days=int(evidence.get("time_span_days", 0) or 0),
                recent_gap_days=int(evidence.get("recent_gap_days", 0) or 0),
                monthly_frequency=float(evidence.get("monthly_frequency", 0.0) or 0.0),
                scene_profile={
                    "scenes": list(evidence.get("scenes", []) or []),
                    "private_scene_ratio": float(evidence.get("private_scene_ratio", 0.0) or 0.0),
                    "dominant_scene_ratio": float(evidence.get("dominant_scene_ratio", 0.0) or 0.0),
                    "with_user_only": bool(evidence.get("with_user_only", False)),
                },
                interaction_signals=list(evidence.get("interaction_behavior", []) or [])
                + list(evidence.get("contact_types", []) or []),
                shared_events=list(evidence.get("rela_events", []) or []),
                trend_detail=dict(evidence.get("trend_detail", {}) or {}),
                co_appearing_persons=list(evidence.get("co_appearing_persons", []) or []),
                anomalies=list(evidence.get("anomalies", []) or []),
                evidence_refs=evidence_refs,
                block_reasons=list(screening.block_reasons if screening else []),
            )
        )
    return dossiers


def infer_relationships_from_dossiers(
    state: MemoryState,
    llm_processor: Any,
    dossiers: List[RelationshipDossier],
) -> Tuple[List[RelationshipRecord], List[RelationshipDossier]]:
    relationships: List[RelationshipRecord] = []
    updated_dossiers: List[RelationshipDossier] = []
    for dossier in dossiers:
        relationship, reflection = _infer_relationship_from_dossier(dossier, llm_processor=llm_processor)
        relationship, reflection = _apply_relationship_type_veto(dossier, relationship, reflection)
        relationship = _apply_status_correction(dossier, relationship)
        relationship = _normalize_relationship_output(dossier, relationship, reflection)

        updated_dossier = replace(dossier)
        updated_dossier.relationship_reflection = reflection
        updated_dossier.retention_decision, updated_dossier.retention_reason = _determine_retention(updated_dossier, relationship)
        updated_dossier.group_eligible, updated_dossier.group_block_reason, updated_dossier.group_weight = _determine_group_eligibility(
            dossier=updated_dossier,
            relationship=relationship,
        )
        updated_dossier.relationship_result = {
            "relationship_type": relationship.relationship_type,
            "status": relationship.status,
            "confidence": relationship.confidence,
            "reasoning": relationship.reasoning,
        }
        updated_dossiers.append(updated_dossier)
        if updated_dossier.retention_decision == "keep":
            relationships.append(relationship)
    return relationships, updated_dossiers


def select_group_candidates(
    relationships: List[RelationshipRecord],
    dossiers: List[RelationshipDossier],
    confidence_threshold: float = 0.75,
) -> List[RelationshipRecord]:
    dossier_by_person_id = {d.person_id: d for d in dossiers}
    selected: List[RelationshipRecord] = []
    for relationship in relationships:
        dossier = dossier_by_person_id.get(relationship.person_id)
        if not dossier:
            continue
        if relationship.confidence < confidence_threshold:
            continue
        if dossier.retention_decision != "keep":
            continue
        if not dossier.group_eligible:
            continue
        selected.append(relationship)
    return selected


def _safe_collect_relationship_evidence(
    person_id: str,
    state: MemoryState,
    llm_processor: Any,
) -> Dict[str, Any]:
    default = {
        "photo_count": 0,
        "time_span": "",
        "time_span_days": 0,
        "recent_gap_days": 0,
        "scenes": [],
        "private_scene_ratio": 0.0,
        "dominant_scene_ratio": 0.0,
        "interaction_behavior": [],
        "weekend_frequency": "none",
        "with_user_only": True,
        "sample_scenes": [],
        "contact_types": [],
        "rela_events": _collect_shared_events_from_state(state, person_id),
        "monthly_frequency": 0.0,
        "trend_detail": {},
        "co_appearing_persons": [],
        "anomalies": [],
    }
    if llm_processor and hasattr(llm_processor, "_collect_relationship_evidence"):
        try:
            evidence = llm_processor._collect_relationship_evidence(person_id, state.vlm_results, state.events)
            if isinstance(evidence, dict):
                merged = dict(default)
                merged.update(evidence)
                if not merged.get("rela_events"):
                    merged["rela_events"] = _collect_shared_events_from_state(state, person_id)
                return merged
        except Exception:
            pass
    return default

def _collect_shared_events_from_state(state: MemoryState, person_id: str) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    for event in state.events or []:
        participants = list(getattr(event, "participants", []) or [])
        if person_id not in participants:
            continue
        collected.append(
            {
                "event_id": getattr(event, "event_id", ""),
                "date": getattr(event, "date", ""),
                "title": getattr(event, "title", ""),
                "location": getattr(event, "location", ""),
                "narrative_synthesis": getattr(event, "narrative_synthesis", ""),
                "description": getattr(event, "description", ""),
                "participants": participants,
                "social_dynamics": list(getattr(event, "social_dynamics", []) or []),
            }
        )
    return collected


def _infer_relationship_from_dossier(
    dossier: RelationshipDossier,
    llm_processor: Any,
) -> Tuple[RelationshipRecord, Dict[str, Any]]:
    response = _run_relationship_llm(dossier, llm_processor)
    baseline_relationship = _run_legacy_relationship_baseline(dossier, llm_processor)
    relationship_type = _normalize_relationship_type(response.get("relationship_type")) if response else ""
    if not relationship_type and baseline_relationship:
        relationship_type = _normalize_relationship_type(baseline_relationship.relationship_type)
    if not relationship_type:
        relationship_type = _heuristic_relationship_type(dossier)
    status = _normalize_status(response, dossier)
    if not response and baseline_relationship and baseline_relationship.status:
        status = baseline_relationship.status
    confidence = _normalize_confidence(
        value=response.get("confidence") if response else None,
        default=(
            baseline_relationship.confidence
            if baseline_relationship
            else _heuristic_confidence(dossier, relationship_type)
        ),
    )
    reasoning = _build_raw_reasoning(dossier, response, relationship_type)
    if not response and baseline_relationship and baseline_relationship.reasoning:
        reasoning = baseline_relationship.reasoning
    intimacy_score = _estimate_intimacy(dossier, relationship_type)
    if baseline_relationship:
        intimacy_score = baseline_relationship.intimacy_score
    shared_events = [
        {
            "event_id": event.get("event_id", ""),
            "date": event.get("date", ""),
            "narrative": event.get("narrative_synthesis") or event.get("title", ""),
        }
        for event in dossier.shared_events
    ]

    relationship = RelationshipRecord(
        person_id=dossier.person_id,
        relationship_type=relationship_type,
        intimacy_score=intimacy_score,
        status=status,
        confidence=confidence,
        reasoning=reasoning,
        shared_events=shared_events,
        evidence={},
    )
    reflection = {
        "triggered": False,
        "issues": [],
        "action": "keep",
        "source": "llm" if response else "heuristic",
    }
    return relationship, reflection


def _run_legacy_relationship_baseline(
    dossier: RelationshipDossier,
    llm_processor: Any,
) -> RelationshipRecord | None:
    if not llm_processor or not hasattr(llm_processor, "_infer_relationship"):
        return None
    if hasattr(llm_processor, "_call_llm_via_official_api"):
        return None
    try:
        return llm_processor._infer_relationship(
            person_id=dossier.person_id,
            evidence=_build_legacy_evidence_from_dossier(dossier),
            vlm_results=[],
            face_db={},
        )
    except Exception:
        return None


def _build_legacy_evidence_from_dossier(dossier: RelationshipDossier) -> Dict[str, Any]:
    return {
        "photo_count": dossier.photo_count,
        "time_span_days": dossier.time_span_days,
        "recent_gap_days": dossier.recent_gap_days,
        "scenes": list(dossier.scene_profile.get("scenes", []) or []),
        "private_scene_ratio": float(dossier.scene_profile.get("private_scene_ratio", 0.0) or 0.0),
        "dominant_scene_ratio": float(dossier.scene_profile.get("dominant_scene_ratio", 0.0) or 0.0),
        "interaction_behavior": list(dossier.interaction_signals or []),
        "with_user_only": bool(dossier.scene_profile.get("with_user_only", False)),
        "contact_types": list(dossier.interaction_signals or []),
        "rela_events": list(dossier.shared_events or []),
        "monthly_frequency": dossier.monthly_frequency,
        "trend_detail": dict(dossier.trend_detail or {}),
        "co_appearing_persons": list(dossier.co_appearing_persons or []),
        "anomalies": list(dossier.anomalies or []),
    }


def _run_relationship_llm(dossier: RelationshipDossier, llm_processor: Any) -> Dict[str, Any] | None:
    if not llm_processor or not hasattr(llm_processor, "_call_llm_via_official_api"):
        return None
    prompt = _build_relationship_prompt(dossier)
    try:
        result = llm_processor._call_llm_via_official_api(prompt, response_mime_type="application/json")
    except Exception:
        return None
    return result if isinstance(result, dict) else None


def _build_relationship_prompt(dossier: RelationshipDossier) -> str:
    event_lines: List[str] = []
    for event in dossier.shared_events[:12]:
        event_lines.append(
            "- {date} {event_id} @ {location}: {title} | participants={participants} | desc={desc}".format(
                date=event.get("date", ""),
                event_id=event.get("event_id", ""),
                location=event.get("location", ""),
                title=event.get("title", ""),
                participants=",".join(event.get("participants", []) or []),
                desc=str(event.get("description", "") or "")[:120],
            )
        )

    prompt = f"""你是关系分析专家，请对主角与目标人物的关系做结构化判断，只返回 JSON。

目标人物：{dossier.person_id}
photo_count={dossier.photo_count}
time_span_days={dossier.time_span_days}
monthly_frequency={dossier.monthly_frequency}
private_scene_ratio={dossier.scene_profile.get("private_scene_ratio", 0.0)}
dominant_scene_ratio={dossier.scene_profile.get("dominant_scene_ratio", 0.0)}
with_user_only={dossier.scene_profile.get("with_user_only", False)}
scenes={dossier.scene_profile.get("scenes", [])}
interaction_signals={dossier.interaction_signals[:20]}

shared events:
{chr(10).join(event_lines) if event_lines else "- none"}

可选 relationship_type:
family, romantic, bestie, close_friend, friend, classmate_colleague, activity_buddy, acquaintance

规则：
1) romantic/family 需要硬信号，否则降级。
2) 单次活动或单场景不能直接升格高亲密关系。
3) 如果证据弱，优先输出 acquaintance。

输出：
{{
  "relationship_type": "one of allowed types",
  "stability": "long_term | short_term",
  "status": "new | growing | stable | fading | gone",
  "confidence": 0-100,
  "strength_summary": "一句话",
  "scenes_summary": "一句话",
  "frequency_summary": "一句话",
  "reasoning": "一句话"
}}
"""
    return prompt


def _apply_relationship_type_veto(
    dossier: RelationshipDossier,
    relationship: RelationshipRecord,
    reflection: Dict[str, Any],
) -> Tuple[RelationshipRecord, Dict[str, Any]]:
    updated = relationship
    issues: List[str] = list(reflection.get("issues", []))

    strong_signal = _has_strong_relationship_signal(dossier, relationship)
    repeated_pattern = (
        dossier.photo_count >= 3
        or len(dossier.shared_events) >= 2
        or len(dossier.scene_profile.get("scenes", [])) >= 2
    )

    if updated.relationship_type in {"romantic", "family"} and not strong_signal:
        issues.append("high_risk_relationship_without_strong_signal")
        updated = replace(
            updated,
            relationship_type=RELATIONSHIP_TYPE_SPECS[updated.relationship_type].downgrade_target or "close_friend",
            confidence=max(0.2, round(updated.confidence - 0.1, 3)),
        )
    elif updated.relationship_type == "close_friend" and not strong_signal and not repeated_pattern:
        issues.append("close_friend_without_strong_signal")
        updated = replace(
            updated,
            relationship_type="friend",
            confidence=max(0.2, round(updated.confidence - 0.08, 3)),
        )
    elif updated.relationship_type == "bestie" and not strong_signal:
        issues.append("bestie_without_strong_signal")
        updated = replace(
            updated,
            relationship_type="close_friend",
            confidence=max(0.2, round(updated.confidence - 0.08, 3)),
        )
    elif updated.relationship_type == "friend" and dossier.photo_count <= 1 and not strong_signal:
        issues.append("friend_without_repeated_signal")
        updated = replace(
            updated,
            relationship_type="acquaintance",
            confidence=max(0.2, round(updated.confidence - 0.08, 3)),
        )

    reflection = dict(reflection)
    reflection["issues"] = issues
    reflection["triggered"] = bool(issues)
    if issues:
        reflection["action"] = "downgrade"
    return updated, reflection


def _normalize_relationship_output(
    dossier: RelationshipDossier,
    relationship: RelationshipRecord,
    reflection: Dict[str, Any],
) -> RelationshipRecord:
    ids = extract_ids_from_refs(dossier.evidence_refs)
    feature_names = ids["feature_names"] + [f"relationship_type:{relationship.relationship_type}"]
    supporting_refs = list(dossier.evidence_refs)
    contradicting_refs = [
        {"source_type": "feature", "source_id": issue, "signal": issue, "why": "relationship_reflection"}
        for issue in reflection.get("issues", [])
    ]
    evidence = build_evidence_payload(
        photo_ids=ids["photo_ids"],
        event_ids=[event.get("event_id") for event in dossier.shared_events] or ids["event_ids"],
        person_ids=[relationship.person_id] + ids["person_ids"],
        feature_names=feature_names,
        supporting_refs=supporting_refs,
        contradicting_refs=contradicting_refs,
    )
    reasoning = _build_relationship_reasoning(dossier, relationship, reflection)
    return replace(relationship, evidence=evidence, reasoning=reasoning)


def _build_relationship_reasoning(
    dossier: RelationshipDossier,
    relationship: RelationshipRecord,
    reflection: Dict[str, Any],
) -> str:
    event_ids = [event.get("event_id") for event in dossier.shared_events if event.get("event_id")]
    scene_count = len(dossier.scene_profile.get("scenes", []))
    base = (
        f"共现 {dossier.photo_count} 张，覆盖 {scene_count} 类场景，月均 {dossier.monthly_frequency:.1f} 次，"
        f"判为 {relationship.relationship_type}。"
    )
    if event_ids:
        base = f"关键事件 {', '.join(event_ids[:3])}；" + base
    if reflection.get("issues"):
        base += f" 已执行降级/纠偏：{', '.join(reflection['issues'])}。"
    return base


def _determine_retention(dossier: RelationshipDossier, relationship: RelationshipRecord) -> Tuple[str, str]:
    if dossier.person_kind in {"service_person", "mediated_person"}:
        return "suppress", f"person_kind={dossier.person_kind}"
    if _looks_mediated_from_dossier(dossier):
        return "suppress", "screen_or_virtual_context"

    strong_signal = _has_strong_relationship_signal(dossier, relationship)
    if dossier.photo_count == 0:
        return "suppress", "no_supporting_photos"
    if dossier.photo_count <= 1 and not strong_signal:
        return "suppress", "single_photo_without_strong_signal"
    if (
        relationship.relationship_type in LOW_SIGNAL_RELATIONSHIP_TYPES
        and relationship.confidence < 0.62
        and not strong_signal
    ):
        return "suppress", "low_confidence_low_signal_relationship"
    if (
        relationship.relationship_type == "acquaintance"
        and dossier.photo_count <= 3
        and len(dossier.shared_events) <= 1
        and not strong_signal
    ):
        return "suppress", "weak_acquaintance_without_repeated_signal"
    if dossier.memory_value == "low_value" and relationship.confidence < 0.68 and not strong_signal:
        return "suppress", "low_value_person_without_strong_signal"
    return "keep", "relationship_retained"


def _determine_group_eligibility(
    dossier: RelationshipDossier,
    relationship: RelationshipRecord,
) -> Tuple[bool, str | None, float]:
    if dossier.retention_decision != "keep":
        return False, f"retention={dossier.retention_decision}", 0.0
    if dossier.person_kind != "real_person":
        return False, f"person_kind={dossier.person_kind}", 0.0
    if relationship.relationship_type in {"acquaintance", "family", "romantic"}:
        return False, f"relationship_type={relationship.relationship_type}", 0.0
    if relationship.confidence < 0.75:
        return False, "low_confidence", 0.0
    if len(dossier.shared_events) < 1:
        return False, "no_shared_events", 0.0
    weight = round((relationship.confidence * 0.6) + min(dossier.photo_count / 10, 0.4), 3)
    return True, None, min(weight, 1.0)


def _apply_status_correction(dossier: RelationshipDossier, relationship: RelationshipRecord) -> RelationshipRecord:
    if relationship.status != "new":
        return relationship
    strong_signal = _has_strong_relationship_signal(dossier, relationship)
    scene_count = len(dossier.scene_profile.get("scenes", []))
    shared_event_count = len(dossier.shared_events)
    trend_direction = str(dossier.trend_detail.get("direction", "") or "").lower()

    should_promote_to_growing = (
        dossier.photo_count >= 3
        and dossier.time_span_days >= 14
        and shared_event_count >= 2
        and (strong_signal or scene_count >= 2 or trend_direction == "up" or dossier.monthly_frequency >= 3)
    )
    if not should_promote_to_growing:
        return relationship

    updated_evidence = dict(relationship.evidence or {})
    supporting_refs = list(updated_evidence.get("supporting_refs", []))
    supporting_refs.append(
        {
            "source_type": "feature",
            "source_id": "status_correction",
            "signal": "new_to_growing",
            "why": "relationship_status_correction",
        }
    )
    updated_evidence["supporting_refs"] = supporting_refs
    return replace(relationship, status="growing", evidence=updated_evidence)


def _looks_mediated_from_dossier(dossier: RelationshipDossier) -> bool:
    dominant_ratio = float(dossier.scene_profile.get("dominant_scene_ratio", 0.0) or 0.0)
    searchable = list(dossier.scene_profile.get("scenes", [])) + list(dossier.interaction_signals)
    haystack = " ".join(str(item or "") for item in searchable).lower()
    has_mediated_keyword = any(keyword.lower() in haystack for keyword in MEDIATED_SIGNAL_KEYWORDS)
    return has_mediated_keyword and dominant_ratio >= 0.75


def _has_strong_relationship_signal(dossier: RelationshipDossier, relationship: RelationshipRecord) -> bool:
    interaction_signals = {str(signal or "").lower() for signal in dossier.interaction_signals}
    shared_event_count = len(dossier.shared_events)
    scene_count = len(dossier.scene_profile.get("scenes", []))
    private_scene_ratio = float(dossier.scene_profile.get("private_scene_ratio", 0.0) or 0.0)
    with_user_only = bool(dossier.scene_profile.get("with_user_only", False))

    return (
        any(signal in STRONG_CONTACT_SIGNALS for signal in interaction_signals)
        or (relationship.relationship_type in {"romantic", "family"} and private_scene_ratio >= 0.25)
        or (with_user_only and private_scene_ratio >= 0.25 and dossier.photo_count >= 2)
        or (shared_event_count >= 2 and scene_count >= 2)
    )


def _heuristic_relationship_type(dossier: RelationshipDossier) -> str:
    scenes = [str(scene or "").lower() for scene in dossier.scene_profile.get("scenes", [])]
    interactions = " ".join(str(signal or "").lower() for signal in dossier.interaction_signals)
    has_strong_contact = any(signal in interactions for signal in STRONG_CONTACT_SIGNALS)
    has_private_scene = any(any(keyword in scene for keyword in INTIMATE_SCENE_KEYWORDS) for scene in scenes)
    functional_dominant = any(any(keyword in scene for keyword in FUNCTIONAL_SCENE_KEYWORDS) for scene in scenes)

    if has_strong_contact and (has_private_scene or dossier.scene_profile.get("with_user_only", False)):
        return "romantic"
    if dossier.photo_count >= 8 and dossier.monthly_frequency >= 3.0 and len(scenes) >= 3:
        return "bestie"
    if dossier.photo_count >= 4 and dossier.monthly_frequency >= 1.5 and len(scenes) >= 2:
        return "close_friend"
    if functional_dominant and dossier.photo_count >= 3:
        return "classmate_colleague"
    if dossier.photo_count >= 3:
        return "friend"
    if dossier.photo_count >= 2:
        return "activity_buddy"
    if _is_emerging_candidate(dossier):
        return "acquaintance"
    return "acquaintance"


def _is_emerging_candidate(dossier: RelationshipDossier) -> bool:
    if dossier.photo_count > 1:
        return False
    haystack = " ".join(str(signal or "") for signal in dossier.interaction_signals).lower()
    return any(keyword in haystack for keyword in EMERGING_INTIMATE_KEYWORDS)


def _normalize_relationship_type(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    if value in RELATIONSHIP_TYPES:
        return value
    return ""


def _normalize_status(response: Dict[str, Any] | None, dossier: RelationshipDossier) -> str:
    if response:
        status = str(response.get("status") or "").strip().lower()
        if status in {"new", "growing", "stable", "fading", "gone"}:
            return status
        stability = str(response.get("stability") or "").strip().lower()
        if stability == "short_term":
            return "new"
        if stability == "long_term":
            return "stable"
    if dossier.recent_gap_days >= 60:
        return "gone"
    if dossier.photo_count >= 3 and dossier.time_span_days >= 30:
        return "stable"
    return "new"


def _normalize_confidence(value: Any, default: float) -> float:
    try:
        score = float(value)
    except Exception:
        return round(default, 3)
    if score > 1:
        score = score / 100.0
    return round(min(max(score, 0.0), 1.0), 3)


def _heuristic_confidence(dossier: RelationshipDossier, relationship_type: str) -> float:
    base = 0.45
    base += min(dossier.photo_count / 20.0, 0.2)
    base += min(dossier.monthly_frequency / 10.0, 0.15)
    base += min(len(dossier.shared_events) / 10.0, 0.1)
    if relationship_type in {"romantic", "family", "bestie"}:
        base -= 0.08
    return round(min(max(base, 0.25), 0.92), 3)


def _estimate_intimacy(dossier: RelationshipDossier, relationship_type: str) -> float:
    score = 0.2
    score += min(dossier.photo_count / 20.0, 0.2)
    score += min(dossier.monthly_frequency / 8.0, 0.25)
    score += min(float(dossier.scene_profile.get("private_scene_ratio", 0.0) or 0.0) * 0.25, 0.25)
    if relationship_type in {"romantic", "bestie"}:
        score += 0.1
    elif relationship_type in {"friend", "close_friend"}:
        score += 0.05
    return round(min(max(score, 0.0), 1.0), 3)


def _build_raw_reasoning(dossier: RelationshipDossier, response: Dict[str, Any] | None, relationship_type: str) -> str:
    if response:
        parts = [
            str(response.get("strength_summary") or "").strip(),
            str(response.get("scenes_summary") or "").strip(),
            str(response.get("frequency_summary") or "").strip(),
            str(response.get("reasoning") or "").strip(),
        ]
        merged = " ".join(part for part in parts if part)
        if merged:
            return merged
    return (
        f"{dossier.person_id} 共现 {dossier.photo_count} 张，场景 {len(dossier.scene_profile.get('scenes', []))} 类，"
        f"月均 {dossier.monthly_frequency:.1f} 次，判为 {relationship_type}。"
    )


def _build_evidence_refs(person_id: str, evidence: Dict[str, Any], vlm_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for item in vlm_results or []:
        analysis = item.get("vlm_analysis", {}) or {}
        people = analysis.get("people", []) or []
        if any(isinstance(person, dict) and person.get("person_id") == person_id for person in people):
            refs.append(
                {
                    "source_type": "photo",
                    "source_id": item.get("photo_id"),
                    "signal": analysis.get("summary", ""),
                    "why": f"co_photo_for_{person_id}",
                }
            )
    for event in evidence.get("rela_events", []) or []:
        refs.append(
            {
                "source_type": "event",
                "source_id": event.get("event_id"),
                "signal": event.get("title") or event.get("narrative_synthesis", ""),
                "why": f"shared_event_for_{person_id}",
            }
        )
    for co_person in evidence.get("co_appearing_persons", []) or []:
        refs.append(
            {
                "source_type": "person",
                "source_id": co_person.get("person_id"),
                "signal": f"co_ratio={co_person.get('co_ratio', 0)}",
                "why": "third_party_co_appearance",
            }
        )
    refs.append(
        {
            "source_type": "feature",
            "source_id": "photo_count",
            "signal": str(evidence.get("photo_count", 0)),
            "why": "relationship_frequency",
        }
    )
    refs.append(
        {
            "source_type": "feature",
            "source_id": "monthly_frequency",
            "signal": str(evidence.get("monthly_frequency", 0)),
            "why": "relationship_frequency",
        }
    )
    return refs
