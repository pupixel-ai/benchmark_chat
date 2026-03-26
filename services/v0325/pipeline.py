"""
Independent v0325 pipeline family.

This family reuses the existing VP1/LP1 normalization from v0323, then runs a new
LP2/LP3 agent-style pipeline with:
- raw-upstream no-drop tracking
- primary person re-judgement
- relationship dossiers
- field-level structured profiling
- lightweight downstream audit + backflow
"""
from __future__ import annotations

import copy
import gzip
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_LLM_MODEL,
    PROFILE_AGENT_ROOT,
    PROFILE_LLM_MODEL,
    PROFILE_LLM_PROVIDER,
    RELATIONSHIP_MAX_RETRIES,
    RELATIONSHIP_REQUEST_TIMEOUT_SECONDS,
    V0323_LP1_MAX_ATTEMPTS,
    V0323_LP1_MAX_OUTPUT_TOKENS,
)
from models import Event
from utils import load_json, save_json

from services.v0325.profile_compaction import compact_structured_profile
from services.v0325.lp3_core import (
    analyze_primary_person_with_reflection as agent_analyze_primary_person_with_reflection,
    build_profile_context as agent_build_profile_context,
    build_relationship_dossiers as agent_build_relationship_dossiers,
    detect_groups as agent_detect_groups,
    generate_structured_profile as agent_generate_structured_profile,
    infer_relationships_from_dossiers as agent_infer_relationships_from_dossiers,
    screen_people as agent_screen_people,
)
from services.v0323.pipeline import (
    LP1_ANALYSIS_TEXT_CHAR_LIMIT,
    LP1_BATCH_SIZE,
    LP1_OVERLAP_SIZE,
    V0323PipelineFamily,
    _append_jsonl,
    _json_default,
    _normalized_text,
    _safe_float,
    _safe_int,
    _sorted_unique_photo_ids,
    _truncate_preview,
    _unique_strings,
    _write_jsonl,
)


PIPELINE_FAMILY_V0325 = "v0325"
PIPELINE_VERSION_V0325 = "v0325"
LP1_PROMPT_VERSION_V0325 = "v0325.lp1.v0139_two_step.v2"
LP1_CONTRACT_VERSION_V0325 = "v0325.lp1.output_window.v1"

SELFIE_KEYWORDS = ("自拍", "selfie", "mirror selfie", "镜前", "前置摄像头")
IDENTITY_KEYWORDS = ("证件", "工牌", "student id", "学生证", "id card", "badge", "证件照")
USER_VIEW_KEYWORDS = ("第一视角", "as photographer", "拍摄者", "主角作为拍摄者", "user view", "recording")
PHOTOGRAPHED_SUBJECT_KEYWORDS = (
    "人像拍摄",
    "portrait",
    "展示照",
    "主角作为拍摄者记录",
    "拍摄者记录",
    "拍别人",
)
PROTAGONIST_LABEL_PATTERNS = (
    re.compile(r"【主角】[（(]?(Person_\d+)[）)]?"),
    re.compile(r"(Person_\d+)[（(]【主角】[）)]"),
    re.compile(r"(Person_\d+).*?【主角】"),
)

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

NON_DAILY_EVENT_KEYWORDS = (
    "旅行",
    "travel",
    "trip",
    "车展",
    "expo",
    "exhibition",
    "festival",
    "展会",
    "主题活动",
    "打卡",
    "concert",
    "parade",
    "fair",
)
SOCIAL_MEDIA_KEYWORDS = (
    "instagram",
    "ig",
    "wechat",
    "朋友圈",
    "moments",
    "小红书",
    "微博",
    "bio",
    "profile",
    "关注了你",
    "ins",
    "story",
)
MATERIAL_SIGNAL_KEYWORDS = (
    "logo",
    "brand",
    "品牌",
    "手机壳",
    "phone case",
    "case",
    "水杯",
    "cup",
    "mug",
    "包",
    "bag",
    "护肤",
    "护肤品",
    "skincare",
    "cosmetic",
    "化妆",
    "lipstick",
    "口红",
    "香水",
    "perfume",
    "卫衣",
    "hoodie",
    "shirt",
    "t-shirt",
    "鞋",
    "shoe",
    "耳机",
    "headphone",
)
LOCATION_STOPWORDS = (
    "office",
    "classroom",
    "campus",
    "restaurant",
    "cafe",
    "museum",
    "park",
    "广场",
    "公园",
    "博物馆",
    "餐厅",
    "咖啡馆",
    "校园",
    "教室",
    "办公室",
    "宿舍",
)


def _path_exists(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def _safe_unlink(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        return


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if isinstance(payload, dict):
                    records.append(payload)
    except Exception:
        return []
    return records


def _dedupe_strs(values: Iterable[str] | None) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values or []:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _extract_partial_event_objects(raw_text: str) -> List[Dict[str, Any]]:
    candidate = str(raw_text or "")
    search_pattern = '"events"'
    anchor = candidate.find(search_pattern)
    if anchor == -1:
        return []
    array_start = candidate.find("[", anchor)
    if array_start == -1:
        return []
    events: List[Dict[str, Any]] = []
    in_string = False
    escape_next = False
    brace_depth = 0
    object_start: Optional[int] = None
    for cursor in range(array_start + 1, len(candidate)):
        char = candidate[cursor]
        if escape_next:
            escape_next = False
            continue
        if char == "\\" and in_string:
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            if brace_depth == 0:
                object_start = cursor
            brace_depth += 1
            continue
        if char == "}":
            if brace_depth == 0:
                continue
            brace_depth -= 1
            if brace_depth == 0 and object_start is not None:
                try:
                    payload = json.loads(candidate[object_start : cursor + 1])
                except Exception:
                    payload = None
                if isinstance(payload, dict):
                    events.append(payload)
                object_start = None
            continue
        if char == "]" and brace_depth == 0:
            break
    return events


def _flatten_ref_buckets(ref_buckets: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    flat: List[Dict[str, Any]] = []
    for refs in ref_buckets.values():
        flat.extend(refs)
    return flat


def _candidate_ref_ids(ref: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for key in ("event_id", "photo_id", "person_id", "group_id", "feature_name"):
        if ref.get(key):
            ids.append(str(ref[key]))
    for key in ("event_ids", "photo_ids", "person_ids", "group_ids", "feature_names"):
        ids.extend(str(item) for item in ref.get(key, []) or [])
    return _dedupe_strs(ids)


def _build_ref_index(ref_buckets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    ref_index: Dict[str, Dict[str, Any]] = {}
    for ref in _flatten_ref_buckets(ref_buckets):
        for ref_id in _candidate_ref_ids(ref):
            ref_index[ref_id] = ref
    return ref_index


def _dedupe_refs(refs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[Tuple[str, ...]] = set()
    ordered: List[Dict[str, Any]] = []
    for ref in refs:
        identity = tuple(_candidate_ref_ids(ref)) or tuple(
            f"{key}:{json.dumps(value, ensure_ascii=False, default=_json_default)}"
            for key, value in sorted(ref.items())
        )
        if identity in seen:
            continue
        seen.add(identity)
        ordered.append(ref)
    return ordered


def _select_refs(ref_index: Dict[str, Dict[str, Any]], ref_ids: Iterable[str] | None) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for ref_id in ref_ids or []:
        normalized = str(ref_id or "").strip()
        if not normalized:
            continue
        ref = ref_index.get(normalized)
        if ref is not None:
            refs.append(ref)
    return _dedupe_refs(refs)


def _filter_bucket(bucket: List[Dict[str, Any]], selected_refs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected_id_set: set[str] = set()
    for ref in selected_refs:
        selected_id_set.update(_candidate_ref_ids(ref))
    if not selected_id_set:
        return []
    filtered: List[Dict[str, Any]] = []
    for ref in bucket:
        if selected_id_set.intersection(_candidate_ref_ids(ref)):
            filtered.append(ref)
    return _dedupe_refs(filtered)


def _extract_ids_from_refs(refs: Iterable[Dict[str, Any]]) -> Dict[str, List[str]]:
    photo_ids: List[str] = []
    event_ids: List[str] = []
    person_ids: List[str] = []
    group_ids: List[str] = []
    feature_names: List[str] = []
    for ref in refs:
        source_type = str(ref.get("source_type", "") or "")
        source_id = ref.get("source_id")
        if source_type == "photo" and source_id:
            photo_ids.append(str(source_id))
        elif source_type == "event" and source_id:
            event_ids.append(str(source_id))
        elif source_type == "person" and source_id:
            person_ids.append(str(source_id))
        elif source_type == "group" and source_id:
            group_ids.append(str(source_id))
        elif source_type == "feature" and source_id:
            feature_names.append(str(source_id))
        if ref.get("photo_id"):
            photo_ids.append(str(ref["photo_id"]))
        if ref.get("event_id"):
            event_ids.append(str(ref["event_id"]))
        if ref.get("person_id"):
            person_ids.append(str(ref["person_id"]))
        if ref.get("group_id"):
            group_ids.append(str(ref["group_id"]))
        if ref.get("feature_name"):
            feature_names.append(str(ref["feature_name"]))
        photo_ids.extend(str(item) for item in list(ref.get("photo_ids", []) or []))
        event_ids.extend(str(item) for item in list(ref.get("event_ids", []) or []))
        person_ids.extend(str(item) for item in list(ref.get("person_ids", []) or []))
        group_ids.extend(str(item) for item in list(ref.get("group_ids", []) or []))
        feature_names.extend(str(item) for item in list(ref.get("feature_names", []) or []))
    return {
        "photo_ids": _dedupe_strs(photo_ids),
        "event_ids": _dedupe_strs(event_ids),
        "person_ids": _dedupe_strs(person_ids),
        "group_ids": _dedupe_strs(group_ids),
        "feature_names": _dedupe_strs(feature_names),
    }


def _build_evidence_payload(
    *,
    photo_ids: Iterable[str] | None = None,
    event_ids: Iterable[str] | None = None,
    person_ids: Iterable[str] | None = None,
    group_ids: Iterable[str] | None = None,
    feature_names: Iterable[str] | None = None,
    supporting_refs: Iterable[Dict[str, Any]] | None = None,
    contradicting_refs: Iterable[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    return {
        "photo_ids": _dedupe_strs(photo_ids),
        "event_ids": _dedupe_strs(event_ids),
        "person_ids": _dedupe_strs(person_ids),
        "group_ids": _dedupe_strs(group_ids),
        "feature_names": _dedupe_strs(feature_names),
        "supporting_refs": list(supporting_refs or []),
        "contradicting_refs": list(contradicting_refs or []),
    }


def _empty_tag_object() -> Dict[str, Any]:
    evidence = _build_evidence_payload()
    evidence["events"] = []
    evidence["relationships"] = []
    evidence["vlm_observations"] = []
    evidence["group_artifacts"] = []
    evidence["feature_refs"] = []
    evidence["constraint_notes"] = []
    evidence["summary"] = ""
    return {
        "value": None,
        "confidence": 0.0,
        "evidence": evidence,
        "reasoning": "",
    }


def _normalize_ref_id_list(values: Any, *, ref_index: Dict[str, Dict[str, Any]] | None = None) -> List[str]:
    normalized = _dedupe_strs(values if isinstance(values, list) else [])
    if not ref_index:
        return normalized
    return [ref_id for ref_id in normalized if ref_id in ref_index]


def _section(keys: Iterable[str]) -> Dict[str, Any]:
    return {key: _empty_tag_object() for key in keys}


def _assign_tag_object(payload: Dict[str, Any], field_key: str, value: Dict[str, Any]) -> None:
    path = field_key.split(".")
    current = payload
    for part in path[:-1]:
        current = current.setdefault(part, {})
    current[path[-1]] = value


def _get_nested(payload: Dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


def _value_to_string(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if str(item or "").strip())
    return str(value)


def _contains_any_keyword(text: str, keywords: Iterable[str]) -> bool:
    normalized = str(text or "").lower()
    return any(keyword.lower() in normalized for keyword in keywords)


def _normalize_string_list(values: Iterable[Any]) -> List[str]:
    normalized: List[str] = []
    for value in values:
        candidate = ""
        if isinstance(value, dict):
            for key in ("person_id", "value", "name", "text"):
                if key in value and value[key]:
                    candidate = str(value[key]).strip()
                    break
        elif value is not None:
            candidate = str(value).strip()
        if candidate:
            normalized.append(candidate)
    return list(dict.fromkeys(normalized))


def _window_key_from_timestamp(timestamp: str) -> str:
    value = str(timestamp or "").strip()
    if len(value) >= 7:
        return value[:7]
    return value


@dataclass
class PersonScreening:
    person_id: str
    person_kind: str
    memory_value: str
    screening_refs: List[Dict[str, Any]] = field(default_factory=list)
    block_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RelationshipTypeSpec:
    type_key: str
    downgrade_target: str | None


@dataclass
class RelationshipRecord:
    person_id: str
    relationship_type: str
    intimacy_score: float
    status: str
    confidence: float
    reasoning: str
    shared_events: List[Dict[str, Any]] = field(default_factory=list)
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["relationship_id"] = f"REL_{self.person_id}"
        payload["supporting_event_ids"] = list(payload["evidence"].get("event_ids", []))
        payload["supporting_photo_ids"] = list(payload["evidence"].get("photo_ids", []))
        payload["reason"] = payload["reasoning"]
        return payload


@dataclass
class RelationshipDossier:
    person_id: str
    person_kind: str
    memory_value: str
    photo_count: int
    time_span_days: int
    recent_gap_days: int
    monthly_frequency: float
    scene_profile: Dict[str, Any]
    interaction_signals: List[str]
    shared_events: List[Dict[str, Any]]
    trend_detail: Dict[str, Any]
    co_appearing_persons: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    evidence_refs: List[Dict[str, Any]]
    block_reasons: List[str] = field(default_factory=list)
    retention_decision: str = "review"
    retention_reason: str = ""
    group_eligible: bool = False
    group_block_reason: Optional[str] = None
    group_weight: float = 0.0
    relationship_result: Dict[str, Any] = field(default_factory=dict)
    relationship_reflection: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GroupArtifact:
    group_id: str
    members: List[str]
    group_type_candidate: str
    confidence: float
    strong_evidence_refs: List[Dict[str, Any]]
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FieldSpec:
    field_key: str
    risk_level: str
    allowed_sources: List[str]
    strong_evidence: List[str]


@dataclass
class ProfileState:
    structured_profile: Dict[str, Any]
    field_decisions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryState:
    photos: List[Any]
    face_db: Dict[str, Any]
    vlm_results: List[Dict[str, Any]]
    raw_upstream: Dict[str, Any] = field(default_factory=dict)
    raw_index: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    primary_alias_bindings: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    screening: Dict[str, PersonScreening] = field(default_factory=dict)
    primary_decision: Dict[str, Any] | None = None
    primary_reflection: Dict[str, Any] | None = None
    events: List[Event] = field(default_factory=list)
    relationships: List[RelationshipRecord] = field(default_factory=list)
    relationship_dossiers: List[RelationshipDossier] = field(default_factory=list)
    groups: List[GroupArtifact] = field(default_factory=list)
    profile_context: Dict[str, Any] | None = None


RELATIONSHIP_TYPE_SPECS: Dict[str, RelationshipTypeSpec] = {
    "family": RelationshipTypeSpec(type_key="family", downgrade_target="close_friend"),
    "romantic": RelationshipTypeSpec(type_key="romantic", downgrade_target="close_friend"),
    "bestie": RelationshipTypeSpec(type_key="bestie", downgrade_target="close_friend"),
    "close_friend": RelationshipTypeSpec(type_key="close_friend", downgrade_target="friend"),
    "friend": RelationshipTypeSpec(type_key="friend", downgrade_target="acquaintance"),
    "classmate_colleague": RelationshipTypeSpec(type_key="classmate_colleague", downgrade_target="friend"),
    "activity_buddy": RelationshipTypeSpec(type_key="activity_buddy", downgrade_target="friend"),
    "acquaintance": RelationshipTypeSpec(type_key="acquaintance", downgrade_target=None),
}

FIELD_SPECS: Dict[str, FieldSpec] = {
    "long_term_facts.identity.name": FieldSpec("long_term_facts.identity.name", "P1", ["vlm", "feature"], ["实名或稳定昵称"]),
    "long_term_facts.identity.gender": FieldSpec("long_term_facts.identity.gender", "P1", ["vlm"], ["多次稳定主角外观"]),
    "long_term_facts.identity.age_range": FieldSpec("long_term_facts.identity.age_range", "P1", ["vlm", "event", "feature"], ["跨事件年龄阶段线索"]),
    "long_term_facts.identity.role": FieldSpec("long_term_facts.identity.role", "P0", ["event", "vlm", "feature"], ["至少2条校园/工作主线证据"]),
    "long_term_facts.identity.race": FieldSpec("long_term_facts.identity.race", "P0", ["vlm", "event"], ["多事件稳定外观"]),
    "long_term_facts.identity.nationality": FieldSpec("long_term_facts.identity.nationality", "P0", ["vlm", "event", "feature"], ["长期地点与文化线索"]),
    "long_term_facts.social_identity.education": FieldSpec("long_term_facts.social_identity.education", "P0", ["event", "vlm", "feature"], ["至少2条跨事件校园证据"]),
    "long_term_facts.social_identity.career": FieldSpec("long_term_facts.social_identity.career", "P0", ["event", "feature"], ["多事件工作场景闭环"]),
    "long_term_facts.social_identity.career_phase": FieldSpec("long_term_facts.social_identity.career_phase", "P1", ["event", "feature"], ["稳定身份主线 + 最近变化"]),
    "long_term_facts.social_identity.professional_dedication": FieldSpec("long_term_facts.social_identity.professional_dedication", "P1", ["event", "feature"], ["连续工作/学习事件"]),
    "long_term_facts.social_identity.language_culture": FieldSpec("long_term_facts.social_identity.language_culture", "P1", ["vlm", "event", "feature"], ["稳定文本/语言环境"]),
    "long_term_facts.material.asset_level": FieldSpec("long_term_facts.material.asset_level", "P0", ["event", "vlm", "feature"], ["多条社经证据"]),
    "long_term_facts.material.spending_style": FieldSpec("long_term_facts.material.spending_style", "P0", ["event", "vlm", "feature"], ["跨事件重复消费模式"]),
    "long_term_facts.material.brand_preference": FieldSpec("long_term_facts.material.brand_preference", "P0", ["event", "vlm", "feature"], ["同一品牌跨事件重复出现"]),
    "long_term_facts.material.income_model": FieldSpec("long_term_facts.material.income_model", "P1", ["event", "feature"], ["role+career+生活方式闭环"]),
    "long_term_facts.material.signature_items": FieldSpec("long_term_facts.material.signature_items", "P1", ["vlm", "event", "feature"], ["物品跨事件重复"]),
    "long_term_facts.geography.location_anchors": FieldSpec("long_term_facts.geography.location_anchors", "P0", ["event", "vlm", "feature"], ["跨事件重复地点锚点"]),
    "long_term_facts.geography.mobility_pattern": FieldSpec("long_term_facts.geography.mobility_pattern", "P0", ["event", "feature"], ["持续位移模式"]),
    "long_term_facts.geography.cross_border": FieldSpec("long_term_facts.geography.cross_border", "P1", ["event", "feature"], ["明确跨境线索"]),
    "long_term_facts.time.life_rhythm": FieldSpec("long_term_facts.time.life_rhythm", "P1", ["event", "feature"], ["稳定作息模式"]),
    "long_term_facts.time.event_cycles": FieldSpec("long_term_facts.time.event_cycles", "P1", ["event", "feature"], ["重复事件模式"]),
    "long_term_facts.time.sleep_pattern": FieldSpec("long_term_facts.time.sleep_pattern", "P1", ["event", "feature"], ["持续深夜/清晨重复"]),
    "long_term_facts.relationships.intimate_partner": FieldSpec("long_term_facts.relationships.intimate_partner", "P0", ["relationship", "feature"], ["LP2稳定 romantic"]),
    "long_term_facts.relationships.close_circle_size": FieldSpec("long_term_facts.relationships.close_circle_size", "P0", ["relationship", "feature"], ["LP2核心关系统计"]),
    "long_term_facts.relationships.social_groups": FieldSpec("long_term_facts.relationships.social_groups", "P1", ["group", "relationship", "event", "feature"], ["GroupArtifact稳定群组"]),
    "long_term_facts.relationships.pets": FieldSpec("long_term_facts.relationships.pets", "P0", ["vlm", "event", "feature"], ["动物跨事件 + 居家/照护线索"]),
    "long_term_facts.relationships.parenting": FieldSpec("long_term_facts.relationships.parenting", "P0", ["event", "vlm", "relationship", "feature"], ["连续照护行为 + family线索"]),
    "long_term_facts.relationships.living_situation": FieldSpec("long_term_facts.relationships.living_situation", "P0", ["event", "relationship", "feature"], ["重复居家事件 + 稳定同住线索"]),
    "long_term_facts.hobbies.interests": FieldSpec("long_term_facts.hobbies.interests", "P1", ["event", "vlm", "feature"], ["同类活动跨事件重复"]),
    "long_term_facts.hobbies.frequent_activities": FieldSpec("long_term_facts.hobbies.frequent_activities", "P1", ["event", "feature"], ["长期 top 活动"]),
    "long_term_facts.hobbies.solo_vs_social": FieldSpec("long_term_facts.hobbies.solo_vs_social", "P1", ["event", "relationship", "feature"], ["事件规模 + 关系结果"]),
    "long_term_facts.physiology.fitness_level": FieldSpec("long_term_facts.physiology.fitness_level", "P1", ["event", "vlm", "feature"], ["持续运动事件"]),
    "long_term_facts.physiology.diet_mode": FieldSpec("long_term_facts.physiology.diet_mode", "P1", ["event", "vlm", "feature"], ["稳定饮食模式"]),
    "short_term_facts.life_events": FieldSpec("short_term_facts.life_events", "P1", ["event", "relationship", "feature"], ["近期窗口重大事件"]),
    "short_term_facts.phase_change": FieldSpec("short_term_facts.phase_change", "P1", ["event", "relationship", "feature"], ["近期与长期基线系统偏移"]),
    "short_term_facts.spending_shift": FieldSpec("short_term_facts.spending_shift", "P1", ["event", "vlm", "feature"], ["近期消费模式持续偏移"]),
    "short_term_facts.current_displacement": FieldSpec("short_term_facts.current_displacement", "P0", ["event", "feature"], ["近期连续临时位移状态"]),
    "short_term_facts.recent_habits": FieldSpec("short_term_facts.recent_habits", "P1", ["event", "feature"], ["近期连续重复行为"]),
    "short_term_facts.recent_interests": FieldSpec("short_term_facts.recent_interests", "P1", ["event", "vlm", "feature"], ["近期主题多次重复"]),
    "long_term_expression.personality_mbti": FieldSpec("long_term_expression.personality_mbti", "P0", ["event", "relationship", "feature"], ["跨事件行为模式"]),
    "long_term_expression.morality": FieldSpec("long_term_expression.morality", "P1", ["event", "relationship", "feature"], ["稳定价值取向二阶特征"]),
    "long_term_expression.philosophy": FieldSpec("long_term_expression.philosophy", "P1", ["event", "relationship", "feature"], ["稳定生活取向二阶特征"]),
    "long_term_expression.attitude_style": FieldSpec("long_term_expression.attitude_style", "P1", ["vlm", "event", "feature"], ["穿搭/场景选择跨事件重复"]),
    "long_term_expression.aesthetic_tendency": FieldSpec("long_term_expression.aesthetic_tendency", "P1", ["vlm", "event", "feature"], ["构图/色调/物品选择跨事件重复"]),
    "long_term_expression.visual_creation_style": FieldSpec("long_term_expression.visual_creation_style", "P1", ["vlm", "event", "feature"], ["拍照模式跨事件重复"]),
    "short_term_expression.current_mood": FieldSpec("short_term_expression.current_mood", "P1", ["event", "vlm", "relationship", "feature"], ["近期多条情绪线索"]),
    "short_term_expression.mental_state": FieldSpec("short_term_expression.mental_state", "P1", ["event", "vlm", "relationship", "feature"], ["近期连续心理状态线索"]),
    "short_term_expression.motivation_shift": FieldSpec("short_term_expression.motivation_shift", "P1", ["event", "relationship", "feature"], ["近期投入方向持续变化"]),
    "short_term_expression.stress_signal": FieldSpec("short_term_expression.stress_signal", "P1", ["event", "relationship", "feature"], ["近期压力信号重复出现"]),
    "short_term_expression.social_energy": FieldSpec("short_term_expression.social_energy", "P1", ["event", "relationship", "feature"], ["近期社交能量模式变化"]),
}

DOMAIN_SPECS: List[Dict[str, Any]] = [
    {
        "domain_key": "foundation_social_identity",
        "display_name": "Foundation & Social Identity",
        "fields": [
            "long_term_facts.identity.name",
            "long_term_facts.identity.gender",
            "long_term_facts.identity.age_range",
            "long_term_facts.identity.role",
            "long_term_facts.identity.race",
            "long_term_facts.identity.nationality",
            "long_term_facts.social_identity.education",
            "long_term_facts.social_identity.career",
            "long_term_facts.social_identity.career_phase",
            "long_term_facts.social_identity.professional_dedication",
            "long_term_facts.social_identity.language_culture",
        ],
    },
    {
        "domain_key": "wealth_consumption",
        "display_name": "Wealth & Consumption",
        "fields": [
            "long_term_facts.material.asset_level",
            "long_term_facts.material.spending_style",
            "long_term_facts.material.brand_preference",
            "long_term_facts.material.income_model",
            "long_term_facts.material.signature_items",
            "short_term_facts.spending_shift",
        ],
    },
    {
        "domain_key": "spatiotemporal_habits",
        "display_name": "Spatio-Temporal Habits",
        "fields": [
            "long_term_facts.geography.location_anchors",
            "long_term_facts.geography.mobility_pattern",
            "long_term_facts.geography.cross_border",
            "long_term_facts.time.life_rhythm",
            "long_term_facts.time.event_cycles",
            "long_term_facts.time.sleep_pattern",
            "short_term_facts.phase_change",
            "short_term_facts.current_displacement",
            "short_term_facts.recent_habits",
        ],
    },
    {
        "domain_key": "relationships_household",
        "display_name": "Relationships & Household",
        "fields": [
            "long_term_facts.relationships.intimate_partner",
            "long_term_facts.relationships.close_circle_size",
            "long_term_facts.relationships.social_groups",
            "long_term_facts.relationships.pets",
            "long_term_facts.relationships.parenting",
            "long_term_facts.relationships.living_situation",
        ],
    },
    {
        "domain_key": "taste_interests",
        "display_name": "Taste & Interests",
        "fields": [
            "long_term_facts.hobbies.interests",
            "long_term_facts.hobbies.frequent_activities",
            "long_term_facts.hobbies.solo_vs_social",
            "long_term_facts.physiology.fitness_level",
            "long_term_facts.physiology.diet_mode",
            "short_term_facts.life_events",
            "short_term_facts.recent_interests",
        ],
    },
    {
        "domain_key": "visual_expression",
        "display_name": "Visual Expression",
        "fields": [
            "long_term_expression.attitude_style",
            "long_term_expression.aesthetic_tendency",
            "long_term_expression.visual_creation_style",
            "short_term_expression.current_mood",
            "short_term_expression.social_energy",
        ],
    },
    {
        "domain_key": "semantic_expression",
        "display_name": "Semantic Expression",
        "fields": [
            "long_term_expression.personality_mbti",
            "long_term_expression.morality",
            "long_term_expression.philosophy",
            "short_term_expression.mental_state",
            "short_term_expression.motivation_shift",
            "short_term_expression.stress_signal",
        ],
    },
]


class V0325LLMRuntimeAdapter:
    def __init__(self, *, llm_processor: Any, primary_person_id_hint: str | None = None) -> None:
        self.llm_processor = llm_processor
        self.primary_person_id = primary_person_id_hint

    def _call_llm_via_official_api(
        self,
        prompt: str,
        response_mime_type: str | None = None,
        model_override: str | None = None,
    ) -> Any:
        _ = model_override
        if response_mime_type == "application/json":
            handler = getattr(self.llm_processor, "_call_json_prompt", None)
            if callable(handler):
                return handler(prompt)
        raw_handler = getattr(self.llm_processor, "_call_json_prompt_raw_text", None)
        if callable(raw_handler):
            return raw_handler(prompt)
        markdown_handler = getattr(self.llm_processor, "_call_markdown_prompt", None)
        if callable(markdown_handler):
            return markdown_handler(prompt)
        fallback_handler = getattr(self.llm_processor, "_call_llm_via_official_api", None)
        if callable(fallback_handler):
            return fallback_handler(prompt)
        raise RuntimeError("LLM processor does not expose a compatible v0325 adapter interface")

    def _collect_relationship_evidence(
        self,
        person_id: str,
        vlm_results: List[Dict[str, Any]],
        events: Sequence[Event] | None = None,
    ) -> Dict[str, Any]:
        evidence = {
            "photo_count": 0,
            "time_span": "",
            "time_span_days": 0,
            "recent_gap_days": 0,
            "scenes": [],
            "private_scene_ratio": 0.0,
            "dominant_scene_ratio": 0.0,
            "interaction_behavior": [],
            "with_user_only": True,
            "sample_scenes": [],
            "contact_types": [],
            "rela_events": [],
            "monthly_frequency": 0.0,
            "trend_detail": {},
            "co_appearing_persons": [],
            "anomalies": [],
        }
        co_photos: List[Dict[str, Any]] = []
        for item in vlm_results or []:
            analysis = item.get("vlm_analysis", {}) or {}
            people = list(analysis.get("people", []) or [])
            face_person_ids = list(item.get("face_person_ids", []) or [])
            if person_id in face_person_ids or any(
                isinstance(person, dict) and str(person.get("person_id") or "").strip() == person_id
                for person in people
            ):
                co_photos.append(item)
        evidence["photo_count"] = len(co_photos)
        if not co_photos:
            evidence["rela_events"] = _collect_shared_events_from_state(events or [], person_id)
            return evidence
        timestamps = []
        for item in co_photos:
            try:
                timestamps.append(datetime.fromisoformat(str(item.get("timestamp") or "")))
            except Exception:
                continue
        timestamps.sort()
        if timestamps:
            first = timestamps[0]
            last = timestamps[-1]
            span_days = (last - first).days + 1
            evidence["time_span"] = f"{span_days}天"
            evidence["time_span_days"] = span_days
            evidence["monthly_frequency"] = round(len(co_photos) / max(span_days / 30, 1), 1)
            dataset_last = max(timestamps)
            evidence["recent_gap_days"] = max((dataset_last - last).days, 0)
        scene_counts: Counter[str] = Counter()
        private_scene_hits = 0
        third_party: Counter[str] = Counter()
        for photo in co_photos:
            analysis = photo.get("vlm_analysis", {}) or {}
            scene_data = analysis.get("scene", {}) or {}
            scene = str(scene_data.get("location_detected") or "").strip()
            if scene:
                if scene not in evidence["scenes"]:
                    evidence["scenes"].append(scene)
                scene_counts[scene] += 1
                if any(keyword in scene.lower() for keyword in INTIMATE_SCENE_KEYWORDS):
                    private_scene_hits += 1
            event_data = analysis.get("event", {}) or {}
            evidence["sample_scenes"].append(
                {
                    "timestamp": photo.get("timestamp"),
                    "scene": scene,
                    "summary": analysis.get("summary", ""),
                    "activity": event_data.get("activity", "") if isinstance(event_data, dict) else "",
                }
            )
            other_people = []
            for person in list(analysis.get("people", []) or []):
                if not isinstance(person, dict):
                    continue
                candidate_id = str(person.get("person_id") or "").strip()
                if not candidate_id:
                    continue
                if candidate_id == person_id:
                    contact_type = str(person.get("contact_type") or "").strip()
                    if contact_type:
                        evidence["contact_types"].append(contact_type)
                elif candidate_id != str(self.primary_person_id or "").strip():
                    other_people.append(candidate_id)
                    third_party[candidate_id] += 1
            if other_people:
                evidence["with_user_only"] = False
        if co_photos:
            evidence["private_scene_ratio"] = round(private_scene_hits / len(co_photos), 2)
        if scene_counts:
            evidence["dominant_scene_ratio"] = round(max(scene_counts.values()) / len(co_photos), 2)

        evidence["rela_events"] = _collect_shared_events_from_state(events or [], person_id)
        for event in evidence["rela_events"]:
            for dyn in list(event.get("social_dynamics", []) or []):
                if dyn.get("target_id") == person_id:
                    interaction = str(dyn.get("interaction_type") or "").strip()
                    if interaction and interaction not in evidence["interaction_behavior"]:
                        evidence["interaction_behavior"].append(interaction)

        if len(timestamps) >= 4:
            midpoint = timestamps[0] + (timestamps[-1] - timestamps[0]) / 2
            first_half = [item for item in timestamps if item <= midpoint]
            second_half = [item for item in timestamps if item > midpoint]
            first_days = max((midpoint - timestamps[0]).days, 1)
            second_days = max((timestamps[-1] - midpoint).days, 1)
            first_freq = round(len(first_half) / (first_days / 30), 1)
            second_freq = round(len(second_half) / (second_days / 30), 1)
            direction = "flat"
            if second_freq > first_freq * 1.5:
                direction = "up"
            elif second_freq < first_freq * 0.5:
                direction = "down"
            evidence["trend_detail"] = {
                "first_half_freq": first_freq,
                "second_half_freq": second_freq,
                "direction": direction,
                "change_ratio": round(second_freq / max(first_freq, 0.1), 1),
            }
        evidence["co_appearing_persons"] = [
            {"person_id": pid, "co_count": count, "co_ratio": round(count / len(co_photos), 2)}
            for pid, count in third_party.most_common(8)
            if count >= 2
        ]
        evidence["contact_types"] = _unique_strings(evidence["contact_types"])
        return evidence


def _build_time_range(started_at: str, ended_at: str) -> str:
    if not started_at:
        return ""
    start = started_at[11:16] if len(started_at) >= 16 else ""
    end = ended_at[11:16] if ended_at and len(ended_at) >= 16 else start
    return f"{start} - {end}".strip(" -")


def _build_empty_structured_profile() -> Dict[str, Any]:
    return {
        "long_term_facts": {
            "identity": _section(["name", "gender", "age_range", "role", "race", "nationality"]),
            "social_identity": _section(["education", "career", "career_phase", "professional_dedication", "language_culture", "political_preference"]),
            "material": _section(["asset_level", "spending_style", "brand_preference", "income_model", "signature_items"]),
            "geography": _section(["location_anchors", "mobility_pattern", "cross_border"]),
            "time": _section(["life_rhythm", "event_cycles", "sleep_pattern"]),
            "relationships": _section(["intimate_partner", "close_circle_size", "social_groups", "pets", "parenting", "living_situation"]),
            "hobbies": _section(["interests", "frequent_activities", "solo_vs_social"]),
            "physiology": _section(["fitness_level", "health_conditions", "diet_mode"]),
        },
        "short_term_facts": _section(["life_events", "phase_change", "spending_shift", "current_displacement", "recent_habits", "recent_interests", "physiological_state"]),
        "long_term_expression": _section(["personality_mbti", "morality", "philosophy", "attitude_style", "aesthetic_tendency", "visual_creation_style"]),
        "short_term_expression": _section(["current_mood", "mental_state", "motivation_shift", "stress_signal", "social_energy"]),
    }


class V0325PipelineFamily(V0323PipelineFamily):
    """v0325 family built on top of v0323 VP1/LP1 normalization."""

    def __init__(
        self,
        *,
        task_id: str,
        task_dir: Path,
        user_id: Optional[str],
        asset_store: Any,
        llm_processor: Any,
        public_url_builder: Callable[[Path], str],
    ) -> None:
        super().__init__(
            task_id=task_id,
            task_dir=task_dir,
            user_id=user_id,
            asset_store=asset_store,
            llm_processor=llm_processor,
            public_url_builder=public_url_builder,
        )
        self.family_dir = self.task_dir / "v0325"
        self.family_dir.mkdir(parents=True, exist_ok=True)
        self.vp1_path = self.family_dir / "vp1_observations.json"
        self.lp1_batch_requests_path = self.family_dir / "lp1_batch_requests.jsonl"
        self.lp1_batch_outputs_path = self.family_dir / "lp1_batch_outputs.jsonl"
        self.lp1_event_cards_path = self.family_dir / "lp1_event_cards.jsonl"
        self.lp1_events_path = self.family_dir / "lp1_events.jsonl"
        self.lp1_events_compact_path = self.family_dir / "lp1_events_compact.json"
        self.lp1_continuation_log_path = self.family_dir / "lp1_event_continuation_log.jsonl"
        self.lp1_parse_failures_path = self.family_dir / "lp1_parse_failures.json"
        self.lp1_salvaged_events_path = self.family_dir / "lp1_salvaged_events.jsonl"
        self.lp1_salvage_report_path = self.family_dir / "lp1_salvage_report.json"
        self.lp2_relationships_jsonl_path = self.family_dir / "lp2_relationships.jsonl"
        self.lp2_relationships_path = self.family_dir / "lp2_relationships.json"
        self.lp3_profile_path = self.family_dir / "lp3_profile.json"
        self.structured_profile_path = self.family_dir / "structured_profile.json"
        self.relationship_dossiers_path = self.family_dir / "relationship_dossiers.json"
        self.group_artifacts_path = self.family_dir / "group_artifacts.json"
        self.profile_fact_decisions_path = self.family_dir / "profile_fact_decisions.json"
        self.downstream_audit_report_path = self.family_dir / "downstream_audit_report.json"
        self.llm_failures_path = self.family_dir / "llm_failures.jsonl"
        self.memory_snapshot_path = self.family_dir / "memory_snapshot.json"
        self.raw_manifest_path = self.family_dir / "raw_upstream_manifest.json"
        self.raw_index_path = self.family_dir / "raw_upstream_index.json"

    def _lp1_hard_output_contract_lines(self) -> List[str]:
        return [
            "HARD_OUTPUT_CONTRACT",
            f"- contract_version: {LP1_CONTRACT_VERSION_V0325}",
            "- Every event must include a non-empty supporting_photo_ids array.",
            "- supporting_photo_ids must come only from PHOTO_INDEX.",
            "- Every event must touch at least one OUTPUT_WINDOW photo.",
            "- If you are unsure, omit the event instead of emitting an invalid event.",
            "- Output must be complete, balanced JSON. Do not stop mid-object, mid-array, or mid-string.",
        ]

    def _call_json_prompt_raw_response(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: Optional[tuple[int | float, int | float]] = None,
    ) -> Dict[str, Any]:
        raw_handler = getattr(self.llm_processor, "_call_json_prompt_raw_response", None)
        if callable(raw_handler):
            payload = raw_handler(
                prompt,
                max_tokens=max_tokens,
                response_format=response_format,
                timeout=timeout,
            )
            if isinstance(payload, dict):
                return {
                    "text": str(payload.get("text") or ""),
                    "provider_response_id": payload.get("provider_response_id"),
                    "provider_finish_reason": payload.get("provider_finish_reason"),
                    "provider_usage": payload.get("provider_usage"),
                }
        return {
            "text": super()._call_json_prompt_raw_text(
                prompt,
                max_tokens=max_tokens,
                response_format=response_format,
                timeout=timeout,
            ),
            "provider_response_id": None,
            "provider_finish_reason": None,
            "provider_usage": None,
        }

    def _persist_lp1_salvage_artifacts(
        self,
        *,
        salvage_reports: Sequence[Dict[str, Any]],
        salvaged_events: Sequence[Dict[str, Any]],
    ) -> None:
        deduped_events: List[Dict[str, Any]] = []
        seen_signatures = set()
        for item in salvaged_events:
            normalized_event = item.get("normalized_event") if isinstance(item, dict) else None
            signature = (
                str(item.get("batch_id") if isinstance(item, dict) else ""),
                str(item.get("event_id") if isinstance(item, dict) else ""),
                json.dumps(normalized_event or {}, ensure_ascii=False, sort_keys=True),
            )
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            deduped_events.append(dict(item))
        if deduped_events:
            _write_jsonl(self.lp1_salvaged_events_path, deduped_events)
        else:
            _safe_unlink(self.lp1_salvaged_events_path)
        if salvage_reports or deduped_events:
            save_json(
                {
                    "summary": {
                        "salvage_detected": bool(salvage_reports),
                        "salvaged_event_count": len(deduped_events),
                        "contract_version": LP1_CONTRACT_VERSION_V0325,
                    },
                    "batches": list(salvage_reports),
                },
                str(self.lp1_salvage_report_path),
            )
        else:
            _safe_unlink(self.lp1_salvage_report_path)

    def _relative_artifact_path(self, path: Path) -> str:
        return path.relative_to(self.task_dir).as_posix()

    def _persist_lp3_sidecar_artifacts(
        self,
        *,
        state: MemoryState,
        field_decisions: Sequence[Dict[str, Any]],
        audit_payload: Dict[str, Any],
        compact_structured: Dict[str, Any],
    ) -> Dict[str, str]:
        relationship_dossiers = [dossier.to_dict() for dossier in state.relationship_dossiers]
        group_artifacts = [group.to_dict() for group in state.groups]
        compact_field_decisions = [self._compact_profile_fact_decision(item) for item in field_decisions]
        save_json(compact_structured, str(self.structured_profile_path))
        save_json(relationship_dossiers, str(self.relationship_dossiers_path))
        save_json(group_artifacts, str(self.group_artifacts_path))
        save_json(compact_field_decisions, str(self.profile_fact_decisions_path))
        if self._debug_profile_trace_enabled():
            with gzip.open(self.profile_fact_decisions_path.with_suffix(".full.json.gz"), "wt", encoding="utf-8") as handle:
                json.dump(list(field_decisions), handle, indent=2, ensure_ascii=False)
        save_json(audit_payload, str(self.downstream_audit_report_path))
        return {
            "structured_profile_path": self._relative_artifact_path(self.structured_profile_path),
            "relationship_dossiers_path": self._relative_artifact_path(self.relationship_dossiers_path),
            "group_artifacts_path": self._relative_artifact_path(self.group_artifacts_path),
            "profile_fact_decisions_path": self._relative_artifact_path(self.profile_fact_decisions_path),
            "downstream_audit_report_path": self._relative_artifact_path(self.downstream_audit_report_path),
        }

    def _debug_profile_trace_enabled(self) -> bool:
        return str(os.getenv("DEBUG_PROFILE_TRACE", "")).strip().lower() in {"1", "true", "yes", "on"}

    def _compact_profile_fact_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        tool_trace = dict(decision.get("tool_trace") or {})
        evidence_bundle = dict(tool_trace.get("evidence_bundle") or {})
        compact = dict(evidence_bundle.get("compact") or {})
        final = dict(decision.get("final") or {})
        draft = dict(decision.get("draft") or {})
        supporting_ids = list(final.get("supporting_ref_ids", []) or draft.get("supporting_ref_ids", []) or [])
        contradicting_ids = list(final.get("contradicting_ref_ids", []) or draft.get("contradicting_ref_ids", []) or [])
        return {
            "field_key": decision.get("field_key"),
            "batch_name": decision.get("batch_name"),
            "final": final,
            "null_reason": decision.get("null_reason") or final.get("null_reason") or draft.get("null_reason"),
            "supporting_ids": supporting_ids,
            "contradicting_ids": contradicting_ids,
            "source_coverage": compact.get("source_coverage", evidence_bundle.get("source_coverage", {})),
            "representative_refs": {
                "events": list(compact.get("representative_events", []) or []),
                "photos": list(compact.get("representative_photos", []) or []),
                "top_candidates": list(compact.get("top_candidates", []) or []),
            },
        }

    def _build_batches(self, observations: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        batches: List[Dict[str, Any]] = []
        total = len(observations)
        if total == 0:
            return batches
        batch_index = 1
        new_start = 0
        while new_start < total:
            new_end = min(total, new_start + LP1_BATCH_SIZE)
            batch_start = 0 if new_start == 0 else max(0, new_start - LP1_OVERLAP_SIZE)
            input_records = list(observations[batch_start:new_end])
            overlap_records = list(observations[batch_start:new_start])
            output_window_records = list(observations[new_start:new_end])
            batches.append(
                {
                    "batch_id": f"BATCH_{batch_index:04d}",
                    "batch_index": batch_index,
                    "input_records": input_records,
                    "overlap_records": overlap_records,
                    "output_window_records": output_window_records,
                    "input_photo_ids": [item["photo_id"] for item in input_records],
                    "overlap_context_photo_ids": [item["photo_id"] for item in overlap_records],
                    "output_window_photo_ids": [item["photo_id"] for item in output_window_records],
                }
            )
            batch_index += 1
            new_start += LP1_BATCH_SIZE
        return batches

    def _run_lp1_batches(
        self,
        observations: Sequence[Dict[str, Any]],
        *,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        batches = self._build_batches(observations)
        for path in (
            self.lp1_batch_requests_path,
            self.lp1_batch_outputs_path,
            self.lp1_event_cards_path,
            self.lp1_events_path,
            self.lp1_events_compact_path,
            self.lp1_continuation_log_path,
            self.lp1_parse_failures_path,
            self.lp1_salvaged_events_path,
            self.lp1_salvage_report_path,
        ):
            _safe_unlink(path)
        final_events_by_id: Dict[str, Dict[str, Any]] = {}
        ordered_event_ids: List[str] = []
        continuation_log: List[Dict[str, Any]] = []
        batch_cards: List[Dict[str, Any]] = []
        parse_failures: List[Dict[str, Any]] = []
        batch_summaries: List[Dict[str, Any]] = []
        salvage_reports: List[Dict[str, Any]] = []
        salvaged_events_log: List[Dict[str, Any]] = []

        for batch in batches:
            carryover_cards = self._select_carryover_event_cards(
                final_events_by_id=final_events_by_id,
                ordered_event_ids=ordered_event_ids,
                overlap_context_photo_ids=batch["overlap_context_photo_ids"],
            )
            prompt, request_record = self._build_lp1_batch_prompt(
                batch=batch,
                carryover_cards=carryover_cards,
            )
            max_attempts = max(1, V0323_LP1_MAX_ATTEMPTS)
            parsed_output: Optional[Dict[str, Any]] = None
            parse_status = "failed"
            last_error: Optional[Exception] = None
            batch_salvage_count = 0
            batch_salvage_status = "none"
            batch_salvage_signatures = set()
            for attempt in range(1, max_attempts + 1):
                attempt_record = dict(request_record)
                attempt_record["attempt"] = attempt
                attempt_record["max_attempts"] = max_attempts
                attempt_record["prompt_kind"] = "primary" if attempt == 1 else "retry"
                _append_jsonl(self.lp1_batch_requests_path, attempt_record)
                attempt_result = self._call_lp1_batch_attempt(
                    prompt=prompt,
                    batch=batch,
                    request_record=attempt_record,
                )
                _append_jsonl(self.lp1_batch_outputs_path, attempt_result["record"])
                salvaged_events = list(attempt_result.get("salvaged_events") or [])
                if salvaged_events:
                    batch_salvage_status = "detected"
                    for salvage_item in salvaged_events:
                        normalized_event = salvage_item.get("normalized_event") if isinstance(salvage_item, dict) else None
                        signature = (
                            str(salvage_item.get("batch_id") if isinstance(salvage_item, dict) else ""),
                            str(salvage_item.get("event_id") if isinstance(salvage_item, dict) else ""),
                            json.dumps(normalized_event or {}, ensure_ascii=False, sort_keys=True),
                        )
                        if signature in batch_salvage_signatures:
                            continue
                        batch_salvage_signatures.add(signature)
                        batch_salvage_count += 1
                    salvaged_events_log.extend(salvaged_events)
                salvage_report = attempt_result.get("salvage_report")
                if isinstance(salvage_report, dict):
                    salvage_reports.append(salvage_report)
                if salvaged_events or salvage_report:
                    self._persist_lp1_salvage_artifacts(
                        salvage_reports=salvage_reports,
                        salvaged_events=salvaged_events_log,
                    )
                if attempt_result["ok"]:
                    parsed_output = attempt_result["parsed_output"]
                    parse_status = "ok" if attempt == 1 else "retry_ok"
                    break
                last_error = attempt_result["error"]

            if parsed_output is None:
                failure = {
                    "batch_id": batch["batch_id"],
                    "batch_index": batch["batch_index"],
                    "error": f"LP1 batch failed after {max_attempts} attempts: {last_error}",
                    "salvage_status": batch_salvage_status,
                    "salvaged_event_count": batch_salvage_count,
                    "contract_version": LP1_CONTRACT_VERSION_V0325,
                }
                parse_failures.append(failure)
                save_json(parse_failures, str(self.lp1_parse_failures_path))
                self._persist_lp1_salvage_artifacts(
                    salvage_reports=salvage_reports,
                    salvaged_events=salvaged_events_log,
                )
                raise RuntimeError(failure["error"])

            continuation_entries = self._apply_lp1_batch_events(
                batch=batch,
                batch_output=parsed_output,
                final_events_by_id=final_events_by_id,
                ordered_event_ids=ordered_event_ids,
            )
            continuation_log.extend(continuation_entries)
            next_cards = self._select_carryover_event_cards(
                final_events_by_id=final_events_by_id,
                ordered_event_ids=ordered_event_ids,
                overlap_context_photo_ids=batch["output_window_photo_ids"][-LP1_OVERLAP_SIZE:],
            )
            batch_cards.extend(
                {
                    "batch_id": batch["batch_id"],
                    "batch_index": batch["batch_index"],
                    **card,
                }
                for card in next_cards
            )
            batch_summary = {
                "batch_id": batch["batch_id"],
                "batch_index": batch["batch_index"],
                "input_photo_ids": list(batch["input_photo_ids"]),
                "overlap_context_photo_ids": list(batch["overlap_context_photo_ids"]),
                "output_window_photo_ids": list(batch["output_window_photo_ids"]),
                "carryover_event_refs": [card["event_id"] for card in carryover_cards],
                "raw_event_count": len(parsed_output["events"]),
                "parse_status": parse_status,
                "prompt_version": LP1_PROMPT_VERSION_V0325,
                "contract_version": LP1_CONTRACT_VERSION_V0325,
                "salvage_status": batch_salvage_status,
                "salvaged_event_count": batch_salvage_count,
            }
            batch_summaries.append(batch_summary)
            self._notify(
                progress_callback,
                {
                    "message": "v0325 LP1 分批事件聚合中",
                    "pipeline_family": PIPELINE_FAMILY_V0325,
                    "substage": "lp1_batch",
                    "batch_id": batch["batch_id"],
                    "batch_index": batch["batch_index"],
                    "batch_count": len(batches),
                    "event_count": len(ordered_event_ids),
                    "continuation_count": len(continuation_log),
                    "percent": round((batch["batch_index"] / max(1, len(batches))) * 100),
                },
            )

        final_events = [final_events_by_id[event_id] for event_id in ordered_event_ids]

        _write_jsonl(self.lp1_event_cards_path, batch_cards)
        _write_jsonl(self.lp1_events_path, final_events)
        _write_jsonl(self.lp1_continuation_log_path, continuation_log)
        save_json(parse_failures, str(self.lp1_parse_failures_path))
        save_json(final_events, str(self.lp1_events_compact_path))
        self._persist_lp1_salvage_artifacts(
            salvage_reports=salvage_reports,
            salvaged_events=salvaged_events_log,
        )

        return {
            "lp1_batches": batch_summaries,
            "lp1_events": final_events,
            "lp1_event_continuation_log": continuation_log,
        }

    def _build_lp1_batch_prompt(
        self,
        *,
        batch: Dict[str, Any],
        carryover_cards: Sequence[Dict[str, Any]],
    ) -> tuple[str, Dict[str, Any]]:
        batch_meta = {
            "batch_id": batch["batch_id"],
            "batch_index": batch["batch_index"],
            "prompt_version": LP1_PROMPT_VERSION_V0325,
            "contract_version": LP1_CONTRACT_VERSION_V0325,
            "input_photo_count": len(batch["input_records"]),
            "overlap_photo_count": len(batch["overlap_records"]),
            "output_window_photo_count": len(batch["output_window_records"]),
        }
        overlap_blocks = [self._format_photo_record(item) for item in batch["overlap_records"]]
        output_window_blocks = [self._format_photo_record(item) for item in batch["output_window_records"]]
        card_blocks = [self._format_event_card(card) for card in carryover_cards]

        sections = [
            "LP1 Batch Analysis Task",
            "You are a senior anthropologist and social-behavior analyst working on a personal photo-memory pipeline.",
            "Your job is to read the ordered photo-level VLM records and reconstruct coherent events using v0139-style time-space-behavior clustering.",
            "",
            "Batch rules:",
            "1. Photos are already stably ordered. Read them strictly in order.",
            "2. OVERLAP_CONTEXT_PHOTOS are context only. They help you decide whether an event continues across the batch boundary.",
            "3. Output analysis only for events that include at least one OUTPUT_WINDOW photo.",
            "4. If an old event clearly continues into the output window, analyze it as one event that spans overlap + output-window photos, but do not separately restate overlap-only events.",
            "5. Always cite exact photo_id groups when describing each event. This is mandatory.",
            "6. Favor fewer, higher-confidence events over speculative over-splitting.",
            "7. Keep the analysis rich but concise. Avoid dumping giant repeated descriptions.",
            "",
            "Write a compact analytical memo in plain text.",
            "For each event, include:",
            "- involved photo_ids",
            "- time span / temporal logic",
            "- location or location nature",
            "- main participants",
            "- objective scene facts",
            "- narrative synthesis",
            "- social dynamics",
            "- persona evidence",
            "- tags",
            "- confidence and reason",
            "",
            "BATCH_META",
            json.dumps(batch_meta, ensure_ascii=False, indent=2),
            "",
            "OVERLAP_CONTEXT_PHOTOS",
            "\n\n".join(overlap_blocks) if overlap_blocks else "NONE",
            "",
            "OUTPUT_WINDOW_PHOTOS",
            "\n\n".join(output_window_blocks),
            "",
            "CARRYOVER_EVENT_CARDS",
            "\n\n".join(card_blocks) if card_blocks else "NONE",
            "",
            *self._lp1_hard_output_contract_lines(),
        ]
        prompt = "\n".join(sections).strip()
        request_record = {
            "batch_id": batch["batch_id"],
            "batch_index": batch["batch_index"],
            "prompt_version": LP1_PROMPT_VERSION_V0325,
            "contract_version": LP1_CONTRACT_VERSION_V0325,
            "prompt_char_count": len(prompt),
            "input_photo_ids": list(batch["input_photo_ids"]),
            "overlap_context_photo_ids": list(batch["overlap_context_photo_ids"]),
            "output_window_photo_ids": list(batch["output_window_photo_ids"]),
            "carryover_event_card_ids": [card["event_id"] for card in carryover_cards],
            "prompt_sections": {
                "batch_meta": batch_meta,
                "overlap_context_photos": [self._prompt_photo_payload(item) for item in batch["overlap_records"]],
                "output_window_photos": [self._prompt_photo_payload(item) for item in batch["output_window_records"]],
                "carryover_event_cards": [dict(card) for card in carryover_cards],
                "hard_output_contract": self._lp1_hard_output_contract_lines()[1:],
            },
        }
        return prompt, request_record

    def _build_lp1_convert_prompt(
        self,
        *,
        batch: Dict[str, Any],
        analysis_text: str,
    ) -> str:
        trimmed_analysis = str(analysis_text or "").strip()
        if len(trimmed_analysis) > LP1_ANALYSIS_TEXT_CHAR_LIMIT:
            trimmed_analysis = trimmed_analysis[:LP1_ANALYSIS_TEXT_CHAR_LIMIT]
        photo_index = [
            {
                "photo_id": item["photo_id"],
                "timestamp": item.get("timestamp"),
                "face_person_ids": list(item.get("face_person_ids", []) or []),
                "location_name": str(dict(item.get("location") or {}).get("name") or "").strip(),
                "summary": str(dict(item.get("vlm_analysis") or {}).get("summary") or "").strip(),
            }
            for item in batch["input_records"]
        ]
        payload_example = {
            "events": [
                {
                    "event_id": "TEMP_EVT_001",
                    "supporting_photo_ids": ["photo_001", "photo_002"],
                    "meta_info": {
                        "title": "事件标题",
                        "location_context": "地点性质",
                        "photo_count": 2,
                    },
                    "objective_fact": {
                        "scene_description": "客观场景事实",
                        "participants": ["Person_001", "Person_002"],
                    },
                    "narrative_synthesis": "一句话深度还原事件。",
                    "social_dynamics": [],
                    "persona_evidence": {
                        "behavioral": [],
                        "aesthetic": [],
                        "socioeconomic": [],
                    },
                    "tags": ["#标签"],
                    "confidence": 0.8,
                    "reason": "时间、地点、人物与行为证据",
                }
            ]
        }
        sections = [
            "Convert the following LP1 batch analysis into JSON.",
            "Return JSON only. No markdown. No explanations. No code fences.",
            "Top-level key must be events.",
            "Rules:",
            "1. Every event must include supporting_photo_ids.",
            "2. supporting_photo_ids must come only from the provided PHOTO_INDEX.",
            "3. Every event must touch at least one OUTPUT_WINDOW photo.",
            "4. If an event spans overlap and output-window photos, include both in supporting_photo_ids.",
            "5. Do not output overlap-only events.",
            "6. Keep strings concise but preserve social_dynamics, persona_evidence, tags, confidence, and reason.",
            "",
            "OUTPUT_WINDOW_PHOTO_IDS",
            json.dumps(list(batch["output_window_photo_ids"]), ensure_ascii=False, indent=2),
            "",
            "PHOTO_INDEX",
            json.dumps(photo_index, ensure_ascii=False, indent=2),
            "",
            "JSON_FORMAT",
            json.dumps(payload_example, ensure_ascii=False, indent=2),
            "",
            "ANALYSIS_TEXT",
            trimmed_analysis or "NONE",
            "",
            *self._lp1_hard_output_contract_lines(),
        ]
        return "\n".join(sections).strip()

    def _normalize_lp1_event(self, *, item: Dict[str, Any], batch: Dict[str, Any], index: int) -> Dict[str, Any]:
        if not isinstance(item, dict):
            raise ValueError("event item must be an object")
        meta_info = dict(item.get("meta_info") or {})
        objective_fact = dict(item.get("objective_fact") or {})
        supporting_photo_ids = _sorted_unique_photo_ids(
            [
                *list(item.get("supporting_photo_ids", []) or []),
                *list(item.get("photo_ids", []) or []),
                *list(meta_info.get("supporting_photo_ids", []) or []),
                *list(objective_fact.get("supporting_photo_ids", []) or []),
            ],
            order_index=self.photo_order_index,
        )
        invalid_photo_ids = [photo_id for photo_id in supporting_photo_ids if photo_id not in set(batch["input_photo_ids"])]
        if invalid_photo_ids:
            raise ValueError(f"supporting_photo_ids outside batch input: {invalid_photo_ids}")
        if not supporting_photo_ids:
            raise ValueError("event must include supporting_photo_ids")
        output_window_support = [photo_id for photo_id in supporting_photo_ids if photo_id in set(batch["output_window_photo_ids"])]
        if not output_window_support:
            raise ValueError("event must touch at least one output-window photo")
        anchor_photo_id = str(item.get("anchor_photo_id") or "").strip()
        if anchor_photo_id not in set(output_window_support):
            anchor_photo_id = output_window_support[0]
        started_at, ended_at = self._derive_event_bounds(
            supporting_photo_ids=supporting_photo_ids,
            started_at="",
            ended_at="",
        )
        persona_evidence = self._normalize_persona_evidence(item.get("persona_evidence"))
        social_dynamics = self._normalize_social_dynamics(item.get("social_dynamics"))
        place_refs = _unique_strings([*list(item.get("place_refs", []) or []), meta_info.get("location_context")])
        if not place_refs:
            place_refs = self._derive_place_refs(supporting_photo_ids)
        participant_person_ids = _unique_strings(
            [
                *list(item.get("participant_person_ids", []) or []),
                *list(item.get("participants", []) or []),
                *list(objective_fact.get("participants", []) or []),
            ]
        )
        depicted_person_ids = _unique_strings(item.get("depicted_person_ids", []))
        if not depicted_person_ids:
            depicted_person_ids = self._derive_depicted_people(supporting_photo_ids)
        if not participant_person_ids:
            participant_person_ids = list(depicted_person_ids)
        title = _normalized_text(meta_info.get("title") or item.get("title"))
        if not title:
            title = f"Batch Event {index:03d}"
        scene_description = _normalized_text(objective_fact.get("scene_description") or item.get("scene_description"))
        narrative_synthesis = _normalized_text(item.get("narrative_synthesis") or item.get("summary") or scene_description)
        return {
            "temp_event_id": str(item.get("event_id") or item.get("temp_event_id") or "").strip() or f"TEMP_EVT_{index:03d}",
            "anchor_photo_id": anchor_photo_id,
            "supporting_photo_ids": supporting_photo_ids,
            "started_at": started_at,
            "ended_at": ended_at,
            "title": title,
            "narrative_synthesis": narrative_synthesis,
            "participant_person_ids": participant_person_ids,
            "depicted_person_ids": depicted_person_ids,
            "place_refs": place_refs,
            "social_dynamics": social_dynamics,
            "persona_evidence": persona_evidence,
            "tags": _unique_strings(item.get("tags", [])),
            "confidence": max(0.0, min(1.0, _safe_float(item.get("confidence"), default=0.0))),
            "reason": _normalized_text(item.get("reason")),
            "meta_info": {
                "title": title,
                "location_context": meta_info.get("location_context") or (place_refs[0] if place_refs else ""),
                "photo_count": len(supporting_photo_ids),
            },
            "objective_fact": {
                "scene_description": scene_description,
                "participants": list(participant_person_ids),
            },
        }

    def _call_lp1_batch_attempt(
        self,
        *,
        prompt: str,
        batch: Dict[str, Any],
        request_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        output_record = {
            "batch_id": batch["batch_id"],
            "batch_index": batch["batch_index"],
            "prompt_kind": request_record.get("prompt_kind"),
            "attempt": _safe_int(request_record.get("attempt"), default=1),
            "max_attempts": _safe_int(request_record.get("max_attempts"), default=1),
            "prompt_char_count": _safe_int(request_record.get("prompt_char_count")),
            "parse_status": "failed",
            "strategy": "v0139_two_step_v0325",
            "prompt_version": LP1_PROMPT_VERSION_V0325,
            "contract_version": LP1_CONTRACT_VERSION_V0325,
            "salvage_status": "none",
            "salvaged_event_count": 0,
        }

        analysis_response = self._call_json_prompt_raw_response(
            prompt,
            max_tokens=V0323_LP1_MAX_OUTPUT_TOKENS,
        )
        analysis_text = str(analysis_response.get("text") or "")
        output_record.update(
            {
                "analysis_response_char_count": len(analysis_text),
                "analysis_response_preview": _truncate_preview(analysis_text, limit=4000),
                "analysis_response_tail": str(analysis_text or "")[-4000:],
                "analysis_response_id": analysis_response.get("provider_response_id"),
                "analysis_finish_reason": analysis_response.get("provider_finish_reason"),
                "analysis_usage": analysis_response.get("provider_usage"),
            }
        )

        analysis_error: Optional[Exception] = None
        try:
            extracted = self._extract_json_from_text(analysis_text, target_key="events")
            if extracted:
                normalized = self._normalize_lp1_batch_output(payload=extracted, batch=batch)
                output_record.update(
                    {
                        "parse_status": "analysis_ok",
                        "event_count": len(normalized["events"]),
                        "output": normalized,
                    }
                )
                return {"ok": True, "parsed_output": normalized, "record": output_record}
        except Exception as exc:
            analysis_error = exc
            output_record.update(
                {
                    "analysis_error_type": type(exc).__name__,
                    "analysis_error": str(exc),
                }
            )

        convert_prompt = self._build_lp1_convert_prompt(batch=batch, analysis_text=analysis_text)
        output_record["convert_prompt_char_count"] = len(convert_prompt)
        convert_response = self._call_json_prompt_raw_response(
            convert_prompt,
            max_tokens=V0323_LP1_MAX_OUTPUT_TOKENS,
            response_format=self._lp1_convert_response_format(),
        )
        convert_text = str(convert_response.get("text") or "")
        output_record.update(
            {
                "convert_response_char_count": len(convert_text),
                "convert_response_preview": _truncate_preview(convert_text, limit=4000),
                "convert_response_tail": str(convert_text or "")[-4000:],
                "convert_response_id": convert_response.get("provider_response_id"),
                "convert_finish_reason": convert_response.get("provider_finish_reason"),
                "convert_usage": convert_response.get("provider_usage"),
            }
        )

        try:
            payload = self._extract_json_from_text(convert_text, target_key="events")
            if not payload:
                raise ValueError("convert step did not produce events JSON")
            normalized = self._normalize_lp1_batch_output(payload=payload, batch=batch)
        except Exception as exc:
            salvaged_output: List[Dict[str, Any]] = []
            salvaged_event_ids: List[str] = []
            salvage_errors: List[str] = []
            for event_index, partial_item in enumerate(_extract_partial_event_objects(convert_text), start=1):
                try:
                    normalized_event = self._normalize_lp1_event(item=partial_item, batch=batch, index=event_index)
                except Exception as salvage_exc:
                    salvage_errors.append(str(salvage_exc))
                    continue
                event_id = str(partial_item.get("event_id") or partial_item.get("temp_event_id") or normalized_event.get("temp_event_id") or "").strip()
                salvaged_event_ids.append(event_id or f"SALVAGED_{event_index:03d}")
                salvaged_output.append(
                    {
                        "debug_only": True,
                        "batch_id": batch["batch_id"],
                        "batch_index": batch["batch_index"],
                        "attempt": output_record["attempt"],
                        "parse_status": "partial_salvage",
                        "event_id": event_id,
                        "normalized_event": normalized_event,
                    }
                )
            salvage_report = None
            if salvaged_output:
                output_record.update(
                    {
                        "salvage_status": "detected",
                        "salvaged_event_count": len(salvaged_output),
                        "salvaged_event_ids": salvaged_event_ids,
                    }
                )
                salvage_report = {
                    "batch_id": batch["batch_id"],
                    "batch_index": batch["batch_index"],
                    "attempt": output_record["attempt"],
                    "salvage_status": "detected",
                    "salvaged_event_count": len(salvaged_output),
                    "salvaged_event_ids": salvaged_event_ids,
                    "error": str(exc),
                    "provider_finish_reason": convert_response.get("provider_finish_reason"),
                    "response_id": convert_response.get("provider_response_id"),
                    "usage": convert_response.get("provider_usage"),
                    "salvage_error_count": len(salvage_errors),
                    "salvage_errors": salvage_errors[:10],
                }
            output_record.update(
                {
                    "parse_status": "convert_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if analysis_error is not None:
                output_record["upstream_analysis_error"] = str(analysis_error)
            return {
                "ok": False,
                "error": exc,
                "record": output_record,
                "salvaged_events": salvaged_output,
                "salvage_report": salvage_report,
            }

        output_record.update(
            {
                "parse_status": "convert_ok",
                "event_count": len(normalized["events"]),
                "output": normalized,
            }
        )
        return {"ok": True, "parsed_output": normalized, "record": output_record}

    def run(
        self,
        *,
        photos: List[Any],
        face_output: Dict[str, Any],
        primary_person_id: Optional[str],
        vlm_results: List[Dict[str, Any]],
        cached_photo_ids: Sequence[str],
        dedupe_report: Dict[str, Any],
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        observations = self._build_vp1_observations(photos, vlm_results)
        return self.run_from_observations(
            observations=observations,
            face_output=face_output,
            primary_person_id=primary_person_id,
            cached_photo_ids=cached_photo_ids,
            dedupe_report=dedupe_report,
            progress_callback=progress_callback,
        )

    def run_from_observations(
        self,
        *,
        observations: Sequence[Dict[str, Any]],
        face_output: Dict[str, Any],
        primary_person_id: Optional[str],
        cached_photo_ids: Sequence[str],
        dedupe_report: Dict[str, Any],
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        observations = [dict(item) for item in list(observations or [])]
        self.observation_index = {item["photo_id"]: item for item in observations}
        self.photo_order_index = {
            item["photo_id"]: int(item["sequence_index"])
            for item in observations
        }
        save_json(observations, str(self.vp1_path))

        lp1_payload = self._run_lp1_batches(observations, progress_callback=progress_callback)
        return self._run_from_lp1_payload(
            observations=observations,
            face_output=face_output,
            primary_person_id=primary_person_id,
            cached_photo_ids=cached_photo_ids,
            dedupe_report=dedupe_report,
            lp1_payload=lp1_payload,
            progress_callback=progress_callback,
        )

    def run_from_precomputed(
        self,
        *,
        observations: Sequence[Dict[str, Any]],
        face_output: Dict[str, Any],
        primary_person_id: Optional[str],
        cached_photo_ids: Sequence[str],
        dedupe_report: Dict[str, Any],
        lp1_events: Sequence[Dict[str, Any]],
        lp1_batches: Sequence[Dict[str, Any]] | None = None,
        lp1_event_continuation_log: Sequence[Dict[str, Any]] | None = None,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        observations = [dict(item) for item in list(observations or [])]
        self.observation_index = {item["photo_id"]: item for item in observations}
        self.photo_order_index = {
            item["photo_id"]: int(item["sequence_index"])
            for item in observations
        }
        save_json(observations, str(self.vp1_path))
        lp1_payload = self._persist_precomputed_lp1_payload(
            observations=observations,
            lp1_events=lp1_events,
            lp1_batches=lp1_batches,
            lp1_event_continuation_log=lp1_event_continuation_log,
        )
        return self._run_from_lp1_payload(
            observations=observations,
            face_output=face_output,
            primary_person_id=primary_person_id,
            cached_photo_ids=cached_photo_ids,
            dedupe_report=dedupe_report,
            lp1_payload=lp1_payload,
            progress_callback=progress_callback,
        )

    def run_from_lp2_checkpoint(
        self,
        *,
        observations: Sequence[Dict[str, Any]],
        face_output: Dict[str, Any],
        primary_person_id: Optional[str],
        cached_photo_ids: Sequence[str],
        dedupe_report: Dict[str, Any],
        lp1_events: Sequence[Dict[str, Any]],
        lp2_relationships: Sequence[Dict[str, Any]],
        relationship_dossiers: Sequence[Dict[str, Any]],
        group_artifacts: Sequence[Dict[str, Any]],
        lp1_batches: Sequence[Dict[str, Any]] | None = None,
        lp1_event_continuation_log: Sequence[Dict[str, Any]] | None = None,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        observations = [dict(item) for item in list(observations or [])]
        self.observation_index = {item["photo_id"]: item for item in observations}
        self.photo_order_index = {
            item["photo_id"]: int(item["sequence_index"])
            for item in observations
        }
        save_json(observations, str(self.vp1_path))
        lp1_payload = self._persist_precomputed_lp1_payload(
            observations=observations,
            lp1_events=lp1_events,
            lp1_batches=lp1_batches,
            lp1_event_continuation_log=lp1_event_continuation_log,
        )
        return self._run_from_lp2_payload(
            observations=observations,
            face_output=face_output,
            primary_person_id=primary_person_id,
            cached_photo_ids=cached_photo_ids,
            dedupe_report=dedupe_report,
            lp1_payload=lp1_payload,
            lp2_relationships=lp2_relationships,
            relationship_dossiers=relationship_dossiers,
            group_artifacts=group_artifacts,
            progress_callback=progress_callback,
        )

    def _run_from_lp1_payload(
        self,
        *,
        observations: Sequence[Dict[str, Any]],
        face_output: Dict[str, Any],
        primary_person_id: Optional[str],
        cached_photo_ids: Sequence[str],
        dedupe_report: Dict[str, Any],
        lp1_payload: Dict[str, Any],
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:

        self._notify(
            progress_callback,
            {
                "message": "v0325 raw bootstrap 中",
                "pipeline_family": PIPELINE_FAMILY_V0325,
                "substage": "lp2_relationship",
                "percent": 5,
            },
        )
        raw_upstream, raw_manifest, raw_index = self._build_raw_upstream_payload(
            face_output=face_output,
            observations=observations,
            lp1_payload=lp1_payload,
            dedupe_report=dedupe_report,
        )
        save_json(raw_manifest, str(self.raw_manifest_path))
        save_json(raw_index, str(self.raw_index_path))

        state = self._build_memory_state(
            face_output=face_output,
            observations=observations,
            lp1_events=lp1_payload["lp1_events"],
            fallback_primary_person_id=primary_person_id,
            raw_upstream=raw_upstream,
            raw_index=raw_index,
        )
        llm_adapter = V0325LLMRuntimeAdapter(
            llm_processor=self.llm_processor,
            primary_person_id_hint=primary_person_id,
        )
        state.screening = agent_screen_people(state)
        primary_decision, primary_reflection = agent_analyze_primary_person_with_reflection(
            state=state,
            fallback_primary_person_id=primary_person_id,
            llm_processor=llm_adapter,
        )
        state.primary_decision = primary_decision.to_dict()
        state.primary_reflection = primary_reflection
        _rebind_primary_aliases(state, previous_primary_person_id=primary_person_id)
        llm_adapter.primary_person_id = (state.primary_decision or {}).get("primary_person_id")

        dossiers = agent_build_relationship_dossiers(state=state, llm_processor=llm_adapter)
        candidate_count = len(dossiers)
        self._notify(
            progress_callback,
            {
                "message": "v0325 LP2 关系推断中",
                "pipeline_family": PIPELINE_FAMILY_V0325,
                "substage": "lp2_relationship",
                "candidate_count": candidate_count,
                "processed_candidates": 0,
                "current_candidate_index": 0,
                "relationship_count": 0,
                "percent": 45 if candidate_count > 0 else 70,
            },
        )

        def _report_relationship_progress(payload: Dict[str, Any]) -> None:
            processed = int(payload.get("processed_candidates") or 0)
            total = int(payload.get("candidate_count") or candidate_count or 0)
            percent = 70 if total <= 0 else round(45 + (processed / total) * 25, 2)
            self._notify(
                progress_callback,
                {
                    "message": "v0325 LP2 关系推断中",
                    "pipeline_family": PIPELINE_FAMILY_V0325,
                    "substage": "lp2_relationship",
                    "candidate_count": total,
                    "processed_candidates": processed,
                    "current_candidate_index": int(payload.get("current_candidate_index") or processed),
                    "last_completed_person_id": payload.get("last_completed_person_id"),
                    "relationship_count": int(payload.get("relationship_count") or 0),
                    "percent": percent,
                },
            )

        relationships, dossiers = agent_infer_relationships_from_dossiers(
            state=state,
            llm_processor=llm_adapter,
            dossiers=dossiers,
            progress_callback=_report_relationship_progress,
        )
        state.relationship_dossiers = dossiers
        state.relationships = relationships
        state.groups = agent_detect_groups(state)
        state.profile_context = agent_build_profile_context(state)

        profile_payload = self._run_lp3_profile(
            state=state,
            llm_processor=llm_adapter,
            progress_callback=progress_callback,
        )
        return self._finalize_memory_payload(
            observations=observations,
            lp1_payload=lp1_payload,
            state=state,
            profile_payload=profile_payload,
            cached_photo_ids=cached_photo_ids,
            dedupe_report=dedupe_report,
            raw_manifest=raw_manifest,
        )

    def _run_from_lp2_payload(
        self,
        *,
        observations: Sequence[Dict[str, Any]],
        face_output: Dict[str, Any],
        primary_person_id: Optional[str],
        cached_photo_ids: Sequence[str],
        dedupe_report: Dict[str, Any],
        lp1_payload: Dict[str, Any],
        lp2_relationships: Sequence[Dict[str, Any]],
        relationship_dossiers: Sequence[Dict[str, Any]],
        group_artifacts: Sequence[Dict[str, Any]],
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        self._notify(
            progress_callback,
            {
                "message": "v0325 LP3 checkpoint 恢复中",
                "pipeline_family": PIPELINE_FAMILY_V0325,
                "substage": "lp3_profile",
                "percent": 72,
            },
        )
        raw_upstream, raw_manifest, raw_index = self._build_raw_upstream_payload(
            face_output=face_output,
            observations=observations,
            lp1_payload=lp1_payload,
            dedupe_report=dedupe_report,
        )
        save_json(raw_manifest, str(self.raw_manifest_path))
        save_json(raw_index, str(self.raw_index_path))

        state = self._build_memory_state(
            face_output=face_output,
            observations=observations,
            lp1_events=lp1_payload["lp1_events"],
            fallback_primary_person_id=primary_person_id,
            raw_upstream=raw_upstream,
            raw_index=raw_index,
        )
        llm_adapter = V0325LLMRuntimeAdapter(
            llm_processor=self.llm_processor,
            primary_person_id_hint=primary_person_id,
        )
        state.screening = agent_screen_people(state)
        primary_decision, primary_reflection = agent_analyze_primary_person_with_reflection(
            state=state,
            fallback_primary_person_id=primary_person_id,
            llm_processor=llm_adapter,
        )
        state.primary_decision = primary_decision.to_dict()
        state.primary_reflection = primary_reflection
        _rebind_primary_aliases(state, previous_primary_person_id=primary_person_id)
        llm_adapter.primary_person_id = (state.primary_decision or {}).get("primary_person_id")

        state.relationship_dossiers = [
            self._relationship_dossier_from_dict(item)
            for item in list(relationship_dossiers or [])
            if isinstance(item, dict)
        ]
        state.relationships = [
            self._relationship_record_from_dict(item)
            for item in list(lp2_relationships or [])
            if isinstance(item, dict)
        ]
        state.groups = [
            self._group_artifact_from_dict(item)
            for item in list(group_artifacts or [])
            if isinstance(item, dict)
        ]
        state.profile_context = agent_build_profile_context(state)

        profile_payload = self._run_lp3_profile(
            state=state,
            llm_processor=llm_adapter,
            progress_callback=progress_callback,
        )
        return self._finalize_memory_payload(
            observations=observations,
            lp1_payload=lp1_payload,
            state=state,
            profile_payload=profile_payload,
            cached_photo_ids=cached_photo_ids,
            dedupe_report=dedupe_report,
            raw_manifest=raw_manifest,
        )

    def _finalize_memory_payload(
        self,
        *,
        observations: Sequence[Dict[str, Any]],
        lp1_payload: Dict[str, Any],
        state: MemoryState,
        profile_payload: Dict[str, Any],
        cached_photo_ids: Sequence[str],
        dedupe_report: Dict[str, Any],
        raw_manifest: Dict[str, Any],
    ) -> Dict[str, Any]:
        lp2_relationships = [relationship.to_dict() for relationship in state.relationships]
        save_json(lp2_relationships, str(self.lp2_relationships_path))
        _write_jsonl(self.lp2_relationships_jsonl_path, lp2_relationships)
        save_json(profile_payload, str(self.lp3_profile_path))

        artifacts = self._artifact_urls(
            {
                "vp1_observations_url": self.vp1_path,
                "lp1_batch_requests_url": self.lp1_batch_requests_path,
                "lp1_batch_outputs_url": self.lp1_batch_outputs_path,
                "lp1_event_cards_url": self.lp1_event_cards_path,
                "lp1_events_url": self.lp1_events_path,
                "lp1_events_compact_url": self.lp1_events_compact_path,
                "lp1_event_continuation_log_url": self.lp1_continuation_log_path,
                "lp1_parse_failures_url": self.lp1_parse_failures_path,
                "lp1_salvaged_events_url": self.lp1_salvaged_events_path,
                "lp1_salvage_report_url": self.lp1_salvage_report_path,
                "lp2_relationships_jsonl_url": self.lp2_relationships_jsonl_path,
                "lp2_relationships_url": self.lp2_relationships_path,
                "lp3_profile_url": self.lp3_profile_path,
                "structured_profile_url": self.structured_profile_path,
                "relationship_dossiers_url": self.relationship_dossiers_path,
                "group_artifacts_url": self.group_artifacts_path,
                "profile_fact_decisions_url": self.profile_fact_decisions_path,
                "downstream_audit_report_url": self.downstream_audit_report_path,
                "llm_failures_url": self.llm_failures_path,
                "memory_snapshot_url": self.memory_snapshot_path,
                "raw_upstream_manifest_url": self.raw_manifest_path,
                "raw_upstream_index_url": self.raw_index_path,
            }
        )

        memory = {
            "pipeline_family": PIPELINE_FAMILY_V0325,
            "summary": {
                "pipeline_family": PIPELINE_FAMILY_V0325,
                "lp1_batch_count": len(lp1_payload["lp1_batches"]),
                "event_count": len(lp1_payload["lp1_events"]),
                "relationship_count": len(lp2_relationships),
                "group_count": len(state.groups),
                "profile_generation_mode": "field_agent_plus_markdown",
                "relationship_generation_mode": "dossier_agent",
                "primary_generation_mode": "rejudged",
                "lp1_prompt_version": LP1_PROMPT_VERSION_V0325,
                "lp1_contract_version": LP1_CONTRACT_VERSION_V0325,
                "cached_photo_count": len(list(cached_photo_ids or [])),
                "dedupe_retained_images": _safe_int(dedupe_report.get("retained_images")),
                "raw_attachment_count": raw_manifest["summary"]["attachment_count"],
                "no_drop_guarantee": raw_manifest["summary"]["no_drop_guarantee"],
            },
            "vp1_observations": observations,
            "lp1_batches": lp1_payload["lp1_batches"],
            "lp1_events": lp1_payload["lp1_events"],
            "lp1_event_continuation_log": lp1_payload["lp1_event_continuation_log"],
            "lp2_relationships": lp2_relationships,
            "lp3_profile": profile_payload,
            "artifacts": artifacts,
            "transparency": {
                "llm_provider": getattr(self.llm_processor, "provider", ""),
                "llm_model": getattr(self.llm_processor, "model", ""),
                "relationship_model": getattr(self.llm_processor, "relationship_model", getattr(self.llm_processor, "model", "")),
                "relationship_timeout_seconds": RELATIONSHIP_REQUEST_TIMEOUT_SECONDS,
                "relationship_max_retries": RELATIONSHIP_MAX_RETRIES,
                "profile_llm_provider": PROFILE_LLM_PROVIDER,
                "profile_llm_model": PROFILE_LLM_MODEL or OPENROUTER_LLM_MODEL,
                "lp1_prompt_version": LP1_PROMPT_VERSION_V0325,
                "lp1_contract_version": LP1_CONTRACT_VERSION_V0325,
            },
        }
        save_json(memory, str(self.memory_snapshot_path))
        return memory

    def _relationship_dossier_from_dict(self, payload: Dict[str, Any]) -> RelationshipDossier:
        return RelationshipDossier(
            person_id=str(payload.get("person_id") or ""),
            person_kind=str(payload.get("person_kind") or "uncertain"),
            memory_value=str(payload.get("memory_value") or "candidate"),
            photo_count=int(payload.get("photo_count") or 0),
            time_span_days=int(payload.get("time_span_days") or 0),
            recent_gap_days=int(payload.get("recent_gap_days") or 0),
            monthly_frequency=float(payload.get("monthly_frequency") or 0.0),
            scene_profile=dict(payload.get("scene_profile") or {}),
            interaction_signals=list(payload.get("interaction_signals", []) or []),
            shared_events=list(payload.get("shared_events", []) or []),
            trend_detail=dict(payload.get("trend_detail") or {}),
            co_appearing_persons=list(payload.get("co_appearing_persons", []) or []),
            anomalies=list(payload.get("anomalies", []) or []),
            evidence_refs=list(payload.get("evidence_refs", []) or []),
            block_reasons=list(payload.get("block_reasons", []) or []),
            retention_decision=str(payload.get("retention_decision") or "review"),
            retention_reason=str(payload.get("retention_reason") or ""),
            group_eligible=bool(payload.get("group_eligible", False)),
            group_block_reason=payload.get("group_block_reason"),
            group_weight=float(payload.get("group_weight") or 0.0),
            relationship_result=dict(payload.get("relationship_result") or {}),
            relationship_reflection=dict(payload.get("relationship_reflection") or {}),
        )

    def _relationship_record_from_dict(self, payload: Dict[str, Any]) -> RelationshipRecord:
        return RelationshipRecord(
            person_id=str(payload.get("person_id") or ""),
            relationship_type=str(payload.get("relationship_type") or ""),
            intimacy_score=float(payload.get("intimacy_score") or 0.0),
            status=str(payload.get("status") or ""),
            confidence=float(payload.get("confidence") or 0.0),
            reasoning=str(payload.get("reasoning") or payload.get("reason") or ""),
            shared_events=list(payload.get("shared_events", []) or []),
            evidence=dict(payload.get("evidence") or {}),
        )

    def _group_artifact_from_dict(self, payload: Dict[str, Any]) -> GroupArtifact:
        return GroupArtifact(
            group_id=str(payload.get("group_id") or ""),
            members=list(payload.get("members", []) or []),
            group_type_candidate=str(payload.get("group_type_candidate") or ""),
            confidence=float(payload.get("confidence") or 0.0),
            strong_evidence_refs=list(payload.get("strong_evidence_refs", []) or []),
            reason=str(payload.get("reason") or ""),
        )

    def _persist_precomputed_lp1_payload(
        self,
        *,
        observations: Sequence[Dict[str, Any]],
        lp1_events: Sequence[Dict[str, Any]],
        lp1_batches: Sequence[Dict[str, Any]] | None = None,
        lp1_event_continuation_log: Sequence[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        final_events = [dict(item) for item in list(lp1_events or []) if isinstance(item, dict)]
        continuation_log = [
            dict(item)
            for item in list(lp1_event_continuation_log or []) or []
            if isinstance(item, dict)
        ]
        batch_summaries = [
            dict(item)
            for item in list(lp1_batches or []) or []
            if isinstance(item, dict)
        ]
        if not batch_summaries:
            event_support_sets = [
                set(_unique_strings(list(event.get("supporting_photo_ids", []) or [])))
                for event in final_events
            ]
            batch_summaries = []
            for batch in self._build_batches(observations):
                output_window_photo_ids = list(batch.get("output_window_photo_ids", []) or [])
                output_window_photo_set = set(output_window_photo_ids)
                raw_event_count = 0
                for support_set in event_support_sets:
                    if support_set and support_set.intersection(output_window_photo_set):
                        raw_event_count += 1
                batch_summaries.append(
                    {
                        "batch_id": batch["batch_id"],
                        "batch_index": batch["batch_index"],
                        "input_photo_ids": list(batch["input_photo_ids"]),
                        "overlap_context_photo_ids": list(batch["overlap_context_photo_ids"]),
                        "output_window_photo_ids": output_window_photo_ids,
                        "carryover_event_refs": [],
                        "raw_event_count": raw_event_count,
                        "parse_status": "reused_precomputed",
                        "prompt_version": LP1_PROMPT_VERSION_V0325,
                        "contract_version": LP1_CONTRACT_VERSION_V0325,
                        "salvage_status": "none",
                        "salvaged_event_count": 0,
                        "reuse_mode": "precomputed_lp1",
                    }
                )

        for path in (
            self.lp1_batch_requests_path,
            self.lp1_batch_outputs_path,
            self.lp1_salvaged_events_path,
            self.lp1_salvage_report_path,
        ):
            _safe_unlink(path)
        _write_jsonl(self.lp1_event_cards_path, [])
        _write_jsonl(self.lp1_events_path, final_events)
        _write_jsonl(self.lp1_continuation_log_path, continuation_log)
        save_json([], str(self.lp1_parse_failures_path))
        save_json(final_events, str(self.lp1_events_compact_path))
        return {
            "lp1_batches": batch_summaries,
            "lp1_events": final_events,
            "lp1_event_continuation_log": continuation_log,
        }

    def _notify(
        self,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]],
        payload: Dict[str, Any],
    ) -> None:
        if progress_callback is None:
            return
        progress_callback("v0325", payload)

    def _build_memory_state(
        self,
        *,
        face_output: Dict[str, Any],
        observations: Sequence[Dict[str, Any]],
        lp1_events: Sequence[Dict[str, Any]],
        fallback_primary_person_id: str | None,
        raw_upstream: Dict[str, Any],
        raw_index: Dict[str, Dict[str, Any]],
    ) -> MemoryState:
        face_db: Dict[str, Dict[str, Any]] = {}
        for person in list(face_output.get("persons", []) or []):
            person_id = str(person.get("person_id") or "").strip()
            if not person_id:
                continue
            face_db[person_id] = {
                "photo_count": int(person.get("photo_count", 0) or 0),
                "first_seen": person.get("first_seen"),
                "last_seen": person.get("last_seen"),
                "avg_confidence": float(person.get("avg_score", 0.0) or 0.0),
                "avg_quality": float(person.get("avg_quality", 0.0) or 0.0),
                "name": person.get("label", ""),
            }
        alias_bindings: Dict[str, List[Dict[str, Any]]] = {"participants": [], "depicted": []}
        events: List[Event] = []
        for raw_event in lp1_events:
            if not isinstance(raw_event, dict):
                continue
            supporting_photo_ids = list(raw_event.get("supporting_photo_ids", []) or [])
            if not supporting_photo_ids:
                continue
            meta_info = dict(raw_event.get("meta_info", {}) or {})
            objective_fact = dict(raw_event.get("objective_fact", {}) or {})
            participants = []
            raw_participants = list(raw_event.get("participant_person_ids", []) or objective_fact.get("participants", []) or [])
            for index, participant in enumerate(raw_participants):
                value = str(participant or "").strip()
                if value == "主角":
                    alias_bindings["participants"].append(
                        {"event_id": str(raw_event.get("event_id") or ""), "index": index, "alias": "主角"}
                    )
                    value = str(fallback_primary_person_id or "").strip() or value
                if value:
                    participants.append(value)
            depicted_people = list(raw_event.get("depicted_person_ids", []) or [])
            for index, participant in enumerate(depicted_people):
                value = str(participant or "").strip()
                if value == "主角":
                    alias_bindings["depicted"].append(
                        {"event_id": str(raw_event.get("event_id") or ""), "index": index, "alias": "主角"}
                    )
                    depicted_people[index] = str(fallback_primary_person_id or "").strip() or value
            trace = dict(meta_info.get("trace", {}) or {})
            trace["supporting_photo_ids"] = supporting_photo_ids
            trace["anchor_photo_id"] = raw_event.get("anchor_photo_id")
            trace["batch_id"] = raw_event.get("batch_id")
            trace["source_temp_event_id"] = raw_event.get("source_temp_event_id")
            meta_info["trace"] = trace
            event = Event(
                event_id=str(raw_event.get("event_id") or ""),
                date=str(raw_event.get("started_at") or "")[:10],
                time_range=_build_time_range(str(raw_event.get("started_at") or ""), str(raw_event.get("ended_at") or "")),
                duration="",
                title=str(raw_event.get("title") or meta_info.get("title") or ""),
                type=str(raw_event.get("type") or "其他"),
                participants=participants,
                location=str(
                    meta_info.get("location_context")
                    or (list(raw_event.get("place_refs", []) or [])[:1] or [""])[0]
                    or raw_event.get("location")
                    or ""
                ),
                description=str(objective_fact.get("scene_description") or raw_event.get("description") or ""),
                photo_count=int(meta_info.get("photo_count") or len(supporting_photo_ids) or 0),
                confidence=float(raw_event.get("confidence", 0.0) or 0.0),
                reason=str(raw_event.get("reason") or ""),
                narrative_synthesis=str(raw_event.get("narrative_synthesis") or ""),
                meta_info=meta_info,
                objective_fact=objective_fact,
                social_dynamics=list(raw_event.get("social_dynamics", []) or []),
                tags=list(raw_event.get("tags", []) or []),
                persona_evidence=dict(raw_event.get("persona_evidence", {}) or {}),
            )
            events.append(event)
        return MemoryState(
            photos=[],
            face_db=face_db,
            vlm_results=[self._normalize_vlm_result(item) for item in observations],
            raw_upstream=raw_upstream,
            raw_index=raw_index,
            primary_alias_bindings=alias_bindings,
            events=events,
        )

    def _normalize_vlm_result(self, item: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {
            "photo_id": item.get("photo_id"),
            "timestamp": item.get("timestamp"),
            "filename": item.get("filename"),
            "location": dict(item.get("location") or {}),
            "face_person_ids": list(item.get("face_person_ids", []) or []),
            "media_kind": item.get("media_kind"),
            "is_reference_like": bool(item.get("is_reference_like")),
            "sequence_index": item.get("sequence_index"),
            "vlm_analysis": dict(item.get("vlm_analysis", {}) or {}),
        }
        analysis = normalized["vlm_analysis"]
        analysis.setdefault("details", list(analysis.get("details", []) or []))
        analysis.setdefault("ocr_hits", list(analysis.get("ocr_hits", []) or []))
        analysis.setdefault("brands", list(analysis.get("brands", []) or []))
        analysis.setdefault("place_candidates", list(analysis.get("place_candidates", []) or []))
        scene = dict(analysis.get("scene") or {})
        if not scene.get("location_detected"):
            scene["location_detected"] = str((normalized["location"] or {}).get("name") or "").strip()
        analysis["scene"] = scene
        people = list(analysis.get("people", []) or [])
        if not people:
            people = [{"person_id": person_id} for person_id in normalized["face_person_ids"]]
        analysis["people"] = people
        normalized["vlm_analysis"] = analysis
        return normalized

    def _build_raw_upstream_payload(
        self,
        *,
        face_output: Dict[str, Any],
        observations: Sequence[Dict[str, Any]],
        lp1_payload: Dict[str, Any],
        dedupe_report: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Dict[str, Any]]]:
        attachment_specs = [
            ("raw_face_output", self.task_dir / "cache" / "face_recognition_output.json", True, "face"),
            ("raw_face_state", self.task_dir / "cache" / "face_recognition_state.json", False, "face"),
            ("raw_vlm_results", self.task_dir / "cache" / "vlm_cache.json", True, "vlm"),
            ("raw_vlm_failures", self.task_dir / "cache" / "vlm_failures.jsonl", False, "vlm"),
            ("raw_dedupe_report", self.task_dir / "cache" / "dedupe_report.json", False, "face"),
            ("raw_vp1_observations", self.vp1_path, True, "vlm"),
            ("raw_lp1_events", self.lp1_events_compact_path, True, "lp1"),
            ("raw_lp1_batch_outputs", self.lp1_batch_outputs_path, False, "lp1"),
            ("raw_lp1_parse_failures", self.lp1_parse_failures_path, False, "lp1"),
        ]
        raw_upstream: Dict[str, Any] = {
            "raw_face_output": copy.deepcopy(face_output),
            "raw_vp1_observations": [dict(item) for item in observations],
            "raw_lp1_events": [dict(item) for item in list(lp1_payload.get("lp1_events", []) or [])],
            "raw_task_metadata": {
                "task_id": self.task_id,
                "task_dir": str(self.task_dir),
                "pipeline_family": PIPELINE_FAMILY_V0325,
                "generated_at": datetime.now().isoformat(),
                "dedupe_report": copy.deepcopy(dedupe_report),
            },
        }
        if _path_exists(self.task_dir / "cache" / "face_recognition_state.json"):
            raw_upstream["raw_face_state"] = load_json(str(self.task_dir / "cache" / "face_recognition_state.json"))
        if _path_exists(self.task_dir / "cache" / "vlm_cache.json"):
            raw_upstream["raw_vlm_results"] = load_json(str(self.task_dir / "cache" / "vlm_cache.json"))
        if _path_exists(self.task_dir / "cache" / "vlm_failures.jsonl"):
            raw_upstream["raw_failures"] = {
                "vlm_failures": _load_jsonl(self.task_dir / "cache" / "vlm_failures.jsonl"),
                "lp1_parse_failures": copy.deepcopy(load_json(str(self.lp1_parse_failures_path))),
            }
        raw_manifest_attachments: List[Dict[str, Any]] = []
        for key, path, participates_in_default_inference, category in attachment_specs:
            exists = path.exists()
            attachment_payload = {
                "attachment_key": key,
                "relative_path": path.relative_to(self.task_dir).as_posix() if exists else path.relative_to(self.task_dir).as_posix(),
                "exists": exists,
                "category": category,
                "participates_in_default_inference": participates_in_default_inference,
                "asset_url": self.public_url_builder(path) if exists else None,
                "size_bytes": int(path.stat().st_size) if exists else 0,
            }
            raw_manifest_attachments.append(attachment_payload)

        raw_index = self._build_raw_index(raw_upstream=raw_upstream)
        raw_manifest = {
            "pipeline_family": PIPELINE_FAMILY_V0325,
            "generated_at": datetime.now().isoformat(),
            "attachments": raw_manifest_attachments,
            "summary": {
                "attachment_count": len(raw_manifest_attachments),
                "existing_attachment_count": sum(1 for item in raw_manifest_attachments if item["exists"]),
                "default_inference_attachment_count": sum(
                    1 for item in raw_manifest_attachments if item["exists"] and item["participates_in_default_inference"]
                ),
                "indexed_photo_count": len(raw_index.get("photo", {})),
                "indexed_event_count": len(raw_index.get("event", {})),
                "indexed_person_count": len(raw_index.get("person", {})),
                "no_drop_guarantee": True,
            },
        }
        return raw_upstream, raw_manifest, raw_index

    def _build_raw_index(self, *, raw_upstream: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        photo_index: Dict[str, Dict[str, Any]] = {}
        event_index: Dict[str, Dict[str, Any]] = {}
        person_index: Dict[str, Dict[str, Any]] = {}
        for image in list(dict(raw_upstream.get("raw_face_output") or {}).get("images", []) or []):
            photo_id = str(image.get("image_id") or image.get("photo_id") or "").strip()
            if not photo_id:
                continue
            bucket = photo_index.setdefault(photo_id, {})
            bucket["face_image"] = image
            for face in list(image.get("faces", []) or []):
                person_id = str(face.get("person_id") or "").strip()
                if not person_id:
                    continue
                person_bucket = person_index.setdefault(person_id, {})
                person_bucket.setdefault("face_instances", []).append(
                    {
                        "photo_id": photo_id,
                        "face_id": face.get("face_id"),
                        "bbox": face.get("bbox") or face.get("bbox_xywh"),
                        "match_decision": face.get("match_decision"),
                        "quality_score": face.get("quality_score"),
                    }
                )
        for person in list(dict(raw_upstream.get("raw_face_output") or {}).get("persons", []) or []):
            person_id = str(person.get("person_id") or "").strip()
            if not person_id:
                continue
            bucket = person_index.setdefault(person_id, {})
            bucket["face_person"] = person
        for item in list(raw_upstream.get("raw_vp1_observations", []) or []):
            photo_id = str(item.get("photo_id") or "").strip()
            if not photo_id:
                continue
            bucket = photo_index.setdefault(photo_id, {})
            bucket["vp1_observation"] = item
        raw_vlm_results = raw_upstream.get("raw_vlm_results")
        if isinstance(raw_vlm_results, dict):
            vlm_photos = list(raw_vlm_results.get("photos") or raw_vlm_results.get("vlm_results") or [])
        else:
            vlm_photos = list(raw_vlm_results or [])
        for item in vlm_photos:
            if not isinstance(item, dict):
                continue
            photo_id = str(item.get("photo_id") or "").strip()
            if not photo_id:
                continue
            bucket = photo_index.setdefault(photo_id, {})
            bucket["vlm_cache_item"] = item
        for event in list(raw_upstream.get("raw_lp1_events", []) or []):
            if not isinstance(event, dict):
                continue
            event_id = str(event.get("event_id") or "").strip()
            if not event_id:
                continue
            bucket = event_index.setdefault(event_id, {})
            bucket["lp1_event"] = event
            for person_id in _unique_strings(
                [
                    *list(event.get("participant_person_ids", []) or []),
                    *list(event.get("depicted_person_ids", []) or []),
                ]
            ):
                person_bucket = person_index.setdefault(person_id, {})
                person_bucket.setdefault("event_ids", []).append(event_id)
        return {"photo": photo_index, "event": event_index, "person": person_index}

    def _run_lp3_profile(
        self,
        *,
        state: MemoryState,
        llm_processor: V0325LLMRuntimeAdapter,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        self._notify(
            progress_callback,
            {
                "message": "v0325 LP3 field-agent 生成中",
                "pipeline_family": PIPELINE_FAMILY_V0325,
                "substage": "lp3_profile",
                "percent": 75,
                "relationship_count": len(state.relationships),
            },
        )
        state.profile_context = agent_build_profile_context(state)
        profile_state = agent_generate_structured_profile(state, llm_processor=llm_processor)
        structured = profile_state["structured"]
        consistency = profile_state["consistency"]
        audit_payload = _run_downstream_audit_and_backflow(
            primary_decision=state.primary_decision or {},
            relationships=state.relationships,
            structured_profile=structured,
            groups=state.groups,
            consistency=consistency,
        )
        final_structured = audit_payload["final_structured_profile"]
        compact_structured = compact_structured_profile(final_structured)
        sidecar_paths = self._persist_lp3_sidecar_artifacts(
            state=state,
            field_decisions=profile_state["field_decisions"],
            audit_payload=audit_payload,
            compact_structured=compact_structured,
        )
        final_consistency = _build_consistency_report(state.events, state.relationships, compact_structured)
        report_markdown = _build_report_markdown(
            primary_decision=state.primary_decision or {},
            structured_profile=compact_structured,
            consistency=final_consistency,
            relationships=state.relationships,
            groups=state.groups,
            audit_report=audit_payload,
        )
        profile_payload = {
            "structured": compact_structured,
            "summary": _build_profile_summary(compact_structured, state.relationships),
            "consistency": final_consistency,
            "report_markdown": report_markdown,
            "internal_artifacts": {
                "screening_count": len(state.screening),
                "primary_decision": state.primary_decision,
                "primary_reflection": state.primary_reflection or {},
                "relationship_dossier_count": len(state.relationship_dossiers),
                "group_artifact_count": len(state.groups),
                "profile_fact_decision_count": len(profile_state["field_decisions"]),
                "audit_flag_count": len(list(audit_payload.get("audit_flags", []) or [])),
                **sidecar_paths,
                "raw_manifest_path": self._relative_artifact_path(self.raw_manifest_path),
                "raw_index_path": self._relative_artifact_path(self.raw_index_path),
            },
        }
        self._notify(
            progress_callback,
            {
                "message": "v0325 LP3 画像生成完成",
                "pipeline_family": PIPELINE_FAMILY_V0325,
                "substage": "lp3_profile",
                "percent": 100,
                "relationship_count": len(state.relationships),
                "group_count": len(state.groups),
            },
        )
        return profile_payload


def _screen_people(state: MemoryState) -> Dict[str, PersonScreening]:
    screenings: Dict[str, PersonScreening] = {}
    appearance_stats = _collect_person_vlm_stats(state.vlm_results)
    for person_id, person_info in state.face_db.items():
        photo_count = int((person_info or {}).get("photo_count", 0) or 0)
        stats = appearance_stats.get(person_id, {})
        refs = list(stats.get("refs", []) or [])
        blocked: List[str] = []
        person_kind = "real_person"
        memory_value = "candidate"
        if stats.get("mediated_ratio", 0.0) >= 0.8:
            person_kind = "mediated_person"
            memory_value = "block"
            blocked.append("mostly_screen_or_poster")
        elif stats.get("service_ratio", 0.0) >= 0.7:
            person_kind = "service_person"
            memory_value = "block"
            blocked.append("mostly_service_context")
        elif photo_count <= 1 and stats.get("group_only_ratio", 0.0) >= 0.8:
            person_kind = "incidental_person"
            memory_value = "low_value"
        elif photo_count >= 8:
            memory_value = "core"
        screenings[person_id] = PersonScreening(
            person_id=person_id,
            person_kind=person_kind,
            memory_value=memory_value,
            screening_refs=refs,
            block_reasons=blocked,
        )
    return screenings


def _collect_person_vlm_stats(vlm_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for item in vlm_results:
        photo_id = item.get("photo_id")
        analysis = item.get("vlm_analysis", {}) or {}
        summary = str(analysis.get("summary", "") or "").lower()
        event_data = analysis.get("event", {}) or {}
        social_context = str(event_data.get("social_context", "") or "").lower() if isinstance(event_data, dict) else ""
        scene = analysis.get("scene", {}) or {}
        location = str(scene.get("location_detected", "") or "").lower() if isinstance(scene, dict) else ""
        for person in list(analysis.get("people", []) or []):
            if not isinstance(person, dict):
                continue
            person_id = str(person.get("person_id") or "").strip()
            if not person_id:
                continue
            bucket = stats.setdefault(person_id, {"total": 0, "mediated": 0, "service": 0, "group_only": 0, "refs": []})
            bucket["total"] += 1
            if any(keyword in summary for keyword in ("poster", "screenshot", "screen", "tv", "电视", "海报")):
                bucket["mediated"] += 1
            if any(keyword in social_context for keyword in ("staff", "cashier", "waiter", "服务员", "店员")):
                bucket["service"] += 1
            if any(keyword in location for keyword in ("stadium", "concert", "广场", "street")) and len(list(analysis.get("people", []) or [])) >= 4:
                bucket["group_only"] += 1
            bucket["refs"].append({"photo_id": photo_id, "signal": summary[:120], "why": "person_screening"})
    for bucket in stats.values():
        total = max(bucket["total"], 1)
        bucket["mediated_ratio"] = round(bucket["mediated"] / total, 2)
        bucket["service_ratio"] = round(bucket["service"] / total, 2)
        bucket["group_only_ratio"] = round(bucket["group_only"] / total, 2)
    return stats


def _extract_protagonist_mentions(text: str) -> List[str]:
    hits: List[str] = []
    for pattern in PROTAGONIST_LABEL_PATTERNS:
        for match in pattern.finditer(str(text or "")):
            person_id = match.group(1)
            if person_id:
                hits.append(person_id)
    return _unique_strings(hits)


def _candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    score = (
        candidate["selfie_count"] * 4.0
        + candidate["identity_anchor_count"] * 3.0
        + candidate["protagonist_label_count"] * 2.2
        + candidate["event_count"] * 1.1
        + candidate["photo_count"] * 0.15
        - candidate["photographed_subject_ratio"] * 3.5
    )
    return (
        round(score, 3),
        float(candidate["selfie_count"]),
        float(candidate["identity_anchor_count"]),
        float(candidate["protagonist_label_count"]),
        float(candidate["photo_count"]),
    )


def _collect_candidate_signals(state: MemoryState) -> List[Dict[str, Any]]:
    signal_map: Dict[str, Dict[str, Any]] = {}
    for person_id, person_info in (state.face_db or {}).items():
        screening = (state.screening or {}).get(person_id)
        if screening and screening.memory_value == "block":
            continue
        signal_map[person_id] = {
            "person_id": person_id,
            "photo_count": int((person_info or {}).get("photo_count", 0) or 0),
            "event_count": 0,
            "protagonist_label_count": 0,
            "selfie_count": 0,
            "identity_anchor_count": 0,
            "first_person_view_count": 0,
            "non_selfie_photo_count": 0,
            "photographed_subject_hits": 0,
            "photographed_subject_ratio": 0.0,
            "supporting_photo_ids": [],
        }
    for event in state.events or []:
        for person_id in list(getattr(event, "participants", []) or []):
            if person_id in signal_map:
                signal_map[person_id]["event_count"] += 1
    for item in state.vlm_results or []:
        analysis = item.get("vlm_analysis", {}) or {}
        summary = str(analysis.get("summary", "") or "")
        event_data = analysis.get("event", {}) or {}
        scene_data = analysis.get("scene", {}) or {}
        activity = event_data.get("activity", "") if isinstance(event_data, dict) else str(event_data or "")
        location = scene_data.get("location_detected", "") if isinstance(scene_data, dict) else str(scene_data or "")
        haystack = " ".join([summary, str(activity), str(location)]).lower()
        photo_id = str(item.get("photo_id") or "")
        people_ids = [
            str(person.get("person_id"))
            for person in (analysis.get("people", []) or [])
            if isinstance(person, dict) and person.get("person_id")
        ]
        protagonist_hits = _extract_protagonist_mentions(" ".join(filter(None, [summary, str(activity)])))
        for hit in protagonist_hits:
            if hit in signal_map:
                signal_map[hit]["protagonist_label_count"] += 1
        is_selfie = any(keyword in haystack for keyword in SELFIE_KEYWORDS)
        is_identity = any(keyword in haystack for keyword in IDENTITY_KEYWORDS)
        is_user_view = (not people_ids) or any(keyword in haystack for keyword in USER_VIEW_KEYWORDS)
        is_photographed_subject = any(keyword in haystack for keyword in PHOTOGRAPHED_SUBJECT_KEYWORDS)
        if is_user_view:
            for candidate in signal_map.values():
                candidate["first_person_view_count"] += 1
        for person_id in people_ids:
            candidate = signal_map.get(person_id)
            if not candidate:
                continue
            if photo_id:
                candidate["supporting_photo_ids"].append(photo_id)
            if is_selfie:
                candidate["selfie_count"] += 1
            else:
                candidate["non_selfie_photo_count"] += 1
            if is_identity:
                candidate["identity_anchor_count"] += 1
            if is_photographed_subject and len(people_ids) == 1:
                candidate["photographed_subject_hits"] += 1
    for candidate in signal_map.values():
        non_selfie = max(candidate["non_selfie_photo_count"], 1)
        candidate["photographed_subject_ratio"] = round(candidate["photographed_subject_hits"] / non_selfie, 3)
        candidate["supporting_photo_ids"] = _unique_strings(candidate["supporting_photo_ids"])
    return list(signal_map.values())


def _run_llm_primary_judgement(
    *,
    state: MemoryState,
    ranked_candidates: List[Dict[str, Any]],
    llm_processor: Any | None,
) -> Dict[str, Any] | None:
    if not llm_processor or not hasattr(llm_processor, "_call_llm_via_official_api") or not ranked_candidates:
        return None
    top_candidates = ranked_candidates[:5]
    stats_rows = []
    for candidate in top_candidates:
        stats_rows.append(
            "| {pid} | {photo} | {event} | {selfie} | {identity} | {protagonist} | {ratio} |".format(
                pid=candidate["person_id"],
                photo=candidate["photo_count"],
                event=candidate["event_count"],
                selfie=candidate["selfie_count"],
                identity=candidate["identity_anchor_count"],
                protagonist=candidate["protagonist_label_count"],
                ratio=candidate["photographed_subject_ratio"],
            )
        )
    event_lines = []
    for event in list(state.events or [])[:15]:
        event_lines.append(
            "- {event_id} {title} @ {location} | participants={participants}".format(
                event_id=getattr(event, "event_id", ""),
                title=getattr(event, "title", ""),
                location=getattr(event, "location", ""),
                participants=",".join(getattr(event, "participants", []) or []),
            )
        )
    prompt = f"""你是相册主人识别专家。请只返回 JSON。

候选统计（Top 5）：
| Person | photo_count | event_count | selfie_count | identity_anchor_count | protagonist_label_count | photographed_subject_ratio |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(stats_rows)}

关键事件/照片线索：
{chr(10).join(event_lines[:30])}

输出：
{{
  "mode": "person_id",
  "primary_person_id": "{top_candidates[0]['person_id']}",
  "confidence": 0.0,
  "reasoning": "一句话"
}}
"""
    try:
        result = llm_processor._call_llm_via_official_api(prompt, response_mime_type="application/json")
    except Exception:
        return None
    return result if isinstance(result, dict) else None


def _build_primary_decision(
    *,
    ranked_candidates: List[Dict[str, Any]],
    llm_result: Dict[str, Any] | None,
    fallback_primary_person_id: str | None,
) -> Dict[str, Any]:
    primary_person_id = None
    mode = "person_id"
    confidence = 0.0
    reasoning = ""
    if isinstance(llm_result, dict):
        primary_person_id = str(llm_result.get("primary_person_id") or "").strip() or None
        mode = str(llm_result.get("mode") or "person_id").strip() or "person_id"
        confidence = float(llm_result.get("confidence", 0.0) or 0.0)
        reasoning = str(llm_result.get("reasoning") or "").strip()
    if not primary_person_id and ranked_candidates:
        primary_person_id = str(ranked_candidates[0]["person_id"] or "").strip() or None
        confidence = max(confidence, 0.75 if ranked_candidates[0]["selfie_count"] > 0 else 0.68)
        if not reasoning:
            reasoning = "根据自拍/证件锚点、事件频次与出镜稳定性排序，当前人物最符合主角。"
    if not primary_person_id and fallback_primary_person_id:
        primary_person_id = str(fallback_primary_person_id or "").strip() or None
        confidence = max(confidence, 0.6)
        if not reasoning:
            reasoning = "主角候选信号不足，回退到上游主角。"
    evidence = _build_evidence_payload(
        person_ids=[primary_person_id] if primary_person_id else [],
        photo_ids=ranked_candidates[0]["supporting_photo_ids"][:8] if ranked_candidates else [],
        supporting_refs=[
            {
                "source_type": "person",
                "source_id": primary_person_id,
                "signal": reasoning or "primary_person_selected",
                "why": "primary_person_selection",
            }
        ]
        if primary_person_id
        else [],
    )
    return {
        "mode": mode,
        "primary_person_id": primary_person_id,
        "confidence": round(min(max(confidence, 0.0), 1.0), 3),
        "evidence": evidence,
        "reasoning": reasoning,
    }


def _reflect_primary_decision(
    *,
    initial_decision: Dict[str, Any],
    ranked_candidates: List[Dict[str, Any]],
    fallback_primary_person_id: str | None,
) -> Dict[str, Any]:
    selected = str(initial_decision.get("primary_person_id") or "").strip()
    reflection = {
        "triggered": False,
        "selected_person_id": selected,
        "issues": [],
        "action": "keep",
    }
    if not selected and fallback_primary_person_id:
        reflection["triggered"] = True
        reflection["issues"].append("no_primary_candidate")
        reflection["action"] = "fallback"
        reflection["selected_person_id"] = fallback_primary_person_id
        return reflection
    if len(ranked_candidates) >= 2:
        top = ranked_candidates[0]
        second = ranked_candidates[1]
        if second["selfie_count"] > top["selfie_count"] and second["photo_count"] >= top["photo_count"]:
            reflection["triggered"] = True
            reflection["issues"].append("second_candidate_has_stronger_selfie_signal")
            reflection["action"] = "switch"
            reflection["selected_person_id"] = second["person_id"]
    return reflection


def _apply_primary_reflection(initial_decision: Dict[str, Any], reflection: Dict[str, Any]) -> Dict[str, Any]:
    updated = dict(initial_decision)
    selected = str(reflection.get("selected_person_id") or "").strip()
    if reflection.get("action") in {"switch", "fallback"} and selected:
        updated["primary_person_id"] = selected
    return updated


def _analyze_primary_person_with_reflection(
    *,
    state: MemoryState,
    fallback_primary_person_id: str | None,
    llm_processor: Any | None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    candidates = _collect_candidate_signals(state)
    ranked = sorted(candidates, key=_candidate_sort_key, reverse=True)
    llm_result = _run_llm_primary_judgement(
        state=state,
        ranked_candidates=ranked,
        llm_processor=llm_processor,
    )
    initial_decision = _build_primary_decision(
        ranked_candidates=ranked,
        llm_result=llm_result,
        fallback_primary_person_id=fallback_primary_person_id,
    )
    reflection = _reflect_primary_decision(
        initial_decision=initial_decision,
        ranked_candidates=ranked,
        fallback_primary_person_id=fallback_primary_person_id,
    )
    final_decision = _apply_primary_reflection(initial_decision, reflection)
    reflection["primary_signal_trace"] = {
        "candidate_signals": ranked,
        "llm_decision": llm_result or {},
        "selected_person_id": final_decision.get("primary_person_id"),
        "selected_mode": final_decision.get("mode"),
    }
    return final_decision, reflection


def _rebind_primary_aliases(state: MemoryState, previous_primary_person_id: str | None) -> None:
    final_primary_person_id = str((state.primary_decision or {}).get("primary_person_id") or "").strip()
    previous = str(previous_primary_person_id or "").strip()
    if not final_primary_person_id or not previous or final_primary_person_id == previous:
        return
    participant_map = {
        (binding.get("event_id"), int(binding.get("index", -1)))
        for binding in list(state.primary_alias_bindings.get("participants", []) or [])
    }
    depicted_map = {
        (binding.get("event_id"), int(binding.get("index", -1)))
        for binding in list(state.primary_alias_bindings.get("depicted", []) or [])
    }
    for event in state.events:
        event_id = getattr(event, "event_id", "")
        participants = list(getattr(event, "participants", []) or [])
        for index, value in enumerate(participants):
            if (event_id, index) in participant_map and str(value or "").strip() == previous:
                participants[index] = final_primary_person_id
        event.participants = participants
        depicted_people = list(getattr(event, "objective_fact", {}).get("depicted_person_ids", []) or [])
        for index, value in enumerate(depicted_people):
            if (event_id, index) in depicted_map and str(value or "").strip() == previous:
                depicted_people[index] = final_primary_person_id
        if depicted_people:
            event.objective_fact["depicted_person_ids"] = depicted_people


def _collect_shared_events_from_state(events: Sequence[Event], person_id: str) -> List[Dict[str, Any]]:
    collected: List[Dict[str, Any]] = []
    for event in events or []:
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


def _safe_collect_relationship_evidence(person_id: str, state: MemoryState, llm_processor: Any) -> Dict[str, Any]:
    default = {
        "photo_count": 0,
        "time_span_days": 0,
        "recent_gap_days": 0,
        "scenes": [],
        "private_scene_ratio": 0.0,
        "dominant_scene_ratio": 0.0,
        "interaction_behavior": [],
        "with_user_only": True,
        "contact_types": [],
        "rela_events": _collect_shared_events_from_state(state.events, person_id),
        "monthly_frequency": 0.0,
        "trend_detail": {},
        "co_appearing_persons": [],
        "anomalies": [],
    }
    if llm_processor and hasattr(llm_processor, "_collect_relationship_evidence"):
        try:
            evidence = llm_processor._collect_relationship_evidence(person_id, state.vlm_results, state.events)
        except Exception:
            evidence = None
        if isinstance(evidence, dict):
            merged = dict(default)
            merged.update(evidence)
            if not merged.get("rela_events"):
                merged["rela_events"] = _collect_shared_events_from_state(state.events, person_id)
            return merged
    return default


def _build_evidence_refs(person_id: str, evidence: Dict[str, Any], state: MemoryState) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for item in state.vlm_results or []:
        analysis = item.get("vlm_analysis", {}) or {}
        people = analysis.get("people", []) or []
        if any(isinstance(person, dict) and person.get("person_id") == person_id for person in people):
            refs.append(
                {
                    "source_type": "photo",
                    "source_id": item.get("photo_id"),
                    "photo_id": item.get("photo_id"),
                    "signal": analysis.get("summary", ""),
                    "why": f"co_photo_for_{person_id}",
                }
            )
    for event in evidence.get("rela_events", []) or []:
        refs.append(
            {
                "source_type": "event",
                "source_id": event.get("event_id"),
                "event_id": event.get("event_id"),
                "signal": event.get("title") or event.get("narrative_synthesis", ""),
                "why": f"shared_event_for_{person_id}",
            }
        )
    for co_person in evidence.get("co_appearing_persons", []) or []:
        refs.append(
            {
                "source_type": "person",
                "source_id": co_person.get("person_id"),
                "person_id": co_person.get("person_id"),
                "signal": f"co_ratio={co_person.get('co_ratio', 0)}",
                "why": "third_party_co_appearance",
            }
        )
    refs.extend(
        [
            {
                "source_type": "feature",
                "source_id": "photo_count",
                "feature_name": "photo_count",
                "signal": str(evidence.get("photo_count", 0)),
                "why": "relationship_frequency",
            },
            {
                "source_type": "feature",
                "source_id": "monthly_frequency",
                "feature_name": "monthly_frequency",
                "signal": str(evidence.get("monthly_frequency", 0)),
                "why": "relationship_frequency",
            },
        ]
    )
    return refs


def _build_relationship_dossiers(state: MemoryState, llm_processor: Any) -> List[RelationshipDossier]:
    primary_person_id = (state.primary_decision or {}).get("primary_person_id")
    dossiers: List[RelationshipDossier] = []
    for person_id in state.face_db.keys():
        if person_id == primary_person_id:
            continue
        screening = (state.screening or {}).get(person_id)
        if screening and screening.memory_value == "block":
            continue
        evidence = _safe_collect_relationship_evidence(person_id, state, llm_processor)
        evidence_refs = _build_evidence_refs(person_id, evidence, state)
        dossiers.append(
            RelationshipDossier(
                person_id=person_id,
                person_kind=screening.person_kind if screening else "uncertain",
                memory_value=screening.memory_value if screening else "candidate",
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
                interaction_signals=list(evidence.get("interaction_behavior", []) or []) + list(evidence.get("contact_types", []) or []),
                shared_events=list(evidence.get("rela_events", []) or []),
                trend_detail=dict(evidence.get("trend_detail", {}) or {}),
                co_appearing_persons=list(evidence.get("co_appearing_persons", []) or []),
                anomalies=list(evidence.get("anomalies", []) or []),
                evidence_refs=evidence_refs,
                block_reasons=list(screening.block_reasons if screening else []),
            )
        )
    return dossiers


def _run_relationship_llm(dossier: RelationshipDossier, llm_processor: Any) -> Dict[str, Any] | None:
    if not llm_processor or not hasattr(llm_processor, "_call_llm_via_official_api"):
        return None
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

输出：
{{
  "relationship_type": "one of allowed types",
  "stability": "long_term | short_term",
  "status": "new | growing | stable | fading | gone",
  "confidence": 0-100,
  "strength_summary": "一句话",
  "reasoning": "一句话"
}}
"""
    try:
        result = llm_processor._call_llm_via_official_api(prompt, response_mime_type="application/json")
    except Exception:
        return None
    return result if isinstance(result, dict) else None


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


def _has_strong_relationship_signal(dossier: RelationshipDossier, relationship_type: str) -> bool:
    interaction_signals = {str(signal or "").lower() for signal in dossier.interaction_signals}
    shared_event_count = len(dossier.shared_events)
    scene_count = len(dossier.scene_profile.get("scenes", []))
    private_scene_ratio = float(dossier.scene_profile.get("private_scene_ratio", 0.0) or 0.0)
    with_user_only = bool(dossier.scene_profile.get("with_user_only", False))
    return (
        any(signal in STRONG_CONTACT_SIGNALS for signal in interaction_signals)
        or (relationship_type in {"romantic", "family"} and private_scene_ratio >= 0.25)
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
    if dossier.photo_count <= 1 and any(keyword in interactions for keyword in EMERGING_INTIMATE_KEYWORDS):
        return "acquaintance"
    return "acquaintance"


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


def _normalize_relationship_output(
    dossier: RelationshipDossier,
    relationship: RelationshipRecord,
    reflection: Dict[str, Any],
) -> RelationshipRecord:
    ids = _extract_ids_from_refs(dossier.evidence_refs)
    feature_names = ids["feature_names"] + [f"relationship_type:{relationship.relationship_type}"]
    supporting_refs = list(dossier.evidence_refs)
    contradicting_refs = [
        {"source_type": "feature", "source_id": issue, "feature_name": issue, "signal": issue, "why": "relationship_reflection"}
        for issue in reflection.get("issues", [])
    ]
    evidence = _build_evidence_payload(
        photo_ids=ids["photo_ids"],
        event_ids=[event.get("event_id") for event in dossier.shared_events] or ids["event_ids"],
        person_ids=[relationship.person_id] + ids["person_ids"],
        feature_names=feature_names,
        supporting_refs=supporting_refs,
        contradicting_refs=contradicting_refs,
    )
    evidence["summary"] = f"relationship:{relationship.person_id}:{relationship.relationship_type}"
    return replace(
        relationship,
        evidence=evidence,
        reasoning=_build_relationship_reasoning(dossier, relationship, reflection),
    )


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
    strong_signal = _has_strong_relationship_signal(dossier, relationship.relationship_type)
    if dossier.photo_count == 0:
        return "suppress", "no_supporting_photos"
    if dossier.photo_count <= 1 and not strong_signal:
        return "suppress", "single_photo_without_strong_signal"
    if relationship.relationship_type in LOW_SIGNAL_RELATIONSHIP_TYPES and relationship.confidence < 0.62 and not strong_signal:
        return "suppress", "low_confidence_low_signal_relationship"
    if relationship.relationship_type == "acquaintance" and dossier.photo_count <= 3 and len(dossier.shared_events) <= 1 and not strong_signal:
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


def _looks_mediated_from_dossier(dossier: RelationshipDossier) -> bool:
    dominant_ratio = float(dossier.scene_profile.get("dominant_scene_ratio", 0.0) or 0.0)
    searchable = list(dossier.scene_profile.get("scenes", [])) + list(dossier.interaction_signals)
    haystack = " ".join(str(item or "") for item in searchable).lower()
    has_mediated_keyword = any(keyword.lower() in haystack for keyword in MEDIATED_SIGNAL_KEYWORDS)
    return has_mediated_keyword and dominant_ratio >= 0.75


def _infer_relationships_from_dossiers(
    *,
    state: MemoryState,
    llm_processor: Any,
    dossiers: List[RelationshipDossier],
) -> Tuple[List[RelationshipRecord], List[RelationshipDossier]]:
    relationships: List[RelationshipRecord] = []
    updated_dossiers: List[RelationshipDossier] = []
    for dossier in dossiers:
        response = _run_relationship_llm(dossier, llm_processor)
        relationship_type = _normalize_relationship_type(response.get("relationship_type")) if response else ""
        if not relationship_type:
            relationship_type = _heuristic_relationship_type(dossier)
        status = _normalize_status(response, dossier)
        confidence = _safe_float(response.get("confidence") if response else 0.0)
        if confidence > 1:
            confidence /= 100.0
        if confidence <= 0:
            confidence = round(min(max(0.45 + min(dossier.photo_count / 20.0, 0.2) + min(dossier.monthly_frequency / 10.0, 0.15), 0.25), 0.92), 3)
        reasoning = str(response.get("reasoning") or response.get("strength_summary") or "").strip() if response else ""
        if not reasoning:
            reasoning = (
                f"{dossier.person_id} 共现 {dossier.photo_count} 张，场景 {len(dossier.scene_profile.get('scenes', []))} 类，"
                f"月均 {dossier.monthly_frequency:.1f} 次，判为 {relationship_type}。"
            )
        relationship = RelationshipRecord(
            person_id=dossier.person_id,
            relationship_type=relationship_type,
            intimacy_score=_estimate_intimacy(dossier, relationship_type),
            status=status,
            confidence=round(min(max(confidence, 0.0), 1.0), 3),
            reasoning=reasoning,
            shared_events=[
                {
                    "event_id": event.get("event_id", ""),
                    "date": event.get("date", ""),
                    "narrative": event.get("narrative_synthesis") or event.get("title", ""),
                }
                for event in dossier.shared_events
            ],
            evidence={},
        )
        reflection = {"triggered": False, "issues": [], "action": "keep", "source": "llm" if response else "heuristic"}
        strong_signal = _has_strong_relationship_signal(dossier, relationship.relationship_type)
        repeated_pattern = (
            dossier.photo_count >= 3
            or len(dossier.shared_events) >= 2
            or len(dossier.scene_profile.get("scenes", [])) >= 2
        )
        if relationship.relationship_type in {"romantic", "family"} and not strong_signal:
            reflection["issues"].append("high_risk_relationship_without_strong_signal")
            relationship = replace(
                relationship,
                relationship_type=RELATIONSHIP_TYPE_SPECS[relationship.relationship_type].downgrade_target or "close_friend",
                confidence=max(0.2, round(relationship.confidence - 0.1, 3)),
            )
        elif relationship.relationship_type == "close_friend" and not strong_signal and not repeated_pattern:
            reflection["issues"].append("close_friend_without_strong_signal")
            relationship = replace(relationship, relationship_type="friend", confidence=max(0.2, round(relationship.confidence - 0.08, 3)))
        elif relationship.relationship_type == "bestie" and not strong_signal:
            reflection["issues"].append("bestie_without_strong_signal")
            relationship = replace(relationship, relationship_type="close_friend", confidence=max(0.2, round(relationship.confidence - 0.08, 3)))
        elif relationship.relationship_type == "friend" and dossier.photo_count <= 1 and not strong_signal:
            reflection["issues"].append("friend_without_repeated_signal")
            relationship = replace(relationship, relationship_type="acquaintance", confidence=max(0.2, round(relationship.confidence - 0.08, 3)))
        if reflection["issues"]:
            reflection["triggered"] = True
            reflection["action"] = "downgrade"
        if relationship.status == "new":
            should_promote = (
                dossier.photo_count >= 3
                and dossier.time_span_days >= 14
                and len(dossier.shared_events) >= 2
                and (
                    strong_signal
                    or len(dossier.scene_profile.get("scenes", [])) >= 2
                    or str(dossier.trend_detail.get("direction", "") or "").lower() == "up"
                    or dossier.monthly_frequency >= 3
                )
            )
            if should_promote:
                relationship = replace(relationship, status="growing")
        relationship = _normalize_relationship_output(dossier, relationship, reflection)
        dossier.relationship_reflection = reflection
        dossier.retention_decision, dossier.retention_reason = _determine_retention(dossier, relationship)
        dossier.group_eligible, dossier.group_block_reason, dossier.group_weight = _determine_group_eligibility(dossier, relationship)
        dossier.relationship_result = {
            "relationship_type": relationship.relationship_type,
            "status": relationship.status,
            "confidence": relationship.confidence,
            "reasoning": relationship.reasoning,
        }
        if dossier.retention_decision == "keep":
            relationships.append(relationship)
        updated_dossiers.append(dossier)
    return relationships, updated_dossiers


def _select_group_candidates(
    relationships: List[RelationshipRecord],
    dossiers: List[RelationshipDossier],
    confidence_threshold: float = 0.75,
) -> List[RelationshipRecord]:
    dossier_by_person_id = {d.person_id: d for d in dossiers}
    selected: List[RelationshipRecord] = []
    for relationship in relationships:
        dossier = dossier_by_person_id.get(relationship.person_id)
        if not dossier or relationship.confidence < confidence_threshold or dossier.retention_decision != "keep" or not dossier.group_eligible:
            continue
        selected.append(relationship)
    return selected


def _detect_groups(state: MemoryState) -> List[GroupArtifact]:
    dossiers = getattr(state, "relationship_dossiers", None) or []
    candidate_relationships = _select_group_candidates(state.relationships or [], dossiers)
    if len(candidate_relationships) < 2:
        return []
    relationship_ids = {relationship.person_id for relationship in candidate_relationships}
    event_memberships: Dict[str, List[str]] = defaultdict(list)
    event_lookup: Dict[str, Any] = {}
    for relationship in candidate_relationships:
        for shared_event in relationship.shared_events:
            event_id = shared_event.get("event_id")
            if event_id:
                event_memberships[event_id].append(relationship.person_id)
    for event in state.events or []:
        event_lookup[event.event_id] = event
    groups: List[GroupArtifact] = []
    counter = 1
    for event_id, members in event_memberships.items():
        unique_members = sorted(set(members))
        if len(unique_members) < 2:
            continue
        event = event_lookup.get(event_id)
        if not event:
            continue
        group_type = _infer_group_type(event)
        groups.append(
            GroupArtifact(
                group_id=f"GRP_{counter:03d}",
                members=unique_members,
                group_type_candidate=group_type,
                confidence=_score_group_confidence(unique_members, event, relationship_ids),
                strong_evidence_refs=[
                    {
                        "event_id": event.event_id,
                        "signal": event.title,
                        "why": f"group_members={','.join(unique_members)}",
                    }
                ],
                reason=f"stable_shared_event:{event.event_id}",
            )
        )
        counter += 1
    return groups


def _infer_group_type(event: Any) -> str:
    haystack = " ".join(
        str(value or "")
        for value in (
            getattr(event, "title", ""),
            getattr(event, "location", ""),
            getattr(event, "description", ""),
            getattr(event, "narrative_synthesis", ""),
        )
    ).lower()
    if any(keyword in haystack for keyword in ("sorority", "greek", "formal", "姐妹会")):
        return "sorority"
    if any(keyword in haystack for keyword in ("lab", "实验室")):
        return "lab"
    if any(keyword in haystack for keyword in ("team", "比赛", "球")):
        return "team"
    if any(keyword in haystack for keyword in ("club", "社团")):
        return "club"
    return "friend_group"


def _score_group_confidence(members: List[str], event: Any, relationship_ids: set[str]) -> float:
    participant_overlap = len([person_id for person_id in getattr(event, "participants", []) if person_id in relationship_ids])
    base = 0.45 + min(0.15 * len(members), 0.3)
    if participant_overlap >= len(members):
        base += 0.1
    if getattr(event, "photo_count", 0) >= 3:
        base += 0.08
    if getattr(event, "confidence", 0.0) >= 0.8:
        base += 0.05
    return round(min(base, 0.95), 3)


def _determine_subject_role(
    primary_person_id: str | None,
    people: List[str],
    face_person_ids: List[str],
    summary: str,
    activity: str,
) -> str:
    primary = str(primary_person_id or "").strip()
    people_set = {person for person in people if person}
    face_person_set = {person for person in face_person_ids if person}
    if primary and primary != "主角" and (primary in people_set or primary in face_person_set):
        return "protagonist_present"
    signal = " ".join(filter(None, [summary, activity]))
    protagonist_view_markers = ("主角", "自拍", "selfie", "first-person", "first person", "POV", "pov")
    if (not people_set and not face_person_set) and any(marker.lower() in signal.lower() for marker in protagonist_view_markers):
        return "protagonist_view"
    if primary == "主角" and "主角" in signal:
        return "protagonist_view"
    return "other_people_only"


def _build_context_feature_refs(
    *,
    events: List[Event],
    relationships: List[RelationshipRecord],
    groups: List[GroupArtifact],
) -> List[Dict[str, Any]]:
    close_types = {"romantic", "family", "bestie", "close_friend"}
    close_circle_size = len(
        {
            relationship.person_id
            for relationship in relationships
            if relationship.relationship_type in close_types and relationship.confidence >= 0.6
        }
    )
    return [
        {"feature_name": "event_count", "value": len(events)},
        {"feature_name": "relationship_count", "value": len(relationships)},
        {"feature_name": "group_count", "value": len(groups)},
        {"feature_name": "close_circle_size", "value": close_circle_size},
    ]


def _build_profile_context(state: MemoryState) -> Dict[str, Any]:
    vlm_observations = []
    primary_person_id = (state.primary_decision or {}).get("primary_person_id")
    for item in state.vlm_results or []:
        analysis = item.get("vlm_analysis", {}) or {}
        scene = analysis.get("scene", {}) or {}
        event = analysis.get("event", {}) or {}
        people = [p.get("person_id") for p in analysis.get("people", []) if isinstance(p, dict) and p.get("person_id")]
        face_person_ids = _normalize_string_list(item.get("face_person_ids", []))
        summary = str(analysis.get("summary", "") or "")
        activity = event.get("activity", "") if isinstance(event, dict) else ""
        vlm_observations.append(
            {
                "photo_id": item.get("photo_id"),
                "timestamp": item.get("timestamp"),
                "summary": summary,
                "location": scene.get("location_detected", "") if isinstance(scene, dict) else "",
                "activity": activity,
                "people": people,
                "details": list(analysis.get("details", []) or []),
                "ocr_hits": _normalize_string_list(analysis.get("ocr_hits", [])),
                "brands": _normalize_string_list(analysis.get("brands", [])),
                "place_candidates": _normalize_string_list(analysis.get("place_candidates", [])),
                "face_person_ids": face_person_ids,
                "media_kind": item.get("media_kind"),
                "is_reference_like": bool(item.get("is_reference_like")),
                "subject_role": _determine_subject_role(primary_person_id, people, face_person_ids, summary, activity),
            }
        )
    return {
        "primary_person_id": primary_person_id,
        "events": list(state.events or []),
        "relationships": list(state.relationships or []),
        "groups": list(state.groups or []),
        "vlm_observations": vlm_observations,
        "feature_refs": _build_context_feature_refs(
            events=list(state.events or []),
            relationships=list(state.relationships or []),
            groups=list(state.groups or []),
        ),
        "social_media_available": _extract_metadata_evidence({"events": state.events, "vlm_observations": vlm_observations})["has_social_media_evidence"],
        "resolved_facts": {},
        "raw_upstream": state.raw_upstream,
        "raw_index": state.raw_index,
    }


def _build_raw_ref_lookup(refs: Iterable[Dict[str, Any]], raw_index: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    lookup: List[Dict[str, Any]] = []
    for ref in refs:
        ids = _extract_ids_from_refs([ref])
        lookup.append(
            {
                "ref": {
                    "source_type": ref.get("source_type"),
                    "source_id": ref.get("source_id"),
                    "signal": ref.get("signal"),
                    "why": ref.get("why"),
                },
                "photo_records": [raw_index.get("photo", {}).get(photo_id, {}) for photo_id in ids["photo_ids"]],
                "event_records": [raw_index.get("event", {}).get(event_id, {}) for event_id in ids["event_ids"]],
                "person_records": [raw_index.get("person", {}).get(person_id, {}) for person_id in ids["person_ids"]],
            }
        )
    return lookup


def _fetch_field_evidence(field_key: str, context: Dict[str, Any]) -> Dict[str, Any]:
    spec = FIELD_SPECS[field_key]
    primary_person_id = context.get("primary_person_id")
    allowed_refs = {"events": [], "relationships": [], "vlm_observations": [], "group_artifacts": [], "feature_refs": []}
    if "event" in spec.allowed_sources:
        for event in context.get("events", []):
            allowed_refs["events"].append(
                {
                    "source_type": "event",
                    "source_id": getattr(event, "event_id", ""),
                    "event_id": getattr(event, "event_id", ""),
                    "date": getattr(event, "date", ""),
                    "title": getattr(event, "title", ""),
                    "signal": f"{getattr(event, 'title', '')} @ {getattr(event, 'location', '')}",
                    "why": field_key,
                    "participants": list(getattr(event, "participants", []) or []),
                    "location": getattr(event, "location", ""),
                    "description": getattr(event, "description", ""),
                    "photo_count": getattr(event, "photo_count", 0),
                    "narrative_synthesis": getattr(event, "narrative_synthesis", ""),
                    "tags": list(getattr(event, "tags", []) or []),
                    "persona_evidence": copy.deepcopy(getattr(event, "persona_evidence", {}) or {}),
                    "timestamp": getattr(event, "date", ""),
                }
            )
    if "relationship" in spec.allowed_sources:
        for relationship in context.get("relationships", []):
            allowed_refs["relationships"].append(
                {
                    "source_type": "person",
                    "source_id": relationship.person_id,
                    "person_id": relationship.person_id,
                    "relationship_type": relationship.relationship_type,
                    "signal": relationship.reasoning,
                    "event_ids": list(relationship.evidence.get("event_ids", []) or []),
                    "photo_ids": list(relationship.evidence.get("photo_ids", []) or []),
                    "feature_names": list(relationship.evidence.get("feature_names", []) or []),
                    "why": field_key,
                }
            )
    if "vlm" in spec.allowed_sources:
        for observation in context.get("vlm_observations", []):
            allowed_refs["vlm_observations"].append(
                {
                    "source_type": "photo",
                    "source_id": observation.get("photo_id"),
                    "photo_id": observation.get("photo_id"),
                    "timestamp": observation.get("timestamp"),
                    "signal": " | ".join(
                        filter(None, [observation.get("summary", ""), observation.get("location", ""), observation.get("activity", "")])
                    ),
                    "why": field_key,
                    "people": list(observation.get("people", []) or []),
                    "face_person_ids": list(observation.get("face_person_ids", []) or []),
                    "location": observation.get("location", ""),
                    "activity": observation.get("activity", ""),
                    "details": list(observation.get("details", []) or []),
                    "ocr_hits": list(observation.get("ocr_hits", []) or []),
                    "brands": list(observation.get("brands", []) or []),
                    "place_candidates": list(observation.get("place_candidates", []) or []),
                    "media_kind": observation.get("media_kind"),
                    "is_reference_like": observation.get("is_reference_like"),
                    "subject_role": observation.get("subject_role", ""),
                }
            )
    if "group" in spec.allowed_sources:
        for group in context.get("groups", []):
            allowed_refs["group_artifacts"].append(
                {
                    "source_type": "group",
                    "source_id": group.group_id,
                    "group_id": group.group_id,
                    "group_type": group.group_type_candidate,
                    "title": group.group_type_candidate,
                    "signal": group.reason,
                    "why": field_key,
                }
            )
    if "feature" in spec.allowed_sources:
        for feature in context.get("feature_refs", []):
            feature_name = str(feature.get("feature_name", "") or "")
            allowed_refs["feature_refs"].append(
                {
                    "source_type": "feature",
                    "source_id": feature_name,
                    "feature_name": feature_name,
                    "value": feature.get("value"),
                    "signal": f"{feature_name}: {feature.get('value')}",
                    "why": field_key,
                }
            )
    supporting_refs = copy.deepcopy(allowed_refs)
    if field_key == "long_term_facts.social_identity.education":
        supporting_refs["events"] = [
            ref for ref in allowed_refs["events"]
            if _contains_any_keyword(ref.get("signal", ""), ("学校", "校园", "class", "campus", "教室"))
            and primary_person_id in ref.get("participants", [])
        ]
        supporting_refs["vlm_observations"] = [
            ref for ref in allowed_refs["vlm_observations"]
            if _contains_any_keyword(ref.get("signal", ""), ("学校", "校园", "class", "campus", "教室"))
            and (
                primary_person_id in ref.get("people", [])
                or primary_person_id in ref.get("face_person_ids", [])
                or ref.get("subject_role") == "protagonist_view"
            )
        ]
        supporting_refs["relationships"] = []
        supporting_refs["group_artifacts"] = []
    elif field_key == "long_term_facts.relationships.intimate_partner":
        supporting_refs["relationships"] = [ref for ref in allowed_refs["relationships"] if ref.get("relationship_type") == "romantic"]
        supporting_refs["events"] = []
        supporting_refs["vlm_observations"] = []
        supporting_refs["group_artifacts"] = []
    elif field_key == "long_term_facts.relationships.social_groups":
        supporting_refs["events"] = []
        supporting_refs["vlm_observations"] = []
        supporting_refs["relationships"] = []
        supporting_refs["group_artifacts"] = list(allowed_refs["group_artifacts"])
        supporting_refs["feature_refs"] = []
    elif field_key == "long_term_facts.relationships.close_circle_size":
        supporting_refs["events"] = []
        supporting_refs["vlm_observations"] = []
        supporting_refs["relationships"] = list(allowed_refs["relationships"])
        supporting_refs["group_artifacts"] = []
    elif field_key in {
        "long_term_facts.material.brand_preference",
        "long_term_facts.material.signature_items",
        "short_term_facts.spending_shift",
    }:
        supporting_refs["vlm_observations"] = [
            ref for ref in allowed_refs["vlm_observations"]
            if ref.get("brands") or _contains_any_keyword(ref.get("signal", ""), MATERIAL_SIGNAL_KEYWORDS)
        ]
        supporting_refs["events"] = [
            ref for ref in allowed_refs["events"]
            if _contains_any_keyword(
                " ".join(
                    filter(
                        None,
                        [
                            ref.get("signal", ""),
                            " ".join(ref.get("tags", []) or []),
                            json.dumps(ref.get("persona_evidence", {}), ensure_ascii=False),
                        ],
                    )
                ),
                MATERIAL_SIGNAL_KEYWORDS,
            )
        ]
        supporting_refs["relationships"] = []
        supporting_refs["group_artifacts"] = []
    elif field_key in {
        "long_term_facts.geography.location_anchors",
        "long_term_facts.geography.mobility_pattern",
        "long_term_facts.geography.cross_border",
        "short_term_facts.current_displacement",
    }:
        supporting_refs["events"] = [ref for ref in allowed_refs["events"] if str(ref.get("location") or "").strip()]
        supporting_refs["vlm_observations"] = [
            ref for ref in allowed_refs["vlm_observations"]
            if str(ref.get("location") or "").strip() or list(ref.get("place_candidates", []) or [])
        ]
        supporting_refs["relationships"] = []
        supporting_refs["group_artifacts"] = []
    elif field_key == "short_term_facts.recent_interests":
        latest_timestamp = max(
            [
                str(ref.get("timestamp") or "")
                for ref in allowed_refs["vlm_observations"]
                if str(ref.get("timestamp") or "").strip()
            ]
            or [""]
        )
        latest_window = _window_key_from_timestamp(latest_timestamp)
        supporting_refs["events"] = [
            ref
            for ref in allowed_refs["events"]
            if _window_key_from_timestamp(str(ref.get("timestamp") or ref.get("date") or "")) == latest_window
            or list(ref.get("tags", []) or [])
        ]
        supporting_refs["vlm_observations"] = [
            ref
            for ref in allowed_refs["vlm_observations"]
            if _window_key_from_timestamp(str(ref.get("timestamp") or "")) == latest_window
            or _contains_any_keyword(ref.get("signal", ""), ("interest", "爱好", "喜欢", "watching", "reading"))
        ]
        supporting_refs["relationships"] = []
        supporting_refs["group_artifacts"] = []
    elif field_key == "long_term_facts.time.sleep_pattern":
        supporting_refs["vlm_observations"] = []
        supporting_refs["relationships"] = []
        supporting_refs["group_artifacts"] = []
    elif field_key == "long_term_facts.relationships.living_situation":
        supporting_refs["vlm_observations"] = []
        supporting_refs["group_artifacts"] = []
    supporting_flat = _flatten_ref_buckets(supporting_refs)
    payload = {
        "field_key": field_key,
        "primary_person_id": primary_person_id,
        "allowed_refs": allowed_refs,
        "supporting_refs": supporting_refs,
        "ref_index": _build_ref_index(allowed_refs),
        "ids": _extract_ids_from_refs(supporting_flat),
        "raw_ref_lookup": _build_raw_ref_lookup(supporting_flat, context.get("raw_index", {})),
    }
    return payload


def _analyze_evidence_stats(field_key: str, evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
    supporting_refs = evidence_bundle["supporting_refs"]
    supporting_events = supporting_refs.get("events", [])
    supporting_vlm = supporting_refs.get("vlm_observations", [])
    supporting_relationships = supporting_refs.get("relationships", [])
    supporting_groups = supporting_refs.get("group_artifacts", [])
    supporting_features = supporting_refs.get("feature_refs", [])
    distinct_event_ids = {ref.get("event_id") for ref in supporting_events if ref.get("event_id")}
    photo_count = len({ref.get("photo_id") for ref in supporting_vlm if ref.get("photo_id")})
    event_count = len(distinct_event_ids)
    support_count = (
        len(supporting_events)
        + len(supporting_vlm)
        + len(supporting_relationships)
        + len(supporting_groups)
        + len(supporting_features)
    )
    timestamps = [
        _window_key_from_timestamp(str(ref.get("timestamp") or ref.get("date") or ""))
        for ref in list(supporting_events) + list(supporting_vlm)
        if _window_key_from_timestamp(str(ref.get("timestamp") or ref.get("date") or ""))
    ]
    window_count = len(set(timestamps))
    is_non_daily_burst = any(
        _contains_any_keyword(
            " ".join(
                filter(
                    None,
                    [
                        ref.get("signal", ""),
                        ref.get("description", ""),
                        ref.get("narrative_synthesis", ""),
                    ],
                )
            ),
            NON_DAILY_EVENT_KEYWORDS,
        )
        for ref in list(supporting_events) + list(supporting_vlm)
    )
    suggested_strong_evidence_met = (
        len(supporting_relationships) >= 1
        if field_key == "long_term_facts.relationships.intimate_partner"
        else len(supporting_groups) >= 1
        if field_key == "long_term_facts.relationships.social_groups"
        else support_count >= (2 if FIELD_SPECS[field_key].risk_level == "P0" else 1)
    )
    if field_key in {"long_term_facts.geography.location_anchors", "short_term_facts.current_displacement"}:
        suggested_strong_evidence_met = bool(supporting_events or supporting_vlm)
    if field_key == "long_term_facts.material.brand_preference":
        suggested_strong_evidence_met = any(ref.get("brands") for ref in supporting_vlm)
    if field_key == "short_term_facts.recent_interests":
        suggested_strong_evidence_met = support_count >= 2
    return {
        "field_key": field_key,
        "photo_count": photo_count,
        "event_count": event_count,
        "window_count": window_count,
        "support_count": support_count,
        "is_non_daily_burst": is_non_daily_burst,
        "suggested_strong_evidence_met": suggested_strong_evidence_met,
    }


def _check_subject_ownership(field_key: str, evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
    primary_person_id = evidence_bundle.get("primary_person_id")
    supporting_refs = evidence_bundle["supporting_refs"]
    protagonist_hits = 0
    ambiguous_hits = 0
    candidate_signals = []
    for ref in list(supporting_refs.get("vlm_observations", []) or []):
        if primary_person_id and (
            primary_person_id in list(ref.get("people", []) or [])
            or primary_person_id in list(ref.get("face_person_ids", []) or [])
            or ref.get("subject_role") in {"protagonist_present", "protagonist_view"}
        ):
            protagonist_hits += 1
            candidate_signals.append({"candidate": ref.get("photo_id"), "signal": "owned_or_used"})
        else:
            ambiguous_hits += 1
            candidate_signals.append({"candidate": ref.get("photo_id"), "signal": "background_or_ambiguous"})
    ownership_signal = "owned_or_used" if protagonist_hits > ambiguous_hits else "background_or_ambiguous"
    if field_key.startswith("long_term_facts.relationships.") or field_key.startswith("short_term_facts."):
        ownership_signal = "owned_or_used"
    return {
        "field_key": field_key,
        "ownership_signal": ownership_signal,
        "candidate_signals": candidate_signals[:8],
    }


def _find_counter_evidence(field_key: str, evidence_bundle: Dict[str, Any], ownership_bundle: Dict[str, Any]) -> Dict[str, Any]:
    conflicts: List[str] = []
    contradicting_refs: List[Dict[str, Any]] = []
    for ref in list(evidence_bundle["allowed_refs"].get("vlm_observations", []) or []):
        if ref.get("is_reference_like"):
            conflicts.append("reference_like_media")
            contradicting_refs.append(ref)
        if ref.get("subject_role") == "other_people_only" and ownership_bundle.get("ownership_signal") != "owned_or_used":
            conflicts.append("subject_not_bound_to_primary")
            contradicting_refs.append(ref)
    if field_key in {"long_term_facts.geography.location_anchors", "short_term_facts.current_displacement"}:
        for ref in list(evidence_bundle["supporting_refs"].get("vlm_observations", []) or []):
            location = str(ref.get("location") or "")
            if location and any(location.lower().endswith(term.lower()) for term in LOCATION_STOPWORDS):
                conflicts.append("generic_location")
                contradicting_refs.append(ref)
    if field_key == "long_term_facts.material.brand_preference" and ownership_bundle.get("ownership_signal") != "owned_or_used":
        conflicts.append("brand_ownership_ambiguous")
    if field_key.startswith("long_term_facts.") and evidence_bundle.get("supporting_refs", {}).get("events"):
        if any(
            _contains_any_keyword(
                " ".join(
                    filter(
                        None,
                        [ref.get("signal", ""), ref.get("description", ""), ref.get("narrative_synthesis", "")]
                    )
                ),
                NON_DAILY_EVENT_KEYWORDS,
            )
            for ref in list(evidence_bundle["supporting_refs"].get("events", []) or [])
        ):
            conflicts.append("non_daily_event_burst_interference")
    ids = _extract_ids_from_refs(contradicting_refs)
    return {
        "field_key": field_key,
        "contradicting_refs": contradicting_refs,
        "conflict_types": _unique_strings(conflicts),
        "conflict_strength": len(set(conflicts)),
        "contradicting_ids": ids,
    }


def _extract_metadata_evidence(context: Dict[str, Any]) -> Dict[str, Any]:
    refs: List[Dict[str, Any]] = []
    source_types: List[str] = []
    for observation in context.get("vlm_observations", []):
        signal = " ".join(
            filter(
                None,
                [
                    observation.get("summary", ""),
                    observation.get("location", ""),
                    observation.get("activity", ""),
                    " ".join(str(item) for item in observation.get("ocr_hits", []) or []),
                ],
            )
        )
        if _contains_any_keyword(signal, SOCIAL_MEDIA_KEYWORDS):
            refs.append({"source_type": "photo", "source_id": observation.get("photo_id"), "photo_id": observation.get("photo_id"), "signal": signal})
            source_types.append("screenshot_or_ui")
    for event in context.get("events", []):
        signal = " ".join(filter(None, [getattr(event, "title", ""), getattr(event, "description", ""), getattr(event, "narrative_synthesis", "")]))
        if _contains_any_keyword(signal, SOCIAL_MEDIA_KEYWORDS):
            refs.append({"source_type": "event", "source_id": getattr(event, "event_id", ""), "event_id": getattr(event, "event_id", ""), "signal": signal})
            source_types.append("event_text")
    return {
        "has_social_media_evidence": bool(refs),
        "source_types": _unique_strings(source_types),
        "metadata_ids": _extract_ids_from_refs(refs),
    }


def _deterministic_field_value(field_key: str, context: Dict[str, Any]) -> Tuple[Any, float]:
    relationships = context.get("relationships", [])
    groups = context.get("groups", [])
    events = context.get("events", [])
    if field_key == "long_term_facts.relationships.intimate_partner":
        romantic = next((rel.person_id for rel in relationships if rel.relationship_type == "romantic"), None)
        return romantic, 0.88 if romantic else 0.0
    if field_key == "long_term_facts.relationships.close_circle_size":
        close_types = {"romantic", "family", "bestie", "close_friend"}
        return len({rel.person_id for rel in relationships if rel.relationship_type in close_types and rel.confidence >= 0.6}), 0.82
    if field_key == "long_term_facts.relationships.social_groups":
        group_names = [group.group_type_candidate for group in groups]
        return group_names or None, 0.76 if group_names else 0.0
    if field_key == "long_term_facts.time.sleep_pattern":
        late_night_events = 0
        for event in events:
            time_range = getattr(event, "time_range", "") or ""
            start_time = time_range.split(" - ")[0].strip()
            if start_time[:2].isdigit() and int(start_time[:2]) >= 22:
                late_night_events += 1
        ratio = round(late_night_events / max(len(events), 1), 2) if events else 0.0
        if ratio >= 0.5:
            return "night_owl", 0.65
        if ratio > 0:
            return "irregular", 0.55
        return None, 0.0
    if field_key == "long_term_facts.relationships.living_situation":
        shared_home_rel_ids = {
            rel.person_id
            for rel in relationships
            if rel.relationship_type in {"romantic", "family"} and rel.confidence >= 0.7
        }
        if shared_home_rel_ids:
            home_event_hits = 0
            for event in events:
                location_text = " ".join([str(getattr(event, "location", "") or ""), str(getattr(event, "title", "") or ""), str(getattr(event, "description", "") or "")]).lower()
                participants = set(getattr(event, "participants", []) or [])
                if participants.intersection(shared_home_rel_ids) and any(keyword in location_text for keyword in ("home", "room", "宿舍", "家", "apartment", "卧室")):
                    home_event_hits += 1
            if home_event_hits >= 2:
                return "shared", 0.78
        return None, 0.0
    return None, 0.0


def _build_tag_evidence(
    evidence_bundle: Dict[str, Any],
    *,
    selected_supporting_ids: List[str] | None = None,
    selected_contradicting_ids: List[str] | None = None,
    constraint_notes: List[str] | None = None,
) -> Dict[str, Any]:
    ref_index = evidence_bundle.get("ref_index") or {}
    supporting_refs = _select_refs(ref_index, selected_supporting_ids) or _flatten_ref_buckets(evidence_bundle["supporting_refs"])
    contradicting_refs = _select_refs(ref_index, selected_contradicting_ids)
    ids = _extract_ids_from_refs(supporting_refs)
    evidence = _build_evidence_payload(
        photo_ids=ids["photo_ids"],
        event_ids=ids["event_ids"],
        person_ids=ids["person_ids"],
        group_ids=ids["group_ids"],
        feature_names=ids["feature_names"],
        supporting_refs=supporting_refs,
        contradicting_refs=contradicting_refs,
    )
    allowed_refs = evidence_bundle["allowed_refs"]
    evidence["events"] = _filter_bucket(allowed_refs["events"], supporting_refs)
    evidence["relationships"] = _filter_bucket(allowed_refs["relationships"], supporting_refs)
    evidence["vlm_observations"] = _filter_bucket(allowed_refs["vlm_observations"], supporting_refs)
    evidence["group_artifacts"] = _filter_bucket(allowed_refs["group_artifacts"], supporting_refs)
    evidence["feature_refs"] = _filter_bucket(allowed_refs["feature_refs"], supporting_refs)
    evidence["constraint_notes"] = _dedupe_strs(constraint_notes)
    evidence["summary"] = f"field_judge:{evidence_bundle.get('field_key')}"
    return evidence


def _llm_field_value(field_key: str, evidence_bundle: Dict[str, Any], stats_bundle: Dict[str, Any], ownership_bundle: Dict[str, Any], counter_bundle: Dict[str, Any], llm_processor: Any) -> Dict[str, Any]:
    prompt = f"""你是客观画像字段判定 agent。
字段: {field_key}
风险等级: {FIELD_SPECS[field_key].risk_level}
强证据: {FIELD_SPECS[field_key].strong_evidence}
主体归属: {ownership_bundle.get('ownership_signal')}
统计: {json.dumps(stats_bundle, ensure_ascii=False)}
冲突: {json.dumps(counter_bundle, ensure_ascii=False)}
允许证据:
{json.dumps(evidence_bundle['supporting_refs'], ensure_ascii=False, default=_json_default)}
可引用锚点 IDs:
{json.dumps(evidence_bundle.get('ids') or {}, ensure_ascii=False)}

请只输出 JSON:
{{
  "value": null,
  "confidence": 0.0,
  "reasoning": "",
  "supporting_ref_ids": [],
  "contradicting_ref_ids": [],
  "null_reason": null
}}"""
    try:
        result = llm_processor._call_llm_via_official_api(prompt, response_mime_type="application/json")
    except Exception:
        return {}
    if not isinstance(result, dict):
        return {}
    ref_index = evidence_bundle.get("ref_index") or {}
    return {
        "value": result.get("value"),
        "confidence": float(result.get("confidence", 0.0) or 0.0),
        "reasoning": str(result.get("reasoning") or "").strip(),
        "supporting_ref_ids": _normalize_ref_id_list(result.get("supporting_ref_ids"), ref_index=ref_index),
        "contradicting_ref_ids": _normalize_ref_id_list(result.get("contradicting_ref_ids"), ref_index=ref_index),
        "null_reason": str(result.get("null_reason") or "").strip() or None,
    }


def _run_field_agent_profile(*, context: Dict[str, Any], llm_processor: Any | None) -> Dict[str, Any]:
    structured = _build_empty_structured_profile()
    profile_state = ProfileState(structured_profile=copy.deepcopy(structured))
    metadata_bundle = _extract_metadata_evidence(context)
    for domain_spec in DOMAIN_SPECS:
        for field_key in domain_spec["fields"]:
            evidence_bundle = _fetch_field_evidence(field_key, context)
            stats_bundle = _analyze_evidence_stats(field_key, evidence_bundle)
            ownership_bundle = _check_subject_ownership(field_key, evidence_bundle)
            counter_bundle = _find_counter_evidence(field_key, evidence_bundle, ownership_bundle)
            deterministic_value, deterministic_confidence = _deterministic_field_value(field_key, context)
            if deterministic_value is not None:
                tag_object = {
                    "value": deterministic_value,
                    "confidence": round(min(max(deterministic_confidence, 0.0), 1.0), 3),
                    "evidence": _build_tag_evidence(evidence_bundle),
                    "reasoning": f"{field_key} 使用确定性规则产出。",
                }
                field_output = {
                    "value": deterministic_value,
                    "confidence": round(min(max(deterministic_confidence, 0.0), 1.0), 3),
                    "supporting_ref_ids": [],
                    "contradicting_ref_ids": [],
                    "null_reason": None,
                }
            elif not stats_bundle.get("suggested_strong_evidence_met"):
                tag_object = _empty_tag_object()
                tag_object["reasoning"] = f"{field_key} 证据不足，保守输出 null。"
                field_output = {
                    "value": None,
                    "confidence": 0.0,
                    "supporting_ref_ids": [],
                    "contradicting_ref_ids": [],
                    "null_reason": "insufficient_evidence",
                }
            else:
                field_output: Dict[str, Any] = {
                    "value": None,
                    "confidence": 0.0,
                    "reasoning": "",
                    "supporting_ref_ids": [],
                    "contradicting_ref_ids": [],
                    "null_reason": None,
                }
                if llm_processor and hasattr(llm_processor, "_call_llm_via_official_api"):
                    field_output = _llm_field_value(
                        field_key,
                        evidence_bundle,
                        stats_bundle,
                        ownership_bundle,
                        counter_bundle,
                        llm_processor,
                    )
                value = field_output.get("value")
                confidence = float(field_output.get("confidence", 0.0) or 0.0)
                tag_object = {
                    "value": value,
                    "confidence": round(min(max(confidence, 0.0), 1.0), 3),
                    "evidence": _build_tag_evidence(
                        evidence_bundle,
                        selected_supporting_ids=field_output.get("supporting_ref_ids", []),
                        selected_contradicting_ids=field_output.get("contradicting_ref_ids", []),
                        constraint_notes=[field_output["null_reason"]] if field_output.get("null_reason") else [],
                    ),
                    "reasoning": (
                        str(field_output.get("reasoning") or "").strip()
                        or (
                            f"{field_key} 基于字段级证据判定完成。"
                            if value not in (None, "", [])
                            else f"{field_key} 未通过字段闸门或冲突检查，输出 null。"
                        )
                    ),
                }
            _assign_tag_object(profile_state.structured_profile, field_key, tag_object)
            profile_state.field_decisions.append(
                {
                    "field_key": field_key,
                    "domain_key": domain_spec["domain_key"],
                    "stats": stats_bundle,
                    "ownership": ownership_bundle,
                    "counter": {
                        "conflict_types": counter_bundle["conflict_types"],
                        "conflict_strength": counter_bundle["conflict_strength"],
                    },
                    "raw_ref_lookup_count": len(evidence_bundle.get("raw_ref_lookup", [])),
                    "selected_supporting_ref_ids": list(field_output.get("supporting_ref_ids", []) or []),
                    "selected_contradicting_ref_ids": list(field_output.get("contradicting_ref_ids", []) or []),
                    "null_reason": field_output.get("null_reason"),
                    "final": {
                        "value": tag_object.get("value"),
                        "confidence": tag_object.get("confidence"),
                    },
                }
            )
    return {
        "structured": profile_state.structured_profile,
        "field_decisions": profile_state.field_decisions,
    }


def _build_consistency_report(events: List[Event], relationships: List[RelationshipRecord], structured_profile: Dict[str, Any]) -> Dict[str, Any]:
    issues = []
    profile_relationships = structured_profile.get("long_term_facts", {}).get("relationships", {})
    profile_partner = (profile_relationships.get("intimate_partner", {}) or {}).get("value")
    profile_circle_size = (profile_relationships.get("close_circle_size", {}) or {}).get("value")
    romantic_partner = next((rel.person_id for rel in relationships if rel.relationship_type == "romantic"), None)
    close_circle_size = sum(1 for rel in relationships if rel.relationship_type in {"romantic", "bestie", "close_friend", "family"})
    if profile_partner and profile_partner != romantic_partner:
        issues.append(
            {
                "code": "INTIMATE_PARTNER_MISMATCH",
                "severity": "high",
                "message": f"profile.intimate_partner={profile_partner} 与 LP2 romantic={romantic_partner} 不一致",
            }
        )
    if profile_circle_size is not None and profile_circle_size != close_circle_size:
        issues.append(
            {
                "code": "CLOSE_CIRCLE_SIZE_MISMATCH",
                "severity": "medium",
                "message": f"profile.close_circle_size={profile_circle_size} 与 LP2 close_circle_size={close_circle_size} 不一致",
            }
        )
    if not events:
        issues.append(
            {
                "code": "NO_EVENTS",
                "severity": "medium",
                "message": "LP1 事件为空，画像仅能基于稀疏关系或视觉线索。",
            }
        )
    return {
        "summary": {
            "issue_count": len(issues),
            "high_risk_issue_count": sum(1 for issue in issues if issue["severity"] == "high"),
        },
        "issues": issues,
    }


def _run_downstream_audit_and_backflow(
    *,
    primary_decision: Dict[str, Any],
    relationships: List[RelationshipRecord],
    structured_profile: Dict[str, Any],
    groups: List[GroupArtifact],
    consistency: Dict[str, Any],
) -> Dict[str, Any]:
    final_structured = copy.deepcopy(structured_profile)
    flags: List[Dict[str, Any]] = []
    patched_fields: List[str] = []
    profile_relationships = final_structured.setdefault("long_term_facts", {}).setdefault("relationships", {})
    partner_object = profile_relationships.get("intimate_partner") or _empty_tag_object()
    close_circle_object = profile_relationships.get("close_circle_size") or _empty_tag_object()
    social_group_object = profile_relationships.get("social_groups") or _empty_tag_object()
    romantic_partner = next((rel.person_id for rel in relationships if rel.relationship_type == "romantic"), None)
    close_types = {"romantic", "family", "bestie", "close_friend"}
    close_circle_size = len({rel.person_id for rel in relationships if rel.relationship_type in close_types and rel.confidence >= 0.6})
    inferred_groups = [group.group_type_candidate for group in groups]
    if partner_object.get("value") != romantic_partner:
        patched_fields.append("long_term_facts.relationships.intimate_partner")
        partner_object["value"] = romantic_partner
        partner_object["confidence"] = 0.88 if romantic_partner else 0.0
        partner_object["reasoning"] = "downstream_audit_backflow: 画像层伴侣字段与关系层 romantic 结果对齐。"
        flags.append({"field_key": "long_term_facts.relationships.intimate_partner", "audit_status": "patched"})
    if close_circle_object.get("value") != close_circle_size:
        patched_fields.append("long_term_facts.relationships.close_circle_size")
        close_circle_object["value"] = close_circle_size
        close_circle_object["confidence"] = 0.82
        close_circle_object["reasoning"] = "downstream_audit_backflow: close_circle_size 与 LP2 close types 计数对齐。"
        flags.append({"field_key": "long_term_facts.relationships.close_circle_size", "audit_status": "patched"})
    if inferred_groups and social_group_object.get("value") != inferred_groups:
        patched_fields.append("long_term_facts.relationships.social_groups")
        social_group_object["value"] = inferred_groups
        social_group_object["confidence"] = 0.76
        social_group_object["reasoning"] = "downstream_audit_backflow: 画像层 social_groups 与 group artifacts 对齐。"
        flags.append({"field_key": "long_term_facts.relationships.social_groups", "audit_status": "patched"})
    profile_relationships["intimate_partner"] = partner_object
    profile_relationships["close_circle_size"] = close_circle_object
    profile_relationships["social_groups"] = social_group_object
    return {
        "metadata": {
            "downstream_engine": "v0325_internal_audit",
            "audit_mode": "rule_based_consistency_backflow",
            "profile_agent_root": PROFILE_AGENT_ROOT,
            "profile_llm_provider": PROFILE_LLM_PROVIDER,
            "profile_llm_model": PROFILE_LLM_MODEL or OPENROUTER_LLM_MODEL,
            "openrouter_base_url": OPENROUTER_BASE_URL,
            "openrouter_key_present": bool(OPENROUTER_API_KEY),
        },
        "summary": {
            "challenged_count": len(flags),
            "patched_count": len(patched_fields),
            "high_risk_issue_count": consistency.get("summary", {}).get("high_risk_issue_count", 0),
        },
        "audit_flags": flags,
        "patched_fields": patched_fields,
        "backflow": {
            "applied": bool(patched_fields),
            "patched_fields": patched_fields,
        },
        "final_structured_profile": final_structured,
    }


def _build_profile_summary(structured_profile: Dict[str, Any], relationships: List[RelationshipRecord]) -> str:
    relationship_count = len(relationships)
    partner = _get_nested(structured_profile, "long_term_facts.relationships.intimate_partner.value")
    interests = _get_nested(structured_profile, "long_term_facts.hobbies.interests.value")
    parts = [f"共识别 {relationship_count} 条保留关系"]
    if partner:
        parts.append(f"伴侣线索指向 {partner}")
    if interests:
        parts.append(f"长期兴趣包括 {_value_to_string(interests)}")
    return "；".join(parts) if parts else "当前画像仍以稀疏线索为主。"


def _build_report_markdown(
    *,
    primary_decision: Dict[str, Any],
    structured_profile: Dict[str, Any],
    consistency: Dict[str, Any],
    relationships: List[RelationshipRecord],
    groups: List[GroupArtifact],
    audit_report: Dict[str, Any],
) -> str:
    partner = _get_nested(structured_profile, "long_term_facts.relationships.intimate_partner.value")
    close_circle = _get_nested(structured_profile, "long_term_facts.relationships.close_circle_size.value")
    anchors = _get_nested(structured_profile, "long_term_facts.geography.location_anchors.value")
    interests = _get_nested(structured_profile, "long_term_facts.hobbies.interests.value")
    recent_interests = _get_nested(structured_profile, "short_term_facts.recent_interests.value")
    lines = [
        "# LP3 画像简报",
        "",
        f"- 主角: {primary_decision.get('primary_person_id') or '未确定'}",
        f"- 保留关系数: {len(relationships)}",
        f"- 群组数: {len(groups)}",
        f"- 审计回流修正字段数: {len(list(audit_report.get('patched_fields', []) or []))}",
        "",
        "## 长期画像",
        "",
        f"- 伴侣: {_value_to_string(partner) if partner not in (None, '', []) else '未确定'}",
        f"- 亲密圈大小: {_value_to_string(close_circle) if close_circle not in (None, '', []) else '未确定'}",
        f"- 地点锚点: {_value_to_string(anchors) if anchors not in (None, '', []) else '未确定'}",
        f"- 长期兴趣: {_value_to_string(interests) if interests not in (None, '', []) else '未确定'}",
        "",
        "## 近期状态",
        "",
        f"- 近期兴趣: {_value_to_string(recent_interests) if recent_interests not in (None, '', []) else '未确定'}",
        "",
        "## 关系摘要",
        "",
    ]
    if relationships:
        for relationship in relationships[:8]:
            lines.append(
                f"- {relationship.person_id}: {relationship.relationship_type} / {relationship.status} ({relationship.confidence:.2f})"
            )
    else:
        lines.append("- 当前没有保留关系。")
    lines.extend(["", "## 一致性", ""])
    if consistency.get("issues"):
        for issue in consistency.get("issues", []):
            lines.append(f"- [{issue.get('severity')}] {issue.get('message')}")
    else:
        lines.append("- 未发现高风险跨层冲突。")
    return "\n".join(lines).strip()
