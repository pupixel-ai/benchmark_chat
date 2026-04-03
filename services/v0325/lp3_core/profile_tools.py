from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import re
from typing import Any, Dict, Iterable, List

from .evidence_utils import extract_ids_from_refs, flatten_ref_buckets

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

SOLITARY_SIGNAL_KEYWORDS = (
    "独处",
    "alone",
    "solo",
    "reading alone",
    "安静",
    "quiet",
    "看书",
)

BRAND_PRODUCT_KEYWORDS = (
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

NON_BRAND_SIGNAL_KEYWORDS = (
    "偶像",
    "idol",
    "海报",
    "poster",
    "花束",
    "bouquet",
    "当前",
    "面部",
    "状态",
    "虚拟",
    "滤镜",
    "头像",
)

BRAND_NOISE_TERMS = {
    "hello",
    "brand",
    "logo",
    "profile",
    "current",
    "花束",
    "偶像",
    "海报",
    "当前面部状态与",
    "有粉色蝴蝶结虚拟",
}

GENERIC_PLACE_TERMS = {
    "办公区",
    "便利店",
    "博物馆",
    "餐厅",
    "咖啡馆",
    "教室",
    "校园",
    "办公室",
    "宿舍",
    "卧室",
    "dorm room",
    "classroom",
    "campus",
    "office",
    "museum",
    "restaurant",
    "cafe",
    "convenience store",
}

SCHOOL_WORK_PLACE_KEYWORDS = (
    "学校",
    "校园",
    "教室",
    "classroom",
    "campus",
    "office",
    "办公室",
    "公司",
    "实验室",
    "library",
    "图书馆",
)

TRAVEL_PLACE_KEYWORDS = (
    "公园",
    "park",
    "景区",
    "景点",
    "museum",
    "博物馆",
    "天坛",
    "hotel",
    "resort",
)

OTHERS_PLACE_KEYWORDS = (
    "朋友家",
    "别人家",
    "室友家",
    "男朋友家",
    "女朋友家",
    "酒店",
    "民宿",
    "hotel",
    "airbnb",
)

PASSBY_PLACE_KEYWORDS = (
    "路过",
    "门口",
    "路边",
    "经过",
    "passing by",
    "outside",
    "pickup",
    "快递柜",
)

TOPIC_STOPWORDS = {
    "and",
    "the",
    "with",
    "watching",
    "continue",
    "again",
    "video",
    "notes",
    "lecture",
    "content",
    "topic",
    "study",
    "class",
}

EXPLICIT_NAME_KEYWORDS = (
    "姓名",
    "名字",
    "昵称",
    "author",
    "账号",
    "@",
    "id card",
    "身份证",
)

EXPLICIT_GENDER_KEYWORDS = (
    "男",
    "女",
    "先生",
    "女士",
    "female",
    "male",
    "woman",
    "man",
    "girl",
    "boy",
)

EXPLICIT_AGE_ROLE_KEYWORDS = (
    "学生",
    "高中",
    "大学",
    "college",
    "campus",
    "teen",
    "adult",
    "毕业",
    "classmate",
    "同学",
    "intern",
    "teacher",
    "employee",
    "游客",
)

CAREER_SIGNAL_KEYWORDS = SCHOOL_WORK_PLACE_KEYWORDS + (
    "meeting",
    "work",
    "project",
    "intern",
    "coworker",
    "职业",
    "工作",
    "实习",
    "上班",
    "会议",
    "工位",
)

SPENDING_SIGNAL_KEYWORDS = BRAND_PRODUCT_KEYWORDS + (
    "shopping",
    "purchase",
    "delivery",
    "外卖",
    "酒店",
    "hotel",
    "消费",
    "购买",
    "买",
    "收货",
    "快递",
    "茶饮",
    "coffee",
)

HOUSEHOLD_SIGNAL_KEYWORDS = (
    "家",
    "home",
    "apartment",
    "宿舍",
    "bedroom",
    "living room",
    "dorm",
    "roommate",
    "同住",
    "室友",
    "pet",
    "dog",
    "cat",
    "baby",
    "child",
    "儿童",
    "小孩",
)

STYLE_SIGNAL_KEYWORDS = (
    "穿搭",
    "风格",
    "ootd",
    "outfit",
    "look",
    "自拍",
    "镜面",
    "镜子",
    "构图",
    "摄影",
    "style",
    "aesthetic",
    "色调",
    "妆容",
    "外套",
    "hoodie",
    "shirt",
    "coat",
    "hat",
)

MOOD_SIGNAL_KEYWORDS = (
    "开心",
    "难过",
    "疲惫",
    "放松",
    "紧张",
    "压力",
    "焦虑",
    "平静",
    "独处",
    "聚会",
    "quiet",
    "happy",
    "sad",
    "tired",
    "relaxed",
    "stressed",
    "anxious",
    "party",
    "alone",
    "social",
)

VALUES_SIGNAL_KEYWORDS = (
    "信仰",
    "价值观",
    "原则",
    "理念",
    "哲学",
    "人生观",
    "belief",
    "moral",
    "ethic",
    "worldview",
)

NATIONALITY_QUERY_KEYWORDS = (
    "国籍",
    "nationality",
    "护照",
    "passport",
    "签证",
    "visa",
    "籍贯",
    "home country",
)

LANGUAGE_CULTURE_QUERY_KEYWORDS = (
    "中文",
    "汉语",
    "汉字",
    "普通话",
    "粤语",
    "英语",
    "english",
    "chinese",
    "cantonese",
    "mandarin",
    "双语",
    "bilingual",
    "文化",
    "culture",
    "传统",
    "书法",
)

ACTIVITY_INTEREST_QUERY_KEYWORDS = (
    "兴趣",
    "爱好",
    "摄影",
    "拍照",
    "travel",
    "旅行",
    "hiking",
    "徒步",
    "篮球",
    "basketball",
    "football",
    "soccer",
    "music",
    "唱歌",
    "reading",
    "看书",
    "游戏",
    "gaming",
    "movie",
    "电影",
    "museum",
    "艺术",
    "绘画",
    "fitness",
    "健身",
    "food",
    "美食",
)

SOCIAL_ACTIVITY_QUERY_KEYWORDS = (
    "独自",
    "alone",
    "solo",
    "朋友",
    "friends",
    "聚会",
    "party",
    "social",
    "一起",
    "同行",
)

DIET_FITNESS_QUERY_KEYWORDS = (
    "健身",
    "fitness",
    "workout",
    "训练",
    "run",
    "running",
    "篮球",
    "basketball",
    "足球",
    "swimming",
    "游泳",
    "饮食",
    "diet",
    "低卡",
    "高蛋白",
    "protein",
    "meal",
    "餐",
    "咖啡",
)

LIFE_EVENT_QUERY_KEYWORDS = (
    "毕业",
    "graduation",
    "入职",
    "job",
    "promotion",
    "搬家",
    "move",
    "分手",
    "breakup",
    "恋爱",
    "engagement",
    "婚礼",
    "birthday",
    "生日",
    "住院",
    "手术",
    "旅行",
    "返程",
    "比赛",
    "演出",
)

IDENTITY_EXPLICIT_FIELDS = {
    "long_term_facts.identity.name",
    "long_term_facts.identity.gender",
    "long_term_facts.identity.age_range",
    "long_term_facts.identity.role",
    "long_term_facts.identity.race",
}

CAREER_FOCUS_FIELDS = {
    "long_term_facts.social_identity.career",
    "long_term_facts.social_identity.career_phase",
    "long_term_facts.social_identity.professional_dedication",
    "long_term_facts.material.income_model",
}

MATERIAL_FOCUS_FIELDS = {
    "long_term_facts.material.asset_level",
    "long_term_facts.material.spending_style",
    "long_term_facts.material.brand_preference",
    "long_term_facts.material.income_model",
    "long_term_facts.material.signature_items",
    "short_term_facts.spending_shift",
}

SPENDING_FOCUS_FIELDS = {
    "long_term_facts.material.asset_level",
    "long_term_facts.material.spending_style",
    "long_term_facts.material.signature_items",
    "short_term_facts.spending_shift",
}

TIME_FOCUS_FIELDS = {
    "long_term_facts.time.life_rhythm",
    "long_term_facts.time.sleep_pattern",
    "long_term_facts.time.event_cycles",
    "short_term_facts.phase_change",
    "short_term_facts.current_displacement",
    "short_term_facts.recent_habits",
}

GEOGRAPHY_FOCUS_FIELDS = {
    "long_term_facts.geography.location_anchors",
    "long_term_facts.geography.mobility_pattern",
    "long_term_facts.geography.cross_border",
    "short_term_facts.current_displacement",
}

EVENT_DRIVEN_GEO_FIELDS = {
    "long_term_facts.geography.cross_border",
    "short_term_facts.current_displacement",
}

OWNERSHIP_SIGNAL_ORDER = {
    "owned_or_used": 0,
    "worn": 1,
    "venue_context": 2,
    "other_person": 3,
    "background_or_ambiguous": 4,
}

HOUSEHOLD_FOCUS_FIELDS = {
    "long_term_facts.relationships.pets",
    "long_term_facts.relationships.parenting",
    "long_term_facts.relationships.living_situation",
}

STYLE_FOCUS_FIELDS = {
    "long_term_expression.attitude_style",
    "long_term_expression.aesthetic_tendency",
    "long_term_expression.visual_creation_style",
}

MOOD_FOCUS_FIELDS = {
    "short_term_expression.current_mood",
    "short_term_expression.social_energy",
    "short_term_expression.mental_state",
    "short_term_expression.motivation_shift",
    "short_term_expression.stress_signal",
}

VALUES_FOCUS_FIELDS = {
    "long_term_expression.personality_mbti",
    "long_term_expression.morality",
    "long_term_expression.philosophy",
}


class FetchFieldEvidenceTool:
    name = "fetch_field_evidence"

    def execute(
        self,
        *,
        field_key: str,
        context: Dict[str, Any],
        profile_state: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return fetch_field_evidence(field_key, context, profile_state=profile_state)


class AnalyzeEvidenceStatsTool:
    name = "analyze_evidence_stats"

    def execute(
        self,
        *,
        field_key: str,
        evidence_bundle: Dict[str, Any],
        ownership_bundle: Dict[str, Any] | None = None,
        profile_state: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return analyze_evidence_stats(
            field_key,
            evidence_bundle,
            ownership_bundle=ownership_bundle,
            profile_state=profile_state,
        )


class CheckSubjectOwnershipTool:
    name = "check_subject_ownership"

    def execute(self, *, field_key: str, evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
        return check_subject_ownership(field_key, evidence_bundle)


class FindCounterEvidenceTool:
    name = "find_counter_evidence"

    def execute(
        self,
        *,
        field_key: str,
        evidence_bundle: Dict[str, Any],
        ownership_bundle: Dict[str, Any] | None = None,
        profile_state: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return find_counter_evidence(
            field_key,
            evidence_bundle,
            ownership_bundle=ownership_bundle,
            profile_state=profile_state,
        )


class ExtractMetadataEvidenceTool:
    name = "extract_metadata_evidence"

    def execute(self, *, context: Dict[str, Any]) -> Dict[str, Any]:
        return extract_metadata_evidence(context)


TOOL_REGISTRY = {
    "fetch_field_evidence": FetchFieldEvidenceTool(),
    "analyze_evidence_stats": AnalyzeEvidenceStatsTool(),
    "check_subject_ownership": CheckSubjectOwnershipTool(),
    "find_counter_evidence": FindCounterEvidenceTool(),
    "extract_metadata_evidence": ExtractMetadataEvidenceTool(),
}


def get_tool(tool_name: str):
    if tool_name not in TOOL_REGISTRY:
        raise KeyError(f"unknown_profile_tool:{tool_name}")
    return TOOL_REGISTRY[tool_name]


def fetch_field_evidence(field_key: str, context: Dict[str, Any], profile_state: Dict[str, Any] | None = None) -> Dict[str, Any]:
    from .profile_fields import FIELD_SPECS

    spec = FIELD_SPECS[field_key]
    primary_person_id = context.get("primary_person_id")
    allowed_refs = {
        "events": [],
        "relationships": [],
        "vlm_observations": [],
        "group_artifacts": [],
        "feature_refs": [],
    }

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
                    "type": getattr(event, "type", ""),
                    "location": getattr(event, "location", ""),
                    "description": getattr(event, "description", ""),
                    "photo_count": getattr(event, "photo_count", 0),
                    "narrative_synthesis": getattr(event, "narrative_synthesis", ""),
                    "tags": list(getattr(event, "tags", []) or []),
                    "persona_evidence": deepcopy(getattr(event, "persona_evidence", {}) or {}),
                }
            )

    if "relationship" in spec.allowed_sources:
        for relationship in context.get("relationships", []):
            relationship_evidence = relationship.evidence or {}
            allowed_refs["relationships"].append(
                {
                    "source_type": "person",
                    "source_id": relationship.person_id,
                    "person_id": relationship.person_id,
                    "relationship_type": relationship.relationship_type,
                    "signal": relationship.reasoning,
                    "event_ids": relationship_evidence.get("event_ids", []),
                    "photo_ids": relationship_evidence.get("photo_ids", []),
                    "feature_names": relationship_evidence.get("feature_names", []),
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
                        filter(
                            None,
                            [
                                observation.get("summary", ""),
                                observation.get("location", ""),
                                observation.get("activity", ""),
                            ],
                        )
                    ),
                    "why": field_key,
                    "people": observation.get("people", []),
                    "face_person_ids": observation.get("face_person_ids", []),
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

    allowed_refs = _index_allowed_refs(allowed_refs, primary_person_id)

    supporting_refs = deepcopy(allowed_refs)
    if field_key == "long_term_facts.social_identity.education":
        supporting_refs["events"] = [
            ref
            for ref in allowed_refs["events"]
            if any(keyword in ref["signal"].lower() for keyword in ("学校", "校园", "class", "campus", "教室"))
            and primary_person_id in ref.get("participants", [])
        ]
        supporting_refs["vlm_observations"] = [
            ref
            for ref in allowed_refs["vlm_observations"]
            if any(keyword in ref["signal"].lower() for keyword in ("学校", "校园", "class", "campus", "教室"))
            and (
                primary_person_id in ref.get("people", [])
                or primary_person_id in ref.get("face_person_ids", [])
                or ref.get("subject_role") == "protagonist_view"
            )
        ]
    elif field_key == "long_term_facts.relationships.intimate_partner":
        supporting_refs["relationships"] = [
            ref for ref in allowed_refs["relationships"] if ref.get("relationship_type") == "romantic"
        ]
    elif field_key == "long_term_facts.relationships.social_groups":
        supporting_refs["group_artifacts"] = list(allowed_refs["group_artifacts"])
    elif field_key == "long_term_facts.material.brand_preference":
        material_candidate_bundle = _collect_material_candidates(
            field_key,
            allowed_refs["events"],
            allowed_refs["vlm_observations"],
        )
        supporting_refs["events"] = [deepcopy(ref) for ref in material_candidate_bundle["event_refs"]]
        supporting_refs["vlm_observations"] = [deepcopy(ref) for ref in material_candidate_bundle["vlm_refs"]]
        supporting_refs["feature_refs"] = []
        allowed_refs["feature_refs"] = []
    elif field_key in GEOGRAPHY_FOCUS_FIELDS:
        location_candidate_bundle = _collect_location_candidates(allowed_refs["events"], allowed_refs["vlm_observations"])
        supporting_refs["events"] = [deepcopy(ref) for ref in location_candidate_bundle["event_refs"]]
        supporting_refs["vlm_observations"] = [deepcopy(ref) for ref in location_candidate_bundle["vlm_refs"]]
        supporting_refs["feature_refs"] = []
        allowed_refs["feature_refs"] = []
    elif field_key == "short_term_facts.recent_interests":
        recent_topic_bundle = _collect_recent_topic_candidates(
            allowed_refs["events"],
            allowed_refs["vlm_observations"],
        )
        supporting_refs["events"] = [deepcopy(ref) for ref in recent_topic_bundle["event_refs"]]
        supporting_refs["vlm_observations"] = [deepcopy(ref) for ref in recent_topic_bundle["vlm_refs"]]
        supporting_refs["feature_refs"] = []
        allowed_refs["feature_refs"] = []

    supporting_refs = _focus_supporting_refs(
        field_key=field_key,
        supporting_refs=supporting_refs,
        primary_person_id=primary_person_id,
    )

    ref_index = _build_ref_index(allowed_refs)
    supporting_ids = extract_ids_from_refs(flatten_ref_buckets(supporting_refs))
    payload = {
        "field_key": field_key,
        "primary_person_id": primary_person_id,
        "allowed_refs": allowed_refs,
        "supporting_refs": supporting_refs,
        "ref_index": ref_index,
        "ids": supporting_ids,
        "source_coverage": {
            bucket: len(refs) for bucket, refs in allowed_refs.items()
        },
    }
    if field_key in MATERIAL_FOCUS_FIELDS:
        payload["material_candidate_bundle"] = _collect_material_candidates(
            field_key,
            supporting_refs["events"],
            supporting_refs["vlm_observations"],
        )
    if field_key == "long_term_facts.material.brand_preference":
        payload["brand_clue_bundle"] = payload.get("material_candidate_bundle", {})
    if field_key in GEOGRAPHY_FOCUS_FIELDS:
        payload["location_candidate_bundle"] = _collect_location_candidates(
            supporting_refs["events"],
            supporting_refs["vlm_observations"],
        )
    if field_key == "short_term_facts.recent_interests":
        payload["recent_topic_bundle"] = _collect_recent_topic_candidates(
            supporting_refs["events"],
            supporting_refs["vlm_observations"],
        )
    payload["compact"] = _build_compact_evidence_payload(payload)
    return payload


def analyze_evidence_stats(
    field_key: str,
    evidence_bundle: Dict[str, Any],
    ownership_bundle: Dict[str, Any] | None = None,
    profile_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    from .profile_fields import FIELD_SPECS

    spec = FIELD_SPECS[field_key]
    supporting_refs = evidence_bundle["supporting_refs"]
    supporting_events = supporting_refs.get("events", [])
    supporting_vlm = supporting_refs.get("vlm_observations", [])
    supporting_relationships = supporting_refs.get("relationships", [])
    supporting_groups = supporting_refs.get("group_artifacts", [])
    supporting_features = supporting_refs.get("feature_refs", [])

    distinct_event_ids = {
        ref.get("event_id")
        for ref in supporting_events
        if ref.get("event_id")
    }
    photo_count = len({ref.get("photo_id") for ref in supporting_vlm if ref.get("photo_id")})
    event_count = len(distinct_event_ids)
    support_count = (
        len(supporting_events)
        + len(supporting_vlm)
        + len(supporting_relationships)
        + len(supporting_groups)
        + len(supporting_features)
    )
    burst_score = _compute_burst_score(supporting_events, supporting_vlm)
    is_non_daily_burst = _is_non_daily_burst(supporting_events, supporting_vlm)
    cross_event_stability = 1.0 if event_count >= 3 else 0.66 if event_count == 2 else 0.33 if event_count == 1 else 0.0
    recent_window_count = event_count or photo_count
    window_count = len(
        {
            _window_key_from_ref(ref)
            for ref in list(supporting_events) + list(supporting_vlm)
            if _window_key_from_ref(ref)
        }
    )
    resolved_facts_summary = ((profile_state or {}).get("resolved_facts_summary")) or {}
    location_summary = _summarize_location_candidates(evidence_bundle.get("location_candidate_bundle", {})) if field_key in GEOGRAPHY_FOCUS_FIELDS else None
    brand_summary = _summarize_brand_distribution(
        evidence_bundle.get("brand_clue_bundle", {}),
        ownership_bundle=ownership_bundle,
    ) if field_key == "long_term_facts.material.brand_preference" else None
    recent_topic_summary = _summarize_recent_topics(
        evidence_bundle.get("recent_topic_bundle", {}),
        resolved_facts_summary,
    ) if field_key == "short_term_facts.recent_interests" else None
    work_signal_summary = _summarize_work_signals(supporting_events, supporting_vlm) if field_key in CAREER_FOCUS_FIELDS else None
    time_summary = _summarize_time_patterns(supporting_events, supporting_vlm) if field_key in TIME_FOCUS_FIELDS else None

    baseline_shift = 0.0
    if field_key.startswith("short_term_facts.") and resolved_facts_summary:
        baseline_shift = 1.0 if support_count > 0 else 0.0

    if field_key == "long_term_facts.social_identity.education":
        suggested_strong_evidence_met = (
            len(supporting_events) + len(supporting_vlm) >= 2
        )
    elif field_key == "long_term_facts.relationships.intimate_partner":
        suggested_strong_evidence_met = len(supporting_relationships) >= 1
    elif field_key == "long_term_facts.relationships.social_groups":
        suggested_strong_evidence_met = len(supporting_groups) >= 1
    elif field_key == "long_term_facts.relationships.close_circle_size":
        suggested_strong_evidence_met = True
    elif field_key in GEOGRAPHY_FOCUS_FIELDS:
        top_city_candidates = location_summary["top_city_candidates"]
        suggested_strong_evidence_met = any(
            item["primary_role"] not in {"travel_place", "others_place", "passby_place", "generic_place"}
            and item["event_count"] >= 1
            and item["window_count"] >= 1
            for item in top_city_candidates
        )
    elif field_key == "long_term_facts.hobbies.frequent_activities":
        suggested_strong_evidence_met = event_count >= 2
    elif field_key == "long_term_facts.material.brand_preference":
        top_brands = brand_summary["top_brands"]
        suggested_strong_evidence_met = any(
            item["event_count"] >= 2 and item["scene_count"] >= 2 and item["source_count"] >= 2
            for item in top_brands
        )
        if is_non_daily_burst:
            suggested_strong_evidence_met = False
    elif field_key == "short_term_facts.recent_interests":
        suggested_strong_evidence_met = any(
            item["event_count"] >= 2 and item["novelty_score"] > 0
            for item in recent_topic_summary["top_topics"]
        )
    elif field_key in CAREER_FOCUS_FIELDS:
        suggested_strong_evidence_met = work_signal_summary["total_signal_count"] >= 2
    elif field_key in TIME_FOCUS_FIELDS:
        suggested_strong_evidence_met = time_summary["distinct_months"] >= 2
    else:
        threshold = 2 if spec.risk_level == "P0" else 1
        suggested_strong_evidence_met = support_count >= threshold

    payload = {
        "field_key": field_key,
        "photo_count": photo_count,
        "event_count": event_count,
        "window_count": window_count,
        "distinct_time_windows": event_count or photo_count,
        "cross_event_stability": cross_event_stability,
        "recent_window_count": recent_window_count,
        "baseline_shift": baseline_shift,
        "burst_score": burst_score,
        "is_non_daily_burst": is_non_daily_burst,
        "support_count": support_count,
        "suggested_strong_evidence_met": suggested_strong_evidence_met,
    }
    if field_key == "long_term_facts.material.brand_preference":
        payload["brand_summary"] = brand_summary
    if field_key in GEOGRAPHY_FOCUS_FIELDS:
        payload["location_summary"] = location_summary
    if field_key == "short_term_facts.recent_interests":
        payload["recent_topic_summary"] = recent_topic_summary
    if field_key in CAREER_FOCUS_FIELDS:
        payload["work_signal_summary"] = work_signal_summary
    if field_key in TIME_FOCUS_FIELDS:
        payload["time_summary"] = time_summary
    return payload


def check_subject_ownership(field_key: str, evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
    supporting_refs = evidence_bundle["supporting_refs"]
    primary_person_id = evidence_bundle.get("primary_person_id")
    candidate_signals: List[Dict[str, Any]] = []

    if field_key in MATERIAL_FOCUS_FIELDS:
        material_candidates = evidence_bundle.get("material_candidate_bundle", {}) or {}
        candidate_signals = _summarize_material_candidate_ownership(material_candidates, primary_person_id)
        ownership_signal = _resolve_overall_ownership_signal(candidate_signals)
    else:
        ownership_signal = _resolve_ref_level_ownership_signal(
            flatten_ref_buckets(supporting_refs),
            primary_person_id,
        )

    return {
        "field_key": field_key,
        "ownership_signal": ownership_signal,
        "candidate_signals": candidate_signals[:8],
    }


def find_counter_evidence(
    field_key: str,
    evidence_bundle: Dict[str, Any],
    ownership_bundle: Dict[str, Any] | None = None,
    profile_state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    conflicts: List[str] = []
    contradicting_refs: List[Dict[str, Any]] = []
    allowed_refs = evidence_bundle["allowed_refs"]
    supporting_refs = evidence_bundle["supporting_refs"]

    if field_key in GEOGRAPHY_FOCUS_FIELDS:
        location_summary = _summarize_location_candidates(evidence_bundle.get("location_candidate_bundle", {}))
        if location_summary["generic_anchor_count"] > 0:
            conflicts.append("generic_place_only")
        if location_summary["travel_anchor_count"] > 0 and location_summary["named_anchor_count"] == 0:
            conflicts.append("travel_landmark_only")
        if any(item["primary_role"] == "others_place" for item in location_summary["top_city_candidates"]):
            conflicts.append("others_place_conflict")
        if location_summary["top_city_candidates"] and location_summary["top_city_candidates"][0]["window_count"] <= 1:
            conflicts.append("single_window_anchor")
        contradicting_refs.extend(
            ref
            for ref in flatten_ref_buckets(supporting_refs)
            if ref.get("candidate_role") in {"generic_place", "travel_place", "others_place", "passby_place"}
        )

    if field_key == "short_term_facts.recent_interests":
        recent_topic_summary = _summarize_recent_topics(
            evidence_bundle.get("recent_topic_bundle", {}),
            ((profile_state or {}).get("resolved_facts_summary")) or {},
        )
        if any(value > 0 for value in recent_topic_summary["overlap_with_long_term"].values()):
            conflicts.append("long_term_interest_bleed")
        if recent_topic_summary["top_topics"] and all(
            item["topic_origin"] == "visual_theme" for item in recent_topic_summary["top_topics"]
        ):
            conflicts.append("visual_theme_without_recent_shift")
        if evidence_bundle.get("recent_topic_bundle", {}).get("single_screenshot_only"):
            conflicts.append("single_screenshot_only")
        if evidence_bundle.get("recent_topic_bundle", {}).get("event_count", 0) <= 1:
            conflicts.append("single_trend_event_only")
        contradicting_refs.extend(
            ref
            for ref in flatten_ref_buckets(supporting_refs)
            if ref.get("topic_origin") == "visual_theme"
        )

    if field_key == "long_term_facts.social_identity.education":
        for ref in allowed_refs.get("vlm_observations", []):
            signal = ref.get("signal", "")
            if "路过" in signal or "walk" in signal.lower():
                contradicting_refs.append(ref)
                conflicts.append("temporary_campus_presence")

    if field_key.startswith("long_term_expression.") or field_key.startswith("short_term_expression."):
        resolved_facts_summary = ((profile_state or {}).get("resolved_facts_summary")) or {}
        long_term_facts = resolved_facts_summary.get("long_term_facts") or {}
        solo_vs_social_payload = (
            long_term_facts.get("hobbies.solo_vs_social")
            or ((long_term_facts.get("hobbies") or {}).get("solo_vs_social"))
            or {}
        )
        solo_vs_social = (solo_vs_social_payload or {}).get("value")
        event_texts = [
            " ".join(
                filter(
                    None,
                    [
                        ref.get("signal", ""),
                        ref.get("description", ""),
                        ref.get("narrative_synthesis", ""),
                    ],
                )
            )
            for ref in allowed_refs.get("events", [])
        ]
        event_texts.extend(ref.get("signal", "") for ref in allowed_refs.get("vlm_observations", []))

    if field_key.startswith("long_term_facts.") and _is_non_daily_burst(supporting_refs.get("events", []), supporting_refs.get("vlm_observations", [])):
        conflicts.append("non_daily_event_burst_interference")
        contradicting_refs.extend(supporting_refs.get("events", []))
        contradicting_refs.extend(supporting_refs.get("vlm_observations", []))

    if field_key in MATERIAL_FOCUS_FIELDS:
        candidate_signals = list((ownership_bundle or {}).get("candidate_signals", []) or [])
        if candidate_signals and all(item.get("signal") == "venue_context" for item in candidate_signals):
            conflicts.append("venue_only")
        if candidate_signals and all(item.get("signal") == "other_person" for item in candidate_signals):
            conflicts.append("other_person_only")
        if len(candidate_signals) <= 1 and len(flatten_ref_buckets(supporting_refs)) <= 1:
            conflicts.append("single_exposure_only")

    contradicting_refs = _dedupe_refs(contradicting_refs)
    return {
        "field_key": field_key,
        "contradicting_refs": contradicting_refs,
        "conflict_types": list(dict.fromkeys(conflicts)),
        "conflict_strength": len(conflicts),
        "conflict_summary": "；".join(dict.fromkeys(conflicts)) if conflicts else "",
        "contradicting_ids": _flatten_id_bundle(extract_ids_from_refs(contradicting_refs)),
    }


def get_resolved_facts(profile_state: Dict[str, Any]) -> Dict[str, Any]:
    structured = (profile_state or {}).get("structured_profile", {}) or {}
    long_term = structured.get("long_term_facts", {})
    short_term = structured.get("short_term_facts", {})
    summary = {
        "long_term_facts": _extract_non_null_tags(long_term),
        "short_term_facts": _extract_non_null_tags(short_term),
    }
    return {
        "resolved_facts_summary": summary,
        "resolved_key_facts": list(summary["long_term_facts"].keys()) + list(summary["short_term_facts"].keys()),
    }


def extract_metadata_evidence(context: Dict[str, Any]) -> Dict[str, Any]:
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
            refs.append(
                {
                    "source_type": "photo",
                    "source_id": observation.get("photo_id"),
                    "photo_id": observation.get("photo_id"),
                    "signal": signal,
                }
            )
            source_types.append("screenshot_or_ui")

    for event in context.get("events", []):
        signal = " ".join(
            filter(
                None,
                [
                    getattr(event, "title", ""),
                    getattr(event, "description", ""),
                    getattr(event, "narrative_synthesis", ""),
                ],
            )
        )
        if _contains_any_keyword(signal, SOCIAL_MEDIA_KEYWORDS):
            refs.append(
                {
                    "source_type": "event",
                    "source_id": getattr(event, "event_id", ""),
                    "event_id": getattr(event, "event_id", ""),
                    "signal": signal,
                }
            )
            source_types.append("event_text")

    return {
        "has_social_media_evidence": bool(refs),
        "source_types": list(dict.fromkeys(source_types)),
        "metadata_ids": _flatten_id_bundle(extract_ids_from_refs(refs)),
    }


def _collect_location_candidates(event_refs: List[Dict[str, Any]], vlm_refs: List[Dict[str, Any]]) -> Dict[str, Any]:
    clue_refs: List[Dict[str, Any]] = []
    event_refs_with_candidates: List[Dict[str, Any]] = []
    vlm_refs_with_candidates: List[Dict[str, Any]] = []

    for ref in event_refs:
        candidates = _extract_location_candidates_from_ref(ref, source_type="event")
        if not candidates:
            continue
        cloned = deepcopy(ref)
        cloned["location_candidates"] = candidates
        event_refs_with_candidates.append(cloned)
        clue_refs.extend(candidates)

    for ref in vlm_refs:
        candidates = _extract_location_candidates_from_ref(ref, source_type="photo")
        if not candidates:
            continue
        cloned = deepcopy(ref)
        cloned["location_candidates"] = candidates
        vlm_refs_with_candidates.append(cloned)
        clue_refs.extend(candidates)

    summary = _summarize_location_candidates({"clue_refs": clue_refs})
    return {
        "clue_refs": clue_refs,
        "event_refs": event_refs_with_candidates,
        "vlm_refs": vlm_refs_with_candidates,
        "top_candidates": summary["top_city_candidates"],
    }


def _extract_location_candidates_from_ref(ref: Dict[str, Any], source_type: str) -> List[Dict[str, Any]]:
    raw_candidates: List[str] = []
    raw_candidates.extend(str(item).strip() for item in ref.get("place_candidates", []) or [] if str(item).strip())
    raw_candidates.extend(filter(None, [ref.get("location", "")]))
    raw_candidates.extend(_extract_named_locations_from_text(ref.get("description", "")))
    raw_candidates.extend(_extract_named_locations_from_text(ref.get("narrative_synthesis", "")))
    raw_candidates.extend(_extract_named_locations_from_text(ref.get("signal", "")))
    for detail in ref.get("details", []) or []:
        raw_candidates.extend(_extract_named_locations_from_text(detail))
    for text in ref.get("ocr_hits", []) or []:
        raw_candidates.extend(_extract_named_locations_from_text(str(text)))

    deduped: List[Dict[str, Any]] = []
    seen = set()
    for raw_name in raw_candidates:
        canonical_name = _canonicalize_location_name(raw_name)
        if not canonical_name:
            continue
        if _looks_like_noise_location(canonical_name):
            continue
        candidate_role = _classify_location_candidate(canonical_name, ref)
        candidate = {
            "raw_name": raw_name,
            "canonical_name": canonical_name,
            "city_name": _city_name_from_location_candidate(canonical_name),
            "source_type": source_type,
            "source_id": ref.get("source_id"),
            "event_id": ref.get("event_id"),
            "photo_id": ref.get("photo_id"),
            "scene_label": _scene_label_from_ref(ref),
            "candidate_role": candidate_role,
            "is_named_place": candidate_role == "named_place",
            "window_key": _window_key_from_ref(ref),
        }
        dedupe_key = (
            canonical_name,
            candidate_role,
            candidate.get("event_id"),
            candidate.get("photo_id"),
            candidate.get("window_key"),
        )
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(candidate)
    return deduped


def _extract_named_locations_from_text(text: str) -> List[str]:
    """
    从文本中提取地点名

    新逻辑：优先使用基于已知地点列表的直接匹配，
    然后使用正则表达式作为补充
    """
    normalized = str(text or "").strip()
    if not normalized:
        return []

    # 使用新的关键词匹配器
    from .keyword_matcher import get_location_matcher
    matcher = get_location_matcher()
    locations = matcher.extract_locations(normalized)

    # 保留原来的正则表达式逻辑作为补充（发现新地点）
    candidates: List[str] = []
    chinese_patterns = [
        r"([\u4e00-\u9fff]{2,20}(?:社区|小区|公园|广场|花园|大厦|大学|学院|学校|商场))",
        r"([\u4e00-\u9fff]{2,12}(?:办公区|便利店|博物馆|餐厅|咖啡馆|宿舍|卧室|教室|校园))",
    ]
    english_patterns = [
        r"([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s(?:Community|Garden|Park|Plaza|Campus|Classroom|Office))",
        r"\b(dorm room|classroom|campus|office|museum|restaurant|cafe|convenience store)\b",
    ]
    for pattern in chinese_patterns + english_patterns:
        for match in re.finditer(pattern, normalized):
            candidate = match.group(1)
            if candidate.lower() not in {l.lower() for l in locations}:
                candidates.append(candidate)

    locations.extend(candidates)
    return list(dict.fromkeys(locations))


def _canonicalize_location_name(raw_name: str) -> str:
    cleaned = re.sub(r"[“”\"'`·•:：,，.。()（）]+", " ", str(raw_name or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _looks_like_noise_location(canonical_name: str) -> bool:
    normalized = str(canonical_name or "").strip().lower()
    if not normalized:
        return True
    noise_terms = (
        "智能手机屏幕",
        "移动设备屏幕",
        "数字空间",
        "wps office",
        "unknown",
        "未知",
    )
    noisy_prefixes = ("与同伴在", "带有", "疑似", "于")
    generic_tail_terms = ("公园", "博物馆", "景区", "展厅")
    return any(term.lower() in normalized for term in noise_terms) or (
        any(str(canonical_name or "").startswith(prefix) for prefix in noisy_prefixes)
        and any(tail in str(canonical_name or "") for tail in generic_tail_terms)
    )


def _classify_location_candidate(canonical_name: str, ref: Dict[str, Any]) -> str:
    lowered = canonical_name.lower()
    signal = " ".join(
        filter(
            None,
            [
                canonical_name,
                ref.get("description", ""),
                ref.get("narrative_synthesis", ""),
                ref.get("signal", ""),
            ],
        )
    ).lower()
    if _contains_any_keyword(signal, PASSBY_PLACE_KEYWORDS):
        return "passby_place"
    if _contains_any_keyword(signal, OTHERS_PLACE_KEYWORDS):
        return "others_place"
    if _looks_generic_place(canonical_name):
        return "generic_place"
    if _looks_specific_named_place(canonical_name):
        return "named_place"
    if _contains_any_keyword(signal, TRAVEL_PLACE_KEYWORDS):
        return "travel_place"
    if _contains_any_keyword(signal, SCHOOL_WORK_PLACE_KEYWORDS):
        return "school_work_place"
    return "named_place"


def _looks_generic_place(canonical_name: str) -> bool:
    lowered = canonical_name.lower()
    return canonical_name in GENERIC_PLACE_TERMS or lowered in GENERIC_PLACE_TERMS


def _looks_specific_named_place(canonical_name: str) -> bool:
    normalized = str(canonical_name or "").strip()
    if not normalized or _looks_generic_place(normalized):
        return False
    if re.search(r"(大学|学院|学校|公园|广场|花园|大厦|商场|社区|小区)$", normalized):
        return True
    if re.search(r"\b(University|College|School|Park|Plaza|Garden|Mall|Museum|Restaurant|Cafe)\b", normalized):
        return True
    return False


def _summarize_location_candidates(location_candidate_bundle: Dict[str, Any]) -> Dict[str, Any]:
    aggregate: Dict[str, Dict[str, Any]] = {}
    for clue_ref in location_candidate_bundle.get("clue_refs", []):
        city_name = clue_ref.get("city_name") or clue_ref.get("canonical_name")
        if not city_name:
            continue
        bucket = aggregate.setdefault(
            city_name,
            {
                "city_name": city_name,
                "candidate_role_counts": {},
                "event_ids": set(),
                "photo_ids": set(),
                "window_keys": set(),
                "scene_labels": set(),
                "example_places": set(),
                "clue_count": 0,
                "first_seen": None,
            },
        )
        role = clue_ref.get("candidate_role", "unknown")
        bucket["candidate_role_counts"][role] = bucket["candidate_role_counts"].get(role, 0) + 1
        if clue_ref.get("event_id"):
            bucket["event_ids"].add(clue_ref["event_id"])
        if clue_ref.get("photo_id"):
            bucket["photo_ids"].add(clue_ref["photo_id"])
        if clue_ref.get("window_key"):
            bucket["window_keys"].add(clue_ref["window_key"])
            first_seen = bucket.get("first_seen")
            window_key = clue_ref.get("window_key")
            if window_key and (first_seen is None or str(window_key) < str(first_seen)):
                bucket["first_seen"] = str(window_key)
        if clue_ref.get("scene_label"):
            bucket["scene_labels"].add(clue_ref["scene_label"])
        if clue_ref.get("canonical_name"):
            bucket["example_places"].add(clue_ref["canonical_name"])
        bucket["clue_count"] += 1

    top_city_candidates: List[Dict[str, Any]] = []
    named_anchor_count = 0
    generic_anchor_count = 0
    travel_anchor_count = 0
    for bucket in aggregate.values():
        primary_role = sorted(
            bucket["candidate_role_counts"].items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
        is_named_place = primary_role not in {"generic_place", "travel_place", "others_place", "passby_place"}
        candidate = {
            "city_name": bucket["city_name"],
            "primary_role": primary_role,
            "is_named_place": is_named_place,
            "city_hit_count": bucket["clue_count"],
            "event_count": len(bucket["event_ids"]),
            "photo_count": len(bucket["photo_ids"]),
            "window_count": len(bucket["window_keys"]),
            "scene_count": len(bucket["scene_labels"]),
            "example_places": sorted(bucket["example_places"])[:5],
            "role_breakdown": dict(sorted(bucket["candidate_role_counts"].items())),
            "event_ids": sorted(bucket["event_ids"]),
            "photo_ids": sorted(bucket["photo_ids"]),
            "first_seen": bucket["first_seen"] or "",
        }
        top_city_candidates.append(candidate)
        if candidate["is_named_place"]:
            named_anchor_count += 1
        if candidate["primary_role"] == "generic_place":
            generic_anchor_count += 1
        if candidate["primary_role"] == "travel_place":
            travel_anchor_count += 1

    top_city_candidates.sort(
        key=lambda item: (
            _location_role_priority(item["primary_role"]),
            -item["city_hit_count"],
            -item["event_count"],
            -item["window_count"],
            -item["scene_count"],
            item["first_seen"],
            item["city_name"],
        )
    )
    return {
        "top_city_candidates": top_city_candidates,
        "named_anchor_count": named_anchor_count,
        "generic_anchor_count": generic_anchor_count,
        "travel_anchor_count": travel_anchor_count,
        "window_count": max((item["window_count"] for item in top_city_candidates), default=0),
        "scene_count": max((item["scene_count"] for item in top_city_candidates), default=0),
        "cross_event_stability": 1.0 if any(item["event_count"] >= 3 for item in top_city_candidates) else 0.66 if any(item["event_count"] >= 2 for item in top_city_candidates) else 0.33 if top_city_candidates else 0.0,
    }


def _collect_recent_topic_candidates(event_refs: List[Dict[str, Any]], vlm_refs: List[Dict[str, Any]]) -> Dict[str, Any]:
    recent_threshold = _recent_threshold(event_refs, vlm_refs)
    clue_refs: List[Dict[str, Any]] = []
    event_refs_with_topics: List[Dict[str, Any]] = []
    vlm_refs_with_topics: List[Dict[str, Any]] = []
    screenshot_only = False

    for ref in event_refs:
        if not _is_within_recent_window(ref, recent_threshold):
            continue
        candidates = _extract_recent_topic_candidates_from_event_ref(ref)
        if not candidates:
            continue
        cloned = deepcopy(ref)
        cloned["recent_topic_candidates"] = candidates
        event_refs_with_topics.append(cloned)
        clue_refs.extend(candidates)

    for ref in vlm_refs:
        if not _is_within_recent_window(ref, recent_threshold):
            continue
        candidates = _extract_recent_topic_candidates_from_vlm_ref(ref)
        if not candidates:
            continue
        cloned = deepcopy(ref)
        cloned["recent_topic_candidates"] = candidates
        vlm_refs_with_topics.append(cloned)
        clue_refs.extend(candidates)
        if len(candidates) == 1 and candidates[0]["topic_origin"] == "visual_theme":
            screenshot_only = True

    return {
        "clue_refs": clue_refs,
        "event_refs": event_refs_with_topics,
        "vlm_refs": vlm_refs_with_topics,
        "single_screenshot_only": screenshot_only and not event_refs_with_topics,
        "event_count": len({ref.get("event_id") for ref in event_refs_with_topics if ref.get("event_id")}),
    }


def _extract_recent_topic_candidates_from_event_ref(ref: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for tag in ref.get("tags", []) or []:
        topic_name = _normalize_topic_name(tag)
        if topic_name:
            candidates.append(
                {
                    "topic_name": topic_name,
                    "topic_origin": "textual_topic",
                    "event_id": ref.get("event_id"),
                    "photo_id": ref.get("photo_id"),
                    "window_key": _window_key_from_ref(ref),
                }
            )
    for text in (ref.get("description", ""), ref.get("narrative_synthesis", ""), ref.get("signal", "")):
        for topic_name in _extract_topic_candidates_from_text(text):
            candidates.append(
                {
                    "topic_name": topic_name,
                    "topic_origin": "textual_topic",
                    "event_id": ref.get("event_id"),
                    "photo_id": ref.get("photo_id"),
                    "window_key": _window_key_from_ref(ref),
                }
            )
    return _dedupe_topic_refs(candidates)


def _extract_recent_topic_candidates_from_vlm_ref(ref: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for text in [ref.get("signal", ""), *list(ref.get("details", []) or []), *list(ref.get("ocr_hits", []) or [])]:
        for topic_name in _extract_topic_candidates_from_text(text):
            candidates.append(
                {
                    "topic_name": topic_name,
                    "topic_origin": "textual_topic",
                    "event_id": ref.get("event_id"),
                    "photo_id": ref.get("photo_id"),
                    "window_key": _window_key_from_ref(ref),
                }
            )
    visual_text = " ".join(str(item) for item in ref.get("details", []) or [])
    for topic_name in _extract_visual_theme_candidates(visual_text):
        candidates.append(
            {
                "topic_name": topic_name,
                "topic_origin": "visual_theme",
                "event_id": ref.get("event_id"),
                "photo_id": ref.get("photo_id"),
                "window_key": _window_key_from_ref(ref),
            }
        )
    return _dedupe_topic_refs(candidates)


def _extract_topic_candidates_from_text(text: str) -> List[str]:
    """
    从文本中提取主题/兴趣

    新逻辑：优先使用基于已知兴趣列表的直接匹配，
    然后使用词组提取作为补充
    """
    normalized = str(text or "").lower().replace("-", " ")
    normalized = normalized.replace("_", " ")
    if not normalized:
        return []

    # 使用新的关键词匹配器
    from .keyword_matcher import get_interest_matcher
    matcher = get_interest_matcher()
    interests = matcher.extract_interests(normalized)

    # 保留原来的词组提取逻辑作为补充
    candidates: List[str] = []
    for phrase in re.findall(r"[a-z]+(?:\s+[a-z]+){0,2}", normalized):
        words = [word for word in phrase.split() if word not in TOPIC_STOPWORDS]
        if len(words) < 2:
            continue
        candidate = _normalize_topic_name(" ".join(words))
        if candidate and candidate.lower() not in {i.lower() for i in interests}:
            candidates.append(candidate)

    interests.extend(candidates)
    return list(dict.fromkeys(interests))


def _extract_visual_theme_candidates(text: str) -> List[str]:
    """
    从文本中提取视觉主题

    新逻辑：使用基于已知主题列表的直接匹配
    """
    normalized = str(text or "").lower()
    if not normalized:
        return []

    # 已知的视觉主题列表
    known_visual_themes = [
        'fashion', 'ootd', 'outfit',
        'gaming', 'game',
        'artistic creation', 'art',
        'makeup', 'beauty',
        'photography', 'photo',
        'travel', 'landscape',
        'food', 'cuisine',
        'sports', 'fitness',
        'nature', 'outdoor',
    ]

    candidates: List[str] = []
    for theme in known_visual_themes:
        if theme in normalized:
            candidates.append(_normalize_topic_name(theme))

    return list(dict.fromkeys(candidates))


def _normalize_topic_name(raw_name: str) -> str:
    cleaned = re.sub(r"[“”\"'`·•:：,，.。()（）]+", " ", str(raw_name or "").lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    return cleaned.replace(" ", "_")


def _dedupe_topic_refs(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for candidate in candidates:
        key = (
            candidate.get("topic_name"),
            candidate.get("topic_origin"),
            candidate.get("event_id"),
            candidate.get("photo_id"),
            candidate.get("window_key"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _summarize_recent_topics(recent_topic_bundle: Dict[str, Any], resolved_facts_summary: Dict[str, Any]) -> Dict[str, Any]:
    baseline_topics = _extract_long_term_topic_baseline(resolved_facts_summary)
    aggregate: Dict[str, Dict[str, Any]] = {}
    for clue_ref in recent_topic_bundle.get("clue_refs", []):
        topic_name = clue_ref.get("topic_name")
        if not topic_name:
            continue
        bucket = aggregate.setdefault(
            topic_name,
            {
                "topic_name": topic_name,
                "topic_origin": clue_ref.get("topic_origin", "textual_topic"),
                "event_ids": set(),
                "photo_ids": set(),
                "window_keys": set(),
            },
        )
        if clue_ref.get("event_id"):
            bucket["event_ids"].add(clue_ref["event_id"])
        if clue_ref.get("photo_id"):
            bucket["photo_ids"].add(clue_ref["photo_id"])
        if clue_ref.get("window_key"):
            bucket["window_keys"].add(clue_ref["window_key"])

    overlap_with_long_term: Dict[str, int] = {}
    top_topics: List[Dict[str, Any]] = []
    for bucket in aggregate.values():
        overlap = 1 if bucket["topic_name"] in baseline_topics else 0
        overlap_with_long_term[bucket["topic_name"]] = overlap
        novelty_score = 0.0 if overlap else min(len(bucket["event_ids"]) / 2, 1.0)
        candidate = {
            "topic_name": bucket["topic_name"],
            "topic_origin": bucket["topic_origin"],
            "event_count": len(bucket["event_ids"]),
            "photo_count": len(bucket["photo_ids"]),
            "window_count": len(bucket["window_keys"]),
            "novelty_score": novelty_score,
        }
        top_topics.append(candidate)

    top_topics.sort(
        key=lambda item: (
            0 if item["topic_origin"] in {"textual_topic", "activity_topic"} else 1,
            -item["event_count"],
            -item["window_count"],
            -item["novelty_score"],
            item["topic_name"],
        )
    )
    return {
        "top_topics": top_topics[:5],
        "recent_topic_frequency": {
            item["topic_name"]: item["event_count"] for item in top_topics
        },
        "recent_topic_event_count": {
            item["topic_name"]: item["event_count"] for item in top_topics
        },
        "overlap_with_long_term": overlap_with_long_term,
        "novelty_score": max((item["novelty_score"] for item in top_topics), default=0.0),
        "life_event_support": any("life_events" in key or "phase_change" in key for key in (resolved_facts_summary.get("short_term_facts") or {}).keys()),
    }


def _extract_long_term_topic_baseline(resolved_facts_summary: Dict[str, Any]) -> set[str]:
    baseline: set[str] = set()
    long_term = (resolved_facts_summary or {}).get("long_term_facts") or {}
    hobbies_payload = long_term.get("hobbies.interests") or ((long_term.get("hobbies") or {}).get("interests")) or {}
    frequent_payload = long_term.get("hobbies.frequent_activities") or ((long_term.get("hobbies") or {}).get("frequent_activities")) or {}
    for item in (hobbies_payload or {}).get("value", []) or []:
        if isinstance(item, dict):
            name = item.get("name")
        else:
            name = item
        normalized = _normalize_topic_name(name)
        if normalized:
            baseline.add(normalized)
    for item in (frequent_payload or {}).get("value", []) or []:
        normalized = _normalize_topic_name(item)
        if normalized:
            baseline.add(normalized)
    return baseline


def _recent_threshold(event_refs: List[Dict[str, Any]], vlm_refs: List[Dict[str, Any]]) -> datetime | None:
    timestamps = [_parse_ref_datetime(ref) for ref in list(event_refs) + list(vlm_refs)]
    timestamps = [item for item in timestamps if item is not None]
    if not timestamps:
        return None
    latest = max(timestamps)
    return latest - timedelta(days=30)


def _is_within_recent_window(ref: Dict[str, Any], threshold: datetime | None) -> bool:
    if threshold is None:
        return True
    ref_dt = _parse_ref_datetime(ref)
    if ref_dt is None:
        return True
    return ref_dt >= threshold


def _parse_ref_datetime(ref: Dict[str, Any]) -> datetime | None:
    raw = ref.get("timestamp") or ref.get("date")
    if not raw:
        return None
    try:
        if "T" in raw:
            return datetime.fromisoformat(raw)
        return datetime.fromisoformat(f"{raw}T00:00:00")
    except ValueError:
        return None


def _window_key_from_ref(ref: Dict[str, Any]) -> str:
    timestamp = ref.get("timestamp")
    if timestamp:
        return str(timestamp)[:10]
    date = ref.get("date") or ref.get("event_id") or ref.get("photo_id") or ""
    return str(date)


def _collect_brand_clues(event_refs: List[Dict[str, Any]], vlm_refs: List[Dict[str, Any]]) -> Dict[str, Any]:
    clue_refs: List[Dict[str, Any]] = []
    event_refs_with_brands: List[Dict[str, Any]] = []
    vlm_refs_with_brands: List[Dict[str, Any]] = []
    rejected_candidates: List[str] = []

    for ref in event_refs:
        brand_clues, rejected = _extract_brand_clues_from_ref(ref)
        rejected_candidates.extend(rejected)
        if not brand_clues:
            continue
        cloned = deepcopy(ref)
        cloned["brand_clues"] = brand_clues
        event_refs_with_brands.append(cloned)
        clue_refs.extend(
            {
                "brand_name": clue["brand_name"],
                "source_type": "event",
                "source_id": cloned.get("source_id"),
                "event_id": cloned.get("event_id"),
                "photo_id": cloned.get("photo_id"),
                "scene_label": _scene_label_from_ref(cloned),
                "signal": clue["signal"],
                "location": cloned.get("location", ""),
                "place_candidates": list(cloned.get("place_candidates", []) or []),
                "participants": list(cloned.get("participants", []) or []),
                "subject_role": cloned.get("subject_role", ""),
                "people": list(cloned.get("people", []) or []),
                "face_person_ids": list(cloned.get("face_person_ids", []) or []),
            }
            for clue in brand_clues
        )

    for ref in vlm_refs:
        brand_clues, rejected = _extract_brand_clues_from_ref(ref)
        rejected_candidates.extend(rejected)
        if not brand_clues:
            continue
        cloned = deepcopy(ref)
        cloned["brand_clues"] = brand_clues
        vlm_refs_with_brands.append(cloned)
        clue_refs.extend(
            {
                "brand_name": clue["brand_name"],
                "source_type": "photo",
                "source_id": cloned.get("source_id"),
                "photo_id": cloned.get("photo_id"),
                "scene_label": _scene_label_from_ref(cloned),
                "signal": clue["signal"],
                "location": cloned.get("location", ""),
                "place_candidates": list(cloned.get("place_candidates", []) or []),
                "participants": list(cloned.get("participants", []) or []),
                "subject_role": cloned.get("subject_role", ""),
                "people": list(cloned.get("people", []) or []),
                "face_person_ids": list(cloned.get("face_person_ids", []) or []),
            }
            for clue in brand_clues
        )

    return {
        "clue_refs": clue_refs,
        "event_refs": event_refs_with_brands,
        "vlm_refs": vlm_refs_with_brands,
        "rejected_candidates": list(dict.fromkeys(candidate for candidate in rejected_candidates if candidate)),
    }


def _extract_brand_clues_from_ref(ref: Dict[str, Any]) -> tuple[List[Dict[str, Any]], List[str]]:
    texts: List[str] = []
    rejected: List[str] = []
    collected: List[Dict[str, Any]] = []

    for brand in ref.get("brands", []) or []:
        normalized_brand = _clean_brand_candidate(str(brand))
        if not normalized_brand:
            continue
        collected.append(
            {
                "brand_name": normalized_brand,
                "signal": f"structured_brand:{normalized_brand}",
            }
        )

    texts.extend(
        filter(
            None,
            [
                ref.get("signal", ""),
                ref.get("description", ""),
                ref.get("narrative_synthesis", ""),
                ref.get("location", ""),
                ref.get("activity", ""),
            ],
        )
    )
    texts.extend(str(item) for item in ref.get("details", []) or [])
    texts.extend(str(item) for item in ref.get("ocr_hits", []) or [])
    texts.extend(str(item) for item in ref.get("tags", []) or [])
    persona_evidence = ref.get("persona_evidence", {}) or {}
    for bucket in ("behavioral", "aesthetic", "socioeconomic"):
        texts.extend(str(item) for item in persona_evidence.get(bucket, []) or [])

    for text in texts:
        candidates, rejected_candidates = _extract_brand_candidates_from_text(text)
        rejected.extend(rejected_candidates)
        for candidate in candidates:
            collected.append(
                {
                    "brand_name": candidate,
                    "signal": text,
                }
            )
    deduped: List[Dict[str, Any]] = []
    seen: Dict[str, Dict[str, Any]] = {}
    for item in collected:
        key = item["brand_name"]
        previous = seen.get(key)
        if previous is None:
            seen[key] = item
            continue
        previous_signal = str(previous.get("signal", "") or "")
        current_signal = str(item.get("signal", "") or "")
        if previous_signal.startswith("structured_brand:") and not current_signal.startswith("structured_brand:"):
            seen[key] = item
        elif len(current_signal) > len(previous_signal):
            seen[key] = item
    deduped.extend(seen.values())
    return deduped, rejected


def _extract_brand_candidates_from_text(text: str) -> tuple[List[str], List[str]]:
    """
    从文本中提取品牌名

    新逻辑：首先使用基于已知品牌列表的直接匹配（高精度），
    然后使用正则表达式模式匹配作为补充（发现新品牌）
    """
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return [], []

    # 使用新的关键词匹配器
    from .keyword_matcher import get_brand_matcher
    matcher = get_brand_matcher()
    candidates, rejected = matcher.extract_brands(normalized)
    filtered_candidates: List[str] = []
    filtered_rejected = list(rejected)
    for candidate in _prune_overlapping_brand_candidates(
        [_clean_brand_candidate(candidate) for candidate in candidates]
    ):
        if not candidate:
            continue
        if _looks_like_non_brand(candidate, normalized):
            filtered_rejected.append(candidate)
            continue
        filtered_candidates.append(candidate)

    return list(dict.fromkeys(filtered_candidates)), list(dict.fromkeys(filtered_rejected))


def _clean_brand_candidate(candidate: str) -> str:
    cleaned = re.sub(r"[“”\"'`·•:：,，.。()（）]+", " ", str(candidate or ""))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _prune_overlapping_brand_candidates(candidates: List[str]) -> List[str]:
    ordered = sorted(list(dict.fromkeys(candidates)), key=len, reverse=True)
    kept: List[str] = []
    for candidate in ordered:
        normalized = candidate.lower()
        if any(normalized != kept_item.lower() and normalized in kept_item.lower() for kept_item in kept):
            continue
        kept.append(candidate)
    return list(reversed(kept))


def _looks_like_non_brand(candidate: str, source_text: str) -> bool:
    normalized_candidate = candidate.strip().lower()
    if not normalized_candidate or normalized_candidate in BRAND_NOISE_TERMS:
        return True
    if normalized_candidate in {keyword.lower() for keyword in BRAND_PRODUCT_KEYWORDS}:
        return True
    if any(keyword in normalized_candidate for keyword in NON_BRAND_SIGNAL_KEYWORDS):
        return True
    if normalized_candidate.isdigit():
        return True
    if len(candidate) == 1:
        return True
    if " " not in candidate and candidate.islower():
        return True
    if _contains_any_keyword(source_text, NON_BRAND_SIGNAL_KEYWORDS) and not _contains_any_keyword(source_text, BRAND_PRODUCT_KEYWORDS):
        return True
    return False


def _scene_label_from_ref(ref: Dict[str, Any]) -> str:
    return (
        ref.get("location")
        or ref.get("type")
        or ref.get("activity")
        or ref.get("event_id")
        or ref.get("photo_id")
        or "unknown"
    )


def _summarize_brand_distribution(
    brand_clue_bundle: Dict[str, Any],
    ownership_bundle: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    aggregate: Dict[str, Dict[str, Any]] = {}
    ownership_map = _candidate_signal_map(ownership_bundle)
    for clue_ref in brand_clue_bundle.get("clue_refs", []):
        brand_name = clue_ref.get("brand_name")
        if not brand_name:
            continue
        bucket = aggregate.setdefault(
            brand_name,
            {
                "brand_name": brand_name,
                "source_count": 0,
                "event_ids": set(),
                "photo_ids": set(),
                "scenes": set(),
                "ownership_signal": ownership_map.get(brand_name, "background_or_ambiguous"),
            },
        )
        bucket["source_count"] += 1
        if clue_ref.get("event_id"):
            bucket["event_ids"].add(clue_ref["event_id"])
        if clue_ref.get("photo_id"):
            bucket["photo_ids"].add(clue_ref["photo_id"])
        if clue_ref.get("scene_label"):
            bucket["scenes"].add(clue_ref["scene_label"])

    top_brands: List[Dict[str, Any]] = []
    suppressed_candidates: List[Dict[str, Any]] = []
    for bucket in aggregate.values():
        item = {
            "brand_name": bucket["brand_name"],
            "ownership_signal": bucket["ownership_signal"],
            "source_count": bucket["source_count"],
            "event_count": len(bucket["event_ids"]),
            "photo_count": len(bucket["photo_ids"]),
            "scene_count": len(bucket["scenes"]),
            "scenes": sorted(bucket["scenes"])[:5],
        }
        if item["ownership_signal"] in {"owned_or_used", "worn", "background_or_ambiguous"}:
            top_brands.append(item)
        else:
            suppressed_candidates.append(item)
    top_brands.sort(key=lambda item: (-item["event_count"], -item["scene_count"], -item["source_count"], item["brand_name"]))
    return {
        "top_brands": top_brands,
        "suppressed_candidates": suppressed_candidates,
        "rejected_candidates": list(brand_clue_bundle.get("rejected_candidates", [])),
    }


def _collect_material_candidates(
    field_key: str,
    event_refs: List[Dict[str, Any]],
    vlm_refs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if field_key in MATERIAL_FOCUS_FIELDS:
        return _collect_brand_clues(event_refs, vlm_refs)
    return {"clue_refs": [], "event_refs": [], "vlm_refs": [], "rejected_candidates": []}


def _build_compact_evidence_payload(evidence_bundle: Dict[str, Any]) -> Dict[str, Any]:
    field_key = evidence_bundle.get("field_key", "")
    supporting_refs = evidence_bundle.get("supporting_refs", {}) or {}
    flat_refs = flatten_ref_buckets(supporting_refs)
    evidence_ids = extract_ids_from_refs(flat_refs)
    top_candidates: List[Dict[str, Any]] = []

    if field_key in GEOGRAPHY_FOCUS_FIELDS:
        top_candidates = _summarize_location_candidates(
            evidence_bundle.get("location_candidate_bundle", {})
        ).get("top_city_candidates", [])[:5]
    elif field_key == "long_term_facts.material.brand_preference":
        top_candidates = [
            {
                "candidate": item.get("brand_name"),
                "event_count": item.get("event_count", 0),
                "scene_count": item.get("scene_count", 0),
            }
            for item in _summarize_brand_distribution(
                evidence_bundle.get("brand_clue_bundle", {})
            ).get("top_brands", [])[:5]
            if item.get("brand_name")
        ]
    elif field_key == "short_term_facts.recent_interests":
        top_candidates = _summarize_recent_topics(
            evidence_bundle.get("recent_topic_bundle", {}),
            {},
        ).get("top_topics", [])[:5]

    summary = _build_compact_summary(field_key, top_candidates, supporting_refs)
    return {
        "schema_version": "lp3_tool_compact_v1",
        "tool_name": "fetch_field_evidence",
        "field_key": field_key,
        "summary": summary,
        "source_coverage": dict(evidence_bundle.get("source_coverage", {}) or {}),
        "top_candidates": top_candidates,
        "evidence_ids": evidence_ids,
        "representative_events": _representative_event_summaries(supporting_refs.get("events", [])),
        "representative_photos": _representative_photo_summaries(supporting_refs.get("vlm_observations", [])),
    }


def _build_compact_summary(
    field_key: str,
    top_candidates: List[Dict[str, Any]],
    supporting_refs: Dict[str, List[Dict[str, Any]]],
) -> str:
    if field_key in GEOGRAPHY_FOCUS_FIELDS and top_candidates:
        city_names = [item.get("city_name") for item in top_candidates if item.get("city_name")]
        return f"城市级地点候选集中在: {', '.join(city_names[:3])}"
    if field_key == "long_term_facts.material.brand_preference" and top_candidates:
        names = [item.get("candidate") for item in top_candidates if item.get("candidate")]
        return f"品牌候选集中在: {', '.join(names[:3])}"
    if field_key == "short_term_facts.recent_interests" and top_candidates:
        names = [item.get("topic_name") for item in top_candidates if item.get("topic_name")]
        return f"近期主题候选集中在: {', '.join(names[:3])}"
    support_count = len(flatten_ref_buckets(supporting_refs))
    return f"已收敛 {support_count} 条高相关证据。"


def _representative_event_summaries(event_refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for ref in event_refs[:10]:
        event_id = str(ref.get("event_id", "") or "")
        if not event_id or event_id in seen:
            continue
        seen.add(event_id)
        entry: Dict[str, Any] = {
            "event_id": event_id,
            "summary": str(
                ref.get("signal")
                or ref.get("description")
                or ref.get("narrative_synthesis")
                or event_id
            ).strip(),
        }
        description = str(ref.get("description", "") or "").strip()
        if description:
            entry["description"] = description
        persona_evidence = ref.get("persona_evidence")
        if isinstance(persona_evidence, dict) and any(persona_evidence.values()):
            entry["persona_evidence"] = persona_evidence
        summaries.append(entry)
    return summaries


def _representative_photo_summaries(vlm_refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for ref in vlm_refs[:10]:
        photo_id = str(ref.get("photo_id", "") or "")
        if not photo_id or photo_id in seen:
            continue
        seen.add(photo_id)
        summaries.append(
            {
                "photo_id": photo_id,
                "summary": str(
                    ref.get("signal")
                    or ref.get("location")
                    or ref.get("activity")
                    or photo_id
                ).strip(),
            }
        )
    return summaries


def _candidate_signal_map(ownership_bundle: Dict[str, Any] | None) -> Dict[str, str]:
    signal_map: Dict[str, str] = {}
    for item in list((ownership_bundle or {}).get("candidate_signals", []) or []):
        candidate = str(item.get("candidate", "") or "").strip()
        signal = str(item.get("signal", "") or "").strip()
        if candidate and signal:
            signal_map[candidate] = signal
    return signal_map


def _summarize_material_candidate_ownership(
    material_candidate_bundle: Dict[str, Any],
    primary_person_id: str | None,
) -> List[Dict[str, Any]]:
    aggregate: Dict[str, List[Dict[str, Any]]] = {}
    for clue_ref in material_candidate_bundle.get("clue_refs", []):
        candidate = str(clue_ref.get("brand_name", "") or "").strip()
        if not candidate:
            continue
        aggregate.setdefault(candidate, []).append(clue_ref)

    candidate_signals: List[Dict[str, Any]] = []
    for candidate, refs in aggregate.items():
        candidate_signals.append(
            {
                "candidate": candidate,
                "signal": _resolve_candidate_ownership_signal(candidate, refs, primary_person_id),
            }
        )
    candidate_signals.sort(
        key=lambda item: (
            OWNERSHIP_SIGNAL_ORDER.get(item.get("signal", "background_or_ambiguous"), 99),
            item.get("candidate", ""),
        )
    )
    return candidate_signals


def _resolve_candidate_ownership_signal(
    candidate: str,
    refs: List[Dict[str, Any]],
    primary_person_id: str | None,
) -> str:
    if refs and all(_is_venue_context_ref(candidate, ref) for ref in refs):
        return "venue_context"
    if refs and all(_is_other_person_ref(ref, primary_person_id) for ref in refs):
        return "other_person"

    primary_hits = 0
    worn_hits = 0
    venue_hits = 0
    other_person_hits = 0
    for ref in refs:
        if _is_venue_context_ref(candidate, ref):
            venue_hits += 1
        if _is_other_person_ref(ref, primary_person_id):
            other_person_hits += 1
        if _ref_points_to_primary(ref, primary_person_id):
            primary_hits += 1
            if _looks_like_worn_signal(ref):
                worn_hits += 1

    if primary_hits > 0:
        return "worn" if worn_hits > 0 else "owned_or_used"
    if venue_hits > 0:
        return "venue_context"
    if other_person_hits > 0:
        return "other_person"
    return "background_or_ambiguous"


def _resolve_overall_ownership_signal(candidate_signals: List[Dict[str, Any]]) -> str:
    if any(item.get("signal") == "owned_or_used" for item in candidate_signals):
        return "owned_or_used"
    if any(item.get("signal") == "worn" for item in candidate_signals):
        return "worn"
    if any(item.get("signal") == "venue_context" for item in candidate_signals):
        return "venue_context"
    if any(item.get("signal") == "other_person" for item in candidate_signals):
        return "other_person"
    return "background_or_ambiguous"


def _resolve_ref_level_ownership_signal(refs: List[Dict[str, Any]], primary_person_id: str | None) -> str:
    if any(_ref_points_to_primary(ref, primary_person_id) for ref in refs):
        return "owned_or_used"
    if any(_is_other_person_ref(ref, primary_person_id) for ref in refs):
        return "other_person"
    return "background_or_ambiguous"


def _ref_points_to_primary(ref: Dict[str, Any], primary_person_id: str | None) -> bool:
    return _primary_binding_score(ref, primary_person_id) > 0


def _is_other_person_ref(ref: Dict[str, Any], primary_person_id: str | None) -> bool:
    if ref.get("subject_role") == "other_people_only":
        return True
    primary = str(primary_person_id or "").strip()
    participants = list(ref.get("participants", []) or [])
    people = list(ref.get("people", []) or [])
    face_person_ids = list(ref.get("face_person_ids", []) or [])
    candidate_people = participants + people + face_person_ids
    candidate_people = [str(item) for item in candidate_people if str(item or "").strip()]
    if not candidate_people:
        return False
    if primary and primary in candidate_people:
        return False
    if "主角" in candidate_people:
        return False
    return True


def _is_venue_context_ref(candidate: str, ref: Dict[str, Any]) -> bool:
    normalized_candidate = str(candidate or "").lower()
    if not normalized_candidate:
        return False
    location_texts = [str(ref.get("location", "") or "")]
    location_texts.extend(str(item) for item in ref.get("place_candidates", []) or [])
    combined_location = " ".join(location_texts).lower()
    if normalized_candidate and normalized_candidate in combined_location:
        return True
    venue_keywords = ("coffee", "cafe", "酒店", "hotel", "mall", "商场", "museum", "博物馆", "resort")
    return any(keyword in normalized_candidate for keyword in venue_keywords)


def _looks_like_worn_signal(ref: Dict[str, Any]) -> bool:
    text = " ".join(
        filter(
            None,
            [
                ref.get("signal", ""),
                ref.get("location", ""),
                ref.get("scene_label", ""),
            ],
        )
    )
    return _contains_any_keyword(
        text,
        ("外套", "卫衣", "hoodie", "shirt", "鞋", "shoe", "bag", "包", "coat", "wear"),
    )


def _flatten_id_bundle(id_bundle: Dict[str, List[str]]) -> List[str]:
    ordered: List[str] = []
    for key in ("event_ids", "photo_ids", "person_ids", "group_ids", "feature_names"):
        for item in id_bundle.get(key, []) or []:
            if item not in ordered:
                ordered.append(item)
    return ordered


def _city_name_from_location_candidate(canonical_name: str) -> str:
    normalized = str(canonical_name or "").strip()
    if not normalized:
        return ""
    city_aliases = {
        "北京": ("北京", "故宫", "天安门", "长城", "国博", "国家博物馆", "清河站", "圆明园站", "雍和宫", "颐和园", "清华大学", "地坛公园"),
        "深圳": ("深圳", "文华东方", "深业上城", "星河双子塔", "龙岗门店"),
        "陆丰": ("陆丰", "泓锋客运站", "陆丰市公安局"),
        "玉环": ("玉环",),
    }
    for city_name, aliases in city_aliases.items():
        if any(alias in normalized for alias in aliases):
            return city_name
    return normalized


def _location_city_keys_from_ref(ref: Dict[str, Any]) -> List[str]:
    city_keys: List[str] = []
    for candidate in ref.get("location_candidates", []) or []:
        city_name = _city_name_from_location_candidate(candidate.get("canonical_name", ""))
        if city_name and city_name not in city_keys:
            city_keys.append(city_name)
    if city_keys:
        return city_keys
    raw_values: List[str] = []
    raw_values.extend(str(item) for item in ref.get("place_candidates", []) or [])
    raw_values.append(str(ref.get("location", "") or ""))
    for raw_value in raw_values:
        city_name = _city_name_from_location_candidate(raw_value)
        if city_name and city_name not in city_keys:
            city_keys.append(city_name)
    return city_keys


def _location_role_priority(role: str) -> int:
    if role in {"named_place", "school_work_place"}:
        return 0
    if role == "generic_place":
        return 2
    if role in {"others_place", "travel_place", "passby_place"}:
        return 3
    return 1


def _location_bucket_priority(ref: Dict[str, Any]) -> int:
    roles = [candidate.get("candidate_role") for candidate in ref.get("location_candidates", []) or []]
    if not roles:
        return 1
    return min(_location_role_priority(str(role or "")) for role in roles)


def _summarize_time_patterns(
    supporting_events: List[Dict[str, Any]],
    supporting_vlm: List[Dict[str, Any]],
) -> Dict[str, Any]:
    timestamps = [
        _parse_ref_datetime(ref)
        for ref in list(supporting_events) + list(supporting_vlm)
    ]
    timestamps = [item for item in timestamps if item is not None]
    month_histogram: Dict[str, int] = {}
    for item in timestamps:
        month_key = item.strftime("%Y-%m")
        month_histogram[month_key] = month_histogram.get(month_key, 0) + 1
    span_days = 0
    if timestamps:
        span_days = (max(timestamps) - min(timestamps)).days

    pattern_counts: Dict[str, int] = {}
    for ref in supporting_events:
        for candidate in (ref.get("type"), ref.get("location")):
            name = str(candidate or "").strip()
            if not name:
                continue
            pattern_counts[name] = pattern_counts.get(name, 0) + 1

    recurring_pattern_candidates = [
        {"name": key, "count": count}
        for key, count in sorted(pattern_counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= 2
    ][:5]
    return {
        "month_histogram": dict(sorted(month_histogram.items())),
        "distinct_months": len(month_histogram),
        "span_days": span_days,
        "recurring_pattern_candidates": recurring_pattern_candidates,
        "evidence_event_ids": sorted(
            {
                ref.get("event_id")
                for ref in supporting_events
                if ref.get("event_id")
            }
        ),
    }


def _summarize_work_signals(
    supporting_events: List[Dict[str, Any]],
    supporting_vlm: List[Dict[str, Any]],
) -> Dict[str, Any]:
    signal_keywords = {
        "payroll_sheet": ("工资", "薪资", "salary", "payroll"),
        "attendance_sheet": ("考勤", "attendance", "签到", "打卡记录"),
        "punch_in_scene": ("打卡", "clock in", "punch"),
        "operation_scene": ("巡查", "值班", "现场", "工位", "办公室", "office", "工作", "meeting", "公安局"),
        "coworker_cluster": ("同事", "coworker", "team", "同伴"),
    }
    signal_counts = {key: 0 for key in signal_keywords}
    signal_ids: Dict[str, List[str]] = {key: [] for key in signal_keywords}
    for ref in list(supporting_events) + list(supporting_vlm):
        text = _ref_focus_text(ref)
        ref_ids = _candidate_ref_ids(ref)
        for signal_name, keywords in signal_keywords.items():
            if _contains_any_keyword(text, keywords):
                signal_counts[signal_name] += 1
                for ref_id in ref_ids:
                    if ref_id not in signal_ids[signal_name]:
                        signal_ids[signal_name].append(ref_id)
    return {
        **signal_counts,
        "signal_ids": signal_ids,
        "total_signal_count": sum(signal_counts.values()),
    }


def _build_ref_index(allowed_refs: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    ref_index: Dict[str, Dict[str, Any]] = {}
    for ref in flatten_ref_buckets(allowed_refs):
        for ref_id in _candidate_ref_ids(ref):
            ref_index[ref_id] = ref
    return ref_index


def _candidate_ref_ids(ref: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    if ref.get("event_id"):
        ids.append(str(ref["event_id"]))
    if ref.get("photo_id"):
        ids.append(str(ref["photo_id"]))
    if ref.get("person_id"):
        ids.append(str(ref["person_id"]))
    if ref.get("group_id"):
        ids.append(str(ref["group_id"]))
    if ref.get("feature_name"):
        ids.append(str(ref["feature_name"]))
    ids.extend(str(item) for item in ref.get("event_ids", []) or [])
    ids.extend(str(item) for item in ref.get("photo_ids", []) or [])
    ids.extend(str(item) for item in ref.get("person_ids", []) or [])
    ids.extend(str(item) for item in ref.get("group_ids", []) or [])
    ids.extend(str(item) for item in ref.get("feature_names", []) or [])
    return list(dict.fromkeys(ids))


def _compute_burst_score(supporting_events: Iterable[Dict[str, Any]], supporting_vlm: Iterable[Dict[str, Any]]) -> float:
    unique_event_ids = {ref.get("event_id") for ref in supporting_events if ref.get("event_id")}
    if not unique_event_ids and not list(supporting_vlm):
        return 0.0
    if len(unique_event_ids) <= 1 and len(list(supporting_vlm)) >= 3:
        return 1.0
    if len(unique_event_ids) <= 1 and any((ref.get("photo_count") or 0) >= 3 for ref in supporting_events):
        return 0.9
    if len(unique_event_ids) == 2:
        return 0.5
    return 0.1


def _is_non_daily_burst(supporting_events: Iterable[Dict[str, Any]], supporting_vlm: Iterable[Dict[str, Any]]) -> bool:
    unique_event_ids = {ref.get("event_id") for ref in supporting_events if ref.get("event_id")}
    concentrated_single_event = len(unique_event_ids) <= 1 and (
        len(list(supporting_vlm)) >= 3 or any((ref.get("photo_count") or 0) >= 3 for ref in supporting_events)
    )
    non_daily_signal = any(_is_non_daily_event_ref(ref) for ref in supporting_events) or any(
        _contains_any_keyword(ref.get("signal", ""), NON_DAILY_EVENT_KEYWORDS)
        for ref in supporting_vlm
    )
    return concentrated_single_event and non_daily_signal


def _is_non_daily_event_ref(ref: Dict[str, Any]) -> bool:
    return _contains_any_keyword(
        " ".join(
            filter(
                None,
                [
                    ref.get("signal", ""),
                    ref.get("type", ""),
                    ref.get("description", ""),
                    ref.get("narrative_synthesis", ""),
                ],
            )
        ),
        NON_DAILY_EVENT_KEYWORDS,
    )


def _contains_any_keyword(text: str, keywords: Iterable[str]) -> bool:
    normalized = str(text or "").lower()
    return any(keyword.lower() in normalized for keyword in keywords)


def _is_meaningful_feature_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple, set, dict, str)):
        return bool(value)
    if isinstance(value, (int, float)):
        return value > 0
    return True


def _extract_non_null_tags(payload: Any, prefix: str = "") -> Dict[str, Any]:
    collected: Dict[str, Any] = {}
    if not isinstance(payload, dict):
        return collected
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and {"value", "confidence", "evidence", "reasoning"} <= set(value.keys()):
            if value.get("value") is not None:
                collected[path] = value
        elif isinstance(value, dict):
            collected.update(_extract_non_null_tags(value, path))
    return collected


def _dedupe_refs(refs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for ref in refs:
        key = "|".join(_candidate_ref_ids(ref)) or str(ref)
        grouped[key] = ref
    return list(grouped.values())


def _build_focus_profile(
    *,
    keywords: Iterable[str] = (),
    require_match: bool = False,
    limits: Dict[str, int] | None = None,
    preferred_structured_clues: Iterable[str] = (),
    preferred_buckets: Iterable[str] = (),
) -> Dict[str, Any]:
    return {
        "keywords": tuple(keywords or ()),
        "require_match": require_match,
        "limits": limits or {"events": 3, "vlm_observations": 4, "relationships": 0, "group_artifacts": 0, "feature_refs": 0},
        "preferred_structured_clues": tuple(preferred_structured_clues or ()),
        "preferred_buckets": tuple(preferred_buckets or ()),
    }


def _index_allowed_refs(
    allowed_refs: Dict[str, List[Dict[str, Any]]],
    primary_person_id: str | None,
) -> Dict[str, List[Dict[str, Any]]]:
    indexed: Dict[str, List[Dict[str, Any]]] = {}
    for bucket, refs in allowed_refs.items():
        indexed[bucket] = [_index_ref(ref, primary_person_id) for ref in refs]
    return indexed


def _index_ref(ref: Dict[str, Any], primary_person_id: str | None) -> Dict[str, Any]:
    indexed = dict(ref)
    indexed["normalized_places"] = _extract_ref_normalized_places(indexed)
    indexed["normalized_brands"] = _extract_ref_normalized_brands(indexed)
    indexed["normalized_topics"] = _extract_ref_normalized_topics(indexed)
    indexed["activity_tags"] = _extract_ref_activity_tags(indexed)
    indexed["work_signals"] = _extract_ref_work_signals(indexed)
    indexed["time_keys"] = _extract_ref_time_keys(indexed)
    indexed["subject_binding"] = _extract_ref_subject_binding(indexed, primary_person_id)
    indexed["relationship_tags"] = _extract_ref_relationship_tags(indexed)
    indexed["group_tags"] = _extract_ref_group_tags(indexed)
    indexed["search_text"] = _build_ref_search_text(indexed)
    return indexed


def _build_ref_search_text(ref: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in (
        "signal",
        "description",
        "narrative_synthesis",
        "location",
        "activity",
        "type",
        "feature_name",
        "relationship_type",
        "group_type",
    ):
        value = ref.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    value = ref.get("value")
    if value is not None:
        parts.append(str(value))
    for key in (
        "details",
        "ocr_hits",
        "brands",
        "place_candidates",
        "tags",
        "normalized_places",
        "normalized_brands",
        "normalized_topics",
        "activity_tags",
        "work_signals",
        "relationship_tags",
        "group_tags",
    ):
        parts.extend(str(item) for item in ref.get(key, []) or [])
    time_keys = ref.get("time_keys") or {}
    parts.extend(str(item) for item in time_keys.values() if item)
    subject_binding = ref.get("subject_binding") or {}
    parts.extend(str(item) for item in subject_binding.values() if item)
    return " ".join(part for part in parts if str(part).strip())


def _extract_ref_normalized_places(ref: Dict[str, Any]) -> List[str]:
    places: List[str] = []
    for place in _location_city_keys_from_ref(ref):
        if place and place not in places:
            places.append(place)
    for raw in list(ref.get("place_candidates", []) or []) + [ref.get("location", "")]:
        text = str(raw or "").strip()
        if not text:
            continue
        normalized = _city_name_from_location_candidate(text)
        if normalized and normalized not in places:
            places.append(normalized)
    return places


def _extract_ref_normalized_brands(ref: Dict[str, Any]) -> List[str]:
    brands: List[str] = []
    for brand in ref.get("brands", []) or []:
        cleaned = _clean_brand_candidate(str(brand))
        if cleaned and cleaned not in brands:
            brands.append(cleaned)
    text = _build_raw_ref_text(ref)
    if brands or _contains_any_keyword(text, BRAND_PRODUCT_KEYWORDS + ("watermark", "shot on", "水印", "设备水印", "手机水印")):
        for brand in _extract_brand_candidates_from_text(text)[0]:
            if brand and brand not in brands:
                brands.append(brand)
    return brands


def _extract_ref_normalized_topics(ref: Dict[str, Any]) -> List[str]:
    topics: List[str] = []
    for raw in list(ref.get("tags", []) or []) + [ref.get("activity", ""), ref.get("type", ""), ref.get("signal", ""), ref.get("description", "")]:
        text = str(raw or "").strip()
        if not text:
            continue
        for candidate in _extract_topic_candidates_from_text(text):
            if candidate and candidate not in topics:
                topics.append(candidate)
        for candidate in _extract_visual_theme_candidates(text):
            if candidate and candidate not in topics:
                topics.append(candidate)
    return topics


def _extract_ref_activity_tags(ref: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    for raw in list(ref.get("tags", []) or []) + [ref.get("activity", ""), ref.get("type", ""), ref.get("title", "")]:
        text = _normalize_topic_name(raw)
        if text and text not in tags:
            tags.append(text)
    return tags


def _extract_ref_work_signals(ref: Dict[str, Any]) -> List[str]:
    text = _build_raw_ref_text(ref)
    signal_keywords = {
        "payroll_sheet": ("工资", "薪资", "salary", "payroll"),
        "attendance_sheet": ("考勤", "attendance", "签到", "打卡记录"),
        "punch_in_scene": ("打卡", "clock in", "punch"),
        "operation_scene": ("巡查", "值班", "现场", "工位", "办公室", "office", "工作", "meeting", "公安局"),
        "coworker_cluster": ("同事", "coworker", "team", "同伴"),
    }
    signals: List[str] = []
    for signal_name, keywords in signal_keywords.items():
        if _contains_any_keyword(text, keywords):
            signals.append(signal_name)
    return signals


def _extract_ref_time_keys(ref: Dict[str, Any]) -> Dict[str, Any]:
    timestamp = str(ref.get("timestamp", "") or "")
    date = str(ref.get("date", "") or "")
    raw = timestamp or date
    month = raw[:7] if len(raw) >= 7 else ""
    hour_bucket = ""
    parsed = _parse_ref_datetime(ref)
    if parsed is not None:
        hour_bucket = f"{parsed.hour:02d}:00"
    return {
        "date": raw[:10] if len(raw) >= 10 else date,
        "month": month,
        "hour_bucket": hour_bucket,
        "window_key": _window_key_from_ref(ref),
    }


def _extract_ref_subject_binding(ref: Dict[str, Any], primary_person_id: str | None) -> Dict[str, Any]:
    primary = str(primary_person_id or "").strip()
    participants = [str(item) for item in ref.get("participants", []) or []]
    people = [str(item) for item in ref.get("people", []) or []]
    face_person_ids = [str(item) for item in ref.get("face_person_ids", []) or []]
    return {
        "subject_role": ref.get("subject_role", ""),
        "participant_hit": primary and primary in participants,
        "people_hit": primary and primary in people,
        "face_hit": primary and primary in face_person_ids,
        "alias_hit": "主角" in participants or "主角" in people or "主角" in face_person_ids,
    }


def _extract_ref_relationship_tags(ref: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    relationship_type = str(ref.get("relationship_type", "") or "").strip()
    if relationship_type:
        tags.append(relationship_type)
    return tags


def _extract_ref_group_tags(ref: Dict[str, Any]) -> List[str]:
    tags: List[str] = []
    group_type = str(ref.get("group_type", "") or "").strip()
    if group_type:
        tags.append(group_type)
    return tags


def _build_raw_ref_text(ref: Dict[str, Any]) -> str:
    texts: List[str] = [
        ref.get("signal", ""),
        ref.get("description", ""),
        ref.get("narrative_synthesis", ""),
        ref.get("location", ""),
        ref.get("activity", ""),
        ref.get("type", ""),
        ref.get("title", ""),
    ]
    texts.extend(str(item) for item in ref.get("details", []) or [])
    texts.extend(str(item) for item in ref.get("ocr_hits", []) or [])
    texts.extend(str(item) for item in ref.get("brands", []) or [])
    texts.extend(str(item) for item in ref.get("place_candidates", []) or [])
    texts.extend(str(item) for item in ref.get("tags", []) or [])
    persona_evidence = ref.get("persona_evidence", {}) or {}
    for bucket in ("behavioral", "aesthetic", "socioeconomic"):
        texts.extend(str(item) for item in persona_evidence.get(bucket, []) or [])
    return " ".join(text for text in texts if text)


def _focus_supporting_refs(
    *,
    field_key: str,
    supporting_refs: Dict[str, List[Dict[str, Any]]],
    primary_person_id: str | None,
) -> Dict[str, List[Dict[str, Any]]]:
    profile = _resolve_focus_profile(field_key)
    if profile is None:
        return supporting_refs

    focused: Dict[str, List[Dict[str, Any]]] = {}
    for bucket, refs in supporting_refs.items():
        limit = profile["limits"].get(bucket, len(refs))
        focused[bucket] = _select_focused_refs(
            field_key=field_key,
            bucket=bucket,
            refs=refs,
            primary_person_id=primary_person_id,
            profile=profile,
            limit=limit,
        )
    return focused


def _resolve_focus_profile(field_key: str) -> Dict[str, Any] | None:
    if field_key == "long_term_facts.social_identity.education":
        return _build_focus_profile(
            keywords=SCHOOL_WORK_PLACE_KEYWORDS,
            require_match=True,
            limits={"events": 3, "vlm_observations": 4, "relationships": 0, "group_artifacts": 0, "feature_refs": 1},
            preferred_structured_clues=("normalized_places", "activity_tags", "subject_binding"),
            preferred_buckets=("events", "vlm_observations", "feature_refs"),
        )
    if field_key in IDENTITY_EXPLICIT_FIELDS:
        keywords = EXPLICIT_NAME_KEYWORDS
        if field_key.endswith(".gender"):
            keywords = EXPLICIT_GENDER_KEYWORDS
        elif field_key.endswith((".age_range", ".role", ".race")):
            keywords = EXPLICIT_AGE_ROLE_KEYWORDS
        return _build_focus_profile(
            keywords=keywords,
            require_match=True,
            limits={"events": 1, "vlm_observations": 3, "relationships": 0, "group_artifacts": 0, "feature_refs": 1},
            preferred_structured_clues=("subject_binding",),
            preferred_buckets=("events", "vlm_observations"),
        )
    if field_key == "long_term_facts.identity.nationality":
        return _build_focus_profile(
            keywords=NATIONALITY_QUERY_KEYWORDS + LANGUAGE_CULTURE_QUERY_KEYWORDS,
            require_match=True,
            limits={"events": 3, "vlm_observations": 4, "relationships": 0, "group_artifacts": 0, "feature_refs": 2},
            preferred_structured_clues=("normalized_places", "normalized_topics", "time_keys"),
            preferred_buckets=("events", "vlm_observations", "feature_refs"),
        )
    if field_key in CAREER_FOCUS_FIELDS:
        return _build_focus_profile(
            keywords=CAREER_SIGNAL_KEYWORDS,
            require_match=True,
            limits={"events": 4, "vlm_observations": 0, "relationships": 0, "group_artifacts": 0, "feature_refs": 2},
            preferred_structured_clues=("work_signals", "subject_binding"),
            preferred_buckets=("events", "feature_refs"),
        )
    if field_key in EVENT_DRIVEN_GEO_FIELDS:
        return _build_focus_profile(
            keywords=(),
            require_match=False,
            limits={"events": 5, "vlm_observations": 0, "relationships": 0, "group_artifacts": 0, "feature_refs": 1},
            preferred_structured_clues=("normalized_places", "time_keys", "subject_binding"),
            preferred_buckets=("events", "feature_refs"),
        )
    if field_key == "long_term_facts.social_identity.language_culture":
        return _build_focus_profile(
            keywords=LANGUAGE_CULTURE_QUERY_KEYWORDS,
            require_match=True,
            limits={"events": 3, "vlm_observations": 4, "relationships": 0, "group_artifacts": 0, "feature_refs": 2},
            preferred_structured_clues=("normalized_places", "normalized_topics", "activity_tags"),
            preferred_buckets=("events", "vlm_observations", "feature_refs"),
        )
    if field_key in SPENDING_FOCUS_FIELDS or field_key == "long_term_facts.material.brand_preference":
        return _build_focus_profile(
            keywords=SPENDING_SIGNAL_KEYWORDS,
            require_match=field_key != "long_term_facts.material.brand_preference",
            limits={"events": 4 if field_key == "long_term_facts.material.brand_preference" else 3, "vlm_observations": 6 if field_key == "long_term_facts.material.brand_preference" else 5, "relationships": 0, "group_artifacts": 0, "feature_refs": 1},
            preferred_structured_clues=("normalized_brands", "subject_binding", "work_signals"),
            preferred_buckets=("events", "vlm_observations"),
        )
    if field_key in GEOGRAPHY_FOCUS_FIELDS:
        return _build_focus_profile(
            keywords=(),
            require_match=False,
            limits={"events": 4, "vlm_observations": 6, "relationships": 0, "group_artifacts": 0, "feature_refs": 0},
            preferred_structured_clues=("normalized_places", "time_keys", "subject_binding"),
            preferred_buckets=("events", "vlm_observations"),
        )
    if field_key == "short_term_facts.recent_interests":
        return _build_focus_profile(
            keywords=ACTIVITY_INTEREST_QUERY_KEYWORDS,
            require_match=False,
            limits={"events": 4, "vlm_observations": 5, "relationships": 0, "group_artifacts": 0, "feature_refs": 1},
            preferred_structured_clues=("normalized_topics", "activity_tags"),
            preferred_buckets=("events", "vlm_observations"),
        )
    if field_key in TIME_FOCUS_FIELDS:
        return _build_focus_profile(
            keywords=(),
            require_match=False,
            limits={"events": 5, "vlm_observations": 2, "relationships": 0, "group_artifacts": 0, "feature_refs": 1},
            preferred_structured_clues=("time_keys", "activity_tags"),
            preferred_buckets=("events", "feature_refs"),
        )
    if field_key == "long_term_facts.relationships.intimate_partner":
        return _build_focus_profile(
            keywords=(),
            require_match=False,
            limits={"events": 0, "vlm_observations": 0, "relationships": 4, "group_artifacts": 0, "feature_refs": 1},
            preferred_structured_clues=("relationship_tags",),
            preferred_buckets=("relationships",),
        )
    if field_key == "long_term_facts.relationships.close_circle_size":
        return _build_focus_profile(
            keywords=(),
            require_match=False,
            limits={"events": 0, "vlm_observations": 0, "relationships": 6, "group_artifacts": 0, "feature_refs": 1},
            preferred_structured_clues=("relationship_tags",),
            preferred_buckets=("relationships",),
        )
    if field_key == "long_term_facts.relationships.social_groups":
        return _build_focus_profile(
            keywords=(),
            require_match=False,
            limits={"events": 0, "vlm_observations": 0, "relationships": 0, "group_artifacts": 3, "feature_refs": 0},
            preferred_structured_clues=("group_tags",),
            preferred_buckets=("group_artifacts",),
        )
    if field_key in HOUSEHOLD_FOCUS_FIELDS:
        return _build_focus_profile(
            keywords=HOUSEHOLD_SIGNAL_KEYWORDS,
            require_match=True,
            limits={"events": 3, "vlm_observations": 4, "relationships": 2, "group_artifacts": 0, "feature_refs": 1},
            preferred_structured_clues=("activity_tags", "subject_binding", "relationship_tags"),
            preferred_buckets=("events", "vlm_observations", "relationships"),
        )
    if field_key in {
        "long_term_facts.hobbies.interests",
        "long_term_facts.hobbies.frequent_activities",
    }:
        return _build_focus_profile(
            keywords=ACTIVITY_INTEREST_QUERY_KEYWORDS,
            require_match=True,
            limits={"events": 4 if field_key.endswith(".interests") else 5, "vlm_observations": 5 if field_key.endswith(".interests") else 0, "relationships": 0, "group_artifacts": 0, "feature_refs": 1},
            preferred_structured_clues=("normalized_topics", "activity_tags", "subject_binding"),
            preferred_buckets=("events", "vlm_observations"),
        )
    if field_key == "long_term_facts.hobbies.solo_vs_social":
        return _build_focus_profile(
            keywords=SOCIAL_ACTIVITY_QUERY_KEYWORDS,
            require_match=True,
            limits={"events": 4, "vlm_observations": 0, "relationships": 3, "group_artifacts": 0, "feature_refs": 1},
            preferred_structured_clues=("relationship_tags", "subject_binding"),
            preferred_buckets=("events", "relationships"),
        )
    if field_key in {
        "long_term_facts.physiology.fitness_level",
        "long_term_facts.physiology.diet_mode",
    }:
        return _build_focus_profile(
            keywords=DIET_FITNESS_QUERY_KEYWORDS,
            require_match=True,
            limits={"events": 4, "vlm_observations": 4, "relationships": 0, "group_artifacts": 0, "feature_refs": 1},
            preferred_structured_clues=("normalized_topics", "activity_tags", "subject_binding"),
            preferred_buckets=("events", "vlm_observations"),
        )
    if field_key == "short_term_facts.life_events":
        return _build_focus_profile(
            keywords=LIFE_EVENT_QUERY_KEYWORDS,
            require_match=False,
            limits={"events": 5, "vlm_observations": 0, "relationships": 2, "group_artifacts": 0, "feature_refs": 1},
            preferred_structured_clues=("time_keys", "relationship_tags", "activity_tags"),
            preferred_buckets=("events", "relationships"),
        )
    if field_key in STYLE_FOCUS_FIELDS:
        return _build_focus_profile(
            keywords=STYLE_SIGNAL_KEYWORDS,
            require_match=True,
            limits={"events": 3, "vlm_observations": 5, "relationships": 0, "group_artifacts": 0, "feature_refs": 1},
            preferred_structured_clues=("normalized_topics", "activity_tags", "subject_binding"),
            preferred_buckets=("vlm_observations", "events"),
        )
    if field_key in MOOD_FOCUS_FIELDS:
        return _build_focus_profile(
            keywords=MOOD_SIGNAL_KEYWORDS,
            require_match=True,
            limits={"events": 3, "vlm_observations": 4, "relationships": 2, "group_artifacts": 0, "feature_refs": 1},
            preferred_structured_clues=("activity_tags", "relationship_tags", "subject_binding"),
            preferred_buckets=("events", "vlm_observations", "relationships"),
        )
    if field_key in VALUES_FOCUS_FIELDS:
        return _build_focus_profile(
            keywords=VALUES_SIGNAL_KEYWORDS,
            require_match=True,
            limits={"events": 2, "vlm_observations": 2, "relationships": 2, "group_artifacts": 0, "feature_refs": 0},
            preferred_structured_clues=("relationship_tags", "activity_tags"),
            preferred_buckets=("events", "relationships"),
        )
    return None


def _select_focused_refs(
    *,
    field_key: str,
    bucket: str,
    refs: List[Dict[str, Any]],
    primary_person_id: str | None,
    profile: Dict[str, Any],
    limit: int,
) -> List[Dict[str, Any]]:
    if limit <= 0 or not refs:
        return []

    if field_key in GEOGRAPHY_FOCUS_FIELDS and bucket in {"events", "vlm_observations"}:
        return _select_location_bucket_refs(
            field_key=field_key,
            bucket=bucket,
            refs=refs,
            primary_person_id=primary_person_id,
            profile=profile,
            limit=limit,
        )

    scored: List[tuple[float, float, Dict[str, Any]]] = []
    for ref in refs:
        score = _score_ref_for_focus(
            field_key=field_key,
            bucket=bucket,
            ref=ref,
            primary_person_id=primary_person_id,
            profile=profile,
        )
        if score is None:
            continue
        scored.append((score, _ref_recency_sort_value(ref), ref))

    if not scored:
        return []

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for _, _, ref in scored:
        ref_key = ref.get("source_id") or ref.get("event_id") or ref.get("photo_id") or str(ref)
        if ref_key in seen:
            continue
        seen.add(str(ref_key))
        selected.append(ref)
        if len(selected) >= limit:
            break
    return selected


def _select_location_bucket_refs(
    *,
    field_key: str,
    bucket: str,
    refs: List[Dict[str, Any]],
    primary_person_id: str | None,
    profile: Dict[str, Any],
    limit: int,
) -> List[Dict[str, Any]]:
    scored_by_city: Dict[str, List[tuple[float, float, Dict[str, Any]]]] = {}
    for ref in refs:
        score = _score_ref_for_focus(
            field_key=field_key,
            bucket=bucket,
            ref=ref,
            primary_person_id=primary_person_id,
            profile=profile,
        )
        if score is None:
            continue
        city_keys = _location_city_keys_from_ref(ref) or ["__unknown__"]
        recency = _ref_recency_sort_value(ref)
        for city_key in city_keys:
            scored_by_city.setdefault(city_key, []).append((score, recency, ref))

    for items in scored_by_city.values():
        items.sort(key=lambda item: (item[0], item[1]), reverse=True)

    selected: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for city_key, items in sorted(
        scored_by_city.items(),
        key=lambda item: (
            min(_location_bucket_priority(entry[2]) for entry in item[1]),
            -len(item[1]),
            -max(entry[0] for entry in item[1]),
            item[0],
        ),
    ):
        for _, _, ref in items:
            ref_key = ref.get("source_id") or ref.get("event_id") or ref.get("photo_id") or str(ref)
            if ref_key in seen:
                continue
            seen.add(str(ref_key))
            selected.append(ref)
            break
        if len(selected) >= limit:
            return selected

    remaining: List[tuple[float, float, Dict[str, Any]]] = []
    for items in scored_by_city.values():
        remaining.extend(items)
    remaining.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for _, _, ref in remaining:
        ref_key = ref.get("source_id") or ref.get("event_id") or ref.get("photo_id") or str(ref)
        if ref_key in seen:
            continue
        seen.add(str(ref_key))
        selected.append(ref)
        if len(selected) >= limit:
            break
    return selected


def _score_ref_for_focus(
    *,
    field_key: str,
    bucket: str,
    ref: Dict[str, Any],
    primary_person_id: str | None,
    profile: Dict[str, Any],
) -> float | None:
    text = _ref_focus_text(ref)
    keyword_hits = _keyword_hit_count(text, profile.get("keywords", ()))
    structured_hits = _structured_focus_hits(field_key, ref, profile=profile)
    semantic_match_hits = _structured_focus_hits(field_key, ref, profile=profile, for_match_gate=True)

    if profile.get("require_match") and keyword_hits == 0 and semantic_match_hits == 0:
        return None

    score = 0.0
    bucket_base = {
        "events": 50.0,
        "vlm_observations": 40.0,
        "relationships": 35.0,
        "group_artifacts": 60.0,
        "feature_refs": 20.0,
    }
    score += bucket_base.get(bucket, 0.0)
    if bucket in set(profile.get("preferred_buckets", ()) or ()):
        score += 6.0
    score += float(keyword_hits * 18)
    score += float(structured_hits * 12)
    score += _primary_binding_score(ref, primary_person_id)

    if bucket == "events":
        score += min(int(ref.get("photo_count", 0) or 0), 6) * 2
    if bucket == "vlm_observations":
        subject_role = ref.get("subject_role")
        if subject_role == "protagonist_present":
            score += 18
        elif subject_role == "protagonist_view":
            score += 12
        if ref.get("is_reference_like"):
            score -= 4
        score += min(len(ref.get("details", []) or []), 4)
        score += min(len(ref.get("ocr_hits", []) or []), 3)

    if field_key in TIME_FOCUS_FIELDS and bucket != "events":
        score -= 10
    if field_key in STYLE_FOCUS_FIELDS and bucket == "vlm_observations":
        score += 8
    if field_key in MOOD_FOCUS_FIELDS and bucket == "events":
        score += 6
    if field_key == "long_term_facts.relationships.social_groups" and bucket != "group_artifacts":
        return None

    return score


def _ref_focus_text(ref: Dict[str, Any]) -> str:
    if ref.get("search_text"):
        return str(ref.get("search_text"))
    texts: List[str] = [
        ref.get("signal", ""),
        ref.get("description", ""),
        ref.get("narrative_synthesis", ""),
        ref.get("location", ""),
        ref.get("activity", ""),
        ref.get("type", ""),
    ]
    texts.extend(str(item) for item in ref.get("details", []) or [])
    texts.extend(str(item) for item in ref.get("ocr_hits", []) or [])
    texts.extend(str(item) for item in ref.get("brands", []) or [])
    texts.extend(str(item) for item in ref.get("place_candidates", []) or [])
    texts.extend(str(item) for item in ref.get("tags", []) or [])
    persona_evidence = ref.get("persona_evidence", {}) or {}
    for bucket in ("behavioral", "aesthetic", "socioeconomic"):
        texts.extend(str(item) for item in persona_evidence.get(bucket, []) or [])
    return " ".join(text for text in texts if text)


def _keyword_hit_count(text: str, keywords: Iterable[str]) -> int:
    normalized = str(text or "").lower()
    hits = 0
    for keyword in keywords or ():
        if keyword.lower() in normalized:
            hits += 1
    return hits


def _structured_focus_hits(
    field_key: str,
    ref: Dict[str, Any],
    profile: Dict[str, Any] | None = None,
    *,
    for_match_gate: bool = False,
) -> int:
    hits = 0
    preferred_clues = set(profile.get("preferred_structured_clues", ()) or ()) if profile else set()
    profile_keywords = tuple(profile.get("keywords", ()) or ()) if profile else ()
    if "normalized_places" in preferred_clues and _structured_clue_matches_profile(ref.get("normalized_places", []), profile_keywords):
        hits += 1
    if "normalized_brands" in preferred_clues and _structured_clue_matches_profile(ref.get("normalized_brands", []), profile_keywords):
        hits += 1
    if "normalized_topics" in preferred_clues and _structured_clue_matches_profile(ref.get("normalized_topics", []), profile_keywords):
        hits += 1
    if "activity_tags" in preferred_clues and _structured_clue_matches_profile(ref.get("activity_tags", []), profile_keywords):
        hits += 1
    if "work_signals" in preferred_clues and _structured_clue_matches_profile(ref.get("work_signals", []), profile_keywords):
        hits += 1
    if not for_match_gate and "time_keys" in preferred_clues and any((ref.get("time_keys") or {}).values()):
        hits += 1
    if not for_match_gate and "subject_binding" in preferred_clues:
        subject_binding = ref.get("subject_binding") or {}
        if any(bool(value) for key, value in subject_binding.items() if key != "subject_role") or subject_binding.get("subject_role"):
            hits += 1
    if "relationship_tags" in preferred_clues and ref.get("relationship_tags"):
        hits += 1
    if "group_tags" in preferred_clues and ref.get("group_tags"):
        hits += 1
    if field_key in SPENDING_FOCUS_FIELDS or field_key == "long_term_facts.material.brand_preference":
        if ref.get("brands"):
            hits += 1
        if ref.get("ocr_hits"):
            hits += 1
    if field_key in GEOGRAPHY_FOCUS_FIELDS and ref.get("place_candidates"):
        hits += 1
    if field_key == "short_term_facts.recent_interests" and (ref.get("tags") or ref.get("details") or ref.get("ocr_hits")):
        hits += 1
    if field_key == "long_term_facts.social_identity.education" and ref.get("place_candidates"):
        hits += 1
    return hits


def _structured_clue_matches_profile(values: Iterable[Any], keywords: Iterable[str]) -> bool:
    normalized_values = [str(value or "").strip() for value in values or () if str(value or "").strip()]
    if not normalized_values:
        return False
    keyword_list = [str(keyword or "").strip().lower() for keyword in keywords or () if str(keyword or "").strip()]
    if not keyword_list:
        return True
    joined = " ".join(normalized_values).lower()
    return any(keyword in joined for keyword in keyword_list)


def _primary_binding_score(ref: Dict[str, Any], primary_person_id: str | None) -> float:
    primary = str(primary_person_id or "").strip()
    if ref.get("subject_role") == "protagonist_present":
        return 16.0
    if ref.get("subject_role") == "protagonist_view":
        return 12.0
    if primary:
        if primary in (ref.get("participants") or []):
            return 12.0
        if primary in (ref.get("people") or []):
            return 12.0
        if primary in (ref.get("face_person_ids") or []):
            return 12.0
    if "主角" in (ref.get("participants") or []):
        return 10.0
    return 0.0


def _ref_recency_sort_value(ref: Dict[str, Any]) -> float:
    for key in ("timestamp", "date"):
        value = ref.get(key)
        if not value:
            continue
        try:
            if key == "date":
                return datetime.fromisoformat(str(value)).timestamp()
            return datetime.fromisoformat(str(value)).timestamp()
        except Exception:
            continue
    return 0.0
