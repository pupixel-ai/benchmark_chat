"""
关系层代码规则：候选关系、状态、置信度
"""
from __future__ import annotations

from typing import Dict, List, Tuple


PRIVATE_SCENE_KEYWORDS = {
    "家", "卧室", "客厅", "公寓", "宿舍",
    "home", "bedroom", "living room", "apartment", "dorm",
}
WORK_OR_SCHOOL_KEYWORDS = {
    "公司", "办公室", "会议室", "工位", "学校", "教室", "校园",
    "office", "meeting", "workspace", "school", "classroom", "campus",
}
HIGH_RISK_RELATIONSHIP_TYPES = {"romantic", "family"}
VALID_RELATIONSHIP_STATUSES = {"new", "growing", "stable", "fading", "gone"}


def infer_relationship_candidates(evidence: Dict, intimacy_score: float) -> List[str]:
    """根据代码证据收缩关系候选集，降低 LLM 自由发挥空间。"""
    candidates: List[str] = []
    scenes = evidence.get("scenes", [])
    scene_count = len(scenes)
    contact_types = set(evidence.get("contact_types", []))
    weekend_high = evidence.get("weekend_frequency") == "高"
    one_on_one = bool(evidence.get("with_user_only", False))
    private_ratio = float(evidence.get("private_scene_ratio", 0.0) or 0.0)
    dominant_scene_ratio = float(evidence.get("dominant_scene_ratio", 0.0) or 0.0)

    romantic_contacts = {"kiss", "hug", "holding_hands", "arm_in_arm"}
    selfie_like_contacts = {"selfie_together", "standing_near", "shoulder_lean"}

    has_work_or_school_scene = any(
        any(keyword in scene.lower() for keyword in WORK_OR_SCHOOL_KEYWORDS)
        for scene in scenes
    )
    has_private_scene = private_ratio >= 0.34 or any(
        any(keyword in scene.lower() for keyword in PRIVATE_SCENE_KEYWORDS)
        for scene in scenes
    )

    if (
        intimacy_score >= 0.68
        and one_on_one
        and has_private_scene
        and contact_types.intersection(romantic_contacts)
    ):
        candidates.append("romantic")

    if has_work_or_school_scene and not weekend_high:
        candidates.append("classmate_colleague")

    if dominant_scene_ratio >= 0.65 and scene_count <= 2:
        candidates.append("activity_buddy")

    if intimacy_score >= 0.58 and weekend_high and (
        contact_types.intersection(selfie_like_contacts) or one_on_one
    ):
        candidates.append("bestie")

    if intimacy_score >= 0.45 and scene_count >= 2:
        candidates.append("close_friend")

    if intimacy_score >= 0.30:
        candidates.append("friend")

    if has_private_scene and evidence.get("time_span_days", 0) >= 180:
        candidates.append("family")

    candidates.append("acquaintance")
    return _dedupe_preserve_order(candidates)


def determine_relationship_status(evidence: Dict) -> str:
    """优先用代码判断关系状态，减少 LLM 对趋势的随意解释。"""
    recent_gap_days = int(evidence.get("recent_gap_days", 0) or 0)
    time_span_days = int(evidence.get("time_span_days", 0) or 0)
    trend_direction = evidence.get("trend_detail", {}).get("direction", "")

    if recent_gap_days > 60:
        return "gone"
    if time_span_days and time_span_days < 30:
        return "new"
    if trend_direction == "up" and recent_gap_days <= 14:
        return "growing"
    if trend_direction == "down":
        return "fading"
    return "stable"


def score_relationship_confidence(
    evidence: Dict,
    intimacy_score: float,
    relationship_type: str,
    candidate_types: List[str],
) -> float:
    """基于代码证据给出置信度，避免完全信任 LLM 自报。"""
    photo_count = int(evidence.get("photo_count", 0) or 0)
    scene_count = len(evidence.get("scenes", []))
    contact_types = evidence.get("contact_types", [])
    anomalies = evidence.get("anomalies", [])
    one_on_one = bool(evidence.get("with_user_only", False))
    weekend_high = evidence.get("weekend_frequency") == "高"

    score = 0.35
    score += min(photo_count / 10, 1.0) * 0.15
    score += min(scene_count / 4, 1.0) * 0.10
    score += intimacy_score * 0.25
    score += min(len(contact_types) / 3, 1.0) * 0.10
    score += 0.05 if one_on_one else 0.0
    score += 0.05 if weekend_high else 0.0
    score -= min(len(anomalies), 3) * 0.05

    if relationship_type not in candidate_types:
        score -= 0.25

    if relationship_type in {"romantic", "family"} and relationship_type != candidate_types[0]:
        score -= 0.05

    return round(max(0.2, min(score, 0.95)), 3)

def apply_relationship_type_veto(llm_relationship_type: str, candidate_types: List[str]) -> Tuple[str, List[str]]:
    """
    Soft-constraint 模式下仅对高风险关系类型保留 veto。
    """
    applied_vetoes: List[str] = []
    fallback_type = candidate_types[0] if candidate_types else "acquaintance"
    final_type = llm_relationship_type or fallback_type

    if final_type in HIGH_RISK_RELATIONSHIP_TYPES and final_type not in candidate_types:
        final_type = fallback_type
        applied_vetoes.append("RELATIONSHIP_TYPE_HIGH_RISK_VETO")

    return final_type, applied_vetoes


def apply_status_redlines(llm_status: str, evidence: Dict) -> Tuple[str, List[str]]:
    """
    Soft-constraint 模式下仅保留时间红线 veto。
    """
    applied_vetoes: List[str] = []
    final_status = llm_status if llm_status in VALID_RELATIONSHIP_STATUSES else "stable"

    recent_gap_days = int(evidence.get("recent_gap_days", 0) or 0)
    time_span_days = int(evidence.get("time_span_days", 0) or 0)

    if recent_gap_days > 60 and final_status != "gone":
        final_status = "gone"
        applied_vetoes.append("STATUS_GONE_REDLINE")
    elif 0 < time_span_days < 30 and final_status != "gone" and final_status != "new":
        final_status = "new"
        applied_vetoes.append("STATUS_NEW_REDLINE")

    return final_status, applied_vetoes


def blend_relationship_confidence(
    llm_confidence: float,
    code_confidence_baseline: float,
    applied_vetoes: List[str],
) -> float:
    """
    融合 LLM 自报和代码基线，触发 veto 时额外降档。
    """
    score = 0.6 * llm_confidence + 0.4 * code_confidence_baseline
    if applied_vetoes:
        score -= 0.1
    return round(max(0.2, min(score, 0.95)), 3)


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered
