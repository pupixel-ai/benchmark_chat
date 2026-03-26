from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, List, Tuple

from .evidence_utils import build_evidence_payload
from .types import MemoryState, PersonScreening


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


@dataclass
class PrimaryDecision:
    mode: str
    primary_person_id: str | None
    confidence: float
    evidence: Dict[str, Any]
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def analyze_primary_person_with_reflection(
    state: MemoryState,
    fallback_primary_person_id: str | None = None,
    llm_processor: Any | None = None,
) -> Tuple[PrimaryDecision, Dict[str, Any]]:
    screening = state.screening or {}
    candidates = _collect_candidate_signals(state, screening)
    ranked = sorted(candidates, key=_candidate_sort_key, reverse=True)

    llm_result = _run_llm_primary_judgement(
        state=state,
        ranked_candidates=ranked,
        llm_processor=llm_processor,
    )
    initial_decision, decision_meta = _build_primary_decision(
        ranked_candidates=ranked,
        llm_result=llm_result,
        screening=screening,
    )
    reflection = _reflect_primary_decision(
        initial_decision=initial_decision,
        decision_meta=decision_meta,
        ranked_candidates=ranked,
        screening=screening,
        llm_result=llm_result,
        fallback_primary_person_id=fallback_primary_person_id,
    )
    final_decision = _apply_primary_reflection(initial_decision, reflection)
    reflection["primary_signal_trace"] = {
        "candidate_signals": ranked,
        "llm_decision": llm_result or {},
        "selected_person_id": final_decision.primary_person_id,
        "selected_mode": final_decision.mode,
    }
    return final_decision, reflection


def analyze_primary_person(
    state: MemoryState,
    fallback_primary_person_id: str | None = None,
    llm_processor: Any | None = None,
) -> PrimaryDecision:
    decision, _ = analyze_primary_person_with_reflection(
        state=state,
        fallback_primary_person_id=fallback_primary_person_id,
        llm_processor=llm_processor,
    )
    return decision


def _collect_candidate_signals(
    state: MemoryState,
    screening: Dict[str, PersonScreening],
) -> List[Dict[str, Any]]:
    signal_map: Dict[str, Dict[str, Any]] = {}
    for person_id, person_info in (state.face_db or {}).items():
        screening_result = screening.get(person_id)
        if screening_result and screening_result.memory_value == "block":
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
            "selfie_photo_ids": [],
            "identity_anchor_photo_ids": [],
            "memory_value": screening_result.memory_value if screening_result else None,
            "person_kind": screening_result.person_kind if screening_result else None,
        }

    for event in state.events or []:
        participants = list(getattr(event, "participants", []) or [])
        for person_id in participants:
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
                if photo_id:
                    candidate["selfie_photo_ids"].append(photo_id)
            else:
                candidate["non_selfie_photo_count"] += 1
            if is_identity:
                candidate["identity_anchor_count"] += 1
                if photo_id:
                    candidate["identity_anchor_photo_ids"].append(photo_id)
            if is_photographed_subject and len(people_ids) == 1:
                candidate["photographed_subject_hits"] += 1

    for candidate in signal_map.values():
        non_selfie = max(candidate["non_selfie_photo_count"], 1)
        candidate["photographed_subject_ratio"] = round(candidate["photographed_subject_hits"] / non_selfie, 3)
        candidate["supporting_photo_ids"] = _dedupe(candidate["supporting_photo_ids"])
        candidate["selfie_photo_ids"] = _dedupe(candidate["selfie_photo_ids"])
        candidate["identity_anchor_photo_ids"] = _dedupe(candidate["identity_anchor_photo_ids"])

    return list(signal_map.values())


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


def _run_llm_primary_judgement(
    state: MemoryState,
    ranked_candidates: List[Dict[str, Any]],
    llm_processor: Any | None,
) -> Dict[str, Any] | None:
    if not llm_processor or not hasattr(llm_processor, "_call_llm_via_official_api"):
        return None
    if not ranked_candidates:
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

    event_lines: List[str] = []
    for event in list(state.events or [])[:15]:
        event_lines.append(
            "- {event_id} {title} @ {location} | participants={participants}".format(
                event_id=getattr(event, "event_id", ""),
                title=getattr(event, "title", ""),
                location=getattr(event, "location", ""),
                participants=",".join(getattr(event, "participants", []) or []),
            )
        )
    if not event_lines:
        for item in list(state.vlm_results or [])[:20]:
            analysis = item.get("vlm_analysis", {}) or {}
            people = [
                str(person.get("person_id"))
                for person in (analysis.get("people", []) or [])
                if isinstance(person, dict) and person.get("person_id")
            ]
            event_lines.append(
                "- {photo_id} {summary} | people={people}".format(
                    photo_id=item.get("photo_id", ""),
                    summary=str(analysis.get("summary", "") or "")[:160],
                    people=",".join(people),
                )
            )

    prompt = f"""你是相册主人识别专家。请只返回 JSON。

候选统计（Top 5）：
| Person | photo_count | event_count | selfie_count | identity_anchor_count | protagonist_label_count | photographed_subject_ratio |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(stats_rows)}

关键事件/照片线索：
{chr(10).join(event_lines[:30])}

输出字段：
{{
  "primary_person_id": "Person_xxx 或 空字符串",
  "confidence": 0-100,
  "key_signals": [{{"signal":"自拍|身份锚点|主角标签|出现频率|拍摄关系","points_to":"Person_xxx","detail":"简述"}}],
  "conflicts": ["冲突说明"],
  "runner_up": {{"person_id":"Person_xxx","relationship_guess":"伴侣/家人/朋友/未知","why_not":"一句话"}},
  "reasoning": "一句话总结"
}}
"""

    try:
        result = llm_processor._call_llm_via_official_api(prompt, response_mime_type="application/json")
        return result if isinstance(result, dict) else None
    except Exception:
        return None


def _build_primary_decision(
    ranked_candidates: List[Dict[str, Any]],
    llm_result: Dict[str, Any] | None,
    screening: Dict[str, PersonScreening],
) -> Tuple[PrimaryDecision, Dict[str, Any]]:
    candidate_by_id = {candidate["person_id"]: candidate for candidate in ranked_candidates}
    if llm_result:
        selected_id = str(llm_result.get("primary_person_id") or "").strip()
        if selected_id in candidate_by_id:
            candidate = candidate_by_id[selected_id]
            confidence = _normalize_confidence(llm_result.get("confidence"), default=0.76)
            feature_names = _extract_feature_names_from_llm(llm_result)
            reasoning = str(llm_result.get("reasoning") or "").strip() or (
                f"{selected_id} 在自拍/身份/主角标签等信号综合后被判为主角。"
            )
            return (
                _person_primary_decision(
                    person_id=selected_id,
                    confidence=confidence,
                    photo_ids=candidate["supporting_photo_ids"],
                    feature_names=feature_names or ["llm_primary_selection"],
                    reasoning=reasoning,
                ),
                {"source": "llm", "selected_person_id": selected_id},
            )

    if not ranked_candidates:
        return (
            _photographer_mode_decision(
                confidence=0.55,
                feature_names=["no_person_candidates"],
                reasoning="没有可用人物候选，回退 photographer_mode。",
            ),
            {"source": "fallback", "selected_person_id": None},
        )

    top = ranked_candidates[0]
    second = ranked_candidates[1] if len(ranked_candidates) > 1 else None
    top_score = _candidate_sort_key(top)[0]
    second_score = _candidate_sort_key(second)[0] if second else -999

    if top["photographed_subject_ratio"] >= 0.6 and top["selfie_count"] == 0 and top["identity_anchor_count"] == 0:
        return (
            _photographer_mode_decision(
                confidence=0.78,
                feature_names=["other_photo_candidate_is_likely_a_photographed_subject"],
                reasoning="Top 候选更像被持续拍摄对象，命中“拍别人”主线，回退 photographer_mode。",
                contradicting_refs=[
                    {
                        "source_type": "feature",
                        "source_id": "other_photo_candidate_is_likely_a_photographed_subject",
                        "signal": str(top["photographed_subject_ratio"]),
                        "why": "primary_candidate_veto",
                    }
                ],
            ),
            {"source": "fallback", "selected_person_id": None},
        )

    if (
        top["selfie_count"] == 0
        and top["identity_anchor_count"] == 0
        and top["protagonist_label_count"] == 0
        and (top_score <= 0.8 or not top["supporting_photo_ids"])
    ):
        photographed_subject_risk = any(
            candidate["photographed_subject_ratio"] >= 0.6 and candidate["non_selfie_photo_count"] >= 3
            for candidate in ranked_candidates[:3]
        )
        feature_names = ["no_stable_anchor_signal"]
        reasoning = "缺少自拍、身份锚点和主角标签等稳定主角信号，保守回退 photographer_mode。"
        if photographed_subject_risk:
            feature_names.append("other_photo_candidate_is_likely_a_photographed_subject")
            reasoning = "候选主要落在“拍别人”主线且缺少稳定主角锚点，保守回退 photographer_mode。"
        return (
            _photographer_mode_decision(
                confidence=0.74,
                feature_names=feature_names,
                reasoning=reasoning,
            ),
            {"source": "fallback", "selected_person_id": None},
        )

    if second and abs(top_score - second_score) <= 0.6 and top["selfie_count"] <= 1 and second["selfie_count"] <= 1:
        return (
            _photographer_mode_decision(
                confidence=0.72,
                feature_names=["ambiguous_top_candidates"],
                reasoning="候选人顶部信号接近且缺少稳定自拍/身份锚点，回退 photographer_mode。",
            ),
            {"source": "fallback", "selected_person_id": None},
        )

    screening_result = screening.get(top["person_id"])
    if screening_result and screening_result.memory_value == "low_value":
        return (
            _photographer_mode_decision(
                confidence=0.7,
                feature_names=["selected_candidate_low_value"],
                reasoning="Top 候选被筛查为 low_value，不作为主角，回退 photographer_mode。",
            ),
            {"source": "fallback", "selected_person_id": None},
        )

    confidence = min(0.9, max(0.62, 0.65 + min(top_score / 20.0, 0.22)))
    return (
        _person_primary_decision(
            person_id=top["person_id"],
            confidence=round(confidence, 2),
            photo_ids=top["supporting_photo_ids"],
            feature_names=["photo_count", "selfie_count", "identity_anchor_count", "protagonist_label_count"],
            reasoning=(
                f"{top['person_id']} 在自拍、身份锚点和主角标签等综合信号中领先，"
                "因此判定为相册主角。"
            ),
        ),
        {"source": "fallback", "selected_person_id": top["person_id"]},
    )


def _reflect_primary_decision(
    initial_decision: PrimaryDecision,
    decision_meta: Dict[str, Any],
    ranked_candidates: List[Dict[str, Any]],
    screening: Dict[str, PersonScreening],
    llm_result: Dict[str, Any] | None,
    fallback_primary_person_id: str | None = None,
) -> Dict[str, Any]:
    issues: List[str] = []
    action = "keep"

    if initial_decision.mode == "photographer_mode":
        feature_names = list((initial_decision.evidence or {}).get("feature_names", []) or [])
        if "other_photo_candidate_is_likely_a_photographed_subject" in feature_names:
            issues.append("other_photo_candidate_is_likely_a_photographed_subject")

    if initial_decision.mode == "person_id" and initial_decision.primary_person_id:
        selected = next((item for item in ranked_candidates if item["person_id"] == initial_decision.primary_person_id), None)
        screening_result = screening.get(initial_decision.primary_person_id)
        if screening_result and screening_result.memory_value == "low_value":
            issues.append("selected_candidate_low_value")
            action = "switch_to_photographer_mode"
        if screening_result and screening_result.person_kind and screening_result.person_kind != "real_person":
            issues.append(f"selected_candidate_kind={screening_result.person_kind}")
            action = "switch_to_photographer_mode"
        if selected and selected["photographed_subject_ratio"] >= 0.6:
            issues.append("other_photo_candidate_is_likely_a_photographed_subject")
            action = "switch_to_photographer_mode"

    if len(ranked_candidates) >= 2:
        top_score = _candidate_sort_key(ranked_candidates[0])[0]
        second_score = _candidate_sort_key(ranked_candidates[1])[0]
        if abs(top_score - second_score) <= 0.6:
            issues.append("ambiguous_top_candidates")
            if initial_decision.mode == "person_id" and ranked_candidates[0]["selfie_count"] <= 1:
                action = "switch_to_photographer_mode"

    llm_conflicts = list((llm_result or {}).get("conflicts") or [])
    if llm_conflicts:
        issues.append("llm_reported_signal_conflicts")
        if initial_decision.mode == "person_id":
            action = "switch_to_photographer_mode" if "ambiguous_top_candidates" in issues else action

    return {
        "triggered": bool(issues),
        "questions": [
            "是否把高频被拍对象误判为主角",
            "是否应回退 photographer_mode",
            "是否存在主角分裂/候选接近风险",
            "候选是否 low_value 或 non-real-person",
            "证据是否足够支撑主角判断",
        ],
        "issues": issues,
        "action": action,
        "fallback_primary_person_id": fallback_primary_person_id,
        "decision_source": decision_meta.get("source", ""),
    }


def _apply_primary_reflection(
    initial_decision: PrimaryDecision,
    reflection: Dict[str, Any],
) -> PrimaryDecision:
    if reflection.get("action") != "switch_to_photographer_mode":
        return initial_decision
    issues = list(reflection.get("issues", []))
    reasoning = "主角复核发现证据冲突或存在“拍别人”风险，回退 photographer_mode。"
    if "other_photo_candidate_is_likely_a_photographed_subject" in issues:
        reasoning = "主角复核后确认候选主要落在“拍别人”主线，回退 photographer_mode。"
    return _photographer_mode_decision(
        confidence=max(0.45, round(initial_decision.confidence - 0.16, 2)),
        feature_names=["primary_reflection"] + issues[:3],
        reasoning=reasoning,
        contradicting_refs=[
            {"source_type": "feature", "source_id": issue, "signal": issue, "why": "primary_reflection"}
            for issue in issues
        ],
    )


def _extract_feature_names_from_llm(llm_result: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for signal in llm_result.get("key_signals", []) or []:
        if not isinstance(signal, dict):
            continue
        key = str(signal.get("signal") or "").strip()
        if not key:
            continue
        names.append(f"llm_signal:{key}")
    return _dedupe(names)


def _extract_protagonist_mentions(text: str) -> List[str]:
    matches: List[str] = []
    if not text:
        return matches
    for pattern in PROTAGONIST_LABEL_PATTERNS:
        matches.extend(pattern.findall(text))
    return _dedupe(matches)


def _normalize_confidence(value: Any, default: float) -> float:
    try:
        score = float(value)
    except Exception:
        return round(default, 2)
    if score > 1:
        score = score / 100.0
    return round(min(max(score, 0.0), 1.0), 2)


def _person_primary_decision(
    *,
    person_id: str,
    confidence: float,
    photo_ids: List[str],
    feature_names: List[str],
    reasoning: str,
) -> PrimaryDecision:
    evidence = build_evidence_payload(
        photo_ids=photo_ids,
        person_ids=[person_id],
        feature_names=feature_names,
    )
    return PrimaryDecision(
        mode="person_id",
        primary_person_id=person_id,
        confidence=confidence,
        evidence=evidence,
        reasoning=reasoning,
    )


def _photographer_mode_decision(
    *,
    confidence: float,
    feature_names: List[str],
    reasoning: str,
    contradicting_refs: List[Dict[str, Any]] | None = None,
) -> PrimaryDecision:
    evidence = build_evidence_payload(
        feature_names=feature_names,
        contradicting_refs=contradicting_refs,
    )
    return PrimaryDecision(
        mode="photographer_mode",
        primary_person_id=None,
        confidence=confidence,
        evidence=evidence,
        reasoning=reasoning,
    )


def _dedupe(values: List[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered
