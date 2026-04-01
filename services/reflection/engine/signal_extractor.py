"""Unified signal extraction.

Ported from evolution._extract_overlooked_signals + _tokenize_signal_text + _extract_structured_refs.
Provides deduplication against historical seen_signal_keys for Nightly convergence detection.

v2: When gt_notes/evidence_summary are empty, actively searches vlm_summaries and
    event_summaries for evidence that correlates with the GT value.
    gt_token is used only as a convergence key, never exposed as a user-facing signal.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SignalResult:
    signals: List[str] = field(default_factory=list)
    is_new: bool = False
    signal_key: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signals": list(self.signals),
            "is_new": self.is_new,
            "signal_key": self.signal_key,
        }


def _dedupe_non_empty(items: List[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for item in items:
        stripped = str(item).strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            result.append(stripped)
    return result


def _tokenize_signal_text(value: Any) -> List[str]:
    if isinstance(value, list):
        merged = " ".join(str(item) for item in value)
    else:
        merged = str(value or "")
    tokens = re.findall(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", merged)
    normalized = []
    for token in tokens:
        lowered = token.strip().lower()
        if lowered:
            normalized.append(lowered)
    return _dedupe_non_empty(normalized)


def _extract_structured_refs(text: str) -> List[str]:
    content = str(text or "")
    refs = re.findall(r"(EVT_\d+|photo_\d+|Person_\d+)", content)
    return _dedupe_non_empty(refs)


# Common EN→ZH keyword expansions for cross-language evidence matching.
# Not exhaustive — covers high-frequency GT value vocabulary.
_EN_ZH_EXPANSIONS: Dict[str, List[str]] = {
    "photography": ["摄影", "拍摄", "拍照", "拍"],
    "artistic": ["艺术", "美术"],
    "documentation": ["记录", "文档"],
    "personal": ["个人", "自己"],
    "eclectic": ["多元", "多样", "折衷"],
    "refined": ["精致", "优雅", "精细", "讲究"],
    "fashion": ["时尚", "潮流", "穿搭"],
    "cute": ["可爱", "萌"],
    "network": ["网络", "互联网"],
    "safety": ["安全", "防护"],
    "selfie": ["自拍"],
    "style": ["风格", "样式"],
    "interest": ["兴趣", "爱好"],
    "hobby": ["爱好", "兴趣"],
    "sport": ["运动", "体育"],
    "food": ["美食", "食物", "吃"],
    "travel": ["旅行", "旅游", "出行"],
    "cat": ["猫", "喵"],
    "dog": ["狗", "犬"],
    "pet": ["宠物"],
    "cooking": ["烹饪", "做饭", "料理"],
    "reading": ["阅读", "读书"],
    "music": ["音乐", "歌"],
    "outdoor": ["户外", "室外"],
    "indoor": ["室内"],
    "snow": ["雪"],
    "work": ["工作", "上班"],
    "study": ["学习", "学业"],
    "campus": ["校园", "学校"],
}


def _expand_tokens_to_search_keywords(gt_tokens: List[str]) -> List[str]:
    """Expand English GT tokens into both English and Chinese search keywords."""
    keywords: List[str] = []
    for token in gt_tokens:
        # Split compound tokens: "artistic_photography" → ["artistic", "photography"]
        sub_tokens = token.lower().replace("_", " ").split()
        for st in sub_tokens:
            if len(st) >= 3:
                keywords.append(st)
                # Add Chinese expansions
                for zh in _EN_ZH_EXPANSIONS.get(st, []):
                    keywords.append(zh)
    return _dedupe_non_empty(keywords)


def _search_evidence_in_summaries(
    gt_tokens: List[str],
    vlm_summaries: Dict[str, str],
    event_summaries: Dict[str, str],
) -> List[str]:
    """Search vlm and event summaries for content related to GT tokens.

    Supports cross-language matching: expands English GT tokens into Chinese keywords
    since VLM/event summaries are typically in Chinese.

    Returns evidence_ref signals like 'evidence_ref:photo_042' or 'evidence_ref:EVT_015'.
    """
    if not gt_tokens:
        return []

    keywords = _expand_tokens_to_search_keywords(gt_tokens)
    if not keywords:
        return []

    found: List[str] = []
    seen: set[str] = set()

    # Score and rank VLM summaries by keyword match count
    scored_photos: List[tuple] = []
    for photo_id, summary in vlm_summaries.items():
        if not summary:
            continue
        summary_lower = summary.lower()
        matched = sum(1 for kw in keywords if kw in summary_lower)
        if matched > 0:
            scored_photos.append((matched, photo_id))

    # Sort by match count descending, take top 3
    scored_photos.sort(key=lambda x: x[0], reverse=True)
    for _, photo_id in scored_photos[:3]:
        ref = f"evidence_ref:{photo_id}"
        if ref not in seen:
            seen.add(ref)
            found.append(ref)

    # Score and rank event summaries
    scored_events: List[tuple] = []
    for event_id, summary in event_summaries.items():
        if not summary:
            continue
        summary_lower = summary.lower()
        matched = sum(1 for kw in keywords if kw in summary_lower)
        if matched > 0:
            scored_events.append((matched, event_id))

    scored_events.sort(key=lambda x: x[0], reverse=True)
    for _, event_id in scored_events[:2]:
        ref = f"evidence_ref:{event_id}"
        if ref not in seen:
            seen.add(ref)
            found.append(ref)

    return found


def extract_signals(
    *,
    field_key: str,
    output_value: Any,
    gt_value: Any,
    gt_notes: str = "",
    evidence_summary: str = "",
    accuracy_note: str = "",
    seen_signal_keys: List[str] | None = None,
    vlm_summaries: Dict[str, str] | None = None,
    event_summaries: Dict[str, str] | None = None,
    excluded_signal_keys: List[str] | None = None,
) -> SignalResult:
    """Extract overlooked signals by comparing system output with GT.

    Priority order:
    1. evidence_ref from gt_notes/evidence_summary (structured refs like photo_/EVT_)
    2. evidence_ref from searching vlm/event summaries using GT value tokens
    3. semantic_anchor by field domain keywords
    4. gt_token as LAST RESORT only for convergence detection (internal key),
       NOT exposed as a user-facing signal when real evidence exists.

    Returns signals found, whether any are new (vs seen_signal_keys),
    and a stable signal_key for deduplication.
    """
    signals: List[str] = []

    # --- Phase 1: Extract structured refs from GT notes/evidence ---
    notes_blob = " ".join(
        str(part)
        for part in [gt_notes, accuracy_note, evidence_summary]
        if part
    )
    for ref in _extract_structured_refs(notes_blob):
        signal = f"evidence_ref:{ref}"
        if signal not in signals:
            signals.append(signal)
        if len(signals) >= 6:
            break

    # --- Phase 2: If no evidence from notes, search vlm/event summaries ---
    output_tokens = set(_tokenize_signal_text(output_value))
    gt_tokens = [
        token
        for token in _tokenize_signal_text(gt_value)
        if token not in output_tokens
    ]

    if not signals and (vlm_summaries or event_summaries):
        evidence_signals = _search_evidence_in_summaries(
            gt_tokens=gt_tokens,
            vlm_summaries=vlm_summaries or {},
            event_summaries=event_summaries or {},
        )
        signals.extend(evidence_signals)

    # --- Phase 3: Semantic anchors by field domain ---
    lower_field = field_key.lower()
    notes_lower = notes_blob.lower()
    _ANCHORS = [
        ("education", ("学生", "校园", "school", "college", "university"), "student_or_campus_context"),
        ("brand_preference", ("品牌", "消费", "喜茶", "coffee", "tea"), "brand_consumption_pattern"),
        ("motivation_shift", ("同步", "近期", "时间", "trend", "shift"), "temporal_shift_signal"),
        ("career", ("工作", "职业", "intern", "job", "company"), "career_context"),
        ("location_anchors", ("城市", "地址", "address", "city", "district"), "location_context"),
        ("interests", ("爱好", "兴趣", "hobby", "interest"), "interest_context"),
    ]
    for field_hint, keywords, anchor_name in _ANCHORS:
        if field_hint in lower_field and any(kw in notes_lower for kw in keywords):
            signals.append(f"semantic_anchor:{anchor_name}")

    # --- Phase 4: gt_token as convergence key only ---
    # gt_token is ONLY used when no real evidence was found,
    # and primarily serves as a stable key for convergence detection.
    # It is appended last so real evidence takes priority in signals list.
    gt_token_signals = [f"gt_token:{token}" for token in gt_tokens[:3]]

    # For signal_key computation, include gt_tokens to ensure convergence detection works.
    # But for user-facing signals list, only include gt_tokens if nothing else was found.
    if not signals:
        signals = gt_token_signals

    signals = _dedupe_non_empty(signals)[:6]

    # Compute signal_key: use gt_tokens for stable dedup regardless of evidence source
    all_key_parts = (signals if signals else gt_token_signals)[:3]
    signal_key = "|".join(all_key_parts).strip()
    excluded = set(excluded_signal_keys or [])
    is_new = bool(signal_key) and (
        seen_signal_keys is None or signal_key not in (seen_signal_keys or [])
    ) and signal_key not in excluded

    return SignalResult(
        signals=signals,
        is_new=is_new,
        signal_key=signal_key,
    )
