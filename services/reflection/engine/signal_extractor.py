"""Unified signal extraction.

Ported from evolution._extract_overlooked_signals + _tokenize_signal_text + _extract_structured_refs.
Provides deduplication against historical seen_signal_keys for Nightly convergence detection.
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


def extract_signals(
    *,
    field_key: str,
    output_value: Any,
    gt_value: Any,
    gt_notes: str = "",
    evidence_summary: str = "",
    accuracy_note: str = "",
    seen_signal_keys: List[str] | None = None,
) -> SignalResult:
    """Extract overlooked signals by comparing system output with GT.

    Returns signals found, whether any are new (vs seen_signal_keys),
    and a stable signal_key for deduplication.
    """
    output_tokens = set(_tokenize_signal_text(output_value))
    gt_tokens = [
        token
        for token in _tokenize_signal_text(gt_value)
        if token not in output_tokens
    ]
    signals: List[str] = [f"gt_token:{token}" for token in gt_tokens[:3]]

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

    # Semantic anchors by field domain
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

    signals = _dedupe_non_empty(signals)[:6]

    # Compute signal_key and novelty
    signal_key = "|".join(signals[:3]).strip()
    is_new = bool(signal_key) and (
        seen_signal_keys is None or signal_key not in (seen_signal_keys or [])
    )

    return SignalResult(
        signals=signals,
        is_new=is_new,
        signal_key=signal_key,
    )
