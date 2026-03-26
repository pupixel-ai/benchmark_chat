from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List

from config import PROFILE_LLM_PROVIDER
from services.memory_pipeline.profile_llm import OpenRouterProfileLLMProcessor


ALLOWED_MATCH_GRADES = {
    "exact_match",
    "close_match",
    "partial_match",
    "mismatch",
    "missing_prediction",
    "missing_gt",
}

TOKEN_ALIASES = {
    "大学生": "college_student",
    "本科生": "college_student",
    "college student": "college_student",
    "college_student": "college_student",
    "学生": "student",
    "student": "student",
    "研究生": "master_student",
    "硕士": "master_student",
    "master student": "master_student",
    "master_student": "master_student",
    "音乐": "music",
    "music": "music",
    "电影": "movie",
    "movie": "movie",
    "movies": "movie",
    "游戏": "game",
    "game": "game",
    "gaming": "game",
}

FIELD_HIERARCHIES = {
    "long_term_facts.social_identity.education": {
        "college_student": "student",
        "master_student": "student",
        "phd_student": "student",
    }
}

LIST_LIKE_FIELDS = {
    "long_term_facts.hobbies.interests",
    "long_term_facts.hobbies.frequent_activities",
    "short_term_facts.recent_interests",
}

TEXT_STYLE_FIELDS = {
    "long_term_expression.attitude_style",
    "long_term_expression.aesthetic_tendency",
    "long_term_expression.visual_creation_style",
    "short_term_expression.current_mood",
    "short_term_expression.mental_state",
    "short_term_expression.motivation_shift",
    "short_term_expression.stress_signal",
    "short_term_expression.social_energy",
}


def compare_profile_field_values(
    *,
    field_key: str,
    predicted_value: Any,
    gt_value: Any,
    llm_processor: Any | None = None,
) -> Dict[str, Any]:
    if _is_missing(predicted_value):
        return {
            "grade": "missing_prediction",
            "score": 0.0,
            "method": "rule_missing_prediction",
            "output_value": predicted_value,
            "gt_value": gt_value,
        }
    if _is_missing(gt_value):
        return {
            "grade": "missing_gt",
            "score": 0.0,
            "method": "rule_missing_gt",
            "output_value": predicted_value,
            "gt_value": gt_value,
        }

    exact_result = _exact_or_hierarchy_match(field_key=field_key, predicted_value=predicted_value, gt_value=gt_value)
    if exact_result:
        return {
            **exact_result,
            "output_value": predicted_value,
            "gt_value": gt_value,
        }

    list_result = _list_overlap_match(field_key=field_key, predicted_value=predicted_value, gt_value=gt_value)
    if list_result:
        return {
            **list_result,
            "output_value": predicted_value,
            "gt_value": gt_value,
        }

    text_result = _text_similarity_match(
        field_key=field_key,
        predicted_value=predicted_value,
        gt_value=gt_value,
        llm_processor=llm_processor,
    )
    if text_result:
        return {
            **text_result,
            "output_value": predicted_value,
            "gt_value": gt_value,
        }

    return {
        "grade": "mismatch",
        "score": 0.0,
        "method": "rule_no_match",
        "output_value": predicted_value,
        "gt_value": gt_value,
    }


def _exact_or_hierarchy_match(*, field_key: str, predicted_value: Any, gt_value: Any) -> Dict[str, Any] | None:
    predicted_token = _normalize_scalar(predicted_value)
    gt_token = _normalize_scalar(gt_value)
    if predicted_token and gt_token and predicted_token == gt_token:
        return {
            "grade": "exact_match",
            "score": 1.0,
            "method": "rule_exact",
        }

    hierarchy = FIELD_HIERARCHIES.get(field_key, {})
    if predicted_token and gt_token and _is_hierarchy_neighbor(predicted_token, gt_token, hierarchy):
        return {
            "grade": "close_match",
            "score": 0.8,
            "method": "rule_hierarchy",
        }
    return None


def _list_overlap_match(*, field_key: str, predicted_value: Any, gt_value: Any) -> Dict[str, Any] | None:
    if field_key not in LIST_LIKE_FIELDS and not isinstance(predicted_value, list) and not isinstance(gt_value, list):
        return None

    predicted_tokens = set(_normalize_multi_value(predicted_value))
    gt_tokens = set(_normalize_multi_value(gt_value))
    if not predicted_tokens or not gt_tokens:
        return None
    if predicted_tokens == gt_tokens:
        return {
            "grade": "exact_match",
            "score": 1.0,
            "method": "rule_set_exact",
        }
    if predicted_tokens < gt_tokens or gt_tokens < predicted_tokens:
        overlap = predicted_tokens & gt_tokens
        precision = len(overlap) / len(predicted_tokens)
        recall = len(overlap) / len(gt_tokens)
        completeness_score = recall if predicted_tokens < gt_tokens else precision
        return {
            "grade": "partial_match",
            "score": round(completeness_score, 4),
            "method": "rule_set_overlap",
        }

    overlap = predicted_tokens & gt_tokens
    if not overlap:
        return {
            "grade": "mismatch",
            "score": 0.0,
            "method": "rule_set_overlap",
        }

    precision = len(overlap) / len(predicted_tokens)
    recall = len(overlap) / len(gt_tokens)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    if f1 >= 0.67:
        grade = "close_match"
    else:
        grade = "partial_match"
    return {
        "grade": grade,
        "score": round(f1, 4),
        "method": "rule_set_overlap",
    }


def _text_similarity_match(
    *,
    field_key: str,
    predicted_value: Any,
    gt_value: Any,
    llm_processor: Any | None = None,
) -> Dict[str, Any] | None:
    if field_key not in TEXT_STYLE_FIELDS:
        return None

    predicted_tokens = set(_tokenize_text(predicted_value))
    gt_tokens = set(_tokenize_text(gt_value))
    if not predicted_tokens or not gt_tokens:
        return None

    overlap = predicted_tokens & gt_tokens
    union = predicted_tokens | gt_tokens
    score = round(len(overlap) / len(union), 4) if union else 0.0
    if score >= 0.67:
        return {"grade": "close_match", "score": score, "method": "rule_text_overlap"}
    if score >= 0.34:
        llm_result = _judge_text_with_llm(
            field_key=field_key,
            predicted_value=predicted_value,
            gt_value=gt_value,
            llm_processor=llm_processor,
        )
        if llm_result:
            return llm_result
        return {"grade": "partial_match", "score": score, "method": "rule_text_overlap"}
    return {"grade": "mismatch", "score": score, "method": "rule_text_overlap"}


def _judge_text_with_llm(*, field_key: str, predicted_value: Any, gt_value: Any, llm_processor: Any | None) -> Dict[str, Any] | None:
    processor = llm_processor or _build_llm_processor()
    if processor is None:
        return None

    prompt = (
        "你是 GT 近似匹配判定器。"
        "只能从 exact_match, close_match, partial_match, mismatch 中选一个。"
        "只返回 JSON，字段必须是 grade, score, reason。"
        + json.dumps(
            {
                "field_key": field_key,
                "predicted_value": predicted_value,
                "gt_value": gt_value,
            },
            ensure_ascii=False,
        )
    )
    try:
        response = processor._call_llm_via_official_api(prompt, response_mime_type="application/json")
    except Exception:
        return None
    if not isinstance(response, dict):
        return None
    grade = str(response.get("grade") or "").strip()
    if grade not in ALLOWED_MATCH_GRADES:
        return None
    try:
        score = float(response.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    return {
        "grade": grade,
        "score": round(max(0.0, min(1.0, score)), 4),
        "method": "llm_semantic_match",
        "reason": str(response.get("reason") or "").strip(),
    }


def _build_llm_processor() -> Any | None:
    if PROFILE_LLM_PROVIDER != "openrouter":
        return None
    try:
        return OpenRouterProfileLLMProcessor()
    except Exception:
        return None


def _is_hierarchy_neighbor(predicted_token: str, gt_token: str, hierarchy: Dict[str, str]) -> bool:
    return hierarchy.get(predicted_token) == gt_token or hierarchy.get(gt_token) == predicted_token


def _normalize_scalar(value: Any) -> str:
    if isinstance(value, list):
        normalized = _normalize_multi_value(value)
        return normalized[0] if len(normalized) == 1 else ""
    if value is None:
        return ""
    text = str(value).strip().lower()
    return TOKEN_ALIASES.get(text, text)


def _normalize_multi_value(value: Any) -> List[str]:
    if isinstance(value, list):
        parts = value
    elif isinstance(value, str):
        parts = [item for item in re.split(r"[、，,/|;；]\s*", value) if item]
    else:
        parts = [value]
    normalized: List[str] = []
    for item in parts:
        text = _normalize_scalar(item)
        if text:
            normalized.append(text)
    return list(dict.fromkeys(normalized))


def _tokenize_text(value: Any) -> List[str]:
    if value is None:
        return []
    text = str(value).strip().lower()
    if not text:
        return []
    raw_tokens = re.split(r"[\s,，、/|;；]+", text)
    normalized = []
    for token in raw_tokens:
        token = token.strip()
        if not token:
            continue
        normalized.append(TOKEN_ALIASES.get(token, token))
    return list(dict.fromkeys(normalized))


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return False
