from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import os
from config import PROFILE_LLM_PROVIDER, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
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
    evidence_count: int | None = None,
) -> Dict[str, Any]:
    if _is_missing(predicted_value):
        result = {
            "grade": "missing_prediction",
            "score": 0.0,
            "method": "rule_missing_prediction",
            "output_value": predicted_value,
            "gt_value": gt_value,
        }
        result["quality_dimensions"] = _compute_quality_dimensions(
            grade="missing_prediction", predicted_value=predicted_value, gt_value=gt_value,
            field_key=field_key, evidence_count=evidence_count,
        )
        return result
    if _is_missing(gt_value):
        return {
            "grade": "missing_gt",
            "score": 0.0,
            "method": "rule_missing_gt",
            "output_value": predicted_value,
            "gt_value": gt_value,
        }

    def _finalize(result: Dict[str, Any]) -> Dict[str, Any]:
        result["output_value"] = predicted_value
        result["gt_value"] = gt_value
        result["quality_dimensions"] = _compute_quality_dimensions(
            grade=result.get("grade", "mismatch"),
            predicted_value=predicted_value,
            gt_value=gt_value,
            field_key=field_key,
            evidence_count=evidence_count,
        )
        return result

    exact_result = _exact_or_hierarchy_match(field_key=field_key, predicted_value=predicted_value, gt_value=gt_value)
    if exact_result:
        return _finalize(exact_result)

    list_result = _list_overlap_match(field_key=field_key, predicted_value=predicted_value, gt_value=gt_value)
    if list_result:
        return _finalize(list_result)

    text_result = _text_similarity_match(
        field_key=field_key,
        predicted_value=predicted_value,
        gt_value=gt_value,
        llm_processor=llm_processor,
    )
    if text_result:
        return _finalize(text_result)

    llm_fallback = _judge_text_with_llm(
        field_key=field_key,
        predicted_value=predicted_value,
        gt_value=gt_value,
        llm_processor=llm_processor,
    )
    if llm_fallback:
        return _finalize(llm_fallback)

    return _finalize({
        "grade": "mismatch",
        "score": 0.0,
        "method": "rule_no_match",
    })


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
    # 非精确匹配的列表交给 LLM 语义判定
    return None


def _text_similarity_match(
    *,
    field_key: str,
    predicted_value: Any,
    gt_value: Any,
    llm_processor: Any | None = None,
) -> Dict[str, Any] | None:

    predicted_tokens = set(_tokenize_text(predicted_value))
    gt_tokens = set(_tokenize_text(gt_value))
    if not predicted_tokens or not gt_tokens:
        return None

    overlap = predicted_tokens & gt_tokens
    union = predicted_tokens | gt_tokens
    score = round(len(overlap) / len(union), 4) if union else 0.0
    if score >= 0.67:
        return {"grade": "close_match", "score": score, "method": "rule_text_overlap"}
    # token overlap 不高时，用 LLM 语义判定
    llm_result = _judge_text_with_llm(
        field_key=field_key,
        predicted_value=predicted_value,
        gt_value=gt_value,
        llm_processor=llm_processor,
    )
    if llm_result:
        return llm_result
    if score >= 0.34:
        return {"grade": "partial_match", "score": score, "method": "rule_text_overlap"}
    return {"grade": "mismatch", "score": score, "method": "rule_text_overlap"}


def _load_human_grade_examples() -> List[Dict[str, Any]]:
    """加载人工修正过的打分案例，作为 LLM 判分参考。"""
    overrides_path = Path(__file__).resolve().parent.parent / "memory" / "reflection" / "gt_grade_overrides.json"
    if not overrides_path.exists():
        return []
    try:
        overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    # 从 gt_comparisons 收集修正案例的上下文
    examples: List[Dict[str, Any]] = []
    reflection_dir = overrides_path.parent
    seen_fields: set[str] = set()
    for comp_file in reflection_dir.glob("gt_comparisons_*.jsonl"):
        for line in comp_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            cr = rec.get("comparison_result", {})
            if not cr.get("human_override"):
                continue
            fk = rec.get("field_key", "")
            # 每个字段最多取 1 条案例，避免 prompt 过长
            if fk in seen_fields:
                continue
            seen_fields.add(fk)
            examples.append({
                "field_key": fk,
                "predicted": cr.get("output_value"),
                "gt": cr.get("gt_value"),
                "human_grade": cr.get("grade", ""),
                "original_grade": cr.get("original_grade", ""),
            })
            if len(examples) >= 8:
                break
    return examples


def _judge_text_with_llm(*, field_key: str, predicted_value: Any, gt_value: Any, llm_processor: Any | None) -> Dict[str, Any] | None:
    processor = llm_processor or _build_llm_processor()
    if processor is None:
        return None

    # 构建人工修正案例作为参考
    examples = _load_human_grade_examples()
    examples_text = ""
    if examples:
        relevant = [e for e in examples if e["field_key"] == field_key]
        others = [e for e in examples if e["field_key"] != field_key][:3]
        selected = (relevant + others)[:5]
        if selected:
            examples_text = (
                "\n\n以下是人工审核过的打分案例供你参考（人工判定优先于规则）：\n"
                + json.dumps(selected, ensure_ascii=False)
            )

    prompt = (
        "你是 GT 近似匹配判定器。只能从 exact_match, close_match, partial_match, mismatch 中选一个。"
        "判定规则："
        "1. 语义等价（如 student 和 学生，domestic_only 和 中国境内）→ exact_match"
        "2. 系统输出比 GT 更具体（如 GT=student，输出=东华大学本科在读）→ close_match，因为输出包含了 GT 的语义且更丰富"
        "3. 系统输出和 GT 有部分重叠但不完全覆盖 → partial_match"
        "4. 语义完全不同 → mismatch"
        + examples_text
        + "\n只返回 JSON，字段必须是 grade, score, reason。"
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


# ── P1: 质量维度 ──────────────────────────────────────────────

_EVIDENCE_DENSITY_BASELINES: Dict[str, int] = {
    "identity": 3, "social_identity": 3, "material": 4, "geography": 3,
    "time": 3, "relationships": 4, "hobbies": 4, "physiology": 2,
    "expression": 3, "short_term": 3,
}


def _compute_quality_dimensions(
    *,
    grade: str,
    predicted_value: Any,
    gt_value: Any,
    field_key: str,
    evidence_count: int | None = None,
) -> Dict[str, float]:
    is_bad = grade in ("mismatch", "missing_prediction")
    dims: Dict[str, float] = {}

    # specificity: 输出有多细粒度（mismatch 时归零）
    if is_bad:
        dims["specificity"] = 0.0
    else:
        pred_tokens = _normalize_multi_value(predicted_value)
        pred_char_len = sum(len(t) for t in pred_tokens)
        dims["specificity"] = round(min(1.0, pred_char_len / 20.0), 3)

    # coverage_breadth: GT 语义空间覆盖了多少
    grade_coverage = {"exact_match": 1.0, "close_match": 0.8, "partial_match": 0.5, "mismatch": 0.0, "missing_prediction": 0.0}
    dims["coverage_breadth"] = grade_coverage.get(grade, 0.0)

    # evidence_density: 证据支撑密度（mismatch 时归零，可选）
    if evidence_count is not None:
        if is_bad:
            dims["evidence_density"] = 0.0
        else:
            domain = field_key.split(".")[1] if "." in field_key else "short_term"
            baseline = _EVIDENCE_DENSITY_BASELINES.get(domain, 3)
            dims["evidence_density"] = round(min(1.0, evidence_count / baseline), 3)

    return dims


def _build_llm_processor() -> Any | None:
    try:
        gt_model = os.getenv("GT_MATCHER_MODEL", "").strip()
        if gt_model:
            return OpenRouterProfileLLMProcessor(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
                model=gt_model,
            )
        return OpenRouterProfileLLMProcessor()
    except Exception:
        return None


def _is_hierarchy_neighbor(predicted_token: str, gt_token: str, hierarchy: Dict[str, str]) -> bool:
    return hierarchy.get(predicted_token) == gt_token or hierarchy.get(gt_token) == predicted_token


def _flatten_complex_value(value: Any) -> Any:
    """将复杂结构（dict/嵌套对象）展平为可比较的字符串或列表。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        # 尝试提取有意义的文本：遍历所有值，收集字符串和列表
        texts: list[str] = []
        for v in value.values():
            flat = _flatten_complex_value(v)
            if isinstance(flat, str) and flat:
                texts.append(flat)
            elif isinstance(flat, list):
                for item in flat:
                    if isinstance(item, str) and item:
                        texts.append(item)
        return texts if texts else str(value)
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            flat = _flatten_complex_value(item)
            if isinstance(flat, str) and flat:
                result.append(flat)
            elif isinstance(flat, list):
                result.extend(flat)
        return result
    return str(value)


def _normalize_scalar(value: Any) -> str:
    if isinstance(value, dict):
        value = _flatten_complex_value(value)
    if isinstance(value, list):
        normalized = _normalize_multi_value(value)
        return normalized[0] if len(normalized) == 1 else ""
    if value is None:
        return ""
    text = str(value).strip().lower()
    return TOKEN_ALIASES.get(text, text)


def _normalize_multi_value(value: Any) -> List[str]:
    if isinstance(value, dict):
        value = _flatten_complex_value(value)
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
