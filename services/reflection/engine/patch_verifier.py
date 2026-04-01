"""P3: 补丁效果自动验证 — 对比补丁前后的 grade/score，输出结构化 diff。"""
from __future__ import annotations

from typing import Any, Dict

_GRADE_PRIORITY = {
    "exact_match": 0,
    "close_match": 1,
    "partial_match": 2,
    "mismatch": 3,
    "missing_prediction": 4,
}

_GRADE_LABELS = {
    0: "exact_match",
    1: "close_match",
    2: "partial_match",
    3: "mismatch",
    4: "missing_prediction",
}


def compute_patch_effect(
    *,
    before_grade: str,
    before_score: float,
    after_grade: str,
    after_score: float,
) -> Dict[str, Any]:
    """比较补丁前后的 grade/score，返回结构化效果评估。"""
    before_rank = _GRADE_PRIORITY.get(before_grade, 3)
    after_rank = _GRADE_PRIORITY.get(after_grade, 3)
    grade_delta = before_rank - after_rank  # 正数 = 改善（rank 越小越好）
    score_delta = round(after_score - before_score, 4)
    grade_improved = grade_delta > 0
    outcome_improved = grade_delta > 0 or (grade_delta == 0 and score_delta > 0.1)

    if grade_improved:
        summary = f"grade 改善: {before_grade} → {after_grade}"
    elif grade_delta < 0:
        summary = f"grade 恶化: {before_grade} → {after_grade}"
    elif score_delta > 0.1:
        summary = f"grade 不变 ({after_grade})，score 提升 {score_delta:+.3f}"
    elif score_delta < -0.1:
        summary = f"grade 不变 ({after_grade})，score 下降 {score_delta:+.3f}"
    else:
        summary = f"无显著变化 ({after_grade}, score delta={score_delta:+.3f})"

    return {
        "grade_improved": grade_improved,
        "grade_delta": grade_delta,
        "score_delta": score_delta,
        "outcome_improved": outcome_improved,
        "before_grade": before_grade,
        "after_grade": after_grade,
        "summary": summary,
    }
