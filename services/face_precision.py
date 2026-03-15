"""
人脸匹配精度增强逻辑。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from config import (
    FACE_POSE_PROFILE_YAW_THRESHOLD,
    FACE_PROFILE_RESCUE_DELTA,
    FACE_PROFILE_RESCUE_MARGIN,
    FACE_PROFILE_RESCUE_MIN_QUALITY,
    FACE_SAME_PHOTO_MATCH_THRESHOLD,
    FACE_MATCH_MARGIN_THRESHOLD,
    FACE_MATCH_MIN_QUALITY_GRAY_ZONE,
    FACE_MATCH_THRESHOLD_PATH,
    FACE_MATCH_WEAK_DELTA,
    FACE_SIM_THRESHOLD,
)
from utils import load_json

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - cv2 is expected in runtime
    cv2 = None


@dataclass(frozen=True)
class CandidateSummary:
    person_id: str
    best_similarity: float
    mean_similarity: float
    support_count: int


def load_strong_threshold(
    fallback: float = FACE_SIM_THRESHOLD,
    threshold_path: str = FACE_MATCH_THRESHOLD_PATH,
) -> float:
    payload = load_json(threshold_path)
    candidates = (
        payload.get("recommended_threshold"),
        payload.get("strong_threshold"),
        payload.get("mean_threshold"),
        payload.get("threshold"),
    )
    for candidate in candidates:
        try:
            return float(candidate)
        except (TypeError, ValueError):
            continue
    return float(fallback)


def compute_face_quality(image_pixels: object, bbox_xywh: Dict[str, int]) -> Dict[str, object]:
    pixels = np.asarray(image_pixels)
    height, width = pixels.shape[:2]

    x = max(0, int(bbox_xywh.get("x", 0)))
    y = max(0, int(bbox_xywh.get("y", 0)))
    w = max(0, int(bbox_xywh.get("w", 0)))
    h = max(0, int(bbox_xywh.get("h", 0)))
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    crop = pixels[y:y2, x:x2]
    if crop.size == 0:
        return {
            "quality_score": 0.0,
            "quality_flags": ["empty_crop"],
            "blur_variance": 0.0,
            "brightness_mean": 0.0,
            "contrast_std": 0.0,
        }

    gray = _to_gray(crop)
    min_dim = float(min(x2 - x, y2 - y))
    laplacian_variance = _laplacian_variance(gray)
    brightness_mean = float(np.mean(gray))
    contrast_std = float(np.std(gray))
    edge_padding = float(min(x, y, max(0, width - x2), max(0, height - y2)))

    size_score = _clip01(min_dim / 160.0)
    blur_score = _clip01(laplacian_variance / 140.0)
    balance_score = _clip01(1.0 - abs(brightness_mean - 128.0) / 128.0)
    contrast_score = _clip01(contrast_std / 64.0)
    exposure_score = (balance_score * 0.7) + (contrast_score * 0.3)
    edge_score = _clip01(edge_padding / 12.0)

    quality_score = round(
        (size_score * 0.30) + (blur_score * 0.35) + (exposure_score * 0.20) + (edge_score * 0.15),
        4,
    )

    flags: List[str] = []
    if min_dim < 64:
        flags.append("small_face")
    if laplacian_variance < 45.0:
        flags.append("blurry")
    if brightness_mean < 45.0:
        flags.append("underexposed")
    elif brightness_mean > 210.0:
        flags.append("overexposed")
    if contrast_std < 20.0:
        flags.append("low_contrast")
    if edge_padding < 3.0:
        flags.append("edge_clipped")

    return {
        "quality_score": quality_score,
        "quality_flags": flags,
        "blur_variance": round(laplacian_variance, 4),
        "brightness_mean": round(brightness_mean, 4),
        "contrast_std": round(contrast_std, 4),
    }


def aggregate_candidate_matches(
    matches: Iterable[object],
    faiss_person_map: Dict[int, str],
    pending_person_map: Dict[int, str],
) -> List[CandidateSummary]:
    grouped: Dict[str, List[float]] = {}
    for match in matches:
        faiss_id = getattr(match, "faiss_id", None)
        score = getattr(match, "score", None)
        if faiss_id is None or score is None:
            continue
        person_id = faiss_person_map.get(faiss_id) or pending_person_map.get(faiss_id)
        if not person_id:
            continue
        grouped.setdefault(person_id, []).append(float(score))

    summaries = [
        CandidateSummary(
            person_id=person_id,
            best_similarity=max(scores),
            mean_similarity=sum(scores) / len(scores),
            support_count=len(scores),
        )
        for person_id, scores in grouped.items()
    ]
    summaries.sort(
        key=lambda item: (item.best_similarity, item.support_count, item.mean_similarity, item.person_id),
        reverse=True,
    )
    return summaries


def decide_match(
    candidates: List[CandidateSummary],
    quality_score: float,
    *,
    strong_threshold: float,
    weak_threshold: Optional[float] = None,
    margin_threshold: float = FACE_MATCH_MARGIN_THRESHOLD,
    min_quality_for_gray_zone: float = FACE_MATCH_MIN_QUALITY_GRAY_ZONE,
    pose_bucket: str | None = None,
    pose_yaw: float | None = None,
    profile_rescue_delta: float = FACE_PROFILE_RESCUE_DELTA,
    profile_rescue_margin: float = FACE_PROFILE_RESCUE_MARGIN,
    profile_rescue_min_quality: float = FACE_PROFILE_RESCUE_MIN_QUALITY,
) -> Dict[str, object]:
    weak_threshold = strong_threshold - FACE_MATCH_WEAK_DELTA if weak_threshold is None else weak_threshold
    if not candidates:
        return {
            "person_id": None,
            "decision": "new_person",
            "reason": "没有可复用候选",
            "best_similarity": 0.0,
            "runner_up_similarity": None,
        }

    best = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None
    runner_up_similarity = runner_up.best_similarity if runner_up else None
    margin = (
        best.best_similarity - runner_up_similarity
        if runner_up_similarity is not None
        else best.best_similarity
    )

    if best.best_similarity >= strong_threshold and margin >= margin_threshold:
        return {
            "person_id": best.person_id,
            "decision": "strong_match",
            "reason": (
                f"强匹配: best={best.best_similarity:.3f}, margin={margin:.3f}, "
                f"support={best.support_count}"
            ),
            "best_similarity": best.best_similarity,
            "runner_up_similarity": runner_up_similarity,
        }

    if _should_allow_profile_rescue(
        pose_bucket=pose_bucket,
        pose_yaw=pose_yaw,
        quality_score=quality_score,
        best_similarity=best.best_similarity,
        margin=margin,
        strong_threshold=strong_threshold,
        profile_rescue_delta=profile_rescue_delta,
        profile_rescue_margin=profile_rescue_margin,
        profile_rescue_min_quality=profile_rescue_min_quality,
    ):
        return {
            "person_id": best.person_id,
            "decision": "profile_rescue_match",
            "reason": (
                f"侧脸补救: best={best.best_similarity:.3f}, margin={margin:.3f}, "
                f"quality={quality_score:.3f}, yaw={0.0 if pose_yaw is None else pose_yaw:.2f}"
            ),
            "best_similarity": best.best_similarity,
            "runner_up_similarity": runner_up_similarity,
        }

    if (
        quality_score >= min_quality_for_gray_zone
        and best.support_count >= 2
        and best.mean_similarity >= weak_threshold
    ):
        return {
            "person_id": best.person_id,
            "decision": "gray_match",
            "reason": (
                f"灰区合并: mean={best.mean_similarity:.3f}, support={best.support_count}, "
                f"quality={quality_score:.3f}"
            ),
            "best_similarity": best.best_similarity,
            "runner_up_similarity": runner_up_similarity,
        }

    return {
        "person_id": None,
        "decision": "new_person_from_ambiguity",
        "reason": (
            f"候选存在但不够稳: best={best.best_similarity:.3f}, "
            f"mean={best.mean_similarity:.3f}, support={best.support_count}, quality={quality_score:.3f}"
        ),
        "best_similarity": best.best_similarity,
        "runner_up_similarity": runner_up_similarity,
    }


def filter_same_photo_candidates(
    candidates: Iterable[CandidateSummary],
    seen_person_ids: Iterable[str],
    *,
    same_photo_match_threshold: float = FACE_SAME_PHOTO_MATCH_THRESHOLD,
) -> List[CandidateSummary]:
    seen = set(seen_person_ids)
    if not seen:
        return list(candidates)
    return [
        candidate
        for candidate in candidates
        if candidate.person_id not in seen or candidate.best_similarity >= same_photo_match_threshold
    ]


def _should_allow_profile_rescue(
    *,
    pose_bucket: str | None,
    pose_yaw: float | None,
    quality_score: float,
    best_similarity: float,
    margin: float,
    strong_threshold: float,
    profile_rescue_delta: float,
    profile_rescue_margin: float,
    profile_rescue_min_quality: float,
) -> bool:
    if pose_bucket not in {"left_profile", "right_profile"}:
        return False
    if pose_yaw is not None and abs(float(pose_yaw)) < FACE_POSE_PROFILE_YAW_THRESHOLD:
        return False
    if quality_score < profile_rescue_min_quality:
        return False
    if best_similarity < strong_threshold - profile_rescue_delta:
        return False
    if margin < profile_rescue_margin:
        return False
    return True


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _laplacian_variance(gray: np.ndarray) -> float:
    if cv2 is None:
        grad_y, grad_x = np.gradient(gray.astype("float32"))
        return float(np.var(grad_x) + np.var(grad_y))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _to_gray(crop: np.ndarray) -> np.ndarray:
    if crop.ndim == 2:
        return crop
    if cv2 is not None:
        return cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return np.dot(crop[..., :3], [0.299, 0.587, 0.114]).astype("uint8")
