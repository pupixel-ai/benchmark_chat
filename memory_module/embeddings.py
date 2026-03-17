"""Shared embedding helpers for graph nodes and evidence segments."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime, timezone
from typing import Dict, List


DEFAULT_EMBEDDING_MODEL = "textual_stub_v1"
DEFAULT_EMBEDDING_VERSION = "v1"


def deterministic_vector(text: str, dim: int) -> List[float]:
    if dim <= 0:
        return []
    if not text:
        return [0.0] * dim

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: List[float] = []
    state = digest
    while len(values) < dim:
        for byte in state:
            values.append((byte / 127.5) - 1.0)
            if len(values) >= dim:
                break
        state = hashlib.sha256(state).digest()

    norm = math.sqrt(sum(value * value for value in values)) or 1.0
    return [round(value / norm, 6) for value in values[:dim]]


def cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
    right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
    return dot / (left_norm * right_norm)


def embedding_text_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def build_embedding_payload(
    text: str,
    *,
    dim: int,
    model: str = DEFAULT_EMBEDDING_MODEL,
    version: str = DEFAULT_EMBEDDING_VERSION,
) -> Dict[str, object]:
    normalized = str(text or "").strip()
    return {
        "search_text": normalized,
        "embedding": deterministic_vector(normalized, dim),
        "embedding_model": model,
        "embedding_dim": dim,
        "embedding_version": version,
        "embedding_text_hash": embedding_text_hash(normalized),
        "embedding_updated_at": datetime.now(timezone.utc).isoformat(),
    }
