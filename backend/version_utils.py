"""
Helpers for translating task version strings into numeric stage versions.
"""
from __future__ import annotations

import re
from typing import Dict, Optional, Tuple


_VERSION_PATTERN = re.compile(r"^v?(?P<body>[0-9][0-9.]*)(?:-(?P<channel>[A-Za-z0-9_-]+))?$")


def parse_numeric_version(version: str | None) -> Tuple[Optional[int], Optional[str]]:
    candidate = str(version or "").strip()
    if not candidate:
        return None, None
    match = _VERSION_PATTERN.match(candidate)
    if match:
        body = match.group("body") or ""
        channel = (match.group("channel") or "").strip().lower() or None
        digits = "".join(ch for ch in body if ch.isdigit())
        return (int(digits) if digits else None), channel
    digits = "".join(ch for ch in candidate if ch.isdigit())
    suffix_match = re.search(r"-([A-Za-z0-9_-]+)$", candidate)
    channel = suffix_match.group(1).strip().lower() if suffix_match else None
    return (int(digits) if digits else None), channel


def build_stage_version_matrix(task_version: str | None, result: dict | None = None) -> Dict[str, Optional[int]]:
    numeric_version, _ = parse_numeric_version(task_version)
    payload = result if isinstance(result, dict) else {}
    memory = payload.get("memory") if isinstance(payload.get("memory"), dict) else {}
    face_version = numeric_version if isinstance(payload.get("face_recognition"), dict) else None
    vlm_version = numeric_version if isinstance(memory, dict) and list(memory.get("vp1_observations", []) or []) else None
    lp1_version = numeric_version if isinstance(memory, dict) and list(memory.get("lp1_events", []) or []) else None
    lp2_version = numeric_version if isinstance(memory, dict) and list(memory.get("lp2_relationships", []) or []) else None
    lp3_version = numeric_version if isinstance(memory, dict) and isinstance(memory.get("lp3_profile"), dict) else None
    judge_version = numeric_version if isinstance(memory, dict) and isinstance(memory.get("lp3_profile"), dict) else None
    return {
        "pipeline_version": numeric_version,
        "face_version": face_version,
        "vlm_version": vlm_version,
        "lp1_version": lp1_version,
        "lp2_version": lp2_version,
        "lp3_version": lp3_version,
        "judge_version": judge_version,
    }
