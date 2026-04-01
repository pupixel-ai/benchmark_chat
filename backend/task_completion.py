from __future__ import annotations

from typing import Any, Dict, List

from config import TASK_VERSION_V0327_DB, TASK_VERSION_V0327_DB_QUERY


_STRICT_COMPLETION_VERSIONS = {
    TASK_VERSION_V0327_DB,
    TASK_VERSION_V0327_DB_QUERY,
}


def missing_completion_outputs(task_version: str | None, result: Dict[str, Any] | None) -> List[str]:
    version = str(task_version or "").strip()
    if version not in _STRICT_COMPLETION_VERSIONS:
        return []

    payload = dict(result or {})
    missing: List[str] = []

    face_payload = payload.get("face_recognition")
    if not isinstance(face_payload, dict):
        missing.append("face_recognition")

    memory_payload = payload.get("memory")
    if not isinstance(memory_payload, dict):
        missing.extend(["vp1_observations", "lp1_events", "lp2_relationships", "lp3_profile"])
        return missing

    required_stage_keys = (
        "vp1_observations",
        "lp1_events",
        "lp2_relationships",
        "lp3_profile",
    )
    for key in required_stage_keys:
        if key not in memory_payload:
            missing.append(key)
            continue
        value = memory_payload.get(key)
        if key == "lp3_profile":
            if not isinstance(value, dict):
                missing.append(key)
        else:
            if not isinstance(value, list):
                missing.append(key)
    return missing


def ensure_completion_outputs(task_version: str | None, result: Dict[str, Any] | None) -> None:
    missing = missing_completion_outputs(task_version, result)
    if not missing:
        return
    raise ValueError(f"任务产物不完整，缺少: {', '.join(missing)}")
