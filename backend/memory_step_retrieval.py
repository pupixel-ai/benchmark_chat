from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List


STEP_ORDER = {
    "lp1_batch": 1,
    "lp2_relationship": 2,
    "lp3_profile": 3,
}


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if isinstance(payload, dict):
                    records.append(payload)
    except Exception:
        return []
    return records


def _artifact_payload(path: Path, task_dir: Path, task_id: str, asset_url_builder: Callable[[str, str], str]) -> Dict[str, Any]:
    try:
        relative_path = path.relative_to(task_dir).as_posix()
    except Exception:
        relative_path = path.name
    exists = path.exists()
    return {
        "relative_path": relative_path,
        "exists": exists,
        "asset_url": asset_url_builder(task_id, relative_path) if exists else None,
        "size_bytes": int(path.stat().st_size) if exists else 0,
        "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat() if exists else None,
    }


def _memory_stage(task: Dict[str, Any]) -> Dict[str, Any]:
    progress = task.get("progress") or {}
    stages = progress.get("stages") or {}
    memory_stage = stages.get("memory") or {}
    if isinstance(memory_stage, dict):
        return copy.deepcopy(memory_stage)
    return {}


def _step_rank(substage: str) -> int:
    return STEP_ORDER.get(str(substage or "").strip(), 0)


def _resolve_step_status(
    *,
    task_status: str,
    current_substage: str,
    step_substage: str,
    has_data: bool,
) -> str:
    current_rank = _step_rank(current_substage)
    step_rank = _step_rank(step_substage)

    if task_status == "completed":
        return "completed" if has_data else "pending"
    if task_status == "failed":
        if current_substage == step_substage:
            return "failed"
        if has_data and step_rank < current_rank:
            return "completed"
        return "pending"
    if task_status in {"running", "queued", "uploading", "draft"}:
        if current_substage == step_substage:
            return "running"
        if has_data and step_rank < current_rank:
            return "completed"
        return "pending"
    return "completed" if has_data else "pending"


def build_task_memory_steps_payload(
    task: Dict[str, Any],
    *,
    asset_url_builder: Callable[[str, str], str],
) -> Dict[str, Any]:
    task_id = str(task.get("task_id") or "").strip()
    task_dir = Path(str(task.get("task_dir") or ""))
    family_dir = task_dir / "output"
    result = task.get("result") or {}
    memory = result.get("memory") or {}
    memory_stage = _memory_stage(task)
    task_status = str(task.get("status") or "").strip()
    current_substage = str(memory_stage.get("substage") or "").strip()

    lp1_events = _load_json(family_dir / "lp1_events_compact.json")
    if not isinstance(lp1_events, list):
        lp1_events = list(memory.get("lp1_events", []) or [])
    lp1_batch_requests = _load_jsonl(family_dir / "lp1_batch_requests.jsonl")
    lp1_batch_outputs = _load_jsonl(family_dir / "lp1_batch_outputs.jsonl")
    lp1_attempt_pairs = {
        (
            str(item.get("batch_id") or "").strip(),
            int(item.get("attempt") or 1),
        )
        for item in lp1_batch_outputs
        if isinstance(item, dict)
    }
    lp1_batch_ids = {
        str(item.get("batch_id") or "").strip()
        for item in lp1_batch_outputs
        if isinstance(item, dict) and str(item.get("batch_id") or "").strip()
    }
    lp1_retry_attempts = [
        item
        for item in lp1_batch_outputs
        if isinstance(item, dict) and int(item.get("attempt") or 1) > 1
    ]
    lp1_last_attempt = lp1_batch_outputs[-1] if lp1_batch_outputs else {}

    lp2_relationships = _load_json(family_dir / "lp2_relationships.json")
    if not isinstance(lp2_relationships, list):
        lp2_relationships = _load_jsonl(family_dir / "lp2_relationships.jsonl")
    if not isinstance(lp2_relationships, list):
        lp2_relationships = list(memory.get("lp2_relationships", []) or [])

    lp3_profile = _load_json(family_dir / "lp3_profile.json")
    if not isinstance(lp3_profile, dict):
        lp3_profile = dict(memory.get("lp3_profile") or {})

    llm_failures = _load_jsonl(family_dir / "llm_failures.jsonl")
    lp1_parse_failures = _load_json(family_dir / "lp1_parse_failures.json")
    if not isinstance(lp1_parse_failures, list):
        lp1_parse_failures = []
    lp2_failures = [item for item in llm_failures if str(item.get("step") or "").strip() == "lp2_relationship"]
    lp3_failures = [item for item in llm_failures if str(item.get("step") or "").strip() == "lp3_profile"]

    call_started_at = str(memory_stage.get("call_started_at") or "").strip() or None
    current_person_id = str(memory_stage.get("current_person_id") or memory_stage.get("person_id") or "").strip() or None
    last_completed_person_id = str(memory_stage.get("last_completed_person_id") or memory_stage.get("person_id") or "").strip() or None

    return {
        "task_id": task_id,
        "version": task.get("version"),
        "pipeline_family": memory.get("pipeline_family") or "v0317",
        "task_status": task_status,
        "current_stage": task.get("stage"),
        "current_substage": current_substage or None,
        "steps": {
            "lp1": {
                "status": _resolve_step_status(
                    task_status=task_status,
                    current_substage=current_substage,
                    step_substage="lp1_batch",
                    has_data=bool(lp1_events),
                ),
                "substage": "lp1_batch",
                "updated_at": _artifact_payload(family_dir / "lp1_events_compact.json", task_dir, task_id, asset_url_builder)["updated_at"]
                or memory_stage.get("updated_at"),
                "summary": {
                    "event_count": len(lp1_events),
                    "batch_count": len(lp1_batch_ids),
                    "attempt_count": len(lp1_attempt_pairs),
                    "retry_count": len(lp1_retry_attempts),
                    "last_parse_status": lp1_last_attempt.get("parse_status"),
                    "parse_failure_count": len(lp1_parse_failures),
                },
                "artifacts": {
                    "lp1_events_compact": _artifact_payload(family_dir / "lp1_events_compact.json", task_dir, task_id, asset_url_builder),
                    "lp1_events_jsonl": _artifact_payload(family_dir / "lp1_events.jsonl", task_dir, task_id, asset_url_builder),
                    "lp1_batch_requests": _artifact_payload(family_dir / "lp1_batch_requests.jsonl", task_dir, task_id, asset_url_builder),
                    "lp1_batch_outputs": _artifact_payload(family_dir / "lp1_batch_outputs.jsonl", task_dir, task_id, asset_url_builder),
                },
                "data": lp1_events,
                "attempts": lp1_batch_outputs,
                "failures": lp1_parse_failures,
            },
            "lp2": {
                "status": _resolve_step_status(
                    task_status=task_status,
                    current_substage=current_substage,
                    step_substage="lp2_relationship",
                    has_data=bool(lp2_relationships),
                ),
                "substage": "lp2_relationship",
                "updated_at": _artifact_payload(family_dir / "lp2_relationships.jsonl", task_dir, task_id, asset_url_builder)["updated_at"]
                or _artifact_payload(family_dir / "lp2_relationships.json", task_dir, task_id, asset_url_builder)["updated_at"]
                or memory_stage.get("updated_at"),
                "summary": {
                    "relationship_count": len(lp2_relationships),
                    "candidate_count": memory_stage.get("candidate_count"),
                    "processed_candidates": memory_stage.get("processed_candidates"),
                    "current_candidate_index": memory_stage.get("current_candidate_index"),
                    "current_person_id": current_person_id,
                    "last_completed_person_id": last_completed_person_id,
                    "call_started_at": call_started_at,
                    "call_timeout_seconds": memory_stage.get("call_timeout_seconds"),
                    "provider": memory_stage.get("provider"),
                    "model": memory_stage.get("model"),
                },
                "artifacts": {
                    "lp2_relationships_json": _artifact_payload(family_dir / "lp2_relationships.json", task_dir, task_id, asset_url_builder),
                    "lp2_relationships_jsonl": _artifact_payload(family_dir / "lp2_relationships.jsonl", task_dir, task_id, asset_url_builder),
                    "llm_failures": _artifact_payload(family_dir / "llm_failures.jsonl", task_dir, task_id, asset_url_builder),
                },
                "data": lp2_relationships,
                "failures": lp2_failures,
            },
            "lp3": {
                "status": _resolve_step_status(
                    task_status=task_status,
                    current_substage=current_substage,
                    step_substage="lp3_profile",
                    has_data=bool(lp3_profile),
                ),
                "substage": "lp3_profile",
                "updated_at": _artifact_payload(family_dir / "lp3_profile.json", task_dir, task_id, asset_url_builder)["updated_at"]
                or memory_stage.get("updated_at"),
                "summary": {
                    "has_profile": bool(lp3_profile),
                    "report_length": len(str(lp3_profile.get("report_markdown") or "")) if isinstance(lp3_profile, dict) else 0,
                },
                "artifacts": {
                    "lp3_profile": _artifact_payload(family_dir / "lp3_profile.json", task_dir, task_id, asset_url_builder),
                    "llm_failures": _artifact_payload(family_dir / "llm_failures.jsonl", task_dir, task_id, asset_url_builder),
                },
                "data": lp3_profile,
                "failures": lp3_failures,
            },
        },
    }
