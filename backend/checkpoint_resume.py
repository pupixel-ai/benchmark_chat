from __future__ import annotations

import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


V0325_RESUME_ACTION_SPECS: Dict[str, Dict[str, Any]] = {
    "vp1": {
        "rerun_stages": ["lp1", "lp2", "lp3"],
        "required": [
            "cache/face_recognition_output.json",
            "cache/face_recognition_state.json",
            "cache/vlm_cache.json",
            "cache/dedupe_report.json",
            "v0325/vp1_observations.json",
        ],
        "optional": [
            "cache/vlm_failures.jsonl",
        ],
        "archive": [
            "v0325/lp1_batch_requests.jsonl",
            "v0325/lp1_batch_outputs.jsonl",
            "v0325/lp1_event_cards.jsonl",
            "v0325/lp1_events.jsonl",
            "v0325/lp1_events_compact.json",
            "v0325/lp1_event_continuation_log.jsonl",
            "v0325/lp1_parse_failures.json",
            "v0325/lp1_salvaged_events.jsonl",
            "v0325/lp1_salvage_report.json",
            "v0325/raw_upstream_manifest.json",
            "v0325/raw_upstream_index.json",
            "v0325/lp2_relationships.json",
            "v0325/lp2_relationships.jsonl",
            "v0325/lp3_profile.json",
            "v0325/structured_profile.json",
            "v0325/relationship_dossiers.json",
            "v0325/group_artifacts.json",
            "v0325/profile_fact_decisions.json",
            "v0325/profile_fact_decisions.full.json.gz",
            "v0325/downstream_audit_report.json",
            "v0325/memory_snapshot.json",
            "v0325/llm_failures.jsonl",
            "output/result.json",
        ],
    },
    "lp1": {
        "rerun_stages": ["lp2", "lp3"],
        "required": [
            "cache/face_recognition_output.json",
            "cache/face_recognition_state.json",
            "cache/vlm_cache.json",
            "cache/dedupe_report.json",
            "v0325/vp1_observations.json",
            "v0325/lp1_events_compact.json",
        ],
        "optional": [
            "cache/vlm_failures.jsonl",
            "v0325/lp1_event_continuation_log.jsonl",
        ],
        "archive": [
            "v0325/raw_upstream_manifest.json",
            "v0325/raw_upstream_index.json",
            "v0325/lp2_relationships.json",
            "v0325/lp2_relationships.jsonl",
            "v0325/lp3_profile.json",
            "v0325/structured_profile.json",
            "v0325/relationship_dossiers.json",
            "v0325/group_artifacts.json",
            "v0325/profile_fact_decisions.json",
            "v0325/profile_fact_decisions.full.json.gz",
            "v0325/downstream_audit_report.json",
            "v0325/memory_snapshot.json",
            "v0325/llm_failures.jsonl",
            "output/result.json",
        ],
    },
    "lp2": {
        "rerun_stages": ["lp3"],
        "required": [
            "cache/face_recognition_output.json",
            "cache/face_recognition_state.json",
            "cache/vlm_cache.json",
            "cache/dedupe_report.json",
            "v0325/vp1_observations.json",
            "v0325/lp1_events_compact.json",
            "v0325/lp2_relationships.json",
            "v0325/relationship_dossiers.json",
            "v0325/group_artifacts.json",
        ],
        "optional": [
            "cache/vlm_failures.jsonl",
            "v0325/lp1_event_continuation_log.jsonl",
        ],
        "archive": [
            "v0325/lp3_profile.json",
            "v0325/structured_profile.json",
            "v0325/profile_fact_decisions.json",
            "v0325/profile_fact_decisions.full.json.gz",
            "v0325/downstream_audit_report.json",
            "v0325/memory_snapshot.json",
            "v0325/llm_failures.jsonl",
            "output/result.json",
        ],
    },
}


def _normalized_resume_from(value: Any) -> str | None:
    candidate = str(value or "").strip().lower()
    if candidate in V0325_RESUME_ACTION_SPECS:
        return candidate
    return None


def _resolved_path(task_dir: Path, relative_path: str) -> Path:
    return task_dir / relative_path


def _existing_relative_paths(task_dir: Path, relative_paths: Iterable[str]) -> List[str]:
    existing: List[str] = []
    for relative_path in relative_paths:
        if _resolved_path(task_dir, relative_path).exists():
            existing.append(relative_path)
    return existing


def build_resume_actions(task: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    version = str(task.get("version") or "").strip()
    if version != "v0325":
        return {}
    task_dir = Path(str(task.get("task_dir") or ""))
    task_status = str(task.get("status") or "").strip().lower()
    target_mode = "fork" if task_status == "completed" else "in_place"
    actions: Dict[str, Dict[str, Any]] = {}
    for resume_from, spec in V0325_RESUME_ACTION_SPECS.items():
        missing = [
            relative_path
            for relative_path in spec["required"]
            if not _resolved_path(task_dir, relative_path).exists()
        ]
        available = not missing and task_status in {"failed", "completed"}
        disabled_reason = ""
        if task_status not in {"failed", "completed"}:
            disabled_reason = "当前任务未处于 completed 或 failed，暂不能执行 checkpoint 恢复。"
        elif missing:
            if resume_from == "lp2":
                disabled_reason = "缺少 lp2_relationships.json / relationship_dossiers.json / group_artifacts.json，需回退到重跑 LP2+。"
            else:
                disabled_reason = f"缺少 checkpoint 文件: {', '.join(missing)}"
        actions[resume_from] = {
            "resume_from": resume_from,
            "available": available,
            "disabled_reason": disabled_reason or None,
            "rerun_stages": list(spec["rerun_stages"]),
            "target_mode": target_mode if available else None,
        }
    return actions


def archive_resume_outputs(task_dir: Path, *, resume_from: str) -> Path | None:
    normalized = _normalized_resume_from(resume_from)
    if normalized is None:
        return None
    spec = V0325_RESUME_ACTION_SPECS[normalized]
    existing = _existing_relative_paths(task_dir, spec["archive"])
    if not existing:
        return None
    archive_dir = task_dir / "resume_archives" / f"{datetime.now().strftime('%Y%m%dT%H%M%S')}_{normalized}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for relative_path in existing:
        source = _resolved_path(task_dir, relative_path)
        target = archive_dir / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
    return archive_dir


def copy_task_uploads(source_task_dir: Path, target_task_dir: Path) -> None:
    source_uploads = source_task_dir / "uploads"
    if source_uploads.exists():
        shutil.copytree(source_uploads, target_task_dir / "uploads", dirs_exist_ok=True)
    source_failures = source_task_dir / "upload_failures.json"
    if source_failures.exists():
        (target_task_dir / "upload_failures.json").parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_failures, target_task_dir / "upload_failures.json")


def copy_resume_checkpoint_files(source_task_dir: Path, target_task_dir: Path, *, resume_from: str) -> None:
    normalized = _normalized_resume_from(resume_from)
    if normalized is None:
        return
    spec = V0325_RESUME_ACTION_SPECS[normalized]
    for relative_path in [*spec["required"], *spec.get("optional", [])]:
        source = _resolved_path(source_task_dir, relative_path)
        if not source.exists():
            continue
        target = _resolved_path(target_task_dir, relative_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def build_resume_checkpoint_archive(source_task_dir: Path, archive_path: Path, *, resume_from: str) -> Path:
    normalized = _normalized_resume_from(resume_from)
    if normalized is None:
        raise ValueError(f"unsupported resume_from: {resume_from}")
    spec = V0325_RESUME_ACTION_SPECS[normalized]
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for relative_path in [*spec["required"], *spec.get("optional", [])]:
            source = _resolved_path(source_task_dir, relative_path)
            if source.exists():
                handle.write(source, arcname=relative_path)
    return archive_path

