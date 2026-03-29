from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile


DOWNLOAD_BUNDLE_MIN_CREATED_AT = datetime.fromisoformat("2026-03-24T13:21:14.908153")
DOWNLOAD_BUNDLE_ROUTE = "analysis-bundle"
DOWNLOAD_BUNDLE_FILENAME_SUFFIX = "face-vlm-lp1-bundle.zip"
DOWNLOAD_BUNDLE_ARCHIVE_RELATIVE_PATH = Path("downloads") / DOWNLOAD_BUNDLE_FILENAME_SUFFIX


@dataclass(frozen=True)
class BundleEntry:
    source_path: Path
    archive_path: str
    category: str


REQUIRED_FILE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("cache/face_recognition_output.json", "face/face_recognition_output.json", "face"),
)

OPTIONAL_FILE_SPECS: tuple[tuple[str, str, str], ...] = (
    ("cache/face_recognition_state.json", "face/face_recognition_state.json", "face"),
    ("cache/vlm_failures.jsonl", "vlm/vlm_failures.jsonl", "vlm"),
    ("cache/dedupe_report.json", "face/dedupe_report.json", "face"),
)

OPTIONAL_DIRECTORY_SPECS: tuple[tuple[str, str, str], ...] = (
    ("cache/boxed_images", "face/boxed_images", "face"),
    ("cache/face_crops", "face/face_crops", "face"),
)


def _parse_created_at(value: object) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _eligible_by_time(task: dict) -> bool:
    created_at = _parse_created_at(task.get("created_at"))
    return created_at is not None and created_at >= DOWNLOAD_BUNDLE_MIN_CREATED_AT


def _bundle_filename(task_id: str) -> str:
    return f"{str(task_id)[:12]}-{DOWNLOAD_BUNDLE_FILENAME_SUFFIX}"


def _task_dir(task: dict) -> Path:
    raw = task.get("task_dir")
    if raw:
        return Path(str(raw))
    raise FileNotFoundError("任务目录不存在")

def _family_name(task: dict) -> str:
    version = str(task.get("version") or "").strip()
    if version in {"v0327-exp", "v0327-db", "v0327-db-query"}:
        return "v0325"
    if version in {"v0323", "v0325"}:
        return version
    return "v0323"


def _file_specs_for_task(task: dict) -> tuple[tuple[tuple[str, str, str], ...], tuple[tuple[str, str, str], ...]]:
    family_name = _family_name(task)
    required_file_specs = (
        *REQUIRED_FILE_SPECS,
        (f"{family_name}/vp1_observations.json", "vlm/vp1_observations.json", "vlm"),
        (f"{family_name}/lp1_events_compact.json", "lp1/lp1_events_compact.json", "lp1"),
    )
    optional_file_specs = (
        *OPTIONAL_FILE_SPECS,
        (f"{family_name}/lp1_batch_outputs.jsonl", "lp1/lp1_batch_outputs.jsonl", "lp1"),
        (f"{family_name}/lp1_parse_failures.json", "lp1/lp1_parse_failures.json", "lp1"),
        (f"{family_name}/lp1_events_raw.json", "lp1/lp1_events_raw.json", "lp1"),
    )
    if family_name == "v0325":
        optional_file_specs = (
            *optional_file_specs,
            (f"{family_name}/structured_profile.json", "lp3/structured_profile.json", "lp3"),
            (f"{family_name}/profile_fact_decisions.json", "lp3/profile_fact_decisions.json", "lp3"),
            (f"{family_name}/downstream_audit_report.json", "lp3/downstream_audit_report.json", "lp3"),
            (f"{family_name}/relationship_dossiers.json", "lp2/relationship_dossiers.json", "lp2"),
            (f"{family_name}/group_artifacts.json", "lp3/group_artifacts.json", "lp3"),
            (f"{family_name}/raw_upstream_manifest.json", "raw/raw_upstream_manifest.json", "raw"),
            (f"{family_name}/raw_upstream_index.json", "raw/raw_upstream_index.json", "raw"),
        )
    return required_file_specs, optional_file_specs


def _collect_entries(task_dir: Path, task: dict) -> tuple[list[BundleEntry], list[str]]:
    entries: list[BundleEntry] = []
    missing_required: list[str] = []
    required_file_specs, optional_file_specs = _file_specs_for_task(task)

    for relative_path, archive_path, category in required_file_specs:
        source = task_dir / relative_path
        if source.exists():
            entries.append(BundleEntry(source, archive_path, category))
        else:
            missing_required.append(relative_path)

    for relative_path, archive_path, category in optional_file_specs:
        source = task_dir / relative_path
        if source.exists():
            entries.append(BundleEntry(source, archive_path, category))

    for relative_dir, archive_dir, category in OPTIONAL_DIRECTORY_SPECS:
        source_dir = task_dir / relative_dir
        if not source_dir.exists() or not source_dir.is_dir():
            continue
        for file_path in sorted(path for path in source_dir.rglob("*") if path.is_file()):
            relative_name = file_path.relative_to(source_dir).as_posix()
            entries.append(BundleEntry(file_path, f"{archive_dir}/{relative_name}", category))

    return entries, missing_required


def describe_task_downloads(task: dict) -> dict | None:
    if str(task.get("status") or "").strip() != "completed":
        return None
    if not _eligible_by_time(task):
        return None

    try:
        task_dir = _task_dir(task)
    except FileNotFoundError:
        return None

    entries, missing_required = _collect_entries(task_dir, task)
    if missing_required:
        return None

    file_count = len(entries)
    categories = sorted({entry.category for entry in entries})
    return {
        "analysis_bundle": {
            "route": DOWNLOAD_BUNDLE_ROUTE,
            "url": f"/api/tasks/{task['task_id']}/downloads/{DOWNLOAD_BUNDLE_ROUTE}",
            "filename": _bundle_filename(str(task["task_id"])),
            "file_count": file_count,
            "categories": categories,
        }
    }


def _latest_mtime(entries: Iterable[BundleEntry]) -> float:
    latest = 0.0
    for entry in entries:
        latest = max(latest, entry.source_path.stat().st_mtime)
    return latest


def build_task_analysis_bundle(task: dict) -> Path:
    task_dir = _task_dir(task)
    downloads = describe_task_downloads(task)
    if not downloads or "analysis_bundle" not in downloads:
        raise FileNotFoundError("当前任务没有可下载的 Face/VLM/LP1 打包结果")

    entries, missing_required = _collect_entries(task_dir, task)
    if missing_required:
        raise FileNotFoundError(f"缺少必要产物: {', '.join(missing_required)}")

    output_path = task_dir / DOWNLOAD_BUNDLE_ARCHIVE_RELATIVE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    latest_source_mtime = _latest_mtime(entries)
    if output_path.exists() and output_path.stat().st_mtime >= latest_source_mtime:
        return output_path

    manifest = {
        "task_id": task.get("task_id"),
        "version": task.get("version"),
        "created_at": task.get("created_at"),
        "generated_at": datetime.utcnow().isoformat(),
        "file_count": len(entries),
        "files": [
            {
                "archive_path": entry.archive_path,
                "category": entry.category,
                "size_bytes": entry.source_path.stat().st_size,
            }
            for entry in entries
        ],
    }

    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        for entry in entries:
            archive.write(entry.source_path, arcname=entry.archive_path)
    return output_path
