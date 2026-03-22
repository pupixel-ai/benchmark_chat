"""
Independent v0321.3 pipeline family.
"""
from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from uuid import NAMESPACE_URL, uuid5

from memory_module.adapters import MemoryStoragePublisher
from utils import save_json


PIPELINE_FAMILY_V0321_3 = "v0321_3"
PIPELINE_VERSION_V0321_3 = "v0321.3"


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default))
            handle.write("\n")


class V03213StagingStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS assets (
                    asset_id TEXT PRIMARY KEY,
                    photo_id TEXT NOT NULL,
                    timestamp TEXT,
                    asset_type TEXT NOT NULL,
                    event_eligible INTEGER NOT NULL,
                    media_event_eligible INTEGER NOT NULL,
                    reference_only INTEGER NOT NULL,
                    place_key TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS person_appearances (
                    appearance_id TEXT PRIMARY KEY,
                    person_id TEXT NOT NULL,
                    photo_id TEXT NOT NULL,
                    timestamp TEXT,
                    appearance_mode TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_person_appearances_person ON person_appearances(person_id, timestamp);
                CREATE TABLE IF NOT EXISTS event_roots (
                    event_root_id TEXT PRIMARY KEY,
                    current_revision_id TEXT,
                    sealed_state TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS event_revisions (
                    event_revision_id TEXT PRIMARY KEY,
                    event_root_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    started_at TEXT,
                    ended_at TEXT,
                    sealed_state TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_event_revisions_root ON event_revisions(event_root_id, revision DESC);
                CREATE INDEX IF NOT EXISTS idx_event_revisions_time ON event_revisions(started_at, ended_at);
                CREATE TABLE IF NOT EXISTS relationship_roots (
                    relationship_root_id TEXT PRIMARY KEY,
                    target_person_id TEXT NOT NULL,
                    current_revision_id TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS relationship_revisions (
                    relationship_revision_id TEXT PRIMARY KEY,
                    relationship_root_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_relationship_revisions_root ON relationship_revisions(relationship_root_id, revision DESC);
                CREATE TABLE IF NOT EXISTS reference_media_signals (
                    signal_id TEXT PRIMARY KEY,
                    profile_bucket TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def upsert_asset(self, record: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO assets (
                    asset_id, photo_id, timestamp, asset_type, event_eligible,
                    media_event_eligible, reference_only, place_key, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["asset_id"],
                    record["photo_id"],
                    record.get("timestamp"),
                    record.get("asset_type"),
                    1 if record.get("event_eligible") else 0,
                    1 if record.get("media_event_eligible") else 0,
                    1 if record.get("reference_only") else 0,
                    record.get("place_key"),
                    json.dumps(record, ensure_ascii=False, default=_json_default),
                ),
            )

    def insert_person_appearance(self, record: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO person_appearances (
                    appearance_id, person_id, photo_id, timestamp, appearance_mode, confidence, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["appearance_id"],
                    record["person_id"],
                    record["photo_id"],
                    record.get("timestamp"),
                    record["appearance_mode"],
                    float(record.get("confidence") or 0.0),
                    json.dumps(record, ensure_ascii=False, default=_json_default),
                ),
            )

    def upsert_event_revision(self, root_payload: Dict[str, Any], revision_payload: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO event_roots (event_root_id, current_revision_id, sealed_state, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    root_payload["event_root_id"],
                    root_payload.get("current_revision_id"),
                    root_payload.get("sealed_state") or "open_frontier",
                    json.dumps(root_payload, ensure_ascii=False, default=_json_default),
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO event_revisions (
                    event_revision_id, event_root_id, revision, started_at, ended_at, sealed_state, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    revision_payload["event_revision_id"],
                    revision_payload["event_root_id"],
                    int(revision_payload.get("revision") or 1),
                    revision_payload.get("started_at"),
                    revision_payload.get("ended_at"),
                    revision_payload.get("sealed_state") or "open_frontier",
                    json.dumps(revision_payload, ensure_ascii=False, default=_json_default),
                ),
            )

    def seal_frontier_before(self, threshold_iso: str) -> List[str]:
        sealed_ids: List[str] = []
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT er.event_revision_id, er.payload_json
                FROM event_revisions er
                JOIN event_roots r ON r.current_revision_id = er.event_revision_id
                WHERE r.sealed_state = 'open_frontier' AND COALESCE(er.ended_at, '') < ?
                """,
                (threshold_iso,),
            ).fetchall()
            for row in rows:
                payload = json.loads(row["payload_json"])
                payload["sealed_state"] = "sealed"
                conn.execute(
                    "UPDATE event_roots SET sealed_state = ?, payload_json = ? WHERE event_root_id = ?",
                    ("sealed", json.dumps({**payload, "event_root_id": payload["event_root_id"], "current_revision_id": payload["event_revision_id"]}, ensure_ascii=False, default=_json_default), payload["event_root_id"]),
                )
                conn.execute(
                    "UPDATE event_revisions SET sealed_state = ?, payload_json = ? WHERE event_revision_id = ?",
                    ("sealed", json.dumps(payload, ensure_ascii=False, default=_json_default), row["event_revision_id"]),
                )
                sealed_ids.append(payload["event_root_id"])
        return sealed_ids

    def list_open_frontier_event_revisions(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT er.payload_json
                FROM event_revisions er
                JOIN event_roots r ON r.current_revision_id = er.event_revision_id
                WHERE r.sealed_state = 'open_frontier'
                ORDER BY COALESCE(er.started_at, '')
                """
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def seal_event_roots(self, event_root_ids: Sequence[str]) -> List[str]:
        sealed_ids: List[str] = []
        unique_root_ids = [str(root_id) for root_id in event_root_ids if str(root_id)]
        if not unique_root_ids:
            return sealed_ids
        with self._connect() as conn:
            for event_root_id in unique_root_ids:
                row = conn.execute(
                    """
                    SELECT er.event_revision_id, er.payload_json
                    FROM event_roots r
                    JOIN event_revisions er ON er.event_revision_id = r.current_revision_id
                    WHERE r.event_root_id = ?
                    """,
                    (event_root_id,),
                ).fetchone()
                if not row:
                    continue
                payload = json.loads(row["payload_json"])
                payload["sealed_state"] = "sealed"
                conn.execute(
                    "UPDATE event_roots SET sealed_state = ?, payload_json = ? WHERE event_root_id = ?",
                    ("sealed", json.dumps({**payload, "event_root_id": payload["event_root_id"], "current_revision_id": payload["event_revision_id"]}, ensure_ascii=False, default=_json_default), payload["event_root_id"]),
                )
                conn.execute(
                    "UPDATE event_revisions SET sealed_state = ?, payload_json = ? WHERE event_revision_id = ?",
                    ("sealed", json.dumps(payload, ensure_ascii=False, default=_json_default), row["event_revision_id"]),
                )
                sealed_ids.append(payload["event_root_id"])
        return sealed_ids

    def list_candidate_event_revisions(
        self,
        *,
        event_started_at: str,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT er.payload_json
                FROM event_revisions er
                JOIN event_roots r ON r.current_revision_id = er.event_revision_id
                ORDER BY ABS(strftime('%s', COALESCE(er.started_at, '')) - strftime('%s', ?)) ASC
                LIMIT ?
                """,
                (event_started_at, limit),
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def list_current_event_revisions(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT er.payload_json
                FROM event_revisions er
                JOIN event_roots r ON r.current_revision_id = er.event_revision_id
                ORDER BY COALESCE(er.started_at, '')
                """
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def list_person_appearances(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT payload_json
                FROM person_appearances
                ORDER BY COALESCE(timestamp, ''), appearance_id
                """
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def get_current_relationship_revision(self, relationship_root_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT rr.payload_json
                FROM relationship_roots r
                JOIN relationship_revisions rr ON rr.relationship_revision_id = r.current_revision_id
                WHERE r.relationship_root_id = ?
                """,
                (relationship_root_id,),
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def upsert_relationship_revision(self, root_payload: Dict[str, Any], revision_payload: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO relationship_roots (
                    relationship_root_id, target_person_id, current_revision_id, payload_json
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    root_payload["relationship_root_id"],
                    root_payload["target_person_id"],
                    root_payload["current_revision_id"],
                    json.dumps(root_payload, ensure_ascii=False, default=_json_default),
                ),
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO relationship_revisions (
                    relationship_revision_id, relationship_root_id, revision, status, payload_json
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    revision_payload["relationship_revision_id"],
                    revision_payload["relationship_root_id"],
                    int(revision_payload.get("revision") or 1),
                    revision_payload.get("status") or "active",
                    json.dumps(revision_payload, ensure_ascii=False, default=_json_default),
                ),
            )

    def list_current_relationship_revisions(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT rr.payload_json
                FROM relationship_roots r
                JOIN relationship_revisions rr ON rr.relationship_revision_id = r.current_revision_id
                ORDER BY rr.relationship_root_id
                """
            ).fetchall()
        return [json.loads(row["payload_json"]) for row in rows]

    def save_reference_signal(self, payload: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO reference_media_signals (signal_id, profile_bucket, payload_json)
                VALUES (?, ?, ?)
                """,
                (
                    payload["signal_id"],
                    payload["profile_bucket"],
                    json.dumps(payload, ensure_ascii=False, default=_json_default),
                ),
            )

    def list_reference_signals(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload_json FROM reference_media_signals ORDER BY signal_id").fetchall()
        return [json.loads(row["payload_json"]) for row in rows]


class V03213PipelineFamily:
    def __init__(
        self,
        *,
        task_id: str,
        task_dir: str | Path,
        user_id: Optional[str],
        asset_store: Any,
        llm_processor: Any | None = None,
        public_url_builder: Optional[Callable[[Path | str], Optional[str]]] = None,
    ) -> None:
        self.task_id = task_id
        self.task_dir = Path(task_dir)
        self.user_id = user_id or f"task:{task_id}"
        self.asset_store = asset_store
        self.llm_processor = llm_processor
        self.public_url_builder = public_url_builder
        self.family_dir = self.task_dir / PIPELINE_FAMILY_V0321_3
        self.family_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = self.family_dir
        self.staging = V03213StagingStore(self.family_dir / "state.db")
        self.family_prefix = f"v0321.3:{self.user_id}"
        self._summary: Dict[str, Any] = {
            "pipeline_family": PIPELINE_FAMILY_V0321_3,
            "asset_count": 0,
            "event_window_count": 0,
            "event_revision_count": 0,
            "relationship_revision_count": 0,
            "reference_media_signal_count": 0,
            "ambiguous_boundary_count": 0,
            "relationship_llm_count": 0,
            "event_llm_count": 0,
            "profile_llm_count": 0,
        }

    def run(
        self,
        *,
        photos: Sequence[Any],
        face_output: Dict[str, Any],
        primary_person_id: Optional[str],
        vlm_results: Sequence[Dict[str, Any]],
        cached_photo_ids: Iterable[str],
        dedupe_report: Optional[Dict[str, Any]],
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        prior_state = self._load_prior_state()
        bootstrap_summary = self._bootstrap_from_prior_state(prior_state)
        self._summary.update(bootstrap_summary)
        cached_photo_ids = set(cached_photo_ids)
        photo_by_id = {str(photo.photo_id): photo for photo in photos}
        observations_by_photo = {
            str(item.get("photo_id")): self._build_photo_observation_packet(
                item,
                photo=photo_by_id.get(str(item.get("photo_id") or "")),
            )
            for item in vlm_results
            if isinstance(item, dict) and item.get("photo_id")
        }
        asset_records = []
        person_appearances = []
        face_anchor_report = []
        for photo in photos:
            observation = observations_by_photo.get(photo.photo_id, self._empty_observation_packet(photo))
            asset = self._classify_asset(photo=photo, observation=observation)
            asset_records.append(asset)
            self.staging.upsert_asset(asset)
            appearances = self._build_person_appearances(photo=photo, asset=asset, observation=observation)
            for appearance in appearances:
                person_appearances.append(appearance)
                self.staging.insert_person_appearance(appearance)
            face_anchor_report.append(
                {
                    "photo_id": photo.photo_id,
                    "face_anchors": self._face_anchors(photo),
                    "asset_type": asset["asset_type"],
                    "event_eligible": asset["event_eligible"],
                }
            )

        reference_signals = self._build_reference_media_signals(asset_records, observations_by_photo, primary_person_id)
        for signal in reference_signals:
            self.staging.save_reference_signal(signal)
        all_reference_signals = self.staging.list_reference_signals()

        event_assets = [asset for asset in asset_records if asset.get("event_eligible")]
        bursts = self._build_bursts(
            event_assets,
            observations_by_photo=observations_by_photo,
            appearances=person_appearances,
        )
        boundaries = self._score_boundaries(bursts)
        boundaries = self._resolve_ambiguous_boundaries(
            bursts=bursts,
            boundaries=boundaries,
            observations_by_photo=observations_by_photo,
            person_appearances=person_appearances,
        )
        windows = self._build_event_windows(bursts, boundaries)
        self._summary["asset_count"] = len(asset_records)
        self._summary["event_window_count"] = len(windows)
        self._summary["ambiguous_boundary_count"] = sum(1 for item in boundaries if item.get("decision") == "ambiguous")
        self._summary["reference_media_signal_count"] = len(all_reference_signals)
        llm_started_at = perf_counter()
        self._emit_progress(
            progress_callback,
            "llm",
            {
                "message": "v0321.3 事件草稿生成中",
                "substage": "event_draft",
                "candidate_count": len(windows),
                "filtered_count": len(windows),
                "processed_candidates": 0,
                "percent": 15,
                "runtime_seconds": 0.0,
            },
        )
        event_drafts: List[Dict[str, Any]] = []
        finalize_candidate_count = 0
        total_windows = len(windows)
        for index, window in enumerate(windows, start=1):
            draft = self._draft_event_window(window, observations_by_photo, person_appearances)
            event_drafts.append(draft)
            if self._window_requires_event_finalize(window):
                finalize_candidate_count += 1
            progress_percent = 15
            if total_windows > 0:
                progress_percent = min(44, 15 + int((index / total_windows) * 29))
            self._emit_progress(
                progress_callback,
                "llm",
                {
                    "message": "v0321.3 事件草稿生成中",
                    "substage": "event_draft",
                    "candidate_count": total_windows,
                    "filtered_count": total_windows,
                    "processed_candidates": index,
                    "finalize_candidate_count": finalize_candidate_count,
                    "percent": progress_percent,
                    "runtime_seconds": round(perf_counter() - llm_started_at, 4),
                },
            )
        self._emit_progress(
            progress_callback,
            "llm",
            {
                "message": "v0321.3 事件定稿中",
                "substage": "event_finalize",
                "candidate_count": finalize_candidate_count,
                "filtered_count": finalize_candidate_count,
                "processed_candidates": finalize_candidate_count,
                "percent": 45,
                "runtime_seconds": round(perf_counter() - llm_started_at, 4),
            },
        )
        event_revisions, atomic_evidence, changed_event_revisions = self._resolve_event_drafts(event_drafts)
        delta_atomic_evidence = self._collect_atomic_evidence(changed_event_revisions)
        profile_input_pack_partial = self._build_profile_input_pack_partial(
            primary_person_id=primary_person_id,
            event_revisions=event_revisions,
            atomic_evidence=atomic_evidence,
            reference_signals=all_reference_signals,
            scope="cumulative",
        )
        delta_profile_input_pack_partial = self._build_profile_input_pack_partial(
            primary_person_id=primary_person_id,
            event_revisions=changed_event_revisions,
            atomic_evidence=delta_atomic_evidence,
            reference_signals=reference_signals,
            scope="current_task",
        )
        self._emit_progress(
            progress_callback,
            "llm",
            {
                "message": "v0321.3 事件定稿完成，关系与画像继续整理中",
                "completed": False,
                "substage": "event_finalize",
                "percent": 58,
                "runtime_seconds": round(perf_counter() - llm_started_at, 4),
                "memory_contract_preview": self._build_llm_preview(
                    event_revisions=changed_event_revisions,
                    atomic_evidence=delta_atomic_evidence,
                    relationship_revisions=[],
                    profile_revision={},
                    profile_input_pack_preview=self._build_profile_input_pack_preview(delta_profile_input_pack_partial),
                ),
                "fact_count": len(changed_event_revisions),
                "relationship_hypothesis_count": 0,
                "profile_evidence_count": 0,
                "profile_markdown_preview": "",
            },
        )
        self._emit_progress(
            progress_callback,
            "llm",
            {
                "message": "v0321.3 关系综合中",
                "substage": "relationship_inference",
                "candidate_count": len(event_revisions),
                "filtered_count": len(changed_event_revisions),
                "processed_candidates": len(changed_event_revisions),
                "percent": 72,
                "runtime_seconds": round(perf_counter() - llm_started_at, 4),
            },
        )
        relationship_revisions, relationship_ledgers, changed_relationship_revisions = self._project_relationships(
            primary_person_id=primary_person_id,
            event_revisions=event_revisions,
            changed_event_revisions=changed_event_revisions,
            atomic_evidence=atomic_evidence,
        )
        profile_input_pack = self._build_profile_input_pack(
            profile_input_pack_partial=profile_input_pack_partial,
            relationship_revisions=relationship_revisions,
        )
        delta_profile_input_pack = self._build_profile_input_pack(
            profile_input_pack_partial=delta_profile_input_pack_partial,
            relationship_revisions=changed_relationship_revisions,
        )
        self._emit_progress(
            progress_callback,
            "llm",
            {
                "message": "v0321.3 关系综合完成，画像继续整理中",
                "completed": False,
                "substage": "relationship_projector",
                "percent": 82,
                "runtime_seconds": round(perf_counter() - llm_started_at, 4),
                "memory_contract_preview": self._build_llm_preview(
                    event_revisions=changed_event_revisions,
                    atomic_evidence=delta_atomic_evidence,
                    relationship_revisions=changed_relationship_revisions,
                    profile_revision={},
                    profile_input_pack_preview=self._build_profile_input_pack_preview(delta_profile_input_pack),
                ),
                "fact_count": len(changed_event_revisions),
                "relationship_hypothesis_count": len(changed_relationship_revisions),
                "profile_evidence_count": 0,
                "profile_markdown_preview": "",
            },
        )
        all_person_appearances = self.staging.list_person_appearances()
        self._emit_progress(
            progress_callback,
            "llm",
            {
                "message": "v0321.3 用户画像整理中",
                "substage": "profile_materialization",
                "candidate_count": len(event_revisions),
                "filtered_count": len(relationship_revisions),
                "processed_candidates": len(relationship_revisions),
                "percent": 88,
                "runtime_seconds": round(perf_counter() - llm_started_at, 4),
            },
        )
        profile_revision, profile_markdown = self._build_profile_revision(
            primary_person_id=primary_person_id,
            event_revisions=event_revisions,
            atomic_evidence=atomic_evidence,
            relationship_revisions=relationship_revisions,
            reference_signals=all_reference_signals,
            profile_input_pack=profile_input_pack,
            revision_key="1",
            scope="cumulative",
        )
        delta_profile_revision, delta_profile_markdown = self._build_profile_revision(
            primary_person_id=primary_person_id,
            event_revisions=changed_event_revisions,
            atomic_evidence=delta_atomic_evidence,
            relationship_revisions=changed_relationship_revisions,
            reference_signals=reference_signals,
            profile_input_pack=delta_profile_input_pack,
            revision_key="delta",
            scope="current_task",
        )
        llm_runtime_seconds = perf_counter() - llm_started_at
        storage = self._build_storage_payload(
            primary_person_id=primary_person_id,
            face_output=face_output,
            person_appearances=all_person_appearances,
            event_revisions=event_revisions,
            relationship_revisions=relationship_revisions,
            relationship_ledgers=relationship_ledgers,
            profile_revision=profile_revision,
            reference_signals=all_reference_signals,
        )
        self._emit_progress(
            progress_callback,
            "llm",
            self._build_llm_progress_payload(
                event_revisions=changed_event_revisions,
                atomic_evidence=delta_atomic_evidence,
                relationship_revisions=changed_relationship_revisions,
                profile_revision=delta_profile_revision,
                profile_markdown=delta_profile_markdown,
                reference_signals=reference_signals,
                profile_input_pack_preview=self._build_profile_input_pack_preview(delta_profile_input_pack),
                runtime_seconds=llm_runtime_seconds,
            ),
        )
        memory_started_at = perf_counter()
        self._emit_progress(
            progress_callback,
            "memory",
            {
                "message": "v0321.3 revision-first 落库中",
                "pipeline_family": PIPELINE_FAMILY_V0321_3,
                "percent": 15,
                "runtime_seconds": 0.0,
            },
        )
        external_publish = MemoryStoragePublisher(
            task_dir=self.task_dir,
            output_dir=self.family_dir,
        ).publish(storage, user_id=self.user_id)

        pipeline_summary = {
            **self._summary,
            "cached_photo_ids": sorted(cached_photo_ids),
            "dedupe_report": dict(dedupe_report or {}),
            "primary_person_id": primary_person_id,
            "over_segmentation_anomaly": len(windows) >= max(10, int(len(event_assets) * 0.75)) if event_assets else False,
        }

        paths = {
            "face_anchor_report": self.artifact_dir / "face_anchor_report.json",
            "person_appearances": self.artifact_dir / "person_appearances.jsonl",
            "asset_triage": self.artifact_dir / "asset_triage.jsonl",
            "burst_manifest": self.artifact_dir / "burst_manifest.json",
            "boundary_decisions": self.artifact_dir / "boundary_decisions.json",
            "event_drafts": self.artifact_dir / "event_drafts.jsonl",
            "event_revisions": self.artifact_dir / "event_revisions.jsonl",
            "atomic_evidence": self.artifact_dir / "atomic_evidence.jsonl",
            "relationship_revisions": self.artifact_dir / "relationship_revisions.jsonl",
            "relationship_ledgers": self.artifact_dir / "relationship_ledgers.json",
            "period_revisions": self.artifact_dir / "period_revisions.jsonl",
            "profile_revision": self.artifact_dir / "profile_revision.json",
            "profile_input_pack_partial": self.artifact_dir / "profile_input_pack_partial.json",
            "profile_input_pack": self.artifact_dir / "profile_input_pack.json",
            "pipeline_summary": self.artifact_dir / "pipeline_summary.json",
            "reference_media": self.artifact_dir / "reference_media.json",
            "memory_payload": self.artifact_dir / "memory_payload.json",
        }
        save_json({"items": face_anchor_report}, str(paths["face_anchor_report"]))
        _write_jsonl(paths["person_appearances"], all_person_appearances)
        _write_jsonl(paths["asset_triage"], asset_records)
        save_json({"bursts": bursts}, str(paths["burst_manifest"]))
        save_json({"boundaries": boundaries}, str(paths["boundary_decisions"]))
        _write_jsonl(paths["event_drafts"], event_drafts)
        _write_jsonl(paths["event_revisions"], event_revisions)
        _write_jsonl(paths["atomic_evidence"], atomic_evidence)
        _write_jsonl(paths["relationship_revisions"], relationship_revisions)
        save_json({"items": relationship_ledgers}, str(paths["relationship_ledgers"]))
        _write_jsonl(paths["period_revisions"], [])
        save_json(profile_revision, str(paths["profile_revision"]))
        save_json(profile_input_pack_partial, str(paths["profile_input_pack_partial"]))
        save_json(profile_input_pack, str(paths["profile_input_pack"]))
        save_json(pipeline_summary, str(paths["pipeline_summary"]))
        save_json({"items": all_reference_signals}, str(paths["reference_media"]))

        payload = {
            "pipeline_family": PIPELINE_FAMILY_V0321_3,
            "summary": {
                "event_count": len(event_revisions),
                "event_revision_count": len(event_revisions),
                "relationship_count": len(relationship_revisions),
                "relationship_revision_count": len(relationship_revisions),
                "reference_media_signal_count": len(all_reference_signals),
                "profile_bucket_count": len(profile_revision.get("buckets", {})),
                "profile_field_count": sum(
                    len(bucket.get("values", []))
                    for bucket in profile_revision.get("buckets", {}).values()
                ),
                **pipeline_summary,
            },
            "event_revisions": event_revisions,
            "delta_event_revisions": changed_event_revisions,
            "atomic_evidence": atomic_evidence,
            "delta_atomic_evidence": delta_atomic_evidence,
            "relationship_revisions": relationship_revisions,
            "delta_relationship_revisions": changed_relationship_revisions,
            "period_revisions": [],
            "profile_revision": profile_revision,
            "profile_markdown": profile_markdown,
            "delta_profile_revision": delta_profile_revision,
            "delta_profile_markdown": delta_profile_markdown,
            "profile_input_pack_partial": profile_input_pack_partial,
            "profile_input_pack": profile_input_pack,
            "person_appearances": all_person_appearances,
            "reference_media_signals": all_reference_signals,
            "delta_reference_media_signals": reference_signals,
            "envelope": {
                "event_revision_ids": [item.get("event_revision_id") for item in event_revisions],
                "relationship_revision_ids": [item.get("relationship_revision_id") for item in relationship_revisions],
                "profile_revision_id": profile_revision.get("profile_revision_id"),
            },
            "storage": storage,
            "external_publish": external_publish,
            "transparency": {
                "vlm_stage": self._build_vlm_stage_transparency(
                    observations_by_photo=observations_by_photo,
                    cached_photo_ids=cached_photo_ids,
                    total_input_photos=len(photos),
                ),
                "segmentation_stage": self._build_segmentation_stage_transparency(
                    bursts=bursts,
                    boundaries=boundaries,
                    windows=windows,
                ),
                "llm_stage": self._build_llm_stage_transparency(
                    event_revisions=changed_event_revisions,
                    atomic_evidence=delta_atomic_evidence,
                    relationship_revisions=changed_relationship_revisions,
                    profile_revision=delta_profile_revision,
                    reference_signals=reference_signals,
                    profile_input_pack_preview=self._build_profile_input_pack_preview(delta_profile_input_pack),
                    runtime_seconds=llm_runtime_seconds,
                ),
                "neo4j_state": self._build_neo4j_state(storage),
                "redis_state": self._build_redis_state(storage),
                "asset_triage": asset_records,
                "bursts": bursts,
                "boundaries": boundaries,
                "open_frontier_count": len([item for item in event_revisions if item.get("sealed_state") == "open_frontier"]),
                "reference_media_signal_count": len(all_reference_signals),
                "relationship_ledgers": relationship_ledgers,
                "bootstrap": prior_state.get("bootstrap") or {},
            },
            "artifacts": {
                f"{key}_path": str(path)
                for key, path in paths.items()
            },
            "evaluation": {},
        }
        payload["artifacts"].update(
            {
                f"{key}_url": self._public_url(path)
                for key, path in paths.items()
            }
        )
        payload["artifacts"]["staging_db_path"] = str(self.staging.db_path)
        payload["artifacts"]["staging_db_url"] = self._public_url(self.staging.db_path)
        save_json(payload, str(paths["memory_payload"]))
        self._emit_progress(
            progress_callback,
            "memory",
            self._build_memory_progress_payload(payload=payload, runtime_seconds=perf_counter() - memory_started_at),
        )
        return payload

    def _build_vlm_stage_transparency(
        self,
        *,
        observations_by_photo: Dict[str, Dict[str, Any]],
        cached_photo_ids: Iterable[str],
        total_input_photos: int,
    ) -> Dict[str, Any]:
        cached_hits = len(list(cached_photo_ids))
        summaries = []
        for photo_id, observation in list(observations_by_photo.items())[:12]:
            summaries.append(
                {
                    "photo_id": photo_id,
                    "summary": observation.get("summary") or observation.get("scene_hint") or observation.get("activity_hint"),
                    "ocr_hits": list(observation.get("ocr_hits", [])[:5]),
                    "brands": list(observation.get("brands", [])[:5]),
                    "place_candidates": list(observation.get("place_candidates", [])[:3]),
                    "embedded_media_signals": list(observation.get("embedded_media_signals", [])[:3]),
                }
            )
        return {
            "processed_photos": len(observations_by_photo),
            "cached_hits": cached_hits,
            "total_input_photos": total_input_photos,
            "representative_photo_count": len(summaries),
            "summaries": summaries,
        }

    def _build_segmentation_stage_transparency(
        self,
        *,
        bursts: Sequence[Dict[str, Any]],
        boundaries: Sequence[Dict[str, Any]],
        windows: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        summaries = []
        for window in list(windows)[:12]:
            summaries.append(
                {
                    "window_id": window.get("window_id"),
                    "started_at": window.get("started_at"),
                    "ended_at": window.get("ended_at"),
                    "photo_count": len(window.get("photo_ids", []) or []),
                    "boundary_reason": window.get("boundary_reason"),
                }
            )
        movement_count = sum(1 for item in boundaries if item.get("decision") in {"split", "ambiguous"})
        return {
            "burst_count": len(bursts),
            "event_count": len(windows),
            "movement_count": movement_count,
            "summaries": summaries,
        }

    def _build_llm_stage_transparency(
        self,
        *,
        event_revisions: Sequence[Dict[str, Any]],
        atomic_evidence: Sequence[Dict[str, Any]],
        relationship_revisions: Sequence[Dict[str, Any]],
        profile_revision: Dict[str, Any],
        reference_signals: Sequence[Dict[str, Any]],
        profile_input_pack_preview: Optional[Dict[str, Any]],
        runtime_seconds: float,
    ) -> Dict[str, Any]:
        summaries: List[Dict[str, Any]] = []
        for event in list(event_revisions)[:8]:
            summaries.append(
                {
                    "kind": "event_revision",
                    "event_revision_id": event.get("event_revision_id"),
                    "title": event.get("title"),
                    "participant_person_ids": list(event.get("participant_person_ids", [])[:4]),
                    "place_refs": list(event.get("place_refs", [])[:3]),
                    "original_photo_ids": list(event.get("original_photo_ids", [])[:6]),
                }
            )
        for relationship in list(relationship_revisions)[:6]:
            summaries.append(
                {
                    "kind": "relationship_revision",
                    "relationship_revision_id": relationship.get("relationship_revision_id"),
                    "target_person_id": relationship.get("target_person_id"),
                    "relationship_type": relationship.get("relationship_type"),
                    "confidence": relationship.get("confidence"),
                }
            )
        for bucket, payload in list((profile_revision.get("buckets") or {}).items())[:4]:
            summaries.append(
                {
                    "kind": "profile_bucket",
                    "bucket": bucket,
                    "values": list((payload or {}).get("values", [])[:5]),
                    "original_photo_ids": list((payload or {}).get("original_photo_ids", [])[:6]),
                }
            )
        return {
            "fact_count": len(event_revisions),
            "relationship_hypothesis_count": len(relationship_revisions),
            "profile_evidence_count": len(reference_signals),
            "observation_count": len(atomic_evidence),
            "profile_delta_count": sum(len((payload or {}).get("values", [])) for payload in (profile_revision.get("buckets") or {}).values()),
            "profile_input_pack_preview": dict(profile_input_pack_preview or {}),
            "runtime_seconds": round(runtime_seconds, 4),
            "summaries": summaries,
        }

    def _build_llm_preview(
        self,
        *,
        event_revisions: Sequence[Dict[str, Any]],
        atomic_evidence: Sequence[Dict[str, Any]],
        relationship_revisions: Sequence[Dict[str, Any]],
        profile_revision: Dict[str, Any],
        profile_input_pack_preview: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "event_revisions": list(event_revisions[:8]),
            "atomic_evidence": list(atomic_evidence[:16]),
            "relationship_revisions": list(relationship_revisions[:8]),
            "profile_revision": profile_revision,
            "profile_input_pack_preview": dict(profile_input_pack_preview or {}),
        }

    def _build_llm_progress_payload(
        self,
        *,
        event_revisions: Sequence[Dict[str, Any]],
        atomic_evidence: Sequence[Dict[str, Any]],
        relationship_revisions: Sequence[Dict[str, Any]],
        profile_revision: Dict[str, Any],
        profile_markdown: str,
        reference_signals: Sequence[Dict[str, Any]],
        profile_input_pack_preview: Optional[Dict[str, Any]],
        runtime_seconds: float,
    ) -> Dict[str, Any]:
        return {
            "message": "v0321.3 LLM revision 生成完成",
            "completed": True,
            "substage": "completed",
            "percent": 100,
            "runtime_seconds": round(runtime_seconds, 4),
            "memory_contract_preview": self._build_llm_preview(
                event_revisions=event_revisions,
                atomic_evidence=atomic_evidence,
                relationship_revisions=relationship_revisions,
                profile_revision=profile_revision,
                profile_input_pack_preview=profile_input_pack_preview,
            ),
            "profile_markdown_preview": profile_markdown[:4000] if profile_markdown else "",
            "fact_count": len(event_revisions),
            "relationship_hypothesis_count": len(relationship_revisions),
            "profile_evidence_count": len(reference_signals),
        }

    def _build_neo4j_state(self, storage: Dict[str, Any]) -> Dict[str, Any]:
        neo4j_storage = dict(storage.get("neo4j", {}) or {})
        nodes = dict(neo4j_storage.get("nodes", {}) or {})
        node_counts = {
            key: len(value) if isinstance(value, list) else 0
            for key, value in nodes.items()
        }
        return {
            "node_counts": node_counts,
            "edge_count": len(list(neo4j_storage.get("edges", []) or [])),
        }

    def _build_redis_state(self, storage: Dict[str, Any]) -> Dict[str, Any]:
        redis_payload = dict(storage.get("redis", {}) or {})
        return {
            "profile_version": int(((redis_payload.get("profile_current") or {}).get("version") or 0)),
            "published_field_count": len(list(((redis_payload.get("profile_current") or {}).get("buckets") or {}).keys())),
            "relationship_count": len([key for key in redis_payload.keys() if str(key).startswith("relationship_ledger_")]),
            "recent_event_count": 0,
            "recent_fact_count": 0,
            "materialized_profile_count": len(list(((redis_payload.get("profile_current") or {}).get("buckets") or {}).keys())),
        }

    def _build_memory_progress_payload(self, *, payload: Dict[str, Any], runtime_seconds: float) -> Dict[str, Any]:
        storage = dict(payload.get("storage", {}) or {})
        neo4j_storage = dict(storage.get("neo4j", {}) or {})
        neo4j_nodes = dict(neo4j_storage.get("nodes", {}) or {})
        neo4j_preview = {
            "nodes": {
                key: list(value[:8]) if isinstance(value, list) else value
                for key, value in neo4j_nodes.items()
            },
            "edges": list((neo4j_storage.get("edges", []) or [])[:20]),
        }
        return {
            "message": "v0321.3 revision-first 落位完成",
            "completed": True,
            "percent": 100,
            "runtime_seconds": round(runtime_seconds, 4),
            "pipeline_family": PIPELINE_FAMILY_V0321_3,
            "redis_preview": dict(storage.get("redis", {}) or {}),
            "neo4j_preview": neo4j_preview,
            "memory_transparency_preview": dict(payload.get("transparency", {}) or {}),
            "artifacts": dict(payload.get("artifacts", {}) or {}),
        }

    def _classify_asset(self, *, photo: Any, observation: Dict[str, Any]) -> Dict[str, Any]:
        name = str(photo.filename or "").lower()
        summary = " ".join(
            str(part).lower()
            for part in [
                observation.get("summary"),
                observation.get("scene_hint"),
                observation.get("activity_hint"),
                " ".join(observation.get("ocr_hits", [])[:5]),
            ]
            if part
        )
        asset_type = "camera_photo"
        reference_only = False
        media_event_eligible = False
        if any(token in name for token in ("screenshot", "screen", "截屏", "截图")):
            asset_type = "screenshot"
            media_event_eligible = self._looks_like_media_action(summary)
        elif any(token in name for token in ("ai", "midjourney", "stable", "wallpaper", "meme", "inspo", "reference", "web")):
            asset_type = "ai_generated_or_reference" if "ai" in name else "saved_web_image"
            reference_only = True
        elif any(token in summary for token in ("poster", "screenshot", "wallpaper", "meme", "海报", "屏幕", "网图")):
            asset_type = "saved_web_image"
            reference_only = True
        if any(token in summary for token in ("polaroid", "instant photo", "printed photo", "相纸", "照片中的照片", "屏幕中的照片")):
            asset_type = "scanned_or_embedded_media"
        event_eligible = asset_type == "camera_photo" or media_event_eligible
        timestamp_iso = photo.timestamp.isoformat() if isinstance(photo.timestamp, datetime) else str(photo.timestamp)
        place_key = str((photo.location or {}).get("name") or observation.get("place_candidates", ["unknown"])[0] or "unknown")
        return {
            "asset_id": self._stable_id("asset", photo.photo_id),
            "photo_id": photo.photo_id,
            "original_photo_id": self._original_photo_id(photo),
            "filename": photo.filename,
            "timestamp": timestamp_iso,
            "asset_type": asset_type,
            "event_eligible": event_eligible,
            "media_event_eligible": media_event_eligible,
            "reference_only": reference_only,
            "place_key": place_key,
            "time_source": "exif" if getattr(photo, "timestamp", None) else "unknown",
            "triage_reason": "filename_or_summary_heuristics",
        }

    def _load_prior_state(self) -> Dict[str, Any]:
        db_state = self._load_prior_state_from_db()
        if db_state.get("event_revisions") or db_state.get("relationship_revisions"):
            return db_state
        sibling_state = self._load_prior_state_from_sibling_snapshot()
        if sibling_state.get("event_revisions") or sibling_state.get("relationship_revisions"):
            return sibling_state
        return {
            "event_revisions": [],
            "relationship_revisions": [],
            "person_appearances": [],
            "reference_signals": [],
            "bootstrap": {
                "applied": False,
                "source": None,
                "source_task_id": None,
            },
        }

    def _load_prior_state_from_db(self) -> Dict[str, Any]:
        try:
            from sqlalchemy import desc, select

            from backend.db import SessionLocal
            from backend.models import TaskRecord
        except Exception:
            return {
                "event_revisions": [],
                "relationship_revisions": [],
                "person_appearances": [],
                "reference_signals": [],
                "bootstrap": {"applied": False, "source": "db_unavailable", "source_task_id": None},
            }

        with SessionLocal() as session:
            stmt = (
                select(TaskRecord)
                .where(
                    TaskRecord.user_id == self.user_id,
                    TaskRecord.version == PIPELINE_VERSION_V0321_3,
                    TaskRecord.task_id != self.task_id,
                    TaskRecord.result.is_not(None),
                )
                .order_by(desc(TaskRecord.updated_at))
                .limit(1)
            )
            record = session.execute(stmt).scalar_one_or_none()

        if record is None:
            return {
                "event_revisions": [],
                "relationship_revisions": [],
                "person_appearances": [],
                "reference_signals": [],
                "bootstrap": {"applied": False, "source": "db", "source_task_id": None},
            }

        memory_payload = dict(((record.result or {}).get("memory") or {}))
        if memory_payload.get("pipeline_family") != PIPELINE_FAMILY_V0321_3:
            return {
                "event_revisions": [],
                "relationship_revisions": [],
                "person_appearances": [],
                "reference_signals": [],
                "bootstrap": {"applied": False, "source": "db", "source_task_id": record.task_id},
            }
        return self._assemble_prior_state(
            source="db",
            task_id=record.task_id,
            task_dir=Path(record.task_dir),
            memory_payload=memory_payload,
        )

    def _load_prior_state_from_sibling_snapshot(self) -> Dict[str, Any]:
        parent = self.task_dir.parent
        if not parent.exists():
            return {
                "event_revisions": [],
                "relationship_revisions": [],
                "person_appearances": [],
                "reference_signals": [],
                "bootstrap": {"applied": False, "source": "sibling_scan", "source_task_id": None},
            }
        candidates = sorted(
            (
                path
                for path in parent.glob(f"*/{PIPELINE_FAMILY_V0321_3}/memory_payload.json")
                if path.parent.parent.resolve() != self.task_dir.resolve()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for payload_path in candidates:
            try:
                memory_payload = json.loads(payload_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if memory_payload.get("pipeline_family") != PIPELINE_FAMILY_V0321_3:
                continue
            return self._assemble_prior_state(
                source="sibling_scan",
                task_id=payload_path.parent.parent.name,
                task_dir=payload_path.parent.parent,
                memory_payload=memory_payload,
            )
        return {
            "event_revisions": [],
            "relationship_revisions": [],
            "person_appearances": [],
            "reference_signals": [],
            "bootstrap": {"applied": False, "source": "sibling_scan", "source_task_id": None},
        }

    def _assemble_prior_state(
        self,
        *,
        source: str,
        task_id: str,
        task_dir: Path,
        memory_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        person_appearances = list(memory_payload.get("person_appearances", []) or [])
        if not person_appearances:
            person_appearances = self._load_jsonl_artifact(
                task_id=task_id,
                task_dir=task_dir,
                relative_path=f"{PIPELINE_FAMILY_V0321_3}/person_appearances.jsonl",
            )
        reference_signals = list(memory_payload.get("reference_media_signals", []) or [])
        if not reference_signals:
            reference_signals = self._load_json_artifact_items(
                task_id=task_id,
                task_dir=task_dir,
                relative_path=f"{PIPELINE_FAMILY_V0321_3}/reference_media.json",
            )
        return {
            "event_revisions": list(memory_payload.get("event_revisions", []) or []),
            "relationship_revisions": list(memory_payload.get("relationship_revisions", []) or []),
            "person_appearances": person_appearances,
            "reference_signals": reference_signals,
            "bootstrap": {
                "applied": True,
                "source": source,
                "source_task_id": task_id,
                "prior_event_revision_count": len(list(memory_payload.get("event_revisions", []) or [])),
                "prior_relationship_revision_count": len(list(memory_payload.get("relationship_revisions", []) or [])),
                "prior_person_appearance_count": len(person_appearances),
                "prior_reference_media_signal_count": len(reference_signals),
            },
        }

    def _load_jsonl_artifact(self, *, task_id: str, task_dir: Path, relative_path: str) -> List[Dict[str, Any]]:
        local_path = task_dir / relative_path
        raw_bytes: Optional[bytes] = None
        if local_path.exists():
            raw_bytes = local_path.read_bytes()
        elif getattr(self.asset_store, "enabled", False):
            try:
                if self.asset_store.has_object(task_id, relative_path):
                    raw_bytes, _ = self.asset_store.read_bytes(task_id, relative_path)
            except Exception:
                raw_bytes = None
        if not raw_bytes:
            return []
        records: List[Dict[str, Any]] = []
        for line in raw_bytes.decode("utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
        return records

    def _load_json_artifact_items(self, *, task_id: str, task_dir: Path, relative_path: str) -> List[Dict[str, Any]]:
        local_path = task_dir / relative_path
        raw_bytes: Optional[bytes] = None
        if local_path.exists():
            raw_bytes = local_path.read_bytes()
        elif getattr(self.asset_store, "enabled", False):
            try:
                if self.asset_store.has_object(task_id, relative_path):
                    raw_bytes, _ = self.asset_store.read_bytes(task_id, relative_path)
            except Exception:
                raw_bytes = None
        if not raw_bytes:
            return []
        try:
            payload = json.loads(raw_bytes.decode("utf-8"))
        except Exception:
            return []
        return list(payload.get("items", []) or [])

    def _bootstrap_from_prior_state(self, prior_state: Dict[str, Any]) -> Dict[str, Any]:
        for appearance in prior_state.get("person_appearances", []) or []:
            self.staging.insert_person_appearance(appearance)
        for signal in prior_state.get("reference_signals", []) or []:
            self.staging.save_reference_signal(signal)
        for event in prior_state.get("event_revisions", []) or []:
            root_payload = {
                "event_root_id": event["event_root_id"],
                "current_revision_id": event["event_revision_id"],
                "sealed_state": event.get("sealed_state") or "sealed",
                "pipeline_family": PIPELINE_FAMILY_V0321_3,
            }
            self.staging.upsert_event_revision(root_payload, event)
        for relationship in prior_state.get("relationship_revisions", []) or []:
            root_payload = {
                "relationship_root_id": relationship["relationship_root_id"],
                "target_person_id": relationship["target_person_id"],
                "current_revision_id": relationship["relationship_revision_id"],
                "pipeline_family": PIPELINE_FAMILY_V0321_3,
            }
            self.staging.upsert_relationship_revision(root_payload, relationship)
        bootstrap = dict(prior_state.get("bootstrap") or {})
        return {
            "bootstrap_applied": bool(bootstrap.get("applied")),
            "bootstrap_source": bootstrap.get("source"),
            "bootstrap_source_task_id": bootstrap.get("source_task_id"),
            "bootstrap_prior_event_revision_count": int(bootstrap.get("prior_event_revision_count") or 0),
            "bootstrap_prior_relationship_revision_count": int(bootstrap.get("prior_relationship_revision_count") or 0),
            "bootstrap_prior_person_appearance_count": int(bootstrap.get("prior_person_appearance_count") or 0),
            "bootstrap_prior_reference_media_signal_count": int(bootstrap.get("prior_reference_media_signal_count") or 0),
        }

    def _looks_like_media_action(self, summary: str) -> bool:
        return any(
            token in summary
            for token in (
                "payment",
                "paid",
                "order",
                "ticket",
                "boarding",
                "itinerary",
                "receipt",
                "confirmation",
                "支付",
                "订单",
                "机票",
                "车票",
                "登机",
                "行程",
                "票据",
                "确认",
            )
        )

    def _build_person_appearances(self, *, photo: Any, asset: Dict[str, Any], observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        timestamp_iso = photo.timestamp.isoformat() if isinstance(photo.timestamp, datetime) else str(photo.timestamp)
        appearances = []
        asset_type = str(asset.get("asset_type") or "camera_photo")
        live_mode = "live_presence" if asset_type == "camera_photo" or asset.get("media_event_eligible") else "embedded_media"
        source_kind = "face_recognition" if live_mode == "live_presence" else "face_recognition_non_camera"
        original_photo_id = self._original_photo_id(photo)
        for face in list(getattr(photo, "faces", []) or []):
            person_id = str(face.get("person_id") or "")
            if not person_id:
                continue
            appearances.append(
                {
                    "appearance_id": self._stable_id("appearance", original_photo_id, person_id, live_mode),
                    "person_id": person_id,
                    "photo_id": photo.photo_id,
                    "original_photo_id": original_photo_id,
                    "timestamp": timestamp_iso,
                    "appearance_mode": live_mode,
                    "confidence": float(face.get("score") or face.get("similarity") or 0.0),
                    "bbox_ref": dict(face.get("bbox_xywh") or {}),
                    "source_kind": source_kind,
                }
            )
        for embedded_person_id in observation.get("embedded_media_person_ids", []) or []:
            appearances.append(
                {
                    "appearance_id": self._stable_id("appearance", original_photo_id, embedded_person_id, "embedded_media"),
                    "person_id": str(embedded_person_id),
                    "photo_id": photo.photo_id,
                    "original_photo_id": original_photo_id,
                    "timestamp": timestamp_iso,
                    "appearance_mode": "embedded_media",
                    "confidence": 0.55,
                    "bbox_ref": {},
                    "source_kind": "vlm_embedded_media",
                }
            )
        return appearances

    def _build_photo_observation_packet(self, item: Dict[str, Any], *, photo: Optional[Any] = None) -> Dict[str, Any]:
        analysis = dict(item.get("vlm_analysis", {}) or {})
        scene = dict(analysis.get("scene", {}) or {})
        event = dict(analysis.get("event", {}) or {})
        details = [str(value) for value in analysis.get("details", []) or [] if value]
        key_objects = [str(value) for value in analysis.get("key_objects", []) or [] if value]
        summary = str(analysis.get("summary") or "")
        ocr_hits = [str(value) for value in analysis.get("ocr_hits", []) or [] if value]
        if not ocr_hits:
            ocr_hits = [detail for detail in details if re.search(r"[A-Za-z0-9]{3,}", detail)]
        brands = [str(value) for value in analysis.get("brands", []) or [] if value]
        place_candidates = [str(value) for value in analysis.get("place_candidates", []) or [] if value]
        if not place_candidates and scene.get("location_detected"):
            place_candidates = [str(scene.get("location_detected"))]
        embedded_ids = [str(value) for value in analysis.get("embedded_media_person_ids", []) or [] if value]
        original_photo_id = self._original_photo_id(photo) if photo is not None else str(item.get("photo_id") or "")
        return {
            "photo_id": str(item.get("photo_id") or ""),
            "summary": summary,
            "scene_hint": str(scene.get("environment_description") or scene.get("location_detected") or ""),
            "activity_hint": str(event.get("activity") or ""),
            "social_hint": str(event.get("social_context") or event.get("interaction") or ""),
            "ocr_hits": ocr_hits[:10],
            "brands": brands[:10],
            "place_candidates": place_candidates[:5],
            "object_clues": key_objects[:10],
            "embedded_media_signals": [summary] if embedded_ids else [],
            "embedded_media_person_ids": embedded_ids,
            "uncertainty": [str(value) for value in analysis.get("uncertainty", []) or [] if value][:5],
            "original_photo_ids": [original_photo_id] if original_photo_id else [],
        }

    def _empty_observation_packet(self, photo: Any) -> Dict[str, Any]:
        return {
            "photo_id": photo.photo_id,
            "summary": "",
            "scene_hint": "",
            "activity_hint": "",
            "social_hint": "",
            "ocr_hits": [],
            "brands": [],
            "place_candidates": [str((photo.location or {}).get("name") or "unknown")],
            "object_clues": [],
            "embedded_media_signals": [],
            "embedded_media_person_ids": [],
            "uncertainty": [],
            "original_photo_ids": [self._original_photo_id(photo)],
        }

    def _build_reference_media_signals(
        self,
        assets: Sequence[Dict[str, Any]],
        observations_by_photo: Dict[str, Dict[str, Any]],
        primary_person_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        signals: List[Dict[str, Any]] = []
        if not primary_person_id:
            return signals
        for asset in assets:
            asset_type = str(asset.get("asset_type") or "")
            if not (
                asset.get("reference_only")
                or (asset_type == "screenshot" and not asset.get("media_event_eligible"))
            ):
                continue
            observation = observations_by_photo.get(asset["photo_id"], {})
            bucket, value = self._summarize_reference_media_signal(asset_type=asset_type, observation=observation)
            signal = {
                "signal_id": self._stable_id("reference-signal", asset["photo_id"], bucket),
                "photo_id": asset["photo_id"],
                "profile_bucket": bucket,
                "evidence_text": value,
                "source_kind": asset["asset_type"],
                "confidence": 0.55,
                "original_photo_ids": [asset["original_photo_id"]],
                "person_id": primary_person_id,
                "provenance": "reference_media_profile_signal",
            }
            signals.append(signal)
        return signals

    def _summarize_reference_media_signal(
        self,
        *,
        asset_type: str,
        observation: Dict[str, Any],
    ) -> Tuple[str, str]:
        text_pool = self._unique(
            [
                str(observation.get("summary") or "").strip(),
                str(observation.get("scene_hint") or "").strip(),
                str(observation.get("activity_hint") or "").strip(),
                str(observation.get("social_hint") or "").strip(),
                *[str(item).strip() for item in list(observation.get("ocr_hits", []) or [])[:5]],
                *[str(item).strip() for item in list(observation.get("brands", []) or [])[:3]],
                *[str(item).strip() for item in list(observation.get("place_candidates", []) or [])[:2]],
            ]
        )
        joined = " ".join(text_pool)
        normalized = joined.lower()

        if any(token in normalized for token in ("douyin", "tiktok", "抖音", "creator", "主页", "profile page", "video page")):
            return "interest", "近期关注短视频内容或创作者主页"
        if any(token in normalized for token in ("video call", "call", "通话", "wechat", "聊天", "message", "messenger", "contact", "聊天记录")):
            return "interest", "近期关注即时通讯或远程沟通内容"
        if any(token in normalized for token in ("flight", "hotel", "boarding", "trip", "travel", "机票", "酒店", "登机", "行程", "车票", "演出票")):
            return "aspiration", "近期关注出行、票务或行程安排"
        if any(token in normalized for token in ("outfit", "style", "fashion", "look", "穿搭", "风格", "审美")):
            return "aesthetic_preference", "近期关注穿搭、风格或审美参考"
        if any(token in normalized for token in ("buy", "shop", "product", "购物", "购买", "种草", "支付", "checkout", "order", "商品")):
            return "consumption_intent", "近期关注商品、购买或支付相关内容"
        if any(token in normalized for token in ("ai", "chatgpt", "midjourney", "prompt", "生成", "模型", "comfyui")):
            return "identity_style", "近期关注 AI 工具或生成内容"

        if observation.get("brands"):
            brands = self._unique(str(item).strip() for item in list(observation.get("brands", []) or []) if str(item).strip())
            if brands:
                return "interest", f"近期保存了与 {', '.join(brands[:2])} 相关的参考内容"
        if observation.get("ocr_hits"):
            ocr_hits = self._unique(
                re.sub(r"\s+", " ", str(item)).strip(" ,.;:：；，。")
                for item in list(observation.get("ocr_hits", []) or [])
                if str(item).strip()
            )
            if ocr_hits:
                snippet = ocr_hits[0]
                if len(snippet) > 32:
                    snippet = f"{snippet[:29]}..."
                return "interest", f"近期保存了与“{snippet}”相关的参考内容"

        if asset_type == "screenshot":
            return "interest", "近期保存了截图类参考内容"
        if asset_type == "ai_generated_or_reference":
            return "identity_style", "近期保存了 AI 生成或创意参考内容"
        return "interest", "近期保存了参考内容"

    def _build_bursts(
        self,
        assets: Sequence[Dict[str, Any]],
        *,
        observations_by_photo: Dict[str, Dict[str, Any]],
        appearances: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ordered = sorted(assets, key=lambda item: item.get("timestamp") or "")
        live_people_by_photo: Dict[str, List[str]] = defaultdict(list)
        for item in appearances:
            if item.get("appearance_mode") != "live_presence":
                continue
            photo_id = str(item.get("photo_id") or "")
            person_id = str(item.get("person_id") or "")
            if not photo_id or not person_id:
                continue
            live_people_by_photo[photo_id].append(person_id)
        bursts: List[Dict[str, Any]] = []
        current: Optional[Dict[str, Any]] = None
        for asset in ordered:
            photo_id = str(asset.get("photo_id") or "")
            observation = observations_by_photo.get(photo_id, {})
            live_person_ids = self._unique(live_people_by_photo.get(photo_id, []))
            if current is None:
                current = self._new_burst(asset, observation=observation, live_person_ids=live_person_ids)
                continue
            if self._can_extend_burst(current, asset, observation=observation, live_person_ids=live_person_ids):
                current["photo_ids"].append(asset["photo_id"])
                current["ended_at"] = asset.get("timestamp") or current["ended_at"]
                current["place_keys"].append(asset.get("place_key") or "")
                current["place_candidates"] = self._unique(
                    [*list(current.get("place_candidates", []) or []), *list(observation.get("place_candidates", []) or [])[:3]]
                )
                current["live_person_ids"] = self._unique(
                    [*list(current.get("live_person_ids", []) or []), *list(live_person_ids or [])]
                )
                current["activity_hints"] = self._unique(
                    [*list(current.get("activity_hints", []) or []), str(observation.get("activity_hint") or "").strip()]
                )
                current["scene_hints"] = self._unique(
                    [*list(current.get("scene_hints", []) or []), str(observation.get("scene_hint") or "").strip()]
                )
            else:
                bursts.append(current)
                current = self._new_burst(asset, observation=observation, live_person_ids=live_person_ids)
        if current is not None:
            bursts.append(current)
        return bursts

    def _new_burst(
        self,
        asset: Dict[str, Any],
        *,
        observation: Dict[str, Any],
        live_person_ids: Sequence[str],
    ) -> Dict[str, Any]:
        return {
            "burst_id": self._stable_id("burst", asset["photo_id"]),
            "photo_ids": [asset["photo_id"]],
            "started_at": asset.get("timestamp"),
            "ended_at": asset.get("timestamp"),
            "place_keys": [asset.get("place_key") or ""],
            "place_candidates": self._unique(list(observation.get("place_candidates", []) or [])[:3]),
            "live_person_ids": self._unique(list(live_person_ids or [])),
            "activity_hints": self._unique([str(observation.get("activity_hint") or "").strip()]),
            "scene_hints": self._unique([str(observation.get("scene_hint") or "").strip()]),
        }

    def _can_extend_burst(
        self,
        burst: Dict[str, Any],
        asset: Dict[str, Any],
        *,
        observation: Dict[str, Any],
        live_person_ids: Sequence[str],
    ) -> bool:
        left = self._parse_dt(burst.get("ended_at"))
        right = self._parse_dt(asset.get("timestamp"))
        if not left or not right:
            return False
        gap = right - left
        if gap > timedelta(minutes=20):
            return False
        last_place = str(burst.get("place_keys", [""])[-1] or "")
        next_place = str(asset.get("place_key") or "")
        if last_place and next_place and last_place == next_place:
            return True
        place_overlap = self._has_overlap(
            [last_place, *list(burst.get("place_candidates", []) or [])],
            [next_place, *list(observation.get("place_candidates", []) or [])[:3]],
        )
        live_overlap = self._has_overlap(
            list(burst.get("live_person_ids", []) or []),
            list(live_person_ids or []),
        )
        activity_overlap = self._has_overlap(
            list(burst.get("activity_hints", []) or []),
            [str(observation.get("activity_hint") or "").strip()],
        )
        scene_overlap = self._has_overlap(
            list(burst.get("scene_hints", []) or []),
            [str(observation.get("scene_hint") or "").strip()],
        )
        if gap <= timedelta(minutes=8) and live_overlap and (place_overlap or activity_overlap or scene_overlap):
            return True
        if gap <= timedelta(minutes=5) and place_overlap and (activity_overlap or scene_overlap):
            return True
        return False

    def _score_boundaries(self, bursts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        decisions: List[Dict[str, Any]] = []
        for index in range(len(bursts) - 1):
            left = bursts[index]
            right = bursts[index + 1]
            left_end = self._parse_dt(left.get("ended_at"))
            right_start = self._parse_dt(right.get("started_at"))
            gap = (right_start - left_end).total_seconds() if left_end and right_start else 0.0
            same_place = str(left.get("place_keys", [""])[-1] or "") == str(right.get("place_keys", [""])[0] or "")
            place_overlap = same_place or self._has_overlap(
                [*list(left.get("place_keys", []) or []), *list(left.get("place_candidates", []) or [])],
                [*list(right.get("place_keys", []) or []), *list(right.get("place_candidates", []) or [])],
            )
            live_overlap = self._has_overlap(
                list(left.get("live_person_ids", []) or []),
                list(right.get("live_person_ids", []) or []),
            )
            activity_overlap = self._has_overlap(
                list(left.get("activity_hints", []) or []),
                list(right.get("activity_hints", []) or []),
            )
            scene_overlap = self._has_overlap(
                list(left.get("scene_hints", []) or []),
                list(right.get("scene_hints", []) or []),
            )
            decision = "ambiguous"
            if gap > 3 * 3600:
                decision = "split"
            elif gap > 45 * 60 and not place_overlap and not live_overlap:
                decision = "split"
            elif place_overlap and gap <= 15 * 60:
                decision = "continue"
            elif gap <= 10 * 60 and live_overlap and (activity_overlap or scene_overlap):
                decision = "continue"
            elif gap <= 25 * 60 and place_overlap and (live_overlap or activity_overlap):
                decision = "continue"
            elif gap > 90 * 60:
                decision = "split"
            decisions.append(
                {
                    "left_burst_id": left["burst_id"],
                    "right_burst_id": right["burst_id"],
                    "gap_seconds": round(gap, 3),
                    "same_place": same_place,
                    "place_overlap": place_overlap,
                    "live_overlap": live_overlap,
                    "activity_overlap": activity_overlap,
                    "scene_overlap": scene_overlap,
                    "decision": decision,
                    "reason": "time_and_place_scoring",
                }
            )
        return decisions

    def _build_event_windows(self, bursts: Sequence[Dict[str, Any]], boundaries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not bursts:
            return []
        boundary_map = {item["right_burst_id"]: item for item in boundaries}
        windows: List[Dict[str, Any]] = []
        current = {
            "window_id": self._stable_id("window", bursts[0]["burst_id"]),
            "burst_ids": [bursts[0]["burst_id"]],
            "photo_ids": list(bursts[0]["photo_ids"]),
            "started_at": bursts[0]["started_at"],
            "ended_at": bursts[0]["ended_at"],
            "boundary_reason": "seed",
        }
        for burst in bursts[1:]:
            boundary = boundary_map.get(burst["burst_id"], {})
            decision = boundary.get("decision", "split")
            if decision == "continue":
                current["burst_ids"].append(burst["burst_id"])
                current["photo_ids"].extend(burst["photo_ids"])
                current["ended_at"] = burst["ended_at"]
                continue
            if decision == "ambiguous":
                current["boundary_reason"] = "ambiguous_resolved_as_split"
            windows.append(current)
            current = {
                "window_id": self._stable_id("window", burst["burst_id"]),
                "burst_ids": [burst["burst_id"]],
                "photo_ids": list(burst["photo_ids"]),
                "started_at": burst["started_at"],
                "ended_at": burst["ended_at"],
                "boundary_reason": decision,
            }
        windows.append(current)
        return windows

    def _call_llm_json_prompt(self, prompt: str) -> Optional[Dict[str, Any]]:
        caller = getattr(self.llm_processor, "_call_json_prompt", None)
        if not callable(caller):
            return None
        try:
            return caller(prompt)
        except Exception:
            return None

    def _call_llm_markdown_prompt(self, prompt: str) -> Optional[str]:
        caller = getattr(self.llm_processor, "_call_markdown_prompt", None)
        if not callable(caller):
            return None
        try:
            response = caller(prompt)
        except Exception:
            return None
        text = str(response or "").strip()
        return text or None

    def _coerce_signal_value(self, value: Any) -> str:
        if value in (None, "", [], {}):
            return ""
        if isinstance(value, str):
            return re.sub(r"\s+", " ", value).strip()
        if isinstance(value, (int, float, bool)):
            return str(value).strip()
        if isinstance(value, dict):
            for key in ("person_id", "photo_id", "place_ref", "place", "name", "label", "value", "id", "text", "title"):
                text = self._coerce_signal_value(value.get(key))
                if text:
                    return text
            try:
                return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)
            except Exception:
                return str(value).strip()
        if isinstance(value, (list, tuple, set)):
            values = list(value)
            if len(values) == 1:
                return self._coerce_signal_value(values[0])
            try:
                return json.dumps(values, ensure_ascii=False, sort_keys=True, default=_json_default)
            except Exception:
                return str(values).strip()
        return str(value).strip()

    def _signal_fingerprint(self, value: Any) -> str:
        text = self._coerce_signal_value(value)
        if text:
            return text
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)
        except Exception:
            return str(value or "")

    def _normalized_signal_set(self, values: Sequence[Any]) -> set[str]:
        normalized = set()
        for value in values:
            text = self._coerce_signal_value(value).lower()
            if text:
                normalized.add(text)
        return normalized

    def _has_overlap(self, left: Sequence[Any], right: Sequence[Any]) -> bool:
        return bool(self._normalized_signal_set(left).intersection(self._normalized_signal_set(right)))

    def _resolve_ambiguous_boundaries(
        self,
        *,
        bursts: Sequence[Dict[str, Any]],
        boundaries: Sequence[Dict[str, Any]],
        observations_by_photo: Dict[str, Dict[str, Any]],
        person_appearances: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        burst_by_id = {burst["burst_id"]: burst for burst in bursts}
        resolved: List[Dict[str, Any]] = []
        for boundary in boundaries:
            if boundary.get("decision") != "ambiguous":
                resolved.append(dict(boundary))
                continue
            left = burst_by_id.get(str(boundary.get("left_burst_id") or ""))
            right = burst_by_id.get(str(boundary.get("right_burst_id") or ""))
            if not left or not right:
                resolved.append(dict(boundary))
                continue
            prompt = self._build_boundary_llm_prompt(
                left=left,
                right=right,
                observations_by_photo=observations_by_photo,
                person_appearances=person_appearances,
            )
            payload = self._call_llm_json_prompt(prompt)
            decision = str((payload or {}).get("decision") or "").strip().lower()
            if decision not in {"continue", "split"}:
                resolved.append(dict(boundary))
                continue
            self._summary["event_llm_count"] = int(self._summary.get("event_llm_count") or 0) + 1
            updated = dict(boundary)
            updated["decision"] = decision
            updated["reason"] = f"llm_boundary:{str((payload or {}).get('reason') or '').strip() or 'resolved'}"
            resolved.append(updated)
        return resolved

    def _build_boundary_llm_prompt(
        self,
        *,
        left: Dict[str, Any],
        right: Dict[str, Any],
        observations_by_photo: Dict[str, Dict[str, Any]],
        person_appearances: Sequence[Dict[str, Any]],
    ) -> str:
        left_context = self._build_event_window_prompt_context(
            {"photo_ids": left.get("photo_ids", []), "started_at": left.get("started_at"), "ended_at": left.get("ended_at")},
            observations_by_photo=observations_by_photo,
            appearances=person_appearances,
        )
        right_context = self._build_event_window_prompt_context(
            {"photo_ids": right.get("photo_ids", []), "started_at": right.get("started_at"), "ended_at": right.get("ended_at")},
            observations_by_photo=observations_by_photo,
            appearances=person_appearances,
        )
        return (
            "You are an event-boundary judge.\n"
            "Decide whether two adjacent photo bursts belong to the same real-life event.\n"
            "Only return JSON with keys decision and reason.\n"
            "decision must be one of: continue, split.\n"
            "Prefer split when evidence is weak. Do not hallucinate people or places.\n\n"
            f"LEFT_BURST={json.dumps(left_context, ensure_ascii=False)}\n"
            f"RIGHT_BURST={json.dumps(right_context, ensure_ascii=False)}\n"
        )

    def _build_event_window_prompt_context(
        self,
        window: Dict[str, Any],
        *,
        observations_by_photo: Dict[str, Dict[str, Any]],
        appearances: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        photo_ids = list(window.get("photo_ids", []) or [])
        photo_appearances: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for item in appearances:
            if item.get("photo_id") in photo_ids:
                photo_appearances[str(item["photo_id"])].append(item)
        photos: List[Dict[str, Any]] = []
        for photo_id in photo_ids:
            observation = observations_by_photo.get(photo_id, {})
            appearance_items = photo_appearances.get(str(photo_id), [])
            photos.append(
                {
                    "photo_id": photo_id,
                    "original_photo_ids": list(observation.get("original_photo_ids", []) or [photo_id]),
                    "summary": str(observation.get("summary") or ""),
                    "scene_hint": str(observation.get("scene_hint") or ""),
                    "activity_hint": str(observation.get("activity_hint") or ""),
                    "social_hint": str(observation.get("social_hint") or ""),
                    "place_candidates": list(observation.get("place_candidates", []) or [])[:3],
                    "ocr_hits": list(observation.get("ocr_hits", []) or [])[:4],
                    "brands": list(observation.get("brands", []) or [])[:3],
                    "object_clues": list(observation.get("object_clues", []) or [])[:4],
                    "embedded_media_person_ids": list(observation.get("embedded_media_person_ids", []) or [])[:3],
                    "live_person_ids": self._unique(
                        item.get("person_id") for item in appearance_items if item.get("appearance_mode") == "live_presence"
                    ),
                    "depicted_person_ids": self._unique(
                        item.get("person_id") for item in appearance_items if item.get("appearance_mode") == "embedded_media"
                    ),
                }
            )
        return {
            "started_at": window.get("started_at"),
            "ended_at": window.get("ended_at"),
            "photo_count": len(photo_ids),
            "photos": photos,
        }

    def _draft_event_window(
        self,
        window: Dict[str, Any],
        observations_by_photo: Dict[str, Dict[str, Any]],
        appearances: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        photo_ids = list(window.get("photo_ids", []))
        related_appearances = [item for item in appearances if item["photo_id"] in photo_ids]
        participants = self._unique(
            item["person_id"]
            for item in related_appearances
            if item["appearance_mode"] == "live_presence"
        )
        depicted = self._unique(
            item["person_id"]
            for item in related_appearances
            if item["appearance_mode"] == "embedded_media"
        )
        place_refs = self._unique(
            candidate
            for photo_id in photo_ids
            for candidate in observations_by_photo.get(photo_id, {}).get("place_candidates", [])[:1]
        )
        activity_hints = self._unique(
            observations_by_photo.get(photo_id, {}).get("activity_hint")
            for photo_id in photo_ids
            if observations_by_photo.get(photo_id, {}).get("activity_hint")
        )
        title = self._event_title(activity_hints, place_refs)
        evidence = self._event_evidence_for_window(window, observations_by_photo)
        draft = {
            "window_id": window["window_id"],
            "started_at": window["started_at"],
            "ended_at": window["ended_at"],
            "participant_person_ids": participants,
            "depicted_person_ids": depicted,
            "place_refs": place_refs,
            "title": title,
            "boundary_reason": window.get("boundary_reason") or "window",
            "original_photo_ids": self._unique(
                original_photo_id
                for photo_id in photo_ids
                for original_photo_id in observations_by_photo.get(photo_id, {}).get("original_photo_ids", [])
            ),
            "atomic_evidence": evidence,
            "confidence": 0.68,
        }
        llm_draft = self._draft_event_window_with_llm(
            window=window,
            observations_by_photo=observations_by_photo,
            appearances=appearances,
            fallback=draft,
        )
        if llm_draft:
            draft = llm_draft
        if len(window.get("burst_ids", []) or []) > 1 or "ambiguous" in str(window.get("boundary_reason") or ""):
            finalized = self._finalize_event_draft_with_llm(
                window=window,
                observations_by_photo=observations_by_photo,
                appearances=appearances,
                draft=draft,
            )
            if finalized:
                draft = finalized
        return draft

    def _window_requires_event_finalize(self, window: Dict[str, Any]) -> bool:
        return bool(len(window.get("burst_ids", []) or []) > 1 or "ambiguous" in str(window.get("boundary_reason") or ""))

    def _draft_event_window_with_llm(
        self,
        *,
        window: Dict[str, Any],
        observations_by_photo: Dict[str, Dict[str, Any]],
        appearances: Sequence[Dict[str, Any]],
        fallback: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        prompt_context = self._build_event_window_prompt_context(
            window,
            observations_by_photo=observations_by_photo,
            appearances=appearances,
        )
        prompt = (
            "You are a world-class multimodal event extraction system.\n"
            "Turn the provided photo-window observations into one grounded event draft.\n"
            "Return JSON only.\n"
            "Use only the person IDs, places, and evidence explicitly present in the input.\n"
            "If uncertain, keep confidence low and prefer fewer claims.\n"
            "Schema:\n"
            "{\n"
            '  "title": string,\n'
            '  "summary": string,\n'
            '  "participant_person_ids": string[],\n'
            '  "depicted_person_ids": string[],\n'
            '  "place_refs": string[],\n'
            '  "confidence": number,\n'
            '  "atomic_evidence": [{"evidence_type": string, "value_or_text": string, "photo_ids": string[], "confidence": number, "provenance": string}]\n'
            "}\n"
            "Allowed evidence_type values: ocr, brand, place_candidate, object_last_seen, route_transport, "
            "health_treatment, person_interaction, profile_signal, embedded_media_person_reference, media_action_receipt.\n"
            f"EVENT_WINDOW={json.dumps(prompt_context, ensure_ascii=False)}\n"
        )
        payload = self._call_llm_json_prompt(prompt)
        if not payload:
            return None
        normalized = self._normalize_llm_event_draft(
            payload=payload,
            fallback=fallback,
            observations_by_photo=observations_by_photo,
            window=window,
        )
        if normalized:
            self._summary["event_llm_count"] = int(self._summary.get("event_llm_count") or 0) + 1
        return normalized

    def _finalize_event_draft_with_llm(
        self,
        *,
        window: Dict[str, Any],
        observations_by_photo: Dict[str, Dict[str, Any]],
        appearances: Sequence[Dict[str, Any]],
        draft: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        prompt_context = self._build_event_window_prompt_context(
            window,
            observations_by_photo=observations_by_photo,
            appearances=appearances,
        )
        prompt = (
            "You are the final event editor.\n"
            "Refine one event draft after boundary resolution. Remove duplicate evidence, keep only grounded claims, "
            "and improve the title/summary if needed.\n"
            "Return JSON with the same schema as the input draft. Do not invent IDs.\n"
            f"EVENT_WINDOW={json.dumps(prompt_context, ensure_ascii=False)}\n"
            f"CURRENT_DRAFT={json.dumps(draft, ensure_ascii=False)}\n"
        )
        payload = self._call_llm_json_prompt(prompt)
        if not payload:
            return None
        normalized = self._normalize_llm_event_draft(
            payload=payload,
            fallback=draft,
            observations_by_photo=observations_by_photo,
            window=window,
        )
        if normalized:
            self._summary["event_llm_count"] = int(self._summary.get("event_llm_count") or 0) + 1
        return normalized

    def _normalize_llm_event_draft(
        self,
        *,
        payload: Dict[str, Any],
        fallback: Dict[str, Any],
        observations_by_photo: Dict[str, Dict[str, Any]],
        window: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        photo_ids = list(window.get("photo_ids", []) or [])
        valid_photo_ids = set(photo_ids)
        photo_id_to_original = {
            photo_id: list(observations_by_photo.get(photo_id, {}).get("original_photo_ids", []) or [photo_id])
            for photo_id in photo_ids
        }
        valid_participants = self._normalized_signal_set(fallback.get("participant_person_ids", []) or [])
        valid_depicted = self._normalized_signal_set(fallback.get("depicted_person_ids", []) or [])
        valid_places = self._normalized_signal_set(fallback.get("place_refs", []) or [])

        title = str(payload.get("title") or fallback.get("title") or "").strip() or str(fallback.get("title") or "")
        summary = str(payload.get("summary") or "").strip()
        participant_person_ids = [
            person_id
            for raw_person_id in list(payload.get("participant_person_ids", []) or [])
            if (person_id := self._coerce_signal_value(raw_person_id)) and person_id.lower() in valid_participants
        ] or list(fallback.get("participant_person_ids", []) or [])
        depicted_person_ids = [
            person_id
            for raw_person_id in list(payload.get("depicted_person_ids", []) or [])
            if (person_id := self._coerce_signal_value(raw_person_id)) and person_id.lower() in valid_depicted
        ] or list(fallback.get("depicted_person_ids", []) or [])
        place_refs = [
            place
            for raw_place in list(payload.get("place_refs", []) or [])
            if (place := self._coerce_signal_value(raw_place)) and place.lower() in valid_places
        ] or list(fallback.get("place_refs", []) or [])
        confidence = payload.get("confidence")
        try:
            confidence_value = float(confidence)
        except Exception:
            confidence_value = float(fallback.get("confidence") or 0.68)
        normalized_evidence: List[Dict[str, Any]] = []
        for item in list(payload.get("atomic_evidence", []) or [])[:16]:
            if not isinstance(item, dict):
                continue
            evidence_type = str(item.get("evidence_type") or "").strip()
            value_or_text = str(item.get("value_or_text") or "").strip()
            candidate_photo_ids = [
                photo_id
                for raw_photo_id in list(item.get("photo_ids", []) or [])
                if (photo_id := self._coerce_signal_value(raw_photo_id)) in valid_photo_ids
            ]
            if not evidence_type or not value_or_text or not candidate_photo_ids:
                continue
            original_photo_ids = self._unique(
                original_photo_id
                for photo_id in candidate_photo_ids
                for original_photo_id in photo_id_to_original.get(photo_id, [photo_id])
            )
            normalized_evidence.append(
                {
                    "evidence_id": self._stable_id("evidence", evidence_type, "|".join(original_photo_ids), value_or_text),
                    "evidence_type": evidence_type,
                    "value_or_text": value_or_text,
                    "original_photo_ids": original_photo_ids,
                    "confidence": float(item.get("confidence") or 0.65),
                    "provenance": str(item.get("provenance") or evidence_type),
                }
            )
        if not normalized_evidence:
            normalized_evidence = list(fallback.get("atomic_evidence", []) or [])
        normalized = dict(fallback)
        normalized.update(
            {
                "title": title,
                "event_summary": summary,
                "participant_person_ids": self._unique(participant_person_ids),
                "depicted_person_ids": self._unique(depicted_person_ids),
                "place_refs": self._unique(place_refs),
                "confidence": max(0.0, min(1.0, confidence_value)),
                "atomic_evidence": normalized_evidence,
            }
        )
        return normalized

    def _event_title(self, activity_hints: Sequence[str], place_refs: Sequence[str]) -> str:
        activity = str(activity_hints[0] if activity_hints else "life event").strip() or "life event"
        place = str(place_refs[0] if place_refs else "unknown place").strip() or "unknown place"
        return f"{activity} @ {place}"

    def _event_evidence_for_window(
        self,
        window: Dict[str, Any],
        observations_by_photo: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        for photo_id in window.get("photo_ids", []):
            observation = observations_by_photo.get(photo_id, {})
            original_photo_ids = list(observation.get("original_photo_ids", []) or [photo_id])
            for text in observation.get("ocr_hits", [])[:3]:
                evidence.append(self._evidence_record("ocr", original_photo_ids, text))
            for text in observation.get("brands", [])[:2]:
                evidence.append(self._evidence_record("brand", original_photo_ids, text))
            for text in observation.get("place_candidates", [])[:1]:
                evidence.append(self._evidence_record("place_candidate", original_photo_ids, text))
            for text in observation.get("embedded_media_person_ids", [])[:2]:
                evidence.append(self._evidence_record("embedded_media_person_reference", original_photo_ids, text))
        return evidence

    def _evidence_record(self, evidence_type: str, original_photo_ids: Sequence[str], value: str) -> Dict[str, Any]:
        photo_key = "|".join(original_photo_ids)
        return {
            "evidence_id": self._stable_id("evidence", evidence_type, photo_key, value),
            "evidence_type": evidence_type,
            "value_or_text": value,
            "original_photo_ids": list(original_photo_ids),
            "confidence": 0.6,
            "provenance": evidence_type,
        }

    def _collect_atomic_evidence(self, event_revisions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        filtered_evidence: List[Dict[str, Any]] = []
        for event in event_revisions:
            for item in list(event.get("atomic_evidence", []) or []):
                payload = dict(item)
                payload["root_event_revision_id"] = event["event_revision_id"]
                filtered_evidence.append(payload)
        return filtered_evidence

    def _resolve_event_drafts(
        self,
        drafts: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        changed_event_revisions: List[Dict[str, Any]] = []
        previous_root_id: Optional[str] = None
        for draft in sorted(drafts, key=lambda item: item.get("started_at") or ""):
            candidates = self.staging.list_candidate_event_revisions(event_started_at=draft.get("started_at") or "", limit=8)
            decision, prior = self._match_event_draft(draft, candidates)
            if prior and decision == "merge":
                root_id = prior["event_root_id"]
                revision = int(prior.get("revision") or 1) + 1
                merged = self._merge_event_revision(prior, draft, revision=revision)
                root_payload = {
                    "event_root_id": root_id,
                    "current_revision_id": merged["event_revision_id"],
                    "sealed_state": "open_frontier",
                    "pipeline_family": PIPELINE_FAMILY_V0321_3,
                }
                self.staging.upsert_event_revision(root_payload, merged)
                changed_event_revisions.append(merged)
                current_root_id = root_id
            else:
                root_id = self._stable_id("event-root", draft["window_id"])
                event_revision = {
                    "event_root_id": root_id,
                    "event_revision_id": self._stable_id("event-revision", root_id, "1"),
                    "revision": 1,
                    "title": draft["title"],
                    "started_at": draft["started_at"],
                    "ended_at": draft["ended_at"],
                    "participant_person_ids": list(draft["participant_person_ids"]),
                    "depicted_person_ids": list(draft["depicted_person_ids"]),
                    "place_refs": list(draft["place_refs"]),
                    "original_photo_ids": list(draft["original_photo_ids"]),
                    "boundary_reason": draft["boundary_reason"],
                    "confidence": draft["confidence"],
                    "status": "active",
                    "sealed_state": "open_frontier",
                    "pipeline_family": PIPELINE_FAMILY_V0321_3,
                    "atomic_evidence": [dict(item) for item in draft["atomic_evidence"]],
                }
                root_payload = {
                    "event_root_id": root_id,
                    "current_revision_id": event_revision["event_revision_id"],
                    "sealed_state": "open_frontier",
                    "pipeline_family": PIPELINE_FAMILY_V0321_3,
                }
                self.staging.upsert_event_revision(root_payload, event_revision)
                changed_event_revisions.append(event_revision)
                current_root_id = root_id
            previous_root_id = current_root_id
            threshold = self._parse_dt(draft.get("started_at"))
            if threshold:
                frontier_threshold = threshold - timedelta(hours=3)
                stale_frontier = [
                    event
                    for event in self.staging.list_open_frontier_event_revisions()
                    if self._parse_dt(event.get("ended_at")) and self._parse_dt(event.get("ended_at")) < frontier_threshold
                ]
                sealable_root_ids = [
                    event["event_root_id"]
                    for event in stale_frontier
                    if not self._should_keep_frontier_open(event, draft)
                ]
                self.staging.seal_event_roots(sealable_root_ids)
        current_events = self.staging.list_current_event_revisions()
        filtered_evidence = self._collect_atomic_evidence(current_events)
        self._summary["event_revision_count"] = len(current_events)
        return current_events, filtered_evidence, changed_event_revisions

    def _match_event_draft(self, draft: Dict[str, Any], candidates: Sequence[Dict[str, Any]]) -> Tuple[str, Optional[Dict[str, Any]]]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for candidate in candidates:
            if self._event_hard_block(draft, candidate):
                continue
            score = 0.0
            if self._has_overlap(draft.get("participant_person_ids", []) or [], candidate.get("participant_person_ids", []) or []):
                score += 0.35
            if self._has_overlap(draft.get("place_refs", []) or [], candidate.get("place_refs", []) or []):
                score += 0.30
            gap = self._time_gap_seconds(draft.get("started_at"), candidate.get("ended_at"))
            if gap is not None and gap <= 1800:
                score += 0.25
            elif gap is not None and gap <= 7200:
                score += 0.10
            if self._same_day(draft.get("started_at"), candidate.get("started_at")):
                score += 0.10
            scored.append((score, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored or scored[0][0] < 0.75:
            return "new", None
        if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.1:
            return "new", None
        return "merge", scored[0][1]

    def _event_hard_block(self, left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        gap = self._time_gap_seconds(left.get("started_at"), right.get("ended_at"))
        if gap is not None and gap > 12 * 3600:
            return True
        left_places = self._normalized_signal_set(left.get("place_refs", []) or [])
        right_places = self._normalized_signal_set(right.get("place_refs", []) or [])
        if left_places and right_places and not left_places.intersection(right_places):
            if not self._has_overlap(left.get("participant_person_ids", []) or [], right.get("participant_person_ids", []) or []):
                return True
        return False

    def _event_signal_tokens(self, event: Dict[str, Any]) -> List[str]:
        raw_tokens: List[str] = []
        raw_tokens.extend(str(place).strip() for place in list(event.get("place_refs", []) or []))
        raw_tokens.extend(
            token.strip().lower()
            for token in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", str(event.get("title") or ""))
            if token.strip()
        )
        raw_tokens.extend(
            token.strip().lower()
            for token in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", str(event.get("event_summary") or ""))
            if token.strip()
        )
        return self._unique(token for token in raw_tokens if len(token) >= 2)

    def _should_keep_frontier_open(self, candidate: Dict[str, Any], current_draft: Dict[str, Any]) -> bool:
        gap = self._time_gap_seconds(current_draft.get("started_at"), candidate.get("ended_at"))
        if gap is None or gap <= 0:
            return True
        if gap > 6 * 3600:
            return False
        live_overlap = self._has_overlap(
            candidate.get("participant_person_ids", []) or [],
            current_draft.get("participant_person_ids", []) or [],
        )
        place_overlap = self._has_overlap(
            candidate.get("place_refs", []) or [],
            current_draft.get("place_refs", []) or [],
        )
        signal_overlap = bool(
            set(self._event_signal_tokens(candidate)).intersection(self._event_signal_tokens(current_draft))
        )
        if live_overlap and place_overlap:
            return True
        if live_overlap and signal_overlap and gap <= 4 * 3600:
            return True
        if place_overlap and signal_overlap and gap <= 3 * 3600:
            return True
        return False

    def _merge_event_revision(self, prior: Dict[str, Any], draft: Dict[str, Any], *, revision: int) -> Dict[str, Any]:
        merged_evidence = list(prior.get("atomic_evidence", [])) + list(draft.get("atomic_evidence", []))
        return {
            "event_root_id": prior["event_root_id"],
            "event_revision_id": self._stable_id("event-revision", prior["event_root_id"], str(revision)),
            "revision": revision,
            "title": prior.get("title") or draft.get("title"),
            "started_at": min(filter(None, [prior.get("started_at"), draft.get("started_at")])),
            "ended_at": max(filter(None, [prior.get("ended_at"), draft.get("ended_at")])),
            "participant_person_ids": self._unique([*prior.get("participant_person_ids", []), *draft.get("participant_person_ids", [])]),
            "depicted_person_ids": self._unique([*prior.get("depicted_person_ids", []), *draft.get("depicted_person_ids", [])]),
            "place_refs": self._unique([*prior.get("place_refs", []), *draft.get("place_refs", [])]),
            "original_photo_ids": self._unique([*prior.get("original_photo_ids", []), *draft.get("original_photo_ids", [])]),
            "boundary_reason": f"merged:{prior.get('boundary_reason')}|{draft.get('boundary_reason')}",
            "confidence": max(float(prior.get("confidence") or 0.0), float(draft.get("confidence") or 0.0)),
            "status": "active",
            "sealed_state": "open_frontier",
            "pipeline_family": PIPELINE_FAMILY_V0321_3,
            "supersedes_event_revision_id": prior.get("event_revision_id"),
            "atomic_evidence": merged_evidence,
        }

    def _project_relationships(
        self,
        *,
        primary_person_id: Optional[str],
        event_revisions: Sequence[Dict[str, Any]],
        changed_event_revisions: Sequence[Dict[str, Any]],
        atomic_evidence: Sequence[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not primary_person_id:
            return self.staging.list_current_relationship_revisions(), [], []
        live_by_person: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        embedded_by_person: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for event in event_revisions:
            if primary_person_id in event.get("participant_person_ids", []):
                for person_id in event.get("participant_person_ids", []):
                    if person_id != primary_person_id:
                        live_by_person[person_id].append(event)
                for person_id in event.get("depicted_person_ids", []):
                    if person_id != primary_person_id:
                        embedded_by_person[person_id].append(event)
        affected_people = self._unique(
            person_id
            for event in changed_event_revisions
            for person_id in [
                *list(event.get("participant_person_ids", []) or []),
                *list(event.get("depicted_person_ids", []) or []),
            ]
            if person_id and person_id != primary_person_id
        )
        ledgers: List[Dict[str, Any]] = []
        changed_revisions: List[Dict[str, Any]] = []
        for person_id in sorted(affected_people):
            root_id = self._stable_id("relationship-root", self.user_id, person_id)
            prior = self.staging.get_current_relationship_revision(root_id)
            live_events = live_by_person.get(person_id, [])
            embedded_events = embedded_by_person.get(person_id, [])
            live_count = len(live_events)
            embedded_count = len(embedded_events)
            relationship_type = "co_presence_only"
            label = "共同出现"
            confidence = 0.35 + min(0.4, 0.08 * live_count)
            status = "active"
            if live_count >= 3:
                relationship_type = "friend"
                label = "朋友"
                confidence = min(0.92, 0.55 + 0.1 * live_count)
            elif live_count == 0 and embedded_count >= 2:
                relationship_type = "remembered_person"
                label = "被反复提及的人"
                confidence = min(0.68, 0.35 + 0.08 * embedded_count)
            llm_relationship = None
            if self._relationship_requires_llm(
                prior=prior,
                live_count=live_count,
                embedded_count=embedded_count,
                relationship_type=relationship_type,
            ):
                llm_relationship = self._infer_relationship_with_llm(
                    primary_person_id=primary_person_id,
                    target_person_id=person_id,
                    prior=prior,
                    live_events=live_events,
                    embedded_events=embedded_events,
                )
            if llm_relationship:
                relationship_type = str(llm_relationship.get("relationship_type") or relationship_type)
                label = str(llm_relationship.get("label") or label)
                try:
                    confidence = float(llm_relationship.get("confidence") or confidence)
                except Exception:
                    confidence = confidence
            revision = int(prior.get("revision") or 0) + 1 if prior else 1
            revision_payload = {
                "relationship_revision_id": self._stable_id("relationship-revision", root_id, str(revision), relationship_type),
                "relationship_root_id": root_id,
                "revision": revision,
                "status": status,
                "relationship_type": relationship_type,
                "label": label,
                "confidence": round(confidence, 4),
                "window_start": self._window_start(live_events, embedded_events),
                "window_end": self._window_end(live_events, embedded_events),
                "live_support_count": live_count,
                "embedded_support_count": embedded_count,
                "supporting_event_ids": self._unique(
                    event["event_revision_id"]
                    for event in [*live_events, *embedded_events]
                ),
                "supporting_photo_count": len(
                    self._unique(
                        photo_id
                        for event in [*live_events, *embedded_events]
                        for photo_id in event.get("original_photo_ids", [])
                    )
                ),
                "reason_summary": str((llm_relationship or {}).get("reason_summary") or f"{person_id}: live={live_count}, embedded={embedded_count}"),
                "feature_snapshot": {
                    "live_support_event_ids": [event["event_revision_id"] for event in live_events],
                    "embedded_support_event_ids": [event["event_revision_id"] for event in embedded_events],
                },
                "score_snapshot": {
                    "live_score": round(min(0.99, 0.12 * live_count), 4),
                    "embedded_score": round(min(0.99, 0.08 * embedded_count), 4),
                },
                "pipeline_family": PIPELINE_FAMILY_V0321_3,
                "target_person_id": person_id,
                "supersedes_relationship_revision_id": prior.get("relationship_revision_id") if prior else None,
            }
            root_payload = {
                "relationship_root_id": root_id,
                "target_person_id": person_id,
                "current_revision_id": revision_payload["relationship_revision_id"],
                "pipeline_family": PIPELINE_FAMILY_V0321_3,
            }
            self.staging.upsert_relationship_revision(root_payload, revision_payload)
            changed_revisions.append(revision_payload)
            ledgers.append(
                {
                    "relationship_revision_id": revision_payload["relationship_revision_id"],
                    "entries": [
                        *[
                            {
                                "event_revision_id": event["event_revision_id"],
                                "photo_ids": event.get("original_photo_ids", []),
                                "appearance_mode": "live_presence",
                            }
                            for event in live_events
                        ],
                        *[
                            {
                                "event_revision_id": event["event_revision_id"],
                                "photo_ids": event.get("original_photo_ids", []),
                                "appearance_mode": "embedded_media",
                            }
                            for event in embedded_events
                        ],
                    ],
                }
            )
        revisions = self.staging.list_current_relationship_revisions()
        self._summary["relationship_revision_count"] = len(revisions)
        return revisions, ledgers, changed_revisions

    def _relationship_requires_llm(
        self,
        *,
        prior: Optional[Dict[str, Any]],
        live_count: int,
        embedded_count: int,
        relationship_type: str,
    ) -> bool:
        if live_count >= 2:
            return True
        if live_count == 0 and embedded_count >= 2:
            return True
        if prior and str(prior.get("relationship_type") or "") != relationship_type:
            return True
        if prior and float(prior.get("confidence") or 0.0) < 0.55 <= (0.35 + min(0.4, 0.08 * live_count)):
            return True
        return False

    def _infer_relationship_with_llm(
        self,
        *,
        primary_person_id: str,
        target_person_id: str,
        prior: Optional[Dict[str, Any]],
        live_events: Sequence[Dict[str, Any]],
        embedded_events: Sequence[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        prompt = (
            "You are a relationship analyst. Infer the current relationship between the primary user and the target person.\n"
            "Return JSON only with keys relationship_type, label, confidence, reason_summary.\n"
            "Use live_presence evidence as stronger than embedded_media evidence.\n"
            "Do not treat embedded_media as direct co-presence.\n"
            "Allowed relationship_type values: co_presence_only, acquaintance, colleague, friend, close_friend, family, remembered_person.\n"
            f"PRIMARY_PERSON_ID={json.dumps(primary_person_id, ensure_ascii=False)}\n"
            f"TARGET_PERSON_ID={json.dumps(target_person_id, ensure_ascii=False)}\n"
            f"PRIOR_RELATIONSHIP={json.dumps(prior or {}, ensure_ascii=False)}\n"
            f"LIVE_EVENTS={json.dumps([self._relationship_event_context(event) for event in live_events], ensure_ascii=False)}\n"
            f"EMBEDDED_MEDIA_EVENTS={json.dumps([self._relationship_event_context(event) for event in embedded_events], ensure_ascii=False)}\n"
        )
        payload = self._call_llm_json_prompt(prompt)
        if not payload:
            return None
        self._summary["relationship_llm_count"] = int(self._summary.get("relationship_llm_count") or 0) + 1
        return payload

    def _relationship_event_context(self, event: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "event_revision_id": event.get("event_revision_id"),
            "title": event.get("title"),
            "started_at": event.get("started_at"),
            "ended_at": event.get("ended_at"),
            "place_refs": list(event.get("place_refs", []) or []),
            "participant_person_ids": list(event.get("participant_person_ids", []) or []),
            "depicted_person_ids": list(event.get("depicted_person_ids", []) or []),
            "original_photo_ids": list(event.get("original_photo_ids", []) or []),
            "event_summary": str(event.get("event_summary") or ""),
        }

    def _build_profile_input_pack_partial(
        self,
        *,
        primary_person_id: Optional[str],
        event_revisions: Sequence[Dict[str, Any]],
        atomic_evidence: Sequence[Dict[str, Any]],
        reference_signals: Sequence[Dict[str, Any]],
        scope: str,
    ) -> Dict[str, Any]:
        return {
            "profile_input_pack_id": self._stable_id("profile-input-pack-partial", scope, len(event_revisions), len(reference_signals)),
            "pipeline_family": PIPELINE_FAMILY_V0321_3,
            "scope": scope,
            "primary_person_id": primary_person_id,
            "time_range": self._build_profile_time_range(event_revisions),
            "baseline_rhythm": self._build_baseline_rhythm(event_revisions),
            "place_patterns": self._build_place_patterns(event_revisions),
            "activity_patterns": self._build_activity_patterns(event_revisions),
            "event_grounded_signals": self._build_event_grounded_signals(
                event_revisions=event_revisions,
                atomic_evidence=atomic_evidence,
            ),
            "reference_media_weak_signals": self._build_reference_media_weak_signals(reference_signals),
            "key_event_refs": self._select_key_event_refs(event_revisions),
            "evidence_guardrails": self._build_profile_evidence_guardrails(),
        }

    def _build_profile_input_pack(
        self,
        *,
        profile_input_pack_partial: Dict[str, Any],
        relationship_revisions: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        pack = dict(profile_input_pack_partial)
        pack["profile_input_pack_id"] = self._stable_id(
            "profile-input-pack",
            pack.get("scope"),
            len(pack.get("key_event_refs", []) or []),
            len(relationship_revisions),
        )
        pack["social_patterns"] = self._build_social_patterns(relationship_revisions)
        pack["change_points"] = self._build_change_points(
            profile_input_pack_partial=profile_input_pack_partial,
            relationship_revisions=relationship_revisions,
        )
        pack["key_relationship_refs"] = self._select_key_relationship_refs(relationship_revisions)
        return pack

    def _build_profile_time_range(self, event_revisions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        starts = [item.get("started_at") for item in event_revisions if item.get("started_at")]
        ends = [item.get("ended_at") for item in event_revisions if item.get("ended_at")]
        return {
            "start": min(starts) if starts else None,
            "end": max(ends) if ends else None,
        }

    def _build_baseline_rhythm(self, event_revisions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        distribution = {"morning": 0, "afternoon": 0, "evening": 0, "late_night": 0}
        weekday_event_count = 0
        weekend_event_count = 0
        late_night_count = 0
        for event in event_revisions:
            started_at = self._parse_dt(event.get("started_at"))
            if not started_at:
                continue
            bucket = self._time_bucket_label(started_at.hour)
            distribution[bucket] += 1
            if started_at.weekday() >= 5:
                weekend_event_count += 1
            else:
                weekday_event_count += 1
            if bucket == "late_night":
                late_night_count += 1
        total = sum(distribution.values())
        dominant = max(distribution, key=distribution.get) if total else "unknown"
        return {
            "active_hour_distribution": distribution,
            "weekday_event_count": weekday_event_count,
            "weekend_event_count": weekend_event_count,
            "dominant_activity_window": dominant,
            "late_night_ratio": round(late_night_count / total, 4) if total else 0.0,
        }

    def _build_place_patterns(self, event_revisions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        place_ref_counts: Dict[str, int] = defaultdict(int)
        place_type_counts: Dict[str, int] = defaultdict(int)
        for event in event_revisions:
            place_refs = list(event.get("place_refs", []) or [])
            if not place_refs:
                place_type_counts["unknown"] += 1
                continue
            for place_ref in place_refs:
                place_text = str(place_ref).strip()
                if not place_text:
                    continue
                place_ref_counts[place_text] += 1
                place_type_counts[self._place_type_for_ref(place_text)] += 1
        total_place_events = sum(place_type_counts.values())
        unique_refs = len(place_ref_counts)
        return {
            "top_place_types": [
                {"place_type": label, "count": count}
                for label, count in sorted(place_type_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
            ],
            "top_place_refs": [
                {"place_ref": label, "count": count}
                for label, count in sorted(place_ref_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
            ],
            "place_diversity_score": round(unique_refs / total_place_events, 4) if total_place_events else 0.0,
        }

    def _build_activity_patterns(self, event_revisions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        activity_counts: Dict[str, int] = defaultdict(int)
        solo_like_events = 0
        social_like_events = 0
        unclear_events = 0
        for event in event_revisions:
            activity_label = self._activity_label_for_event(event)
            activity_counts[activity_label] += 1
            participant_count = len(list(event.get("participant_person_ids", []) or []))
            if participant_count >= 2:
                social_like_events += 1
            elif participant_count == 1:
                solo_like_events += 1
            else:
                unclear_events += 1
        repeated_small_patterns = [
            label
            for label, count in sorted(activity_counts.items(), key=lambda item: (-item[1], item[0]))
            if count >= 2
        ][:5]
        return {
            "top_activities": [
                {"activity_type": label, "count": count}
                for label, count in sorted(activity_counts.items(), key=lambda item: (-item[1], item[0]))[:6]
            ],
            "solo_like_events": solo_like_events,
            "social_like_events": social_like_events,
            "unclear_events": unclear_events,
            "repeated_small_patterns": repeated_small_patterns,
        }

    def _build_event_grounded_signals(
        self,
        *,
        event_revisions: Sequence[Dict[str, Any]],
        atomic_evidence: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        grouped: Dict[str, Dict[str, Dict[str, Any]]] = {
            "interest_signals": {},
            "aesthetic_signals": {},
            "identity_style_signals": {},
        }
        event_by_id = {
            str(event.get("event_revision_id") or ""): event
            for event in event_revisions
            if str(event.get("event_revision_id") or "")
        }
        for event in event_revisions:
            event_id = str(event.get("event_revision_id") or "")
            activity_label = self._activity_label_for_event(event)
            self._accumulate_profile_signal(
                grouped["interest_signals"],
                label=activity_label,
                supporting_event_id=event_id,
            )
            inferred_aesthetic = self._infer_event_aesthetic_signal(event)
            if inferred_aesthetic:
                self._accumulate_profile_signal(
                    grouped["aesthetic_signals"],
                    label=inferred_aesthetic,
                    supporting_event_id=event_id,
                )
            inferred_identity_style = self._infer_event_identity_style_signal(event)
            if inferred_identity_style:
                self._accumulate_profile_signal(
                    grouped["identity_style_signals"],
                    label=inferred_identity_style,
                    supporting_event_id=event_id,
                )
        for evidence in atomic_evidence:
            if evidence.get("evidence_type") != "profile_signal":
                continue
            event_id = str(evidence.get("root_event_revision_id") or "")
            signal_type = self._event_profile_signal_kind(str(evidence.get("value_or_text") or ""))
            target_group = grouped[signal_type]
            self._accumulate_profile_signal(
                target_group,
                label=str(evidence.get("value_or_text") or "").strip(),
                supporting_event_id=event_id if event_id in event_by_id else None,
            )
        return {
            key: [
                {
                    "label": label,
                    "count": payload["count"],
                    "supporting_event_ids": payload["supporting_event_ids"][:8],
                }
                for label, payload in sorted(value.items(), key=lambda item: (-item[1]["count"], item[0]))[:6]
            ]
            for key, value in grouped.items()
        }

    def _build_reference_media_weak_signals(
        self,
        reference_signals: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        bucket_mapping = {
            "interest": "interest_hints",
            "aesthetic_preference": "aesthetic_hints",
            "aspiration": "aspiration_hints",
            "consumption_intent": "aspiration_hints",
            "identity_style": "identity_style_hints",
        }
        grouped: Dict[str, Dict[str, Dict[str, Any]]] = {
            "interest_hints": {},
            "aesthetic_hints": {},
            "aspiration_hints": {},
            "identity_style_hints": {},
        }
        for signal in reference_signals:
            bucket = bucket_mapping.get(str(signal.get("profile_bucket") or "interest"), "interest_hints")
            label = str(signal.get("evidence_text") or "").strip()
            if not label:
                continue
            payload = grouped[bucket].setdefault(
                label,
                {"count": 0, "supporting_signal_ids": [], "source_kinds": []},
            )
            payload["count"] += 1
            if signal.get("signal_id"):
                payload["supporting_signal_ids"].append(str(signal["signal_id"]))
            if signal.get("source_kind"):
                payload["source_kinds"].append(str(signal["source_kind"]))
        return {
            key: [
                {
                    "label": label,
                    "count": payload["count"],
                    "supporting_signal_ids": self._unique(payload["supporting_signal_ids"])[:8],
                    "source_kinds": self._unique(payload["source_kinds"])[:3],
                }
                for label, payload in sorted(value.items(), key=lambda item: (-item[1]["count"], item[0]))[:6]
            ]
            for key, value in grouped.items()
        }

    def _select_key_event_refs(self, event_revisions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored: List[Tuple[float, Dict[str, Any], str]] = []
        for event in event_revisions:
            participant_count = len(list(event.get("participant_person_ids", []) or []))
            evidence_count = len(list(event.get("atomic_evidence", []) or []))
            photo_count = len(list(event.get("original_photo_ids", []) or []))
            score = participant_count * 3 + evidence_count * 1.5 + photo_count
            why_selected = "高代表性的事件样本"
            if participant_count >= 2:
                why_selected = "社交关系代表样本"
            elif evidence_count >= 4:
                why_selected = "细节较丰富的事件样本"
            scored.append((score, event, why_selected))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("started_at") or ""), str(item[1].get("event_revision_id") or "")))
        return [
            {
                "event_revision_id": item.get("event_revision_id"),
                "title": item.get("title"),
                "why_selected": why_selected,
                "supporting_photo_ids": list(item.get("original_photo_ids", []) or [])[:6],
            }
            for _, item, why_selected in scored[:12]
        ]

    def _build_profile_evidence_guardrails(self) -> Dict[str, Any]:
        return {
            "strong_evidence_only_sections": [
                "常驻地/通勤模式",
                "是否单身/伴侣是谁",
                "核心身份/阶层",
            ],
            "weak_signal_only_sections": [
                "兴趣爱好补充观察",
                "审美偏好补充观察",
                "向往与计划补充观察",
            ],
            "forbidden_direct_inference_from_reference_media": [
                "真实到访",
                "真实关系",
                "真实职业",
                "真实消费能力",
            ],
        }

    def _build_social_patterns(self, relationship_revisions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        close_relationship_count = 0
        co_presence_only_count = 0
        one_on_one_score = 0.0
        group_score = 0.0
        top_relationships = []
        for relationship in sorted(
            relationship_revisions,
            key=lambda item: (
                -float(item.get("confidence") or 0.0),
                -int(item.get("live_support_count") or 0),
                str(item.get("target_person_id") or ""),
            ),
        ):
            relationship_type = str(relationship.get("relationship_type") or "")
            if relationship_type in {"friend", "close_friend", "family"}:
                close_relationship_count += 1
            if relationship_type == "co_presence_only":
                co_presence_only_count += 1
            live_support_count = int(relationship.get("live_support_count") or 0)
            embedded_support_count = int(relationship.get("embedded_support_count") or 0)
            if live_support_count >= 2:
                one_on_one_score += live_support_count
            else:
                group_score += max(1, live_support_count + embedded_support_count)
            top_relationships.append(
                {
                    "relationship_revision_id": relationship.get("relationship_revision_id"),
                    "target_person_id": relationship.get("target_person_id"),
                    "relationship_type": relationship_type,
                    "label": relationship.get("label"),
                    "confidence": relationship.get("confidence"),
                    "live_support_count": live_support_count,
                    "embedded_support_count": embedded_support_count,
                    "status": relationship.get("status"),
                    "supporting_event_ids": list(relationship.get("supporting_event_ids", []) or [])[:6],
                }
            )
        total_bias = one_on_one_score + group_score
        one_on_one_bias = round(one_on_one_score / total_bias, 4) if total_bias else 0.0
        group_bias = round(group_score / total_bias, 4) if total_bias else 0.0
        active_count = sum(1 for item in relationship_revisions if str(item.get("status") or "") == "active")
        stability = "low"
        if active_count >= 3:
            stability = "high"
        elif active_count >= 1:
            stability = "medium"
        return {
            "top_relationships": top_relationships[:8],
            "relationship_summary": {
                "close_relationship_count": close_relationship_count,
                "co_presence_only_count": co_presence_only_count,
                "active_relationship_count": active_count,
            },
            "social_style_hints": {
                "one_on_one_bias": one_on_one_bias,
                "group_bias": group_bias,
                "relationship_stability": stability,
            },
        }

    def _build_change_points(
        self,
        *,
        profile_input_pack_partial: Dict[str, Any],
        relationship_revisions: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        change_points: List[Dict[str, Any]] = []
        top_activities = list((profile_input_pack_partial.get("activity_patterns") or {}).get("top_activities", []) or [])
        if len(top_activities) >= 2 and int(top_activities[0].get("count") or 0) >= int(top_activities[1].get("count") or 0) + 2:
            change_points.append(
                {
                    "change_type": "activity_dominance",
                    "description": f"{top_activities[0].get('activity_type')} 是当前最强的重复活动模式",
                    "supporting_event_ids": [
                        item.get("event_revision_id")
                        for item in list(profile_input_pack_partial.get("key_event_refs", []) or [])[:3]
                        if item.get("event_revision_id")
                    ],
                }
            )
        for relationship in sorted(
            relationship_revisions,
            key=lambda item: (-int(item.get("live_support_count") or 0), -float(item.get("confidence") or 0.0)),
        ):
            if int(relationship.get("live_support_count") or 0) < 2:
                continue
            change_points.append(
                {
                    "change_type": "new_person_prominence",
                    "description": f"{relationship.get('target_person_id')} 在当前关系网络中开始变得更显著",
                    "supporting_relationship_ids": [relationship.get("relationship_revision_id")],
                    "supporting_event_ids": list(relationship.get("supporting_event_ids", []) or [])[:4],
                }
            )
            break
        return change_points[:6]

    def _select_key_relationship_refs(self, relationship_revisions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        scored = sorted(
            relationship_revisions,
            key=lambda item: (
                -float(item.get("confidence") or 0.0),
                -int(item.get("live_support_count") or 0),
                str(item.get("target_person_id") or ""),
            ),
        )
        return [
            {
                "relationship_revision_id": item.get("relationship_revision_id"),
                "target_person_id": item.get("target_person_id"),
                "why_selected": "最高置信且支持事件较多的关系样本",
                "supporting_event_ids": list(item.get("supporting_event_ids", []) or [])[:6],
            }
            for item in scored[:8]
        ]

    def _build_profile_input_pack_preview(self, profile_input_pack: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not profile_input_pack:
            return {}
        activity_patterns = dict(profile_input_pack.get("activity_patterns") or {})
        social_patterns = dict(profile_input_pack.get("social_patterns") or {})
        return {
            "scope": profile_input_pack.get("scope"),
            "key_event_ref_count": len(list(profile_input_pack.get("key_event_refs", []) or [])),
            "key_relationship_ref_count": len(list(profile_input_pack.get("key_relationship_refs", []) or [])),
            "top_activities": [item.get("activity_type") for item in list(activity_patterns.get("top_activities", []) or [])[:3]],
            "top_relationships": [
                {
                    "target_person_id": item.get("target_person_id"),
                    "relationship_type": item.get("relationship_type"),
                }
                for item in list(social_patterns.get("top_relationships", []) or [])[:3]
            ],
            "weak_signal_count": sum(len(list(value or [])) for value in dict(profile_input_pack.get("reference_media_weak_signals") or {}).values()),
        }

    def _build_profile_revision(
        self,
        *,
        primary_person_id: Optional[str],
        event_revisions: Sequence[Dict[str, Any]],
        atomic_evidence: Sequence[Dict[str, Any]],
        relationship_revisions: Sequence[Dict[str, Any]],
        reference_signals: Sequence[Dict[str, Any]],
        profile_input_pack: Dict[str, Any],
        revision_key: str = "1",
        scope: str = "cumulative",
    ) -> Tuple[Dict[str, Any], str]:
        buckets: Dict[str, Dict[str, Any]] = {}
        has_reference_only = bool(reference_signals)
        has_event_grounded = False
        for signal in reference_signals:
            bucket = signal["profile_bucket"]
            bucket_payload = buckets.setdefault(
                bucket,
                {"values": [], "original_photo_ids": [], "evidence_refs": [], "source_kinds": []},
            )
            bucket_payload["values"].append(signal["evidence_text"])
            bucket_payload["original_photo_ids"].extend(signal["original_photo_ids"])
            bucket_payload["source_kinds"].append(signal.get("source_kind") or "reference_media")
            bucket_payload["evidence_refs"].append(
                {"ref_type": "reference_media", "ref_id": signal["signal_id"]}
            )
        for evidence in atomic_evidence:
            if evidence.get("evidence_type") != "profile_signal":
                continue
            has_event_grounded = True
            bucket_payload = buckets.setdefault(
                "interest",
                {"values": [], "original_photo_ids": [], "evidence_refs": [], "source_kinds": []},
            )
            bucket_payload["values"].append(str(evidence.get("value_or_text") or ""))
            bucket_payload["original_photo_ids"].extend(evidence.get("original_photo_ids", []))
            bucket_payload["source_kinds"].append(evidence.get("provenance") or "event_profile_signal")
            bucket_payload["evidence_refs"].append(
                {"ref_type": "evidence", "ref_id": evidence["evidence_id"]}
            )
        normalized_buckets = {
            key: {
                "values": self._unique(value["values"]),
                "original_photo_ids": self._unique(value["original_photo_ids"]),
                "evidence_refs": value["evidence_refs"][:20],
                "source_kinds": self._unique(value.get("source_kinds", [])),
            }
            for key, value in buckets.items()
        }
        profile_revision = {
            "profile_revision_id": self._stable_id("profile-revision", self.user_id, revision_key),
            "pipeline_family": PIPELINE_FAMILY_V0321_3,
            "version": 1,
            "scope": scope,
            "primary_person_id": primary_person_id,
            "profile_input_pack_id": profile_input_pack.get("profile_input_pack_id"),
            "buckets": normalized_buckets,
            "original_photo_ids": self._unique(
                photo_id
                for bucket in normalized_buckets.values()
                for photo_id in bucket.get("original_photo_ids", [])
            ),
        }
        llm_markdown = self._generate_profile_markdown_with_llm(
            primary_person_id=primary_person_id,
            event_revisions=event_revisions,
            relationship_revisions=relationship_revisions,
            profile_input_pack=profile_input_pack,
            scope=scope,
        )
        if llm_markdown:
            self._summary["profile_llm_count"] = int(self._summary.get("profile_llm_count") or 0) + 1
            return profile_revision, llm_markdown
        lines = ["# Profile", ""]
        if not normalized_buckets:
            lines.append("- 当前任务还没有足够稳定的画像信号。")
            return profile_revision, "\n".join(lines).strip()

        if has_reference_only and not has_event_grounded:
            lines.append("- 本轮画像主要来自截图、网图或参考图，只能作为弱画像线索，不代表已经发生的真实经历。")
            lines.append("- 这些内容更适合用于判断近期关注、审美偏好和潜在兴趣，不适合直接推断真实生活事件。")
            lines.append("")

        bucket_labels = {
            "interest": "近期关注内容",
            "aesthetic_preference": "审美与风格偏好",
            "consumption_intent": "潜在消费/种草方向",
            "aspiration": "向往与计划线索",
            "identity_style": "身份与表达风格",
        }
        for bucket, payload in sorted(normalized_buckets.items()):
            values = [str(item).strip() for item in payload.get("values", []) if str(item).strip()]
            if not values:
                continue
            headline = bucket_labels.get(bucket, bucket)
            lines.append(f"## {headline}")
            lines.append(f"- 线索数量：{len(values)}")
            if payload.get("source_kinds"):
                lines.append(f"- 来源：{', '.join(payload['source_kinds'][:3])}")
            lines.append(f"- 代表信号：{self._summarize_profile_values(values)}")
            lines.append("")
        return profile_revision, "\n".join(lines).strip()

    def _generate_profile_markdown_with_llm(
        self,
        *,
        primary_person_id: Optional[str],
        event_revisions: Sequence[Dict[str, Any]],
        relationship_revisions: Sequence[Dict[str, Any]],
        profile_input_pack: Dict[str, Any],
        scope: str,
    ) -> Optional[str]:
        markdown = self._call_llm_markdown_prompt(
            self._build_profile_llm_prompt(
                primary_person_id=primary_person_id,
                event_revisions=event_revisions,
                relationship_revisions=relationship_revisions,
                profile_input_pack=profile_input_pack,
                scope=scope,
            )
        )
        return markdown

    def _build_profile_llm_prompt(
        self,
        *,
        primary_person_id: Optional[str],
        event_revisions: Sequence[Dict[str, Any]],
        relationship_revisions: Sequence[Dict[str, Any]],
        profile_input_pack: Dict[str, Any],
        scope: str,
    ) -> str:
        key_event_ids = [
            str(item.get("event_revision_id") or "")
            for item in list(profile_input_pack.get("key_event_refs", []) or [])
            if str(item.get("event_revision_id") or "")
        ]
        key_relationship_ids = [
            str(item.get("relationship_revision_id") or "")
            for item in list(profile_input_pack.get("key_relationship_refs", []) or [])
            if str(item.get("relationship_revision_id") or "")
        ]
        event_lookup = {
            str(event.get("event_revision_id") or ""): event
            for event in event_revisions
            if str(event.get("event_revision_id") or "")
        }
        relationship_lookup = {
            str(item.get("relationship_revision_id") or ""): item
            for item in relationship_revisions
            if str(item.get("relationship_revision_id") or "")
        }
        key_event_payload = [
            self._profile_event_context(event_lookup[event_id])
            for event_id in key_event_ids[:12]
            if event_id in event_lookup
        ]
        key_relationship_payload = [
            self._profile_relationship_context(relationship_lookup[relationship_id])
            for relationship_id in key_relationship_ids[:8]
            if relationship_id in relationship_lookup
        ]
        scope_label = "本轮当前任务增量画像" if scope == "current_task" else "累计长期画像"
        return (
            "# Role\n"
            "你是一位世界级的行为分析专家和 FBI 级别的人格画像师，擅长通过行为残迹还原人类灵魂。\n\n"
            "# Task\n"
            f"请基于提供的结构化人物档案、关键事件样本和关键关系样本，生成《用户全维画像分析报告》。当前任务目标是：{scope_label}。\n\n"
            "# Special Rules\n"
            "1. 只能把 PROFILE_INPUT_PACK 当作主输入，不要回头假设还有未提供的全量事件。\n"
            "2. 没有 supporting_event_ids 或 supporting_relationship_ids 的强结论禁止出现。\n"
            "3. reference_media_weak_signals 只能进入“补充观察/弱线索”，不能直接当成真实经历、真实关系、真实职业或真实消费能力。\n"
            "4. MBTI、伴侣判断、职业判断、阶层判断、心理状态判断如果证据不够，必须显式写“待进一步观察”或“证据不足”。\n"
            "5. 输出必须使用中文 Markdown，并尽量遵守以下结构：推理草稿箱、基础画像、社交关系图谱、深度人格分析。\n\n"
            "Output Format (尽量遵守)\n"
            "推理草稿箱 (Reasoning Scratchpad)\n"
            "1. 时空锚点确认\n"
            "2. 社交实体挖掘\n"
            "3. 职业逻辑修正\n\n"
            "1. 基础画像\n"
            "1.1 姓名/称呼推测\n"
            "1.2 估算年龄/生命周期\n"
            "1.3 核心身份/阶层\n"
            "1.4 常驻地/通勤模式\n"
            "1.5 兴趣爱好\n"
            "1.6 是否单身/伴侣是谁\n\n"
            "2. 社交关系图谱\n"
            "2.1 重要关系识别表\n"
            "2.2 社交性格总结\n\n"
            "3. 深度人格分析\n"
            "3.1 性格特征与 MBTI\n"
            "3.2 价值观与底层驱动力\n"
            "3.3 审美偏好和对外展示\n"
            "3.4 底层人格侧写\n\n"
            f"PRIMARY_PERSON_ID={json.dumps(primary_person_id, ensure_ascii=False)}\n"
            f"PROFILE_INPUT_PACK={json.dumps(profile_input_pack, ensure_ascii=False)}\n"
            f"KEY_EVENT_CONTEXT={json.dumps(key_event_payload, ensure_ascii=False)}\n"
            f"KEY_RELATIONSHIP_CONTEXT={json.dumps(key_relationship_payload, ensure_ascii=False)}\n"
        )

    def _time_bucket_label(self, hour: int) -> str:
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 18:
            return "afternoon"
        if 18 <= hour < 24:
            return "evening"
        return "late_night"

    def _place_type_for_ref(self, place_ref: str) -> str:
        normalized = str(place_ref or "").lower()
        if any(token in normalized for token in ("home", "居家", "卧室", "客厅", "室内")):
            return "indoor_home_like"
        if any(token in normalized for token in ("cafe", "coffee", "restaurant", "餐", "咖啡", "食堂")):
            return "restaurant_cafe"
        if any(token in normalized for token in ("mall", "shop", "store", "商场", "店", "展")):
            return "mall_commercial"
        if any(token in normalized for token in ("office", "company", "school", "campus", "办公室", "学校")):
            return "work_or_study"
        if any(token in normalized for token in ("station", "airport", "hotel", "trip", "车站", "机场", "酒店")):
            return "travel_transit"
        return "other"

    def _event_text_blob(self, event: Dict[str, Any]) -> str:
        evidence_values = [
            str(item.get("value_or_text") or "")
            for item in list(event.get("atomic_evidence", []) or [])[:12]
            if item.get("value_or_text")
        ]
        return " ".join(
            str(part).lower()
            for part in [
                event.get("title"),
                event.get("event_summary"),
                " ".join(str(place) for place in list(event.get("place_refs", []) or [])),
                " ".join(evidence_values),
            ]
            if part
        )

    def _activity_label_for_event(self, event: Dict[str, Any]) -> str:
        text = self._event_text_blob(event)
        if any(token in text for token in ("selfie", "自拍", "portrait", "pose")):
            return "自拍/自我展示"
        if any(token in text for token in ("cat", "dog", "pet", "猫", "狗", "宠物")):
            return "宠物陪伴"
        if any(token in text for token in ("meal", "dining", "breakfast", "lunch", "dinner", "coffee", "restaurant", "用餐", "吃饭", "咖啡")):
            return "用餐/轻社交"
        if any(token in text for token in ("shopping", "mall", "museum", "exhibition", "jewelry", "展览", "商场", "购物")):
            return "购物/展览"
        if any(token in text for token in ("ticket", "boarding", "flight", "hotel", "trip", "出行", "机票", "行程", "酒店")):
            return "出行/移动"
        if any(token in text for token in ("meeting", "office", "study", "school", "work", "会议", "学习", "工作")):
            return "工作/学习"
        if any(token in text for token in ("call", "chat", "video call", "message", "聊天", "通话")):
            return "数字社交"
        return "日常生活记录"

    def _accumulate_profile_signal(
        self,
        target_group: Dict[str, Dict[str, Any]],
        *,
        label: str,
        supporting_event_id: Optional[str],
    ) -> None:
        cleaned_label = re.sub(r"\s+", " ", str(label or "")).strip(" ,.;:：；，。")
        if not cleaned_label:
            return
        payload = target_group.setdefault(
            cleaned_label,
            {"count": 0, "supporting_event_ids": []},
        )
        payload["count"] += 1
        if supporting_event_id:
            payload["supporting_event_ids"].append(str(supporting_event_id))

    def _infer_event_aesthetic_signal(self, event: Dict[str, Any]) -> Optional[str]:
        text = self._event_text_blob(event)
        if any(token in text for token in ("cute", "hello kitty", "动漫", "可爱", "娃娃", "粉色", "口罩")):
            return "可爱/萌系视觉元素在线下事件中重复出现"
        if any(token in text for token in ("jewelry", "museum", "gold", "饰品", "展览", "古风")):
            return "风格化装饰与展陈内容反复出现"
        if any(token in text for token in ("outfit", "style", "fashion", "穿搭", "风格")):
            return "风格与穿搭相关线索在线下事件中出现"
        return None

    def _infer_event_identity_style_signal(self, event: Dict[str, Any]) -> Optional[str]:
        text = self._event_text_blob(event)
        if any(token in text for token in ("selfie", "自拍", "portrait")):
            return "更偏自我记录而非纯风景记录"
        if any(token in text for token in ("note", "reflection", "备忘", "记录")):
            return "有持续记录和整理生活片段的倾向"
        return None

    def _event_profile_signal_kind(self, text: str) -> str:
        normalized = str(text or "").lower()
        if any(token in normalized for token in ("style", "fashion", "aesthetic", "穿搭", "风格", "审美")):
            return "aesthetic_signals"
        if any(token in normalized for token in ("self", "expression", "record", "自拍", "表达", "记录")):
            return "identity_style_signals"
        return "interest_signals"

    def _profile_event_context(self, event: Dict[str, Any]) -> Dict[str, Any]:
        evidence = []
        for item in list(event.get("atomic_evidence", []) or [])[:8]:
            evidence.append(
                {
                    "evidence_type": item.get("evidence_type"),
                    "value_or_text": item.get("value_or_text"),
                    "original_photo_ids": list(item.get("original_photo_ids", []) or []),
                    "evidence_id": item.get("evidence_id"),
                }
            )
        return {
            "event_revision_id": event.get("event_revision_id"),
            "title": event.get("title"),
            "event_summary": event.get("event_summary"),
            "started_at": event.get("started_at"),
            "ended_at": event.get("ended_at"),
            "participant_person_ids": list(event.get("participant_person_ids", []) or []),
            "depicted_person_ids": list(event.get("depicted_person_ids", []) or []),
            "place_refs": list(event.get("place_refs", []) or []),
            "original_photo_ids": list(event.get("original_photo_ids", []) or []),
            "atomic_evidence": evidence,
        }

    def _profile_relationship_context(self, relationship: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "relationship_revision_id": relationship.get("relationship_revision_id"),
            "target_person_id": relationship.get("target_person_id"),
            "relationship_type": relationship.get("relationship_type"),
            "label": relationship.get("label"),
            "confidence": relationship.get("confidence"),
            "live_support_count": relationship.get("live_support_count"),
            "embedded_support_count": relationship.get("embedded_support_count"),
            "supporting_event_ids": list(relationship.get("supporting_event_ids", []) or []),
            "reason_summary": relationship.get("reason_summary"),
        }

    def _summarize_profile_values(self, values: Sequence[str]) -> str:
        cleaned = self._unique(
            re.sub(r"\s+", " ", str(value)).strip(" ,.;:：；，。")
            for value in values
            if str(value).strip()
        )
        if not cleaned:
            return "暂无可用线索"
        compact = cleaned[:3]
        normalized = []
        for item in compact:
            if len(item) > 80:
                normalized.append(f"{item[:77]}...")
            else:
                normalized.append(item)
        return "；".join(normalized)

    def _build_storage_payload(
        self,
        *,
        primary_person_id: Optional[str],
        face_output: Dict[str, Any],
        person_appearances: Sequence[Dict[str, Any]],
        event_revisions: Sequence[Dict[str, Any]],
        relationship_revisions: Sequence[Dict[str, Any]],
        relationship_ledgers: Sequence[Dict[str, Any]],
        profile_revision: Dict[str, Any],
        reference_signals: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        person_ids = self._unique(item["person_id"] for item in person_appearances)
        persons = []
        redis_payload: Dict[str, Any] = {
            "profile_current": {
                "key": f"{self.family_prefix}:profile_current",
                **profile_revision,
            },
            "profile_revision": {
                "key": f"{self.family_prefix}:profile_revision:{profile_revision['version']}",
                **profile_revision,
            },
            "reference_media_catalog": {
                "key": f"{self.family_prefix}:reference_media",
                "items": list(reference_signals),
            },
        }
        for person_id in person_ids:
            person_appearance_items = [item for item in person_appearances if item["person_id"] == person_id]
            original_photo_ids = self._unique(
                item.get("original_photo_id") or item["photo_id"]
                for item in person_appearance_items
            )
            segments: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for item in person_appearance_items:
                month_key = str(item.get("timestamp") or "")[:7] or "unknown"
                segments[month_key].append(
                    {
                        "photo_id": item["photo_id"],
                        "original_photo_id": item.get("original_photo_id"),
                        "appearance_mode": item["appearance_mode"],
                        "timestamp": item.get("timestamp"),
                    }
                )
            persons.append(
                {
                    "person_uuid": person_id,
                    "labels": ["Person"],
                    "properties": {
                        "user_id": self.user_id,
                        "person_id": person_id,
                        "pipeline_family": PIPELINE_FAMILY_V0321_3,
                        "is_primary_candidate": person_id == primary_person_id,
                        "original_image_ids": self._unique(
                            item.get("original_photo_id") or item["photo_id"]
                            for item in person_appearance_items
                        ),
                        "photo_count": len(original_photo_ids),
                    },
                }
            )
            redis_payload[f"person_photo_index_{person_id}"] = {
                "key": f"{self.family_prefix}:person_photo_index:{person_id}",
                "person_id": person_id,
                "photo_count": len(original_photo_ids),
                "segments": dict(segments),
            }
        for ledger in relationship_ledgers:
            redis_payload[f"relationship_ledger_{ledger['relationship_revision_id']}"] = {
                "key": f"{self.family_prefix}:relationship_ledger:{ledger['relationship_revision_id']}",
                **ledger,
            }
        event_root_nodes = []
        event_revision_nodes = []
        edges: List[Dict[str, Any]] = []
        for event in event_revisions:
            root_id = event["event_root_id"]
            event_root_nodes.append(
                {
                    "event_root_id": root_id,
                    "labels": ["EventRoot"],
                    "properties": {
                        "user_id": self.user_id,
                        "pipeline_family": PIPELINE_FAMILY_V0321_3,
                        "current_revision_id": event["event_revision_id"],
                        "sealed_state": event.get("sealed_state") or "sealed",
                    },
                }
            )
            event_revision_nodes.append(
                {
                    "event_revision_id": event["event_revision_id"],
                    "labels": ["EventRevision"],
                    "properties": {
                        **{k: v for k, v in event.items() if k != "atomic_evidence"},
                        "user_id": self.user_id,
                    },
                }
            )
            edges.append(
                {
                    "edge_id": self._stable_id("edge", root_id, event["event_revision_id"], "HAS_REVISION"),
                    "from_id": root_id,
                    "to_id": event["event_revision_id"],
                    "edge_type": "HAS_REVISION",
                    "properties": {"pipeline_family": PIPELINE_FAMILY_V0321_3},
                }
            )
            edges.append(
                {
                    "edge_id": self._stable_id("edge", root_id, event["event_revision_id"], "CURRENT"),
                    "from_id": root_id,
                    "to_id": event["event_revision_id"],
                    "edge_type": "CURRENT",
                    "properties": {"pipeline_family": PIPELINE_FAMILY_V0321_3},
                }
            )
            if event.get("supersedes_event_revision_id"):
                edges.append(
                    {
                        "edge_id": self._stable_id("edge", event["event_revision_id"], event["supersedes_event_revision_id"], "SUPERSEDES"),
                        "from_id": event["event_revision_id"],
                        "to_id": event["supersedes_event_revision_id"],
                        "edge_type": "SUPERSEDES",
                        "properties": {"pipeline_family": PIPELINE_FAMILY_V0321_3},
                    }
                )
        relationship_root_nodes = []
        relationship_revision_nodes = []
        for relationship in relationship_revisions:
            root_id = relationship["relationship_root_id"]
            relationship_root_nodes.append(
                {
                    "relationship_root_id": root_id,
                    "labels": ["RelationshipRoot"],
                    "properties": {
                        "user_id": self.user_id,
                        "target_person_id": relationship["target_person_id"],
                        "current_revision_id": relationship["relationship_revision_id"],
                        "pipeline_family": PIPELINE_FAMILY_V0321_3,
                    },
                }
            )
            relationship_revision_nodes.append(
                {
                    "relationship_revision_id": relationship["relationship_revision_id"],
                    "labels": ["RelationshipRevision"],
                    "properties": {
                        **relationship,
                        "user_id": self.user_id,
                    },
                }
            )
            edges.append(
                {
                    "edge_id": self._stable_id("edge", root_id, relationship["target_person_id"], "TARGET_PERSON"),
                    "from_id": root_id,
                    "to_id": relationship["target_person_id"],
                    "edge_type": "TARGET_PERSON",
                    "properties": {"pipeline_family": PIPELINE_FAMILY_V0321_3},
                }
            )
            edges.append(
                {
                    "edge_id": self._stable_id("edge", root_id, relationship["relationship_revision_id"], "HAS_REVISION"),
                    "from_id": root_id,
                    "to_id": relationship["relationship_revision_id"],
                    "edge_type": "HAS_REVISION",
                    "properties": {"pipeline_family": PIPELINE_FAMILY_V0321_3},
                }
            )
            edges.append(
                {
                    "edge_id": self._stable_id("edge", root_id, relationship["relationship_revision_id"], "CURRENT"),
                    "from_id": root_id,
                    "to_id": relationship["relationship_revision_id"],
                    "edge_type": "CURRENT",
                    "properties": {"pipeline_family": PIPELINE_FAMILY_V0321_3},
                }
            )
            if relationship.get("supersedes_relationship_revision_id"):
                edges.append(
                    {
                        "edge_id": self._stable_id(
                            "edge",
                            relationship["relationship_revision_id"],
                            relationship["supersedes_relationship_revision_id"],
                            "SUPERSEDES",
                        ),
                        "from_id": relationship["relationship_revision_id"],
                        "to_id": relationship["supersedes_relationship_revision_id"],
                        "edge_type": "SUPERSEDES",
                        "properties": {"pipeline_family": PIPELINE_FAMILY_V0321_3},
                    }
                )
            for event_id in relationship.get("supporting_event_ids", []) or []:
                edges.append(
                    {
                        "edge_id": self._stable_id("edge", relationship["relationship_revision_id"], event_id, "SUPPORTED_BY_EVENT"),
                        "from_id": relationship["relationship_revision_id"],
                        "to_id": event_id,
                        "edge_type": "SUPPORTED_BY_EVENT",
                        "properties": {"pipeline_family": PIPELINE_FAMILY_V0321_3},
                    }
                )

        ordered_events = sorted(event_revisions, key=lambda item: (item.get("started_at") or "", item.get("event_root_id") or ""))
        for left, right in zip(ordered_events, ordered_events[1:]):
            if left.get("event_root_id") == right.get("event_root_id"):
                continue
            edges.append(
                {
                    "edge_id": self._stable_id("edge", left["event_root_id"], right["event_root_id"], "NEXT_EVENT"),
                    "from_id": left["event_root_id"],
                    "to_id": right["event_root_id"],
                    "edge_type": "NEXT_EVENT",
                    "properties": {"pipeline_family": PIPELINE_FAMILY_V0321_3},
                }
            )
        neo4j_nodes = {
            "user": [
                {
                    "user_id": self.user_id,
                    "labels": ["User"],
                    "properties": {
                        "pipeline_family": PIPELINE_FAMILY_V0321_3,
                        "primary_person_id": primary_person_id,
                    },
                }
            ],
            "persons": persons,
            "event_roots": event_root_nodes,
            "event_revisions": event_revision_nodes,
            "relationship_roots": relationship_root_nodes,
            "relationship_revisions": relationship_revision_nodes,
            "period_revisions": [],
        }
        return {
            "redis": redis_payload,
            "neo4j": {
                "nodes": neo4j_nodes,
                "edges": edges,
            },
            "milvus": {},
        }

    def _face_anchors(self, photo: Any) -> List[Dict[str, Any]]:
        anchors = []
        for face in list(getattr(photo, "faces", []) or []):
            anchors.append(
                {
                    "person_id": face.get("person_id"),
                    "normalized_bbox": dict(face.get("bbox_xywh") or {}),
                    "face_confidence": float(face.get("score") or 0.0),
                    "identity_confidence": float(face.get("similarity") or 0.0),
                    "primary_candidate": face.get("person_id") == getattr(photo, "primary_person_id", None),
                    "visibility_bucket": "clear" if float(face.get("quality_score") or 0.0) >= 0.75 else "medium",
                }
            )
        return anchors

    def _public_url(self, path: Path | str) -> Optional[str]:
        if self.public_url_builder is None:
            return None
        return self.public_url_builder(path)

    def _parse_dt(self, value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return None

    def _time_gap_seconds(self, left: Any, right: Any) -> Optional[float]:
        left_dt = self._parse_dt(left)
        right_dt = self._parse_dt(right)
        if not left_dt or not right_dt:
            return None
        return abs((left_dt - right_dt).total_seconds())

    def _same_day(self, left: Any, right: Any) -> bool:
        left_dt = self._parse_dt(left)
        right_dt = self._parse_dt(right)
        if not left_dt or not right_dt:
            return False
        return left_dt.date() == right_dt.date()

    def _window_start(self, live_events: Sequence[Dict[str, Any]], embedded_events: Sequence[Dict[str, Any]]) -> str:
        values = [item.get("started_at") for item in [*live_events, *embedded_events] if item.get("started_at")]
        return min(values) if values else ""

    def _window_end(self, live_events: Sequence[Dict[str, Any]], embedded_events: Sequence[Dict[str, Any]]) -> str:
        values = [item.get("ended_at") for item in [*live_events, *embedded_events] if item.get("ended_at")]
        return max(values) if values else ""

    def _stable_id(self, namespace: str, *parts: Any) -> str:
        raw = "|".join(str(part or "") for part in (PIPELINE_FAMILY_V0321_3, self.user_id, self.task_id, namespace, *parts))
        return uuid5(NAMESPACE_URL, raw).hex

    def _original_photo_id(self, photo: Any) -> str:
        return str(getattr(photo, "source_hash", None) or getattr(photo, "photo_id", "") or "")

    def _unique(self, values: Iterable[Any]) -> List[Any]:
        items: List[Any] = []
        seen = set()
        for value in values:
            fingerprint = self._signal_fingerprint(value)
            if not fingerprint:
                continue
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            if isinstance(value, (dict, list, tuple, set)):
                items.append(self._coerce_signal_value(value))
            else:
                items.append(value)
        return items

    def _emit_progress(
        self,
        callback: Optional[Callable[[str, Dict[str, Any]], None]],
        stage: str,
        payload: Dict[str, Any],
    ) -> None:
        if callback:
            callback(stage, payload)
