"""
Independent v0323 pipeline family.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from config import (
    RELATIONSHIP_MAX_RETRIES,
    RELATIONSHIP_MIN_CO_OCCURRENCE,
    RELATIONSHIP_MIN_DISTINCT_DAYS,
    RELATIONSHIP_MIN_INTIMACY_SCORE,
    RELATIONSHIP_REQUEST_TIMEOUT_SECONDS,
    RETRY_DELAY,
    V0323_LP1_MAX_ATTEMPTS,
    V0323_LP2_TIMEOUT_SCHEDULE_SECONDS,
    V0323_LP1_MAX_OUTPUT_TOKENS,
)
from utils import save_json


PIPELINE_FAMILY_V0323 = "v0323"
PIPELINE_VERSION_V0323 = "v0323"
LP1_BATCH_SIZE = 200
LP1_OVERLAP_SIZE = 8
LP1_PROMPT_VERSION = "v0323.lp1.v0139_two_step.v1"
LP1_ANALYSIS_TEXT_CHAR_LIMIT = 30000
LP2_PROMPT_VERSION = "v0323.lp2.person.v1"
LP3_STRUCTURED_PROMPT_VERSION = "v0323.lp3.structured.v1"
LP3_REPORT_PROMPT_VERSION = "v0323.lp3.report.v1"


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


def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=_json_default))
        handle.write("\n")


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _normalized_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _truncate_text(value: Any, *, limit: int) -> str:
    text = _normalized_text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def _truncate_preview(value: Any, *, limit: int = 4000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit]


def _unique_strings(values: Iterable[Any]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _sorted_unique_photo_ids(values: Iterable[Any], *, order_index: Dict[str, int]) -> List[str]:
    unique = _unique_strings(values)
    return sorted(unique, key=lambda item: (order_index.get(item, 10**9), item))

class V0323PipelineFamily:
    """LP snapshot family based on v0139-style LP1 aggregation."""

    def __init__(
        self,
        *,
        task_id: str,
        task_dir: Path,
        user_id: Optional[str],
        asset_store: Any,
        llm_processor: Any,
        public_url_builder: Callable[[Path], str],
    ) -> None:
        self.task_id = task_id
        self.task_dir = Path(task_dir)
        self.user_id = user_id or ""
        self.asset_store = asset_store
        self.llm_processor = llm_processor
        self.public_url_builder = public_url_builder
        self.family_dir = self.task_dir / "v0323"
        self.family_dir.mkdir(parents=True, exist_ok=True)
        self.observation_index: Dict[str, Dict[str, Any]] = {}
        self.photo_order_index: Dict[str, int] = {}

        self.vp1_path = self.family_dir / "vp1_observations.json"
        self.lp1_batch_requests_path = self.family_dir / "lp1_batch_requests.jsonl"
        self.lp1_batch_outputs_path = self.family_dir / "lp1_batch_outputs.jsonl"
        self.lp1_event_cards_path = self.family_dir / "lp1_event_cards.jsonl"
        self.lp1_events_path = self.family_dir / "lp1_events.jsonl"
        self.lp1_events_compact_path = self.family_dir / "lp1_events_compact.json"
        self.lp1_continuation_log_path = self.family_dir / "lp1_event_continuation_log.jsonl"
        self.lp1_parse_failures_path = self.family_dir / "lp1_parse_failures.json"
        self.lp2_relationships_jsonl_path = self.family_dir / "lp2_relationships.jsonl"
        self.lp2_relationships_path = self.family_dir / "lp2_relationships.json"
        self.lp3_profile_path = self.family_dir / "lp3_profile.json"
        self.llm_failures_path = self.family_dir / "llm_failures.jsonl"
        self.memory_snapshot_path = self.family_dir / "memory_snapshot.json"

    def run(
        self,
        *,
        photos: List[Any],
        face_output: Dict[str, Any],
        primary_person_id: Optional[str],
        vlm_results: List[Dict[str, Any]],
        cached_photo_ids: Sequence[str],
        dedupe_report: Dict[str, Any],
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        observations = self._build_vp1_observations(photos, vlm_results)
        return self.run_from_observations(
            observations=observations,
            face_output=face_output,
            primary_person_id=primary_person_id,
            cached_photo_ids=cached_photo_ids,
            dedupe_report=dedupe_report,
            progress_callback=progress_callback,
        )

    def run_from_observations(
        self,
        *,
        observations: Sequence[Dict[str, Any]],
        face_output: Dict[str, Any],
        primary_person_id: Optional[str],
        cached_photo_ids: Sequence[str],
        dedupe_report: Dict[str, Any],
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        observations = [dict(item) for item in list(observations or [])]
        self.observation_index = {item["photo_id"]: item for item in observations}
        self.photo_order_index = {
            item["photo_id"]: int(item["sequence_index"])
            for item in observations
        }
        save_json(observations, str(self.vp1_path))

        lp1_payload = self._run_lp1_batches(observations, progress_callback=progress_callback)
        lp2_payload = self._run_lp2_relationships(
            observations=observations,
            face_output=face_output,
            primary_person_id=primary_person_id,
            lp1_events=lp1_payload["lp1_events"],
            progress_callback=progress_callback,
        )
        lp2_relationships = list(lp2_payload.get("relationships", []) or [])
        save_json(lp2_relationships, str(self.lp2_relationships_path))

        lp3_profile = self._run_lp3_profile(
            primary_person_id=primary_person_id,
            lp1_events=lp1_payload["lp1_events"],
            lp2_relationships=lp2_relationships,
            progress_callback=progress_callback,
        )
        save_json(lp3_profile, str(self.lp3_profile_path))

        artifacts = self._artifact_urls(
            {
                "vp1_observations_url": self.vp1_path,
                "lp1_batch_requests_url": self.lp1_batch_requests_path,
                "lp1_batch_outputs_url": self.lp1_batch_outputs_path,
                "lp1_event_cards_url": self.lp1_event_cards_path,
                "lp1_events_url": self.lp1_events_path,
                "lp1_events_compact_url": self.lp1_events_compact_path,
                "lp1_event_continuation_log_url": self.lp1_continuation_log_path,
                "lp1_parse_failures_url": self.lp1_parse_failures_path,
                "lp2_relationships_jsonl_url": self.lp2_relationships_jsonl_path,
                "lp2_relationships_url": self.lp2_relationships_path,
                "lp3_profile_url": self.lp3_profile_path,
                "llm_failures_url": self.llm_failures_path,
                "memory_snapshot_url": self.memory_snapshot_path,
            }
        )

        memory = {
            "pipeline_family": PIPELINE_FAMILY_V0323,
            "summary": {
                "pipeline_family": PIPELINE_FAMILY_V0323,
                "lp1_batch_count": len(lp1_payload["lp1_batches"]),
                "event_count": len(lp1_payload["lp1_events"]),
                "relationship_count": len(lp2_relationships),
                "lp2_candidate_count": _safe_int(lp2_payload.get("candidate_count")),
                "lp2_failed_candidate_count": _safe_int(lp2_payload.get("failed_candidate_count")),
                "lp2_retry_count": _safe_int(lp2_payload.get("retry_count")),
                "profile_generation_mode": "structured_plus_markdown",
                "cached_photo_count": len(list(cached_photo_ids or [])),
                "dedupe_retained_images": _safe_int(dedupe_report.get("retained_images")),
            },
            "vp1_observations": observations,
            "lp1_batches": lp1_payload["lp1_batches"],
            "lp1_events": lp1_payload["lp1_events"],
            "lp1_event_continuation_log": lp1_payload["lp1_event_continuation_log"],
            "lp2_relationships": lp2_relationships,
            "lp3_profile": lp3_profile,
            "artifacts": artifacts,
            "transparency": {
                "llm_provider": getattr(self.llm_processor, "provider", ""),
                "llm_model": getattr(self.llm_processor, "model", ""),
                "relationship_model": getattr(self.llm_processor, "relationship_model", getattr(self.llm_processor, "model", "")),
                "relationship_timeout_seconds": max(self._lp2_timeout_schedule(max(1, RELATIONSHIP_MAX_RETRIES))),
                "relationship_timeout_schedule_seconds": self._lp2_timeout_schedule(max(1, RELATIONSHIP_MAX_RETRIES)),
                "relationship_max_retries": RELATIONSHIP_MAX_RETRIES,
                "lp2_candidate_count": _safe_int(lp2_payload.get("candidate_count")),
                "lp2_failed_candidate_count": _safe_int(lp2_payload.get("failed_candidate_count")),
                "lp2_retry_count": _safe_int(lp2_payload.get("retry_count")),
                "lp1_prompt_version": LP1_PROMPT_VERSION,
                "lp2_prompt_version": LP2_PROMPT_VERSION,
                "lp3_prompt_versions": {
                    "structured": LP3_STRUCTURED_PROMPT_VERSION,
                    "report": LP3_REPORT_PROMPT_VERSION,
                },
            },
        }
        save_json(memory, str(self.memory_snapshot_path))
        return memory

    def _artifact_urls(self, mapping: Dict[str, Path]) -> Dict[str, str]:
        artifacts: Dict[str, str] = {}
        for key, path in mapping.items():
            if path.exists():
                artifacts[key] = self.public_url_builder(path)
        return artifacts

    def _notify(
        self,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]],
        payload: Dict[str, Any],
    ) -> None:
        if progress_callback is None:
            return
        progress_callback("v0323", payload)

    def _build_vp1_observations(self, photos: Sequence[Any], vlm_results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        photo_index = {
            str(getattr(photo, "photo_id", "") or ""): photo
            for photo in photos
            if str(getattr(photo, "photo_id", "") or "")
        }
        combined: List[Dict[str, Any]] = []
        for item in vlm_results:
            photo_id = str(item.get("photo_id") or "").strip()
            if not photo_id:
                continue
            photo = photo_index.get(photo_id)
            timestamp = str(item.get("timestamp") or getattr(photo, "timestamp", "") or "").strip()
            location = dict(item.get("location") or getattr(photo, "location", {}) or {})
            analysis = dict(item.get("vlm_analysis") or {})
            source_type = str(item.get("source_type") or analysis.get("source_type") or "").strip()
            if not source_type:
                source_type = self._detect_source_type(photo=photo, analysis=analysis)
            media_kind = self._detect_media_kind(photo=photo, analysis=analysis, source_type=source_type)
            raw_face_person_ids = list(item.get("face_person_ids", []) or [])
            if not raw_face_person_ids and photo is not None:
                raw_face_person_ids = [
                    face.get("person_id")
                    for face in list(getattr(photo, "faces", []) or [])
                    if isinstance(face, dict) and face.get("person_id")
                ]
            combined.append(
                {
                    "photo_id": photo_id,
                    "filename": str(item.get("filename") or getattr(photo, "filename", "") or ""),
                    "timestamp": timestamp,
                    "location": location,
                    "face_person_ids": _unique_strings(raw_face_person_ids),
                    "sequence_hint": _safe_int(getattr(photo, "sequence_index", 0)),
                    "source_type": source_type,
                    "media_kind": media_kind,
                    "is_reference_like": media_kind in {"reference_media", "screenshot", "embedded_media"},
                    "vlm_analysis": analysis,
                }
            )

        combined.sort(
            key=lambda item: (
                str(item.get("timestamp") or ""),
                _safe_int(item.get("sequence_hint")),
                str(item.get("photo_id") or ""),
            )
        )

        observations: List[Dict[str, Any]] = []
        for index, item in enumerate(combined, start=1):
            analysis = dict(item.get("vlm_analysis") or {})
            face_person_ids = []
            for value in item.get("face_person_ids", []) or []:
                if isinstance(value, dict):
                    person_id = str(value.get("person_id") or "").strip()
                else:
                    person_id = str(value or "").strip()
                if person_id:
                    face_person_ids.append(person_id)
            observation = {
                "photo_id": item["photo_id"],
                "filename": item["filename"],
                "sequence_index": index,
                "timestamp": item["timestamp"],
                "location": item["location"],
                "face_person_ids": _unique_strings(face_person_ids),
                "source_type": item["source_type"],
                "media_kind": item["media_kind"],
                "is_reference_like": bool(item["is_reference_like"]),
                "vlm_analysis": analysis,
            }
            observations.append(observation)
        return observations

    def _detect_source_type(self, *, photo: Any, analysis: Dict[str, Any]) -> str:
        filename = str(getattr(photo, "filename", "") or "").lower()
        uncertainty = " ".join(str(item).lower() for item in list(analysis.get("uncertainty", []) or []))
        ocr_hits = " ".join(str(item).lower() for item in list(analysis.get("ocr_hits", []) or []))
        summary = " ".join(
            part
            for part in [
                str(analysis.get("summary") or ""),
                " ".join(str(item) for item in list(analysis.get("details", []) or [])),
                str(dict(analysis.get("scene") or {}).get("environment_description") or ""),
                str(dict(analysis.get("event") or {}).get("activity") or ""),
            ]
            if part
        ).lower()
        if any(token in filename for token in ("screenshot", "screen shot", "截屏", "截图")):
            return "screenshot"
        if any(token in ocr_hits for token in ("student id", "id card", "passport", "学生证", "身份证")):
            return "document"
        if any(
            token in " ".join([filename, summary, uncertainty])
            for token in (
                "midjourney",
                "dall-e",
                "dalle",
                "stable diffusion",
                "sdxl",
                "comfyui",
                "flux",
                "ai生成",
                "生成图",
                "ai图",
                "ai 图",
                "ai绘画",
                "generated illustration",
                "generated portrait",
                "ai generated image",
            )
        ):
            return "ai_generated_image"
        if any(
            token in summary
            for token in (
                "polaroid",
                "instant photo",
                "printed photo",
                "相纸",
                "照片中的照片",
                "屏幕中的照片",
            )
        ):
            return "embedded_media"
        if "reference-only" in uncertainty or any(
            token in summary for token in ("reference media", "poster", "wallpaper", "meme", "海报", "网图", "参考图")
        ):
            return "reference_media"
        return "camera_photo"

    def _detect_media_kind(self, *, photo: Any, analysis: Dict[str, Any], source_type: str = "") -> str:
        normalized_source_type = str(source_type or "").strip().lower()
        if normalized_source_type == "screenshot":
            return "screenshot"
        if normalized_source_type == "document":
            return "document"
        if normalized_source_type in {"reference_media", "ai_generated_image"}:
            return "reference_media"
        if normalized_source_type == "embedded_media":
            return "embedded_media"
        return "live_photo"

    def _run_lp1_batches(
        self,
        observations: Sequence[Dict[str, Any]],
        *,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        batches = self._build_batches(observations)
        for path in (
            self.lp1_batch_requests_path,
            self.lp1_batch_outputs_path,
            self.lp1_event_cards_path,
            self.lp1_events_path,
            self.lp1_events_compact_path,
            self.lp1_continuation_log_path,
            self.lp1_parse_failures_path,
        ):
            if path.exists():
                path.unlink()
        final_events_by_id: Dict[str, Dict[str, Any]] = {}
        ordered_event_ids: List[str] = []
        continuation_log: List[Dict[str, Any]] = []
        batch_cards: List[Dict[str, Any]] = []
        parse_failures: List[Dict[str, Any]] = []
        batch_summaries: List[Dict[str, Any]] = []

        for batch in batches:
            carryover_cards = self._select_carryover_event_cards(
                final_events_by_id=final_events_by_id,
                ordered_event_ids=ordered_event_ids,
                overlap_context_photo_ids=batch["overlap_context_photo_ids"],
            )
            prompt, request_record = self._build_lp1_batch_prompt(
                batch=batch,
                carryover_cards=carryover_cards,
            )
            max_attempts = max(1, V0323_LP1_MAX_ATTEMPTS)
            parsed_output: Optional[Dict[str, Any]] = None
            parse_status = "failed"
            last_error: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                attempt_record = dict(request_record)
                attempt_record["attempt"] = attempt
                attempt_record["max_attempts"] = max_attempts
                attempt_record["prompt_kind"] = "primary" if attempt == 1 else "retry"
                _append_jsonl(self.lp1_batch_requests_path, attempt_record)
                attempt_result = self._call_lp1_batch_attempt(
                    prompt=prompt,
                    batch=batch,
                    request_record=attempt_record,
                )
                _append_jsonl(self.lp1_batch_outputs_path, attempt_result["record"])
                if attempt_result["ok"]:
                    parsed_output = attempt_result["parsed_output"]
                    parse_status = "ok" if attempt == 1 else "retry_ok"
                    break
                last_error = attempt_result["error"]

            if parsed_output is None:
                failure = {
                    "batch_id": batch["batch_id"],
                    "batch_index": batch["batch_index"],
                    "error": f"LP1 batch failed after {max_attempts} attempts: {last_error}",
                }
                parse_failures.append(failure)
                save_json(parse_failures, str(self.lp1_parse_failures_path))
                raise RuntimeError(failure["error"])

            continuation_entries = self._apply_lp1_batch_events(
                batch=batch,
                batch_output=parsed_output,
                final_events_by_id=final_events_by_id,
                ordered_event_ids=ordered_event_ids,
            )
            continuation_log.extend(continuation_entries)
            next_cards = self._select_carryover_event_cards(
                final_events_by_id=final_events_by_id,
                ordered_event_ids=ordered_event_ids,
                overlap_context_photo_ids=batch["new_region_photo_ids"][-LP1_OVERLAP_SIZE:],
            )
            batch_cards.extend(
                {
                    "batch_id": batch["batch_id"],
                    "batch_index": batch["batch_index"],
                    **card,
                }
                for card in next_cards
            )
            batch_summary = {
                "batch_id": batch["batch_id"],
                "batch_index": batch["batch_index"],
                "input_photo_ids": list(batch["input_photo_ids"]),
                "overlap_context_photo_ids": list(batch["overlap_context_photo_ids"]),
                "new_region_photo_ids": list(batch["new_region_photo_ids"]),
                "carryover_event_refs": [card["event_id"] for card in carryover_cards],
                "raw_event_count": len(parsed_output["events"]),
                "parse_status": parse_status,
                "prompt_version": LP1_PROMPT_VERSION,
            }
            batch_summaries.append(batch_summary)
            self._notify(
                progress_callback,
                {
                    "message": "v0323 LP1 分批事件聚合中",
                    "pipeline_family": PIPELINE_FAMILY_V0323,
                    "substage": "lp1_batch",
                    "batch_id": batch["batch_id"],
                    "batch_index": batch["batch_index"],
                    "batch_count": len(batches),
                    "event_count": len(ordered_event_ids),
                    "continuation_count": len(continuation_log),
                    "percent": round((batch["batch_index"] / max(1, len(batches))) * 100),
                },
            )

        final_events = [final_events_by_id[event_id] for event_id in ordered_event_ids]

        _write_jsonl(self.lp1_event_cards_path, batch_cards)
        _write_jsonl(self.lp1_events_path, final_events)
        _write_jsonl(self.lp1_continuation_log_path, continuation_log)
        save_json(parse_failures, str(self.lp1_parse_failures_path))
        save_json(final_events, str(self.lp1_events_compact_path))

        return {
            "lp1_batches": batch_summaries,
            "lp1_events": final_events,
            "lp1_event_continuation_log": continuation_log,
        }

    def _build_batches(self, observations: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        batches: List[Dict[str, Any]] = []
        total = len(observations)
        if total == 0:
            return batches
        batch_index = 1
        new_start = 0
        while new_start < total:
            new_end = min(total, new_start + LP1_BATCH_SIZE)
            batch_start = 0 if new_start == 0 else max(0, new_start - LP1_OVERLAP_SIZE)
            input_records = list(observations[batch_start:new_end])
            overlap_records = list(observations[batch_start:new_start])
            new_records = list(observations[new_start:new_end])
            batches.append(
                {
                    "batch_id": f"BATCH_{batch_index:04d}",
                    "batch_index": batch_index,
                    "input_records": input_records,
                    "overlap_records": overlap_records,
                    "new_records": new_records,
                    "input_photo_ids": [item["photo_id"] for item in input_records],
                    "overlap_context_photo_ids": [item["photo_id"] for item in overlap_records],
                    "new_region_photo_ids": [item["photo_id"] for item in new_records],
                }
            )
            batch_index += 1
            new_start += LP1_BATCH_SIZE
        return batches

    def _build_lp1_batch_prompt(
        self,
        *,
        batch: Dict[str, Any],
        carryover_cards: Sequence[Dict[str, Any]],
    ) -> tuple[str, Dict[str, Any]]:
        batch_meta = {
            "batch_id": batch["batch_id"],
            "batch_index": batch["batch_index"],
            "prompt_version": LP1_PROMPT_VERSION,
            "input_photo_count": len(batch["input_records"]),
            "overlap_photo_count": len(batch["overlap_records"]),
            "new_region_photo_count": len(batch["new_records"]),
        }
        overlap_blocks = [self._format_photo_record(item) for item in batch["overlap_records"]]
        new_region_blocks = [self._format_photo_record(item) for item in batch["new_records"]]
        card_blocks = [self._format_event_card(card) for card in carryover_cards]

        sections = [
            "LP1 Batch Analysis Task",
            "You are a senior anthropologist and social-behavior analyst working on a personal photo-memory pipeline.",
            "Your job is to read the ordered photo-level VLM records and reconstruct coherent events using v0139-style time-space-behavior clustering.",
            "",
            "Batch rules:",
            "1. Photos are already stably ordered. Read them strictly in order.",
            "2. OVERLAP_CONTEXT_PHOTOS are context only. They help you decide whether an event continues across the batch boundary.",
            "3. Output analysis only for events that include at least one NEW_REGION photo.",
            "4. If an old event clearly continues into the new region, analyze it as one event that spans overlap + new photos, but do not separately restate overlap-only events.",
            "5. Always cite exact photo_id groups when describing each event. This is mandatory.",
            "6. Favor fewer, higher-confidence events over speculative over-splitting.",
            "7. Keep the analysis rich but concise. Avoid dumping giant repeated descriptions.",
            "",
            "Write a compact analytical memo in plain text.",
            "For each event, include:",
            "- involved photo_ids",
            "- time span / temporal logic",
            "- location or location nature",
            "- main participants",
            "- objective scene facts",
            "- narrative synthesis",
            "- social dynamics",
            "- persona evidence",
            "- tags",
            "- confidence and reason",
            "",
            "BATCH_META",
            json.dumps(batch_meta, ensure_ascii=False, indent=2),
            "",
            "OVERLAP_CONTEXT_PHOTOS",
            "\n\n".join(overlap_blocks) if overlap_blocks else "NONE",
            "",
            "NEW_REGION_PHOTOS",
            "\n\n".join(new_region_blocks),
            "",
            "CARRYOVER_EVENT_CARDS",
            "\n\n".join(card_blocks) if card_blocks else "NONE",
        ]
        prompt = "\n".join(sections).strip()
        request_record = {
            "batch_id": batch["batch_id"],
            "batch_index": batch["batch_index"],
            "prompt_version": LP1_PROMPT_VERSION,
            "prompt_char_count": len(prompt),
            "input_photo_ids": list(batch["input_photo_ids"]),
            "overlap_context_photo_ids": list(batch["overlap_context_photo_ids"]),
            "new_region_photo_ids": list(batch["new_region_photo_ids"]),
            "carryover_event_card_ids": [card["event_id"] for card in carryover_cards],
            "prompt_sections": {
                "batch_meta": batch_meta,
                "overlap_context_photos": [self._prompt_photo_payload(item) for item in batch["overlap_records"]],
                "new_region_photos": [self._prompt_photo_payload(item) for item in batch["new_records"]],
                "carryover_event_cards": [dict(card) for card in carryover_cards],
            },
        }
        return prompt, request_record

    def _build_lp1_convert_prompt(
        self,
        *,
        batch: Dict[str, Any],
        analysis_text: str,
    ) -> str:
        trimmed_analysis = str(analysis_text or "").strip()
        if len(trimmed_analysis) > LP1_ANALYSIS_TEXT_CHAR_LIMIT:
            trimmed_analysis = trimmed_analysis[:LP1_ANALYSIS_TEXT_CHAR_LIMIT]
        photo_index = [
            {
                "photo_id": item["photo_id"],
                "timestamp": item.get("timestamp"),
                "face_person_ids": list(item.get("face_person_ids", []) or []),
                "location_name": str(dict(item.get("location") or {}).get("name") or "").strip(),
                "summary": str(dict(item.get("vlm_analysis") or {}).get("summary") or "").strip(),
            }
            for item in batch["input_records"]
        ]
        payload_example = {
            "events": [
                {
                    "event_id": "TEMP_EVT_001",
                    "supporting_photo_ids": ["photo_001", "photo_002"],
                    "meta_info": {
                        "title": "事件标题",
                        "location_context": "地点性质",
                        "photo_count": 2,
                    },
                    "objective_fact": {
                        "scene_description": "客观场景事实",
                        "participants": ["Person_001", "Person_002"],
                    },
                    "narrative_synthesis": "一句话深度还原事件。",
                    "social_dynamics": [],
                    "persona_evidence": {
                        "behavioral": [],
                        "aesthetic": [],
                        "socioeconomic": [],
                    },
                    "tags": ["#标签"],
                    "confidence": 0.8,
                    "reason": "时间、地点、人物与行为证据",
                }
            ]
        }
        sections = [
            "Convert the following LP1 batch analysis into JSON.",
            "Return JSON only. No markdown. No explanations. No code fences.",
            "Top-level key must be events.",
            "Rules:",
            "1. Every event must include supporting_photo_ids.",
            "2. supporting_photo_ids must come only from the provided PHOTO_INDEX.",
            "3. Every event must touch at least one NEW_REGION photo.",
            "4. If an event spans overlap and new-region photos, include both in supporting_photo_ids.",
            "5. Do not output overlap-only events.",
            "6. Keep strings concise but preserve social_dynamics, persona_evidence, tags, confidence, and reason.",
            "",
            "NEW_REGION_PHOTO_IDS",
            json.dumps(list(batch["new_region_photo_ids"]), ensure_ascii=False, indent=2),
            "",
            "PHOTO_INDEX",
            json.dumps(photo_index, ensure_ascii=False, indent=2),
            "",
            "JSON_FORMAT",
            json.dumps(payload_example, ensure_ascii=False, indent=2),
            "",
            "ANALYSIS_TEXT",
            trimmed_analysis or "NONE",
        ]
        return "\n".join(sections).strip()

    def _format_photo_record(self, observation: Dict[str, Any]) -> str:
        payload = self._prompt_photo_payload(observation)
        people_lines = []
        for person in payload.get("people", []):
            if not isinstance(person, dict):
                continue
            bits = [
                f"person_id={person.get('person_id') or ''}",
                f"appearance={_normalized_text(person.get('appearance'))}",
                f"clothing={_normalized_text(person.get('clothing'))}",
                f"activity={_normalized_text(person.get('activity'))}",
                f"interaction={_normalized_text(person.get('interaction'))}",
                f"expression={_normalized_text(person.get('expression'))}",
            ]
            people_lines.append("  - " + "; ".join(bit for bit in bits if bit and not bit.endswith("=")))
        relations = []
        for relation in payload.get("relations", []):
            if not isinstance(relation, dict):
                continue
            relations.append(
                "  - "
                + " -> ".join(
                    [
                        _normalized_text(relation.get("subject")),
                        _normalized_text(relation.get("relation")),
                        _normalized_text(relation.get("object")),
                    ]
                ).strip(" ->")
            )
        lines = [
            f"【照片 {payload['photo_id']}】",
            f"时间: {payload['timestamp']}",
            f"地点: {payload['location_name'] or '未知'}",
            f"来源类型: {payload['source_type']}",
            f"媒体类型: {payload['media_kind']}",
            f"人物ID: {', '.join(payload['face_person_ids']) if payload['face_person_ids'] else '无'}",
            f"VLM描述: {payload['summary'] or ''}",
            f"场景细节: {payload['scene_details'] or ''}",
            f"活动: {payload['event_activity'] or ''}",
            f"社交背景: {payload['event_social'] or ''}",
            f"氛围: {payload['event_mood'] or ''}",
            f"故事线索: {payload['story_hints'] or ''}",
            f"细节: {payload['details'] or ''}",
        ]
        if people_lines:
            lines.append("人物详情:")
            lines.extend(people_lines)
        if relations:
            lines.append("实体关系:")
            lines.extend(relations)
        return "\n".join(lines).strip()

    def _prompt_photo_payload(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        analysis = dict(observation.get("vlm_analysis") or {})
        scene = analysis.get("scene")
        event = analysis.get("event")
        details = analysis.get("details")
        people = list(analysis.get("people", []) or [])
        relations = list(analysis.get("relations", []) or [])
        if isinstance(scene, dict):
            scene_details = ", ".join(
                _unique_strings(
                    [
                        scene.get("location_detected"),
                        scene.get("location_type"),
                        *list(scene.get("environment_details", []) or []),
                        scene.get("environment_description"),
                    ]
                )
            )
        else:
            scene_details = _normalized_text(scene)
        if isinstance(event, dict):
            story_hints = ", ".join(_unique_strings(event.get("story_hints", []) or []))
            event_activity = _normalized_text(event.get("activity"))
            event_social = _normalized_text(event.get("social_context"))
            event_mood = _normalized_text(event.get("mood"))
        else:
            story_hints = ""
            event_activity = _normalized_text(event)
            event_social = ""
            event_mood = ""
        if isinstance(details, list):
            detail_text = ", ".join(_unique_strings(details))
        elif isinstance(details, dict):
            detail_text = _normalized_text(json.dumps(details, ensure_ascii=False))
        else:
            detail_text = _normalized_text(details)
        location_name = str(dict(observation.get("location") or {}).get("name") or "").strip()
        scene_location = ""
        if isinstance(scene, dict):
            scene_location = str(scene.get("location_detected") or "").strip()
        return {
            "photo_id": observation["photo_id"],
            "sequence_index": observation["sequence_index"],
            "timestamp": observation["timestamp"],
            "source_type": observation.get("source_type"),
            "media_kind": observation.get("media_kind"),
            "location_name": scene_location or location_name,
            "face_person_ids": list(observation.get("face_person_ids", []) or []),
            "summary": _normalized_text(analysis.get("summary")),
            "scene_details": scene_details,
            "event_activity": event_activity,
            "event_social": event_social,
            "event_mood": event_mood,
            "story_hints": story_hints,
            "details": detail_text,
            "people": people,
            "relations": relations,
        }

    def _format_event_card(self, card: Dict[str, Any]) -> str:
        return f"EVENT_CARD {card['event_id']}\n{json.dumps(card, ensure_ascii=False, indent=2)}"

    def _select_carryover_event_cards(
        self,
        *,
        final_events_by_id: Dict[str, Dict[str, Any]],
        ordered_event_ids: Sequence[str],
        overlap_context_photo_ids: Sequence[str],
    ) -> List[Dict[str, Any]]:
        overlap_set = {str(photo_id) for photo_id in overlap_context_photo_ids if str(photo_id).strip()}
        selected: List[str] = []
        for event_id in ordered_event_ids:
            event = final_events_by_id.get(str(event_id))
            if not event:
                continue
            support = set(event.get("supporting_photo_ids", []) or [])
            if overlap_set and support.intersection(overlap_set):
                selected.append(str(event_id))
        for event_id in list(ordered_event_ids)[-3:]:
            if event_id not in selected:
                selected.append(str(event_id))
        return [self._build_event_card(final_events_by_id[event_id]) for event_id in selected if event_id in final_events_by_id]

    def _build_event_card(self, event: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "event_id": event["event_id"],
            "anchor_photo_id": event.get("anchor_photo_id"),
            "supporting_photo_ids": list(event.get("supporting_photo_ids", []) or []),
            "started_at": event.get("started_at"),
            "ended_at": event.get("ended_at"),
            "title": event.get("title"),
            "participant_person_ids": list(event.get("participant_person_ids", []) or []),
            "place_refs": list(event.get("place_refs", []) or []),
            "short_narrative": _truncate_text(event.get("narrative_synthesis"), limit=240),
        }

    def _lp1_convert_response_format(self) -> Dict[str, Any]:
        return {"type": "json_object"}

    def _call_lp1_batch_attempt(
        self,
        *,
        prompt: str,
        batch: Dict[str, Any],
        request_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        output_record = {
            "batch_id": batch["batch_id"],
            "batch_index": batch["batch_index"],
            "prompt_kind": request_record.get("prompt_kind"),
            "attempt": _safe_int(request_record.get("attempt"), default=1),
            "max_attempts": _safe_int(request_record.get("max_attempts"), default=1),
            "prompt_char_count": _safe_int(request_record.get("prompt_char_count")),
            "parse_status": "failed",
            "strategy": "v0139_two_step",
        }

        analysis_text = self._call_json_prompt_raw_text(
            prompt,
            max_tokens=V0323_LP1_MAX_OUTPUT_TOKENS,
        )
        output_record.update(
            {
                "analysis_response_char_count": len(analysis_text),
                "analysis_response_preview": _truncate_preview(analysis_text, limit=4000),
                "analysis_response_tail": str(analysis_text or "")[-4000:],
            }
        )

        analysis_error: Optional[Exception] = None
        try:
            extracted = self._extract_json_from_text(analysis_text, target_key="events")
            if extracted:
                normalized = self._normalize_lp1_batch_output(payload=extracted, batch=batch)
                output_record.update(
                    {
                        "parse_status": "analysis_ok",
                        "event_count": len(normalized["events"]),
                        "output": normalized,
                    }
                )
                return {"ok": True, "parsed_output": normalized, "record": output_record}
        except Exception as exc:
            analysis_error = exc
            output_record.update(
                {
                    "analysis_error_type": type(exc).__name__,
                    "analysis_error": str(exc),
                }
            )

        convert_prompt = self._build_lp1_convert_prompt(batch=batch, analysis_text=analysis_text)
        output_record["convert_prompt_char_count"] = len(convert_prompt)
        convert_text = self._call_json_prompt_raw_text(
            convert_prompt,
            max_tokens=V0323_LP1_MAX_OUTPUT_TOKENS,
            response_format=self._lp1_convert_response_format(),
        )
        output_record.update(
            {
                "convert_response_char_count": len(convert_text),
                "convert_response_preview": _truncate_preview(convert_text, limit=4000),
                "convert_response_tail": str(convert_text or "")[-4000:],
            }
        )

        try:
            payload = self._extract_json_from_text(convert_text, target_key="events")
            if not payload:
                raise ValueError("convert step did not produce events JSON")
            normalized = self._normalize_lp1_batch_output(payload=payload, batch=batch)
        except Exception as exc:
            output_record.update(
                {
                    "parse_status": "convert_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if analysis_error is not None:
                output_record["upstream_analysis_error"] = str(analysis_error)
            return {"ok": False, "error": exc, "record": output_record}

        output_record.update(
            {
                "parse_status": "convert_ok",
                "event_count": len(normalized["events"]),
                "output": normalized,
            }
        )
        return {"ok": True, "parsed_output": normalized, "record": output_record}

    def _call_json_prompt(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: Optional[tuple[int | float, int | float]] = None,
    ) -> Dict[str, Any]:
        handler = getattr(self.llm_processor, "_call_json_prompt", None)
        if handler is None:
            raise RuntimeError("LLM processor does not expose _call_json_prompt")
        payload = handler(prompt, max_tokens=max_tokens, response_format=response_format, timeout=timeout)
        if isinstance(payload, dict):
            return payload
        if payload is None:
            raise RuntimeError("LLM returned empty JSON payload")
        raise RuntimeError(f"Unexpected JSON payload type: {type(payload).__name__}")

    def _call_json_prompt_raw_text(
        self,
        prompt: str,
        *,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        timeout: Optional[tuple[int | float, int | float]] = None,
    ) -> str:
        raw_handler = getattr(self.llm_processor, "_call_json_prompt_raw_text", None)
        if callable(raw_handler):
            return str(
                raw_handler(
                    prompt,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    timeout=timeout,
                )
                or ""
            )
        payload = self._call_json_prompt(
            prompt,
            max_tokens=max_tokens,
            response_format=response_format,
            timeout=timeout,
        )
        return json.dumps(payload, ensure_ascii=False, default=_json_default)

    def _extract_json_payload(self, raw_text: str) -> Dict[str, Any]:
        extractor = getattr(self.llm_processor, "_extract_json_payload", None)
        if callable(extractor):
            payload = extractor(raw_text)
        else:
            payload = json.loads(str(raw_text or ""))
        if isinstance(payload, dict):
            return payload
        return {"items": payload}

    def _parse_json_candidate(self, candidate: str) -> Optional[Dict[str, Any]]:
        extractor = getattr(self.llm_processor, "_extract_json_payload", None)
        try:
            if callable(extractor):
                payload = extractor(candidate)
            else:
                payload = json.loads(candidate)
        except Exception:
            return None
        return payload if isinstance(payload, dict) else None

    def _extract_json_from_text(self, text: str, *, target_key: str = "events") -> Optional[Dict[str, Any]]:
        candidate = str(text or "").strip()
        if not candidate:
            return None

        direct = self._parse_json_candidate(candidate)
        if isinstance(direct, dict) and target_key in direct:
            return direct

        json_blocks = re.findall(r"```json\s*\n(.*?)\n\s*```", candidate, re.DOTALL)
        for block in json_blocks:
            parsed = self._parse_json_candidate(block.strip())
            if isinstance(parsed, dict) and target_key in parsed:
                return parsed

        search_pattern = f'"{target_key}"'
        idx = candidate.find(search_pattern)
        while idx != -1:
            brace_start = candidate.rfind("{", 0, idx)
            if brace_start == -1:
                idx = candidate.find(search_pattern, idx + 1)
                continue
            depth = 0
            in_string = False
            escape_next = False
            for cursor in range(brace_start, len(candidate)):
                char = candidate[cursor]
                if escape_next:
                    escape_next = False
                    continue
                if char == "\\" and in_string:
                    escape_next = True
                    continue
                if char == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        parsed = self._parse_json_candidate(candidate[brace_start : cursor + 1])
                        if isinstance(parsed, dict) and target_key in parsed:
                            return parsed
                        break
            idx = candidate.find(search_pattern, idx + 1)
        return None

    def _call_markdown_prompt(self, prompt: str) -> str:
        handler = getattr(self.llm_processor, "_call_markdown_prompt", None)
        if handler is None:
            raise RuntimeError("LLM processor does not expose _call_markdown_prompt")
        result = handler(prompt)
        return str(result or "").strip()

    def _normalize_lp1_batch_output(
        self,
        *,
        payload: Dict[str, Any],
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        events = payload.get("events")
        if not isinstance(events, list):
            raise ValueError("events must be a list")
        normalized = [
            self._normalize_lp1_event(item=item, batch=batch, index=index)
            for index, item in enumerate(events, start=1)
        ]
        return {
            "batch_id": batch["batch_id"],
            "events": normalized,
        }

    def _normalize_lp1_event(self, *, item: Dict[str, Any], batch: Dict[str, Any], index: int) -> Dict[str, Any]:
        if not isinstance(item, dict):
            raise ValueError("event item must be an object")
        meta_info = dict(item.get("meta_info") or {})
        objective_fact = dict(item.get("objective_fact") or {})
        supporting_photo_ids = _sorted_unique_photo_ids(
            [
                *list(item.get("supporting_photo_ids", []) or []),
                *list(item.get("photo_ids", []) or []),
                *list(meta_info.get("supporting_photo_ids", []) or []),
                *list(objective_fact.get("supporting_photo_ids", []) or []),
            ],
            order_index=self.photo_order_index,
        )
        invalid_photo_ids = [photo_id for photo_id in supporting_photo_ids if photo_id not in set(batch["input_photo_ids"])]
        if invalid_photo_ids:
            raise ValueError(f"supporting_photo_ids outside batch input: {invalid_photo_ids}")
        if not supporting_photo_ids:
            raise ValueError("event must include supporting_photo_ids")
        new_region_support = [photo_id for photo_id in supporting_photo_ids if photo_id in set(batch["new_region_photo_ids"])]
        if not new_region_support:
            raise ValueError("event must touch at least one new-region photo")
        anchor_photo_id = str(item.get("anchor_photo_id") or "").strip()
        if anchor_photo_id not in set(new_region_support):
            anchor_photo_id = new_region_support[0]
        started_at, ended_at = self._derive_event_bounds(
            supporting_photo_ids=supporting_photo_ids,
            started_at="",
            ended_at="",
        )
        persona_evidence = self._normalize_persona_evidence(item.get("persona_evidence"))
        social_dynamics = self._normalize_social_dynamics(item.get("social_dynamics"))
        place_refs = _unique_strings([*list(item.get("place_refs", []) or []), meta_info.get("location_context")])
        if not place_refs:
            place_refs = self._derive_place_refs(supporting_photo_ids)
        participant_person_ids = _unique_strings(
            [
                *list(item.get("participant_person_ids", []) or []),
                *list(item.get("participants", []) or []),
                *list(objective_fact.get("participants", []) or []),
            ]
        )
        depicted_person_ids = _unique_strings(item.get("depicted_person_ids", []))
        if not depicted_person_ids:
            depicted_person_ids = self._derive_depicted_people(supporting_photo_ids)
        if not participant_person_ids:
            participant_person_ids = list(depicted_person_ids)
        title = _normalized_text(meta_info.get("title") or item.get("title"))
        if not title:
            title = f"Batch Event {index:03d}"
        scene_description = _normalized_text(objective_fact.get("scene_description") or item.get("scene_description"))
        narrative_synthesis = _normalized_text(
            item.get("narrative_synthesis")
            or item.get("summary")
            or scene_description
        )
        return {
            "temp_event_id": str(item.get("event_id") or item.get("temp_event_id") or "").strip() or f"TEMP_EVT_{index:03d}",
            "anchor_photo_id": anchor_photo_id,
            "supporting_photo_ids": supporting_photo_ids,
            "started_at": started_at,
            "ended_at": ended_at,
            "title": title,
            "narrative_synthesis": narrative_synthesis,
            "participant_person_ids": participant_person_ids,
            "depicted_person_ids": depicted_person_ids,
            "place_refs": place_refs,
            "social_dynamics": social_dynamics,
            "persona_evidence": persona_evidence,
            "tags": _unique_strings(item.get("tags", [])),
            "confidence": max(0.0, min(1.0, _safe_float(item.get("confidence"), default=0.0))),
            "reason": _normalized_text(item.get("reason")),
            "meta_info": {
                "title": title,
                "location_context": meta_info.get("location_context") or (place_refs[0] if place_refs else ""),
                "photo_count": len(supporting_photo_ids),
            },
            "objective_fact": {
                "scene_description": scene_description,
                "participants": list(participant_person_ids),
            },
        }

    def _derive_event_bounds(
        self,
        *,
        supporting_photo_ids: Sequence[str],
        started_at: Any,
        ended_at: Any,
    ) -> tuple[str, str]:
        timestamps = [
            str(self.observation_index.get(photo_id, {}).get("timestamp") or "").strip()
            for photo_id in supporting_photo_ids
        ]
        timestamps = [item for item in timestamps if item]
        derived_started = min(timestamps) if timestamps else ""
        derived_ended = max(timestamps) if timestamps else derived_started
        started_text = str(started_at or "").strip() or derived_started
        ended_text = str(ended_at or "").strip() or derived_ended
        return started_text, ended_text or started_text

    def _derive_place_refs(self, supporting_photo_ids: Sequence[str]) -> List[str]:
        values: List[str] = []
        for photo_id in supporting_photo_ids:
            observation = self.observation_index.get(photo_id, {})
            location = dict(observation.get("location") or {})
            analysis = dict(observation.get("vlm_analysis") or {})
            scene = dict(analysis.get("scene") or {})
            values.extend(
                [
                    str(location.get("name") or "").strip(),
                    str(scene.get("location_detected") or "").strip(),
                ]
            )
        return _unique_strings(values)

    def _derive_depicted_people(self, supporting_photo_ids: Sequence[str]) -> List[str]:
        values: List[str] = []
        for photo_id in supporting_photo_ids:
            observation = self.observation_index.get(photo_id, {})
            values.extend(list(observation.get("face_person_ids", []) or []))
            analysis = dict(observation.get("vlm_analysis") or {})
            for person in list(analysis.get("people", []) or []):
                if isinstance(person, dict):
                    values.append(person.get("person_id"))
        return _unique_strings(values)

    def _normalize_persona_evidence(self, payload: Any) -> Dict[str, List[str]]:
        if not isinstance(payload, dict):
            payload = {}
        return {
            "behavioral": _unique_strings(payload.get("behavioral", [])),
            "aesthetic": _unique_strings(payload.get("aesthetic", [])),
            "socioeconomic": _unique_strings(payload.get("socioeconomic", [])),
        }

    def _normalize_social_dynamics(self, payload: Any) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for item in list(payload or []):
            if not isinstance(item, dict):
                continue
            records.append(
                {
                    "target_id": str(item.get("target_id") or "").strip(),
                    "interaction_type": _normalized_text(item.get("interaction_type")),
                    "social_clue": _normalized_text(item.get("social_clue")),
                    "relation_hypothesis": _normalized_text(item.get("relation_hypothesis")),
                    "confidence": round(max(0.0, min(1.0, _safe_float(item.get("confidence")))), 4),
                }
            )
        return records

    def _apply_lp1_batch_events(
        self,
        *,
        batch: Dict[str, Any],
        batch_output: Dict[str, Any],
        final_events_by_id: Dict[str, Dict[str, Any]],
        ordered_event_ids: List[str],
    ) -> List[Dict[str, Any]]:
        merge_entries: List[Dict[str, Any]] = []
        for item in batch_output.get("events", []):
            target_event_id = self._find_lp1_merge_target(
                batch=batch,
                incoming_event=item,
                final_events_by_id=final_events_by_id,
                ordered_event_ids=ordered_event_ids,
            )
            if not target_event_id:
                event_id = f"EVT_{len(ordered_event_ids) + 1:04d}"
                event_payload = {
                    "event_id": event_id,
                    "batch_id": batch["batch_id"],
                    "anchor_photo_id": item["anchor_photo_id"],
                    "supporting_photo_ids": list(item["supporting_photo_ids"]),
                    "started_at": item["started_at"],
                    "ended_at": item["ended_at"],
                    "title": item["title"],
                    "narrative_synthesis": item["narrative_synthesis"],
                    "participant_person_ids": list(item["participant_person_ids"]),
                    "depicted_person_ids": list(item["depicted_person_ids"]),
                    "place_refs": list(item["place_refs"]),
                    "social_dynamics": list(item["social_dynamics"]),
                    "persona_evidence": dict(item["persona_evidence"]),
                    "tags": list(item["tags"]),
                    "confidence": item["confidence"],
                    "reason": item["reason"],
                    "meta_info": dict(item.get("meta_info") or {}),
                    "objective_fact": dict(item.get("objective_fact") or {}),
                    "source_temp_event_id": item["temp_event_id"],
                }
                final_events_by_id[event_id] = event_payload
                ordered_event_ids.append(event_id)
                continue

            event = final_events_by_id[target_event_id]
            before = {
                "title": event.get("title"),
                "ended_at": event.get("ended_at"),
                "supporting_photo_ids": list(event.get("supporting_photo_ids", [])),
                "tags": list(event.get("tags", [])),
            }
            event["supporting_photo_ids"] = _sorted_unique_photo_ids(
                [*event.get("supporting_photo_ids", []), *item.get("supporting_photo_ids", [])],
                order_index=self.photo_order_index,
            )
            event["started_at"], event["ended_at"] = self._derive_event_bounds(
                supporting_photo_ids=event.get("supporting_photo_ids", []),
                started_at=event.get("started_at"),
                ended_at=item.get("ended_at"),
            )
            if item.get("title") and (
                len(item.get("supporting_photo_ids", [])) >= len(before["supporting_photo_ids"])
                or not _normalized_text(event.get("title"))
                or str(item.get("ended_at") or "") > str(before.get("ended_at") or "")
            ):
                event["title"] = item["title"]
            event["narrative_synthesis"] = self._merge_text(
                base=event.get("narrative_synthesis"),
                patch=item.get("narrative_synthesis"),
            )
            event["reason"] = self._merge_text(base=event.get("reason"), patch=item.get("reason"))
            event["social_dynamics"] = self._merge_social_dynamics(
                base=event.get("social_dynamics", []),
                extra=item.get("social_dynamics", []),
            )
            event["persona_evidence"] = self._merge_persona_evidence(
                base=event.get("persona_evidence", {}),
                extra=item.get("persona_evidence", {}),
            )
            event["tags"] = _unique_strings([*event.get("tags", []), *item.get("tags", [])])
            event["participant_person_ids"] = _unique_strings(
                [*event.get("participant_person_ids", []), *item.get("participant_person_ids", [])]
            )
            event["depicted_person_ids"] = self._derive_depicted_people(event.get("supporting_photo_ids", []))
            event["place_refs"] = _unique_strings([*event.get("place_refs", []), *item.get("place_refs", [])])
            if not event["place_refs"]:
                event["place_refs"] = self._derive_place_refs(event.get("supporting_photo_ids", []))
            event["confidence"] = max(_safe_float(event.get("confidence")), _safe_float(item.get("confidence")))
            event["meta_info"] = {
                "title": event.get("title"),
                "location_context": (event.get("place_refs") or [""])[0],
                "photo_count": len(event.get("supporting_photo_ids", [])),
            }
            merged_scene = self._merge_text(
                base=dict(event.get("objective_fact") or {}).get("scene_description"),
                patch=dict(item.get("objective_fact") or {}).get("scene_description"),
            )
            event["objective_fact"] = {
                "scene_description": merged_scene,
                "participants": list(event.get("participant_person_ids", [])),
            }
            merge_entries.append(
                {
                    "batch_id": batch["batch_id"],
                    "target_event_id": target_event_id,
                    "merge_mode": "overlap_support_merge",
                    "source_temp_event_id": item["temp_event_id"],
                    "before": before,
                    "after": {
                        "title": event.get("title"),
                        "ended_at": event.get("ended_at"),
                        "supporting_photo_ids": list(event.get("supporting_photo_ids", [])),
                        "tags": list(event.get("tags", [])),
                    },
                }
            )
        return merge_entries

    def _find_lp1_merge_target(
        self,
        *,
        batch: Dict[str, Any],
        incoming_event: Dict[str, Any],
        final_events_by_id: Dict[str, Dict[str, Any]],
        ordered_event_ids: Sequence[str],
    ) -> Optional[str]:
        overlap_set = set(batch["overlap_context_photo_ids"])
        incoming_support = set(incoming_event.get("supporting_photo_ids", []) or [])
        if not overlap_set or not incoming_support.intersection(overlap_set):
            return None
        best_event_id: Optional[str] = None
        best_score = 0
        for event_id in reversed(list(ordered_event_ids)):
            event = final_events_by_id.get(event_id)
            if not event:
                continue
            support = set(event.get("supporting_photo_ids", []) or [])
            score = len(support.intersection(incoming_support))
            if score > best_score:
                best_event_id = event_id
                best_score = score
        return best_event_id

    def _merge_text(self, *, base: Any, patch: Any) -> str:
        base_text = _normalized_text(base)
        patch_text = _normalized_text(patch)
        if not patch_text:
            return base_text
        if not base_text:
            return patch_text
        if patch_text in base_text:
            return base_text
        if base_text in patch_text and len(patch_text) >= len(base_text):
            return patch_text
        return f"{base_text} {patch_text}".strip()

    def _merge_social_dynamics(self, *, base: Sequence[Dict[str, Any]], extra: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        keyed: Dict[tuple[str, str], Dict[str, Any]] = {}
        for item in [*list(base or []), *list(extra or [])]:
            if not isinstance(item, dict):
                continue
            key = (str(item.get("target_id") or ""), str(item.get("interaction_type") or ""))
            keyed[key] = dict(item)
        return list(keyed.values())

    def _merge_persona_evidence(self, *, base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, List[str]]:
        merged = self._normalize_persona_evidence(base)
        extra_normalized = self._normalize_persona_evidence(extra)
        for key in ("behavioral", "aesthetic", "socioeconomic"):
            merged[key] = _unique_strings([*merged.get(key, []), *extra_normalized.get(key, [])])
        return merged

    def _run_lp2_relationships(
        self,
        *,
        observations: Sequence[Dict[str, Any]],
        face_output: Dict[str, Any],
        primary_person_id: Optional[str],
        lp1_events: Sequence[Dict[str, Any]],
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        lp2_max_retries = max(1, RELATIONSHIP_MAX_RETRIES)
        lp2_timeout_schedule = self._lp2_timeout_schedule(lp2_max_retries)
        if self.lp2_relationships_jsonl_path.exists():
            self.lp2_relationships_jsonl_path.unlink()
        if self.llm_failures_path.exists():
            self.llm_failures_path.unlink()
        face_person_ids = _unique_strings(
            item.get("person_id")
            for item in list(face_output.get("persons", []) or [])
            if isinstance(item, dict)
        )
        event_person_ids = _unique_strings(
            person_id
            for event in lp1_events
            for person_id in [
                *list(event.get("participant_person_ids", []) or []),
                *list(event.get("depicted_person_ids", []) or []),
            ]
        )
        candidate_person_ids = [
            person_id
            for person_id in _unique_strings([*face_person_ids, *event_person_ids])
            if person_id and person_id != str(primary_person_id or "").strip()
        ]
        relationships: List[Dict[str, Any]] = []
        failed_candidate_count = 0
        total_retry_count = 0
        events_by_person = self._index_lp2_events_by_person(lp1_events)
        for index, person_id in enumerate(candidate_person_ids, start=1):
            evidence = self._build_relationship_evidence(
                person_id=person_id,
                observations=observations,
                primary_person_id=primary_person_id,
                related_events=events_by_person.get(person_id, []),
            )
            if not evidence["shared_event_ids"]:
                continue
            if not self._relationship_candidate_requires_llm(evidence):
                continue
            prompt = self._build_relationship_prompt(
                person_id=person_id,
                primary_person_id=primary_person_id,
                evidence=evidence,
            )
            call_started_at = datetime.now().isoformat()
            relationship_provider = str(
                getattr(
                    self.llm_processor,
                    "_active_relationship_provider",
                    lambda: getattr(self.llm_processor, "relationship_provider", getattr(self.llm_processor, "provider", "")),
                )()
            ).strip()
            relationship_model = str(
                getattr(
                    self.llm_processor,
                    "_active_relationship_model",
                    lambda: getattr(self.llm_processor, "relationship_model", getattr(self.llm_processor, "model", "")),
                )()
            ).strip()
            self._notify(
                progress_callback,
                {
                    "message": "v0323 LP2 正在调用关系推断",
                    "pipeline_family": PIPELINE_FAMILY_V0323,
                    "substage": "lp2_relationship",
                    "processed_candidates": len(relationships),
                    "candidate_count": len(candidate_person_ids),
                    "relationship_count": len(relationships),
                    "current_candidate_index": index,
                    "current_person_id": person_id,
                    "call_started_at": call_started_at,
                    "call_timeout_seconds": lp2_timeout_schedule[0],
                    "provider": relationship_provider,
                    "model": relationship_model,
                    "current_attempt": 1,
                    "max_attempts": lp2_max_retries,
                    "retry_count": total_retry_count,
                    "failed_candidate_count": failed_candidate_count,
                    "percent": round(((max(0, index - 1)) / max(1, len(candidate_person_ids))) * 100),
                },
            )
            payload, candidate_retry_count, failure_record, final_timeout_seconds = self._call_lp2_candidate_with_retries(
                prompt=prompt,
                person_id=person_id,
                candidate_index=index,
                candidate_count=len(candidate_person_ids),
                relationship_provider=relationship_provider,
                relationship_model=relationship_model,
                evidence=evidence,
                timeout_schedule=lp2_timeout_schedule,
                progress_callback=progress_callback,
            )
            total_retry_count += candidate_retry_count
            if payload is None:
                failed_candidate_count += 1
                self._notify(
                    progress_callback,
                    {
                        "message": "v0323 LP2 候选人关系推断失败，已跳过并继续",
                        "pipeline_family": PIPELINE_FAMILY_V0323,
                        "substage": "lp2_relationship",
                        "processed_candidates": len(relationships),
                        "candidate_count": len(candidate_person_ids),
                        "relationship_count": len(relationships),
                        "current_candidate_index": index,
                        "current_person_id": person_id,
                        "call_started_at": call_started_at,
                        "call_finished_at": str((failure_record or {}).get("call_finished_at") or datetime.now().isoformat()),
                        "call_timeout_seconds": _safe_int((failure_record or {}).get("call_timeout_seconds")),
                        "provider": relationship_provider,
                        "model": relationship_model,
                        "retry_count": total_retry_count,
                        "failed_candidate_count": failed_candidate_count,
                        "error": str((failure_record or {}).get("error") or "LP2 candidate failed"),
                        "percent": round((index / max(1, len(candidate_person_ids))) * 100),
                    },
                )
                continue
            relationship = self._normalize_relationship_payload(
                person_id=person_id,
                payload=payload,
                evidence=evidence,
            )
            relationships.append(relationship)
            _write_jsonl(self.lp2_relationships_jsonl_path, relationships)
            self._notify(
                progress_callback,
                {
                    "message": "v0323 LP2 逐人关系推断中",
                    "pipeline_family": PIPELINE_FAMILY_V0323,
                    "substage": "lp2_relationship",
                    "processed_candidates": index,
                    "candidate_count": len(candidate_person_ids),
                    "relationship_count": len(relationships),
                    "person_id": person_id,
                    "current_candidate_index": index,
                    "current_person_id": person_id,
                    "last_completed_person_id": person_id,
                    "call_started_at": call_started_at,
                    "call_finished_at": datetime.now().isoformat(),
                    "call_timeout_seconds": final_timeout_seconds,
                    "provider": relationship_provider,
                    "model": relationship_model,
                    "candidate_retry_count": candidate_retry_count,
                    "retry_count": total_retry_count,
                    "failed_candidate_count": failed_candidate_count,
                    "percent": round((index / max(1, len(candidate_person_ids))) * 100),
                },
            )
        return {
            "relationships": relationships,
            "candidate_count": len(candidate_person_ids),
            "failed_candidate_count": failed_candidate_count,
            "retry_count": total_retry_count,
        }

    def _call_lp2_candidate_with_retries(
        self,
        *,
        prompt: str,
        person_id: str,
        candidate_index: int,
        candidate_count: int,
        relationship_provider: str,
        relationship_model: str,
        evidence: Dict[str, Any],
        timeout_schedule: Sequence[int | float],
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> tuple[Optional[Dict[str, Any]], int, Optional[Dict[str, Any]], int]:
        retry_count = 0
        last_failure_record: Optional[Dict[str, Any]] = None
        max_attempts = max(1, len(list(timeout_schedule or [])))
        schedule = [max(1, _safe_int(value, default=RELATIONSHIP_REQUEST_TIMEOUT_SECONDS)) for value in list(timeout_schedule or [])]
        if not schedule:
            schedule = [RELATIONSHIP_REQUEST_TIMEOUT_SECONDS]
            max_attempts = 1
        for attempt, read_timeout_seconds in enumerate(schedule, start=1):
            call_started_at = datetime.now().isoformat()
            try:
                payload = self._call_json_prompt(prompt, timeout=(15, read_timeout_seconds))
                return payload, retry_count, last_failure_record, read_timeout_seconds
            except Exception as exc:
                is_retryable = self._is_retryable_relationship_error(exc)
                will_retry = attempt < max_attempts and is_retryable
                last_failure_record = {
                    "timestamp": datetime.now().isoformat(),
                    "step": "lp2_relationship",
                    "person_id": person_id,
                    "candidate_index": candidate_index,
                    "candidate_count": candidate_count,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "provider": relationship_provider,
                    "model": relationship_model,
                    "call_started_at": call_started_at,
                    "call_finished_at": datetime.now().isoformat(),
                    "call_timeout_seconds": read_timeout_seconds,
                    "prompt_char_count": len(prompt),
                    "shared_event_count": len(list(evidence.get("shared_event_ids", []) or [])),
                    "shared_photo_count": len(list(evidence.get("supporting_photo_ids", []) or [])),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "is_retryable": is_retryable,
                    "outcome": "retrying" if will_retry else "skipped_after_retries",
                }
                _append_jsonl(self.llm_failures_path, last_failure_record)
                if will_retry:
                    retry_count += 1
                    retry_delay_seconds = RETRY_DELAY * attempt
                    self._notify(
                        progress_callback,
                        {
                            "message": "v0323 LP2 关系推断调用失败，准备重试",
                            "pipeline_family": PIPELINE_FAMILY_V0323,
                            "substage": "lp2_relationship",
                            "current_candidate_index": candidate_index,
                            "candidate_count": candidate_count,
                            "current_person_id": person_id,
                            "call_started_at": call_started_at,
                            "call_finished_at": last_failure_record["call_finished_at"],
                            "call_timeout_seconds": read_timeout_seconds,
                            "provider": relationship_provider,
                            "model": relationship_model,
                            "current_attempt": attempt,
                            "max_attempts": max_attempts,
                            "next_retry_delay_seconds": retry_delay_seconds,
                            "next_call_timeout_seconds": schedule[attempt],
                            "error": str(exc),
                            "will_retry": True,
                        },
                    )
                    time.sleep(retry_delay_seconds)
                    continue
                return None, retry_count, last_failure_record, read_timeout_seconds
        return None, retry_count, last_failure_record, schedule[-1]

    def _lp2_timeout_schedule(self, max_attempts: int) -> List[int]:
        schedule = [max(1, _safe_int(value, default=RELATIONSHIP_REQUEST_TIMEOUT_SECONDS)) for value in list(V0323_LP2_TIMEOUT_SCHEDULE_SECONDS or [])]
        if not schedule:
            schedule = [RELATIONSHIP_REQUEST_TIMEOUT_SECONDS]
        if len(schedule) >= max_attempts:
            return schedule[:max_attempts]
        return [*schedule, *([schedule[-1]] * (max_attempts - len(schedule)))]

    def _is_retryable_relationship_error(self, exc: Exception) -> bool:
        checker = getattr(self.llm_processor, "_is_retryable_error", None)
        if callable(checker):
            try:
                return bool(checker(exc))
            except Exception:
                pass
        message = str(exc).lower()
        retry_keywords = [
            "429",
            "rate limit",
            "connection",
            "timeout",
            "timed out",
            "temporarily unavailable",
            "reset by peer",
            "throttl",
            "too many requests",
            "bad gateway",
            "response ended prematurely",
        ]
        return any(keyword in message for keyword in retry_keywords)

    def _index_lp2_events_by_person(self, lp1_events: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        index: Dict[str, List[Dict[str, Any]]] = {}
        for event in lp1_events:
            person_ids = _unique_strings(
                [
                    *list(event.get("participant_person_ids", []) or []),
                    *list(event.get("depicted_person_ids", []) or []),
                ]
            )
            for person_id in person_ids:
                index.setdefault(person_id, []).append(event)
        return index

    def _build_relationship_evidence(
        self,
        *,
        person_id: str,
        observations: Sequence[Dict[str, Any]],
        primary_person_id: Optional[str],
        related_events: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        supporting_event_ids = [event["event_id"] for event in related_events]
        supporting_photo_ids = _sorted_unique_photo_ids(
            [
                photo_id
                for event in related_events
                for photo_id in list(event.get("supporting_photo_ids", []) or [])
            ],
            order_index=self.photo_order_index,
        )
        observation_map = self.observation_index or {item["photo_id"]: item for item in observations}
        related_observations = [observation_map[photo_id] for photo_id in supporting_photo_ids if photo_id in observation_map]
        day_keys = sorted(
            {
                str(item.get("timestamp") or "").split("T", 1)[0]
                for item in related_observations
                if "T" in str(item.get("timestamp") or "")
            }
        )
        scene_values = _unique_strings(
            [
                *[
                    value
                    for event in related_events
                    for value in list(event.get("place_refs", []) or [])
                ],
                *[
                    str((observation.get("location") or {}).get("name") or "").strip()
                    for observation in related_observations
                ],
            ]
        )
        contact_types = _unique_strings(
            person.get("contact_type")
            for observation in related_observations
            for person in list((observation.get("vlm_analysis") or {}).get("people", []) or [])
            if isinstance(person, dict) and str(person.get("person_id") or "").strip() == person_id
        )
        interactions = _unique_strings(
            [
                item.get("interaction_type")
                for event in related_events
                for item in list(event.get("social_dynamics", []) or [])
                if isinstance(item, dict) and str(item.get("target_id") or "").strip() == person_id
            ]
        )
        shared_facts = [
            {
                "fact_id": event["event_id"],
                "title": event.get("title"),
                "narrative_synthesis": event.get("narrative_synthesis"),
                "social_dynamics": [
                    item
                    for item in list(event.get("social_dynamics", []) or [])
                    if isinstance(item, dict) and str(item.get("target_id") or "").strip() == person_id
                ],
                "original_image_ids": list(event.get("supporting_photo_ids", []) or []),
            }
            for event in related_events[:10]
        ]
        third_party_counts: Counter[str] = Counter()
        exclusive_one_on_one = 0
        for event in related_events:
            participants = _unique_strings(
                [
                    *list(event.get("participant_person_ids", []) or []),
                    *list(event.get("depicted_person_ids", []) or []),
                ]
            )
            others = [
                participant
                for participant in participants
                if participant not in {person_id, str(primary_person_id or "").strip()}
            ]
            if not others and person_id in participants and primary_person_id and str(primary_person_id).strip() in participants:
                exclusive_one_on_one += 1
            for other in others:
                third_party_counts[other] += 1
        intimacy_weights = {
            "kiss": 1.0,
            "hug": 0.9,
            "holding_hands": 0.9,
            "arm_in_arm": 0.75,
            "selfie_together": 0.6,
            "shoulder_lean": 0.65,
            "sitting_close": 0.55,
            "standing_near": 0.35,
            "no_contact": 0.1,
        }
        intimacy_score = 0.0
        if contact_types:
            intimacy_score = round(
                sum(intimacy_weights.get(contact_type, 0.2) for contact_type in contact_types) / len(contact_types),
                3,
            )
        monthly_average = 0.0
        if day_keys:
            try:
                span_days = max(1, (datetime.fromisoformat(day_keys[-1]) - datetime.fromisoformat(day_keys[0])).days + 1)
            except Exception:
                span_days = max(1, len(day_keys))
            monthly_average = round(len(supporting_photo_ids) / max(1.0, span_days / 30.0), 2)
        evidence = {
            "shared_event_ids": supporting_event_ids,
            "co_occurrence_count": len(supporting_photo_ids),
            "distinct_days": len(day_keys),
            "monthly_average": monthly_average,
            "scenes": scene_values[:12],
            "contact_types": contact_types[:12],
            "interaction": interactions[:12],
            "exclusive_one_on_one": exclusive_one_on_one,
            "co_appearing_third_parties": [
                {"person_id": person, "count": count}
                for person, count in third_party_counts.most_common(5)
            ],
            "intimacy_score": intimacy_score,
            "trend": "stable" if len(day_keys) <= 1 else "longitudinal_observed",
            "shared_facts": shared_facts,
            "supporting_photo_ids": supporting_photo_ids,
        }
        return evidence

    def _relationship_candidate_requires_llm(self, evidence: Dict[str, Any]) -> bool:
        if int(evidence.get("distinct_days") or 0) >= RELATIONSHIP_MIN_DISTINCT_DAYS:
            return True
        if int(evidence.get("co_occurrence_count") or 0) >= RELATIONSHIP_MIN_CO_OCCURRENCE:
            return True
        if bool(evidence.get("contact_types")):
            return True
        if bool(evidence.get("interaction")):
            return True
        if int(evidence.get("exclusive_one_on_one") or 0) > 0:
            return True
        return float(evidence.get("intimacy_score") or 0.0) >= RELATIONSHIP_MIN_INTIMACY_SCORE

    def _build_relationship_prompt(
        self,
        *,
        person_id: str,
        primary_person_id: Optional[str],
        evidence: Dict[str, Any],
    ) -> str:
        return f"""You are a gossip-savvy social analyst who reads photo albums like a reality TV producer reads footage. Your job: figure out how {person_id} fits into {primary_person_id or "the authenticated user"}'s life.

# Evidence
- Co-occurrence: {evidence.get("co_occurrence_count", 0)} photos over {evidence.get("distinct_days", 0)} days (monthly avg: {evidence.get("monthly_average", 0)}/mo)
- Scenes: {json.dumps(evidence.get("scenes", []), ensure_ascii=False)}
- Contact types: {json.dumps(evidence.get("contact_types", []), ensure_ascii=False)}
- Interaction: {json.dumps(evidence.get("interaction", []), ensure_ascii=False)}
- Exclusive 1-on-1: {evidence.get("exclusive_one_on_one", 0)}
- Co-appearing third parties: {json.dumps(evidence.get("co_appearing_third_parties", []), ensure_ascii=False)}
- Intimacy score: {evidence.get("intimacy_score", 0.0)}
- Trend: {evidence.get("trend", "stable")}
- Shared facts: {json.dumps(evidence.get("shared_facts", []), ensure_ascii=False)}

# Relationship Types (pick exactly one)
- family
- romantic
- bestie
- close_friend
- friend
- classmate_colleague
- activity_buddy
- acquaintance

# Relationship Status (pick exactly one)
- new
- growing
- stable
- fading
- gone

Output JSON only:
{{
  "relationship_type": "friend",
  "status": "stable",
  "confidence": 0.0,
  "reason": "why"
}}"""

    def _normalize_relationship_payload(
        self,
        *,
        person_id: str,
        payload: Dict[str, Any],
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        relationship_type = str(payload.get("relationship_type") or "acquaintance").strip() or "acquaintance"
        status = str(payload.get("status") or "stable").strip() or "stable"
        confidence = round(max(0.0, min(1.0, _safe_float(payload.get("confidence")))), 4)
        reason = _normalized_text(payload.get("reason"))
        return {
            "relationship_id": f"REL_{person_id}",
            "person_id": person_id,
            "relationship_type": relationship_type,
            "status": status,
            "confidence": confidence,
            "reason": reason,
            "supporting_event_ids": list(evidence.get("shared_event_ids", [])),
            "supporting_photo_ids": list(evidence.get("supporting_photo_ids", [])),
            "evidence_snapshot": {
                "co_occurrence_count": evidence.get("co_occurrence_count", 0),
                "distinct_days": evidence.get("distinct_days", 0),
                "monthly_average": evidence.get("monthly_average", 0.0),
                "intimacy_score": evidence.get("intimacy_score", 0.0),
                "scenes": list(evidence.get("scenes", [])),
                "contact_types": list(evidence.get("contact_types", [])),
                "interaction": list(evidence.get("interaction", [])),
                "shared_facts": list(evidence.get("shared_facts", [])),
            },
        }

    def _run_lp3_profile(
        self,
        *,
        primary_person_id: Optional[str],
        lp1_events: Sequence[Dict[str, Any]],
        lp2_relationships: Sequence[Dict[str, Any]],
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        event_payload = [
            {
                "event_id": event.get("event_id"),
                "title": event.get("title"),
                "started_at": event.get("started_at"),
                "ended_at": event.get("ended_at"),
                "place_refs": event.get("place_refs"),
                "participant_person_ids": event.get("participant_person_ids"),
                "persona_evidence": event.get("persona_evidence"),
                "tags": event.get("tags"),
                "narrative_synthesis": event.get("narrative_synthesis"),
            }
            for event in lp1_events[:60]
        ]
        relationship_payload = [
            {
                "person_id": relationship.get("person_id"),
                "relationship_type": relationship.get("relationship_type"),
                "status": relationship.get("status"),
                "confidence": relationship.get("confidence"),
                "reason": relationship.get("reason"),
                "supporting_event_ids": relationship.get("supporting_event_ids"),
            }
            for relationship in lp2_relationships[:20]
        ]
        structured_prompt = f"""You are LP3 for a personal memory system.

Generate a structured profile grounded in LP1 events and LP2 relationships.

Requirements:
1. Output JSON only.
2. Separate event-grounded and relationship-grounded signals.
3. Do not invent facts that are absent from the input.
4. weak_reference should stay empty unless the input explicitly contains weak evidence.

Output schema:
{{
  "primary_person_id": "{primary_person_id or ''}",
  "event_grounded": {{
    "behavioral_traits": [],
    "aesthetic_traits": [],
    "socioeconomic_traits": [],
    "activity_patterns": [],
    "place_patterns": []
  }},
  "relationship_grounded": {{
    "top_relationships": [],
    "social_style": []
  }},
  "weak_reference": [],
  "summary": ""
}}

Events:
{json.dumps(event_payload, ensure_ascii=False, indent=2)}

Relationships:
{json.dumps(relationship_payload, ensure_ascii=False, indent=2)}
"""
        try:
            structured = self._call_json_prompt(structured_prompt)
            if not isinstance(structured, dict):
                structured = {}
        except Exception:
            structured = {}

        report_prompt = f"""Write a concise Markdown profile report for the authenticated user.

Requirements:
1. Mention only evidence grounded in the input.
2. Use short sections.
3. If evidence is sparse, say so directly.

Structured profile:
{json.dumps(structured, ensure_ascii=False, indent=2)}

Events:
{json.dumps(event_payload[:20], ensure_ascii=False, indent=2)}

Relationships:
{json.dumps(relationship_payload[:10], ensure_ascii=False, indent=2)}
"""
        report_markdown = ""
        try:
            report_markdown = self._call_markdown_prompt(report_prompt)
        except Exception:
            report_markdown = ""
        if not report_markdown:
            report_markdown = self._fallback_profile_markdown(
                primary_person_id=primary_person_id,
                lp1_events=lp1_events,
                lp2_relationships=lp2_relationships,
            )
        self._notify(
            progress_callback,
            {
                "message": "v0323 LP3 画像生成完成",
                "pipeline_family": PIPELINE_FAMILY_V0323,
                "substage": "lp3_profile",
                "event_count": len(lp1_events),
                "relationship_count": len(lp2_relationships),
                "percent": 100,
            },
        )
        return {
            "structured": structured,
            "report_markdown": report_markdown,
        }

    def _fallback_profile_markdown(
        self,
        *,
        primary_person_id: Optional[str],
        lp1_events: Sequence[Dict[str, Any]],
        lp2_relationships: Sequence[Dict[str, Any]],
    ) -> str:
        lines = [
            "# 用户画像",
            "",
            f"- 主角: {primary_person_id or 'unknown'}",
            f"- 事件数: {len(lp1_events)}",
            f"- 关系数: {len(lp2_relationships)}",
        ]
        if lp1_events:
            lines.append("")
            lines.append("## 事件线索")
            for event in list(lp1_events)[:5]:
                lines.append(f"- {event.get('title') or event.get('event_id')}: {_truncate_text(event.get('narrative_synthesis'), limit=120)}")
        if lp2_relationships:
            lines.append("")
            lines.append("## 关系线索")
            for relationship in list(lp2_relationships)[:5]:
                lines.append(
                    f"- {relationship.get('person_id')}: {relationship.get('relationship_type')} / {relationship.get('status')} ({relationship.get('confidence', 0):.2f})"
                )
        return "\n".join(lines).strip()
