from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def load_feedback_cases(queue_path: str) -> List[Dict[str, Any]]:
    path = Path(queue_path)
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def load_pending_feedback_cases(queue_path: str) -> List[Dict[str, Any]]:
    records = load_feedback_cases(queue_path)
    return [record for record in records if str(record.get("status") or "pending") == "pending"]


def mark_feedback_cases_processed(
    *,
    queue_path: str,
    case_ids: Iterable[str] | None = None,
) -> Dict[str, int]:
    records = load_feedback_cases(queue_path)
    if not records:
        return {"updated": 0, "total": 0}
    target_case_ids = {str(case_id) for case_id in list(case_ids or []) if str(case_id)}
    updated = 0
    for record in records:
        current_status = str(record.get("status") or "pending")
        if current_status != "pending":
            continue
        if target_case_ids and str(record.get("case_id") or "") not in target_case_ids:
            continue
        record["status"] = "processed"
        updated += 1
    _write_jsonl(Path(queue_path), records)
    return {"updated": updated, "total": len(records)}


def mirror_pending_cases_to_downstream(
    *,
    pending_cases: Iterable[Dict[str, Any]],
    profile_agent_root: str,
) -> Tuple[str, int]:
    root = Path(profile_agent_root)
    hard_cases_path = root / "data" / "hard_cases.json"
    hard_cases_path.parent.mkdir(parents=True, exist_ok=True)
    mirrored: List[Dict[str, Any]] = []
    for case in pending_cases:
        mirrored.append(
            {
                "album_id": case.get("album_id"),
                "source": case.get("source") or "judge",
                "dimension": case.get("dimension"),
                "extractor_said": case.get("extractor_v2_value")
                if case.get("extractor_v2_value") not in (None, "")
                else case.get("extractor_v1_value"),
                "failure_mode": case.get("failure_mode", ""),
                "evidence_used": list(case.get("evidence_used") or []),
                "timestamp": str(case.get("created_at") or "")[:10],
                "status": "pending",
                "agent_type": case.get("agent_type"),
                "case_id": case.get("case_id"),
            }
        )
    with hard_cases_path.open("w", encoding="utf-8") as handle:
        json.dump(mirrored, handle, ensure_ascii=False, indent=2)
    return str(hard_cases_path), len(mirrored)


def persist_downstream_feedback_cases(
    *,
    downstream_audit_report: Dict[str, Any],
    user_name: str,
    run_timestamp: str,
    album_id: str,
    queue_path: str,
    run_output_path: str,
    profile_agent_root: str,
) -> Dict[str, Any]:
    extracted = _extract_cases_from_report(
        downstream_audit_report=downstream_audit_report,
        user_name=user_name,
        run_timestamp=run_timestamp,
        album_id=album_id,
    )
    existing = load_feedback_cases(queue_path)
    existing_keys = {_build_dedupe_key(record) for record in existing}
    new_records: List[Dict[str, Any]] = []
    for candidate in extracted:
        dedupe_key = _build_dedupe_key(candidate)
        if dedupe_key in existing_keys:
            continue
        existing_keys.add(dedupe_key)
        new_records.append(candidate)

    all_records = existing + new_records
    if new_records:
        _append_jsonl(Path(queue_path), new_records)

    pending_records = [record for record in all_records if str(record.get("status") or "pending") == "pending"]
    mirror_path, mirrored_count = mirror_pending_cases_to_downstream(
        pending_cases=pending_records,
        profile_agent_root=profile_agent_root,
    )

    run_output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "user_name": user_name,
            "run_timestamp": run_timestamp,
            "album_id": album_id,
            "queue_path": queue_path,
            "mirror_path": mirror_path,
        },
        "summary": {
            "extracted_count": len(extracted),
            "written_count": len(new_records),
            "dedup_skipped_count": len(extracted) - len(new_records),
            "pending_count": len(pending_records),
            "mirrored_count": mirrored_count,
        },
        "cases": new_records,
    }
    run_output_file = Path(run_output_path)
    run_output_file.parent.mkdir(parents=True, exist_ok=True)
    with run_output_file.open("w", encoding="utf-8") as handle:
        json.dump(run_output, handle, ensure_ascii=False, indent=2)

    return {
        "run_output_path": str(run_output_file),
        "queue_path": queue_path,
        "mirror_path": mirror_path,
        "written_count": len(new_records),
        "pending_count": len(pending_records),
        "mirrored_count": mirrored_count,
        "case_ids": [record.get("case_id") for record in new_records if record.get("case_id")],
    }


def _extract_cases_from_report(
    *,
    downstream_audit_report: Dict[str, Any],
    user_name: str,
    run_timestamp: str,
    album_id: str,
) -> List[Dict[str, Any]]:
    metadata = downstream_audit_report.get("metadata", {}) if isinstance(downstream_audit_report, dict) else {}
    audit_status = str(metadata.get("audit_status") or "")
    if audit_status == "skipped_init_failure":
        failure_mode = (
            f"skipped_init_failure: {metadata.get('audit_error_type', 'RuntimeError')}: "
            f"{metadata.get('audit_error_message', '')}"
        ).strip()
        return [
            _build_case_record(
                user_name=user_name,
                run_timestamp=run_timestamp,
                album_id=album_id,
                agent_type="system",
                dimension="system>audit_runtime_failure",
                verdict="runtime_failure",
                failure_mode=failure_mode,
                extractor_v1_value=None,
                extractor_v2_value=None,
                evidence_used=[],
                source="audit_runtime_failure",
            )
        ]

    cases: List[Dict[str, Any]] = []
    for agent_type in ("protagonist", "relationship", "profile"):
        section = downstream_audit_report.get(agent_type, {})
        if not isinstance(section, dict):
            continue
        decisions = list((section.get("judge_output") or {}).get("decisions") or [])
        v1_tags = list((section.get("extractor_output") or {}).get("tags") or [])
        v2_tags = list((section.get("extractor_v2_output") or {}).get("tags") or v1_tags)
        v1_by_dimension = {str(tag.get("dimension") or ""): tag for tag in v1_tags}
        v2_by_dimension = {str(tag.get("dimension") or ""): tag for tag in v2_tags}

        for decision in decisions:
            verdict = str(decision.get("verdict") or "accept")
            if verdict not in {"nullify", "downgrade"}:
                continue
            dimension = str(decision.get("dimension") or "")
            if not dimension:
                continue
            reason = str(decision.get("reason") or "").strip()
            failure_mode = f"{verdict}: {reason}" if reason else verdict
            v1_tag = v1_by_dimension.get(dimension, {})
            v2_tag = v2_by_dimension.get(dimension, {})
            evidence_used = _normalize_evidence_used(
                list(v2_tag.get("evidence") or v1_tag.get("evidence") or [])
            )
            cases.append(
                _build_case_record(
                    user_name=user_name,
                    run_timestamp=run_timestamp,
                    album_id=album_id,
                    agent_type=agent_type,
                    dimension=dimension,
                    verdict=verdict,
                    failure_mode=failure_mode,
                    extractor_v1_value=v1_tag.get("value"),
                    extractor_v2_value=v2_tag.get("value"),
                    evidence_used=evidence_used,
                    source="judge",
                )
            )
    return cases


def _build_case_record(
    *,
    user_name: str,
    run_timestamp: str,
    album_id: str,
    agent_type: str,
    dimension: str,
    verdict: str,
    failure_mode: str,
    extractor_v1_value: Any,
    extractor_v2_value: Any,
    evidence_used: List[Dict[str, Any]],
    source: str,
) -> Dict[str, Any]:
    created_at = datetime.now().isoformat()
    failure_mode_hash = _hash_text(failure_mode)
    evidence_hash = _hash_payload(evidence_used)
    dedupe_key = "|".join(
        [
            str(album_id),
            str(agent_type),
            str(dimension),
            str(verdict),
            failure_mode_hash,
            evidence_hash,
        ]
    )
    case_id = f"fc_{_hash_text(dedupe_key)}"
    return {
        "case_id": case_id,
        "user_name": user_name,
        "run_timestamp": run_timestamp,
        "album_id": album_id,
        "agent_type": agent_type,
        "dimension": dimension,
        "verdict": verdict,
        "failure_mode": failure_mode,
        "extractor_v1_value": extractor_v1_value,
        "extractor_v2_value": extractor_v2_value,
        "evidence_used": evidence_used,
        "source": source,
        "status": "pending",
        "created_at": created_at,
    }


def _build_dedupe_key(case: Dict[str, Any]) -> str:
    failure_mode = str(case.get("failure_mode") or "")
    evidence_used = list(case.get("evidence_used") or [])
    return "|".join(
        [
            str(case.get("album_id") or ""),
            str(case.get("agent_type") or ""),
            str(case.get("dimension") or ""),
            str(case.get("verdict") or ""),
            _hash_text(failure_mode),
            _hash_payload(evidence_used),
        ]
    )


def _normalize_evidence_used(evidence_used: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    seen = set()
    for item in evidence_used:
        if not isinstance(item, dict):
            continue
        try:
            inference_depth = int(item.get("inference_depth") or 1)
        except (TypeError, ValueError):
            inference_depth = 1
        payload = {
            "event_id": str(item.get("event_id") or "").strip() or None,
            "photo_id": str(item.get("photo_id") or "").strip() or None,
            "person_id": str(item.get("person_id") or "").strip() or None,
            "feature_names": [
                str(feature_name).strip()
                for feature_name in list(item.get("feature_names") or [])
                if str(feature_name).strip()
            ],
            "description": str(item.get("description") or "").strip(),
            "evidence_type": str(item.get("evidence_type") or "direct"),
            "inference_depth": inference_depth,
        }
        identity = (
            payload["event_id"],
            payload["photo_id"],
            payload["person_id"],
            tuple(payload["feature_names"]),
            payload["description"],
        )
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(payload)
    return normalized


def _hash_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _hash_payload(payload: Any) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return _hash_text(serialized)


def _append_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
