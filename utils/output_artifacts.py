"""
轻量输出 helper：负责序列化和落盘，不依赖重型服务模块。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

from . import save_json


def _normalize_generated_at(generated_at: str | datetime | None = None) -> str:
    if isinstance(generated_at, datetime):
        return generated_at.isoformat()
    if isinstance(generated_at, str) and generated_at:
        return generated_at
    return datetime.now().isoformat()


def serialize_event(event: Any) -> Dict[str, Any]:
    return {
        "event_id": event.event_id,
        "date": event.date,
        "time_range": event.time_range,
        "duration": event.duration,
        "title": event.title,
        "type": event.type,
        "participants": event.participants,
        "location": event.location,
        "description": event.description,
        "photo_count": event.photo_count,
        "confidence": event.confidence,
        "reason": event.reason,
        "narrative": event.narrative,
        "narrative_synthesis": event.narrative_synthesis,
        "meta_info": event.meta_info,
        "objective_fact": event.objective_fact,
        "social_interaction": event.social_interaction,
        "social_dynamics": event.social_dynamics,
        "lifestyle_tags": event.lifestyle_tags,
        "tags": event.tags,
        "social_slices": event.social_slices,
        "persona_evidence": event.persona_evidence,
    }


def serialize_relationship(relationship: Any) -> Dict[str, Any]:
    return {
        "person_id": relationship.person_id,
        "relationship_type": relationship.relationship_type,
        "intimacy_score": relationship.intimacy_score,
        "status": relationship.status,
        "confidence": relationship.confidence,
        "reasoning": relationship.reasoning,
        "shared_events": relationship.shared_events,
        "evidence": relationship.evidence,
    }


def serialize_face_db(face_db: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        person_id: {
            "name": person.name if hasattr(person, "name") else "",
            "photo_count": person.photo_count if hasattr(person, "photo_count") else person.get("photo_count", 0) if isinstance(person, dict) else 0,
            "first_seen": person.first_seen.isoformat() if hasattr(person, "first_seen") and person.first_seen else None,
            "last_seen": person.last_seen.isoformat() if hasattr(person, "last_seen") and person.last_seen else None,
            "avg_confidence": person.avg_confidence if hasattr(person, "avg_confidence") else person.get("avg_confidence", 0) if isinstance(person, dict) else 0,
        }
        for person_id, person in face_db.items()
    }


def build_relationships_artifact(
    relationships: Iterable[Any],
    primary_person_id: str | None,
    generated_at: str | datetime | None = None,
    version: str = "2.0",
) -> Dict[str, Any]:
    serialized_relationships = [serialize_relationship(rel) for rel in relationships]
    return {
        "metadata": {
            "generated_at": _normalize_generated_at(generated_at),
            "version": version,
            "primary_person_id": primary_person_id,
            "total_relationships": len(serialized_relationships),
        },
        "relationships": serialized_relationships,
    }


def build_profile_debug_artifact(
    profile_result: Dict[str, Any] | None,
    primary_person_id: str | None,
    total_events: int,
    total_relationships: int,
    generated_at: str | datetime | None = None,
    version: str = "2.0",
) -> Dict[str, Any]:
    profile_result = profile_result or {}
    return {
        "metadata": {
            "generated_at": _normalize_generated_at(generated_at),
            "version": version,
            "primary_person_id": primary_person_id,
            "total_events": total_events,
            "total_relationships": total_relationships,
        },
        "debug": profile_result.get("debug", {}),
        "consistency": profile_result.get("consistency", {}),
    }


def build_artifacts_manifest(**paths: str | None) -> Dict[str, str | None]:
    manifest = {}
    for key, value in paths.items():
        manifest[key] = str(Path(value)) if value else None
    return manifest


def build_internal_artifact(
    artifact_name: str,
    payload: Any,
    generated_at: str | datetime | None = None,
    version: str = "2.0",
    **metadata: Any,
) -> Dict[str, Any]:
    return {
        "metadata": {
            "generated_at": _normalize_generated_at(generated_at),
            "version": version,
            **metadata,
        },
        artifact_name: payload,
    }


def build_final_output_payload(
    events: Iterable[Any],
    relationships: Iterable[Any],
    face_db: Dict[str, Any],
    artifacts: Dict[str, str | None],
    models: Dict[str, str],
    generated_at: str | datetime | None = None,
    version: str = "2.0",
) -> Dict[str, Any]:
    serialized_events = [serialize_event(event) for event in events]
    serialized_relationships = [serialize_relationship(rel) for rel in relationships]
    return {
        "metadata": {
            "generated_at": _normalize_generated_at(generated_at),
            "version": version,
            "total_events": len(serialized_events),
            "total_relationships": len(serialized_relationships),
            "models": models,
        },
        "events": serialized_events,
        "relationships": serialized_relationships,
        "face_db": serialize_face_db(face_db),
        "artifacts": artifacts,
    }


def save_json_artifact(payload: Dict[str, Any], path: str) -> str:
    save_json(payload, path)
    return path


def save_markdown_report(result: Dict[str, Any], path: str) -> str:
    lines = [
        "# 记忆工程 v2.0 输出摘要",
        "",
        f"- 生成时间: {result.get('metadata', {}).get('generated_at', '未知')}",
        f"- 主角 person_id: {result.get('summary', {}).get('primary_person_id', '未知')}",
        f"- 事件数: {result.get('summary', {}).get('total_events', 0)}",
        f"- 关系数: {result.get('summary', {}).get('total_relationships', 0)}",
        "",
        "## 关系摘要",
    ]

    relationships = result.get("relationships", [])
    if relationships:
        for relationship in relationships:
            lines.append(
                f"- {relationship.get('person_id', '未知')}: {relationship.get('relationship_type', '未知')} / {relationship.get('status', '未知')} / {relationship.get('confidence', 0):.0%}"
            )
    else:
        lines.append("- 无")

    lines.extend(["", "## Artifacts"])
    for key, value in result.get("artifacts", {}).items():
        lines.append(f"- {key}: {value}")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(output_path)
