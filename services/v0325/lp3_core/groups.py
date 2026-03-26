from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from .relationships import select_group_candidates
from .types import GroupArtifact, MemoryState


def detect_groups(state: MemoryState) -> List[GroupArtifact]:
    dossiers = getattr(state, "relationship_dossiers", None) or []
    candidate_relationships = select_group_candidates(state.relationships or [], dossiers)
    if len(candidate_relationships) < 2:
        return []

    relationship_ids = {relationship.person_id for relationship in candidate_relationships}
    event_memberships: Dict[str, List[str]] = defaultdict(list)
    event_lookup: Dict[str, Any] = {}

    for relationship in candidate_relationships:
        for shared_event in relationship.shared_events:
            event_id = shared_event.get("event_id")
            if not event_id:
                continue
            event_memberships[event_id].append(relationship.person_id)

    for event in state.events or []:
        event_lookup[event.event_id] = event

    groups: List[GroupArtifact] = []
    counter = 1
    for event_id, members in event_memberships.items():
        unique_members = sorted(set(members))
        if len(unique_members) < 2:
            continue
        event = event_lookup.get(event_id)
        if not event:
            continue
        group_type = _infer_group_type(event)
        strong_evidence_refs = [
            {
                "event_id": event.event_id,
                "signal": event.title,
                "why": f"group_members={','.join(unique_members)}",
            }
        ]
        groups.append(
            GroupArtifact(
                group_id=f"GRP_{counter:03d}",
                members=unique_members,
                group_type_candidate=group_type,
                confidence=_score_group_confidence(unique_members, event, relationship_ids),
                strong_evidence_refs=strong_evidence_refs,
                reason=f"stable_shared_event:{event.event_id}",
            )
        )
        counter += 1

    return groups


def _infer_group_type(event: Any) -> str:
    haystack = " ".join(
        str(value or "")
        for value in (
            getattr(event, "title", ""),
            getattr(event, "location", ""),
            getattr(event, "description", ""),
            getattr(event, "narrative_synthesis", ""),
        )
    ).lower()
    if any(keyword in haystack for keyword in ("sorority", "greek", "formal", "姐妹会")):
        return "sorority"
    if any(keyword in haystack for keyword in ("lab", "实验室")):
        return "lab"
    if any(keyword in haystack for keyword in ("team", "比赛", "球")):
        return "team"
    if any(keyword in haystack for keyword in ("club", "社团")):
        return "club"
    return "friend_group"


def _score_group_confidence(members: List[str], event: Any, relationship_ids: set[str]) -> float:
    participant_overlap = len([person_id for person_id in getattr(event, "participants", []) if person_id in relationship_ids])
    base = 0.45 + min(0.15 * len(members), 0.3)
    if participant_overlap >= len(members):
        base += 0.1
    if getattr(event, "photo_count", 0) >= 3:
        base += 0.08
    if getattr(event, "confidence", 0.0) >= 0.8:
        base += 0.05
    return round(min(base, 0.95), 3)
