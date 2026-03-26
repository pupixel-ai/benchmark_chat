from __future__ import annotations

from typing import Any, Dict

from .events import extract_events_from_state
from .groups import detect_groups
from .person_screening import screen_people
from .primary_person import analyze_primary_person_with_reflection
from .profile_fields import build_profile_context, generate_structured_profile
from .relationships import build_relationship_dossiers, infer_relationships_from_dossiers
from .types import MemoryState


def run_memory_pipeline(
    state: MemoryState,
    llm_processor: Any,
    fallback_primary_person_id: str | None = None,
) -> Dict[str, Any]:
    if not state.screening:
        state.screening = screen_people(state)

    if not state.primary_decision:
        primary_decision, primary_reflection = analyze_primary_person_with_reflection(
            state=state,
            fallback_primary_person_id=fallback_primary_person_id,
            llm_processor=llm_processor,
        )
        state.primary_decision = primary_decision.to_dict()
        state.primary_reflection = primary_reflection
    elif not state.primary_reflection:
        fallback_primary_person_id = (state.primary_decision or {}).get("primary_person_id")
        _, primary_reflection = analyze_primary_person_with_reflection(
            state=state,
            fallback_primary_person_id=fallback_primary_person_id,
            llm_processor=llm_processor,
        )
        state.primary_reflection = primary_reflection

    if not state.events:
        state.events = extract_events_from_state(state, llm_processor)

    dossiers = build_relationship_dossiers(state=state, llm_processor=llm_processor)
    relationships, dossiers = infer_relationships_from_dossiers(
        state=state,
        llm_processor=llm_processor,
        dossiers=dossiers,
    )
    state.relationship_dossiers = dossiers
    state.relationships = relationships

    state.groups = detect_groups(state)
    state.profile_context = build_profile_context(state)
    profile_result = generate_structured_profile(state, llm_processor=llm_processor)

    return _build_pipeline_result(state, profile_result)


def rerun_pipeline_from_primary_backflow(
    *,
    state: MemoryState,
    llm_processor: Any,
) -> Dict[str, Any]:
    dossiers = build_relationship_dossiers(state=state, llm_processor=llm_processor)
    relationships, dossiers = infer_relationships_from_dossiers(
        state=state,
        llm_processor=llm_processor,
        dossiers=dossiers,
    )
    state.relationship_dossiers = dossiers
    state.relationships = relationships
    state.groups = detect_groups(state)
    state.profile_context = build_profile_context(state)
    profile_result = generate_structured_profile(state, llm_processor=llm_processor)
    return _build_pipeline_result(state, profile_result)


def rerun_pipeline_from_relationship_backflow(
    *,
    state: MemoryState,
    llm_processor: Any,
) -> Dict[str, Any]:
    state.groups = detect_groups(state)
    state.profile_context = build_profile_context(state)
    profile_result = generate_structured_profile(state, llm_processor=llm_processor)
    return _build_pipeline_result(state, profile_result)


def _build_pipeline_result(state: MemoryState, profile_result: Dict[str, Any]) -> Dict[str, Any]:

    return {
        "events": state.events,
        "relationships": state.relationships,
        "structured": profile_result["structured"],
        "report": "",
        "debug": {
            "field_decision_count": len(profile_result.get("field_decisions", [])),
            "report_reasoning": {},
        },
        "consistency": profile_result["consistency"],
        "internal_artifacts": {
            "screening": {person_id: screening.to_dict() for person_id, screening in state.screening.items()},
            "primary_decision": state.primary_decision,
            "primary_reflection": state.primary_reflection or {},
            "relationship_dossiers": [dossier.to_dict() for dossier in state.relationship_dossiers],
            "group_artifacts": [group.to_dict() for group in state.groups],
            "profile_fact_decisions": profile_result["field_decisions"],
            "profile_llm_batch_debug": profile_result.get("llm_batch_debug", []),
        },
    }


def build_memory_state(
    *,
    photos: list,
    face_db: dict,
    vlm_results: list,
) -> MemoryState:
    return MemoryState(
        photos=photos,
        face_db=face_db,
        vlm_results=vlm_results,
    )
