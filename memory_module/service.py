"""Task-scoped memory ingestion/materialization service."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence
from uuid import NAMESPACE_URL, uuid5

from memory_module.adapters import MemoryStoragePublisher
from memory_module.domain import MaterializationInputBundle, ProfileEvidenceItem
from memory_module.dto import (
    ArtifactRefDTO,
    ConceptDTO,
    ChangeLogEntryDTO,
    EventCandidateDTO,
    EventCandidateDTO,
    FaceObservationDTO,
    MoodStateHypothesisDTO,
    MemoryIngestionEnvelopeDTO,
    PeriodHypothesisDTO,
    PersonObservationDTO,
    PhotoFactDTO,
    PlaceAnchorDTO,
    PrimaryPersonHypothesisDTO,
    ProfileEvidenceDTO,
    RelationshipHypothesisDTO,
    ScopeDTO,
    VLMPhotoObservationDTO,
)
from memory_module.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_VERSION,
    EmbeddingProvider,
    build_embedding_payload,
)
from memory_module.evaluation import ErrorBucketDTO, EvaluationRunDTO, MetricSnapshotDTO
from memory_module.ontology import collect_concepts, concept_metadata, normalize_concept, suggest_candidate_concept
from memory_module.records import (
    MilvusSegmentRecord,
    Neo4jConceptNodeRecord,
    Neo4jEventNodeRecord,
    Neo4jMoodStateNodeRecord,
    Neo4jPersonNodeRecord,
    Neo4jPlaceNodeRecord,
    Neo4jPrimaryPersonHypothesisNodeRecord,
    Neo4jRelationshipEdgeRecord,
    Neo4jRelationshipHypothesisNodeRecord,
    Neo4jSessionNodeRecord,
    Neo4jPeriodHypothesisNodeRecord,
    Neo4jTimelineNodeRecord,
    Neo4jUserNodeRecord,
    PersonIdentityMapRecord,
    PhotoIdentityMapRecord,
    RedisProfileCoreRecord,
    RedisProfileDebugRefsRecord,
    RedisProfileMetaRecord,
    RedisProfileRecentEventsRecord,
    RedisProfileRecentTimelinesRecord,
    RedisProfileRelationshipsRecord,
)
from memory_module.sequencing import SequenceBundle, build_sequences
from memory_module.views import (
    EvidenceChainView,
    FaceStageView,
    FocusGraphView,
    GraphEdgeView,
    GraphNodeView,
    LLMStageView,
    MilvusStateView,
    Neo4jStateView,
    ObjectDiffView,
    PublishDecisionView,
    RedisStateView,
    SequenceStageView,
    TraceView,
    VLMStageView,
)
from models import Event, Relationship
from utils import save_json


RELATIONSHIP_LABEL_THRESHOLDS = {
    "friend": {"upgrade": 0.60, "keep": 0.50},
    "close_friend": {"upgrade": 0.72, "keep": 0.62},
    "colleague": {"upgrade": 0.68, "keep": 0.58},
    "partner": {"upgrade": 0.78, "keep": 0.68},
    "family_generic": {"upgrade": 0.82, "keep": 0.72},
    "father": {"upgrade": 0.86, "keep": 0.76},
    "mother": {"upgrade": 0.86, "keep": 0.76},
}


class MemoryModuleService:
    """Builds task-scoped memory outputs from face/VLM/LLM artifacts."""

    def __init__(
        self,
        task_id: str,
        task_dir: str,
        user_id: Optional[str] = None,
        source_system: str = "memory_engineering",
        pipeline_version: str = "",
        public_url_builder: Optional[Callable[[Path | str], Optional[str]]] = None,
    ) -> None:
        self.task_id = task_id
        self.task_dir = Path(task_dir)
        self.user_id = user_id or f"task:{task_id}"
        self.source_system = source_system
        self.pipeline_version = pipeline_version
        self.public_url_builder = public_url_builder
        self.output_dir = self.task_dir / "output" / "memory"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scope_key = f"{self.user_id}:{self.task_id}:{self.pipeline_version or 'unknown'}"
        self.embedder = EmbeddingProvider.from_config()

    def materialize(
        self,
        photos: Sequence,
        face_output: Dict[str, Any],
        vlm_results: Sequence[Dict[str, Any]],
        events: Sequence[Event],
        relationships: Sequence[Relationship],
        profile_markdown: str,
        cached_photo_ids: Optional[set[str]] = None,
    ) -> Dict[str, Any]:
        generated_at = datetime.now(timezone.utc).isoformat()
        cached_photo_ids = cached_photo_ids or set()
        vlm_by_photo_id = {
            item.get("photo_id"): item
            for item in vlm_results
            if isinstance(item, dict) and item.get("photo_id")
        }

        photo_uuid_map = {
            photo.photo_id: self._stable_uuid("photo", photo.photo_id)
            for photo in photos
        }
        face_person_ids = self._collect_face_person_ids(photos, face_output, relationships)
        person_uuid_map = {
            face_person_id: self._canonical_person_uuid(face_person_id)
            for face_person_id in sorted(face_person_ids)
        }
        primary_face_person_id = self._resolve_primary_face_person_id(photos, face_output)
        primary_person_uuid = person_uuid_map.get(primary_face_person_id) if primary_face_person_id else None

        photos_dto = self._build_photo_facts(photos, photo_uuid_map, person_uuid_map, vlm_by_photo_id)
        sequences = build_sequences(photos, photo_uuid_map, person_uuid_map, self.scope_key)
        session_lookup = {session.session_id: session for session in sequences.sessions}
        sessions_by_day = defaultdict(list)
        for session in sequences.sessions:
            sessions_by_day[session.day_key].append(session)

        event_candidates = self._build_event_candidates(
            events=events,
            sessions=sequences.sessions,
            photo_uuid_map=photo_uuid_map,
            person_uuid_map=person_uuid_map,
        )
        self._refine_event_candidates(
            event_candidates=event_candidates,
            photos=photos_dto,
            sessions=sequences.sessions,
        )
        event_candidates = self._merge_event_candidates(event_candidates)
        prior_storage = self._load_prior_storage()
        place_anchors = self._build_place_anchors(sequences.sessions)
        concept_catalog = self._build_concepts(
            sessions=sequences.sessions,
            event_candidates=event_candidates,
            relationships=relationships,
        )
        primary_person_hypothesis = self._build_primary_person_hypothesis(
            sequences=sequences,
            primary_person_uuid=primary_person_uuid,
        )
        relationship_hypotheses = self._build_relationships(
            relationships=relationships,
            person_uuid_map=person_uuid_map,
            sequences=sequences,
            event_candidates=event_candidates,
            primary_person_uuid=primary_person_uuid,
            primary_face_person_id=primary_face_person_id,
            prior_storage=prior_storage,
        )
        mood_hypotheses = self._build_mood_hypotheses(sequences)
        period_hypotheses = self._build_period_hypotheses(
            sequences=sequences,
            concept_catalog=concept_catalog,
        )
        profile_evidence = self._build_profile_evidence(
            events=events,
            event_candidates=event_candidates,
            relationship_hypotheses=relationship_hypotheses,
        )

        change_log = self._build_change_log(
            photos_dto=photos_dto,
            sequences=sequences,
            event_candidates=event_candidates,
            relationship_hypotheses=relationship_hypotheses,
            profile_evidence=profile_evidence,
        )

        artifacts = self._build_top_level_artifacts(photos, profile_markdown)
        scope = ScopeDTO(
            user_id=self.user_id,
            source_system=self.source_system,
            pipeline_version=self.pipeline_version,
            task_id=self.task_id,
            ingestion_id=self._stable_uuid("ingestion", self.task_id),
            generated_at=generated_at,
        )
        envelope = MemoryIngestionEnvelopeDTO(
            scope=scope,
            photos=photos_dto,
            bursts=sequences.bursts,
            sessions=sequences.sessions,
            movements=sequences.movements,
            day_timelines=sequences.day_timelines,
            event_candidates=event_candidates,
            relationship_hypotheses=relationship_hypotheses,
            profile_evidence=profile_evidence,
            artifacts=artifacts,
            change_log=change_log,
        )

        profile_version = 1
        bundle = self._build_materialization_bundle(photos_dto, event_candidates, relationship_hypotheses, profile_evidence)
        storage = self._build_storage_records(
            scope=scope,
            photos=photos_dto,
            sequences=sequences,
            event_candidates=event_candidates,
            place_anchors=place_anchors,
            concept_catalog=concept_catalog,
            relationship_hypotheses=relationship_hypotheses,
            mood_hypotheses=mood_hypotheses,
            primary_person_hypothesis=primary_person_hypothesis,
            period_hypotheses=period_hypotheses,
            profile_evidence=profile_evidence,
            bundle=bundle,
            profile_markdown=profile_markdown,
            person_uuid_map=person_uuid_map,
            primary_face_person_id=primary_face_person_id,
            primary_person_uuid=primary_person_uuid,
            profile_version=profile_version,
        )
        focus_graph = self._build_focus_graph(
            photos=photos_dto,
            sequences=sequences,
            event_candidates=event_candidates,
            relationship_hypotheses=relationship_hypotheses,
            person_uuid_map=person_uuid_map,
            primary_face_person_id=primary_face_person_id,
            primary_person_uuid=primary_person_uuid,
        )
        storage["neo4j"]["focus_graph"] = self._serialize(focus_graph)
        transparency = self._build_transparency(
            face_output=face_output,
            vlm_results=vlm_results,
            sequences=sequences,
            event_candidates=event_candidates,
            relationship_hypotheses=relationship_hypotheses,
            profile_evidence=profile_evidence,
            storage=storage,
            change_log=change_log,
            cached_photo_ids=cached_photo_ids,
            focus_graph=focus_graph,
        )
        evaluation = self._build_evaluation(
            scope=scope,
            profile_evidence=profile_evidence,
            relationships=relationship_hypotheses,
            events=event_candidates,
        )

        memory_payload = {
            "summary": {
                "ingestion_id": scope.ingestion_id,
                "profile_version": profile_version,
                "photo_count": len(photos_dto),
                "person_count": len(person_uuid_map),
                "burst_count": len(sequences.bursts),
                "session_count": len(sequences.sessions),
                "movement_count": len(sequences.movements),
                "timeline_count": len(sequences.day_timelines),
                "event_candidate_count": len(event_candidates),
                "relationship_count": len(relationship_hypotheses),
                "mood_hypothesis_count": len(mood_hypotheses),
                "period_hypothesis_count": len(period_hypotheses),
                "concept_count": len(concept_catalog),
                "profile_field_count": len(storage["redis"]["profile_core"]["fields"]),
                "segment_count": len(storage["milvus"]["segments"]),
                "generated_at": generated_at,
            },
            "envelope": self._serialize(envelope),
            "storage": storage,
            "transparency": transparency,
            "evaluation": evaluation,
            "query_capabilities": {
                "intents": ["event_search", "relationship_explore", "relationship_rank_query", "mood_lookup"],
                "entrypoint": "memory_module.query.MemoryQueryService",
                "answer_type": "AnswerDTO",
            },
        }
        external_publish = MemoryStoragePublisher(task_dir=self.task_dir).publish(storage, user_id=self.user_id)
        memory_payload["external_publish"] = external_publish
        memory_payload["summary"]["external_sinks_published"] = sum(
            1
            for sink_name in ("redis", "neo4j", "milvus")
            if external_publish.get(sink_name, {}).get("status") == "published"
        )
        artifact_paths = self._save_outputs(memory_payload)
        memory_payload["artifacts"] = artifact_paths
        return memory_payload

    def _build_photo_facts(
        self,
        photos: Sequence,
        photo_uuid_map: Dict[str, str],
        person_uuid_map: Dict[str, str],
        vlm_by_photo_id: Dict[str, Dict[str, Any]],
    ) -> List[PhotoFactDTO]:
        photo_facts: List[PhotoFactDTO] = []
        for photo in sorted(photos, key=lambda item: item.timestamp):
            face_dtos = []
            for face in photo.faces:
                face_id = str(face.get("face_id") or self._stable_uuid("face-id", photo.photo_id, str(face.get("person_id") or "")))
                face_person_id = str(face.get("person_id") or "unknown")
                face_dtos.append(
                    FaceObservationDTO(
                        face_id=face_id,
                        face_uuid=self._stable_uuid("face", photo.photo_id, face_id),
                        face_person_id=face_person_id,
                        person_uuid=person_uuid_map.get(face_person_id, self._canonical_person_uuid(face_person_id)),
                        upstream_ref={"object_type": "face", "object_id": face_id},
                        bbox_xywh=dict(face.get("bbox_xywh") or self._bbox_to_xywh(face.get("bbox"))),
                        confidence=float(face.get("score") or 0.0),
                        similarity=float(face.get("similarity") or 0.0),
                        quality_score=float(face.get("quality_score") or 0.0),
                        quality_flags=list(face.get("quality_flags") or []),
                        pose_bucket=face.get("pose_bucket"),
                        pose_yaw=face.get("pose_yaw"),
                        pose_pitch=face.get("pose_pitch"),
                        pose_roll=face.get("pose_roll"),
                        landmark_detected=bool(face.get("landmark_detected", False)),
                        landmark_source=face.get("landmark_source"),
                    )
                )

            raw_vlm = vlm_by_photo_id.get(photo.photo_id, {}).get("vlm_analysis")
            if not raw_vlm and isinstance(photo.vlm_analysis, dict):
                raw_vlm = photo.vlm_analysis
            vlm_observation = self._build_vlm_observation(raw_vlm)
            people_observations = self._build_people_observations(
                photo=photo,
                raw_vlm=raw_vlm,
                person_uuid_map=person_uuid_map,
            )
            photo_facts.append(
                PhotoFactDTO(
                    photo_id=photo.photo_id,
                    photo_uuid=photo_uuid_map[photo.photo_id],
                    upstream_ref={"object_type": "photo", "object_id": photo.photo_id},
                    filename=photo.filename,
                    source_hash=str(photo.source_hash or ""),
                    captured_at_original=photo.timestamp.isoformat(),
                    captured_at_utc=photo.timestamp.isoformat(),
                    timezone_guess="local",
                    time_confidence=1.0,
                    location=dict(photo.location or {}),
                    primary_face_person_id=photo.primary_person_id,
                    faces=face_dtos,
                    vlm_observation=vlm_observation,
                    people_observations=people_observations,
                    artifact_refs=self._build_photo_artifacts(photo),
                )
            )
        return photo_facts

    def _build_vlm_observation(self, raw_vlm: Optional[Dict[str, Any]]) -> Optional[VLMPhotoObservationDTO]:
        if not isinstance(raw_vlm, dict):
            return None
        scene = raw_vlm.get("scene")
        event = raw_vlm.get("event")
        details = raw_vlm.get("details")
        key_objects = raw_vlm.get("key_objects")
        return VLMPhotoObservationDTO(
            summary=str(raw_vlm.get("summary") or ""),
            scene=scene if isinstance(scene, dict) else {},
            event=event if isinstance(event, dict) else {},
            details=[str(item) for item in details] if isinstance(details, list) else [],
            key_objects=[str(item) for item in key_objects] if isinstance(key_objects, list) else [],
            raw=raw_vlm,
        )

    def _build_people_observations(
        self,
        photo,
        raw_vlm: Optional[Dict[str, Any]],
        person_uuid_map: Dict[str, str],
    ) -> List[PersonObservationDTO]:
        if not isinstance(raw_vlm, dict):
            return []

        people = raw_vlm.get("people")
        if not isinstance(people, list):
            return []

        face_by_person_id = {}
        for face in photo.faces:
            person_id = face.get("person_id")
            if person_id and person_id not in face_by_person_id:
                face_by_person_id[person_id] = face

        observations: List[PersonObservationDTO] = []
        for index, person in enumerate(people, start=1):
            if not isinstance(person, dict):
                continue
            face_person_id = person.get("person_id")
            matched_face = face_by_person_id.get(face_person_id)
            observations.append(
                PersonObservationDTO(
                    observation_id=f"{photo.photo_id}_person_{index:02d}",
                    upstream_ref={"object_type": "photo_person_observation", "object_id": f"{photo.photo_id}_{index:02d}"},
                    face_id=matched_face.get("face_id") if matched_face else None,
                    face_person_id=face_person_id,
                    person_uuid=person_uuid_map.get(face_person_id) if face_person_id else None,
                    appearance=str(person.get("appearance") or ""),
                    clothing=str(person.get("clothing") or ""),
                    activity=str(person.get("activity") or person.get("interaction") or ""),
                    interaction=str(person.get("interaction") or ""),
                    expression=str(person.get("expression") or ""),
                    confidence=0.8 if face_person_id else 0.4,
                )
            )
        return observations

    def _build_event_candidates(
        self,
        events: Sequence[Event],
        sessions: Sequence,
        photo_uuid_map: Dict[str, str],
        person_uuid_map: Dict[str, str],
    ) -> List[EventCandidateDTO]:
        candidates: List[EventCandidateDTO] = []
        for index, event in enumerate(events, start=1):
            started_at, ended_at = self._event_bounds(event)
            matched_sessions = [
                session
                for session in sessions
                if self._time_windows_overlap(started_at, ended_at, session.started_at, session.ended_at, event.date)
            ]
            if not matched_sessions and sessions and event.date:
                matched_sessions = [session for session in sessions if session.day_key == event.date]
            photo_ids = self._unique(event.evidence_photos or [photo_id for session in matched_sessions for photo_id in session.photo_ids])
            participant_face_person_ids = [participant for participant in event.participants if participant in person_uuid_map]
            evidence_refs = [{"ref_type": "session", "ref_id": session.session_id} for session in matched_sessions]
            evidence_refs.extend({"ref_type": "photo", "ref_id": photo_id} for photo_id in photo_ids)
            candidates.append(
                EventCandidateDTO(
                    event_id=event.event_id or f"event_{index:03d}",
                    event_uuid=self._stable_uuid("event", event.event_id or f"event_{index:03d}"),
                    upstream_ref={"object_type": "event_candidate", "object_id": event.event_id or f"event_{index:03d}"},
                    title=event.title,
                    event_type=event.type,
                    time_range=event.time_range,
                    started_at=started_at,
                    ended_at=ended_at,
                    location=event.location,
                    participant_face_person_ids=participant_face_person_ids,
                    participant_person_uuids=[person_uuid_map[person_id] for person_id in participant_face_person_ids],
                    photo_ids=photo_ids,
                    photo_uuids=[photo_uuid_map[photo_id] for photo_id in photo_ids if photo_id in photo_uuid_map],
                    session_ids=[session.session_id for session in matched_sessions],
                    session_uuids=[session.session_uuid for session in matched_sessions],
                    description=event.description,
                    narrative_synthesis=event.narrative_synthesis,
                    tags=list(event.tags or []),
                    persona_evidence=dict(event.persona_evidence or {}),
                    confidence=float(event.confidence or 0.0),
                    evidence_refs=evidence_refs,
                )
            )
        return candidates

    def _refine_event_candidates(
        self,
        event_candidates: Sequence[EventCandidateDTO],
        photos: Sequence[PhotoFactDTO],
        sessions: Sequence[SessionDTO],
    ) -> None:
        photo_by_id = {photo.photo_id: photo for photo in photos}
        session_by_id = {session.session_id: session for session in sessions}
        for event in event_candidates:
            normalized_event_type, event_subtype, refined_title, refined_tags = self._infer_event_normalization(
                event,
                photo_by_id=photo_by_id,
                session_by_id=session_by_id,
            )
            event.event_type = normalized_event_type
            if refined_title:
                event.title = refined_title
            if refined_tags:
                event.tags = self._unique([*event.tags, *refined_tags])

    def _merge_event_candidates(
        self,
        event_candidates: Sequence[EventCandidateDTO],
    ) -> List[EventCandidateDTO]:
        merged: List[EventCandidateDTO] = []
        for event in sorted(event_candidates, key=lambda item: (item.started_at, item.event_uuid)):
            matched = None
            for existing in merged:
                if self._should_merge_events(existing, event):
                    matched = existing
                    break
            if matched is None:
                merged.append(event)
                continue
            self._merge_event_into_target(matched, event)
        return merged

    def _should_merge_events(
        self,
        left: EventCandidateDTO,
        right: EventCandidateDTO,
    ) -> bool:
        left_sessions = set(left.session_uuids or left.session_ids)
        right_sessions = set(right.session_uuids or right.session_ids)
        if not left_sessions or not right_sessions:
            return False
        if not left_sessions.intersection(right_sessions):
            return False
        live_music_types = {"concert", "music_live_event", "music_festival_performance"}
        left_type = normalize_concept(left.event_type, preferred_type="event") or left.event_type
        right_type = normalize_concept(right.event_type, preferred_type="event") or right.event_type
        if left_type not in live_music_types or right_type not in live_music_types:
            return False
        return True

    def _merge_event_into_target(
        self,
        target: EventCandidateDTO,
        incoming: EventCandidateDTO,
    ) -> None:
        target.started_at = min(filter(None, [target.started_at, incoming.started_at]), default=target.started_at)
        target.ended_at = max(filter(None, [target.ended_at, incoming.ended_at]), default=target.ended_at)
        target.time_range = self._format_event_time_range(target.started_at, target.ended_at)
        target.confidence = max(float(target.confidence or 0.0), float(incoming.confidence or 0.0))
        target.event_type = self._preferred_event_type(target.event_type, incoming.event_type)
        target.title = self._preferred_event_title(target.title, incoming.title, event_type=target.event_type)
        target.location = self._preferred_text(target.location, incoming.location)
        target.description = self._join_event_text(target.description, incoming.description)
        target.narrative_synthesis = self._join_event_text(target.narrative_synthesis, incoming.narrative_synthesis)
        target.participant_face_person_ids = self._unique([*target.participant_face_person_ids, *incoming.participant_face_person_ids])
        target.participant_person_uuids = self._unique([*target.participant_person_uuids, *incoming.participant_person_uuids])
        target.photo_ids = self._unique([*target.photo_ids, *incoming.photo_ids])
        target.photo_uuids = self._unique([*target.photo_uuids, *incoming.photo_uuids])
        target.session_ids = self._unique([*target.session_ids, *incoming.session_ids])
        target.session_uuids = self._unique([*target.session_uuids, *incoming.session_uuids])
        target.tags = self._unique([*target.tags, *incoming.tags])
        target.evidence_refs = self._unique_refs([*target.evidence_refs, *incoming.evidence_refs])
        target.persona_evidence = self._merge_persona_evidence(target.persona_evidence, incoming.persona_evidence)
        target.upstream_ref = {
            "object_type": "merged_event_candidate",
            "object_id": target.event_id,
        }

    def _infer_event_normalization(
        self,
        event: EventCandidateDTO,
        *,
        photo_by_id: Dict[str, PhotoFactDTO],
        session_by_id: Dict[str, SessionDTO],
    ) -> tuple[str, str, str, List[str]]:
        signal_texts = self._event_signal_texts(event, photo_by_id=photo_by_id, session_by_id=session_by_id)
        signal_blob = " ".join(signal_texts)
        normalized_blob = signal_blob.lower()
        matched_concepts = collect_concepts(signal_texts, preferred_type="event") or collect_concepts(signal_texts)

        has_stage = any(token in signal_blob for token in ("舞台", "钢结构", "灯光", "stage"))
        has_poster = any(token in signal_blob for token in ("海报", "宣传海报", "poster"))
        has_music_keywords = any(
            token in signal_blob
            for token in ("演唱会", "音乐会", "音乐节", "现场演出", "live", "concert", "festival", "show")
        )
        has_artist_signal = any(
            token in normalized_blob
            for token in ("rapeter", "next", "rapper", "band", "dj")
        )
        artist_hint = self._extract_artist_hint(signal_texts)

        normalized_event_type = normalize_concept(event.event_type, preferred_type="event") or ""
        event_subtype = event.event_type or ""
        refined_tags: List[str] = []

        if "音乐节" in signal_blob or (has_stage and has_poster and (artist_hint or has_artist_signal)):
            normalized_event_type = "music_festival_performance"
            event_subtype = "festival_poster"
        elif "music_festival_performance" in matched_concepts:
            normalized_event_type = "music_festival_performance"
            event_subtype = "music_festival_performance"
        elif "concert" in matched_concepts:
            normalized_event_type = "concert"
            event_subtype = "concert"
        elif "music_live_event" in matched_concepts:
            normalized_event_type = "music_live_event"
            event_subtype = "music_live_event"

        if not normalized_event_type and (has_music_keywords or has_stage):
            normalized_event_type = "music_festival_performance" if (has_poster and (artist_hint or has_artist_signal)) else "concert"
            event_subtype = "festival_poster" if normalized_event_type == "music_festival_performance" else "concert_scene"

        if normalized_event_type == "music_festival_performance":
            refined_tags.extend(["music_festival_performance", "concert"])
        elif normalized_event_type == "concert":
            refined_tags.append("concert")

        current_title = (event.title or "").strip()
        generic_title = self._is_generic_event_title(current_title)
        refined_title = current_title
        if normalized_event_type == "music_festival_performance":
            if artist_hint:
                refined_title = f"{artist_hint}相关演出活动"
            elif generic_title or not current_title:
                refined_title = "相关演出活动记录"
        elif normalized_event_type == "concert" and (generic_title or not current_title):
            refined_title = f"{artist_hint}相关演出活动" if artist_hint else "相关演出活动记录"
        elif normalized_event_type == "music_live_event" and (generic_title or not current_title):
            refined_title = f"{artist_hint}相关演出活动" if artist_hint else "相关演出活动记录"

        if not normalized_event_type:
            normalized_event_type = event.event_type or "其他"
        if not event_subtype:
            event_subtype = normalized_event_type
        return normalized_event_type, event_subtype, refined_title, refined_tags

    def _event_signal_texts(
        self,
        event: EventCandidateDTO,
        *,
        photo_by_id: Dict[str, PhotoFactDTO],
        session_by_id: Dict[str, SessionDTO],
    ) -> List[str]:
        texts: List[str] = [
            event.event_type,
            event.title,
            event.location,
            event.description,
            event.narrative_synthesis,
            *event.tags,
        ]

        for session_id in event.session_ids:
            session = session_by_id.get(session_id)
            if not session:
                continue
            texts.extend(
                [
                    str((session.location_hint or {}).get("name") or ""),
                    session.summary_hint,
                    *session.activity_hints,
                ]
            )

        for photo_id in event.photo_ids:
            photo = photo_by_id.get(photo_id)
            if not photo or not photo.vlm_observation:
                continue
            observation = photo.vlm_observation
            scene = observation.scene or {}
            event_meta = observation.event or {}
            texts.extend(
                [
                    observation.summary,
                    str(scene.get("environment_description") or ""),
                    str(scene.get("location_detected") or ""),
                    str(scene.get("weather") or ""),
                    str(event_meta.get("activity") or ""),
                    str(event_meta.get("social_context") or ""),
                    str(event_meta.get("interaction") or ""),
                    str(event_meta.get("mood") or ""),
                    *[str(item) for item in observation.details],
                    *[str(item) for item in observation.key_objects],
                ]
            )

        return [text.strip() for text in texts if str(text or "").strip()]

    def _extract_artist_hint(self, signal_texts: Sequence[str]) -> str:
        combined = " ".join(signal_texts)
        for pattern in (
            r"[A-Za-z]+\s*([一-龥]{2,4})",
            r"[‘'\"“]([一-龥]{2,4})[”'\"’]",
        ):
            match = re.search(pattern, combined)
            if match:
                candidate = match.group(1).strip()
                if candidate and candidate not in {"拍摄者", "宣传海报", "户外舞台", "音乐活动"}:
                    return candidate
        for candidate in re.findall(r"[一-龥]{2,4}", combined):
            if candidate in {"拍摄者", "宣传海报", "户外舞台", "夜间街道", "室内咖啡", "同伴密友"}:
                continue
            if candidate in combined and any(prefix in combined for prefix in (candidate, f"Rapeter{candidate}", f"NEXT{candidate}")):
                return candidate
        return ""

    def _is_generic_event_title(self, title: str) -> bool:
        normalized = str(title or "").strip()
        if not normalized:
            return True
        generic_tokens = ("活动记录", "聚会", "街头", "日常", "其他", "复合场景")
        return any(token in normalized for token in generic_tokens)

    def _preferred_event_type(self, left: str, right: str) -> str:
        rank = {
            "music_festival_performance": 3,
            "concert": 2,
            "music_live_event": 1,
        }
        left_norm = normalize_concept(left, preferred_type="event") or left
        right_norm = normalize_concept(right, preferred_type="event") or right
        return left if rank.get(left_norm, 0) >= rank.get(right_norm, 0) else right

    def _preferred_event_title(self, left: str, right: str, *, event_type: str) -> str:
        left = str(left or "").strip()
        right = str(right or "").strip()
        preferred_generic = "相关演出活动记录" if normalize_concept(event_type, preferred_type="event") in {
            "concert",
            "music_live_event",
            "music_festival_performance",
        } else ""
        for candidate in (left, right):
            if candidate and "相关演出活动" in candidate:
                return candidate
        if left and not self._is_generic_event_title(left):
            return left
        if right and not self._is_generic_event_title(right):
            return right
        return left or right or preferred_generic

    def _preferred_text(self, left: str, right: str) -> str:
        left = str(left or "").strip()
        right = str(right or "").strip()
        if left and left.lower() != "unknown":
            return left
        return right

    def _join_event_text(self, left: str, right: str) -> str:
        parts = self._unique([str(left or "").strip(), str(right or "").strip()])
        return "；".join(part for part in parts if part)

    def _merge_persona_evidence(self, left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        for key in set(left or {}).union(right or {}):
            left_value = (left or {}).get(key)
            right_value = (right or {}).get(key)
            if isinstance(left_value, list) or isinstance(right_value, list):
                merged[key] = self._unique([*(left_value or []), *(right_value or [])])
            elif left_value is not None:
                merged[key] = left_value
            else:
                merged[key] = right_value
        return merged

    def _format_event_time_range(self, started_at: str, ended_at: str) -> str:
        try:
            start_dt = datetime.fromisoformat(started_at)
            end_dt = datetime.fromisoformat(ended_at)
        except ValueError:
            return ""
        return f"{start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')}"

    def _build_relationships(
        self,
        relationships: Sequence[Relationship],
        person_uuid_map: Dict[str, str],
        sequences: SequenceBundle,
        event_candidates: Sequence[EventCandidateDTO],
        primary_person_uuid: Optional[str],
        primary_face_person_id: Optional[str],
        prior_storage: Optional[Dict[str, Any]],
    ) -> List[RelationshipHypothesisDTO]:
        prior_revisions = self._prior_relationship_revisions(prior_storage)
        hypotheses: List[RelationshipHypothesisDTO] = []
        for index, relationship in enumerate(relationships, start=1):
            target_person_uuid = person_uuid_map.get(relationship.person_id, self._canonical_person_uuid(relationship.person_id))
            evidence = dict(relationship.evidence or {})
            evidence_refs = []
            for sample in evidence.get("sample_scenes", []):
                timestamp = sample.get("timestamp")
                if timestamp:
                    evidence_refs.append({"ref_type": "relationship_scene", "ref_id": str(timestamp)})
            feature_snapshot = self._relationship_feature_snapshot(
                target_person_uuid=target_person_uuid,
                sequences=sequences,
                event_candidates=event_candidates,
                evidence=evidence,
            )
            score_snapshot = self._relationship_score_snapshot(
                relationship=relationship,
                feature_snapshot=feature_snapshot,
            )
            relationship_key = self._stable_uuid(
                "relationship-key",
                primary_person_uuid or self.user_id,
                target_person_uuid,
            )
            prior_revision = prior_revisions.get(relationship_key)
            inherited_metrics = self._relationship_inherited_metrics(prior_revision, feature_snapshot)
            revision = int(prior_revision.get("revision") or 0) + 1 if prior_revision else 1
            status = self._relationship_status(
                relationship_type=relationship.relationship_type,
                score_snapshot=score_snapshot,
                prior_revision=prior_revision,
                feature_snapshot=feature_snapshot,
            )
            window_start = feature_snapshot.get("window_start") or ""
            window_end = feature_snapshot.get("window_end") or ""
            relationship_uuid = self._stable_uuid(
                "relationship-revision",
                relationship_key,
                str(revision),
                relationship.relationship_type,
            )
            hypotheses.append(
                RelationshipHypothesisDTO(
                    relationship_uuid=relationship_uuid,
                    relationship_key=relationship_key,
                    upstream_ref={"object_type": "relationship_hypothesis", "object_id": relationship.person_id},
                    anchor_person_uuid=primary_person_uuid,
                    target_person_uuid=target_person_uuid,
                    target_face_person_id=relationship.person_id,
                    relationship_type=relationship.relationship_type,
                    label=relationship.label,
                    confidence=float(relationship.confidence or 0.0),
                    revision=revision,
                    status=status,
                    window_start=window_start,
                    window_end=window_end,
                    model_version=self.pipeline_version or "vnext",
                    reason_summary=relationship.reason,
                    feature_snapshot=feature_snapshot,
                    score_snapshot=score_snapshot,
                    inherited_metrics=inherited_metrics,
                    evidence=evidence,
                    evidence_refs=evidence_refs,
                    prior_revision_uuid=prior_revision.get("relationship_uuid") if prior_revision else None,
                )
            )
        return hypotheses

    def _build_profile_evidence(
        self,
        events: Sequence[Event],
        event_candidates: Sequence[EventCandidateDTO],
        relationship_hypotheses: Sequence[RelationshipHypothesisDTO],
    ) -> List[ProfileEvidenceDTO]:
        event_by_id = {candidate.event_id: candidate for candidate in event_candidates}
        evidence_items: List[ProfileEvidenceDTO] = []

        for event in events:
            candidate = event_by_id.get(event.event_id)
            supporting_event_ids = [event.event_id] if candidate else []
            supporting_event_uuids = [candidate.event_uuid] if candidate else []
            for tag in event.tags or []:
                evidence_items.append(
                    self._profile_evidence_item(
                        field_key="interests",
                        field_value=str(tag).lstrip("#"),
                        category="interest",
                        confidence=float(event.confidence or 0.0),
                        supporting_event_ids=supporting_event_ids,
                        supporting_event_uuids=supporting_event_uuids,
                        evidence_refs=candidate.evidence_refs if candidate else [],
                    )
                )

            persona_evidence = event.persona_evidence or {}
            for field_key, category in (
                ("behavioral", "behavioral_trait"),
                ("aesthetic", "aesthetic_trait"),
                ("socioeconomic", "socioeconomic_signal"),
            ):
                for value in persona_evidence.get(field_key, []) or []:
                    evidence_items.append(
                        self._profile_evidence_item(
                            field_key=field_key,
                            field_value=str(value),
                            category=category,
                            confidence=float(event.confidence or 0.0),
                            supporting_event_ids=supporting_event_ids,
                            supporting_event_uuids=supporting_event_uuids,
                            evidence_refs=candidate.evidence_refs if candidate else [],
                        )
                    )

            if event.title:
                evidence_items.append(
                    self._profile_evidence_item(
                        field_key="recent_focus",
                        field_value=event.title,
                        category="recent_focus",
                        confidence=float(event.confidence or 0.0),
                        supporting_event_ids=supporting_event_ids,
                        supporting_event_uuids=supporting_event_uuids,
                        evidence_refs=candidate.evidence_refs if candidate else [],
                    )
                )

        for relationship in relationship_hypotheses:
            evidence_items.append(
                self._profile_evidence_item(
                    field_key="social_graph",
                    field_value=f"{relationship.target_face_person_id}:{relationship.label}",
                    category="relationship",
                    confidence=relationship.confidence,
                    supporting_person_uuids=[relationship.target_person_uuid],
                    evidence_refs=relationship.evidence_refs,
                )
            )

        return evidence_items

    def _profile_evidence_item(
        self,
        field_key: str,
        field_value: str,
        category: str,
        confidence: float,
        supporting_event_ids: Optional[List[str]] = None,
        supporting_event_uuids: Optional[List[str]] = None,
        supporting_person_uuids: Optional[List[str]] = None,
        supporting_session_uuids: Optional[List[str]] = None,
        evidence_refs: Optional[List[Dict[str, str]]] = None,
    ) -> ProfileEvidenceDTO:
        evidence_id = self._stable_uuid("profile-evidence", field_key, field_value)
        return ProfileEvidenceDTO(
            evidence_id=evidence_id,
            upstream_ref={"object_type": "profile_evidence", "object_id": evidence_id},
            field_key=field_key,
            field_value=field_value,
            category=category,
            confidence=confidence,
            supporting_event_ids=supporting_event_ids or [],
            supporting_event_uuids=supporting_event_uuids or [],
            supporting_person_uuids=supporting_person_uuids or [],
            supporting_session_uuids=supporting_session_uuids or [],
            evidence_refs=evidence_refs or [],
        )

    def _build_change_log(
        self,
        photos_dto: Sequence[PhotoFactDTO],
        sequences: SequenceBundle,
        event_candidates: Sequence[EventCandidateDTO],
        relationship_hypotheses: Sequence[RelationshipHypothesisDTO],
        profile_evidence: Sequence[ProfileEvidenceDTO],
    ) -> List[ChangeLogEntryDTO]:
        changes: List[ChangeLogEntryDTO] = []

        def append_change(object_type: str, object_id: str, summary: str, metadata: Optional[Dict[str, Any]] = None) -> None:
            changes.append(
                ChangeLogEntryDTO(
                    change_id=self._stable_uuid("change", object_type, object_id),
                    object_type=object_type,
                    object_id=object_id,
                    change_type="upsert",
                    summary=summary,
                    upstream_ref={"object_type": object_type, "object_id": object_id},
                    metadata=metadata or {},
                )
            )

        for photo in photos_dto:
            append_change("photo", photo.photo_id, f"ingested {photo.filename}", {"face_count": len(photo.faces)})
        for burst in sequences.bursts:
            append_change("burst", burst.burst_id, f"grouped {len(burst.photo_ids)} photos into burst")
        for session in sequences.sessions:
            append_change("session", session.session_id, f"materialized session with {len(session.photo_ids)} photos")
        for movement in sequences.movements:
            append_change("movement", movement.movement_id, f"linked {movement.from_session_id} -> {movement.to_session_id}")
        for timeline in sequences.day_timelines:
            append_change("day_timeline", timeline.timeline_id, f"built timeline for {timeline.day_key}")
        for event in event_candidates:
            append_change("event_candidate", event.event_id, f"generated event {event.title}", {"confidence": event.confidence})
        for relationship in relationship_hypotheses:
            append_change(
                "relationship_hypothesis",
                relationship.relationship_uuid,
                f"inferred {relationship.label} rev {relationship.revision}",
                {"confidence": relationship.confidence, "status": relationship.status},
            )
        grouped_profile_fields = Counter(item.field_key for item in profile_evidence)
        for field_key, count in grouped_profile_fields.items():
            append_change("profile_field", field_key, f"published {count} evidence items for {field_key}")
        return changes

    def _load_prior_storage(self) -> Optional[Dict[str, Any]]:
        path = self.output_dir / "memory_storage.json"
        if not path.exists():
            return None
        try:
            import json

            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _prior_relationship_revisions(self, prior_storage: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        if not prior_storage:
            return {}
        active_revisions: Dict[str, Dict[str, Any]] = {}
        nodes = prior_storage.get("neo4j", {}).get("nodes", {}).get("relationship_hypotheses", [])
        for record in nodes:
            props = record.get("properties", {})
            relationship_key = props.get("relationship_key")
            if not relationship_key:
                continue
            if props.get("status") != "active":
                continue
            active_revisions[str(relationship_key)] = {
                "relationship_uuid": record.get("relationship_uuid"),
                "revision": int(props.get("revision") or 0),
                "status": props.get("status"),
                "relationship_type": props.get("relationship_type"),
                "label": props.get("label"),
                "confidence": float(props.get("confidence") or 0.0),
                "feature_snapshot": props.get("feature_snapshot", {}),
                "score_snapshot": props.get("score_snapshot", {}),
                "window_start": props.get("window_start"),
                "window_end": props.get("window_end"),
            }
        return active_revisions

    def _build_place_anchors(self, sessions: Sequence) -> List[PlaceAnchorDTO]:
        anchors: List[PlaceAnchorDTO] = []
        seen = set()
        for session in sessions:
            location_hint = dict(session.location_hint or {})
            canonical_name = str(location_hint.get("name") or "unknown").strip() or "unknown"
            lat = location_hint.get("lat")
            lng = location_hint.get("lng")
            geo_hash = f"{round(float(lat), 3) if lat is not None else 'na'}:{round(float(lng), 3) if lng is not None else 'na'}"
            key = (canonical_name, geo_hash)
            if key in seen:
                continue
            seen.add(key)
            place_type = "poi" if canonical_name not in {"unknown", ""} else "unknown"
            anchors.append(
                PlaceAnchorDTO(
                    place_uuid=self._stable_uuid("place", canonical_name, geo_hash),
                    user_id=self.user_id,
                    canonical_name=canonical_name,
                    aliases=[canonical_name] if canonical_name != "unknown" else [],
                    place_type=place_type,
                    geo_hash=geo_hash,
                    lat=float(lat) if lat is not None else None,
                    lng=float(lng) if lng is not None else None,
                    source="session_location_hint",
                    confidence=0.9 if canonical_name != "unknown" else 0.4,
                )
            )
        return anchors

    def _build_concepts(
        self,
        sessions: Sequence,
        event_candidates: Sequence[EventCandidateDTO],
        relationships: Sequence[Relationship],
    ) -> List[ConceptDTO]:
        concept_names: Dict[str, ConceptDTO] = {}

        def add_concept(canonical_name: str, *, concept_type: Optional[str] = None, scope: str = "canonical") -> None:
            if canonical_name in concept_names:
                return
            meta = concept_metadata(canonical_name)
            final_type = concept_type or str(meta.get("concept_type") or "unknown")
            concept_names[canonical_name] = ConceptDTO(
                concept_uuid=self._stable_uuid("concept", canonical_name, scope),
                canonical_name=canonical_name,
                aliases=list(meta.get("aliases", [])),
                concept_type=final_type,
                scope=scope,
                status="active" if scope == "canonical" else "proposed",
                version="v1",
                user_id=None if scope == "canonical" else self.user_id,
                description=str(meta.get("description") or ""),
                parent_concepts=list(meta.get("parents", [])),
            )

        def add_candidate(raw_text: str, concept_type: str) -> None:
            candidate = suggest_candidate_concept(raw_text, concept_type=concept_type, user_id=self.user_id)
            canonical_name = str(candidate["canonical_name"])
            if canonical_name in concept_names:
                return
            concept_names[canonical_name] = ConceptDTO(
                concept_uuid=self._stable_uuid("candidate-concept", canonical_name),
                canonical_name=canonical_name,
                aliases=list(candidate["aliases"]),
                concept_type=concept_type,
                scope="candidate",
                status="proposed",
                version="v1",
                user_id=self.user_id,
            )

        for session in sessions:
            for raw_value in [*session.activity_hints, str((session.location_hint or {}).get("name") or ""), session.summary_hint]:
                if not raw_value:
                    continue
                matched = collect_concepts([str(raw_value)])
                if matched:
                    for canonical in matched:
                        add_concept(canonical)
                elif len(str(raw_value).strip()) > 2:
                    add_candidate(str(raw_value), "context")

        for event in event_candidates:
            for raw_value in [
                event.event_type,
                event.title,
                event.location,
                event.description,
                event.narrative_synthesis,
                *event.tags,
            ]:
                if not raw_value:
                    continue
                matched = collect_concepts([str(raw_value)])
                if matched:
                    for canonical in matched:
                        add_concept(canonical)
                elif len(str(raw_value).strip()) > 2:
                    add_candidate(str(raw_value), "event")

        for relationship in relationships:
            for raw_value in [relationship.relationship_type, relationship.label]:
                matched = collect_concepts([str(raw_value)], preferred_type="relationship") or collect_concepts([str(raw_value)])
                for canonical in matched:
                    add_concept(canonical)

        add_concept("recent_period")
        add_concept("happy_mood")
        add_concept("sad_mood")
        add_concept("neutral_mood")
        return list(concept_names.values())

    def _build_primary_person_hypothesis(
        self,
        sequences: SequenceBundle,
        primary_person_uuid: Optional[str],
    ) -> Optional[PrimaryPersonHypothesisDTO]:
        if not primary_person_uuid or not sequences.sessions:
            return None
        return PrimaryPersonHypothesisDTO(
            primary_person_hypothesis_uuid=self._stable_uuid("primary-person-hypothesis", primary_person_uuid),
            upstream_ref={"object_type": "primary_person_hypothesis", "object_id": primary_person_uuid},
            user_id=self.user_id,
            person_uuid=primary_person_uuid,
            confidence=0.95,
            window_start=sequences.sessions[0].started_at,
            window_end=sequences.sessions[-1].ended_at,
            model_version=self.pipeline_version or "vnext",
        )

    def _build_mood_hypotheses(self, sequences: SequenceBundle) -> List[MoodStateHypothesisDTO]:
        moods: List[MoodStateHypothesisDTO] = []
        for session in sequences.sessions:
            summary = (session.summary_hint or "").lower()
            label = "neutral_mood"
            score = 0.5
            confidence = 0.55
            if any(token in summary for token in ("happy", "warm", "joy", "开心", "愉快")):
                label = "happy_mood"
                score = 0.82
                confidence = 0.72
            elif any(token in summary for token in ("sad", "down", "tense", "难过", "冲突")):
                label = "sad_mood"
                score = 0.28
                confidence = 0.7
            moods.append(
                MoodStateHypothesisDTO(
                    mood_uuid=self._stable_uuid("mood", session.session_uuid, label),
                    upstream_ref={"object_type": "mood_state_hypothesis", "object_id": session.session_uuid},
                    session_uuid=session.session_uuid,
                    mood_label=label,
                    mood_score=score,
                    confidence=confidence,
                    window_start=session.started_at,
                    window_end=session.ended_at,
                    model_version=self.pipeline_version or "vnext",
                    reason_summary=session.summary_hint or label,
                    artifact_ref_ids=[self._stable_uuid("session-artifact", session.session_uuid)],
                    evidence_refs=[{"ref_type": "session", "ref_id": session.session_id}],
                )
            )
        return moods

    def _build_period_hypotheses(
        self,
        sequences: SequenceBundle,
        concept_catalog: Sequence[ConceptDTO],
    ) -> List[PeriodHypothesisDTO]:
        periods: List[PeriodHypothesisDTO] = []
        if sequences.sessions:
            recent_sessions = sorted(sequences.sessions, key=lambda item: item.started_at)[-5:]
            periods.append(
                PeriodHypothesisDTO(
                    period_uuid=self._stable_uuid("period", "recent"),
                    upstream_ref={"object_type": "period_hypothesis", "object_id": "recent_period"},
                    user_id=self.user_id,
                    period_type="recent_period",
                    label="最近",
                    window_start=recent_sessions[0].started_at,
                    window_end=recent_sessions[-1].ended_at,
                    confidence=0.9,
                    reason_summary="latest five sessions window",
                    artifact_ref_ids=[self._stable_uuid("period-artifact", "recent")],
                    evidence_refs=[{"ref_type": "session", "ref_id": session.session_id} for session in recent_sessions],
                )
            )

        campus_sessions = [session for session in sequences.sessions if "campus" in self._session_concepts(session)]
        if campus_sessions:
            periods.append(
                PeriodHypothesisDTO(
                    period_uuid=self._stable_uuid("period", "college"),
                    upstream_ref={"object_type": "period_hypothesis", "object_id": "college_period"},
                    user_id=self.user_id,
                    period_type="college_period",
                    label="大学时期",
                    window_start=campus_sessions[0].started_at,
                    window_end=campus_sessions[-1].ended_at,
                    confidence=0.8,
                    reason_summary="campus concept sessions",
                    artifact_ref_ids=[self._stable_uuid("period-artifact", "college")],
                    evidence_refs=[{"ref_type": "session", "ref_id": session.session_id} for session in campus_sessions],
                )
            )

        job_sessions = [session for session in sequences.sessions if "work" in self._session_concepts(session)]
        if job_sessions:
            periods.append(
                PeriodHypothesisDTO(
                    period_uuid=self._stable_uuid("period", "job"),
                    upstream_ref={"object_type": "period_hypothesis", "object_id": "job_period"},
                    user_id=self.user_id,
                    period_type="job_period",
                    label="工作时期",
                    window_start=job_sessions[0].started_at,
                    window_end=job_sessions[-1].ended_at,
                    confidence=0.75,
                    reason_summary="work concept sessions",
                    artifact_ref_ids=[self._stable_uuid("period-artifact", "job")],
                    evidence_refs=[{"ref_type": "session", "ref_id": session.session_id} for session in job_sessions],
                )
            )

        return periods

    def _relationship_feature_snapshot(
        self,
        target_person_uuid: str,
        sequences: SequenceBundle,
        event_candidates: Sequence[EventCandidateDTO],
        evidence: Dict[str, Any],
    ) -> Dict[str, Any]:
        supporting_sessions = []
        for session in sequences.sessions:
            participant_uuids = set(session.dominant_person_uuids)
            if target_person_uuid in participant_uuids:
                supporting_sessions.append(session)

        supporting_session_uuids = [session.session_uuid for session in supporting_sessions]
        supporting_event_uuids = [
            event.event_uuid
            for event in event_candidates
            if target_person_uuid in event.participant_person_uuids
        ]
        distinct_days = {session.day_key for session in supporting_sessions}
        scene_diversity = len(
            {
                normalize_concept(hint) or hint
                for session in supporting_sessions
                for hint in session.activity_hints
                if hint
            }
        )
        exclusive_sessions = [
            session
            for session in supporting_sessions
            if len(session.dominant_person_uuids) <= 2
        ]
        work_sessions = [session for session in supporting_sessions if "work" in self._session_concepts(session)]
        leisure_sessions = [session for session in supporting_sessions if "leisure" in self._session_concepts(session)]
        home_sessions = [session for session in supporting_sessions if "home" in self._session_concepts(session)]
        campus_sessions = [session for session in supporting_sessions if "campus" in self._session_concepts(session)]
        weekend_sessions = [
            session
            for session in supporting_sessions
            if datetime.fromisoformat(session.started_at).weekday() >= 5
        ]
        time_span_days = 0
        if supporting_sessions:
            first_dt = datetime.fromisoformat(supporting_sessions[0].started_at)
            last_dt = datetime.fromisoformat(supporting_sessions[-1].ended_at)
            time_span_days = max(0, (last_dt - first_dt).days)

        conflict_count = sum(1 for item in evidence.get("interaction_behavior", []) if "conflict" in str(item).lower() or "争" in str(item))
        affection_count = sum(1 for item in evidence.get("interaction_behavior", []) if "hug" in str(item).lower() or "亲" in str(item))
        photo_count = int(evidence.get("photo_count") or 0)
        return {
            "co_present_session_count": len(supporting_sessions),
            "co_present_event_count": len(supporting_event_uuids),
            "distinct_days": len(distinct_days),
            "scene_diversity": float(scene_diversity),
            "exclusive_ratio": round(len(exclusive_sessions) / len(supporting_sessions), 4) if supporting_sessions else 0.0,
            "weekend_ratio": round(len(weekend_sessions) / len(supporting_sessions), 4) if supporting_sessions else 0.0,
            "work_scene_ratio": round(len(work_sessions) / len(supporting_sessions), 4) if supporting_sessions else 0.0,
            "leisure_scene_ratio": round(len(leisure_sessions) / len(supporting_sessions), 4) if supporting_sessions else 0.0,
            "home_scene_ratio": round(len(home_sessions) / len(supporting_sessions), 4) if supporting_sessions else 0.0,
            "campus_scene_ratio": round(len(campus_sessions) / len(supporting_sessions), 4) if supporting_sessions else 0.0,
            "avg_user_mood_score": 0.5 + (0.15 if leisure_sessions else 0.0) - (0.10 if conflict_count else 0.0),
            "conflict_signal_count": conflict_count,
            "affection_signal_count": affection_count,
            "place_diversity": len({str((session.location_hint or {}).get("name") or "") for session in supporting_sessions}),
            "time_span_days": time_span_days,
            "total_photo_evidence_count": photo_count,
            "supporting_session_uuids": supporting_session_uuids,
            "supporting_event_uuids": supporting_event_uuids,
            "window_start": supporting_sessions[0].started_at if supporting_sessions else "",
            "window_end": supporting_sessions[-1].ended_at if supporting_sessions else "",
        }

    def _relationship_score_snapshot(
        self,
        relationship: Relationship,
        feature_snapshot: Dict[str, Any],
    ) -> Dict[str, float]:
        session_count = float(feature_snapshot.get("co_present_session_count") or 0.0)
        distinct_days = float(feature_snapshot.get("distinct_days") or 0.0)
        exclusive_ratio = float(feature_snapshot.get("exclusive_ratio") or 0.0)
        leisure_ratio = float(feature_snapshot.get("leisure_scene_ratio") or 0.0)
        work_ratio = float(feature_snapshot.get("work_scene_ratio") or 0.0)
        home_ratio = float(feature_snapshot.get("home_scene_ratio") or 0.0)
        affection_count = float(feature_snapshot.get("affection_signal_count") or 0.0)
        conflict_count = float(feature_snapshot.get("conflict_signal_count") or 0.0)
        base = float(relationship.confidence or 0.0)
        friend_score = min(0.99, 0.25 + 0.08 * session_count + 0.05 * distinct_days + 0.20 * leisure_ratio + 0.15 * base)
        close_friend_score = min(0.99, friend_score + 0.10 * exclusive_ratio + 0.12 * leisure_ratio + 0.03 * feature_snapshot.get("scene_diversity", 0.0))
        colleague_score = min(0.99, 0.18 + 0.12 * session_count + 0.35 * work_ratio + 0.12 * base)
        partner_score = min(0.99, 0.10 + 0.20 * session_count + 0.28 * exclusive_ratio + 0.25 * leisure_ratio + 0.10 * affection_count - 0.12 * work_ratio)
        family_score = min(0.99, 0.12 + 0.22 * home_ratio + 0.18 * base)
        if relationship.relationship_type in {"friend", "close_friend"}:
            friend_score = min(0.99, friend_score + 0.10)
            close_friend_score = min(0.99, close_friend_score + 0.08)
        if relationship.relationship_type == "colleague":
            colleague_score = min(0.99, colleague_score + 0.12)
        if relationship.relationship_type == "partner":
            partner_score = min(0.99, partner_score + 0.18)
        normalized_label = normalize_concept(relationship.label)
        if relationship.relationship_type in {"family_generic", "father", "mother"} or normalized_label in {"family_generic", "father", "mother"}:
            family_score = min(0.99, family_score + 0.35)
        if normalized_label == "father":
            family_score = min(0.99, family_score + 0.12)
        if normalized_label == "mother":
            family_score = min(0.99, family_score + 0.12)
        friend_score = max(0.0, friend_score - 0.05 * conflict_count)
        close_friend_score = max(0.0, close_friend_score - 0.05 * conflict_count)
        partner_score = max(0.0, partner_score - 0.08 * conflict_count)
        return {
            "friend": round(friend_score, 4),
            "close_friend": round(close_friend_score, 4),
            "colleague": round(colleague_score, 4),
            "partner": round(partner_score, 4),
            "family_generic": round(family_score, 4),
            "father": round(min(0.99, family_score + (0.25 if normalized_label == "father" or relationship.relationship_type == "father" else 0.0)), 4),
            "mother": round(min(0.99, family_score + (0.25 if normalized_label == "mother" or relationship.relationship_type == "mother" else 0.0)), 4),
        }

    def _relationship_inherited_metrics(
        self,
        prior_revision: Optional[Dict[str, Any]],
        feature_snapshot: Dict[str, Any],
    ) -> Dict[str, Any]:
        inherited = {
            "first_seen_at": feature_snapshot.get("window_start"),
            "last_seen_at": feature_snapshot.get("window_end"),
            "total_co_present_sessions": feature_snapshot.get("co_present_session_count", 0),
            "total_co_present_events": feature_snapshot.get("co_present_event_count", 0),
            "total_distinct_days": feature_snapshot.get("distinct_days", 0),
            "total_shared_place_count": feature_snapshot.get("place_diversity", 0),
            "recent_90d_sessions": feature_snapshot.get("co_present_session_count", 0),
            "recent_weekend_ratio": feature_snapshot.get("weekend_ratio", 0.0),
            "recent_work_scene_ratio": feature_snapshot.get("work_scene_ratio", 0.0),
            "recent_leisure_scene_ratio": feature_snapshot.get("leisure_scene_ratio", 0.0),
            "recent_home_scene_ratio": feature_snapshot.get("home_scene_ratio", 0.0),
            "recent_campus_scene_ratio": feature_snapshot.get("campus_scene_ratio", 0.0),
            "recent_exclusive_ratio": feature_snapshot.get("exclusive_ratio", 0.0),
            "recent_avg_mood_score": feature_snapshot.get("avg_user_mood_score", 0.0),
            "recent_conflict_signal_count": feature_snapshot.get("conflict_signal_count", 0),
            "recent_affection_signal_count": feature_snapshot.get("affection_signal_count", 0),
            "prior_active_label": prior_revision.get("label") if prior_revision else None,
            "prior_active_confidence": prior_revision.get("confidence") if prior_revision else None,
            "prior_label_streak": 1 if prior_revision else 0,
            "revision_count": int(prior_revision.get("revision") or 0) + 1 if prior_revision else 1,
            "historical_best_friend_score": max(
                float(prior_revision.get("score_snapshot", {}).get("friend", 0.0)) if prior_revision else 0.0,
                float(feature_snapshot.get("co_present_session_count", 0)) * 0.1,
            ),
            "historical_best_partner_score": float(prior_revision.get("score_snapshot", {}).get("partner", 0.0)) if prior_revision else 0.0,
            "historical_best_colleague_score": float(prior_revision.get("score_snapshot", {}).get("colleague", 0.0)) if prior_revision else 0.0,
            "contradiction_count": 0,
        }
        return inherited

    def _relationship_status(
        self,
        relationship_type: str,
        score_snapshot: Dict[str, float],
        prior_revision: Optional[Dict[str, Any]],
        feature_snapshot: Dict[str, Any],
    ) -> str:
        thresholds = RELATIONSHIP_LABEL_THRESHOLDS.get(relationship_type, RELATIONSHIP_LABEL_THRESHOLDS["friend"])
        score = float(score_snapshot.get(relationship_type, 0.0))
        if not prior_revision:
            return "active" if score >= thresholds["upgrade"] else "rejected"

        prior_confidence = float(prior_revision.get("confidence") or 0.0)
        margin = 0.18 if relationship_type in {"partner", "family_generic", "father", "mother"} else 0.12
        if score >= thresholds["upgrade"] and score >= prior_confidence + margin:
            return "active"
        if score < thresholds["keep"]:
            previous_status = str(prior_revision.get("status") or "")
            return "superseded" if previous_status == "cooling" else "cooling"
        return "active"

    def _session_participants(
        self,
        sessions: Sequence,
        photos: Sequence[PhotoFactDTO],
    ) -> Dict[str, List[str]]:
        participants: Dict[str, List[str]] = {}
        photo_lookup = {photo.photo_id: photo for photo in photos}
        for session in sessions:
            person_uuids: List[str] = []
            for photo_id in session.photo_ids:
                photo = photo_lookup.get(photo_id)
                if not photo:
                    continue
                for face in photo.faces:
                    if face.person_uuid not in person_uuids:
                        person_uuids.append(face.person_uuid)
            participants[session.session_uuid] = person_uuids
        return participants

    def _session_artifact_ref_ids(self, session, photos: Sequence[PhotoFactDTO]) -> List[str]:
        photo_lookup = {photo.photo_id: photo for photo in photos}
        refs = []
        for photo_id in session.photo_ids[:3]:
            photo = photo_lookup.get(photo_id)
            if not photo:
                continue
            for artifact in photo.artifact_refs[:2]:
                refs.append(self._stable_uuid("artifact-ref", artifact.path))
        return refs

    def _event_artifact_ref_ids(self, event: EventCandidateDTO, photos: Sequence[PhotoFactDTO]) -> List[str]:
        photo_lookup = {photo.photo_id: photo for photo in photos}
        refs = []
        for photo_id in event.photo_ids[:3]:
            photo = photo_lookup.get(photo_id)
            if not photo:
                continue
            for artifact in photo.artifact_refs[:2]:
                refs.append(self._stable_uuid("artifact-ref", artifact.path))
        return refs

    def _relationship_artifact_ref_ids(
        self,
        relationship: RelationshipHypothesisDTO,
        sequences: SequenceBundle,
        photos: Sequence[PhotoFactDTO],
    ) -> List[str]:
        supporting_session_uuids = set(relationship.feature_snapshot.get("supporting_session_uuids", []))
        refs = []
        for session in sequences.sessions:
            if session.session_uuid not in supporting_session_uuids:
                continue
            refs.extend(self._session_artifact_ref_ids(session, photos))
        return self._unique(refs)

    def _session_concepts(self, session) -> List[str]:
        raw_values = [*session.activity_hints, str((session.location_hint or {}).get("name") or "")]
        concepts = collect_concepts(raw_values)
        if not concepts and session.summary_hint:
            concepts = collect_concepts([session.summary_hint])
        return concepts

    def _event_concepts(self, event: EventCandidateDTO) -> List[str]:
        raw_values = [
            event.event_type,
            event.title,
            event.location,
            event.description,
            event.narrative_synthesis,
            *event.tags,
        ]
        return collect_concepts([str(raw_value) for raw_value in raw_values if raw_value])

    def _relationship_concepts(self, relationship: RelationshipHypothesisDTO) -> List[str]:
        concepts = collect_concepts(
            [relationship.relationship_type, relationship.label],
            preferred_type="relationship",
        )
        if not concepts:
            concepts = collect_concepts([relationship.relationship_type, relationship.label])
        return concepts

    def _mood_concepts(self, mood: MoodStateHypothesisDTO) -> List[str]:
        return [mood.mood_label] if mood.mood_label else []

    def _period_concepts(self, period: PeriodHypothesisDTO) -> List[str]:
        concepts = collect_concepts([period.period_type])
        if period.period_type == "college_period":
            concepts.extend(concept for concept in collect_concepts(["campus"]) if concept not in concepts)
        if period.period_type == "job_period":
            concepts.extend(concept for concept in collect_concepts(["work"]) if concept not in concepts)
        return self._unique(concepts)

    def _period_session_uuids(self, period: PeriodHypothesisDTO, sessions: Sequence) -> List[str]:
        start_dt = datetime.fromisoformat(period.window_start)
        end_dt = datetime.fromisoformat(period.window_end)
        session_uuids = []
        for session in sessions:
            session_start = datetime.fromisoformat(session.started_at)
            session_end = datetime.fromisoformat(session.ended_at)
            if session_end < start_dt or end_dt < session_start:
                continue
            session_uuids.append(session.session_uuid)
        return session_uuids

    def _embedding_props(self, search_text: str) -> Dict[str, Any]:
        return build_embedding_payload(
            search_text,
            dim=self.embedder.dim,
            model=DEFAULT_EMBEDDING_MODEL,
            version=DEFAULT_EMBEDDING_VERSION,
            embedder=self.embedder,
            task_type="document",
        )

    def _build_materialization_bundle(
        self,
        photos_dto: Sequence[PhotoFactDTO],
        event_candidates: Sequence[EventCandidateDTO],
        relationship_hypotheses: Sequence[RelationshipHypothesisDTO],
        profile_evidence: Sequence[ProfileEvidenceDTO],
    ) -> MaterializationInputBundle:
        return MaterializationInputBundle(
            user_id=self.user_id,
            facts={
                "photo_facts": [self._serialize(item) for item in photos_dto],
            },
            hypotheses={
                "event_hypotheses": [self._serialize(item) for item in event_candidates],
                "relationship_hypotheses": [self._serialize(item) for item in relationship_hypotheses],
            },
            profile_evidence_items=[
                ProfileEvidenceItem(
                    evidence_id=item.evidence_id,
                    field_key=item.field_key,
                    field_value=item.field_value,
                    confidence=item.confidence,
                    evidence_refs=item.evidence_refs,
                )
                for item in profile_evidence
            ],
        )

    def _build_storage_records(
        self,
        scope: ScopeDTO,
        photos: Sequence[PhotoFactDTO],
        sequences: SequenceBundle,
        event_candidates: Sequence[EventCandidateDTO],
        place_anchors: Sequence[PlaceAnchorDTO],
        concept_catalog: Sequence[ConceptDTO],
        relationship_hypotheses: Sequence[RelationshipHypothesisDTO],
        mood_hypotheses: Sequence[MoodStateHypothesisDTO],
        primary_person_hypothesis: Optional[PrimaryPersonHypothesisDTO],
        period_hypotheses: Sequence[PeriodHypothesisDTO],
        profile_evidence: Sequence[ProfileEvidenceDTO],
        bundle: MaterializationInputBundle,
        profile_markdown: str,
        person_uuid_map: Dict[str, str],
        primary_face_person_id: Optional[str],
        primary_person_uuid: Optional[str],
        profile_version: int,
    ) -> Dict[str, Any]:
        people_records = [
            PersonIdentityMapRecord(
                user_id=self.user_id,
                tenant_id=scope.tenant_id,
                face_person_id=face_person_id,
                person_uuid=person_uuid,
                source_system=self.source_system,
            )
            for face_person_id, person_uuid in sorted(person_uuid_map.items())
        ]
        photo_identity_records = [
            PhotoIdentityMapRecord(
                user_id=self.user_id,
                tenant_id=scope.tenant_id,
                photo_id=photo.photo_id,
                photo_uuid=photo.photo_uuid,
                source_hash=photo.source_hash,
                source_system=self.source_system,
            )
            for photo in photos
        ]
        place_uuid_by_name = {place.canonical_name: place.place_uuid for place in place_anchors}
        session_participants = self._session_participants(sequences.sessions, photos)
        concept_lookup = {concept.canonical_name: concept for concept in concept_catalog}

        neo4j_nodes = {
            "user": [
                Neo4jUserNodeRecord(
                    user_id=self.user_id,
                    tenant_id=scope.tenant_id,
                    properties={
                        "source_system": self.source_system,
                        "profile_version": profile_version,
                        "task_id": self.task_id,
                        "primary_face_person_id": primary_face_person_id,
                        "primary_person_uuid": primary_person_uuid,
                    },
                )
            ],
            "persons": [],
            "places": [],
            "sessions": [],
            "timelines": [],
            "events": [],
            "relationship_hypotheses": [],
            "mood_states": [],
            "primary_person_hypotheses": [],
            "period_hypotheses": [],
            "concepts": [],
        }

        for record in people_records:
            related_sessions = [
                session for session in sequences.sessions
                if record.person_uuid in session_participants.get(session.session_uuid, [])
            ]
            first_seen = min((session.started_at for session in related_sessions), default="")
            last_seen = max((session.ended_at for session in related_sessions), default="")
            search_text = " ".join(
                part
                for part in [
                    record.face_person_id,
                    "primary person" if record.person_uuid == primary_person_uuid else "known person",
                ]
                if part
            )
            neo4j_nodes["persons"].append(
                Neo4jPersonNodeRecord(
                    person_uuid=record.person_uuid,
                    labels=["Person"],
                    properties={
                        "user_id": self.user_id,
                        "face_person_id": record.face_person_id,
                        "display_name_hint": record.face_person_id,
                        "first_seen_at": first_seen,
                        "last_seen_at": last_seen,
                        "is_primary_candidate": record.person_uuid == primary_person_uuid,
                        **self._embedding_props(search_text),
                    },
                )
            )

        for place in place_anchors:
            search_text = " ".join(
                item for item in [place.canonical_name, " ".join(place.aliases), place.place_type] if item
            )
            neo4j_nodes["places"].append(
                Neo4jPlaceNodeRecord(
                    place_uuid=place.place_uuid,
                    properties={
                        "user_id": self.user_id,
                        "canonical_name": place.canonical_name,
                        "aliases": place.aliases,
                        "place_type": place.place_type,
                        "geo_hash": place.geo_hash,
                        "lat": place.lat,
                        "lng": place.lng,
                        "source": place.source,
                        "confidence": place.confidence,
                        **self._embedding_props(search_text),
                    },
                )
            )

        for session in sequences.sessions:
            place_name = str((session.location_hint or {}).get("name") or "unknown")
            place_uuid = place_uuid_by_name.get(place_name)
            concept_names = self._session_concepts(session)
            search_text = " ".join(
                part
                for part in [
                    place_name,
                    session.summary_hint,
                    " ".join(session.activity_hints),
                    " ".join(concept_names),
                ]
                if part
            )
            artifact_ref_ids = self._session_artifact_ref_ids(session, photos)
            neo4j_nodes["sessions"].append(
                Neo4jSessionNodeRecord(
                    session_uuid=session.session_uuid,
                    properties={
                        "user_id": self.user_id,
                        "session_id": session.session_id,
                        "day_key": session.day_key,
                        "started_at": session.started_at,
                        "ended_at": session.ended_at,
                        "duration_seconds": session.duration_seconds,
                        "place_uuid": place_uuid,
                        "participant_count": len(session_participants.get(session.session_uuid, [])),
                        "dominant_person_uuids": session.dominant_person_uuids,
                        "photo_count": len(session.photo_ids),
                        "representative_photo_ids": session.photo_ids[:3],
                        "artifact_ref_ids": artifact_ref_ids,
                        "summary_hint": session.summary_hint,
                        **self._embedding_props(search_text),
                    },
                )
            )

        for timeline in sequences.day_timelines:
            neo4j_nodes["timelines"].append(
                Neo4jTimelineNodeRecord(
                    timeline_uuid=timeline.timeline_uuid,
                    properties={
                        "user_id": self.user_id,
                        "timeline_id": timeline.timeline_id,
                        "day_key": timeline.day_key,
                        "started_at": timeline.started_at,
                        "ended_at": timeline.ended_at,
                    },
                )
            )

        for event in event_candidates:
            concept_names = self._event_concepts(event)
            place_uuid = place_uuid_by_name.get(event.location or "unknown")
            short_narrative = event.narrative_synthesis or event.description or event.title
            search_text = " ".join(
                part
                for part in [
                    event.title,
                    event.event_type,
                    event.location,
                    " ".join(event.tags),
                    " ".join(concept_names),
                    short_narrative,
                ]
                if part
            )
            neo4j_nodes["events"].append(
                Neo4jEventNodeRecord(
                    event_uuid=event.event_uuid,
                    properties={
                        "user_id": self.user_id,
                        "event_id": event.event_id,
                        "title": event.title,
                        "normalized_event_type": normalize_concept(event.event_type, preferred_type="event") or event.event_type,
                        "event_subtype": event.event_type,
                        "started_at": event.started_at,
                        "ended_at": event.ended_at,
                        "place_uuid": place_uuid,
                        "confidence": event.confidence,
                        "participant_count": len(event.participant_person_uuids),
                        "photo_count": len(event.photo_ids),
                        "representative_photo_ids": event.photo_ids[:3],
                        "artifact_ref_ids": self._event_artifact_ref_ids(event, photos),
                        "model_version": self.pipeline_version or "vnext",
                        **self._embedding_props(search_text),
                    },
                )
            )

        for relationship in relationship_hypotheses:
            concept_names = self._relationship_concepts(relationship)
            search_text = " ".join(
                part
                for part in [
                    relationship.relationship_type,
                    relationship.label,
                    relationship.reason_summary,
                    relationship.target_face_person_id,
                    " ".join(concept_names),
                ]
                if part
            )
            neo4j_nodes["relationship_hypotheses"].append(
                Neo4jRelationshipHypothesisNodeRecord(
                    relationship_uuid=relationship.relationship_uuid,
                    properties={
                        "user_id": self.user_id,
                        "relationship_key": relationship.relationship_key,
                        "revision": relationship.revision,
                        "status": relationship.status,
                        "anchor_person_uuid": relationship.anchor_person_uuid,
                        "target_person_uuid": relationship.target_person_uuid,
                        "target_face_person_id": relationship.target_face_person_id,
                        "relationship_type": relationship.relationship_type,
                        "label": relationship.label,
                        "confidence": relationship.confidence,
                        "window_start": relationship.window_start,
                        "window_end": relationship.window_end,
                        "model_version": relationship.model_version,
                        "reason_summary": relationship.reason_summary,
                        "feature_snapshot": relationship.feature_snapshot,
                        "score_snapshot": relationship.score_snapshot,
                        "inherited_metrics": relationship.inherited_metrics,
                        "prior_revision_uuid": relationship.prior_revision_uuid,
                        "artifact_ref_ids": self._relationship_artifact_ref_ids(relationship, sequences, photos),
                        **self._embedding_props(search_text),
                    },
                )
            )

        for mood in mood_hypotheses:
            neo4j_nodes["mood_states"].append(
                Neo4jMoodStateNodeRecord(
                    mood_uuid=mood.mood_uuid,
                    properties={
                        "user_id": self.user_id,
                        "session_uuid": mood.session_uuid,
                        "event_uuid": mood.event_uuid,
                        "mood_label": mood.mood_label,
                        "mood_score": mood.mood_score,
                        "confidence": mood.confidence,
                        "window_start": mood.window_start,
                        "window_end": mood.window_end,
                        "model_version": mood.model_version,
                        "reason_summary": mood.reason_summary,
                        "artifact_ref_ids": mood.artifact_ref_ids,
                    },
                )
            )

        if primary_person_hypothesis:
            neo4j_nodes["primary_person_hypotheses"].append(
                Neo4jPrimaryPersonHypothesisNodeRecord(
                    primary_person_hypothesis_uuid=primary_person_hypothesis.primary_person_hypothesis_uuid,
                    properties={
                        "user_id": self.user_id,
                        "person_uuid": primary_person_hypothesis.person_uuid,
                        "confidence": primary_person_hypothesis.confidence,
                        "window_start": primary_person_hypothesis.window_start,
                        "window_end": primary_person_hypothesis.window_end,
                        "model_version": primary_person_hypothesis.model_version,
                    },
                )
            )

        for period in period_hypotheses:
            neo4j_nodes["period_hypotheses"].append(
                Neo4jPeriodHypothesisNodeRecord(
                    period_uuid=period.period_uuid,
                    properties={
                        "user_id": self.user_id,
                        "period_type": period.period_type,
                        "label": period.label,
                        "window_start": period.window_start,
                        "window_end": period.window_end,
                        "confidence": period.confidence,
                        "reason_summary": period.reason_summary,
                        "artifact_ref_ids": period.artifact_ref_ids,
                    },
                )
            )

        for concept in concept_catalog:
            meta = concept_metadata(concept.canonical_name)
            search_text = " ".join(
                part
                for part in [
                    concept.canonical_name,
                    " ".join(concept.aliases),
                    str(meta.get("description") or concept.description or ""),
                    " ".join(meta.get("parents", []) or concept.parent_concepts),
                ]
                if part
            )
            neo4j_nodes["concepts"].append(
                Neo4jConceptNodeRecord(
                    concept_uuid=concept.concept_uuid,
                    properties={
                        "user_id": concept.user_id,
                        "canonical_name": concept.canonical_name,
                        "aliases": concept.aliases,
                        "concept_type": concept.concept_type,
                        "scope": concept.scope,
                        "status": concept.status,
                        "version": concept.version,
                        "description": concept.description or meta.get("description") or "",
                        "parents": concept.parent_concepts or meta.get("parents", []),
                        **self._embedding_props(search_text),
                    },
                )
            )

        neo4j_edges: List[Neo4jRelationshipEdgeRecord] = []
        if primary_person_hypothesis:
            neo4j_edges.append(
                Neo4jRelationshipEdgeRecord(
                    edge_id=self._stable_uuid("edge", "user-primary-hypothesis", self.user_id, primary_person_hypothesis.primary_person_hypothesis_uuid),
                    from_id=self.user_id,
                    to_id=primary_person_hypothesis.primary_person_hypothesis_uuid,
                    edge_type="PRIMARY_PERSON_HYPOTHESIS",
                )
            )
            neo4j_edges.append(
                Neo4jRelationshipEdgeRecord(
                    edge_id=self._stable_uuid("edge", "primary-hypothesis-person", primary_person_hypothesis.primary_person_hypothesis_uuid, primary_person_hypothesis.person_uuid),
                    from_id=primary_person_hypothesis.primary_person_hypothesis_uuid,
                    to_id=primary_person_hypothesis.person_uuid,
                    edge_type="TARGET_PERSON",
                )
            )

        for timeline in sequences.day_timelines:
            for session_uuid in timeline.session_uuids:
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "timeline-session", timeline.timeline_uuid, session_uuid),
                        from_id=timeline.timeline_uuid,
                        to_id=session_uuid,
                        edge_type="HAS_SESSION",
                    )
                )

        for session in sequences.sessions:
            place_name = str((session.location_hint or {}).get("name") or "unknown")
            place_uuid = place_uuid_by_name.get(place_name)
            if place_uuid:
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "session-place", session.session_uuid, place_uuid),
                        from_id=session.session_uuid,
                        to_id=place_uuid,
                        edge_type="IN_PLACE",
                    )
                )
            for person_uuid in session_participants.get(session.session_uuid, []):
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "person-session", person_uuid, session.session_uuid),
                        from_id=person_uuid,
                        to_id=session.session_uuid,
                        edge_type="CO_PRESENT_IN",
                    )
                )
            for concept_name in self._session_concepts(session):
                concept = concept_lookup.get(concept_name)
                if not concept:
                    continue
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "session-concept", session.session_uuid, concept.concept_uuid),
                        from_id=session.session_uuid,
                        to_id=concept.concept_uuid,
                        edge_type="HAS_CONCEPT",
                    )
                )

        for event in event_candidates:
            if not primary_person_uuid:
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "user-event-observed", self.user_id, event.event_uuid),
                        from_id=self.user_id,
                        to_id=event.event_uuid,
                        edge_type="OBSERVED_EVENT",
                        properties={"confidence": event.confidence},
                    )
                )
            place_uuid = place_uuid_by_name.get(event.location or "unknown")
            if place_uuid:
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "event-place", event.event_uuid, place_uuid),
                        from_id=event.event_uuid,
                        to_id=place_uuid,
                        edge_type="IN_PLACE",
                    )
                )
            for session_uuid in event.session_uuids:
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "event-session", event.event_uuid, session_uuid),
                        from_id=event.event_uuid,
                        to_id=session_uuid,
                        edge_type="DERIVED_FROM_SESSION",
                    )
                )
            for concept_name in self._event_concepts(event):
                concept = concept_lookup.get(concept_name)
                if not concept:
                    continue
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "event-concept", event.event_uuid, concept.concept_uuid),
                        from_id=event.event_uuid,
                        to_id=concept.concept_uuid,
                        edge_type="HAS_CONCEPT",
                    )
                )

        for relationship in relationship_hypotheses:
            if relationship.anchor_person_uuid:
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "person-relationship", relationship.anchor_person_uuid, relationship.relationship_uuid),
                        from_id=relationship.anchor_person_uuid,
                        to_id=relationship.relationship_uuid,
                        edge_type="HAS_RELATIONSHIP",
                    )
                )
            neo4j_edges.append(
                Neo4jRelationshipEdgeRecord(
                    edge_id=self._stable_uuid("edge", "relationship-target", relationship.relationship_uuid, relationship.target_person_uuid),
                    from_id=relationship.relationship_uuid,
                    to_id=relationship.target_person_uuid,
                    edge_type="TARGET_PERSON",
                )
            )
            for session_uuid in relationship.feature_snapshot.get("supporting_session_uuids", []):
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "relationship-session", relationship.relationship_uuid, session_uuid),
                        from_id=relationship.relationship_uuid,
                        to_id=session_uuid,
                        edge_type="SUPPORTED_BY_SESSION",
                    )
                )
            for event_uuid in relationship.feature_snapshot.get("supporting_event_uuids", []):
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "relationship-event", relationship.relationship_uuid, event_uuid),
                        from_id=relationship.relationship_uuid,
                        to_id=event_uuid,
                        edge_type="SUPPORTED_BY_EVENT",
                    )
                )
            for concept_name in self._relationship_concepts(relationship):
                concept = concept_lookup.get(concept_name)
                if not concept:
                    continue
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "relationship-concept", relationship.relationship_uuid, concept.concept_uuid),
                        from_id=relationship.relationship_uuid,
                        to_id=concept.concept_uuid,
                        edge_type="HAS_CONCEPT",
                    )
                )

        for mood in mood_hypotheses:
            if mood.session_uuid:
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "mood-session", mood.mood_uuid, mood.session_uuid),
                        from_id=mood.mood_uuid,
                        to_id=mood.session_uuid,
                        edge_type="DESCRIBES_SESSION",
                    )
                )
            for concept_name in self._mood_concepts(mood):
                concept = concept_lookup.get(concept_name)
                if not concept:
                    continue
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "mood-concept", mood.mood_uuid, concept.concept_uuid),
                        from_id=mood.mood_uuid,
                        to_id=concept.concept_uuid,
                        edge_type="HAS_CONCEPT",
                    )
                )

        for period in period_hypotheses:
            for session_uuid in self._period_session_uuids(period, sequences.sessions):
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "period-session", period.period_uuid, session_uuid),
                        from_id=period.period_uuid,
                        to_id=session_uuid,
                        edge_type="COVERS_SESSION",
                    )
                )
            for concept_name in self._period_concepts(period):
                concept = concept_lookup.get(concept_name)
                if not concept:
                    continue
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "period-concept", period.period_uuid, concept.concept_uuid),
                        from_id=period.period_uuid,
                        to_id=concept.concept_uuid,
                        edge_type="HAS_CONCEPT",
                    )
                )

        for concept in concept_catalog:
            parent_names = concept.parent_concepts or concept_metadata(concept.canonical_name).get("parents", [])
            for parent_name in parent_names:
                parent = concept_lookup.get(parent_name)
                if not parent:
                    continue
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "concept-parent", concept.concept_uuid, parent.concept_uuid),
                        from_id=concept.concept_uuid,
                        to_id=parent.concept_uuid,
                        edge_type="IS_A",
                    )
                )

        milvus_segments = self._build_milvus_segments(
            photos=photos,
            sequences=sequences,
            event_candidates=event_candidates,
            relationship_hypotheses=relationship_hypotheses,
            profile_evidence=profile_evidence,
            mood_hypotheses=mood_hypotheses,
            concept_catalog=concept_catalog,
        )
        redis = self._build_redis_records(
            profile_evidence=profile_evidence,
            relationships=relationship_hypotheses,
            events=event_candidates,
            day_timelines=sequences.day_timelines,
            profile_markdown=profile_markdown,
            generated_at=scope.generated_at,
            profile_version=profile_version,
        )

        return {
            "identity_maps": {
                "persons": [self._serialize(record) for record in people_records],
                "photos": [self._serialize(record) for record in photo_identity_records],
            },
            "neo4j": {
                "nodes": {name: [self._serialize(item) for item in items] for name, items in neo4j_nodes.items()},
                "edges": [self._serialize(edge) for edge in neo4j_edges],
            },
            "milvus": {
                "segments": [self._serialize(segment) for segment in milvus_segments],
            },
            "redis": redis,
            "materialization_bundle": self._serialize(bundle),
        }

    def _build_focus_graph(
        self,
        photos: Sequence[PhotoFactDTO],
        sequences: SequenceBundle,
        event_candidates: Sequence[EventCandidateDTO],
        relationship_hypotheses: Sequence[RelationshipHypothesisDTO],
        person_uuid_map: Dict[str, str],
        primary_face_person_id: Optional[str],
        primary_person_uuid: Optional[str],
    ) -> FocusGraphView:
        face_person_by_uuid = {
            person_uuid: face_person_id
            for face_person_id, person_uuid in person_uuid_map.items()
        }
        nodes: List[GraphNodeView] = []
        edges: List[GraphEdgeView] = []
        node_index: Dict[str, GraphNodeView] = {}

        def add_node(
            node_id: str,
            label: str,
            node_type: str,
            ring: int,
            *,
            is_primary: bool = False,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> None:
            if not node_id:
                return
            current = node_index.get(node_id)
            payload = metadata or {}
            if current:
                current.ring = min(current.ring, ring)
                current.is_primary = current.is_primary or is_primary
                current.metadata.update(payload)
                if len(label) > len(current.label):
                    current.label = label
                return
            item = GraphNodeView(
                node_id=node_id,
                label=label,
                node_type=node_type,
                ring=ring,
                is_primary=is_primary,
                metadata=dict(payload),
            )
            node_index[node_id] = item
            nodes.append(item)

        def add_edge(
            source_id: str,
            target_id: str,
            edge_type: str,
            *,
            label: str = "",
            confidence: Optional[float] = None,
            metadata: Optional[Dict[str, Any]] = None,
        ) -> None:
            if not source_id or not target_id:
                return
            edges.append(
                GraphEdgeView(
                    edge_id=self._stable_uuid("focus-edge", source_id, target_id, edge_type, label or ""),
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=edge_type,
                    label=label,
                    confidence=confidence,
                    metadata=dict(metadata or {}),
                )
            )

        center_node_id = primary_person_uuid or self.user_id
        if primary_person_uuid:
            add_node(
                primary_person_uuid,
                f"{primary_face_person_id or 'Primary'} · 主用户",
                "primary_person",
                0,
                is_primary=True,
                metadata={"face_person_id": primary_face_person_id},
            )
            add_node(
                self.user_id,
                self.user_id,
                "user_account",
                1,
                metadata={"role": "account"},
            )
            add_edge(
                self.user_id,
                primary_person_uuid,
                "PRIMARY_USER",
                label="主角锚点",
                metadata={"face_person_id": primary_face_person_id},
            )
        else:
            add_node(
                self.user_id,
                self.user_id,
                "user_account",
                0,
                metadata={"role": "account"},
            )

        for relationship in sorted(
            relationship_hypotheses,
            key=lambda item: (-item.confidence, item.target_face_person_id),
        ):
            add_node(
                relationship.target_person_uuid,
                f"{relationship.target_face_person_id} · {relationship.label}",
                "related_person",
                1,
                metadata={
                    "face_person_id": relationship.target_face_person_id,
                    "relationship_type": relationship.relationship_type,
                    "status": relationship.status,
                    "revision": relationship.revision,
                },
            )
            add_edge(
                center_node_id,
                relationship.target_person_uuid,
                "RELATIONSHIP_HYPOTHESIS",
                label=relationship.label,
                confidence=relationship.confidence,
                metadata={"relationship_type": relationship.relationship_type, "status": relationship.status},
            )

        relevant_event_uuids = set()
        relevant_session_uuids = set()
        relevant_photo_uuids = set()
        relevant_timeline_uuids = set()

        for event in sorted(
            event_candidates,
            key=lambda item: (item.started_at, item.title),
        ):
            involves_primary = bool(
                primary_person_uuid and primary_person_uuid in event.participant_person_uuids
            )
            if primary_person_uuid and not involves_primary:
                continue

            relevant_event_uuids.add(event.event_uuid)
            add_node(
                event.event_uuid,
                event.title or event.event_id,
                "event",
                1,
                metadata={"event_type": event.event_type, "location": event.location},
            )
            add_edge(
                center_node_id,
                event.event_uuid,
                "PARTICIPATED_IN" if primary_person_uuid else "OBSERVED_EVENT",
                label=event.event_type,
                confidence=event.confidence,
                metadata={"location": event.location},
            )

            for person_uuid in event.participant_person_uuids:
                if person_uuid == primary_person_uuid:
                    continue
                add_node(
                    person_uuid,
                    face_person_by_uuid.get(person_uuid, "Person"),
                    "participant_person",
                    1,
                    metadata={"face_person_id": face_person_by_uuid.get(person_uuid)},
                )
                add_edge(
                    event.event_uuid,
                    person_uuid,
                    "INVOLVES_PERSON",
                    label="参与人物",
                )

            for session_uuid in event.session_uuids:
                relevant_session_uuids.add(session_uuid)

            for photo_uuid in event.photo_uuids:
                relevant_photo_uuids.add(photo_uuid)

        for photo in photos:
            contains_primary = bool(
                primary_person_uuid and any(face.person_uuid == primary_person_uuid for face in photo.faces)
            )
            if contains_primary:
                relevant_photo_uuids.add(photo.photo_uuid)

        session_by_uuid = {session.session_uuid: session for session in sequences.sessions}
        for photo in photos:
            if photo.photo_uuid not in relevant_photo_uuids:
                continue
            add_node(
                photo.photo_uuid,
                photo.filename,
                "photo",
                2,
                metadata={"photo_id": photo.photo_id, "captured_at": photo.captured_at_utc},
            )
            if primary_person_uuid and any(face.person_uuid == primary_person_uuid for face in photo.faces):
                add_edge(
                    center_node_id,
                    photo.photo_uuid,
                    "APPEARS_IN",
                    label="出现于照片",
                )
            for session in sequences.sessions:
                if photo.photo_id not in session.photo_ids:
                    continue
                relevant_session_uuids.add(session.session_uuid)
                add_edge(
                    photo.photo_uuid,
                    session.session_uuid,
                    "IN_SESSION",
                    label=session.day_key,
                )
                break

        for session_uuid in sorted(relevant_session_uuids):
            session = session_by_uuid.get(session_uuid)
            if not session:
                continue
            add_node(
                session.session_uuid,
                session.summary_hint or session.session_id,
                "session",
                2,
                metadata={"day_key": session.day_key, "location_hint": session.location_hint},
            )
            if primary_person_uuid and primary_person_uuid in session.dominant_person_uuids:
                add_edge(
                    center_node_id,
                    session.session_uuid,
                    "ACTIVE_IN_SESSION",
                    label=session.day_key,
                )
            for timeline in sequences.day_timelines:
                if session.session_uuid not in timeline.session_uuids:
                    continue
                relevant_timeline_uuids.add(timeline.timeline_uuid)

        for timeline in sequences.day_timelines:
            if timeline.timeline_uuid not in relevant_timeline_uuids:
                continue
            add_node(
                timeline.timeline_uuid,
                timeline.day_key,
                "timeline",
                3,
                metadata={"day_key": timeline.day_key},
            )
            for session_uuid in timeline.session_uuids:
                if session_uuid not in relevant_session_uuids:
                    continue
                add_edge(
                    timeline.timeline_uuid,
                    session_uuid,
                    "PART_OF_DAY_TIMELINE",
                    label="时间线",
                )

        mermaid = self._build_focus_graph_mermaid(nodes, edges)
        return FocusGraphView(
            center_node_id=center_node_id,
            primary_face_person_id=primary_face_person_id,
            primary_person_uuid=primary_person_uuid,
            node_count=len(nodes),
            edge_count=len(edges),
            nodes=nodes,
            edges=edges,
            mermaid=mermaid,
        )

    def _build_focus_graph_mermaid(
        self,
        nodes: Sequence[GraphNodeView],
        edges: Sequence[GraphEdgeView],
    ) -> str:
        if not nodes:
            return "graph LR\n"

        lines = ["graph LR"]
        for node in nodes:
            label = node.label.replace('"', "'")
            lines.append(f'  {self._mermaid_id(node.node_id)}["{label}"]')
        for edge in edges:
            edge_label = edge.label or edge.edge_type
            edge_label = edge_label.replace('"', "'")
            lines.append(
                f"  {self._mermaid_id(edge.source_id)} -->|{edge_label}| {self._mermaid_id(edge.target_id)}"
            )
        return "\n".join(lines)

    def _build_milvus_segments(
        self,
        photos: Sequence[PhotoFactDTO],
        sequences: SequenceBundle,
        event_candidates: Sequence[EventCandidateDTO],
        relationship_hypotheses: Sequence[RelationshipHypothesisDTO],
        profile_evidence: Sequence[ProfileEvidenceDTO],
        mood_hypotheses: Sequence[MoodStateHypothesisDTO],
        concept_catalog: Sequence[ConceptDTO],
    ) -> List[MilvusSegmentRecord]:
        segments: List[MilvusSegmentRecord] = []
        session_by_uuid = {session.session_uuid: session for session in sequences.sessions}
        session_by_photo_id = {
            photo_id: session
            for session in sequences.sessions
            for photo_id in session.photo_ids
        }
        event_by_uuid = {event.event_uuid: event for event in event_candidates}

        for photo in photos:
            session = session_by_photo_id.get(photo.photo_id)
            if photo.vlm_observation and photo.vlm_observation.summary:
                segments.append(
                    self._segment_record(
                        photo_uuid=photo.photo_uuid,
                        segment_type="scene_summary",
                        text=photo.vlm_observation.summary,
                        session_uuid=self._photo_session_uuid(photo.photo_id, sequences.sessions),
                        started_at=session.started_at if session else None,
                        ended_at=session.ended_at if session else None,
                        place_uuid=self._session_place_uuid(session),
                        location_hint=self._session_location_name(session),
                        evidence_refs=[{"ref_type": "photo", "ref_id": photo.photo_id}],
                    )
                )

            if photo.vlm_observation:
                activity = photo.vlm_observation.event.get("activity") if isinstance(photo.vlm_observation.event, dict) else None
                if activity:
                    segments.append(
                        self._segment_record(
                            photo_uuid=photo.photo_uuid,
                            segment_type="activity",
                            text=str(activity),
                            session_uuid=self._photo_session_uuid(photo.photo_id, sequences.sessions),
                            started_at=session.started_at if session else None,
                            ended_at=session.ended_at if session else None,
                            place_uuid=self._session_place_uuid(session),
                            location_hint=self._session_location_name(session),
                            evidence_refs=[{"ref_type": "photo", "ref_id": photo.photo_id}],
                        )
                    )
                interaction = photo.vlm_observation.event.get("interaction") if isinstance(photo.vlm_observation.event, dict) else None
                if interaction:
                    segments.append(
                        self._segment_record(
                            photo_uuid=photo.photo_uuid,
                            segment_type="interaction",
                            text=str(interaction),
                            session_uuid=self._photo_session_uuid(photo.photo_id, sequences.sessions),
                            started_at=session.started_at if session else None,
                            ended_at=session.ended_at if session else None,
                            place_uuid=self._session_place_uuid(session),
                            location_hint=self._session_location_name(session),
                            evidence_refs=[{"ref_type": "photo", "ref_id": photo.photo_id}],
                        )
                    )

            for person in photo.people_observations:
                text = " | ".join(part for part in [person.appearance, person.clothing, person.interaction] if part)
                if not text:
                    continue
                segments.append(
                    self._segment_record(
                        photo_uuid=photo.photo_uuid,
                        segment_type="person_observation",
                        text=text,
                        person_uuid=person.person_uuid,
                        session_uuid=self._photo_session_uuid(photo.photo_id, sequences.sessions),
                        started_at=session.started_at if session else None,
                        ended_at=session.ended_at if session else None,
                        place_uuid=self._session_place_uuid(session),
                        location_hint=self._session_location_name(session),
                        evidence_refs=[{"ref_type": "photo_person_observation", "ref_id": person.observation_id}],
                    )
                )
                if person.interaction:
                    segments.append(
                        self._segment_record(
                            photo_uuid=photo.photo_uuid,
                            segment_type="interaction",
                            text=str(person.interaction),
                            person_uuid=person.person_uuid,
                            session_uuid=self._photo_session_uuid(photo.photo_id, sequences.sessions),
                            started_at=session.started_at if session else None,
                            ended_at=session.ended_at if session else None,
                            place_uuid=self._session_place_uuid(session),
                            location_hint=self._session_location_name(session),
                            evidence_refs=[{"ref_type": "photo_person_observation", "ref_id": person.observation_id}],
                        )
                    )

            if photo.vlm_observation and photo.vlm_observation.raw:
                raw_text_candidates = []
                for detail in photo.vlm_observation.details:
                    raw_text_candidates.append(detail)
                for obj in photo.vlm_observation.key_objects:
                    raw_text_candidates.append(obj)
                for raw_text in raw_text_candidates[:5]:
                    segments.append(
                        self._segment_record(
                            photo_uuid=photo.photo_uuid,
                            segment_type="ocr_snippet",
                            text=str(raw_text),
                            session_uuid=self._photo_session_uuid(photo.photo_id, sequences.sessions),
                            started_at=session.started_at if session else None,
                            ended_at=session.ended_at if session else None,
                            place_uuid=self._session_place_uuid(session),
                            location_hint=self._session_location_name(session),
                            evidence_refs=[{"ref_type": "photo", "ref_id": photo.photo_id}],
                        )
                    )

        for session in sequences.sessions:
            if session.summary_hint:
                segments.append(
                    self._segment_record(
                        photo_uuid=session.photo_uuids[0] if session.photo_uuids else "",
                        segment_type="session_summary",
                        text=session.summary_hint,
                        session_uuid=session.session_uuid,
                        started_at=session.started_at,
                        ended_at=session.ended_at,
                        place_uuid=self._session_place_uuid(session),
                        location_hint=self._session_location_name(session),
                        evidence_refs=[{"ref_type": "session", "ref_id": session.session_id}],
                    )
                )

        for event in event_candidates:
            text = event.narrative_synthesis or event.description or event.title
            if text:
                anchor_session = session_by_uuid.get(event.session_uuids[0]) if event.session_uuids else None
                segments.append(
                    self._segment_record(
                        photo_uuid=event.photo_uuids[0] if event.photo_uuids else "",
                        segment_type="event_narrative",
                        text=text,
                        event_uuid=event.event_uuid,
                        session_uuid=event.session_uuids[0] if event.session_uuids else None,
                        started_at=event.started_at,
                        ended_at=event.ended_at,
                        place_uuid=self._event_place_uuid(event, anchor_session),
                        location_hint=self._event_location_name(event, anchor_session),
                        evidence_refs=event.evidence_refs,
                    )
                )

        for relationship in relationship_hypotheses:
            if relationship.reason_summary:
                anchor_session_uuid = relationship.feature_snapshot.get("supporting_session_uuids", [None])[0]
                anchor_event_uuid = relationship.feature_snapshot.get("supporting_event_uuids", [None])[0]
                anchor_session = session_by_uuid.get(anchor_session_uuid) if anchor_session_uuid else None
                anchor_event = event_by_uuid.get(anchor_event_uuid) if anchor_event_uuid else None
                segments.append(
                    self._segment_record(
                        photo_uuid=None,
                        segment_type="relationship_reason",
                        text=relationship.reason_summary,
                        relationship_uuid=relationship.relationship_uuid,
                        person_uuid=relationship.target_person_uuid,
                        session_uuid=anchor_session_uuid,
                        event_uuid=anchor_event_uuid,
                        started_at=anchor_event.started_at if anchor_event else (anchor_session.started_at if anchor_session else relationship.window_start),
                        ended_at=anchor_event.ended_at if anchor_event else (anchor_session.ended_at if anchor_session else relationship.window_end),
                        place_uuid=self._event_place_uuid(anchor_event, anchor_session),
                        location_hint=self._event_location_name(anchor_event, anchor_session),
                        evidence_refs=relationship.evidence_refs,
                    )
                )

        for mood in mood_hypotheses:
            if mood.reason_summary:
                mood_session = session_by_uuid.get(mood.session_uuid) if mood.session_uuid else None
                mood_event = event_by_uuid.get(mood.event_uuid) if mood.event_uuid else None
                segments.append(
                    self._segment_record(
                        photo_uuid=None,
                        segment_type="profile_evidence_snippet",
                        text=f"mood: {mood.reason_summary}",
                        session_uuid=mood.session_uuid,
                        event_uuid=mood.event_uuid,
                        started_at=mood.window_start or (mood_event.started_at if mood_event else (mood_session.started_at if mood_session else None)),
                        ended_at=mood.window_end or (mood_event.ended_at if mood_event else (mood_session.ended_at if mood_session else None)),
                        place_uuid=self._event_place_uuid(mood_event, mood_session),
                        location_hint=self._event_location_name(mood_event, mood_session),
                        evidence_refs=mood.evidence_refs,
                    )
                )

        for item in profile_evidence:
            if item.supporting_event_uuids:
                anchor_event_uuid = item.supporting_event_uuids[0]
            else:
                anchor_event_uuid = None
            anchor_event = event_by_uuid.get(anchor_event_uuid) if anchor_event_uuid else None
            anchor_session_uuid = item.supporting_session_uuids[0] if item.supporting_session_uuids else None
            anchor_session = session_by_uuid.get(anchor_session_uuid) if anchor_session_uuid else None
            segments.append(
                self._segment_record(
                    photo_uuid="",
                    segment_type="profile_evidence_snippet",
                    text=f"{item.field_key}: {item.field_value}",
                    event_uuid=anchor_event_uuid,
                    person_uuid=item.supporting_person_uuids[0] if item.supporting_person_uuids else None,
                    session_uuid=anchor_session_uuid,
                    started_at=anchor_event.started_at if anchor_event else (anchor_session.started_at if anchor_session else None),
                    ended_at=anchor_event.ended_at if anchor_event else (anchor_session.ended_at if anchor_session else None),
                    place_uuid=self._event_place_uuid(anchor_event, anchor_session),
                    location_hint=self._event_location_name(anchor_event, anchor_session),
                    evidence_refs=item.evidence_refs,
                )
            )

        for concept in concept_catalog:
            if concept.scope != "candidate":
                continue
            segments.append(
                self._segment_record(
                    photo_uuid=None,
                    segment_type="profile_evidence_snippet",
                    text=f"candidate concept: {concept.canonical_name}",
                    concept_uuid=concept.concept_uuid,
                    evidence_refs=[{"ref_type": "concept", "ref_id": concept.canonical_name}],
                )
            )

        return segments

    def _segment_record(
        self,
        photo_uuid: Optional[str],
        segment_type: str,
        text: str,
        event_uuid: Optional[str] = None,
        person_uuid: Optional[str] = None,
        session_uuid: Optional[str] = None,
        relationship_uuid: Optional[str] = None,
        concept_uuid: Optional[str] = None,
        started_at: Optional[str] = None,
        ended_at: Optional[str] = None,
        place_uuid: Optional[str] = None,
        location_hint: str = "",
        evidence_refs: Optional[List[Dict[str, str]]] = None,
    ) -> MilvusSegmentRecord:
        return MilvusSegmentRecord(
            segment_uuid=self._stable_uuid(
                "segment",
                segment_type,
                photo_uuid or "",
                event_uuid or "",
                person_uuid or "",
                relationship_uuid or "",
                concept_uuid or "",
                text[:80],
            ),
            tenant_id=None,
            user_id=self.user_id,
            photo_uuid=photo_uuid,
            event_uuid=event_uuid,
            person_uuid=person_uuid,
            session_uuid=session_uuid,
            relationship_uuid=relationship_uuid,
            concept_uuid=concept_uuid,
            started_at=started_at,
            ended_at=ended_at,
            place_uuid=place_uuid,
            location_hint=location_hint,
            segment_type=segment_type,
            text=text,
            sparse_terms=self._tokenize(text),
            importance_score=min(1.0, 0.35 + (len(text) / 200.0)),
            evidence_refs=evidence_refs or [],
        )

    def _session_place_uuid(self, session: Optional[SessionDTO]) -> Optional[str]:
        if not session:
            return None
        location_hint = dict(session.location_hint or {})
        canonical_name = str(location_hint.get("name") or "unknown").strip() or "unknown"
        lat = location_hint.get("lat")
        lng = location_hint.get("lng")
        geo_hash = f"{round(float(lat), 3) if lat is not None else 'na'}:{round(float(lng), 3) if lng is not None else 'na'}"
        return self._stable_uuid("place", canonical_name, geo_hash)

    def _session_location_name(self, session: Optional[SessionDTO]) -> str:
        if not session:
            return ""
        return str((session.location_hint or {}).get("name") or "")

    def _event_place_uuid(self, event: Optional[EventCandidateDTO], session: Optional[SessionDTO]) -> Optional[str]:
        if event and event.location:
            return self._stable_uuid("place", event.location, "na:na")
        return self._session_place_uuid(session)

    def _event_location_name(self, event: Optional[EventCandidateDTO], session: Optional[SessionDTO]) -> str:
        if event and event.location:
            return event.location
        return self._session_location_name(session)

    def _build_redis_records(
        self,
        profile_evidence: Sequence[ProfileEvidenceDTO],
        relationships: Sequence[RelationshipHypothesisDTO],
        events: Sequence[EventCandidateDTO],
        day_timelines: Sequence,
        profile_markdown: str,
        generated_at: str,
        profile_version: int,
    ) -> Dict[str, Any]:
        grouped: Dict[str, List[ProfileEvidenceDTO]] = defaultdict(list)
        for item in profile_evidence:
            grouped[item.field_key].append(item)

        profile_fields = {}
        publish_decisions = []
        debug_refs = {}
        for field_key, items in grouped.items():
            top_values = Counter(item.field_value for item in items).most_common(5)
            avg_confidence = sum(item.confidence for item in items) / len(items)
            supporting_event_ids = self._unique(
                event_id
                for item in items
                for event_id in item.supporting_event_ids
            )
            evidence_refs = self._unique_refs(ref for item in items for ref in item.evidence_refs)
            field_payload = {
                "field_key": field_key,
                "values": [value for value, _count in top_values],
                "confidence": round(avg_confidence, 4),
                "evidence_refs": evidence_refs,
                "supporting_event_ids": supporting_event_ids,
                "generated_at": generated_at,
                "profile_version": profile_version,
                "evaluation_status": "pending_gold_review",
            }
            profile_fields[field_key] = field_payload
            debug_refs[field_key] = evidence_refs
            publish_decisions.append(
                PublishDecisionView(
                    field_key=field_key,
                    status="published",
                    confidence=round(avg_confidence, 4),
                    reason=f"{len(items)} evidence items",
                    evidence_refs=evidence_refs,
                )
            )

        recent_events = [
            {
                "event_id": event.event_id,
                "event_uuid": event.event_uuid,
                "title": event.title,
                "location": event.location,
                "started_at": event.started_at,
                "ended_at": event.ended_at,
                "confidence": event.confidence,
                "participants": event.participant_face_person_ids,
                "evidence_refs": event.evidence_refs,
            }
            for event in sorted(events, key=lambda item: (item.started_at, item.title), reverse=True)[:10]
        ]
        recent_timelines = [
            {
                "timeline_id": timeline.timeline_id,
                "timeline_uuid": timeline.timeline_uuid,
                "day_key": timeline.day_key,
                "session_ids": timeline.session_ids,
                "movement_ids": timeline.movement_ids,
                "started_at": timeline.started_at,
                "ended_at": timeline.ended_at,
            }
            for timeline in day_timelines
        ]
        relationship_items = [
            {
                "relationship_uuid": relationship.relationship_uuid,
                "relationship_key": relationship.relationship_key,
                "revision": relationship.revision,
                "status": relationship.status,
                "target_face_person_id": relationship.target_face_person_id,
                "target_person_uuid": relationship.target_person_uuid,
                "relationship_type": relationship.relationship_type,
                "label": relationship.label,
                "confidence": relationship.confidence,
                "reason_summary": relationship.reason_summary,
                "window_start": relationship.window_start,
                "window_end": relationship.window_end,
                "feature_snapshot": relationship.feature_snapshot,
                "score_snapshot": relationship.score_snapshot,
                "evidence_refs": relationship.evidence_refs,
            }
            for relationship in relationships
            if relationship.status in {"active", "cooling"}
        ]

        profile_core = RedisProfileCoreRecord(
            key=f"profile:{self.user_id}:core",
            payload={
                "fields": profile_fields,
                "profile_markdown": profile_markdown,
            },
        )
        profile_relationships = RedisProfileRelationshipsRecord(
            key=f"profile:{self.user_id}:relationships",
            payload={"items": relationship_items},
        )
        profile_recent_events = RedisProfileRecentEventsRecord(
            key=f"profile:{self.user_id}:recent_events",
            payload={"items": recent_events},
        )
        profile_recent_timelines = RedisProfileRecentTimelinesRecord(
            key=f"profile:{self.user_id}:recent_timelines",
            payload={"items": recent_timelines},
        )
        profile_meta = RedisProfileMetaRecord(
            key=f"profile:{self.user_id}:meta",
            payload={
                "profile_version": profile_version,
                "generated_at": generated_at,
                "source_ingestion_ids": [self._stable_uuid("ingestion", self.task_id)],
                "state": "published",
                "staleness": "fresh",
                "evaluation_status": "pending_gold_review",
            },
        )
        profile_debug_refs = RedisProfileDebugRefsRecord(
            key=f"profile:{self.user_id}:debug_refs",
            payload={"fields": debug_refs, "publish_decisions": [self._serialize(item) for item in publish_decisions]},
        )

        return {
            "profile_core": {"key": profile_core.key, **self._serialize(profile_core.payload)},
            "profile_relationships": {"key": profile_relationships.key, **self._serialize(profile_relationships.payload)},
            "profile_recent_events": {"key": profile_recent_events.key, **self._serialize(profile_recent_events.payload)},
            "profile_recent_timelines": {"key": profile_recent_timelines.key, **self._serialize(profile_recent_timelines.payload)},
            "profile_meta": {"key": profile_meta.key, **self._serialize(profile_meta.payload)},
            "profile_debug_refs": {"key": profile_debug_refs.key, **self._serialize(profile_debug_refs.payload)},
        }

    def _build_transparency(
        self,
        face_output: Dict[str, Any],
        vlm_results: Sequence[Dict[str, Any]],
        sequences: SequenceBundle,
        event_candidates: Sequence[EventCandidateDTO],
        relationship_hypotheses: Sequence[RelationshipHypothesisDTO],
        profile_evidence: Sequence[ProfileEvidenceDTO],
        storage: Dict[str, Any],
        change_log: Sequence[ChangeLogEntryDTO],
        cached_photo_ids: set[str],
        focus_graph: FocusGraphView,
    ) -> Dict[str, Any]:
        face_stage = FaceStageView(
            total_faces=int(face_output.get("metrics", {}).get("total_faces", 0)),
            total_persons=int(face_output.get("metrics", {}).get("total_persons", 0)),
            primary_face_person_id=face_output.get("primary_person_id"),
            failed_images=len(face_output.get("failed_images", []) or []),
        )
        vlm_stage = VLMStageView(
            processed_photos=len(vlm_results),
            cached_hits=len(cached_photo_ids),
            summaries=[
                {
                    "photo_id": item.get("photo_id"),
                    "summary": item.get("vlm_analysis", {}).get("summary"),
                    "activity": item.get("vlm_analysis", {}).get("event", {}).get("activity"),
                }
                for item in vlm_results[:10]
            ],
        )
        sequence_stage = SequenceStageView(
            burst_count=len(sequences.bursts),
            session_count=len(sequences.sessions),
            movement_count=len(sequences.movements),
            timeline_count=len(sequences.day_timelines),
            summaries=[
                {
                    "session_id": session.session_id,
                    "day_key": session.day_key,
                    "photo_count": len(session.photo_ids),
                    "location_hint": session.location_hint,
                    "summary_hint": session.summary_hint,
                }
                for session in sequences.sessions[:10]
            ],
        )
        llm_stage = LLMStageView(
            event_candidate_count=len(event_candidates),
            relationship_hypothesis_count=len(relationship_hypotheses),
            profile_evidence_count=len(profile_evidence),
            summaries=[
                {
                    "event_id": event.event_id,
                    "title": event.title,
                    "confidence": event.confidence,
                }
                for event in event_candidates[:10]
            ],
        )
        neo4j_nodes = storage["neo4j"]["nodes"]
        neo4j_state = Neo4jStateView(
            node_counts={name: len(items) for name, items in neo4j_nodes.items()},
            edge_count=len(storage["neo4j"]["edges"]),
        )
        segment_counts = Counter(segment["segment_type"] for segment in storage["milvus"]["segments"])
        milvus_state = MilvusStateView(
            segment_count=len(storage["milvus"]["segments"]),
            segment_type_counts=dict(segment_counts),
        )
        profile_core = storage["redis"]["profile_core"]
        redis_state = RedisStateView(
            profile_version=int(storage["redis"]["profile_meta"].get("profile_version", 0)),
            published_field_count=len(profile_core.get("fields", {})),
            relationship_count=len(storage["redis"]["profile_relationships"].get("items", [])),
            recent_event_count=len(storage["redis"]["profile_recent_events"].get("items", [])),
            recent_timeline_count=len(storage["redis"]["profile_recent_timelines"].get("items", [])),
        )
        object_diff = ObjectDiffView(
            change_count=len(change_log),
            changes=[self._serialize(change) for change in change_log[:50]],
        )

        traces = []
        evidence_chains = []
        for field_key, payload in profile_core.get("fields", {}).items():
            trace_id = self._stable_uuid("trace", field_key)
            evidence_chain = [
                {"ref_type": ref.get("ref_type"), "ref_id": ref.get("ref_id")}
                for ref in payload.get("evidence_refs", [])
            ]
            traces.append(
                TraceView(
                    trace_id=trace_id,
                    trace_type="profile_field",
                    summary=f"{field_key} -> {', '.join(payload.get('values', []))}",
                    evidence_chain=evidence_chain,
                )
            )
            evidence_chains.append(
                EvidenceChainView(
                    target_id=field_key,
                    chain=evidence_chain,
                )
            )

        return {
            "face_stage": self._serialize(face_stage),
            "vlm_stage": self._serialize(vlm_stage),
            "sequence_stage": self._serialize(sequence_stage),
            "llm_stage": self._serialize(llm_stage),
            "neo4j_state": self._serialize(neo4j_state),
            "focus_graph": self._serialize(focus_graph),
            "milvus_state": self._serialize(milvus_state),
            "redis_state": self._serialize(redis_state),
            "object_diff": self._serialize(object_diff),
            "traces": [self._serialize(item) for item in traces],
            "evidence_chains": [self._serialize(item) for item in evidence_chains],
            "publish_decisions": storage["redis"]["profile_debug_refs"].get("publish_decisions", []),
        }

    def _build_evaluation(
        self,
        scope: ScopeDTO,
        profile_evidence: Sequence[ProfileEvidenceDTO],
        relationships: Sequence[RelationshipHypothesisDTO],
        events: Sequence[EventCandidateDTO],
    ) -> Dict[str, Any]:
        evaluated_at = scope.generated_at
        profile_fields = len({item.field_key for item in profile_evidence})
        metric_snapshots = [
            MetricSnapshotDTO(
                metric_name="profile_field_coverage",
                value=float(profile_fields),
                evaluated_at=evaluated_at,
                scope="task",
                metadata={"unit": "fields"},
            ),
            MetricSnapshotDTO(
                metric_name="relationship_candidate_count",
                value=float(len(relationships)),
                evaluated_at=evaluated_at,
                scope="task",
            ),
            MetricSnapshotDTO(
                metric_name="event_candidate_count",
                value=float(len(events)),
                evaluated_at=evaluated_at,
                scope="task",
            ),
        ]
        run = EvaluationRunDTO(
            evaluation_run_id=self._stable_uuid("evaluation", self.task_id),
            user_id=self.user_id,
            evaluated_at=evaluated_at,
            metric_snapshots=metric_snapshots,
            error_buckets=[
                ErrorBucketDTO(bucket_id="pending_manual_review", bucket_name="pending_manual_review", count=0, examples=[]),
            ],
        )
        return self._serialize(run)

    def _build_photo_artifacts(self, photo) -> List[ArtifactRefDTO]:
        artifacts = []
        for artifact_type, path in (
            ("original", photo.original_path or photo.path),
            ("compressed", getattr(photo, "compressed_path", None)),
            ("boxed", getattr(photo, "boxed_path", None)),
        ):
            if not path:
                continue
            path_obj = Path(path)
            artifacts.append(
                ArtifactRefDTO(
                    artifact_type=artifact_type,
                    path=str(path_obj),
                    url=self._public_url(path_obj),
                    mime_type=self._guess_mime_type(path_obj),
                )
            )
        return artifacts

    def _build_top_level_artifacts(self, photos: Sequence, profile_markdown: str) -> List[ArtifactRefDTO]:
        artifacts = []
        if profile_markdown:
            profile_path = self.task_dir / "output" / "user_profile_report.md"
            if profile_path.exists():
                artifacts.append(
                    ArtifactRefDTO(
                        artifact_type="profile_markdown",
                        path=str(profile_path),
                        url=self._public_url(profile_path),
                        mime_type="text/markdown",
                    )
                )

        for photo in photos[:10]:
            for artifact in self._build_photo_artifacts(photo):
                artifacts.append(artifact)

        return artifacts

    def _save_outputs(self, memory_payload: Dict[str, Any]) -> Dict[str, Any]:
        paths = {
            "envelope": self.output_dir / "memory_envelope.json",
            "storage": self.output_dir / "memory_storage.json",
            "transparency": self.output_dir / "memory_transparency.json",
            "evaluation": self.output_dir / "memory_evaluation.json",
            "external_publish": self.output_dir / "external_publish_report.json",
            "focus_graph": self.output_dir / "memory_focus_graph.json",
            "focus_graph_mermaid": self.output_dir / "memory_focus_graph.mmd",
        }
        save_json(memory_payload["envelope"], str(paths["envelope"]))
        save_json(memory_payload["storage"], str(paths["storage"]))
        save_json(memory_payload["transparency"], str(paths["transparency"]))
        save_json(memory_payload["evaluation"], str(paths["evaluation"]))
        focus_graph = memory_payload.get("transparency", {}).get("focus_graph")
        if focus_graph:
            save_json(focus_graph, str(paths["focus_graph"]))
            paths["focus_graph_mermaid"].write_text(
                str(focus_graph.get("mermaid") or ""),
                encoding="utf-8",
            )
        if "external_publish" in memory_payload:
            save_json(memory_payload["external_publish"], str(paths["external_publish"]))
        return {
            "envelope_path": str(paths["envelope"]),
            "storage_path": str(paths["storage"]),
            "transparency_path": str(paths["transparency"]),
            "evaluation_path": str(paths["evaluation"]),
            "external_publish_path": str(paths["external_publish"]),
            "focus_graph_path": str(paths["focus_graph"]),
            "focus_graph_mermaid_path": str(paths["focus_graph_mermaid"]),
            "envelope_url": self._public_url(paths["envelope"]),
            "storage_url": self._public_url(paths["storage"]),
            "transparency_url": self._public_url(paths["transparency"]),
            "evaluation_url": self._public_url(paths["evaluation"]),
            "external_publish_url": self._public_url(paths["external_publish"]),
            "focus_graph_url": self._public_url(paths["focus_graph"]),
            "focus_graph_mermaid_url": self._public_url(paths["focus_graph_mermaid"]),
        }

    def _event_bounds(self, event: Event) -> tuple[str, str]:
        if isinstance(event.meta_info, dict):
            timestamp = str(event.meta_info.get("timestamp") or "").strip()
            if timestamp:
                parts = [part.strip() for part in timestamp.split(" - ")]
                if len(parts) == 2:
                    return self._normalize_datetime_string(parts[0], event.date), self._normalize_datetime_string(parts[1], event.date)
                normalized = self._normalize_datetime_string(timestamp, event.date)
                return normalized, normalized

        if event.date and event.time_range:
            pieces = [part.strip() for part in event.time_range.split("-")]
            if len(pieces) == 2:
                return self._normalize_datetime_string(f"{event.date} {pieces[0]}", event.date), self._normalize_datetime_string(f"{event.date} {pieces[1]}", event.date)

        fallback = self._normalize_datetime_string(event.date or datetime.now().date().isoformat(), event.date)
        return fallback, fallback

    def _normalize_datetime_string(self, raw: str, fallback_date: Optional[str]) -> str:
        candidate = str(raw).strip()
        if not candidate:
            candidate = fallback_date or datetime.now().date().isoformat()
        if "T" in candidate:
            return candidate
        if " " in candidate:
            date_part, time_part = candidate.split(" ", 1)
            if len(time_part) == 5:
                return f"{date_part}T{time_part}:00"
            return f"{date_part}T{time_part}"
        if fallback_date and ":" in candidate:
            return f"{fallback_date}T{candidate}:00" if len(candidate) == 5 else f"{fallback_date}T{candidate}"
        return f"{candidate}T00:00:00"

    def _time_windows_overlap(
        self,
        left_start: str,
        left_end: str,
        right_start: str,
        right_end: str,
        event_date: str,
    ) -> bool:
        try:
            left_start_dt = datetime.fromisoformat(left_start)
            left_end_dt = datetime.fromisoformat(left_end)
            right_start_dt = datetime.fromisoformat(right_start)
            right_end_dt = datetime.fromisoformat(right_end)
        except ValueError:
            return bool(event_date) and event_date == right_start[:10]
        return not (left_end_dt < right_start_dt or right_end_dt < left_start_dt)

    def _photo_session_uuid(self, photo_id: str, sessions: Iterable) -> Optional[str]:
        for session in sessions:
            if photo_id in session.photo_ids:
                return session.session_uuid
        return None

    def _collect_face_person_ids(
        self,
        photos: Sequence,
        face_output: Dict[str, Any],
        relationships: Sequence[Relationship],
    ) -> set[str]:
        person_ids = set()
        for photo in photos:
            if getattr(photo, "primary_person_id", None):
                person_ids.add(str(photo.primary_person_id))
            for face in photo.faces:
                person_id = face.get("person_id")
                if person_id:
                    person_ids.add(str(person_id))
        for person in face_output.get("persons", []):
            person_id = person.get("person_id")
            if person_id:
                person_ids.add(str(person_id))
        for relationship in relationships:
            if relationship.person_id:
                person_ids.add(str(relationship.person_id))
        return person_ids

    def _resolve_primary_face_person_id(self, photos: Sequence, face_output: Dict[str, Any]) -> Optional[str]:
        primary_person_id = str(face_output.get("primary_person_id") or "").strip()
        if primary_person_id:
            return primary_person_id

        candidates = Counter(
            str(getattr(photo, "primary_person_id", "") or "").strip()
            for photo in photos
            if getattr(photo, "primary_person_id", None)
        )
        if not candidates:
            return None
        primary_person_id, _count = candidates.most_common(1)[0]
        return primary_person_id or None

    def _bbox_to_xywh(self, bbox: Any) -> Dict[str, int]:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return {"x": 0, "y": 0, "w": 0, "h": 0}
        x1, y1, x2, y2 = [int(value) for value in bbox]
        return {"x": x1, "y": y1, "w": max(0, x2 - x1), "h": max(0, y2 - y1)}

    def _mermaid_id(self, raw: str) -> str:
        sanitized = "".join(char if char.isalnum() else "_" for char in str(raw))
        if sanitized and sanitized[0].isdigit():
            sanitized = f"n_{sanitized}"
        return sanitized or "node"

    def _public_url(self, path: Path | str) -> Optional[str]:
        if not self.public_url_builder:
            return None
        return self.public_url_builder(path)

    def _guess_mime_type(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".webp":
            return "image/webp"
        if suffix in {".jpg", ".jpeg"}:
            return "image/jpeg"
        if suffix == ".png":
            return "image/png"
        if suffix in {".md", ".markdown"}:
            return "text/markdown"
        return "application/octet-stream"

    def _tokenize(self, text: str) -> List[str]:
        seen = set()
        tokens = []
        for raw in text.replace("|", " ").replace(",", " ").split():
            token = raw.strip().lower()
            if len(token) < 2 or token in seen:
                continue
            seen.add(token)
            tokens.append(token)
        return tokens[:24]

    def _unique(self, values: Iterable[str]) -> List[str]:
        seen = set()
        ordered = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    def _unique_refs(self, values: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
        seen = set()
        ordered = []
        for value in values:
            key = (value.get("ref_type"), value.get("ref_id"))
            if key in seen:
                continue
            seen.add(key)
            ordered.append({"ref_type": key[0], "ref_id": key[1]})
        return ordered

    def _serialize(self, value: Any) -> Any:
        if is_dataclass(value):
            return {key: self._serialize(item) for key, item in asdict(value).items()}
        if isinstance(value, dict):
            return {key: self._serialize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._serialize(item) for item in value]
        return value

    def _stable_uuid(self, object_type: str, *parts: str) -> str:
        payload = ":".join([self.scope_key, object_type, *[str(part) for part in parts]])
        return str(uuid5(NAMESPACE_URL, payload))

    def _canonical_person_uuid(self, face_person_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"{self.user_id}:person:{face_person_id}"))
