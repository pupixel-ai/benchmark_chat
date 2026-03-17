"""Task-scoped memory ingestion/materialization service."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence
from uuid import NAMESPACE_URL, uuid5

from memory_module.adapters import MemoryStoragePublisher
from memory_module.domain import MaterializationInputBundle, ProfileEvidenceItem
from memory_module.dto import (
    ArtifactRefDTO,
    ChangeLogEntryDTO,
    EventCandidateDTO,
    FaceObservationDTO,
    MemoryIngestionEnvelopeDTO,
    PersonObservationDTO,
    PhotoFactDTO,
    ProfileEvidenceDTO,
    RelationshipHypothesisDTO,
    ScopeDTO,
    VLMPhotoObservationDTO,
)
from memory_module.evaluation import ErrorBucketDTO, EvaluationRunDTO, MetricSnapshotDTO
from memory_module.records import (
    MilvusSegmentRecord,
    Neo4jEventNodeRecord,
    Neo4jMovementNodeRecord,
    Neo4jPersonNodeRecord,
    Neo4jPhotoNodeRecord,
    Neo4jRelationshipEdgeRecord,
    Neo4jSessionNodeRecord,
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
        relationship_hypotheses = self._build_relationships(
            relationships=relationships,
            person_uuid_map=person_uuid_map,
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
            relationship_hypotheses=relationship_hypotheses,
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
                "profile_field_count": len(storage["redis"]["profile_core"]["fields"]),
                "segment_count": len(storage["milvus"]["segments"]),
                "generated_at": generated_at,
            },
            "envelope": self._serialize(envelope),
            "storage": storage,
            "transparency": transparency,
            "evaluation": evaluation,
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

    def _build_relationships(
        self,
        relationships: Sequence[Relationship],
        person_uuid_map: Dict[str, str],
    ) -> List[RelationshipHypothesisDTO]:
        hypotheses: List[RelationshipHypothesisDTO] = []
        for index, relationship in enumerate(relationships, start=1):
            person_uuid = person_uuid_map.get(relationship.person_id, self._canonical_person_uuid(relationship.person_id))
            evidence = dict(relationship.evidence or {})
            evidence_refs = []
            for sample in evidence.get("sample_scenes", []):
                timestamp = sample.get("timestamp")
                if timestamp:
                    evidence_refs.append({"ref_type": "relationship_scene", "ref_id": str(timestamp)})
            hypotheses.append(
                RelationshipHypothesisDTO(
                    relationship_id=f"relationship_{index:03d}",
                    upstream_ref={"object_type": "relationship_hypothesis", "object_id": relationship.person_id},
                    face_person_id=relationship.person_id,
                    person_uuid=person_uuid,
                    relationship_type=relationship.relationship_type,
                    label=relationship.label,
                    confidence=float(relationship.confidence or 0.0),
                    evidence=evidence,
                    evidence_refs=evidence_refs,
                    reason=relationship.reason,
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
                    field_value=f"{relationship.face_person_id}:{relationship.label}",
                    category="relationship",
                    confidence=relationship.confidence,
                    supporting_person_uuids=[relationship.person_uuid],
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
            append_change("relationship_hypothesis", relationship.relationship_id, f"inferred {relationship.label}", {"confidence": relationship.confidence})
        grouped_profile_fields = Counter(item.field_key for item in profile_evidence)
        for field_key, count in grouped_profile_fields.items():
            append_change("profile_field", field_key, f"published {count} evidence items for {field_key}")
        return changes

    def _build_materialization_bundle(
        self,
        photos_dto: Sequence[PhotoFactDTO],
        event_candidates: Sequence[EventCandidateDTO],
        relationship_hypotheses: Sequence[RelationshipHypothesisDTO],
        profile_evidence: Sequence[ProfileEvidenceDTO],
    ) -> MaterializationInputBundle:
        return MaterializationInputBundle(
            user_id=self.user_id,
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
        relationship_hypotheses: Sequence[RelationshipHypothesisDTO],
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

        neo4j_nodes = {
            "user": [
                Neo4jUserNodeRecord(
                    user_id=self.user_id,
                    tenant_id=scope.tenant_id,
                    properties={
                        "task_id": self.task_id,
                        "primary_face_person_id": primary_face_person_id,
                        "primary_person_uuid": primary_person_uuid,
                    },
                )
            ],
            "persons": [
                Neo4jPersonNodeRecord(
                    person_uuid=record.person_uuid,
                    labels=["Person", "PrimaryUser"] if record.person_uuid == primary_person_uuid else ["Person"],
                    properties={
                        "face_person_id": record.face_person_id,
                        "is_primary_user": record.person_uuid == primary_person_uuid,
                    },
                )
                for record in people_records
            ],
            "photos": [
                Neo4jPhotoNodeRecord(
                    photo_uuid=photo.photo_uuid,
                    properties={
                        "photo_id": photo.photo_id,
                        "filename": photo.filename,
                        "captured_at": photo.captured_at_utc,
                        "location": photo.location,
                        "contains_primary_user": bool(
                            primary_person_uuid and any(face.person_uuid == primary_person_uuid for face in photo.faces)
                        ),
                    },
                )
                for photo in photos
            ],
            "sessions": [
                Neo4jSessionNodeRecord(
                    session_uuid=session.session_uuid,
                    properties={
                        "session_id": session.session_id,
                        "day_key": session.day_key,
                        "location_hint": session.location_hint,
                    },
                )
                for session in sequences.sessions
            ],
            "movements": [
                Neo4jMovementNodeRecord(
                    movement_uuid=movement.movement_uuid,
                    properties={
                        "movement_id": movement.movement_id,
                        "distance_km": movement.distance_km,
                    },
                )
                for movement in sequences.movements
            ],
            "timelines": [
                Neo4jTimelineNodeRecord(
                    timeline_uuid=timeline.timeline_uuid,
                    properties={
                        "timeline_id": timeline.timeline_id,
                        "day_key": timeline.day_key,
                    },
                )
                for timeline in sequences.day_timelines
            ],
            "events": [
                Neo4jEventNodeRecord(
                    event_uuid=event.event_uuid,
                    properties={
                        "event_id": event.event_id,
                        "title": event.title,
                        "event_type": event.event_type,
                        "location": event.location,
                        "confidence": event.confidence,
                        "has_primary_user": bool(primary_person_uuid and primary_person_uuid in event.participant_person_uuids),
                    },
                )
                for event in event_candidates
            ],
        }

        neo4j_edges = []
        if primary_person_uuid:
            neo4j_edges.append(
                Neo4jRelationshipEdgeRecord(
                    edge_id=self._stable_uuid("edge", "primary-user", self.user_id, primary_person_uuid),
                    from_id=self.user_id,
                    to_id=primary_person_uuid,
                    edge_type="PRIMARY_USER",
                    properties={"face_person_id": primary_face_person_id},
                )
            )
        for photo in photos:
            neo4j_edges.append(
                Neo4jRelationshipEdgeRecord(
                    edge_id=self._stable_uuid("edge", "user-photo", photo.photo_uuid),
                    from_id=self.user_id,
                    to_id=photo.photo_uuid,
                    edge_type="HAS_PHOTO",
                )
            )
            for face in photo.faces:
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "photo-person", photo.photo_uuid, face.person_uuid),
                        from_id=photo.photo_uuid,
                        to_id=face.person_uuid,
                        edge_type="VISIBLE_IN",
                        properties={"face_id": face.face_id, "quality_score": face.quality_score},
                    )
                )
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "person-photo", face.person_uuid, photo.photo_uuid, face.face_id),
                        from_id=face.person_uuid,
                        to_id=photo.photo_uuid,
                        edge_type="APPEARS_IN",
                        properties={"face_id": face.face_id, "quality_score": face.quality_score},
                    )
                )

        for session in sequences.sessions:
            for photo_uuid in session.photo_uuids:
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "photo-session", photo_uuid, session.session_uuid),
                        from_id=photo_uuid,
                        to_id=session.session_uuid,
                        edge_type="IN_SESSION",
                    )
                )
            for person_uuid in session.dominant_person_uuids:
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "session-person", session.session_uuid, person_uuid),
                        from_id=session.session_uuid,
                        to_id=person_uuid,
                        edge_type="HAS_DOMINANT_PERSON",
                    )
                )

        for movement in sequences.movements:
            neo4j_edges.append(
                Neo4jRelationshipEdgeRecord(
                    edge_id=self._stable_uuid("edge", "movement", movement.from_session_uuid, movement.to_session_uuid),
                    from_id=movement.from_session_uuid,
                    to_id=movement.to_session_uuid,
                    edge_type="NEXT_SESSION",
                    properties={"distance_km": movement.distance_km},
                )
            )

        for timeline in sequences.day_timelines:
            for session_uuid in timeline.session_uuids:
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "timeline-session", timeline.timeline_uuid, session_uuid),
                        from_id=timeline.timeline_uuid,
                        to_id=session_uuid,
                        edge_type="PART_OF_DAY_TIMELINE",
                    )
                )

        for event in event_candidates:
            if not primary_person_uuid:
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "user-event", self.user_id, event.event_uuid),
                        from_id=self.user_id,
                        to_id=event.event_uuid,
                        edge_type="OBSERVED_EVENT",
                        properties={"event_type": event.event_type, "confidence": event.confidence},
                    )
                )
            for person_uuid in event.participant_person_uuids:
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "person-event", person_uuid, event.event_uuid),
                        from_id=person_uuid,
                        to_id=event.event_uuid,
                        edge_type="PARTICIPATED_IN",
                        properties={"event_type": event.event_type, "confidence": event.confidence},
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
            for photo_uuid in event.photo_uuids:
                neo4j_edges.append(
                    Neo4jRelationshipEdgeRecord(
                        edge_id=self._stable_uuid("edge", "event-photo", event.event_uuid, photo_uuid),
                        from_id=event.event_uuid,
                        to_id=photo_uuid,
                        edge_type="EVIDENCED_BY_PHOTO",
                    )
                )

        for relationship in relationship_hypotheses:
            from_id = primary_person_uuid or self.user_id
            neo4j_edges.append(
                Neo4jRelationshipEdgeRecord(
                    edge_id=self._stable_uuid("edge", "relationship", from_id, relationship.person_uuid),
                    from_id=from_id,
                    to_id=relationship.person_uuid,
                    edge_type="RELATIONSHIP_HYPOTHESIS",
                    properties={
                        "relationship_type": relationship.relationship_type,
                        "label": relationship.label,
                        "confidence": relationship.confidence,
                        "evidence_refs": relationship.evidence_refs,
                    },
                )
            )

        milvus_segments = self._build_milvus_segments(
            photos=photos,
            sequences=sequences,
            event_candidates=event_candidates,
            profile_evidence=profile_evidence,
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
            key=lambda item: (-item.confidence, item.face_person_id),
        ):
            add_node(
                relationship.person_uuid,
                f"{relationship.face_person_id} · {relationship.label}",
                "related_person",
                1,
                metadata={
                    "face_person_id": relationship.face_person_id,
                    "relationship_type": relationship.relationship_type,
                },
            )
            add_edge(
                center_node_id,
                relationship.person_uuid,
                "RELATIONSHIP_HYPOTHESIS",
                label=relationship.label,
                confidence=relationship.confidence,
                metadata={"relationship_type": relationship.relationship_type},
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
        profile_evidence: Sequence[ProfileEvidenceDTO],
    ) -> List[MilvusSegmentRecord]:
        segments: List[MilvusSegmentRecord] = []

        for photo in photos:
            if photo.vlm_observation and photo.vlm_observation.summary:
                segments.append(
                    self._segment_record(
                        photo_uuid=photo.photo_uuid,
                        segment_type="scene",
                        text=photo.vlm_observation.summary,
                        session_uuid=self._photo_session_uuid(photo.photo_id, sequences.sessions),
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
                        evidence_refs=[{"ref_type": "photo_person_observation", "ref_id": person.observation_id}],
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
                        evidence_refs=[{"ref_type": "session", "ref_id": session.session_id}],
                    )
                )

        for event in event_candidates:
            text = event.narrative_synthesis or event.description or event.title
            if text:
                segments.append(
                    self._segment_record(
                        photo_uuid=event.photo_uuids[0] if event.photo_uuids else "",
                        segment_type="event_narrative",
                        text=text,
                        event_uuid=event.event_uuid,
                        session_uuid=event.session_uuids[0] if event.session_uuids else None,
                        evidence_refs=event.evidence_refs,
                    )
                )

        for item in profile_evidence:
            if item.supporting_event_uuids:
                anchor_event_uuid = item.supporting_event_uuids[0]
            else:
                anchor_event_uuid = None
            segments.append(
                self._segment_record(
                    photo_uuid="",
                    segment_type="profile_evidence_snippet",
                    text=f"{item.field_key}: {item.field_value}",
                    event_uuid=anchor_event_uuid,
                    person_uuid=item.supporting_person_uuids[0] if item.supporting_person_uuids else None,
                    evidence_refs=item.evidence_refs,
                )
            )

        return segments

    def _segment_record(
        self,
        photo_uuid: str,
        segment_type: str,
        text: str,
        event_uuid: Optional[str] = None,
        person_uuid: Optional[str] = None,
        session_uuid: Optional[str] = None,
        evidence_refs: Optional[List[Dict[str, str]]] = None,
    ) -> MilvusSegmentRecord:
        return MilvusSegmentRecord(
            segment_uuid=self._stable_uuid("segment", segment_type, photo_uuid, event_uuid or "", person_uuid or "", text[:80]),
            tenant_id=None,
            user_id=self.user_id,
            photo_uuid=photo_uuid,
            event_uuid=event_uuid,
            person_uuid=person_uuid,
            session_uuid=session_uuid,
            segment_type=segment_type,
            text=text,
            sparse_terms=self._tokenize(text),
            importance_score=min(1.0, 0.35 + (len(text) / 200.0)),
            evidence_refs=evidence_refs or [],
        )

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
                "relationship_id": relationship.relationship_id,
                "face_person_id": relationship.face_person_id,
                "person_uuid": relationship.person_uuid,
                "relationship_type": relationship.relationship_type,
                "label": relationship.label,
                "confidence": relationship.confidence,
                "reason": relationship.reason,
                "evidence_refs": relationship.evidence_refs,
            }
            for relationship in relationships
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
