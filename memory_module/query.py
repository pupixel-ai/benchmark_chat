"""Agent-facing memory query planning and answer synthesis."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional
from uuid import NAMESPACE_URL, uuid5

from memory_module.dto import (
    AgentMemoryQueryRequestDTO,
    AnswerDTO,
    EntityRecallCandidateDTO,
    GraphDebugTraceDTO,
    OperatorPlanDTO,
    QueryDSLDTO,
    TimeScopeDTO,
)
from memory_module.embeddings import EmbeddingProvider, cosine_similarity
from memory_module.ontology import canonical_concept_names, concept_metadata, normalize_concept


NODE_GROUP_ID_FIELDS = {
    "user": "user_id",
    "persons": "person_uuid",
    "places": "place_uuid",
    "sessions": "session_uuid",
    "timelines": "timeline_uuid",
    "events": "event_uuid",
    "relationship_hypotheses": "relationship_uuid",
    "mood_states": "mood_uuid",
    "primary_person_hypotheses": "primary_person_hypothesis_uuid",
    "period_hypotheses": "period_uuid",
    "concepts": "concept_uuid",
}


class MemoryQueryService:
    """Transforms natural-language agent questions into graph-backed answers."""

    def __init__(
        self,
        now: Optional[datetime] = None,
        vector_dim: int = 32,
        embedder: Optional[EmbeddingProvider] = None,
    ) -> None:
        self.now = now or datetime.now()
        self.embedder = embedder or EmbeddingProvider.from_config(dim=vector_dim)
        self.vector_dim = self.embedder.dim

    def answer(
        self,
        memory_payload: Dict[str, Any],
        question: str,
        *,
        user_id: Optional[str] = None,
        context_hints: Optional[Dict[str, Any]] = None,
        time_hint: Optional[str] = None,
        answer_shape_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        memory = memory_payload.get("memory", memory_payload)
        storage = memory.get("storage", {})
        storage_graph = storage.get("neo4j", {})
        request = AgentMemoryQueryRequestDTO(
            user_id=user_id or memory.get("envelope", {}).get("scope", {}).get("user_id") or "",
            question=question,
            query_id=str(uuid5(NAMESPACE_URL, question)),
            context_hints=dict(context_hints or {}),
            time_hint=time_hint,
            answer_shape_hint=answer_shape_hint,
        )
        indexes = self._build_indexes(storage_graph)
        segments = storage.get("milvus", {}).get("segments", [])
        operator_plan = self._build_operator_plan(question, indexes, time_hint=time_hint, answer_shape_hint=answer_shape_hint)
        recall_candidates = self._recall(indexes, question, operator_plan)
        dsl = self._build_dsl(operator_plan, recall_candidates, indexes)
        answer = self._execute(memory, indexes, segments, operator_plan, dsl, question)
        trace = GraphDebugTraceDTO(
            operator_plan=self._serialize(operator_plan),
            recall_candidates=[self._serialize(item) for item in recall_candidates],
            dsl=self._serialize(dsl),
            executed_cypher=self._pseudo_cypher(operator_plan, dsl),
            evidence_fill={
                "segment_count": len(answer.evidence_segment_ids),
                "segment_ids": answer.evidence_segment_ids,
            },
        )
        return {
            "request": self._serialize(request),
            "answer": self._serialize(answer),
            "debug_trace": self._serialize(trace),
        }

    def _build_operator_plan(
        self,
        question: str,
        indexes: Dict[str, Any],
        *,
        time_hint: Optional[str],
        answer_shape_hint: Optional[str],
    ) -> OperatorPlanDTO:
        normalized = str(question or "").strip().lower()
        time_scope = self._resolve_time_scope(question, indexes, hint=time_hint)
        ordinal = self._extract_ordinal(normalized)
        threshold = self._extract_threshold(normalized)
        target_concepts = self._extract_target_concepts(question)
        target_entities = []
        output_shape = answer_shape_hint or ("ranked_list" if ordinal else "summary")

        if any(token in normalized for token in ("心情", "mood", "开心", "难过")):
            intent = "mood_lookup"
            metric = "latest_mood"
            group_by = None
        elif any(token in normalized for token in ("好友", "best friend", "第", "top")) and any(
            token in normalized for token in ("朋友", "friend", "好友")
        ):
            intent = "relationship_rank_query"
            metric = "friendship_score"
            group_by = "person"
        elif any(token in normalized for token in ("父亲", "爸爸", "father", "dad", "关系", "relationship")):
            intent = "relationship_explore"
            metric = "relationship_confidence"
            group_by = "relationship"
        else:
            intent = "event_search"
            metric = "event_confidence"
            group_by = "event"

        if "大学" in question or "college" in normalized or "campus" in normalized:
            target_concepts.extend(item for item in ["college_period", "campus"] if item not in target_concepts)

        return OperatorPlanDTO(
            intent=intent,
            time_scope=time_scope,
            ordinal=ordinal,
            threshold=threshold,
            group_by=group_by,
            metric=metric,
            target_concepts=target_concepts,
            target_entities=target_entities,
            output_shape=output_shape,
            fallback_policy="concept_vector_then_graph",
        )

    def _resolve_time_scope(self, question: str, indexes: Dict[str, Any], hint: Optional[str] = None) -> TimeScopeDTO:
        raw_text = hint or question
        normalized = raw_text.lower()
        if "过去" in raw_text and "个月" in raw_text:
            months = self._extract_number_between(raw_text, "过去", "个月") or 1
            end_at = self.now
            start_at = self._subtract_months(end_at, months)
            return TimeScopeDTO(
                raw_text=raw_text,
                start_at=start_at.isoformat(),
                end_at=end_at.isoformat(),
                resolution="relative_month_range",
                confidence=0.95,
            )
        if "最近一次" in raw_text or "latest" in normalized or "most recent" in normalized:
            start_at = self.now - timedelta(days=90)
            return TimeScopeDTO(
                raw_text=raw_text,
                start_at=start_at.isoformat(),
                end_at=self.now.isoformat(),
                resolution="recent_single",
                confidence=0.8,
            )
        if "最近" in raw_text or "recent" in normalized:
            for period in indexes["nodes_by_group"].get("period_hypotheses", []):
                props = period.get("properties", {})
                if props.get("period_type") == "recent_period":
                    return TimeScopeDTO(
                        raw_text=raw_text,
                        start_at=str(props.get("window_start") or ""),
                        end_at=str(props.get("window_end") or ""),
                        resolution="period_hypothesis",
                        confidence=float(props.get("confidence") or 0.8),
                    )
            start_at = self.now - timedelta(days=30)
            return TimeScopeDTO(
                raw_text=raw_text,
                start_at=start_at.isoformat(),
                end_at=self.now.isoformat(),
                resolution="fallback_recent_30d",
                confidence=0.6,
            )
        if "前年春天" in raw_text:
            target_year = self.now.year - 2
            return TimeScopeDTO(
                raw_text=raw_text,
                start_at=f"{target_year}-03-01T00:00:00",
                end_at=f"{target_year}-05-31T23:59:59",
                resolution="relative_named_period",
                confidence=0.9,
            )
        if "大学" in raw_text or "college" in normalized or "campus" in normalized:
            for period in indexes["nodes_by_group"].get("period_hypotheses", []):
                props = period.get("properties", {})
                if props.get("period_type") == "college_period":
                    return TimeScopeDTO(
                        raw_text=raw_text,
                        start_at=str(props.get("window_start") or ""),
                        end_at=str(props.get("window_end") or ""),
                        resolution="period_hypothesis",
                        confidence=float(props.get("confidence") or 0.8),
                    )
        return TimeScopeDTO(raw_text=raw_text, resolution="unbounded", confidence=0.4)

    def _extract_ordinal(self, normalized_question: str) -> Optional[int]:
        for marker in ("第", "top "):
            if marker == "第" and marker in normalized_question:
                digits = []
                started = False
                for char in normalized_question:
                    if char == "第":
                        started = True
                        continue
                    if started and char.isdigit():
                        digits.append(char)
                        continue
                    if digits:
                        break
                if digits:
                    return int("".join(digits))
            if marker in normalized_question:
                suffix = normalized_question.split(marker, 1)[1]
                digits = "".join(char for char in suffix if char.isdigit())
                if digits:
                    return int(digits)
        return None

    def _extract_threshold(self, normalized_question: str) -> Optional[float]:
        if "超过" in normalized_question and "次" in normalized_question:
            value = self._extract_number_between(normalized_question, "超过", "次")
            return float(value) if value is not None else None
        return None

    def _extract_target_concepts(self, question: str) -> List[str]:
        targets: List[str] = []
        for canonical_name in canonical_concept_names():
            meta = concept_metadata(canonical_name)
            aliases = [canonical_name, *[str(item) for item in meta.get("aliases", [])]]
            if any(alias.lower() in question.lower() for alias in aliases):
                if canonical_name not in targets:
                    targets.append(canonical_name)
        if not targets:
            for raw_token in question.replace("？", " ").replace("?", " ").split():
                concept = normalize_concept(raw_token)
                if concept and concept not in targets:
                    targets.append(concept)
        return targets

    def _recall(
        self,
        indexes: Dict[str, Any],
        question: str,
        operator_plan: OperatorPlanDTO,
    ) -> List[EntityRecallCandidateDTO]:
        candidates: List[EntityRecallCandidateDTO] = []
        query_vector = self.embedder.embed_query(question)

        for concept_node in indexes["nodes_by_group"].get("concepts", []):
            props = concept_node.get("properties", {})
            score = self._match_score(question, props, query_vector)
            if score < 0.25 and props.get("canonical_name") not in operator_plan.target_concepts:
                continue
            candidates.append(
                EntityRecallCandidateDTO(
                    entity_type="concept",
                    entity_id=str(concept_node.get("concept_uuid")),
                    score=score,
                    matched_concept=str(props.get("canonical_name") or ""),
                    source="exact" if score >= 0.95 else "index",
                    metadata={"canonical_name": props.get("canonical_name"), "concept_type": props.get("concept_type")},
                )
            )

        for group_name in ("events", "sessions", "relationship_hypotheses", "places", "persons"):
            for node in indexes["nodes_by_group"].get(group_name, []):
                props = node.get("properties", {})
                score = self._match_score(question, props, query_vector)
                if score < 0.3:
                    continue
                entity_id = node.get(NODE_GROUP_ID_FIELDS[group_name])
                candidates.append(
                    EntityRecallCandidateDTO(
                        entity_type=group_name.rstrip("s"),
                        entity_id=str(entity_id),
                        score=score,
                        matched_concept=None,
                        source="index",
                        metadata={"label": props.get("title") or props.get("canonical_name") or props.get("label")},
                    )
                )

        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:20]

    def _build_dsl(
        self,
        operator_plan: OperatorPlanDTO,
        recall_candidates: List[EntityRecallCandidateDTO],
        indexes: Dict[str, Any],
    ) -> QueryDSLDTO:
        concept_filters = list(operator_plan.target_concepts)
        for candidate in recall_candidates:
            if candidate.entity_type == "concept" and candidate.matched_concept and candidate.matched_concept not in concept_filters:
                concept_filters.append(candidate.matched_concept)

        graph_filters = {
            "concept_filters": concept_filters,
            "status_filters": ["active", "cooling"],
            "user_id": indexes["user_id"],
        }
        ranking_rule = {
            "ordinal": operator_plan.ordinal,
            "threshold": operator_plan.threshold,
            "metric": operator_plan.metric,
            "group_by": operator_plan.group_by,
        }
        evidence_fill = {
            "segment_types": [
                "event_narrative",
                "relationship_reason",
                "session_summary",
                "profile_evidence_snippet",
            ]
        }
        return QueryDSLDTO(
            intent=operator_plan.intent,
            graph_filters=graph_filters,
            time_scope=asdict(operator_plan.time_scope),
            ranking_rule=ranking_rule,
            evidence_fill=evidence_fill,
        )

    def _execute(
        self,
        memory: Dict[str, Any],
        indexes: Dict[str, Any],
        segments: List[Dict[str, Any]],
        operator_plan: OperatorPlanDTO,
        dsl: QueryDSLDTO,
        question: str,
    ) -> AnswerDTO:
        if operator_plan.intent == "mood_lookup":
            return self._answer_mood(indexes, segments, operator_plan)
        if operator_plan.intent == "relationship_explore":
            return self._answer_relationship(indexes, segments, operator_plan)
        if operator_plan.intent == "relationship_rank_query":
            return self._answer_relationship_rank(indexes, segments, operator_plan)
        return self._answer_events(indexes, segments, operator_plan, question)

    def _answer_events(
        self,
        indexes: Dict[str, Any],
        segments: List[Dict[str, Any]],
        operator_plan: OperatorPlanDTO,
        question: str,
    ) -> AnswerDTO:
        time_scope = operator_plan.time_scope
        concept_filters = set(operator_plan.target_concepts)
        events = []
        for event in indexes["nodes_by_group"].get("events", []):
            props = event.get("properties", {})
            if props.get("user_id") != indexes["user_id"]:
                continue
            if time_scope.start_at and not self._overlaps_time_scope(
                props.get("started_at"),
                props.get("ended_at") or props.get("started_at"),
                time_scope,
            ):
                continue
            event_concepts = set(self._outgoing_concepts(indexes, event.get("event_uuid")))
            normalized_event_type = str(props.get("normalized_event_type") or props.get("event_type") or "")
            if concept_filters and normalized_event_type not in concept_filters and not event_concepts.intersection(concept_filters):
                continue
            events.append(event)

        events.sort(key=lambda item: str(item.get("properties", {}).get("started_at") or ""), reverse=True)
        if "最近一次" in question or "latest" in question.lower() or "most recent" in question.lower():
            events = events[:1]

        supporting_events = [self._event_payload(indexes, event) for event in events]
        supporting_sessions = self._supporting_sessions_for_events(indexes, events)
        representative_photo_ids = self._unique(
            photo_id
            for event in events
            for photo_id in event.get("properties", {}).get("representative_photo_ids", [])
        )
        evidence_segment_ids = self._segment_ids_for_objects(
            segments,
            event_uuids=[event.get("event_uuid") for event in events],
            session_uuids=[session["session_uuid"] for session in supporting_sessions],
        )
        resolved_concepts = list(concept_filters)
        if events:
            summary = f"共找到 {len(events)} 个事件: " + "；".join(event["title"] for event in supporting_events[:5])
            confidence = round(sum(event["confidence"] for event in supporting_events) / len(supporting_events), 4)
        else:
            summary = "没有找到符合条件的事件。"
            confidence = 0.0
        explanation = "先按时间窗过滤事件，再按 canonical concepts 过滤，最后补充会话和语义证据。"
        uncertainty_flags = [] if events else ["no_matching_events"]
        return AnswerDTO(
            answer_type="event_search",
            summary=summary,
            confidence=confidence,
            resolved_entities=supporting_events,
            resolved_concepts=resolved_concepts,
            time_window=asdict(time_scope),
            supporting_sessions=supporting_sessions,
            supporting_events=supporting_events,
            representative_photo_ids=representative_photo_ids,
            evidence_segment_ids=evidence_segment_ids,
            explanation=explanation,
            uncertainty_flags=uncertainty_flags,
        )

    def _answer_relationship(
        self,
        indexes: Dict[str, Any],
        segments: List[Dict[str, Any]],
        operator_plan: OperatorPlanDTO,
    ) -> AnswerDTO:
        concept_filters = set(operator_plan.target_concepts)
        relationships = []
        for relationship in indexes["nodes_by_group"].get("relationship_hypotheses", []):
            props = relationship.get("properties", {})
            if props.get("status") not in {"active", "cooling"}:
                continue
            if concept_filters:
                rel_concepts = set(self._outgoing_concepts(indexes, relationship.get("relationship_uuid")))
                if not rel_concepts.intersection(concept_filters) and props.get("relationship_type") not in concept_filters:
                    continue
            relationships.append(relationship)

        relationships.sort(
            key=lambda item: (
                float(item.get("properties", {}).get("confidence") or 0.0),
                float(item.get("properties", {}).get("co_present_session_count") or 0.0),
            ),
            reverse=True,
        )
        primary = relationships[:1]
        supporting_relationships = [self._relationship_payload(indexes, item) for item in primary]
        supporting_sessions = self._supporting_sessions_for_relationships(indexes, primary)
        supporting_events = self._supporting_events_for_relationships(indexes, primary)
        evidence_segment_ids = self._segment_ids_for_objects(
            segments,
            relationship_uuids=[item.get("relationship_uuid") for item in primary],
            session_uuids=[item["session_uuid"] for item in supporting_sessions],
            event_uuids=[item["event_uuid"] for item in supporting_events],
        )
        representative_photo_ids = self._unique(
            photo_id
            for item in supporting_relationships
            for photo_id in item.get("representative_photo_ids", [])
        )
        if supporting_relationships:
            top = supporting_relationships[0]
            summary = f"{top['target_face_person_id']} 当前最像 {top['label']}，置信度 {top['confidence']:.2f}。"
            confidence = top["confidence"]
            uncertainty_flags = []
        else:
            summary = "没有找到符合条件的关系假设。"
            confidence = 0.0
            uncertainty_flags = ["no_matching_relationship"]
        return AnswerDTO(
            answer_type="relationship_explore",
            summary=summary,
            confidence=confidence,
            resolved_entities=supporting_relationships,
            resolved_concepts=list(concept_filters),
            time_window=asdict(operator_plan.time_scope),
            supporting_sessions=supporting_sessions,
            supporting_events=supporting_events,
            supporting_relationships=supporting_relationships,
            representative_photo_ids=representative_photo_ids,
            evidence_segment_ids=evidence_segment_ids,
            explanation="先找 active/cooling 的关系修订版，再沿支持的会话和事件展开证据。",
            uncertainty_flags=uncertainty_flags,
        )

    def _answer_relationship_rank(
        self,
        indexes: Dict[str, Any],
        segments: List[Dict[str, Any]],
        operator_plan: OperatorPlanDTO,
    ) -> AnswerDTO:
        time_scope = operator_plan.time_scope
        relationships = []
        for relationship in indexes["nodes_by_group"].get("relationship_hypotheses", []):
            props = relationship.get("properties", {})
            if props.get("status") not in {"active", "cooling"}:
                continue
            if props.get("relationship_type") not in {"friend", "close_friend"}:
                continue
            if time_scope.start_at and props.get("window_end") and not self._overlaps_time_scope(
                props.get("window_start"),
                props.get("window_end"),
                time_scope,
            ):
                continue
            if operator_plan.threshold is not None and float(props.get("co_present_session_count") or 0.0) <= operator_plan.threshold:
                continue
            relationships.append(relationship)

        relationships.sort(
            key=lambda item: (
                float(item.get("properties", {}).get("feature_snapshot", {}).get("co_present_session_count", 0.0)),
                float(item.get("properties", {}).get("score_snapshot", {}).get("close_friend", 0.0)),
                float(item.get("properties", {}).get("confidence") or 0.0),
            ),
            reverse=True,
        )
        if operator_plan.ordinal and operator_plan.ordinal > 0:
            ranked = relationships[operator_plan.ordinal - 1 : operator_plan.ordinal]
        else:
            limit = operator_plan.ordinal or 3
            ranked = relationships[:limit]

        supporting_relationships = [self._relationship_payload(indexes, item) for item in ranked]
        supporting_sessions = self._supporting_sessions_for_relationships(indexes, ranked)
        supporting_events = self._supporting_events_for_relationships(indexes, ranked)
        evidence_segment_ids = self._segment_ids_for_objects(
            segments,
            relationship_uuids=[item.get("relationship_uuid") for item in ranked],
            session_uuids=[item["session_uuid"] for item in supporting_sessions],
            event_uuids=[item["event_uuid"] for item in supporting_events],
        )
        representative_photo_ids = self._unique(
            photo_id
            for rel in supporting_relationships
            for photo_id in rel.get("representative_photo_ids", [])
        )
        if supporting_relationships:
            summary = "；".join(
                f"{item['target_face_person_id']} ({item['label']}, {item['confidence']:.2f})"
                for item in supporting_relationships
            )
            confidence = round(sum(item["confidence"] for item in supporting_relationships) / len(supporting_relationships), 4)
            uncertainty_flags = []
        else:
            summary = "没有找到符合条件的好友排序结果。"
            confidence = 0.0
            uncertainty_flags = ["no_ranked_relationships"]
        return AnswerDTO(
            answer_type="relationship_rank_query",
            summary=summary,
            confidence=confidence,
            resolved_entities=supporting_relationships,
            resolved_concepts=["friend", "close_friend"],
            time_window=asdict(time_scope),
            supporting_sessions=supporting_sessions,
            supporting_events=supporting_events,
            supporting_relationships=supporting_relationships,
            representative_photo_ids=representative_photo_ids,
            evidence_segment_ids=evidence_segment_ids,
            explanation="按活动窗口筛选 active 关系修订版，并按共现次数、亲密度分数和置信度排序。",
            uncertainty_flags=uncertainty_flags,
        )

    def _answer_mood(
        self,
        indexes: Dict[str, Any],
        segments: List[Dict[str, Any]],
        operator_plan: OperatorPlanDTO,
    ) -> AnswerDTO:
        time_scope = operator_plan.time_scope
        moods = []
        for mood in indexes["nodes_by_group"].get("mood_states", []):
            props = mood.get("properties", {})
            if time_scope.start_at and not self._overlaps_time_scope(props.get("window_start"), props.get("window_end"), time_scope):
                continue
            moods.append(mood)

        moods.sort(key=lambda item: str(item.get("properties", {}).get("window_end") or ""), reverse=True)
        top = moods[:1]
        supporting_sessions = []
        for mood in top:
            session_uuid = mood.get("properties", {}).get("session_uuid")
            if session_uuid and session_uuid in indexes["nodes"]:
                supporting_sessions.append(self._session_payload(indexes["nodes"][session_uuid]))
        evidence_segment_ids = self._segment_ids_for_objects(
            segments,
            session_uuids=[item["session_uuid"] for item in supporting_sessions],
        )
        if top:
            props = top[0]["properties"]
            summary = f"最近的情绪更接近 {props.get('mood_label')}，分数 {float(props.get('mood_score') or 0.0):.2f}。"
            confidence = float(props.get("confidence") or 0.0)
            resolved_concepts = self._outgoing_concepts(indexes, top[0].get("mood_uuid"))
            uncertainty_flags = []
        else:
            summary = "没有找到最近的情绪状态。"
            confidence = 0.0
            resolved_concepts = []
            uncertainty_flags = ["no_mood_hypothesis"]
        return AnswerDTO(
            answer_type="mood_lookup",
            summary=summary,
            confidence=confidence,
            resolved_entities=[self._serialize(self._node_payload(item)) for item in top],
            resolved_concepts=resolved_concepts,
            time_window=asdict(time_scope),
            supporting_sessions=supporting_sessions,
            evidence_segment_ids=evidence_segment_ids,
            explanation="优先读取 mood hypotheses，并用会话级证据片段回填解释。",
            uncertainty_flags=uncertainty_flags,
        )

    def _build_indexes(self, neo4j_payload: Dict[str, Any]) -> Dict[str, Any]:
        nodes_by_group = neo4j_payload.get("nodes", {})
        nodes: Dict[str, Dict[str, Any]] = {}
        user_id = ""
        for group_name, records in nodes_by_group.items():
            id_field = NODE_GROUP_ID_FIELDS.get(group_name)
            if not id_field:
                continue
            for record in records:
                node_id = str(record.get(id_field))
                nodes[node_id] = {"group": group_name, **record}
                if group_name == "user" and not user_id:
                    user_id = node_id

        outgoing: Dict[str, List[Dict[str, Any]]] = {}
        incoming: Dict[str, List[Dict[str, Any]]] = {}
        for edge in neo4j_payload.get("edges", []):
            outgoing.setdefault(str(edge.get("from_id")), []).append(edge)
            incoming.setdefault(str(edge.get("to_id")), []).append(edge)

        return {
            "user_id": user_id,
            "nodes_by_group": nodes_by_group,
            "nodes": nodes,
            "outgoing": outgoing,
            "incoming": incoming,
        }

    def _match_score(self, question: str, props: Dict[str, Any], query_vector: List[float]) -> float:
        aliases = [str(item).lower() for item in props.get("aliases", [])]
        candidates = [
            str(props.get("canonical_name") or "").lower(),
            str(props.get("title") or "").lower(),
            str(props.get("label") or "").lower(),
            str(props.get("search_text") or "").lower(),
            *aliases,
        ]
        if any(candidate and candidate in question.lower() for candidate in candidates):
            return 0.99
        embedding = props.get("embedding")
        if isinstance(embedding, list):
            return max(0.0, cosine_similarity(query_vector, [float(item) for item in embedding]))
        return 0.0

    def _pseudo_cypher(self, operator_plan: OperatorPlanDTO, dsl: QueryDSLDTO) -> str:
        if operator_plan.intent == "event_search":
            return (
                "MATCH (e:Event)-[:HAS_CONCEPT]->(c:Concept) "
                "WHERE e.user_id = $user_id AND e.started_at >= $start_at AND c.canonical_name IN $concepts "
                "RETURN e ORDER BY e.started_at DESC"
            )
        if operator_plan.intent == "relationship_rank_query":
            return (
                "MATCH (p:Person)-[:HAS_RELATIONSHIP]->(r:RelationshipHypothesis) "
                "WHERE r.status IN ['active','cooling'] "
                "RETURN r ORDER BY r.co_present_session_count DESC"
            )
        if operator_plan.intent == "mood_lookup":
            return "MATCH (m:MoodStateHypothesis)-[:DESCRIBES_SESSION]->(s:Session) RETURN m, s ORDER BY m.window_end DESC"
        return (
            "MATCH (p:Person)-[:HAS_RELATIONSHIP]->(r:RelationshipHypothesis) "
            "WHERE r.status IN ['active','cooling'] RETURN r"
        )

    def _event_payload(self, indexes: Dict[str, Any], node: Dict[str, Any]) -> Dict[str, Any]:
        props = node.get("properties", {})
        participants = self._participants_for_event(indexes, node.get("event_uuid"))
        return {
            "event_uuid": node.get("event_uuid"),
            "title": props.get("title"),
            "normalized_event_type": props.get("normalized_event_type"),
            "started_at": props.get("started_at"),
            "ended_at": props.get("ended_at"),
            "confidence": float(props.get("confidence") or 0.0),
            "participants": participants,
            "representative_photo_ids": props.get("representative_photo_ids", []),
        }

    def _relationship_payload(self, indexes: Dict[str, Any], node: Dict[str, Any]) -> Dict[str, Any]:
        props = node.get("properties", {})
        target_face_person_id = props.get("target_face_person_id")
        if not target_face_person_id:
            target_node = self._first_neighbor(indexes, node.get("relationship_uuid"), "TARGET_PERSON")
            target_face_person_id = target_node.get("properties", {}).get("face_person_id") if target_node else None
        representative_photo_ids = self._unique(
            photo_id
            for session in self._supporting_sessions_for_relationships(indexes, [node])
            for photo_id in session.get("representative_photo_ids", [])
        )
        return {
            "relationship_uuid": node.get("relationship_uuid"),
            "label": props.get("label"),
            "relationship_type": props.get("relationship_type"),
            "confidence": float(props.get("confidence") or 0.0),
            "target_face_person_id": target_face_person_id,
            "window_start": props.get("window_start"),
            "window_end": props.get("window_end"),
            "feature_snapshot": props.get("feature_snapshot", {}),
            "score_snapshot": props.get("score_snapshot", {}),
            "representative_photo_ids": representative_photo_ids,
        }

    def _session_payload(self, node: Dict[str, Any]) -> Dict[str, Any]:
        props = node.get("properties", {})
        return {
            "session_uuid": node.get("session_uuid"),
            "started_at": props.get("started_at"),
            "ended_at": props.get("ended_at"),
            "place_uuid": props.get("place_uuid"),
            "participant_count": props.get("participant_count"),
            "representative_photo_ids": props.get("representative_photo_ids", []),
        }

    def _node_payload(self, node: Dict[str, Any]) -> Dict[str, Any]:
        id_field = NODE_GROUP_ID_FIELDS.get(node.get("group", ""))
        payload = {id_field: node.get(id_field)} if id_field else {}
        payload.update(node.get("properties", {}))
        return payload

    def _participants_for_event(self, indexes: Dict[str, Any], event_uuid: str) -> List[str]:
        session_uuids = [
            edge.get("to_id")
            for edge in indexes["outgoing"].get(str(event_uuid), [])
            if edge.get("edge_type") == "DERIVED_FROM_SESSION"
        ]
        participants = []
        for session_uuid in session_uuids:
            for edge in indexes["incoming"].get(str(session_uuid), []):
                if edge.get("edge_type") != "CO_PRESENT_IN":
                    continue
                node = indexes["nodes"].get(str(edge.get("from_id")))
                if not node:
                    continue
                face_person_id = node.get("properties", {}).get("face_person_id")
                if face_person_id and face_person_id not in participants:
                    participants.append(face_person_id)
        return participants

    def _supporting_sessions_for_events(self, indexes: Dict[str, Any], events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sessions = []
        seen = set()
        for event in events:
            for edge in indexes["outgoing"].get(str(event.get("event_uuid")), []):
                if edge.get("edge_type") != "DERIVED_FROM_SESSION":
                    continue
                session_node = indexes["nodes"].get(str(edge.get("to_id")))
                if session_node and session_node.get("session_uuid") not in seen:
                    seen.add(session_node.get("session_uuid"))
                    sessions.append(self._session_payload(session_node))
        return sessions

    def _supporting_sessions_for_relationships(self, indexes: Dict[str, Any], relationships: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sessions = []
        seen = set()
        for relationship in relationships:
            for edge in indexes["outgoing"].get(str(relationship.get("relationship_uuid")), []):
                if edge.get("edge_type") != "SUPPORTED_BY_SESSION":
                    continue
                session_node = indexes["nodes"].get(str(edge.get("to_id")))
                if session_node and session_node.get("session_uuid") not in seen:
                    seen.add(session_node.get("session_uuid"))
                    sessions.append(self._session_payload(session_node))
        return sessions

    def _supporting_events_for_relationships(self, indexes: Dict[str, Any], relationships: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        events = []
        seen = set()
        for relationship in relationships:
            for edge in indexes["outgoing"].get(str(relationship.get("relationship_uuid")), []):
                if edge.get("edge_type") != "SUPPORTED_BY_EVENT":
                    continue
                event_node = indexes["nodes"].get(str(edge.get("to_id")))
                if event_node and event_node.get("event_uuid") not in seen:
                    seen.add(event_node.get("event_uuid"))
                    events.append(self._event_payload(indexes, event_node))
        return events

    def _segment_ids_for_objects(
        self,
        segments: List[Dict[str, Any]],
        *,
        event_uuids: Optional[List[str]] = None,
        session_uuids: Optional[List[str]] = None,
        relationship_uuids: Optional[List[str]] = None,
    ) -> List[str]:
        event_set = {str(item) for item in event_uuids or [] if item}
        session_set = {str(item) for item in session_uuids or [] if item}
        relationship_set = {str(item) for item in relationship_uuids or [] if item}
        selected = []
        for segment in segments:
            if event_set and str(segment.get("event_uuid") or "") in event_set:
                selected.append(str(segment.get("segment_uuid")))
                continue
            if session_set and str(segment.get("session_uuid") or "") in session_set:
                selected.append(str(segment.get("segment_uuid")))
                continue
            if relationship_set and str(segment.get("relationship_uuid") or "") in relationship_set:
                selected.append(str(segment.get("segment_uuid")))
        return self._unique(selected)

    def _outgoing_concepts(self, indexes: Dict[str, Any], source_id: Optional[str]) -> List[str]:
        concepts = []
        if not source_id:
            return concepts
        for edge in indexes["outgoing"].get(str(source_id), []):
            if edge.get("edge_type") != "HAS_CONCEPT":
                continue
            concept_node = indexes["nodes"].get(str(edge.get("to_id")))
            if not concept_node:
                continue
            canonical_name = concept_node.get("properties", {}).get("canonical_name")
            if canonical_name and canonical_name not in concepts:
                concepts.append(str(canonical_name))
        return concepts

    def _first_neighbor(self, indexes: Dict[str, Any], source_id: Optional[str], edge_type: str) -> Optional[Dict[str, Any]]:
        if not source_id:
            return None
        for edge in indexes["outgoing"].get(str(source_id), []):
            if edge.get("edge_type") == edge_type:
                return indexes["nodes"].get(str(edge.get("to_id")))
        return None

    def _in_time_scope(self, raw_started_at: Any, time_scope: TimeScopeDTO) -> bool:
        if not time_scope.start_at:
            return True
        started_at = self._parse_datetime(raw_started_at)
        scope_start = self._parse_datetime(time_scope.start_at)
        scope_end = self._parse_datetime(time_scope.end_at)
        if not started_at or not scope_start or not scope_end:
            return False
        return scope_start <= started_at <= scope_end

    def _overlaps_time_scope(self, raw_start: Any, raw_end: Any, time_scope: TimeScopeDTO) -> bool:
        if not time_scope.start_at:
            return True
        node_start = self._parse_datetime(raw_start)
        node_end = self._parse_datetime(raw_end)
        scope_start = self._parse_datetime(time_scope.start_at)
        scope_end = self._parse_datetime(time_scope.end_at)
        if not node_start or not node_end or not scope_start or not scope_end:
            return False
        return not (node_end < scope_start or scope_end < node_start)

    def _parse_datetime(self, raw_value: Any) -> Optional[datetime]:
        if not raw_value:
            return None
        try:
            return datetime.fromisoformat(str(raw_value))
        except ValueError:
            return None

    def _subtract_months(self, dt: datetime, months: int) -> datetime:
        year = dt.year
        month = dt.month - months
        while month <= 0:
            month += 12
            year -= 1
        day = min(dt.day, self._days_in_month(year, month))
        return dt.replace(year=year, month=month, day=day)

    def _days_in_month(self, year: int, month: int) -> int:
        if month == 2:
            leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
            return 29 if leap else 28
        if month in {4, 6, 9, 11}:
            return 30
        return 31

    def _extract_number_between(self, text: str, left: str, right: str) -> Optional[int]:
        if left not in text or right not in text:
            return None
        middle = text.split(left, 1)[1].split(right, 1)[0]
        digits = "".join(char for char in middle if char.isdigit())
        return int(digits) if digits else None

    def _unique(self, values: Iterable[str]) -> List[str]:
        ordered = []
        seen = set()
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            ordered.append(value)
        return ordered

    def _serialize(self, value: Any) -> Any:
        if is_dataclass(value):
            return {key: self._serialize(item) for key, item in asdict(value).items()}
        if isinstance(value, dict):
            return {key: self._serialize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._serialize(item) for item in value]
        return value
