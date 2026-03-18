"""Agent-facing memory query planning and answer synthesis."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
import re
from typing import Any, Dict, Iterable, List, Optional
from uuid import NAMESPACE_URL, uuid5

from memory_module.dto import (
    AgentMemoryQueryRequestDTO,
    AnswerDTO,
    EntityRecallCandidateDTO,
    GraphDebugTraceDTO,
    OperatorPlanDTO,
    QueryPlanDTO,
    QueryDSLDTO,
    TimeScopeDTO,
)
from memory_module.embeddings import EmbeddingProvider, cosine_similarity
from memory_module.ontology import canonical_concept_names, concept_metadata, expand_concepts, normalize_concept


NODE_GROUP_ID_FIELDS = {
    "user": "user_id",
    "persons": "person_uuid",
    "places": "place_uuid",
    "events": "session_uuid",
    "facts": "event_uuid",
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
        vector_dim: Optional[int] = None,
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
        redis_payload = storage.get("redis", {})
        request = AgentMemoryQueryRequestDTO(
            user_id=user_id or memory.get("envelope", {}).get("scope", {}).get("user_id") or "",
            question=question,
            query_id=str(uuid5(NAMESPACE_URL, question)),
            context_hints=dict(context_hints or {}),
            time_hint=time_hint,
            answer_shape_hint=answer_shape_hint,
        )
        indexes = self._build_indexes(storage_graph, redis_payload)
        segments = storage.get("milvus", {}).get("segments", [])
        operator_plan = self._build_operator_plan(question, indexes, time_hint=time_hint, answer_shape_hint=answer_shape_hint)
        query_plan = self._build_query_plan(question, operator_plan, context_hints=context_hints)
        recall_candidates = self._recall(indexes, question, operator_plan)
        dsl = self._build_dsl(operator_plan, recall_candidates, indexes)
        answer, execution_details = self._execute(memory, indexes, segments, operator_plan, query_plan, dsl, question)
        trace = GraphDebugTraceDTO(
            operator_plan=self._serialize(operator_plan),
            recall_candidates=[self._serialize(item) for item in recall_candidates],
            dsl=self._serialize(dsl),
            executed_cypher=self._pseudo_cypher(operator_plan, dsl),
            evidence_fill={
                "segment_count": len(answer.evidence_segment_ids),
                "segment_ids": answer.evidence_segment_ids,
                "source_path": execution_details["source_path"],
            },
        )
        return {
            "request": self._serialize(request),
            "query_plan": self._serialize(query_plan),
            "source_order": list(query_plan.source_order),
            "answer": self._serialize(answer),
            "structured_answer": self._serialize(answer),
            "supporting_redis_profiles": execution_details["supporting_redis_profiles"],
            "supporting_graph_entities": execution_details["supporting_graph_entities"],
            "supporting_segments": execution_details["supporting_segments"],
            "abstain_reason": execution_details["abstain_reason"],
            "debug_trace": self._serialize(trace),
        }

    def _build_query_plan(
        self,
        question: str,
        operator_plan: OperatorPlanDTO,
        *,
        context_hints: Optional[Dict[str, Any]] = None,
    ) -> QueryPlanDTO:
        normalized = str(question or "").lower()
        subject_binding = "authenticated_user" if any(token in question for token in ("我", "我的")) else "unknown"
        operators = self._infer_operators(normalized)
        source_order = self._source_order_for_question(normalized, operator_plan)
        target_spec = {
            "raw_question": question,
            "target_concepts": list(operator_plan.target_concepts),
            "target_entities": list(operator_plan.target_entities),
            "context_hints": dict(context_hints or {}),
        }
        constraints = {
            "time_scope": asdict(operator_plan.time_scope),
            "ordinal": operator_plan.ordinal,
            "threshold": operator_plan.threshold,
        }
        answer_schema = {
            "shape": operator_plan.output_shape,
            "group_by": operator_plan.group_by,
            "metric": operator_plan.metric,
        }
        evidence_requirements = self._evidence_requirements_for_source_order(source_order)
        return QueryPlanDTO(
            subject_binding=subject_binding,
            target_spec=target_spec,
            operators=operators,
            constraints=constraints,
            source_order=source_order,
            answer_schema=answer_schema,
            evidence_requirements=evidence_requirements,
        )

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

    def _infer_operators(self, normalized_question: str) -> List[str]:
        operators: List[str] = []
        keyword_map = {
            "locate": ("哪里", "哪儿", "在哪", "在哪儿", "去了哪"),
            "list": ("哪些", "有什么", "列举", "列出"),
            "count": ("几次", "多少次", "多少个"),
            "rank": ("top", "第", "最"),
            "compare": ("变化", "相比", "对比"),
            "trace_change": ("变化", "经历了怎样的变化", "轨迹"),
            "infer": ("是否", "是不是", "判定", "分析", "推断", "画像"),
            "recommend": ("推荐", "策划", "建议"),
            "summarize": ("总结", "概括", "描述"),
        }
        for operator, tokens in keyword_map.items():
            if any(token in normalized_question for token in tokens):
                operators.append(operator)
        if not operators:
            operators.append("summarize")
        return operators

    def _source_order_for_question(self, normalized_question: str, operator_plan: OperatorPlanDTO) -> List[str]:
        redis_tokens = (
            "喜欢",
            "偏好",
            "评价",
            "画像",
            "风格",
            "身份",
            "职业",
            "公司",
            "股权",
            "消费",
            "审美",
            "推荐",
            "轨迹",
            "变化",
        )
        milvus_tokens = (
            "品牌",
            "菜",
            "饮料",
            "价格",
            "围巾",
            "玩具",
            "鱼缸",
            "动作",
            "材质",
            "ocr",
            "票据",
            "产品",
            "药",
            "治疗",
            "路线",
        )
        if any(token in normalized_question for token in redis_tokens):
            return ["redis_first", "neo4j", "milvus"]
        if any(token in normalized_question for token in milvus_tokens):
            return ["milvus_first", "neo4j", "redis"]
        if operator_plan.intent in {"relationship_explore", "relationship_rank_query", "mood_lookup"}:
            return ["neo4j_first", "milvus", "redis"]
        return ["neo4j_first", "milvus", "redis"]

    def _evidence_requirements_for_source_order(self, source_order: List[str]) -> List[str]:
        requirements: List[str] = []
        if source_order and source_order[0] == "redis_first":
            requirements.extend(["materialized_profile_fields", "supporting_segments"])
        elif source_order and source_order[0] == "milvus_first":
            requirements.extend(["semantic_segments", "graph_backfill"])
        else:
            requirements.extend(["graph_entities", "supporting_segments"])
        return requirements

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
        normalized_question = question.lower()
        for canonical_name in canonical_concept_names():
            meta = concept_metadata(canonical_name)
            aliases = [canonical_name, *[str(item) for item in meta.get("aliases", [])]]
            if any(alias.lower() in normalized_question for alias in aliases):
                if canonical_name not in targets:
                    targets.append(canonical_name)
        if not targets:
            for raw_token in question.replace("？", " ").replace("?", " ").split():
                concept = normalize_concept(raw_token)
                if concept and concept not in targets:
                    targets.append(concept)
        live_music_patterns = (
            r"\blive\b",
            r"\bshow\b",
            r"\bgig\b",
            r"\bconcert\b",
            r"演唱会",
            r"音乐会",
            r"音乐节",
            r"巡演",
            r"现场演出",
            r"演出现场",
        )
        if any(re.search(pattern, normalized_question) for pattern in live_music_patterns):
            for concept_name in ("concert", "music_live_event"):
                if concept_name not in targets:
                    targets.append(concept_name)
        if any(token in normalized_question for token in ("festival", "音乐节")):
            if "music_festival_performance" not in targets:
                targets.append("music_festival_performance")
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

        for group_name in ("events", "facts", "relationship_hypotheses", "places", "persons"):
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
        concept_filters = expand_concepts(concept_filters)

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
        query_plan: QueryPlanDTO,
        dsl: QueryDSLDTO,
        question: str,
    ) -> tuple[AnswerDTO, Dict[str, Any]]:
        if operator_plan.intent == "mood_lookup":
            answer = self._answer_mood(indexes, segments, operator_plan)
            return answer, self._execution_details(answer, segments, source_path=["neo4j_first"])
        if operator_plan.intent == "relationship_explore":
            answer = self._answer_relationship(indexes, segments, operator_plan)
            return answer, self._execution_details(answer, segments, source_path=["neo4j_first"])
        if operator_plan.intent == "relationship_rank_query":
            answer = self._answer_relationship_rank(indexes, segments, operator_plan)
            return answer, self._execution_details(answer, segments, source_path=["neo4j_first"])

        source_handlers = {
            "neo4j_first": lambda: self._answer_events(indexes, segments, operator_plan, question),
            "redis_first": lambda: self._answer_profiles(memory, indexes, segments, operator_plan, query_plan, question),
            "milvus_first": lambda: self._answer_evidence_lookup(indexes, segments, operator_plan, query_plan, question),
            "neo4j": lambda: self._answer_events(indexes, segments, operator_plan, question),
            "redis": lambda: self._answer_profiles(memory, indexes, segments, operator_plan, query_plan, question),
            "milvus": lambda: self._answer_evidence_lookup(indexes, segments, operator_plan, query_plan, question),
        }
        source_path: List[str] = []
        first_answer: Optional[AnswerDTO] = None
        for source in query_plan.source_order:
            handler = source_handlers.get(source)
            if handler is None:
                continue
            source_path.append(source)
            candidate = handler()
            if candidate is None:
                continue
            if first_answer is None:
                first_answer = candidate
            if not self._is_empty_answer(candidate):
                return candidate, self._execution_details(candidate, segments, source_path=source_path)

        fallback = first_answer or self._memory_abstain_answer(operator_plan, query_plan, question)
        return fallback, self._execution_details(fallback, segments, source_path=source_path)

    def _answer_profiles(
        self,
        memory: Dict[str, Any],
        indexes: Dict[str, Any],
        segments: List[Dict[str, Any]],
        operator_plan: OperatorPlanDTO,
        query_plan: QueryPlanDTO,
        question: str,
    ) -> Optional[AnswerDTO]:
        _ = memory
        _ = query_plan
        redis_payload = indexes.get("redis", {})
        profile_core = redis_payload.get("profile_core", {})
        fields = profile_core.get("fields", {}) if isinstance(profile_core, dict) else {}
        structured_profiles = profile_core.get("profiles", {}) if isinstance(profile_core, dict) else {}
        query_vector = self.embedder.embed_query(question)
        candidates: List[Dict[str, Any]] = []

        for field_key, payload in fields.items():
            if not isinstance(payload, dict):
                continue
            values = [str(item) for item in payload.get("values", []) or [] if item]
            text = " ".join([str(field_key), *values])
            candidates.append(
                {
                    "source": "profile_field",
                    "profile_key": "profile_core",
                    "field_key": str(field_key),
                    "text": text,
                    "summary": " / ".join(values),
                    "confidence": float(payload.get("confidence") or 0.0),
                    "supporting_event_ids": [str(item) for item in payload.get("supporting_event_ids", []) or []],
                    "evidence_refs": payload.get("evidence_refs", []) or [],
                }
            )

        for profile_key, profile_fields in structured_profiles.items():
            if not isinstance(profile_fields, dict):
                continue
            for field_key, payload in profile_fields.items():
                if not isinstance(payload, dict):
                    continue
                summary = str(payload.get("summary") or payload.get("value") or "")
                text = " ".join([str(profile_key), str(field_key), summary])
                candidates.append(
                    {
                        "source": "materialized_profile",
                        "profile_key": str(profile_key),
                        "field_key": str(field_key),
                        "text": text,
                        "summary": summary,
                        "confidence": float(payload.get("confidence") or 0.0),
                        "supporting_event_ids": [str(item) for item in payload.get("supporting_event_ids", []) or []],
                        "evidence_refs": payload.get("evidence_refs", []) or [],
                    }
                )

        scored: List[Dict[str, Any]] = []
        for item in candidates:
            score = self._text_or_embedding_score(question, item["text"], query_vector)
            score += self._profile_candidate_bonus(question, item)
            if score < 0.18:
                continue
            payload = dict(item)
            payload["match_score"] = round(score, 4)
            scored.append(payload)

        scored.sort(key=lambda item: (float(item["match_score"]), float(item["confidence"])), reverse=True)
        top = scored[:5]
        if not top:
            return None

        supporting_events = self._events_from_ids(indexes, [event_id for item in top for event_id in item["supporting_event_ids"]])
        evidence_segment_ids = self._segment_ids_for_profile_candidates(
            segments,
            profile_candidates=top,
            event_uuids=[item["event_uuid"] for item in supporting_events],
        )
        avg_confidence = round(sum(float(item["confidence"]) for item in top) / max(len(top), 1), 4)
        summary = "；".join(
            f"{item['profile_key']}.{item['field_key']}: {item['summary'] or 'n/a'}"
            for item in top[:3]
        )
        return AnswerDTO(
            answer_type="profile_lookup",
            summary=summary,
            confidence=avg_confidence,
            resolved_entities=[
                {
                    "source": item["source"],
                    "profile_key": item["profile_key"],
                    "field_key": item["field_key"],
                    "summary": item["summary"],
                    "confidence": item["confidence"],
                    "match_score": item["match_score"],
                    "supporting_event_ids": item["supporting_event_ids"],
                    "evidence_refs": item["evidence_refs"],
                }
                for item in top
            ],
            resolved_concepts=self._unique(str(item["profile_key"]) for item in top),
            time_window=asdict(operator_plan.time_scope),
            supporting_events=supporting_events,
            evidence_segment_ids=evidence_segment_ids,
            explanation="优先查询 Redis 中物化的画像字段，再补充事件与证据段。",
            uncertainty_flags=[] if avg_confidence >= 0.45 else ["profile_low_confidence"],
        )

    def _answer_evidence_lookup(
        self,
        indexes: Dict[str, Any],
        segments: List[Dict[str, Any]],
        operator_plan: OperatorPlanDTO,
        query_plan: QueryPlanDTO,
        question: str,
    ) -> Optional[AnswerDTO]:
        _ = query_plan
        if not segments:
            return None
        query_vector = self.embedder.embed_query(question)
        scored_segments: List[Dict[str, Any]] = []
        for segment in segments:
            if not self._segment_in_time_scope(segment, operator_plan.time_scope):
                continue
            search_text = " ".join(
                [
                    str(segment.get("segment_type") or ""),
                    str(segment.get("text") or ""),
                    str(segment.get("location_hint") or ""),
                ]
            )
            score = self._text_or_embedding_score(question, search_text, query_vector)
            score += self._segment_candidate_bonus(question, segment)
            if score < 0.22:
                continue
            payload = dict(segment)
            payload["match_score"] = round(score, 4)
            scored_segments.append(payload)

        scored_segments.sort(key=lambda item: float(item["match_score"]), reverse=True)
        top_segments = scored_segments[:6]
        if not top_segments:
            return None

        fact_uuids = [str(item.get("event_uuid") or "") for item in top_segments if item.get("event_uuid")]
        event_uuids = [str(item.get("session_uuid") or "") for item in top_segments if item.get("session_uuid")]
        relationship_uuids = [str(item.get("relationship_uuid") or "") for item in top_segments if item.get("relationship_uuid")]
        supporting_facts = self._events_by_uuids(indexes, fact_uuids)
        supporting_events = self._sessions_by_uuids(indexes, event_uuids)
        supporting_relationships = self._relationships_by_uuids(indexes, relationship_uuids)
        evidence_segment_ids = [str(item.get("segment_uuid")) for item in top_segments if item.get("segment_uuid")]
        summary = "；".join(
            f"{item.get('segment_type')}: {str(item.get('text') or '')[:80]}"
            for item in top_segments[:3]
        )
        confidence = round(sum(float(item["match_score"]) for item in top_segments) / len(top_segments), 4)
        return AnswerDTO(
            answer_type="evidence_lookup",
            summary=summary,
            confidence=confidence,
            resolved_entities=[
                {
                    "segment_uuid": item.get("segment_uuid"),
                    "segment_type": item.get("segment_type"),
                    "text": item.get("text"),
                    "match_score": item.get("match_score"),
                    "location_hint": item.get("location_hint"),
                }
                for item in top_segments
            ],
            resolved_concepts=[],
            time_window=asdict(operator_plan.time_scope),
            supporting_events=supporting_events,
            supporting_facts=supporting_facts,
            supporting_relationships=supporting_relationships,
            evidence_segment_ids=evidence_segment_ids,
            explanation="优先在 Milvus 证据段中做开放语义召回，再回填图谱中的 Event、Fact 与关系骨架。",
            uncertainty_flags=[] if confidence >= 0.32 else ["evidence_low_confidence"],
        )

    def _memory_abstain_answer(
        self,
        operator_plan: OperatorPlanDTO,
        query_plan: QueryPlanDTO,
        question: str,
    ) -> AnswerDTO:
        _ = query_plan
        return AnswerDTO(
            answer_type="memory_abstain",
            summary=f"当前 memory 中没有足够证据回答：{question}",
            confidence=0.0,
            resolved_entities=[],
            resolved_concepts=list(operator_plan.target_concepts),
            time_window=asdict(operator_plan.time_scope),
            explanation="已按 source_order 查询 Redis / Neo4j / Milvus，但没有找到足够强的记忆证据。",
            uncertainty_flags=["not_answerable_from_memory"],
        )

    def _is_empty_answer(self, answer: Optional[AnswerDTO]) -> bool:
        if answer is None:
            return True
        if answer.answer_type == "memory_abstain":
            return True
        if answer.confidence > 0.0 and answer.resolved_entities:
            return False
        return bool(
            not answer.resolved_entities
            and not answer.supporting_events
            and not answer.supporting_facts
            and not answer.supporting_relationships
            and not answer.evidence_segment_ids
        )

    def _execution_details(
        self,
        answer: AnswerDTO,
        segments: List[Dict[str, Any]],
        *,
        source_path: List[str],
    ) -> Dict[str, Any]:
        supporting_segments = self._segments_by_ids(segments, answer.evidence_segment_ids)
        supporting_redis_profiles = [
            entity for entity in answer.resolved_entities
            if isinstance(entity, dict) and entity.get("profile_key")
        ]
        abstain_reason = answer.summary if "not_answerable_from_memory" in answer.uncertainty_flags else ""
        return {
            "source_path": list(source_path),
            "supporting_redis_profiles": self._serialize(supporting_redis_profiles),
            "supporting_graph_entities": self._serialize(answer.resolved_entities),
            "supporting_segments": self._serialize(supporting_segments),
            "abstain_reason": abstain_reason,
        }

    def _answer_events(
        self,
        indexes: Dict[str, Any],
        segments: List[Dict[str, Any]],
        operator_plan: OperatorPlanDTO,
        question: str,
    ) -> AnswerDTO:
        time_scope = operator_plan.time_scope
        concept_filters = set(operator_plan.target_concepts)
        facts = []
        for fact in indexes["nodes_by_group"].get("facts", []):
            props = fact.get("properties", {})
            if props.get("user_id") != indexes["user_id"]:
                continue
            if time_scope.start_at and not self._overlaps_time_scope(
                props.get("started_at"),
                props.get("ended_at") or props.get("started_at"),
                time_scope,
            ):
                continue
            fact_concepts = set(self._outgoing_concepts(indexes, fact.get("event_uuid")))
            event_text = " ".join(
                str(part or "")
                for part in (
                    props.get("title"),
                    props.get("coarse_event_type"),
                    props.get("day_key"),
                    props.get("location"),
                )
            ).lower()
            if concept_filters and not fact_concepts.intersection(concept_filters) and not any(
                concept in event_text for concept in concept_filters
            ):
                continue
            facts.append(fact)

        facts.sort(key=lambda item: str(item.get("properties", {}).get("started_at") or ""), reverse=True)
        if "最近一次" in question or "latest" in question.lower() or "most recent" in question.lower():
            facts = facts[:1]

        supporting_facts = [self._event_payload(indexes, fact) for fact in facts]
        supporting_events = self._supporting_sessions_for_events(indexes, facts)
        representative_photo_ids = self._unique(
            photo_id
            for fact in facts
            for photo_id in fact.get("properties", {}).get("representative_photo_ids", [])
        )
        evidence_segment_ids = self._segment_ids_for_objects(
            segments,
            session_uuids=[event["event_uuid"] for event in supporting_events],
            event_uuids=[fact.get("event_uuid") for fact in facts],
        )
        resolved_concepts = list(concept_filters)
        if facts:
            summary = f"共找到 {len(facts)} 个事实: " + "；".join(fact["title"] for fact in supporting_facts[:5])
            confidence = round(sum(fact["confidence"] for fact in supporting_facts) / len(supporting_facts), 4)
            explanation = "先按时间窗过滤 Facts，再按 canonical concepts 过滤，最后补充 Events 和语义证据。"
            uncertainty_flags = []
        else:
            fallback = self._fallback_conflict_answer(indexes, segments, operator_plan)
            if fallback is not None:
                return fallback
            summary = "没有找到符合条件的事实。"
            confidence = 0.0
            explanation = "先按时间窗过滤 Facts，再按 canonical concepts 过滤；当前未命中。"
            uncertainty_flags = ["no_matching_events"]
        return AnswerDTO(
            answer_type="event_search",
            summary=summary,
            confidence=confidence,
            resolved_entities=supporting_facts,
            resolved_concepts=resolved_concepts,
            time_window=asdict(time_scope),
            supporting_events=supporting_events,
            supporting_facts=supporting_facts,
            representative_photo_ids=representative_photo_ids,
            evidence_segment_ids=evidence_segment_ids,
            explanation=explanation,
            uncertainty_flags=uncertainty_flags,
        )

    def _fallback_conflict_answer(
        self,
        indexes: Dict[str, Any],
        segments: List[Dict[str, Any]],
        operator_plan: OperatorPlanDTO,
    ) -> Optional[AnswerDTO]:
        concept_filters = set(operator_plan.target_concepts)
        if "conflict" not in concept_filters:
            return None

        relationships = []
        time_scope = operator_plan.time_scope
        for relationship in indexes["nodes_by_group"].get("relationship_hypotheses", []):
            props = relationship.get("properties", {})
            if props.get("status") not in {"active", "cooling"}:
                continue
            if time_scope.start_at and props.get("window_end") and not self._overlaps_time_scope(
                props.get("window_start"),
                props.get("window_end"),
                time_scope,
            ):
                continue
            rel_concepts = set(self._outgoing_concepts(indexes, relationship.get("relationship_uuid")))
            reason_summary = str(props.get("reason_summary") or "")
            feature_snapshot = props.get("feature_snapshot", {})
            conflict_score = float(feature_snapshot.get("conflict_signal_count") or 0.0)
            if (
                "conflict" in rel_concepts
                or conflict_score > 0
                or self._contains_conflict_text(reason_summary)
            ):
                relationships.append(relationship)

        relationships.sort(
            key=lambda item: (
                float(item.get("properties", {}).get("feature_snapshot", {}).get("conflict_signal_count", 0.0)),
                str(item.get("properties", {}).get("window_end") or ""),
                float(item.get("properties", {}).get("confidence") or 0.0),
            ),
            reverse=True,
        )

        if not relationships:
            return None

        supporting_relationships = [self._relationship_payload(indexes, item) for item in relationships[:3]]
        supporting_events = self._supporting_sessions_for_relationships(indexes, relationships[:3])
        supporting_facts = self._supporting_events_for_relationships(indexes, relationships[:3])
        representative_photo_ids = self._unique(
            photo_id
            for relationship in supporting_relationships
            for photo_id in relationship.get("representative_photo_ids", [])
        )
        evidence_segment_ids = self._segment_ids_for_conflict_fallback(
            segments,
            relationship_uuids=[item.get("relationship_uuid") for item in relationships[:3]],
            session_uuids=[item.get("event_uuid") for item in supporting_events],
            event_uuids=[item.get("event_uuid") for item in supporting_facts],
        )
        target_names = [item.get("target_face_person_id") for item in supporting_relationships if item.get("target_face_person_id")]
        if target_names:
            summary = f"没有找到显式冲突事件，已回退到关系与交互证据：最近更值得关注的是与 {'、'.join(target_names[:3])} 的冲突线索。"
        else:
            summary = "没有找到显式冲突事件，已回退到关系与交互证据。"
        confidence = round(
            sum(float(item.get("confidence") or 0.0) for item in supporting_relationships) / len(supporting_relationships),
            4,
        )
        return AnswerDTO(
            answer_type="event_search",
            summary=summary,
            confidence=confidence,
            resolved_entities=supporting_relationships,
            resolved_concepts=list(concept_filters),
            time_window=asdict(time_scope),
            supporting_events=supporting_events,
            supporting_facts=supporting_facts,
            supporting_relationships=supporting_relationships,
            representative_photo_ids=representative_photo_ids,
            evidence_segment_ids=evidence_segment_ids,
            explanation="先查 Event(conflict)；未命中后回退到 RelationshipHypothesis.reason_summary，再补充 Milvus 的 interaction/relationship_reason/negative-mood 证据。",
            uncertainty_flags=["derived_from_relationship_fallback"],
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
        supporting_events = self._supporting_sessions_for_relationships(indexes, primary)
        supporting_facts = self._supporting_events_for_relationships(indexes, primary)
        evidence_segment_ids = self._segment_ids_for_objects(
            segments,
            relationship_uuids=[item.get("relationship_uuid") for item in primary],
            session_uuids=[item["event_uuid"] for item in supporting_events],
            event_uuids=[item["event_uuid"] for item in supporting_facts],
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
            supporting_events=supporting_events,
            supporting_facts=supporting_facts,
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
        supporting_events = self._supporting_sessions_for_relationships(indexes, ranked)
        supporting_facts = self._supporting_events_for_relationships(indexes, ranked)
        evidence_segment_ids = self._segment_ids_for_objects(
            segments,
            relationship_uuids=[item.get("relationship_uuid") for item in ranked],
            session_uuids=[item["event_uuid"] for item in supporting_events],
            event_uuids=[item["event_uuid"] for item in supporting_facts],
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
            supporting_events=supporting_events,
            supporting_facts=supporting_facts,
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
        supporting_events = []
        for mood in top:
            session_uuid = mood.get("properties", {}).get("session_uuid")
            if session_uuid and session_uuid in indexes["nodes"]:
                supporting_events.append(self._session_payload(indexes["nodes"][session_uuid]))
        evidence_segment_ids = self._segment_ids_for_objects(
            segments,
            session_uuids=[item["event_uuid"] for item in supporting_events],
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
            supporting_events=supporting_events,
            evidence_segment_ids=evidence_segment_ids,
            explanation="优先读取 mood hypotheses，并用会话级证据片段回填解释。",
            uncertainty_flags=uncertainty_flags,
        )

    def _build_indexes(self, neo4j_payload: Dict[str, Any], redis_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            "redis": dict(redis_payload or {}),
        }

    def _text_or_embedding_score(self, question: str, candidate_text: str, query_vector: List[float]) -> float:
        normalized_question = str(question or "").lower()
        normalized_candidate = str(candidate_text or "").lower()
        if normalized_candidate and normalized_candidate in normalized_question:
            return 0.99
        overlap_tokens = [
            token
            for token in re.split(r"[\s,，。！？?!.:：/_-]+", normalized_question)
            if token and token in normalized_candidate
        ]
        lexical_bonus = min(0.45, len(overlap_tokens) * 0.08)
        candidate_vector = self.embedder.embed_text(candidate_text, task_type="document")[0]
        semantic_score = max(0.0, cosine_similarity(query_vector, candidate_vector))
        return round(max(semantic_score, semantic_score + lexical_bonus), 4)

    def _profile_candidate_bonus(self, question: str, candidate: Dict[str, Any]) -> float:
        normalized_question = str(question or "").lower()
        profile_key = str(candidate.get("profile_key") or "").lower()
        field_key = str(candidate.get("field_key") or "").lower()
        bonus = 0.0
        if any(token in normalized_question for token in ("喜欢", "偏好", "爱喝", "饮料", "喝什么", "吃什么")):
            if "preference" in profile_key:
                bonus += 0.22
            if any(token in field_key for token in ("drink", "beverage", "food", "favorite")):
                bonus += 0.22
        if any(token in normalized_question for token in ("评价", "差评", "口味", "商家")) and "opinion" in profile_key:
            bonus += 0.28
        if any(token in normalized_question for token in ("风格", "穿搭", "材质", "品牌")) and "style" in profile_key:
            bonus += 0.28
        if any(token in normalized_question for token in ("职业", "公司", "股权", "身份")) and ("career" in profile_key or "identity" in profile_key):
            bonus += 0.28
        if any(token in normalized_question for token in ("消费", "阶层")) and "consumption" in profile_key:
            bonus += 0.28
        if any(token in normalized_question for token in ("审美", "精神", "命运", "超自然")) and "aesthetic" in profile_key:
            bonus += 0.28
        if "推荐" in normalized_question and "recommendation" in profile_key:
            bonus += 0.28
        return round(bonus, 4)

    def _segment_candidate_bonus(self, question: str, segment: Dict[str, Any]) -> float:
        normalized_question = str(question or "").lower()
        segment_type = str(segment.get("segment_type") or "").lower()
        segment_text = str(segment.get("text") or "").lower()
        bonus = 0.0
        if any(token in normalized_question for token in ("饮料", "喝什么", "产品", "品牌")):
            if any(token in segment_type for token in ("brand", "dish", "price", "observation", "claim")):
                bonus += 0.2
            if any(token in segment_text for token in ("drink", "beverage", "latte", "coffee", "brand", "product")):
                bonus += 0.25
        if any(token in normalized_question for token in ("围巾", "材质", "穿搭", "品牌")):
            if any(token in segment_type for token in ("clothing", "brand", "observation", "claim")):
                bonus += 0.2
            if any(token in segment_text for token in ("scarf", "material", "cotton", "wool", "brand")):
                bonus += 0.25
        if any(token in normalized_question for token in ("玩具", "鱼缸", "在哪里", "最后")):
            if any(token in segment_type for token in ("object", "observation", "claim")):
                bonus += 0.2
            if any(token in segment_text for token in ("last seen", "object", "toy", "aquarium")):
                bonus += 0.25
        if any(token in normalized_question for token in ("路线", "规划", "景点")) and any(token in segment_type for token in ("route", "plan", "observation", "claim")):
            bonus += 0.24
        if any(token in normalized_question for token in ("治疗", "生病", "药")) and any(token in segment_type for token in ("health", "observation", "claim")):
            bonus += 0.24
        return round(bonus, 4)

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
                "MATCH (f:Fact)-[:HAS_CONCEPT]->(c:Concept) "
                "WHERE f.user_id = $user_id AND f.started_at >= $start_at AND c.canonical_name IN $concepts "
                "RETURN f ORDER BY f.started_at DESC"
            )
        if operator_plan.intent == "relationship_rank_query":
            return (
                "MATCH (p:Person)-[:HAS_RELATIONSHIP]->(r:RelationshipHypothesis) "
                "WHERE r.status IN ['active','cooling'] "
                "RETURN r ORDER BY r.co_present_session_count DESC"
            )
        if operator_plan.intent == "mood_lookup":
            return "MATCH (m:MoodStateHypothesis)-[:DESCRIBES_EVENT]->(e:Event) RETURN m, e ORDER BY m.window_end DESC"
        return (
            "MATCH (p:Person)-[:HAS_RELATIONSHIP]->(r:RelationshipHypothesis) "
            "WHERE r.status IN ['active','cooling'] RETURN r"
        )

    def _event_payload(self, indexes: Dict[str, Any], node: Dict[str, Any]) -> Dict[str, Any]:
        props = node.get("properties", {})
        participants = self._participants_for_event(indexes, node.get("event_uuid"))
        return {
            "event_uuid": node.get("event_uuid"),
            "fact_id": props.get("fact_id"),
            "title": props.get("title"),
            "normalized_event_type": props.get("normalized_event_type"),
            "coarse_event_type": props.get("coarse_event_type"),
            "started_at": props.get("started_at"),
            "ended_at": props.get("ended_at"),
            "confidence": float(props.get("confidence") or 0.0),
            "participants": participants,
            "original_image_ids": props.get("original_image_ids", []),
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
            "event_uuid": node.get("session_uuid"),
            "event_id": props.get("event_id"),
            "title": props.get("summary_hint") or props.get("event_id"),
            "started_at": props.get("started_at"),
            "ended_at": props.get("ended_at"),
            "place_uuid": props.get("place_uuid"),
            "participant_count": props.get("participant_count"),
            "confidence": float(props.get("confidence") or 0.6),
            "original_image_ids": props.get("original_image_ids", []),
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
            if edge.get("edge_type") == "DERIVED_FROM_EVENT"
        ]
        participants = []
        for session_uuid in session_uuids:
            for edge in indexes["incoming"].get(str(session_uuid), []):
                if edge.get("edge_type") != "CO_PRESENT_IN_EVENT":
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
                if edge.get("edge_type") != "DERIVED_FROM_EVENT":
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

    def _segment_ids_for_conflict_fallback(
        self,
        segments: List[Dict[str, Any]],
        *,
        relationship_uuids: List[str],
        session_uuids: List[str],
        event_uuids: List[str],
    ) -> List[str]:
        relationship_set = {str(item) for item in relationship_uuids if item}
        session_set = {str(item) for item in session_uuids if item}
        event_set = {str(item) for item in event_uuids if item}
        selected: List[str] = []
        fallback: List[str] = []
        for segment in segments:
            if (
                str(segment.get("relationship_uuid") or "") not in relationship_set
                and str(segment.get("session_uuid") or "") not in session_set
                and str(segment.get("event_uuid") or "") not in event_set
            ):
                continue
            segment_id = str(segment.get("segment_uuid"))
            segment_type = str(segment.get("segment_type") or "")
            text = str(segment.get("text") or "")
            if segment_type in {"interaction", "relationship_reason"} and self._contains_conflict_text(text):
                selected.append(segment_id)
                continue
            if segment_type == "profile_evidence_snippet" and self._contains_negative_mood_text(text):
                selected.append(segment_id)
                continue
            if segment_type in {"relationship_reason", "session_summary", "event_narrative"}:
                fallback.append(segment_id)
        return self._unique(selected or fallback)

    def _facts_for_events(self, indexes: Dict[str, Any], events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        facts = []
        seen = set()
        for event in events:
            for edge in indexes["incoming"].get(str(event.get("session_uuid")), []):
                if edge.get("edge_type") != "DERIVED_FROM_EVENT":
                    continue
                fact_node = indexes["nodes"].get(str(edge.get("from_id")))
                if fact_node and fact_node.get("event_uuid") not in seen:
                    seen.add(fact_node.get("event_uuid"))
                    facts.append(self._event_payload(indexes, fact_node))
        return facts

    def _segment_ids_for_profile_candidates(
        self,
        segments: List[Dict[str, Any]],
        *,
        profile_candidates: List[Dict[str, Any]],
        event_uuids: List[str],
    ) -> List[str]:
        event_set = {str(item) for item in event_uuids if item}
        evidence_tokens = {
            str(ref.get("ref_id") or "").lower()
            for candidate in profile_candidates
            for ref in candidate.get("evidence_refs", []) or []
            if isinstance(ref, dict)
        }
        selected: List[str] = []
        for segment in segments:
            segment_uuid = str(segment.get("segment_uuid") or "")
            if event_set and str(segment.get("event_uuid") or "") in event_set:
                selected.append(segment_uuid)
                continue
            segment_text = str(segment.get("text") or "").lower()
            if evidence_tokens and any(token and token in segment_text for token in evidence_tokens):
                selected.append(segment_uuid)
                continue
            if str(segment.get("segment_type") or "") == "profile_evidence_snippet":
                for candidate in profile_candidates:
                    field_key = str(candidate.get("field_key") or "").lower()
                    summary = str(candidate.get("summary") or "").lower()
                    if field_key and field_key in segment_text:
                        selected.append(segment_uuid)
                        break
                    if summary and summary in segment_text:
                        selected.append(segment_uuid)
                        break
        return self._unique(selected)

    def _events_from_ids(self, indexes: Dict[str, Any], event_ids: List[str]) -> List[Dict[str, Any]]:
        event_uuid_by_id = {
            str(node.get("properties", {}).get("fact_id") or ""): node.get("event_uuid")
            for node in indexes["nodes_by_group"].get("facts", [])
        }
        event_uuids = [event_uuid_by_id.get(str(event_id) or "") for event_id in event_ids if event_uuid_by_id.get(str(event_id) or "")]
        return self._events_by_uuids(indexes, event_uuids)

    def _events_by_uuids(self, indexes: Dict[str, Any], event_uuids: List[str]) -> List[Dict[str, Any]]:
        seen = set()
        payloads: List[Dict[str, Any]] = []
        for event_uuid in event_uuids:
            node = indexes["nodes"].get(str(event_uuid))
            if not node or str(event_uuid) in seen:
                continue
            seen.add(str(event_uuid))
            payloads.append(self._event_payload(indexes, node))
        return payloads

    def _sessions_by_uuids(self, indexes: Dict[str, Any], session_uuids: List[str]) -> List[Dict[str, Any]]:
        seen = set()
        payloads: List[Dict[str, Any]] = []
        for session_uuid in session_uuids:
            node = indexes["nodes"].get(str(session_uuid))
            if not node or str(session_uuid) in seen:
                continue
            seen.add(str(session_uuid))
            payloads.append(self._session_payload(node))
        return payloads

    def _relationships_by_uuids(self, indexes: Dict[str, Any], relationship_uuids: List[str]) -> List[Dict[str, Any]]:
        seen = set()
        payloads: List[Dict[str, Any]] = []
        for relationship_uuid in relationship_uuids:
            node = indexes["nodes"].get(str(relationship_uuid))
            if not node or str(relationship_uuid) in seen:
                continue
            seen.add(str(relationship_uuid))
            payloads.append(self._relationship_payload(indexes, node))
        return payloads

    def _segment_in_time_scope(self, segment: Dict[str, Any], time_scope: TimeScopeDTO) -> bool:
        if not time_scope.start_at:
            return True
        return self._overlaps_time_scope(
            segment.get("started_at"),
            segment.get("ended_at") or segment.get("started_at"),
            time_scope,
        )

    def _segments_by_ids(self, segments: List[Dict[str, Any]], segment_ids: List[str]) -> List[Dict[str, Any]]:
        segment_set = {str(item) for item in segment_ids if item}
        return [
            {
                "segment_uuid": segment.get("segment_uuid"),
                "segment_type": segment.get("segment_type"),
                "text": segment.get("text"),
                "location_hint": segment.get("location_hint"),
                "event_uuid": segment.get("event_uuid"),
                "session_uuid": segment.get("session_uuid"),
                "relationship_uuid": segment.get("relationship_uuid"),
            }
            for segment in segments
            if str(segment.get("segment_uuid") or "") in segment_set
        ]

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

    def _contains_conflict_text(self, text: str) -> bool:
        normalized = str(text or "").lower()
        return any(
            token in normalized
            for token in ("conflict", "argument", "fight", "tension", "dispute", "disagree", "冲突", "争吵", "矛盾", "冷战", "紧张")
        )

    def _contains_negative_mood_text(self, text: str) -> bool:
        normalized = str(text or "").lower()
        return any(
            token in normalized
            for token in ("sad", "down", "tense", "upset", "negative", "难过", "低落", "压抑", "紧张", "不开心")
        )

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
