"""Photo-backed query engine for v0325 / v0327-db tasks."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from memory_module.embeddings import EmbeddingProvider, cosine_similarity

from .materializer import SCHEMA_VERSION, SUPPORTED_QUERY_V1_VERSIONS, materialize_v0325_to_query_store
from .planner import StructuredQueryPlanner
from .store import QueryStore
from .writer import AnswerWriterLLM


class QueryEngineV1:
    """Query v1 facade with on-demand materialization + routed retrieval."""

    def __init__(
        self,
        *,
        store: Optional[QueryStore] = None,
        planner: Optional[StructuredQueryPlanner] = None,
        embedder: Optional[EmbeddingProvider] = None,
        writer: Optional[AnswerWriterLLM] = None,
        now: Optional[datetime] = None,
    ) -> None:
        self.store = store or QueryStore()
        self.now = now or datetime.now()
        self.planner = planner or StructuredQueryPlanner(now=self.now)
        self.embedder = embedder or EmbeddingProvider.from_config()
        self.writer = writer or AnswerWriterLLM()

    def supports_task(self, task: Dict[str, Any]) -> bool:
        return str(task.get("version") or "").strip() in SUPPORTED_QUERY_V1_VERSIONS

    def answer_task(
        self,
        task: Dict[str, Any],
        question: str,
        *,
        user_id: str,
        context_hints: Optional[Dict[str, Any]] = None,
        time_hint: Optional[str] = None,
        answer_shape_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        materialization = materialize_v0325_to_query_store(task, user_id, self.store)
        bundle = self.store.fetch_scope(user_id=user_id, source_task_id=str(task.get("task_id") or ""))
        prepared = self._prepare_bundle(bundle)

        known_person_ids = self._unique(
            [item["person_id"] for item in bundle["relationships"] if item.get("person_id")]
            + [item["person_id"] for item in bundle["event_people"] if item.get("person_id")]
        )
        known_group_ids = [item["group_id"] for item in bundle["groups"] if item.get("group_id")]
        known_relationship_types = [item["relationship_type"] for item in bundle["relationships"] if item.get("relationship_type")]
        known_profile_fact_keys = [item["field_key"] for item in bundle["profile_facts"] if item.get("field_key")]

        plan = self.plan_query_v1(
            question,
            user_id=user_id,
            task_scope=str(task.get("task_id") or ""),
            known_person_ids=known_person_ids,
            known_group_ids=known_group_ids,
            known_relationship_types=known_relationship_types,
            known_profile_fact_keys=known_profile_fact_keys,
        )
        if context_hints:
            plan["context_hints"] = dict(context_hints)
        if answer_shape_hint:
            plan["answer_shape_hint"] = answer_shape_hint
        if time_hint:
            plan["time_hint"] = time_hint

        clause_results = []
        for clause in list(plan.get("clauses") or []):
            clause_results.append(self._execute_clause(clause, plan=plan, prepared=prepared, time_hint=time_hint))

        return self.answer_query_v1(
            plan,
            clause_results,
            bundle=bundle,
            materialization=materialization,
            question=question,
        )

    def plan_query_v1(
        self,
        question: str,
        *,
        user_id: str,
        task_scope: Optional[str],
        known_person_ids: Iterable[str],
        known_group_ids: Iterable[str],
        known_relationship_types: Iterable[str],
        known_profile_fact_keys: Iterable[str],
    ) -> Dict[str, Any]:
        return self.planner.plan(
            question,
            user_id=user_id,
            task_scope=task_scope,
            known_person_ids=known_person_ids,
            known_group_ids=known_group_ids,
            known_relationship_types=known_relationship_types,
            known_profile_fact_keys=known_profile_fact_keys,
        )

    def answer_query_v1(
        self,
        plan: Dict[str, Any],
        clause_results: Sequence[Dict[str, Any]],
        *,
        bundle: Dict[str, Any],
        materialization: Dict[str, Any],
        question: str,
    ) -> Dict[str, Any]:
        matched_events = self._aggregate_matched_events(clause_results)
        supporting_evidence = self._aggregate_unique_rows(clause_results, "supporting_evidence", id_key="evidence_id")
        supporting_photos = self._aggregate_unique_rows(clause_results, "supporting_photos", id_key="photo_id")
        supporting_facts = self._aggregate_unique_rows(clause_results, "supporting_facts", id_key="fact_id")
        supporting_relationships = self._aggregate_unique_rows(
            clause_results,
            "supporting_relationships",
            id_key="relationship_id",
        )
        graph_support = supporting_relationships + self._aggregate_unique_rows(clause_results, "group_support", id_key="group_id")
        clause_results_payload = [self._serialize_clause_result(item) for item in list(clause_results or [])]
        judgement_status, abstain_reason = self._final_judgement(clause_results)
        aggregation_result = self._aggregate_clause_aggregations(clause_results)
        matched_event_ids = [str(item.get("event_id") or "") for item in matched_events if item.get("event_id")]
        top_event_ids_for_summary = matched_event_ids[:6]
        summary_payload = self.writer.write(
            question=question,
            route_plan=plan,
            clause_results=clause_results_payload,
            matched_events=matched_events,
            top_event_ids_for_summary=top_event_ids_for_summary,
            supporting_photos=supporting_photos,
            supporting_facts=supporting_facts,
            supporting_relationships=supporting_relationships,
            judgement_status=judgement_status,
            abstain_reason=abstain_reason,
        )

        answer_type = self._answer_type(plan, clause_results)
        confidence = self._answer_confidence(matched_events, supporting_relationships, supporting_facts)
        original_photo_ids = self._original_photo_ids(supporting_photos)

        answer = {
            "answer_type": answer_type,
            "summary": summary_payload.get("summary_text") or "",
            "confidence": confidence,
            "original_photo_ids": original_photo_ids,
            "matched_events": matched_events,
            "matched_event_count": len(matched_events),
            "matched_event_ids": matched_event_ids,
            "top_event_ids_for_summary": top_event_ids_for_summary,
            "supporting_photos": supporting_photos,
            "supporting_evidence": supporting_evidence,
            "supporting_facts": supporting_facts,
            "supporting_relationships": supporting_relationships,
            "graph_support": graph_support,
            "clause_results": clause_results_payload,
            "aggregation_result": aggregation_result,
            "judgement_status": judgement_status,
            "abstain_reason": abstain_reason,
            "materialization_id": materialization.get("materialization_id"),
            "engine": "query_v1",
            "summary_event_ids": summary_payload.get("summary_event_ids") or [],
            "summary_fact_keys": summary_payload.get("summary_fact_keys") or [],
            "summary_relationship_ids": summary_payload.get("summary_relationship_ids") or [],
            "writer_source": summary_payload.get("writer_source") or "template",
        }
        return {
            "query_plan": {
                "engine": "query_v1",
                "schema_version": SCHEMA_VERSION,
                "plan_type": self._plan_type(plan),
                **plan,
            },
            "answer": answer,
            "matched_events": matched_events,
            "supporting_units": [self._legacy_unit(item) for item in matched_events],
            "supporting_evidence": supporting_evidence,
            "supporting_facts": supporting_facts,
            "supporting_relationships": supporting_relationships,
            "supporting_graph_entities": graph_support,
            "supporting_photos": supporting_photos,
            "graph_support": graph_support,
            "clause_results": clause_results_payload,
            "aggregation_result": aggregation_result,
            "judgement_status": judgement_status,
            "abstain_reason": abstain_reason,
        }

    def _prepare_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        prepared: Dict[str, Any] = {}
        prepared["event_lookup"] = {
            str(item.get("event_id") or ""): dict(item)
            for item in bundle["events"]
            if item.get("event_id")
        }
        prepared["event_people"] = self._group_rows(bundle["event_people"], "event_id")
        prepared["event_places"] = self._group_rows(bundle["event_places"], "event_id")
        prepared["event_photos"] = self._group_rows(bundle["event_photos"], "event_id")
        prepared["evidence_by_event"] = self._group_rows(bundle["evidence"], "event_id")
        prepared["photos_by_id"] = {
            str(item.get("photo_id") or ""): dict(item)
            for item in bundle["photos"]
            if item.get("photo_id")
        }
        prepared["relationships"] = [dict(item) for item in bundle["relationships"]]
        prepared["relationship_support"] = self._group_rows(bundle["relationship_support"], "relationship_id")
        prepared["groups"] = [dict(item) for item in bundle["groups"]]
        prepared["group_members"] = self._group_rows(bundle["group_members"], "group_id")
        prepared["profile_facts"] = [dict(item) for item in bundle["profile_facts"]]
        prepared["event_views"] = self._event_views(bundle)
        prepared["evidence_docs"] = self._evidence_docs(bundle)
        prepared["photo_to_events"] = self._group_rows(bundle["event_photos"], "photo_id")
        return prepared

    def _execute_clause(
        self,
        clause: Dict[str, Any],
        *,
        plan: Dict[str, Any],
        prepared: Dict[str, Any],
        time_hint: Optional[str],
    ) -> Dict[str, Any]:
        clause_plan = self._search_plan_for_clause(clause, time_hint=time_hint)
        route = str(clause.get("route") or "event-first")
        if route == "fact-first":
            return self._execute_fact_clause(clause, clause_plan=clause_plan, prepared=prepared)
        if route == "relationship-first":
            return self._execute_relationship_clause(clause, clause_plan=clause_plan, prepared=prepared)
        if route == "hybrid-judgement":
            return self._execute_hybrid_clause(clause, clause_plan=clause_plan, prepared=prepared)
        return self._execute_event_clause(clause, clause_plan=clause_plan, prepared=prepared)

    def _search_plan_for_clause(self, clause: Dict[str, Any], *, time_hint: Optional[str]) -> Dict[str, Any]:
        time_windows = self._normalize_time_windows(clause.get("time_windows"))
        if time_hint and not time_windows:
            time_windows = self._normalize_time_windows(time_hint)
        detail_clues = [str(item) for item in list(clause.get("detail_clues") or []) if str(item).strip()]
        clause_text = str(clause.get("clause_text") or "").strip()
        rewrites = [clause_text] if clause_text else []
        if detail_clues:
            rewrites.append(" ".join(detail_clues))
        if clause.get("target_fact_keys"):
            rewrites.append(" ".join(list(clause.get("target_fact_keys") or [])[:4]))
        if clause.get("target_relationship_types"):
            rewrites.append(" ".join(list(clause.get("target_relationship_types") or [])[:4]))
        return {
            "detail_clues": detail_clues,
            "rewrites": self._unique(rewrites),
            "time_windows": time_windows,
            "hard_filters": {
                "person_ids": list(clause.get("target_person_ids") or []),
                "group_ids": list(clause.get("target_group_ids") or []),
                "relationship_types": list(clause.get("target_relationship_types") or []),
            },
        }

    def _execute_fact_clause(
        self,
        clause: Dict[str, Any],
        *,
        clause_plan: Dict[str, Any],
        prepared: Dict[str, Any],
    ) -> Dict[str, Any]:
        facts = self._fact_candidates(clause, prepared["profile_facts"])
        event_ids = []
        photo_ids = []
        for fact in facts:
            event_ids.extend(list(fact.get("evidence_event_ids") or []))
            photo_ids.extend(list(fact.get("evidence_photo_ids") or []))
        matched_event_bundles = self._event_bundles_from_refs(
            event_ids=event_ids,
            photo_ids=photo_ids,
            prepared=prepared,
            fact_weight=max((float(item.get("confidence") or 0.0) for item in facts), default=0.0),
        )
        if not matched_event_bundles and facts:
            fact_texts = []
            for fact in facts[:4]:
                fact_texts.append(str(fact.get("field_key") or ""))
                fact_texts.append(self._fact_value_text(fact))
            semantic_plan = dict(clause_plan)
            semantic_plan["detail_clues"] = self._unique(list(clause_plan.get("detail_clues") or []) + fact_texts)
            semantic_plan["rewrites"] = self._unique(list(clause_plan.get("rewrites") or []) + fact_texts)
            matched_event_bundles = self._semantic_event_candidates(semantic_plan, prepared=prepared)
        matched_events = [self._serialize_matched_event(item) for item in matched_event_bundles]
        supporting_evidence = self._serialize_supporting_evidence(matched_event_bundles)
        supporting_photos = self._supporting_photos(matched_event_bundles, prepared["photos_by_id"])
        supporting_facts = [self._serialize_fact(item) for item in facts]
        judgement_status, abstain_reason = self._fact_judgement(clause, facts)
        return {
            "clause_id": clause.get("clause_id"),
            "route": clause.get("route"),
            "intent": clause.get("intent"),
            "clause_text": clause.get("clause_text"),
            "matched_events": matched_events,
            "supporting_evidence": supporting_evidence,
            "supporting_photos": supporting_photos,
            "supporting_facts": supporting_facts,
            "supporting_relationships": [],
            "group_support": [],
            "judgement_status": judgement_status,
            "abstain_reason": abstain_reason,
            "aggregation_result": {
                "kind": "fact_lookup",
                "matched_fact_count": len(supporting_facts),
                "field_keys": [item.get("field_key") for item in supporting_facts],
            },
            "result_hint": self._fact_result_hint(supporting_facts, judgement_status, abstain_reason),
        }

    def _execute_relationship_clause(
        self,
        clause: Dict[str, Any],
        *,
        clause_plan: Dict[str, Any],
        prepared: Dict[str, Any],
    ) -> Dict[str, Any]:
        candidates = self._relationship_candidates(
            clause_plan,
            prepared["relationships"],
            prepared["relationship_support"],
            prepared["event_lookup"],
        )
        intent = str(clause.get("intent") or "lookup")
        if intent == "rank":
            candidates = candidates[:1]
        elif intent not in {"existence", "count"}:
            candidates = candidates[:5]
        event_ids = []
        photo_ids = []
        for candidate in candidates:
            relationship = dict(candidate.get("relationship") or {})
            relationship_id = str(relationship.get("relationship_id") or "")
            for support in prepared["relationship_support"].get(relationship_id, []):
                event_id = str(support.get("event_id") or "")
                photo_id = str(support.get("photo_id") or "")
                if event_id:
                    event_ids.append(event_id)
                if photo_id:
                    photo_ids.append(photo_id)
        matched_event_bundles = self._event_bundles_from_refs(
            event_ids=event_ids,
            photo_ids=photo_ids,
            prepared=prepared,
            fact_weight=max((float(item.get("score") or 0.0) for item in candidates), default=0.0),
        )
        matched_events = [self._serialize_matched_event(item) for item in matched_event_bundles]
        supporting_evidence = self._serialize_supporting_evidence(matched_event_bundles)
        supporting_photos = self._supporting_photos(matched_event_bundles, prepared["photos_by_id"])
        supporting_relationships = [self._serialize_graph_entity(item) for item in candidates]
        groups = self._group_candidates(
            clause_plan,
            prepared["groups"],
            prepared["group_members"],
            matched_event_bundles,
        )
        group_support = [self._serialize_group_entity(item) for item in groups]
        judgement_status = "supported" if supporting_relationships else "insufficient_evidence"
        abstain_reason = None if supporting_relationships else "insufficient_relationship_evidence"
        aggregation_result = {
            "kind": "relationship",
            "match_count": len(supporting_relationships),
            "target_person_ids": [item.get("target_person_id") for item in supporting_relationships],
        }
        return {
            "clause_id": clause.get("clause_id"),
            "route": clause.get("route"),
            "intent": clause.get("intent"),
            "clause_text": clause.get("clause_text"),
            "matched_events": matched_events,
            "supporting_evidence": supporting_evidence,
            "supporting_photos": supporting_photos,
            "supporting_facts": [],
            "supporting_relationships": supporting_relationships,
            "group_support": group_support,
            "judgement_status": judgement_status,
            "abstain_reason": abstain_reason,
            "aggregation_result": aggregation_result,
            "result_hint": self._relationship_result_hint(supporting_relationships, judgement_status, abstain_reason),
        }

    def _execute_event_clause(
        self,
        clause: Dict[str, Any],
        *,
        clause_plan: Dict[str, Any],
        prepared: Dict[str, Any],
    ) -> Dict[str, Any]:
        matched_event_bundles = self._semantic_event_candidates(clause_plan, prepared=prepared)
        matched_events = [self._serialize_matched_event(item) for item in matched_event_bundles]
        supporting_evidence = self._serialize_supporting_evidence(matched_event_bundles)
        supporting_photos = self._supporting_photos(matched_event_bundles, prepared["photos_by_id"])
        intent = str(clause.get("intent") or "lookup")
        if intent == "count":
            aggregation_result = {"kind": "count", "count": len(matched_events)}
            judgement_status = "supported" if matched_events else "insufficient_evidence"
            abstain_reason = None if matched_events else "insufficient_event_evidence"
        elif intent == "existence":
            aggregation_result = {"kind": "existence", "exists": bool(matched_events)}
            judgement_status = "supported" if matched_events else "insufficient_evidence"
            abstain_reason = None if matched_events else "insufficient_coverage_for_negative"
        elif intent == "compare":
            aggregation_result = {"kind": "compare", "matched_event_count": len(matched_events)}
            judgement_status = "supported" if matched_events else "insufficient_evidence"
            abstain_reason = None if matched_events else "insufficient_compare_evidence"
        else:
            aggregation_result = {"kind": "event_search", "matched_event_count": len(matched_events)}
            judgement_status = "supported" if matched_events else "insufficient_evidence"
            abstain_reason = None if matched_events else "insufficient_event_evidence"
        return {
            "clause_id": clause.get("clause_id"),
            "route": clause.get("route"),
            "intent": clause.get("intent"),
            "clause_text": clause.get("clause_text"),
            "matched_events": matched_events,
            "supporting_evidence": supporting_evidence,
            "supporting_photos": supporting_photos,
            "supporting_facts": [],
            "supporting_relationships": [],
            "group_support": [],
            "judgement_status": judgement_status,
            "abstain_reason": abstain_reason,
            "aggregation_result": aggregation_result,
            "result_hint": self._event_result_hint(matched_events, judgement_status, abstain_reason),
        }

    def _execute_hybrid_clause(
        self,
        clause: Dict[str, Any],
        *,
        clause_plan: Dict[str, Any],
        prepared: Dict[str, Any],
    ) -> Dict[str, Any]:
        fact_result = self._execute_fact_clause(clause, clause_plan=clause_plan, prepared=prepared)
        relationship_result = self._execute_relationship_clause(clause, clause_plan=clause_plan, prepared=prepared)
        event_result = self._execute_event_clause(clause, clause_plan=clause_plan, prepared=prepared)
        supporting_facts = self._aggregate_unique_rows([fact_result], "supporting_facts", id_key="fact_id")
        supporting_relationships = self._aggregate_unique_rows([relationship_result], "supporting_relationships", id_key="relationship_id")
        matched_events = self._aggregate_matched_events([fact_result, relationship_result, event_result])
        supporting_evidence = self._aggregate_unique_rows(
            [fact_result, relationship_result, event_result],
            "supporting_evidence",
            id_key="evidence_id",
        )
        supporting_photos = self._aggregate_unique_rows(
            [fact_result, relationship_result, event_result],
            "supporting_photos",
            id_key="photo_id",
        )
        group_support = self._aggregate_unique_rows([relationship_result], "group_support", id_key="group_id")
        judgement_status, abstain_reason = self._hybrid_judgement(
            clause,
            supporting_facts=supporting_facts,
            supporting_relationships=supporting_relationships,
            matched_events=matched_events,
        )
        aggregation_result = {
            "kind": "judgement",
            "matched_event_count": len(matched_events),
            "supporting_fact_count": len(supporting_facts),
            "supporting_relationship_count": len(supporting_relationships),
        }
        return {
            "clause_id": clause.get("clause_id"),
            "route": clause.get("route"),
            "intent": clause.get("intent"),
            "clause_text": clause.get("clause_text"),
            "matched_events": matched_events,
            "supporting_evidence": supporting_evidence,
            "supporting_photos": supporting_photos,
            "supporting_facts": supporting_facts,
            "supporting_relationships": supporting_relationships,
            "group_support": group_support,
            "judgement_status": judgement_status,
            "abstain_reason": abstain_reason,
            "aggregation_result": aggregation_result,
            "result_hint": self._hybrid_result_hint(judgement_status, supporting_facts, supporting_relationships, abstain_reason),
        }

    def _semantic_event_candidates(
        self,
        clause_plan: Dict[str, Any],
        *,
        prepared: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        event_scores: Dict[str, Dict[str, Any]] = {}
        query_vectors = [
            self.embedder.embed_query(text)
            for text in list(clause_plan.get("rewrites") or [])[:6]
            if str(text or "").strip()
        ]
        for view in list(prepared["event_views"] or []):
            event_id = str(view.get("event_id") or "")
            if not event_id or event_id not in prepared["event_lookup"]:
                continue
            if not self._event_matches_hard_filters(
                clause_plan,
                prepared["event_lookup"][event_id],
                prepared["event_people"].get(event_id, []),
                prepared["event_places"].get(event_id, []),
            ):
                continue
            retrieval_text = str(view.get("retrieval_text") or "").strip()
            if not retrieval_text:
                continue
            view_vector = self.embedder.embed_query(retrieval_text)
            semantic_score = max((cosine_similarity(query_vector, view_vector) for query_vector in query_vectors), default=0.0)
            lexical_bonus = self._lexical_bonus(clause_plan, retrieval_text)
            score = semantic_score + lexical_bonus
            current = event_scores.setdefault(
                event_id,
                {
                    "event_score": 0.0,
                    "evidence_score": 0.0,
                    "relationship_score": 0.0,
                    "reasons": [],
                    "supporting_evidence_ids": set(),
                },
            )
            if score > current["event_score"]:
                current["event_score"] = score
                current["reasons"].append(f"view:{view.get('view_type')}:{score:.3f}")

        for doc in list(prepared["evidence_docs"] or []):
            event_id = str(doc.get("event_id") or "")
            if not event_id or event_id not in prepared["event_lookup"]:
                continue
            if not self._event_matches_hard_filters(
                clause_plan,
                prepared["event_lookup"][event_id],
                prepared["event_people"].get(event_id, []),
                prepared["event_places"].get(event_id, []),
            ):
                continue
            retrieval_text = str(doc.get("retrieval_text") or "").strip()
            if not retrieval_text:
                continue
            doc_vector = self.embedder.embed_query(retrieval_text)
            semantic_score = max((cosine_similarity(query_vector, doc_vector) for query_vector in query_vectors), default=0.0)
            lexical_bonus = self._lexical_bonus(clause_plan, retrieval_text)
            score = semantic_score + lexical_bonus
            if score < 0.16:
                continue
            current = event_scores.setdefault(
                event_id,
                {
                    "event_score": 0.0,
                    "evidence_score": 0.0,
                    "relationship_score": 0.0,
                    "reasons": [],
                    "supporting_evidence_ids": set(),
                },
            )
            current["evidence_score"] += score
            current["supporting_evidence_ids"].add(str(doc.get("evidence_id") or ""))
            current["reasons"].append(f"evidence:{doc.get('evidence_type')}:{score:.3f}")

        verified_events: List[Dict[str, Any]] = []
        for event_id, score_payload in event_scores.items():
            event = dict(prepared["event_lookup"][event_id])
            if not prepared["event_photos"].get(event_id):
                continue
            final_score = self._final_event_score(event, score_payload)
            if final_score < 0.14:
                continue
            verified_events.append(
                {
                    "event": event,
                    "event_score": final_score,
                    "event_people": [dict(item) for item in prepared["event_people"].get(event_id, [])],
                    "event_places": [dict(item) for item in prepared["event_places"].get(event_id, [])],
                    "event_photos": [dict(item) for item in prepared["event_photos"].get(event_id, [])],
                    "supporting_evidence": [
                        dict(item)
                        for item in prepared["evidence_by_event"].get(event_id, [])
                        if str(item.get("evidence_id") or "") in score_payload["supporting_evidence_ids"]
                    ],
                    "reasons": list(score_payload["reasons"]),
                }
            )
        verified_events.sort(
            key=lambda item: (-float(item.get("event_score") or 0.0), str((item.get("event") or {}).get("start_ts") or "")),
        )
        return verified_events

    def _event_bundles_from_refs(
        self,
        *,
        event_ids: Sequence[str],
        photo_ids: Sequence[str],
        prepared: Dict[str, Any],
        fact_weight: float,
    ) -> List[Dict[str, Any]]:
        resolved_event_ids = set(str(item) for item in list(event_ids or []) if str(item).strip())
        for photo_id in list(photo_ids or []):
            for row in list(prepared["photo_to_events"].get(str(photo_id), []) or []):
                event_id = str(row.get("event_id") or "")
                if event_id:
                    resolved_event_ids.add(event_id)
        bundles: List[Dict[str, Any]] = []
        for event_id in resolved_event_ids:
            event = dict(prepared["event_lookup"].get(event_id) or {})
            if not event:
                continue
            bundles.append(
                {
                    "event": event,
                    "event_score": 0.65 + min(max(fact_weight, 0.0), 0.3),
                    "event_people": [dict(item) for item in prepared["event_people"].get(event_id, [])],
                    "event_places": [dict(item) for item in prepared["event_places"].get(event_id, [])],
                    "event_photos": [dict(item) for item in prepared["event_photos"].get(event_id, [])],
                    "supporting_evidence": [dict(item) for item in prepared["evidence_by_event"].get(event_id, [])[:12]],
                    "reasons": ["canonical_support"],
                }
            )
        bundles.sort(
            key=lambda item: (-float(item.get("event_score") or 0.0), str((item.get("event") or {}).get("start_ts") or "")),
        )
        return bundles

    def _fact_candidates(self, clause: Dict[str, Any], profile_facts: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        detail_clues = [str(item).lower() for item in list(clause.get("detail_clues") or []) if str(item).strip()]
        target_fact_keys = set(str(item) for item in list(clause.get("target_fact_keys") or []) if str(item).strip())
        candidates: List[Dict[str, Any]] = []
        for fact in list(profile_facts or []):
            field_key = str(fact.get("field_key") or "")
            if not field_key:
                continue
            value_text = self._fact_value_text(fact).lower()
            score = 0.0
            if target_fact_keys and field_key in target_fact_keys:
                score += 0.85
            if not target_fact_keys and any(clue in field_key.lower() for clue in detail_clues):
                score += 0.5
            if any(clue in value_text for clue in detail_clues):
                score += 0.35
            if score <= 0:
                continue
            event_ids, photo_ids = self._extract_evidence_refs(fact)
            candidates.append(
                {
                    **dict(fact),
                    "match_score": round(score + min(float(fact.get("confidence") or 0.0), 1.0) * 0.1, 4),
                    "evidence_event_ids": event_ids,
                    "evidence_photo_ids": photo_ids,
                }
            )
        candidates.sort(key=lambda item: (-float(item.get("match_score") or 0.0), -float(item.get("confidence") or 0.0), str(item.get("field_key") or "")))
        return candidates[:8]

    def _relationship_candidates(
        self,
        clause_plan: Dict[str, Any],
        relationships: Sequence[Dict[str, Any]],
        relationship_support: Dict[str, List[Dict[str, Any]]],
        event_lookup: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        target_person_ids = {str(item) for item in list((clause_plan.get("hard_filters") or {}).get("person_ids", []) or [])}
        target_relationship_types = {str(item) for item in list((clause_plan.get("hard_filters") or {}).get("relationship_types", []) or [])}
        candidates: List[Dict[str, Any]] = []
        question_vector = self.embedder.embed_query(" ".join(list(clause_plan.get("rewrites") or [])[:2]) or "")
        max_shared_count = max((int(item.get("shared_event_count") or 0) for item in relationships), default=1)
        max_photo_count = max((int(item.get("photo_count") or 0) for item in relationships), default=1)
        for relationship in list(relationships or []):
            person_id = str(relationship.get("person_id") or "")
            if target_person_ids and person_id not in target_person_ids:
                continue
            relationship_type = str(relationship.get("relationship_type") or "")
            if target_relationship_types and relationship_type not in target_relationship_types:
                continue
            reasoning = " ".join(
                [
                    person_id,
                    relationship_type,
                    str(relationship.get("status") or ""),
                    str(relationship.get("reasoning") or ""),
                ]
            )
            reasoning_vector = self.embedder.embed_query(reasoning)
            semantic_score = cosine_similarity(question_vector, reasoning_vector)
            intimacy_score = float(relationship.get("intimacy_score") or 0.0)
            confidence = float(relationship.get("confidence") or 0.0)
            shared_event_count = int(relationship.get("shared_event_count") or 0)
            photo_count = int(relationship.get("photo_count") or 0)
            rank_score = (
                intimacy_score * 0.45
                + confidence * 0.25
                + (shared_event_count / max_shared_count) * 0.18
                + (photo_count / max_photo_count) * 0.12
            )
            score = semantic_score * 0.4 + rank_score
            supported_events = [
                row
                for row in list(relationship_support.get(str(relationship.get("relationship_id") or ""), []) or [])
                if str(row.get("event_id") or "") in event_lookup or str(row.get("photo_id") or "")
            ]
            if not supported_events:
                continue
            candidates.append(
                {
                    "relationship": dict(relationship),
                    "score": round(score, 4),
                }
            )
        candidates.sort(key=lambda item: (-float(item["score"]), -float((item["relationship"]).get("intimacy_score") or 0.0)))
        return candidates

    def _group_candidates(
        self,
        clause_plan: Dict[str, Any],
        groups: Sequence[Dict[str, Any]],
        group_members: Dict[str, List[Dict[str, Any]]],
        verified_events: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        group_ids = {str(item) for item in list((clause_plan.get("hard_filters") or {}).get("group_ids", []) or [])}
        if not group_ids:
            return []
        event_ids = {str(item["event"].get("event_id") or "") for item in list(verified_events or [])}
        candidates: List[Dict[str, Any]] = []
        for group in list(groups or []):
            group_id = str(group.get("group_id") or "")
            if group_ids and group_id not in group_ids:
                continue
            payload = dict(group.get("group_payload") or {})
            supporting = [
                ref
                for ref in list(payload.get("strong_evidence_refs", []) or [])
                if str((ref or {}).get("event_id") or "") in event_ids or not event_ids
            ]
            candidates.append(
                {
                    "group": dict(group),
                    "members": [dict(item) for item in group_members.get(group_id, [])],
                    "supporting_refs": supporting,
                }
            )
        candidates.sort(key=lambda item: (-float((item["group"]).get("confidence") or 0.0), str((item["group"]).get("group_id") or "")))
        return candidates

    def _event_views(self, bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
        events = {str(item.get("event_id") or ""): dict(item) for item in bundle["events"]}
        event_people = self._group_rows(bundle["event_people"], "event_id")
        event_places = self._group_rows(bundle["event_places"], "event_id")
        evidence = self._group_rows(bundle["evidence"], "event_id")
        relationship_support = self._group_rows(bundle["relationship_support"], "event_id")
        relationship_lookup = {str(item.get("relationship_id") or ""): dict(item) for item in bundle["relationships"]}
        docs: List[Dict[str, Any]] = []
        for event_id, event in events.items():
            persons = [item.get("person_id") for item in event_people.get(event_id, []) if item.get("person_id")]
            places = [item.get("place_ref") for item in event_places.get(event_id, []) if item.get("place_ref")]
            relationship_texts: List[str] = []
            for support in relationship_support.get(event_id, []):
                relationship = relationship_lookup.get(str(support.get("relationship_id") or ""))
                if not relationship:
                    continue
                relationship_texts.append(
                    f"{relationship.get('person_id')} {relationship.get('relationship_type')} intimacy {relationship.get('intimacy_score')}"
                )
            event_evidence = evidence.get(event_id, [])
            grouped = {
                "summary": " ".join([str(event.get("title") or ""), str(event.get("summary") or "")]).strip(),
                "people_relation": " ".join(persons + relationship_texts).strip(),
                "time_place": " ".join([str(event.get("start_ts") or ""), str(event.get("end_ts") or ""), " ".join(places)]).strip(),
                "activity_scene": " ".join(
                    str(item.get("text") or "")
                    for item in event_evidence
                    if str(item.get("evidence_type") or "") in {"scene", "activity", "interaction"}
                ).strip(),
                "object_ocr": " ".join(
                    str(item.get("text") or "")
                    for item in event_evidence
                    if str(item.get("evidence_type") or "") in {"object", "ocr", "brand"}
                ).strip(),
                "mood_style": " ".join(
                    str(item.get("text") or "")
                    for item in event_evidence
                    if str(item.get("evidence_type") or "") in {"mood_style", "relationship_signal"}
                ).strip(),
            }
            for view_type, retrieval_text in grouped.items():
                if retrieval_text:
                    docs.append({"event_id": event_id, "view_type": view_type, "retrieval_text": retrieval_text})
        return docs

    def _evidence_docs(self, bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
        docs = []
        for row in list(bundle["evidence"] or []):
            retrieval_text = str(row.get("text") or row.get("normalized_value") or "").strip()
            if retrieval_text:
                docs.append(
                    {
                        "evidence_id": row.get("evidence_id"),
                        "event_id": row.get("event_id"),
                        "evidence_type": row.get("evidence_type"),
                        "retrieval_text": retrieval_text,
                    }
                )
        return docs

    def _event_matches_hard_filters(
        self,
        clause_plan: Dict[str, Any],
        event: Dict[str, Any],
        event_people_rows: Sequence[Dict[str, Any]],
        event_place_rows: Sequence[Dict[str, Any]],
    ) -> bool:
        hard_filters = dict(clause_plan.get("hard_filters") or {})
        person_ids = {str(item) for item in list(hard_filters.get("person_ids", []) or [])}
        if person_ids:
            event_person_ids = {str(item.get("person_id") or "") for item in list(event_people_rows or [])}
            if not person_ids.intersection(event_person_ids):
                return False
        time_windows = self._normalize_time_windows(clause_plan.get("time_windows"))
        if time_windows:
            event_start = self._parse_dt(event.get("start_ts"))
            if event_start is None:
                return False
            if not any(self._event_in_window(event_start, window) for window in time_windows):
                return False
        return True

    def _event_in_window(self, event_start: datetime, window: Dict[str, Any]) -> bool:
        start_at = self._parse_dt(window.get("start_at"))
        end_at = self._parse_dt(window.get("end_at"))
        if start_at and event_start < start_at:
            return False
        if end_at and event_start > end_at:
            return False
        return True

    def _normalize_time_windows(self, value: Any) -> List[Dict[str, str]]:
        if not value:
            return []
        if isinstance(value, str):
            return self.planner._extract_time_windows(value)
        if not isinstance(value, (list, tuple)):
            return []
        normalized: List[Dict[str, str]] = []
        for item in value:
            if isinstance(item, str):
                normalized.extend(self.planner._extract_time_windows(item))
                continue
            if not isinstance(item, dict):
                continue
            start_at = str(item.get("start_at") or "").strip()
            end_at = str(item.get("end_at") or "").strip()
            raw = str(item.get("raw") or "").strip()
            if not start_at and not end_at and raw:
                normalized.extend(self.planner._extract_time_windows(raw))
                continue
            normalized.append(
                {
                    "label": str(item.get("label") or "window"),
                    "start_at": start_at,
                    "end_at": end_at,
                }
            )
        return normalized

    def _parse_dt(self, value: Any) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None

    def _lexical_bonus(self, clause_plan: Dict[str, Any], text: str) -> float:
        lowered_text = text.lower()
        bonus = 0.0
        for clue in list(clause_plan.get("detail_clues") or []):
            normalized = str(clue or "").strip().lower()
            if normalized and normalized in lowered_text:
                bonus += 0.04
        return min(bonus, 0.24)

    def _final_event_score(self, event: Dict[str, Any], score_payload: Dict[str, Any]) -> float:
        photo_bonus = 0.08 if int(event.get("photo_count") or 0) > 0 else -0.25
        confidence = float(event.get("confidence") or 0.0)
        return (
            float(score_payload.get("event_score") or 0.0) * 0.55
            + float(score_payload.get("evidence_score") or 0.0) * 0.25
            + float(score_payload.get("relationship_score") or 0.0) * 0.15
            + confidence * 0.1
            + photo_bonus
        )

    def _supporting_photos(self, verified_events: Sequence[Dict[str, Any]], photos_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen = set()
        for event_bundle in list(verified_events or []):
            for row in list(event_bundle.get("event_photos") or []):
                photo_id = str(row.get("photo_id") or "")
                if not photo_id or photo_id in seen:
                    continue
                seen.add(photo_id)
                payload = dict(photos_by_id.get(photo_id) or {"photo_id": photo_id})
                payload["supporting_event_id"] = event_bundle["event"].get("event_id")
                rows.append(payload)
        return rows

    def _serialize_matched_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        event = dict(payload["event"])
        photo_ids = [str(item.get("photo_id") or "") for item in payload.get("event_photos", []) if item.get("photo_id")]
        person_ids = [str(item.get("person_id") or "") for item in payload.get("event_people", []) if item.get("person_id")]
        place_refs = [str(item.get("place_ref") or "") for item in payload.get("event_places", []) if item.get("place_ref")]
        return {
            "event_id": event.get("event_id"),
            "title": event.get("title"),
            "summary": event.get("summary"),
            "start_ts": event.get("start_ts"),
            "end_ts": event.get("end_ts"),
            "photo_ids": self._unique(photo_ids),
            "person_ids": self._unique(person_ids),
            "place_refs": self._unique(place_refs),
            "confidence": event.get("confidence"),
            "score": round(float(payload.get("event_score") or 0.0), 4),
            "reasons": list(payload.get("reasons") or []),
        }

    def _serialize_supporting_evidence(self, verified_events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        seen = set()
        for event_bundle in list(verified_events or []):
            for evidence in list(event_bundle.get("supporting_evidence") or []):
                evidence_id = str(evidence.get("evidence_id") or "")
                if not evidence_id or evidence_id in seen:
                    continue
                seen.add(evidence_id)
                rows.append(
                    {
                        "evidence_id": evidence_id,
                        "event_id": evidence.get("event_id"),
                        "photo_id": evidence.get("photo_id"),
                        "evidence_type": evidence.get("evidence_type"),
                        "text": evidence.get("text"),
                        "confidence": evidence.get("confidence"),
                    }
                )
        return rows[:120]

    def _serialize_graph_entity(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        relationship = dict(candidate.get("relationship") or {})
        return {
            "entity_type": "relationship",
            "relationship_id": relationship.get("relationship_id"),
            "target_person_id": relationship.get("person_id"),
            "person_id": relationship.get("person_id"),
            "relationship_type": relationship.get("relationship_type"),
            "status": relationship.get("status"),
            "confidence": relationship.get("confidence"),
            "intimacy_score": relationship.get("intimacy_score"),
            "shared_event_count": relationship.get("shared_event_count"),
            "photo_count": relationship.get("photo_count"),
            "reasoning": relationship.get("reasoning"),
            "match_score": candidate.get("score"),
        }

    def _serialize_group_entity(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        group = dict(candidate.get("group") or {})
        return {
            "entity_type": "group",
            "group_id": group.get("group_id"),
            "group_type": group.get("group_type"),
            "confidence": group.get("confidence"),
            "members": [item.get("person_id") for item in list(candidate.get("members") or []) if item.get("person_id")],
            "supporting_refs": list(candidate.get("supporting_refs") or []),
        }

    def _serialize_fact(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        value = self._fact_value(fact)
        return {
            "fact_id": fact.get("fact_id"),
            "field_key": fact.get("field_key"),
            "value": value,
            "confidence": fact.get("confidence"),
            "source_level": fact.get("source_level"),
            "match_score": fact.get("match_score"),
            "evidence_event_ids": list(fact.get("evidence_event_ids") or []),
            "evidence_photo_ids": list(fact.get("evidence_photo_ids") or []),
        }

    def _serialize_clause_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "clause_id": result.get("clause_id"),
            "route": result.get("route"),
            "intent": result.get("intent"),
            "clause_text": result.get("clause_text"),
            "matched_event_ids": [item.get("event_id") for item in list(result.get("matched_events") or []) if item.get("event_id")],
            "matched_event_count": len(list(result.get("matched_events") or [])),
            "supporting_fact_keys": [item.get("field_key") for item in list(result.get("supporting_facts") or []) if item.get("field_key")],
            "supporting_relationship_ids": [
                item.get("relationship_id")
                for item in list(result.get("supporting_relationships") or [])
                if item.get("relationship_id")
            ],
            "supporting_photo_ids": [item.get("photo_id") for item in list(result.get("supporting_photos") or []) if item.get("photo_id")],
            "judgement_status": result.get("judgement_status"),
            "abstain_reason": result.get("abstain_reason"),
            "aggregation_result": result.get("aggregation_result"),
            "result_hint": result.get("result_hint"),
        }

    def _fact_value(self, fact: Dict[str, Any]) -> Any:
        if "value" in fact:
            return fact.get("value")
        value_json = fact.get("value_json")
        if isinstance(value_json, dict) and "value" in value_json:
            return value_json.get("value")
        return value_json

    def _fact_value_text(self, fact: Dict[str, Any]) -> str:
        value = self._fact_value(fact)
        if isinstance(value, list):
            return " ".join(str(item) for item in value[:8])
        if isinstance(value, dict):
            return " ".join(f"{key}:{value[key]}" for key in list(value.keys())[:8])
        return str(value or "")

    def _extract_evidence_refs(self, fact: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        event_ids: List[str] = []
        photo_ids: List[str] = []
        for payload in (fact.get("fact_payload"), fact.get("value_json")):
            self._walk_refs(payload, event_ids, photo_ids)
        return self._unique(event_ids), self._unique(photo_ids)

    def _walk_refs(self, payload: Any, event_ids: List[str], photo_ids: List[str]) -> None:
        if payload is None:
            return
        if isinstance(payload, dict):
            for key, value in payload.items():
                key_text = str(key).lower()
                if key_text == "event_id":
                    normalized = str(value or "").strip()
                    if normalized:
                        event_ids.append(normalized)
                    continue
                if key_text == "photo_id":
                    normalized = str(value or "").strip()
                    if normalized:
                        photo_ids.append(normalized)
                    continue
                if key_text in {"event_ids", "supporting_event_ids"} and isinstance(value, (list, tuple)):
                    event_ids.extend(str(item).strip() for item in value if str(item).strip())
                    continue
                if key_text in {"photo_ids", "supporting_photo_ids"} and isinstance(value, (list, tuple)):
                    photo_ids.extend(str(item).strip() for item in value if str(item).strip())
                    continue
                self._walk_refs(value, event_ids, photo_ids)
            return
        if isinstance(payload, (list, tuple)):
            for item in payload:
                self._walk_refs(item, event_ids, photo_ids)

    def _fact_judgement(self, clause: Dict[str, Any], facts: Sequence[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
        if not facts:
            return "insufficient_evidence", "fact_missing"
        best = facts[0]
        if float(best.get("confidence") or 0.0) < 0.35:
            return "insufficient_evidence", "fact_confidence_too_low"
        return "supported", None

    def _fact_result_hint(
        self,
        supporting_facts: Sequence[Dict[str, Any]],
        judgement_status: str,
        abstain_reason: Optional[str],
    ) -> str:
        if not supporting_facts:
            return f"当前没有足够画像事实。原因是 {abstain_reason or 'fact_missing'}。"
        fact = supporting_facts[0]
        value = fact.get("value")
        if isinstance(value, list):
            value_text = "、".join(str(item) for item in value[:4])
        else:
            value_text = str(value)
        return f"{fact.get('field_key')} 的结果是 {value_text}。"

    def _relationship_result_hint(
        self,
        supporting_relationships: Sequence[Dict[str, Any]],
        judgement_status: str,
        abstain_reason: Optional[str],
    ) -> str:
        if not supporting_relationships:
            return f"当前没有足够关系证据。原因是 {abstain_reason or 'insufficient_relationship_evidence'}。"
        relationship = supporting_relationships[0]
        return (
            f"{relationship.get('target_person_id') or relationship.get('person_id')} 当前更像 "
            f"{relationship.get('relationship_type') or 'unknown'}。"
        )

    def _event_result_hint(
        self,
        matched_events: Sequence[Dict[str, Any]],
        judgement_status: str,
        abstain_reason: Optional[str],
    ) -> str:
        if not matched_events:
            return f"当前没有召回到足够有照片支撑的相关事件。原因是 {abstain_reason or 'insufficient_event_evidence'}。"
        titles = "、".join(str(item.get("title") or item.get("event_id")) for item in list(matched_events)[:3])
        return f"共召回 {len(list(matched_events or []))} 个相关事件，重点包括 {titles}。"

    def _hybrid_judgement(
        self,
        clause: Dict[str, Any],
        *,
        supporting_facts: Sequence[Dict[str, Any]],
        supporting_relationships: Sequence[Dict[str, Any]],
        matched_events: Sequence[Dict[str, Any]],
    ) -> Tuple[str, Optional[str]]:
        clause_text = str(clause.get("clause_text") or "")
        lowered = clause_text.lower()
        if "工作" in clause_text or "career" in lowered:
            return self._binary_fact_judgement(
                supporting_facts,
                positive_markers=("正式工作", "已经工作", "上班", "在职", "employed", "full-time"),
                negative_markers=("学生", "在校", "尚未正式步入职场", "未工作", "school", "student"),
                empty_reason="career_fact_missing",
                prefer_negative=True,
            )
        if "上学" in clause_text or "学生" in clause_text or "school" in lowered or "student" in lowered:
            return self._binary_fact_judgement(
                supporting_facts,
                positive_markers=("学生", "在校", "school", "student"),
                negative_markers=("工作", "职场", "career", "employed"),
                empty_reason="education_fact_missing",
                prefer_negative=False,
            )
        if supporting_relationships:
            return "supported", None
        if matched_events:
            return "supported", None
        return "insufficient_evidence", "insufficient_evidence"

    def _binary_fact_judgement(
        self,
        supporting_facts: Sequence[Dict[str, Any]],
        *,
        positive_markers: Sequence[str],
        negative_markers: Sequence[str],
        empty_reason: str,
        prefer_negative: bool,
    ) -> Tuple[str, Optional[str]]:
        if not supporting_facts:
            return "insufficient_evidence", empty_reason
        text = " ".join(self._fact_value_text(fact) for fact in supporting_facts).lower()
        if prefer_negative and any(marker.lower() in text for marker in negative_markers):
            return "contradicted", None
        if any(marker.lower() in text for marker in positive_markers):
            return "supported", None
        if any(marker.lower() in text for marker in negative_markers):
            return "contradicted", None
        return "insufficient_evidence", "weak_fact_signal"

    def _hybrid_result_hint(
        self,
        judgement_status: str,
        supporting_facts: Sequence[Dict[str, Any]],
        supporting_relationships: Sequence[Dict[str, Any]],
        abstain_reason: Optional[str],
    ) -> str:
        if judgement_status == "supported":
            if supporting_facts:
                return self._fact_result_hint(supporting_facts, judgement_status, abstain_reason)
            if supporting_relationships:
                return self._relationship_result_hint(supporting_relationships, judgement_status, abstain_reason)
            return "当前证据足以支持这个判断。"
        if judgement_status == "contradicted":
            return "现有可审计证据更倾向于相反结论。"
        return f"当前没有足够可审计证据稳定判断。原因是 {abstain_reason or 'insufficient_evidence'}。"

    def _aggregate_matched_events(self, clause_results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for clause_result in list(clause_results or []):
            clause_id = str(clause_result.get("clause_id") or "")
            for item in list(clause_result.get("matched_events") or []):
                event_id = str(item.get("event_id") or "")
                if not event_id:
                    continue
                current = merged.get(event_id)
                if current is None:
                    payload = dict(item)
                    payload["clause_ids"] = [clause_id] if clause_id else []
                    merged[event_id] = payload
                    continue
                current["score"] = round(max(float(current.get("score") or 0.0), float(item.get("score") or 0.0)), 4)
                current["reasons"] = self._unique(list(current.get("reasons") or []) + list(item.get("reasons") or []))
                current["photo_ids"] = self._unique(list(current.get("photo_ids") or []) + list(item.get("photo_ids") or []))
                current["person_ids"] = self._unique(list(current.get("person_ids") or []) + list(item.get("person_ids") or []))
                current["place_refs"] = self._unique(list(current.get("place_refs") or []) + list(item.get("place_refs") or []))
                current["clause_ids"] = self._unique(list(current.get("clause_ids") or []) + ([clause_id] if clause_id else []))
        result = list(merged.values())
        result.sort(key=lambda item: (-float(item.get("score") or 0.0), str(item.get("start_ts") or "")))
        return result

    def _aggregate_unique_rows(self, clause_results: Sequence[Dict[str, Any]], key: str, *, id_key: str) -> List[Dict[str, Any]]:
        seen = set()
        rows: List[Dict[str, Any]] = []
        for clause_result in list(clause_results or []):
            for item in list(clause_result.get(key) or []):
                normalized = str(item.get(id_key) or "")
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                rows.append(dict(item))
        return rows

    def _final_judgement(self, clause_results: Sequence[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
        statuses = [str(item.get("judgement_status") or "") for item in list(clause_results or []) if item.get("judgement_status")]
        if not statuses:
            return "insufficient_evidence", "missing_clause_results"
        if all(status == "supported" for status in statuses):
            return "supported", None
        if any(status == "contradicted" for status in statuses) and not any(status == "supported" for status in statuses):
            return "contradicted", None
        if any(status == "insufficient_evidence" for status in statuses):
            first_reason = next((item.get("abstain_reason") for item in clause_results if item.get("abstain_reason")), None)
            if any(status == "supported" for status in statuses):
                return "supported", None
            return "insufficient_evidence", first_reason or "insufficient_evidence"
        return statuses[0], None

    def _aggregate_clause_aggregations(self, clause_results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        results = [item.get("aggregation_result") for item in list(clause_results or []) if item.get("aggregation_result")]
        if len(results) == 1:
            return dict(results[0])
        return {"kind": "composite", "clauses": results}

    def _answer_type(self, plan: Dict[str, Any], clause_results: Sequence[Dict[str, Any]]) -> str:
        composition_mode = str(plan.get("composition_mode") or "single")
        if composition_mode != "single" or len(list(clause_results or [])) > 1:
            return "composite_query"
        if not clause_results:
            return "event_search"
        route = str(clause_results[0].get("route") or "")
        intent = str(clause_results[0].get("intent") or "")
        if route == "fact-first":
            return "fact_lookup_query"
        if route == "relationship-first":
            return "relationship_rank_query" if intent == "rank" else "relationship_explore"
        if route == "hybrid-judgement":
            return "judgement_query"
        if intent == "count":
            return "count_query"
        if intent == "existence":
            return "existence_query"
        if intent == "compare":
            return "compare_query"
        return "event_search"

    def _plan_type(self, plan: Dict[str, Any]) -> str:
        routes = {str(item.get("route") or "") for item in list(plan.get("clauses") or [])}
        if len(routes) == 1:
            route = next(iter(routes))
            return route.replace("-", "_")
        return "composite"

    def _answer_confidence(
        self,
        matched_events: Sequence[Dict[str, Any]],
        supporting_relationships: Sequence[Dict[str, Any]],
        supporting_facts: Sequence[Dict[str, Any]],
    ) -> float:
        scores = [float(item.get("score") or 0.0) for item in list(matched_events or [])[:8]]
        scores.extend(float(item.get("confidence") or 0.0) for item in list(supporting_relationships or [])[:5])
        scores.extend(float(item.get("confidence") or 0.0) for item in list(supporting_facts or [])[:5])
        if not scores:
            return 0.0
        return round(min(max(sum(scores) / len(scores), 0.0), 1.0), 4)

    def _original_photo_ids(self, supporting_photos: Sequence[Dict[str, Any]]) -> List[str]:
        rows = [str(item.get("photo_id") or item.get("image_id") or "") for item in list(supporting_photos or [])]
        return self._unique(rows)

    def _legacy_unit(self, event: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "unit_id": event.get("event_id"),
            "source_type": "query_v1_event",
            "title": event.get("title"),
            "summary": event.get("summary"),
            "started_at": event.get("start_ts"),
            "ended_at": event.get("end_ts"),
            "original_photo_ids": list(event.get("photo_ids") or []),
            "confidence": event.get("confidence"),
        }

    def _group_rows(self, rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in list(rows or []):
            value = str(row.get(key) or "").strip()
            if not value:
                continue
            grouped[value].append(dict(row))
        return grouped

    def _unique(self, values: Iterable[Any]) -> List[str]:
        seen = set()
        result: List[str] = []
        for value in values:
            normalized = str(value or "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            result.append(normalized)
        return result
