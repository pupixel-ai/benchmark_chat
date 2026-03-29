"""Photo-backed query engine for v0325 / v0327-db tasks."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from memory_module.embeddings import EmbeddingProvider, cosine_similarity

from .materializer import SCHEMA_VERSION, SUPPORTED_QUERY_V1_VERSIONS, materialize_v0325_to_query_store
from .planner import StructuredQueryPlanner
from .store import QueryStore


class QueryEngineV1:
    """Query v1 facade with on-demand materialization + photo-backed retrieval."""

    def __init__(
        self,
        *,
        store: Optional[QueryStore] = None,
        planner: Optional[StructuredQueryPlanner] = None,
        embedder: Optional[EmbeddingProvider] = None,
        now: Optional[datetime] = None,
    ) -> None:
        self.store = store or QueryStore()
        self.now = now or datetime.now()
        self.planner = planner or StructuredQueryPlanner(now=self.now)
        self.embedder = embedder or EmbeddingProvider.from_config()

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

        known_person_ids = [item["person_id"] for item in bundle["relationships"] if item.get("person_id")]
        known_person_ids.extend(item["person_id"] for item in bundle["event_people"] if item.get("person_id"))
        known_group_ids = [item["group_id"] for item in bundle["groups"] if item.get("group_id")]
        known_relationship_types = [item["relationship_type"] for item in bundle["relationships"] if item.get("relationship_type")]

        plan = self.plan_query_v1(
            question,
            user_id=user_id,
            task_scope=str(task.get("task_id") or ""),
            known_person_ids=known_person_ids,
            known_group_ids=known_group_ids,
            known_relationship_types=known_relationship_types,
        )
        if time_hint and not plan.get("time_windows"):
            plan["time_windows"] = [{"label": "hint", "start_at": "", "end_at": "", "raw": time_hint}]
        if answer_shape_hint:
            plan["answer_shape_hint"] = answer_shape_hint
        if context_hints:
            plan["context_hints"] = dict(context_hints)

        retrieved = self.retrieve_events_v1(plan, bundle)
        return self.answer_query_v1(plan, retrieved, bundle, materialization)

    def plan_query_v1(
        self,
        question: str,
        *,
        user_id: str,
        task_scope: Optional[str],
        known_person_ids: Iterable[str],
        known_group_ids: Iterable[str],
        known_relationship_types: Iterable[str],
    ) -> Dict[str, Any]:
        return self.planner.plan(
            question,
            user_id=user_id,
            task_scope=task_scope,
            known_person_ids=known_person_ids,
            known_group_ids=known_group_ids,
            known_relationship_types=known_relationship_types,
        )

    def retrieve_events_v1(self, plan: Dict[str, Any], bundle: Dict[str, Any]) -> Dict[str, Any]:
        event_lookup = {str(item.get("event_id") or ""): dict(item) for item in bundle["events"] if item.get("event_id")}
        event_people = self._group_rows(bundle["event_people"], "event_id")
        event_places = self._group_rows(bundle["event_places"], "event_id")
        event_photos = self._group_rows(bundle["event_photos"], "event_id")
        evidence_by_event = self._group_rows(bundle["evidence"], "event_id")
        photos_by_id = {str(item.get("photo_id") or ""): dict(item) for item in bundle["photos"] if item.get("photo_id")}

        event_views = self._event_views(bundle)
        evidence_docs = self._evidence_docs(bundle)
        relationships = [dict(item) for item in bundle["relationships"]]
        relationship_support = self._group_rows(bundle["relationship_support"], "relationship_id")
        groups = [dict(item) for item in bundle["groups"]]
        group_members = self._group_rows(bundle["group_members"], "group_id")

        relationship_candidates = self._relationship_candidates(plan, relationships, relationship_support, event_lookup)
        relationship_event_ids: set[str] = set()
        query_mode = str(plan.get("query_mode") or "")
        if query_mode.startswith("relationship"):
            max_relationships = 1 if query_mode == "relationship_rank" else 3
            for item in relationship_candidates[:max_relationships]:
                relationship_id = str(((item.get("relationship") or {}).get("relationship_id")) or "")
                if not relationship_id:
                    continue
                for support in relationship_support.get(relationship_id, []):
                    event_id = str(support.get("event_id") or "")
                    if event_id:
                        relationship_event_ids.add(event_id)
        event_scores: Dict[str, Dict[str, Any]] = {}
        query_vectors = [self.embedder.embed_query(text) for text in list(plan.get("rewrites") or [])[:6] if str(text or "").strip()]
        for view in event_views:
            event_id = str(view.get("event_id") or "")
            if not event_id or event_id not in event_lookup:
                continue
            if not self._event_matches_hard_filters(plan, event_lookup[event_id], event_people.get(event_id, []), event_places.get(event_id, [])):
                continue
            retrieval_text = str(view.get("retrieval_text") or "").strip()
            if not retrieval_text:
                continue
            view_vector = self.embedder.embed_query(retrieval_text)
            semantic_score = max((cosine_similarity(query_vector, view_vector) for query_vector in query_vectors), default=0.0)
            lexical_bonus = self._lexical_bonus(plan, retrieval_text)
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

        for doc in evidence_docs:
            event_id = str(doc.get("event_id") or "")
            if not event_id or event_id not in event_lookup:
                continue
            if not self._event_matches_hard_filters(plan, event_lookup[event_id], event_people.get(event_id, []), event_places.get(event_id, [])):
                continue
            retrieval_text = str(doc.get("retrieval_text") or "").strip()
            if not retrieval_text:
                continue
            doc_vector = self.embedder.embed_query(retrieval_text)
            semantic_score = max((cosine_similarity(query_vector, doc_vector) for query_vector in query_vectors), default=0.0)
            lexical_bonus = self._lexical_bonus(plan, retrieval_text)
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

        graph_support: List[Dict[str, Any]] = []
        if relationship_candidates:
            graph_support.extend([item["relationship"] for item in relationship_candidates[:5]])
            for item in relationship_candidates[:5]:
                relationship = item["relationship"]
                for support in relationship_support.get(str(relationship.get("relationship_id") or ""), []):
                    event_id = str(support.get("event_id") or "")
                    if not event_id or event_id not in event_lookup:
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
                    current["relationship_score"] += float(item["score"])
                    current["reasons"].append(f"relationship:{relationship.get('person_id')}:{item['score']:.3f}")

        verified_events: List[Dict[str, Any]] = []
        for event_id, score_payload in event_scores.items():
            if relationship_event_ids and event_id not in relationship_event_ids:
                continue
            event = dict(event_lookup[event_id])
            if not event_photos.get(event_id):
                continue
            final_score = self._final_event_score(event, score_payload)
            if final_score < 0.14:
                continue
            verified_events.append(
                {
                    "event": event,
                    "event_score": final_score,
                    "event_people": [dict(item) for item in event_people.get(event_id, [])],
                    "event_places": [dict(item) for item in event_places.get(event_id, [])],
                    "event_photos": [dict(item) for item in event_photos.get(event_id, [])],
                    "supporting_evidence": [
                        dict(item)
                        for item in evidence_by_event.get(event_id, [])
                        if str(item.get("evidence_id") or "") in score_payload["supporting_evidence_ids"]
                    ],
                    "reasons": list(score_payload["reasons"]),
                }
            )
        verified_events.sort(key=lambda item: (-float(item["event_score"]), str(item["event"].get("start_ts") or "")), reverse=False)

        supporting_photos = self._supporting_photos(verified_events, photos_by_id)
        return {
            "events": verified_events[:12],
            "relationships": relationship_candidates[:5],
            "groups": self._group_candidates(plan, groups, group_members, verified_events)[:5],
            "supporting_photos": supporting_photos,
            "photos_by_id": photos_by_id,
        }

    def answer_query_v1(
        self,
        plan: Dict[str, Any],
        retrieved: Dict[str, Any],
        bundle: Dict[str, Any],
        materialization: Dict[str, Any],
    ) -> Dict[str, Any]:
        matched_events = [self._serialize_matched_event(item) for item in retrieved["events"]]
        supporting_evidence = self._serialize_supporting_evidence(retrieved["events"])
        supporting_photos = list(retrieved["supporting_photos"])
        supporting_graph_entities = [self._serialize_graph_entity(item) for item in retrieved["relationships"]]
        if retrieved["groups"]:
            supporting_graph_entities.extend(self._serialize_group_entity(item) for item in retrieved["groups"])

        query_mode = str(plan.get("query_mode") or "event_search")
        answer_type = {
            "event_search": "event_search",
            "count": "count_query",
            "existence": "existence_query",
            "compare": "compare_query",
            "trend": "trend_query",
            "relationship_rank": "relationship_rank_query",
            "relationship_explain": "relationship_explore",
        }.get(query_mode, "event_search")

        summary, aggregation_result, abstain_reason = self._build_summary(
            plan,
            matched_events,
            supporting_graph_entities,
        )
        confidence = self._answer_confidence(matched_events, supporting_graph_entities)
        original_photo_ids = self._original_photo_ids(supporting_photos)

        answer = {
            "answer_type": answer_type,
            "summary": summary,
            "confidence": confidence,
            "original_photo_ids": original_photo_ids,
            "matched_events": matched_events,
            "supporting_photos": supporting_photos,
            "supporting_evidence": supporting_evidence,
            "graph_support": supporting_graph_entities,
            "aggregation_result": aggregation_result,
            "abstain_reason": abstain_reason,
            "materialization_id": materialization.get("materialization_id"),
            "engine": "query_v1",
        }
        return {
            "query_plan": {
                "engine": "query_v1",
                "schema_version": SCHEMA_VERSION,
                "plan_type": "graph_first_exact" if query_mode.startswith("relationship") else "hybrid",
                **plan,
            },
            "answer": answer,
            "matched_events": matched_events,
            "supporting_units": [self._legacy_unit(item) for item in matched_events],
            "supporting_evidence": supporting_evidence,
            "supporting_graph_entities": supporting_graph_entities,
            "supporting_photos": supporting_photos,
            "graph_support": supporting_graph_entities,
            "aggregation_result": aggregation_result,
            "abstain_reason": abstain_reason,
        }

    def _group_rows(self, rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in list(rows or []):
            value = str(row.get(key) or "").strip()
            if not value:
                continue
            grouped[value].append(dict(row))
        return grouped

    def _relationship_candidates(
        self,
        plan: Dict[str, Any],
        relationships: Sequence[Dict[str, Any]],
        relationship_support: Dict[str, List[Dict[str, Any]]],
        event_lookup: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        query_mode = str(plan.get("query_mode") or "")
        hard_filters = dict(plan.get("hard_filters") or {})
        target_person_ids = {str(item) for item in list(hard_filters.get("person_ids", []) or [])}
        target_relationship_types = {str(item) for item in list(hard_filters.get("relationship_types", []) or [])}
        candidates: List[Dict[str, Any]] = []
        if query_mode not in {"relationship_rank", "relationship_explain"} and not target_relationship_types:
            return candidates
        question_vector = self.embedder.embed_query(" ".join(list(plan.get("rewrites") or [])[:2]) or "")
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
            score = rank_score if query_mode == "relationship_rank" else (semantic_score + rank_score * 0.6)
            supported_events = [
                row
                for row in list(relationship_support.get(str(relationship.get("relationship_id") or ""), []) or [])
                if str(row.get("event_id") or "") in event_lookup
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
        plan: Dict[str, Any],
        groups: Sequence[Dict[str, Any]],
        group_members: Dict[str, List[Dict[str, Any]]],
        verified_events: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        group_ids = {str(item) for item in list((plan.get("hard_filters") or {}).get("group_ids", []) or [])}
        if not group_ids and str(plan.get("query_mode") or "") != "event_search":
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
            if group_ids and not supporting:
                continue
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
                "activity_scene": " ".join(str(item.get("text") or "") for item in event_evidence if str(item.get("evidence_type") or "") in {"scene", "activity", "interaction"}).strip(),
                "object_ocr": " ".join(str(item.get("text") or "") for item in event_evidence if str(item.get("evidence_type") or "") in {"object", "ocr", "brand"}).strip(),
                "mood_style": " ".join(str(item.get("text") or "") for item in event_evidence if str(item.get("evidence_type") or "") in {"mood_style", "relationship_signal"}).strip(),
            }
            for view_type, retrieval_text in grouped.items():
                if retrieval_text:
                    docs.append(
                        {
                            "event_id": event_id,
                            "view_type": view_type,
                            "retrieval_text": retrieval_text,
                        }
                    )
        return docs

    def _evidence_docs(self, bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
        docs = []
        for row in list(bundle["evidence"] or []):
            retrieval_text = str(row.get("text") or row.get("normalized_value") or "").strip()
            if retrieval_text:
                docs.append({"evidence_id": row.get("evidence_id"), "event_id": row.get("event_id"), "evidence_type": row.get("evidence_type"), "retrieval_text": retrieval_text})
        return docs

    def _event_matches_hard_filters(
        self,
        plan: Dict[str, Any],
        event: Dict[str, Any],
        event_people_rows: Sequence[Dict[str, Any]],
        event_place_rows: Sequence[Dict[str, Any]],
    ) -> bool:
        hard_filters = dict(plan.get("hard_filters") or {})
        person_ids = {str(item) for item in list(hard_filters.get("person_ids", []) or [])}
        if person_ids:
            event_person_ids = {str(item.get("person_id") or "") for item in list(event_people_rows or [])}
            if not person_ids.intersection(event_person_ids):
                return False
        time_windows = list(plan.get("time_windows") or [])
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

    def _parse_dt(self, value: Any) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            return None

    def _lexical_bonus(self, plan: Dict[str, Any], text: str) -> float:
        lowered_text = text.lower()
        bonus = 0.0
        for clue in list(plan.get("detail_clues") or []):
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
        return rows[:30]

    def _serialize_graph_entity(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        relationship = dict(candidate.get("relationship") or {})
        return {
            "entity_type": "relationship",
            "relationship_id": relationship.get("relationship_id"),
            "target_person_id": relationship.get("person_id"),
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

    def _build_summary(
        self,
        plan: Dict[str, Any],
        matched_events: Sequence[Dict[str, Any]],
        supporting_graph_entities: Sequence[Dict[str, Any]],
    ) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
        query_mode = str(plan.get("query_mode") or "event_search")
        if query_mode == "relationship_rank":
            if supporting_graph_entities:
                top = supporting_graph_entities[0]
                return (
                    f"当前最强关系对象是 {top.get('target_person_id')}，关系类型更像 {top.get('relationship_type')}，亲密度 {float(top.get('intimacy_score') or 0.0):.2f}。",
                    {
                        "kind": "relationship_rank",
                        "top_person_id": top.get("target_person_id"),
                        "relationship_type": top.get("relationship_type"),
                        "intimacy_score": top.get("intimacy_score"),
                    },
                    None,
                )
            return "当前没有足够证据判断最强关系。", None, "insufficient_evidence"
        if query_mode == "relationship_explain":
            if supporting_graph_entities:
                top = supporting_graph_entities[0]
                return (
                    f"{top.get('target_person_id')} 当前更像 {top.get('relationship_type')}，状态 {top.get('status') or 'unknown'}。",
                    {
                        "kind": "relationship_explain",
                        "target_person_id": top.get("target_person_id"),
                        "relationship_type": top.get("relationship_type"),
                    },
                    None,
                )
            return "当前没有足够关系证据。", None, "insufficient_evidence"
        if query_mode == "count":
            count = len(matched_events)
            return f"共找到 {count} 个相关事件。", {"kind": "count", "count": count}, None
        if query_mode == "existence":
            exists = bool(matched_events)
            return ("找到了相关事件。" if exists else "没有找到足够证据支持相关事件。"), {"kind": "existence", "exists": exists}, None if exists else "insufficient_coverage"
        if query_mode in {"compare", "trend"}:
            if matched_events:
                titles = "、".join(item.get("title") or item.get("event_id") for item in list(matched_events)[:3])
                return f"当前先召回到这些关键事件：{titles}。", {"kind": query_mode, "matched_event_count": len(matched_events)}, None
            return "当前没有召回到足够事件做比较或趋势判断。", None, "insufficient_evidence"
        if matched_events:
            titles = "、".join(item.get("title") or item.get("event_id") for item in list(matched_events)[:3])
            return f"找到 {len(matched_events)} 个相关事件：{titles}。", {"kind": "event_search", "matched_event_count": len(matched_events)}, None
        return "没有找到足够有照片支撑的相关事件。", None, "insufficient_evidence"

    def _answer_confidence(self, matched_events: Sequence[Dict[str, Any]], supporting_graph_entities: Sequence[Dict[str, Any]]) -> float:
        scores = [float(item.get("score") or 0.0) for item in list(matched_events or [])]
        scores.extend(float(item.get("confidence") or 0.0) for item in list(supporting_graph_entities or []))
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
