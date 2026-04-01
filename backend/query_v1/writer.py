"""Final summary writer for query v1."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import requests

from config import OPENROUTER_AGENT_MODEL, OPENROUTER_API_KEY, OPENROUTER_APP_NAME, OPENROUTER_BASE_URL, OPENROUTER_SITE_URL


class AnswerWriterLLM:
    """Writes a short summary after retrieval without changing retrieved events."""

    def write(
        self,
        *,
        question: str,
        route_plan: Dict[str, Any],
        clause_results: Sequence[Dict[str, Any]],
        matched_events: Sequence[Dict[str, Any]],
        top_event_ids_for_summary: Sequence[str],
        supporting_photos: Sequence[Dict[str, Any]],
        supporting_facts: Sequence[Dict[str, Any]],
        supporting_relationships: Sequence[Dict[str, Any]],
        judgement_status: str,
        abstain_reason: Optional[str],
    ) -> Dict[str, Any]:
        payload = {
            "summary_text": "",
            "summary_event_ids": list(top_event_ids_for_summary or [])[:6],
            "summary_fact_keys": [item.get("field_key") for item in list(supporting_facts or [])[:6] if item.get("field_key")],
            "summary_relationship_ids": [
                item.get("relationship_id")
                for item in list(supporting_relationships or [])[:6]
                if item.get("relationship_id")
            ],
            "writer_source": "template",
        }
        llm_payload = self._llm_summary(
            question=question,
            route_plan=route_plan,
            clause_results=clause_results,
            matched_events=matched_events,
            top_event_ids_for_summary=top_event_ids_for_summary,
            supporting_photos=supporting_photos,
            supporting_facts=supporting_facts,
            supporting_relationships=supporting_relationships,
            judgement_status=judgement_status,
            abstain_reason=abstain_reason,
        )
        if isinstance(llm_payload, dict):
            payload.update({key: value for key, value in llm_payload.items() if value not in (None, "")})
            payload["writer_source"] = "llm"
        if not str(payload.get("summary_text") or "").strip():
            payload["summary_text"] = self._template_summary(
                question=question,
                clause_results=clause_results,
                matched_events=matched_events,
                top_event_ids_for_summary=top_event_ids_for_summary,
                supporting_facts=supporting_facts,
                supporting_relationships=supporting_relationships,
                judgement_status=judgement_status,
                abstain_reason=abstain_reason,
            )
            payload["writer_source"] = payload.get("writer_source") or "template"
        payload["summary_event_ids"] = [str(item) for item in list(payload.get("summary_event_ids") or []) if str(item).strip()][:6]
        payload["summary_fact_keys"] = [str(item) for item in list(payload.get("summary_fact_keys") or []) if str(item).strip()][:6]
        payload["summary_relationship_ids"] = [str(item) for item in list(payload.get("summary_relationship_ids") or []) if str(item).strip()][:6]
        return payload

    def _llm_summary(
        self,
        *,
        question: str,
        route_plan: Dict[str, Any],
        clause_results: Sequence[Dict[str, Any]],
        matched_events: Sequence[Dict[str, Any]],
        top_event_ids_for_summary: Sequence[str],
        supporting_photos: Sequence[Dict[str, Any]],
        supporting_facts: Sequence[Dict[str, Any]],
        supporting_relationships: Sequence[Dict[str, Any]],
        judgement_status: str,
        abstain_reason: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        if not OPENROUTER_API_KEY:
            return None
        top_events = [
            item
            for item in list(matched_events or [])
            if str(item.get("event_id") or "") in set(str(event_id) for event_id in list(top_event_ids_for_summary or []))
        ]
        payload = {
            "model": OPENROUTER_AGENT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the final summary writer for a photo-backed retrieval API. "
                        "Return JSON only with keys: summary_text, summary_event_ids, summary_fact_keys, summary_relationship_ids. "
                        "Do not invent events, people, dates, locations, numbers, or facts. "
                        "Only summarize evidence that already appears in the provided payload."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "question": question,
                            "route_plan": route_plan,
                            "clause_results": clause_results,
                            "top_events": top_events,
                            "top_event_ids_for_summary": list(top_event_ids_for_summary or []),
                            "supporting_photos": list(supporting_photos or [])[:12],
                            "supporting_facts": list(supporting_facts or [])[:12],
                            "supporting_relationships": list(supporting_relationships or [])[:12],
                            "judgement_status": judgement_status,
                            "abstain_reason": abstain_reason,
                            "constraints": {
                                "max_sentences": 4,
                                "must_cover_all_clauses": True,
                                "must_not_change_retrieved_events": True,
                            },
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-Title": OPENROUTER_APP_NAME,
        }
        try:
            response = requests.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=20,
            )
            response.raise_for_status()
            body = response.json()
            content = ((((body.get("choices") or [{}])[0].get("message") or {}).get("content")) or "{}")
            parsed = json.loads(content)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _template_summary(
        self,
        *,
        question: str,
        clause_results: Sequence[Dict[str, Any]],
        matched_events: Sequence[Dict[str, Any]],
        top_event_ids_for_summary: Sequence[str],
        supporting_facts: Sequence[Dict[str, Any]],
        supporting_relationships: Sequence[Dict[str, Any]],
        judgement_status: str,
        abstain_reason: Optional[str],
    ) -> str:
        if judgement_status == "insufficient_evidence":
            return f"当前没有足够可审计证据稳定回答这个问题。原因是 {abstain_reason or 'insufficient_evidence'}。"
        clause_hints = [str(item.get("result_hint") or "").strip() for item in list(clause_results or []) if str(item.get("result_hint") or "").strip()]
        if clause_hints:
            summary = " ".join(clause_hints[:3]).strip()
            if summary:
                return summary
        if supporting_relationships:
            relationship = supporting_relationships[0]
            return (
                f"关系侧最强命中是 {relationship.get('target_person_id') or relationship.get('person_id')}，"
                f"类型更像 {relationship.get('relationship_type') or 'unknown'}。"
            )
        if supporting_facts:
            fact = supporting_facts[0]
            value = fact.get("value")
            if isinstance(value, list):
                value_text = "、".join(str(item) for item in value[:4])
            else:
                value_text = str(value)
            return f"{fact.get('field_key')} 的当前结果是 {value_text}。"
        if matched_events:
            titles = "、".join(
                str(item.get("title") or item.get("event_id") or "")
                for item in list(matched_events or [])
                if str(item.get("event_id") or "") in set(str(event_id) for event_id in list(top_event_ids_for_summary or []))
            )
            if titles:
                return f"共召回 {len(list(matched_events or []))} 个相关事件，重点包括 {titles}。"
        return f"当前问题是：{question}。系统还没有生成更具体的摘要。"
