"""Structured planner for query v1."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

import requests

from config import (
    GEMINI_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_LLM_MODEL,
    OPENROUTER_SITE_URL,
)


KINSHIP_RELATION_MAP = {
    "父亲": "father",
    "爸爸": "father",
    "dad": "father",
    "father": "father",
    "母亲": "mother",
    "妈妈": "mother",
    "mother": "mother",
    "伴侣": "romantic",
    "对象": "romantic",
    "romantic": "romantic",
    "情侣": "romantic",
    "朋友": "friend",
    "好友": "friend",
    "friend": "friend",
    "bestie": "bestie",
    "闺蜜": "bestie",
    "同事": "coworker",
    "coworker": "coworker",
    "家人": "family",
    "family": "family",
}


class StructuredQueryPlanner:
    """Produces a structured plan without relying on regex-heavy search logic."""

    def __init__(self, *, now: Optional[datetime] = None) -> None:
        self.now = now or datetime.now()

    def plan(
        self,
        question: str,
        *,
        user_id: str,
        task_scope: Optional[str],
        known_person_ids: Iterable[str],
        known_group_ids: Iterable[str],
        known_relationship_types: Iterable[str],
    ) -> Dict[str, Any]:
        anchor_plan = self._deterministic_plan(
            question,
            task_scope=task_scope,
            known_person_ids=known_person_ids,
            known_group_ids=known_group_ids,
            known_relationship_types=known_relationship_types,
        )
        llm_plan = self._llm_plan(
            question,
            user_id=user_id,
            task_scope=task_scope,
            anchor_plan=anchor_plan,
            known_person_ids=known_person_ids,
            known_group_ids=known_group_ids,
            known_relationship_types=known_relationship_types,
        )
        if llm_plan:
            merged = dict(anchor_plan)
            merged.update({key: value for key, value in llm_plan.items() if value not in (None, "", [], {})})
            merged["planner_source"] = "hybrid_llm"
            merged.setdefault("rewrites", [question])
            if question not in merged["rewrites"]:
                merged["rewrites"] = [question] + list(merged["rewrites"])
            return merged
        anchor_plan["planner_source"] = "deterministic"
        return anchor_plan

    def _deterministic_plan(
        self,
        question: str,
        *,
        task_scope: Optional[str],
        known_person_ids: Iterable[str],
        known_group_ids: Iterable[str],
        known_relationship_types: Iterable[str],
    ) -> Dict[str, Any]:
        normalized = str(question or "").strip()
        lowered = normalized.lower()
        query_mode = "event_search"
        aggregation = "none"
        if any(token in normalized for token in ("几次", "多少次", "多少个", "count", "how many")):
            query_mode = "count"
            aggregation = "count"
        elif any(token in normalized for token in ("有没有", "是否", "有没有过", "有无", "exist")):
            query_mode = "existence"
            aggregation = "existence"
        elif any(token in normalized for token in ("对比", "相比", "compare", "比", "更")):
            query_mode = "compare"
            aggregation = "compare"
        elif any(token in normalized for token in ("趋势", "变化", "变得", "trend", "change")):
            query_mode = "trend"
            aggregation = "trend"
        if any(token in normalized for token in ("关系最好", "最亲密", "closest", "best relationship", "best friend")):
            query_mode = "relationship_rank"
            aggregation = "rank"
        elif any(token in normalized for token in ("关系", "relationship", "父亲", "母亲", "伴侣", "同事", "好友")):
            query_mode = "relationship_explain"

        hard_filters: Dict[str, Any] = {}
        target_person_ids = [
            person_id
            for person_id in known_person_ids
            if person_id and person_id.lower() in lowered
        ]
        if target_person_ids:
            hard_filters["person_ids"] = target_person_ids
        target_group_ids = [
            group_id
            for group_id in known_group_ids
            if group_id and group_id.lower() in lowered
        ]
        if target_group_ids:
            hard_filters["group_ids"] = target_group_ids
        target_relationship_types = [
            rel_type
            for rel_type in known_relationship_types
            if rel_type and rel_type.lower() in lowered
        ]
        for token, rel_type in KINSHIP_RELATION_MAP.items():
            if token in lowered or token in normalized:
                if rel_type not in target_relationship_types:
                    target_relationship_types.append(rel_type)
        if target_relationship_types:
            hard_filters["relationship_types"] = target_relationship_types

        time_windows = self._extract_time_windows(normalized)
        if time_windows:
            hard_filters["time_windows"] = time_windows

        same_event_required = aggregation == "none"
        detail_clues = self._detail_clues(normalized)
        soft_clues = [normalized]
        rewrites = [normalized]
        if detail_clues:
            rewrites.append(" ".join(detail_clues))
        if target_relationship_types:
            rewrites.append(" ".join(target_relationship_types + detail_clues[:4]))
        return {
            "query_mode": query_mode,
            "task_scope": task_scope,
            "time_windows": time_windows,
            "hard_filters": hard_filters,
            "soft_clues": soft_clues,
            "detail_clues": detail_clues,
            "aggregation": aggregation,
            "same_event_required": same_event_required,
            "photo_required": True,
            "abstain_policy": "strict_negative_requires_coverage",
            "rewrites": [item for item in rewrites if item],
        }

    def _extract_time_windows(self, text: str) -> List[Dict[str, str]]:
        lowered = text.lower()
        windows: List[Dict[str, str]] = []
        month_match = re.search(r"过去\s*(\d+)\s*个?月", text)
        if month_match:
            months = max(1, int(month_match.group(1)))
            end_at = self.now
            start_at = end_at - timedelta(days=months * 30)
            windows.append({"label": f"last_{months}_months", "start_at": start_at.isoformat(), "end_at": end_at.isoformat()})
        if "最近" in text or "recent" in lowered:
            end_at = self.now
            start_at = end_at - timedelta(days=30)
            windows.append({"label": "recent_30d", "start_at": start_at.isoformat(), "end_at": end_at.isoformat()})
        if "去年" in text or "last year" in lowered:
            start_at = datetime(self.now.year - 1, 1, 1)
            end_at = datetime(self.now.year - 1, 12, 31, 23, 59, 59)
            windows.append({"label": "last_year", "start_at": start_at.isoformat(), "end_at": end_at.isoformat()})
        year_match = re.search(r"\b(20\d{2})\b", text)
        if year_match:
            year = int(year_match.group(1))
            start_at = datetime(year, 1, 1)
            end_at = datetime(year, 12, 31, 23, 59, 59)
            windows.append({"label": f"year_{year}", "start_at": start_at.isoformat(), "end_at": end_at.isoformat()})
        deduped: List[Dict[str, str]] = []
        seen = set()
        for item in windows:
            key = (item.get("start_at"), item.get("end_at"))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _detail_clues(self, text: str) -> List[str]:
        parts = [item for item in re.split(r"[\s,，。？！;；/]+", text) if item]
        stopwords = {"我", "用户", "主角", "现在", "最近", "这个", "那个", "一下", "请", "帮我", "帮", "是否", "有没有"}
        result: List[str] = []
        for part in parts:
            normalized = part.strip()
            if len(normalized) <= 1 or normalized in stopwords:
                continue
            if normalized not in result:
                result.append(normalized)
        return result[:12]

    def _llm_plan(
        self,
        question: str,
        *,
        user_id: str,
        task_scope: Optional[str],
        anchor_plan: Dict[str, Any],
        known_person_ids: Iterable[str],
        known_group_ids: Iterable[str],
        known_relationship_types: Iterable[str],
    ) -> Optional[Dict[str, Any]]:
        if not OPENROUTER_API_KEY and not GEMINI_API_KEY:
            return None
        if not OPENROUTER_API_KEY:
            return None
        payload = {
            "model": OPENROUTER_LLM_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a query planner for a photo-backed memory retrieval system. "
                        "Return JSON only. Do not answer the question. "
                        "The output schema keys are: query_mode, task_scope, time_windows, hard_filters, soft_clues, "
                        "detail_clues, aggregation, same_event_required, photo_required, abstain_policy, rewrites."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_id": user_id,
                            "task_scope": task_scope,
                            "question": question,
                            "anchor_plan": anchor_plan,
                            "known_person_ids": list(known_person_ids),
                            "known_group_ids": list(known_group_ids),
                            "known_relationship_types": list(known_relationship_types),
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
            content = (
                (((body.get("choices") or [{}])[0].get("message") or {}).get("content"))
                or "{}"
            )
            plan = json.loads(content)
            return plan if isinstance(plan, dict) else None
        except Exception:
            return None
