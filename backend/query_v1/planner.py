"""Structured router for query v1."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests

from config import (
    GEMINI_API_KEY,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_LLM_MODEL,
    OPENROUTER_SITE_URL,
)


ROUTER_VERSION = "route_plan_v1"

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
    "恋人": "romantic",
    "老公": "romantic",
    "老婆": "romantic",
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

FACT_KEY_ALIASES: Dict[str, Sequence[str]] = {
    "宠物": (
        "long_term_facts.relationships.pets",
        "long_term_facts.life.pet_companions",
        "pets",
        "pet",
    ),
    "pet": (
        "long_term_facts.relationships.pets",
        "long_term_facts.life.pet_companions",
        "pets",
    ),
    "地点锚点": (
        "long_term_facts.geography.location_anchors",
        "location_anchor",
        "location_anchors",
        "geography.location_anchor",
    ),
    "location anchor": (
        "long_term_facts.geography.location_anchors",
        "location_anchor",
        "location_anchors",
    ),
    "亲密圈大小": (
        "long_term_facts.relationships.close_circle_size",
        "close_circle_size",
        "close_circle.count",
    ),
    "close circle": (
        "long_term_facts.relationships.close_circle_size",
        "close_circle_size",
    ),
    "长期兴趣": (
        "long_term_facts.interests",
        "interests",
        "hobbies",
    ),
    "兴趣": (
        "long_term_facts.interests",
        "interests",
        "hobbies",
    ),
    "hobby": (
        "long_term_facts.interests",
        "interests",
        "hobbies",
    ),
    "工作": (
        "social_identity.career",
        "social_identity.career_phase",
        "identity.role",
        "material.income_model",
        "career",
        "career_phase",
        "work_status",
    ),
    "职场": (
        "social_identity.career",
        "social_identity.career_phase",
        "career",
        "career_phase",
    ),
    "career": (
        "social_identity.career",
        "social_identity.career_phase",
        "career",
        "career_phase",
    ),
    "上学": (
        "social_identity.education_phase",
        "social_identity.career_phase",
        "education",
        "student_status",
        "schooling",
    ),
    "学生": (
        "social_identity.education_phase",
        "social_identity.career_phase",
        "education",
        "student_status",
    ),
    "school": (
        "social_identity.education_phase",
        "education",
        "student_status",
    ),
}

CLAUSE_QUERY_MARKERS = (
    "谁",
    "什么",
    "哪里",
    "在哪",
    "多少",
    "几次",
    "有没有",
    "是否",
    "能不能",
    "请说明",
    "分别",
    "关系",
    "做了哪些事情",
    "做了什么",
)


class StructuredQueryPlanner:
    """Produces a route+resolve plan for photo-backed retrieval."""

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
        known_profile_fact_keys: Iterable[str],
    ) -> Dict[str, Any]:
        deterministic = self._deterministic_plan(
            question,
            task_scope=task_scope,
            known_person_ids=known_person_ids,
            known_group_ids=known_group_ids,
            known_relationship_types=known_relationship_types,
            known_profile_fact_keys=known_profile_fact_keys,
        )
        llm_plan = self._llm_route(
            question,
            user_id=user_id,
            task_scope=task_scope,
            deterministic_plan=deterministic,
            known_person_ids=known_person_ids,
            known_group_ids=known_group_ids,
            known_relationship_types=known_relationship_types,
            known_profile_fact_keys=known_profile_fact_keys,
        )
        if llm_plan:
            normalized = self._normalize_route_plan(
                llm_plan,
                question=question,
                task_scope=task_scope,
                known_person_ids=known_person_ids,
                known_group_ids=known_group_ids,
                known_relationship_types=known_relationship_types,
                known_profile_fact_keys=known_profile_fact_keys,
            )
            if normalized:
                return self._merge_plans(deterministic, normalized)
        deterministic["router_source"] = "deterministic"
        deterministic["router_fallback"] = "deterministic"
        return deterministic

    def _deterministic_plan(
        self,
        question: str,
        *,
        task_scope: Optional[str],
        known_person_ids: Iterable[str],
        known_group_ids: Iterable[str],
        known_relationship_types: Iterable[str],
        known_profile_fact_keys: Iterable[str],
    ) -> Dict[str, Any]:
        normalized_question = str(question or "").strip()
        composition_mode, clause_texts = self._split_clauses(normalized_question)
        clauses = []
        for index, clause_text in enumerate(clause_texts, start=1):
            clauses.append(
                self._build_clause(
                    clause_id=f"clause_{index}",
                    clause_text=clause_text,
                    task_scope=task_scope,
                    known_person_ids=known_person_ids,
                    known_group_ids=known_group_ids,
                    known_relationship_types=known_relationship_types,
                    known_profile_fact_keys=known_profile_fact_keys,
                    depends_on_clause_ids=[],
                )
            )
        needs_abstain = any(bool(item.get("can_abstain")) for item in clauses)
        return {
            "router_version": ROUTER_VERSION,
            "normalized_question": normalized_question,
            "composition_mode": composition_mode,
            "global_constraints": {
                "task_scope": task_scope,
                "photo_required": True,
            },
            "clauses": clauses,
            "router_confidence": self._router_confidence(clauses),
            "needs_abstain_judgement": needs_abstain,
            "router_source": "deterministic",
            "router_fallback": None,
        }

    def _split_clauses(self, question: str) -> tuple[str, List[str]]:
        normalized = str(question or "").strip()
        lowered = normalized.lower()
        if "既" in normalized and "也" in normalized:
            cleaned = (
                normalized.replace("分别是什么", "")
                .replace("分别是哪些", "")
                .replace("分别", "")
                .replace("吗", "")
                .replace("？", "")
                .replace("?", "")
            )
            fragment = cleaned.split("既", 1)[-1]
            left, right = fragment.split("也", 1)
            left = left.strip(" ，,。")
            right = right.strip(" ，,。")
            if left and right:
                return "and_then_enumerate", [left, right]
        if "如果不能" in normalized or "如果没有" in normalized:
            parts = [part.strip() for part in re.split(r"[？?]", normalized) if part.strip()]
            if parts:
                return "judge_then_explain", parts
        if any(token in lowered for token in ("对比", "compare", "相比")):
            return "compare", [normalized]

        clause_texts: List[str] = []
        for chunk in re.split(r"[？?]", normalized):
            chunk = chunk.strip(" ，,。;；")
            if not chunk:
                continue
            comma_parts = [part.strip(" ，,。;；") for part in re.split(r"[，,]", chunk) if part.strip(" ，,。;；")]
            if len(comma_parts) <= 1:
                clause_texts.append(chunk)
                continue
            segmented = []
            for part in comma_parts:
                if any(marker in part for marker in CLAUSE_QUERY_MARKERS):
                    segmented.append(part)
                elif segmented:
                    segmented[-1] = f"{segmented[-1]}，{part}"
                else:
                    segmented.append(part)
            clause_texts.extend(segmented)
        clause_texts = [item for item in clause_texts if item]
        if not clause_texts:
            clause_texts = [normalized]
        composition_mode = "parallel" if len(clause_texts) > 1 else "single"
        return composition_mode, clause_texts

    def _build_clause(
        self,
        *,
        clause_id: str,
        clause_text: str,
        task_scope: Optional[str],
        known_person_ids: Iterable[str],
        known_group_ids: Iterable[str],
        known_relationship_types: Iterable[str],
        known_profile_fact_keys: Iterable[str],
        depends_on_clause_ids: Sequence[str],
    ) -> Dict[str, Any]:
        detail_clues = self._detail_clues(clause_text)
        target_person_ids = self._resolve_known_ids(clause_text, known_person_ids)
        target_group_ids = self._resolve_known_ids(clause_text, known_group_ids)
        target_relationship_types = self._resolve_relationship_types(clause_text, known_relationship_types)
        target_fact_keys = self._resolve_fact_keys(clause_text, known_profile_fact_keys)
        intent = self._infer_intent(clause_text)
        route = self._infer_route(
            clause_text,
            intent=intent,
            target_fact_keys=target_fact_keys,
            target_relationship_types=target_relationship_types,
        )
        time_windows = self._extract_time_windows(clause_text)
        same_event_required = route == "event-first" and intent in {"lookup", "explain"}
        can_abstain = route in {"fact-first", "hybrid-judgement"} or intent in {"judgement", "existence"}
        return {
            "clause_id": clause_id,
            "clause_text": clause_text,
            "route": route,
            "intent": intent,
            "subject_scope": "task",
            "task_scope": task_scope,
            "target_fact_keys": target_fact_keys,
            "target_person_ids": target_person_ids,
            "target_group_ids": target_group_ids,
            "target_relationship_types": target_relationship_types,
            "time_windows": time_windows,
            "detail_clues": detail_clues,
            "same_event_required": same_event_required,
            "requires_photos": True,
            "can_abstain": can_abstain,
            "depends_on_clause_ids": list(depends_on_clause_ids),
        }

    def _infer_intent(self, clause_text: str) -> str:
        normalized = str(clause_text or "").strip()
        lowered = normalized.lower()
        if any(token in normalized for token in ("能不能判断", "可不可以判断", "是否足以判断", "如果只看可审计证据")):
            return "judgement"
        if any(token in normalized for token in ("关系最好", "最亲密", "最好", "top", "最")) and "关系" in normalized:
            return "rank"
        if any(token in normalized for token in ("多少次", "几次", "多少个", "几个", "count", "how many")):
            return "count"
        if any(token in normalized for token in ("有没有", "是否存在", "有无", "存在不止一条", "exist")):
            return "existence"
        if any(token in normalized for token in ("对比", "相比", "compare")):
            return "compare"
        if any(token in normalized for token in ("请说明", "为什么", "解释")):
            return "explain"
        if "分别" in normalized:
            return "explain"
        if "relationship" in lowered and "best" in lowered:
            return "rank"
        return "lookup"

    def _infer_route(
        self,
        clause_text: str,
        *,
        intent: str,
        target_fact_keys: Sequence[str],
        target_relationship_types: Sequence[str],
    ) -> str:
        normalized = str(clause_text or "").strip()
        lowered = normalized.lower()
        if intent == "judgement":
            return "hybrid-judgement"
        if target_fact_keys and not any(token in normalized for token in ("事件", "经历", "做了", "去过", "哪些")):
            return "fact-first"
        if target_relationship_types or any(
            token in normalized for token in ("伴侣", "对象", "关系", "romantic", "friend", "bestie", "亲密")
        ):
            return "relationship-first"
        if any(token in lowered for token in ("pet", "hobby", "interest", "student", "career")):
            return "fact-first"
        return "event-first"

    def _resolve_known_ids(self, text: str, candidates: Iterable[str]) -> List[str]:
        lowered = str(text or "").lower()
        resolved = [
            value
            for value in candidates
            if value and value.lower() in lowered
        ]
        explicit = re.findall(r"\bPerson[_-]?\d+\b", text, flags=re.IGNORECASE)
        for value in explicit:
            if value not in resolved:
                resolved.append(value)
        return resolved

    def _resolve_relationship_types(self, text: str, known_relationship_types: Iterable[str]) -> List[str]:
        lowered = str(text or "").lower()
        resolved: List[str] = []
        for rel_type in known_relationship_types:
            normalized = str(rel_type or "").strip().lower()
            if normalized and normalized in lowered and normalized not in resolved:
                resolved.append(normalized)
        for token, rel_type in KINSHIP_RELATION_MAP.items():
            if token in lowered or token in text:
                if rel_type not in resolved:
                    resolved.append(rel_type)
        return resolved

    def _resolve_fact_keys(self, text: str, known_profile_fact_keys: Iterable[str]) -> List[str]:
        lowered = str(text or "").lower()
        known_keys = [str(item or "").strip() for item in known_profile_fact_keys if str(item or "").strip()]
        resolved: List[str] = []
        for alias, candidates in FACT_KEY_ALIASES.items():
            if alias.lower() not in lowered:
                continue
            for candidate in candidates:
                for known_key in known_keys:
                    if candidate in known_key and known_key not in resolved:
                        resolved.append(known_key)
                if candidate not in resolved and "." in candidate:
                    resolved.append(candidate)
        if resolved:
            return resolved

        detail_clues = self._detail_clues(text)
        if not detail_clues:
            return []
        scored: List[tuple[int, str]] = []
        for known_key in known_keys:
            key_lower = known_key.lower()
            score = sum(1 for clue in detail_clues if clue.lower() in key_lower)
            if score > 0:
                scored.append((score, known_key))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [item[1] for item in scored[:4]]

    def _extract_time_windows(self, text: str) -> List[Dict[str, str]]:
        lowered = str(text or "").lower()
        windows: List[Dict[str, str]] = []
        explicit_date = re.search(r"(20\d{2})[年/-](\d{1,2})[月/-](\d{1,2})日?", text)
        if explicit_date:
            year = int(explicit_date.group(1))
            month = int(explicit_date.group(2))
            day = int(explicit_date.group(3))
            start_at = datetime(year, month, day)
            end_at = start_at + timedelta(days=1) - timedelta(seconds=1)
            windows.append(
                {
                    "label": f"date_{year:04d}-{month:02d}-{day:02d}",
                    "start_at": start_at.isoformat(),
                    "end_at": end_at.isoformat(),
                }
            )
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
        if year_match and not explicit_date:
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
        stopwords = {
            "我",
            "用户",
            "主角",
            "现在",
            "最近",
            "这个",
            "那个",
            "一下",
            "请",
            "帮我",
            "帮",
            "是否",
            "有没有",
            "如果",
            "只看",
            "可审计证据",
        }
        result: List[str] = []
        for part in parts:
            normalized = part.strip()
            if len(normalized) <= 1 or normalized in stopwords:
                continue
            if normalized not in result:
                result.append(normalized)
        return result[:12]

    def _router_confidence(self, clauses: Sequence[Dict[str, Any]]) -> float:
        if not clauses:
            return 0.25
        route_bonus = sum(0.1 for clause in clauses if clause.get("route") != "event-first")
        resolve_bonus = sum(0.08 for clause in clauses if clause.get("target_fact_keys") or clause.get("target_person_ids"))
        return round(min(0.45 + route_bonus + resolve_bonus, 0.92), 3)

    def _llm_route(
        self,
        question: str,
        *,
        user_id: str,
        task_scope: Optional[str],
        deterministic_plan: Dict[str, Any],
        known_person_ids: Iterable[str],
        known_group_ids: Iterable[str],
        known_relationship_types: Iterable[str],
        known_profile_fact_keys: Iterable[str],
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
                        "You are an always-on router for a photo-backed memory retrieval API. "
                        "Return JSON only. Do not answer the user's question. "
                        "You must output a RoutePlanV1 object with keys: router_version, normalized_question, "
                        "composition_mode, global_constraints, clauses, router_confidence, needs_abstain_judgement. "
                        "Each clause must contain: clause_id, clause_text, route, intent, subject_scope, task_scope, "
                        "target_fact_keys, target_person_ids, target_group_ids, target_relationship_types, time_windows, "
                        "detail_clues, same_event_required, requires_photos, can_abstain, depends_on_clause_ids. "
                        "Routes are one of fact-first, relationship-first, event-first, hybrid-judgement."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "user_id": user_id,
                            "task_scope": task_scope,
                            "question": question,
                            "deterministic_plan": deterministic_plan,
                            "known_person_ids": list(known_person_ids),
                            "known_group_ids": list(known_group_ids),
                            "known_relationship_types": list(known_relationship_types),
                            "known_profile_fact_keys": list(known_profile_fact_keys),
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
            plan = json.loads(content)
            return plan if isinstance(plan, dict) else None
        except Exception:
            return None

    def _normalize_route_plan(
        self,
        payload: Dict[str, Any],
        *,
        question: str,
        task_scope: Optional[str],
        known_person_ids: Iterable[str],
        known_group_ids: Iterable[str],
        known_relationship_types: Iterable[str],
        known_profile_fact_keys: Iterable[str],
    ) -> Optional[Dict[str, Any]]:
        clauses_payload = list(payload.get("clauses") or [])
        if not clauses_payload:
            return None
        clauses = []
        for index, clause_payload in enumerate(clauses_payload, start=1):
            clause_text = str((clause_payload or {}).get("clause_text") or "").strip()
            if not clause_text:
                clause_text = str(question or "").strip()
            base = self._build_clause(
                clause_id=str((clause_payload or {}).get("clause_id") or f"clause_{index}"),
                clause_text=clause_text,
                task_scope=task_scope,
                known_person_ids=known_person_ids,
                known_group_ids=known_group_ids,
                known_relationship_types=known_relationship_types,
                known_profile_fact_keys=known_profile_fact_keys,
                depends_on_clause_ids=list((clause_payload or {}).get("depends_on_clause_ids") or []),
            )
            clause = dict(base)
            clause["route"] = self._normalize_enum(
                (clause_payload or {}).get("route"),
                allowed={"fact-first", "relationship-first", "event-first", "hybrid-judgement"},
                fallback=base["route"],
            )
            clause["intent"] = self._normalize_enum(
                (clause_payload or {}).get("intent"),
                allowed={"lookup", "explain", "rank", "count", "existence", "compare", "judgement"},
                fallback=base["intent"],
            )
            clause["subject_scope"] = str((clause_payload or {}).get("subject_scope") or base.get("subject_scope") or "task")
            clause["task_scope"] = str((clause_payload or {}).get("task_scope") or task_scope or "")
            clause["target_person_ids"] = self._normalize_str_list((clause_payload or {}).get("target_person_ids"), fallback=base["target_person_ids"])
            clause["target_group_ids"] = self._normalize_str_list((clause_payload or {}).get("target_group_ids"), fallback=base["target_group_ids"])
            clause["target_relationship_types"] = self._normalize_str_list(
                (clause_payload or {}).get("target_relationship_types"),
                fallback=base["target_relationship_types"],
            )
            clause["target_fact_keys"] = self._normalize_str_list(
                (clause_payload or {}).get("target_fact_keys"),
                fallback=base["target_fact_keys"],
            )
            clause["detail_clues"] = self._normalize_str_list((clause_payload or {}).get("detail_clues"), fallback=base["detail_clues"])
            clause["time_windows"] = self._normalize_time_windows((clause_payload or {}).get("time_windows"), fallback=base["time_windows"])
            clause["same_event_required"] = bool((clause_payload or {}).get("same_event_required", base["same_event_required"]))
            clause["requires_photos"] = bool((clause_payload or {}).get("requires_photos", True))
            clause["can_abstain"] = bool((clause_payload or {}).get("can_abstain", base["can_abstain"]))
            clause["depends_on_clause_ids"] = self._normalize_str_list(
                (clause_payload or {}).get("depends_on_clause_ids"),
                fallback=base["depends_on_clause_ids"],
            )
            clauses.append(clause)
        composition_mode = self._normalize_enum(
            payload.get("composition_mode"),
            allowed={"single", "parallel", "and_then_enumerate", "compare", "judge_then_explain"},
            fallback="parallel" if len(clauses) > 1 else "single",
        )
        return {
            "router_version": str(payload.get("router_version") or ROUTER_VERSION),
            "normalized_question": str(payload.get("normalized_question") or question or "").strip(),
            "composition_mode": composition_mode,
            "global_constraints": dict(payload.get("global_constraints") or {"task_scope": task_scope, "photo_required": True}),
            "clauses": clauses,
            "router_confidence": float(payload.get("router_confidence") or 0.75),
            "needs_abstain_judgement": bool(payload.get("needs_abstain_judgement", any(item.get("can_abstain") for item in clauses))),
            "router_source": "llm",
            "router_fallback": None,
        }

    def _merge_plans(self, deterministic: Dict[str, Any], llm_plan: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(llm_plan)
        merged["router_version"] = str(llm_plan.get("router_version") or deterministic.get("router_version") or ROUTER_VERSION)
        merged["normalized_question"] = str(llm_plan.get("normalized_question") or deterministic.get("normalized_question") or "")
        merged["global_constraints"] = {
            **dict(deterministic.get("global_constraints") or {}),
            **dict(llm_plan.get("global_constraints") or {}),
        }
        merged["router_confidence"] = round(
            max(float(deterministic.get("router_confidence") or 0.0), float(llm_plan.get("router_confidence") or 0.0)),
            3,
        )
        merged["needs_abstain_judgement"] = bool(
            llm_plan.get("needs_abstain_judgement", deterministic.get("needs_abstain_judgement"))
        )
        merged["router_source"] = "llm"
        merged["router_fallback"] = None
        return merged

    def _normalize_str_list(self, value: Any, *, fallback: Sequence[str]) -> List[str]:
        if isinstance(value, str):
            parts = [part.strip() for part in re.split(r"[,，;；\s]+", value) if part.strip()]
            return parts or list(fallback)
        if isinstance(value, (list, tuple)):
            parts = [str(item).strip() for item in value if str(item).strip()]
            return parts or list(fallback)
        return list(fallback)

    def _normalize_time_windows(self, value: Any, *, fallback: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
        if isinstance(value, str):
            parsed = self._extract_time_windows(value)
            return parsed or list(fallback)
        if not isinstance(value, (list, tuple)):
            return list(fallback)
        normalized: List[Dict[str, str]] = []
        for item in value:
            if isinstance(item, str):
                normalized.extend(self._extract_time_windows(item))
                continue
            if not isinstance(item, dict):
                continue
            start_at = str(item.get("start_at") or "").strip()
            end_at = str(item.get("end_at") or "").strip()
            label = str(item.get("label") or "window").strip()
            if not start_at and not end_at and item.get("raw"):
                normalized.extend(self._extract_time_windows(str(item.get("raw") or "")))
                continue
            normalized.append({"label": label or "window", "start_at": start_at, "end_at": end_at})
        return normalized or list(fallback)

    def _normalize_enum(self, value: Any, *, allowed: set[str], fallback: str) -> str:
        normalized = str(value or "").strip()
        return normalized if normalized in allowed else fallback

