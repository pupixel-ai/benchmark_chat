"""Canonical concept registry for the memory graph."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional


CANONICAL_CONCEPTS: Dict[str, Dict[str, object]] = {
    "concert": {
        "aliases": ["演唱会", "concert", "live", "show", "live show", "gig", "巡演", "音乐会", "live music"],
        "concept_type": "event",
        "description": "Live music performances attended or observed by the user.",
        "parents": ["music_live_event"],
    },
    "music_festival_performance": {
        "aliases": ["音乐节", "music festival", "festival set", "festival performance", "音乐节演出", "音乐节活动"],
        "concept_type": "event",
        "description": "Festival-style live music performance or related on-site activity.",
        "parents": ["concert", "music_live_event"],
    },
    "music_live_event": {
        "aliases": ["演出", "现场演出", "演出现场", "音乐节", "festival", "live performance", "showcase"],
        "concept_type": "event",
        "description": "Broad live music event umbrella concept.",
        "parents": [],
    },
    "conflict": {
        "aliases": ["冲突", "争吵", "吵架", "矛盾", "disagreement", "argument"],
        "concept_type": "relationship_signal",
        "description": "Conflict or disagreement signals.",
        "parents": [],
    },
    "friend": {
        "aliases": ["朋友", "friend", "好友"],
        "concept_type": "relationship",
        "description": "General friendship relationship.",
        "parents": [],
    },
    "close_friend": {
        "aliases": ["密友", "闺蜜", "close friend", "best friend", "最好的好友"],
        "concept_type": "relationship",
        "description": "High-confidence close friendship relationship.",
        "parents": ["friend"],
    },
    "colleague": {
        "aliases": ["同事", "colleague", "coworker", "工作伙伴"],
        "concept_type": "relationship",
        "description": "Work relationship.",
        "parents": [],
    },
    "partner": {
        "aliases": ["伴侣", "女朋友", "男朋友", "partner", "girlfriend", "boyfriend"],
        "concept_type": "relationship",
        "description": "Romantic relationship.",
        "parents": [],
    },
    "family_generic": {
        "aliases": ["家人", "family", "亲人"],
        "concept_type": "relationship",
        "description": "Generic family relationship.",
        "parents": [],
    },
    "father": {
        "aliases": ["父亲", "爸爸", "father", "dad"],
        "concept_type": "relationship",
        "description": "Father relationship.",
        "parents": ["family_generic"],
    },
    "mother": {
        "aliases": ["母亲", "妈妈", "mother", "mom"],
        "concept_type": "relationship",
        "description": "Mother relationship.",
        "parents": ["family_generic"],
    },
    "happy_mood": {
        "aliases": ["开心", "高兴", "愉快", "快乐", "happy", "joyful"],
        "concept_type": "mood",
        "description": "Positive affect.",
        "parents": [],
    },
    "sad_mood": {
        "aliases": ["难过", "沮丧", "伤心", "sad", "down"],
        "concept_type": "mood",
        "description": "Negative affect.",
        "parents": [],
    },
    "neutral_mood": {
        "aliases": ["平静", "一般", "neutral", "calm"],
        "concept_type": "mood",
        "description": "Neutral affect.",
        "parents": [],
    },
    "recent_period": {
        "aliases": ["最近", "recent"],
        "concept_type": "period",
        "description": "Recent period inferred from latest sessions.",
        "parents": [],
    },
    "college_period": {
        "aliases": ["大学", "college", "campus"],
        "concept_type": "period",
        "description": "College or campus-related period.",
        "parents": [],
    },
    "job_period": {
        "aliases": ["工作", "job", "office", "career"],
        "concept_type": "period",
        "description": "Job or office-related period.",
        "parents": [],
    },
    "campus": {
        "aliases": ["校园", "campus", "大学", "college"],
        "concept_type": "context",
        "description": "Campus context.",
        "parents": ["college_period"],
    },
    "home": {
        "aliases": ["家", "home"],
        "concept_type": "place_context",
        "description": "Home context.",
        "parents": [],
    },
    "work": {
        "aliases": ["工作", "办公", "office", "meeting"],
        "concept_type": "context",
        "description": "Work context.",
        "parents": ["job_period"],
    },
    "leisure": {
        "aliases": ["休闲", "聚会", "leisure", "outing", "dinner"],
        "concept_type": "context",
        "description": "Leisure context.",
        "parents": [],
    },
}


def canonical_concept_names() -> List[str]:
    return sorted(CANONICAL_CONCEPTS.keys())


def concept_metadata(canonical_name: str) -> Dict[str, object]:
    return dict(CANONICAL_CONCEPTS.get(canonical_name, {}))


def match_concepts(raw_text: str, *, preferred_type: Optional[str] = None) -> List[str]:
    normalized = str(raw_text or "").strip().lower()
    if not normalized:
        return []

    matches: List[str] = []
    for canonical_name, payload in CANONICAL_CONCEPTS.items():
        if preferred_type and payload.get("concept_type") != preferred_type:
            continue
        aliases = [canonical_name, *[str(item).lower() for item in payload.get("aliases", [])]]
        if normalized in aliases:
            matches.append(canonical_name)
            continue
        if any(alias in normalized or normalized in alias for alias in aliases):
            matches.append(canonical_name)
    return matches


def normalize_concept(raw_text: str, *, preferred_type: Optional[str] = None) -> Optional[str]:
    matches = match_concepts(raw_text, preferred_type=preferred_type)
    return matches[0] if matches else None


def suggest_candidate_concept(raw_text: str, *, concept_type: str, user_id: Optional[str] = None) -> Dict[str, object]:
    raw = str(raw_text or "").strip()
    canonical_name = raw.lower().replace(" ", "_")
    return {
        "canonical_name": canonical_name,
        "aliases": [raw] if raw else [],
        "concept_type": concept_type,
        "scope": "candidate",
        "status": "proposed",
        "version": "v1",
        "user_id": user_id,
        "description": "",
        "parents": [],
    }


def collect_concepts(values: Iterable[str], *, preferred_type: Optional[str] = None) -> List[str]:
    concepts: List[str] = []
    for value in values:
        for concept in match_concepts(value, preferred_type=preferred_type):
            if concept not in concepts:
                concepts.append(concept)
    return concepts


def expand_concepts(concepts: Iterable[str]) -> List[str]:
    expanded: List[str] = []
    pending = [str(item) for item in concepts if item]
    while pending:
        current = pending.pop(0)
        if current in expanded:
            continue
        expanded.append(current)
        meta = CANONICAL_CONCEPTS.get(current, {})
        for parent in meta.get("parents", []):
            if parent and parent not in expanded:
                pending.append(str(parent))
        for candidate, payload in CANONICAL_CONCEPTS.items():
            parents = [str(item) for item in payload.get("parents", [])]
            if current in parents and candidate not in expanded:
                pending.append(candidate)
    return expanded
