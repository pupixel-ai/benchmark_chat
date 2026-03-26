from __future__ import annotations

from pathlib import Path
import os
import re
from typing import Any, Dict, List

from dotenv import load_dotenv

from services.llm_processor import LLMProcessor

from .profile_llm import OpenRouterProfileLLMProcessor


DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "google/gemini-3.1-flash-lite-preview"


class ReusableSmokeOpenRouterLLMProcessor(OpenRouterProfileLLMProcessor):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = DEFAULT_OPENROUTER_BASE_URL,
        model: str = DEFAULT_OPENROUTER_MODEL,
        primary_person_id: str | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            primary_person_id=primary_person_id,
        )
        self._relationship_helper = object.__new__(LLMProcessor)
        self._relationship_helper.primary_person_id = primary_person_id

    def _recall_shared_events(self, person_id: str, events: List[Any]) -> List[Dict[str, Any]]:
        return LLMProcessor._recall_shared_events(self._relationship_helper, person_id, events)

    def _collect_relationship_evidence(self, person_id: str, vlm_results: List[Dict], events=None) -> Dict:
        return LLMProcessor._collect_relationship_evidence(
            self._relationship_helper,
            person_id,
            vlm_results,
            events,
        )


def resolve_reusable_smoke_llm_processor(
    *,
    primary_person_id: str | None,
    repo_root: str | Path | None = None,
) -> ReusableSmokeOpenRouterLLMProcessor | None:
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)

    api_key = (
        (os.getenv("OPENROUTER_API_KEY") or "").strip()
        or _read_first_openrouter_key(root / "open router key.md")
    )
    if not api_key:
        return None

    base_url = (os.getenv("OPENROUTER_BASE_URL") or DEFAULT_OPENROUTER_BASE_URL).strip() or DEFAULT_OPENROUTER_BASE_URL
    model = (os.getenv("PROFILE_LLM_MODEL") or DEFAULT_OPENROUTER_MODEL).strip() or DEFAULT_OPENROUTER_MODEL

    return ReusableSmokeOpenRouterLLMProcessor(
        api_key=api_key,
        base_url=base_url,
        model=model,
        primary_person_id=primary_person_id,
    )


def _read_first_openrouter_key(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8")
    matches = re.findall(r"sk-or-v1-[A-Za-z0-9]+", text)
    return matches[0] if matches else ""
