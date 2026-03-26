from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL, PROFILE_LLM_MODEL


class OpenRouterProfileLLMProcessor:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        primary_person_id: str | None = None,
    ) -> None:
        self.api_key = (api_key or OPENROUTER_API_KEY or "").strip()
        self.base_url = (base_url or OPENROUTER_BASE_URL or "").strip().rstrip("/")
        self.model = (model or PROFILE_LLM_MODEL or "google/gemini-3.1-flash-lite-preview").strip()
        self.primary_person_id = primary_person_id

        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY 未配置")
        if not self.base_url:
            raise ValueError("OPENROUTER_BASE_URL 未配置")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self._last_call_debug: dict[str, Any] | None = None

    def _consume_last_call_debug(self) -> dict[str, Any]:
        payload = dict(self._last_call_debug or {})
        self._last_call_debug = None
        return payload

    def _call_llm_via_official_api(
        self,
        prompt: str,
        response_mime_type: str = None,
        model_override: str = None,
    ) -> Any:
        model = (model_override or self.model).strip() or self.model
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        }
        if response_mime_type == "application/json":
            payload["response_format"] = {"type": "json_object"}

        url = f"{self.base_url}/chat/completions"
        max_retries = 3
        timeout = 300
        self._last_call_debug = {
            "api_call_attempted": False,
            "http_status_code": None,
            "raw_response_preview": "",
            "raw_response_truncated": False,
            "model": model,
            "url": url,
            "attempt_count": 0,
            "exception_type": None,
            "exception_message": None,
        }

        for attempt in range(max_retries):
            try:
                self._last_call_debug["api_call_attempted"] = True
                self._last_call_debug["attempt_count"] = attempt + 1
                response = requests.post(
                    url,
                    json=payload,
                    headers=self.headers,
                    timeout=timeout,
                )
                self._last_call_debug["http_status_code"] = response.status_code
                preview, truncated = _truncate_debug_text(response.text)
                self._last_call_debug["raw_response_preview"] = preview
                self._last_call_debug["raw_response_truncated"] = truncated
                if response.status_code != 200:
                    error_msg = f"OpenRouter API 返回状态码 {response.status_code}"
                    if response.text:
                        try:
                            error_data = response.json()
                            if isinstance(error_data, dict) and error_data.get("error"):
                                error_msg += f": {error_data['error']}"
                        except Exception:
                            error_msg += f": {response.text[:200]}"
                    self._last_call_debug["exception_type"] = "HTTPError"
                    self._last_call_debug["exception_message"] = error_msg
                    raise Exception(error_msg)

                response_data = response.json()
                content = self._extract_message_content(response_data)
                if not content:
                    return None
                return self._parse_content(content)
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                self._last_call_debug["exception_type"] = exc.__class__.__name__
                self._last_call_debug["exception_message"] = str(exc)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise Exception(f"OpenRouter 调用重试{max_retries}次仍失败: {exc}") from exc

    def _extract_message_content(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            return ""
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "\n".join(part for part in parts if part)
        return ""

    def _parse_content(self, text: str) -> Any:
        normalized = str(text or "").strip()
        if not normalized:
            return None

        candidates = [normalized]
        fenced_blocks = re.findall(r"```(?:json)?\s*(.*?)\s*```", normalized, re.DOTALL)
        candidates.extend(block.strip() for block in fenced_blocks if block.strip())

        first_object = normalized.find("{")
        last_object = normalized.rfind("}")
        if 0 <= first_object < last_object:
            candidates.append(normalized[first_object:last_object + 1])

        first_array = normalized.find("[")
        last_array = normalized.rfind("]")
        if 0 <= first_array < last_array:
            candidates.append(normalized[first_array:last_array + 1])

        seen: set[str] = set()
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

        return {"text": normalized}


def _truncate_debug_text(text: Any, limit: int = 2000) -> tuple[str, bool]:
    normalized = str(text or "")
    if len(normalized) <= limit:
        return normalized, False
    return normalized[:limit], True
