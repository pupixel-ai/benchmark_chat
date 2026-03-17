from __future__ import annotations

import unittest
from unittest.mock import patch

from services.llm_processor import LLMProcessor
from services.vlm_analyzer import VLMAnalyzer


class _FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self) -> dict:
        return self._payload


class _FakeRequests:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls: list[dict] = []

    def post(self, url: str, json=None, headers=None, timeout=None):
        self.calls.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return _FakeResponse(self.payload)


class OpenRouterProviderTests(unittest.TestCase):
    def test_json_payload_parser_handles_wrappers_and_trailing_text(self) -> None:
        analyzer = VLMAnalyzer.__new__(VLMAnalyzer)
        processor = LLMProcessor.__new__(LLMProcessor)

        vlm_payload = analyzer._extract_json_payload(
            '<|begin_of_box|>{"ok": true, "items": []}<|end_of_box|>'
        )
        llm_payload = processor._extract_json_payload(
            '{"events": [], "relationships": []}.'
        )

        self.assertEqual(vlm_payload["ok"], True)
        self.assertEqual(vlm_payload["items"], [])
        self.assertEqual(llm_payload["events"], [])
        self.assertEqual(llm_payload["relationships"], [])

    def test_vlm_openrouter_request_stays_single_image(self) -> None:
        with patch.multiple(
            "services.vlm_analyzer",
            MODEL_PROVIDER="openrouter",
            OPENROUTER_API_KEY="test-openrouter-key",
            OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
            OPENROUTER_SITE_URL="http://localhost:8000",
            OPENROUTER_APP_NAME="Memory Engineering Test",
            OPENROUTER_VLM_MODEL="google/gemini-2.0-flash-001",
            GEMINI_API_KEY="",
        ):
            analyzer = VLMAnalyzer(cache_path="cache/test_openrouter_vlm.json")
            fake_requests = _FakeRequests(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "{\"summary\": \"ok\", \"people\": [], \"scene\": {}, \"event\": {}, \"details\": [], \"key_objects\": []}"
                            }
                        }
                    ]
                }
            )
            analyzer.requests = fake_requests

            result = analyzer._analyze_via_openrouter(
                "只返回 JSON",
                b"image-bytes",
                "image/jpeg",
            )

        self.assertEqual(result["summary"], "ok")
        self.assertEqual(len(fake_requests.calls), 1)
        payload = fake_requests.calls[0]["json"]
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_llm_openrouter_json_and_markdown_calls(self) -> None:
        with patch.multiple(
            "services.llm_processor",
            MODEL_PROVIDER="openrouter",
            OPENROUTER_API_KEY="test-openrouter-key",
            OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
            OPENROUTER_SITE_URL="http://localhost:8000",
            OPENROUTER_APP_NAME="Memory Engineering Test",
            OPENROUTER_LLM_MODEL="google/gemini-2.5-flash",
            GEMINI_API_KEY="",
        ):
            processor = LLMProcessor()
            json_requests = _FakeRequests(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "{\"events\": [], \"relationships\": []}"
                            }
                        }
                    ]
                }
            )
            processor.requests = json_requests
            llm_result = processor._call_llm_via_openrouter("只返回 JSON")

            markdown_requests = _FakeRequests(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "# Markdown Profile\n\nOK"
                            }
                        }
                    ]
                }
            )
            processor.requests = markdown_requests
            profile_result = processor._call_profile_via_openrouter("输出 Markdown")

        self.assertEqual(llm_result["events"], [])
        self.assertEqual(profile_result, "# Markdown Profile\n\nOK")
        self.assertEqual(json_requests.calls[0]["json"]["messages"][0]["content"], "只返回 JSON")
        self.assertEqual(markdown_requests.calls[0]["json"]["messages"][0]["content"], "输出 Markdown")


if __name__ == "__main__":
    unittest.main()
