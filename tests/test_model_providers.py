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


class _FakeBedrockClient:
    def __init__(self, payload_text: str) -> None:
        self.payload_text = payload_text
        self.calls: list[dict] = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "output": {
                "message": {
                    "content": [
                        {"text": self.payload_text},
                    ]
                }
            }
        }


class OpenRouterProviderTests(unittest.TestCase):
    def test_json_payload_parser_handles_wrappers_and_trailing_text(self) -> None:
        analyzer = VLMAnalyzer.__new__(VLMAnalyzer)
        processor = LLMProcessor.__new__(LLMProcessor)

        vlm_payload = analyzer._extract_json_payload(
            '<|begin_of_box|>{"ok": true, "items": []}<|end_of_box|>'
        )
        llm_payload = processor._extract_json_payload(
            '{"facts": [], "relationship_hypotheses": []}.'
        )

        self.assertEqual(vlm_payload["ok"], True)
        self.assertEqual(vlm_payload["items"], [])
        self.assertEqual(llm_payload["facts"], [])
        self.assertEqual(llm_payload["relationship_hypotheses"], [])

    def test_bedrock_vlm_request_uses_converse_image_blocks(self) -> None:
        with patch.multiple(
            "services.vlm_analyzer",
            VLM_PROVIDER="bedrock",
            VLM_MODEL="amazon.nova-2-pro-preview-20251202-v1:0",
            BEDROCK_REGION="ap-southeast-1",
        ), patch("services.vlm_analyzer.build_bedrock_client", return_value=_FakeBedrockClient('{"summary":"ok","people":[],"scene":{},"event":{},"details":[],"key_objects":[]}')), patch(
            "services.vlm_analyzer.resolve_bedrock_model_candidates",
            return_value=["amazon.nova-2-pro-preview-20251202-v1:0"],
        ):
            analyzer = VLMAnalyzer(cache_path="cache/test_bedrock_vlm.json")
            result = analyzer._analyze_via_bedrock("只返回 JSON", b"image-bytes", "image/jpeg")

        self.assertEqual(result["summary"], "ok")
        call = analyzer.bedrock_client.calls[0]
        self.assertEqual(call["modelId"], "amazon.nova-2-pro-preview-20251202-v1:0")
        content = call["messages"][0]["content"]
        self.assertEqual(content[0]["text"], "只返回 JSON")
        self.assertEqual(content[1]["image"]["format"], "jpeg")

    def test_bedrock_llm_request_returns_structured_contract(self) -> None:
        payload_text = '{"facts":[],"observations":[],"claims":[],"relationship_hypotheses":[],"profile_deltas":[],"uncertainty":[]}'
        with patch.multiple(
            "services.llm_processor",
            LLM_PROVIDER="bedrock",
            LLM_MODEL="anthropic.claude-sonnet-4-6",
            BEDROCK_REGION="ap-southeast-1",
        ), patch("services.llm_processor.build_bedrock_client", return_value=_FakeBedrockClient(payload_text)), patch(
            "services.llm_processor.resolve_bedrock_model_candidates",
            return_value=["anthropic.claude-sonnet-4-6"],
        ):
            processor = LLMProcessor()
            result = processor._call_llm_via_bedrock("只返回 JSON")

        self.assertEqual(result["facts"], [])
        self.assertEqual(result["observations"], [])
        call = processor.bedrock_client.calls[0]
        self.assertEqual(call["modelId"], "anthropic.claude-sonnet-4-6")
        self.assertEqual(call["messages"][0]["content"][0]["text"], "只返回 JSON")

    def test_vlm_openrouter_request_stays_single_image(self) -> None:
        with patch.multiple(
            "services.vlm_analyzer",
            VLM_PROVIDER="openrouter",
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
            LLM_PROVIDER="openrouter",
            OPENROUTER_API_KEY="test-openrouter-key",
            OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
            OPENROUTER_SITE_URL="http://localhost:8000",
            OPENROUTER_APP_NAME="Memory Engineering Test",
            OPENROUTER_LLM_MODEL="minimax/minimax-m2.7",
            GEMINI_API_KEY="",
        ):
            processor = LLMProcessor()
            json_requests = _FakeRequests(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "{\"facts\": [], \"relationship_hypotheses\": []}"
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

        self.assertEqual(llm_result["facts"], [])
        self.assertEqual(profile_result, "# Markdown Profile\n\nOK")
        self.assertEqual(json_requests.calls[0]["json"]["messages"][0]["content"], "只返回 JSON")
        self.assertEqual(markdown_requests.calls[0]["json"]["messages"][0]["content"], "输出 Markdown")


if __name__ == "__main__":
    unittest.main()
