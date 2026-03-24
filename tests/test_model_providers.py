from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

from models import Photo
from services.llm_processor import LLMProcessor
from services.vlm_analyzer import VLMAnalyzer, VLMCallError
import config


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


class _FlakyRequests:
    def __init__(self, sequence) -> None:
        self.sequence = list(sequence)
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
        if not self.sequence:
            raise AssertionError("No more fake responses configured")
        item = self.sequence.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


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
    def test_fallback_load_dotenv_sets_missing_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dotenv_path = os.path.join(tmpdir, ".env")
            with open(dotenv_path, "w", encoding="utf-8") as handle:
                handle.write("OPENROUTER_LLM_MODEL=nvidia/nemotron-3-super-120b-a12b\n")
                handle.write("LLM_PROVIDER=openrouter\n")
            with patch.dict(os.environ, {}, clear=True):
                config._fallback_load_dotenv(dotenv_path)
                self.assertEqual(os.environ["OPENROUTER_LLM_MODEL"], "nvidia/nemotron-3-super-120b-a12b")
                self.assertEqual(os.environ["LLM_PROVIDER"], "openrouter")

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

    def test_vlm_normalize_result_supports_new_visual_archive_schema(self) -> None:
        analyzer = VLMAnalyzer.__new__(VLMAnalyzer)
        payload = {
            "summary": "2026-03-23 傍晚，【主角】在校园咖啡店靠窗座位与 Person_002 自拍合影，桌面摆有两杯饮品。",
            "people": [
                {
                    "person_id": "Person_002",
                    "appearance": "青年女性，长发",
                    "clothing": "灰色连帽卫衣",
                    "activity": "坐在窗边合影",
                    "interaction": "与【主角】并肩自拍",
                    "contact_type": "selfie_together",
                    "expression": "微笑",
                }
            ],
            "relations": [
                {"subject": "Person_002", "relation": "holding", "object": "奶茶杯"},
                {"subject": "奶茶杯", "relation": "placed_on", "object": "木质桌子"},
            ],
            "scene": {
                "environment_details": ["木质桌子", "落地窗", "暖色灯光"],
                "location_detected": "校园咖啡店",
                "location_type": "室内",
            },
            "event": {
                "activity": "喝咖啡",
                "social_context": "和朋友",
                "mood": "轻松",
                "story_hints": ["可能是课间短暂休息"],
            },
            "details": ["学生证 学号 20261234", "WPS AI"],
        }

        normalized = analyzer._normalize_result(payload)

        self.assertEqual(normalized["people"][0]["contact_type"], "selfie_together")
        self.assertEqual(normalized["scene"]["location_type"], "室内")
        self.assertEqual(normalized["event"]["story_hints"], ["可能是课间短暂休息"])
        self.assertIn("校园咖啡店", normalized["place_candidates"])
        self.assertIn("木质桌子", normalized["key_objects"])
        self.assertIn("学生证 学号 20261234", normalized["ocr_hits"])

    def test_vlm_build_result_entry_derives_source_type(self) -> None:
        analyzer = VLMAnalyzer.__new__(VLMAnalyzer)
        screenshot_photo = Photo(
            photo_id="photo_001",
            filename="Screenshot 2026-03-25 at 18.12.11.png",
            path="/tmp/photo_001.png",
            timestamp=datetime(2026, 3, 25, 18, 12, 11),
            location={},
        )
        ai_photo = Photo(
            photo_id="photo_002",
            filename="midjourney_cake_reference.jpg",
            path="/tmp/photo_002.jpg",
            timestamp=datetime(2026, 3, 25, 18, 12, 11),
            location={},
        )

        screenshot_entry = analyzer.build_result_entry(
            screenshot_photo,
            {
                "summary": "聊天截图，包含蛋糕图片和聊天记录",
                "scene": {},
                "event": {},
                "details": ["screen capture"],
            },
        )
        ai_entry = analyzer.build_result_entry(
            ai_photo,
            {
                "summary": "AI generated image of a decorated cake",
                "scene": {},
                "event": {},
                "details": ["midjourney concept art"],
            },
        )

        self.assertEqual(screenshot_entry["source_type"], "screenshot")
        self.assertEqual(screenshot_entry["vlm_analysis"]["source_type"], "screenshot")
        self.assertEqual(ai_entry["source_type"], "ai_generated_image")
        self.assertEqual(ai_entry["vlm_analysis"]["source_type"], "ai_generated_image")

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
        ):
            with patch("services.llm_processor.build_bedrock_client", return_value=_FakeBedrockClient(payload_text)):
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
            OPENROUTER_VLM_MODEL="google/gemini-3.1-flash-lite-preview",
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
            analyzer.http_session = fake_requests

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

    def test_vlm_openrouter_parse_failure_preserves_raw_preview(self) -> None:
        with patch.multiple(
            "services.vlm_analyzer",
            VLM_PROVIDER="openrouter",
            OPENROUTER_API_KEY="test-openrouter-key",
            OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
            OPENROUTER_SITE_URL="http://localhost:8000",
            OPENROUTER_APP_NAME="Memory Engineering Test",
            OPENROUTER_VLM_MODEL="google/gemini-3.1-flash-lite-preview",
            GEMINI_API_KEY="",
        ):
            analyzer = VLMAnalyzer(cache_path="cache/test_openrouter_vlm.json")
            fake_requests = _FakeRequests(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "{\"summary\": \"broken json"
                            }
                        }
                    ]
                }
            )
            analyzer.requests = fake_requests
            analyzer.http_session = fake_requests

            with self.assertRaises(VLMCallError) as ctx:
                analyzer._analyze_via_openrouter(
                    "只返回 JSON",
                    b"image-bytes",
                    "image/jpeg",
                )

        self.assertEqual(ctx.exception.details["response_provider"], "openrouter")
        self.assertEqual(ctx.exception.details["response_status_code"], 200)
        self.assertEqual(ctx.exception.details["parse_error_type"], "JSONDecodeError")
        self.assertIn("\"summary\": \"broken json", ctx.exception.details["raw_response_preview"])

    def test_llm_openrouter_json_and_markdown_calls(self) -> None:
        with patch.multiple(
            "services.llm_processor",
            LLM_PROVIDER="openrouter",
            OPENROUTER_API_KEY="test-openrouter-key",
            OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
            OPENROUTER_SITE_URL="http://localhost:8000",
            OPENROUTER_APP_NAME="Memory Engineering Test",
            OPENROUTER_LLM_MODEL="minimax/minimax-m2.5",
            OPENROUTER_REASONING_EFFORT="minimal",
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
        self.assertEqual(json_requests.calls[0]["json"]["reasoning"]["effort"], "minimal")
        self.assertEqual(markdown_requests.calls[0]["json"]["reasoning"]["effort"], "minimal")

    def test_llm_openrouter_json_call_accepts_response_format_and_max_tokens(self) -> None:
        with patch.multiple(
            "services.llm_processor",
            LLM_PROVIDER="openrouter",
            OPENROUTER_API_KEY="test-openrouter-key",
            OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
            OPENROUTER_SITE_URL="http://localhost:8000",
            OPENROUTER_APP_NAME="Memory Engineering Test",
            OPENROUTER_LLM_MODEL="minimax/minimax-m2.5",
            GEMINI_API_KEY="",
        ):
            processor = LLMProcessor()
            requests = _FakeRequests(
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
            processor.requests = requests

            result = processor._call_json_prompt(
                "只返回 JSON",
                max_tokens=24576,
                response_format={"type": "json_object"},
            )

        self.assertEqual(result["facts"], [])
        payload = requests.calls[0]["json"]
        self.assertEqual(payload["max_tokens"], 24576)
        self.assertEqual(payload["response_format"], {"type": "json_object"})

    def test_llm_openrouter_retries_premature_response(self) -> None:
        with patch.multiple(
            "services.llm_processor",
            LLM_PROVIDER="openrouter",
            OPENROUTER_API_KEY="test-openrouter-key",
            OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
            OPENROUTER_SITE_URL="http://localhost:8000",
            OPENROUTER_APP_NAME="Memory Engineering Test",
            OPENROUTER_LLM_MODEL="minimax/minimax-m2.5",
            GEMINI_API_KEY="",
        ):
            processor = LLMProcessor()
            processor.requests = _FlakyRequests(
                [
                    RuntimeError("Response ended prematurely"),
                    _FakeResponse(
                        {
                            "choices": [
                                {
                                    "message": {
                                        "content": "{\"facts\": [], \"observations\": [], \"claims\": [], \"relationship_hypotheses\": [], \"profile_deltas\": [], \"uncertainty\": []}"
                                    }
                                }
                            ]
                        }
                    ),
                ]
            )

            result = processor._call_with_retries(lambda: processor._call_llm_via_openrouter("只返回 JSON"))

        self.assertEqual(result["facts"], [])
        self.assertEqual(len(processor.requests.calls), 2)

    def test_relationship_provider_follows_main_openrouter(self) -> None:
        with patch.multiple(
            "services.llm_processor",
            LLM_PROVIDER="openrouter",
            RELATIONSHIP_FOLLOWS_MAIN_LLM=True,
            RELATIONSHIP_PROVIDER="openrouter",
            OPENROUTER_API_KEY="test-openrouter-key",
            OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
            OPENROUTER_SITE_URL="http://localhost:8000",
            OPENROUTER_APP_NAME="Memory Engineering Test",
            OPENROUTER_LLM_MODEL="minimax/minimax-m2.5",
            GEMINI_API_KEY="",
        ):
            processor = LLMProcessor()
            processor.requests = _FakeRequests(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "{\"relationship_type\": \"friend\", \"status\": \"stable\", \"confidence\": 0.8, \"reason\": \"ok\"}"
                            }
                        }
                    ]
                }
            )

            result = processor._call_relationship_prompt("relationship prompt")

        self.assertEqual(processor.relationship_provider, "openrouter")
        self.assertEqual(result["relationship_type"], "friend")

    def test_retryable_error_detects_premature_openrouter_response(self) -> None:
        processor = LLMProcessor.__new__(LLMProcessor)
        self.assertTrue(processor._is_retryable_error(RuntimeError("Response ended prematurely")))
        self.assertTrue(processor._is_retryable_error(RuntimeError("OpenRouter 返回状态码 502: bad gateway")))


if __name__ == "__main__":
    unittest.main()
