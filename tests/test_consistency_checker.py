from __future__ import annotations

import unittest

from models import Event, Relationship
from services.consistency_checker import build_consistency_report
from services.llm_processor import LLMProcessor


class ConsistencyCheckerTests(unittest.TestCase):
    def test_build_consistency_report_flags_intimate_partner_mismatch(self) -> None:
        relationships = [
            Relationship(
                person_id="Person_002",
                relationship_type="romantic",
                intimacy_score=0.8,
                status="stable",
                confidence=0.9,
                reasoning="",
            )
        ]
        structured_profile = {
            "long_term_facts": {
                "relationships": {
                    "intimate_partner": {"value": "Person_999", "confidence": 0.9},
                    "close_circle_size": {"value": 1, "confidence": 0.8},
                }
            }
        }

        report = build_consistency_report([], relationships, structured_profile)

        self.assertEqual(report["summary"]["issue_count"], 1)
        self.assertEqual(report["issues"][0]["code"], "INTIMATE_PARTNER_MISMATCH")

    def test_build_consistency_report_flags_close_circle_size_mismatch(self) -> None:
        relationships = [
            Relationship(
                person_id="Person_002",
                relationship_type="bestie",
                intimacy_score=0.8,
                status="stable",
                confidence=0.9,
                reasoning="",
            )
        ]
        structured_profile = {
            "long_term_facts": {
                "relationships": {
                    "intimate_partner": {"value": None, "confidence": 0.0},
                    "close_circle_size": {"value": 4, "confidence": 0.8},
                }
            }
        }

        report = build_consistency_report([], relationships, structured_profile)

        self.assertEqual(report["summary"]["issue_count"], 1)
        self.assertEqual(report["issues"][0]["code"], "CLOSE_CIRCLE_SIZE_MISMATCH")

    def test_generate_profile_returns_debug_and_consistency_payload(self) -> None:
        processor = LLMProcessor.__new__(LLMProcessor)
        def fake_call(prompt, response_mime_type=None, model_override=None):
            if response_mime_type == "application/json" and "输出 JSON Schema" in prompt:
                return {
                    "long_term_facts": {
                        "relationships": {
                            "intimate_partner": {"value": "Person_999", "confidence": 0.9},
                            "close_circle_size": {"value": 4, "confidence": 0.8},
                        }
                    }
                }
            if response_mime_type == "application/json" and "\"public_report\"" in prompt:
                return {
                    "public_report": "# 用户全维画像分析报告\n\n结论。",
                    "reasoning_trace": {"reasoning_steps": ["step"]},
                }
            raise AssertionError("unexpected prompt")

        processor._call_llm_via_official_api = fake_call
        processor._create_structured_profile_prompt = lambda events_str, relationships_str, face_context="", raw_vlm_context="": "prompt\n输出 JSON Schema"
        processor._create_profile_report_prompt = lambda events_str, relationships_str, structured_profile: 'prompt\n"public_report"'

        events = [
            Event(
                event_id="EVT_001",
                date="2026-03-01",
                time_range="23:00 - 23:30",
                duration="",
                title="深夜散步",
                type="休闲",
                participants=["Person_001", "Person_003"],
                location="街区",
                description="一起散步",
                photo_count=2,
                confidence=0.8,
                reason="",
            )
        ]
        relationships = [
            Relationship(
                person_id="Person_003",
                relationship_type="friend",
                intimacy_score=0.32,
                status="stable",
                confidence=0.7,
                reasoning="",
            )
        ]

        result = processor.generate_profile(events, relationships, vlm_results=[], face_db={})

        self.assertIn("debug", result)
        self.assertIn("consistency", result)
        self.assertIn("report_reasoning", result["debug"])
        self.assertGreaterEqual(result["consistency"]["summary"]["issue_count"], 0)


if __name__ == "__main__":
    unittest.main()
