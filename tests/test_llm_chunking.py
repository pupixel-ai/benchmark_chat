from __future__ import annotations

import unittest
from unittest.mock import patch

from services.llm_processor import LLMProcessor


class LLMChunkingTests(unittest.TestCase):
    def _build_vlm_result(
        self,
        photo_id: str,
        timestamp: str,
        *,
        location_name: str,
        people: list[str],
        activity: str,
        ocr_hits: list[str] | None = None,
    ) -> dict:
        return {
            "photo_id": photo_id,
            "timestamp": timestamp,
            "location": {"name": location_name},
            "face_person_ids": people,
            "vlm_analysis": {
                "summary": f"summary {photo_id}",
                "scene": {
                    "location_detected": location_name,
                    "environment_description": "stub scene",
                },
                "event": {
                    "activity": activity,
                    "social_context": "group",
                },
                "details": ["detail"],
                "key_objects": ["object"],
                "ocr_hits": ocr_hits or [],
                "brands": [],
                "place_candidates": [{"name": location_name, "confidence": 0.8, "reason": "stub"}],
                "route_plan_clues": [],
                "transport_clues": [],
                "health_treatment_clues": [],
                "object_last_seen_clues": [],
                "raw_structured_observations": [],
                "uncertainty": [],
            },
        }

    def test_session_slice_generation_uses_bursts_and_overlap(self) -> None:
        processor = LLMProcessor.__new__(LLMProcessor)
        vlm_results = [
            self._build_vlm_result("photo_001", "2025-11-02T19:00:00", location_name="place_a", people=["Person_001"], activity="walk", ocr_hits=["alpha"]),
            self._build_vlm_result("photo_002", "2025-11-02T19:00:20", location_name="place_a", people=["Person_001"], activity="walk"),
            self._build_vlm_result("photo_003", "2025-11-02T19:02:10", location_name="place_a", people=["Person_001"], activity="walk", ocr_hits=["beta"]),
            self._build_vlm_result("photo_004", "2025-11-02T19:02:30", location_name="place_a", people=["Person_001"], activity="walk"),
            self._build_vlm_result("photo_005", "2025-11-02T19:04:20", location_name="place_a", people=["Person_001"], activity="view", ocr_hits=["gamma"]),
            self._build_vlm_result("photo_006", "2025-11-02T19:04:40", location_name="place_a", people=["Person_001"], activity="view"),
        ]

        facts = processor._build_photo_fact_buffer(vlm_results)
        bursts = processor._build_bursts(facts)
        raw_sessions = processor._build_raw_sessions(bursts)

        with patch.multiple(
            "services.llm_processor",
            LLM_SLICE_MAX_BURSTS=2,
            LLM_SLICE_HARD_MAX_BURSTS=3,
            LLM_SLICE_OVERLAP_BURSTS=1,
            LLM_SLICE_MAX_PHOTOS=4,
            LLM_SLICE_MAX_RARE_CLUES=8,
            LLM_SLICE_MIN_PHOTOS=2,
        ):
            session_slices = processor._build_session_slices(raw_sessions)

        self.assertEqual(len(bursts), 3)
        self.assertEqual(len(raw_sessions), 1)
        self.assertEqual(len(session_slices), 2)
        self.assertEqual(session_slices[0]["burst_ids"], ["burst_0001", "burst_0002"])
        self.assertEqual(session_slices[1]["overlap_burst_ids"], ["burst_0002"])
        self.assertEqual(session_slices[1]["burst_ids"], ["burst_0002", "burst_0003"])
        self.assertTrue(session_slices[0]["evidence_packet"]["fact_inventory"])
        self.assertTrue(session_slices[0]["evidence_packet"]["change_points"])
        self.assertIn("information_score", session_slices[0])
        self.assertIn("density_score", session_slices[0])
        self.assertIn("slice_budget_metrics", session_slices[0]["evidence_packet"])


if __name__ == "__main__":
    unittest.main()
