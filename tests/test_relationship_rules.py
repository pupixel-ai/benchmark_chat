from __future__ import annotations

import unittest
from unittest.mock import patch

from services.llm_processor import LLMProcessor
from services.relationship_rules import (
    determine_relationship_status,
    infer_relationship_candidates,
    score_relationship_confidence,
)


class RelationshipRulesTests(unittest.TestCase):
    def test_infer_relationship_candidates_prefers_romantic_when_private_one_on_one(self) -> None:
        evidence = {
            "photo_count": 8,
            "time_span_days": 120,
            "scenes": ["公寓", "咖啡馆", "海边"],
            "contact_types": ["hug", "holding_hands"],
            "with_user_only": True,
            "weekend_frequency": "高",
            "private_scene_ratio": 0.5,
            "dominant_scene_ratio": 0.34,
            "interaction_behavior": ["合影", "共同进餐"],
        }

        candidates = infer_relationship_candidates(evidence, intimacy_score=0.78)

        self.assertEqual(candidates[0], "romantic")
        self.assertNotIn("family", candidates)

    def test_determine_relationship_status_prefers_gone_when_recent_gap_is_large(self) -> None:
        evidence = {
            "time_span_days": 180,
            "recent_gap_days": 75,
            "trend_detail": {"direction": "up"},
        }

        status = determine_relationship_status(evidence)

        self.assertEqual(status, "gone")

    def test_score_relationship_confidence_penalizes_llm_decision_outside_candidates(self) -> None:
        evidence = {
            "photo_count": 8,
            "scenes": ["公寓", "咖啡馆", "海边"],
            "contact_types": ["hug"],
            "with_user_only": True,
            "weekend_frequency": "高",
            "co_appearing_persons": [],
            "anomalies": [],
            "trend_detail": {"direction": "flat"},
        }

        confidence = score_relationship_confidence(
            evidence=evidence,
            intimacy_score=0.76,
            relationship_type="family",
            candidate_types=["romantic", "bestie", "close_friend"],
        )

        self.assertLess(confidence, 0.7)

    def test_infer_relationship_uses_high_risk_veto_but_keeps_llm_status_without_redline(self) -> None:
        processor = LLMProcessor.__new__(LLMProcessor)
        processor.primary_person_id = "Person_001"

        evidence = {
            "photo_count": 8,
            "time_span": "120天",
            "time_span_days": 120,
            "scenes": ["公寓", "咖啡馆", "海边"],
            "interaction_behavior": ["合影", "共同进餐"],
            "weekend_frequency": "高",
            "with_user_only": True,
            "sample_scenes": [],
            "contact_types": ["hug", "holding_hands"],
            "rela_events": [
                {
                    "event_id": "EVT_001",
                    "date": "2026-01-03",
                    "title": "海边约会",
                    "location": "海边",
                    "photo_count": 4,
                    "description": "一起在海边散步",
                    "participants": ["Person_001", "Person_002"],
                    "narrative_synthesis": "两人在周末海边约会",
                    "social_dynamics": [
                        {
                            "target_id": "Person_002",
                            "interaction_type": "约会",
                            "social_clue": "1v1 私密场景",
                            "relation_hypothesis": "romantic",
                            "confidence": 0.9,
                        }
                    ],
                }
            ],
            "monthly_frequency": 2.0,
            "trend_detail": {"direction": "up", "first_half_freq": 1.0, "second_half_freq": 2.0, "change_ratio": 2.0},
            "co_appearing_persons": [],
            "anomalies": [],
            "private_scene_ratio": 0.5,
            "dominant_scene_ratio": 0.34,
            "recent_gap_days": 3,
        }

        processor._compute_intimacy_score = lambda person_id, evidence, vlm_results, face_db: 0.78
        processor._call_llm_via_official_api = lambda prompt: {
            "relationship_type": "family",
            "status": "gone",
            "confidence": 0.99,
            "reason": "hallucinated",
        }
        processor._create_relationship_prompt = lambda person_id, evidence, intimacy_score, candidate_types=None, code_status_suggestion="": "prompt"

        relationship = processor._infer_relationship(
            person_id="Person_002",
            evidence=evidence,
            vlm_results=[],
            face_db={},
        )

        self.assertEqual(relationship.relationship_type, "romantic")
        self.assertEqual(relationship.status, "gone")
        self.assertLess(relationship.confidence, 0.99)
        self.assertGreater(relationship.confidence, 0.7)
        self.assertIn("候选关系建议", relationship.reasoning)

    def test_infer_relationship_allows_low_risk_llm_override_and_blends_confidence(self) -> None:
        processor = LLMProcessor.__new__(LLMProcessor)
        processor.primary_person_id = "Person_001"
        processor._compute_intimacy_score = lambda person_id, evidence, vlm_results, face_db: 0.41
        processor._create_relationship_prompt = lambda person_id, evidence, intimacy_score, candidate_types=None, code_status_suggestion="": "prompt"
        processor._call_llm_via_official_api = lambda prompt: {
            "relationship_type": "friend",
            "status": "stable",
            "confidence": 0.9,
            "reason": "low-risk override",
        }

        evidence = {
            "photo_count": 5,
            "time_span": "120天",
            "time_span_days": 120,
            "recent_gap_days": 4,
            "scenes": ["办公室"],
            "interaction_behavior": ["共同工作"],
            "weekend_frequency": "低",
            "with_user_only": False,
            "sample_scenes": [],
            "contact_types": ["no_contact"],
            "rela_events": [],
            "monthly_frequency": 1.2,
            "trend_detail": {"direction": "flat"},
            "co_appearing_persons": [],
            "anomalies": [],
        }

        with (
            patch("services.llm_processor.infer_relationship_candidates", return_value=["classmate_colleague", "acquaintance"]),
            patch("services.llm_processor.determine_relationship_status", return_value="growing"),
            patch("services.llm_processor.score_relationship_confidence", return_value=0.6),
        ):
            relationship = processor._infer_relationship(
                person_id="Person_002",
                evidence=evidence,
                vlm_results=[],
                face_db={},
            )

        self.assertEqual(relationship.relationship_type, "friend")
        self.assertEqual(relationship.status, "stable")
        self.assertAlmostEqual(relationship.confidence, 0.78, places=3)
        self.assertEqual(relationship.evidence["decision_trace"]["candidate_types"], ["classmate_colleague", "acquaintance"])
        self.assertEqual(relationship.evidence["decision_trace"]["code_status_suggestion"], "growing")
        self.assertEqual(relationship.evidence["decision_trace"]["final_relationship_type"], "friend")
        self.assertEqual(relationship.evidence["decision_trace"]["applied_vetoes"], [])

    def test_infer_relationship_vetoes_high_risk_type_outside_candidates(self) -> None:
        processor = LLMProcessor.__new__(LLMProcessor)
        processor.primary_person_id = "Person_001"
        processor._compute_intimacy_score = lambda person_id, evidence, vlm_results, face_db: 0.75
        processor._create_relationship_prompt = lambda person_id, evidence, intimacy_score, candidate_types=None, code_status_suggestion="": "prompt"
        processor._call_llm_via_official_api = lambda prompt: {
            "relationship_type": "romantic",
            "status": "stable",
            "confidence": 0.9,
            "reason": "overcalled romantic",
        }

        evidence = {
            "photo_count": 7,
            "time_span": "120天",
            "time_span_days": 120,
            "recent_gap_days": 2,
            "scenes": ["咖啡馆", "街区"],
            "interaction_behavior": ["合影"],
            "weekend_frequency": "高",
            "with_user_only": False,
            "sample_scenes": [],
            "contact_types": ["standing_near"],
            "rela_events": [],
            "monthly_frequency": 1.8,
            "trend_detail": {"direction": "flat"},
            "co_appearing_persons": [],
            "anomalies": [],
        }

        with (
            patch("services.llm_processor.infer_relationship_candidates", return_value=["bestie", "close_friend"]),
            patch("services.llm_processor.determine_relationship_status", return_value="stable"),
            patch("services.llm_processor.score_relationship_confidence", return_value=0.6),
        ):
            relationship = processor._infer_relationship(
                person_id="Person_002",
                evidence=evidence,
                vlm_results=[],
                face_db={},
            )

        self.assertEqual(relationship.relationship_type, "bestie")
        self.assertEqual(relationship.status, "stable")
        self.assertAlmostEqual(relationship.confidence, 0.68, places=3)
        self.assertIn("RELATIONSHIP_TYPE_HIGH_RISK_VETO", relationship.evidence["decision_trace"]["applied_vetoes"])

    def test_infer_relationship_enforces_time_redlines_on_status(self) -> None:
        processor = LLMProcessor.__new__(LLMProcessor)
        processor.primary_person_id = "Person_001"
        processor._compute_intimacy_score = lambda person_id, evidence, vlm_results, face_db: 0.33
        processor._create_relationship_prompt = lambda person_id, evidence, intimacy_score, candidate_types=None, code_status_suggestion="": "prompt"
        processor._call_llm_via_official_api = lambda prompt: {
            "relationship_type": "friend",
            "status": "stable",
            "confidence": 0.8,
            "reason": "ignored time redline",
        }

        evidence = {
            "photo_count": 4,
            "time_span": "180天",
            "time_span_days": 180,
            "recent_gap_days": 75,
            "scenes": ["校园"],
            "interaction_behavior": ["聊天"],
            "weekend_frequency": "低",
            "with_user_only": False,
            "sample_scenes": [],
            "contact_types": ["no_contact"],
            "rela_events": [],
            "monthly_frequency": 0.7,
            "trend_detail": {"direction": "up"},
            "co_appearing_persons": [],
            "anomalies": [],
        }

        with (
            patch("services.llm_processor.infer_relationship_candidates", return_value=["friend", "acquaintance"]),
            patch("services.llm_processor.determine_relationship_status", return_value="growing"),
            patch("services.llm_processor.score_relationship_confidence", return_value=0.5),
        ):
            relationship = processor._infer_relationship(
                person_id="Person_002",
                evidence=evidence,
                vlm_results=[],
                face_db={},
            )

        self.assertEqual(relationship.relationship_type, "friend")
        self.assertEqual(relationship.status, "gone")
        self.assertAlmostEqual(relationship.confidence, 0.58, places=3)
        self.assertIn("STATUS_GONE_REDLINE", relationship.evidence["decision_trace"]["applied_vetoes"])


if __name__ == "__main__":
    unittest.main()
