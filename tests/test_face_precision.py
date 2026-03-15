from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from services.face_precision import (
    aggregate_candidate_matches,
    compute_face_quality,
    decide_cluster_merge,
    decide_match,
    filter_same_photo_candidates,
    load_strong_threshold,
)
from vendor.face_recognition_src.face_recognition.index_store import SearchMatch


class FacePrecisionTests(unittest.TestCase):
    def test_compute_face_quality_flags_small_blurry_edge_face(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        result = compute_face_quality(image, {"x": 0, "y": 0, "w": 20, "h": 20})
        self.assertLess(result["quality_score"], 0.4)
        self.assertIn("small_face", result["quality_flags"])
        self.assertIn("blurry", result["quality_flags"])
        self.assertIn("edge_clipped", result["quality_flags"])

    def test_aggregate_candidate_matches_groups_hits_by_person(self) -> None:
        matches = [
            SearchMatch(score=0.55, faiss_id=1),
            SearchMatch(score=0.54, faiss_id=2),
            SearchMatch(score=0.52, faiss_id=3),
        ]
        candidates = aggregate_candidate_matches(matches, {1: "Person_001", 2: "Person_001", 3: "Person_002"}, {})
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].person_id, "Person_001")
        self.assertEqual(candidates[0].support_count, 2)
        self.assertAlmostEqual(candidates[0].best_similarity, 0.55)
        self.assertAlmostEqual(candidates[0].mean_similarity, 0.545)

    def test_decide_match_prefers_strong_match(self) -> None:
        decision = decide_match(
            candidates=[
                type("Candidate", (), {"person_id": "Person_001", "best_similarity": 0.62, "mean_similarity": 0.60, "support_count": 2})(),
                type("Candidate", (), {"person_id": "Person_002", "best_similarity": 0.56, "mean_similarity": 0.56, "support_count": 1})(),
            ],
            quality_score=0.78,
            strong_threshold=0.60,
            weak_threshold=0.56,
            margin_threshold=0.03,
            min_quality_for_gray_zone=0.40,
        )
        self.assertEqual(decision["decision"], "strong_match")
        self.assertEqual(decision["person_id"], "Person_001")

    def test_decide_match_allows_gray_match_with_support(self) -> None:
        decision = decide_match(
            candidates=[
                type("Candidate", (), {"person_id": "Person_001", "best_similarity": 0.58, "mean_similarity": 0.57, "support_count": 2})(),
                type("Candidate", (), {"person_id": "Person_002", "best_similarity": 0.56, "mean_similarity": 0.56, "support_count": 1})(),
            ],
            quality_score=0.72,
            strong_threshold=0.60,
            weak_threshold=0.56,
            margin_threshold=0.03,
            min_quality_for_gray_zone=0.40,
        )
        self.assertEqual(decision["decision"], "gray_match")
        self.assertEqual(decision["person_id"], "Person_001")

    def test_decide_match_creates_new_person_when_ambiguous(self) -> None:
        decision = decide_match(
            candidates=[
                type("Candidate", (), {"person_id": "Person_001", "best_similarity": 0.58, "mean_similarity": 0.55, "support_count": 1})(),
                type("Candidate", (), {"person_id": "Person_002", "best_similarity": 0.57, "mean_similarity": 0.57, "support_count": 1})(),
            ],
            quality_score=0.35,
            strong_threshold=0.60,
            weak_threshold=0.56,
            margin_threshold=0.03,
            min_quality_for_gray_zone=0.40,
        )
        self.assertEqual(decision["decision"], "new_person_from_ambiguity")
        self.assertIsNone(decision["person_id"])

    def test_decide_match_allows_profile_rescue_when_side_face_is_clear(self) -> None:
        decision = decide_match(
            candidates=[
                type("Candidate", (), {"person_id": "Person_001", "best_similarity": 0.57, "mean_similarity": 0.55, "support_count": 1})(),
                type("Candidate", (), {"person_id": "Person_002", "best_similarity": 0.54, "mean_similarity": 0.54, "support_count": 1})(),
            ],
            quality_score=0.66,
            strong_threshold=0.60,
            weak_threshold=0.56,
            margin_threshold=0.03,
            min_quality_for_gray_zone=0.40,
            pose_bucket="left_profile",
            pose_yaw=-28.0,
        )
        self.assertEqual(decision["decision"], "profile_rescue_match")
        self.assertEqual(decision["person_id"], "Person_001")

    def test_load_strong_threshold_reads_benchmark_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "latest.json"
            path.write_text('{"recommended_threshold": 0.33}', encoding="utf-8")
            self.assertAlmostEqual(load_strong_threshold(fallback=0.5, threshold_path=str(path)), 0.33)

    def test_filter_same_photo_candidates_prefers_new_person_unless_similarity_is_very_high(self) -> None:
        candidates = [
            type("Candidate", (), {"person_id": "Person_001", "best_similarity": 0.34, "mean_similarity": 0.34, "support_count": 1})(),
            type("Candidate", (), {"person_id": "Person_002", "best_similarity": 0.28, "mean_similarity": 0.28, "support_count": 1})(),
            type("Candidate", (), {"person_id": "Person_003", "best_similarity": 0.60, "mean_similarity": 0.60, "support_count": 1})(),
        ]
        filtered = filter_same_photo_candidates(candidates, {"Person_001"}, same_photo_match_threshold=0.52)
        self.assertEqual([candidate.person_id for candidate in filtered], ["Person_002", "Person_003"])

    def test_decide_cluster_merge_uses_profile_bridge_with_two_supporting_links(self) -> None:
        embeddings = {
            1: np.asarray([1.0, 0.0], dtype=np.float32),
            2: np.asarray([0.95, 0.05], dtype=np.float32),
            3: np.asarray([0.93, 0.07], dtype=np.float32),
        }
        decision = decide_cluster_merge(
            "Person_002",
            [
                {
                    "faiss_id": 1,
                    "image_id": "img-010",
                    "quality_score": 0.72,
                    "score": 0.95,
                    "pose_bucket": "frontal",
                },
                {
                    "faiss_id": 2,
                    "image_id": "img-872",
                    "quality_score": 0.66,
                    "score": 0.91,
                    "pose_bucket": "left_profile",
                },
            ],
            "Person_009",
            [
                {
                    "faiss_id": 3,
                    "image_id": "img-003",
                    "quality_score": 0.69,
                    "score": 0.93,
                    "pose_bucket": "right_profile",
                }
            ],
            embedding_lookup=lambda faiss_id: embeddings.get(faiss_id),
            strong_threshold=0.30,
            high_quality_threshold=0.40,
        )
        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertIn(decision.decision, {"strong_cluster_merge", "supported_cluster_merge"})
        self.assertTrue(decision.profile_bridge)

    def test_decide_cluster_merge_rejects_people_that_coexist_in_same_photo(self) -> None:
        embeddings = {
            1: np.asarray([1.0, 0.0], dtype=np.float32),
            2: np.asarray([0.95, 0.05], dtype=np.float32),
        }
        decision = decide_cluster_merge(
            "Person_010",
            [
                {
                    "faiss_id": 1,
                    "image_id": "img-shared",
                    "quality_score": 0.72,
                    "score": 0.95,
                    "pose_bucket": "frontal",
                }
            ],
            "Person_011",
            [
                {
                    "faiss_id": 2,
                    "image_id": "img-shared",
                    "quality_score": 0.74,
                    "score": 0.94,
                    "pose_bucket": "frontal",
                }
            ],
            embedding_lookup=lambda faiss_id: embeddings.get(faiss_id),
            strong_threshold=0.30,
            high_quality_threshold=0.40,
        )
        self.assertIsNone(decision)


if __name__ == "__main__":
    unittest.main()
