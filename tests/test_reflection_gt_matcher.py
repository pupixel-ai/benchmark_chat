from __future__ import annotations

import unittest


class ReflectionGTMatcherTests(unittest.TestCase):
    def test_compare_profile_field_values_marks_student_and_college_student_as_close_match(self) -> None:
        from services.reflection.gt_matcher import compare_profile_field_values

        result = compare_profile_field_values(
            field_key="long_term_facts.social_identity.education",
            predicted_value="college_student",
            gt_value="student",
        )

        self.assertEqual(result["grade"], "close_match")
        self.assertGreaterEqual(result["score"], 0.67)
        self.assertEqual(result["method"], "rule_hierarchy")

    def test_compare_profile_field_values_marks_subset_interest_list_as_partial_match(self) -> None:
        from services.reflection.gt_matcher import compare_profile_field_values

        result = compare_profile_field_values(
            field_key="long_term_facts.hobbies.interests",
            predicted_value=["音乐", "游戏"],
            gt_value=["音乐", "电影", "游戏"],
        )

        self.assertEqual(result["grade"], "partial_match")
        self.assertGreater(result["score"], 0.0)
        self.assertLess(result["score"], 0.67)
        self.assertEqual(result["method"], "rule_set_overlap")

    def test_compare_profile_field_values_marks_unrelated_values_as_mismatch(self) -> None:
        from services.reflection.gt_matcher import compare_profile_field_values

        result = compare_profile_field_values(
            field_key="long_term_facts.social_identity.education",
            predicted_value="worker",
            gt_value="student",
        )

        self.assertEqual(result["grade"], "mismatch")
        self.assertEqual(result["score"], 0.0)

