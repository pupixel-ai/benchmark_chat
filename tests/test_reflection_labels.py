from __future__ import annotations

import unittest


class ReflectionLabelLookupTests(unittest.TestCase):
    def test_lookup_bilingual_label_reads_profile_field_mapping(self) -> None:
        from services.reflection.labels import lookup_bilingual_label

        label = lookup_bilingual_label("profile_field", "long_term_facts.social_identity.education")

        self.assertEqual(label, "社会身份：教育背景")

    def test_lookup_bilingual_label_reads_experiment_field_mapping(self) -> None:
        from services.reflection.labels import lookup_bilingual_label

        label = lookup_bilingual_label("comparison_grade", "partial_match")

        self.assertEqual(label, "部分匹配")

    def test_describe_profile_field_keeps_both_chinese_and_english(self) -> None:
        from services.reflection.labels import describe_profile_field

        label = describe_profile_field("long_term_facts.social_identity.education")

        self.assertEqual(label, "社会身份：教育背景（long_term_facts.social_identity.education）")
