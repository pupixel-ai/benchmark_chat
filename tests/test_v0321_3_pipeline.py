from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from services.v0321_3.pipeline import V03213PipelineFamily


class NullAssetStore:
    enabled = False
    bucket = None

    def upload_file(self, *args, **kwargs):
        return None

    def asset_url(self, *args, **kwargs):
        return None

    def sync_task_directory(self, *args, **kwargs):
        return None


class V03213PipelineFamilyTests(unittest.TestCase):
    def _build_family(self, task_dir: Path) -> V03213PipelineFamily:
        return V03213PipelineFamily(
            task_id="task_v03213_test",
            task_dir=task_dir,
            user_id="user_v03213_test",
            asset_store=NullAssetStore(),
            llm_processor=object(),
        )

    def test_unique_dedupes_object_wrapped_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            family = self._build_family(Path(tmpdir))

            values = family._unique(
                [
                    {"name": "Home"},
                    {"name": "Home"},
                    {"label": "Cafe"},
                    {"value": "Cafe"},
                    {"text": "Bookstore"},
                ]
            )

            self.assertEqual(values, ["Home", "Cafe", "Bookstore"])

    def test_normalize_llm_event_draft_coerces_object_wrapped_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            family = self._build_family(Path(tmpdir))
            fallback = {
                "title": "meal @ Home",
                "participant_person_ids": ["Person_001", "Person_002"],
                "depicted_person_ids": ["Person_003"],
                "place_refs": ["Home"],
                "confidence": 0.68,
                "atomic_evidence": [],
            }
            payload = {
                "title": "Breakfast",
                "summary": "Breakfast at home",
                "participant_person_ids": [{"person_id": "Person_001"}, {"id": "Person_999"}],
                "depicted_person_ids": [{"person_id": "Person_003"}],
                "place_refs": [{"name": "Home"}, {"name": "Elsewhere"}],
                "confidence": 0.9,
                "atomic_evidence": [
                    {
                        "evidence_type": "ocr",
                        "value_or_text": "Cafe receipt",
                        "photo_ids": [{"photo_id": "photo_001"}],
                        "confidence": 0.7,
                        "provenance": "ocr",
                    }
                ],
            }
            observations_by_photo = {
                "photo_001": {
                    "original_photo_ids": ["orig_001"],
                }
            }
            window = {"photo_ids": ["photo_001"]}

            normalized = family._normalize_llm_event_draft(
                payload=payload,
                fallback=fallback,
                observations_by_photo=observations_by_photo,
                window=window,
            )

            self.assertIsNotNone(normalized)
            assert normalized is not None
            self.assertEqual(normalized["participant_person_ids"], ["Person_001"])
            self.assertEqual(normalized["depicted_person_ids"], ["Person_003"])
            self.assertEqual(normalized["place_refs"], ["Home"])
            self.assertEqual(normalized["atomic_evidence"][0]["original_photo_ids"], ["orig_001"])

    def test_match_event_draft_handles_object_wrapped_overlap_signals(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            family = self._build_family(Path(tmpdir))
            draft = {
                "participant_person_ids": ["Person_001"],
                "place_refs": ["Home"],
                "started_at": "2026-03-15T09:40:00",
                "ended_at": "2026-03-15T10:00:00",
            }
            candidate = {
                "participant_person_ids": [{"person_id": "Person_001"}],
                "place_refs": [{"name": "Home"}],
                "started_at": "2026-03-15T09:00:00",
                "ended_at": "2026-03-15T09:30:00",
            }

            decision, matched = family._match_event_draft(draft, [candidate])

            self.assertEqual(decision, "merge")
            self.assertEqual(matched, candidate)


if __name__ == "__main__":
    unittest.main()
