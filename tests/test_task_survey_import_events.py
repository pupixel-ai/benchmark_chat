from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.task_survey_import_events import build_survey_import_event_payload


class TaskSurveyImportEventsTests(unittest.TestCase):
    def test_build_survey_import_event_payload_emits_minimal_contract(self) -> None:
        task = self._build_task()

        with patch(
            "backend.memory_full_retrieval._survey_asset_store.presigned_get_url",
            side_effect=lambda task_id, relative_path, expires_in=86400: f"https://signed.example.com/{task_id}/{relative_path}",
        ):
            event = build_survey_import_event_payload(task)

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(event["event_type"], "survey.import.ready")
        self.assertEqual(event["username"], "alice")
        self.assertIn("profile", event["payload"])
        self.assertIn("relationships", event["payload"])
        self.assertIn("events", event["payload"])
        self.assertIn("vlm_observations", event["payload"])
        self.assertIn("face_recognition", event["payload"])
        self.assertIn("snapshot_mode", event["payload"])
        self.assertIn("api_bundle_url", event["payload"])
        self.assertEqual(event["schema_version"], "v2")
        self.assertEqual(event["payload"]["profile"]["report_markdown"], "# Profile\n\nConcert-heavy social life.")
        self.assertEqual(
            event["payload"]["profile"]["structured_profile"]["long_term_facts"]["identity"]["name"]["value"],
            "Alice",
        )
        relationship = event["payload"]["relationships"][0]
        self.assertEqual(
            set(relationship.keys()),
            {"person_id", "photo_count", "photo_ids_sample", "boxed_image_url", "sample_scenes"},
        )
        self.assertEqual(relationship["person_id"], "Person_002")
        self.assertEqual(relationship["photo_count"], 1)
        self.assertEqual(relationship["photo_ids_sample"], ["photo_001"])
        self.assertEqual(
            relationship["boxed_image_url"],
            "https://signed.example.com/task-1/cache/boxed_images/photo_001_boxed.webp",
        )
        self.assertEqual(relationship["sample_scenes"][0]["summary"], "Live concert with a close friend.")

    def test_build_survey_import_event_payload_falls_back_to_related_event_boxed_image(self) -> None:
        task = self._build_task(
            supporting_photo_ids=[],
            boxed_image_url="/api/assets/task-1/cache/boxed_images/photo_001_boxed.webp",
        )

        with patch(
            "backend.memory_full_retrieval._survey_asset_store.presigned_get_url",
            side_effect=lambda task_id, relative_path, expires_in=86400: f"https://signed.example.com/{task_id}/{relative_path}",
        ):
            event = build_survey_import_event_payload(task)

        self.assertIsNotNone(event)
        assert event is not None
        relationship = event["payload"]["relationships"][0]
        self.assertEqual(relationship["photo_count"], 1)
        self.assertEqual(relationship["photo_ids_sample"], ["photo_001"])
        self.assertEqual(
            relationship["boxed_image_url"],
            "https://signed.example.com/task-1/cache/boxed_images/photo_001_boxed.webp",
        )

    def test_build_survey_import_event_payload_returns_none_without_survey_username(self) -> None:
        task = self._build_task(survey_username=None)

        event = build_survey_import_event_payload(task)

        self.assertIsNone(event)

    def test_build_survey_import_event_payload_keeps_structured_only_profile(self) -> None:
        task = self._build_task()
        task["result"]["memory"]["lp3_profile"]["report_markdown"] = ""
        task["result"]["memory"].pop("delta_event_revisions", None)
        task["result"]["memory"].pop("delta_relationship_revisions", None)

        event = build_survey_import_event_payload(task)

        self.assertIsNotNone(event)
        assert event is not None
        self.assertEqual(
            event["payload"]["profile"]["structured_profile"]["long_term_facts"]["identity"]["name"]["value"],
            "Alice",
        )
        self.assertEqual(event["payload"]["relationships"], [])

    def _build_task(
        self,
        *,
        survey_username: str | None = "alice",
        supporting_photo_ids: list[str] | None = None,
        boxed_image_url: str = "/api/assets/task-1/cache/boxed_images/photo_001_boxed.webp",
    ) -> dict:
        relationship_supporting_photo_ids = ["hash-001"] if supporting_photo_ids is None else supporting_photo_ids
        return {
            "task_id": "task-1",
            "user_id": "user-1",
            "version": "v0327-db-query",
            "status": "completed",
            "stage": "completed",
            "created_at": "2026-03-29T00:00:00",
            "updated_at": "2026-03-29T01:00:00",
            "options": {
                "creation_source": "directory",
                "survey_username": survey_username,
            },
            "uploads": [
                {
                    "image_id": "photo_001",
                    "filename": "sample.png",
                    "path": "uploads/001_sample.png",
                    "url": "/api/assets/task-1/uploads/001_sample.png",
                    "preview_url": None,
                    "source_hash": "hash-001",
                    "timestamp": "2026-03-20T20:00:00",
                }
            ],
            "result": {
                "face_recognition": {
                    "images": [
                        {
                            "image_id": "photo_001",
                            "filename": "sample.png",
                            "source_hash": "hash-001",
                            "timestamp": "2026-03-20T20:00:00",
                            "display_image_url": "/api/assets/task-1/uploads/001_sample.png",
                            "boxed_image_url": boxed_image_url,
                            "faces": [],
                        }
                    ]
                },
                "memory": {
                    "pipeline_family": "v0321_3",
                    "lp3_profile": {
                        "report_markdown": "# Profile\n\nConcert-heavy social life.",
                        "structured": {
                            "long_term_facts": {
                                "identity": {
                                    "name": {
                                        "value": "Alice",
                                        "confidence": 0.91,
                                        "reasoning": "Repeated self-reference across albums.",
                                    }
                                }
                            }
                        },
                    },
                    "delta_event_revisions": [
                        {
                            "event_revision_id": "event_rev_001",
                            "title": "Concert Night",
                            "event_summary": "Live concert with a close friend.",
                            "started_at": "2026-03-20T20:00:00",
                            "participant_person_ids": ["Person_001", "Person_002"],
                            "depicted_person_ids": ["Person_001", "Person_002"],
                            "place_refs": ["Shanghai"],
                            "original_photo_ids": ["hash-001"],
                        }
                    ],
                    "delta_relationship_revisions": [
                        {
                            "relationship_revision_id": "rel_rev_001",
                            "target_person_id": "Person_002",
                            "supporting_photo_ids": relationship_supporting_photo_ids,
                        }
                    ],
                },
            },
        }
