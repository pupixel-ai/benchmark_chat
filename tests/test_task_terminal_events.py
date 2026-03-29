from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.task_terminal_events import build_terminal_event_payload


class _FaceReviewStoreStub:
    def get_task_feedback(self, *, task_id: str, user_id: str, source_hashes: set[str]) -> dict:
        del task_id, user_id, source_hashes
        return {"reviews": {}, "policies": {}}


class TaskTerminalEventsTests(unittest.TestCase):
    def test_payload_switches_to_reduced_mode_when_message_is_too_large(self) -> None:
        task = {
            "task_id": "task-1",
            "user_id": "user-1",
            "version": "v0327-db-query",
            "status": "completed",
            "stage": "completed",
            "updated_at": "2026-03-29T00:00:00",
        }
        large_payload = {
            "task": {
                "task_id": "task-1",
                "user_id": "user-1",
                "version": "v0327-db-query",
                "status": "completed",
                "stage": "completed",
                "upload_count": 1,
                "worker_status": "running",
                "created_at": "2026-03-29T00:00:00",
                "updated_at": "2026-03-29T00:00:00",
            },
            "summary": {"primary_person_id": "Person_001"},
            "memory_core": {"items": ["x" * 2048]},
            "steps": {
                "pipeline_family": "v0325",
                "steps": {
                    "lp1": {
                        "status": "completed",
                        "summary": {"event_count": 2},
                        "data": ["x" * 2048 for _ in range(8)],
                        "attempts": ["x" * 1024 for _ in range(4)],
                        "failures": [],
                    }
                },
            },
            "reviews": {"reviews": {}, "policies": {}},
            "artifacts": {
                "artifact_count": 50,
                "files": [{"relative_path": f"artifacts/{index}.json", "asset_url": "/asset"} for index in range(50)],
                "named_urls": {"report_url": "/asset/report.json"},
            },
            "failure": None,
        }

        with patch("backend.task_terminal_events._build_business_payload", return_value=large_payload):
            event = build_terminal_event_payload(
                task,
                face_review_store=_FaceReviewStoreStub(),
                asset_url_builder=lambda *_: "/asset",
                message_max_bytes=700,
            )

        self.assertEqual(event["snapshot_mode"], "reduced")
        self.assertEqual(event["payload"]["task"]["task_id"], "task-1")
        self.assertEqual(event["payload"]["summary"]["primary_person_id"], "Person_001")
        self.assertIn("named_urls", event["payload"]["artifacts"])
        self.assertIn("memory_core", event["payload"])
