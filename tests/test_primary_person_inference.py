from __future__ import annotations

import unittest

from services.face_recognition import FaceRecognition


class PrimaryPersonInferenceTests(unittest.TestCase):
    def test_face_recognition_primary_is_none_when_top_photo_count_ties(self) -> None:
        service = FaceRecognition.__new__(FaceRecognition)
        service.photo_results = {
            "photo_001": {
                "timestamp": "2026-03-16T01:00:00",
                "faces": [{"person_id": "Person_001"}],
            },
            "photo_002": {
                "timestamp": "2026-03-16T01:00:01",
                "faces": [{"person_id": "Person_002"}],
            },
        }
        service.state = {"persons": {}}

        primary_person_id = FaceRecognition._compute_primary_person_id(service)

        self.assertIsNone(primary_person_id)

    def test_face_recognition_primary_requires_multiple_photos(self) -> None:
        service = FaceRecognition.__new__(FaceRecognition)
        service.photo_results = {
            "photo_001": {
                "timestamp": "2026-03-16T01:00:00",
                "faces": [{"person_id": "Person_001"}],
            },
            "photo_002": {
                "timestamp": "2026-03-16T01:00:01",
                "faces": [{"person_id": "Person_001"}],
            },
            "photo_003": {
                "timestamp": "2026-03-16T01:00:02",
                "faces": [{"person_id": "Person_002"}],
            },
        }
        service.state = {"persons": {}}

        primary_person_id = FaceRecognition._compute_primary_person_id(service)

        self.assertEqual(primary_person_id, "Person_001")


if __name__ == "__main__":
    unittest.main()
