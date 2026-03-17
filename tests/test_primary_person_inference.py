from __future__ import annotations

import unittest
from datetime import datetime

from models import Person, Photo
from services.face_recognition import FaceRecognition
from services.vlm_analyzer import VLMAnalyzer


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

    def test_vlm_analyzer_does_not_guess_primary_when_batch_is_ambiguous(self) -> None:
        analyzer = VLMAnalyzer.__new__(VLMAnalyzer)
        face_db = {
            "Person_001": Person(person_id="Person_001", name="", photo_count=1, first_seen=datetime(2026, 3, 16, 1, 0, 0)),
            "Person_002": Person(person_id="Person_002", name="", photo_count=1, first_seen=datetime(2026, 3, 16, 1, 0, 1)),
        }

        primary_person_id = VLMAnalyzer._infer_reliable_primary_person_id(analyzer, face_db)

        self.assertIsNone(primary_person_id)

    def test_vlm_prompt_warns_against_binding_visible_person_to_primary_when_primary_unknown(self) -> None:
        analyzer = VLMAnalyzer.__new__(VLMAnalyzer)
        photo = Photo(
            photo_id="photo_001",
            filename="IMG_3483.JPG",
            path="/tmp/IMG_3483.JPG",
            timestamp=datetime(2026, 3, 16, 1, 25, 0),
            location={},
        )
        photo.faces = [{"person_id": "Person_003"}]
        face_db = {
            "Person_003": Person(person_id="Person_003", name="", photo_count=1, first_seen=datetime(2026, 3, 16, 1, 25, 0)),
        }

        prompt = VLMAnalyzer._create_prompt(analyzer, photo, face_db, None)

        self.assertIn("没有可靠的主用户身份锚点", prompt)
        self.assertIn('summary 中使用"【拍摄者】"', prompt)
        self.assertIn("不要把任何 person_id 写成【主用户】", prompt)
        self.assertIn("镜面反射", prompt)
        self.assertIn("疑似载体中的人物", prompt)


if __name__ == "__main__":
    unittest.main()
