from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from models import Photo
from services.image_processor import ImageProcessor


class ImageProcessorTests(unittest.TestCase):
    def test_preprocess_keeps_all_photos_for_vlm(self) -> None:
        photos = [
            Photo(
                photo_id="photo_001",
                filename="a.jpg",
                path="/tmp/a.jpg",
                timestamp=datetime(2026, 3, 16, 12, 0, 0),
                location={},
            ),
            Photo(
                photo_id="photo_002",
                filename="b.jpg",
                path="/tmp/b.jpg",
                timestamp=datetime(2026, 3, 16, 12, 0, 1),
                location={},
            ),
            Photo(
                photo_id="photo_003",
                filename="c.jpg",
                path="/tmp/c.jpg",
                timestamp=datetime(2026, 3, 16, 12, 0, 2),
                location={},
            ),
        ]

        processor = ImageProcessor(cache_dir="runtime/test-image-processor-cache")

        with patch.object(ImageProcessor, "_compress_photos", return_value=photos) as compress_mock:
            result = processor.preprocess(photos)

        compress_mock.assert_called_once_with(photos)
        self.assertEqual([photo.photo_id for photo in result], ["photo_001", "photo_002", "photo_003"])

    def test_dedupe_before_face_recognition_removes_burst_duplicates(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_a = temp_path / "a.jpg"
            image_b = temp_path / "b.jpg"
            image_c = temp_path / "c.jpg"

            Image.new("RGB", (64, 64), color="red").save(image_a)
            Image.new("RGB", (64, 64), color="red").save(image_b)
            contrasting = Image.new("RGB", (64, 64), color="black")
            for x in range(32, 64):
                for y in range(64):
                    contrasting.putpixel((x, y), (255, 255, 255))
            contrasting.save(image_c)

            photos = [
                Photo(
                    photo_id="photo_001",
                    filename="a.jpg",
                    path=str(image_a),
                    timestamp=datetime(2026, 3, 16, 12, 0, 0),
                    location={"name": "livehouse"},
                    source_hash="hash-1",
                ),
                Photo(
                    photo_id="photo_002",
                    filename="b.jpg",
                    path=str(image_b),
                    timestamp=datetime(2026, 3, 16, 12, 0, 2),
                    location={"name": "livehouse"},
                    source_hash="hash-2",
                ),
                Photo(
                    photo_id="photo_003",
                    filename="c.jpg",
                    path=str(image_c),
                    timestamp=datetime(2026, 3, 16, 12, 0, 3),
                    location={"name": "livehouse"},
                    source_hash="hash-3",
                ),
            ]

            processor = ImageProcessor(cache_dir=str(temp_path / "cache"))
            result = processor.dedupe_before_face_recognition(photos)

        self.assertEqual([photo.photo_id for photo in result], ["photo_001", "photo_003"])


if __name__ == "__main__":
    unittest.main()
