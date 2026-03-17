from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
