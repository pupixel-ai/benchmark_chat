from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from services.face_landmarks import (
    FaceLandmarkAnalyzer,
    build_insightface_fallback_pose,
    classify_pose_bucket,
    compute_eye_visibility_ratio,
    euler_from_rotation_matrix,
)


class FaceLandmarkTests(unittest.TestCase):
    def test_classify_pose_bucket_marks_frontal_and_profiles(self) -> None:
        self.assertEqual(classify_pose_bucket(0.0), "frontal")
        self.assertEqual(classify_pose_bucket(21.0), "right_profile")
        self.assertEqual(classify_pose_bucket(-24.0), "left_profile")
        self.assertEqual(classify_pose_bucket(None), "unknown")

    def test_compute_eye_visibility_ratio_is_symmetric(self) -> None:
        self.assertAlmostEqual(compute_eye_visibility_ratio(0.4, 0.2), 0.5)
        self.assertAlmostEqual(compute_eye_visibility_ratio(0.2, 0.4), 0.5)

    def test_euler_from_rotation_matrix_extracts_yaw(self) -> None:
        yaw = math.radians(30.0)
        rotation = np.asarray(
            [
                [math.cos(yaw), 0.0, math.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-math.sin(yaw), 0.0, math.cos(yaw)],
            ],
            dtype="float32",
        )
        yaw_deg, pitch_deg, roll_deg = euler_from_rotation_matrix(rotation)
        self.assertAlmostEqual(yaw_deg, 30.0, delta=0.5)
        self.assertAlmostEqual(pitch_deg, 0.0, delta=0.5)
        self.assertAlmostEqual(roll_deg, 0.0, delta=0.5)

    def test_build_insightface_fallback_pose_maps_pitch_yaw_roll(self) -> None:
        pose = build_insightface_fallback_pose(
            [-7.0, -67.8, 3.1],
            insight_landmark_2d_106=[[0.0, 0.0]],
        )
        self.assertIsNotNone(pose)
        assert pose is not None
        self.assertAlmostEqual(pose.pose_pitch, -7.0, delta=1e-4)
        self.assertAlmostEqual(pose.pose_yaw, -67.8, delta=1e-4)
        self.assertAlmostEqual(pose.pose_roll, 3.1, delta=1e-4)
        self.assertEqual(pose.pose_bucket, "left_profile")
        self.assertTrue(pose.landmark_detected)
        self.assertEqual(pose.landmark_source, "insightface-pose-fallback")

    def test_landmarker_uses_model_buffer_instead_of_path(self) -> None:
        class FakeFaceLandmarker:
            captured_options = None

            @classmethod
            def create_from_options(cls, options):
                cls.captured_options = options
                return object()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "face_landmarker.task"
            model_path.write_bytes(b"fake-model")

            analyzer = FaceLandmarkAnalyzer(
                enabled=True,
                model_path=str(model_path),
                model_url="https://example.com/model",
            )

            with patch("services.face_landmarks.ensure_model_downloaded", return_value=model_path), patch(
                "services.face_landmarks.FaceLandmarker", FakeFaceLandmarker
            ), patch(
                "services.face_landmarks.FaceLandmarkerOptions", lambda **kwargs: kwargs
            ), patch(
                "services.face_landmarks.BaseOptions", lambda **kwargs: kwargs
            ), patch(
                "services.face_landmarks.RunningMode",
                type("RunningModeStub", (), {"IMAGE": "IMAGE"}),
            ):
                landmarker = analyzer._get_landmarker()

        self.assertIsNotNone(landmarker)
        assert FakeFaceLandmarker.captured_options is not None
        base_options = FakeFaceLandmarker.captured_options["base_options"]
        self.assertEqual(base_options["model_asset_buffer"], b"fake-model")
        self.assertNotIn("model_asset_path", base_options)

    def test_landmarker_init_failure_degrades_to_none_once(self) -> None:
        class FailingFaceLandmarker:
            calls = 0

            @classmethod
            def create_from_options(cls, options):
                cls.calls += 1
                raise RuntimeError("Unable to map file to memory buffer, errno=22")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "face_landmarker.task"
            model_path.write_bytes(b"fake-model")

            analyzer = FaceLandmarkAnalyzer(
                enabled=True,
                model_path=str(model_path),
                model_url="https://example.com/model",
            )

            with patch("services.face_landmarks.ensure_model_downloaded", return_value=model_path), patch(
                "services.face_landmarks.FaceLandmarker", FailingFaceLandmarker
            ), patch(
                "services.face_landmarks.FaceLandmarkerOptions", lambda **kwargs: kwargs
            ), patch(
                "services.face_landmarks.BaseOptions", lambda **kwargs: kwargs
            ), patch(
                "services.face_landmarks.RunningMode",
                type("RunningModeStub", (), {"IMAGE": "IMAGE"}),
            ):
                first = analyzer._get_landmarker()
                second = analyzer._get_landmarker()

        self.assertIsNone(first)
        self.assertIsNone(second)
        self.assertTrue(analyzer._landmarker_init_failed)
        self.assertEqual(FailingFaceLandmarker.calls, 1)


if __name__ == "__main__":
    unittest.main()
