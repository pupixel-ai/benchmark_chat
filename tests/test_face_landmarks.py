from __future__ import annotations

import math
import unittest

import numpy as np

from services.face_landmarks import (
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


if __name__ == "__main__":
    unittest.main()
