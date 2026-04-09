"""Tests for cam_detection_to_world coordinate transform and quaternion-to-rotmat.

These functions live in perception/frame_transforms.py and are the bridge between
SimBallDetector (camera frame) and the EKF (world frame).
"""

import importlib
import math
import os
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose

# Direct import of frame_transforms without triggering go1_ball_balance.__init__
_PERCEPTION_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "source", "go1_ball_balance", "go1_ball_balance", "perception")
)
_spec = importlib.util.spec_from_file_location("frame_transforms", os.path.join(_PERCEPTION_DIR, "frame_transforms.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_quat_to_rotmat = _mod.quat_to_rotmat
cam_detection_to_world = _mod.cam_detection_to_world


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z). Test helper."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


class TestQuatToRotmat(unittest.TestCase):
    """Test quaternion (w,x,y,z) to rotation matrix conversion."""

    def test_identity(self):
        R = _quat_to_rotmat(np.array([1.0, 0.0, 0.0, 0.0]))
        assert_allclose(R, np.eye(3), atol=1e-10)

    def test_90_deg_about_z(self):
        """90 deg about Z: X->Y, Y->-X, Z->Z."""
        c = math.cos(math.radians(45))
        s = math.sin(math.radians(45))
        q = np.array([c, 0.0, 0.0, s])
        R = _quat_to_rotmat(q)
        assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-10)
        assert_allclose(R @ [0, 1, 0], [-1, 0, 0], atol=1e-10)
        assert_allclose(R @ [0, 0, 1], [0, 0, 1], atol=1e-10)

    def test_180_deg_about_x(self):
        """180 deg about X: Y->-Y, Z->-Z."""
        q = np.array([0.0, 1.0, 0.0, 0.0])
        R = _quat_to_rotmat(q)
        assert_allclose(R @ [1, 0, 0], [1, 0, 0], atol=1e-10)
        assert_allclose(R @ [0, 1, 0], [0, -1, 0], atol=1e-10)
        assert_allclose(R @ [0, 0, 1], [0, 0, -1], atol=1e-10)

    def test_orthogonal_and_det_1(self):
        """Random quaternions produce proper rotation matrices."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            R = _quat_to_rotmat(q)
            assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
            assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_70_deg_body_pitch(self):
        """Config quaternion for -70 deg about Y tilts body +X upward.

        This is the body-frame rotation only (not quat_w_ros).
        """
        angle_deg = -70.0
        half = math.radians(angle_deg / 2)
        q = np.array([math.cos(half), 0.0, math.sin(half), 0.0])
        R = _quat_to_rotmat(q)
        body_fwd_world = R @ np.array([1.0, 0.0, 0.0])
        elevation = math.degrees(math.asin(np.clip(body_fwd_world[2], -1, 1)))
        self.assertAlmostEqual(elevation, 70.0, places=0)

    def test_inverse_property(self):
        """q and conjugate(q) give transpose rotation matrices."""
        rng = np.random.default_rng(123)
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
        R = _quat_to_rotmat(q)
        R_conj = _quat_to_rotmat(q_conj)
        assert_allclose(R_conj, R.T, atol=1e-10)

    def test_roundtrip_rotmat_to_quat(self):
        """rotmat -> quat -> rotmat roundtrip."""
        rng = np.random.default_rng(77)
        for _ in range(20):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            R = _quat_to_rotmat(q)
            q2 = _rotmat_to_quat(R)
            R2 = _quat_to_rotmat(q2)
            assert_allclose(R2, R, atol=1e-10)


class TestCamDetectionToWorld(unittest.TestCase):
    """Test the full camera-to-world coordinate transform."""

    def test_identity_camera(self):
        """Identity rotation at origin: pos_cam = pos_world."""
        pos_cam = np.array([0.1, -0.2, 0.5])
        cam_pos = np.zeros(3)
        cam_quat = np.array([1.0, 0.0, 0.0, 0.0])
        pos_w = cam_detection_to_world(pos_cam, cam_pos, cam_quat)
        assert_allclose(pos_w, pos_cam, atol=1e-10)

    def test_pure_translation(self):
        """Camera translated, no rotation: world = cam + offset."""
        pos_cam = np.array([0.0, 0.0, 0.5])
        cam_pos = np.array([1.0, 2.0, 3.0])
        cam_quat = np.array([1.0, 0.0, 0.0, 0.0])
        pos_w = cam_detection_to_world(pos_cam, cam_pos, cam_quat)
        assert_allclose(pos_w, [1.0, 2.0, 3.5], atol=1e-10)

    def test_camera_looking_straight_up(self):
        """Camera at (0,0,0.4) with identity quat_w_ros (optical = world +Z).

        With identity quat_w_ros: cam +Z maps to world +Z.
        Ball 0.3m along optical axis -> 0.3m above camera.
        """
        q = np.array([1.0, 0.0, 0.0, 0.0])
        cam_pos = np.array([0.0, 0.0, 0.4])
        pos_cam = np.array([0.0, 0.0, 0.3])
        pos_w = cam_detection_to_world(pos_cam, cam_pos, q)
        assert_allclose(pos_w, [0.0, 0.0, 0.7], atol=1e-6)

    def test_70_deg_tilt_detection(self):
        """Camera with optical axis at 70 deg elevation above horizontal.

        Construct quat_w_ros so cam +Z -> world (cos70, 0, sin70).
        Ball 0.3m along optical axis should land at camera + 0.3 * direction.
        """
        elev = math.radians(70.0)
        # Build rotation matrix: columns = where cam axes go in world
        z_axis = np.array([math.cos(elev), 0.0, math.sin(elev)])  # optical
        x_axis = np.array([0.0, -1.0, 0.0])  # cam right -> world -Y
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        R = np.column_stack([x_axis, y_axis, z_axis])
        assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

        q = _rotmat_to_quat(R)
        cam_pos = np.array([-0.08, 0.0, 0.46])
        pos_cam = np.array([0.0, 0.0, 0.3])
        pos_w = cam_detection_to_world(pos_cam, cam_pos, q)

        expected = cam_pos + 0.3 * z_axis
        assert_allclose(pos_w, expected, atol=1e-10)
        self.assertGreater(pos_w[2], cam_pos[2])

    def test_ball_offset_in_camera_x(self):
        """Ball off-centre in camera +X (right in ROS frame)."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        cam_pos = np.zeros(3)
        pos_cam = np.array([0.1, 0.0, 0.5])
        pos_w = cam_detection_to_world(pos_cam, cam_pos, q)
        assert_allclose(pos_w, [0.1, 0.0, 0.5], atol=1e-10)

    def test_roundtrip_world_to_cam_to_world(self):
        """Place point in world, project to cam, recover via our function."""
        rng = np.random.default_rng(99)
        for _ in range(10):
            cam_pos = rng.uniform(-1, 1, size=3)
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            R = _quat_to_rotmat(q)
            point_w = rng.uniform(-2, 2, size=3)
            # World -> camera: pos_cam = R^T @ (point_w - cam_pos)
            pos_cam = R.T @ (point_w - cam_pos)
            # Camera -> world
            recovered = cam_detection_to_world(pos_cam, cam_pos, q)
            assert_allclose(recovered, point_w, atol=1e-10)

    def test_batch_consistency_across_heights(self):
        """Transform is consistent across different trunk heights."""
        elev = math.radians(70.0)
        z_axis = np.array([math.cos(elev), 0.0, math.sin(elev)])
        x_axis = np.array([0.0, -1.0, 0.0])
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        R = np.column_stack([x_axis, y_axis, z_axis])
        q = _rotmat_to_quat(R)

        for trunk_z in [0.30, 0.35, 0.40, 0.45]:
            cam_pos = np.array([-0.08, 0.0, trunk_z + 0.06])
            pos_cam = np.array([0.0, 0.0, 0.2])
            pos_w = cam_detection_to_world(pos_cam, cam_pos, q)
            expected = cam_pos + 0.2 * z_axis
            assert_allclose(pos_w, expected, atol=1e-10)
            self.assertGreater(pos_w[2], cam_pos[2])


class TestEdgeCases(unittest.TestCase):
    """Edge cases and numerical robustness."""

    def test_zero_detection(self):
        """Detection at camera origin -> world = camera position."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        cam_pos = np.array([1.0, 2.0, 3.0])
        pos_w = cam_detection_to_world(np.zeros(3), cam_pos, q)
        assert_allclose(pos_w, cam_pos, atol=1e-10)

    def test_negative_quaternion(self):
        """q and -q represent the same rotation."""
        rng = np.random.default_rng(7)
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        pos_cam = rng.uniform(-1, 1, size=3)
        cam_pos = rng.uniform(-1, 1, size=3)
        pos_w1 = cam_detection_to_world(pos_cam, cam_pos, q)
        pos_w2 = cam_detection_to_world(pos_cam, cam_pos, -q)
        assert_allclose(pos_w1, pos_w2, atol=1e-10)

    def test_slightly_non_unit_quaternion(self):
        """Small floating point error in quat norm is tolerated."""
        q = np.array([1.0001, 0.0, 0.0, 0.0])
        cam_pos = np.zeros(3)
        pos_cam = np.array([0.0, 0.0, 0.5])
        pos_w = cam_detection_to_world(pos_cam, cam_pos, q)
        assert_allclose(pos_w, pos_cam, atol=1e-3)

    def test_detection_with_all_axes(self):
        """Detection with non-zero x, y, z in camera frame."""
        # Camera rotated 90 deg about Z: cam +X -> world +Y, cam +Y -> world -X
        c = math.cos(math.radians(45))
        s = math.sin(math.radians(45))
        q = np.array([c, 0.0, 0.0, s])
        cam_pos = np.array([1.0, 0.0, 0.0])

        pos_cam = np.array([0.1, 0.2, 0.3])
        pos_w = cam_detection_to_world(pos_cam, cam_pos, q)
        R = _quat_to_rotmat(q)
        expected = R @ pos_cam + cam_pos
        assert_allclose(pos_w, expected, atol=1e-10)
        # Manual: R rotates +X->+Y, +Y->-X, +Z->+Z
        # R @ [0.1, 0.2, 0.3] = [-0.2, 0.1, 0.3]
        assert_allclose(pos_w, [1.0 - 0.2, 0.1, 0.3], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
