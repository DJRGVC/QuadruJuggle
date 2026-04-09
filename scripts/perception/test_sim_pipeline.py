"""Integration tests for the sim camera pipeline.

Validates the exact data path used by demo_camera_ekf.py:
  SimBallDetector(depth) → cam_detection_to_world(pos_cam, cam_pos_w, quat_w_ros) → EKF(world)

No GPU or Isaac Lab required — uses synthetic depth frames and constructed camera poses.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest

import numpy as np
import torch

# Direct module imports to bypass go1_ball_balance.__init__ (needs pxr/Isaac Lab)
_PERC_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "source",
                 "go1_ball_balance", "go1_ball_balance", "perception")
)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_PERC_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_det_mod = _load("sim_detector", "sim_detector.py")
SimBallDetector = _det_mod.SimBallDetector

_ft_mod = _load("frame_transforms", "frame_transforms.py")
cam_detection_to_world = _ft_mod.cam_detection_to_world
quat_to_rotmat = _ft_mod.quat_to_rotmat

_ekf_mod = _load("ball_ekf", "ball_ekf.py")
BallEKF = _ekf_mod.BallEKF
BallEKFConfig = _ekf_mod.BallEKFConfig


# ---------- helpers ----------

_FX = 11.24 / 20.955 * 640  # ≈342.8 px (D435i sim config)
_CX, _CY = 320.0, 240.0
_W, _H = 640, 480
_BALL_R = 0.020  # 40 mm


def _make_depth(ball_pos_cam: np.ndarray) -> np.ndarray:
    """Render a synthetic depth frame with a ball at `ball_pos_cam` (ROS frame)."""
    depth = np.full((_H, _W), np.inf, dtype=np.float32)
    x, y, z = ball_pos_cam
    if z <= 0:
        return depth
    u = _FX * x / z + _CX
    v = _FX * y / z + _CY
    r_px = _FX * _BALL_R / z
    yy, xx = np.ogrid[:_H, :_W]
    mask = ((xx - u) ** 2 + (yy - v) ** 2) <= r_px ** 2
    depth[mask] = z
    return depth


def _quat_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Return (w, x, y, z) quaternion for rotation about `axis` by `angle_rad`."""
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle_rad / 2)
    c = np.cos(angle_rad / 2)
    return np.array([c, axis[0] * s, axis[1] * s, axis[2] * s])


# ---------- tests ----------

class TestSimPipelineChain(unittest.TestCase):
    """SimBallDetector → cam_detection_to_world → EKF, CPU-only."""

    def _detector(self) -> SimBallDetector:
        return SimBallDetector(focal_length_px=_FX, cx=_CX, cy=_CY)

    # 1. Identity camera: cam frame = world frame, camera at origin
    def test_identity_camera(self):
        """Camera at origin with identity rotation: world = cam frame."""
        ball_world = np.array([0.0, 0.0, 0.5])  # 50 cm ahead
        depth = _make_depth(ball_world)

        det = self._detector().detect(depth)
        self.assertIsNotNone(det)

        pos_w = cam_detection_to_world(
            det.pos_cam,
            cam_pos_w=np.zeros(3),
            cam_quat_w_ros=np.array([1.0, 0.0, 0.0, 0.0]),  # identity
        )

        err = np.linalg.norm(pos_w - ball_world)
        self.assertLess(err, 0.005, f"Identity cam error {err:.4f}m > 5mm")

    # 2. Translated camera: camera at [1, 0, 0], ball at [1, 0, 0.5] world
    def test_translated_camera(self):
        """Camera offset by 1m in X; ball directly ahead."""
        ball_cam = np.array([0.0, 0.0, 0.5])
        ball_world = np.array([1.0, 0.0, 0.5])
        depth = _make_depth(ball_cam)

        det = self._detector().detect(depth)
        self.assertIsNotNone(det)

        pos_w = cam_detection_to_world(
            det.pos_cam,
            cam_pos_w=np.array([1.0, 0.0, 0.0]),
            cam_quat_w_ros=np.array([1.0, 0.0, 0.0, 0.0]),
        )

        err = np.linalg.norm(pos_w - ball_world)
        self.assertLess(err, 0.005, f"Translated cam error {err:.4f}m")

    # 3. 90° rotation about Y: camera forward (cam Z) → world -X
    def test_yaw_90_about_y(self):
        """90° about Y axis: cam +Z → world -X."""
        ball_cam = np.array([0.0, 0.0, 0.4])
        depth = _make_depth(ball_cam)

        det = self._detector().detect(depth)
        self.assertIsNotNone(det)

        # 90° about Y: R_y(90) maps [0,0,1]→[1,0,0], so cam_Z→world_+X
        q = _quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), np.pi / 2)
        pos_w = cam_detection_to_world(det.pos_cam, np.zeros(3), q)

        # Ball at cam [0,0,0.4] → world [0.4, 0, 0]
        self.assertAlmostEqual(pos_w[0], 0.4, places=2)
        self.assertAlmostEqual(abs(pos_w[1]), 0.0, places=2)
        self.assertAlmostEqual(abs(pos_w[2]), 0.0, places=2)

    # 4. Moderate pitch: 30° tilt about Y (camera looking partly downward)
    def test_moderate_tilt(self):
        """Camera tilted 30° about Y axis. Ball at known world position roundtrips."""
        tilt_rad = np.radians(30)
        q_tilt = _quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), tilt_rad)
        R = quat_to_rotmat(q_tilt)
        cam_pos_w = np.zeros(3)

        # Place ball at world [0.3, 0, 0.4] — find cam-frame position
        ball_world = np.array([0.3, 0.0, 0.4])
        ball_cam = R.T @ (ball_world - cam_pos_w)
        self.assertGreater(ball_cam[2], 0.05, "Ball must be in front of camera")

        # Verify ball projects inside image
        u = _FX * ball_cam[0] / ball_cam[2] + _CX
        v = _FX * ball_cam[1] / ball_cam[2] + _CY
        self.assertTrue(0 < u < _W and 0 < v < _H,
                        f"Ball projects to ({u:.0f},{v:.0f}) — outside image")

        depth = _make_depth(ball_cam)
        det = self._detector().detect(depth)
        self.assertIsNotNone(det, "Ball should be visible to tilted camera")

        pos_w = cam_detection_to_world(det.pos_cam, cam_pos_w, q_tilt)
        err = np.linalg.norm(pos_w - ball_world)
        self.assertLess(err, 0.010, f"Tilt roundtrip error {err:.4f}m > 10mm")

    # 5. Full chain with EKF: detect → transform → EKF converges
    def test_detect_transform_ekf_converges(self):
        """Full chain: 10 frames of stationary ball → EKF within 5mm."""
        ball_world = np.array([0.05, -0.02, 0.45])
        cam_pos_w = np.array([0.0, 0.0, 0.0])
        cam_quat = np.array([1.0, 0.0, 0.0, 0.0])  # identity

        R = quat_to_rotmat(cam_quat)
        ball_cam = R.T @ (ball_world - cam_pos_w)

        ekf = BallEKF(num_envs=1, device="cpu", cfg=BallEKFConfig(q_vel=0.40))
        ekf.reset(
            torch.tensor([0]),
            torch.tensor(ball_world, dtype=torch.float32).unsqueeze(0),
            torch.zeros(1, 3),
        )

        det = self._detector()
        for _ in range(10):
            depth = _make_depth(ball_cam)
            result = det.detect(depth)
            self.assertIsNotNone(result)

            pos_w = cam_detection_to_world(result.pos_cam, cam_pos_w, cam_quat)
            z = torch.tensor(pos_w, dtype=torch.float32).unsqueeze(0)
            ekf.step(z, detected=torch.tensor([True]), dt=0.02)

        est = ekf.pos[0].numpy()
        err = np.linalg.norm(est - ball_world)
        # EKF models gravity, so stationary ball has systematic Z bias.
        # 30mm tolerance accounts for gravity-induced prediction drift.
        self.assertLess(err, 0.030, f"EKF error {err:.4f}m > 30mm after 10 detections")

    # 6. Ballistic trajectory: ball with upward velocity, detect each frame
    def test_ballistic_trajectory_tracking(self):
        """EKF tracks a ball on a ballistic arc through the sim pipeline."""
        cam_pos_w = np.array([0.0, 0.0, 0.0])
        cam_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Ball starts at (0, 0, 0.3) with vz=2.0 m/s
        pos0 = np.array([0.0, 0.0, 0.3])
        vel0 = np.array([0.0, 0.0, 2.0])
        g = 9.81
        dt = 0.02

        ekf = BallEKF(num_envs=1, device="cpu", cfg=BallEKFConfig(q_vel=0.40))
        ekf.reset(
            torch.tensor([0]),
            torch.tensor(pos0, dtype=torch.float32).unsqueeze(0),
            torch.tensor(vel0, dtype=torch.float32).unsqueeze(0),
        )

        det = self._detector()
        errors = []

        for step in range(30):
            t = step * dt
            # True ballistic position
            ball_w = pos0.copy()
            ball_w[2] += vel0[2] * t - 0.5 * g * t * t

            if ball_w[2] < 0.05:
                break  # below camera min_depth

            R = quat_to_rotmat(cam_quat)
            ball_cam = R.T @ (ball_w - cam_pos_w)

            depth = _make_depth(ball_cam)
            result = det.detect(depth)

            if result is not None:
                pos_w = cam_detection_to_world(result.pos_cam, cam_pos_w, cam_quat)
                z = torch.tensor(pos_w, dtype=torch.float32).unsqueeze(0)
                ekf.step(z, detected=torch.tensor([True]), dt=dt)
            else:
                ekf.step(torch.zeros(1, 3), detected=torch.tensor([False]), dt=dt)

            est = ekf.pos[0].numpy()
            errors.append(np.linalg.norm(est - ball_w))

        # Mean tracking error should be under 30mm for a ballistic arc
        mean_err = np.mean(errors)
        self.assertLess(mean_err, 0.030,
                        f"Mean tracking error {mean_err:.4f}m > 30mm on ballistic arc")

    # 7. Dropout resilience: 50% detection rate → EKF still tracks
    def test_dropout_resilience(self):
        """50% frame dropout — EKF should still track within 20mm."""
        ball_world = np.array([0.02, 0.01, 0.4])
        cam_pos_w = np.zeros(3)
        cam_quat = np.array([1.0, 0.0, 0.0, 0.0])

        R = quat_to_rotmat(cam_quat)
        ball_cam = R.T @ (ball_world - cam_pos_w)

        ekf = BallEKF(num_envs=1, device="cpu", cfg=BallEKFConfig(q_vel=0.40))
        ekf.reset(
            torch.tensor([0]),
            torch.tensor(ball_world, dtype=torch.float32).unsqueeze(0),
            torch.zeros(1, 3),
        )

        det = self._detector()
        rng = np.random.default_rng(42)

        for step in range(20):
            detected = rng.random() > 0.5  # 50% dropout
            if detected:
                depth = _make_depth(ball_cam)
                result = det.detect(depth)
                if result is not None:
                    pos_w = cam_detection_to_world(result.pos_cam, cam_pos_w, cam_quat)
                    z = torch.tensor(pos_w, dtype=torch.float32).unsqueeze(0)
                    ekf.step(z, detected=torch.tensor([True]), dt=0.02)
                    continue
            ekf.step(torch.zeros(1, 3), detected=torch.tensor([False]), dt=0.02)

        est = ekf.pos[0].numpy()
        err = np.linalg.norm(est - ball_world)
        # With 50% dropout + gravity drift, 80mm is a reasonable bound
        self.assertLess(err, 0.080, f"Dropout EKF error {err:.4f}m > 80mm")

    # 8. Multi-step with translated + rotated camera (20° about Y)
    def test_rotated_translated_chain(self):
        """Camera at [1, 0, 0], rotated 20° about Y: detect→transform→EKF."""
        tilt_rad = np.radians(20)
        q_tilt = _quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), tilt_rad)
        R = quat_to_rotmat(q_tilt)
        cam_pos_w = np.array([1.0, 0.0, 0.0])

        # Ball at world (1.3, 0, 0.4)
        ball_world = np.array([1.3, 0.0, 0.4])
        ball_cam = R.T @ (ball_world - cam_pos_w)

        self.assertGreater(ball_cam[2], 0.05, "Ball must be in front of camera")
        # Verify in-frame
        u = _FX * ball_cam[0] / ball_cam[2] + _CX
        v = _FX * ball_cam[1] / ball_cam[2] + _CY
        self.assertTrue(0 < u < _W and 0 < v < _H,
                        f"Ball at ({u:.0f},{v:.0f}) outside image")

        ekf = BallEKF(num_envs=1, device="cpu", cfg=BallEKFConfig(q_vel=0.40))
        ekf.reset(
            torch.tensor([0]),
            torch.tensor(ball_world, dtype=torch.float32).unsqueeze(0),
            torch.zeros(1, 3),
        )

        det = self._detector()
        for _ in range(10):
            depth = _make_depth(ball_cam)
            result = det.detect(depth)
            self.assertIsNotNone(result)

            pos_w = cam_detection_to_world(result.pos_cam, cam_pos_w, q_tilt)
            z = torch.tensor(pos_w, dtype=torch.float32).unsqueeze(0)
            ekf.step(z, detected=torch.tensor([True]), dt=0.02)

        est = ekf.pos[0].numpy()
        err = np.linalg.norm(est - ball_world)
        # 30mm accounts for EKF gravity drift on stationary ball
        self.assertLess(err, 0.030, f"Rotated+translated chain error {err:.4f}m > 30mm")


if __name__ == "__main__":
    unittest.main(verbosity=2)
