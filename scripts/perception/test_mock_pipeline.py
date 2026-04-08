"""Integration tests for MockCamera + MockDetector + calibration + EKF chain.

Validates the full real pipeline path without hardware:
  MockCamera → MockDetector → CameraExtrinsics → EKF → body-frame output

Run: python scripts/perception/test_mock_pipeline.py
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch

# Direct imports — bypass go1_ball_balance.__init__ which pulls Isaac Lab / pxr
# Direct imports — bypass perception/__init__ and go1_ball_balance/__init__
# which pull in Isaac Lab (pxr) via ball_obs_spec.py.
import importlib.util

_PERC_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "source", "go1_ball_balance", "go1_ball_balance", "perception",
))


def _load_module(name: str, path: str):
    """Import a single .py file as a module, avoiding __init__ chains."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load only the modules we need (no Isaac Lab dependency)
_camera_mod = _load_module(
    "perception.real.camera",
    os.path.join(_PERC_DIR, "real", "camera.py"),
)
CameraIntrinsics = _camera_mod.CameraIntrinsics

_calib_mod = _load_module(
    "perception.real.calibration",
    os.path.join(_PERC_DIR, "real", "calibration.py"),
)
CameraCalibrator = _calib_mod.CameraCalibrator
CameraExtrinsics = _calib_mod.CameraExtrinsics

_detector_mod = _load_module(
    "perception.real.detector",
    os.path.join(_PERC_DIR, "real", "detector.py"),
)

_mock_mod = _load_module(
    "perception.real.mock",
    os.path.join(_PERC_DIR, "real", "mock.py"),
)
MockCamera = _mock_mod.MockCamera
MockCameraConfig = _mock_mod.MockCameraConfig
MockDetector = _mock_mod.MockDetector

_ekf_mod = _load_module(
    "perception.ball_ekf",
    os.path.join(_PERC_DIR, "ball_ekf.py"),
)
BallEKF = _ekf_mod.BallEKF
BallEKFConfig = _ekf_mod.BallEKFConfig


class TestMockCamera(unittest.TestCase):
    """MockCamera generates correct depth frames."""

    def test_start_stop(self):
        cam = MockCamera()
        cam.start()
        self.assertIsNotNone(cam.get_frame())
        cam.stop()

    def test_empty_scene(self):
        """No ball → all-zero depth frame."""
        cam = MockCamera(MockCameraConfig(background_depth_mm=0))
        cam.start()
        depth, ts = cam.get_frame()
        self.assertEqual(depth.shape, (480, 848))
        self.assertEqual(depth.dtype, np.uint16)
        np.testing.assert_array_equal(depth, 0)

    def test_ball_at_known_position(self):
        """Ball at 0.5m depth produces non-zero pixels at correct location."""
        cam = MockCamera(MockCameraConfig(depth_noise_std_mm=0.0))
        cam.start()
        cam.set_ball_pos_cam(np.array([0.0, 0.0, 0.5]))
        depth, ts = cam.get_frame()

        intr = cam.get_intrinsics()
        # Ball should be at principal point (cx, cy)
        u_c, v_c = int(intr.cx), int(intr.cy)

        # Centre pixel should have ball depth (500mm)
        self.assertAlmostEqual(depth[v_c, u_c], 500, delta=1)

        # Non-zero pixels should form a cluster around centre
        nonzero = np.argwhere(depth > 0)
        self.assertGreater(len(nonzero), 5)

        # Centroid of nonzero pixels should be near principal point
        centroid_v = np.mean(nonzero[:, 0])
        centroid_u = np.mean(nonzero[:, 1])
        self.assertAlmostEqual(centroid_u, intr.cx, delta=2.0)
        self.assertAlmostEqual(centroid_v, intr.cy, delta=2.0)

    def test_ball_off_centre(self):
        """Ball offset in X projects to correct pixel column."""
        cam = MockCamera(MockCameraConfig(depth_noise_std_mm=0.0))
        cam.start()
        # Ball at X=0.1m, Z=0.5m → u = fx * 0.1/0.5 + cx = 425*0.2 + 424 = 509
        cam.set_ball_pos_cam(np.array([0.1, 0.0, 0.5]))
        depth, _ = cam.get_frame()
        nonzero = np.argwhere(depth > 0)
        centroid_u = np.mean(nonzero[:, 1])
        expected_u = 425.0 * 0.1 / 0.5 + 424.0  # = 509
        self.assertAlmostEqual(centroid_u, expected_u, delta=2.0)

    def test_clear_ball(self):
        """clear_ball removes the ball from subsequent frames."""
        cam = MockCamera(MockCameraConfig(depth_noise_std_mm=0.0))
        cam.start()
        cam.set_ball_pos_cam(np.array([0.0, 0.0, 0.5]))
        depth1, _ = cam.get_frame()
        self.assertTrue(np.any(depth1 > 0))

        cam.clear_ball()
        depth2, _ = cam.get_frame()
        np.testing.assert_array_equal(depth2, 0)

    def test_frame_counter(self):
        cam = MockCamera()
        cam.start()
        self.assertEqual(cam.frame_count, 0)
        cam.get_frame()
        cam.get_frame()
        self.assertEqual(cam.frame_count, 2)

    def test_not_started_returns_none(self):
        cam = MockCamera()
        self.assertIsNone(cam.get_frame())


class TestMockDetector(unittest.TestCase):
    """MockDetector finds ball blob in synthetic depth frames."""

    def _make_frame_with_ball(self, pos_cam, noise=False):
        """Helper: generate a depth frame + intrinsics with ball at pos_cam."""
        cfg = MockCameraConfig(depth_noise_std_mm=1.0 if noise else 0.0)
        cam = MockCamera(cfg)
        cam.start()
        cam.set_ball_pos_cam(pos_cam)
        depth, _ = cam.get_frame()
        intr = cam.get_intrinsics()
        return depth, intr

    def test_detect_ball(self):
        """Detector finds ball at known position with low error."""
        pos = np.array([0.0, 0.0, 0.5])
        depth, intr = self._make_frame_with_ball(pos, noise=False)
        det = MockDetector(pos_noise_std=0.0)
        result = det.detect(depth, intr)

        self.assertIsNotNone(result)
        self.assertEqual(result.method, "mock")
        self.assertGreater(result.confidence, 0.5)

        # Position should match within a few mm (pixel quantisation)
        err = np.linalg.norm(result.pos_cam - pos)
        self.assertLess(err, 0.005, f"Detection error {err:.4f}m > 5mm")

    def test_detect_off_centre(self):
        """Detector finds ball at off-centre position."""
        pos = np.array([0.05, -0.03, 0.4])
        depth, intr = self._make_frame_with_ball(pos, noise=False)
        det = MockDetector(pos_noise_std=0.0)
        result = det.detect(depth, intr)

        self.assertIsNotNone(result)
        err = np.linalg.norm(result.pos_cam - pos)
        self.assertLess(err, 0.010, f"Detection error {err:.4f}m > 10mm")

    def test_dropout(self):
        """100% dropout always returns None."""
        pos = np.array([0.0, 0.0, 0.5])
        depth, intr = self._make_frame_with_ball(pos)
        det = MockDetector(dropout_rate=1.0)
        self.assertIsNone(det.detect(depth, intr))

    def test_empty_frame(self):
        """No ball → no detection (not a false positive)."""
        depth = np.zeros((480, 848), dtype=np.uint16)
        intr = CameraIntrinsics(fx=425, fy=425, cx=424, cy=240, width=848, height=480)
        det = MockDetector()
        self.assertIsNone(det.detect(depth, intr))


class TestFullChain(unittest.TestCase):
    """End-to-end: MockCamera → MockDetector → Calibration → EKF → body frame."""

    def test_stationary_ball_identity_calibration(self):
        """Ball at 50cm, identity extrinsics: EKF converges to true position."""
        # Setup mock camera + detector
        ball_pos_cam = np.array([0.0, 0.0, 0.5])
        cam = MockCamera(MockCameraConfig(depth_noise_std_mm=0.0))
        cam.start()
        cam.set_ball_pos_cam(ball_pos_cam)

        det = MockDetector(pos_noise_std=0.001)

        # Identity extrinsics (camera frame = body frame)
        extr = CameraExtrinsics(
            R_cam_body=np.eye(3),
            t_cam_body=np.zeros(3),
        )

        # Single-env EKF on CPU
        ekf = BallEKF(num_envs=1, device="cpu", cfg=BallEKFConfig(q_vel=7.0))
        init_pos = torch.tensor([[0.0, 0.0, 0.5]])
        init_vel = torch.zeros(1, 3)
        ekf.reset(torch.tensor([0]), init_pos, init_vel)

        # Run 10 steps
        dt = 0.02  # 50 Hz
        for _ in range(10):
            depth, ts = cam.get_frame()
            detection = det.detect(depth, cam.get_intrinsics())
            self.assertIsNotNone(detection)

            # Transform camera → body
            pos_body = extr.transform_to_body(detection.pos_cam)

            # EKF update
            z = torch.tensor(pos_body, dtype=torch.float32).unsqueeze(0)
            ekf.step(z, detected=torch.tensor([True]), dt=dt)

        # Check convergence
        est_pos = ekf.pos[0].numpy()
        err = np.linalg.norm(est_pos - ball_pos_cam)
        self.assertLess(err, 0.010, f"EKF position error {err:.4f}m > 10mm")

    def test_known_mount_calibration(self):
        """Ball in camera frame → body frame via known mount transform."""
        # Camera mounted at body position [0, 0, 0.07] with 45° pitch up
        extr = CameraCalibrator.from_known_mount(
            mount_position_body=np.array([0.0, 0.0, 0.07]),
            mount_orientation_rpy=np.array([0.0, np.pi / 4, 0.0]),
        )

        # Ball at 50cm along camera Z axis (straight ahead in camera frame)
        ball_cam = np.array([0.0, 0.0, 0.5])
        ball_body = extr.transform_to_body(ball_cam)

        # With 45° pitch, camera Z maps to body X*cos45 + Z*sin45
        expected_x = 0.5 * np.cos(np.pi / 4)
        expected_z = 0.07 + 0.5 * np.sin(np.pi / 4)
        self.assertAlmostEqual(ball_body[0], expected_x, places=3)
        self.assertAlmostEqual(ball_body[2], expected_z, places=3)

    def test_moving_ball_velocity_tracking(self):
        """EKF tracks velocity of a ball moving through the camera FOV."""
        cam = MockCamera(MockCameraConfig(depth_noise_std_mm=0.0))
        cam.start()

        det = MockDetector(pos_noise_std=0.001)
        extr = CameraExtrinsics(R_cam_body=np.eye(3), t_cam_body=np.zeros(3))

        ekf = BallEKF(num_envs=1, device="cpu", cfg=BallEKFConfig(q_vel=7.0))
        init_pos = torch.tensor([[0.0, 0.0, 0.5]])
        ekf.reset(torch.tensor([0]), init_pos, torch.zeros(1, 3))

        dt = 0.02
        vx_true = 0.5  # m/s rightward

        for step in range(20):
            t = step * dt
            ball_x = vx_true * t
            cam.set_ball_pos_cam(np.array([ball_x, 0.0, 0.5]))

            depth, ts = cam.get_frame()
            detection = det.detect(depth, cam.get_intrinsics())
            if detection is None:
                continue

            pos_body = extr.transform_to_body(detection.pos_cam)
            z = torch.tensor(pos_body, dtype=torch.float32).unsqueeze(0)
            ekf.step(z, detected=torch.tensor([True]), dt=dt)

        # After 20 steps the EKF should have a reasonable velocity estimate
        est_vel = ekf.vel[0].numpy()
        # Velocity X should be close to 0.5 m/s (allow 30% tolerance —
        # EKF with q_vel=7.0 is noisy but should track the trend)
        self.assertGreater(est_vel[0], 0.2, f"vx={est_vel[0]:.2f} too low")
        self.assertLess(est_vel[0], 0.8, f"vx={est_vel[0]:.2f} too high")

    def test_dropout_handling(self):
        """EKF predict-through-dropout produces degraded but bounded estimate."""
        cam = MockCamera(MockCameraConfig(depth_noise_std_mm=0.0))
        cam.start()
        cam.set_ball_pos_cam(np.array([0.0, 0.0, 0.5]))

        det_good = MockDetector(pos_noise_std=0.001, dropout_rate=0.0)
        det_bad = MockDetector(pos_noise_std=0.001, dropout_rate=1.0)

        extr = CameraExtrinsics(R_cam_body=np.eye(3), t_cam_body=np.zeros(3))
        ekf = BallEKF(num_envs=1, device="cpu", cfg=BallEKFConfig(q_vel=7.0))
        ekf.reset(
            torch.tensor([0]),
            torch.tensor([[0.0, 0.0, 0.5]]),
            torch.zeros(1, 3),
        )

        dt = 0.02

        # 5 good measurements to initialise
        for _ in range(5):
            depth, _ = cam.get_frame()
            detection = det_good.detect(depth, cam.get_intrinsics())
            pos_body = extr.transform_to_body(detection.pos_cam)
            z = torch.tensor(pos_body, dtype=torch.float32).unsqueeze(0)
            ekf.step(z, detected=torch.tensor([True]), dt=dt)

        pos_before = ekf.pos[0].numpy().copy()

        # 5 dropout steps (predict only)
        for _ in range(5):
            ekf.step(
                torch.zeros(1, 3),
                detected=torch.tensor([False]),
                dt=dt,
            )

        pos_after = ekf.pos[0].numpy()

        # Position should drift due to gravity (z decreases) but not explode
        drift = np.linalg.norm(pos_after - pos_before)
        self.assertLess(drift, 0.1, f"Drift {drift:.4f}m during 100ms dropout — too large")
        # Z should decrease (gravity)
        self.assertLessEqual(pos_after[2], pos_before[2] + 0.001)


if __name__ == "__main__":
    unittest.main(verbosity=2)
