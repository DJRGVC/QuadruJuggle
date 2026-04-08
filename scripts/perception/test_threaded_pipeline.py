"""Integration tests for threaded RealPerceptionPipeline.

Tests the full camera->detector->EKF->body-frame pipeline using
MockCamera + MockDetector, validating threading, timing, EKF convergence,
dropout handling, and body-frame transformation.

Run: uv run --active python scripts/perception/test_threaded_pipeline.py -v
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
import unittest

import numpy as np

# ── Direct module loading (bypasses Isaac Lab import chain) ──────────────

_PERC_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "source", "go1_ball_balance", "go1_ball_balance", "perception",
))


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_camera_mod = _load_module(
    "perception.real.camera",
    os.path.join(_PERC_DIR, "real", "camera.py"),
)
CameraIntrinsics = _camera_mod.CameraIntrinsics

_calib_mod = _load_module(
    "perception.real.calibration",
    os.path.join(_PERC_DIR, "real", "calibration.py"),
)
CameraExtrinsics = _calib_mod.CameraExtrinsics

_config_mod = _load_module(
    "perception.real.config",
    os.path.join(_PERC_DIR, "real", "config.py"),
)
HardwarePipelineConfig = _config_mod.HardwarePipelineConfig

_detector_mod = _load_module(
    "perception.real.detector",
    os.path.join(_PERC_DIR, "real", "detector.py"),
)
BallDetector = _detector_mod.BallDetector
Detection = _detector_mod.Detection

_ekf_mod = _load_module(
    "perception.ball_ekf",
    os.path.join(_PERC_DIR, "ball_ekf.py"),
)
BallEKFConfig = _ekf_mod.BallEKFConfig
BallEKF = _ekf_mod.BallEKF

# Mock module depends on camera + detector — patch sys.modules first
# so relative imports work
sys.modules["go1_ball_balance.perception.real.camera"] = _camera_mod
sys.modules["go1_ball_balance.perception.real.detector"] = _detector_mod
sys.modules["go1_ball_balance.perception.real.calibration"] = _calib_mod
sys.modules["go1_ball_balance.perception.real.config"] = _config_mod
sys.modules["go1_ball_balance.perception.ball_ekf"] = _ekf_mod

_mock_mod = _load_module(
    "perception.real.mock",
    os.path.join(_PERC_DIR, "real", "mock.py"),
)
MockCamera = _mock_mod.MockCamera
MockCameraConfig = _mock_mod.MockCameraConfig
MockDetector = _mock_mod.MockDetector

_pipeline_mod = _load_module(
    "perception.real.pipeline",
    os.path.join(_PERC_DIR, "real", "pipeline.py"),
)
RealPerceptionPipeline = _pipeline_mod.RealPerceptionPipeline
PipelineObservation = _pipeline_mod.PipelineObservation
_quat_to_rotmat = _pipeline_mod._quat_to_rotmat

# Check cv2
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


# ── Helpers ──────────────────────────────────────────────────────────────

def _identity_extrinsics() -> CameraExtrinsics:
    """Camera = body frame (no rotation, no offset)."""
    return CameraExtrinsics(
        R_cam_body=np.eye(3),
        t_cam_body=np.zeros(3),
    )


def _identity_quat() -> np.ndarray:
    """Identity quaternion [w, x, y, z]."""
    return np.array([1.0, 0.0, 0.0, 0.0])


# ── Tests ────────────────────────────────────────────────────────────────

class TestQuatToRotmat(unittest.TestCase):
    """Test the quaternion-to-rotation-matrix utility."""

    def test_identity(self):
        R = _quat_to_rotmat(np.array([1.0, 0, 0, 0]))
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_90deg_z(self):
        q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        R = _quat_to_rotmat(q)
        np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-10)
        np.testing.assert_allclose(R @ [0, 1, 0], [-1, 0, 0], atol=1e-10)

    def test_orthogonal(self):
        q = np.array([0.5, 0.5, 0.5, 0.5])
        R = _quat_to_rotmat(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)


class TestPipelineStartStop(unittest.TestCase):
    """Test pipeline lifecycle (start/stop/restart)."""

    def _make_pipeline(self) -> RealPerceptionPipeline:
        config = HardwarePipelineConfig()
        camera = MockCamera()
        detector = MockDetector()
        return RealPerceptionPipeline(
            config, camera=camera, detector=detector,
            extrinsics=_identity_extrinsics(),
        )

    def test_start_stop(self):
        p = self._make_pipeline()
        self.assertFalse(p.is_running)
        p.start()
        self.assertTrue(p.is_running)
        p.stop()
        self.assertFalse(p.is_running)

    def test_double_start_raises(self):
        p = self._make_pipeline()
        p.start()
        with self.assertRaises(RuntimeError):
            p.start()
        p.stop()

    def test_get_observation_before_start_raises(self):
        p = self._make_pipeline()
        with self.assertRaises(RuntimeError):
            p.get_observation(_identity_quat(), np.zeros(3))

    def test_restart(self):
        p = self._make_pipeline()
        p.start()
        p.stop()
        p.start()
        self.assertTrue(p.is_running)
        p.stop()


class TestPipelineDetection(unittest.TestCase):
    """Test that the pipeline detects and tracks a stationary ball."""

    def test_stationary_ball_convergence(self):
        """EKF should converge to true ball position within a few frames."""
        config = HardwarePipelineConfig()
        cam = MockCamera()
        det = MockDetector(pos_noise_std=0.001)
        cam.set_ball_pos_cam(np.array([0.0, 0.0, 0.5]))

        p = RealPerceptionPipeline(
            config, camera=cam, detector=det,
            extrinsics=_identity_extrinsics(),
        )
        p.start()
        time.sleep(0.1)

        obs = None
        for _ in range(20):
            obs = p.get_observation(_identity_quat(), np.zeros(3))
            time.sleep(0.01)

        p.stop()

        self.assertIsNotNone(obs)
        np.testing.assert_allclose(obs.ball_pos_b, [0, 0, 0.5], atol=0.05)

    def test_stats_populated(self):
        config = HardwarePipelineConfig()
        cam = MockCamera()
        det = MockDetector(dropout_rate=0.0)
        cam.set_ball_pos_cam(np.array([0.0, 0.0, 0.5]))

        p = RealPerceptionPipeline(
            config, camera=cam, detector=det,
            extrinsics=_identity_extrinsics(),
        )
        p.start()
        time.sleep(0.05)
        p.get_observation(_identity_quat(), np.zeros(3))
        time.sleep(0.05)
        p.stop()

        stats = p.stats
        self.assertGreater(stats["total_frames"], 0)
        self.assertGreater(stats["total_detections"], 0)
        self.assertGreater(stats["detection_rate"], 0.0)


class TestPipelineDropout(unittest.TestCase):
    """Test ball_lost flag under dropout conditions."""

    def test_full_dropout_triggers_ball_lost(self):
        config = HardwarePipelineConfig(lost_ball_frames=3)
        cam = MockCamera()
        det = MockDetector(dropout_rate=1.0)
        cam.set_ball_pos_cam(np.array([0.0, 0.0, 0.5]))

        p = RealPerceptionPipeline(
            config, camera=cam, detector=det,
            extrinsics=_identity_extrinsics(),
        )
        p.start()
        time.sleep(0.05)

        obs = None
        for _ in range(10):
            obs = p.get_observation(_identity_quat(), np.zeros(3))
            time.sleep(0.01)

        p.stop()
        self.assertTrue(obs.ball_lost)
        self.assertFalse(obs.detected)

    def test_no_ball_in_scene(self):
        config = HardwarePipelineConfig(lost_ball_frames=3)
        cam = MockCamera()
        det = MockDetector()
        # No ball set -> empty frame

        p = RealPerceptionPipeline(
            config, camera=cam, detector=det,
            extrinsics=_identity_extrinsics(),
        )
        p.start()
        time.sleep(0.05)

        obs = None
        for _ in range(10):
            obs = p.get_observation(_identity_quat(), np.zeros(3))
            time.sleep(0.01)

        p.stop()
        self.assertTrue(obs.ball_lost)


class TestPipelineBodyFrame(unittest.TestCase):
    """Test body-frame transformation of EKF output."""

    def test_rotated_robot(self):
        config = HardwarePipelineConfig()
        cam = MockCamera()
        det = MockDetector(pos_noise_std=0.001)
        # Ball at [0.1, 0, 0.5] in camera frame — needs valid depth (Z>0.168)
        cam.set_ball_pos_cam(np.array([0.1, 0.0, 0.5]))

        p = RealPerceptionPipeline(
            config, camera=cam, detector=det,
            extrinsics=_identity_extrinsics(),
        )
        p.start()
        time.sleep(0.1)

        q_90z = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])

        obs = None
        for _ in range(20):
            obs = p.get_observation(q_90z, np.zeros(3))
            time.sleep(0.01)

        p.stop()

        # cam=body (identity extr), ball_cam=[0.1,0,0.5]
        # R_body_world(90° Z) maps body [1,0,0] -> world [0,1,0]
        # ball_world = R_body_world @ [0.1,0,0.5] = [-0, 0.1, 0.5] (approx)
        # ball_body = R_world_body @ ball_world = [0.1, 0, 0.5]
        np.testing.assert_allclose(obs.ball_pos_b, [0.1, 0, 0.5], atol=0.05)

    def test_robot_offset(self):
        config = HardwarePipelineConfig()
        cam = MockCamera()
        det = MockDetector(pos_noise_std=0.001)
        cam.set_ball_pos_cam(np.array([0.0, 0.0, 0.5]))

        p = RealPerceptionPipeline(
            config, camera=cam, detector=det,
            extrinsics=_identity_extrinsics(),
        )
        p.start()
        time.sleep(0.1)

        obs = None
        for _ in range(20):
            obs = p.get_observation(_identity_quat(), np.array([1.0, 0, 0]))
            time.sleep(0.01)

        p.stop()

        # ball_world = [0,0,0.5] + [1,0,0] = [1,0,0.5]
        # ball_body = R_I @ ([1,0,0.5] - [1,0,0]) = [0,0,0.5]
        np.testing.assert_allclose(obs.ball_pos_b, [0, 0, 0.5], atol=0.05)


class TestPipelineExtrinsics(unittest.TestCase):
    """Test non-identity camera extrinsics."""

    def test_camera_offset(self):
        config = HardwarePipelineConfig()
        cam = MockCamera()
        det = MockDetector(pos_noise_std=0.001)
        cam.set_ball_pos_cam(np.array([0.0, 0.0, 0.3]))

        extr = CameraExtrinsics(
            R_cam_body=np.eye(3),
            t_cam_body=np.array([0.0, 0.0, 0.1]),
        )

        p = RealPerceptionPipeline(
            config, camera=cam, detector=det, extrinsics=extr,
        )
        p.start()
        time.sleep(0.1)

        obs = None
        for _ in range(20):
            obs = p.get_observation(_identity_quat(), np.zeros(3))
            time.sleep(0.01)

        p.stop()

        # ball_cam=[0,0,0.3], R=I, t=[0,0,0.1]
        # ball_body = I @ [0,0,0.3] + [0,0,0.1] = [0,0,0.4]
        np.testing.assert_allclose(obs.ball_pos_b, [0, 0, 0.4], atol=0.05)


class TestPipelineEKFReset(unittest.TestCase):
    """Test EKF reset during pipeline operation."""

    def test_reset_ekf_converges_back(self):
        # Use higher process noise for faster convergence after reset
        ekf_cfg = BallEKFConfig(q_vel=5.0, contact_aware=False)
        config = HardwarePipelineConfig()
        cam = MockCamera()
        det = MockDetector(pos_noise_std=0.001)
        cam.set_ball_pos_cam(np.array([0.0, 0.0, 0.5]))

        p = RealPerceptionPipeline(
            config, camera=cam, detector=det,
            extrinsics=_identity_extrinsics(),
            ekf_config=ekf_cfg,
        )
        p.start()
        time.sleep(0.05)

        for _ in range(5):
            p.get_observation(_identity_quat(), np.zeros(3))
            time.sleep(0.01)

        # Reset EKF to a wrong position
        p.reset_ekf(init_pos=np.array([0.0, 0.0, 0.2]))

        # After more updates, should converge back to [0,0,0.5]
        obs = None
        for _ in range(50):
            obs = p.get_observation(_identity_quat(), np.zeros(3))
            time.sleep(0.01)

        p.stop()
        np.testing.assert_allclose(obs.ball_pos_b, [0, 0, 0.5], atol=0.10)


class TestPipelineObservationFields(unittest.TestCase):
    """Test PipelineObservation data structure."""

    def test_all_fields_populated(self):
        config = HardwarePipelineConfig()
        cam = MockCamera()
        det = MockDetector(pos_noise_std=0.001)
        cam.set_ball_pos_cam(np.array([0.0, 0.0, 0.5]))

        p = RealPerceptionPipeline(
            config, camera=cam, detector=det,
            extrinsics=_identity_extrinsics(),
        )
        p.start()
        time.sleep(0.05)

        obs = p.get_observation(_identity_quat(), np.zeros(3))
        p.stop()

        self.assertEqual(obs.ball_pos_b.shape, (3,))
        self.assertEqual(obs.ball_vel_b.shape, (3,))
        self.assertIsInstance(obs.ball_lost, bool)
        self.assertIsInstance(obs.detected, bool)
        self.assertGreater(obs.timestamp, 0.0)
        self.assertEqual(obs.ekf_pos_w.shape, (3,))
        self.assertEqual(obs.ekf_vel_w.shape, (3,))


@unittest.skipUnless(_HAS_CV2, "cv2 not installed")
class TestPipelineWithHoughDetector(unittest.TestCase):
    """Test pipeline using real BallDetector (Hough fallback) on MockCamera frames."""

    def test_hough_on_mock_frames(self):
        config = HardwarePipelineConfig()
        cam = MockCamera(MockCameraConfig(depth_noise_std_mm=1.0))
        det = BallDetector(model_path=None, hough_fallback=True)
        cam.set_ball_pos_cam(np.array([0.0, 0.0, 0.5]))

        p = RealPerceptionPipeline(
            config, camera=cam, detector=det,
            extrinsics=_identity_extrinsics(),
        )
        p.start()
        time.sleep(0.15)

        obs = None
        for _ in range(20):
            obs = p.get_observation(_identity_quat(), np.zeros(3))
            time.sleep(0.01)

        p.stop()
        np.testing.assert_allclose(obs.ball_pos_b, [0, 0, 0.5], atol=0.10)


if __name__ == "__main__":
    unittest.main()
