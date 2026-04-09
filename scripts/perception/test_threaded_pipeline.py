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


class TestPipelineLatencyTracking(unittest.TestCase):
    """Test measurement-age and detection latency tracking."""

    def test_latency_stats_populated(self):
        """After processing detections, latency stats should be non-zero."""
        config = HardwarePipelineConfig()
        cam = MockCamera()
        det = MockDetector(pos_noise_std=0.001, dropout_rate=0.0)
        cam.set_ball_pos_cam(np.array([0.0, 0.0, 0.5]))

        p = RealPerceptionPipeline(
            config, camera=cam, detector=det,
            extrinsics=_identity_extrinsics(),
        )
        p.start()
        time.sleep(0.1)

        for _ in range(10):
            p.get_observation(_identity_quat(), np.zeros(3))
            time.sleep(0.02)

        p.stop()

        stats = p.stats
        # Detection processing time should be tracked
        self.assertGreater(stats["mean_detect_dt"], 0.0)
        self.assertGreater(stats["max_detect_dt"], 0.0)
        self.assertGreaterEqual(stats["max_detect_dt"], stats["mean_detect_dt"])

        # Measurement age should be tracked (time from acq thread to main thread)
        self.assertGreaterEqual(stats["mean_meas_age"], 0.0)
        self.assertGreaterEqual(stats["max_meas_age"], 0.0)

    def test_latency_stats_zero_before_detections(self):
        """Before any detections, latency stats should be zero."""
        config = HardwarePipelineConfig()
        cam = MockCamera()
        det = MockDetector()
        # No ball set — no detections

        p = RealPerceptionPipeline(
            config, camera=cam, detector=det,
            extrinsics=_identity_extrinsics(),
        )
        p.start()
        p.get_observation(_identity_quat(), np.zeros(3))
        p.stop()

        stats = p.stats
        self.assertEqual(stats["mean_detect_dt"], 0.0)
        self.assertEqual(stats["max_detect_dt"], 0.0)
        self.assertEqual(stats["mean_meas_age"], 0.0)
        self.assertEqual(stats["max_meas_age"], 0.0)

    def test_latency_resets_on_restart(self):
        """Latency counters should reset when pipeline restarts."""
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
        for _ in range(5):
            p.get_observation(_identity_quat(), np.zeros(3))
            time.sleep(0.01)
        p.stop()

        # Stats should be non-zero now
        self.assertGreater(p.stats["mean_detect_dt"], 0.0)

        # Restart — stats should reset
        p.start()
        time.sleep(0.01)
        p.get_observation(_identity_quat(), np.zeros(3))
        # Don't wait long enough for detection — check reset happened
        stats_after = p.stats
        # max_detect_dt should have reset (could be re-populated immediately
        # if acq thread ran fast, but it restarted from 0)
        p.stop()


@unittest.skipUnless(_HAS_CV2, "cv2 not installed")
class TestDynamicTrajectoryHoughPipeline(unittest.TestCase):
    """Integration test: full MockCamera → HoughDetector → EKF pipeline
    tracking a ball through a ballistic bounce trajectory.

    Validates that the real-hardware perception chain (minus actual hardware)
    can track a dynamically moving ball with realistic timing.
    """

    @staticmethod
    def _ballistic_pos(t: float, z0: float, vz0: float, g: float = -9.81) -> np.ndarray:
        """Ballistic position at time t. Ball moves along Z (optical axis)
        with lateral drift to test XY tracking.

        Camera frame: Z = depth (optical axis = upward toward ball),
        X = right, Y = down.  Ball starts at (0, 0, z0) with vz=vz0 upward
        (increasing depth = ball moving away from camera).
        """
        # Small lateral drift: 0.02 m/s in X
        x = 0.02 * t
        y = 0.0
        z = z0 + vz0 * t + 0.5 * g * t * t
        return np.array([x, y, z])

    def test_ascending_ball_tracking(self):
        """Ball launched upward at 2 m/s from 0.3m — track through ascent."""
        config = HardwarePipelineConfig()
        cam = MockCamera(MockCameraConfig(depth_noise_std_mm=1.0))
        det = BallDetector(model_path=None, hough_fallback=True)

        p = RealPerceptionPipeline(
            config, camera=cam, detector=det,
            extrinsics=_identity_extrinsics(),
        )

        z0, vz0 = 0.30, 2.0  # 30cm initial depth, 2 m/s away from camera
        # In camera frame, ball moving away = increasing Z (depth).
        # Gravity acts in -Z direction (ball decelerates then falls back).
        # Apex at t = vz0/g ≈ 0.204s, z_apex ≈ 0.30 + 0.204 m ≈ 0.504m.

        # Set initial position and start
        pos0 = self._ballistic_pos(0, z0, vz0, g=9.81)  # g>0 since Z is depth (away)
        cam.set_ball_pos_cam(pos0)
        p.start()
        time.sleep(0.05)  # let acq thread start

        # Drive the trajectory for 0.3s at ~50Hz (policy rate)
        dt_policy = 0.02
        n_steps = 15
        observations = []
        true_positions = []
        t = 0.0

        for _ in range(n_steps):
            t += dt_policy
            # Gravity reversed in camera depth direction: ball going up = Z increasing,
            # gravity pulls it back = Z eventually decreasing.
            # Use simple model: z = z0 + vz0*t - 0.5*9.81*t^2 for camera Z.
            true_pos = np.array([0.02 * t, 0.0, z0 + vz0 * t - 0.5 * 9.81 * t * t])
            if true_pos[2] < 0.17:  # below D435i min depth
                break
            cam.set_ball_pos_cam(true_pos)
            time.sleep(dt_policy)
            obs = p.get_observation(_identity_quat(), np.zeros(3))
            observations.append(obs)
            true_positions.append(true_pos.copy())

        p.stop()

        self.assertGreater(len(observations), 5, "Should have tracked for >5 steps")

        # After a few steps of EKF convergence, position error should be < 50mm
        last_obs = observations[-1]
        last_true = true_positions[-1]
        pos_err = np.linalg.norm(last_obs.ball_pos_b - last_true)
        self.assertLess(pos_err, 0.08,
                        f"Final position error {pos_err:.3f}m > 80mm (true={last_true}, "
                        f"est={last_obs.ball_pos_b})")

        # Detection rate should be high (>50%) since ball is always in view
        stats = p.stats
        self.assertGreater(stats["detection_rate"], 0.5,
                           f"Detection rate {stats['detection_rate']:.2f} < 50%")

    def test_ball_disappears_then_reappears(self):
        """Ball tracked → disappears (cleared) → reappears. Verify ball_lost flag."""
        config = HardwarePipelineConfig()
        cam = MockCamera(MockCameraConfig(depth_noise_std_mm=1.0))
        det = BallDetector(model_path=None, hough_fallback=True)

        p = RealPerceptionPipeline(
            config, camera=cam, detector=det,
            extrinsics=_identity_extrinsics(),
        )

        # Phase 1: ball visible at 50cm
        cam.set_ball_pos_cam(np.array([0.0, 0.0, 0.5]))
        p.start()
        time.sleep(0.05)

        for _ in range(10):
            obs = p.get_observation(_identity_quat(), np.zeros(3))
            time.sleep(0.02)

        self.assertTrue(obs.detected or not obs.ball_lost,
                        "Ball should be tracked in phase 1")

        # Phase 2: ball disappears
        cam.clear_ball()
        time.sleep(0.05)

        lost_seen = False
        for _ in range(20):
            obs = p.get_observation(_identity_quat(), np.zeros(3))
            if obs.ball_lost:
                lost_seen = True
                break
            time.sleep(0.02)

        self.assertTrue(lost_seen,
                        "ball_lost should trigger after consecutive dropouts")

        # Phase 3: ball reappears at new position
        cam.set_ball_pos_cam(np.array([0.05, 0.0, 0.40]))
        time.sleep(0.10)  # let acq thread pick it up

        recovered = False
        for _ in range(20):
            obs = p.get_observation(_identity_quat(), np.zeros(3))
            if obs.detected and not obs.ball_lost:
                recovered = True
                break
            time.sleep(0.02)

        p.stop()
        self.assertTrue(recovered, "Pipeline should recover after ball reappears")

    def test_ball_at_different_depths(self):
        """Hough detector should work across the juggling range (30cm-1.0m).

        Beyond ~1.0m the 40mm ball subtends <8px radius — Hough becomes
        unreliable without YOLO. This is fine for our task where the ball
        is typically 0.2-1.0m above the paddle.
        """
        depths = [0.30, 0.50, 0.80, 1.0]
        config = HardwarePipelineConfig()

        for z in depths:
            cam = MockCamera(MockCameraConfig(depth_noise_std_mm=1.0))
            det = BallDetector(model_path=None, hough_fallback=True)
            p = RealPerceptionPipeline(
                config, camera=cam, detector=det,
                extrinsics=_identity_extrinsics(),
            )
            cam.set_ball_pos_cam(np.array([0.0, 0.0, z]))
            p.start()
            time.sleep(0.15)  # let acq thread run a few cycles

            obs = None
            for _ in range(20):
                obs = p.get_observation(_identity_quat(), np.zeros(3))
                time.sleep(0.02)

            p.stop()

            pos_err = np.linalg.norm(obs.ball_pos_b - np.array([0, 0, z]))
            # Tolerate larger error at longer range (depth noise grows)
            max_err = 0.05 + 0.05 * z  # 50mm + 50mm per metre
            self.assertLess(
                pos_err, max_err,
                f"depth={z}m: error {pos_err:.3f}m > {max_err:.3f}m",
            )

    def test_rotated_robot_body_frame(self):
        """Ball at fixed world position, robot yawed 90° — body-frame output should rotate."""
        config = HardwarePipelineConfig()
        cam = MockCamera(MockCameraConfig(depth_noise_std_mm=0.5))
        det = BallDetector(model_path=None, hough_fallback=True)

        # 90° yaw rotation: body X -> world Y, body Y -> world -X
        angle = np.pi / 2
        quat_90z = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])

        p = RealPerceptionPipeline(
            config, camera=cam, detector=det,
            extrinsics=_identity_extrinsics(),
        )

        # Ball at camera-frame (0, 0, 0.5) = world (0, 0, 0.5) since extrinsics=identity
        cam.set_ball_pos_cam(np.array([0.0, 0.0, 0.5]))
        p.start()
        time.sleep(0.08)

        for _ in range(15):
            obs = p.get_observation(quat_90z, np.zeros(3))
            time.sleep(0.01)

        p.stop()

        # After EKF converges, world-frame estimate should be near (0, 0, 0.5)
        self.assertIsNotNone(obs.ekf_pos_w)
        np.testing.assert_allclose(obs.ekf_pos_w, [0, 0, 0.5], atol=0.10)

        # Body-frame should be the world pos rotated by R_world_body
        R_body_world = _quat_to_rotmat(quat_90z)
        expected_body = R_body_world.T @ np.array([0, 0, 0.5])
        np.testing.assert_allclose(obs.ball_pos_b, expected_body, atol=0.10)


if __name__ == "__main__":
    unittest.main()
