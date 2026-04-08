"""Tests for Hough circle fallback detector + CameraCalibrator.from_yaml().

Tests the BallDetector._detect_hough() method using MockCamera-generated
depth frames with known ball positions. No hardware or YOLO model required.

Run: python scripts/perception/test_hough_detector.py -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

# Direct imports — bypass Isaac Lab init chain
import importlib.util

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
CameraCalibrator = _calib_mod.CameraCalibrator
CameraExtrinsics = _calib_mod.CameraExtrinsics

_detector_mod = _load_module(
    "perception.real.detector",
    os.path.join(_PERC_DIR, "real", "detector.py"),
)
BallDetector = _detector_mod.BallDetector
Detection = _detector_mod.Detection

_mock_mod = _load_module(
    "perception.real.mock",
    os.path.join(_PERC_DIR, "real", "mock.py"),
)
MockCamera = _mock_mod.MockCamera
MockCameraConfig = _mock_mod.MockCameraConfig

# Check cv2 availability — skip Hough tests if not installed
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


def _make_depth_with_ball(
    pos_cam: np.ndarray,
    noise_std: float = 0.0,
) -> tuple[np.ndarray, CameraIntrinsics]:
    """Generate a synthetic depth frame with a ball at pos_cam."""
    cfg = MockCameraConfig(depth_noise_std_mm=noise_std)
    cam = MockCamera(cfg)
    cam.start()
    cam.set_ball_pos_cam(pos_cam)
    depth, _ = cam.get_frame()
    intr = cam.get_intrinsics()
    cam.stop()
    return depth, intr


# ── Hough circle detector tests ──────────────────────────────────────────


@unittest.skipUnless(_HAS_CV2, "cv2 not installed")
class TestHoughDetector(unittest.TestCase):
    """Hough circle fallback detection on synthetic depth frames."""

    def setUp(self):
        # Detector with no YOLO model — relies entirely on Hough
        self.det = BallDetector(model_path=None, hough_fallback=True)

    def test_detect_ball_at_50cm(self):
        """Ball at 50cm along optical axis is detected within 10mm."""
        pos = np.array([0.0, 0.0, 0.5])
        depth, intr = _make_depth_with_ball(pos, noise_std=0.0)
        result = self.det.detect(depth, intr)

        self.assertIsNotNone(result, "Hough failed to detect ball at 50cm")
        self.assertEqual(result.method, "hough")
        err = np.linalg.norm(result.pos_cam - pos)
        self.assertLess(err, 0.010, f"Position error {err:.4f}m > 10mm")

    def test_detect_ball_at_30cm(self):
        """Ball at near range (30cm) — larger apparent size."""
        pos = np.array([0.0, 0.0, 0.30])
        depth, intr = _make_depth_with_ball(pos, noise_std=0.0)
        result = self.det.detect(depth, intr)

        self.assertIsNotNone(result, "Hough failed to detect ball at 30cm")
        err = np.linalg.norm(result.pos_cam - pos)
        self.assertLess(err, 0.010, f"Position error {err:.4f}m > 10mm")

    def test_detect_ball_at_1m(self):
        """Ball at far range (1m) — small apparent size."""
        pos = np.array([0.0, 0.0, 1.0])
        depth, intr = _make_depth_with_ball(pos, noise_std=0.0)
        result = self.det.detect(depth, intr)

        self.assertIsNotNone(result, "Hough failed to detect ball at 1m")
        err = np.linalg.norm(result.pos_cam - pos)
        self.assertLess(err, 0.015, f"Position error {err:.4f}m > 15mm")

    def test_detect_ball_off_centre(self):
        """Ball offset from optical axis."""
        pos = np.array([0.05, -0.03, 0.5])
        depth, intr = _make_depth_with_ball(pos, noise_std=0.0)
        result = self.det.detect(depth, intr)

        self.assertIsNotNone(result, "Hough failed to detect off-centre ball")
        err = np.linalg.norm(result.pos_cam - pos)
        self.assertLess(err, 0.015, f"Position error {err:.4f}m > 15mm")

    def test_detect_ball_with_noise(self):
        """Ball detection still works with realistic depth noise."""
        pos = np.array([0.0, 0.0, 0.5])
        depth, intr = _make_depth_with_ball(pos, noise_std=2.0)
        result = self.det.detect(depth, intr)

        self.assertIsNotNone(result, "Hough failed with 2mm depth noise")
        err = np.linalg.norm(result.pos_cam - pos)
        self.assertLess(err, 0.015, f"Position error {err:.4f}m > 15mm with noise")

    def test_empty_frame_no_detection(self):
        """No ball in frame → no detection."""
        depth = np.zeros((480, 848), dtype=np.uint16)
        intr = CameraIntrinsics(fx=425, fy=425, cx=424, cy=240, width=848, height=480)
        result = self.det.detect(depth, intr)
        self.assertIsNone(result)

    def test_hough_disabled(self):
        """With hough_fallback=False and no YOLO, detect returns None."""
        det = BallDetector(model_path=None, hough_fallback=False)
        pos = np.array([0.0, 0.0, 0.5])
        depth, intr = _make_depth_with_ball(pos, noise_std=0.0)
        result = det.detect(depth, intr)
        self.assertIsNone(result)

    def test_confidence_is_valid(self):
        """Hough detection confidence is in [0, 1]."""
        pos = np.array([0.0, 0.0, 0.5])
        depth, intr = _make_depth_with_ball(pos, noise_std=0.0)
        result = self.det.detect(depth, intr)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_bbox_is_valid(self):
        """Bounding box is non-empty and within image bounds."""
        pos = np.array([0.0, 0.0, 0.5])
        depth, intr = _make_depth_with_ball(pos, noise_std=0.0)
        result = self.det.detect(depth, intr)
        self.assertIsNotNone(result)
        x1, y1, x2, y2 = result.bbox
        self.assertLess(x1, x2)
        self.assertLess(y1, y2)
        self.assertGreaterEqual(x1, 0)
        self.assertGreaterEqual(y1, 0)
        self.assertLessEqual(x2, intr.width)
        self.assertLessEqual(y2, intr.height)

    def test_detect_logic_uses_hough_when_no_model(self):
        """BallDetector.detect() calls _detect_hough when model is None."""
        pos = np.array([0.0, 0.0, 0.5])
        depth, intr = _make_depth_with_ball(pos, noise_std=0.0)
        result = self.det.detect(depth, intr)
        self.assertIsNotNone(result)
        self.assertEqual(result.method, "hough")

    def test_depth_range_consistency(self):
        """Detected depth matches known ball distance."""
        for z in [0.25, 0.5, 0.8, 1.2]:
            if z < self.det._min_depth or z > self.det._max_depth:
                continue
            pos = np.array([0.0, 0.0, z])
            depth, intr = _make_depth_with_ball(pos, noise_std=0.0)
            result = self.det.detect(depth, intr)
            if result is not None:
                z_det = result.pos_cam[2]
                self.assertAlmostEqual(
                    z_det, z, delta=0.015,
                    msg=f"Depth mismatch at z={z}: detected {z_det:.4f}m"
                )


# ── CameraCalibrator.from_yaml tests ─────────────────────────────────────


class TestCalibratorFromYaml(unittest.TestCase):
    """CameraCalibrator.from_yaml() loads extrinsics from YAML."""

    def test_identity_yaml(self):
        """Load identity extrinsics from YAML."""
        yaml_content = """\
R_cam_body:
  - [1.0, 0.0, 0.0]
  - [0.0, 1.0, 0.0]
  - [0.0, 0.0, 1.0]
t_cam_body: [0.0, 0.0, 0.0]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            ext = CameraCalibrator.from_yaml(path)
            np.testing.assert_allclose(ext.R_cam_body, np.eye(3), atol=1e-10)
            np.testing.assert_allclose(ext.t_cam_body, np.zeros(3), atol=1e-10)
        finally:
            os.unlink(path)

    def test_nonidentity_yaml(self):
        """Load non-trivial extrinsics and verify transform."""
        # 90-degree rotation about Z: X→Y, Y→-X
        yaml_content = """\
R_cam_body:
  - [0.0, -1.0, 0.0]
  - [1.0, 0.0, 0.0]
  - [0.0, 0.0, 1.0]
t_cam_body: [0.1, -0.05, 0.07]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            ext = CameraCalibrator.from_yaml(path)
            # Camera X=[1,0,0] → body [-1,0,0]... wait, R @ [1,0,0] = [0,1,0]
            pt = ext.transform_to_body(np.array([1.0, 0.0, 0.0]))
            expected = np.array([0.0, 1.0, 0.0]) + np.array([0.1, -0.05, 0.07])
            np.testing.assert_allclose(pt, expected, atol=1e-10)
        finally:
            os.unlink(path)

    def test_bad_shape_raises(self):
        """Invalid R shape raises ValueError."""
        yaml_content = """\
R_cam_body:
  - [1.0, 0.0]
  - [0.0, 1.0]
t_cam_body: [0.0, 0.0, 0.0]
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            path = f.name

        try:
            with self.assertRaises(ValueError):
                CameraCalibrator.from_yaml(path)
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        """Missing file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            CameraCalibrator.from_yaml("/nonexistent/path.yaml")


if __name__ == "__main__":
    unittest.main(verbosity=2)
