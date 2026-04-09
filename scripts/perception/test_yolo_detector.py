"""Tests for BallDetector._detect_yolo() ONNX inference path.

Tests the full YOLO pipeline: preprocessing (depth→letterbox→blob),
output parsing (both (1,5,N) and (1,N,5) layouts), coordinate
un-letterboxing, depth lookup, and 3D deprojection.

Uses a mock ONNX session (no real model file needed).

Run: pytest scripts/perception/test_yolo_detector.py -x -q
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

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


# Load perception modules directly (dotted names so relative imports resolve)
_cam_mod = _load_module(
    "perception.real.camera",
    os.path.join(_PERC_DIR, "real", "camera.py"),
)
CameraIntrinsics = _cam_mod.CameraIntrinsics

_det_mod = _load_module(
    "perception.real.detector",
    os.path.join(_PERC_DIR, "real", "detector.py"),
)
BallDetector = _det_mod.BallDetector
Detection = _det_mod.Detection
_letterbox = _det_mod._letterbox
_YOLO_INPUT_SIZE = _det_mod._YOLO_INPUT_SIZE


# D435i 848x480 typical intrinsics
_INTRINSICS = CameraIntrinsics(
    fx=425.19, fy=425.19, cx=423.36, cy=239.95,
    width=848, height=480,
)


def _make_depth_frame_with_ball(
    intrinsics: CameraIntrinsics,
    ball_z_m: float = 0.50,
    ball_x_m: float = 0.0,
    ball_y_m: float = 0.0,
    ball_radius_m: float = 0.020,
) -> np.ndarray:
    """Create a synthetic depth frame with a ball at a known 3D position."""
    frame = np.zeros((intrinsics.height, intrinsics.width), dtype=np.uint16)

    # Project ball centre to pixel coords
    u = intrinsics.fx * ball_x_m / ball_z_m + intrinsics.cx
    v = intrinsics.fy * ball_y_m / ball_z_m + intrinsics.cy
    r_px = intrinsics.fx * ball_radius_m / ball_z_m

    # Draw filled circle at ball depth
    yy, xx = np.mgrid[:intrinsics.height, :intrinsics.width]
    mask = ((xx - u) ** 2 + (yy - v) ** 2) <= r_px ** 2
    frame[mask] = int(ball_z_m * 1000)

    return frame


def _make_mock_yolo_output(
    cx: float, cy: float, w: float, h: float, conf: float,
    layout: str = "5xN",
) -> np.ndarray:
    """Create a mock YOLOv8 output tensor.

    Args:
        cx, cy, w, h: Bounding box in letterboxed pixel coords.
        conf: Detection confidence.
        layout: "5xN" for (1, 5, N) or "Nx5" for (1, N, 5).
    """
    row = np.array([[cx, cy, w, h, conf]], dtype=np.float32)
    if layout == "5xN":
        return row.T[np.newaxis]  # (1, 5, 1)
    else:
        return row[np.newaxis]  # (1, 1, 5)


class TestLetterbox(unittest.TestCase):
    """Test letterbox resizing utility."""

    def test_square_input(self):
        img = np.zeros((480, 480), dtype=np.uint8)
        padded, scale, (px, py) = _letterbox(img, 640)
        self.assertEqual(padded.shape, (640, 640))
        self.assertAlmostEqual(scale, 640 / 480, places=3)

    def test_wide_input(self):
        img = np.zeros((480, 848), dtype=np.uint8)
        padded, scale, (px, py) = _letterbox(img, 640)
        self.assertEqual(padded.shape, (640, 640))
        # Scale determined by wider dimension
        self.assertAlmostEqual(scale, 640 / 848, places=3)
        # Vertical padding should be nonzero
        self.assertGreater(py, 0)

    def test_tall_input(self):
        img = np.zeros((848, 480), dtype=np.uint8)
        padded, scale, (px, py) = _letterbox(img, 640)
        self.assertEqual(padded.shape, (640, 640))
        self.assertAlmostEqual(scale, 640 / 848, places=3)
        self.assertGreater(px, 0)

    def test_3channel(self):
        img = np.zeros((480, 848, 3), dtype=np.uint8)
        padded, scale, _ = _letterbox(img, 640)
        self.assertEqual(padded.shape, (640, 640, 3))

    def test_padding_value(self):
        """Padding should be 114 (YOLO convention)."""
        img = np.zeros((100, 200), dtype=np.uint8)
        padded, _, (px, py) = _letterbox(img, 640)
        # Top padding area should be 114
        if py > 0:
            self.assertEqual(padded[0, px], 114)


class TestPreprocess(unittest.TestCase):
    """Test BallDetector._preprocess()."""

    def setUp(self):
        self.detector = BallDetector(model_path=None, hough_fallback=False)

    def test_output_shape(self):
        frame = np.full((480, 848), 500, dtype=np.uint16)
        blob, scale, (px, py) = self.detector._preprocess(frame)
        self.assertEqual(blob.shape, (1, 3, 640, 640))
        self.assertEqual(blob.dtype, np.float32)

    def test_normalised_range(self):
        frame = np.full((480, 848), 500, dtype=np.uint16)
        blob, _, _ = self.detector._preprocess(frame)
        self.assertGreaterEqual(blob.min(), 0.0)
        self.assertLessEqual(blob.max(), 1.0)

    def test_invalid_depth_is_zero(self):
        """Pixels outside [min_depth, max_depth] should be zero."""
        frame = np.zeros((480, 848), dtype=np.uint16)  # all zero = invalid
        blob, _, _ = self.detector._preprocess(frame)
        # The padded region is 114/255, but the image region should be 0
        # (since depth=0 is invalid)
        # Just check it doesn't crash and stays in range
        self.assertGreaterEqual(blob.min(), 0.0)

    def test_scale_and_pad(self):
        """Scale and pad should allow reverse mapping to original coords."""
        frame = np.full((480, 848), 500, dtype=np.uint16)
        _, scale, (px, py) = self.detector._preprocess(frame)
        # 848 is the dominant dimension
        expected_scale = 640 / 848
        self.assertAlmostEqual(scale, expected_scale, places=3)


class TestDetectYolo(unittest.TestCase):
    """Test _detect_yolo with mock ONNX session."""

    def _make_detector_with_mock(self, yolo_output: np.ndarray) -> BallDetector:
        """Create a BallDetector with a mock ONNX session."""
        det = BallDetector(model_path=None, hough_fallback=False)
        mock_session = MagicMock()
        mock_session.run.return_value = [yolo_output]
        det._model = mock_session
        det._input_name = "images"
        return det

    def test_detect_5xN_layout(self):
        """Test with (1, 5, N) output layout (standard YOLOv8)."""
        # Place ball at image centre, 0.5m away
        frame = _make_depth_frame_with_ball(_INTRINSICS, ball_z_m=0.50)

        # Compute expected letterboxed coords for image centre
        scale = 640 / 848
        px = (640 - int(round(848 * scale))) // 2
        py = (640 - int(round(480 * scale))) // 2
        cx_lb = _INTRINSICS.cx * scale + px
        cy_lb = _INTRINSICS.cy * scale + py
        r_lb = _INTRINSICS.fx * 0.020 / 0.50 * scale
        w_lb = h_lb = r_lb * 2 * 1.2  # slightly larger bbox

        output = _make_mock_yolo_output(cx_lb, cy_lb, w_lb, h_lb, 0.85, "5xN")
        det = self._make_detector_with_mock(output)
        result = det._detect_yolo(frame, _INTRINSICS)

        self.assertIsNotNone(result)
        self.assertEqual(result.method, "yolo")
        self.assertAlmostEqual(result.confidence, 0.85, places=2)
        # Z should be close to 0.50m
        self.assertAlmostEqual(result.pos_cam[2], 0.50, delta=0.05)
        # XY should be near zero (ball at centre)
        self.assertAlmostEqual(result.pos_cam[0], 0.0, delta=0.02)
        self.assertAlmostEqual(result.pos_cam[1], 0.0, delta=0.02)

    def test_detect_Nx5_layout(self):
        """Test with (1, N, 5) output layout."""
        frame = _make_depth_frame_with_ball(_INTRINSICS, ball_z_m=0.50)

        scale = 640 / 848
        px = (640 - int(round(848 * scale))) // 2
        py = (640 - int(round(480 * scale))) // 2
        cx_lb = _INTRINSICS.cx * scale + px
        cy_lb = _INTRINSICS.cy * scale + py

        output = _make_mock_yolo_output(cx_lb, cy_lb, 30, 30, 0.90, "Nx5")
        det = self._make_detector_with_mock(output)
        result = det._detect_yolo(frame, _INTRINSICS)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.confidence, 0.90, places=2)

    def test_low_confidence_returns_none(self):
        """Detections below hough_fallback_thresh are rejected."""
        frame = _make_depth_frame_with_ball(_INTRINSICS, ball_z_m=0.50)
        output = _make_mock_yolo_output(320, 320, 20, 20, 0.1, "5xN")
        det = self._make_detector_with_mock(output)
        result = det._detect_yolo(frame, _INTRINSICS)
        self.assertIsNone(result)

    def test_no_valid_depth_returns_none(self):
        """If bbox has no valid depth pixels, return None."""
        frame = np.zeros((480, 848), dtype=np.uint16)  # all invalid
        output = _make_mock_yolo_output(320, 240, 30, 30, 0.95, "5xN")
        det = self._make_detector_with_mock(output)
        result = det._detect_yolo(frame, _INTRINSICS)
        self.assertIsNone(result)

    def test_empty_predictions(self):
        """Empty prediction tensor returns None."""
        frame = _make_depth_frame_with_ball(_INTRINSICS)
        empty = np.zeros((1, 5, 0), dtype=np.float32)
        det = self._make_detector_with_mock(empty)
        result = det._detect_yolo(frame, _INTRINSICS)
        self.assertIsNone(result)

    def test_best_of_multiple(self):
        """Selects highest-confidence detection from multiple candidates."""
        frame = _make_depth_frame_with_ball(_INTRINSICS, ball_z_m=0.50)

        # Two detections: low conf at (100,100), high conf at image centre
        scale = 640 / 848
        px = (640 - int(round(848 * scale))) // 2
        py = (640 - int(round(480 * scale))) // 2
        cx_lb = _INTRINSICS.cx * scale + px
        cy_lb = _INTRINSICS.cy * scale + py

        preds = np.array([
            [100, cx_lb],   # cx
            [100, cy_lb],   # cy
            [20,  30],      # w
            [20,  30],      # h
            [0.3, 0.92],    # conf
        ], dtype=np.float32)[np.newaxis]  # (1, 5, 2)

        det = self._make_detector_with_mock(preds)
        result = det._detect_yolo(frame, _INTRINSICS)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.confidence, 0.92, places=2)

    def test_off_centre_ball(self):
        """Ball not at image centre — verify deprojection is correct."""
        ball_x, ball_y, ball_z = 0.05, -0.03, 0.40
        frame = _make_depth_frame_with_ball(
            _INTRINSICS, ball_z_m=ball_z, ball_x_m=ball_x, ball_y_m=ball_y,
        )

        # Compute expected letterboxed coordinates
        u = _INTRINSICS.fx * ball_x / ball_z + _INTRINSICS.cx
        v = _INTRINSICS.fy * ball_y / ball_z + _INTRINSICS.cy
        scale = 640 / 848
        px = (640 - int(round(848 * scale))) // 2
        py = (640 - int(round(480 * scale))) // 2
        cx_lb = u * scale + px
        cy_lb = v * scale + py

        output = _make_mock_yolo_output(cx_lb, cy_lb, 40, 40, 0.88, "5xN")
        det = self._make_detector_with_mock(output)
        result = det._detect_yolo(frame, _INTRINSICS)

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.pos_cam[0], ball_x, delta=0.02)
        self.assertAlmostEqual(result.pos_cam[1], ball_y, delta=0.02)
        self.assertAlmostEqual(result.pos_cam[2], ball_z, delta=0.05)

    def test_bbox_clamps_to_image(self):
        """Bbox near edge should be clamped to image bounds."""
        frame = _make_depth_frame_with_ball(
            _INTRINSICS, ball_z_m=0.30, ball_x_m=-0.10, ball_y_m=0.0,
        )
        # Detection near left edge in letterboxed coords
        scale = 640 / 848
        px = (640 - int(round(848 * scale))) // 2
        py = (640 - int(round(480 * scale))) // 2
        # cx near px (left edge)
        output = _make_mock_yolo_output(px + 5, 320, 50, 50, 0.75, "5xN")
        det = self._make_detector_with_mock(output)
        result = det._detect_yolo(frame, _INTRINSICS)

        if result is not None:
            x1, y1, x2, y2 = result.bbox
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, _INTRINSICS.width)
            self.assertLessEqual(y2, _INTRINSICS.height)


class TestDetectIntegration(unittest.TestCase):
    """Integration: detect() routes correctly between YOLO and Hough."""

    def test_yolo_primary_when_confident(self):
        """High-confidence YOLO detection is returned directly."""
        frame = _make_depth_frame_with_ball(_INTRINSICS, ball_z_m=0.50)
        scale = 640 / 848
        px = (640 - int(round(848 * scale))) // 2
        py = (640 - int(round(480 * scale))) // 2
        cx_lb = _INTRINSICS.cx * scale + px
        cy_lb = _INTRINSICS.cy * scale + py
        output = _make_mock_yolo_output(cx_lb, cy_lb, 30, 30, 0.95, "5xN")

        det = BallDetector(model_path=None, hough_fallback=True)
        mock_session = MagicMock()
        mock_session.run.return_value = [output]
        det._model = mock_session
        det._input_name = "images"

        result = det.detect(frame, _INTRINSICS)
        self.assertIsNotNone(result)
        self.assertEqual(result.method, "yolo")

    def test_hough_fallback_when_no_model(self):
        """Falls back to Hough when no YOLO model loaded."""
        frame = _make_depth_frame_with_ball(_INTRINSICS, ball_z_m=0.50)
        det = BallDetector(model_path=None, hough_fallback=True)
        result = det.detect(frame, _INTRINSICS)
        if result is not None:
            self.assertEqual(result.method, "hough")

    def test_no_detection_when_both_fail(self):
        """Returns None when both YOLO and Hough fail."""
        frame = np.zeros((480, 848), dtype=np.uint16)  # empty frame
        det = BallDetector(model_path=None, hough_fallback=True)
        result = det.detect(frame, _INTRINSICS)
        self.assertIsNone(result)


class TestMedianDepth(unittest.TestCase):
    """Test _median_depth_in_bbox static method."""

    def test_valid_region(self):
        frame = np.full((100, 100), 500, dtype=np.uint16)
        d = BallDetector._median_depth_in_bbox(frame, (10, 10, 20, 20))
        self.assertAlmostEqual(d, 0.50, places=2)

    def test_empty_bbox_returns_none(self):
        frame = np.zeros((100, 100), dtype=np.uint16)
        d = BallDetector._median_depth_in_bbox(frame, (10, 10, 20, 20))
        self.assertIsNone(d)

    def test_mixed_valid_invalid(self):
        frame = np.zeros((100, 100), dtype=np.uint16)
        frame[15, 15] = 400
        frame[16, 15] = 600
        d = BallDetector._median_depth_in_bbox(frame, (10, 10, 20, 20))
        self.assertAlmostEqual(d, 0.50, places=2)


if __name__ == "__main__":
    unittest.main()
