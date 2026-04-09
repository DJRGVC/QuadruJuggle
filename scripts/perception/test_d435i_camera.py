"""Tests for D435iCamera pyrealsense2 wrapper implementation.

Tests the D435iCamera methods (start, get_frame, stop) by mocking the
pyrealsense2 library. No hardware required.

Run: pytest scripts/perception/test_d435i_camera.py -v
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np

# Direct imports — bypass Isaac Lab __init__ chain
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
    "perception.real.camera_test",
    os.path.join(_PERC_DIR, "real", "camera.py"),
)
CameraIntrinsics = _camera_mod.CameraIntrinsics
D435iCamera = _camera_mod.D435iCamera


def _make_mock_rs():
    """Create a mock pyrealsense2 module with all required attributes."""
    mock_rs = MagicMock()

    # Mock stream types
    mock_rs.stream.depth = "depth_stream"
    mock_rs.format.z16 = "z16_format"

    # Mock intrinsics
    mock_intrinsics = MagicMock()
    mock_intrinsics.fx = 425.0
    mock_intrinsics.fy = 425.0
    mock_intrinsics.ppx = 424.0
    mock_intrinsics.ppy = 240.0
    mock_intrinsics.width = 848
    mock_intrinsics.height = 480

    # Mock profile chain
    mock_video_profile = MagicMock()
    mock_video_profile.get_intrinsics.return_value = mock_intrinsics

    mock_stream_profile = MagicMock()
    mock_stream_profile.as_video_stream_profile.return_value = mock_video_profile

    mock_profile = MagicMock()
    mock_profile.get_stream.return_value = mock_stream_profile

    # Mock pipeline
    mock_pipeline_instance = MagicMock()
    mock_pipeline_instance.start.return_value = mock_profile
    mock_rs.pipeline.return_value = mock_pipeline_instance

    return mock_rs, mock_pipeline_instance


class TestD435iCameraStart(unittest.TestCase):
    """D435iCamera.start() configures and starts the pipeline."""

    def test_start_extracts_intrinsics(self):
        """start() reads intrinsics from the depth stream profile."""
        mock_rs, mock_pipeline = _make_mock_rs()

        # Temporarily replace the rs module in the camera module
        original_rs = _camera_mod.rs
        _camera_mod.rs = mock_rs

        try:
            cam = D435iCamera(width=848, height=480, fps=90)
            cam.start()

            intr = cam.get_intrinsics()
            self.assertAlmostEqual(intr.fx, 425.0)
            self.assertAlmostEqual(intr.fy, 425.0)
            self.assertAlmostEqual(intr.cx, 424.0)
            self.assertAlmostEqual(intr.cy, 240.0)
            self.assertEqual(intr.width, 848)
            self.assertEqual(intr.height, 480)
        finally:
            _camera_mod.rs = original_rs

    def test_start_configures_depth_stream(self):
        """start() enables the depth stream with correct resolution/fps."""
        mock_rs, mock_pipeline = _make_mock_rs()
        original_rs = _camera_mod.rs
        _camera_mod.rs = mock_rs

        try:
            cam = D435iCamera(width=848, height=480, fps=90)
            cam.start()

            # Verify config.enable_stream was called
            mock_config = mock_rs.config.return_value
            mock_config.enable_stream.assert_called_once_with(
                mock_rs.stream.depth, 848, 480, mock_rs.format.z16, 90
            )
        finally:
            _camera_mod.rs = original_rs

    def test_start_with_serial(self):
        """start() enables specific device when serial is provided."""
        mock_rs, mock_pipeline = _make_mock_rs()
        original_rs = _camera_mod.rs
        _camera_mod.rs = mock_rs

        try:
            cam = D435iCamera(serial="123456789", width=848, height=480, fps=90)
            cam.start()

            mock_config = mock_rs.config.return_value
            mock_config.enable_device.assert_called_once_with("123456789")
        finally:
            _camera_mod.rs = original_rs

    def test_start_waits_for_first_frame(self):
        """start() blocks until first frame arrives."""
        mock_rs, mock_pipeline = _make_mock_rs()
        original_rs = _camera_mod.rs
        _camera_mod.rs = mock_rs

        try:
            cam = D435iCamera()
            cam.start()
            mock_pipeline.wait_for_frames.assert_called_once_with(timeout_ms=5000)
        finally:
            _camera_mod.rs = original_rs


class TestD435iCameraGetFrame(unittest.TestCase):
    """D435iCamera.get_frame() polls for depth frames."""

    def test_get_frame_returns_depth_and_timestamp(self):
        """get_frame() returns (depth_u16, timestamp_s) when frame available."""
        mock_rs, mock_pipeline = _make_mock_rs()

        # Create fake depth data
        fake_depth_data = np.full(480 * 848, 500, dtype=np.uint16)  # 500mm

        mock_depth_frame = MagicMock()
        mock_depth_frame.get_timestamp.return_value = 1234.5  # ms
        mock_depth_frame.get_data.return_value = fake_depth_data

        mock_frames = MagicMock()
        mock_frames.__bool__ = MagicMock(return_value=True)
        mock_frames.get_depth_frame.return_value = mock_depth_frame

        mock_pipeline.poll_for_frames.return_value = mock_frames

        original_rs = _camera_mod.rs
        _camera_mod.rs = mock_rs

        try:
            cam = D435iCamera(width=848, height=480)
            cam.start()

            result = cam.get_frame()
            self.assertIsNotNone(result)
            depth, ts = result
            self.assertEqual(depth.shape, (480, 848))
            self.assertEqual(depth.dtype, np.uint16)
            self.assertAlmostEqual(ts, 1.2345)  # ms → s
        finally:
            _camera_mod.rs = original_rs

    def test_get_frame_returns_none_when_no_frame(self):
        """get_frame() returns None when no frame is ready (non-blocking)."""
        mock_rs, mock_pipeline = _make_mock_rs()

        mock_frames = MagicMock()
        mock_frames.__bool__ = MagicMock(return_value=False)
        mock_pipeline.poll_for_frames.return_value = mock_frames

        original_rs = _camera_mod.rs
        _camera_mod.rs = mock_rs

        try:
            cam = D435iCamera(width=848, height=480)
            cam.start()

            result = cam.get_frame()
            self.assertIsNone(result)
        finally:
            _camera_mod.rs = original_rs

    def test_get_frame_before_start_returns_none(self):
        """get_frame() before start() returns None (pipeline is None)."""
        mock_rs, _ = _make_mock_rs()
        original_rs = _camera_mod.rs
        _camera_mod.rs = mock_rs

        try:
            cam = D435iCamera()
            # Don't call start()
            result = cam.get_frame()
            self.assertIsNone(result)
        finally:
            _camera_mod.rs = original_rs


class TestD435iCameraStop(unittest.TestCase):
    """D435iCamera.stop() releases resources."""

    def test_stop_calls_pipeline_stop(self):
        """stop() calls pipeline.stop()."""
        mock_rs, mock_pipeline = _make_mock_rs()
        original_rs = _camera_mod.rs
        _camera_mod.rs = mock_rs

        try:
            cam = D435iCamera()
            cam.start()
            cam.stop()
            mock_pipeline.stop.assert_called_once()
        finally:
            _camera_mod.rs = original_rs

    def test_stop_sets_pipeline_none(self):
        """After stop(), pipeline is None (prevents get_frame from using it)."""
        mock_rs, mock_pipeline = _make_mock_rs()
        original_rs = _camera_mod.rs
        _camera_mod.rs = mock_rs

        try:
            cam = D435iCamera()
            cam.start()
            cam.stop()
            # get_frame should return None now
            self.assertIsNone(cam.get_frame())
        finally:
            _camera_mod.rs = original_rs

    def test_stop_idempotent(self):
        """Calling stop() when already stopped doesn't raise."""
        mock_rs, mock_pipeline = _make_mock_rs()
        original_rs = _camera_mod.rs
        _camera_mod.rs = mock_rs

        try:
            cam = D435iCamera()
            cam.stop()  # Never started — should not raise
        finally:
            _camera_mod.rs = original_rs


class TestD435iCameraImportGuard(unittest.TestCase):
    """D435iCamera raises ImportError when pyrealsense2 is missing."""

    def test_raises_without_pyrealsense2(self):
        """Constructor raises ImportError if rs is None."""
        original_rs = _camera_mod.rs
        _camera_mod.rs = None

        try:
            with self.assertRaises(ImportError):
                D435iCamera()
        finally:
            _camera_mod.rs = original_rs


if __name__ == "__main__":
    unittest.main(verbosity=2)
