"""Tests for DepthFrameVisualizer.

Tests cover:
- Colorization of uint16 depth frames
- Detection bbox and marker rendering
- SimDetection adapter
- float32 → uint16 conversion path
- Telemetry text overlay
- Edge cases (no detection, empty frame, all-invalid depth)
- Panel size consistency
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

import numpy as np
import pytest

# ── Ensure cv2 is available ──────────────────────────────────────────
cv2 = pytest.importorskip("cv2")

# ── Direct module loading (bypass Isaac Lab init chain) ──────────────
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


# Load sim_detector (needed for SimDetection)
_sim_det_mod = _load_module(
    "perception.sim_detector",
    os.path.join(_PERC_DIR, "sim_detector.py"),
)
SimDetection = _sim_det_mod.SimDetection

# Load depth_viz module
_depth_viz_mod = _load_module(
    "perception.debug.depth_viz",
    os.path.join(_PERC_DIR, "debug", "depth_viz.py"),
)
DepthFrameVisualizer = _depth_viz_mod.DepthFrameVisualizer
VizConfig = _depth_viz_mod.VizConfig
_SimDetAsDetection = _depth_viz_mod._SimDetAsDetection


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return VizConfig(width=320, height=240)


@pytest.fixture
def viz(cfg):
    return DepthFrameVisualizer(cfg)


@pytest.fixture
def depth_u16():
    """Realistic depth frame: ball at ~500mm, background at ~1000mm."""
    frame = np.full((480, 640), 1000, dtype=np.uint16)  # background
    # Ball blob at centre
    frame[235:245, 315:325] = 500
    return frame


@pytest.fixture
def depth_f32():
    """Float32 depth frame in metres."""
    frame = np.full((480, 640), 1.0, dtype=np.float32)
    frame[235:245, 315:325] = 0.5
    return frame


class _FakeDetection:
    """Minimal Detection-like object."""

    def __init__(self, pos_cam, confidence, bbox, method="yolo"):
        self.pos_cam = pos_cam
        self.confidence = confidence
        self.bbox = bbox
        self.method = method


# ── Tests ────────────────────────────────────────────────────────────

class TestOutputShape:
    def test_render_u16_output_shape(self, viz, depth_u16, cfg):
        panel = viz.render(depth_u16)
        assert panel.shape == (cfg.height, cfg.width, 3)
        assert panel.dtype == np.uint8

    def test_render_f32_output_shape(self, viz, depth_f32, cfg):
        panel = viz.render_f32(depth_f32)
        assert panel.shape == (cfg.height, cfg.width, 3)
        assert panel.dtype == np.uint8

    def test_custom_output_size(self, depth_u16):
        small_cfg = VizConfig(width=160, height=120)
        v = DepthFrameVisualizer(small_cfg)
        panel = v.render(depth_u16)
        assert panel.shape == (120, 160, 3)


class TestColorization:
    def test_valid_depth_not_black(self, viz, depth_u16):
        panel = viz.render(depth_u16)
        assert panel.sum() > 0

    def test_all_invalid_depth(self, viz):
        frame = np.zeros((480, 640), dtype=np.uint16)  # all zero = invalid
        panel = viz.render(frame)
        assert panel.shape == (240, 320, 3)

    def test_all_max_depth(self, viz):
        frame = np.full((480, 640), 60000, dtype=np.uint16)  # beyond max
        panel = viz.render(frame)
        assert panel.shape == (240, 320, 3)


class TestDetectionOverlay:
    def test_bbox_drawn(self, viz, depth_u16):
        det = _FakeDetection(
            pos_cam=np.array([0.0, 0.0, 0.5]),
            confidence=0.95,
            bbox=(310, 230, 330, 250),
        )
        panel_no_det = viz.render(depth_u16)
        panel_det = viz.render(depth_u16, detection=det)
        assert not np.array_equal(panel_no_det, panel_det)

    def test_marker_color(self, viz, depth_u16):
        det = _FakeDetection(
            pos_cam=np.array([0.0, 0.0, 0.5]),
            confidence=0.85,
            bbox=(310, 230, 330, 250),
        )
        panel = viz.render(depth_u16, detection=det)
        assert panel[:, :, 2].max() > 200

    def test_no_detection_indicator(self, viz, depth_u16):
        panel = viz.render(depth_u16, detection=None)
        assert panel[:, :, 2].max() > 200


class TestSimDetectionAdapter:
    def test_adapter_fields(self):
        sim_det = SimDetection(
            pos_cam=np.array([0.05, -0.03, 0.4]),
            pixel_uv=(320.0, 240.0),
            depth_m=0.4,
            num_pixels=80,
        )
        wrapped = _SimDetAsDetection(sim_det)
        assert wrapped.confidence == 1.0
        assert wrapped.method == "sim"
        np.testing.assert_array_equal(wrapped.pos_cam, sim_det.pos_cam)
        x1, y1, x2, y2 = wrapped.bbox
        assert x1 < 320 < x2
        assert y1 < 240 < y2

    def test_render_with_sim_detection(self, viz, depth_f32):
        sim_det = SimDetection(
            pos_cam=np.array([0.0, 0.0, 0.5]),
            pixel_uv=(320.0, 240.0),
            depth_m=0.5,
            num_pixels=100,
        )
        panel = viz.render_f32(depth_f32, sim_detection=sim_det)
        assert panel.shape == (240, 320, 3)


class TestFloat32Conversion:
    def test_f32_output_valid(self, viz, depth_f32):
        panel = viz.render_f32(depth_f32)
        assert panel.shape == (240, 320, 3)
        assert panel.dtype == np.uint8

    def test_nan_handling(self, viz):
        frame = np.full((480, 640), np.nan, dtype=np.float32)
        panel = viz.render_f32(frame)
        assert panel.shape == (240, 320, 3)

    def test_negative_depth(self, viz):
        frame = np.full((480, 640), -1.0, dtype=np.float32)
        panel = viz.render_f32(frame)
        assert panel.shape == (240, 320, 3)


class TestTelemetry:
    def test_telemetry_overlay(self, viz, depth_u16):
        telem = {"depth": "0.50m", "conf": "95%", "method": "YOLO"}
        panel = viz.render(depth_u16, telemetry=telem)
        assert panel.shape == (240, 320, 3)

    def test_telemetry_disabled(self, depth_u16):
        cfg = VizConfig(show_telemetry=False)
        v = DepthFrameVisualizer(cfg)
        panel1 = v.render(depth_u16, telemetry={"x": "y"})
        panel2 = v.render(depth_u16)
        np.testing.assert_array_equal(panel1, panel2)


class TestEdgeCases:
    def test_tiny_frame(self, viz):
        frame = np.array([[500]], dtype=np.uint16)
        panel = viz.render(frame)
        assert panel.shape == (240, 320, 3)

    def test_large_frame(self, viz):
        frame = np.full((1080, 1920), 800, dtype=np.uint16)
        panel = viz.render(frame)
        assert panel.shape == (240, 320, 3)

    def test_bbox_outside_frame(self, viz, depth_u16):
        det = _FakeDetection(
            pos_cam=np.array([0.0, 0.0, 0.5]),
            confidence=0.9,
            bbox=(-10, -10, 650, 490),
        )
        panel = viz.render(depth_u16, detection=det)
        assert panel.shape == (240, 320, 3)

    def test_frame_count_increments(self, viz, depth_u16):
        assert viz._frame_count == 0
        viz.render(depth_u16)
        assert viz._frame_count == 1
        viz.render(depth_u16)
        assert viz._frame_count == 2
