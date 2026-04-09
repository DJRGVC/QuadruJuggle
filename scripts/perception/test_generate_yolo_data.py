"""Tests for synthetic YOLO training data generator."""

from __future__ import annotations

import numpy as np
import pytest

from generate_yolo_data import (
    _DEFAULT_CX,
    _DEFAULT_CY,
    _DEFAULT_FX,
    _DEFAULT_FY,
    _DEFAULT_H,
    _DEFAULT_W,
    bbox_to_yolo,
    render_ball_depth_frame,
)


class TestRenderBallDepthFrame:
    """Tests for depth frame rendering."""

    def test_ball_at_centre_produces_valid_bbox(self):
        """Ball at optical axis should produce a bbox near frame centre."""
        depth, bbox = render_ball_depth_frame(
            ball_x=0.0, ball_y=0.0, ball_z=0.5,
            rng=np.random.default_rng(42),
        )
        assert bbox is not None
        x1, y1, x2, y2 = bbox
        # Bbox centre should be near optical centre
        cx_px = (x1 + x2) / 2
        cy_px = (y1 + y2) / 2
        assert abs(cx_px - _DEFAULT_CX) < 5
        assert abs(cy_px - _DEFAULT_CY) < 5

    def test_ball_size_scales_with_depth(self):
        """Closer ball should produce larger bbox."""
        _, bbox_close = render_ball_depth_frame(
            ball_x=0.0, ball_y=0.0, ball_z=0.3,
            rng=np.random.default_rng(42),
        )
        _, bbox_far = render_ball_depth_frame(
            ball_x=0.0, ball_y=0.0, ball_z=1.0,
            rng=np.random.default_rng(42),
        )
        assert bbox_close is not None and bbox_far is not None
        close_w = bbox_close[2] - bbox_close[0]
        far_w = bbox_far[2] - bbox_far[0]
        assert close_w > far_w * 2  # ~3.3x larger at 0.3 vs 1.0

    def test_ball_outside_frame_returns_none(self):
        """Ball far off-axis should return None bbox."""
        _, bbox = render_ball_depth_frame(
            ball_x=10.0, ball_y=10.0, ball_z=0.3,
            rng=np.random.default_rng(42),
        )
        assert bbox is None

    def test_output_shape_and_dtype(self):
        """Depth frame should be uint16 at expected resolution."""
        depth, _ = render_ball_depth_frame(
            ball_x=0.0, ball_y=0.0, ball_z=0.5,
            rng=np.random.default_rng(42),
        )
        assert depth.shape == (_DEFAULT_H, _DEFAULT_W)
        assert depth.dtype == np.uint16

    def test_ball_depth_is_approximately_correct(self):
        """Ball pixel depths should be near the specified depth."""
        z = 0.5  # 500mm
        depth, bbox = render_ball_depth_frame(
            ball_x=0.0, ball_y=0.0, ball_z=z,
            noise_sigma_base=0.0, noise_sigma_quad=0.0,  # no noise
            rng=np.random.default_rng(42),
        )
        assert bbox is not None
        x1, y1, x2, y2 = bbox
        roi = depth[y1:y2, x1:x2]
        ball_pixels = roi[roi > 0]
        assert len(ball_pixels) > 0
        median_mm = np.median(ball_pixels)
        # Should be within ~20mm of 500mm (ball surface is sphere, not flat)
        assert abs(median_mm - z * 1000) < 25

    def test_bbox_within_frame(self):
        """Bbox should be clamped to frame boundaries."""
        depth, bbox = render_ball_depth_frame(
            ball_x=0.0, ball_y=0.0, ball_z=0.5,
            rng=np.random.default_rng(42),
        )
        assert bbox is not None
        x1, y1, x2, y2 = bbox
        assert x1 >= 0 and y1 >= 0
        assert x2 <= _DEFAULT_W and y2 <= _DEFAULT_H

    def test_reproducible_with_seed(self):
        """Same seed should produce identical output."""
        d1, b1 = render_ball_depth_frame(
            ball_x=0.1, ball_y=-0.05, ball_z=0.7,
            rng=np.random.default_rng(123),
        )
        d2, b2 = render_ball_depth_frame(
            ball_x=0.1, ball_y=-0.05, ball_z=0.7,
            rng=np.random.default_rng(123),
        )
        np.testing.assert_array_equal(d1, d2)
        assert b1 == b2

    def test_background_has_invalid_pixels(self):
        """Background should contain zeros (invalid depth)."""
        depth, _ = render_ball_depth_frame(
            ball_x=0.0, ball_y=0.0, ball_z=0.5,
            rng=np.random.default_rng(42),
        )
        # ~30% of background should be invalid
        n_zero = np.count_nonzero(depth == 0)
        total = depth.size
        zero_frac = n_zero / total
        assert 0.15 < zero_frac < 0.50  # ~30% ± margin

    def test_depth_at_various_distances(self):
        """Ball should be detectable from 0.2 to 1.5m."""
        for z in [0.2, 0.4, 0.7, 1.0, 1.3]:
            _, bbox = render_ball_depth_frame(
                ball_x=0.0, ball_y=0.0, ball_z=z,
                rng=np.random.default_rng(42),
            )
            assert bbox is not None, f"Ball not detected at z={z}m"


class TestBboxToYolo:
    """Tests for YOLO format conversion."""

    def test_centre_of_frame(self):
        """Bbox at centre should produce cx≈0.5, cy≈0.5."""
        line = bbox_to_yolo((400, 200, 500, 300), 1000, 500)
        parts = line.split()
        assert parts[0] == "0"  # class
        cx, cy = float(parts[1]), float(parts[2])
        assert abs(cx - 0.45) < 0.01
        assert abs(cy - 0.50) < 0.01

    def test_full_frame_bbox(self):
        """Full frame bbox should produce cx=0.5, cy=0.5, w=1.0, h=1.0."""
        line = bbox_to_yolo((0, 0, 100, 100), 100, 100)
        parts = line.split()
        assert abs(float(parts[1]) - 0.5) < 1e-5
        assert abs(float(parts[2]) - 0.5) < 1e-5
        assert abs(float(parts[3]) - 1.0) < 1e-5
        assert abs(float(parts[4]) - 1.0) < 1e-5

    def test_normalised_range(self):
        """All YOLO values should be in [0, 1]."""
        line = bbox_to_yolo((10, 20, 50, 80), 848, 480)
        parts = line.split()
        for p in parts[1:]:
            v = float(p)
            assert 0.0 <= v <= 1.0


class TestGenerateDataset:
    """Integration tests for the full generation pipeline."""

    def test_small_batch(self, tmp_path):
        """Generate a small batch and verify output structure."""
        from generate_yolo_data import generate_dataset

        out = str(tmp_path / "yolo_test")
        stats = generate_dataset(out, n_images=20, seed=42)

        assert stats["valid"] > 0
        assert stats["valid"] + stats["skipped"] == stats["total"]

        # Check output files exist
        from pathlib import Path
        img_dir = Path(out) / "images"
        lbl_dir = Path(out) / "labels"
        assert img_dir.exists()
        assert lbl_dir.exists()

        # Each valid image should have a corresponding label
        imgs = sorted(img_dir.glob("*.png"))
        lbls = sorted(lbl_dir.glob("*.txt"))
        assert len(imgs) == stats["valid"]
        assert len(lbls) == stats["valid"]

        # Check dataset.yaml
        yaml_path = Path(out) / "dataset.yaml"
        assert yaml_path.exists()
        content = yaml_path.read_text()
        assert "nc: 1" in content
        assert "ball" in content

    def test_label_format(self, tmp_path):
        """Labels should be valid YOLO format."""
        from generate_yolo_data import generate_dataset
        from pathlib import Path

        out = str(tmp_path / "yolo_fmt")
        generate_dataset(out, n_images=10, seed=99)

        for lbl_path in (Path(out) / "labels").glob("*.txt"):
            text = lbl_path.read_text().strip()
            parts = text.split()
            assert len(parts) == 5, f"Bad label: {text}"
            assert parts[0] == "0"  # class 0
            for p in parts[1:]:
                v = float(p)
                assert 0.0 <= v <= 1.0, f"Out of range: {v} in {text}"
