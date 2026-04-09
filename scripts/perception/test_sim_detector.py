"""Tests for SimBallDetector."""

import importlib
import os
import sys
import unittest

import numpy as np

# Direct import of the module without triggering go1_ball_balance.__init__ (needs pxr/Isaac Lab)
_PERCEPTION_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "source", "go1_ball_balance", "go1_ball_balance", "perception")
)
_spec = importlib.util.spec_from_file_location("sim_detector", os.path.join(_PERCEPTION_DIR, "sim_detector.py"))
_mod = importlib.util.module_from_spec(_spec)
sys.modules["sim_detector"] = _mod  # register so dataclass can find the module
_spec.loader.exec_module(_mod)
SimBallDetector = _mod.SimBallDetector
SimDetection = _mod.SimDetection


class TestSimBallDetector(unittest.TestCase):
    """Unit tests for SimBallDetector."""

    def _make_detector(self, **kwargs) -> SimBallDetector:
        return SimBallDetector.from_tiled_camera_cfg(**kwargs)

    def _make_depth_with_ball(
        self,
        ball_depth: float = 0.5,
        ball_u: float = 320.0,
        ball_v: float = 240.0,
        ball_radius_px: float = 14.0,
        width: int = 640,
        height: int = 480,
    ) -> np.ndarray:
        """Create a synthetic depth image with a ball blob."""
        depth = np.full((height, width), np.inf, dtype=np.float32)
        # Draw circular blob at ball_depth
        yy, xx = np.ogrid[:height, :width]
        mask = ((xx - ball_u) ** 2 + (yy - ball_v) ** 2) <= ball_radius_px ** 2
        depth[mask] = ball_depth
        return depth

    def test_detect_centered_ball(self):
        """Ball at image centre, 0.5m depth."""
        det = SimBallDetector.from_tiled_camera_cfg()
        # Expected pixel radius at 0.5m: fx * 0.02 / 0.5
        fx = 11.24 / 20.955 * 640
        r_px = fx * 0.02 / 0.5

        depth = self._make_depth_with_ball(ball_depth=0.5, ball_radius_px=r_px)
        result = det.detect(depth)

        self.assertIsNotNone(result)
        self.assertIsInstance(result, SimDetection)
        # Centre should be near (320, 240) → camera X,Y ≈ 0
        self.assertAlmostEqual(result.pos_cam[0], 0.0, places=1)
        self.assertAlmostEqual(result.pos_cam[1], 0.0, places=1)
        self.assertAlmostEqual(result.pos_cam[2], 0.5, places=1)

    def test_detect_off_centre_ball(self):
        """Ball offset from centre — check deprojection."""
        det = SimBallDetector.from_tiled_camera_cfg()
        fx = 11.24 / 20.955 * 640
        ball_depth = 0.3
        ball_u, ball_v = 400.0, 300.0
        r_px = fx * 0.02 / ball_depth

        depth = self._make_depth_with_ball(
            ball_depth=ball_depth, ball_u=ball_u, ball_v=ball_v, ball_radius_px=r_px
        )
        result = det.detect(depth)

        self.assertIsNotNone(result)
        # Expected X = (400 - 320) * 0.3 / fx
        expected_x = (ball_u - 320) * ball_depth / fx
        expected_y = (ball_v - 240) * ball_depth / fx
        self.assertAlmostEqual(result.pos_cam[0], expected_x, places=2)
        self.assertAlmostEqual(result.pos_cam[1], expected_y, places=2)
        self.assertAlmostEqual(result.pos_cam[2], ball_depth, places=2)

    def test_no_ball_empty_depth(self):
        """All-inf depth → no detection."""
        det = self._make_detector()
        depth = np.full((480, 640), np.inf, dtype=np.float32)
        result = det.detect(depth)
        self.assertIsNone(result)

    def test_no_ball_all_far(self):
        """All depth > max_depth → no detection."""
        det = self._make_detector(max_depth=2.0)
        depth = np.full((480, 640), 5.0, dtype=np.float32)
        result = det.detect(depth)
        self.assertIsNone(result)

    def test_ball_with_ground_plane(self):
        """Ball + ground plane — should detect ball (closer, smaller)."""
        det = self._make_detector()
        fx = 11.24 / 20.955 * 640
        ball_depth = 0.4
        r_px = fx * 0.02 / ball_depth

        # Ground plane at bottom of image, far depth
        depth = np.full((480, 640), np.inf, dtype=np.float32)
        depth[400:480, :] = 2.5  # ground at 2.5m (beyond max_depth=2.0 → filtered out)

        # Ball blob
        yy, xx = np.ogrid[:480, :640]
        ball_mask = ((xx - 320) ** 2 + (yy - 200) ** 2) <= r_px ** 2
        depth[ball_mask] = ball_depth

        result = det.detect(depth)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.depth_m, ball_depth, places=1)

    def test_ball_close_range(self):
        """Ball at 0.15m (close range) — large in image."""
        det = self._make_detector(min_depth=0.10)
        fx = 11.24 / 20.955 * 640
        ball_depth = 0.15
        r_px = fx * 0.02 / ball_depth  # ~45 pixels

        depth = self._make_depth_with_ball(ball_depth=ball_depth, ball_radius_px=r_px)
        result = det.detect(depth)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.depth_m, ball_depth, places=1)

    def test_from_tiled_camera_cfg(self):
        """Factory method produces correct focal length."""
        det = SimBallDetector.from_tiled_camera_cfg(
            focal_length_cm=11.24, horizontal_aperture_cm=20.955, width=640
        )
        expected_fx = 11.24 / 20.955 * 640
        self.assertAlmostEqual(det._fx, expected_fx, places=2)

    def test_blob_too_small(self):
        """Tiny blob below min_blob_px → rejected."""
        det = self._make_detector(min_blob_px=10)
        depth = np.full((480, 640), np.inf, dtype=np.float32)
        # 2x2 pixel blob (area=4 < min_blob_px=10)
        depth[240:242, 320:322] = 0.5
        result = det.detect(depth)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
