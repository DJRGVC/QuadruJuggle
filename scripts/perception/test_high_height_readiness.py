"""Tests for high-height readiness analysis.

Validates that the D435i noise model + EKF pipeline produces accurate
ball tracking at juggling heights (0.50-1.00m apex). Key criteria:
  - Apex position error < 10mm (pi1 needs height for energy modulation)
  - Flight RMSE < 60mm (reasonable for fast transit with dropout)
  - Detection rate > 50% during flight (enough for EKF convergence)
  - Camera visible fraction increases with height (expected: more time above FOV)
"""

from __future__ import annotations

import os
import sys
import unittest

# Add scripts dir to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from analyze_high_height_readiness import (
    TrajectoryResult,
    simulate_trajectory,
    vz0_for_apex,
    ballistic_trajectory,
    CAMERA_MIN_HEIGHT,
    GRAVITY,
)

import numpy as np


class TestVz0Calculation(unittest.TestCase):
    """Test initial velocity calculation."""

    def test_vz0_for_half_metre(self):
        vz = vz0_for_apex(0.50)
        # v = sqrt(2*9.81*0.5) ≈ 3.13 m/s
        self.assertAlmostEqual(vz, 3.13, places=1)

    def test_vz0_for_one_metre(self):
        vz = vz0_for_apex(1.00)
        # v = sqrt(2*9.81*1.0) ≈ 4.43 m/s
        self.assertAlmostEqual(vz, 4.43, places=1)

    def test_vz0_increases_with_height(self):
        self.assertGreater(vz0_for_apex(1.0), vz0_for_apex(0.5))
        self.assertGreater(vz0_for_apex(0.5), vz0_for_apex(0.2))


class TestBallisticTrajectory(unittest.TestCase):
    """Test trajectory generation."""

    def test_apex_height_approximately_correct(self):
        """Trajectory apex should be close to target (drag reduces it slightly)."""
        for target in [0.50, 0.70, 1.00]:
            vz0 = vz0_for_apex(target)
            pos0 = np.array([0.0, 0.0, 0.025])
            vel0 = np.array([0.0, 0.0, vz0])
            n = int(2.0 * vz0 / abs(GRAVITY) / 0.02 * 1.2) + 10
            gt_pos, _ = ballistic_trajectory(pos0, vel0, 0.02, n)
            actual_apex = gt_pos[:, 2].max()
            # Drag reduces apex; should be within 15% of target
            self.assertGreater(actual_apex, target * 0.80,
                               f"Apex {actual_apex:.3f} too low for target {target}")
            self.assertLess(actual_apex, target * 1.05,
                            f"Apex {actual_apex:.3f} too high for target {target}")

    def test_trajectory_returns_to_ground(self):
        """Ball should descend back to paddle level."""
        vz0 = vz0_for_apex(0.50)
        pos0 = np.array([0.0, 0.0, 0.025])
        vel0 = np.array([0.0, 0.0, vz0])
        gt_pos, _ = ballistic_trajectory(pos0, vel0, 0.02, 100)
        # Should have values below 0.025 after going up
        after_apex = gt_pos[20:, 2]
        self.assertTrue(np.any(after_apex < 0.025),
                        "Ball never returns below paddle")


class TestSimulateTrajectory(unittest.TestCase):
    """Integration tests for full D435i + EKF simulation."""

    def test_apex_error_under_10mm(self):
        """Apex position error should be < 10mm at all heights."""
        for target in [0.50, 0.70, 1.00]:
            r = simulate_trajectory(target, seed=42)
            self.assertLess(r.apex_pos_err, 0.010,
                            f"Apex error {r.apex_pos_err*1000:.1f}mm > 10mm at {target}m")

    def test_flight_rmse_under_60mm(self):
        """Flight RMSE should be < 60mm (reasonable for fast transit + dropout)."""
        for target in [0.50, 0.70, 1.00]:
            r = simulate_trajectory(target, seed=42)
            self.assertLess(r.pos_rmse_flight, 0.060,
                            f"Flight RMSE {r.pos_rmse_flight*1000:.1f}mm > 60mm at {target}m")

    def test_detection_rate_above_50_pct(self):
        """Detection rate during flight should be > 50%."""
        for target in [0.50, 0.70, 1.00]:
            r = simulate_trajectory(target, seed=42)
            self.assertGreater(r.detection_rate, 0.50,
                               f"Detection rate {r.detection_rate*100:.1f}% < 50% at {target}m")

    def test_camera_visibility_increases_with_height(self):
        """Higher targets should have larger camera-visible windows."""
        results = [simulate_trajectory(t, seed=42) for t in [0.50, 0.70, 1.00]]
        for i in range(len(results) - 1):
            self.assertGreater(results[i + 1].camera_visible_frac,
                               results[i].camera_visible_frac,
                               f"Camera visible didn't increase from {results[i].target_apex}m "
                               f"to {results[i+1].target_apex}m")

    def test_dropout_increases_with_height(self):
        """D435i dropout should increase with ball height (as expected)."""
        results = [simulate_trajectory(t, seed=42) for t in [0.50, 0.70, 1.00]]
        for i in range(len(results) - 1):
            self.assertGreater(results[i + 1].dropout_apex,
                               results[i].dropout_apex,
                               f"Dropout didn't increase with height")

    def test_actual_apex_less_than_target(self):
        """Actual apex should be less than target (drag effect)."""
        for target in [0.50, 0.70, 1.00]:
            r = simulate_trajectory(target, seed=42)
            self.assertLess(r.actual_apex, target,
                            f"Actual apex {r.actual_apex:.3f} >= target {target}")

    def test_result_fields_populated(self):
        """All TrajectoryResult fields should have valid values."""
        r = simulate_trajectory(0.50, seed=42)
        self.assertGreater(r.flight_steps, 10)
        self.assertGreater(r.total_steps, r.flight_steps)
        self.assertEqual(len(r.gt_z), r.total_steps)
        self.assertEqual(len(r.ekf_z), r.total_steps)
        self.assertEqual(len(r.time_s), r.total_steps)
        self.assertFalse(np.isnan(r.pos_rmse_flight))
        self.assertFalse(np.isnan(r.vel_rmse_flight))


class TestMultiSeedRobustness(unittest.TestCase):
    """Verify results are robust across different random seeds."""

    def test_apex_error_robust_across_seeds(self):
        """Apex error should be < 15mm for all seeds at 0.70m."""
        errors = []
        for seed in range(5):
            r = simulate_trajectory(0.70, seed=seed)
            errors.append(r.apex_pos_err)
            self.assertLess(r.apex_pos_err, 0.015,
                            f"Apex error {r.apex_pos_err*1000:.1f}mm > 15mm at seed {seed}")
        mean_err = np.mean(errors) * 1000
        self.assertLess(mean_err, 10.0,
                        f"Mean apex error {mean_err:.1f}mm > 10mm across seeds")


if __name__ == "__main__":
    unittest.main()
