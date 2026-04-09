"""Tests for analyze_ekf_vs_raw.py — height-binned EKF vs raw detection analysis."""

import numpy as np
import pytest

from analyze_ekf_vs_raw import compute_height_binned_metrics, _PADDLE_Z


def _make_trajectory(n_steps=200, n_det=150, paddle_z=_PADDLE_Z):
    """Create synthetic trajectory data for testing.

    Ball oscillates sinusoidally between paddle_z and paddle_z + 0.6m.
    EKF tracks GT with small noise; raw detections have height-dependent noise.
    """
    t = np.linspace(0, 4 * np.pi, n_steps)
    gt_z = paddle_z + 0.3 + 0.3 * np.sin(t)  # 0 to 0.6m above paddle
    gt = np.column_stack([np.zeros(n_steps), np.zeros(n_steps), gt_z])

    # EKF: small constant noise
    rng = np.random.default_rng(42)
    ekf_noise = rng.normal(0, 0.01, (n_steps, 3))
    ekf = gt + ekf_noise

    # Detections: subsample, with height-dependent noise (worse at distance)
    det_steps = np.sort(rng.choice(n_steps, n_det, replace=False))
    height_above_paddle = gt_z[det_steps] - paddle_z
    det_noise_std = 0.01 + 0.3 * height_above_paddle  # noise grows with height
    det_noise = rng.normal(0, 1, (n_det, 3)) * det_noise_std[:, None]
    det = gt[det_steps] + det_noise

    rmse_ekf = np.linalg.norm(ekf - gt, axis=1)
    rmse_det = np.linalg.norm(det - gt[det_steps], axis=1)

    return {
        "gt": gt, "ekf": ekf, "steps": np.arange(n_steps),
        "det": det, "det_steps": det_steps,
        "rmse_ekf": rmse_ekf, "rmse_det": rmse_det, "dt": 0.02,
    }


class TestHeightBinnedMetrics:
    def test_basic_output_shape(self):
        traj = _make_trajectory()
        bins = np.arange(0.0, 0.75, 0.10)
        results = compute_height_binned_metrics(traj, bins)
        n_bins = len(bins) - 1
        assert len(results["bin_centres"]) == n_bins
        assert len(results["ekf_rmse"]) == n_bins
        assert len(results["det_rmse"]) == n_bins
        assert len(results["det_rate"]) == n_bins
        assert len(results["count"]) == n_bins

    def test_all_steps_accounted(self):
        traj = _make_trajectory()
        bins = np.arange(-0.1, 0.85, 0.10)  # wide enough to capture all
        results = compute_height_binned_metrics(traj, bins)
        assert results["count"].sum() == len(traj["gt"])

    def test_ekf_beats_raw_at_high_height(self):
        """EKF (low constant noise) should beat raw (height-dependent noise) at high heights."""
        traj = _make_trajectory(n_steps=2000, n_det=1500)
        bins = np.arange(0.0, 0.75, 0.10)
        results = compute_height_binned_metrics(traj, bins)

        # At the highest populated bin, EKF RMSE should be lower than det RMSE
        valid = results["count"] > 10
        if valid.any():
            highest_valid = np.where(valid)[0][-1]
            ekf_r = results["ekf_rmse"][highest_valid]
            det_r = results["det_rmse"][highest_valid]
            if not np.isnan(det_r):
                assert ekf_r < det_r, f"EKF ({ekf_r:.4f}) should beat raw ({det_r:.4f}) at high height"

    def test_detection_rate_bounded(self):
        traj = _make_trajectory()
        bins = np.arange(0.0, 0.75, 0.10)
        results = compute_height_binned_metrics(traj, bins)
        valid = ~np.isnan(results["det_rate"])
        assert np.all(results["det_rate"][valid] >= 0)
        assert np.all(results["det_rate"][valid] <= 1.0)

    def test_empty_bins_are_nan(self):
        traj = _make_trajectory()
        # Use bins way above the trajectory
        bins = np.array([2.0, 3.0, 4.0])
        results = compute_height_binned_metrics(traj, bins)
        assert results["count"].sum() == 0
        assert np.all(np.isnan(results["ekf_rmse"]))

    def test_no_detections(self):
        traj = _make_trajectory()
        traj["det"] = np.zeros((0, 3))
        traj["det_steps"] = np.zeros((0,), dtype=int)
        traj["rmse_det"] = np.zeros((0,))
        bins = np.arange(0.0, 0.75, 0.10)
        results = compute_height_binned_metrics(traj, bins)
        # Detection rate should be 0 where we have steps
        valid = results["count"] > 0
        assert np.all(results["det_rate"][valid] == 0.0)
        assert np.all(np.isnan(results["det_rmse"][valid]))

    def test_custom_paddle_z(self):
        traj = _make_trajectory(paddle_z=0.50)
        bins = np.arange(0.0, 0.75, 0.10)
        results = compute_height_binned_metrics(traj, bins, paddle_z=0.50)
        assert results["count"].sum() > 0
