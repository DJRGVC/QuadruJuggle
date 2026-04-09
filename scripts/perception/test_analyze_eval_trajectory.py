"""Tests for analyze_eval_trajectory.py — npz-based camera eval analysis."""
import os
import tempfile

import numpy as np
import pytest

from analyze_eval_trajectory import (
    compute_height_binned,
    compute_overall_metrics,
    compute_phase_metrics,
    load_npz,
)

# Paddle Z for test data
_PADDLE_Z = 0.47


def _make_trajectory(n_steps=200, n_det=50, seed=42):
    """Create a synthetic trajectory.npz dict for testing."""
    rng = np.random.default_rng(seed)
    # Ball oscillates vertically above paddle
    t = np.arange(n_steps)
    gt_z = _PADDLE_Z + 0.05 + 0.3 * np.abs(np.sin(t * 0.05))  # 0.05-0.35m above paddle
    gt = np.column_stack([rng.normal(0, 0.01, n_steps),
                          rng.normal(0, 0.01, n_steps),
                          gt_z])
    # EKF has small tracking error
    ekf = gt + rng.normal(0, 0.005, gt.shape)
    # Detections at random steps with larger error
    det_steps = np.sort(rng.choice(n_steps, n_det, replace=False))
    det = gt[det_steps] + rng.normal(0, 0.02, (n_det, 3))
    return {
        "gt": gt,
        "ekf": ekf,
        "det": det,
        "det_steps": det_steps.astype(float),
        "steps": np.arange(n_steps, dtype=float),
        "dt": np.array(0.02),
        "rmse_ekf": np.zeros(0),
        "rmse_det": np.zeros(0),
    }


def _save_npz(traj: dict, path: str):
    np.savez_compressed(path, **traj)


class TestLoadNpz:
    def test_load_valid(self, tmp_path):
        traj = _make_trajectory()
        p = str(tmp_path / "trajectory.npz")
        _save_npz(traj, p)
        loaded = load_npz(p)
        assert "gt" in loaded
        assert loaded["gt"].shape == (200, 3)

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_npz(str(tmp_path / "nonexistent.npz"))


class TestOverallMetrics:
    def test_basic_metrics(self):
        traj = _make_trajectory(n_steps=100, n_det=25)
        m = compute_overall_metrics(traj)
        assert m["total_steps"] == 100
        assert m["n_detections"] == 25
        assert abs(m["det_rate_pct"] - 25.0) < 0.1
        assert m["ekf_rmse_mm"] > 0
        assert m["det_rmse_mm"] > 0
        # EKF should be more accurate than raw detection
        assert m["ekf_rmse_mm"] < m["det_rmse_mm"]

    def test_no_detections(self):
        traj = _make_trajectory(n_steps=100, n_det=0)
        # Fix: n_det=0 means empty arrays
        traj["det"] = np.zeros((0, 3))
        traj["det_steps"] = np.zeros((0,))
        m = compute_overall_metrics(traj)
        assert m["n_detections"] == 0
        assert m["det_rate_pct"] == 0.0
        assert np.isnan(m["det_rmse_mm"])

    def test_perfect_ekf(self):
        traj = _make_trajectory()
        traj["ekf"] = traj["gt"].copy()  # perfect tracking
        m = compute_overall_metrics(traj)
        assert m["ekf_rmse_mm"] < 0.01  # effectively zero


class TestHeightBinned:
    def test_bins_cover_data(self):
        traj = _make_trajectory()
        hb = compute_height_binned(traj, paddle_z=_PADDLE_Z)
        assert hb["count"].sum() == traj["gt"].shape[0]

    def test_detection_rate_bounded(self):
        traj = _make_trajectory()
        hb = compute_height_binned(traj, paddle_z=_PADDLE_Z)
        valid = hb["count"] > 0
        assert np.all(hb["det_rate"][valid] >= 0)
        assert np.all(hb["det_rate"][valid] <= 100)

    def test_empty_bins_are_nan(self):
        traj = _make_trajectory(n_steps=50)
        # With high max_height, some bins will be empty
        hb = compute_height_binned(traj, paddle_z=_PADDLE_Z, max_height=2.0)
        empty = hb["count"] == 0
        assert empty.any()
        assert np.all(np.isnan(hb["ekf_rmse"][empty]))

    def test_custom_bin_width(self):
        traj = _make_trajectory()
        hb = compute_height_binned(traj, paddle_z=_PADDLE_Z, bin_width=0.05, max_height=0.40)
        # More bins with finer width
        assert len(hb["centres"]) == 8  # 0.0-0.40 in 0.05 steps

    def test_ekf_more_accurate_than_det(self):
        traj = _make_trajectory(n_steps=500, n_det=200)
        hb = compute_height_binned(traj, paddle_z=_PADDLE_Z)
        valid = (hb["count"] > 10) & ~np.isnan(hb["ekf_rmse"]) & ~np.isnan(hb["det_rmse"])
        if valid.any():
            # On average, EKF should beat raw detection
            assert np.mean(hb["ekf_rmse"][valid]) < np.mean(hb["det_rmse"][valid])


class TestPlotting:
    """Smoke tests for plot functions — just verify they don't crash."""

    def test_plot_single(self, tmp_path):
        traj = _make_trajectory()
        hb = compute_height_binned(traj, paddle_z=_PADDLE_Z)
        m = compute_overall_metrics(traj)
        out = str(tmp_path / "test_single.png")
        from analyze_eval_trajectory import plot_single
        plot_single(traj, hb, m, out, "Test")
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 1000

    def test_plot_comparison(self, tmp_path):
        traj1 = _make_trajectory(seed=42)
        traj2 = _make_trajectory(seed=99)
        hb1 = compute_height_binned(traj1, paddle_z=_PADDLE_Z)
        hb2 = compute_height_binned(traj2, paddle_z=_PADDLE_Z)
        m1 = compute_overall_metrics(traj1)
        m2 = compute_overall_metrics(traj2)
        out = str(tmp_path / "test_compare.png")
        from analyze_eval_trajectory import plot_comparison
        plot_comparison([traj1, traj2], [hb1, hb2], [m1, m2],
                        ["Run A", "Run B"], out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 1000


class TestPhaseMetrics:
    """Tests for ascending/descending/contact phase analysis."""

    def test_all_phases_present(self):
        traj = _make_trajectory()
        ph = compute_phase_metrics(traj, paddle_z=_PADDLE_Z)
        assert "ascending" in ph
        assert "descending" in ph
        assert "contact" in ph

    def test_phase_counts_sum_to_total(self):
        traj = _make_trajectory(n_steps=300)
        ph = compute_phase_metrics(traj, paddle_z=_PADDLE_Z)
        total = sum(ph[p]["count"] for p in ("ascending", "descending", "contact"))
        assert total == traj["gt"].shape[0]

    def test_oscillating_ball_has_both_phases(self):
        """Sinusoidal trajectory should have roughly equal ascending/descending."""
        traj = _make_trajectory(n_steps=500, n_det=200)
        ph = compute_phase_metrics(traj, paddle_z=_PADDLE_Z)
        # Ball oscillates — both phases should have substantial counts
        assert ph["ascending"]["count"] > 50
        assert ph["descending"]["count"] > 50

    def test_stationary_ball_all_contact(self):
        """Ball sitting on paddle should be classified as contact."""
        n = 100
        gt = np.column_stack([
            np.zeros(n), np.zeros(n),
            np.full(n, _PADDLE_Z + 0.01),  # 10mm above paddle = within contact threshold
        ])
        ekf = gt.copy()
        traj = {
            "gt": gt, "ekf": ekf,
            "det": np.zeros((0, 3)), "det_steps": np.zeros(0),
            "dt": np.array(0.02),
        }
        ph = compute_phase_metrics(traj, paddle_z=_PADDLE_Z, contact_threshold=0.025)
        assert ph["contact"]["count"] == n
        assert ph["ascending"]["count"] == 0
        assert ph["descending"]["count"] == 0

    def test_det_rate_bounded(self):
        traj = _make_trajectory(n_steps=200, n_det=100)
        ph = compute_phase_metrics(traj, paddle_z=_PADDLE_Z)
        for name in ("ascending", "descending", "contact"):
            if ph[name]["count"] > 0:
                assert 0 <= ph[name]["det_rate_pct"] <= 100

    def test_ekf_rmse_positive(self):
        traj = _make_trajectory()
        ph = compute_phase_metrics(traj, paddle_z=_PADDLE_Z)
        for name in ("ascending", "descending"):
            if ph[name]["count"] > 0:
                assert ph[name]["ekf_rmse_mm"] > 0

    def test_perfect_ekf_phase(self):
        """Perfect EKF should have near-zero RMSE in all phases."""
        traj = _make_trajectory()
        traj["ekf"] = traj["gt"].copy()
        ph = compute_phase_metrics(traj, paddle_z=_PADDLE_Z)
        for name in ("ascending", "descending", "contact"):
            if ph[name]["count"] > 0:
                assert ph[name]["ekf_rmse_mm"] < 0.01
