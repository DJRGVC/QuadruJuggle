"""Tests for analyze_flight_windows.py."""

import os
import tempfile

import numpy as np
import pytest

from analyze_flight_windows import (
    FlightMetrics,
    PHASE_ASCENDING,
    PHASE_CONTACT,
    PHASE_DESCENDING,
    _count_flight_windows,
    analyze_trajectory,
    load_eval_dir,
)


def _make_trajectory_npz(
    path: str,
    n_steps: int = 100,
    n_flight: int = 10,
    n_det: int = 5,
    flight_start: int = 20,
    ball_z_contact: float = 0.48,
    ball_z_flight: float = 0.65,
):
    """Create a synthetic trajectory.npz for testing."""
    phase = np.full(n_steps, PHASE_CONTACT, dtype=np.int64)
    if n_flight > 0:
        half = n_flight // 2
        phase[flight_start : flight_start + half] = PHASE_ASCENDING
        phase[flight_start + half : flight_start + n_flight] = PHASE_DESCENDING

    gt = np.zeros((n_steps, 3), dtype=np.float32)
    gt[:, 2] = ball_z_contact
    gt[phase > 0, 2] = ball_z_flight

    # EKF tracks perfectly during contact, with some error during flight
    ekf = gt.copy()
    ekf[phase > 0, 0] += 0.05  # 5cm X error during flight

    # Detections during flight
    det_steps = np.linspace(flight_start, flight_start + n_flight - 1, n_det).astype(np.int64)
    det = np.zeros((n_det, 3), dtype=np.float64)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        phase=phase,
        gt=gt,
        ekf=ekf,
        det_steps=det_steps,
        det=det,
        ball_h=gt[:, 2],
        dt=0.02,
        rmse_det=np.zeros(n_det),
        rmse_ekf=np.zeros(n_steps, dtype=np.float32),
        sched_active=np.ones(n_steps, dtype=np.int64),
        steps=np.arange(n_steps),
        anchored_step=np.zeros(n_steps, dtype=np.int64),
    )


class TestCountFlightWindows:
    def test_no_flight(self):
        phase = np.zeros(50, dtype=np.int64)
        n, mean = _count_flight_windows(phase)
        assert n == 0
        assert mean == 0.0

    def test_single_window(self):
        phase = np.zeros(50, dtype=np.int64)
        phase[10:20] = PHASE_ASCENDING
        n, mean = _count_flight_windows(phase)
        assert n == 1
        assert mean == 10.0

    def test_two_windows(self):
        phase = np.zeros(50, dtype=np.int64)
        phase[5:10] = PHASE_ASCENDING
        phase[20:30] = PHASE_DESCENDING
        n, mean = _count_flight_windows(phase)
        assert n == 2
        assert mean == 7.5  # (5 + 10) / 2

    def test_mixed_phases_count_as_one(self):
        phase = np.zeros(50, dtype=np.int64)
        phase[5:8] = PHASE_ASCENDING
        phase[8:12] = PHASE_DESCENDING
        n, mean = _count_flight_windows(phase)
        assert n == 1
        assert mean == 7.0


class TestAnalyzeTrajectory:
    def test_basic_metrics(self, tmp_path):
        npz_path = str(tmp_path / "trajectory.npz")
        _make_trajectory_npz(npz_path, n_steps=100, n_flight=10, n_det=5)
        m = analyze_trajectory(npz_path, target=0.50)

        assert m.target == 0.50
        assert m.n_total == 100
        assert m.n_flight == 10
        assert m.n_contact == 90
        assert m.flight_fraction == pytest.approx(0.10)
        assert m.n_det_total == 5
        assert m.n_det_flight == 5
        assert m.n_det_contact == 0
        assert m.det_per_flight_step == pytest.approx(0.50)

    def test_rmse_flight_vs_contact(self, tmp_path):
        npz_path = str(tmp_path / "trajectory.npz")
        _make_trajectory_npz(npz_path)
        m = analyze_trajectory(npz_path, target=0.50)

        # Flight has 5cm X error, contact has 0 error
        assert m.rmse_flight > 0.04
        assert m.rmse_contact == pytest.approx(0.0, abs=1e-6)

    def test_no_flight(self, tmp_path):
        npz_path = str(tmp_path / "trajectory.npz")
        _make_trajectory_npz(npz_path, n_flight=0, n_det=0, flight_start=0)
        m = analyze_trajectory(npz_path, target=0.10)

        assert m.n_flight == 0
        assert m.flight_fraction == 0.0
        assert np.isnan(m.rmse_flight)
        assert np.isnan(m.mean_ball_z_flight)

    def test_height_metrics(self, tmp_path):
        npz_path = str(tmp_path / "trajectory.npz")
        _make_trajectory_npz(npz_path, ball_z_contact=0.48, ball_z_flight=0.70)
        m = analyze_trajectory(npz_path, target=0.50)

        assert m.mean_ball_z_flight == pytest.approx(0.70, abs=0.01)
        assert m.mean_ball_z_contact == pytest.approx(0.48, abs=0.01)

    def test_flight_windows_count(self, tmp_path):
        npz_path = str(tmp_path / "trajectory.npz")
        _make_trajectory_npz(npz_path)
        m = analyze_trajectory(npz_path, target=0.50)

        assert m.n_flight_windows == 1
        assert m.mean_window_length == 10.0


class TestLoadEvalDir:
    def test_loads_multiple_targets(self, tmp_path):
        for target_str in ["target_0_10", "target_0_50", "target_1_00"]:
            _make_trajectory_npz(str(tmp_path / target_str / "trajectory.npz"))

        results = load_eval_dir(str(tmp_path))
        assert len(results) == 3
        assert results[0].target == pytest.approx(0.10)
        assert results[1].target == pytest.approx(0.50)
        assert results[2].target == pytest.approx(1.00)

    def test_empty_dir(self, tmp_path):
        results = load_eval_dir(str(tmp_path))
        assert len(results) == 0

    def test_skips_non_target_dirs(self, tmp_path):
        _make_trajectory_npz(str(tmp_path / "target_0_50" / "trajectory.npz"))
        os.makedirs(str(tmp_path / "other_dir"))
        results = load_eval_dir(str(tmp_path))
        assert len(results) == 1


class TestPlot:
    def test_plot_single(self, tmp_path):
        for t in ["target_0_10", "target_0_50"]:
            _make_trajectory_npz(str(tmp_path / "eval_a" / t / "trajectory.npz"))

        from analyze_flight_windows import plot_comparison

        metrics = load_eval_dir(str(tmp_path / "eval_a"))
        out = str(tmp_path / "test_plot.png")
        plot_comparison(metrics, None, "Test", None, out)
        assert os.path.isfile(out)

    def test_plot_comparison(self, tmp_path):
        for t in ["target_0_10", "target_0_50"]:
            _make_trajectory_npz(str(tmp_path / "eval_a" / t / "trajectory.npz"))
            _make_trajectory_npz(str(tmp_path / "eval_b" / t / "trajectory.npz"), n_flight=20)

        from analyze_flight_windows import plot_comparison

        metrics_a = load_eval_dir(str(tmp_path / "eval_a"))
        metrics_b = load_eval_dir(str(tmp_path / "eval_b"))
        out = str(tmp_path / "test_compare.png")
        plot_comparison(metrics_a, metrics_b, "A", "B", out)
        assert os.path.isfile(out)
