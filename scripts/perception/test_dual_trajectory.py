"""Tests for plot_dual_trajectory.py."""

import os
import tempfile

import numpy as np
import pytest

from plot_dual_trajectory import compute_stats, discover_targets


def _make_trajectory_npz(path: str, n_steps: int = 100, n_det: int = 5,
                         flight_steps: tuple = (30, 60)) -> None:
    """Create a minimal trajectory.npz for testing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    gt = np.random.randn(n_steps, 3).astype(np.float32)
    gt[:, 2] = np.abs(gt[:, 2]) + 0.4  # z always positive
    ekf = gt + np.random.randn(n_steps, 3).astype(np.float32) * 0.01

    phase = np.zeros(n_steps, dtype=np.int64)
    if flight_steps:
        phase[flight_steps[0]:flight_steps[1]] = 1  # ascending

    det_steps = np.linspace(flight_steps[0], flight_steps[1], n_det).astype(np.int64)
    det = np.random.randn(n_det, 3)

    anchored = np.zeros(n_steps, dtype=np.int64)
    anchored[:flight_steps[0]] = 1

    ball_h = gt[:, 2] - 0.49
    sched_active = np.ones(n_steps, dtype=np.int64)

    np.savez(path, gt=gt, ekf=ekf, steps=np.arange(n_steps),
             dt=0.02, det=det, det_steps=det_steps,
             rmse_ekf=np.zeros(n_steps, dtype=np.float32),
             rmse_det=np.zeros(n_det),
             ball_h=ball_h, anchored_step=anchored,
             phase=phase, sched_active=sched_active)


class TestDiscoverTargets:
    def test_finds_targets(self, tmp_path):
        _make_trajectory_npz(str(tmp_path / "target_0_30" / "trajectory.npz"))
        _make_trajectory_npz(str(tmp_path / "target_0_50" / "trajectory.npz"))
        targets = discover_targets(str(tmp_path))
        assert len(targets) == 2
        assert 0.30 in targets
        assert 0.50 in targets

    def test_empty_dir(self, tmp_path):
        targets = discover_targets(str(tmp_path))
        assert len(targets) == 0

    def test_missing_dir(self):
        targets = discover_targets("/nonexistent/path")
        assert len(targets) == 0

    def test_ignores_non_target_dirs(self, tmp_path):
        os.makedirs(tmp_path / "some_other_dir")
        targets = discover_targets(str(tmp_path))
        assert len(targets) == 0


class TestComputeStats:
    def test_basic_stats(self, tmp_path):
        npz_path = str(tmp_path / "target_0_30" / "trajectory.npz")
        _make_trajectory_npz(npz_path, n_steps=100, n_det=5, flight_steps=(30, 60))
        d = np.load(npz_path)
        stats = compute_stats(d, 100, 0.30)
        assert stats["flight_pct"] == pytest.approx(30.0, abs=1.0)
        assert stats["n_det"] == 5
        assert stats["bounces"] >= 1
        assert stats["peak_h_m"] > 0
        assert not np.isnan(stats["rmse_flight_mm"])

    def test_no_flight(self, tmp_path):
        npz_path = str(tmp_path / "target_0_10" / "trajectory.npz")
        _make_trajectory_npz(npz_path, n_steps=50, n_det=0, flight_steps=(50, 50))
        d = np.load(npz_path)
        stats = compute_stats(d, 50, 0.10)
        assert stats["flight_pct"] == 0.0
        assert stats["n_det"] == 0
        assert np.isnan(stats["rmse_flight_mm"])

    def test_all_flight(self, tmp_path):
        npz_path = str(tmp_path / "target_0_50" / "trajectory.npz")
        _make_trajectory_npz(npz_path, n_steps=80, n_det=10, flight_steps=(0, 80))
        d = np.load(npz_path)
        stats = compute_stats(d, 80, 0.50)
        assert stats["flight_pct"] == 100.0


class TestPlotDualComparison:
    def test_generates_figure(self, tmp_path):
        """End-to-end test: create two eval dirs, generate comparison figure."""
        oracle_dir = tmp_path / "oracle"
        d435i_dir = tmp_path / "d435i"
        for h in [0.10, 0.30, 0.50]:
            h_str = f"{h:.2f}".replace(".", "_")
            _make_trajectory_npz(str(oracle_dir / f"target_{h_str}" / "trajectory.npz"))
            _make_trajectory_npz(str(d435i_dir / f"target_{h_str}" / "trajectory.npz"))

        out_path = str(tmp_path / "comparison.png")

        from plot_dual_trajectory import plot_dual_comparison
        plot_dual_comparison(str(oracle_dir), str(d435i_dir), out_path, max_steps=50)
        assert os.path.isfile(out_path)

    def test_no_shared_targets(self, tmp_path, capsys):
        oracle_dir = tmp_path / "oracle"
        d435i_dir = tmp_path / "d435i"
        _make_trajectory_npz(str(oracle_dir / "target_0_10" / "trajectory.npz"))
        _make_trajectory_npz(str(d435i_dir / "target_0_50" / "trajectory.npz"))

        out_path = str(tmp_path / "comparison.png")
        from plot_dual_trajectory import plot_dual_comparison
        plot_dual_comparison(str(oracle_dir), str(d435i_dir), out_path)
        assert not os.path.isfile(out_path)
        captured = capsys.readouterr()
        assert "No shared targets" in captured.out
