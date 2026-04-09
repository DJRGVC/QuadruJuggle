"""Tests for compare_multi_target.py."""

import os
import tempfile

import numpy as np
import pytest

from compare_multi_target import discover_targets, load_eval_dir, print_single


def _make_npz(path: str, n_steps: int = 100, n_det: int = 5, peak_z: float = 0.7):
    """Create a minimal trajectory.npz for testing."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    gt = np.random.randn(n_steps, 3) * 0.1
    gt[:, 2] = np.abs(gt[:, 2]) + 0.4  # positive Z
    gt[n_steps // 2, 2] = peak_z  # set peak
    ekf = gt + np.random.randn(n_steps, 3) * 0.02
    det_steps = np.sort(np.random.choice(n_steps, size=n_det, replace=False))
    det = gt[det_steps] + np.random.randn(n_det, 3) * 0.01
    steps = np.arange(n_steps)
    np.savez(
        path,
        gt=gt, ekf=ekf, det=det, det_steps=det_steps,
        steps=steps, dt=np.float64(0.02), n_episodes=np.int64(3),
    )


class TestDiscoverTargets:
    def test_finds_targets(self, tmp_path):
        for tag in ["target_0_10", "target_0_50", "target_1_00"]:
            _make_npz(str(tmp_path / tag / "trajectory.npz"))
        targets = discover_targets(str(tmp_path))
        assert len(targets) == 3
        assert targets[0][0] == pytest.approx(0.10)
        assert targets[1][0] == pytest.approx(0.50)
        assert targets[2][0] == pytest.approx(1.00)

    def test_skips_dirs_without_npz(self, tmp_path):
        _make_npz(str(tmp_path / "target_0_30" / "trajectory.npz"))
        os.makedirs(tmp_path / "target_0_70")  # no npz
        targets = discover_targets(str(tmp_path))
        assert len(targets) == 1
        assert targets[0][0] == pytest.approx(0.30)

    def test_empty_dir(self, tmp_path):
        targets = discover_targets(str(tmp_path))
        assert targets == []

    def test_nonexistent_dir(self):
        with pytest.raises(FileNotFoundError):
            discover_targets("/nonexistent/path")


class TestLoadEvalDir:
    def test_loads_all_targets(self, tmp_path):
        for tag in ["target_0_10", "target_0_50"]:
            _make_npz(str(tmp_path / tag / "trajectory.npz"))
        data = load_eval_dir(str(tmp_path))
        assert 0.10 in data
        assert 0.50 in data
        assert "metrics" in data[0.10]
        assert "flight_frac" in data[0.10]
        assert "peak_z" in data[0.10]

    def test_metrics_reasonable(self, tmp_path):
        _make_npz(str(tmp_path / "target_0_30" / "trajectory.npz"),
                   n_steps=200, n_det=10, peak_z=0.8)
        data = load_eval_dir(str(tmp_path))
        m = data[0.30]["metrics"]
        assert m["total_steps"] == 200
        assert m["n_detections"] == 10
        assert 0 < m["det_rate_pct"] < 100
        assert m["ekf_rmse_mm"] > 0

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(ValueError):
            load_eval_dir(str(tmp_path))


class TestPrintSingle:
    def test_prints_without_error(self, tmp_path, capsys):
        for tag in ["target_0_10", "target_0_50"]:
            _make_npz(str(tmp_path / tag / "trajectory.npz"))
        data = load_eval_dir(str(tmp_path))
        print_single(data, "Test Run")
        captured = capsys.readouterr()
        assert "Test Run" in captured.out
        assert "0.10" in captured.out
        assert "0.50" in captured.out
