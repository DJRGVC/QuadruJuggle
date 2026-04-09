"""Tests for analyze_anchor_ablation.py."""

import os
import tempfile

import numpy as np
import pytest

from analyze_anchor_ablation import (
    compute_per_step_error,
    cumulative_rmse,
    load_npz,
    phase_mask,
    phase_rmse,
    plot_ablation,
    print_ablation_summary,
)


def _make_traj(T=200, n_det=10, paddle_z=0.47, with_anchor=True, seed=42):
    """Create a synthetic trajectory dict mimicking demo_camera_ekf.py output."""
    rng = np.random.RandomState(seed)
    gt = np.zeros((T, 3))
    gt[:, 2] = paddle_z  # default: ball on paddle
    # Add a flight phase (scaled to trajectory length)
    f_start = T // 4
    f_end = T // 2
    flight = np.arange(f_start, min(f_end, T))
    if len(flight) > 0:
        gt[flight, 2] = paddle_z + 0.15 * np.sin(np.pi * (flight - f_start) / max(1, f_end - f_start))

    ekf = gt + rng.randn(T, 3) * 0.005  # small EKF noise
    det = gt[::20] + rng.randn(T // 20, 3) * 0.003
    det_steps = np.arange(0, T, 20, dtype=int)

    traj = {
        "gt": gt,
        "ekf": ekf,
        "dt": np.float64(0.02),
        "det": det,
        "det_steps": det_steps,
        "ball_h": gt[:, 2] - paddle_z,
    }
    if with_anchor:
        anchored = np.zeros(T, dtype=np.float64)
        # Anchor fires during contact phases (low height)
        anchored[gt[:, 2] - paddle_z < 0.03] = 1.0
        # But not first 5 steps (min_starve)
        anchored[:5] = 0.0
        traj["anchored_step"] = anchored
    return traj


def _save(traj, path):
    np.savez_compressed(path, **traj)


class TestComputePerStepError:
    def test_zero_error(self):
        traj = {"gt": np.ones((10, 3)), "ekf": np.ones((10, 3))}
        err = compute_per_step_error(traj)
        np.testing.assert_allclose(err, 0.0, atol=1e-10)

    def test_known_error(self):
        gt = np.zeros((5, 3))
        ekf = np.zeros((5, 3))
        ekf[:, 0] = 0.001  # 1mm X error
        traj = {"gt": gt, "ekf": ekf}
        err = compute_per_step_error(traj)
        np.testing.assert_allclose(err, 1.0, atol=0.01)  # 1mm


class TestPhaseMask:
    def test_all_contact(self):
        traj = {"gt": np.zeros((10, 3))}
        traj["gt"][:, 2] = 0.47 + 0.01  # 10mm above paddle
        c, f = phase_mask(traj, paddle_z=0.47, threshold=0.03)
        assert c.all()
        assert not f.any()

    def test_all_flight(self):
        traj = {"gt": np.zeros((10, 3))}
        traj["gt"][:, 2] = 0.47 + 0.10  # 100mm above
        c, f = phase_mask(traj, paddle_z=0.47, threshold=0.03)
        assert not c.any()
        assert f.all()

    def test_mixed(self):
        traj = _make_traj()
        c, f = phase_mask(traj, paddle_z=0.47)
        assert c.sum() > 0
        assert f.sum() > 0
        assert (c | f).all()
        assert not (c & f).any()


class TestPhaseRMSE:
    def test_all_zero(self):
        err = np.zeros(10)
        mask = np.ones(10, dtype=bool)
        assert phase_rmse(err, mask) == 0.0

    def test_nan_on_empty(self):
        err = np.array([1.0, 2.0])
        mask = np.zeros(2, dtype=bool)
        assert np.isnan(phase_rmse(err, mask))

    def test_known_value(self):
        err = np.array([3.0, 4.0])
        mask = np.ones(2, dtype=bool)
        expected = np.sqrt((9 + 16) / 2)
        assert abs(phase_rmse(err, mask) - expected) < 1e-6


class TestCumulativeRMSE:
    def test_constant_error(self):
        err = np.full(10, 5.0)
        cum = cumulative_rmse(err)
        np.testing.assert_allclose(cum, 5.0, atol=1e-6)

    def test_increasing(self):
        err = np.array([0.0, 0.0, 0.0, 10.0, 10.0])
        cum = cumulative_rmse(err)
        # cumulative at step 5: sqrt((0+0+0+100+100)/5) = sqrt(40)
        assert abs(cum[-1] - np.sqrt(40)) < 1e-6
        # Should be monotonically >= previous after spike
        assert cum[3] > cum[2]


class TestLoadNpz:
    def test_loads(self, tmp_path):
        traj = _make_traj()
        p = str(tmp_path / "test.npz")
        _save(traj, p)
        loaded = load_npz(p)
        np.testing.assert_array_equal(loaded["gt"], traj["gt"])

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_npz("/nonexistent/path.npz")


class TestPrintSummary:
    def test_runs_without_error(self, capsys):
        on = _make_traj(with_anchor=True)
        off = _make_traj(with_anchor=False, seed=99)
        print_ablation_summary(on, off, paddle_z=0.47)
        captured = capsys.readouterr()
        assert "Anchor Ablation Summary" in captured.out
        assert "Contact RMSE" in captured.out
        assert "Anchor fires" in captured.out


class TestPlotAblation:
    def test_creates_figure(self, tmp_path):
        on = _make_traj(with_anchor=True)
        off = _make_traj(with_anchor=False, seed=99)
        out = str(tmp_path / "test_fig.png")
        plot_ablation(on, off, paddle_z=0.47, out_path=out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 1000  # not empty

    def test_without_anchor_data(self, tmp_path):
        """Works even if anchor_step data is missing (OFF trajectory)."""
        on = _make_traj(with_anchor=False)  # no anchor data
        off = _make_traj(with_anchor=False, seed=99)
        out = str(tmp_path / "test_fig2.png")
        plot_ablation(on, off, paddle_z=0.47, out_path=out)
        assert os.path.isfile(out)

    def test_short_trajectory(self, tmp_path):
        """Handles very short trajectories."""
        on = _make_traj(T=20, with_anchor=True)
        off = _make_traj(T=20, with_anchor=False, seed=99)
        out = str(tmp_path / "test_fig3.png")
        plot_ablation(on, off, paddle_z=0.47, out_path=out)
        assert os.path.isfile(out)
