"""Tests for analyze_perception_gap.py."""

import os
import tempfile

import numpy as np
import pytest

from analyze_perception_gap import (
    compute_obs_staleness,
    compute_velocity_error,
    decompose_gap,
    discover_targets,
)


def _make_trajectory_npz(
    path: str,
    n_steps: int = 200,
    n_det: int = 10,
    paddle_z: float = 0.47,
    peak_height: float = 0.25,
    include_phase: bool = True,
    include_anchor: bool = True,
):
    """Create a realistic trajectory.npz with phase and anchor data."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Simulate a ball that bounces once in the middle
    gt = np.zeros((n_steps, 3), dtype=np.float32)
    gt[:, 0] = np.random.randn(n_steps) * 0.005  # small XY jitter
    gt[:, 1] = np.random.randn(n_steps) * 0.005

    # Z trajectory: contact → bounce → land → contact
    bounce_start = n_steps // 4
    bounce_end = 3 * n_steps // 4
    bounce_len = bounce_end - bounce_start
    t_norm = np.linspace(0, 1, bounce_len)
    # Parabolic arc
    arc = 4 * peak_height * t_norm * (1 - t_norm)
    gt[:bounce_start, 2] = paddle_z
    gt[bounce_start:bounce_end, 2] = paddle_z + arc
    gt[bounce_end:, 2] = paddle_z

    # EKF with small noise
    ekf = gt.copy()
    ekf += np.random.randn(n_steps, 3).astype(np.float32) * 0.005

    # Detections during flight
    flight_steps = np.arange(bounce_start, bounce_end)
    if len(flight_steps) > n_det:
        det_steps = np.sort(np.random.choice(flight_steps, size=n_det, replace=False))
    else:
        det_steps = flight_steps[:n_det]
    det = gt[det_steps] + np.random.randn(len(det_steps), 3) * 0.002

    # Phase: 0=contact, 1=ascending, 2=descending
    phase = np.zeros(n_steps, dtype=np.int64)
    mid_bounce = (bounce_start + bounce_end) // 2
    phase[bounce_start:mid_bounce] = 1  # ascending
    phase[mid_bounce:bounce_end] = 2    # descending

    # Anchored: during contact phases
    anchored = np.zeros(n_steps, dtype=np.int64)
    anchored[:bounce_start] = 1
    anchored[bounce_end:] = 1

    # Ball height above paddle
    ball_h = (gt[:, 2] - paddle_z).astype(np.float64)

    data = dict(
        gt=gt, ekf=ekf, det=det.astype(np.float64),
        det_steps=det_steps.astype(np.int64),
        steps=np.arange(n_steps, dtype=np.int64),
        dt=np.float64(0.02),
        rmse_ekf=np.linalg.norm(ekf - gt, axis=1).astype(np.float32),
        rmse_det=np.linalg.norm(det - gt[det_steps], axis=1),
        ball_h=ball_h,
        sched_active=np.ones(n_steps, dtype=np.int64),
    )
    if include_phase:
        data["phase"] = phase
    if include_anchor:
        data["anchored_step"] = anchored
    np.savez(path, **data)


def _make_eval_dir(base_dir: str, targets: list[float], **kwargs):
    """Create eval dir with trajectory.npz for each target."""
    for t in targets:
        tag = f"target_{int(t)}_{int((t % 1) * 100):02d}"
        _make_trajectory_npz(os.path.join(base_dir, tag, "trajectory.npz"), **kwargs)


class TestDiscoverTargets:
    def test_finds_matching_targets(self, tmp_path):
        _make_eval_dir(str(tmp_path), [0.10, 0.30, 0.50])
        targets = discover_targets(str(tmp_path))
        assert len(targets) == 3
        assert targets[0][0] == pytest.approx(0.10)
        assert targets[2][0] == pytest.approx(0.50)

    def test_empty_dir(self, tmp_path):
        targets = discover_targets(str(tmp_path))
        assert targets == []

    def test_nonexistent_dir(self):
        with pytest.raises(FileNotFoundError):
            discover_targets("/nonexistent/path")


class TestComputeObsStaleness:
    def test_staleness_with_detections(self, tmp_path):
        npz_path = str(tmp_path / "trajectory.npz")
        _make_trajectory_npz(npz_path, n_steps=100, n_det=5)
        traj = dict(np.load(npz_path))
        result = compute_obs_staleness(traj)
        assert "mean_staleness" in result
        assert result["mean_staleness"] >= 0

    def test_staleness_by_phase(self, tmp_path):
        npz_path = str(tmp_path / "trajectory.npz")
        _make_trajectory_npz(npz_path, n_steps=200, n_det=10)
        traj = dict(np.load(npz_path))
        result = compute_obs_staleness(traj)
        # Contact phase should have low staleness (anchored)
        assert "staleness_contact" in result
        assert "staleness_ascending" in result
        assert "staleness_descending" in result

    def test_no_phase_data(self, tmp_path):
        npz_path = str(tmp_path / "trajectory.npz")
        _make_trajectory_npz(npz_path, include_phase=False)
        traj = dict(np.load(npz_path))
        result = compute_obs_staleness(traj)
        assert "mean_staleness" in result
        # Phase-specific keys should not exist when no phase data
        assert "staleness_contact" not in result


class TestComputeVelocityError:
    def test_basic_velocity_error(self, tmp_path):
        npz_path = str(tmp_path / "trajectory.npz")
        _make_trajectory_npz(npz_path, n_steps=200)
        traj = dict(np.load(npz_path))
        result = compute_velocity_error(traj)
        assert "vz_rmse_mps" in result
        assert result["vz_rmse_mps"] >= 0
        # Small noise → small velocity error
        assert result["vz_rmse_mps"] < 5.0

    def test_phase_velocity_breakdown(self, tmp_path):
        npz_path = str(tmp_path / "trajectory.npz")
        _make_trajectory_npz(npz_path, n_steps=200)
        traj = dict(np.load(npz_path))
        result = compute_velocity_error(traj)
        assert "vz_rmse_contact" in result
        assert "vz_rmse_ascending" in result

    def test_apex_accuracy(self, tmp_path):
        npz_path = str(tmp_path / "trajectory.npz")
        _make_trajectory_npz(npz_path, n_steps=200, peak_height=0.30)
        traj = dict(np.load(npz_path))
        result = compute_velocity_error(traj)
        assert "apex_pos_rmse_mm" in result
        # With 5mm noise, apex RMSE should be reasonable
        if not np.isnan(result["apex_pos_rmse_mm"]):
            assert result["apex_pos_rmse_mm"] < 50  # 50mm tolerance

    def test_short_trajectory(self, tmp_path):
        npz_path = str(tmp_path / "trajectory.npz")
        _make_trajectory_npz(npz_path, n_steps=2)
        traj = dict(np.load(npz_path))
        result = compute_velocity_error(traj)
        assert np.isnan(result["vz_rmse_mps"])


class TestDecomposeGap:
    def test_basic_decomposition(self, tmp_path):
        oracle_dir = str(tmp_path / "oracle")
        d435i_dir = str(tmp_path / "d435i")
        targets = [0.10, 0.30, 0.50]
        _make_eval_dir(oracle_dir, targets, n_det=5)
        _make_eval_dir(d435i_dir, targets, n_det=10)

        results = decompose_gap(oracle_dir, d435i_dir)
        assert len(results) == 3
        assert "0.1" in results
        assert "0.5" in results

        for t in results:
            r = results[t]
            assert "oracle" in r
            assert "d435i" in r
            assert "phase_metrics" in r["oracle"]
            assert "staleness" in r["d435i"]
            assert "velocity" in r["oracle"]

    def test_partial_target_overlap(self, tmp_path):
        oracle_dir = str(tmp_path / "oracle")
        d435i_dir = str(tmp_path / "d435i")
        _make_eval_dir(oracle_dir, [0.10, 0.30, 0.50])
        _make_eval_dir(d435i_dir, [0.30, 0.50, 0.70])

        results = decompose_gap(oracle_dir, d435i_dir)
        assert len(results) == 2  # only 0.30 and 0.50 overlap

    def test_no_overlap_raises(self, tmp_path):
        oracle_dir = str(tmp_path / "oracle")
        d435i_dir = str(tmp_path / "d435i")
        _make_eval_dir(oracle_dir, [0.10])
        _make_eval_dir(d435i_dir, [0.50])
        with pytest.raises(ValueError, match="No matching targets"):
            decompose_gap(oracle_dir, d435i_dir)

    def test_nonexistent_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            decompose_gap("/nonexistent", str(tmp_path))


class TestPlotGapDecomposition:
    def test_figure_saved(self, tmp_path):
        """Test that figure generation works without error."""
        from analyze_perception_gap import plot_gap_decomposition

        oracle_dir = str(tmp_path / "oracle")
        d435i_dir = str(tmp_path / "d435i")
        targets = [0.10, 0.30, 0.50]
        _make_eval_dir(oracle_dir, targets, n_det=8, peak_height=0.20)
        _make_eval_dir(d435i_dir, targets, n_det=8, peak_height=0.20)

        results = decompose_gap(oracle_dir, d435i_dir)
        out_path = str(tmp_path / "fig.png")
        plot_gap_decomposition(results, out_path)
        assert os.path.isfile(out_path)
        assert os.path.getsize(out_path) > 1000  # non-trivial file
