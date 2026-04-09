"""Tests for plot_stage_comparison.py."""

import os
import tempfile

import numpy as np
import pytest

from plot_stage_comparison import load_results, VARIANTS


def _make_trajectory_npz(path: str, n_steps: int = 100, n_det: int = 50):
    """Create a minimal trajectory.npz."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        steps=np.arange(n_steps),
        det_steps=np.arange(n_det),
        gt=np.random.randn(n_steps, 3) * 0.1 + np.array([0, 0, 0.3]),
        ekf=np.random.randn(n_steps, 3) * 0.1 + np.array([0, 0, 0.3]),
    )


class TestLoadResults:
    def test_empty_dir(self, tmp_path):
        results = load_results(str(tmp_path))
        assert results == {}

    def test_single_variant_single_target(self, tmp_path):
        npz_path = tmp_path / "d435i_anchor" / "target_0_50" / "trajectory.npz"
        _make_trajectory_npz(str(npz_path))
        results = load_results(str(tmp_path))
        assert "d435i_anchor" in results
        assert "0.50" in results["d435i_anchor"]
        r = results["d435i_anchor"]["0.50"]
        assert 0 <= r["det_rate"] <= 100
        assert r["rmse_pos"] >= 0
        assert r["total_steps"] == 100

    def test_multiple_variants(self, tmp_path):
        for variant in ["d435i_anchor", "oracle_baseline"]:
            for target in ["0_10", "0_50", "1_00"]:
                npz_path = tmp_path / variant / f"target_{target}" / "trajectory.npz"
                _make_trajectory_npz(str(npz_path))
        results = load_results(str(tmp_path))
        assert len(results) == 2
        assert len(results["d435i_anchor"]) == 3
        assert "0.10" in results["d435i_anchor"]
        assert "1.00" in results["oracle_baseline"]

    def test_missing_npz_skipped(self, tmp_path):
        (tmp_path / "d435i_anchor" / "target_0_50").mkdir(parents=True)
        # No npz file
        results = load_results(str(tmp_path))
        assert results.get("d435i_anchor", {}) == {}

    def test_det_rate_calculation(self, tmp_path):
        npz_path = tmp_path / "oracle_anchor" / "target_0_30" / "trajectory.npz"
        _make_trajectory_npz(str(npz_path), n_steps=200, n_det=100)
        results = load_results(str(tmp_path))
        assert abs(results["oracle_anchor"]["0.30"]["det_rate"] - 50.0) < 0.1


class TestPlotComparison:
    def test_generates_figure(self, tmp_path):
        for variant in VARIANTS:
            npz_path = tmp_path / "data" / variant / "target_0_50" / "trajectory.npz"
            _make_trajectory_npz(str(npz_path))
        from plot_stage_comparison import plot_comparison
        results = load_results(str(tmp_path / "data"))
        out = str(tmp_path / "out" / "test.png")
        plot_comparison(results, out)
        assert os.path.isfile(out)
        assert os.path.getsize(out) > 1000  # non-trivial image

    def test_empty_results_no_crash(self, tmp_path):
        from plot_stage_comparison import plot_comparison
        out = str(tmp_path / "empty.png")
        plot_comparison({}, out)
        assert not os.path.isfile(out)  # nothing to plot
