"""Tests for predict_perception_gap.py — noise-to-gap prediction model."""

import json
import math
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from predict_perception_gap import (
    DROPOUT_BASE,
    DROPOUT_RANGE,
    DROPOUT_SCALE,
    G,
    SIGMA_XY_FLOOR,
    SIGMA_XY_PER_METRE,
    SIGMA_Z_BASE,
    SIGMA_Z_QUADRATIC,
    compute_effective_noise_exposure,
    compute_flight_fraction,
    compute_noise_at_height,
    predict_gap,
)


class TestComputeNoiseAtHeight:
    """Test D435i noise computation."""

    def test_zero_height_uses_floor(self):
        r = compute_noise_at_height(0.0)
        assert r["sigma_xy_mm"] == pytest.approx(SIGMA_XY_FLOOR * 1000, abs=0.01)

    def test_noise_increases_with_height(self):
        r1 = compute_noise_at_height(0.10)
        r2 = compute_noise_at_height(0.50)
        assert r2["sigma_z_mm"] > r1["sigma_z_mm"]
        assert r2["dropout_pct"] > r1["dropout_pct"]

    def test_sigma_z_formula(self):
        h = 0.50
        expected_z = (SIGMA_Z_BASE + SIGMA_Z_QUADRATIC * h**2) * 1000
        r = compute_noise_at_height(h)
        assert r["sigma_z_mm"] == pytest.approx(expected_z, rel=0.001)

    def test_sigma_xy_formula(self):
        h = 1.0  # high enough to exceed floor
        expected_xy = max(SIGMA_XY_FLOOR, SIGMA_XY_PER_METRE * h) * 1000
        r = compute_noise_at_height(h)
        assert r["sigma_xy_mm"] == pytest.approx(expected_xy, rel=0.001)

    def test_dropout_formula(self):
        h = 0.50
        expected = (DROPOUT_BASE + DROPOUT_RANGE * (1.0 - math.exp(-h / DROPOUT_SCALE))) * 100
        r = compute_noise_at_height(h)
        assert r["dropout_pct"] == pytest.approx(expected, rel=0.001)

    def test_sigma_3d_is_rss(self):
        r = compute_noise_at_height(0.30)
        expected = math.sqrt(2 * (r["sigma_xy_mm"])**2 + (r["sigma_z_mm"])**2)
        assert r["sigma_3d_mm"] == pytest.approx(expected, rel=0.001)

    def test_all_positive(self):
        for h in [0.0, 0.01, 0.1, 0.5, 1.0, 2.0]:
            r = compute_noise_at_height(h)
            assert r["sigma_xy_mm"] > 0
            assert r["sigma_z_mm"] > 0
            assert r["sigma_3d_mm"] > 0
            assert 0 < r["dropout_pct"] < 100


class TestComputeFlightFraction:
    """Test ballistic flight dynamics."""

    def test_zero_height_no_flight(self):
        r = compute_flight_fraction(0.0)
        assert r["flight_fraction"] == 0.0
        assert r["launch_vel"] == 0.0

    def test_launch_vel_correct(self):
        h = 0.50
        expected_v = math.sqrt(2 * G * h)
        r = compute_flight_fraction(h)
        assert r["launch_vel"] == pytest.approx(expected_v, rel=0.001)

    def test_flight_time_correct(self):
        h = 0.50
        v = math.sqrt(2 * G * h)
        expected_t = 2 * v / G
        r = compute_flight_fraction(h)
        assert r["flight_time_s"] == pytest.approx(expected_t, rel=0.001)

    def test_flight_fraction_increases_with_height(self):
        r1 = compute_flight_fraction(0.10)
        r2 = compute_flight_fraction(0.50)
        r3 = compute_flight_fraction(1.00)
        assert r2["flight_fraction"] > r1["flight_fraction"]
        assert r3["flight_fraction"] > r2["flight_fraction"]

    def test_flight_fraction_bounded(self):
        for h in [0.01, 0.1, 0.5, 1.0, 5.0]:
            r = compute_flight_fraction(h)
            assert 0.0 <= r["flight_fraction"] <= 1.0

    def test_apex_time_is_half_flight(self):
        r = compute_flight_fraction(0.50)
        assert r["apex_time_s"] == pytest.approx(r["flight_time_s"] / 2, rel=0.001)


class TestComputeEffectiveNoiseExposure:
    """Test combined noise × flight duration metric."""

    def test_exposure_increases_with_height(self):
        r1 = compute_effective_noise_exposure(0.10)
        r2 = compute_effective_noise_exposure(0.50)
        assert r2["noise_exposure"] > r1["noise_exposure"]

    def test_contains_all_fields(self):
        r = compute_effective_noise_exposure(0.30)
        expected_keys = [
            "target_h", "sigma_xy_mm", "sigma_z_mm", "dropout_pct",
            "flight_fraction", "effective_noise_mm", "noise_exposure",
            "mean_flight_h", "predict_drift_mm",
        ]
        for key in expected_keys:
            assert key in r, f"Missing key: {key}"

    def test_mean_flight_height(self):
        h = 0.60
        r = compute_effective_noise_exposure(h)
        assert r["mean_flight_h"] == pytest.approx(2.0 / 3.0 * h, rel=0.001)

    def test_effective_noise_positive(self):
        for h in [0.01, 0.1, 0.3, 0.5, 1.0]:
            r = compute_effective_noise_exposure(h)
            assert r["effective_noise_mm"] > 0
            assert r["noise_exposure"] > 0


class TestPredictGap:
    """Test gap prediction with and without observed data."""

    def test_basic_prediction(self):
        targets = [0.10, 0.30, 0.50]
        results = predict_gap(targets)
        assert len(results) == 3
        assert results[0]["target_h"] < results[1]["target_h"]

    def test_with_observed_data(self):
        targets = [0.10, 0.30, 0.50]
        observed = {"0.10": 0.3, "0.30": 3.6, "0.50": 18.3}
        results = predict_gap(targets, observed)
        assert results[0]["observed_gap_pct"] == pytest.approx(0.3)
        assert results[2]["observed_gap_pct"] == pytest.approx(18.3)

    def test_linear_fit_with_observed(self):
        targets = [0.10, 0.20, 0.30, 0.40, 0.50]
        observed = {"0.10": 0.3, "0.20": 0.0, "0.30": 3.6, "0.40": 10.0, "0.50": 18.3}
        results = predict_gap(targets, observed)
        # Should have predicted gap and fit parameters
        assert "predicted_gap_pct" in results[0]
        assert "fit_slope" in results[0]
        assert results[0]["fit_slope"] > 0  # gap should increase with exposure

    def test_sorted_output(self):
        targets = [0.50, 0.10, 0.30]
        results = predict_gap(targets)
        for i in range(len(results) - 1):
            assert results[i]["target_h"] <= results[i + 1]["target_h"]

    def test_no_observed_no_fit(self):
        results = predict_gap([0.10, 0.50])
        assert "predicted_gap_pct" not in results[0]
        assert "fit_slope" not in results[0]


class TestOutputs:
    """Test figure and JSON output."""

    def test_json_output(self):
        targets = [0.10, 0.30, 0.50]
        results = predict_gap(targets)
        # Should be JSON-serializable
        json_str = json.dumps(results, default=str)
        parsed = json.loads(json_str)
        assert len(parsed) == 3

    def test_plot_output(self):
        """Test figure generation (requires matplotlib)."""
        from predict_perception_gap import plot_results

        targets = [0.10, 0.30, 0.50]
        observed = {"0.10": 0.3, "0.30": 3.6, "0.50": 18.3}
        results = predict_gap(targets, observed)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            out_path = f.name
        try:
            plot_results(results, out_path)
            assert os.path.exists(out_path)
            assert os.path.getsize(out_path) > 1000  # non-trivial PNG
        finally:
            os.unlink(out_path)
