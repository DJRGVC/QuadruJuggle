"""Tests for plot_noise_profile.py noise model functions."""

import numpy as np
import pytest

# Import functions under test
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from plot_noise_profile import (
    position_noise,
    velocity_noise,
    dropout_rate,
    peak_velocity,
)


class TestPositionNoise:
    def test_sigma_xy_floor(self):
        """At very close range, sigma_xy should hit the floor."""
        z = np.array([0.01])
        sigma_xy, _ = position_noise(z)
        assert sigma_xy[0] == pytest.approx(0.001, abs=1e-6)

    def test_sigma_xy_linear(self):
        """At moderate distance, sigma_xy should be linear in z."""
        z = np.array([1.0])
        sigma_xy, _ = position_noise(z)
        assert sigma_xy[0] == pytest.approx(0.0025, abs=1e-6)

    def test_sigma_z_quadratic(self):
        """Sigma_z should have a quadratic z² term."""
        z = np.array([1.0])
        _, sigma_z = position_noise(z)
        expected = 0.001 + 0.005 * 1.0  # base + quadratic * z^2
        assert sigma_z[0] == pytest.approx(expected, abs=1e-6)

    def test_scale_multiplies_noise(self):
        """noise_scale should linearly scale noise amplitudes."""
        z = np.array([0.5])
        s_xy_1, s_z_1 = position_noise(z, scale=1.0)
        s_xy_half, s_z_half = position_noise(z, scale=0.5)
        assert s_xy_half[0] == pytest.approx(s_xy_1[0] * 0.5, abs=1e-6)
        assert s_z_half[0] == pytest.approx(s_z_1[0] * 0.5, abs=1e-6)

    def test_monotonic_in_z(self):
        """Noise should increase with distance."""
        z = np.linspace(0.1, 1.5, 50)
        _, sigma_z = position_noise(z)
        assert np.all(np.diff(sigma_z) >= 0)


class TestVelocityNoise:
    def test_sqrt2_scaling(self):
        """Velocity noise = sqrt(2) * pos_noise / dt."""
        z = np.array([0.5])
        s_xy, s_z = position_noise(z)
        sv_xy, sv_z = velocity_noise(z, fps=30.0)
        dt = 1.0 / 30.0
        assert sv_xy[0] == pytest.approx(np.sqrt(2) * s_xy[0] / dt, rel=1e-5)
        assert sv_z[0] == pytest.approx(np.sqrt(2) * s_z[0] / dt, rel=1e-5)

    def test_higher_scale_more_noise(self):
        """Higher noise scale should produce more velocity noise."""
        z = np.array([0.5])
        sv_half_xy, sv_half_z = velocity_noise(z, scale=0.5)
        sv_full_xy, sv_full_z = velocity_noise(z, scale=1.0)
        assert sv_full_z[0] > sv_half_z[0]


class TestDropout:
    def test_baseline_at_close_range(self):
        """Below 0.5m, dropout should equal baseline."""
        z = np.array([0.1, 0.3, 0.49])
        dr = dropout_rate(z)
        np.testing.assert_allclose(dr, 0.20, atol=1e-6)

    def test_increases_above_half_metre(self):
        """Dropout should increase above 0.5m."""
        dr_close = dropout_rate(np.array([0.3]))[0]
        dr_far = dropout_rate(np.array([1.0]))[0]
        assert dr_far > dr_close

    def test_clipped_to_1(self):
        """Dropout should never exceed 1.0."""
        z = np.array([100.0])
        dr = dropout_rate(z)
        assert dr[0] <= 1.0


class TestPeakVelocity:
    def test_energy_conservation(self):
        """v_peak = sqrt(2*g*h)."""
        h = np.array([0.5])
        v = peak_velocity(h)
        assert v[0] == pytest.approx(np.sqrt(2 * 9.81 * 0.5), rel=1e-5)

    def test_monotonic(self):
        """Higher targets should have higher peak velocity."""
        h = np.array([0.1, 0.3, 0.5, 1.0])
        v = peak_velocity(h)
        assert np.all(np.diff(v) > 0)
