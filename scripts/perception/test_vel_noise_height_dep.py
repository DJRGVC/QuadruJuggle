"""Tests for height-dependent velocity noise in d435i mode.

The velocity noise should scale with ball-camera distance (z in paddle frame),
matching the position noise model. Previously, velocity noise used a fixed
z_nominal=0.5m, which underestimates noise at higher altitudes (Stage G: 1m+).
"""

import importlib.util
import os
import sys
import unittest

import torch

# Stub isaaclab imports so we can test without Isaac Lab installed
class _StubModule:
    def __getattr__(self, name):
        return _StubModule()
    def __call__(self, *a, **kw):
        return _StubModule()

for mod_name in [
    "isaaclab", "isaaclab.utils", "isaaclab.utils.math",
    "isaaclab.assets", "isaaclab.managers", "isaaclab.envs",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = _StubModule()

_PERC_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "source", "go1_ball_balance", "go1_ball_balance", "perception",
)

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_ekf_mod = _load_module("perception.ball_ekf", os.path.join(_PERC_DIR, "ball_ekf.py"))
_noise_mod = _load_module("perception.noise_model", os.path.join(_PERC_DIR, "noise_model.py"))
_obs_mod = _load_module("perception.ball_obs_spec", os.path.join(_PERC_DIR, "ball_obs_spec.py"))

_apply_d435i_vel_noise = _obs_mod._apply_d435i_vel_noise
D435iNoiseParams = _obs_mod.D435iNoiseParams


class TestVelNoiseHeightDependence(unittest.TestCase):
    """Verify velocity noise scales with ball height above paddle."""

    def setUp(self):
        self.params = D435iNoiseParams()
        self.N = 10000
        torch.manual_seed(42)

    def _measure_vel_noise_std(self, z_height: float) -> tuple[float, float]:
        """Measure empirical velocity noise std at given height."""
        vel_b = torch.zeros(self.N, 3)
        pos_b = torch.zeros(self.N, 3)
        pos_b[:, 2] = z_height

        noisy = _apply_d435i_vel_noise(vel_b, self.params, pos_b)
        non_dropout = noisy[noisy.abs().sum(dim=-1) > 0]
        if len(non_dropout) < 100:
            return 0.0, 0.0
        std_xy = non_dropout[:, :2].std().item()
        std_z = non_dropout[:, 2].std().item()
        return std_xy, std_z

    def test_noise_increases_with_height(self):
        """Velocity noise at 1.0m should be larger than at 0.1m."""
        std_xy_low, std_z_low = self._measure_vel_noise_std(0.1)
        std_xy_high, std_z_high = self._measure_vel_noise_std(1.0)
        self.assertGreater(std_z_high, std_z_low * 1.5,
                           "Z velocity noise should be significantly larger at 1.0m than 0.1m")
        self.assertGreater(std_xy_high, std_xy_low * 1.5,
                           "XY velocity noise should be significantly larger at 1.0m than 0.1m")

    def test_noise_at_half_metre_matches_nominal(self):
        """At z=0.5m, height-dependent noise should match the old nominal."""
        dt = 1.0 / 30.0
        z = 0.5
        expected_sigma_xy = max(self.params.sigma_xy_floor,
                                self.params.sigma_xy_per_metre * z)
        expected_sigma_z = self.params.sigma_z_base + self.params.sigma_z_quadratic * z * z
        expected_vel_xy = (2 ** 0.5) * expected_sigma_xy / dt
        expected_vel_z = (2 ** 0.5) * expected_sigma_z / dt

        std_xy, std_z = self._measure_vel_noise_std(0.5)
        self.assertAlmostEqual(std_xy, expected_vel_xy, delta=expected_vel_xy * 0.2)
        self.assertAlmostEqual(std_z, expected_vel_z, delta=expected_vel_z * 0.2)

    def test_backward_compat_no_pos(self):
        """When pos_b=None, should use nominal z=0.5m (backward compat)."""
        vel_b = torch.zeros(self.N, 3)
        noisy_no_pos = _apply_d435i_vel_noise(vel_b, self.params, pos_b=None)
        non_dropout = noisy_no_pos[noisy_no_pos.abs().sum(dim=-1) > 0]
        self.assertGreater(len(non_dropout), self.N * 0.5,
                           "Without pos_b, should still apply noise with nominal z")

    def test_z_noise_quadratic_scaling(self):
        """Z velocity noise should show quadratic growth (σ_z ∝ z²)."""
        heights = [0.2, 0.5, 1.0]
        stds = []
        for h in heights:
            _, std_z = self._measure_vel_noise_std(h)
            stds.append(std_z)

        ratio = stds[2] / max(stds[0], 1e-9)
        self.assertGreater(ratio, 2.0,
                           f"Z noise ratio (1.0m/0.2m) = {ratio:.2f}, expected > 2.0")

    def test_dropout_increases_with_distance(self):
        """Dropout rate should increase for balls far from camera."""
        vel_b = torch.zeros(self.N, 3)

        pos_close = torch.zeros(self.N, 3)
        pos_close[:, 2] = 0.1
        noisy_close = _apply_d435i_vel_noise(vel_b, self.params, pos_close)
        dropout_close = (noisy_close.abs().sum(dim=-1) == 0).float().mean().item()

        pos_far = torch.zeros(self.N, 3)
        pos_far[:, 2] = 2.0
        noisy_far = _apply_d435i_vel_noise(vel_b, self.params, pos_far)
        dropout_far = (noisy_far.abs().sum(dim=-1) == 0).float().mean().item()

        self.assertGreater(dropout_far, dropout_close + 0.05,
                           "Dropout should be higher for distant ball")

    def test_height_dep_vs_nominal_at_1m(self):
        """At z=1.0m, height-dep noise should be larger than nominal (z=0.5m)."""
        vel_b = torch.zeros(self.N, 3)

        # Height-dependent at 1.0m
        pos_1m = torch.zeros(self.N, 3)
        pos_1m[:, 2] = 1.0
        noisy_1m = _apply_d435i_vel_noise(vel_b.clone(), self.params, pos_1m)
        non_drop_1m = noisy_1m[noisy_1m.abs().sum(dim=-1) > 0]

        # Nominal (no pos_b, uses z=0.5m)
        noisy_nom = _apply_d435i_vel_noise(vel_b.clone(), self.params, pos_b=None)
        non_drop_nom = noisy_nom[noisy_nom.abs().sum(dim=-1) > 0]

        std_z_1m = non_drop_1m[:, 2].std().item()
        std_z_nom = non_drop_nom[:, 2].std().item()

        self.assertGreater(std_z_1m, std_z_nom * 1.3,
                           f"Z noise at 1.0m ({std_z_1m:.4f}) should exceed nominal ({std_z_nom:.4f})")


if __name__ == "__main__":
    unittest.main()
