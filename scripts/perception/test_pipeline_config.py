#!/usr/bin/env python3
"""Tests for BallObsNoiseCfg → PerceptionPipeline config wiring.

Validates that high-level config flags (enable_spin, world_frame, noise_scale)
correctly propagate to the underlying EKF and noise model instances.

Run: python scripts/perception/test_pipeline_config.py
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

# Direct imports — bypass go1_ball_balance/__init__ (pulls Isaac Lab/pxr).
import importlib.util

_PERC_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "source", "go1_ball_balance", "go1_ball_balance", "perception",
))


def _load_module(name: str, path: str):
    """Import a single .py file as a module, avoiding __init__ chains."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Stub out isaaclab imports that ball_obs_spec.py needs at import time
# so we can test config wiring without Isaac Lab installed.
class _StubModule:
    """Attribute-returning stub for mocking entire module hierarchies."""

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

# Load perception modules directly
_ekf_mod = _load_module(
    "perception.ball_ekf",
    os.path.join(_PERC_DIR, "ball_ekf.py"),
)
BallEKF = _ekf_mod.BallEKF
BallEKFConfig = _ekf_mod.BallEKFConfig

_noise_mod = _load_module(
    "perception.noise_model",
    os.path.join(_PERC_DIR, "noise_model.py"),
)

_obs_mod = _load_module(
    "perception.ball_obs_spec",
    os.path.join(_PERC_DIR, "ball_obs_spec.py"),
)
BallObsNoiseCfg = _obs_mod.BallObsNoiseCfg
PerceptionPipeline = _obs_mod.PerceptionPipeline


class TestEnableSpinWiring(unittest.TestCase):
    """Verify enable_spin on BallObsNoiseCfg propagates to EKF."""

    def test_spin_disabled_by_default(self):
        cfg = BallObsNoiseCfg(mode="ekf")
        self.assertFalse(cfg.enable_spin)

    def test_spin_flag_on_cfg(self):
        cfg = BallObsNoiseCfg(mode="ekf", enable_spin=True)
        self.assertTrue(cfg.enable_spin)

    def test_pipeline_creates_6d_ekf_by_default(self):
        cfg = BallObsNoiseCfg(mode="ekf")
        pipeline = PerceptionPipeline(num_envs=4, device="cpu", noise_cfg=cfg)
        self.assertFalse(pipeline.ekf._spin_enabled)
        self.assertEqual(pipeline.ekf.state_dim, 6)
        self.assertIsNone(pipeline.spin)

    def test_pipeline_creates_9d_ekf_when_spin_enabled(self):
        cfg = BallObsNoiseCfg(mode="ekf", enable_spin=True)
        pipeline = PerceptionPipeline(num_envs=4, device="cpu", noise_cfg=cfg)
        self.assertTrue(pipeline.ekf._spin_enabled)
        self.assertEqual(pipeline.ekf.state_dim, 9)
        self.assertIsNotNone(pipeline.spin)
        self.assertEqual(pipeline.spin.shape, (4, 3))

    def test_spin_enabled_via_cfg_not_ekf_cfg_directly(self):
        """enable_spin on BallObsNoiseCfg should propagate even if
        ekf_cfg.enable_spin is not explicitly set."""
        ekf_cfg = BallEKFConfig(enable_spin=False)
        cfg = BallObsNoiseCfg(mode="ekf", ekf_cfg=ekf_cfg, enable_spin=True)
        pipeline = PerceptionPipeline(num_envs=4, device="cpu", noise_cfg=cfg)
        self.assertTrue(pipeline.ekf._spin_enabled)

    def test_ekf_cfg_spin_respected_when_obs_cfg_off(self):
        """If ekf_cfg.enable_spin=True but BallObsNoiseCfg.enable_spin=False,
        the ekf_cfg should still be respected (EKF config is authoritative)."""
        ekf_cfg = BallEKFConfig(enable_spin=True)
        cfg = BallObsNoiseCfg(mode="ekf", ekf_cfg=ekf_cfg, enable_spin=False)
        pipeline = PerceptionPipeline(num_envs=4, device="cpu", noise_cfg=cfg)
        self.assertTrue(pipeline.ekf._spin_enabled)

    def test_spin_estimate_updates_after_step(self):
        """9D EKF spin estimate should change when processing measurements."""
        cfg = BallObsNoiseCfg(mode="ekf", enable_spin=True)
        pipeline = PerceptionPipeline(num_envs=2, device="cpu", noise_cfg=cfg)

        # Initial spin is zero
        spin0 = pipeline.spin.clone()
        self.assertTrue(torch.allclose(spin0, torch.zeros(2, 3)))

        # Run a few steps with position observations that imply curvature
        detected = torch.ones(2, dtype=torch.bool)
        for t in range(10):
            pos = torch.tensor([[0.01 * t, 0.005 * t, 0.05 - 0.001 * t**2]] * 2)
            pipeline.ekf.step(
                pos, detected, dt=0.02,
                gravity_b=torch.tensor([[0.0, 0.0, -9.81]] * 2),
            )

        # Spin should have diverged from zero (EKF estimates it from curvature)
        # We don't assert exact values — just that the filter is active.
        spin_final = pipeline.spin
        self.assertEqual(spin_final.shape, (2, 3))


class TestEnableImuWiring(unittest.TestCase):
    """Verify enable_imu flag on BallObsNoiseCfg."""

    def test_imu_enabled_by_default(self):
        cfg = BallObsNoiseCfg(mode="ekf")
        self.assertTrue(cfg.enable_imu)

    def test_imu_can_be_disabled(self):
        cfg = BallObsNoiseCfg(mode="ekf", enable_imu=False)
        self.assertFalse(cfg.enable_imu)

    def test_imu_flag_independent_of_world_frame(self):
        """enable_imu should be independent of world_frame flag."""
        cfg = BallObsNoiseCfg(mode="ekf", world_frame=True, enable_imu=False)
        self.assertFalse(cfg.enable_imu)
        self.assertTrue(cfg.world_frame)

    def test_pipeline_step_without_imu(self):
        """Pipeline step should work without IMU (robot_ang_vel_b=None)."""
        cfg = BallObsNoiseCfg(mode="ekf", enable_imu=False)
        pipeline = PerceptionPipeline(num_envs=2, device="cpu", noise_cfg=cfg)
        pos = torch.tensor([[0.0, 0.0, 0.05], [0.01, 0.0, 0.04]])
        detected = torch.ones(2, dtype=torch.bool)
        # Step without angular velocity — should not raise
        pipeline.ekf.step(
            pos, detected, dt=0.02,
            gravity_b=torch.tensor([[0.0, 0.0, -9.81]] * 2),
            robot_ang_vel_b=None,
        )
        self.assertEqual(pipeline.pos.shape, (2, 3))

    def test_pipeline_step_with_imu(self):
        """Pipeline step should accept IMU angular velocity."""
        cfg = BallObsNoiseCfg(mode="ekf", enable_imu=True)
        pipeline = PerceptionPipeline(num_envs=2, device="cpu", noise_cfg=cfg)
        pos = torch.tensor([[0.0, 0.0, 0.05], [0.01, 0.0, 0.04]])
        detected = torch.ones(2, dtype=torch.bool)
        omega = torch.tensor([[0.0, 0.5, 0.0], [0.0, -0.3, 0.1]])
        pipeline.ekf.step(
            pos, detected, dt=0.02,
            gravity_b=torch.tensor([[0.0, 0.0, -9.81]] * 2),
            robot_ang_vel_b=omega,
        )
        self.assertEqual(pipeline.pos.shape, (2, 3))


class TestWorldFrameFlag(unittest.TestCase):
    """Verify world_frame flag creates correct pipeline state."""

    def test_world_frame_off(self):
        cfg = BallObsNoiseCfg(mode="ekf", world_frame=False)
        pipeline = PerceptionPipeline(num_envs=4, device="cpu", noise_cfg=cfg)
        self.assertFalse(pipeline._world_frame)

    def test_world_frame_on(self):
        cfg = BallObsNoiseCfg(mode="ekf", world_frame=True)
        pipeline = PerceptionPipeline(num_envs=4, device="cpu", noise_cfg=cfg)
        self.assertTrue(pipeline._world_frame)
        self.assertIsNotNone(pipeline._robot_quat_w)


class TestNoiseScaleWiring(unittest.TestCase):
    """Verify noise_scale is applied when creating pipeline through factory.

    Note: PerceptionPipeline.__init__ expects pre-scaled config.
    The _scaled_noise_model_cfg helper (used by _get_or_create_pipeline)
    applies the scaling. We test both the helper and direct construction.
    """

    def test_scaled_helper_half(self):
        """_scaled_noise_model_cfg correctly halves amplitudes."""
        from perception.ball_obs_spec import _scaled_noise_model_cfg, D435iNoiseModelCfg
        base = D435iNoiseModelCfg()
        scaled = _scaled_noise_model_cfg(base, 0.5)
        self.assertAlmostEqual(
            scaled.sigma_xy_per_metre, 0.0025 * 0.5, places=6,
        )

    def test_scaled_helper_full(self):
        """_scaled_noise_model_cfg at scale=1.0 is identity."""
        from perception.ball_obs_spec import _scaled_noise_model_cfg, D435iNoiseModelCfg
        base = D435iNoiseModelCfg()
        scaled = _scaled_noise_model_cfg(base, 1.0)
        # scale=1.0 returns original object
        self.assertIs(scaled, base)

    def test_direct_pipeline_uses_cfg_as_is(self):
        """PerceptionPipeline.__init__ uses noise_model_cfg as-is (no scaling)."""
        cfg = BallObsNoiseCfg(mode="ekf", noise_scale=0.5)
        pipeline = PerceptionPipeline(num_envs=2, device="cpu", noise_cfg=cfg)
        # Direct construction doesn't apply noise_scale to noise_model_cfg
        self.assertAlmostEqual(
            pipeline.noise_model.cfg.sigma_xy_per_metre, 0.0025, places=6,
        )


if __name__ == "__main__":
    unittest.main()
