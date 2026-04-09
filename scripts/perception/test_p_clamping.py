#!/usr/bin/env python3
"""Unit tests for EKF covariance clamping (sparse-measurement regime).

Tests that P diagonals are bounded during long predict-only sequences,
and that the measurement starvation counter tracks correctly.

Usage:
    python scripts/perception/test_p_clamping.py
"""

import sys
import os
import unittest

import torch

# Direct import to avoid Isaac Lab __init__.py chain (needs pxr/sim)
_PERCEPTION = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "source", "go1_ball_balance", "go1_ball_balance", "perception",
))
sys.path.insert(0, _PERCEPTION)

from ball_ekf import BallEKF, BallEKFConfig


class TestCovarianceClamping(unittest.TestCase):
    """Tests for P diagonal clamping during predict-only sequences."""

    def test_p_bounded_after_many_predicts(self):
        """P diagonals stay below p_max after 500 predict-only steps."""
        cfg = BallEKFConfig(
            p_clamp_enabled=True,
            p_max_pos=0.25,
            p_max_vel=5.0,
            contact_aware=False,  # simplify — no contact logic
        )
        ekf = BallEKF(num_envs=2, device="cpu", cfg=cfg)
        init_pos = torch.tensor([[0.0, 0.0, 0.30]] * 2)
        ekf.reset(torch.arange(2), init_pos)

        # Run 500 predict-only steps (10s at 50Hz)
        for _ in range(500):
            ekf.predict(dt=0.02)

        # Check position diagonals
        for i in range(3):
            max_var = ekf._P[:, i, i].max().item()
            self.assertLessEqual(max_var, cfg.p_max_pos ** 2 + 1e-6,
                                 f"P[{i},{i}] = {max_var:.4f} exceeds p_max_pos² = {cfg.p_max_pos**2}")

        # Check velocity diagonals
        for i in range(3, 6):
            max_var = ekf._P[:, i, i].max().item()
            self.assertLessEqual(max_var, cfg.p_max_vel ** 2 + 1e-6,
                                 f"P[{i},{i}] = {max_var:.4f} exceeds p_max_vel² = {cfg.p_max_vel**2}")

    def test_p_unbounded_when_disabled(self):
        """Without clamping, P diagonals grow beyond clamped limits."""
        cfg = BallEKFConfig(
            p_clamp_enabled=False,
            contact_aware=True,  # contact-aware inflates q_vel in contact zone
            q_vel_contact=50.0,
            contact_z_threshold=0.025,
        )
        ekf = BallEKF(num_envs=1, device="cpu", cfg=cfg)
        # Place ball in contact zone so q_vel=50.0 (large process noise)
        init_pos = torch.tensor([[0.0, 0.0, 0.01]])
        ekf.reset(torch.arange(1), init_pos)

        for _ in range(200):
            ekf.predict(dt=0.02)

        # With q_vel_contact=50.0, vel diags should be large
        vel_diags = [ekf._P[0, i, i].item() for i in range(3, 6)]
        max_vel_var = max(vel_diags)
        self.assertGreater(max_vel_var, 10.0,  # well above flight-only growth
                           f"Expected large P_vel after 200 contact predicts, got {max_vel_var}")

    def test_clamping_preserves_measurement_absorption(self):
        """After clamping, a valid measurement still corrects the state."""
        cfg = BallEKFConfig(
            p_clamp_enabled=True,
            p_max_pos=0.25,
            p_max_vel=5.0,
            contact_aware=False,
            nis_gate_enabled=False,  # disable gating to ensure update happens
        )
        ekf = BallEKF(num_envs=1, device="cpu", cfg=cfg)
        init_pos = torch.tensor([[0.0, 0.0, 0.30]])
        ekf.reset(torch.arange(1), init_pos)

        # 200 predict-only steps — P should be clamped but not zero
        for _ in range(200):
            ekf.predict(dt=0.02)

        pos_before = ekf.pos.clone()

        # Now provide a measurement at a different location
        z_meas = torch.tensor([[0.05, -0.03, 0.25]])
        detected = torch.tensor([True])
        ekf.update(z_meas, detected)

        pos_after = ekf.pos.clone()

        # State should have moved toward the measurement
        delta = (pos_after - pos_before).abs().sum().item()
        self.assertGreater(delta, 0.01,
                           f"Expected state correction after measurement, got delta={delta}")

    def test_9d_spin_clamping(self):
        """Spin diagonals are also clamped in 9D mode."""
        cfg = BallEKFConfig(
            enable_spin=True,
            p_clamp_enabled=True,
            p_max_pos=0.25,
            p_max_vel=5.0,
            p_max_spin=50.0,
            contact_aware=False,
        )
        ekf = BallEKF(num_envs=1, device="cpu", cfg=cfg)
        init_pos = torch.tensor([[0.0, 0.0, 0.30]])
        ekf.reset(torch.arange(1), init_pos)

        for _ in range(500):
            ekf.predict(dt=0.02)

        # Check spin diagonals
        for i in range(6, 9):
            max_var = ekf._P[0, i, i].item()
            self.assertLessEqual(max_var, cfg.p_max_spin ** 2 + 1e-6,
                                 f"P[{i},{i}] = {max_var:.4f} exceeds p_max_spin² = {cfg.p_max_spin**2}")


class TestMeasurementStarvation(unittest.TestCase):
    """Tests for the steps_since_measurement counter."""

    def test_counter_increments_on_predict_only(self):
        """Counter goes up each step when no measurement is provided."""
        ekf = BallEKF(num_envs=2, device="cpu")
        init_pos = torch.tensor([[0.0, 0.0, 0.30]] * 2)
        ekf.reset(torch.arange(2), init_pos)

        z = torch.zeros(2, 3)
        detected = torch.tensor([False, False])

        for i in range(1, 11):
            ekf.step(z, detected, dt=0.02)
            self.assertEqual(ekf.steps_since_measurement[0].item(), i)
            self.assertEqual(ekf.steps_since_measurement[1].item(), i)

    def test_counter_resets_on_measurement(self):
        """Counter resets to 0 for envs that receive a measurement."""
        ekf = BallEKF(num_envs=2, device="cpu",
                       cfg=BallEKFConfig(nis_gate_enabled=False))
        init_pos = torch.tensor([[0.0, 0.0, 0.30]] * 2)
        ekf.reset(torch.arange(2), init_pos)

        z = torch.zeros(2, 3)

        # 5 steps with no measurement
        for _ in range(5):
            ekf.step(z, torch.tensor([False, False]), dt=0.02)
        self.assertEqual(ekf.steps_since_measurement[0].item(), 5)

        # Env 0 gets measurement, env 1 doesn't
        ekf.step(z, torch.tensor([True, False]), dt=0.02)
        self.assertEqual(ekf.steps_since_measurement[0].item(), 0)
        self.assertEqual(ekf.steps_since_measurement[1].item(), 6)

    def test_counter_resets_on_env_reset(self):
        """Counter resets when env is reset."""
        ekf = BallEKF(num_envs=2, device="cpu")
        init_pos = torch.tensor([[0.0, 0.0, 0.30]] * 2)
        ekf.reset(torch.arange(2), init_pos)

        z = torch.zeros(2, 3)
        for _ in range(10):
            ekf.step(z, torch.tensor([False, False]), dt=0.02)

        self.assertEqual(ekf.steps_since_measurement[0].item(), 10)

        # Reset env 0 only
        ekf.reset(torch.tensor([0]), torch.tensor([[0.0, 0.0, 0.30]]))
        self.assertEqual(ekf.steps_since_measurement[0].item(), 0)
        self.assertEqual(ekf.steps_since_measurement[1].item(), 10)

    def test_counter_mixed_detection(self):
        """Counter handles mixed detection patterns correctly."""
        ekf = BallEKF(num_envs=3, device="cpu",
                       cfg=BallEKFConfig(nis_gate_enabled=False))
        init_pos = torch.tensor([[0.0, 0.0, 0.30]] * 3)
        ekf.reset(torch.arange(3), init_pos)

        z = torch.zeros(3, 3)

        # Step 1: all no measurement
        ekf.step(z, torch.tensor([False, False, False]), dt=0.02)
        torch.testing.assert_close(
            ekf.steps_since_measurement,
            torch.tensor([1, 1, 1]),
        )

        # Step 2: env 1 gets measurement
        ekf.step(z, torch.tensor([False, True, False]), dt=0.02)
        torch.testing.assert_close(
            ekf.steps_since_measurement,
            torch.tensor([2, 0, 2]),
        )

        # Step 3: env 0 and 2 get measurements
        ekf.step(z, torch.tensor([True, False, True]), dt=0.02)
        torch.testing.assert_close(
            ekf.steps_since_measurement,
            torch.tensor([0, 1, 0]),
        )


if __name__ == "__main__":
    unittest.main()
