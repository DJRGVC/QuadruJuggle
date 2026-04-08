#!/usr/bin/env python3
"""Unit tests for contact-aware EKF.

Tests that the EKF inflates process noise when ball Z < contact_z_threshold,
and uses normal (low) process noise during free flight. CPU-only, no sim.

Usage:
    python scripts/perception/test_contact_aware_ekf.py
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


class TestContactAwareEKF(unittest.TestCase):
    """Tests for contact-aware process noise inflation."""

    def test_free_flight_low_q_vel(self):
        """During free flight (ball Z > threshold), q_vel should be the low
        free-flight value, not the contact value."""
        cfg = BallEKFConfig(
            q_vel=0.40,
            q_vel_contact=50.0,
            contact_aware=True,
            contact_z_threshold=0.025,
        )
        ekf = BallEKF(num_envs=4, device="cpu", cfg=cfg)

        # Place ball well above paddle (free flight)
        init_pos = torch.tensor([[0.0, 0.0, 0.20]] * 4)
        init_vel = torch.tensor([[0.0, 0.0, 2.0]] * 4)
        ekf.reset(torch.arange(4), init_pos, init_vel)

        # Record P before predict
        P_before = ekf._P.clone()
        ekf.predict(dt=0.02)
        P_after = ekf._P.clone()

        # The velocity covariance growth should be based on q_vel=0.40
        # q_vel_sq = (0.40 * 0.02)^2 = 6.4e-5
        # With contact: would be (50.0 * 0.02)^2 = 1.0
        # Check that P[3,3] grew by roughly q_vel^2*dt^2, not q_vel_contact
        dP_vel = P_after[0, 3, 3] - P_before[0, 3, 3]
        # Should be close to 6.4e-5 (from Q) + F@P@F' contribution
        # Key test: it should be FAR less than 1.0 (contact noise)
        self.assertLess(dP_vel.item(), 0.01,
                        f"Free-flight P growth {dP_vel:.4f} too large (contact noise leak?)")

    def test_contact_high_q_vel(self):
        """During contact (ball Z < threshold), q_vel should be inflated."""
        cfg = BallEKFConfig(
            q_vel=0.40,
            q_vel_contact=50.0,
            contact_aware=True,
            contact_z_threshold=0.025,
        )
        ekf = BallEKF(num_envs=4, device="cpu", cfg=cfg)

        # Place ball ON paddle (contact)
        init_pos = torch.tensor([[0.0, 0.0, 0.020]] * 4)  # Z=20mm < 25mm threshold
        init_vel = torch.zeros(4, 3)
        ekf.reset(torch.arange(4), init_pos, init_vel)

        P_before = ekf._P.clone()
        ekf.predict(dt=0.02)
        P_after = ekf._P.clone()

        # q_vel_contact = 50.0 → q_vel_sq = (50.0 * 0.02)^2 = 1.0
        dP_vel = P_after[0, 3, 3] - P_before[0, 3, 3]
        # Should be at least 0.5 (from Q contribution alone ~1.0)
        self.assertGreater(dP_vel.item(), 0.5,
                           f"Contact P growth {dP_vel:.4f} too small (contact noise not applied?)")

    def test_mixed_envs_contact_and_flight(self):
        """Different envs can be in contact vs flight simultaneously."""
        cfg = BallEKFConfig(
            q_vel=0.40,
            q_vel_contact=50.0,
            contact_aware=True,
            contact_z_threshold=0.025,
        )
        ekf = BallEKF(num_envs=4, device="cpu", cfg=cfg)

        # Envs 0,1: contact (Z=0.02). Envs 2,3: flight (Z=0.20)
        init_pos = torch.tensor([
            [0.0, 0.0, 0.020],  # contact
            [0.0, 0.0, 0.015],  # contact
            [0.0, 0.0, 0.200],  # flight
            [0.0, 0.0, 0.500],  # flight
        ])
        ekf.reset(torch.arange(4), init_pos)

        P_before = ekf._P.clone()
        ekf.predict(dt=0.02)
        P_after = ekf._P.clone()

        # Contact envs should have much larger P growth
        dP_contact = P_after[0, 3, 3] - P_before[0, 3, 3]
        dP_flight = P_after[2, 3, 3] - P_before[2, 3, 3]
        ratio = dP_contact / dP_flight
        # Expected ratio: (50.0/0.40)^2 ≈ 15625 (just Q contribution)
        # In practice F@P@F' contributes too, but ratio should be >>10
        self.assertGreater(ratio.item(), 100.0,
                           f"Contact/flight P ratio {ratio:.1f} too small")

    def test_contact_aware_disabled(self):
        """When contact_aware=False, all envs use the same q_vel."""
        cfg = BallEKFConfig(
            q_vel=7.0,
            q_vel_contact=50.0,
            contact_aware=False,
        )
        ekf = BallEKF(num_envs=4, device="cpu", cfg=cfg)

        # One env at contact height, one in flight — should get same Q
        init_pos = torch.tensor([
            [0.0, 0.0, 0.020],  # would be contact if enabled
            [0.0, 0.0, 0.020],
            [0.0, 0.0, 0.200],
            [0.0, 0.0, 0.200],
        ])
        ekf.reset(torch.arange(4), init_pos)

        P_before = ekf._P.clone()
        ekf.predict(dt=0.02)
        P_after = ekf._P.clone()

        # All envs should have similar P growth (only differing by F due to drag)
        dP_0 = (P_after[0, 3, 3] - P_before[0, 3, 3]).item()
        dP_2 = (P_after[2, 3, 3] - P_before[2, 3, 3]).item()
        # Should be roughly equal (drag difference is tiny)
        self.assertAlmostEqual(dP_0, dP_2, delta=0.01,
                               msg=f"contact_aware=False but P differs: {dP_0:.4f} vs {dP_2:.4f}")

    def test_ekf_accuracy_free_flight_with_contact_aware(self):
        """EKF with contact_aware=True should track a free-flight ball accurately
        (low q_vel → good smoothing)."""
        cfg = BallEKFConfig(
            q_vel=0.40,
            q_vel_contact=50.0,
            contact_aware=True,
            contact_z_threshold=0.025,
        )
        ekf = BallEKF(num_envs=1, device="cpu", cfg=cfg)

        # Simulate ball dropping from Z=0.10 with no noise
        pos = torch.tensor([[0.0, 0.0, 0.10]])
        vel = torch.tensor([[0.0, 0.0, 0.0]])
        ekf.reset(torch.tensor([0]), pos, vel)

        dt = 0.02
        g = -9.81
        for step in range(10):
            t = (step + 1) * dt
            # True position (free-fall)
            true_z = 0.10 + 0.5 * g * t**2
            true_vz = g * t
            # Add small noise to measurement
            noise = torch.randn(1) * 0.002
            z_meas = torch.tensor([[0.0, 0.0, true_z]]) + noise.unsqueeze(-1) * torch.tensor([[0, 0, 1.0]])
            detected = torch.tensor([True])
            ekf.step(z_meas, detected, dt=dt)

        # After 10 steps (0.2s), ball should be at Z ≈ 0.10 - 0.5*9.81*0.04 ≈ -0.096
        # EKF should track closely (low q_vel = trust dynamics)
        pos_err = abs(ekf.pos[0, 2].item() - (0.10 + 0.5 * g * 0.2**2))
        self.assertLess(pos_err, 0.01,
                        f"Free-flight tracking error {pos_err*1000:.1f}mm too large")

    def test_ekf_tracks_through_contact_bounce(self):
        """EKF should handle a ball bouncing (free → contact → free) without diverging."""
        cfg = BallEKFConfig(
            q_vel=0.40,
            q_vel_contact=50.0,
            contact_aware=True,
            contact_z_threshold=0.025,
        )
        ekf = BallEKF(num_envs=1, device="cpu", cfg=cfg)

        # Start in free flight above paddle
        pos = torch.tensor([[0.0, 0.0, 0.10]])
        vel = torch.tensor([[0.0, 0.0, -1.0]])  # falling
        ekf.reset(torch.tensor([0]), pos, vel)

        dt = 0.02
        # Simulate: falling → contact at Z≈0.02 → bounce up
        trajectory_z = [
            0.08, 0.06, 0.04, 0.025, 0.020,   # falling
            0.020, 0.020,                        # contact (2 steps on paddle)
            0.025, 0.04, 0.06, 0.08, 0.10,     # bouncing up
        ]
        for z in trajectory_z:
            z_meas = torch.tensor([[0.0, 0.0, z]])
            detected = torch.tensor([True])
            ekf.step(z_meas, detected, dt=dt)

        # EKF should not have diverged (pos should be finite and near last measurement)
        self.assertTrue(torch.isfinite(ekf.pos).all(), "EKF diverged (NaN/Inf)")
        z_err = abs(ekf.pos[0, 2].item() - 0.10)
        self.assertLess(z_err, 0.05,
                        f"Post-bounce tracking error {z_err*1000:.1f}mm too large")

    def test_backward_compat_default_config(self):
        """Default BallEKFConfig has contact_aware=True and reasonable defaults."""
        cfg = BallEKFConfig()
        self.assertTrue(cfg.contact_aware)
        self.assertLess(cfg.q_vel, cfg.q_vel_contact)
        self.assertGreater(cfg.contact_z_threshold, 0.0)


if __name__ == "__main__":
    unittest.main()
