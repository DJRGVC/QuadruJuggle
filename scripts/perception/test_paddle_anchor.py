#!/usr/bin/env python3
"""Unit tests for EKF paddle-anchor virtual measurement.

Tests that the anchor measurement:
1. Fires only when both starved AND in contact zone
2. Corrects position drift toward paddle position
3. Zeros velocity for anchored envs
4. Respects anchor_enabled flag
5. Uses the anchor's own R (not camera R)
6. Does not reset steps_since_measurement

Usage:
    python scripts/perception/test_paddle_anchor.py
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


class TestPaddleAnchor(unittest.TestCase):
    """Tests for paddle_anchor_update()."""

    def _make_ekf(self, **kwargs) -> BallEKF:
        """Create an EKF with anchor enabled and contact-aware defaults."""
        defaults = dict(
            anchor_enabled=True,
            anchor_r_pos=0.005,
            anchor_min_starve_steps=5,
            contact_aware=True,
            contact_z_threshold=0.025,
            p_clamp_enabled=False,  # disable clamping to isolate anchor effect
        )
        defaults.update(kwargs)
        cfg = BallEKFConfig(**defaults)
        return BallEKF(num_envs=4, device="cpu", cfg=cfg)

    def test_anchor_fires_when_starved_and_contact(self):
        """Anchor fires for envs that are both starved and in contact zone."""
        ekf = self._make_ekf()
        # Place all envs in contact zone (z=0.02 < 0.025 threshold)
        init_pos = torch.tensor([[0.0, 0.0, 0.02]] * 4)
        ekf.reset(torch.arange(4), init_pos)

        # Run predict-only steps to reach starve threshold
        for _ in range(6):
            ekf.predict(0.005)
            ekf._steps_since_measurement += 1

        paddle_pos = torch.tensor([[0.0, 0.0, 0.02]] * 4)
        n = ekf.paddle_anchor_update(paddle_pos)
        self.assertEqual(n, 4, "All 4 envs should be anchored")

    def test_anchor_skips_non_starved(self):
        """Envs with recent camera measurements are not anchored."""
        ekf = self._make_ekf(anchor_min_starve_steps=5)
        init_pos = torch.tensor([[0.0, 0.0, 0.02]] * 4)
        ekf.reset(torch.arange(4), init_pos)

        # Only starve envs 0,1 (6 steps); envs 2,3 have recent measurement (2 steps)
        ekf._steps_since_measurement[:] = torch.tensor([6, 6, 2, 2])

        paddle_pos = torch.tensor([[0.0, 0.0, 0.02]] * 4)
        n = ekf.paddle_anchor_update(paddle_pos)
        self.assertEqual(n, 2, "Only starved envs 0,1 should be anchored")

    def test_anchor_skips_flight_envs(self):
        """Envs with ball in flight (z > contact threshold) are not anchored."""
        ekf = self._make_ekf()
        init_pos = torch.tensor([
            [0.0, 0.0, 0.02],   # contact
            [0.0, 0.0, 0.30],   # flight
            [0.0, 0.0, 0.02],   # contact
            [0.0, 0.0, 0.50],   # flight
        ])
        ekf.reset(torch.arange(4), init_pos)
        ekf._steps_since_measurement[:] = 10  # all starved

        paddle_pos = torch.tensor([[0.0, 0.0, 0.02]] * 4)
        n = ekf.paddle_anchor_update(paddle_pos)
        self.assertEqual(n, 2, "Only contact envs 0,2 should be anchored")

    def test_anchor_corrects_position_drift(self):
        """Position drifts toward paddle position after anchor update."""
        ekf = self._make_ekf()
        # Start ball at contact height but offset in XY
        init_pos = torch.tensor([[0.05, -0.03, 0.02]] * 4)
        ekf.reset(torch.arange(4), init_pos)

        # Starve then anchor at paddle centre
        ekf._steps_since_measurement[:] = 10
        paddle_pos = torch.tensor([[0.0, 0.0, 0.02]] * 4)

        pos_before = ekf.pos.clone()
        ekf.paddle_anchor_update(paddle_pos)
        pos_after = ekf.pos.clone()

        # Position should move toward paddle centre
        dist_before = (pos_before - paddle_pos).norm(dim=-1)
        dist_after = (pos_after - paddle_pos).norm(dim=-1)
        for i in range(4):
            self.assertLess(
                dist_after[i].item(), dist_before[i].item(),
                f"Env {i}: position should move closer to paddle after anchor"
            )

    def test_anchor_zeros_velocity(self):
        """Anchored envs have velocity zeroed (ball on paddle is stationary)."""
        ekf = self._make_ekf()
        init_pos = torch.tensor([[0.0, 0.0, 0.02]] * 4)
        init_vel = torch.tensor([[0.5, -0.3, 1.0]] * 4)
        ekf.reset(torch.arange(4), init_pos, init_vel)
        ekf._steps_since_measurement[:] = 10

        paddle_pos = torch.tensor([[0.0, 0.0, 0.02]] * 4)
        ekf.paddle_anchor_update(paddle_pos)

        for i in range(4):
            vel_norm = ekf.vel[i].norm().item()
            self.assertAlmostEqual(vel_norm, 0.0, places=5,
                                   msg=f"Env {i}: velocity should be zero after anchor")

    def test_anchor_disabled_returns_zero(self):
        """When anchor_enabled=False, no envs are anchored."""
        ekf = self._make_ekf(anchor_enabled=False)
        init_pos = torch.tensor([[0.0, 0.0, 0.02]] * 4)
        ekf.reset(torch.arange(4), init_pos)
        ekf._steps_since_measurement[:] = 10

        paddle_pos = torch.tensor([[0.0, 0.0, 0.02]] * 4)
        n = ekf.paddle_anchor_update(paddle_pos)
        self.assertEqual(n, 0)

    def test_anchor_does_not_reset_starve_counter(self):
        """steps_since_measurement is not reset by anchor (only camera resets it)."""
        ekf = self._make_ekf()
        init_pos = torch.tensor([[0.0, 0.0, 0.02]] * 4)
        ekf.reset(torch.arange(4), init_pos)
        ekf._steps_since_measurement[:] = 10

        paddle_pos = torch.tensor([[0.0, 0.0, 0.02]] * 4)
        ekf.paddle_anchor_update(paddle_pos)

        # steps_since_measurement should still be 10 (not reset)
        for i in range(4):
            self.assertEqual(ekf.steps_since_measurement[i].item(), 10)

    def test_anchor_reduces_covariance(self):
        """Anchor measurement should reduce P (inject information)."""
        ekf = self._make_ekf()
        init_pos = torch.tensor([[0.0, 0.0, 0.02]] * 4)
        ekf.reset(torch.arange(4), init_pos)

        # Let P grow via predictions
        for _ in range(20):
            ekf.predict(0.005)
            ekf._steps_since_measurement += 1
        # Force ball back into contact zone for anchor eligibility
        ekf._x[:, 2] = 0.02

        P_trace_before = ekf._P[:, :3, :3].diagonal(dim1=-2, dim2=-1).sum(dim=-1).clone()

        paddle_pos = torch.tensor([[0.0, 0.0, 0.02]] * 4)
        ekf.paddle_anchor_update(paddle_pos)

        P_trace_after = ekf._P[:, :3, :3].diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        for i in range(4):
            self.assertLess(
                P_trace_after[i].item(), P_trace_before[i].item(),
                f"Env {i}: position covariance trace should decrease after anchor"
            )

    def test_anchor_mixed_envs(self):
        """Only eligible envs are modified; non-eligible envs are untouched."""
        ekf = self._make_ekf()
        # Env 0: starved + contact → anchored
        # Env 1: starved + flight → NOT anchored
        # Env 2: not starved + contact → NOT anchored
        # Env 3: starved + contact → anchored
        init_pos = torch.tensor([
            [0.05, 0.0, 0.02],   # contact, will drift
            [0.0, 0.0, 0.30],    # flight
            [0.05, 0.0, 0.02],   # contact
            [0.05, 0.0, 0.02],   # contact, will drift
        ])
        init_vel = torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ])
        ekf.reset(torch.arange(4), init_pos, init_vel)
        ekf._steps_since_measurement[:] = torch.tensor([10, 10, 2, 10])

        pos_before = ekf.pos.clone()
        vel_before = ekf.vel.clone()

        paddle_pos = torch.tensor([[0.0, 0.0, 0.02]] * 4)
        n = ekf.paddle_anchor_update(paddle_pos)
        self.assertEqual(n, 2, "Envs 0,3 should be anchored")

        # Env 1 (flight): unchanged
        torch.testing.assert_close(ekf.pos[1], pos_before[1])
        torch.testing.assert_close(ekf.vel[1], vel_before[1])
        # Env 2 (not starved): unchanged
        torch.testing.assert_close(ekf.pos[2], pos_before[2])
        torch.testing.assert_close(ekf.vel[2], vel_before[2])

        # Env 0 (anchored): velocity zeroed
        self.assertAlmostEqual(ekf.vel[0].norm().item(), 0.0, places=5)
        # Env 3 (anchored): velocity zeroed
        self.assertAlmostEqual(ekf.vel[3].norm().item(), 0.0, places=5)


    def test_anchor_world_frame_threshold(self):
        """Anchor fires with world-frame Z coordinates and matching contact_z_threshold."""
        # Simulate world-frame setup: paddle at Z=0.47, ball centre at Z=0.49
        paddle_z_world = 0.47 + 0.020  # paddle surface + ball radius
        contact_z_world = paddle_z_world + 0.010  # 10mm margin → 0.50
        ekf = self._make_ekf(contact_z_threshold=contact_z_world)

        # Ball resting on paddle in world frame
        init_pos = torch.tensor([[0.01, -0.01, paddle_z_world]] * 4)
        ekf.reset(torch.arange(4), init_pos)

        # Starve all envs
        for _ in range(6):
            ekf.predict(0.005)
            ekf._steps_since_measurement += 1
        # Force ball Z back to paddle (gravity would pull it down during predict)
        ekf._x[:, 2] = paddle_z_world

        paddle_pos = torch.tensor([[0.0, 0.0, paddle_z_world]] * 4)
        n = ekf.paddle_anchor_update(paddle_pos)
        self.assertEqual(n, 4, "All 4 envs should anchor in world frame")

        # Ball in flight at Z=0.65 (above contact_z=0.50) should NOT anchor
        ekf._x[:, 2] = 0.65
        n = ekf.paddle_anchor_update(paddle_pos)
        self.assertEqual(n, 0, "Flight envs should not anchor")


if __name__ == "__main__":
    unittest.main()
