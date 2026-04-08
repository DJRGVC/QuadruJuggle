"""Tests for phase-separated NIS tracking in BallEKF.

NIS is split into free-flight and contact phases so we can validate
that q_vel (free-flight) and q_vel_contact are independently well-tuned.

Run:  uv run --active python scripts/perception/test_nis_phase.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

_PERCEPTION = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "source", "go1_ball_balance", "go1_ball_balance", "perception",
))
sys.path.insert(0, _PERCEPTION)

from ball_ekf import BallEKF, BallEKFConfig


def _phase_cfg(**overrides) -> BallEKFConfig:
    """Config with contact-aware enabled, no gravity/adaptive-R."""
    defaults = dict(
        contact_aware=True,
        contact_z_threshold=0.025,
        q_vel=0.40,
        q_vel_contact=50.0,
        adaptive_r=False,
        gravity_z=0.0,
        nis_gate_enabled=False,
    )
    defaults.update(overrides)
    return BallEKFConfig(**defaults)


class TestPhaseNISProperties(unittest.TestCase):
    """Basic property access and initial values."""

    def test_initial_values_zero(self):
        ekf = BallEKF(num_envs=4, device="cpu", cfg=_phase_cfg())
        self.assertEqual(ekf.mean_nis_flight, 0.0)
        self.assertEqual(ekf.mean_nis_contact, 0.0)

    def test_properties_exist(self):
        ekf = BallEKF(num_envs=4, device="cpu", cfg=_phase_cfg())
        _ = ekf.mean_nis_flight
        _ = ekf.mean_nis_contact


class TestPhaseNISSeparation(unittest.TestCase):
    """NIS accumulates separately for flight vs contact envs."""

    def test_flight_only(self):
        """All envs in free-flight → only flight NIS accumulated."""
        cfg = _phase_cfg()
        ekf = BallEKF(num_envs=4, device="cpu", cfg=cfg)
        # Init ball at z=0.10 (above contact threshold 0.025)
        ekf.reset(
            torch.arange(4),
            torch.tensor([[0, 0, 0.10]] * 4),
        )
        z = torch.tensor([[0, 0, 0.10]] * 4) + torch.randn(4, 3) * 0.005
        ekf.step(z, torch.ones(4, dtype=torch.bool), dt=0.02)

        self.assertGreater(ekf._nis_count_flight, 0)
        self.assertEqual(ekf._nis_count_contact, 0)
        self.assertGreater(ekf.mean_nis_flight, 0.0)
        self.assertEqual(ekf.mean_nis_contact, 0.0)

    def test_contact_only(self):
        """All envs in contact → only contact NIS accumulated."""
        cfg = _phase_cfg()
        ekf = BallEKF(num_envs=4, device="cpu", cfg=cfg)
        # Init ball at z=0.015 (below contact threshold 0.025)
        ekf.reset(
            torch.arange(4),
            torch.tensor([[0, 0, 0.015]] * 4),
        )
        z = torch.tensor([[0, 0, 0.015]] * 4) + torch.randn(4, 3) * 0.005
        ekf.step(z, torch.ones(4, dtype=torch.bool), dt=0.02)

        self.assertEqual(ekf._nis_count_flight, 0)
        self.assertGreater(ekf._nis_count_contact, 0)
        self.assertEqual(ekf.mean_nis_flight, 0.0)
        self.assertGreater(ekf.mean_nis_contact, 0.0)

    def test_mixed_envs(self):
        """Some envs in flight, some in contact → both accumulators populated."""
        cfg = _phase_cfg()
        ekf = BallEKF(num_envs=4, device="cpu", cfg=cfg)
        # Envs 0,1 in flight (z=0.10), envs 2,3 in contact (z=0.015)
        init_pos = torch.tensor([
            [0, 0, 0.10],
            [0, 0, 0.10],
            [0, 0, 0.015],
            [0, 0, 0.015],
        ])
        ekf.reset(torch.arange(4), init_pos)
        z = init_pos + torch.randn(4, 3) * 0.003
        ekf.step(z, torch.ones(4, dtype=torch.bool), dt=0.02)

        self.assertGreater(ekf._nis_count_flight, 0)
        self.assertGreater(ekf._nis_count_contact, 0)
        # Total should equal sum
        self.assertEqual(
            ekf._nis_count,
            ekf._nis_count_flight + ekf._nis_count_contact,
        )

    def test_contact_nis_lower_than_flight(self):
        """Contact NIS should be lower because q_vel_contact is much larger."""
        cfg = _phase_cfg()
        ekf = BallEKF(num_envs=100, device="cpu", cfg=cfg)
        torch.manual_seed(42)
        # Half flight, half contact
        init_pos = torch.zeros(100, 3)
        init_pos[:50, 2] = 0.10   # flight
        init_pos[50:, 2] = 0.015  # contact
        ekf.reset(torch.arange(100), init_pos)

        # Run several steps with consistent small noise
        for _ in range(20):
            z = ekf.pos.clone() + torch.randn(100, 3) * 0.003
            # Keep z heights in the right phase region
            z[:50, 2] = z[:50, 2].clamp(min=0.05)
            z[50:, 2] = z[50:, 2].clamp(max=0.020)
            ekf.step(z, torch.ones(100, dtype=torch.bool), dt=0.02)

        # Contact phase should have lower NIS due to inflated Q
        self.assertGreater(ekf.mean_nis_flight, 0.0)
        self.assertGreater(ekf.mean_nis_contact, 0.0)
        self.assertGreater(ekf.mean_nis_flight, ekf.mean_nis_contact)


class TestPhaseNISReset(unittest.TestCase):
    """reset_nis() clears phase accumulators too."""

    def test_reset_clears_phase(self):
        cfg = _phase_cfg()
        ekf = BallEKF(num_envs=4, device="cpu", cfg=cfg)
        ekf.reset(torch.arange(4), torch.tensor([[0, 0, 0.10]] * 4))
        z = torch.tensor([[0, 0, 0.10]] * 4) + torch.randn(4, 3) * 0.005
        ekf.step(z, torch.ones(4, dtype=torch.bool), dt=0.02)
        self.assertGreater(ekf._nis_count_flight, 0)

        ekf.reset_nis()
        self.assertEqual(ekf._nis_count_flight, 0)
        self.assertEqual(ekf._nis_count_contact, 0)
        self.assertEqual(ekf._nis_sum_flight, 0.0)
        self.assertEqual(ekf._nis_sum_contact, 0.0)
        self.assertEqual(ekf.mean_nis_flight, 0.0)
        self.assertEqual(ekf.mean_nis_contact, 0.0)


class TestPhaseNISNoContactAware(unittest.TestCase):
    """When contact_aware=False, all NIS goes to flight bucket."""

    def test_all_goes_to_flight(self):
        cfg = _phase_cfg(contact_aware=False)
        ekf = BallEKF(num_envs=4, device="cpu", cfg=cfg)
        ekf.reset(torch.arange(4), torch.tensor([[0, 0, 0.015]] * 4))
        z = torch.tensor([[0, 0, 0.015]] * 4) + torch.randn(4, 3) * 0.005
        ekf.step(z, torch.ones(4, dtype=torch.bool), dt=0.02)

        # Even though z < threshold, contact_aware=False → all flight
        self.assertGreater(ekf._nis_count_flight, 0)
        self.assertEqual(ekf._nis_count_contact, 0)


class TestPhaseNISUndetected(unittest.TestCase):
    """Undetected envs don't contribute to phase NIS."""

    def test_undetected_excluded(self):
        cfg = _phase_cfg()
        ekf = BallEKF(num_envs=4, device="cpu", cfg=cfg)
        ekf.reset(torch.arange(4), torch.tensor([[0, 0, 0.10]] * 4))
        detected = torch.tensor([True, True, False, False])
        z = torch.tensor([[0, 0, 0.10]] * 4) + torch.randn(4, 3) * 0.005
        ekf.step(z, detected, dt=0.02)

        self.assertEqual(ekf._nis_count_flight, 2)  # only detected
        self.assertEqual(ekf._nis_count_contact, 0)


class TestPhaseNISDiagnostics(unittest.TestCase):
    """Pipeline diagnostics dict includes phase NIS keys."""

    def test_diagnostics_keys(self):
        """Verify PerceptionPipeline.diagnostics includes phase NIS fields."""
        # We test the BallEKF-level fields since PerceptionPipeline needs Isaac Lab
        cfg = _phase_cfg()
        ekf = BallEKF(num_envs=4, device="cpu", cfg=cfg)
        ekf.reset(torch.arange(4), torch.tensor([[0, 0, 0.10]] * 4))
        z = torch.tensor([[0, 0, 0.10]] * 4) + torch.randn(4, 3) * 0.003
        ekf.step(z, torch.ones(4, dtype=torch.bool), dt=0.02)

        # Simulate what diagnostics property does
        nis_flight = ekf.mean_nis_flight
        nis_contact = ekf.mean_nis_contact
        count_flight = ekf._nis_count_flight
        count_contact = ekf._nis_count_contact
        ekf.reset_nis()

        self.assertGreater(nis_flight, 0.0)
        self.assertEqual(nis_contact, 0.0)
        self.assertEqual(count_flight, 4)
        self.assertEqual(count_contact, 0)
        # After reset all zero
        self.assertEqual(ekf.mean_nis_flight, 0.0)


if __name__ == "__main__":
    unittest.main()
