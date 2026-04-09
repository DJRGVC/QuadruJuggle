"""Tests for NIS (chi-squared) gating in BallEKF.

NIS gating rejects outlier measurements where the Normalized Innovation
Squared exceeds a chi-squared threshold, preventing detector glitches
from corrupting the EKF state estimate.

Run:  uv run --active python scripts/perception/test_nis_gating.py -v
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

# Direct import to avoid Isaac Lab __init__.py chain (needs pxr/sim)
_PERCEPTION = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "source", "go1_ball_balance", "go1_ball_balance", "perception",
))
sys.path.insert(0, _PERCEPTION)

from ball_ekf import BallEKF, BallEKFConfig


def _gating_cfg(**overrides) -> BallEKFConfig:
    """Config with gating active, no gravity/contact/adaptive-R noise.
    warmup=0 so gating is active from the first update (tests control
    convergence explicitly)."""
    defaults = dict(
        nis_gate_enabled=True,
        nis_gate_threshold=11.345,
        nis_gate_warmup=0,
        contact_aware=False,
        adaptive_r=False,
        gravity_z=0.0,
    )
    defaults.update(overrides)
    return BallEKFConfig(**defaults)


class TestNISGateConfig(unittest.TestCase):
    """Config defaults and toggle."""

    def test_gate_enabled_by_default(self):
        cfg = BallEKFConfig()
        self.assertTrue(cfg.nis_gate_enabled)

    def test_default_threshold_is_chi2_99(self):
        cfg = BallEKFConfig()
        self.assertAlmostEqual(cfg.nis_gate_threshold, 11.345, places=2)

    def test_default_warmup_is_50(self):
        cfg = BallEKFConfig()
        self.assertEqual(cfg.nis_gate_warmup, 50)

    def test_gate_can_be_disabled(self):
        cfg = BallEKFConfig(nis_gate_enabled=False)
        ekf = BallEKF(num_envs=1, cfg=cfg)
        self.assertFalse(ekf.cfg.nis_gate_enabled)


class TestNISGateRejectsOutliers(unittest.TestCase):
    """Outlier measurements are rejected; state stays clean."""

    def setUp(self):
        self.cfg = _gating_cfg()
        self.ekf = BallEKF(num_envs=4, cfg=self.cfg)
        pos = torch.zeros(4, 3)
        self.ekf.reset(torch.arange(4), pos)

    def test_consistent_measurement_accepted(self):
        """Small innovation -> NIS < threshold -> measurement accepted."""
        z = torch.tensor([[0.005, 0.005, 0.005]] * 4)
        det = torch.ones(4, dtype=torch.bool)
        self.ekf.predict(0.02)
        self.ekf.update(z, det)
        self.assertTrue((self.ekf.pos[:, 0] > 0).all())
        self.assertEqual(self.ekf.gate_rejection_count, 0)

    def test_outlier_measurement_rejected(self):
        """Huge innovation -> NIS >> threshold -> measurement rejected."""
        for _ in range(5):
            self.ekf.predict(0.02)
            self.ekf.update(torch.zeros(4, 3), torch.ones(4, dtype=torch.bool))

        pos_before = self.ekf.pos.clone()
        z_outlier = torch.tensor([[5.0, 5.0, 5.0]] * 4)
        self.ekf.predict(0.02)
        self.ekf.update(z_outlier, torch.ones(4, dtype=torch.bool))

        drift = (self.ekf.pos - pos_before).norm(dim=-1)
        self.assertTrue((drift < 0.5).all(), f"State drifted {drift} toward outlier")
        self.assertEqual(self.ekf.gate_rejection_count, 4)

    def test_mixed_accept_reject(self):
        """Some envs have outliers, others don't."""
        for _ in range(5):
            self.ekf.predict(0.02)
            self.ekf.update(torch.zeros(4, 3), torch.ones(4, dtype=torch.bool))

        pos_before = self.ekf.pos.clone()
        z = torch.zeros(4, 3)
        z[0] = torch.tensor([5.0, 5.0, 5.0])   # outlier
        z[1] = torch.tensor([0.001, 0.001, 0.001])  # fine
        z[2] = torch.tensor([-5.0, -5.0, -5.0])  # outlier
        z[3] = torch.tensor([0.002, -0.001, 0.0])   # fine
        self.ekf.predict(0.02)
        self.ekf.update(z, torch.ones(4, dtype=torch.bool))

        drift_0 = (self.ekf.pos[0] - pos_before[0]).norm()
        drift_2 = (self.ekf.pos[2] - pos_before[2]).norm()
        self.assertLess(drift_0, 0.5)
        self.assertLess(drift_2, 0.5)
        self.assertTrue(self.ekf.pos[1, 0] > pos_before[1, 0])
        self.assertEqual(self.ekf.gate_rejection_count, 2)


class TestNISGateDisabled(unittest.TestCase):
    """When gating is disabled, all measurements are accepted."""

    def test_outlier_accepted_when_disabled(self):
        cfg = _gating_cfg(nis_gate_enabled=False)
        ekf = BallEKF(num_envs=1, cfg=cfg)
        ekf.reset(torch.tensor([0]), torch.zeros(1, 3))

        for _ in range(5):
            ekf.predict(0.02)
            ekf.update(torch.zeros(1, 3), torch.ones(1, dtype=torch.bool))

        pos_before = ekf.pos.clone()
        ekf.predict(0.02)
        ekf.update(torch.tensor([[5.0, 5.0, 5.0]]), torch.ones(1, dtype=torch.bool))

        drift = (ekf.pos - pos_before).norm()
        self.assertGreater(drift, 0.1)
        self.assertEqual(ekf.gate_rejection_count, 0)


class TestNISGateWarmup(unittest.TestCase):
    """Per-env warm-up: gating skipped until update_count >= warmup."""

    def test_warmup_skips_gating(self):
        """During warm-up, even large innovations are accepted."""
        cfg = _gating_cfg(nis_gate_warmup=10)
        ekf = BallEKF(num_envs=1, cfg=cfg)
        ekf.reset(torch.tensor([0]), torch.zeros(1, 3))

        # 5 updates (within warm-up) — outlier should be accepted
        for _ in range(5):
            ekf.predict(0.02)
            ekf.update(torch.zeros(1, 3), torch.ones(1, dtype=torch.bool))

        pos_before = ekf.pos.clone()
        ekf.predict(0.02)
        ekf.update(torch.tensor([[5.0, 5.0, 5.0]]), torch.ones(1, dtype=torch.bool))

        # Should jump — still in warm-up
        drift = (ekf.pos - pos_before).norm()
        self.assertGreater(drift, 0.1, "Outlier should be accepted during warm-up")
        self.assertEqual(ekf.gate_rejection_count, 0)

    def test_gating_activates_after_warmup(self):
        """After warm-up, outliers are rejected."""
        warmup = 10
        cfg = _gating_cfg(nis_gate_warmup=warmup)
        ekf = BallEKF(num_envs=1, cfg=cfg)
        ekf.reset(torch.tensor([0]), torch.zeros(1, 3))

        # Run past warm-up with consistent measurements
        for _ in range(warmup + 5):
            ekf.predict(0.02)
            ekf.update(torch.zeros(1, 3), torch.ones(1, dtype=torch.bool))

        pos_before = ekf.pos.clone()
        ekf.predict(0.02)
        ekf.update(torch.tensor([[5.0, 5.0, 5.0]]), torch.ones(1, dtype=torch.bool))

        drift = (ekf.pos - pos_before).norm()
        self.assertLess(drift, 0.5, "Outlier should be rejected after warm-up")
        self.assertEqual(ekf.gate_rejection_count, 1)

    def test_warmup_resets_on_env_reset(self):
        """Resetting an env restarts its warm-up counter."""
        cfg = _gating_cfg(nis_gate_warmup=5)
        ekf = BallEKF(num_envs=2, cfg=cfg)
        ekf.reset(torch.arange(2), torch.zeros(2, 3))

        # Run past warm-up
        for _ in range(10):
            ekf.predict(0.02)
            ekf.update(torch.zeros(2, 3), torch.ones(2, dtype=torch.bool))

        # Reset env 0 only
        ekf.reset(torch.tensor([0]), torch.zeros(1, 3))
        self.assertEqual(ekf._update_count[0].item(), 0)
        self.assertGreater(ekf._update_count[1].item(), 0)


class TestNISGateDiagnostics(unittest.TestCase):
    """Diagnostic counters and reset."""

    def test_rejection_rate_zero_for_clean_data(self):
        cfg = _gating_cfg()
        ekf = BallEKF(num_envs=8, cfg=cfg)
        ekf.reset(torch.arange(8), torch.zeros(8, 3))

        for _ in range(20):
            z = torch.randn(8, 3) * 0.001
            ekf.predict(0.02)
            ekf.update(z, torch.ones(8, dtype=torch.bool))

        self.assertAlmostEqual(ekf.gate_rejection_rate, 0.0, places=2)

    def test_reset_gate_stats(self):
        cfg = _gating_cfg()
        ekf = BallEKF(num_envs=1, cfg=cfg)
        ekf.reset(torch.tensor([0]), torch.zeros(1, 3))

        for _ in range(5):
            ekf.predict(0.02)
            ekf.update(torch.zeros(1, 3), torch.ones(1, dtype=torch.bool))

        rejected, total = ekf.reset_gate_stats()
        self.assertEqual(rejected, 0)
        self.assertEqual(total, 5)
        self.assertEqual(ekf.gate_rejection_count, 0)

    def test_rejection_count_with_outliers(self):
        cfg = _gating_cfg()
        ekf = BallEKF(num_envs=2, cfg=cfg)
        ekf.reset(torch.arange(2), torch.zeros(2, 3))

        for _ in range(10):
            ekf.predict(0.02)
            ekf.update(torch.zeros(2, 3), torch.ones(2, dtype=torch.bool))

        z = torch.zeros(2, 3)
        z[0] = torch.tensor([10.0, 10.0, 10.0])
        ekf.predict(0.02)
        ekf.update(z, torch.ones(2, dtype=torch.bool))

        self.assertEqual(ekf._gate_total_count, 22)  # 11 steps * 2 envs
        self.assertEqual(ekf.gate_rejection_count, 1)


class TestNISGateWithSpin(unittest.TestCase):
    """Gating works correctly in 9D spin mode."""

    def test_gating_in_9d_mode(self):
        cfg = _gating_cfg(enable_spin=True)
        ekf = BallEKF(num_envs=2, cfg=cfg)
        ekf.reset(torch.arange(2), torch.zeros(2, 3))
        self.assertEqual(ekf.state_dim, 9)

        for _ in range(10):
            ekf.predict(0.02)
            ekf.update(torch.zeros(2, 3), torch.ones(2, dtype=torch.bool))

        pos_before = ekf.pos.clone()
        z_outlier = torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        ekf.predict(0.02)
        ekf.update(z_outlier, torch.ones(2, dtype=torch.bool))

        drift_1 = (ekf.pos[1] - pos_before[1]).norm()
        self.assertLess(drift_1, 0.5)
        self.assertGreaterEqual(ekf.gate_rejection_count, 1)


class TestNISGateContactAware(unittest.TestCase):
    """Gating interacts correctly with contact-aware process noise."""

    def test_contact_inflated_Q_allows_larger_innovations(self):
        """During contact, process noise inflated -> S larger -> NIS smaller."""
        cfg = _gating_cfg(contact_aware=True)
        ekf = BallEKF(num_envs=1, cfg=cfg)
        ekf.reset(torch.tensor([0]), torch.tensor([[0.0, 0.0, 0.02]]))

        for _ in range(3):
            ekf.predict(0.02)
            ekf.update(torch.tensor([[0.0, 0.0, 0.02]]), torch.ones(1, dtype=torch.bool))

        z_moderate = torch.tensor([[0.0, 0.0, 0.5]])
        ekf.predict(0.02)
        ekf.update(z_moderate, torch.ones(1, dtype=torch.bool))
        # During contact, q_vel=50 -> large S -> small NIS -> not gated


class TestNISGateCustomThreshold(unittest.TestCase):
    """Custom threshold values work correctly."""

    def test_very_tight_threshold_rejects_more(self):
        cfg = _gating_cfg(nis_gate_threshold=1.0)
        ekf = BallEKF(num_envs=4, cfg=cfg)
        ekf.reset(torch.arange(4), torch.zeros(4, 3))

        for _ in range(10):
            ekf.predict(0.02)
            ekf.update(torch.zeros(4, 3), torch.ones(4, dtype=torch.bool))

        z = torch.ones(4, 3) * 0.1
        ekf.predict(0.02)
        ekf.update(z, torch.ones(4, dtype=torch.bool))

        self.assertGreater(ekf.gate_rejection_count, 0)

    def test_very_loose_threshold_accepts_more(self):
        cfg = _gating_cfg(nis_gate_threshold=1e9)
        ekf = BallEKF(num_envs=4, cfg=cfg)
        ekf.reset(torch.arange(4), torch.zeros(4, 3))

        for _ in range(5):
            ekf.predict(0.02)
            ekf.update(torch.zeros(4, 3), torch.ones(4, dtype=torch.bool))

        z = torch.ones(4, 3) * 1.0
        ekf.predict(0.02)
        ekf.update(z, torch.ones(4, dtype=torch.bool))

        self.assertEqual(ekf.gate_rejection_count, 0)


class TestNISGateUndetectedUnaffected(unittest.TestCase):
    """Undetected envs not counted in gate stats."""

    def test_undetected_not_counted(self):
        cfg = _gating_cfg()
        ekf = BallEKF(num_envs=4, cfg=cfg)
        ekf.reset(torch.arange(4), torch.zeros(4, 3))

        det = torch.tensor([True, False, True, False])
        ekf.predict(0.02)
        ekf.update(torch.zeros(4, 3), det)

        # Only 2 detected envs should be counted
        self.assertEqual(ekf._gate_total_count, 2)


class TestAdaptiveR(unittest.TestCase):
    """Adaptive R scales XY and Z noise with estimated ball height."""

    def test_adaptive_r_xy_scales_with_height(self):
        """R_xy should be smaller at low z and larger at high z."""
        cfg = BallEKFConfig(
            adaptive_r=True,
            r_xy_per_metre=0.0025,
            r_xy_floor=0.0005,
            q_vel=0.40,
            contact_aware=False,
            nis_gate_enabled=False,
        )
        ekf = BallEKF(num_envs=2, cfg=cfg)
        # Env 0 at z=0.10, Env 1 at z=0.80
        pos_low = torch.tensor([[0.0, 0.0, 0.10]])
        pos_high = torch.tensor([[0.0, 0.0, 0.80]])
        ekf.reset(torch.tensor([0]), pos_low)
        ekf.reset(torch.tensor([1]), pos_high)

        # Run one predict+update cycle with perfect measurements
        ekf.predict(0.02)
        z = torch.stack([pos_low[0], pos_high[0]])
        ekf.update(z, torch.ones(2, dtype=torch.bool))

        # After update, P should differ: env 1 (high z) has larger R → less
        # measurement trust → larger posterior P
        P_low = ekf._P[0, :3, :3].diag()
        P_high = ekf._P[1, :3, :3].diag()
        # XY posterior covariance should be smaller at low z (tighter R)
        self.assertLess(P_low[0].item(), P_high[0].item(),
                        "P_xx at z=0.10 should be < P_xx at z=0.80")

    def test_adaptive_r_xy_floor(self):
        """R_xy should not go below r_xy_floor even at z≈0."""
        cfg = BallEKFConfig(
            adaptive_r=True,
            r_xy_per_metre=0.0025,
            r_xy_floor=0.0005,
            q_vel=0.40,
            contact_aware=False,
            nis_gate_enabled=False,
        )
        ekf = BallEKF(num_envs=1, cfg=cfg)
        ekf.reset(torch.tensor([0]), torch.tensor([[0.0, 0.0, 0.01]]))

        ekf.predict(0.02)
        z = torch.tensor([[0.0, 0.0, 0.01]])
        ekf.update(z, torch.ones(1, dtype=torch.bool))

        # At z=0.01, sigma_xy = max(0.0025*0.01, 0.0005) = 0.0005
        # P should still be finite and reasonable
        self.assertTrue(torch.isfinite(ekf._P).all())
        self.assertGreater(ekf._P[0, 0, 0].item(), 0.0)

    def test_non_adaptive_r_ignores_height(self):
        """With adaptive_r=False, R is constant regardless of height."""
        cfg = BallEKFConfig(adaptive_r=False, q_vel=0.40, contact_aware=False,
                            nis_gate_enabled=False)
        ekf = BallEKF(num_envs=2, cfg=cfg)
        ekf.reset(torch.tensor([0]), torch.tensor([[0.0, 0.0, 0.10]]))
        ekf.reset(torch.tensor([1]), torch.tensor([[0.0, 0.0, 0.80]]))

        ekf.predict(0.02)
        z = torch.tensor([[0.0, 0.0, 0.10], [0.0, 0.0, 0.80]])
        ekf.update(z, torch.ones(2, dtype=torch.bool))

        # Both envs should have identical XY posterior P (same fixed R)
        P0_xx = ekf._P[0, 0, 0].item()
        P1_xx = ekf._P[1, 0, 0].item()
        self.assertAlmostEqual(P0_xx, P1_xx, places=8,
                               msg="Non-adaptive R should give identical P regardless of height")


if __name__ == "__main__":
    unittest.main()
