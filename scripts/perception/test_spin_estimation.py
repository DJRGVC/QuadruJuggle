#!/usr/bin/env python3
"""Unit tests for 9D spin estimation (Magnus effect) in BallEKF.

Tests cover:
- 9D state initialisation and properties
- Magnus force direction and magnitude
- Spin decay dynamics
- Spin observability from curved trajectories
- Contact-aware spin process noise
- Backward compatibility (6D mode unchanged)

CPU-only, no simulator required.

Usage:
    python scripts/perception/test_spin_estimation.py
    pytest scripts/perception/test_spin_estimation.py -v
"""

import math
import os
import sys
import unittest

import torch

# Direct import to avoid Isaac Lab __init__.py chain
_PERCEPTION = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "source", "go1_ball_balance", "go1_ball_balance", "perception",
))
sys.path.insert(0, _PERCEPTION)
from ball_ekf import BallEKF, BallEKFConfig


class TestSpinInit(unittest.TestCase):
    """Test 9D state initialisation."""

    def test_state_dim_6d(self):
        ekf = BallEKF(4, cfg=BallEKFConfig(enable_spin=False))
        self.assertEqual(ekf.state_dim, 6)
        self.assertEqual(ekf.state.shape, (4, 6))

    def test_state_dim_9d(self):
        ekf = BallEKF(4, cfg=BallEKFConfig(enable_spin=True))
        self.assertEqual(ekf.state_dim, 9)
        self.assertEqual(ekf.state.shape, (4, 9))

    def test_spin_property_9d(self):
        ekf = BallEKF(4, cfg=BallEKFConfig(enable_spin=True))
        self.assertEqual(ekf.spin.shape, (4, 3))
        torch.testing.assert_close(ekf.spin, torch.zeros(4, 3))

    def test_spin_property_6d_returns_zeros(self):
        ekf = BallEKF(4, cfg=BallEKFConfig(enable_spin=False))
        self.assertEqual(ekf.spin.shape, (4, 3))
        torch.testing.assert_close(ekf.spin, torch.zeros(4, 3))

    def test_covariance_shape_9d(self):
        ekf = BallEKF(4, cfg=BallEKFConfig(enable_spin=True))
        self.assertEqual(ekf._P.shape, (4, 9, 9))

    def test_initial_spin_covariance(self):
        cfg = BallEKFConfig(enable_spin=True, p_spin_init=10.0)
        ekf = BallEKF(2, cfg=cfg)
        # Spin block of covariance should be p_spin_init^2 * I
        spin_P = ekf._P[0, 6:9, 6:9]
        expected = torch.eye(3) * 100.0  # 10^2
        torch.testing.assert_close(spin_P, expected)

    def test_measurement_matrix_shape_9d(self):
        ekf = BallEKF(2, cfg=BallEKFConfig(enable_spin=True))
        self.assertEqual(ekf._H.shape, (3, 9))
        # H should select position only
        expected = torch.zeros(3, 9)
        expected[0, 0] = expected[1, 1] = expected[2, 2] = 1.0
        torch.testing.assert_close(ekf._H, expected)


class TestSpinReset(unittest.TestCase):
    """Test reset with spin initialisation."""

    def test_reset_default_zero_spin(self):
        ekf = BallEKF(4, cfg=BallEKFConfig(enable_spin=True))
        ids = torch.tensor([0, 2])
        pos = torch.tensor([[0.1, 0.0, 0.5], [0.0, 0.1, 0.3]])
        ekf.reset(ids, pos)
        torch.testing.assert_close(ekf.spin[0], torch.zeros(3))
        torch.testing.assert_close(ekf.spin[2], torch.zeros(3))

    def test_reset_with_spin(self):
        ekf = BallEKF(4, cfg=BallEKFConfig(enable_spin=True))
        ids = torch.tensor([1])
        pos = torch.tensor([[0.0, 0.0, 0.5]])
        spin = torch.tensor([[10.0, 20.0, 30.0]])
        ekf.reset(ids, pos, init_spin=spin)
        torch.testing.assert_close(ekf.spin[1], spin[0])

    def test_reset_covariance_9d(self):
        cfg = BallEKFConfig(enable_spin=True, p_spin_init=5.0)
        ekf = BallEKF(4, cfg=cfg)
        # Corrupt P
        ekf._P[:] = 999.0
        ids = torch.tensor([0])
        ekf.reset(ids, torch.zeros(1, 3))
        # Check spin block is reset
        spin_P = ekf._P[0, 6:9, 6:9]
        expected = torch.eye(3) * 25.0  # 5^2
        torch.testing.assert_close(spin_P, expected)
        # Check env 1 is still corrupted
        self.assertAlmostEqual(ekf._P[1, 0, 0].item(), 999.0)


class TestMagnusForce(unittest.TestCase):
    """Test Magnus force direction and magnitude."""

    def _make_ekf(self, **kwargs):
        cfg = BallEKFConfig(
            enable_spin=True, drag_coeff=0.0, gravity_z=0.0,
            contact_aware=False, **kwargs,
        )
        return BallEKF(1, cfg=cfg)

    def test_magnus_direction_spin_z_vel_x(self):
        """spin=[0,0,wz] × vel=[vx,0,0] → Magnus in +y direction."""
        ekf = self._make_ekf(magnus_coeff=1.0)
        ekf._x[0, 3] = 1.0   # vx = 1
        ekf._x[0, 8] = 1.0   # wz = 1
        ekf._x[0, 2] = 0.5   # z > contact threshold
        vel_before = ekf.vel.clone()
        ekf.predict(dt=1.0)
        # spin × vel = [0,0,1] × [1,0,0] = [0,1,0]
        # a_magnus = 1.0 * [0, 1, 0]
        dv = ekf.vel[0] - vel_before[0]
        self.assertAlmostEqual(dv[0].item(), 0.0, places=3)
        self.assertGreater(dv[1].item(), 0.5)  # should be ~1.0
        self.assertAlmostEqual(dv[2].item(), 0.0, places=3)

    def test_magnus_direction_spin_x_vel_z(self):
        """spin=[wx,0,0] × vel=[0,0,vz] → Magnus in -y direction."""
        ekf = self._make_ekf(magnus_coeff=1.0)
        ekf._x[0, 5] = 1.0   # vz = 1
        ekf._x[0, 6] = 1.0   # wx = 1
        ekf._x[0, 2] = 0.5   # z > contact threshold
        vel_before = ekf.vel.clone()
        ekf.predict(dt=1.0)
        # spin × vel = [1,0,0] × [0,0,1] = [0,-1,0]
        dv = ekf.vel[0] - vel_before[0]
        self.assertAlmostEqual(dv[0].item(), 0.0, places=3)
        self.assertLess(dv[1].item(), -0.5)  # should be ~-1.0
        self.assertAlmostEqual(dv[2].item(), 0.0, places=3)

    def test_magnus_magnitude_physical(self):
        """Verify Magnus acceleration matches physics for 40mm ball."""
        # Default magnus_coeff ≈ 0.0149
        ekf = self._make_ekf()  # uses default magnus_coeff
        Cm = ekf.cfg.magnus_coeff
        omega = 20.0  # rad/s (moderate spin)
        v = 3.0  # m/s (typical juggle velocity)
        ekf._x[0, 3] = v    # vx
        ekf._x[0, 8] = omega  # wz
        ekf._x[0, 2] = 0.5
        vel_before = ekf.vel.clone()
        dt = 0.001  # small dt for accuracy
        ekf.predict(dt=dt)
        dv = ekf.vel[0] - vel_before[0]
        # Expected: a_magnus = Cm * omega * v = 0.0149 * 20 * 3 = 0.894 m/s²
        a_expected = Cm * omega * v
        a_actual = dv[1].item() / dt  # y-direction acceleration
        self.assertAlmostEqual(a_actual, a_expected, places=1)

    def test_no_magnus_without_spin(self):
        """Zero spin should produce zero Magnus force."""
        ekf = self._make_ekf()
        ekf._x[0, 3] = 3.0   # vx
        ekf._x[0, 2] = 0.5
        vel_before = ekf.vel.clone()
        ekf.predict(dt=0.01)
        dv = ekf.vel[0] - vel_before[0]
        # Only drag (but drag=0 in config), so dv should be ~0
        torch.testing.assert_close(dv, torch.zeros(3), atol=1e-6, rtol=0)

    def test_magnus_perpendicular_to_spin_and_vel(self):
        """Magnus force is always perpendicular to both spin and velocity."""
        ekf = self._make_ekf(magnus_coeff=1.0)
        # Random spin and velocity
        torch.manual_seed(42)
        spin = torch.randn(3)
        vel = torch.randn(3)
        ekf._x[0, 3:6] = vel
        ekf._x[0, 6:9] = spin
        ekf._x[0, 2] = 0.5
        vel_before = ekf.vel.clone()
        dt = 0.0001  # very small dt for linear approximation
        ekf.predict(dt=dt)
        dv = ekf.vel[0] - vel_before[0]
        a = dv / dt  # acceleration
        # a should be perpendicular to both spin and vel
        dot_spin = torch.dot(a, spin).item()
        dot_vel = torch.dot(a, vel).item()
        self.assertAlmostEqual(dot_spin, 0.0, places=2)
        self.assertAlmostEqual(dot_vel, 0.0, places=2)


class TestSpinDecay(unittest.TestCase):
    """Test exponential spin decay."""

    def test_spin_decays(self):
        cfg = BallEKFConfig(
            enable_spin=True, drag_coeff=0.0, gravity_z=0.0,
            contact_aware=False, spin_decay_rate=1.0,  # fast decay for test
        )
        ekf = BallEKF(1, cfg=cfg)
        ekf._x[0, 6:9] = torch.tensor([10.0, 20.0, 30.0])
        ekf._x[0, 2] = 0.5
        dt = 0.1
        ekf.predict(dt=dt)
        expected_factor = math.exp(-1.0 * dt)  # exp(-0.1) ≈ 0.905
        expected = torch.tensor([10.0, 20.0, 30.0]) * expected_factor
        torch.testing.assert_close(ekf.spin[0], expected, atol=1e-5, rtol=1e-5)

    def test_spin_barely_decays_with_real_params(self):
        """Physical spin_decay_rate ≈ 0.008 → almost no decay over 1s."""
        cfg = BallEKFConfig(enable_spin=True, drag_coeff=0.0, gravity_z=0.0,
                            contact_aware=False)
        ekf = BallEKF(1, cfg=cfg)
        ekf._x[0, 6] = 50.0  # 50 rad/s
        ekf._x[0, 2] = 0.5
        # 50 steps at 0.02s = 1 second
        for _ in range(50):
            ekf.predict(dt=0.02)
        # After 1s: spin = 50 * exp(-0.008) ≈ 49.6 rad/s
        expected = 50.0 * math.exp(-0.008 * 1.0)
        self.assertAlmostEqual(ekf.spin[0, 0].item(), expected, places=1)

    def test_spin_preserved_in_free_flight(self):
        """Spin direction should not change during free flight (only magnitude)."""
        cfg = BallEKFConfig(enable_spin=True, drag_coeff=0.0, gravity_z=0.0,
                            contact_aware=False, spin_decay_rate=0.01)
        ekf = BallEKF(1, cfg=cfg)
        spin_init = torch.tensor([1.0, 2.0, 3.0])
        ekf._x[0, 6:9] = spin_init.clone()
        ekf._x[0, 2] = 0.5
        for _ in range(100):
            ekf.predict(dt=0.02)
        # Direction should be preserved
        spin_now = ekf.spin[0]
        direction = spin_now / spin_now.norm()
        expected_dir = spin_init / spin_init.norm()
        torch.testing.assert_close(direction, expected_dir, atol=1e-5, rtol=1e-5)


class TestSpinObservability(unittest.TestCase):
    """Test that spin can be estimated from curved trajectories."""

    def test_topspin_curves_downward(self):
        """Ball with topspin (spin_y>0 moving in +x) should curve downward faster."""
        cfg = BallEKFConfig(
            enable_spin=True, drag_coeff=0.0, gravity_z=-9.81,
            contact_aware=False,
        )
        # Run two simulations: with and without spin
        ekf_spin = BallEKF(1, cfg=cfg)
        ekf_nospin = BallEKF(1, cfg=BallEKFConfig(
            enable_spin=True, drag_coeff=0.0, gravity_z=-9.81,
            contact_aware=False, magnus_coeff=0.0,
        ))

        # Ball launched upward with forward velocity and topspin
        init_pos = torch.tensor([[0.0, 0.0, 0.5]])
        init_vel = torch.tensor([[3.0, 0.0, 2.0]])
        init_spin = torch.tensor([[0.0, 30.0, 0.0]])  # topspin about y-axis
        ekf_spin.reset(torch.tensor([0]), init_pos, init_vel, init_spin)
        ekf_nospin.reset(torch.tensor([0]), init_pos, init_vel, init_spin)

        # Simulate 0.5s (no measurements — prediction only)
        for _ in range(25):
            ekf_spin.predict(dt=0.02)
            ekf_nospin.predict(dt=0.02)

        # Topspin (ω_y with v_x): ω × v = [0,30,0] × [3,0,*] → Z component < 0
        # So ball with topspin should be lower
        z_spin = ekf_spin.pos[0, 2].item()
        z_nospin = ekf_nospin.pos[0, 2].item()
        self.assertLess(z_spin, z_nospin,
                        "Topspin should make ball curve downward faster")

    def test_ekf_estimates_spin_from_curvature(self):
        """EKF should infer spin from position measurements on a curved trajectory."""
        cfg = BallEKFConfig(
            enable_spin=True, drag_coeff=0.0, gravity_z=-9.81,
            contact_aware=False, q_spin=0.5, p_spin_init=20.0,
        )
        ekf = BallEKF(1, cfg=cfg)

        # Ground truth: ball with significant sidespin
        omega_true = torch.tensor([0.0, 0.0, 40.0])  # 40 rad/s sidespin
        Cm = cfg.magnus_coeff
        pos_gt = torch.tensor([0.0, 0.0, 0.5])
        vel_gt = torch.tensor([3.0, 0.0, 2.0])

        # Reset EKF with correct position but NO spin knowledge
        ekf.reset(torch.tensor([0]),
                  pos_gt.unsqueeze(0), vel_gt.unsqueeze(0))
        # Spin should start at 0
        self.assertAlmostEqual(ekf.spin[0].norm().item(), 0.0)

        dt = 0.02
        steps = 100
        detected = torch.ones(1, dtype=torch.bool)

        for _ in range(steps):
            # Propagate ground truth with Magnus
            a_mag = Cm * torch.linalg.cross(omega_true, vel_gt)
            g = torch.tensor([0.0, 0.0, -9.81])
            a = g + a_mag
            pos_gt = pos_gt + vel_gt * dt + 0.5 * a * dt**2
            vel_gt = vel_gt + a * dt
            omega_true = omega_true * math.exp(-cfg.spin_decay_rate * dt)

            # Give perfect position measurements to EKF
            ekf.step(pos_gt.unsqueeze(0), detected, dt=dt)

        # EKF should have estimated some spin in the z-direction
        spin_est = ekf.spin[0]
        # The z-component should be positive (correct sign)
        # With perfect measurements and 100 steps, it should converge somewhat
        self.assertGreater(spin_est[2].item(), 5.0,
                           f"EKF should estimate positive z-spin, got {spin_est}")


class TestSpinContactNoise(unittest.TestCase):
    """Test contact-aware process noise for spin."""

    def test_spin_process_noise_inflated_during_contact(self):
        cfg = BallEKFConfig(
            enable_spin=True, contact_aware=True,
            q_spin=1.0, q_spin_contact=100.0,
            contact_z_threshold=0.025, p_spin_init=0.01,  # small init P
        )
        ekf = BallEKF(2, cfg=cfg)
        # Reset to small covariance so Q dominates
        ekf.reset(torch.tensor([0, 1]), torch.zeros(2, 3))
        # Env 0: ball on paddle (z=0.02 < 0.025 threshold)
        ekf._x[0, 2] = 0.02
        ekf._x[0, 6] = 10.0
        # Env 1: ball in air (z=0.5 > threshold)
        ekf._x[1, 2] = 0.5
        ekf._x[1, 6] = 10.0

        ekf.predict(dt=0.02)

        # Spin covariance should grow much faster for env 0 (contact)
        P_spin_0 = ekf._P[0, 6:9, 6:9].diag()
        P_spin_1 = ekf._P[1, 6:9, 6:9].diag()
        # Contact noise (100) >> free-flight (1), so P growth is ~10000× larger
        self.assertGreater(P_spin_0[0].item(), P_spin_1[0].item() * 10,
                           "Contact spin noise should be much larger than free-flight")


class TestBackwardCompatibility(unittest.TestCase):
    """Verify 6D mode is completely unchanged."""

    def test_6d_vel_property_still_works(self):
        ekf = BallEKF(2, cfg=BallEKFConfig(enable_spin=False))
        ekf._x[0, 3:6] = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(ekf.vel[0], torch.tensor([1.0, 2.0, 3.0]))

    def test_6d_predict_unchanged(self):
        """Same config → same prediction as before."""
        cfg = BallEKFConfig(enable_spin=False, drag_coeff=0.0, contact_aware=False)
        ekf = BallEKF(1, cfg=cfg)
        ekf._x[0] = torch.tensor([0.0, 0.0, 0.5, 1.0, 0.0, 2.0])
        ekf.predict(dt=0.02)
        # pos = [0+1*0.02, 0, 0.5+2*0.02+0.5*(-9.81)*0.0004]
        self.assertAlmostEqual(ekf.pos[0, 0].item(), 0.02, places=4)
        expected_z = 0.5 + 2.0 * 0.02 + 0.5 * (-9.81) * 0.02**2
        self.assertAlmostEqual(ekf.pos[0, 2].item(), expected_z, places=4)

    def test_6d_reset_accepts_no_spin(self):
        ekf = BallEKF(2, cfg=BallEKFConfig(enable_spin=False))
        ekf.reset(torch.tensor([0]), torch.zeros(1, 3))
        # Should not raise

    def test_6d_state_shape(self):
        ekf = BallEKF(2, cfg=BallEKFConfig(enable_spin=False))
        self.assertEqual(ekf.state.shape, (2, 6))


if __name__ == "__main__":
    unittest.main()
