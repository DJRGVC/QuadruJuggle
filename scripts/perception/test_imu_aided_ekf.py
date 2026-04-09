#!/usr/bin/env python3
"""Unit tests for IMU-aided EKF (Coriolis + centrifugal corrections).

Tests that the body-frame EKF correctly compensates for platform rotation
using robot angular velocity. CPU-only, no sim.

Usage:
    python scripts/perception/test_imu_aided_ekf.py
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

from ball_ekf import BallEKF, BallEKFConfig, _batch_skew

DEVICE = "cpu"
DT = 0.02  # 50 Hz policy rate


class TestBatchSkew(unittest.TestCase):
    """Tests for _batch_skew helper."""

    def test_skew_cross_product(self):
        """[v]_x @ u should equal v x u."""
        v = torch.tensor([[1.0, 2.0, 3.0], [0.5, -1.0, 0.3]])
        u = torch.tensor([[0.1, -0.2, 0.4], [1.0, 1.0, 1.0]])
        S = _batch_skew(v)
        result = torch.bmm(S, u.unsqueeze(-1)).squeeze(-1)
        expected = torch.linalg.cross(v, u)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-5)

    def test_skew_antisymmetric(self):
        """Skew matrix should be antisymmetric: S^T = -S."""
        v = torch.randn(8, 3)
        S = _batch_skew(v)
        torch.testing.assert_close(S.transpose(-1, -2), -S, atol=1e-7, rtol=1e-6)

    def test_skew_zero(self):
        """Zero vector → zero skew matrix."""
        v = torch.zeros(3, 3)
        S = _batch_skew(v)
        torch.testing.assert_close(S, torch.zeros(3, 3, 3), atol=1e-7, rtol=1e-6)


class TestIMUAidedPredict(unittest.TestCase):
    """Tests for Coriolis + centrifugal corrections in EKF predict."""

    def _make_ekf(self, num_envs=4, **kwargs):
        cfg = BallEKFConfig(
            q_vel=0.40, contact_aware=False, drag_coeff=0.0, **kwargs
        )
        return BallEKF(num_envs=num_envs, device=DEVICE, cfg=cfg)

    def test_no_omega_unchanged(self):
        """Without angular velocity, predict behaves identically to baseline."""
        ekf_a = self._make_ekf()
        ekf_b = self._make_ekf()
        pos = torch.tensor([[0.0, 0.0, 0.10]] * 4)
        vel = torch.tensor([[0.0, 0.0, 1.0]] * 4)
        ekf_a.reset(torch.arange(4), pos, vel)
        ekf_b.reset(torch.arange(4), pos, vel)

        ekf_a.predict(DT, robot_ang_vel_b=None)
        ekf_b.predict(DT, robot_ang_vel_b=torch.zeros(4, 3))

        torch.testing.assert_close(ekf_a.state, ekf_b.state, atol=1e-6, rtol=1e-5)

    def test_coriolis_deflection(self):
        """Pure Coriolis: ball moving +X, platform rotating +Z → deflection in -Y.

        Coriolis acc = -2 * omega x vel
        omega = (0, 0, omega_z), vel = (vx, 0, 0)
        → a_cor = -2 * (0, 0, omega_z) x (vx, 0, 0) = -2 * (0, vx*omega_z, 0) * (-1 in y)
        Wait: (0,0,w) x (v,0,0) = (0*0 - w*0, w*v - 0*0, 0*0 - 0*v) = (0, wv, 0)
        So a_cor = -2 * (0, wv, 0) → deflection in -Y direction.
        """
        ekf = self._make_ekf()
        pos = torch.tensor([[0.0, 0.0, 0.50]] * 4)  # above contact zone
        vel = torch.tensor([[1.0, 0.0, 0.0]] * 4)   # moving +X
        ekf.reset(torch.arange(4), pos, vel)

        omega = torch.tensor([[0.0, 0.0, 2.0]] * 4)  # rotating +Z at 2 rad/s

        # Record state before
        ekf.predict(DT, robot_ang_vel_b=omega)

        # Coriolis should add -Y velocity component: -2 * 2.0 * 1.0 = -4.0 m/s^2
        # After dt=0.02: delta_vy = -4.0 * 0.02 = -0.08 m/s
        # (plus gravity in Z, but we check Y only)
        vy = ekf.vel[0, 1].item()
        self.assertLess(vy, -0.05,
                        f"Expected negative Y velocity from Coriolis, got vy={vy:.4f}")

    def test_centrifugal_outward(self):
        """Centrifugal: ball at +X offset, platform rotating +Z → outward +X force.

        Centrifugal acc = -omega x (omega x pos)
        omega = (0, 0, w), pos = (r, 0, 0)
        omega x pos = (0, wr, 0)
        omega x (omega x pos) = (0,0,w) x (0,wr,0) = (-w^2*r, 0, 0)
        -omega x (omega x pos) = (w^2*r, 0, 0) → outward force
        """
        ekf = self._make_ekf()
        pos = torch.tensor([[0.10, 0.0, 0.50]] * 4)  # 10cm +X, above contact
        vel = torch.tensor([[0.0, 0.0, 0.0]] * 4)     # stationary
        ekf.reset(torch.arange(4), pos, vel)

        omega = torch.tensor([[0.0, 0.0, 3.0]] * 4)  # 3 rad/s around Z
        ekf.predict(DT, robot_ang_vel_b=omega)

        # Centrifugal: a_x = w^2 * r = 9.0 * 0.10 = 0.9 m/s^2
        # delta_vx = 0.9 * 0.02 = 0.018 m/s
        vx = ekf.vel[0, 0].item()
        self.assertGreater(vx, 0.01,
                           f"Expected positive X velocity from centrifugal, got vx={vx:.4f}")

    def test_coriolis_magnitude(self):
        """Verify Coriolis force magnitude matches analytical formula."""
        ekf = self._make_ekf(num_envs=1)
        pos = torch.tensor([[0.0, 0.0, 0.50]])  # above contact
        vx = 2.0
        vel = torch.tensor([[vx, 0.0, 0.0]])
        ekf.reset(torch.tensor([0]), pos, vel)

        omega_z = 1.5
        omega = torch.tensor([[0.0, 0.0, omega_z]])
        ekf.predict(DT, robot_ang_vel_b=omega)

        # Expected Coriolis: a_y = -2 * omega_z * vx = -2 * 1.5 * 2.0 = -6.0
        # delta_vy = -6.0 * 0.02 = -0.12
        expected_vy = -2.0 * omega_z * vx * DT
        actual_vy = ekf.vel[0, 1].item()
        self.assertAlmostEqual(actual_vy, expected_vy, places=3,
                               msg=f"Coriolis vy: expected {expected_vy:.4f}, got {actual_vy:.4f}")

    def test_centrifugal_magnitude(self):
        """Verify centrifugal force magnitude matches analytical formula."""
        ekf = self._make_ekf(num_envs=1)
        r = 0.08  # 8cm offset
        pos = torch.tensor([[r, 0.0, 0.50]])  # above contact
        vel = torch.tensor([[0.0, 0.0, 0.0]])
        ekf.reset(torch.tensor([0]), pos, vel)

        omega_z = 2.0
        omega = torch.tensor([[0.0, 0.0, omega_z]])
        ekf.predict(DT, robot_ang_vel_b=omega)

        # Expected centrifugal: a_x = omega_z^2 * r = 4.0 * 0.08 = 0.32 m/s^2
        # delta_vx = 0.32 * 0.02 = 0.0064
        expected_vx = omega_z**2 * r * DT
        actual_vx = ekf.vel[0, 0].item()
        self.assertAlmostEqual(actual_vx, expected_vx, places=3,
                               msg=f"Centrifugal vx: expected {expected_vx:.4f}, got {actual_vx:.4f}")

    def test_combined_coriolis_centrifugal(self):
        """Both forces applied simultaneously: verify they superpose correctly."""
        ekf = self._make_ekf(num_envs=1)
        r = 0.05
        pos = torch.tensor([[r, 0.0, 0.50]])
        vx = 1.0
        vel = torch.tensor([[vx, 0.0, 0.0]])
        ekf.reset(torch.tensor([0]), pos, vel)

        omega_z = 2.0
        omega = torch.tensor([[0.0, 0.0, omega_z]])
        ekf.predict(DT, robot_ang_vel_b=omega)

        # Centrifugal X: omega^2 * r = 4 * 0.05 = 0.20
        # Coriolis Y: -2 * omega * vx = -4.0
        expected_dvx = omega_z**2 * r * DT  # 0.004
        expected_dvy = -2.0 * omega_z * vx * DT  # -0.08

        actual_vx = ekf.vel[0, 0].item()
        actual_vy = ekf.vel[0, 1].item()

        self.assertAlmostEqual(actual_vx - vx, expected_dvx, places=3)
        self.assertAlmostEqual(actual_vy, expected_dvy, places=3)

    def test_3d_angular_velocity(self):
        """Non-trivial 3D omega: verify cross-product direction is correct."""
        ekf = self._make_ekf(num_envs=1)
        pos = torch.tensor([[0.05, 0.03, 0.50]])
        vel = torch.tensor([[0.5, -0.3, 0.8]])
        ekf.reset(torch.tensor([0]), pos, vel)

        omega = torch.tensor([[1.0, -0.5, 0.8]])

        # Compute expected accelerations analytically
        a_cor = -2.0 * torch.linalg.cross(omega, vel)
        a_cent = -torch.linalg.cross(omega, torch.linalg.cross(omega, pos))
        a_gravity = torch.tensor([[0.0, 0.0, -9.81]])
        a_total = a_gravity + a_cor + a_cent

        expected_vel = vel + a_total * DT
        expected_pos = pos + vel * DT + 0.5 * a_total * DT**2

        ekf.predict(DT, robot_ang_vel_b=omega)

        torch.testing.assert_close(ekf.vel, expected_vel, atol=1e-4, rtol=1e-3)
        torch.testing.assert_close(ekf.pos, expected_pos, atol=1e-5, rtol=1e-4)

    def test_covariance_grows_with_omega(self):
        """Angular velocity corrections should affect F, changing P growth."""
        ekf_no_omega = self._make_ekf()
        ekf_with_omega = self._make_ekf()
        pos = torch.tensor([[0.05, 0.0, 0.50]] * 4)
        vel = torch.tensor([[1.0, 0.0, 0.0]] * 4)
        ekf_no_omega.reset(torch.arange(4), pos, vel)
        ekf_with_omega.reset(torch.arange(4), pos, vel)

        ekf_no_omega.predict(DT)
        omega = torch.tensor([[0.0, 0.0, 3.0]] * 4)
        ekf_with_omega.predict(DT, robot_ang_vel_b=omega)

        # Covariances should differ due to different F matrices
        P_diff = (ekf_with_omega._P - ekf_no_omega._P).abs().max().item()
        self.assertGreater(P_diff, 1e-6,
                           "Covariance should differ with angular velocity")

    def test_jacobian_pos_vel_coupling(self):
        """With omega, F should have off-diagonal blocks F[3:,:3] (centrifugal Jacobian)."""
        ekf = self._make_ekf(num_envs=1)
        pos = torch.tensor([[0.10, 0.0, 0.50]])
        vel = torch.tensor([[0.0, 0.0, 0.0]])
        ekf.reset(torch.tensor([0]), pos, vel)

        # Save P before predict
        P_before = ekf._P.clone()
        omega = torch.tensor([[0.0, 0.0, 2.0]])
        ekf.predict(DT, robot_ang_vel_b=omega)

        # The centrifugal Jacobian d(a_cent)/d(pos) introduces vel-pos coupling
        # in the F matrix. This manifests as cross-covariance growth in P.
        # Specifically, P[3:, :3] (vel-pos cross-covariance) should be non-zero.
        P_cross = ekf._P[0, 3:, :3]
        cross_norm = P_cross.abs().max().item()
        # Without omega, this cross-term would be zero (only F[:3,3:] is non-zero)
        self.assertGreater(cross_norm, 1e-8,
                           "vel-pos cross-covariance should exist with centrifugal Jacobian")

    def test_multi_step_tracking_with_rotation(self):
        """Run 50 steps with constant omega; EKF should track GT ball trajectory
        generated using the same physics (gravity + Coriolis + centrifugal)."""
        N = 1
        ekf = self._make_ekf(num_envs=N)

        omega = torch.tensor([[0.0, 0.0, 1.5]])  # moderate yaw rate

        # GT simulation: same physics as EKF
        gt_pos = torch.tensor([[0.03, -0.02, 0.30]])
        gt_vel = torch.tensor([[0.0, 0.0, 2.0]])  # launched upward
        ekf.reset(torch.tensor([0]), gt_pos, gt_vel)

        noise_std = 0.002  # 2mm measurement noise

        for _ in range(50):
            # Advance GT with same dynamics
            a_grav = torch.tensor([[0.0, 0.0, -9.81]])
            a_cor = -2.0 * torch.linalg.cross(omega, gt_vel)
            a_cent = -torch.linalg.cross(omega, torch.linalg.cross(omega, gt_pos))
            a = a_grav + a_cor + a_cent
            gt_pos = gt_pos + gt_vel * DT + 0.5 * a * DT**2
            gt_vel = gt_vel + a * DT

            # Noisy measurement
            noise = torch.randn_like(gt_pos) * noise_std
            z = gt_pos + noise
            detected = torch.ones(N, dtype=torch.bool)

            ekf.step(z, detected, DT, robot_ang_vel_b=omega)

        pos_err = (ekf.pos - gt_pos).norm().item()
        vel_err = (ekf.vel - gt_vel).norm().item()
        self.assertLess(pos_err, 0.010, f"Position RMSE {pos_err:.4f}m too large")
        self.assertLess(vel_err, 0.50, f"Velocity RMSE {vel_err:.4f} m/s too large")

    def test_tracking_degrades_without_imu(self):
        """Without omega compensation, tracking under rotation should be worse."""
        N = 1
        omega = torch.tensor([[0.0, 0.0, 4.0]])  # fast yaw rotation

        def run_tracking(use_imu: bool):
            cfg = BallEKFConfig(q_vel=0.40, contact_aware=False, drag_coeff=0.0)
            ekf = BallEKF(num_envs=N, device=DEVICE, cfg=cfg)
            gt_pos = torch.tensor([[0.03, 0.0, 0.50]])
            gt_vel = torch.tensor([[0.0, 0.0, 1.5]])
            ekf.reset(torch.tensor([0]), gt_pos, gt_vel)

            for _ in range(60):
                a_grav = torch.tensor([[0.0, 0.0, -9.81]])
                a_cor = -2.0 * torch.linalg.cross(omega, gt_vel)
                a_cent = -torch.linalg.cross(omega, torch.linalg.cross(omega, gt_pos))
                a = a_grav + a_cor + a_cent
                gt_pos_new = gt_pos + gt_vel * DT + 0.5 * a * DT**2
                gt_vel_new = gt_vel + a * DT
                gt_pos, gt_vel = gt_pos_new, gt_vel_new

                z = gt_pos + torch.randn_like(gt_pos) * 0.002
                detected = torch.ones(N, dtype=torch.bool)

                if use_imu:
                    ekf.step(z, detected, DT, robot_ang_vel_b=omega)
                else:
                    ekf.step(z, detected, DT)

            return (ekf.pos - gt_pos).norm().item()

        torch.manual_seed(42)
        err_with_imu = run_tracking(True)
        torch.manual_seed(42)
        err_without_imu = run_tracking(False)

        self.assertLess(err_with_imu, err_without_imu,
                        f"IMU-aided ({err_with_imu:.4f}) should be better than no-IMU ({err_without_imu:.4f})")

    def test_step_passthrough(self):
        """step() correctly passes robot_ang_vel_b to predict()."""
        ekf = self._make_ekf(num_envs=1)
        pos = torch.tensor([[0.05, 0.0, 0.50]])
        vel = torch.tensor([[1.0, 0.0, 0.0]])
        ekf.reset(torch.tensor([0]), pos, vel)

        z = torch.tensor([[0.05, 0.0, 0.50]])
        detected = torch.ones(1, dtype=torch.bool)
        omega = torch.tensor([[0.0, 0.0, 2.0]])

        # Should not raise
        ekf.step(z, detected, DT, robot_ang_vel_b=omega)

        # Verify Coriolis effect is present (vy should be non-zero)
        vy = ekf.vel[0, 1].item()
        self.assertNotAlmostEqual(vy, 0.0, places=3,
                                  msg="step() should pass omega to predict()")

    def test_per_env_omega(self):
        """Different envs can have different angular velocities."""
        N = 4
        ekf = self._make_ekf(num_envs=N)
        pos = torch.tensor([[0.05, 0.0, 0.50]] * N)
        vel = torch.tensor([[1.0, 0.0, 0.0]] * N)
        ekf.reset(torch.arange(N), pos, vel)

        omega = torch.zeros(N, 3)
        omega[0, 2] = 1.0   # env 0: slow rotation
        omega[1, 2] = 3.0   # env 1: fast rotation
        # env 2, 3: no rotation

        ekf.predict(DT, robot_ang_vel_b=omega)

        # Coriolis effect should scale with omega_z
        vy_0 = ekf.vel[0, 1].item()
        vy_1 = ekf.vel[1, 1].item()
        vy_2 = ekf.vel[2, 1].item()

        # env 1 should have 3x the Coriolis deflection of env 0
        self.assertAlmostEqual(vy_1 / vy_0, 3.0, places=1,
                               msg="Coriolis should scale linearly with omega")
        self.assertAlmostEqual(vy_2, 0.0, places=5,
                               msg="No rotation → no Coriolis")


if __name__ == "__main__":
    unittest.main()
