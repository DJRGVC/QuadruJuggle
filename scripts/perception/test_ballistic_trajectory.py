"""Ballistic trajectory tests for EKF + mock pipeline.

Simulates realistic juggling parabolic arcs (launch → apex → descent → contact)
and validates that the EKF tracks position and velocity accurately throughout
all flight phases. Tests contact-aware mode switching at paddle surface.

Run: python scripts/perception/test_ballistic_trajectory.py
"""

from __future__ import annotations

import math
import os
import sys
import unittest

import numpy as np
import torch

# Direct imports — bypass go1_ball_balance.__init__ which pulls Isaac Lab / pxr
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


_camera_mod = _load_module(
    "perception.real.camera",
    os.path.join(_PERC_DIR, "real", "camera.py"),
)
CameraIntrinsics = _camera_mod.CameraIntrinsics

_calib_mod = _load_module(
    "perception.real.calibration",
    os.path.join(_PERC_DIR, "real", "calibration.py"),
)
CameraExtrinsics = _calib_mod.CameraExtrinsics

_detector_mod = _load_module(
    "perception.real.detector",
    os.path.join(_PERC_DIR, "real", "detector.py"),
)

_mock_mod = _load_module(
    "perception.real.mock",
    os.path.join(_PERC_DIR, "real", "mock.py"),
)
MockCamera = _mock_mod.MockCamera
MockCameraConfig = _mock_mod.MockCameraConfig
MockDetector = _mock_mod.MockDetector

_ekf_mod = _load_module(
    "perception.ball_ekf",
    os.path.join(_PERC_DIR, "ball_ekf.py"),
)
BallEKF = _ekf_mod.BallEKF
BallEKFConfig = _ekf_mod.BallEKFConfig


# --- Trajectory generators ---

GRAVITY = -9.81  # m/s²
DRAG_COEFF = 0.112  # matches BallEKFConfig default


def ballistic_trajectory(
    pos0: np.ndarray,
    vel0: np.ndarray,
    dt: float,
    n_steps: int,
    drag: float = DRAG_COEFF,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ground-truth ballistic trajectory with drag.

    Returns:
        positions: (n_steps, 3)
        velocities: (n_steps, 3)
    """
    positions = np.zeros((n_steps, 3))
    velocities = np.zeros((n_steps, 3))
    pos = pos0.copy()
    vel = vel0.copy()

    for i in range(n_steps):
        positions[i] = pos
        velocities[i] = vel
        # Acceleration: gravity + quadratic drag
        speed = np.linalg.norm(vel)
        a_grav = np.array([0.0, 0.0, GRAVITY])
        a_drag = -drag * speed * vel if speed > 1e-8 else np.zeros(3)
        a = a_grav + a_drag
        # Euler integration (matches EKF predict)
        pos = pos + vel * dt + 0.5 * a * dt ** 2
        vel = vel + a * dt

    return positions, velocities


class TestBallisticFreeFlightEKF(unittest.TestCase):
    """EKF tracks a full parabolic arc (free flight only, no contact)."""

    def _run_arc(
        self,
        vz0: float,
        pos_noise_std: float = 0.002,
        dropout_rate: float = 0.0,
        contact_aware: bool = True,
    ) -> dict:
        """Launch ball upward from paddle and track full arc.

        Args:
            vz0: Initial upward velocity (m/s).
            pos_noise_std: Measurement noise std (m).
            dropout_rate: Fraction of missed detections.
            contact_aware: Enable contact-aware EKF mode.

        Returns dict with pos_rmse, vel_rmse, apex_z_err, apex_vz_err.
        """
        dt = 0.02  # 50 Hz policy rate
        # Ball starts just above paddle surface
        pos0 = np.array([0.0, 0.0, 0.025])
        vel0 = np.array([0.0, 0.0, vz0])

        # Compute flight time: ball returns to z=0.025 (symmetric minus drag)
        # Rough: t_flight ≈ 2 * vz0 / |g|
        t_flight = 2.0 * vz0 / abs(GRAVITY) * 1.1  # 10% margin for drag
        n_steps = int(t_flight / dt) + 5

        gt_pos, gt_vel = ballistic_trajectory(pos0, vel0, dt, n_steps)

        # Only track while ball is above paddle (z > 0.02)
        valid = gt_pos[:, 2] > 0.02
        if not valid.any():
            self.fail("Ball never above paddle")

        # Setup mock pipeline
        cam = MockCamera(MockCameraConfig(depth_noise_std_mm=0.0))
        cam.start()
        det = MockDetector(pos_noise_std=pos_noise_std, dropout_rate=dropout_rate)
        extr = CameraExtrinsics(R_cam_body=np.eye(3), t_cam_body=np.zeros(3))

        ekf = BallEKF(
            num_envs=1,
            device="cpu",
            cfg=BallEKFConfig(contact_aware=contact_aware, nis_gate_enabled=False),
        )
        ekf.reset(
            torch.tensor([0]),
            torch.tensor([pos0], dtype=torch.float32),
            torch.tensor([vel0], dtype=torch.float32),
        )

        pos_errors = []
        vel_errors = []
        apex_z_err = None
        apex_vz_err = None
        apex_idx = np.argmax(gt_pos[:, 2])

        for i in range(n_steps):
            if not valid[i]:
                continue

            # Set ball position in camera (identity extrinsics → same as body)
            cam.set_ball_pos_cam(gt_pos[i])
            depth, ts = cam.get_frame()
            detection = det.detect(depth, cam.get_intrinsics())

            if detection is not None:
                pos_body = extr.transform_to_body(detection.pos_cam)
                z = torch.tensor(pos_body, dtype=torch.float32).unsqueeze(0)
                ekf.step(z, detected=torch.tensor([True]), dt=dt)
            else:
                ekf.step(torch.zeros(1, 3), detected=torch.tensor([False]), dt=dt)

            # Record errors (skip first 3 steps for EKF warmup)
            if i >= 3:
                p_err = np.linalg.norm(ekf.pos[0].numpy() - gt_pos[i])
                v_err = np.linalg.norm(ekf.vel[0].numpy() - gt_vel[i])
                pos_errors.append(p_err)
                vel_errors.append(v_err)

            if i == apex_idx:
                apex_z_err = abs(ekf.pos[0, 2].item() - gt_pos[i, 2])
                apex_vz_err = abs(ekf.vel[0, 2].item() - gt_vel[i, 2])

        return {
            "pos_rmse": float(np.sqrt(np.mean(np.array(pos_errors) ** 2))),
            "vel_rmse": float(np.sqrt(np.mean(np.array(vel_errors) ** 2))),
            "pos_max": float(np.max(pos_errors)) if pos_errors else 999.0,
            "apex_z_err": apex_z_err or 999.0,
            "apex_vz_err": apex_vz_err or 999.0,
            "n_valid_steps": len(pos_errors),
            "mean_nis": ekf.mean_nis,
        }

    def test_low_arc_stage_a(self):
        """Stage A: 10cm target apex → vz0 ≈ 1.4 m/s."""
        vz0 = math.sqrt(2 * 9.81 * 0.10)  # ≈ 1.4 m/s
        r = self._run_arc(vz0)
        # Low arcs spend most time near contact threshold → looser bound
        self.assertLess(r["pos_rmse"], 0.020, f"pos RMSE {r['pos_rmse']:.4f}m > 20mm")
        self.assertLess(r["vel_rmse"], 0.5, f"vel RMSE {r['vel_rmse']:.3f}m/s > 0.5")
        self.assertGreater(r["n_valid_steps"], 5, "Too few valid steps")

    def test_medium_arc_stage_d(self):
        """Stage D: 45cm target apex → vz0 ≈ 2.97 m/s."""
        vz0 = math.sqrt(2 * 9.81 * 0.45)  # ≈ 2.97 m/s
        r = self._run_arc(vz0)
        self.assertLess(r["pos_rmse"], 0.015, f"pos RMSE {r['pos_rmse']:.4f}m > 15mm")
        self.assertLess(r["vel_rmse"], 0.5, f"vel RMSE {r['vel_rmse']:.3f}m/s > 0.5")
        self.assertLess(r["apex_z_err"], 0.020, f"apex Z err {r['apex_z_err']:.4f}m > 20mm")

    def test_high_arc_stage_g(self):
        """Stage G: 1.0m target apex → vz0 ≈ 4.43 m/s."""
        vz0 = math.sqrt(2 * 9.81 * 1.0)  # ≈ 4.43 m/s
        r = self._run_arc(vz0)
        self.assertLess(r["pos_rmse"], 0.020, f"pos RMSE {r['pos_rmse']:.4f}m > 20mm")
        self.assertLess(r["vel_rmse"], 0.8, f"vel RMSE {r['vel_rmse']:.3f}m/s > 0.8")
        self.assertLess(r["apex_z_err"], 0.030, f"apex Z err {r['apex_z_err']:.4f}m > 30mm")

    def test_apex_velocity_near_zero(self):
        """At apex, vz should be approximately zero."""
        vz0 = math.sqrt(2 * 9.81 * 0.45)
        r = self._run_arc(vz0)
        self.assertLess(r["apex_vz_err"], 0.5, f"apex vz err {r['apex_vz_err']:.3f}m/s > 0.5")

    def test_noisy_measurements(self):
        """Realistic D435i noise (5mm std) — EKF should still track."""
        vz0 = math.sqrt(2 * 9.81 * 0.45)
        r = self._run_arc(vz0, pos_noise_std=0.005)
        self.assertLess(r["pos_rmse"], 0.025, f"pos RMSE {r['pos_rmse']:.4f}m > 25mm (noisy)")
        self.assertLess(r["vel_rmse"], 1.0, f"vel RMSE {r['vel_rmse']:.3f}m/s > 1.0 (noisy)")

    def test_with_dropout(self):
        """20% dropout during flight — EKF bridges gaps via prediction."""
        vz0 = math.sqrt(2 * 9.81 * 0.45)
        r = self._run_arc(vz0, dropout_rate=0.2)
        # Allow slightly worse performance with dropout
        self.assertLess(r["pos_rmse"], 0.030, f"pos RMSE {r['pos_rmse']:.4f}m > 30mm (dropout)")
        self.assertLess(r["vel_rmse"], 1.2, f"vel RMSE {r['vel_rmse']:.3f}m/s > 1.2 (dropout)")

    def test_contact_aware_vs_uniform(self):
        """Contact-aware mode should not degrade free-flight tracking vs uniform."""
        vz0 = math.sqrt(2 * 9.81 * 0.45)
        r_ca = self._run_arc(vz0, contact_aware=True)
        r_uni = self._run_arc(vz0, contact_aware=False)
        # Contact-aware should be at least as good (free flight uses same q_vel=0.40)
        # Allow 50% tolerance since trajectory randomness from mock detector noise
        self.assertLess(
            r_ca["pos_rmse"], r_uni["pos_rmse"] * 1.5 + 0.005,
            f"Contact-aware ({r_ca['pos_rmse']:.4f}) much worse than uniform ({r_uni['pos_rmse']:.4f})"
        )


class TestBallisticWithContact(unittest.TestCase):
    """EKF handles the contact→flight→contact transition (full bounce cycle)."""

    def test_single_bounce_cycle(self):
        """Ball rests on paddle, launches up, falls back. Track the whole thing."""
        dt = 0.02
        vz_launch = math.sqrt(2 * 9.81 * 0.30)  # 30cm apex
        n_steps = 80  # ~1.6s covers full cycle

        # Phase 1: ball on paddle (contact) — 10 steps
        # Phase 2: launch + flight — ~50 steps
        # Phase 3: ball back near paddle — remaining

        ekf = BallEKF(num_envs=1, device="cpu", cfg=BallEKFConfig(contact_aware=True, nis_gate_enabled=False))
        pos0 = np.array([0.0, 0.0, 0.020])  # resting on paddle
        ekf.reset(
            torch.tensor([0]),
            torch.tensor([pos0], dtype=torch.float32),
            torch.zeros(1, 3),
        )

        # Build ground truth: contact → ballistic
        contact_steps = 10
        flight_pos0 = np.array([0.0, 0.0, 0.025])
        flight_vel0 = np.array([0.0, 0.0, vz_launch])
        flight_steps = n_steps - contact_steps
        flight_gt_pos, flight_gt_vel = ballistic_trajectory(
            flight_pos0, flight_vel0, dt, flight_steps
        )

        pos_errors_contact = []
        pos_errors_flight = []

        for i in range(n_steps):
            if i < contact_steps:
                # Ball resting on paddle (z ≈ 0.020)
                true_pos = np.array([0.0, 0.0, 0.020])
                true_vel = np.zeros(3)
            else:
                fi = i - contact_steps
                if fi >= flight_steps:
                    break
                true_pos = flight_gt_pos[fi]
                true_vel = flight_gt_vel[fi]
                # Stop if ball below paddle
                if true_pos[2] < 0.0:
                    break

            # Noisy measurement
            meas = true_pos + np.random.normal(0, 0.002, 3)
            z = torch.tensor(meas, dtype=torch.float32).unsqueeze(0)
            ekf.step(z, detected=torch.tensor([True]), dt=dt)

            err = np.linalg.norm(ekf.pos[0].numpy() - true_pos)
            if i < contact_steps:
                pos_errors_contact.append(err)
            elif i >= contact_steps + 3:  # skip 3 warmup after launch
                pos_errors_flight.append(err)

        contact_rmse = float(np.sqrt(np.mean(np.array(pos_errors_contact) ** 2)))
        flight_rmse = float(np.sqrt(np.mean(np.array(pos_errors_flight) ** 2)))

        self.assertLess(contact_rmse, 0.010,
                        f"Contact RMSE {contact_rmse:.4f}m > 10mm")
        self.assertLess(flight_rmse, 0.020,
                        f"Flight RMSE {flight_rmse:.4f}m > 20mm")

    def test_contact_phase_high_q_vel(self):
        """During contact (z < 25mm), q_vel should inflate so EKF trusts measurements."""
        ekf = BallEKF(num_envs=1, device="cpu", cfg=BallEKFConfig(contact_aware=True, nis_gate_enabled=False))
        pos0 = torch.tensor([[0.0, 0.0, 0.020]])  # on paddle
        ekf.reset(torch.tensor([0]), pos0, torch.zeros(1, 3))

        # Record P before and after predict
        P_before = ekf._P.clone()
        ekf.predict(0.02)
        P_after = ekf._P.clone()

        # During contact, P velocity block should grow substantially
        # (q_vel_contact=50.0 vs q_vel=0.40)
        vel_p_growth = (P_after[0, 3:, 3:] - P_before[0, 3:, 3:]).diagonal()
        # q_vel_contact * dt = 50.0 * 0.02 = 1.0 → Q_vel = 1.0^2 = 1.0
        self.assertTrue(
            (vel_p_growth > 0.5).all(),
            f"Velocity P growth too small during contact: {vel_p_growth}"
        )

    def test_flight_phase_low_q_vel(self):
        """During flight (z > 25mm), q_vel is low for good smoothing."""
        ekf = BallEKF(num_envs=1, device="cpu", cfg=BallEKFConfig(contact_aware=True, nis_gate_enabled=False))
        pos0 = torch.tensor([[0.0, 0.0, 0.50]])  # high up
        ekf.reset(torch.tensor([0]), pos0, torch.zeros(1, 3))

        P_before = ekf._P.clone()
        ekf.predict(0.02)
        P_after = ekf._P.clone()

        # During flight, P velocity block should grow very little
        # q_vel * dt = 0.40 * 0.02 = 0.008 → Q_vel = 6.4e-5
        vel_p_growth = (P_after[0, 3:, 3:] - P_before[0, 3:, 3:]).diagonal()
        self.assertTrue(
            (vel_p_growth < 0.01).all(),
            f"Velocity P growth too large during flight: {vel_p_growth}"
        )


class TestMultipleBounces(unittest.TestCase):
    """EKF tracks multiple sequential bounces (realistic juggling)."""

    def test_three_bounces(self):
        """Simulate 3 consecutive bounces with decreasing height."""
        dt = 0.02
        ekf = BallEKF(num_envs=1, device="cpu", cfg=BallEKFConfig(contact_aware=True, nis_gate_enabled=False))

        # Bounce heights: 30cm, 25cm, 20cm (losing energy each bounce)
        heights = [0.30, 0.25, 0.20]
        pos0 = np.array([0.0, 0.0, 0.025])
        all_pos_errors = []

        ekf.reset(
            torch.tensor([0]),
            torch.tensor([pos0], dtype=torch.float32),
            torch.zeros(1, 3),
        )

        for bounce_idx, h in enumerate(heights):
            vz0 = math.sqrt(2 * 9.81 * h)
            t_flight = 2.0 * vz0 / 9.81 * 1.05
            n_steps = int(t_flight / dt)

            gt_pos, gt_vel = ballistic_trajectory(
                pos0, np.array([0.0, 0.0, vz0]), dt, n_steps
            )

            for i in range(n_steps):
                if gt_pos[i, 2] < 0.0:
                    break

                meas = gt_pos[i] + np.random.normal(0, 0.003, 3)
                z = torch.tensor(meas, dtype=torch.float32).unsqueeze(0)
                ekf.step(z, detected=torch.tensor([True]), dt=dt)

                if i >= 3:  # skip warmup
                    err = np.linalg.norm(ekf.pos[0].numpy() - gt_pos[i])
                    all_pos_errors.append(err)

        overall_rmse = float(np.sqrt(np.mean(np.array(all_pos_errors) ** 2)))
        self.assertLess(overall_rmse, 0.020,
                        f"Multi-bounce RMSE {overall_rmse:.4f}m > 20mm")

    def test_nis_stays_bounded(self):
        """NIS should stay in reasonable range across multiple bounces."""
        dt = 0.02
        ekf = BallEKF(num_envs=1, device="cpu", cfg=BallEKFConfig(contact_aware=True, nis_gate_enabled=False))

        pos0 = np.array([0.0, 0.0, 0.025])
        ekf.reset(
            torch.tensor([0]),
            torch.tensor([pos0], dtype=torch.float32),
            torch.zeros(1, 3),
        )

        # Run two bounces
        for h in [0.30, 0.30]:
            vz0 = math.sqrt(2 * 9.81 * h)
            gt_pos, _ = ballistic_trajectory(
                pos0, np.array([0.0, 0.0, vz0]), dt, 60
            )
            for i in range(60):
                if gt_pos[i, 2] < 0.0:
                    break
                meas = gt_pos[i] + np.random.normal(0, 0.003, 3)
                z = torch.tensor(meas, dtype=torch.float32).unsqueeze(0)
                ekf.step(z, detected=torch.tensor([True]), dt=dt)

        nis = ekf.mean_nis
        # With contact-aware mode, overall NIS should be reasonable
        # (contact phases have high NIS but free-flight phases pull it down)
        self.assertLess(nis, 50.0, f"NIS {nis:.1f} too high — EKF divergent")
        self.assertGreater(nis, 0.01, f"NIS {nis:.4f} too low — over-conservative")


class TestOffAxisLaunch(unittest.TestCase):
    """Ball launched with lateral velocity (realistic — ball rarely goes straight up)."""

    def test_diagonal_launch(self):
        """Ball launched at angle: vx=0.3, vy=0.1, vz=2.5 m/s."""
        dt = 0.02
        pos0 = np.array([0.0, 0.0, 0.025])
        vel0 = np.array([0.3, 0.1, 2.5])

        t_flight = 2.0 * 2.5 / 9.81 * 1.1
        n_steps = int(t_flight / dt)
        gt_pos, gt_vel = ballistic_trajectory(pos0, vel0, dt, n_steps)

        ekf = BallEKF(num_envs=1, device="cpu", cfg=BallEKFConfig(contact_aware=True, nis_gate_enabled=False))
        ekf.reset(
            torch.tensor([0]),
            torch.tensor([pos0], dtype=torch.float32),
            torch.tensor([vel0], dtype=torch.float32),
        )

        cam = MockCamera(MockCameraConfig(depth_noise_std_mm=0.0))
        cam.start()
        det = MockDetector(pos_noise_std=0.003)
        extr = CameraExtrinsics(R_cam_body=np.eye(3), t_cam_body=np.zeros(3))

        pos_errors = []
        for i in range(n_steps):
            if gt_pos[i, 2] < 0.0:
                break

            cam.set_ball_pos_cam(gt_pos[i])
            depth, _ = cam.get_frame()
            detection = det.detect(depth, cam.get_intrinsics())

            if detection is not None:
                pos_body = extr.transform_to_body(detection.pos_cam)
                z = torch.tensor(pos_body, dtype=torch.float32).unsqueeze(0)
                ekf.step(z, detected=torch.tensor([True]), dt=dt)
            else:
                ekf.step(torch.zeros(1, 3), detected=torch.tensor([False]), dt=dt)

            if i >= 3:
                pos_errors.append(np.linalg.norm(ekf.pos[0].numpy() - gt_pos[i]))

        rmse = float(np.sqrt(np.mean(np.array(pos_errors) ** 2)))
        self.assertLess(rmse, 0.020, f"Diagonal launch RMSE {rmse:.4f}m > 20mm")

        # Check lateral velocity tracking
        est_vx = ekf.vel[0, 0].item()
        est_vy = ekf.vel[0, 1].item()
        # At the end, true vx ≈ 0.3 (drag slows slightly), vy ≈ 0.1
        self.assertGreater(est_vx, 0.1, f"vx={est_vx:.2f} too low")
        self.assertLess(est_vx, 0.5, f"vx={est_vx:.2f} too high")


if __name__ == "__main__":
    # Set seed for reproducibility of noise
    np.random.seed(42)
    torch.manual_seed(42)
    unittest.main(verbosity=2)
