"""Latency injection tests for D435i noise model + EKF pipeline.

Verifies that observation delays (1-3 policy steps) are correctly modelled
and that EKF tracking degrades gracefully with increasing latency.

Key scenarios:
- Latency buffer delays observations by exactly N steps
- EKF position RMSE increases monotonically with latency
- Velocity estimation still usable at 3-frame delay
- Combined latency + dropout doesn't diverge
- Reset clears latency buffer per-environment

Run: python scripts/perception/test_latency_injection.py
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


_noise_mod = _load_module(
    "perception.noise_model",
    os.path.join(_PERC_DIR, "noise_model.py"),
)
D435iNoiseModel = _noise_mod.D435iNoiseModel
D435iNoiseModelCfg = _noise_mod.D435iNoiseModelCfg

_ekf_mod = _load_module(
    "perception.ball_ekf",
    os.path.join(_PERC_DIR, "ball_ekf.py"),
)
BallEKF = _ekf_mod.BallEKF
BallEKFConfig = _ekf_mod.BallEKFConfig


# --- Trajectory generator (from test_ballistic_trajectory.py) ---

GRAVITY = -9.81
DRAG_COEFF = 0.112


def ballistic_trajectory(
    pos0: np.ndarray,
    vel0: np.ndarray,
    dt: float,
    n_steps: int,
    drag: float = DRAG_COEFF,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate ground-truth ballistic trajectory with drag."""
    positions = np.zeros((n_steps, 3))
    velocities = np.zeros((n_steps, 3))
    pos = pos0.copy()
    vel = vel0.copy()
    for i in range(n_steps):
        positions[i] = pos
        velocities[i] = vel
        speed = np.linalg.norm(vel)
        acc = np.array([0.0, 0.0, GRAVITY])
        if speed > 1e-6:
            acc -= drag * speed * vel
        vel = vel + acc * dt
        pos = pos + vel * dt
    return positions, velocities


class TestLatencyBuffer(unittest.TestCase):
    """Test the D435iNoiseModel latency buffer in isolation."""

    def test_zero_latency_passthrough(self):
        """latency_steps=0 should return current-step observation."""
        cfg = D435iNoiseModelCfg(
            sigma_xy_base=0.0, sigma_z_base=0.0, sigma_z_per_metre=0.0,
            dropout_prob=0.0, latency_steps=0,
        )
        noise = D435iNoiseModel(num_envs=4, cfg=cfg)
        gt = torch.tensor([[0.01, 0.02, 0.05]] * 4)
        out, det = noise.sample(gt)
        torch.testing.assert_close(out, gt, atol=1e-6, rtol=0)
        self.assertTrue(det.all())

    def test_one_step_delay(self):
        """latency_steps=1: output at step T is the measurement from step T-1."""
        cfg = D435iNoiseModelCfg(
            sigma_xy_base=0.0, sigma_z_base=0.0, sigma_z_per_metre=0.0,
            dropout_prob=0.0, latency_steps=1,
        )
        noise = D435iNoiseModel(num_envs=1, cfg=cfg)

        pos_a = torch.tensor([[0.10, 0.00, 0.05]])
        pos_b = torch.tensor([[0.20, 0.00, 0.05]])

        # Step 0: feed pos_a, get the pre-fill zeros (or init value)
        out0, _ = noise.sample(pos_a)
        # Step 1: feed pos_b, should get pos_a (delayed by 1)
        out1, _ = noise.sample(pos_b)
        torch.testing.assert_close(out1, pos_a, atol=1e-6, rtol=0)

    def test_two_step_delay(self):
        """latency_steps=2: output at step T is from step T-2."""
        cfg = D435iNoiseModelCfg(
            sigma_xy_base=0.0, sigma_z_base=0.0, sigma_z_per_metre=0.0,
            dropout_prob=0.0, latency_steps=2,
        )
        noise = D435iNoiseModel(num_envs=1, cfg=cfg)

        pos_a = torch.tensor([[0.10, 0.00, 0.05]])
        pos_b = torch.tensor([[0.20, 0.00, 0.05]])
        pos_c = torch.tensor([[0.30, 0.00, 0.05]])

        out0, _ = noise.sample(pos_a)  # returns pre-fill
        out1, _ = noise.sample(pos_b)  # returns pre-fill
        out2, _ = noise.sample(pos_c)  # returns pos_a (2 steps ago)
        torch.testing.assert_close(out2, pos_a, atol=1e-6, rtol=0)

    def test_three_step_delay(self):
        """latency_steps=3: output at step T is from step T-3."""
        cfg = D435iNoiseModelCfg(
            sigma_xy_base=0.0, sigma_z_base=0.0, sigma_z_per_metre=0.0,
            dropout_prob=0.0, latency_steps=3,
        )
        noise = D435iNoiseModel(num_envs=1, cfg=cfg)

        positions = [
            torch.tensor([[float(i) * 0.1, 0.0, 0.05]]) for i in range(6)
        ]
        outputs = []
        for p in positions:
            out, _ = noise.sample(p)
            outputs.append(out.clone())

        # Step 3 should return step 0's value, step 4 → step 1, step 5 → step 2
        torch.testing.assert_close(outputs[3], positions[0], atol=1e-6, rtol=0)
        torch.testing.assert_close(outputs[4], positions[1], atol=1e-6, rtol=0)
        torch.testing.assert_close(outputs[5], positions[2], atol=1e-6, rtol=0)

    def test_reset_clears_buffer(self):
        """After reset, latency buffer should contain the init position."""
        cfg = D435iNoiseModelCfg(
            sigma_xy_base=0.0, sigma_z_base=0.0, sigma_z_per_metre=0.0,
            dropout_prob=0.0, latency_steps=2,
        )
        noise = D435iNoiseModel(num_envs=2, cfg=cfg)

        # Feed some values
        for _ in range(5):
            noise.sample(torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]))

        # Reset env 0 only
        init = torch.tensor([[0.0, 0.0, 0.02]])
        noise.reset(torch.tensor([0]), init)

        # After reset, env 0's buffer entries should be init_pos
        out, _ = noise.sample(torch.tensor([[0.9, 0.9, 0.9], [0.9, 0.9, 0.9]]))
        # env 0: delayed by 2, buffer was reset to init → should get init or close
        self.assertAlmostEqual(out[0, 0].item(), 0.0, places=2)
        # env 1: was not reset, should still have old value (0.5)
        self.assertAlmostEqual(out[1, 0].item(), 0.5, places=2)

    def test_latency_with_dropout(self):
        """Latency + dropout: delayed dropout events propagate correctly."""
        cfg = D435iNoiseModelCfg(
            sigma_xy_base=0.0, sigma_z_base=0.0, sigma_z_per_metre=0.0,
            dropout_prob=1.0,  # 100% dropout — always hold last
            latency_steps=1,
        )
        noise = D435iNoiseModel(num_envs=1, cfg=cfg)

        pos_a = torch.tensor([[0.10, 0.00, 0.05]])
        pos_b = torch.tensor([[0.20, 0.00, 0.05]])

        _, det0 = noise.sample(pos_a)
        _, det1 = noise.sample(pos_b)

        # With 100% dropout, detected should be False
        self.assertFalse(det1.any().item())


class TestLatencyEKFDegradation(unittest.TestCase):
    """Test EKF tracking quality under increasing observation latency."""

    def _run_trajectory_with_latency(
        self,
        latency_steps: int,
        n_steps: int = 28,
        dt: float = 0.02,
        dropout_prob: float = 0.0,
        noise_std: float = 0.002,
    ) -> dict:
        """Run EKF on a ballistic trajectory with given latency.

        Uses a free-flight arc that stays above contact_z_threshold (0.025m)
        for the entire test duration, avoiding contact-mode Q inflation.

        Returns dict with pos_rmse, vel_rmse, max_pos_error.
        """
        # Generate trajectory: Stage D-like arc, start at z=0.10 (above contact zone)
        # Apex at ~step 15 (~z=0.56), returns to z=0.10 at ~step 30
        # Run 28 steps to stay safely in free flight
        pos0 = np.array([0.0, 0.0, 0.10])
        vel0 = np.array([0.05, -0.02, 3.0])  # ~46cm above start → apex ~0.56m
        gt_pos, gt_vel = ballistic_trajectory(pos0, vel0, dt, n_steps)

        # Noise model with specified latency
        noise_cfg = D435iNoiseModelCfg(
            sigma_xy_base=noise_std,
            sigma_z_base=noise_std,
            sigma_z_per_metre=0.0,
            dropout_prob=dropout_prob,
            latency_steps=latency_steps,
        )
        noise = D435iNoiseModel(num_envs=1, cfg=noise_cfg)

        # EKF
        ekf_cfg = BallEKFConfig(contact_aware=True)
        ekf = BallEKF(num_envs=1, cfg=ekf_cfg)
        ekf.reset(
            torch.tensor([0]),
            torch.tensor([pos0], dtype=torch.float32),
            torch.tensor([vel0], dtype=torch.float32),
        )

        pos_errors = []
        vel_errors = []

        # Skip first latency_steps (warmup) for fair comparison
        for i in range(n_steps):
            gt = torch.tensor([gt_pos[i]], dtype=torch.float32)
            noisy, detected = noise.sample(gt)
            ekf.step(noisy, detected, dt=dt)

            if i >= latency_steps + 5:  # skip warmup
                pe = (ekf.pos[0] - torch.tensor(gt_pos[i])).norm().item()
                ve = (ekf.vel[0] - torch.tensor(gt_vel[i])).norm().item()
                pos_errors.append(pe)
                vel_errors.append(ve)

        pos_errors = np.array(pos_errors)
        vel_errors = np.array(vel_errors)

        return {
            "pos_rmse": float(np.sqrt(np.mean(pos_errors**2))),
            "vel_rmse": float(np.sqrt(np.mean(vel_errors**2))),
            "max_pos_error": float(np.max(pos_errors)),
        }

    def test_rmse_increases_with_latency(self):
        """Position RMSE should increase monotonically with latency."""
        results = {}
        for lat in [0, 1, 2, 3]:
            results[lat] = self._run_trajectory_with_latency(lat)

        # Monotonic increase in pos RMSE
        for lat in [1, 2, 3]:
            self.assertGreater(
                results[lat]["pos_rmse"],
                results[lat - 1]["pos_rmse"] * 0.95,  # 5% tolerance for stochasticity
                f"RMSE should increase: lat={lat} ({results[lat]['pos_rmse']:.4f}) "
                f"vs lat={lat-1} ({results[lat-1]['pos_rmse']:.4f})",
            )

    def test_latency_0_baseline(self):
        """Zero latency should achieve <15mm pos RMSE on Stage D arc."""
        r = self._run_trajectory_with_latency(0)
        self.assertLess(r["pos_rmse"], 0.015, f"pos RMSE={r['pos_rmse']:.4f}")

    def test_latency_1_bounded(self):
        """1-frame delay (20ms): pos RMSE should be <40mm."""
        r = self._run_trajectory_with_latency(1)
        self.assertLess(r["pos_rmse"], 0.040, f"pos RMSE={r['pos_rmse']:.4f}")

    def test_latency_2_bounded(self):
        """2-frame delay (40ms): pos RMSE should be <80mm."""
        r = self._run_trajectory_with_latency(2)
        self.assertLess(r["pos_rmse"], 0.080, f"pos RMSE={r['pos_rmse']:.4f}")

    def test_latency_3_bounded(self):
        """3-frame delay (60ms): pos RMSE should be <120mm."""
        r = self._run_trajectory_with_latency(3)
        self.assertLess(r["pos_rmse"], 0.120, f"pos RMSE={r['pos_rmse']:.4f}")

    def test_velocity_still_usable_at_3_frames(self):
        """Even at 3-frame delay, velocity RMSE should be <2 m/s."""
        r = self._run_trajectory_with_latency(3)
        self.assertLess(r["vel_rmse"], 2.0, f"vel RMSE={r['vel_rmse']:.4f}")

    def test_latency_plus_dropout(self):
        """2-frame latency + 10% dropout: RMSE bounded and no divergence."""
        torch.manual_seed(42)
        r = self._run_trajectory_with_latency(
            latency_steps=2, dropout_prob=0.10,
        )
        # Combined degradation: should still be <100mm
        self.assertLess(r["pos_rmse"], 0.100, f"pos RMSE={r['pos_rmse']:.4f}")
        # No divergence: max error bounded
        self.assertLess(r["max_pos_error"], 0.200, f"max error={r['max_pos_error']:.4f}")

    def test_latency_plus_high_dropout(self):
        """2-frame latency + 20% dropout: EKF bridges gaps without diverging."""
        torch.manual_seed(42)
        r = self._run_trajectory_with_latency(
            latency_steps=2, dropout_prob=0.20,
        )
        self.assertLess(r["pos_rmse"], 0.150, f"pos RMSE={r['pos_rmse']:.4f}")
        self.assertLess(r["max_pos_error"], 0.300, f"max error={r['max_pos_error']:.4f}")

    def test_latency_multi_env_independence(self):
        """Different envs with different trajectories should not cross-talk through latency buffer."""
        cfg = D435iNoiseModelCfg(
            sigma_xy_base=0.0, sigma_z_base=0.0, sigma_z_per_metre=0.0,
            dropout_prob=0.0, latency_steps=2,
        )
        noise = D435iNoiseModel(num_envs=3, cfg=cfg)

        # Feed distinct positions per env
        for i in range(5):
            gt = torch.tensor([
                [0.1 * i, 0.0, 0.05],
                [0.0, 0.1 * i, 0.05],
                [0.0, 0.0, 0.05 + 0.1 * i],
            ])
            out, _ = noise.sample(gt)

        # At step 4 (latency=2), output should be step 2's values
        # Step 2 values: env0=[0.2,0,0.05], env1=[0,0.2,0.05], env2=[0,0,0.25]
        self.assertAlmostEqual(out[0, 0].item(), 0.2, places=3)
        self.assertAlmostEqual(out[1, 1].item(), 0.2, places=3)
        self.assertAlmostEqual(out[2, 2].item(), 0.25, places=3)


class TestLatencyPolicyImpact(unittest.TestCase):
    """Test the practical impact of latency on policy-relevant metrics."""

    def test_apex_detection_delay(self):
        """At apex (vz=0), latency causes the EKF to report position from N steps earlier.

        At 3-frame delay, the stale observation is from 60ms ago.
        At vz ≈ 0 (apex), position drift is minimal — latency is least harmful at apex.
        """
        dt = 0.02
        pos0 = np.array([0.0, 0.0, 0.10])  # above contact zone
        vel0 = np.array([0.0, 0.0, 3.0])
        n_steps = 28  # stay in free flight

        gt_pos, gt_vel = ballistic_trajectory(pos0, vel0, dt, n_steps)

        # Find apex step (where vz closest to 0)
        apex_step = np.argmin(np.abs(gt_vel[:, 2]))
        apex_z = gt_pos[apex_step, 2]

        # Run with 3-frame latency
        noise_cfg = D435iNoiseModelCfg(
            sigma_xy_base=0.0, sigma_z_base=0.0, sigma_z_per_metre=0.0,
            dropout_prob=0.0, latency_steps=3,
        )
        noise = D435iNoiseModel(num_envs=1, cfg=noise_cfg)
        ekf = BallEKF(num_envs=1, cfg=BallEKFConfig(contact_aware=True))
        ekf.reset(
            torch.tensor([0]),
            torch.tensor([pos0], dtype=torch.float32),
            torch.tensor([vel0], dtype=torch.float32),
        )

        for i in range(n_steps):
            gt = torch.tensor([gt_pos[i]], dtype=torch.float32)
            noisy, det = noise.sample(gt)
            ekf.step(noisy, det, dt=dt)

        # Find the step nearest the apex (vz ≈ 0) and check EKF error there
        # At apex, ball is near-stationary → latency causes minimal position drift
        apex_step = np.argmin(np.abs(gt_vel[:, 2]))

        # Re-run to record per-step EKF estimates
        ekf2 = BallEKF(num_envs=1, cfg=BallEKFConfig(contact_aware=True))
        ekf2.reset(
            torch.tensor([0]),
            torch.tensor([pos0], dtype=torch.float32),
            torch.tensor([vel0], dtype=torch.float32),
        )
        noise2 = D435iNoiseModel(num_envs=1, cfg=noise_cfg)
        ekf_z = []
        for i in range(n_steps):
            gt = torch.tensor([gt_pos[i]], dtype=torch.float32)
            noisy, det = noise2.sample(gt)
            ekf2.step(noisy, det, dt=dt)
            ekf_z.append(ekf2.pos[0, 2].item())

        # At apex, EKF's ballistic predictor is well-matched (vz≈0 → small prediction error)
        # Even with 3-frame latency, the error near apex should be small (<50mm)
        error_at_apex = abs(ekf_z[apex_step] - gt_pos[apex_step, 2])
        self.assertLess(error_at_apex, 0.050,
                        f"Z error at apex (step {apex_step}): {error_at_apex:.4f}m")


if __name__ == "__main__":
    # Run with verbose output
    print("=" * 70)
    print("LATENCY INJECTION TESTS")
    print("=" * 70)
    unittest.main(verbosity=2)
