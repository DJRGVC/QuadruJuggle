"""High-height readiness analysis: D435i noise model + EKF at juggling heights.

Simulates ballistic trajectories at target apex heights (0.50, 0.70, 1.00m)
through the full D435i noise model → EKF pipeline. Measures:
  - Position/velocity RMSE during flight
  - Detection rate and dropout at each height
  - EKF accuracy at apex (where pi1 needs it most)
  - Camera detection window (fraction of flight visible to camera)

This validates that the perception pipeline is ready for when the policy
learns to juggle, not just balance.

Usage:
    python scripts/perception/analyze_high_height_readiness.py \
        --out images/perception/high_height_readiness.png
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# --- Direct imports to avoid Isaac Lab dependency ---

_PERC_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "..", "..", "source", "go1_ball_balance", "go1_ball_balance", "perception",
))


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_ekf_mod = _load_module("perception.ball_ekf", os.path.join(_PERC_DIR, "ball_ekf.py"))
BallEKF = _ekf_mod.BallEKF
BallEKFConfig = _ekf_mod.BallEKFConfig

_noise_mod = _load_module("perception.noise_model", os.path.join(_PERC_DIR, "noise_model.py"))
D435iNoiseModel = _noise_mod.D435iNoiseModel
D435iNoiseModelCfg = _noise_mod.D435iNoiseModelCfg

# --- Constants ---

GRAVITY = -9.81
DRAG_COEFF = 0.112  # BallEKFConfig default
DT = 0.02           # 50 Hz policy rate
CAMERA_MIN_HEIGHT = 0.20  # Camera FOV lower bound (m above paddle)


@dataclass
class TrajectoryResult:
    """Results from a single trajectory simulation."""
    target_apex: float
    actual_apex: float
    vz0: float
    flight_steps: int
    total_steps: int
    detection_rate: float
    camera_visible_frac: float  # fraction of flight where ball > CAMERA_MIN_HEIGHT
    pos_rmse_flight: float      # RMSE during free flight only
    vel_rmse_flight: float
    apex_pos_err: float         # position error at apex
    apex_vel_err: float         # velocity error at apex
    pos_rmse_visible: float     # RMSE during camera-visible window
    vel_rmse_visible: float
    # Noise characteristics at apex
    sigma_xy_apex: float
    sigma_z_apex: float
    dropout_apex: float
    # Time series for plotting
    gt_z: np.ndarray
    ekf_z: np.ndarray
    noisy_z: np.ndarray
    detected_mask: np.ndarray
    time_s: np.ndarray


def vz0_for_apex(target_h: float) -> float:
    """Compute initial vertical velocity for a given apex height (approx, ignoring drag)."""
    # h = vz0^2 / (2*|g|)  =>  vz0 = sqrt(2*|g|*h)
    return math.sqrt(2 * abs(GRAVITY) * target_h)


def ballistic_trajectory(
    pos0: np.ndarray, vel0: np.ndarray, dt: float, n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate GT ballistic trajectory with drag."""
    positions = np.zeros((n_steps, 3))
    velocities = np.zeros((n_steps, 3))
    pos, vel = pos0.copy(), vel0.copy()
    for i in range(n_steps):
        positions[i] = pos
        velocities[i] = vel
        speed = np.linalg.norm(vel)
        a = np.array([0.0, 0.0, GRAVITY])
        if speed > 1e-8:
            a -= DRAG_COEFF * speed * vel
        pos = pos + vel * dt + 0.5 * a * dt**2
        vel = vel + a * dt
    return positions, velocities


def simulate_trajectory(target_apex: float, seed: int = 42) -> TrajectoryResult:
    """Simulate a full juggle arc through D435i noise + EKF."""
    torch.manual_seed(seed)

    vz0 = vz0_for_apex(target_apex)
    pos0 = np.array([0.0, 0.0, 0.025])  # just above paddle
    vel0 = np.array([0.0, 0.0, vz0])

    # Generous flight time
    t_flight = 2.0 * vz0 / abs(GRAVITY) * 1.15
    n_steps = max(int(t_flight / DT) + 10, 50)

    gt_pos, gt_vel = ballistic_trajectory(pos0, vel0, DT, n_steps)

    # Find flight phase (z > paddle surface)
    in_flight = gt_pos[:, 2] > 0.022
    flight_end = np.where(~in_flight & (np.arange(n_steps) > 5))[0]
    if len(flight_end) > 0:
        n_steps = min(n_steps, flight_end[0] + 5)
        gt_pos = gt_pos[:n_steps]
        gt_vel = gt_vel[:n_steps]
        in_flight = in_flight[:n_steps]

    # Setup noise model + EKF
    noise = D435iNoiseModel(num_envs=1, device="cpu")
    ekf = BallEKF(
        num_envs=1, device="cpu",
        cfg=BallEKFConfig(
            contact_aware=True,
            nis_gate_enabled=False,
            anchor_enabled=False,  # no anchor — we want to see pure EKF flight tracking
        ),
    )
    ekf.reset(
        torch.tensor([0]),
        torch.from_numpy(pos0.astype("float32")).unsqueeze(0),
        torch.from_numpy(vel0.astype("float32")).unsqueeze(0),
    )

    # Run simulation
    ekf_positions = np.zeros((n_steps, 3))
    ekf_velocities = np.zeros((n_steps, 3))
    noisy_positions = np.zeros((n_steps, 3))
    detected_mask = np.zeros(n_steps, dtype=bool)

    for i in range(n_steps):
        gt_b = torch.from_numpy(gt_pos[i].astype("float32")).unsqueeze(0)
        noisy_pos, detected = noise.sample(gt_b)

        noisy_positions[i] = noisy_pos[0].numpy()
        detected_mask[i] = detected[0].item()

        ekf.step(noisy_pos, detected, dt=DT)
        ekf_positions[i] = ekf.pos[0].detach().numpy()
        ekf_velocities[i] = ekf.vel[0].detach().numpy()

    # Compute metrics
    flight_mask = in_flight[:n_steps]
    flight_steps = int(flight_mask.sum())

    # Camera-visible window (ball > 200mm above paddle)
    visible_mask = gt_pos[:n_steps, 2] > CAMERA_MIN_HEIGHT
    camera_visible_frac = float(visible_mask.sum()) / max(flight_steps, 1)

    # Detection rate during flight
    det_in_flight = detected_mask & flight_mask
    detection_rate = float(det_in_flight.sum()) / max(flight_steps, 1)

    # RMSE during flight
    if flight_steps > 3:
        flight_idx = np.where(flight_mask)[0][3:]  # skip warmup
        pos_err = np.linalg.norm(ekf_positions[flight_idx] - gt_pos[flight_idx], axis=1)
        vel_err = np.linalg.norm(ekf_velocities[flight_idx] - gt_vel[flight_idx], axis=1)
        pos_rmse_flight = float(np.sqrt(np.mean(pos_err**2)))
        vel_rmse_flight = float(np.sqrt(np.mean(vel_err**2)))
    else:
        pos_rmse_flight = vel_rmse_flight = float("nan")

    # RMSE during camera-visible window
    vis_flight = visible_mask & flight_mask
    vis_idx = np.where(vis_flight)[0]
    if len(vis_idx) > 2:
        pos_err_v = np.linalg.norm(ekf_positions[vis_idx] - gt_pos[vis_idx], axis=1)
        vel_err_v = np.linalg.norm(ekf_velocities[vis_idx] - gt_vel[vis_idx], axis=1)
        pos_rmse_visible = float(np.sqrt(np.mean(pos_err_v**2)))
        vel_rmse_visible = float(np.sqrt(np.mean(vel_err_v**2)))
    else:
        pos_rmse_visible = vel_rmse_visible = float("nan")

    # Apex metrics
    apex_idx = np.argmax(gt_pos[:n_steps, 2])
    actual_apex = float(gt_pos[apex_idx, 2])
    apex_pos_err = float(np.linalg.norm(ekf_positions[apex_idx] - gt_pos[apex_idx]))
    apex_vel_err = float(np.linalg.norm(ekf_velocities[apex_idx] - gt_vel[apex_idx]))

    # Noise characteristics at apex height
    z_apex = actual_apex
    cfg = noise.cfg
    sigma_xy_apex = max(cfg.sigma_xy_per_metre * z_apex, cfg.sigma_xy_floor)
    sigma_z_apex = cfg.sigma_z_base + cfg.sigma_z_quadratic * z_apex**2
    z_excess = max(z_apex - 0.5, 0.0)
    dropout_apex = cfg.dropout_base + cfg.dropout_range * (1.0 - math.exp(-z_excess / cfg.dropout_scale))

    time_s = np.arange(n_steps) * DT

    return TrajectoryResult(
        target_apex=target_apex, actual_apex=actual_apex, vz0=vz0,
        flight_steps=flight_steps, total_steps=n_steps,
        detection_rate=detection_rate, camera_visible_frac=camera_visible_frac,
        pos_rmse_flight=pos_rmse_flight, vel_rmse_flight=vel_rmse_flight,
        apex_pos_err=apex_pos_err, apex_vel_err=apex_vel_err,
        pos_rmse_visible=pos_rmse_visible, vel_rmse_visible=vel_rmse_visible,
        sigma_xy_apex=sigma_xy_apex, sigma_z_apex=sigma_z_apex,
        dropout_apex=dropout_apex,
        gt_z=gt_pos[:n_steps, 2], ekf_z=ekf_positions[:n_steps, 2],
        noisy_z=noisy_positions[:n_steps, 2], detected_mask=detected_mask[:n_steps],
        time_s=time_s,
    )


def plot_readiness(results: list[TrajectoryResult], out_path: str) -> None:
    """Generate 6-panel readiness figure."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("D435i + EKF High-Height Readiness Analysis", fontsize=14, fontweight="bold")

    colors = ["#2196F3", "#FF9800", "#E91E63"]

    # Panel 1-3: Trajectory tracking for each target height
    for idx, (r, color) in enumerate(zip(results, colors)):
        ax = axes[0, idx]
        t = r.time_s

        # GT trajectory
        ax.plot(t, r.gt_z * 1000, "k-", lw=1.5, label="GT", alpha=0.8)

        # EKF estimate
        ax.plot(t, r.ekf_z * 1000, color=color, lw=1.2, label="EKF", alpha=0.9)

        # Noisy measurements (only detected)
        det_t = t[r.detected_mask]
        det_z = r.noisy_z[r.detected_mask]
        ax.scatter(det_t, det_z * 1000, c=color, s=8, alpha=0.3, label="Meas", zorder=3)

        # Camera FOV line
        ax.axhline(CAMERA_MIN_HEIGHT * 1000, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.text(t[-1] * 0.95, CAMERA_MIN_HEIGHT * 1000 + 10, "camera FOV",
                ha="right", va="bottom", fontsize=7, color="gray")

        ax.set_title(f"Target: {r.target_apex:.2f}m (actual: {r.actual_apex*1000:.0f}mm)",
                     fontsize=10)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Ball Z (mm)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=-20)

    # Panel 4: Position RMSE comparison
    ax = axes[1, 0]
    targets = [r.target_apex for r in results]
    flight_rmse = [r.pos_rmse_flight * 1000 for r in results]
    vis_rmse = [r.pos_rmse_visible * 1000 for r in results]
    apex_err = [r.apex_pos_err * 1000 for r in results]
    x = np.arange(len(targets))
    w = 0.25
    ax.bar(x - w, flight_rmse, w, label="Flight RMSE", color="#2196F3", alpha=0.8)
    ax.bar(x, vis_rmse, w, label="Visible RMSE", color="#FF9800", alpha=0.8)
    ax.bar(x + w, apex_err, w, label="Apex error", color="#E91E63", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}m" for t in targets])
    ax.set_ylabel("Position error (mm)")
    ax.set_title("EKF Position Accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 5: Detection rate + camera window
    ax = axes[1, 1]
    det_rates = [r.detection_rate * 100 for r in results]
    cam_frac = [r.camera_visible_frac * 100 for r in results]
    ax.bar(x - 0.15, det_rates, 0.3, label="Detection rate", color="#4CAF50", alpha=0.8)
    ax.bar(x + 0.15, cam_frac, 0.3, label="Camera visible", color="#9C27B0", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}m" for t in targets])
    ax.set_ylabel("Fraction (%)")
    ax.set_title("Detection & Camera Window")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 6: Noise characteristics at apex
    ax = axes[1, 2]
    sigma_xy = [r.sigma_xy_apex * 1000 for r in results]
    sigma_z = [r.sigma_z_apex * 1000 for r in results]
    dropout = [r.dropout_apex * 100 for r in results]

    ax2 = ax.twinx()
    bars1 = ax.bar(x - 0.15, sigma_xy, 0.15, label="σ_xy", color="#2196F3", alpha=0.8)
    bars2 = ax.bar(x, sigma_z, 0.15, label="σ_z", color="#FF9800", alpha=0.8)
    line = ax2.plot(x + 0.15, dropout, "s-", color="#E91E63", markersize=8, label="Dropout %")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.2f}m" for t in targets])
    ax.set_ylabel("Noise std (mm)")
    ax2.set_ylabel("Dropout (%)")
    ax.set_title("D435i Noise at Apex Height")
    lines = [bars1, bars2, line[0]]
    labels = ["σ_xy", "σ_z", "Dropout"]
    ax.legend(lines, labels, fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_summary(results: list[TrajectoryResult]) -> None:
    """Print summary table."""
    print("\n" + "=" * 90)
    print("HIGH-HEIGHT READINESS ANALYSIS — D435i + EKF")
    print("=" * 90)
    header = (
        f"{'Target':>8} {'Apex':>8} {'vz0':>6} {'Flight':>7} "
        f"{'Det%':>6} {'CamVis%':>8} {'FlightRMSE':>11} {'VisRMSE':>9} "
        f"{'ApexErr':>8} {'σ_xy':>6} {'σ_z':>6} {'Drop%':>6}"
    )
    print(header)
    print("-" * 90)
    for r in results:
        print(
            f"{r.target_apex:>7.2f}m {r.actual_apex*1000:>6.0f}mm {r.vz0:>5.1f} "
            f"{r.flight_steps:>6}st "
            f"{r.detection_rate*100:>5.1f}% {r.camera_visible_frac*100:>7.1f}% "
            f"{r.pos_rmse_flight*1000:>9.1f}mm {r.pos_rmse_visible*1000:>7.1f}mm "
            f"{r.apex_pos_err*1000:>6.1f}mm "
            f"{r.sigma_xy_apex*1000:>4.1f}mm {r.sigma_z_apex*1000:>4.1f}mm "
            f"{r.dropout_apex*100:>5.1f}%"
        )
    print("=" * 90)

    # Readiness verdict
    all_ok = all(r.pos_rmse_flight < 0.020 for r in results)  # < 20mm
    print(f"\nREADINESS: {'✓ PASS' if all_ok else '⚠ NEEDS WORK'} — "
          f"All flight RMSE < 20mm: {all_ok}")
    for r in results:
        status = "✓" if r.pos_rmse_flight < 0.020 else "✗"
        print(f"  {status} {r.target_apex:.2f}m: flight RMSE = {r.pos_rmse_flight*1000:.1f}mm, "
              f"apex err = {r.apex_pos_err*1000:.1f}mm, "
              f"det rate = {r.detection_rate*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="High-height readiness analysis")
    parser.add_argument("--out", default="images/perception/high_height_readiness.png",
                        help="Output figure path")
    parser.add_argument("--targets", default="0.50,0.70,1.00",
                        help="Comma-separated target apex heights (m)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    targets = [float(t) for t in args.targets.split(",")]
    results = []
    for target in targets:
        r = simulate_trajectory(target, seed=args.seed)
        results.append(r)

    print_summary(results)
    plot_readiness(results, args.out)


if __name__ == "__main__":
    main()
