#!/usr/bin/env python3
"""Decompose EKF estimation error into position vs velocity components at different bounce heights.

For each target height, simulates N full bounce cycles with D435i noise model,
runs the EKF, and computes:
  - Position RMSE (xyz, per-axis)
  - Velocity RMSE (xyz, per-axis)
  - Error by flight phase (ascending, descending, pre-landing, contact)
  - Contribution of each error type to the overall perception gap

This informs whether to focus EKF tuning on R (measurement, position) or Q (process, velocity).

Usage:
    python scripts/perception/analyze_ekf_error_decomposition.py \
        --out images/perception/ekf_error_decomposition_iter138.png
"""

import argparse
import math
import os
import sys

import numpy as np
import torch

# Direct imports to avoid Isaac Lab import chain (pxr not available outside sim)
import importlib.util

_perception_dir = os.path.join(
    os.path.dirname(__file__),
    "../../source/go1_ball_balance/go1_ball_balance/perception",
)

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_ekf_mod = _load_module("ball_ekf", os.path.join(_perception_dir, "ball_ekf.py"))
_noise_mod = _load_module("noise_model", os.path.join(_perception_dir, "noise_model.py"))
BallEKF = _ekf_mod.BallEKF
BallEKFConfig = _ekf_mod.BallEKFConfig
D435iNoiseModel = _noise_mod.D435iNoiseModel
D435iNoiseModelCfg = _noise_mod.D435iNoiseModelCfg


def simulate_bounce_cycle(target_h: float, dt: float = 0.02, restitution: float = 0.85,
                          g: float = 9.81, drag_mode: str = "linear",
                          drag_coeff: float = 0.112, linear_damping: float = 0.1) -> dict:
    """Generate ground-truth ballistic trajectory for one bounce cycle.

    Ball starts at z=0 (paddle surface) with upward velocity to reach target_h.
    Returns arrays of (pos, vel) at each timestep through the full up-down cycle.

    Args:
        drag_mode: "linear" (PhysX sim) or "quadratic" (real aerodynamic drag)
        drag_coeff: quadratic drag coefficient (a = -drag_coeff * |v| * v)
        linear_damping: PhysX linear damping (a = -linear_damping * v)
    """
    # Launch velocity to reach target_h: v0 = sqrt(2*g*h) (ignoring drag for launch)
    # With drag, need slightly more — iterate
    v0 = math.sqrt(2 * g * target_h) * 1.05  # 5% overshoot to compensate drag

    pos_list = []
    vel_list = []

    x, y, z = 0.0, 0.0, 0.02  # ball radius above paddle
    vx, vy, vz = 0.0, 0.0, v0

    max_steps = int(2.0 / dt)  # max 2 seconds per cycle
    for _ in range(max_steps):
        pos_list.append([x, y, z])
        vel_list.append([vx, vy, vz])

        # Ballistic dynamics with drag
        if drag_mode == "quadratic":
            speed = math.sqrt(vx**2 + vy**2 + vz**2)
            ax = -drag_coeff * speed * vx
            ay = -drag_coeff * speed * vy
            az = -g - drag_coeff * speed * vz
        else:  # linear (PhysX)
            ax = -linear_damping * vx
            ay = -linear_damping * vy
            az = -g - linear_damping * vz

        vx += ax * dt
        vy += ay * dt
        vz += az * dt
        x += vx * dt
        y += vy * dt
        z += vz * dt

        # Stop when ball returns to paddle surface
        if z <= 0.02 and vz < 0 and len(pos_list) > 5:
            # One more point at contact
            pos_list.append([x, y, max(z, 0.02)])
            vel_list.append([vx, vy, vz])
            break

    return {
        "pos": np.array(pos_list),
        "vel": np.array(vel_list),
        "target_h": target_h,
        "actual_apex": max(p[2] for p in pos_list),
        "n_steps": len(pos_list),
    }


def run_ekf_with_noise(traj: dict, n_trials: int = 50, seed: int = 42,
                       drag_mode: str = "linear") -> dict:
    """Run EKF on trajectory with D435i noise, collect per-step errors.

    Runs n_trials independent noise realizations to average out stochastic effects.
    """
    dt = 0.02
    n_steps = traj["n_steps"]
    pos_gt = torch.tensor(traj["pos"], dtype=torch.float32)  # (T, 3)
    vel_gt = torch.tensor(traj["vel"], dtype=torch.float32)  # (T, 3)

    torch.manual_seed(seed)

    # Accumulate errors across trials
    all_pos_errors = []  # (n_trials, T, 3)
    all_vel_errors = []
    all_phases = []  # (T,) phase labels

    # Compute phase for each step (same for all trials since GT is deterministic)
    contact_thresh = 0.025
    pre_landing_thresh = 0.08
    phases = []
    for t in range(n_steps):
        z = pos_gt[t, 2].item()
        vz = vel_gt[t, 2].item()
        if z < contact_thresh:
            phases.append("contact")
        elif vz < 0 and z < pre_landing_thresh:
            phases.append("pre_landing")
        elif vz < 0:
            phases.append("descending")
        else:
            phases.append("ascending")

    cfg = BallEKFConfig(drag_mode=drag_mode)
    noise_cfg = D435iNoiseModelCfg()

    noise_model = D435iNoiseModel(num_envs=1, device="cpu", cfg=noise_cfg)

    for trial in range(n_trials):
        ekf = BallEKF(num_envs=1, device="cpu", cfg=cfg)
        noise_model.reset(torch.tensor([0]), pos_gt[0:1].clone())

        # Reset EKF with initial state (with small noise on init)
        init_pos = pos_gt[0:1].clone()
        init_vel = vel_gt[0:1].clone()
        # Add small init noise (realistic: we know approximate launch from paddle)
        init_pos += torch.randn_like(init_pos) * 0.005
        init_vel += torch.randn_like(init_vel) * 0.1
        ekf.reset(torch.tensor([0]), init_pos, init_vel)

        trial_pos_err = []
        trial_vel_err = []

        for t in range(n_steps):
            gt_pos = pos_gt[t:t+1]  # (1, 3)
            gt_vel = vel_gt[t:t+1]  # (1, 3)

            # Apply noise model
            noisy_pos, detected = noise_model.sample(gt_pos)

            # EKF step
            ekf.step(noisy_pos, detected, dt=dt)

            # Record errors
            pos_err = (ekf.pos - gt_pos).squeeze(0).detach().numpy()
            vel_err = (ekf.vel - gt_vel).squeeze(0).detach().numpy()
            trial_pos_err.append(pos_err)
            trial_vel_err.append(vel_err)

        all_pos_errors.append(np.array(trial_pos_err))
        all_vel_errors.append(np.array(trial_vel_err))

    # Stack: (n_trials, T, 3)
    pos_errors = np.array(all_pos_errors)
    vel_errors = np.array(all_vel_errors)

    # Compute RMSE per step, then per phase
    pos_rmse_per_step = np.sqrt(np.mean(pos_errors**2, axis=0))  # (T, 3)
    vel_rmse_per_step = np.sqrt(np.mean(vel_errors**2, axis=0))  # (T, 3)

    # Overall RMSE
    pos_rmse_3d = np.sqrt(np.mean(np.sum(pos_errors**2, axis=2), axis=0))  # (T,)
    vel_rmse_3d = np.sqrt(np.mean(np.sum(vel_errors**2, axis=2), axis=0))  # (T,)

    # Per-phase RMSE
    phase_stats = {}
    for phase in ["ascending", "descending", "pre_landing", "contact"]:
        mask = np.array([p == phase for p in phases])
        if mask.any():
            phase_stats[phase] = {
                "pos_rmse_mm": np.sqrt(np.mean(pos_errors[:, mask, :]**2)) * 1000,
                "vel_rmse_mps": np.sqrt(np.mean(vel_errors[:, mask, :]**2)),
                "pos_rmse_xy_mm": np.sqrt(np.mean(pos_errors[:, mask, :2]**2)) * 1000,
                "pos_rmse_z_mm": np.sqrt(np.mean(pos_errors[:, mask, 2:3]**2)) * 1000,
                "vel_rmse_vxy_mps": np.sqrt(np.mean(vel_errors[:, mask, :2]**2)),
                "vel_rmse_vz_mps": np.sqrt(np.mean(vel_errors[:, mask, 2:3]**2)),
                "n_steps": int(mask.sum()),
                "frac": float(mask.mean()),
            }

    # Overall stats
    overall = {
        "pos_rmse_mm": np.sqrt(np.mean(pos_errors**2)) * 1000,
        "vel_rmse_mps": np.sqrt(np.mean(vel_errors**2)),
        "pos_rmse_xy_mm": np.sqrt(np.mean(pos_errors[:, :, :2]**2)) * 1000,
        "pos_rmse_z_mm": np.sqrt(np.mean(pos_errors[:, :, 2:3]**2)) * 1000,
        "vel_rmse_vxy_mps": np.sqrt(np.mean(vel_errors[:, :, :2]**2)),
        "vel_rmse_vz_mps": np.sqrt(np.mean(vel_errors[:, :, 2:3]**2)),
    }

    return {
        "target_h": traj["target_h"],
        "actual_apex": traj["actual_apex"],
        "n_steps": n_steps,
        "phases": phases,
        "pos_rmse_per_step": pos_rmse_per_step,
        "vel_rmse_per_step": vel_rmse_per_step,
        "pos_rmse_3d": pos_rmse_3d,
        "vel_rmse_3d": vel_rmse_3d,
        "phase_stats": phase_stats,
        "overall": overall,
    }


def print_results(results: list[dict]) -> None:
    """Print formatted decomposition table."""
    print("\n" + "=" * 110)
    print("  EKF ERROR DECOMPOSITION — Position vs Velocity across target heights")
    print("=" * 110)

    # Overall table
    print("\n  Overall RMSE:")
    print(f"  {'Target':>8s}  {'Apex':>6s}  {'Pos 3D':>8s}  {'Pos XY':>8s}  {'Pos Z':>8s}"
          f"  {'Vel 3D':>8s}  {'Vel XY':>8s}  {'Vel Z':>8s}")
    print(f"  {'(m)':>8s}  {'(m)':>6s}  {'(mm)':>8s}  {'(mm)':>8s}  {'(mm)':>8s}"
          f"  {'(m/s)':>8s}  {'(m/s)':>8s}  {'(m/s)':>8s}")
    print(f"  {'─' * 8}  {'─' * 6}  {'─' * 8}  {'─' * 8}  {'─' * 8}"
          f"  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    for r in results:
        o = r["overall"]
        print(f"  {r['target_h']:8.2f}  {r['actual_apex']:6.3f}  {o['pos_rmse_mm']:8.2f}"
              f"  {o['pos_rmse_xy_mm']:8.2f}  {o['pos_rmse_z_mm']:8.2f}"
              f"  {o['vel_rmse_mps']:8.3f}  {o['vel_rmse_vxy_mps']:8.3f}"
              f"  {o['vel_rmse_vz_mps']:8.3f}")

    # Per-phase tables
    for phase in ["ascending", "descending", "pre_landing", "contact"]:
        print(f"\n  Phase: {phase}")
        print(f"  {'Target':>8s}  {'Frac':>6s}  {'Pos 3D':>8s}  {'Pos XY':>8s}  {'Pos Z':>8s}"
              f"  {'Vel 3D':>8s}  {'Vel XY':>8s}  {'Vel Z':>8s}")
        print(f"  {'(m)':>8s}  {'(%)':>6s}  {'(mm)':>8s}  {'(mm)':>8s}  {'(mm)':>8s}"
              f"  {'(m/s)':>8s}  {'(m/s)':>8s}  {'(m/s)':>8s}")
        print(f"  {'─' * 8}  {'─' * 6}  {'─' * 8}  {'─' * 8}  {'─' * 8}"
              f"  {'─' * 8}  {'─' * 8}  {'─' * 8}")
        for r in results:
            ps = r["phase_stats"].get(phase)
            if ps:
                print(f"  {r['target_h']:8.2f}  {ps['frac'] * 100:6.1f}"
                      f"  {ps['pos_rmse_mm']:8.2f}  {ps['pos_rmse_xy_mm']:8.2f}"
                      f"  {ps['pos_rmse_z_mm']:8.2f}  {ps['vel_rmse_mps']:8.3f}"
                      f"  {ps['vel_rmse_vxy_mps']:8.3f}  {ps['vel_rmse_vz_mps']:8.3f}")
            else:
                print(f"  {r['target_h']:8.2f}  {'--':>6s}  {'--':>8s}  {'--':>8s}  {'--':>8s}"
                      f"  {'--':>8s}  {'--':>8s}  {'--':>8s}")

    # Velocity-to-position error ratio (indicator of which drives policy gap)
    print(f"\n  Velocity / Position error ratio (higher = velocity dominates):")
    print(f"  {'Target':>8s}  {'V/P ratio':>10s}  {'Interpretation':>40s}")
    print(f"  {'─' * 8}  {'─' * 10}  {'─' * 40}")
    for r in results:
        o = r["overall"]
        # Convert velocity error to position-equivalent: vel_err * reaction_time
        # Policy operates at 50Hz, so the velocity error translates to position
        # error over one policy step (20ms). But the relevant timescale for
        # catching the ball is ~100ms (5 steps), so use that.
        reaction_time = 0.10  # 100ms = 5 policy steps
        vel_as_pos_mm = o["vel_rmse_mps"] * reaction_time * 1000
        pos_mm = o["pos_rmse_mm"]
        ratio = vel_as_pos_mm / max(pos_mm, 0.01)
        if ratio > 2.0:
            interp = "velocity-dominated"
        elif ratio > 0.5:
            interp = "mixed"
        else:
            interp = "position-dominated"
        print(f"  {r['target_h']:8.2f}  {ratio:10.2f}  "
              f"pos={pos_mm:.1f}mm, vel→pos={vel_as_pos_mm:.1f}mm → {interp}")

    print("\n" + "=" * 110)


def plot_results(results: list[dict], out_path: str) -> None:
    """Generate decomposition figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = [r["target_h"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("EKF Error Decomposition: Position vs Velocity", fontsize=14, fontweight="bold")

    # Panel 1: Overall RMSE vs height
    ax = axes[0, 0]
    ax.plot(targets, [r["overall"]["pos_rmse_xy_mm"] for r in results],
            "o-", label="Pos XY", color="#4477AA")
    ax.plot(targets, [r["overall"]["pos_rmse_z_mm"] for r in results],
            "s-", label="Pos Z", color="#66CCEE")
    ax2 = ax.twinx()
    ax2.plot(targets, [r["overall"]["vel_rmse_vxy_mps"] for r in results],
             "^--", label="Vel XY", color="#EE6677")
    ax2.plot(targets, [r["overall"]["vel_rmse_vz_mps"] for r in results],
             "D--", label="Vel Z", color="#CCBB44")
    ax.set_xlabel("Target height (m)")
    ax.set_ylabel("Position RMSE (mm)", color="#4477AA")
    ax2.set_ylabel("Velocity RMSE (m/s)", color="#EE6677")
    ax.set_title("Overall RMSE vs Target Height")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Stacked bar — pos vs vel contribution
    ax = axes[0, 1]
    reaction_time = 0.10
    pos_contrib = [r["overall"]["pos_rmse_mm"] for r in results]
    vel_contrib = [r["overall"]["vel_rmse_mps"] * reaction_time * 1000 for r in results]
    x = np.arange(len(targets))
    width = 0.35
    ax.bar(x - width/2, pos_contrib, width, label="Position error (mm)",
           color="#4477AA", alpha=0.8)
    ax.bar(x + width/2, vel_contrib, width, label="Velocity→position (mm, 100ms)",
           color="#EE6677", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.1f}" for t in targets])
    ax.set_xlabel("Target height (m)")
    ax.set_ylabel("Effective position error (mm)")
    ax.set_title("Position vs Velocity Error Contribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Per-phase RMSE for selected heights
    ax = axes[1, 0]
    phase_order = ["contact", "ascending", "descending", "pre_landing"]
    phase_colors = {"contact": "#228833", "ascending": "#4477AA",
                    "descending": "#CCBB44", "pre_landing": "#EE6677"}
    x = np.arange(len(targets))
    bar_width = 0.18
    for i, phase in enumerate(phase_order):
        vals = []
        for r in results:
            ps = r["phase_stats"].get(phase)
            vals.append(ps["vel_rmse_vz_mps"] if ps else 0)
        ax.bar(x + i * bar_width - 1.5 * bar_width, vals, bar_width,
               label=phase, color=phase_colors[phase], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:.1f}" for t in targets])
    ax.set_xlabel("Target height (m)")
    ax.set_ylabel("Velocity-Z RMSE (m/s)")
    ax.set_title("Velocity-Z Error by Flight Phase")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: Time-series for 0.50m target (representative)
    ax = axes[1, 1]
    # Find 0.50m result (or closest)
    r50 = min(results, key=lambda r: abs(r["target_h"] - 0.50))
    steps = np.arange(r50["n_steps"])
    times_ms = steps * 20  # dt=20ms
    ax.plot(times_ms, r50["pos_rmse_3d"] * 1000, label="Pos RMSE (mm)",
            color="#4477AA", linewidth=1.5)
    ax2 = ax.twinx()
    ax2.plot(times_ms, r50["vel_rmse_3d"], label="Vel RMSE (m/s)",
             color="#EE6677", linewidth=1.5, linestyle="--")
    # Shade phases
    phase_shading = {"contact": "#228833", "ascending": "#4477AA",
                     "descending": "#CCBB44", "pre_landing": "#EE6677"}
    prev_phase = r50["phases"][0]
    start = 0
    for t_idx in range(1, len(r50["phases"])):
        if r50["phases"][t_idx] != prev_phase or t_idx == len(r50["phases"]) - 1:
            ax.axvspan(start * 20, t_idx * 20, alpha=0.08,
                       color=phase_shading.get(prev_phase, "gray"))
            start = t_idx
            prev_phase = r50["phases"][t_idx]
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Position RMSE (mm)", color="#4477AA")
    ax2.set_ylabel("Velocity RMSE (m/s)", color="#EE6677")
    ax.set_title(f"Error Timeline — {r50['target_h']:.1f}m bounce")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", type=str, default="0.10,0.20,0.30,0.40,0.50,0.70,1.00",
                        help="Comma-separated target heights in metres")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of noise trials per height")
    parser.add_argument("--out", type=str, default=None,
                        help="Output figure path (PNG)")
    parser.add_argument("--drag-mode", type=str, default="linear",
                        choices=["linear", "quadratic"],
                        help="Drag model for ground truth and EKF (default: linear = PhysX)")
    args = parser.parse_args()

    targets = [float(t.strip()) for t in args.targets.split(",")]

    results = []
    for h in targets:
        print(f"  Simulating {h:.2f}m bounce ({args.n_trials} trials, drag={args.drag_mode})...")
        traj = simulate_bounce_cycle(h, drag_mode=args.drag_mode)
        r = run_ekf_with_noise(traj, n_trials=args.n_trials, drag_mode=args.drag_mode)
        results.append(r)

    print_results(results)

    if args.out:
        plot_results(results, args.out)


if __name__ == "__main__":
    main()
