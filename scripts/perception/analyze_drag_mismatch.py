#!/usr/bin/env python3
"""Analyze drag model mismatch between PhysX simulation and EKF prediction.

PhysX uses linear_damping=0.1: a_drag = -0.1 * v  (linear in velocity)
EKF uses quadratic drag:       a_drag = -0.112 * |v| * v  (quadratic in velocity)

These diverge at high speeds, which may explain the ascending vz RMSE growth
with target height found in iter 138.

At crossover speed: 0.1 = 0.112 * |v| → |v| = 0.89 m/s
Above 0.89 m/s, EKF over-predicts drag (expects ball to slow faster than it does).

This script:
1. Simulates bounce cycles under both drag models
2. Runs the EKF (which uses quadratic) on PhysX-linear ground truth
3. Computes velocity prediction error due purely to the drag mismatch
4. Generates comparison figure

Usage:
    python scripts/perception/analyze_drag_mismatch.py \
        --out images/perception/drag_mismatch_iter139.png
"""

import argparse
import math
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib required")
    sys.exit(1)


def simulate_bounce(target_h: float, dt: float = 0.005, g: float = 9.81,
                    drag_model: str = "quadratic", drag_coeff: float = 0.112,
                    linear_damping: float = 0.1) -> dict:
    """Simulate a single vertical bounce cycle.

    drag_model: "quadratic" (EKF) or "linear" (PhysX)
    """
    # Launch velocity for target height (no-drag approximation, then overshoot)
    v0 = math.sqrt(2 * g * target_h) * 1.10  # overshoot to compensate drag

    pos_z = [0.02]  # start at ball radius above paddle
    vel_z = [v0]
    times = [0.0]

    z, vz = 0.02, v0
    t = 0.0
    max_steps = int(3.0 / dt)

    for _ in range(max_steps):
        # Compute drag deceleration
        speed = abs(vz)
        if drag_model == "quadratic":
            # a_drag = -drag_coeff * |v| * v_hat (in z direction)
            a_drag_z = -drag_coeff * speed * vz
        elif drag_model == "linear":
            # PhysX: a_drag = -linear_damping * v
            a_drag_z = -linear_damping * vz
        else:
            a_drag_z = 0.0

        az = -g + a_drag_z
        vz += az * dt
        z += vz * dt
        t += dt

        pos_z.append(z)
        vel_z.append(vz)
        times.append(t)

        # Stop when ball returns to paddle
        if z <= 0.02 and vz < 0 and t > 0.05:
            break

    return {
        "z": np.array(pos_z),
        "vz": np.array(vel_z),
        "t": np.array(times),
        "apex": max(pos_z),
        "target_h": target_h,
    }


def compute_ekf_prediction_error(target_h: float, dt_sim: float = 0.005,
                                 dt_ekf: float = 0.02) -> dict:
    """Compute the pure drag-mismatch error.

    Ground truth: PhysX linear damping
    EKF prediction: quadratic drag model

    No measurement noise — just the dynamics model mismatch.
    """
    # Ground truth from PhysX model
    gt = simulate_bounce(target_h, dt=dt_sim, drag_model="linear")
    # EKF model prediction from same initial conditions
    ekf_pred = simulate_bounce(target_h, dt=dt_sim, drag_model="quadratic")

    # Align by time (both use same dt and start)
    n = min(len(gt["t"]), len(ekf_pred["t"]))

    vz_error = ekf_pred["vz"][:n] - gt["vz"][:n]
    z_error = ekf_pred["z"][:n] - gt["z"][:n]

    # Find ascending phase (vz > 0 in ground truth)
    asc_mask = gt["vz"][:n] > 0
    desc_mask = gt["vz"][:n] <= 0

    return {
        "target_h": target_h,
        "gt_apex": gt["apex"],
        "ekf_apex": ekf_pred["apex"],
        "t": gt["t"][:n],
        "gt_vz": gt["vz"][:n],
        "ekf_vz": ekf_pred["vz"][:n],
        "gt_z": gt["z"][:n],
        "ekf_z": ekf_pred["z"][:n],
        "vz_error": vz_error,
        "z_error": z_error,
        "vz_rmse_asc": np.sqrt(np.mean(vz_error[asc_mask]**2)) if asc_mask.any() else 0,
        "vz_rmse_desc": np.sqrt(np.mean(vz_error[desc_mask]**2)) if desc_mask.any() else 0,
        "z_rmse_total": np.sqrt(np.mean(z_error**2)),
        "vz_rmse_total": np.sqrt(np.mean(vz_error**2)),
        "max_vz_error": float(np.max(np.abs(vz_error))),
        "max_z_error": float(np.max(np.abs(z_error))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="images/perception/drag_mismatch_iter139.png")
    args = parser.parse_args()

    targets = [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00]

    # --- Analysis ---
    results = []
    print("Drag model mismatch analysis: PhysX linear_damping=0.1 vs EKF quadratic drag_coeff=0.112")
    print(f"{'Target':>8} {'GT apex':>8} {'EKF apex':>9} {'Δapex':>7} "
          f"{'vz RMSE↑':>9} {'vz RMSE↓':>9} {'max|Δvz|':>9} {'max|Δz|':>9}")
    print("-" * 82)

    for h in targets:
        r = compute_ekf_prediction_error(h)
        results.append(r)
        print(f"{h:>8.2f} {r['gt_apex']:>8.3f} {r['ekf_apex']:>9.3f} "
              f"{r['ekf_apex']-r['gt_apex']:>+7.3f} "
              f"{r['vz_rmse_asc']:>9.4f} {r['vz_rmse_desc']:>9.4f} "
              f"{r['max_vz_error']:>9.4f} {r['max_z_error']:>9.4f}")

    # Crossover speed
    crossover = 0.1 / 0.112
    print(f"\nCrossover speed (linear=quadratic): {crossover:.2f} m/s")
    print(f"Launch velocity for 0.50m: {math.sqrt(2*9.81*0.50):.2f} m/s")
    print(f"Launch velocity for 1.00m: {math.sqrt(2*9.81*1.00):.2f} m/s")

    # Drag ratio at different speeds
    print(f"\nDrag deceleration comparison (m/s²):")
    for v in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]:
        a_linear = 0.1 * v
        a_quad = 0.112 * v * v
        print(f"  v={v:.1f} m/s: PhysX={a_linear:.3f}, EKF={a_quad:.3f}, "
              f"ratio={a_quad/a_linear:.2f}x, Δ={a_quad-a_linear:+.3f}")

    # --- Figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Drag Model Mismatch: PhysX linear_damping=0.1 vs EKF quadratic drag=0.112",
                 fontsize=14, fontweight="bold")

    # Panel 1: vz time series for a few heights
    ax = axes[0, 0]
    for h, color in [(0.20, "tab:blue"), (0.50, "tab:orange"), (1.00, "tab:red")]:
        r = compute_ekf_prediction_error(h, dt_sim=0.002)
        ax.plot(r["t"], r["gt_vz"], "-", color=color, label=f"PhysX h={h}m", linewidth=1.5)
        ax.plot(r["t"], r["ekf_vz"], "--", color=color, label=f"EKF h={h}m", linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("vz (m/s)")
    ax.set_title("Vertical velocity: PhysX (solid) vs EKF (dashed)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)

    # Panel 2: vz RMSE vs target height
    ax = axes[0, 1]
    hs = [r["target_h"] for r in results]
    vz_asc = [r["vz_rmse_asc"] for r in results]
    vz_desc = [r["vz_rmse_desc"] for r in results]
    vz_max = [r["max_vz_error"] for r in results]
    ax.plot(hs, vz_asc, "o-", label="vz RMSE ascending", color="tab:red", linewidth=2)
    ax.plot(hs, vz_desc, "s-", label="vz RMSE descending", color="tab:blue", linewidth=2)
    ax.plot(hs, vz_max, "^--", label="max |Δvz|", color="tab:orange", linewidth=1.5)
    ax.set_xlabel("Target height (m)")
    ax.set_ylabel("Velocity error (m/s)")
    ax.set_title("EKF velocity prediction error (drag mismatch only)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Position error vs height
    ax = axes[1, 0]
    z_rmse = [r["z_rmse_total"] for r in results]
    z_max = [r["max_z_error"] for r in results]
    ax.plot(hs, [e * 1000 for e in z_rmse], "o-", label="z RMSE", color="tab:green", linewidth=2)
    ax.plot(hs, [e * 1000 for e in z_max], "^--", label="max |Δz|", color="tab:purple", linewidth=1.5)
    ax.set_xlabel("Target height (m)")
    ax.set_ylabel("Position error (mm)")
    ax.set_title("EKF position prediction error (drag mismatch only)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Drag force comparison
    ax = axes[1, 1]
    v_range = np.linspace(0, 5, 100)
    a_linear = 0.1 * v_range
    a_quad = 0.112 * v_range**2
    ax.plot(v_range, a_linear, "-", label="PhysX: 0.1·v", color="tab:blue", linewidth=2)
    ax.plot(v_range, a_quad, "-", label="EKF: 0.112·v²", color="tab:red", linewidth=2)
    ax.fill_between(v_range, a_linear, a_quad, alpha=0.15, color="gray")
    ax.axvline(crossover, color="gray", linestyle=":", linewidth=1.5,
               label=f"Crossover: {crossover:.2f} m/s")
    # Mark launch velocities
    for h, marker in [(0.50, "o"), (1.00, "s")]:
        v = math.sqrt(2 * 9.81 * h)
        ax.axvline(v, color="tab:orange", linestyle="--", linewidth=1, alpha=0.7)
        ax.text(v + 0.05, 0.5, f"v₀={v:.1f}\n(h={h}m)", fontsize=8, color="tab:orange")
    ax.set_xlabel("Ball speed (m/s)")
    ax.set_ylabel("Drag deceleration (m/s²)")
    ax.set_title("Drag model comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {args.out}")


if __name__ == "__main__":
    main()
