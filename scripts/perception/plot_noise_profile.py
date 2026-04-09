#!/usr/bin/env python3
"""Plot D435i noise model profile across ball heights.

Visualizes position noise, velocity noise, dropout rate, and signal-to-noise
ratio as a function of ball-camera distance. Helps diagnose which target
heights are most affected by perception noise.

Usage:
    python scripts/perception/plot_noise_profile.py \
        --out images/perception/d435i_noise_profile.png
"""

import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt


# D435i noise model defaults (from D435iNoiseParams dataclass)
SIGMA_XY_PER_METRE = 0.0025
SIGMA_XY_FLOOR = 0.001
SIGMA_Z_BASE = 0.001
SIGMA_Z_QUADRATIC = 0.005
DROPOUT_BASE = 0.20
DROPOUT_RANGE = 0.30
DROPOUT_SCALE = 0.80

# Scene constants
PADDLE_OFFSET_Z = 0.070  # paddle above trunk centre
CAMERA_ON_PADDLE = True   # camera is upward-facing on paddle


def position_noise(z_dist: np.ndarray, scale: float = 1.0):
    """Compute position noise σ_xy and σ_z vs distance."""
    sigma_xy = np.maximum(SIGMA_XY_FLOOR * scale, SIGMA_XY_PER_METRE * scale * z_dist)
    sigma_z = SIGMA_Z_BASE * scale + SIGMA_Z_QUADRATIC * scale * z_dist ** 2
    return sigma_xy, sigma_z


def velocity_noise(z_dist: np.ndarray, scale: float = 1.0, fps: float = 30.0):
    """Compute velocity noise (finite-difference approximation) vs distance."""
    dt = 1.0 / fps
    sigma_xy, sigma_z = position_noise(z_dist, scale)
    sigma_vel_xy = np.sqrt(2) * sigma_xy / dt
    sigma_vel_z = np.sqrt(2) * sigma_z / dt
    return sigma_vel_xy, sigma_vel_z


def dropout_rate(z_dist: np.ndarray, scale: float = 1.0):
    """Compute dropout probability vs distance."""
    z_excess = np.maximum(z_dist - 0.5, 0.0)
    p = DROPOUT_BASE * scale + DROPOUT_RANGE * scale * (
        1.0 - np.exp(-z_excess / DROPOUT_SCALE)
    )
    return np.clip(p, 0.0, 1.0)


def peak_velocity(target_height: np.ndarray):
    """Peak ball velocity at a given apex target height (energy conservation)."""
    g = 9.81
    return np.sqrt(2 * g * target_height)


def snr_position(z_dist: np.ndarray, target_heights: np.ndarray, scale: float = 1.0):
    """Signal-to-noise ratio for position obs at each target height."""
    sigma_xy, sigma_z = position_noise(z_dist, scale)
    # Signal: target height is the amplitude of z variation
    snr_z = target_heights / sigma_z
    # XY signal is the paddle extent (0.085m)
    snr_xy = 0.085 / sigma_xy
    return snr_xy, snr_z


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=str, required=True, help="Output figure path")
    parser.add_argument("--scale", type=float, default=1.0, help="Noise scale factor")
    args = parser.parse_args()

    z_dist = np.linspace(0.02, 1.5, 500)  # ball-camera distance
    target_heights = np.array([0.10, 0.20, 0.30, 0.40, 0.50])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"D435i Noise Model Profile (noise_scale={args.scale:.2f})",
        fontsize=14, fontweight="bold",
    )

    # --- Panel 1: Position noise ---
    ax = axes[0, 0]
    sigma_xy, sigma_z = position_noise(z_dist, args.scale)
    ax.plot(z_dist * 100, sigma_xy * 1000, label="σ_xy", color="C0", linewidth=2)
    ax.plot(z_dist * 100, sigma_z * 1000, label="σ_z", color="C1", linewidth=2)
    ax.set_xlabel("Ball–camera distance (cm)")
    ax.set_ylabel("Position noise σ (mm)")
    ax.set_title("Position Noise")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Mark typical distances for each target
    for h in target_heights:
        ax.axvline(h * 100, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
        ax.text(h * 100, ax.get_ylim()[1] * 0.95, f"{h:.1f}m", fontsize=7,
                ha="center", color="gray")

    # --- Panel 2: Velocity noise ---
    ax = axes[0, 1]
    sv_xy, sv_z = velocity_noise(z_dist, args.scale)
    ax.plot(z_dist * 100, sv_xy, label="σ_vel_xy", color="C0", linewidth=2)
    ax.plot(z_dist * 100, sv_z, label="σ_vel_z", color="C1", linewidth=2)
    ax.set_xlabel("Ball–camera distance (cm)")
    ax.set_ylabel("Velocity noise σ (m/s)")
    ax.set_title("Velocity Noise (30 Hz)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    for h in target_heights:
        ax.axvline(h * 100, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    # --- Panel 3: Dropout rate ---
    ax = axes[0, 2]
    dr = dropout_rate(z_dist, args.scale)
    ax.plot(z_dist * 100, dr * 100, color="C3", linewidth=2)
    ax.set_xlabel("Ball–camera distance (cm)")
    ax.set_ylabel("Dropout rate (%)")
    ax.set_title("Measurement Dropout")
    ax.set_ylim(0, 60)
    ax.grid(True, alpha=0.3)
    ax.axhline(DROPOUT_BASE * args.scale * 100, color="C3", linestyle=":",
               alpha=0.5, label=f"baseline {DROPOUT_BASE * args.scale * 100:.0f}%")
    ax.legend()
    for h in target_heights:
        ax.axvline(h * 100, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    # --- Panel 4: Velocity SNR ---
    ax = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(target_heights)))
    for i, h in enumerate(target_heights):
        v_peak = peak_velocity(np.array([h]))[0]
        _, sv_z_arr = velocity_noise(z_dist, args.scale)
        snr = v_peak / sv_z_arr
        ax.plot(z_dist * 100, snr, label=f"target={h:.1f}m (v={v_peak:.2f}m/s)",
                color=colors[i], linewidth=2)
    ax.set_xlabel("Ball–camera distance (cm)")
    ax.set_ylabel("Velocity SNR (peak_vel / σ_vel_z)")
    ax.set_title("Velocity Signal-to-Noise Ratio")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.axhline(5, color="red", linestyle=":", alpha=0.5, label="SNR=5 threshold")
    ax.set_ylim(0, 80)

    # --- Panel 5: Effective noise budget per target ---
    ax = axes[1, 1]
    # For each target, compute noise at the mean ball-camera distance during flight
    # Mean distance ≈ target_height / 2 (ball spends equal time ascending/descending)
    bar_heights_pos = []
    bar_heights_vel = []
    bar_labels = []
    for h in target_heights:
        z_mean = h / 2  # mean distance during flight
        s_xy, s_z = position_noise(np.array([z_mean]), args.scale)
        sv_xy_val, sv_z_val = velocity_noise(np.array([z_mean]), args.scale)
        bar_heights_pos.append(s_z[0] * 1000)
        bar_heights_vel.append(sv_z_val[0])
        bar_labels.append(f"{h:.1f}")
    x = np.arange(len(target_heights))
    width = 0.35
    bars1 = ax.bar(x - width / 2, bar_heights_pos, width, label="σ_z (mm)",
                   color="C0", alpha=0.7)
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width / 2, bar_heights_vel, width, label="σ_vel_z (m/s)",
                    color="C1", alpha=0.7)
    ax.set_xlabel("Target height (m)")
    ax.set_ylabel("Position noise σ_z (mm)", color="C0")
    ax2.set_ylabel("Velocity noise σ_vel_z (m/s)", color="C1")
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels)
    ax.set_title("Noise at Mean Flight Distance")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 6: Noise impact table ---
    ax = axes[1, 2]
    ax.axis("off")
    table_data = []
    headers = ["Target\n(m)", "Peak vel\n(m/s)", "σ_z\n(mm)", "σ_vel_z\n(m/s)",
               "Vel SNR", "Dropout\n(%)"]
    for h in target_heights:
        z_mean = h / 2
        v_peak = np.sqrt(2 * 9.81 * h)
        s_xy, s_z = position_noise(np.array([z_mean]), args.scale)
        sv_xy_val, sv_z_val = velocity_noise(np.array([z_mean]), args.scale)
        dr_val = dropout_rate(np.array([z_mean]), args.scale)
        snr = v_peak / sv_z_val[0]
        table_data.append([
            f"{h:.2f}", f"{v_peak:.2f}", f"{s_z[0] * 1000:.2f}",
            f"{sv_z_val[0]:.3f}", f"{snr:.1f}", f"{dr_val[0] * 100:.1f}",
        ])

    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    ax.set_title("Noise Summary per Target", pad=20)

    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")

    # Print summary
    print("\n--- Noise Summary (noise_scale={:.2f}) ---".format(args.scale))
    print(f"{'Target':>8s} {'Peak v':>8s} {'σ_z':>8s} {'σ_vel_z':>10s} {'SNR':>6s} {'Drop':>6s}")
    for row in table_data:
        print(f"{row[0]:>8s} {row[1]:>8s} {row[2]:>8s} {row[3]:>10s} {row[4]:>6s} {row[5]:>6s}")


if __name__ == "__main__":
    main()
