#!/usr/bin/env python3
"""Plot ball height time series from eval trajectories.

Creates a multi-panel figure showing ground-truth ball height, EKF estimate,
detection events, and phase coloring for each target height. Useful for
diagnosing whether the policy is balancing vs juggling and how well the
perception pipeline tracks during flight windows.

Usage:
    python scripts/perception/plot_trajectory_timeseries.py \
        --eval-dir logs/perception/eval_stage_g_d435i \
        --out images/perception/trajectory_timeseries.png
"""

import argparse
import os
import re
import sys

import numpy as np

PHASE_NAMES = {0: "contact", 1: "ascending", 2: "descending"}
PHASE_COLORS = {0: "#E0E0E0", 1: "#A8D5BA", 2: "#F5C6AA"}


def discover_targets(eval_dir: str) -> list[tuple[float, str]]:
    """Find target_X_XX subdirs, return sorted (target_height, npz_path)."""
    targets = []
    for entry in os.listdir(eval_dir):
        m = re.match(r"target_(\d+)_(\d+)", entry)
        if m:
            npz = os.path.join(eval_dir, entry, "trajectory.npz")
            if os.path.isfile(npz):
                h = float(f"{m.group(1)}.{m.group(2)}")
                targets.append((h, npz))
    return sorted(targets, key=lambda x: x[0])


def plot_timeseries(eval_dir: str, out_path: str, max_steps: int = 500) -> None:
    """Create multi-panel time series figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    targets = discover_targets(eval_dir)
    n_targets = len(targets)
    if n_targets == 0:
        print(f"No trajectories found in {eval_dir}")
        return

    fig, axes = plt.subplots(n_targets, 1, figsize=(14, 3 * n_targets),
                              sharex=True, squeeze=False)

    for i, (target_h, npz_path) in enumerate(targets):
        ax = axes[i, 0]
        d = np.load(npz_path)

        gt = d["gt"]
        ekf = d["ekf"]
        phase = d["phase"]
        det_steps = d["det_steps"]
        anchored = d["anchored_step"]
        ball_h = d["ball_h"]

        n = min(len(gt), max_steps)
        t = np.arange(n) * 0.02  # 50Hz → seconds

        # Ball height above ground (z-coordinate)
        gt_z = gt[:n, 2]
        ekf_z = ekf[:n, 2]

        # Phase background coloring
        for step in range(n - 1):
            ph = int(phase[step])
            ax.axvspan(t[step], t[step + 1], alpha=0.3,
                       color=PHASE_COLORS.get(ph, "#FFFFFF"), linewidth=0)

        # Ground truth and EKF
        ax.plot(t, gt_z, color="#2166AC", linewidth=1.0, alpha=0.9, label="GT ball z")
        ax.plot(t, ekf_z, color="#B2182B", linewidth=0.8, alpha=0.7, label="EKF ball z",
                linestyle="--")

        # Target height line (approximate: paddle height ~0.49m + target)
        # ball_h is height above paddle, gt_z is absolute → paddle_z ≈ gt_z - ball_h
        if len(ball_h) > 10:
            paddle_z = gt_z[:n] - ball_h[:n].astype(np.float32)
            median_paddle_z = np.median(paddle_z[paddle_z > 0.3])
            target_abs = median_paddle_z + target_h
        else:
            target_abs = 0.49 + target_h
        ax.axhline(target_abs, color="#4DAF4A", linewidth=1.5, linestyle=":",
                    alpha=0.8, label=f"target ({target_h}m)")

        # Detection events
        valid_dets = det_steps[det_steps < n]
        if len(valid_dets) > 0:
            ax.scatter(valid_dets * 0.02, gt_z[valid_dets.astype(int)],
                       color="#FF7F00", s=30, zorder=5, marker="v",
                       label=f"detections ({len(valid_dets)})")

        # Anchor indicator (small dots on bottom)
        anchor_mask = anchored[:n] > 0
        ax.scatter(t[anchor_mask], np.full(anchor_mask.sum(), gt_z.min() - 0.02),
                   color="#984EA3", s=2, alpha=0.3, marker="|")

        ax.set_ylabel(f"z (m)\ntgt={target_h}m", fontsize=9)
        ax.set_ylim(max(0, gt_z.min() - 0.05), max(gt_z.max(), target_abs) + 0.1)
        ax.legend(fontsize=7, loc="upper right", ncol=4)
        ax.grid(True, alpha=0.2)

        # Stats annotation
        flight_frac = ((phase[:n] == 1) | (phase[:n] == 2)).sum() / n * 100
        anchor_frac = anchor_mask.sum() / n * 100
        n_det = len(valid_dets)
        ax.text(0.01, 0.95, f"flight: {flight_frac:.1f}%  anchor: {anchor_frac:.0f}%  det: {n_det}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    axes[-1, 0].set_xlabel("Time (s)", fontsize=10)

    # Phase legend at top
    legend_elements = [
        Patch(facecolor=PHASE_COLORS[0], alpha=0.5, label="Contact"),
        Patch(facecolor=PHASE_COLORS[1], alpha=0.5, label="Ascending"),
        Patch(facecolor=PHASE_COLORS[2], alpha=0.5, label="Descending"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, 1.0))

    label = os.path.basename(eval_dir)
    fig.suptitle(f"Ball Height Time Series — {label}\n(first {max_steps} steps)",
                 fontsize=12, y=1.03)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True, help="Eval directory with target_X_XX/trajectory.npz")
    parser.add_argument("--out", required=True, help="Output figure path")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps to show (default 500)")
    args = parser.parse_args()
    plot_timeseries(args.eval_dir, args.out, args.max_steps)


if __name__ == "__main__":
    main()
