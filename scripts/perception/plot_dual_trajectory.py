#!/usr/bin/env python3
"""Side-by-side oracle vs d435i trajectory comparison.

Produces a grid figure: rows = target heights, columns = [oracle, d435i].
Each cell shows ball height time series with phase coloring, EKF estimate,
detection events, and summary stats. Designed for Quarto experiment pages.

Usage:
    python scripts/perception/plot_dual_trajectory.py \
        --oracle-dir logs/perception/eval_stage_g_final_oracle \
        --d435i-dir logs/perception/eval_stage_g_final_d435i \
        --out images/perception/dual_trajectory_comparison.png \
        --max-steps 1000
"""

import argparse
import os
import re
import sys

import numpy as np

PHASE_NAMES = {0: "contact", 1: "ascending", 2: "descending"}
PHASE_COLORS = {0: "#E0E0E0", 1: "#A8D5BA", 2: "#F5C6AA"}


def discover_targets(eval_dir: str) -> dict[float, str]:
    """Find target_X_XX subdirs, return {target_height: npz_path}."""
    targets = {}
    if not os.path.isdir(eval_dir):
        return targets
    for entry in os.listdir(eval_dir):
        m = re.match(r"target_(\d+)_(\d+)", entry)
        if m:
            npz = os.path.join(eval_dir, entry, "trajectory.npz")
            if os.path.isfile(npz):
                h = float(f"{m.group(1)}.{m.group(2)}")
                targets[h] = npz
    return targets


def compute_stats(d: dict, n: int, target_h: float) -> dict:
    """Compute summary statistics from trajectory data."""
    phase = d["phase"][:n]
    det_steps = d["det_steps"]
    anchored = d["anchored_step"][:n]
    ball_h = d["ball_h"][:n]
    gt = d["gt"][:n]
    ekf = d["ekf"][:n]

    flight_mask = (phase == 1) | (phase == 2)
    flight_frac = flight_mask.sum() / n * 100
    anchor_frac = (anchored > 0).sum() / n * 100
    n_det = (det_steps < n).sum()
    det_rate = n_det / n * 100

    # EKF RMSE during flight only
    if flight_mask.sum() > 0:
        err = np.linalg.norm(gt[flight_mask] - ekf[flight_mask], axis=1)
        rmse_flight = np.sqrt(np.mean(err**2)) * 1000  # mm
    else:
        rmse_flight = float("nan")

    # Peak ball height above paddle
    peak_h = np.max(ball_h) if len(ball_h) > 0 else 0.0

    # Number of bounces (ascending phase transitions)
    phase_diff = np.diff(phase)
    bounces = np.sum(phase_diff == 1)  # contact→ascending

    return {
        "flight_pct": flight_frac,
        "anchor_pct": anchor_frac,
        "n_det": n_det,
        "det_rate": det_rate,
        "rmse_flight_mm": rmse_flight,
        "peak_h_m": peak_h,
        "bounces": bounces,
    }


def plot_cell(ax, npz_path: str, target_h: float, max_steps: int,
              mode_label: str, show_ylabel: bool = True) -> dict:
    """Plot a single trajectory cell. Returns stats dict."""
    d = np.load(npz_path)
    gt = d["gt"]
    ekf = d["ekf"]
    phase = d["phase"]
    det_steps = d["det_steps"]
    anchored = d["anchored_step"]
    ball_h = d["ball_h"]

    n = min(len(gt), max_steps)
    t = np.arange(n) * 0.02  # 50Hz

    gt_z = gt[:n, 2]
    ekf_z = ekf[:n, 2]

    # Phase background
    for step in range(n - 1):
        ph = int(phase[step])
        ax.axvspan(t[step], t[step + 1], alpha=0.25,
                   color=PHASE_COLORS.get(ph, "#FFFFFF"), linewidth=0)

    # Ground truth and EKF
    ax.plot(t, gt_z, color="#2166AC", linewidth=0.8, alpha=0.9, label="GT")
    ax.plot(t, ekf_z, color="#B2182B", linewidth=0.6, alpha=0.7,
            label="EKF", linestyle="--")

    # Target height line
    if len(ball_h) > 10:
        paddle_z = gt_z - ball_h[:n].astype(np.float32)
        valid = paddle_z[paddle_z > 0.3]
        median_paddle_z = np.median(valid) if len(valid) > 0 else 0.49
        target_abs = median_paddle_z + target_h
    else:
        target_abs = 0.49 + target_h
    ax.axhline(target_abs, color="#4DAF4A", linewidth=1.2, linestyle=":",
               alpha=0.7)

    # Detection events
    valid_dets = det_steps[det_steps < n]
    if len(valid_dets) > 0:
        ax.scatter(valid_dets * 0.02, gt_z[valid_dets.astype(int)],
                   color="#FF7F00", s=15, zorder=5, marker="v", alpha=0.8)

    stats = compute_stats(d, n, target_h)

    # Stats annotation
    stat_text = (f"flight:{stats['flight_pct']:.0f}%  "
                 f"det:{stats['n_det']}  "
                 f"peak:{stats['peak_h_m']:.2f}m")
    ax.text(0.02, 0.95, stat_text, transform=ax.transAxes, fontsize=6,
            va="top", bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                                alpha=0.8))

    if show_ylabel:
        ax.set_ylabel(f"z (m)", fontsize=8)
    ax.set_ylim(max(0.3, gt_z.min() - 0.03), max(gt_z.max(), target_abs) + 0.08)
    ax.grid(True, alpha=0.15)
    ax.tick_params(labelsize=7)

    return stats


def plot_dual_comparison(oracle_dir: str, d435i_dir: str, out_path: str,
                         max_steps: int = 1000) -> None:
    """Create side-by-side oracle vs d435i trajectory comparison."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    oracle_targets = discover_targets(oracle_dir)
    d435i_targets = discover_targets(d435i_dir)

    # Use shared target set
    shared = sorted(set(oracle_targets.keys()) & set(d435i_targets.keys()))
    if not shared:
        print("No shared targets found between oracle and d435i dirs")
        return

    n_rows = len(shared)
    fig, axes = plt.subplots(n_rows, 2, figsize=(16, 2.5 * n_rows),
                              sharex=True, squeeze=False)

    all_stats = {"oracle": [], "d435i": []}

    for i, target_h in enumerate(shared):
        # Oracle column
        stats_o = plot_cell(axes[i, 0], oracle_targets[target_h], target_h,
                            max_steps, "Oracle", show_ylabel=True)
        all_stats["oracle"].append({"target": target_h, **stats_o})

        # D435i column
        stats_d = plot_cell(axes[i, 1], d435i_targets[target_h], target_h,
                            max_steps, "D435i", show_ylabel=False)
        all_stats["d435i"].append({"target": target_h, **stats_d})

        # Row label (target height)
        axes[i, 0].text(-0.12, 0.5, f"tgt\n{target_h}m",
                         transform=axes[i, 0].transAxes, fontsize=9,
                         ha="center", va="center", fontweight="bold")

    # Column headers
    axes[0, 0].set_title("Oracle (no perception noise)", fontsize=10,
                          fontweight="bold", color="#2166AC")
    axes[0, 1].set_title("D435i (perception noise injected)", fontsize=10,
                          fontweight="bold", color="#B2182B")

    # X-axis labels
    axes[-1, 0].set_xlabel("Time (s)", fontsize=9)
    axes[-1, 1].set_xlabel("Time (s)", fontsize=9)

    # Legend on first row
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#2166AC", linewidth=1, label="Ground truth"),
        Line2D([0], [0], color="#B2182B", linewidth=0.8, linestyle="--",
               label="EKF estimate"),
        Line2D([0], [0], color="#4DAF4A", linewidth=1, linestyle=":",
               label="Target height"),
        Line2D([0], [0], marker="v", color="#FF7F00", linewidth=0, markersize=5,
               label="Camera detection"),
        Patch(facecolor=PHASE_COLORS[0], alpha=0.4, label="Contact"),
        Patch(facecolor=PHASE_COLORS[1], alpha=0.4, label="Ascending"),
        Patch(facecolor=PHASE_COLORS[2], alpha=0.4, label="Descending"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=7,
               fontsize=8, bbox_to_anchor=(0.5, 1.01))

    fig.suptitle("Oracle vs D435i — Ball Trajectory Comparison (Stage G eval)",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

    # Print summary table
    print("\n" + "=" * 85)
    print("TRAJECTORY COMPARISON SUMMARY")
    print("=" * 85)
    print(f"{'Target':>8} | {'Oracle':^35} | {'D435i':^35}")
    print(f"{'':>8} | {'flight%':>8} {'det':>5} {'peak(m)':>8} {'bounces':>8} | "
          f"{'flight%':>8} {'det':>5} {'peak(m)':>8} {'bounces':>8}")
    print("-" * 85)
    for o, d in zip(all_stats["oracle"], all_stats["d435i"]):
        print(f"{o['target']:>7.2f}m | "
              f"{o['flight_pct']:>7.1f}% {o['n_det']:>5d} {o['peak_h_m']:>8.3f} {o['bounces']:>8d} | "
              f"{d['flight_pct']:>7.1f}% {d['n_det']:>5d} {d['peak_h_m']:>8.3f} {d['bounces']:>8d}")
    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--oracle-dir", required=True,
                        help="Oracle eval directory with target_X_XX/trajectory.npz")
    parser.add_argument("--d435i-dir", required=True,
                        help="D435i eval directory with target_X_XX/trajectory.npz")
    parser.add_argument("--out", required=True, help="Output figure path")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Max timesteps to show (default 1000)")
    args = parser.parse_args()
    plot_dual_comparison(args.oracle_dir, args.d435i_dir, args.out, args.max_steps)


if __name__ == "__main__":
    main()
