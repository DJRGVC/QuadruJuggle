#!/usr/bin/env python3
"""Compare two multi-target eval runs from run_perception_eval.sh.

Takes two eval directories (each containing target_X_XX/trajectory.npz) and
produces a dashboard figure comparing per-target metrics side-by-side.

Usage:
    python scripts/perception/compare_multi_target.py \
        --dir-a logs/perception/eval_stage_g_d435i \
        --dir-b logs/perception/eval_stage_g_starve10 \
        --labels "D435i (limit=50)" "D435i (limit=10)" \
        --out images/perception/stage_g_comparison.png

    # With Quarto copy:
    python scripts/perception/compare_multi_target.py \
        --dir-a logs/perception/eval_oracle \
        --dir-b logs/perception/eval_d435i \
        --labels Oracle D435i \
        --out images/perception/oracle_vs_d435i_multi.png
"""

import argparse
import os
import re
import sys

import numpy as np

# Reuse existing trajectory analysis
sys.path.insert(0, os.path.dirname(__file__))
from analyze_eval_trajectory import (
    compute_overall_metrics,
    compute_phase_metrics,
    load_npz,
)


def discover_targets(eval_dir: str) -> list[tuple[float, str]]:
    """Find target_X_XX subdirs and return sorted (target_height, npz_path)."""
    targets = []
    if not os.path.isdir(eval_dir):
        raise FileNotFoundError(f"Eval directory not found: {eval_dir}")
    for entry in os.listdir(eval_dir):
        m = re.match(r"target_(\d+)_(\d+)", entry)
        if m:
            npz = os.path.join(eval_dir, entry, "trajectory.npz")
            if os.path.isfile(npz):
                h = float(f"{m.group(1)}.{m.group(2)}")
                targets.append((h, npz))
    targets.sort(key=lambda x: x[0])
    return targets


def load_eval_dir(eval_dir: str) -> dict:
    """Load all targets from an eval directory into a dict."""
    targets = discover_targets(eval_dir)
    if not targets:
        raise ValueError(f"No target_X_XX/trajectory.npz found in {eval_dir}")
    result = {}
    for h, npz_path in targets:
        traj = load_npz(npz_path)
        metrics = compute_overall_metrics(traj)
        phase = compute_phase_metrics(traj)
        # Episode stats from trajectory
        gt = traj["gt"]
        total_steps = gt.shape[0]
        # Approximate episode count from done signals if available
        n_episodes = int(traj["n_episodes"]) if "n_episodes" in traj else -1
        # Flight fraction from phase metrics
        flight_steps = phase["ascending"]["count"] + phase["descending"]["count"]
        flight_frac = flight_steps / max(1, total_steps)
        # Peak height
        peak_z = float(gt[:, 2].max())
        result[h] = {
            "metrics": metrics,
            "phase": phase,
            "flight_frac": flight_frac,
            "peak_z": peak_z,
            "total_steps": total_steps,
            "n_episodes": n_episodes,
            "traj": traj,
        }
    return result


def print_comparison(data_a: dict, data_b: dict, label_a: str, label_b: str) -> None:
    """Print per-target comparison table."""
    all_targets = sorted(set(data_a.keys()) | set(data_b.keys()))

    print(f"\n{'='*90}")
    print(f"MULTI-TARGET COMPARISON: {label_a} vs {label_b}")
    print(f"{'='*90}")
    print(f"{'Target':>8} | {'Det%':>7} {'Det%':>7} | {'EKF(mm)':>8} {'EKF(mm)':>8} | "
          f"{'Flight%':>8} {'Flight%':>8} | {'Peak(m)':>8} {'Peak(m)':>8}")
    print(f"{'':>8} | {label_a[:7]:>7} {label_b[:7]:>7} | {label_a[:8]:>8} {label_b[:8]:>8} | "
          f"{label_a[:8]:>8} {label_b[:8]:>8} | {label_a[:8]:>8} {label_b[:8]:>8}")
    print("-" * 90)

    for t in all_targets:
        a = data_a.get(t)
        b = data_b.get(t)

        def fmt_det(d):
            return f"{d['metrics']['det_rate_pct']:6.1f}%" if d else "   —   "

        def fmt_ekf(d):
            return f"{d['metrics']['ekf_rmse_mm']:7.1f}" if d else "     —  "

        def fmt_flight(d):
            return f"{d['flight_frac']*100:7.1f}%" if d else "    —   "

        def fmt_peak(d):
            return f"{d['peak_z']:7.3f}" if d else "     —  "

        print(f"{t:8.2f} | {fmt_det(a)} {fmt_det(b)} | {fmt_ekf(a)} {fmt_ekf(b)} | "
              f"{fmt_flight(a)} {fmt_flight(b)} | {fmt_peak(a)} {fmt_peak(b)}")

    print(f"{'='*90}")


def plot_dashboard(
    data_a: dict, data_b: dict, label_a: str, label_b: str, out_path: str
) -> None:
    """4-panel dashboard: det rate, EKF RMSE, flight fraction, peak height."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_targets = sorted(set(data_a.keys()) | set(data_b.keys()))
    x = np.arange(len(all_targets))
    width = 0.35
    colors = ("#4C72B0", "#DD8452")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Helper to extract per-target values
    def extract(data, key_fn):
        return [key_fn(data.get(t)) if t in data else np.nan for t in all_targets]

    panels = [
        (axes[0, 0], "Detection Rate (%)",
         lambda d: d["metrics"]["det_rate_pct"] if d else np.nan, (0, None)),
        (axes[0, 1], "EKF RMSE (mm)",
         lambda d: d["metrics"]["ekf_rmse_mm"] if d else np.nan, (0, None)),
        (axes[1, 0], "Flight Fraction (%)",
         lambda d: d["flight_frac"] * 100 if d else np.nan, (0, None)),
        (axes[1, 1], "Peak Ball Height (m)",
         lambda d: d["peak_z"] if d else np.nan, (0, None)),
    ]

    for ax, title, key_fn, (ymin, ymax) in panels:
        vals_a = extract(data_a, key_fn)
        vals_b = extract(data_b, key_fn)

        mask_a = ~np.isnan(vals_a)
        mask_b = ~np.isnan(vals_b)

        bars_a = ax.bar(x[mask_a] - width / 2, np.array(vals_a)[mask_a], width,
                        color=colors[0], alpha=0.85, label=label_a, edgecolor="white")
        bars_b = ax.bar(x[mask_b] + width / 2, np.array(vals_b)[mask_b], width,
                        color=colors[1], alpha=0.85, label=label_b, edgecolor="white")

        # Value labels (skip if bars overlap too much)
        for bars in [bars_a, bars_b]:
            for bar in bars:
                h = bar.get_height()
                if h > 0 and not np.isnan(h):
                    if h >= 100:
                        fmt = f"{h:.0f}"
                    elif h >= 1:
                        fmt = f"{h:.1f}"
                    else:
                        fmt = f"{h:.2f}"
                    ax.text(bar.get_x() + bar.get_width() / 2, h,
                            fmt, ha="center", va="bottom", fontsize=6.5)

        ax.set_ylabel(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t:.2f}" for t in all_targets], fontsize=9)
        ax.set_xlabel("Target Height (m)", fontsize=9)
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        if ymax is not None:
            ax.set_ylim(top=ymax)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Identity line on peak height panel
    ax_peak = axes[1, 1]
    ax_peak.plot(x, all_targets, "k--", alpha=0.4, lw=1, label="Target = Peak")
    ax_peak.legend(fontsize=8, loc="upper left")

    fig.suptitle(f"Multi-Target Eval: {label_a} vs {label_b}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] Dashboard saved: {out_path}")


def print_single(data: dict, label: str) -> None:
    """Print per-target metrics for a single eval directory."""
    print(f"\n{'='*70}")
    print(f"MULTI-TARGET EVAL: {label}")
    print(f"{'='*70}")
    print(f"{'Target':>8} | {'Det%':>7} | {'EKF RMSE':>10} | {'Flight%':>8} | {'Peak(m)':>8} | {'Steps':>8}")
    print("-" * 70)
    for t in sorted(data.keys()):
        d = data[t]
        m = d["metrics"]
        print(f"{t:8.2f} | {m['det_rate_pct']:6.1f}% | {m['ekf_rmse_mm']:8.1f}mm | "
              f"{d['flight_frac']*100:7.1f}% | {d['peak_z']:7.3f} | {d['total_steps']:8d}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two multi-target perception eval runs")
    parser.add_argument("--dir-a", required=True,
                        help="First eval directory (e.g. logs/perception/eval_oracle)")
    parser.add_argument("--dir-b", default=None,
                        help="Second eval directory for comparison (optional)")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Labels for the runs (1 for single, 2 for comparison)")
    parser.add_argument("--out", default="images/perception/multi_target_comparison.png",
                        help="Output figure path")
    args = parser.parse_args()

    data_a = load_eval_dir(args.dir_a)

    if args.dir_b:
        data_b = load_eval_dir(args.dir_b)
        if len(args.labels) < 2:
            args.labels.append("Run B")
        label_a, label_b = args.labels[0], args.labels[1]
        print_comparison(data_a, data_b, label_a, label_b)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        plot_dashboard(data_a, data_b, label_a, label_b, args.out)
    else:
        print_single(data_a, args.labels[0])
        print("\n(No --dir-b provided; skipping figure generation.)")


if __name__ == "__main__":
    main()
