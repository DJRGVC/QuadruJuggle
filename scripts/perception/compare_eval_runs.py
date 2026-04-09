#!/usr/bin/env python3
"""Compare multiple camera eval runs side-by-side.

Takes 2+ log files from demo_camera_ekf.py and produces:
  1. A text summary table comparing detection rate, RMSE, episode stats
  2. A comparison bar chart for Quarto

Usage:
    python scripts/perception/compare_eval_runs.py \
        --labels "D435i policy" "Oracle policy" \
        --logs logs/perception/d435i_eval.log logs/perception/oracle_eval.log \
        --out images/perception/oracle_vs_d435i_comparison.png

If trajectory .npz files exist alongside logs, also generates height-binned comparison.
"""

import argparse
import os
import sys

import numpy as np

# Reuse the existing log parser
sys.path.insert(0, os.path.dirname(__file__))
from parse_oracle_eval import parse_log


def compare_runs(logs: list[str], labels: list[str]) -> list[dict]:
    """Parse multiple logs and return list of result dicts."""
    results = []
    for log_path, label in zip(logs, labels):
        r = parse_log(log_path)
        r["label"] = label
        r["log_path"] = log_path
        results.append(r)
    return results


def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 80)
    print("CAMERA PIPELINE EVAL — MULTI-RUN COMPARISON")
    print("=" * 80)

    # Header
    header = f"{'Metric':<30}"
    for r in results:
        header += f" | {r['label']:>20}"
    print(header)
    print("-" * (30 + 23 * len(results)))

    # Detection rate
    row = f"{'Detection rate':<30}"
    for r in results:
        if "det_rate_pct" in r:
            row += f" | {r['det_rate_pct']:>18.1f}%"
        else:
            row += f" | {'—':>20}"
    print(row)

    # Detection RMSE
    row = f"{'Det RMSE (mm)':<30}"
    for r in results:
        if "det_rmse_mm" in r:
            row += f" | {r['det_rmse_mm']:>15.1f}±{r['det_rmse_std_mm']:.1f}"
        else:
            row += f" | {'—':>20}"
    print(row)

    # EKF RMSE
    row = f"{'EKF RMSE (mm)':<30}"
    for r in results:
        if "ekf_rmse_mm" in r:
            row += f" | {r['ekf_rmse_mm']:>15.1f}±{r['ekf_rmse_std_mm']:.1f}"
        else:
            row += f" | {'—':>20}"
    print(row)

    # Episodes
    row = f"{'Episodes':<30}"
    for r in results:
        if "episodes" in r:
            row += f" | {r['episodes']:>20d}"
        else:
            row += f" | {'—':>20}"
    print(row)

    # Timeout %
    row = f"{'Timeout %':<30}"
    for r in results:
        if "timeout_pct" in r:
            row += f" | {r['timeout_pct']:>18.1f}%"
        else:
            row += f" | {'—':>20}"
    print(row)

    # Noise mode
    row = f"{'Noise mode':<30}"
    for r in results:
        row += f" | {r.get('noise_mode', '—'):>20}"
    print(row)

    # Target height
    row = f"{'Target height (m)':<30}"
    for r in results:
        if "target_height_m" in r:
            row += f" | {r['target_height_m']:>20.2f}"
        else:
            row += f" | {'—':>20}"
    print(row)

    # Ball height from step logs
    for r in results:
        if "step_logs" in r:
            heights = [s["ball_h_m"] for s in r["step_logs"]]
            above_02 = sum(1 for h in heights if h > 0.20)
            r["ball_above_02_pct"] = 100.0 * above_02 / max(1, len(heights))
    row = f"{'Ball >0.2m above paddle %':<30}"
    for r in results:
        if "ball_above_02_pct" in r:
            row += f" | {r['ball_above_02_pct']:>18.1f}%"
        else:
            row += f" | {'—':>20}"
    print(row)

    print("=" * (30 + 23 * len(results)))


def plot_comparison(results: list[dict], out_path: str) -> None:
    """Generate a multi-panel comparison figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Metrics to compare
    metrics = [
        ("det_rate_pct", "Detection Rate (%)", "%", 0, 105),
        ("det_rmse_mm", "Detection RMSE (mm)", "mm", 0, None),
        ("ekf_rmse_mm", "EKF RMSE (mm)", "mm", 0, None),
        ("timeout_pct", "Timeout Rate (%)", "%", 0, 105),
    ]

    # Filter to metrics that have data in at least one run
    available = []
    for key, title, unit, ymin, ymax in metrics:
        if any(key in r for r in results):
            available.append((key, title, unit, ymin, ymax))

    if not available:
        print("[compare] No plottable metrics found.")
        return

    n_panels = len(available)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    x = np.arange(len(results))
    width = 0.6

    for ax, (key, title, unit, ymin, ymax) in zip(axes, available):
        vals = []
        errs = []
        for r in results:
            vals.append(r.get(key, 0))
            # Add error bars for RMSE metrics
            std_key = key.replace("_mm", "_std_mm")
            errs.append(r.get(std_key, 0))

        bars = ax.bar(x, vals, width, color=colors[:len(results)],
                       edgecolor="white", linewidth=1.5, alpha=0.85)

        # Error bars for RMSE
        if any(e > 0 for e in errs):
            ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor="black",
                       capsize=4, capthick=1.5)

        # Value labels on bars
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                       f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels([r["label"] for r in results], rotation=15, ha="right", fontsize=9)
        if ymin is not None:
            ax.set_ylim(bottom=ymin)
        if ymax is not None:
            ax.set_ylim(top=ymax)
        ax.grid(True, alpha=0.3, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Camera Pipeline Eval: Policy Comparison", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] Figure saved: {out_path}")


def plot_step_timeseries(results: list[dict], out_path: str) -> None:
    """Plot detection rate over time for each run (from step logs)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_step_logs = any("step_logs" in r for r in results)
    if not has_step_logs:
        return

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for i, r in enumerate(results):
        if "step_logs" not in r:
            continue
        steps = [s["step"] for s in r["step_logs"]]
        det_rates = [s["det_rate_pct"] for s in r["step_logs"]]
        ball_heights = [s["ball_h_m"] * 1000 for s in r["step_logs"]]  # mm

        ax1.plot(steps, det_rates, color=colors[i % len(colors)],
                label=r["label"], linewidth=1.5, alpha=0.8)
        ax2.plot(steps, ball_heights, color=colors[i % len(colors)],
                label=r["label"], linewidth=1.5, alpha=0.8)

    ax1.set_ylabel("Detection Rate (%)")
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=50, color="gray", linestyle="--", alpha=0.4, label="50%")

    ax2.set_ylabel("Ball Height Above Paddle (mm)")
    ax2.set_xlabel("Simulation Step")
    ax2.axhline(y=200, color="red", linestyle="--", alpha=0.4, label="Camera FOV floor")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Camera Pipeline Over Time", fontsize=12, fontweight="bold")
    plt.tight_layout()

    ts_path = out_path.replace(".png", "_timeseries.png")
    fig.savefig(ts_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[compare] Timeseries figure saved: {ts_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare camera eval runs")
    parser.add_argument("--logs", nargs="+", required=True,
                       help="Log files from demo_camera_ekf.py")
    parser.add_argument("--labels", nargs="+", required=True,
                       help="Labels for each run (same order as --logs)")
    parser.add_argument("--out", type=str,
                       default="images/perception/eval_comparison.png",
                       help="Output figure path")
    args = parser.parse_args()

    if len(args.logs) != len(args.labels):
        print("ERROR: --logs and --labels must have the same number of arguments")
        sys.exit(1)

    results = compare_runs(args.logs, args.labels)

    # Check if any results were parsed
    non_empty = [r for r in results if len(r) > 2]  # >2 = has more than label+path
    if not non_empty:
        print("ERROR: No results could be parsed from any log file.")
        print("(Files may be empty — GPU runs haven't completed yet.)")
        sys.exit(1)

    print_comparison_table(results)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plot_comparison(results, args.out)
    plot_step_timeseries(results, args.out)


if __name__ == "__main__":
    main()
