#!/usr/bin/env python3
"""Plot Stage G comparison: 4-variant × N-target grid.

Reads trajectory.npz files from a comparison directory produced by
run_stage_g_comparison.sh and generates a multi-panel figure.

Usage:
    python scripts/perception/plot_stage_comparison.py \
        --dir logs/perception/comparison_YYYYMMDD_HHMMSS \
        --out images/perception/stage_g_comparison.png
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


VARIANTS = ["d435i_anchor", "d435i_no_anchor", "oracle_anchor", "oracle_baseline"]
VARIANT_LABELS = {
    "d435i_anchor": "D435i + anchor",
    "d435i_no_anchor": "D435i (no anchor)",
    "oracle_anchor": "Oracle + anchor",
    "oracle_baseline": "Oracle baseline",
}
VARIANT_COLORS = {
    "d435i_anchor": "#2196F3",
    "d435i_no_anchor": "#90CAF9",
    "oracle_anchor": "#4CAF50",
    "oracle_baseline": "#A5D6A7",
}


def load_results(comparison_dir: str) -> dict:
    """Load all trajectory.npz results into a nested dict."""
    results = {}
    for variant in VARIANTS:
        vdir = Path(comparison_dir) / variant
        if not vdir.exists():
            continue
        results[variant] = {}
        for target_dir in sorted(vdir.glob("target_*")):
            npz = target_dir / "trajectory.npz"
            if not npz.exists():
                continue
            target = target_dir.name.replace("target_", "").replace("_", ".")
            d = np.load(str(npz))
            total = len(d["steps"])
            num_det = len(d["det_steps"])
            gt = d["gt"]
            ekf = d["ekf"]
            n = min(len(ekf), len(gt))
            rmse = np.sqrt(np.mean((ekf[:n] - gt[:n]) ** 2)) if n > 0 else float("nan")
            rmse_z = np.sqrt(np.mean((ekf[:n, 2] - gt[:n, 2]) ** 2)) if n > 0 else float("nan")
            results[variant][target] = {
                "det_rate": num_det / max(total, 1) * 100,
                "mean_h": gt[:, 2].mean(),
                "max_h": gt[:, 2].max(),
                "rmse_pos": rmse,
                "rmse_z": rmse_z,
                "total_steps": total,
            }
    return results


def plot_comparison(results: dict, out_path: str) -> None:
    """Generate 4-panel comparison figure."""
    targets = sorted(
        set(t for v in results.values() for t in v.keys()),
        key=float,
    )
    if not targets:
        print("No results to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Stage G Perception Pipeline Comparison", fontsize=14, fontweight="bold")

    metrics = [
        ("det_rate", "Detection rate (%)", axes[0, 0]),
        ("max_h", "Max ball height (m)", axes[0, 1]),
        ("rmse_pos", "Position RMSE (m)", axes[1, 0]),
        ("rmse_z", "Height RMSE (m)", axes[1, 1]),
    ]

    x = np.arange(len(targets))
    width = 0.18

    for metric_key, metric_label, ax in metrics:
        for i, variant in enumerate(VARIANTS):
            if variant not in results:
                continue
            values = [results[variant].get(t, {}).get(metric_key, 0) for t in targets]
            bars = ax.bar(
                x + i * width - 1.5 * width,
                values,
                width,
                label=VARIANT_LABELS[variant],
                color=VARIANT_COLORS[variant],
                edgecolor="white",
                linewidth=0.5,
            )
        ax.set_xlabel("Target height (m)")
        ax.set_ylabel(metric_label)
        ax.set_xticks(x)
        ax.set_xticklabels(targets)
        ax.legend(fontsize=7, loc="best")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Stage G comparison")
    parser.add_argument("--dir", required=True, help="Comparison directory")
    parser.add_argument(
        "--out",
        default="images/perception/stage_g_comparison.png",
        help="Output figure path",
    )
    args = parser.parse_args()

    results = load_results(args.dir)
    if not results:
        print(f"No results found in {args.dir}")
        return

    # Print text table
    print("\n=== Numeric summary ===")
    for variant in VARIANTS:
        if variant not in results:
            continue
        print(f"\n--- {VARIANT_LABELS[variant]} ---")
        print(f"{'Target':>8s}  {'Det%':>6s}  {'MaxH':>6s}  {'RMSE':>8s}  {'RMSE_z':>8s}")
        for t in sorted(results[variant], key=float):
            r = results[variant][t]
            print(
                f"{t:>8s}  {r['det_rate']:5.1f}%  {r['max_h']:6.3f}  "
                f"{r['rmse_pos']:8.4f}  {r['rmse_z']:8.4f}"
            )

    plot_comparison(results, args.out)


if __name__ == "__main__":
    main()
