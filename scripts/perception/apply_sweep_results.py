#!/usr/bin/env python3
"""Read sweep_q_vel results JSON(s) and print recommended EKF config changes.

Supports merging multiple sweep files (e.g., high-range + low-range).

Usage:
    python scripts/perception/apply_sweep_results.py logs/perception/sweep_q_vel_*.json
    python scripts/perception/apply_sweep_results.py --plot  # auto-find + generate figure
"""

import json
import sys
import glob
import os


def load_and_merge(paths):
    """Load one or more sweep JSON files and merge by q_vel (dedup, sorted)."""
    all_results = {}
    config = None
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        if config is None:
            config = data.get("config", {})
        for r in data["sweep"]:
            qv = r["q_vel"]
            # Keep latest result for each q_vel (later file wins)
            all_results[qv] = r
    sorted_results = sorted(all_results.values(), key=lambda r: r["q_vel"])
    return sorted_results, config


def find_nis_crossing(results, target=3.0):
    """Find q_vel where flight NIS crosses target via linear interpolation."""
    for i in range(len(results) - 1):
        lo, hi = results[i], results[i + 1]
        nis_lo, nis_hi = lo["flight_nis"], hi["flight_nis"]
        # NIS decreases as q_vel increases (more process noise → lower NIS)
        if (nis_lo >= target and nis_hi < target) or (nis_lo <= target and nis_hi > target):
            # Linear interpolation on log(q_vel) scale
            import math
            log_lo = math.log(lo["q_vel"])
            log_hi = math.log(hi["q_vel"])
            frac = (target - nis_lo) / (nis_hi - nis_lo)
            log_cross = log_lo + frac * (log_hi - log_lo)
            return math.exp(log_cross)
    return None


def print_table(results, best_q_vel=None):
    """Print formatted table of sweep results."""
    print(f"{'q_vel':>10s}  {'NIS':>8s}  {'Flight':>8s}  {'Contact':>8s}  "
          f"{'EKF mm':>8s}  {'Raw mm':>8s}  {'Impr%':>7s}")
    print(f"{'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}")
    for r in results:
        marker = " ←" if best_q_vel and abs(r["q_vel"] - best_q_vel) < 0.001 else ""
        print(f"{r['q_vel']:10.4f}  {r['mean_nis']:8.3f}  {r['flight_nis']:8.3f}  "
              f"{r['contact_nis']:8.3f}  {r['ekf_rmse_mm']:8.2f}  {r['raw_rmse_mm']:8.2f}  "
              f"{r['improvement_pct']:7.1f}{marker}")


def make_figure(results, output_path, crossing_q_vel=None):
    """Generate publication-quality NIS + RMSE vs q_vel figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    q_vals = [r["q_vel"] for r in results]
    flight_nis = [r["flight_nis"] for r in results]
    contact_nis = [r["contact_nis"] for r in results]
    ekf_rmse = [r["ekf_rmse_mm"] for r in results]
    raw_rmse = [r["raw_rmse_mm"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: NIS vs q_vel (log scale)
    ax1.semilogx(q_vals, flight_nis, "o-", color="#2196F3", label="Flight NIS", linewidth=2, markersize=6)
    ax1.semilogx(q_vals, contact_nis, "s--", color="#FF9800", label="Contact NIS", linewidth=1.5, markersize=5)
    ax1.axhline(y=3.0, color="#E53935", linestyle=":", linewidth=1.5, label="Target NIS = 3.0")
    ax1.axhline(y=1.0, color="#888888", linestyle=":", linewidth=1.0, alpha=0.5, label="Under-confident")
    if crossing_q_vel is not None:
        ax1.axvline(x=crossing_q_vel, color="#4CAF50", linestyle="--", linewidth=1.5,
                     label=f"Crossing: q_vel={crossing_q_vel:.3f}")
    ax1.set_xlabel("q_vel (m/s/√s)", fontsize=12)
    ax1.set_ylabel("Normalized Innovation Squared (NIS)", fontsize=12)
    ax1.set_title("EKF Calibration: NIS vs Process Noise", fontsize=13)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Right panel: RMSE comparison
    ax2.semilogx(q_vals, ekf_rmse, "o-", color="#2196F3", label="EKF RMSE", linewidth=2, markersize=6)
    ax2.semilogx(q_vals, raw_rmse, "s--", color="#FF9800", label="Raw d435i RMSE", linewidth=1.5, markersize=5)
    if crossing_q_vel is not None:
        ax2.axvline(x=crossing_q_vel, color="#4CAF50", linestyle="--", linewidth=1.5,
                     label=f"Optimal q_vel={crossing_q_vel:.3f}")
    ax2.set_xlabel("q_vel (m/s/√s)", fontsize=12)
    ax2.set_ylabel("Position RMSE (mm)", fontsize=12)
    ax2.set_title("Position Accuracy: EKF vs Raw Noise", fontsize=13)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Contact-Aware EKF: q_vel Sweep Results (active juggling policy)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")
    plt.close(fig)


def apply_to_config(q_vel_value, config_path=None):
    """Patch BallEKFConfig.q_vel in ball_ekf.py with the recommended value."""
    import re
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "source/go1_ball_balance/go1_ball_balance/perception/ball_ekf.py"
        )
        config_path = os.path.normpath(config_path)

    with open(config_path) as f:
        content = f.read()

    pattern = r"(q_vel:\s*float\s*=\s*)[\d.]+(\s*#.*FREE-FLIGHT)"
    new_val = f"{q_vel_value:.4f}"
    new_content, n = re.subn(pattern, rf"\g<1>{new_val}\2", content)
    if n == 0:
        print(f"WARNING: Could not find q_vel pattern in {config_path}")
        return False

    with open(config_path, "w") as f:
        f.write(new_content)
    print(f"Updated {config_path}: q_vel = {new_val}")
    return True


def main():
    do_plot = "--plot" in sys.argv
    do_apply = "--apply" in sys.argv
    args = [a for a in sys.argv[1:] if a not in ("--plot", "--apply")]

    if not args:
        files = sorted(glob.glob("logs/perception/sweep_q_vel_*.json"))
        # Filter out known all-zeros files (pre-fix)
        files = [f for f in files if "fixed" in f or "low_range" in f or "combined" in f]
        if not files:
            # Fallback: any sweep file
            files = sorted(glob.glob("logs/perception/sweep_q_vel_*.json"))
        if not files:
            print("No sweep results found in logs/perception/")
            sys.exit(1)
    else:
        files = args

    results, config = load_and_merge(files)

    print(f"Merged {len(files)} sweep file(s): {', '.join(os.path.basename(f) for f in files)}")
    print(f"Total q_vel points: {len(results)}")
    print()

    print_table(results)

    # Find NIS=3.0 crossing
    crossing = find_nis_crossing(results, target=3.0)
    if crossing:
        print(f"\nNIS=3.0 crossing at q_vel ≈ {crossing:.4f}")
    else:
        # Check if all below or all above
        max_nis = max(r["flight_nis"] for r in results)
        min_nis = min(r["flight_nis"] for r in results)
        if max_nis < 3.0:
            print(f"\nNo NIS=3.0 crossing: max flight NIS = {max_nis:.3f} (all below 3.0)")
            print(f"  → Need even lower q_vel to find crossing")
        elif min_nis > 3.0:
            print(f"\nNo NIS=3.0 crossing: min flight NIS = {min_nis:.3f} (all above 3.0)")
            print(f"  → Need higher q_vel to find crossing")

    # Find best q_vel: closest to NIS=3.0 that still beats raw RMSE
    best = min(results, key=lambda r: abs(r["flight_nis"] - 3.0))
    print(f"\nClosest to NIS=3.0: q_vel={best['q_vel']:.4f}")
    print(f"  Flight NIS = {best['flight_nis']:.3f}")
    print(f"  EKF RMSE = {best['ekf_rmse_mm']:.1f}mm vs raw {best['raw_rmse_mm']:.1f}mm")
    ekf_beats = best["ekf_rmse_mm"] < best["raw_rmse_mm"]
    print(f"  EKF {'beats' if ekf_beats else 'LOSES to'} raw d435i")

    recommended_q_vel = crossing if crossing else best["q_vel"]
    print(f"\nRecommended BallEKFConfig changes:")
    print(f"  q_vel = {recommended_q_vel:.4f}  (was 0.40)")
    if config:
        print(f"  q_vel_contact = {config.get('q_vel_contact', 50.0)}")
        print(f"  q_vel_post_contact = {config.get('q_vel_post_contact', 20.0)}")
        print(f"  post_contact_steps = {config.get('post_contact_steps', 10)}")

    if do_plot:
        fig_path = "images/perception/q_vel_sweep_combined.png"
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)
        make_figure(results, fig_path, crossing_q_vel=crossing)

    if do_apply:
        print(f"\nApplying q_vel = {recommended_q_vel:.4f} to BallEKFConfig...")
        ok = apply_to_config(recommended_q_vel)
        if ok:
            print("Done. Run tests to verify: pytest scripts/perception/ -q")
        else:
            print("Failed to apply. Update ball_ekf.py manually.")


if __name__ == "__main__":
    main()
