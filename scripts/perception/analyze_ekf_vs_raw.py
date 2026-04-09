"""Analyze EKF vs raw camera detection accuracy binned by ball height.

Reads trajectory.npz files produced by demo_camera_ekf.py, bins data by GT
height above paddle, and produces a comparison figure showing:
  - Detection RMSE vs EKF RMSE at each height bin
  - Detection rate at each height bin

Usage:
    python scripts/perception/analyze_ekf_vs_raw.py [--data-dir DIR] [--out DIR]

If --data-dir is not specified, searches for the latest trajectory.npz in
the default demo output directory.
"""

import argparse
import os
import sys

import numpy as np


# Approximate paddle height in world frame (trunk ~0.40m + paddle offset 0.07m)
_PADDLE_Z = 0.47


def load_trajectory(data_dir: str) -> dict:
    """Load trajectory.npz from data_dir."""
    path = os.path.join(data_dir, "trajectory.npz")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No trajectory.npz found at {path}")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def compute_height_binned_metrics(
    traj: dict,
    height_bins: np.ndarray | None = None,
    paddle_z: float = _PADDLE_Z,
) -> dict:
    """Compute RMSE for EKF and raw detection, binned by GT height above paddle.

    Args:
        traj: Dict with keys gt, ekf, steps, det, det_steps, rmse_ekf, rmse_det.
        height_bins: Bin edges in metres above paddle. Default: 0 to 0.8 in 0.1m steps.
        paddle_z: Paddle z coordinate in world frame.

    Returns:
        Dict with bin_edges, bin_centres, ekf_rmse_per_bin, det_rmse_per_bin,
        det_rate_per_bin, count_per_bin.
    """
    if height_bins is None:
        height_bins = np.arange(0.0, 0.85, 0.10)

    gt = traj["gt"]               # (T, 3)
    ekf = traj["ekf"]             # (T, 3)
    det = traj["det"]             # (D, 3)
    det_steps = traj["det_steps"].astype(int)  # (D,)

    # Height above paddle for all steps
    gt_h = gt[:, 2] - paddle_z   # (T,)

    n_bins = len(height_bins) - 1
    ekf_rmse_per_bin = np.full(n_bins, np.nan)
    det_rmse_per_bin = np.full(n_bins, np.nan)
    det_rate_per_bin = np.full(n_bins, np.nan)
    count_per_bin = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = height_bins[i], height_bins[i + 1]

        # All steps in this height bin
        mask = (gt_h >= lo) & (gt_h < hi)
        count = mask.sum()
        count_per_bin[i] = count

        if count == 0:
            continue

        # EKF error for steps in this bin
        ekf_err = np.linalg.norm(ekf[mask] - gt[mask], axis=1)
        ekf_rmse_per_bin[i] = np.sqrt(np.mean(ekf_err ** 2))

        # Detection error for detections that fall in this height bin
        if len(det) > 0:
            det_gt_h = gt[det_steps, 2] - paddle_z  # height at detection steps
            det_mask = (det_gt_h >= lo) & (det_gt_h < hi)
            n_det = det_mask.sum()
            det_rate_per_bin[i] = n_det / count

            if n_det > 0:
                det_err = np.linalg.norm(det[det_mask] - gt[det_steps[det_mask]], axis=1)
                det_rmse_per_bin[i] = np.sqrt(np.mean(det_err ** 2))
        else:
            det_rate_per_bin[i] = 0.0

    bin_centres = (height_bins[:-1] + height_bins[1:]) / 2

    return {
        "bin_edges": height_bins,
        "bin_centres": bin_centres,
        "ekf_rmse": ekf_rmse_per_bin,
        "det_rmse": det_rmse_per_bin,
        "det_rate": det_rate_per_bin,
        "count": count_per_bin,
    }


def plot_comparison(results: dict, out_path: str):
    """Generate a 2-panel figure: RMSE comparison + detection rate by height."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    centres = results["bin_centres"] * 1000  # convert to mm for x-axis
    ekf_rmse = results["ekf_rmse"] * 1000   # mm
    det_rmse = results["det_rmse"] * 1000   # mm
    det_rate = results["det_rate"] * 100    # %
    count = results["count"]

    # Only plot bins with data
    valid = count > 0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    # Panel 1: RMSE comparison
    width = (centres[1] - centres[0]) * 0.35 if len(centres) > 1 else 20
    ekf_valid = valid & ~np.isnan(ekf_rmse)
    det_valid = valid & ~np.isnan(det_rmse)

    ax1.bar(centres[ekf_valid] - width / 2, ekf_rmse[ekf_valid], width,
            color="steelblue", alpha=0.85, label="EKF", edgecolor="white")
    ax1.bar(centres[det_valid] + width / 2, det_rmse[det_valid], width,
            color="coral", alpha=0.85, label="Raw Detection", edgecolor="white")

    ax1.set_ylabel("Position RMSE (mm)")
    ax1.set_title("EKF vs Raw Detection Accuracy by Ball Height Above Paddle")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    # Annotate sample counts
    for i in range(len(centres)):
        if count[i] > 0:
            y_max = max(
                ekf_rmse[i] if not np.isnan(ekf_rmse[i]) else 0,
                det_rmse[i] if not np.isnan(det_rmse[i]) else 0,
            )
            ax1.text(centres[i], y_max + 3, f"n={count[i]}", ha="center",
                     fontsize=7, color="gray")

    # Panel 2: Detection rate
    ax2.bar(centres[valid], det_rate[valid],
            width * 2, color="green", alpha=0.6, edgecolor="white")
    ax2.set_ylabel("Detection Rate (%)")
    ax2.set_xlabel("Height Above Paddle (mm)")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] Comparison figure saved: {out_path}")


def print_table(results: dict):
    """Print a text table of the results."""
    print("\nHeight bin (mm) | Count | Det Rate | Det RMSE (mm) | EKF RMSE (mm) | Winner")
    print("-" * 80)
    for i in range(len(results["bin_centres"])):
        lo = results["bin_edges"][i] * 1000
        hi = results["bin_edges"][i + 1] * 1000
        n = results["count"][i]
        dr = results["det_rate"][i]
        d_rmse = results["det_rmse"][i]
        e_rmse = results["ekf_rmse"][i]

        if n == 0:
            print(f"  {lo:5.0f}-{hi:5.0f}   |   {n:4d} |    —     |      —        |      —        |  —")
            continue

        dr_str = f"{dr * 100:5.1f}%" if not np.isnan(dr) else "  —  "
        d_str = f"{d_rmse * 1000:7.1f}" if not np.isnan(d_rmse) else "    —  "
        e_str = f"{e_rmse * 1000:7.1f}" if not np.isnan(e_rmse) else "    —  "

        if np.isnan(d_rmse) or np.isnan(e_rmse):
            winner = "—"
        elif e_rmse < d_rmse:
            winner = "EKF"
        elif d_rmse < e_rmse:
            winner = "Raw"
        else:
            winner = "Tie"

        print(f"  {lo:5.0f}-{hi:5.0f}   |   {n:4d} | {dr_str}  |   {d_str}     |   {e_str}     | {winner}")


def main():
    parser = argparse.ArgumentParser(description="Analyze EKF vs raw detection by height")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing trajectory.npz")
    parser.add_argument("--out", type=str, default=None,
                        help="Output directory for figure (default: same as data-dir)")
    parser.add_argument("--paddle-z", type=float, default=_PADDLE_Z,
                        help="Paddle z-coordinate in world frame (m)")
    parser.add_argument("--bin-width", type=float, default=0.10,
                        help="Height bin width in metres (default: 0.10)")
    parser.add_argument("--max-height", type=float, default=0.80,
                        help="Maximum height above paddle to analyze (m)")
    args = parser.parse_args()

    # Find data directory
    if args.data_dir is None:
        default_dir = os.path.normpath(os.path.join(
            os.path.dirname(__file__), "..", "..",
            "source", "go1_ball_balance", "go1_ball_balance", "perception", "debug", "demo"
        ))
        args.data_dir = default_dir

    traj = load_trajectory(args.data_dir)
    print(f"[analyze] Loaded trajectory: {traj['gt'].shape[0]} steps, "
          f"{traj['det'].shape[0]} detections")

    height_bins = np.arange(0.0, args.max_height + args.bin_width, args.bin_width)
    results = compute_height_binned_metrics(traj, height_bins, args.paddle_z)

    print_table(results)

    out_dir = args.out or args.data_dir
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "ekf_vs_raw_by_height.png")
    plot_comparison(results, fig_path)

    # Also copy to Quarto images
    repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    quarto_dst = os.path.join(repo_root, "images", "perception", "ekf_vs_raw_by_height.png")
    if os.path.isdir(os.path.dirname(quarto_dst)):
        import shutil
        shutil.copy2(fig_path, quarto_dst)
        print(f"[analyze] Copied to Quarto: {quarto_dst}")


if __name__ == "__main__":
    main()
