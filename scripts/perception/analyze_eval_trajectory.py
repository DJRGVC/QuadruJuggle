#!/usr/bin/env python3
"""Analyze camera eval trajectory from .npz files (stdout-independent).

Reads trajectory.npz produced by demo_camera_ekf.py and computes all metrics
directly from the saved arrays. This is the robust analysis path — it does NOT
depend on captured stdout text, so it works even when the demo was run as a
background process.

Produces:
  1. Text summary (detection rate, RMSE, EKF vs raw comparison)
  2. Height-binned RMSE comparison table
  3. 3-panel publication figure: (a) Z trajectory, (b) RMSE by height, (c) det rate

Can compare two trajectory.npz files (e.g. oracle vs d435i eval) with --compare.

Usage:
    # Single trajectory analysis:
    python scripts/perception/analyze_eval_trajectory.py \
        --npz source/.../perception/debug/demo/trajectory.npz

    # Compare two trajectories:
    python scripts/perception/analyze_eval_trajectory.py \
        --npz logs/perception/oracle_run/trajectory.npz \
        --compare logs/perception/d435i_run/trajectory.npz \
        --labels "Oracle policy" "D435i policy" \
        --out images/perception/oracle_vs_d435i_comparison.png
"""

import argparse
import os
import sys

import numpy as np


# Approximate paddle height in world frame (trunk ~0.40m + paddle offset 0.07m)
_PADDLE_Z_DEFAULT = 0.47


def load_npz(path: str) -> dict:
    """Load trajectory.npz and return as a plain dict."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No trajectory.npz at {path}")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def compute_overall_metrics(traj: dict) -> dict:
    """Compute aggregate metrics from a trajectory."""
    gt = traj["gt"]       # (T, 3) ground truth positions
    ekf = traj["ekf"]     # (T, 3) EKF estimates
    det = traj["det"]     # (D, 3) raw detections
    det_steps = traj["det_steps"].astype(int)  # (D,)
    total_steps = gt.shape[0]
    n_det = det.shape[0]

    result = {
        "total_steps": total_steps,
        "n_detections": n_det,
        "det_rate_pct": n_det / max(1, total_steps) * 100,
    }

    # EKF RMSE (over all steps)
    ekf_err = np.linalg.norm(ekf - gt, axis=1)
    result["ekf_rmse_mm"] = np.sqrt(np.mean(ekf_err ** 2)) * 1000
    result["ekf_rmse_std_mm"] = np.std(ekf_err) * 1000

    # Detection RMSE (only at detected steps)
    if n_det > 0 and len(det_steps) > 0:
        det_gt = gt[det_steps]
        det_err = np.linalg.norm(det - det_gt, axis=1)
        result["det_rmse_mm"] = np.sqrt(np.mean(det_err ** 2)) * 1000
        result["det_rmse_std_mm"] = np.std(det_err) * 1000
    else:
        result["det_rmse_mm"] = float("nan")
        result["det_rmse_std_mm"] = float("nan")

    return result


def compute_height_binned(
    traj: dict,
    paddle_z: float = _PADDLE_Z_DEFAULT,
    bin_width: float = 0.10,
    max_height: float = 0.80,
) -> dict:
    """Compute RMSE and detection rate binned by GT height above paddle."""
    gt = traj["gt"]
    ekf = traj["ekf"]
    det = traj["det"]
    det_steps = traj["det_steps"].astype(int)

    gt_h = gt[:, 2] - paddle_z
    bins = np.arange(0.0, max_height + bin_width, bin_width)
    n_bins = len(bins) - 1

    ekf_rmse = np.full(n_bins, np.nan)
    det_rmse = np.full(n_bins, np.nan)
    det_rate = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (gt_h >= lo) & (gt_h < hi)
        n = mask.sum()
        count[i] = n
        if n == 0:
            continue

        ekf_err = np.linalg.norm(ekf[mask] - gt[mask], axis=1)
        ekf_rmse[i] = np.sqrt(np.mean(ekf_err ** 2)) * 1000  # mm

        if len(det) > 0:
            det_gt_h = gt[det_steps, 2] - paddle_z
            det_mask = (det_gt_h >= lo) & (det_gt_h < hi)
            n_det = det_mask.sum()
            det_rate[i] = n_det / n * 100  # %
            if n_det > 0:
                d_err = np.linalg.norm(det[det_mask] - gt[det_steps[det_mask]], axis=1)
                det_rmse[i] = np.sqrt(np.mean(d_err ** 2)) * 1000
        else:
            det_rate[i] = 0.0

    return {
        "bins": bins,
        "centres": (bins[:-1] + bins[1:]) / 2,
        "ekf_rmse": ekf_rmse,
        "det_rmse": det_rmse,
        "det_rate": det_rate,
        "count": count,
    }


def compute_phase_metrics(
    traj: dict,
    paddle_z: float = _PADDLE_Z_DEFAULT,
    contact_threshold: float = 0.025,
) -> dict:
    """Compute metrics split by flight phase: ascending, descending, contact.

    Phases:
        contact:    ball height above paddle < contact_threshold
        ascending:  vz > 0 and not in contact
        descending: vz <= 0 and not in contact

    Velocity is estimated via finite differences of GT positions.
    """
    gt = traj["gt"]       # (T, 3)
    ekf = traj["ekf"]     # (T, 3)
    det = traj["det"]     # (D, 3)
    det_steps = traj["det_steps"].astype(int)
    dt = float(traj["dt"]) if "dt" in traj else 0.02
    T = gt.shape[0]

    # Compute velocity via finite differences (central where possible)
    vz = np.zeros(T)
    if T > 2:
        vz[1:-1] = (gt[2:, 2] - gt[:-2, 2]) / (2 * dt)
        vz[0] = (gt[1, 2] - gt[0, 2]) / dt
        vz[-1] = (gt[-1, 2] - gt[-2, 2]) / dt
    elif T == 2:
        vz[:] = (gt[1, 2] - gt[0, 2]) / dt

    gt_h = gt[:, 2] - paddle_z
    in_contact = gt_h < contact_threshold
    ascending = (~in_contact) & (vz > 0)
    descending = (~in_contact) & (vz <= 0)

    # Detection mask per step (True if a detection exists at that step)
    det_at_step = np.zeros(T, dtype=bool)
    valid_det_steps = det_steps[(det_steps >= 0) & (det_steps < T)]
    det_at_step[valid_det_steps] = True

    phases = {"ascending": ascending, "descending": descending, "contact": in_contact}
    result = {}

    for name, mask in phases.items():
        n = mask.sum()
        r = {"count": int(n)}
        if n == 0:
            r["ekf_rmse_mm"] = float("nan")
            r["det_rmse_mm"] = float("nan")
            r["det_rate_pct"] = float("nan")
            result[name] = r
            continue

        # EKF RMSE in this phase
        ekf_err = np.linalg.norm(ekf[mask] - gt[mask], axis=1)
        r["ekf_rmse_mm"] = float(np.sqrt(np.mean(ekf_err ** 2)) * 1000)

        # Detection rate and RMSE in this phase
        n_det_in_phase = det_at_step[mask].sum()
        r["det_rate_pct"] = float(n_det_in_phase / n * 100)

        # Detection RMSE: match detections that fall within this phase
        phase_det_mask = np.zeros(len(det_steps), dtype=bool)
        for di, ds in enumerate(det_steps):
            if 0 <= ds < T and mask[ds]:
                phase_det_mask[di] = True
        if phase_det_mask.any():
            d_err = np.linalg.norm(
                det[phase_det_mask] - gt[det_steps[phase_det_mask]], axis=1
            )
            r["det_rmse_mm"] = float(np.sqrt(np.mean(d_err ** 2)) * 1000)
        else:
            r["det_rmse_mm"] = float("nan")

        result[name] = r

    return result


def print_phase_table(phase: dict, label: str = "") -> None:
    """Print phase-separated metrics table."""
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}Phase          | Count | Det Rate | Det RMSE (mm) | EKF RMSE (mm)")
    print("-" * 72)
    for name in ("ascending", "descending", "contact"):
        p = phase[name]
        n = p["count"]
        if n == 0:
            print(f"  {name:<14s} | {n:5d} |    —     |      —        |      —")
            continue
        dr = p["det_rate_pct"]
        d = p["det_rmse_mm"]
        e = p["ekf_rmse_mm"]
        dr_s = f"{dr:5.1f}%" if not np.isnan(dr) else "  —  "
        d_s = f"{d:7.1f}" if not np.isnan(d) else "    —  "
        e_s = f"{e:7.1f}" if not np.isnan(e) else "    —  "
        print(f"  {name:<14s} | {n:5d} | {dr_s}  |   {d_s}     |   {e_s}")


def print_summary(metrics: dict, label: str = "") -> None:
    """Print a text summary of overall metrics."""
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}=== Camera Eval Summary ===")
    print(f"  Steps: {metrics['total_steps']}")
    print(f"  Detections: {metrics['n_detections']}/{metrics['total_steps']} "
          f"({metrics['det_rate_pct']:.1f}%)")
    print(f"  EKF RMSE: {metrics['ekf_rmse_mm']:.1f} ± {metrics['ekf_rmse_std_mm']:.1f} mm")
    if not np.isnan(metrics["det_rmse_mm"]):
        print(f"  Det RMSE: {metrics['det_rmse_mm']:.1f} ± {metrics['det_rmse_std_mm']:.1f} mm")
    else:
        print("  Det RMSE: — (no detections)")


def print_height_table(hb: dict, label: str = "") -> None:
    """Print height-binned table."""
    prefix = f"[{label}] " if label else ""
    print(f"\n{prefix}Height bin (mm) | Count | Det Rate | Det RMSE (mm) | EKF RMSE (mm) | Winner")
    print("-" * 80)
    for i in range(len(hb["centres"])):
        lo = hb["bins"][i] * 1000
        hi = hb["bins"][i + 1] * 1000
        n = hb["count"][i]
        if n == 0:
            print(f"  {lo:5.0f}-{hi:5.0f}   |   {n:4d} |    —     |      —        |      —        |  —")
            continue
        dr = hb["det_rate"][i]
        d = hb["det_rmse"][i]
        e = hb["ekf_rmse"][i]
        dr_s = f"{dr:5.1f}%" if not np.isnan(dr) else "  —  "
        d_s = f"{d:7.1f}" if not np.isnan(d) else "    —  "
        e_s = f"{e:7.1f}" if not np.isnan(e) else "    —  "
        if np.isnan(d) or np.isnan(e):
            w = "—"
        elif e < d:
            w = "EKF"
        elif d < e:
            w = "Raw"
        else:
            w = "Tie"
        print(f"  {lo:5.0f}-{hi:5.0f}   |   {n:4d} | {dr_s}  |   {d_s}     |   {e_s}     | {w}")


def plot_single(traj: dict, hb: dict, metrics: dict, out_path: str, label: str = ""):
    """3-panel figure for a single trajectory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(10, 9),
                              gridspec_kw={"height_ratios": [2.5, 1.5, 1]})
    title_suffix = f" — {label}" if label else ""

    # Panel 1: Z trajectory over time
    ax = axes[0]
    gt = traj["gt"]
    ekf = traj["ekf"]
    det = traj["det"]
    det_steps = traj["det_steps"].astype(int)
    dt = float(traj["dt"]) if "dt" in traj else 0.02
    t = np.arange(gt.shape[0]) * dt

    ax.plot(t, gt[:, 2], "k-", lw=1.0, alpha=0.7, label="Ground truth")
    ax.plot(t, ekf[:, 2], "b-", lw=0.8, alpha=0.8, label="EKF estimate")
    if len(det) > 0:
        ax.scatter(det_steps * dt, det[:, 2], s=6, c="coral", alpha=0.5,
                   label="Raw detection", zorder=3)
    ax.set_ylabel("Ball Z (m)")
    ax.set_title(f"Camera Pipeline Eval{title_suffix}")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 2: RMSE by height
    ax = axes[1]
    centres = hb["centres"] * 1000
    valid = hb["count"] > 0
    w = (centres[1] - centres[0]) * 0.35 if len(centres) > 1 else 20
    ekf_valid = valid & ~np.isnan(hb["ekf_rmse"])
    det_valid = valid & ~np.isnan(hb["det_rmse"])
    ax.bar(centres[ekf_valid] - w / 2, hb["ekf_rmse"][ekf_valid], w,
           color="steelblue", alpha=0.85, label="EKF", edgecolor="white")
    ax.bar(centres[det_valid] + w / 2, hb["det_rmse"][det_valid], w,
           color="coral", alpha=0.85, label="Raw", edgecolor="white")
    ax.set_ylabel("RMSE (mm)")
    ax.set_xlabel("Height above paddle (mm)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Detection rate
    ax = axes[2]
    ax.bar(centres[valid], hb["det_rate"][valid], w * 2,
           color="green", alpha=0.6, edgecolor="white")
    ax.set_ylabel("Det rate (%)")
    ax.set_xlabel("Height above paddle (mm)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] Figure saved: {out_path}")


def plot_comparison(
    trajs: list[dict], hbs: list[dict], metrics_list: list[dict],
    labels: list[str], out_path: str,
):
    """Side-by-side comparison figure for two trajectories."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = [("steelblue", "royalblue"), ("coral", "tomato")]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9),
                              gridspec_kw={"height_ratios": [2.5, 1.5, 1]})

    # Panel 1: Z trajectories overlaid
    ax = axes[0]
    for i, (traj, label) in enumerate(zip(trajs, labels)):
        gt = traj["gt"]
        dt = float(traj["dt"]) if "dt" in traj else 0.02
        t = np.arange(gt.shape[0]) * dt
        c = colors[i % len(colors)]
        ax.plot(t, gt[:, 2], "-", lw=0.8, alpha=0.6, color=c[0], label=f"{label} GT")
        ekf = traj["ekf"]
        ax.plot(t, ekf[:, 2], "--", lw=0.7, alpha=0.5, color=c[1], label=f"{label} EKF")
    ax.set_ylabel("Ball Z (m)")
    ax.set_title("Camera Pipeline — Multi-Run Comparison")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: EKF RMSE comparison by height
    ax = axes[1]
    for i, (hb, label) in enumerate(zip(hbs, labels)):
        centres = hb["centres"] * 1000
        valid = (hb["count"] > 0) & ~np.isnan(hb["ekf_rmse"])
        n = len(labels)
        w = (centres[1] - centres[0]) * 0.35 / n if len(centres) > 1 else 20
        offset = (i - (n - 1) / 2) * w
        c = colors[i % len(colors)][0]
        ax.bar(centres[valid] + offset, hb["ekf_rmse"][valid], w,
               color=c, alpha=0.8, label=label, edgecolor="white")
    ax.set_ylabel("EKF RMSE (mm)")
    ax.set_xlabel("Height above paddle (mm)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 3: Detection rate comparison
    ax = axes[2]
    for i, (hb, label) in enumerate(zip(hbs, labels)):
        centres = hb["centres"] * 1000
        valid = (hb["count"] > 0) & ~np.isnan(hb["det_rate"])
        n = len(labels)
        w = (centres[1] - centres[0]) * 0.35 / n if len(centres) > 1 else 20
        offset = (i - (n - 1) / 2) * w
        c = colors[i % len(colors)][0]
        ax.bar(centres[valid] + offset, hb["det_rate"][valid], w,
               color=c, alpha=0.7, label=label, edgecolor="white")
    ax.set_ylabel("Det rate (%)")
    ax.set_xlabel("Height above paddle (mm)")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[analyze] Comparison figure saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze camera eval trajectory from .npz (stdout-independent)")
    parser.add_argument("--npz", type=str, required=True,
                        help="Path to trajectory.npz (primary)")
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to second trajectory.npz for comparison")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Labels for the runs (e.g. 'Oracle' 'D435i')")
    parser.add_argument("--out", type=str, default=None,
                        help="Output figure path (default: alongside npz)")
    parser.add_argument("--paddle-z", type=float, default=_PADDLE_Z_DEFAULT,
                        help="Paddle z in world frame (m)")
    parser.add_argument("--bin-width", type=float, default=0.10,
                        help="Height bin width (m)")
    parser.add_argument("--max-height", type=float, default=0.80,
                        help="Max height above paddle to analyze (m)")
    parser.add_argument("--quarto-copy", action="store_true",
                        help="Also copy figure to images/perception/")
    args = parser.parse_args()

    traj1 = load_npz(args.npz)
    m1 = compute_overall_metrics(traj1)
    hb1 = compute_height_binned(traj1, args.paddle_z, args.bin_width, args.max_height)

    ph1 = compute_phase_metrics(traj1, args.paddle_z)

    if args.compare:
        # Comparison mode
        traj2 = load_npz(args.compare)
        m2 = compute_overall_metrics(traj2)
        hb2 = compute_height_binned(traj2, args.paddle_z, args.bin_width, args.max_height)
        ph2 = compute_phase_metrics(traj2, args.paddle_z)

        labels = args.labels or ["Run A", "Run B"]
        if len(labels) < 2:
            labels = labels + [f"Run {chr(65 + i)}" for i in range(len(labels), 2)]

        print_summary(m1, labels[0])
        print_height_table(hb1, labels[0])
        print_phase_table(ph1, labels[0])
        print_summary(m2, labels[1])
        print_height_table(hb2, labels[1])
        print_phase_table(ph2, labels[1])

        out = args.out or os.path.join(os.path.dirname(args.npz), "comparison.png")
        plot_comparison([traj1, traj2], [hb1, hb2], [m1, m2], labels, out)
    else:
        # Single trajectory
        label = args.labels[0] if args.labels else ""
        print_summary(m1, label)
        print_height_table(hb1, label)
        print_phase_table(ph1, label)

        out = args.out or os.path.join(os.path.dirname(args.npz), "eval_analysis.png")
        plot_single(traj1, hb1, m1, out, label)

    # Quarto copy
    if args.quarto_copy and args.out:
        repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
        quarto_dst = os.path.join(repo_root, "images", "perception", os.path.basename(args.out))
        if os.path.isdir(os.path.dirname(quarto_dst)):
            import shutil
            shutil.copy2(args.out, quarto_dst)
            print(f"[analyze] Copied to Quarto: {quarto_dst}")


if __name__ == "__main__":
    main()
