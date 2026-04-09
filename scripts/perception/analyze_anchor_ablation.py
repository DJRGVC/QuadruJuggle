#!/usr/bin/env python3
"""Analyze paddle-anchor ablation: compare trajectory.npz with/without anchor.

Produces a 4-panel figure:
  (a) EKF position error over time (ON vs OFF)
  (b) Contact vs flight phase RMSE bar chart
  (c) Anchor fire rate and covariance trace over time
  (d) Cumulative RMSE divergence

Usage:
    python scripts/perception/analyze_anchor_ablation.py \
        --on  logs/perception/anchor_on_eval/trajectory.npz \
        --off logs/perception/anchor_off_eval/trajectory.npz \
        --out images/perception/anchor_ablation_detail.png
"""

import argparse
import os
import sys

import numpy as np


_PADDLE_Z_DEFAULT = 0.47
_CONTACT_THRESHOLD_M = 0.030  # 30mm above paddle = contact phase


def load_npz(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No trajectory at {path}")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def compute_per_step_error(traj: dict) -> np.ndarray:
    """Return per-step 3D position error in mm."""
    return np.linalg.norm(traj["ekf"] - traj["gt"], axis=1) * 1000


def phase_mask(traj: dict, paddle_z: float = _PADDLE_Z_DEFAULT,
               threshold: float = _CONTACT_THRESHOLD_M) -> tuple[np.ndarray, np.ndarray]:
    """Return (contact_mask, flight_mask) boolean arrays."""
    h = traj["gt"][:, 2] - paddle_z
    contact = h < threshold
    flight = ~contact
    return contact, flight


def phase_rmse(err_mm: np.ndarray, mask: np.ndarray) -> float:
    """RMSE over masked steps, or NaN if no steps."""
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean(err_mm[mask] ** 2)))


def cumulative_rmse(err_mm: np.ndarray) -> np.ndarray:
    """Cumulative RMSE up to each step."""
    cumsum = np.cumsum(err_mm ** 2)
    counts = np.arange(1, len(err_mm) + 1)
    return np.sqrt(cumsum / counts)


def print_ablation_summary(on: dict, off: dict, paddle_z: float):
    """Print text summary comparing anchor ON vs OFF."""
    err_on = compute_per_step_error(on)
    err_off = compute_per_step_error(off)
    c_on, f_on = phase_mask(on, paddle_z)
    c_off, f_off = phase_mask(off, paddle_z)

    print("\n=== Anchor Ablation Summary ===")
    print(f"  Steps:   ON={len(err_on)}, OFF={len(err_off)}")

    on_anchored = on.get("anchored_step", np.array([]))
    if len(on_anchored) > 0:
        n_anchored = int(on_anchored.sum())
        print(f"  Anchor fires: {n_anchored}/{len(on_anchored)} steps "
              f"({n_anchored / max(1, len(on_anchored)) * 100:.1f}%)")

    print(f"\n  {'Metric':<25s} | {'Anchor ON':>12s} | {'Anchor OFF':>12s} | {'Delta':>10s}")
    print("  " + "-" * 68)

    rows = [
        ("Overall RMSE (mm)", phase_rmse(err_on, np.ones(len(err_on), bool)),
         phase_rmse(err_off, np.ones(len(err_off), bool))),
        ("Contact RMSE (mm)", phase_rmse(err_on, c_on), phase_rmse(err_off, c_off)),
        ("Flight RMSE (mm)", phase_rmse(err_on, f_on), phase_rmse(err_off, f_off)),
        ("Contact steps", float(c_on.sum()), float(c_off.sum())),
        ("Flight steps", float(f_on.sum()), float(f_off.sum())),
    ]
    for label, v_on, v_off in rows:
        if np.isnan(v_on) or np.isnan(v_off):
            delta_s = "—"
        else:
            delta = v_on - v_off
            delta_s = f"{delta:+.1f}"
        v_on_s = f"{v_on:.1f}" if not np.isnan(v_on) else "—"
        v_off_s = f"{v_off:.1f}" if not np.isnan(v_off) else "—"
        print(f"  {label:<25s} | {v_on_s:>12s} | {v_off_s:>12s} | {delta_s:>10s}")


def plot_ablation(on: dict, off: dict, paddle_z: float, out_path: str):
    """4-panel publication figure comparing anchor ON vs OFF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    err_on = compute_per_step_error(on)
    err_off = compute_per_step_error(off)
    dt_on = float(on.get("dt", 0.02))
    dt_off = float(off.get("dt", 0.02))
    t_on = np.arange(len(err_on)) * dt_on
    t_off = np.arange(len(err_off)) * dt_off

    c_on, f_on = phase_mask(on, paddle_z)
    c_off, f_off = phase_mask(off, paddle_z)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # ─── Panel (a): Position error over time ───
    ax = axes[0, 0]
    # Smoothed with rolling window for readability
    win = min(50, len(err_on) // 10)
    if win > 1:
        kernel = np.ones(win) / win
        smooth_on = np.convolve(err_on, kernel, mode="same")
        smooth_off = np.convolve(err_off, kernel, mode="same")
    else:
        smooth_on, smooth_off = err_on, err_off
    ax.plot(t_on, smooth_on, color="steelblue", lw=1.2, alpha=0.9, label="Anchor ON")
    ax.plot(t_off, smooth_off, color="coral", lw=1.2, alpha=0.9, label="Anchor OFF")
    # Shade contact regions (from ON trajectory as reference)
    _shade_contact(ax, c_on, dt_on)
    ax.set_ylabel("Position error (mm)")
    ax.set_xlabel("Time (s)")
    ax.set_title("(a) EKF position error over time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ─── Panel (b): Phase RMSE bar chart ───
    ax = axes[0, 1]
    phases = ["Contact", "Flight", "Overall"]
    rmse_on = [
        phase_rmse(err_on, c_on),
        phase_rmse(err_on, f_on),
        phase_rmse(err_on, np.ones(len(err_on), bool)),
    ]
    rmse_off = [
        phase_rmse(err_off, c_off),
        phase_rmse(err_off, f_off),
        phase_rmse(err_off, np.ones(len(err_off), bool)),
    ]
    x = np.arange(len(phases))
    w = 0.35
    bars_on = ax.bar(x - w / 2, rmse_on, w, color="steelblue", alpha=0.85,
                     label="Anchor ON", edgecolor="white")
    bars_off = ax.bar(x + w / 2, rmse_off, w, color="coral", alpha=0.85,
                      label="Anchor OFF", edgecolor="white")
    # Value labels
    for bars in (bars_on, bars_off):
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 2,
                        f"{h:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel("RMSE (mm)")
    ax.set_title("(b) Phase-separated RMSE")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # ─── Panel (c): Ball height + anchor fires ───
    ax = axes[1, 0]
    gt_h_on = (on["gt"][:, 2] - paddle_z) * 1000  # mm
    gt_h_off = (off["gt"][:, 2] - paddle_z) * 1000
    ax.plot(t_on, gt_h_on, color="steelblue", lw=0.8, alpha=0.7, label="Ball h (ON)")
    ax.plot(t_off, gt_h_off, color="coral", lw=0.8, alpha=0.7, label="Ball h (OFF)")
    ax.axhline(_CONTACT_THRESHOLD_M * 1000, color="gray", ls="--", lw=0.7, alpha=0.5,
               label="Contact threshold")
    # Mark anchor fires
    anchored = on.get("anchored_step", np.array([]))
    if len(anchored) > 0 and len(anchored) == len(t_on):
        anchor_mask = anchored.astype(bool)
        ax.scatter(t_on[anchor_mask], gt_h_on[anchor_mask], s=3, c="green",
                   alpha=0.4, zorder=3, label="Anchor fired")
    ax.set_ylabel("Height above paddle (mm)")
    ax.set_xlabel("Time (s)")
    ax.set_title("(c) Ball height + anchor events")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ─── Panel (d): Cumulative RMSE ───
    ax = axes[1, 1]
    cum_on = cumulative_rmse(err_on)
    cum_off = cumulative_rmse(err_off)
    ax.plot(t_on, cum_on, color="steelblue", lw=1.5, label="Anchor ON")
    ax.plot(t_off, cum_off, color="coral", lw=1.5, label="Anchor OFF")
    ax.set_ylabel("Cumulative RMSE (mm)")
    ax.set_xlabel("Time (s)")
    ax.set_title("(d) Cumulative RMSE divergence")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Paddle Anchor Ablation — EKF Accuracy", fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[anchor_ablation] Figure saved: {out_path}")


def _shade_contact(ax, contact_mask: np.ndarray, dt: float):
    """Shade contact regions on a time-series axis."""
    # Find contiguous contact blocks
    diff = np.diff(contact_mask.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if contact_mask[0]:
        starts = np.concatenate([[0], starts])
    if contact_mask[-1]:
        ends = np.concatenate([ends, [len(contact_mask)]])
    for s, e in zip(starts, ends):
        ax.axvspan(s * dt, e * dt, color="lightyellow", alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Analyze anchor ablation trajectories")
    parser.add_argument("--on", required=True, help="trajectory.npz with anchor ON")
    parser.add_argument("--off", required=True, help="trajectory.npz with anchor OFF")
    parser.add_argument("--out", default="images/perception/anchor_ablation_detail.png",
                        help="Output figure path")
    parser.add_argument("--paddle-z", type=float, default=_PADDLE_Z_DEFAULT,
                        help="Paddle Z in world frame (m)")
    args = parser.parse_args()

    traj_on = load_npz(args.on)
    traj_off = load_npz(args.off)

    print_ablation_summary(traj_on, traj_off, args.paddle_z)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plot_ablation(traj_on, traj_off, args.paddle_z, args.out)


if __name__ == "__main__":
    main()
