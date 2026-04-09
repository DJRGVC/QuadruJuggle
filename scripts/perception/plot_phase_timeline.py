#!/usr/bin/env python3
"""Plot a phase-annotated EKF timeline from trajectory.npz.

Produces a 4-panel figure showing the full perception pipeline state over time:
  (a) Ball height above paddle with phase-colored background
  (b) 3D EKF position error with anchor/detection event markers
  (c) Camera scheduling: detection attempts vs skipped (stacked area)
  (d) EKF covariance proxy (RMSE accumulation rate)

Usage:
    python scripts/perception/plot_phase_timeline.py \
        --npz logs/perception/eval_run/trajectory.npz \
        --out images/perception/phase_timeline.png
"""

import argparse
import os
import sys

import numpy as np


# Phase constants (must match phase_tracker.py)
CONTACT = 0
ASCENDING = 1
DESCENDING = 2

_PHASE_COLORS = {
    CONTACT: "#2196F3",     # blue
    ASCENDING: "#4CAF50",   # green
    DESCENDING: "#FF9800",  # orange
}
_PHASE_LABELS = {
    CONTACT: "Contact",
    ASCENDING: "Ascending",
    DESCENDING: "Descending",
}


def load_npz(path: str) -> dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No trajectory at {path}")
    data = np.load(path)
    return {k: data[k] for k in data.files}


def compute_phase_spans(phase: np.ndarray) -> list:
    """Convert per-step phase array into (start, end, phase_id) spans."""
    if len(phase) == 0:
        return []
    spans = []
    cur_phase = int(phase[0])
    start = 0
    for i in range(1, len(phase)):
        p = int(phase[i])
        if p != cur_phase:
            spans.append((start, i, cur_phase))
            cur_phase = p
            start = i
    spans.append((start, len(phase), cur_phase))
    return spans


def compute_per_step_error(gt: np.ndarray, ekf: np.ndarray) -> np.ndarray:
    """3D position error in mm."""
    return np.linalg.norm(gt - ekf, axis=1) * 1000


def compute_bounce_events(phase: np.ndarray) -> np.ndarray:
    """Return step indices where ascending phase begins (bounce moments)."""
    if len(phase) < 2:
        return np.array([], dtype=int)
    transitions = []
    for i in range(1, len(phase)):
        if phase[i] == ASCENDING and phase[i - 1] != ASCENDING:
            transitions.append(i)
    return np.array(transitions, dtype=int)


def plot_phase_timeline(traj: dict, out_path: str, dt: float = 0.005,
                        paddle_z: float = 0.47, title: str = ""):
    """Generate 4-panel phase-annotated timeline figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    gt = traj["gt"]          # (T, 3)
    ekf = traj["ekf"]        # (T, 3)
    steps = traj["steps"]    # (T,)
    phase = traj.get("phase", np.zeros(len(gt), dtype=int))
    anchored = traj.get("anchored_step", np.zeros(len(gt), dtype=int))
    ball_h = traj.get("ball_h", gt[:, 2] - paddle_z)
    sched_active = traj.get("sched_active", np.ones(len(gt), dtype=int))
    det = traj.get("det", np.zeros((0, 3)))
    det_steps = traj.get("det_steps", np.array([]))

    t = steps * dt  # seconds
    T = len(t)

    error_mm = compute_per_step_error(gt, ekf)
    spans = compute_phase_spans(phase)
    bounces = compute_bounce_events(phase)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1, 1]})
    fig.suptitle(title or "Phase-Annotated EKF Timeline", fontsize=13, y=0.98)

    # --- Panel (a): Ball height with phase coloring ---
    ax = axes[0]
    for s, e, pid in spans:
        ax.axvspan(t[s], t[min(e - 1, T - 1)], alpha=0.15,
                   color=_PHASE_COLORS.get(pid, "#999"))
    ax.plot(t, ball_h * 1000, color="black", linewidth=0.8, label="Ball h (mm)")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    # Mark bounce events
    if len(bounces) > 0 and len(bounces) < 200:
        bounce_t = t[bounces] if max(bounces) < T else []
        bounce_h = ball_h[bounces] * 1000 if max(bounces) < T else []
        ax.scatter(bounce_t, bounce_h, marker="^", color="red", s=20,
                   zorder=5, label=f"Bounces ({len(bounces)})")
    ax.set_ylabel("Height above paddle (mm)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("(a) Ball height + phase classification", fontsize=10, loc="left")

    # --- Panel (b): EKF error with event markers ---
    ax = axes[1]
    ax.plot(t, error_mm, color="#E91E63", linewidth=0.6, alpha=0.8, label="EKF error")
    # Rolling mean
    if T > 50:
        kernel = np.ones(50) / 50
        smooth = np.convolve(error_mm, kernel, mode="same")
        ax.plot(t, smooth, color="#880E4F", linewidth=1.5, label="50-step avg")
    # Mark anchor fires
    anchor_mask = anchored.astype(bool)
    if anchor_mask.any():
        anchor_t = t[anchor_mask]
        anchor_err = error_mm[anchor_mask]
        ax.scatter(anchor_t, anchor_err, marker="|", color="#2196F3", s=15,
                   alpha=0.5, label=f"Anchor ({anchor_mask.sum()})")
    # Mark detections
    if len(det_steps) > 0:
        det_t = det_steps * dt
        valid = det_steps.astype(int)
        valid = valid[valid < T]
        if len(valid) > 0:
            det_err = error_mm[valid]
            ax.scatter(det_t[:len(valid)], det_err, marker=".", color="#4CAF50",
                       s=10, alpha=0.5, label=f"Detections ({len(det_steps)})")
    ax.set_ylabel("Position error (mm)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("(b) EKF error + anchor/detection events", fontsize=10, loc="left")

    # --- Panel (c): Camera scheduling ---
    ax = axes[2]
    if sched_active is not None and len(sched_active) == T:
        detected = sched_active.astype(float)
        skipped = 1.0 - detected
        ax.fill_between(t, 0, detected, color="#4CAF50", alpha=0.5, label="Detected")
        ax.fill_between(t, detected, 1, color="#9E9E9E", alpha=0.3, label="Skipped")
        pct_skip = skipped.mean() * 100
        ax.set_ylabel("Camera")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Skip", "Detect"])
        ax.legend(loc="upper right", fontsize=8,
                  title=f"{pct_skip:.0f}% skipped")
    else:
        ax.text(0.5, 0.5, "No scheduling data", transform=ax.transAxes,
                ha="center", va="center", color="gray")
    ax.set_title("(c) Camera scheduling", fontsize=10, loc="left")

    # --- Panel (d): Cumulative RMSE ---
    ax = axes[3]
    cum_sq = np.cumsum(error_mm ** 2)
    cum_count = np.arange(1, T + 1)
    cum_rmse = np.sqrt(cum_sq / cum_count)
    ax.plot(t, cum_rmse, color="#673AB7", linewidth=1.2)
    ax.set_ylabel("Cumulative RMSE (mm)")
    ax.set_xlabel("Time (s)")
    ax.set_title("(d) Cumulative RMSE", fontsize=10, loc="left")

    # Phase legend
    phase_patches = [Patch(facecolor=_PHASE_COLORS[p], alpha=0.3, label=_PHASE_LABELS[p])
                     for p in [CONTACT, ASCENDING, DESCENDING]]
    axes[0].legend(handles=phase_patches + axes[0].get_legend_handles_labels()[0][:2],
                   loc="upper right", fontsize=7, ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # Print phase summary
    if len(phase) > 0:
        for pid, label in _PHASE_LABELS.items():
            frac = (phase == pid).mean() * 100
            print(f"  {label}: {frac:.1f}%")
        print(f"  Bounces: {len(bounces)}")
        print(f"  Mean error: {error_mm.mean():.1f} mm")
        print(f"  Final cumulative RMSE: {cum_rmse[-1]:.1f} mm")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--npz", required=True, help="Path to trajectory.npz")
    parser.add_argument("--out", default="images/perception/phase_timeline.png",
                        help="Output figure path")
    parser.add_argument("--dt", type=float, default=0.005, help="Timestep (s)")
    parser.add_argument("--paddle-z", type=float, default=0.47, help="Paddle Z in world frame")
    parser.add_argument("--title", default="", help="Figure title")
    args = parser.parse_args()

    traj = load_npz(args.npz)
    plot_phase_timeline(traj, args.out, dt=args.dt, paddle_z=args.paddle_z, title=args.title)


if __name__ == "__main__":
    main()
