#!/usr/bin/env python3
"""Plot training curves from TensorBoard event files.

Unlike plot_training_curves.py (which parses RSL-RL text logs), this reads
the TensorBoard event files directly, enabling live monitoring of active runs.

Usage:
    python scripts/perception/plot_training_tb.py \
        --logdir logs/rsl_rl/go1_ball_juggle_hier/2026-04-09_07-38-27/ \
        --out images/perception/stage_g_training_live.png

    # Compare two runs:
    python scripts/perception/plot_training_tb.py \
        --logdir dir1/ dir2/ --labels "Run A" "Run B" \
        --out images/perception/training_comparison.png

    # Also dump CSV summary:
    python scripts/perception/plot_training_tb.py \
        --logdir dir1/ --out fig.png --csv summary.csv
"""

import argparse
import os
import sys

import numpy as np


def load_tb_scalars(logdir: str, tags: list[str]) -> dict[str, list[tuple[int, float]]]:
    """Load scalar events from a TensorBoard logdir."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(logdir)
    try:
        ea.Reload()
    except Exception:
        return {tag: [] for tag in tags}

    available = set(ea.Tags().get("scalars", []))
    result = {}
    for tag in tags:
        if tag in available:
            events = ea.Scalars(tag)
            result[tag] = [(e.step, e.value) for e in events]
        else:
            result[tag] = []
    return result


def smooth(arr: np.ndarray, window: int = 20) -> np.ndarray:
    """Simple moving average."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


TAGS = [
    "Episode_Termination/time_out",
    "Train/mean_episode_length",
    "Train/mean_reward",
    "Episode_Reward/ball_apex_height",
    "Policy/mean_noise_std",
    "Curriculum/es_no_improve",
    "Episode_Reward/ball_release_vel",
    "Episode_Reward/ball_xy_dist",
]


def plot_training_tb(
    logdirs: list[str],
    labels: list[str],
    out_path: str,
    window: int = 20,
    threshold: float = 0.80,
    csv_path: str | None = None,
):
    """Plot multi-panel training curves from TensorBoard event files."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_data = []
    for logdir in logdirs:
        all_data.append(load_tb_scalars(logdir, TAGS))

    colors = plt.cm.tab10.colors

    panels = [
        ("Episode_Termination/time_out", "Timeout %", "Episode Survival", 100, True),
        ("Train/mean_episode_length", "Mean Episode Length", "Episode Length", 1, False),
        ("Episode_Reward/ball_apex_height", "Apex Height Reward", "Ball Apex Tracking", 1, False),
        ("Policy/mean_noise_std", "Action Noise Std", "Policy Exploration", 1, False),
        ("Curriculum/es_no_improve", "ES No-Improve Counter", "Early Stopping Progress", 1, False),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3 * len(panels)), sharex=False)
    fig.suptitle("Stage G Training — D435i Noise Mode", fontsize=14, fontweight="bold")

    for pidx, (tag, ylabel, title, scale, show_thresh) in enumerate(panels):
        ax = axes[pidx]
        for i, (data, label) in enumerate(zip(all_data, labels)):
            pts = data.get(tag, [])
            if not pts:
                continue
            steps = np.array([p[0] for p in pts])
            vals = np.array([p[1] for p in pts]) * scale
            ax.plot(steps, vals, alpha=0.2, color=colors[i % len(colors)])
            if len(vals) >= window:
                s = smooth(vals, window)
                s_steps = steps[window // 2 : window // 2 + len(s)]
                ax.plot(s_steps, s, color=colors[i % len(colors)], label=label, linewidth=2)
            else:
                ax.plot(steps, vals, color=colors[i % len(colors)], label=label, linewidth=2)

        if show_thresh:
            ax.axhline(
                y=threshold * scale,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Threshold ({threshold*100:.0f}%)",
            )
            ax.set_ylim(0, 100)

        if tag == "Curriculum/es_no_improve":
            ax.axhline(y=1500, color="red", linestyle="--", alpha=0.7, label="ES patience (1500)")

        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlabel("Training Iteration")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {out_path}")

    # Print summary
    for data, label in zip(all_data, labels):
        print(f"\n--- {label} ---")
        timeout_pts = data.get("Episode_Termination/time_out", [])
        if timeout_pts:
            vals = np.array([p[1] for p in timeout_pts])
            steps = np.array([p[0] for p in timeout_pts])
            print(f"  Steps: {steps[0]}-{steps[-1]} ({len(vals)} iterations)")
            last20 = vals[-20:] if len(vals) >= 20 else vals
            print(f"  Timeout (last 20): {last20.mean()*100:.1f}% ± {last20.std()*100:.1f}%")
            over80 = (vals >= threshold).sum()
            print(f"  Iters ≥ {threshold*100:.0f}%: {over80}/{len(vals)}")

        apex_pts = data.get("Episode_Reward/ball_apex_height", [])
        if apex_pts:
            vals = np.array([p[1] for p in apex_pts])
            last20 = vals[-20:] if len(vals) >= 20 else vals
            print(f"  Apex reward (last 20): {last20.mean():.3f} ± {last20.std():.3f}")

        noise_pts = data.get("Policy/mean_noise_std", [])
        if noise_pts:
            print(f"  Noise std (final): {noise_pts[-1][1]:.4f}")

        es_pts = data.get("Curriculum/es_no_improve", [])
        if es_pts:
            es_val = es_pts[-1][1]
            print(f"  ES no-improve: {es_val:.0f}/1500 ({es_val/1500*100:.1f}%)")

    # CSV export
    if csv_path:
        with open(csv_path, "w") as f:
            f.write("label,total_iters,timeout_last20_mean,timeout_last20_std,apex_last20_mean,noise_std_final,es_no_improve\n")
            for data, label in zip(all_data, labels):
                timeout_vals = np.array([p[1] for p in data.get("Episode_Termination/time_out", [])])
                apex_vals = np.array([p[1] for p in data.get("Episode_Reward/ball_apex_height", [])])
                noise_pts = data.get("Policy/mean_noise_std", [])
                es_pts = data.get("Curriculum/es_no_improve", [])

                t_mean = timeout_vals[-20:].mean() * 100 if len(timeout_vals) >= 20 else float("nan")
                t_std = timeout_vals[-20:].std() * 100 if len(timeout_vals) >= 20 else float("nan")
                a_mean = apex_vals[-20:].mean() if len(apex_vals) >= 20 else float("nan")
                ns = noise_pts[-1][1] if noise_pts else float("nan")
                es = es_pts[-1][1] if es_pts else float("nan")
                f.write(f"{label},{len(timeout_vals)},{t_mean:.2f},{t_std:.2f},{a_mean:.3f},{ns:.4f},{es:.0f}\n")
        print(f"Saved CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from TensorBoard events")
    parser.add_argument("--logdir", nargs="+", required=True, help="TensorBoard logdir(s)")
    parser.add_argument("--labels", nargs="+", default=None, help="Labels for each run")
    parser.add_argument("--out", required=True, help="Output figure path")
    parser.add_argument("--window", type=int, default=20, help="Smoothing window")
    parser.add_argument("--threshold", type=float, default=0.80, help="Curriculum threshold")
    parser.add_argument("--csv", default=None, help="Optional CSV summary output")
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(d.rstrip("/")) for d in args.logdir]
    plot_training_tb(args.logdir, labels, args.out, args.window, args.threshold, args.csv)


if __name__ == "__main__":
    main()
