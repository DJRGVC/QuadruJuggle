#!/usr/bin/env python3
"""Plot d435i vs oracle training history across curriculum stages.

Reads TensorBoard event files from multiple training runs and produces a
multi-panel figure showing the complete training progression.

Usage:
    python scripts/perception/plot_training_history.py \
        --out images/perception/training_history_iter124.png
"""

import argparse
import os
import sys

import numpy as np


def load_tb_scalars(logdir: str, tags: list[str]) -> dict[str, list[tuple[int, float]]]:
    """Load scalar events from a TensorBoard logdir."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
    ea.Reload()

    result: dict[str, list[tuple[int, float]]] = {}
    available = ea.Tags().get("scalars", [])
    for tag in tags:
        if tag in available:
            events = ea.Scalars(tag)
            result[tag] = [(e.step, e.value) for e in events]
    return result


# Tags to load
TAGS = [
    "Episode_Termination/time_out",
    "Train/mean_episode_length",
    "Episode_Reward/ball_apex_height",
    "Policy/mean_noise_std",
]

TAG_LABELS = {
    "Episode_Termination/time_out": "Timeout %",
    "Train/mean_episode_length": "Episode Length",
    "Episode_Reward/ball_apex_height": "Apex Reward (per episode)",
    "Policy/mean_noise_std": "Policy Noise Std",
}

# Run metadata: (logdir_suffix, label, color, linestyle)
POLICY_LOG_BASE = (
    "/home/daniel-grant/Research/QuadruJuggle-policy/logs/rsl_rl/go1_ball_juggle_hier"
)

RUNS = [
    ("2026-04-08_19-19-41", "Oracle (A→F)", "#2E86C1", "-"),
    ("2026-04-08_21-16-05", "D435i Stage E", "#E67E22", "--"),
    ("2026-04-08_22-51-56", "D435i Stage F", "#E67E22", "-"),
    ("2026-04-09_07-38-27", "D435i Stage G", "#C0392B", "-"),
]


def smooth(values: np.ndarray, window: int = 20) -> np.ndarray:
    """Simple moving average."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser(description="Training history dashboard")
    parser.add_argument(
        "--out",
        default="images/perception/training_history.png",
        help="Output figure path",
    )
    parser.add_argument("--window", type=int, default=20, help="Smoothing window")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    panel_tags = list(TAG_LABELS.keys())

    for run_dir, label, color, ls in RUNS:
        logdir = os.path.join(POLICY_LOG_BASE, run_dir)
        if not os.path.isdir(logdir):
            print(f"[WARN] Skipping missing dir: {logdir}")
            continue

        data = load_tb_scalars(logdir, TAGS)

        for idx, tag in enumerate(panel_tags):
            ax = axes[idx // 2, idx % 2]
            if tag not in data or not data[tag]:
                continue
            steps, vals = zip(*data[tag])
            steps = np.array(steps)
            vals = np.array(vals)

            # Convert timeout to percentage
            if "time_out" in tag:
                vals = vals * 100

            # Smooth
            s_vals = smooth(vals, args.window)
            offset = len(vals) - len(s_vals)
            s_steps = steps[offset:]

            ax.plot(s_steps, s_vals, color=color, ls=ls, lw=1.5, alpha=0.85, label=label)

    # Configure panels
    for idx, tag in enumerate(panel_tags):
        ax = axes[idx // 2, idx % 2]
        ax.set_ylabel(TAG_LABELS[tag], fontsize=10)
        ax.set_xlabel("Training Step", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=7, loc="best")

    # Add 80% threshold line on timeout panel
    axes[0, 0].axhline(80, color="green", ls=":", lw=1, alpha=0.5, label="Stage advance (80%)")
    axes[0, 0].set_ylim(0, 105)
    axes[0, 0].legend(fontsize=7, loc="best")

    fig.suptitle(
        "D435i vs Oracle Training Progression (Stages E→G)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.out}")

    # Print summary stats for each run
    print("\n--- Summary (last 50 iters) ---")
    for run_dir, label, _, _ in RUNS:
        logdir = os.path.join(POLICY_LOG_BASE, run_dir)
        if not os.path.isdir(logdir):
            continue
        data = load_tb_scalars(logdir, TAGS)
        timeout_tag = "Episode_Termination/time_out"
        if timeout_tag in data and data[timeout_tag]:
            steps, vals = zip(*data[timeout_tag])
            vals = np.array(vals) * 100
            n = min(50, len(vals))
            last_step = steps[-1]
            total = len(vals)
            print(
                f"  {label:25s}: {total:5d} iters, "
                f"step {last_step:5d}, "
                f"timeout {vals[-n:].mean():.1f}% ± {vals[-n:].std():.1f}%"
            )


if __name__ == "__main__":
    main()
