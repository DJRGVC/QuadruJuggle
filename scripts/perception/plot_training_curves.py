#!/usr/bin/env python3
"""Plot training curves from an RSL-RL training log file.

Extracts key metrics (timeout %, episode length, reward, apex reward, noise std)
and produces a multi-panel figure showing training progression. Useful for
diagnosing plateaus and comparing training runs.

Usage:
    python scripts/perception/plot_training_curves.py \
        --log experiments/iter_032_stage_g_entropy_fix/train.log \
        --out images/perception/stage_g_training_curves.png

    # Compare two training logs:
    python scripts/perception/plot_training_curves.py \
        --log experiments/run_a/train.log experiments/run_b/train.log \
        --labels "Run A" "Run B" \
        --out images/perception/training_comparison.png
"""

import argparse
import os
import re
import sys

import numpy as np


def parse_training_log(path: str) -> dict:
    """Extract per-iteration metrics from an RSL-RL training log."""
    metrics = {
        "timeout": [],
        "ep_length": [],
        "reward": [],
        "apex_reward": [],
        "noise_std": [],
        "ball_xy_dist": [],
    }

    with open(path) as f:
        for line in f:
            line = line.strip()
            if "Episode_Termination/time_out:" in line:
                val = float(line.split(":")[-1].strip())
                metrics["timeout"].append(val)
            elif "Mean episode length:" in line:
                val = float(line.split(":")[-1].strip())
                metrics["ep_length"].append(val)
            elif line.startswith("Mean reward:"):
                val = float(line.split(":")[-1].strip())
                metrics["reward"].append(val)
            elif "Episode_Reward/ball_apex_height:" in line:
                val = float(line.split(":")[-1].strip())
                metrics["apex_reward"].append(val)
            elif "Mean action noise std:" in line:
                val = float(line.split(":")[-1].strip())
                metrics["noise_std"].append(val)
            elif "Episode_Reward/ball_xy_dist:" in line:
                val = float(line.split(":")[-1].strip())
                metrics["ball_xy_dist"].append(val)

    # Convert to numpy
    for k in metrics:
        metrics[k] = np.array(metrics[k]) if metrics[k] else np.array([])

    return metrics


def smooth(arr: np.ndarray, window: int = 20) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_training_curves(
    logs: list[str],
    labels: list[str],
    out_path: str,
    window: int = 20,
    threshold: float = 0.80,
):
    """Plot multi-panel training curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_metrics = []
    for log_path in logs:
        all_metrics.append(parse_training_log(log_path))

    n_panels = 4
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), sharex=False)
    fig.suptitle("Training Curves — Stage G D435i", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10.colors

    # Panel 1: Timeout %
    ax = axes[0]
    for i, (m, label) in enumerate(zip(all_metrics, labels)):
        if len(m["timeout"]) > 0:
            x = np.arange(len(m["timeout"]))
            ax.plot(x, m["timeout"] * 100, alpha=0.2, color=colors[i])
            s = smooth(m["timeout"] * 100, window)
            ax.plot(np.arange(len(s)) + window // 2, s, color=colors[i], label=label, linewidth=2)
    ax.axhline(y=threshold * 100, color="red", linestyle="--", alpha=0.7, label=f"Curriculum threshold ({threshold*100:.0f}%)")
    ax.set_ylabel("Timeout %")
    ax.set_title("Episode Survival (timeout = reached max steps)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # Panel 2: Episode length
    ax = axes[1]
    for i, (m, label) in enumerate(zip(all_metrics, labels)):
        if len(m["ep_length"]) > 0:
            x = np.arange(len(m["ep_length"]))
            ax.plot(x, m["ep_length"], alpha=0.2, color=colors[i])
            s = smooth(m["ep_length"], window)
            ax.plot(np.arange(len(s)) + window // 2, s, color=colors[i], label=label, linewidth=2)
    ax.set_ylabel("Mean Episode Length (steps)")
    ax.set_title("Episode Length")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Panel 3: Apex reward
    ax = axes[2]
    for i, (m, label) in enumerate(zip(all_metrics, labels)):
        if len(m["apex_reward"]) > 0:
            x = np.arange(len(m["apex_reward"]))
            ax.plot(x, m["apex_reward"], alpha=0.2, color=colors[i])
            s = smooth(m["apex_reward"], window)
            ax.plot(np.arange(len(s)) + window // 2, s, color=colors[i], label=label, linewidth=2)
    ax.set_ylabel("Apex Height Reward")
    ax.set_title("Ball Apex Height Reward (target tracking)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Panel 4: Noise std
    ax = axes[3]
    for i, (m, label) in enumerate(zip(all_metrics, labels)):
        if len(m["noise_std"]) > 0:
            x = np.arange(len(m["noise_std"]))
            ax.plot(x, m["noise_std"], color=colors[i], label=label, linewidth=2)
    ax.set_ylabel("Action Noise Std")
    ax.set_title("Policy Exploration (noise std)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xlabel("Training Iteration")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Print summary
    for m, label in zip(all_metrics, labels):
        print(f"\n--- {label} ---")
        if len(m["timeout"]) > 0:
            print(f"  Iterations: {len(m['timeout'])}")
            print(f"  Timeout: {m['timeout'][-1]*100:.1f}% (last), {m['timeout'].mean()*100:.1f}% (mean)")
            last20 = m["timeout"][-20:] if len(m["timeout"]) >= 20 else m["timeout"]
            print(f"  Timeout (last 20): {last20.mean()*100:.1f}% ± {last20.std()*100:.1f}%")
            over80 = (m["timeout"] >= threshold).sum()
            print(f"  Iters ≥ {threshold*100:.0f}%: {over80}/{len(m['timeout'])}")
        if len(m["apex_reward"]) > 0:
            last20 = m["apex_reward"][-20:] if len(m["apex_reward"]) >= 20 else m["apex_reward"]
            print(f"  Apex reward (last 20): {last20.mean():.2f} ± {last20.std():.2f}")
        if len(m["noise_std"]) > 0:
            print(f"  Noise std: {m['noise_std'][-1]:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from RSL-RL log")
    parser.add_argument("--log", nargs="+", required=True, help="Training log file(s)")
    parser.add_argument("--labels", nargs="+", default=None, help="Labels for each log")
    parser.add_argument("--out", required=True, help="Output figure path")
    parser.add_argument("--window", type=int, default=20, help="Smoothing window (default: 20)")
    parser.add_argument("--threshold", type=float, default=0.80, help="Curriculum threshold")
    args = parser.parse_args()

    labels = args.labels or [os.path.basename(os.path.dirname(p)) for p in args.log]
    if len(labels) < len(args.log):
        labels.extend([f"Log {i}" for i in range(len(labels), len(args.log))])

    plot_training_curves(args.log, labels, args.out, args.window, args.threshold)


if __name__ == "__main__":
    main()
