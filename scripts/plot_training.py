#!/usr/bin/env python3
"""Plot training metrics from an RSL-RL go1_ball_balance run.

Usage:
    # Latest run
    python scripts/plot_training.py

    # Specific run name
    python scripts/plot_training.py --run 2026-02-25_10-00-00

    # Full path to event file or run directory
    python scripts/plot_training.py --log-dir /path/to/run/dir

Output:
    <run_dir>/training_plots.png
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Resolve log directory
# ---------------------------------------------------------------------------
ISAACLAB_DIR = Path.home() / "IsaacLab"
DEFAULT_LOG_ROOT = ISAACLAB_DIR / "logs" / "rsl_rl" / "go1_ball_balance"

parser = argparse.ArgumentParser()
parser.add_argument("--run",     default=None, help="Run name inside the log root")
parser.add_argument("--log-dir", default=None, help="Explicit path to run directory")
args = parser.parse_args()

if args.log_dir:
    run_dir = Path(args.log_dir)
elif args.run:
    run_dir = DEFAULT_LOG_ROOT / args.run
else:
    if not DEFAULT_LOG_ROOT.exists():
        sys.exit(f"[ERROR] Log root not found: {DEFAULT_LOG_ROOT}\n"
                 "Run training first with ./scripts/train.sh")
    runs = sorted(DEFAULT_LOG_ROOT.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    runs = [r for r in runs if r.is_dir()]
    if not runs:
        sys.exit(f"[ERROR] No runs found in {DEFAULT_LOG_ROOT}")
    run_dir = runs[0]

print(f"[plot_training] Reading run: {run_dir.name}")

# ---------------------------------------------------------------------------
# Find the tensorboard event file
# ---------------------------------------------------------------------------
event_files = list(run_dir.rglob("events.out.tfevents.*"))
if not event_files:
    sys.exit(f"[ERROR] No tensorboard event file found under {run_dir}")

event_file = sorted(event_files)[-1]
print(f"[plot_training] Event file: {event_file.name}")

# ---------------------------------------------------------------------------
# Read events — try tensorboard's event_accumulator first, fall back to
# manual protobuf parsing so the script works even without TensorFlow.
# ---------------------------------------------------------------------------
scalars = {}   # tag -> [(step, value), ...]

try:
    from tensorboard.backend.event_processing import event_accumulator as ea
    acc = ea.EventAccumulator(str(run_dir), size_guidance={ea.SCALARS: 0})
    acc.Reload()
    for tag in acc.Tags().get("scalars", []):
        scalars[tag] = [(e.step, e.value) for e in acc.Scalars(tag)]
    print(f"[plot_training] Loaded {len(scalars)} scalar tags via tensorboard")

except Exception as tb_err:
    print(f"[plot_training] tensorboard reader failed ({tb_err}), trying struct reader")
    import struct

    def _read_events(path):
        """Minimal tfrecord reader — no dependencies."""
        import gzip
        data = open(path, "rb").read()
        offset = 0
        events = []
        while offset < len(data):
            if offset + 8 > len(data):
                break
            length = struct.unpack_from("<Q", data, offset)[0]
            offset += 8 + 4   # length + masked_crc32_of_length
            if offset + length > len(data):
                break
            raw = data[offset: offset + length]
            offset += length + 4  # data + masked_crc32_of_data
            events.append(raw)
        return events

    try:
        from google.protobuf import descriptor_pool, descriptor_pb2
        import tensorflow as tf
        for e in tf.compat.v1.train.summary_iterator(str(event_file)):
            for v in e.summary.value:
                scalars.setdefault(v.tag, []).append((e.step, v.simple_value))
    except Exception:
        print("[plot_training] Could not read event file — install tensorboard: pip install tensorboard")
        sys.exit(1)

if not scalars:
    sys.exit("[ERROR] No scalar data found in event file.")

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def get(tag):
    """Return (steps, values) arrays for a tag, or (None, None) if missing."""
    for key in scalars:
        if tag.lower() in key.lower():
            pairs = scalars[key]
            if pairs:
                steps, vals = zip(*pairs)
                return np.array(steps), np.array(vals)
    return None, None

def smooth(y, w=20):
    if len(y) < w:
        return y
    kernel = np.ones(w) / w
    return np.convolve(y, kernel, mode="same")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(f"Training: {run_dir.name}", fontsize=13)

tag_axes = [
    ("Train/mean_reward",          axes[0, 0], "Mean Reward",           "reward / step"),
    ("Train/mean_episode_length",  axes[0, 1], "Mean Episode Length",   "steps"),
    ("Loss/value_function",        axes[1, 0], "Value Function Loss",   "loss"),
    ("Loss/surrogate",             axes[1, 1], "Surrogate (Policy) Loss","loss"),
    ("Loss/entropy",               axes[1, 2], "Entropy",               "nats"),
    ("Policy/mean_noise_std",      axes[0, 2], "Policy Noise Std",      "std"),
]

for tag, ax, title, ylabel in tag_axes:
    steps, vals = get(tag)
    if steps is None:
        ax.set_visible(False)
        continue
    ax.plot(steps, vals, alpha=0.3, color="steelblue", linewidth=0.8)
    if len(vals) > 5:
        ax.plot(steps, smooth(vals), color="steelblue", linewidth=1.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

out_path = run_dir / "training_plots.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"[plot_training] Saved: {out_path}")

# Print summary stats
steps, vals = get("Train/mean_reward")
if vals is not None:
    print(f"\n  Reward — final: {vals[-1]:.3f}   best: {vals.max():.3f}   "
          f"@ iter {steps[vals.argmax()]}")
steps, vals = get("Train/mean_episode_length")
if vals is not None:
    print(f"  Episode len — final: {vals[-1]:.1f}   best: {vals.max():.1f}")
