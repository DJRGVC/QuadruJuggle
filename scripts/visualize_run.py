"""
Visualize Isaac Lab / RSL-RL training runs from TensorBoard event files.

Usage:
    python visualize_run.py                          # latest run
    python visualize_run.py --run 2026-02-24_12-24-41
    python visualize_run.py --all                    # overlay all runs on key plots
    python visualize_run.py --save                   # save PNGs instead of showing
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
if os.environ.get("DISPLAY"):
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Config ────────────────────────────────────────────────────────────────────

LOG_BASE = Path(__file__).parent / "logs" / "rsl_rl" / "unitree_go1_flat"
SMOOTH_WEIGHT = 0.85        # exponential moving average for smoothing curves
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_run(run_dir: Path) -> dict[str, tuple[list, list]]:
    """Load all scalars from a TensorBoard event file. Returns {tag: (steps, values)}."""
    ea = EventAccumulator(str(run_dir))
    ea.Reload()
    data = {}
    for tag in ea.Tags().get("scalars", []):
        events = ea.Scalars(tag)
        steps  = [e.step  for e in events]
        values = [e.value for e in events]
        data[tag] = (steps, values)
    return data


def smooth(values: list[float], weight: float = SMOOTH_WEIGHT) -> np.ndarray:
    """Exponential moving average smoothing (same as TensorBoard's default)."""
    out = np.zeros(len(values))
    last = values[0]
    for i, v in enumerate(values):
        last = weight * last + (1 - weight) * v
        out[i] = last
    return out


def plot_tag(ax, data, tag, label=None, color=None, alpha_raw=0.2, show_raw=True):
    if tag not in data:
        return
    steps, values = data[tag]
    s = smooth(values)
    kw = dict(color=color) if color else {}
    if show_raw:
        ax.plot(steps, values, alpha=alpha_raw, **kw)
    line, = ax.plot(steps, s, label=label or tag.split("/")[-1], linewidth=2, **kw)
    return line


def list_runs() -> list[Path]:
    return sorted(LOG_BASE.iterdir()) if LOG_BASE.exists() else []


def pick_run(run_name: str | None) -> Path:
    runs = list_runs()
    if not runs:
        sys.exit(f"No runs found in {LOG_BASE}")
    if run_name:
        match = LOG_BASE / run_name
        if not match.exists():
            sys.exit(f"Run not found: {match}")
        return match
    return runs[-1]   # latest


def summarise(run_dir: Path, data: dict) -> str:
    lines = [f"\n{'='*60}", f"  Run: {run_dir.name}", f"{'='*60}"]

    def last(tag):
        if tag in data:
            return data[tag][1][-1]
        return None

    def iters(tag):
        if tag in data:
            return data[tag][0][-1]
        return None

    r  = last("Train/mean_reward")
    ep = last("Train/mean_episode_length")
    to = last("Episode_Termination/time_out")
    fc = last("Episode_Termination/base_contact")
    vt = last("Episode_Reward/track_lin_vel_xy_exp")
    yt = last("Episode_Reward/track_ang_vel_z_exp")
    ns = last("Policy/mean_noise_std")
    lr = last("Loss/learning_rate")
    fp = last("Perf/total_fps")
    ev = last("Metrics/base_velocity/error_vel_xy")

    if r  is not None: lines.append(f"  Mean reward          : {r:.2f}")
    if ep is not None: lines.append(f"  Mean episode length  : {ep:.0f} / 1000 steps")
    if to is not None: lines.append(f"  Timeout rate         : {to*100:.1f}%  (not falling)")
    if fc is not None: lines.append(f"  Fall rate            : {fc*100:.2f}%")
    if vt is not None: lines.append(f"  Lin vel tracking     : {vt:.3f} / 1.500  ({vt/1.5*100:.0f}%)")
    if yt is not None: lines.append(f"  Ang vel tracking     : {yt:.3f} / 0.750  ({yt/0.75*100:.0f}%)")
    if ev is not None: lines.append(f"  Velocity XY error    : {ev:.4f} m/s")
    if ns is not None: lines.append(f"  Policy noise std     : {ns:.4f}")
    if lr is not None: lines.append(f"  Final learning rate  : {lr:.2e}")
    if fp is not None: lines.append(f"  Throughput           : {fp/1e3:.0f}k env-steps/sec")
    if iters("Train/mean_reward") is not None:
        lines.append(f"  Total iterations     : {iters('Train/mean_reward')}")

    lines.append("="*60)
    return "\n".join(lines)


# ── Main plot function ────────────────────────────────────────────────────────

def plot_single(run_dir: Path, save: bool = False):
    data = load_run(run_dir)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"Training run: {run_dir.name}", fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0: key training signals ──────────────────────────────────────────
    ax_reward = fig.add_subplot(gs[0, :2])
    ax_reward.set_title("Mean Reward  ↑")
    plot_tag(ax_reward, data, "Train/mean_reward", label="mean reward")
    ax_reward.set_xlabel("Iteration")
    ax_reward.set_ylabel("Reward")
    ax_reward.legend()
    ax_reward.grid(alpha=0.3)

    ax_eplen = fig.add_subplot(gs[0, 2:])
    ax_eplen.set_title("Mean Episode Length  ↑  (max=1000)")
    plot_tag(ax_eplen, data, "Train/mean_episode_length", label="ep length", color=COLORS[1])
    ax_eplen.axhline(1000, color="grey", linestyle="--", linewidth=1, label="max")
    ax_eplen.set_xlabel("Iteration")
    ax_eplen.set_ylabel("Steps")
    ax_eplen.legend()
    ax_eplen.grid(alpha=0.3)

    # ── Row 1: reward components ──────────────────────────────────────────────
    tracking_tags = [
        ("Episode_Reward/track_lin_vel_xy_exp", "lin vel XY  (max 1.5)"),
        ("Episode_Reward/track_ang_vel_z_exp",  "ang vel Z   (max 0.75)"),
    ]
    penalty_tags = [
        ("Episode_Reward/flat_orientation_l2", "flat orientation"),
        ("Episode_Reward/lin_vel_z_l2",        "vert vel"),
        ("Episode_Reward/ang_vel_xy_l2",       "roll/pitch rate"),
        ("Episode_Reward/dof_torques_l2",      "joint torques"),
        ("Episode_Reward/dof_acc_l2",          "joint accel"),
        ("Episode_Reward/action_rate_l2",      "action rate"),
    ]

    ax_track = fig.add_subplot(gs[1, :2])
    ax_track.set_title("Velocity Tracking Rewards  ↑")
    for i, (tag, lbl) in enumerate(tracking_tags):
        plot_tag(ax_track, data, tag, label=lbl, color=COLORS[i])
    ax_track.set_xlabel("Iteration")
    ax_track.legend(fontsize=8)
    ax_track.grid(alpha=0.3)

    ax_pen = fig.add_subplot(gs[1, 2:])
    ax_pen.set_title("Penalty Terms  ↑ (closer to 0 = better)")
    for i, (tag, lbl) in enumerate(penalty_tags):
        plot_tag(ax_pen, data, tag, label=lbl, color=COLORS[i % len(COLORS)])
    ax_pen.set_xlabel("Iteration")
    ax_pen.legend(fontsize=7)
    ax_pen.grid(alpha=0.3)

    # ── Row 2: losses, LR, noise, FPS ────────────────────────────────────────
    ax_loss = fig.add_subplot(gs[2, 0])
    ax_loss.set_title("Actor & Critic Loss  ↓")
    plot_tag(ax_loss, data, "Loss/surrogate",       label="surrogate (actor)", color=COLORS[0])
    plot_tag(ax_loss, data, "Loss/value_function",  label="value fn (critic)", color=COLORS[1])
    ax_loss.set_xlabel("Iteration")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(alpha=0.3)

    ax_entropy = fig.add_subplot(gs[2, 1])
    ax_entropy.set_title("Entropy  (exploration)")
    plot_tag(ax_entropy, data, "Loss/entropy", label="entropy", color=COLORS[2])
    ax_entropy.set_xlabel("Iteration")
    ax_entropy.legend(fontsize=8)
    ax_entropy.grid(alpha=0.3)

    ax_lr = fig.add_subplot(gs[2, 2])
    ax_lr.set_title("Learning Rate  (adaptive)")
    plot_tag(ax_lr, data, "Loss/learning_rate",     label="LR",         color=COLORS[3], show_raw=False)
    ax_lr2 = ax_lr.twinx()
    plot_tag(ax_lr2, data, "Policy/mean_noise_std", label="noise std",  color=COLORS[4], show_raw=False)
    ax_lr.set_xlabel("Iteration")
    ax_lr.set_ylabel("LR", color=COLORS[3])
    ax_lr2.set_ylabel("noise σ", color=COLORS[4])
    ax_lr.grid(alpha=0.3)

    ax_fps = fig.add_subplot(gs[2, 3])
    ax_fps.set_title("Throughput (env steps/sec)")
    plot_tag(ax_fps, data, "Perf/total_fps", label="fps", color=COLORS[5], show_raw=False)
    ax_fps.set_xlabel("Iteration")
    ax_fps.set_ylabel("steps/sec")
    ax_fps.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax_fps.grid(alpha=0.3)

    plt.tight_layout()

    print(summarise(run_dir, data))

    if save:
        out = Path(run_dir) / "training_plots.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\nSaved to: {out}")
    else:
        plt.show()


def plot_comparison(save: bool = False):
    """Overlay all runs on the two key metrics for comparison."""
    runs = list_runs()
    if not runs:
        sys.exit("No runs found.")

    fig, (ax_r, ax_e) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("All runs — comparison", fontsize=13, fontweight="bold")

    for i, run_dir in enumerate(runs):
        try:
            data = load_run(run_dir)
        except Exception:
            continue
        color = COLORS[i % len(COLORS)]
        label = run_dir.name
        plot_tag(ax_r, data, "Train/mean_reward",        label=label, color=color)
        plot_tag(ax_e, data, "Train/mean_episode_length", label=label, color=color)

    for ax, title, ylabel in [
        (ax_r, "Mean Reward  ↑",           "Reward"),
        (ax_e, "Mean Episode Length  ↑",   "Steps"),
    ]:
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    ax_e.axhline(1000, color="grey", linestyle="--", linewidth=1, label="max")

    plt.tight_layout()

    if save:
        out = LOG_BASE / "comparison.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved to: {out}")
    else:
        plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize RSL-RL training runs")
    parser.add_argument("--run",  type=str,  default=None,  help="Run name (default: latest)")
    parser.add_argument("--all",  action="store_true",      help="Compare all runs")
    parser.add_argument("--save", action="store_true",      help="Save PNG instead of showing")
    args = parser.parse_args()

    if args.all:
        plot_comparison(save=args.save)
    else:
        run_dir = pick_run(args.run)
        print(f"Loading run: {run_dir.name}")
        plot_single(run_dir, save=args.save)
