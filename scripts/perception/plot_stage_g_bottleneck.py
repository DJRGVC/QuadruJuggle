#!/usr/bin/env python3
"""Plot Stage G eval results showing policy bottleneck (iter 116).

Creates a 2-panel figure:
  Left:  Flight fraction + detection rate vs target height
  Right: Ball peak height vs target height (with identity line)

Demonstrates that low detection rate is caused by lack of flight time
(policy balances instead of juggling), not camera/EKF failure.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def load_eval_data(eval_dir: Path):
    """Load trajectory data from eval directory."""
    targets = []
    det_rates = []
    flight_fracs = []
    peak_heights = []
    timeout_rates = []
    bounce_counts = []
    ekf_rmses = []

    for target_dir in sorted(eval_dir.iterdir()):
        if not target_dir.is_dir() or not target_dir.name.startswith("target_"):
            continue
        traj_file = target_dir / "trajectory.npz"
        if not traj_file.exists():
            continue

        # Parse target from dir name: target_0_30 → 0.30
        parts = target_dir.name.replace("target_", "").split("_")
        target = float(f"{parts[0]}.{parts[1]}")
        targets.append(target)

        d = np.load(traj_file, allow_pickle=True)
        n_steps = len(d["gt"])
        n_det = len(d["det"])
        det_rates.append(n_det / n_steps * 100)

        # Flight fraction from phase data (1=ascending, 2=descending, 0=contact)
        phase = d["phase"]
        flight_steps = np.sum(phase > 0)
        flight_fracs.append(flight_steps / n_steps * 100)

        # Peak height above paddle (ball_h is height above paddle surface)
        ball_h = d["ball_h"]
        peak_heights.append(np.max(ball_h) if len(ball_h) > 0 else 0.0)

        # EKF RMSE (mean across steps)
        ekf_rmses.append(np.mean(d["rmse_ekf"]))

    return {
        "targets": np.array(targets),
        "det_rate": np.array(det_rates),
        "flight_frac": np.array(flight_fracs),
        "peak_height": np.array(peak_heights),
        "ekf_rmse": np.array(ekf_rmses),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path,
                        default=Path("logs/perception/eval_stage_g_starve10"),
                        help="Directory with per-target trajectory.npz files")
    parser.add_argument("--output", type=Path,
                        default=Path("images/perception/stage_g_bottleneck_iter116.png"))
    args = parser.parse_args()

    data = load_eval_data(args.eval_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Flight fraction + detection rate
    color_flight = "#2196F3"
    color_det = "#FF5722"

    ax1.bar(data["targets"] - 0.015, data["flight_frac"], width=0.03,
            color=color_flight, alpha=0.8, label="Flight fraction")
    ax1.bar(data["targets"] + 0.015, data["det_rate"], width=0.03,
            color=color_det, alpha=0.8, label="Detection rate")

    ax1.set_xlabel("Target apex height (m)", fontsize=12)
    ax1.set_ylabel("Percentage of episode (%)", fontsize=12)
    ax1.set_title("Flight time vs detection rate", fontsize=13)
    ax1.legend(fontsize=11, loc="upper left")
    ax1.set_xticks(data["targets"])
    ax1.set_xticklabels([f"{t:.2f}" for t in data["targets"]])
    ax1.set_ylim(0, max(30, max(data["flight_frac"]) * 1.3))
    ax1.grid(axis="y", alpha=0.3)

    # Annotate: detection tracks flight, not target
    ax1.annotate("Detection rate tracks\nflight fraction, not target",
                 xy=(0.85, 15), fontsize=9, fontstyle="italic",
                 ha="center", color="#666")

    # Panel 2: Peak height vs target (with identity line)
    ax2.bar(data["targets"], data["peak_height"], width=0.06,
            color="#4CAF50", alpha=0.8, label="Achieved peak height")
    ax2.plot([0, 1.1], [0, 1.1], "k--", alpha=0.4, label="Target = achieved")

    ax2.set_xlabel("Target apex height (m)", fontsize=12)
    ax2.set_ylabel("Peak ball height above paddle (m)", fontsize=12)
    ax2.set_title("Policy undershoots high targets", fontsize=13)
    ax2.legend(fontsize=11, loc="upper left")
    ax2.set_xticks(data["targets"])
    ax2.set_xticklabels([f"{t:.2f}" for t in data["targets"]])
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis="y", alpha=0.3)

    # Annotate balance vs juggle zones
    ax2.axvspan(0.0, 0.55, alpha=0.05, color="blue")
    ax2.axvspan(0.55, 1.1, alpha=0.05, color="red")
    ax2.text(0.30, 1.02, "Balances\n(no bounce)", fontsize=9, ha="center",
             fontstyle="italic", color="#1565C0")
    ax2.text(0.85, 1.02, "Attempts juggle\n(drops ball)", fontsize=9,
             ha="center", fontstyle="italic", color="#C62828")

    fig.suptitle("Stage G Eval — Perception Pipeline vs Policy Bottleneck (iter 116)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
