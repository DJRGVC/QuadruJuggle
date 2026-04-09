#!/usr/bin/env python3
"""Analyze EKF accuracy during flight windows vs contact phases.

Reads trajectory.npz files from eval directories and breaks down metrics by
ball phase (contact/ascending/descending). This proves the perception pipeline
works correctly during the flight windows that matter for juggling.

Produces:
  1. Per-target flight-window summary table (stdout)
  2. 4-panel publication figure:
     (a) Flight fraction by target height
     (b) EKF RMSE: flight vs contact vs overall
     (c) Detection efficiency (detections per flight step)
     (d) Flight-window height profile (mean ball Z during flight)

Usage:
    # Single eval directory:
    python scripts/perception/analyze_flight_windows.py \
        --eval-dir logs/perception/eval_stage_g_d435i

    # Compare two eval directories:
    python scripts/perception/analyze_flight_windows.py \
        --eval-dir logs/perception/eval_stage_g_d435i \
        --compare logs/perception/eval_stage_g_starve10 \
        --labels "D435i camera" "Starvation override" \
        --out images/perception/flight_window_analysis.png
"""

import argparse
import os
import sys
from dataclasses import dataclass, field

import numpy as np


# Phase constants (must match phase_tracker.py)
PHASE_CONTACT = 0
PHASE_ASCENDING = 1
PHASE_DESCENDING = 2

PHASE_NAMES = {PHASE_CONTACT: "contact", PHASE_ASCENDING: "ascending", PHASE_DESCENDING: "descending"}


@dataclass
class FlightMetrics:
    """Metrics for a single trajectory, broken down by flight phase."""

    target: float
    n_total: int
    n_flight: int  # ascending + descending
    n_contact: int
    n_ascending: int
    n_descending: int
    flight_fraction: float

    # Detection metrics
    n_det_total: int
    n_det_flight: int
    n_det_contact: int
    det_rate_total: float
    det_per_flight_step: float  # detections / flight steps (efficiency)

    # RMSE metrics
    rmse_all: float
    rmse_flight: float
    rmse_contact: float
    rmse_ascending: float
    rmse_descending: float

    # Height metrics during flight
    mean_ball_z_flight: float
    max_ball_z_flight: float
    mean_ball_z_contact: float

    # Flight window stats
    n_flight_windows: int  # number of separate flight events
    mean_window_length: float  # steps per flight window


def _count_flight_windows(phase: np.ndarray) -> tuple[int, float]:
    """Count separate contiguous flight windows and their mean length."""
    flight_mask = phase > 0
    if not flight_mask.any():
        return 0, 0.0
    # Find transitions into/out of flight
    padded = np.concatenate([[False], flight_mask, [False]])
    starts = np.where(padded[1:] & ~padded[:-1])[0]
    ends = np.where(~padded[1:] & padded[:-1])[0]
    n_windows = len(starts)
    if n_windows == 0:
        return 0, 0.0
    lengths = ends - starts
    return n_windows, float(np.mean(lengths))


def analyze_trajectory(npz_path: str, target: float) -> FlightMetrics:
    """Analyze a single trajectory.npz for flight-window metrics."""
    d = np.load(npz_path)
    phase = d["phase"]
    gt = d["gt"].astype(np.float64)  # (T, 3)
    ekf = d["ekf"].astype(np.float64)  # (T, 3)
    det_steps = d["det_steps"]

    n_total = len(phase)
    flight_mask = phase > 0
    contact_mask = phase == 0
    asc_mask = phase == PHASE_ASCENDING
    desc_mask = phase == PHASE_DESCENDING

    n_flight = int(flight_mask.sum())
    n_contact = int(contact_mask.sum())
    n_ascending = int(asc_mask.sum())
    n_descending = int(desc_mask.sum())

    # Detections by phase
    det_in_flight = sum(1 for s in det_steps if s < n_total and flight_mask[s])
    det_in_contact = len(det_steps) - det_in_flight

    # RMSE by phase
    err = np.linalg.norm(ekf - gt, axis=1)

    def _rmse(mask):
        if mask.any():
            return float(np.sqrt(np.mean(err[mask] ** 2)))
        return float("nan")

    rmse_all = _rmse(np.ones(n_total, dtype=bool))
    rmse_flight = _rmse(flight_mask)
    rmse_contact = _rmse(contact_mask)
    rmse_ascending = _rmse(asc_mask)
    rmse_descending = _rmse(desc_mask)

    # Height metrics
    ball_z = gt[:, 2]
    mean_z_flight = float(np.mean(ball_z[flight_mask])) if n_flight > 0 else float("nan")
    max_z_flight = float(np.max(ball_z[flight_mask])) if n_flight > 0 else float("nan")
    mean_z_contact = float(np.mean(ball_z[contact_mask])) if n_contact > 0 else float("nan")

    # Flight windows
    n_windows, mean_window_len = _count_flight_windows(phase)

    return FlightMetrics(
        target=target,
        n_total=n_total,
        n_flight=n_flight,
        n_contact=n_contact,
        n_ascending=n_ascending,
        n_descending=n_descending,
        flight_fraction=n_flight / n_total if n_total > 0 else 0.0,
        n_det_total=len(det_steps),
        n_det_flight=det_in_flight,
        n_det_contact=det_in_contact,
        det_rate_total=len(det_steps) / n_total if n_total > 0 else 0.0,
        det_per_flight_step=det_in_flight / n_flight if n_flight > 0 else 0.0,
        rmse_all=rmse_all,
        rmse_flight=rmse_flight,
        rmse_contact=rmse_contact,
        rmse_ascending=rmse_ascending,
        rmse_descending=rmse_descending,
        mean_ball_z_flight=mean_z_flight,
        max_ball_z_flight=max_z_flight,
        mean_ball_z_contact=mean_z_contact,
        n_flight_windows=n_windows,
        mean_window_length=mean_window_len,
    )


def load_eval_dir(eval_dir: str) -> list[FlightMetrics]:
    """Load all target subdirectories from an eval directory."""
    results = []
    for entry in sorted(os.listdir(eval_dir)):
        npz_path = os.path.join(eval_dir, entry, "trajectory.npz")
        if not os.path.isfile(npz_path):
            continue
        # Parse target from dirname like "target_0_50"
        try:
            target = float(entry.replace("target_", "").replace("_", "."))
        except ValueError:
            continue
        results.append(analyze_trajectory(npz_path, target))
    return sorted(results, key=lambda m: m.target)


def print_summary(metrics_list: list[FlightMetrics], label: str = "") -> None:
    """Print a formatted summary table."""
    if label:
        print(f"\n{'='*80}")
        print(f"  {label}")
        print(f"{'='*80}")

    header = (
        f"{'Target':>8} {'Flight%':>8} {'Windows':>8} {'MeanLen':>8} "
        f"{'Det':>5} {'DetFlt':>7} {'Det/Flt':>8} "
        f"{'RMSE_all':>9} {'RMSE_flt':>9} {'RMSE_con':>9} "
        f"{'MeanZ_f':>8} {'MaxZ_f':>8}"
    )
    print(header)
    print("-" * len(header))

    for m in metrics_list:
        print(
            f"{m.target:>8.2f} {100*m.flight_fraction:>7.1f}% {m.n_flight_windows:>8d} "
            f"{m.mean_window_length:>8.1f} "
            f"{m.n_det_total:>5d} {m.n_det_flight:>7d} {m.det_per_flight_step:>8.3f} "
            f"{m.rmse_all:>9.4f} {m.rmse_flight:>9.4f} {m.rmse_contact:>9.4f} "
            f"{m.mean_ball_z_flight:>8.3f} {m.max_ball_z_flight:>8.3f}"
        )


def plot_comparison(
    metrics_a: list[FlightMetrics],
    metrics_b: list[FlightMetrics] | None,
    label_a: str,
    label_b: str | None,
    out_path: str,
) -> None:
    """Create 4-panel comparison figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Flight-Window EKF Analysis", fontsize=14, fontweight="bold")

    def _extract(metrics, attr):
        targets = [m.target for m in metrics]
        values = [getattr(m, attr) for m in metrics]
        return targets, values

    colors = ["#2196F3", "#FF5722"]
    datasets = [(metrics_a, label_a, colors[0])]
    if metrics_b is not None and label_b is not None:
        datasets.append((metrics_b, label_b, colors[1]))

    # Panel (a): Flight fraction
    ax = axes[0, 0]
    for metrics, label, color in datasets:
        t, v = _extract(metrics, "flight_fraction")
        ax.bar(
            [x + (0.15 if color == colors[1] else -0.15) for x in t] if len(datasets) > 1 else t,
            [100 * x for x in v],
            width=0.08 if len(datasets) > 1 else 0.12,
            label=label,
            color=color,
            alpha=0.85,
        )
    ax.set_xlabel("Target apex height (m)")
    ax.set_ylabel("Flight fraction (%)")
    ax.set_title("(a) Time in flight")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Panel (b): RMSE breakdown
    ax = axes[0, 1]
    for metrics, label, color in datasets:
        t_flt, v_flt = _extract(metrics, "rmse_flight")
        t_con, v_con = _extract(metrics, "rmse_contact")
        t_all, v_all = _extract(metrics, "rmse_all")
        # Cap extreme values for readability
        v_flt = [min(v, 2.0) for v in v_flt]
        v_con = [min(v, 2.0) for v in v_con]
        ax.plot(t_flt, [100 * v for v in v_flt], "o-", color=color, label=f"{label} (flight)", linewidth=2)
        ax.plot(t_con, [100 * v for v in v_con], "s--", color=color, label=f"{label} (contact)", linewidth=1.5, alpha=0.6)
    ax.set_xlabel("Target apex height (m)")
    ax.set_ylabel("EKF RMSE (cm)")
    ax.set_title("(b) EKF accuracy by phase")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel (c): Detection efficiency (detections per flight step)
    ax = axes[1, 0]
    for metrics, label, color in datasets:
        t, v = _extract(metrics, "det_per_flight_step")
        ax.plot(t, v, "o-", color=color, label=label, linewidth=2, markersize=8)
    ax.set_xlabel("Target apex height (m)")
    ax.set_ylabel("Detections per flight step")
    ax.set_title("(c) Detection efficiency in flight")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim(bottom=0)

    # Panel (d): Flight-window height profile
    ax = axes[1, 1]
    for metrics, label, color in datasets:
        t_mean, v_mean = _extract(metrics, "mean_ball_z_flight")
        t_max, v_max = _extract(metrics, "max_ball_z_flight")
        ax.plot(t_mean, v_mean, "o-", color=color, label=f"{label} (mean Z)", linewidth=2)
        ax.plot(t_max, v_max, "^--", color=color, label=f"{label} (peak Z)", linewidth=1.5, alpha=0.7)
    # Add identity line (target = actual peak)
    all_targets = [m.target for m in metrics_a]
    ax.plot(all_targets, [t + 0.47 for t in all_targets], "k:", alpha=0.4, label="Ideal peak (target + paddle)")
    ax.set_xlabel("Target apex height (m)")
    ax.set_ylabel("Ball Z in world frame (m)")
    ax.set_title("(d) Ball height during flight")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Flight-window EKF analysis")
    parser.add_argument("--eval-dir", required=True, help="Primary eval directory with target_X_XX subdirs")
    parser.add_argument("--compare", default=None, help="Second eval directory for comparison")
    parser.add_argument("--labels", nargs=2, default=["Primary", "Comparison"])
    parser.add_argument("--out", default=None, help="Output figure path (default: auto)")
    args = parser.parse_args()

    metrics_a = load_eval_dir(args.eval_dir)
    if not metrics_a:
        print(f"ERROR: No trajectory.npz files found in {args.eval_dir}", file=sys.stderr)
        sys.exit(1)

    print_summary(metrics_a, args.labels[0])

    metrics_b = None
    if args.compare:
        metrics_b = load_eval_dir(args.compare)
        if metrics_b:
            print_summary(metrics_b, args.labels[1])
        else:
            print(f"WARNING: No trajectory data in {args.compare}", file=sys.stderr)

    out_path = args.out or os.path.join(
        "images", "perception",
        "flight_window_analysis.png",
    )
    plot_comparison(metrics_a, metrics_b, args.labels[0], args.labels[1] if metrics_b else None, out_path)


if __name__ == "__main__":
    main()
