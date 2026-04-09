#!/usr/bin/env python3
"""Predict the perception gap (oracle vs d435i timeout %) from noise model first principles.

For each target height, computes:
  1. D435i noise characteristics (σ_xy, σ_z, dropout rate)
  2. Expected flight time fraction (ballistic physics: t_flight ∝ √h)
  3. Effective noise exposure = flight_fraction × noise_level × (1 - anchor_fraction)
  4. Correlation with observed gap from eval data

This helps determine whether the observed gap is "expected" given the noise model,
or whether there's a specific EKF/pipeline issue to address.

Usage:
    python scripts/perception/predict_perception_gap.py \
        --out images/perception/gap_prediction_iter131.png

    # With observed data for overlay:
    python scripts/perception/predict_perception_gap.py \
        --observed '{"0.10": 0.3, "0.20": 0.0, "0.30": 3.6, "0.40": 10.0, "0.50": 18.3}' \
        --out images/perception/gap_prediction_iter131.png
"""

import argparse
import json
import math
import os
import sys

import numpy as np


# ── D435i noise model parameters (from noise_model.py) ──
SIGMA_XY_PER_METRE = 0.0025  # σ_xy = 0.0025·z
SIGMA_XY_FLOOR = 0.001       # 1mm floor
SIGMA_Z_BASE = 0.001         # 1mm constant floor
SIGMA_Z_QUADRATIC = 0.005    # 0.005·z²
DROPOUT_BASE = 0.20          # 20% baseline
DROPOUT_RANGE = 0.30         # 30% additional
DROPOUT_SCALE = 0.80         # metres

# Physics constants
G = 9.81  # m/s²
POLICY_HZ = 50.0  # policy rate
PADDLE_RESTITUTION = 0.85  # from scene config (effective restitution)

# Paddle anchor parameters (from paddle_anchor)
ANCHOR_R_POS = 0.005  # 5mm anchor measurement noise
MIN_STARVE_FOR_ANCHOR = 5  # steps without detection before anchor kicks in


def compute_noise_at_height(h: float) -> dict:
    """Compute D435i noise characteristics at height h above paddle.

    Args:
        h: ball height above paddle surface in metres

    Returns:
        dict with sigma_xy, sigma_z, dropout_rate, sigma_3d (RSS of xyz)
    """
    h = max(h, 0.001)  # avoid zero
    sigma_xy = max(SIGMA_XY_FLOOR, SIGMA_XY_PER_METRE * h)
    sigma_z = SIGMA_Z_BASE + SIGMA_Z_QUADRATIC * h * h
    dropout = DROPOUT_BASE + DROPOUT_RANGE * (1.0 - math.exp(-h / DROPOUT_SCALE))
    sigma_3d = math.sqrt(2 * sigma_xy**2 + sigma_z**2)
    return {
        "sigma_xy_mm": sigma_xy * 1000,
        "sigma_z_mm": sigma_z * 1000,
        "sigma_3d_mm": sigma_3d * 1000,
        "dropout_pct": dropout * 100,
    }


def compute_flight_fraction(target_h: float) -> dict:
    """Estimate flight time fraction for a ballistic bounce to target_h.

    Assumes ball bounces off paddle with restitution e and needs to reach
    height h above paddle. Contact time is ~1-2 policy steps.

    Returns:
        dict with flight_time_s, contact_time_s, flight_fraction,
        launch_vel, apex_time_s
    """
    if target_h <= 0.001:
        return {
            "flight_time_s": 0.0,
            "contact_time_s": 1.0,
            "flight_fraction": 0.0,
            "launch_vel": 0.0,
            "apex_time_s": 0.0,
        }

    # v₀ = √(2gh) to reach height h
    launch_vel = math.sqrt(2 * G * target_h)

    # Total flight time = 2·v₀/g (up + down)
    flight_time = 2 * launch_vel / G

    # Apex time = v₀/g
    apex_time = launch_vel / G

    # Contact phase ~ 2 policy steps (40ms) for bounce
    contact_time = 2.0 / POLICY_HZ

    # Total cycle time
    cycle_time = flight_time + contact_time
    flight_fraction = flight_time / cycle_time

    return {
        "flight_time_s": flight_time,
        "contact_time_s": contact_time,
        "flight_fraction": flight_fraction,
        "launch_vel": launch_vel,
        "apex_time_s": apex_time,
    }


def compute_effective_noise_exposure(target_h: float) -> dict:
    """Combined metric: noise × flight fraction = effective noise exposure.

    During contact phase, the paddle anchor provides clean measurements (R=5mm).
    During flight phase, the ball relies on D435i measurements + EKF prediction.
    The effective noise exposure captures both the noise level AND the duration
    of exposure.
    """
    noise = compute_noise_at_height(target_h)
    flight = compute_flight_fraction(target_h)

    # During flight: noise from D435i + dropout-induced stale measurements
    # Effective position uncertainty grows with dropout: when measurement is
    # missed, EKF predicts with growing covariance
    #
    # Mean height during flight ≈ 2/3 * target_h (parabolic mean)
    mean_flight_h = (2.0 / 3.0) * target_h
    flight_noise = compute_noise_at_height(mean_flight_h)

    # EKF prediction drift during dropout: σ_predict ≈ q_vel * dt * n_steps_missed
    # With 20-35% dropout at 50Hz, mean consecutive misses ≈ 1/(1-dropout) ≈ 1.25-1.54
    dropout_rate = flight_noise["dropout_pct"] / 100.0
    # Expected consecutive misses (geometric distribution mean)
    mean_consec_misses = dropout_rate / max(1.0 - dropout_rate, 0.01)
    # EKF prediction drift per miss (q_vel=0.40 m/s per step, dt=0.02s)
    q_vel = 0.40
    dt = 1.0 / POLICY_HZ
    predict_drift_mm = q_vel * dt * mean_consec_misses * 1000  # mm

    # Effective noise = measurement noise + prediction drift, weighted by flight fraction
    effective_noise_mm = (
        flight["flight_fraction"] * (flight_noise["sigma_3d_mm"] + predict_drift_mm)
        + (1.0 - flight["flight_fraction"]) * ANCHOR_R_POS * 1000
    )

    # Noise exposure metric: effective_noise × sqrt(flight_time)
    # sqrt(flight_time) because longer flights accumulate more error
    noise_exposure = effective_noise_mm * math.sqrt(max(flight["flight_time_s"], 0.001))

    return {
        "target_h": target_h,
        **noise,
        **flight,
        "mean_flight_h": mean_flight_h,
        "mean_consec_misses": mean_consec_misses,
        "predict_drift_mm": predict_drift_mm,
        "effective_noise_mm": effective_noise_mm,
        "noise_exposure": noise_exposure,
    }


def predict_gap(targets: list[float], observed: dict[str, float] | None = None) -> list[dict]:
    """Compute noise exposure for each target and optionally fit to observed gap.

    If observed gap data is provided, fits a linear model:
        predicted_gap = a * noise_exposure + b

    Args:
        targets: list of target heights in metres
        observed: optional dict mapping target_height_str → gap_percentage

    Returns:
        list of dicts with all computed metrics + predicted gap
    """
    results = []
    for h in sorted(targets):
        r = compute_effective_noise_exposure(h)
        if observed and str(h) in observed:
            r["observed_gap_pct"] = observed[str(h)]
        elif observed:
            # Try formatting with fewer decimals
            for fmt in [f"{h:.1f}", f"{h:.2f}"]:
                if fmt in observed:
                    r["observed_gap_pct"] = observed[fmt]
                    break
        results.append(r)

    # Fit linear model if observed data available
    if observed:
        xs = [r["noise_exposure"] for r in results if "observed_gap_pct" in r]
        ys = [r["observed_gap_pct"] for r in results if "observed_gap_pct" in r]
        if len(xs) >= 2:
            xs_arr = np.array(xs)
            ys_arr = np.array(ys)
            # Linear regression: gap = a * exposure + b
            A = np.column_stack([xs_arr, np.ones_like(xs_arr)])
            coeffs, residuals, _, _ = np.linalg.lstsq(A, ys_arr, rcond=None)
            a, b = coeffs
            r_squared = 1.0 - (np.sum((ys_arr - (a * xs_arr + b))**2) /
                                np.sum((ys_arr - ys_arr.mean())**2)) if len(xs) > 2 else float("nan")

            for r in results:
                r["predicted_gap_pct"] = a * r["noise_exposure"] + b
                r["fit_slope"] = a
                r["fit_intercept"] = b
                r["fit_r_squared"] = r_squared

    return results


def print_results(results: list[dict]) -> None:
    """Print a formatted results table."""
    print("\n" + "=" * 100)
    print("  PERCEPTION GAP PREDICTION — D435i noise model → expected timeout gap")
    print("=" * 100)

    # Noise characteristics table
    print("\n  Noise at target height:")
    print(f"  {'Target':>8s}  {'σ_xy':>8s}  {'σ_z':>8s}  {'σ_3D':>8s}  {'Dropout':>8s}")
    print(f"  {'(m)':>8s}  {'(mm)':>8s}  {'(mm)':>8s}  {'(mm)':>8s}  {'(%)':>8s}")
    print(f"  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    for r in results:
        print(f"  {r['target_h']:8.2f}  {r['sigma_xy_mm']:8.3f}  {r['sigma_z_mm']:8.3f}"
              f"  {r['sigma_3d_mm']:8.3f}  {r['dropout_pct']:8.1f}")

    # Flight dynamics table
    print("\n  Ballistic flight dynamics:")
    print(f"  {'Target':>8s}  {'v_launch':>8s}  {'t_flight':>8s}  {'t_apex':>8s}  {'Flight%':>8s}")
    print(f"  {'(m)':>8s}  {'(m/s)':>8s}  {'(ms)':>8s}  {'(ms)':>8s}  {'':>8s}")
    print(f"  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}  {'─' * 8}")
    for r in results:
        print(f"  {r['target_h']:8.2f}  {r['launch_vel']:8.3f}  "
              f"{r['flight_time_s'] * 1000:8.1f}  {r['apex_time_s'] * 1000:8.1f}"
              f"  {r['flight_fraction'] * 100:8.1f}")

    # Effective noise exposure table
    print("\n  Effective noise exposure (noise × flight duration):")
    print(f"  {'Target':>8s}  {'Eff.Noise':>10s}  {'Drift':>8s}  {'Exposure':>10s}", end="")
    has_obs = any("observed_gap_pct" in r for r in results)
    has_pred = any("predicted_gap_pct" in r for r in results)
    if has_obs:
        print(f"  {'Obs.Gap':>8s}", end="")
    if has_pred:
        print(f"  {'Pred.Gap':>9s}", end="")
    print()
    print(f"  {'(m)':>8s}  {'(mm)':>10s}  {'(mm)':>8s}  {'(mm·√s)':>10s}", end="")
    if has_obs:
        print(f"  {'(%)':>8s}", end="")
    if has_pred:
        print(f"  {'(%)':>9s}", end="")
    print()
    print(f"  {'─' * 8}  {'─' * 10}  {'─' * 8}  {'─' * 10}", end="")
    if has_obs:
        print(f"  {'─' * 8}", end="")
    if has_pred:
        print(f"  {'─' * 9}", end="")
    print()
    for r in results:
        print(f"  {r['target_h']:8.2f}  {r['effective_noise_mm']:10.3f}"
              f"  {r['predict_drift_mm']:8.3f}  {r['noise_exposure']:10.3f}", end="")
        if has_obs and "observed_gap_pct" in r:
            print(f"  {r['observed_gap_pct']:8.1f}", end="")
        elif has_obs:
            print(f"  {'--':>8s}", end="")
        if has_pred and "predicted_gap_pct" in r:
            print(f"  {r['predicted_gap_pct']:9.1f}", end="")
        elif has_pred:
            print(f"  {'--':>9s}", end="")
        print()

    if has_pred and "fit_r_squared" in results[0]:
        r2 = results[0].get("fit_r_squared", float("nan"))
        slope = results[0].get("fit_slope", 0)
        intercept = results[0].get("fit_intercept", 0)
        print(f"\n  Linear fit: gap = {slope:.3f} × exposure + {intercept:.1f}  (R² = {r2:.3f})")

    print("\n" + "=" * 100)


def plot_results(results: list[dict], out_path: str) -> None:
    """Generate a 2×2 figure: noise profile, flight dynamics, exposure, gap prediction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = [r["target_h"] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("D435i Perception Gap Prediction", fontsize=14, fontweight="bold")

    # Panel 1: Noise vs height
    ax = axes[0, 0]
    ax.plot(targets, [r["sigma_xy_mm"] for r in results], "o-", label="σ_xy", color="#4477AA")
    ax.plot(targets, [r["sigma_z_mm"] for r in results], "s-", label="σ_z", color="#EE6677")
    ax.plot(targets, [r["sigma_3d_mm"] for r in results], "^-", label="σ_3D", color="#228833")
    ax.set_xlabel("Target height (m)")
    ax.set_ylabel("Noise std (mm)")
    ax.set_title("D435i Noise vs Target Height")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Flight dynamics
    ax = axes[0, 1]
    ax_twin = ax.twinx()
    ax.plot(targets, [r["flight_fraction"] * 100 for r in results], "o-",
            label="Flight %", color="#4477AA")
    ax_twin.plot(targets, [r["dropout_pct"] for r in results], "s--",
                 label="Dropout %", color="#EE6677")
    ax.set_xlabel("Target height (m)")
    ax.set_ylabel("Flight fraction (%)", color="#4477AA")
    ax_twin.set_ylabel("Dropout rate (%)", color="#EE6677")
    ax.set_title("Flight Dynamics & Dropout")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 3: Effective noise exposure
    ax = axes[1, 0]
    ax.bar([str(t) for t in targets], [r["noise_exposure"] for r in results],
           color="#4477AA", alpha=0.7, label="Noise exposure")
    ax.set_xlabel("Target height (m)")
    ax.set_ylabel("Noise exposure (mm·√s)")
    ax.set_title("Effective Noise Exposure")
    ax.grid(True, alpha=0.3, axis="y")

    # Panel 4: Gap prediction vs observed
    ax = axes[1, 1]
    has_obs = any("observed_gap_pct" in r for r in results)
    has_pred = any("predicted_gap_pct" in r for r in results)

    if has_obs:
        obs_t = [r["target_h"] for r in results if "observed_gap_pct" in r]
        obs_g = [r["observed_gap_pct"] for r in results if "observed_gap_pct" in r]
        ax.scatter(obs_t, obs_g, s=80, zorder=5, color="#EE6677", label="Observed gap")

    if has_pred:
        pred_t = [r["target_h"] for r in results]
        pred_g = [r.get("predicted_gap_pct", 0) for r in results]
        ax.plot(pred_t, pred_g, "--", color="#4477AA", label="Predicted gap (linear fit)")
        if "fit_r_squared" in results[0]:
            r2 = results[0]["fit_r_squared"]
            ax.text(0.05, 0.95, f"R² = {r2:.3f}", transform=ax.transAxes,
                    fontsize=10, va="top", ha="left",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax.set_xlabel("Target height (m)")
    ax.set_ylabel("Timeout gap: oracle − d435i (%)")
    ax.set_title("Observed vs Predicted Gap")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--targets", type=str, default="0.10,0.20,0.30,0.40,0.50,0.70,1.00",
                        help="Comma-separated target heights in metres")
    parser.add_argument("--observed", type=str, default=None,
                        help='JSON dict of observed gaps: \'{"0.10": 0.3, "0.50": 18.3}\'')
    parser.add_argument("--out", type=str, default=None,
                        help="Output figure path (PNG)")
    parser.add_argument("--json-out", type=str, default=None,
                        help="Output JSON path")
    args = parser.parse_args()

    targets = [float(t.strip()) for t in args.targets.split(",")]
    observed = json.loads(args.observed) if args.observed else None

    results = predict_gap(targets, observed)
    print_results(results)

    if args.out:
        plot_results(results, args.out)

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"  JSON saved: {args.json_out}")


if __name__ == "__main__":
    main()
