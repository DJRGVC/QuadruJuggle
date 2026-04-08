#!/usr/bin/env python3
"""Read sweep_q_vel results JSON and print recommended EKF config changes.

Usage:
    python scripts/perception/apply_sweep_results.py logs/perception/sweep_q_vel_*.json
"""

import json
import sys
import glob


def main():
    if len(sys.argv) < 2:
        # Auto-find latest sweep result
        files = sorted(glob.glob("logs/perception/sweep_q_vel_*.json"))
        if not files:
            print("No sweep results found in logs/perception/")
            sys.exit(1)
        path = files[-1]
    else:
        path = sys.argv[1]

    with open(path) as f:
        data = json.load(f)

    print(f"Sweep results from: {path}")
    print(f"Config: {json.dumps(data['config'], indent=2)}")
    print()

    # Summary table
    print(f"{'q_vel':>10s}  {'NIS':>8s}  {'Flight':>8s}  {'Contact':>8s}  "
          f"{'EKF mm':>8s}  {'Raw mm':>8s}  {'Impr%':>7s}")
    print(f"{'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}")
    for r in data["sweep"]:
        marker = " ←" if r["q_vel"] == data["best_q_vel"] else ""
        print(f"{r['q_vel']:10.4f}  {r['mean_nis']:8.3f}  {r['flight_nis']:8.3f}  "
              f"{r['contact_nis']:8.3f}  {r['ekf_rmse_mm']:8.2f}  {r['raw_rmse_mm']:8.2f}  "
              f"{r['improvement_pct']:7.1f}{marker}")

    best = data["best_result"]
    print(f"\nBest q_vel = {data['best_q_vel']:.4f}")
    print(f"  Flight NIS = {best['flight_nis']:.3f} (target ~3.0)")
    print(f"  EKF RMSE = {best['ekf_rmse_mm']:.1f}mm vs raw {best['raw_rmse_mm']:.1f}mm")
    print(f"  Improvement = {best['improvement_pct']:.1f}%")

    ekf_beats_raw = best["ekf_rmse_mm"] < best["raw_rmse_mm"]
    print(f"\n  EKF {'beats' if ekf_beats_raw else 'LOSES to'} raw d435i noise")

    print(f"\nRecommended BallEKFConfig changes:")
    print(f"  q_vel = {data['best_q_vel']:.4f}  (was 0.40)")
    print(f"  q_vel_contact = {data['config']['q_vel_contact']}")
    print(f"  q_vel_post_contact = {data['config']['q_vel_post_contact']}")
    print(f"  post_contact_steps = {data['config']['post_contact_steps']}")


if __name__ == "__main__":
    main()
