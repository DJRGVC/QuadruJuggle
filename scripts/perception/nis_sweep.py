#!/usr/bin/env python3
"""NIS parameter sweep — runs nis_diagnostic.py for multiple q_vel values.

Produces a comparison table to find the q_vel that puts mean NIS closest to 3.0.

Usage:
    $C3R_BIN/gpu_lock.sh uv run --active python scripts/perception/nis_sweep.py \
        --pi2-checkpoint <path> --num_envs 256 --steps 300 --headless
"""

import argparse
import json
import os
import subprocess
import sys

SWEEP_Q_VEL = [0.15, 0.25, 0.30, 0.35, 0.50]


def main():
    parser = argparse.ArgumentParser(description="NIS q_vel sweep")
    parser.add_argument("--pi2-checkpoint", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--q_vel_values",
        type=str,
        default=None,
        help="Comma-separated q_vel values to test (default: 0.15,0.25,0.30,0.35,0.50)",
    )
    args = parser.parse_args()

    q_vals = (
        [float(v) for v in args.q_vel_values.split(",")]
        if args.q_vel_values
        else SWEEP_Q_VEL
    )

    script = os.path.join(os.path.dirname(__file__), "nis_diagnostic.py")
    results = []

    for q_vel in q_vals:
        print(f"\n{'='*70}")
        print(f"  Running q_vel = {q_vel}")
        print(f"{'='*70}\n")

        cmd = [
            sys.executable,
            script,
            "--pi2-checkpoint",
            args.pi2_checkpoint,
            "--num_envs",
            str(args.num_envs),
            "--steps",
            str(args.steps),
            "--log_interval",
            "50",
            "--q_vel",
            str(q_vel),
        ]
        if args.headless:
            cmd.append("--headless")

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        stdout = proc.stdout
        stderr = proc.stderr

        # Parse summary from stdout
        mean_nis = None
        in_band = None
        diagnosis = None
        for line in stdout.split("\n"):
            if "Overall mean NIS:" in line:
                try:
                    mean_nis = float(line.split("Overall mean NIS:")[1].split("(")[0].strip())
                except (ValueError, IndexError):
                    pass
            if "In 95% band" in line:
                try:
                    in_band = line.split("In 95% band")[1].strip()
                except IndexError:
                    pass
            if "DIAGNOSIS:" in line:
                diagnosis = line.split("DIAGNOSIS:")[1].strip()

        result = {
            "q_vel": q_vel,
            "mean_nis": mean_nis,
            "in_band": in_band,
            "diagnosis": diagnosis,
            "returncode": proc.returncode,
        }
        results.append(result)

        if proc.returncode != 0:
            print(f"  FAILED (rc={proc.returncode})")
            if stderr:
                # Print last 5 lines of stderr
                for line in stderr.strip().split("\n")[-5:]:
                    print(f"    {line}")
        else:
            print(stdout)

    # Summary table
    print(f"\n{'='*70}")
    print(f"  NIS SWEEP SUMMARY")
    print(f"{'='*70}")
    print(f"  {'q_vel':>8s}  {'mean_NIS':>10s}  {'band':>15s}  {'diagnosis'}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*15}  {'-'*30}")
    for r in results:
        nis_str = f"{r['mean_nis']:.3f}" if r["mean_nis"] is not None else "FAIL"
        band_str = r["in_band"] or "N/A"
        diag_str = r["diagnosis"] or ("ERROR rc=" + str(r["returncode"]))
        print(f"  {r['q_vel']:8.2f}  {nis_str:>10s}  {band_str:>15s}  {diag_str}")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "nis_sweep_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
