#!/usr/bin/env python3
"""Compare two pi1 checkpoints on a fixed eval protocol.

Runs eval_juggle_hier.py sequentially on checkpoint A and B, parses
the summary tables, and prints a side-by-side relative comparison.

Usage:
    uv run --active python scripts/rsl_rl/compare_pi1.py \\
        --task Isaac-BallJuggleHier-Go1-Play-v0 \\
        --pi2-checkpoint <pi2_ckpt> \\
        --ckpt-a logs/rsl_rl/go1_ball_juggle_hier/A/model_best.pt \\
        --ckpt-b logs/rsl_rl/go1_ball_juggle_hier/B/model_best.pt \\
        [--label-a "oracle"] [--label-b "d435i"] \\
        [--noise-mode-a oracle] [--noise-mode-b d435i] \\
        [--num_envs 256] [--episodes 50] [--targets "0.10,0.20,0.30"]

The script does NOT run Isaac Lab itself — it shells out to eval_juggle_hier.py
twice (eval is GPU-intensive; this script is just the comparison layer).

Output: one combined table printed to stdout with columns:
  Target | σ | A:Timeout | B:Timeout | Δpp | A:ApexRew | B:ApexRew | Δ% | A:MeanLen | B:MeanLen | Δ%
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def _build_eval_cmd(
    task: str,
    pi2_checkpoint: str,
    pi1_checkpoint: str,
    noise_mode: str,
    num_envs: int,
    episodes: int,
    targets: str | None,
    device: str,
) -> list[str]:
    script = str(Path(__file__).parent / "eval_juggle_hier.py")
    cmd = [
        "uv", "run", "--active", "python", script,
        "--task", task,
        "--pi2-checkpoint", pi2_checkpoint,
        "--checkpoint", pi1_checkpoint,
        "--num_envs", str(num_envs),
        "--episodes", str(episodes),
        "--device", device,
        "--headless",
    ]
    if noise_mode:
        cmd += ["--noise-mode", noise_mode]
    if targets:
        cmd += ["--targets", targets]
    return cmd


def _parse_eval_output(text: str) -> list[dict]:
    """Parse lines like:
      Stage D  target=0.20m  σ=0.080m  |  timeout= 85.2%  apex_rew=  3.04  mean_len= 1438.6  episodes= 100
    Returns list of dicts with keys: target, sigma, timeout_pct, apex_rew, mean_len, episodes.
    """
    results = []
    pattern = re.compile(
        r"target=([0-9.]+)m\s+σ=([0-9.]+)m\s+\|\s+"
        r"timeout=\s*([0-9.]+)%\s+"
        r"apex_rew=\s*([0-9.]+)\s+"
        r"mean_len=\s*([0-9.]+)\s+"
        r"episodes=\s*([0-9]+)"
    )
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            results.append({
                "target": float(m.group(1)),
                "sigma": float(m.group(2)),
                "timeout_pct": float(m.group(3)),
                "apex_rew": float(m.group(4)),
                "mean_len": float(m.group(5)),
                "episodes": int(m.group(6)),
            })
    return results


def _run_eval(cmd: list[str], label: str) -> tuple[str, list[dict]]:
    print(f"\n{'='*80}", flush=True)
    print(f"  Running eval for: {label}", flush=True)
    print(f"  Cmd: {' '.join(cmd)}", flush=True)
    print(f"{'='*80}", flush=True)

    result = subprocess.run(cmd, capture_output=False, text=True, stdout=subprocess.PIPE)
    output = result.stdout
    print(output, flush=True)
    if result.returncode != 0:
        print(f"[WARN] eval exited with code {result.returncode}", flush=True)
    rows = _parse_eval_output(output)
    return output, rows


def _delta_pp(a: float, b: float) -> str:
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}pp"


def _delta_pct(a: float, b: float) -> str:
    if abs(a) < 1e-6:
        return "N/A"
    d = 100.0 * (b - a) / abs(a)
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.1f}%"


def _print_comparison(rows_a: list[dict], rows_b: list[dict], label_a: str, label_b: str) -> None:
    # Align by target height
    b_by_target = {r["target"]: r for r in rows_b}
    rows_matched = [(r, b_by_target.get(r["target"])) for r in rows_a]

    col_w = 10
    header = (
        f"  {'Target':>7s}  {'σ':>6s}  "
        f"{label_a+':TO':>{col_w}s}  {label_b+':TO':>{col_w}s}  {'ΔTO':>8s}  "
        f"{label_a+':apex':>{col_w}s}  {label_b+':apex':>{col_w}s}  {'Δapex':>7s}  "
        f"{label_a+':len':>{col_w}s}  {label_b+':len':>{col_w}s}  {'Δlen':>7s}"
    )
    sep = "  " + "-" * (len(header) - 2)

    print(f"\n{'='*80}")
    print(f"  COMPARISON: {label_a} vs {label_b}")
    print(f"{'='*80}")
    print(header)
    print(sep)

    timeout_deltas = []
    for ra, rb in rows_matched:
        target = ra["target"]
        sigma = ra["sigma"]
        a_to = ra["timeout_pct"]
        a_apex = ra["apex_rew"]
        a_len = ra["mean_len"]

        if rb is None:
            print(f"  {target:7.2f}  {sigma:6.3f}  {a_to:{col_w}.1f}%  {'N/A':>{col_w}s}  {'N/A':>8s}  "
                  f"{a_apex:{col_w}.2f}  {'N/A':>{col_w}s}  {'N/A':>7s}  "
                  f"{a_len:{col_w}.1f}  {'N/A':>{col_w}s}  {'N/A':>7s}")
            continue

        b_to = rb["timeout_pct"]
        b_apex = rb["apex_rew"]
        b_len = rb["mean_len"]
        d_to = _delta_pp(a_to, b_to)
        d_apex = _delta_pct(a_apex, b_apex)
        d_len = _delta_pct(a_len, b_len)
        timeout_deltas.append(b_to - a_to)

        print(
            f"  {target:7.2f}  {sigma:6.3f}  "
            f"{a_to:{col_w}.1f}%  {b_to:{col_w}.1f}%  {d_to:>8s}  "
            f"{a_apex:{col_w}.2f}  {b_apex:{col_w}.2f}  {d_apex:>7s}  "
            f"{a_len:{col_w}.1f}  {b_len:{col_w}.1f}  {d_len:>7s}"
        )

    if timeout_deltas:
        mean_delta_to = sum(timeout_deltas) / len(timeout_deltas)
        sign = "+" if mean_delta_to >= 0 else ""
        print(sep)
        print(f"  Mean ΔTimeout: {sign}{mean_delta_to:.1f}pp  ({label_b} vs {label_a})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare two pi1 checkpoints via eval.")
    parser.add_argument("--task", default="Isaac-BallJuggleHier-Go1-Play-v0")
    parser.add_argument("--pi2-checkpoint", required=True, help="Path to frozen pi2 checkpoint")
    parser.add_argument("--ckpt-a", required=True, help="Checkpoint A (reference)")
    parser.add_argument("--ckpt-b", required=True, help="Checkpoint B (comparison)")
    parser.add_argument("--label-a", default="A", help="Label for checkpoint A")
    parser.add_argument("--label-b", default="B", help="Label for checkpoint B")
    parser.add_argument("--noise-mode-a", default="oracle", choices=["oracle", "d435i", "ekf"],
                        help="Noise mode for checkpoint A eval")
    parser.add_argument("--noise-mode-b", default="oracle", choices=["oracle", "d435i", "ekf"],
                        help="Noise mode for checkpoint B eval")
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--targets", type=str, default=None,
                        help="Comma-separated target heights, e.g. '0.10,0.20,0.30'")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    pi2_ckpt = os.path.abspath(args.pi2_checkpoint)
    ckpt_a = os.path.abspath(args.ckpt_a)
    ckpt_b = os.path.abspath(args.ckpt_b)

    cmd_a = _build_eval_cmd(
        args.task, pi2_ckpt, ckpt_a, args.noise_mode_a,
        args.num_envs, args.episodes, args.targets, args.device,
    )
    cmd_b = _build_eval_cmd(
        args.task, pi2_ckpt, ckpt_b, args.noise_mode_b,
        args.num_envs, args.episodes, args.targets, args.device,
    )

    _, rows_a = _run_eval(cmd_a, args.label_a)
    _, rows_b = _run_eval(cmd_b, args.label_b)

    if not rows_a or not rows_b:
        print("[ERROR] Could not parse eval output for one or both checkpoints.")
        sys.exit(1)

    _print_comparison(rows_a, rows_b, args.label_a, args.label_b)


if __name__ == "__main__":
    main()
