#!/usr/bin/env python3
"""Compare perception modes (oracle vs d435i vs ekf) on ball_juggle_hier.

Runs short training (50 iters default) for each mode back-to-back, collects
training metrics AND perception diagnostics (EKF estimation error vs GT),
then prints a comparison table.

Usage:
    $C3R_BIN/gpu_lock.sh uv run --active python scripts/perception/compare_perception_modes.py \
        --pi2-checkpoint logs/rsl_rl/go1_torso_tracking/<run>/model_best.pt \
        --num_envs 4096 --headless --max_iterations 50

    # Specific modes only:
    ... --modes ekf d435i
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare perception modes.")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=50)
parser.add_argument(
    "--pi2-checkpoint", type=str, required=True,
    help="Path to the trained pi2 (torso-tracking) checkpoint.",
)
parser.add_argument(
    "--modes", nargs="+", default=["oracle", "d435i", "ekf"],
    help="Perception modes to compare.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports after AppLauncher ──────────────────────────────────────────────

import json
import os
import time

# Worktree isolation
_OUR_SRC = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "source", "go1_ball_balance",
))
sys.path.insert(0, _OUR_SRC)

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import go1_ball_balance  # noqa: F401 — registers gym envs

from go1_ball_balance.perception import (
    BallObsNoiseCfg,
    ball_pos_perceived,
    ball_vel_perceived,
    reset_perception_pipeline,
)
from go1_ball_balance.tasks.ball_juggle_hier.ball_juggle_hier_env_cfg import (
    BallJuggleHierEnvCfg,
    _PADDLE_OFFSET_B,
)
from go1_ball_balance.tasks.ball_juggle_hier.agents.rsl_rl_ppo_cfg import (
    BallJuggleHierPPORunnerCfg,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DIAG_LOG_INTERVAL = 10  # log perception diagnostics every N iterations


def patch_env_cfg(env_cfg: BallJuggleHierEnvCfg, mode: str) -> None:
    """Patch env_cfg observation terms for the given perception mode."""
    noise_cfg = BallObsNoiseCfg(mode=mode)

    env_cfg.observations.policy.ball_pos = ObsTerm(
        func=ball_pos_perceived,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "noise_cfg": noise_cfg,
        },
    )
    env_cfg.observations.policy.ball_vel = ObsTerm(
        func=ball_vel_perceived,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "noise_cfg": noise_cfg,
        },
    )

    if mode == "ekf":
        env_cfg.events.reset_perception = EventTerm(
            func=reset_perception_pipeline,
            mode="reset",
            params={
                "ball_cfg": SceneEntityCfg("ball"),
                "robot_cfg": SceneEntityCfg("robot"),
                "paddle_offset_b": _PADDLE_OFFSET_B,
            },
        )


def run_mode(mode: str) -> dict:
    """Run training for one perception mode, return collected metrics."""
    print(f"\n{'='*60}")
    print(f"  Running mode: {mode}")
    print(f"  num_envs={args_cli.num_envs}, max_iterations={args_cli.max_iterations}")
    print(f"{'='*60}\n")

    env_cfg = BallJuggleHierEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = "cuda:0"

    pi2_path = os.path.abspath(args_cli.pi2_checkpoint)
    if not os.path.isfile(pi2_path):
        raise FileNotFoundError(f"pi2 checkpoint not found: {pi2_path}")
    env_cfg.actions.torso_cmd.pi2_checkpoint = pi2_path

    patch_env_cfg(env_cfg, mode)

    env = gym.make("Isaac-BallJuggleHier-Go1-v0", cfg=env_cfg)

    # Enable diagnostics for EKF mode
    if mode == "ekf":
        env.unwrapped._perception_diagnostics_enabled = True

    agent_cfg = BallJuggleHierPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.device = "cuda:0"
    agent_cfg.experiment_name = f"perception_compare_{mode}"

    log_dir = os.path.join(
        "logs", "rsl_rl", "perception_compare", mode,
    )
    os.makedirs(log_dir, exist_ok=True)

    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(
        env_wrapped, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device,
    )

    # Metric collection
    metrics = {"mode": mode}
    ep_lengths = []
    reward_sums = []
    perception_diags = []
    iteration_count = [0]

    original_log = runner.log

    def _metric_log(locs, *args, **kwargs):
        original_log(locs, *args, **kwargs)
        iteration_count[0] += 1
        ep_infos = locs.get("ep_infos", [])
        if ep_infos:
            for key in ep_infos[0]:
                if "len" in key.lower() or "length" in key.lower():
                    vals = [float(ep[key]) for ep in ep_infos if key in ep]
                    if vals:
                        ep_lengths.append(sum(vals) / len(vals))
                if "rew" in key.lower() and "mean" not in key.lower():
                    vals = [float(ep[key]) for ep in ep_infos if key in ep]
                    if vals:
                        reward_sums.append(sum(vals) / len(vals))

        # Collect perception diagnostics periodically
        if mode == "ekf" and iteration_count[0] % DIAG_LOG_INTERVAL == 0:
            pipeline = getattr(env.unwrapped, "_perception_pipeline", None)
            if pipeline is not None:
                diag = pipeline.diagnostics
                if diag:
                    diag["iteration"] = iteration_count[0]
                    perception_diags.append(diag)
                    print(f"  [DIAG iter {iteration_count[0]}] "
                          f"EKF pos RMSE: {diag['pos_rmse_ekf_mm']:.1f}mm, "
                          f"raw: {diag['pos_rmse_raw_mm']:.1f}mm, "
                          f"improvement: {diag['ekf_improvement_pct']:.0f}%")

    runner.log = _metric_log

    start = time.time()
    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
    elapsed = time.time() - start

    metrics["elapsed_s"] = round(elapsed, 1)
    if ep_lengths:
        metrics["mean_ep_len_final10"] = round(sum(ep_lengths[-10:]) / len(ep_lengths[-10:]), 1)
        metrics["mean_ep_len_all"] = round(sum(ep_lengths) / len(ep_lengths), 1)
        metrics["max_ep_len"] = round(max(ep_lengths), 1)
    if reward_sums:
        metrics["mean_reward_final10"] = round(sum(reward_sums[-10:]) / len(reward_sums[-10:]), 1)

    if perception_diags:
        # Average of all diagnostic snapshots
        avg_diag = {}
        for key in perception_diags[0]:
            if key == "iteration" or key == "num_samples":
                continue
            vals = [d[key] for d in perception_diags if key in d]
            avg_diag[f"avg_{key}"] = round(sum(vals) / len(vals), 2)
        metrics.update(avg_diag)

        # Save full diagnostic log
        diag_path = os.path.join(log_dir, "perception_diagnostics.json")
        with open(diag_path, "w") as f:
            json.dump(perception_diags, f, indent=2)
        print(f"  [DIAG] Full diagnostics saved to {diag_path}")

    env.close()
    return metrics


def main():
    all_results = {}
    for mode in args_cli.modes:
        all_results[mode] = run_mode(mode)

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*70}")

    # Collect all keys
    all_keys = set()
    for r in all_results.values():
        all_keys.update(r.keys())
    all_keys.discard("mode")
    sorted_keys = sorted(all_keys)

    # Header
    modes = list(all_results.keys())
    header = f"  {'metric':<30s}" + "".join(f"  {m:>12s}" for m in modes)
    print(header)
    print("  " + "-" * (30 + 14 * len(modes)))

    for key in sorted_keys:
        row = f"  {key:<30s}"
        for m in modes:
            val = all_results[m].get(key, "—")
            if isinstance(val, float):
                row += f"  {val:>12.2f}"
            else:
                row += f"  {str(val):>12s}"
        print(row)

    print(f"{'='*70}\n")

    # Save results
    results_path = os.path.join("logs", "rsl_rl", "perception_compare", "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
    simulation_app.close()
