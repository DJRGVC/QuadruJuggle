#!/usr/bin/env python3
"""Integration test: ball_juggle_hier with EKF perception pipeline.

Patches the env_cfg to use perception.ball_pos_perceived / ball_vel_perceived
with mode="ekf", adds the reset_perception_pipeline event, and runs a short
training (50 iterations by default).

Compares against the oracle baseline (policy agent's iter_001/002: mean_len≈1470-1500).

Usage:
    uv run --active python scripts/perception/test_ekf_integration.py \
        --pi2-checkpoint logs/rsl_rl/go1_torso_tracking/<run>/model_best.pt \
        --num_envs 4096 --headless --max_iterations 50
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="EKF perception integration test.")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=50)
parser.add_argument(
    "--pi2-checkpoint", type=str, required=True,
    help="Path to the trained pi2 (torso-tracking) checkpoint.",
)
parser.add_argument(
    "--perception-mode", type=str, default="ekf",
    choices=["oracle", "d435i", "ekf"],
    help="Perception mode to test.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── imports after AppLauncher ──────────────────────────────────────────────

import os
import sys
import time

# Worktree isolation: ensure our perception code is found, not the policy
# agent's worktree (which the editable install points to).
_OUR_SRC = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "source", "go1_ball_balance",
))
sys.path.insert(0, _OUR_SRC)

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.io import dump_yaml

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


def patch_env_cfg_for_perception(
    env_cfg: BallJuggleHierEnvCfg,
    mode: str = "ekf",
) -> None:
    """Patch observation and event terms to use perception pipeline."""
    noise_cfg = BallObsNoiseCfg(mode=mode)

    # Replace ball obs terms with perception-aware versions
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

    # Add EKF reset event (only for ekf mode)
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

    print(f"[PERCEPTION] Patched obs to mode={mode!r}")


def main():
    # ── Build env config ──────────────────────────────────────────────────
    env_cfg = BallJuggleHierEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = "cuda:0"

    # Inject pi2 checkpoint
    pi2_path = os.path.abspath(args_cli.pi2_checkpoint)
    if not os.path.isfile(pi2_path):
        raise FileNotFoundError(f"pi2 checkpoint not found: {pi2_path}")
    env_cfg.actions.torso_cmd.pi2_checkpoint = pi2_path
    print(f"[INFO] pi2 checkpoint: {pi2_path}")

    # Patch for perception
    patch_env_cfg_for_perception(env_cfg, mode=args_cli.perception_mode)

    # ── Create env ────────────────────────────────────────────────────────
    env = gym.make(
        "Isaac-BallJuggleHier-Go1-v0",
        cfg=env_cfg,
    )

    # ── Agent config ──────────────────────────────────────────────────────
    agent_cfg = BallJuggleHierPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations
    agent_cfg.device = "cuda:0"
    agent_cfg.experiment_name = "go1_ball_juggle_hier_ekf_test"

    log_dir = os.path.join(
        "logs", "rsl_rl", agent_cfg.experiment_name,
        f"test_{args_cli.perception_mode}",
    )
    os.makedirs(log_dir, exist_ok=True)

    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(
        env_wrapped, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device,
    )

    # ── Collect metrics ───────────────────────────────────────────────────
    metrics = {"mode": args_cli.perception_mode, "num_envs": args_cli.num_envs}

    original_log = runner.log
    ep_lengths = []

    def _metric_log(locs, *args, **kwargs):
        original_log(locs, *args, **kwargs)
        ep_infos = locs.get("ep_infos", [])
        if ep_infos:
            for key in ep_infos[0]:
                if "len" in key.lower() or "length" in key.lower():
                    vals = [float(ep[key]) for ep in ep_infos if key in ep]
                    if vals:
                        mean_len = sum(vals) / len(vals)
                        ep_lengths.append(mean_len)
                if "time_out" in key.lower() or "timeout" in key.lower():
                    vals = [float(ep[key]) for ep in ep_infos if key in ep]
                    if vals:
                        metrics["last_timeout_frac"] = sum(vals) / len(vals)

    runner.log = _metric_log

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EKF Integration Test — mode={args_cli.perception_mode}")
    print(f"  num_envs={args_cli.num_envs}, max_iterations={args_cli.max_iterations}")
    print(f"{'='*60}\n")

    start_time = time.time()
    runner.learn(num_learning_iterations=args_cli.max_iterations, init_at_random_ep_len=True)
    elapsed = time.time() - start_time

    # ── Report ────────────────────────────────────────────────────────────
    metrics["elapsed_s"] = round(elapsed, 1)
    if ep_lengths:
        metrics["mean_ep_len_final10"] = round(sum(ep_lengths[-10:]) / len(ep_lengths[-10:]), 1)
        metrics["mean_ep_len_all"] = round(sum(ep_lengths) / len(ep_lengths), 1)
        metrics["max_ep_len"] = round(max(ep_lengths), 1)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}\n")

    env.close()
    return metrics


if __name__ == "__main__":
    results = main()
    simulation_app.close()
