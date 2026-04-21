# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train pi1 (ball planner) for the hierarchical ball-juggle task.

pi1 outputs 6D torso commands; frozen pi2 (torso tracker) converts them
to 12D joint targets.  The pi2 checkpoint must be provided via --pi2-checkpoint.

Uses the same 14-stage juggling curriculum as train_juggle.py.

Usage:
    uv run --active python scripts/rsl_rl/train_juggle_hier.py \\
        --task Isaac-BallJuggleHier-Go1-v0 \\
        --pi2-checkpoint logs/rsl_rl/go1_torso_tracking/<run>/model_best.pt \\
        --num_envs 12288 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train pi1 for hierarchical ball juggling.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Isaac-BallJuggleHier-Go1-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--export_io_descriptors", action="store_true", default=False)
parser.add_argument("--ray-proc-id", "-rid", type=int, default=None)
parser.add_argument(
    "--pi2-checkpoint", type=str, required=True,
    help="Path to the trained pi2 (torso-tracking) checkpoint.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── version check ─────────────────────────────────────────────────────────────
import importlib.metadata as metadata
import platform

from packaging import version

RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install RSL-RL {RSL_RL_VERSION} (installed: {installed_version}).\n"
        f"Run: {' '.join(cmd)}"
    )
    exit(1)

import logging
import os
import time
from datetime import datetime

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

logger = logging.getLogger(__name__)

import go1_ball_balance  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# ── Juggling curriculum (same 14 stages as flat train_juggle.py) ─────────────
_BJ_STAGES = [
    # target_h  sigma   xy_std  vel_xy_std
    (0.10,      0.20,   0.020,  0.00),   # A
    (0.17,      0.18,   0.025,  0.00),   # B
    (0.24,      0.15,   0.028,  0.00),   # C
    (0.32,      0.13,   0.032,  0.00),   # D
    (0.40,      0.11,   0.036,  0.00),   # E
    (0.48,      0.10,   0.040,  0.00),   # F
    (0.56,      0.09,   0.045,  0.02),   # G
    (0.64,      0.08,   0.050,  0.04),   # H
    (0.72,      0.07,   0.055,  0.06),   # I
    (0.80,      0.07,   0.060,  0.08),   # J
    (0.86,      0.06,   0.068,  0.10),   # K
    (0.92,      0.06,   0.075,  0.12),   # L
    (0.96,      0.05,   0.082,  0.15),   # M
    (1.00,      0.05,   0.090,  0.18),   # N
]
_BJ_THRESHOLD      = 0.75
_BJ_APEX_THRESHOLD = 5.0
_BJ_SUSTAIN    = 15
_BJ_TRANSITION = 10

# Early stopping
_ES_PATIENCE = 300     # longer patience for hierarchical (pi2 adds lag)
_ES_MIN_DELTA = 0.5


class EarlyStopException(Exception):
    pass


def _bj_set_params(
    rl_env, target_h: float, sigma: float, xy_std: float, vel_xy_std: float
) -> None:
    """Write reward and event params directly."""
    for i, name in enumerate(rl_env.reward_manager._term_names):
        if name == "ball_apex_height":
            rl_env.reward_manager._term_cfgs[i].params["target_height"] = target_h
            rl_env.reward_manager._term_cfgs[i].params["std"] = sigma
            break
    for term_cfg in rl_env.event_manager._mode_term_cfgs.get("reset", []):
        if hasattr(term_cfg, "params") and "xy_std" in (term_cfg.params or {}):
            term_cfg.params["xy_std"] = xy_std
        if hasattr(term_cfg, "params") and "vel_xy_std" in (term_cfg.params or {}):
            term_cfg.params["vel_xy_std"] = vel_xy_std


def _bj_apply_stage(rl_env, stage_idx: int) -> None:
    target_h, sigma, xy_std, vel_xy_std = _BJ_STAGES[stage_idx]
    _bj_set_params(rl_env, target_h, sigma, xy_std, vel_xy_std)
    print(
        f"\n[HIER-JUGGLE-CURRICULUM] Stage {stage_idx}/{len(_BJ_STAGES) - 1}  "
        f"target={target_h:.2f} m  sigma={sigma:.3f} m  "
        f"xy_std={xy_std:.3f} m  vel_xy={vel_xy_std:.3f} m/s\n"
    )


def _bj_install_curriculum(runner) -> None:
    """Monkey-patch runner.log() for juggling curriculum + early stopping."""
    try:
        rl_env = runner.env.unwrapped
    except AttributeError:
        print("[HIER-JUGGLE-CURRICULUM] Could not access unwrapped env — curriculum disabled.")
        return

    original_log = runner.log
    _STAGE_LETTERS = "ABCDEFGHIJKLMN"
    state = {
        "stage":       0,
        "above_count": 0,
        "stage_iter":  0,
        "old_stage":   None,
        "trans_iter":  0,
        "best_reward": float("-inf"),
        "no_improve":  0,
    }

    log_dir = getattr(runner, "log_dir", None)

    def _wrapped_log(locs, *args, **kwargs):
        original_log(locs, *args, **kwargs)

        state["stage_iter"] += 1
        s      = state["stage"]
        s_iter = state["stage_iter"]
        label  = _STAGE_LETTERS[s] if s < len(_STAGE_LETTERS) else str(s)
        final  = s >= len(_BJ_STAGES) - 1

        # ── continue in-progress parameter transition ─────────────────────────
        if state["old_stage"] is not None:
            state["trans_iter"] += 1
            alpha = min(state["trans_iter"] / _BJ_TRANSITION, 1.0)
            o = _BJ_STAGES[state["old_stage"]]
            n = _BJ_STAGES[s]
            _bj_set_params(
                rl_env,
                target_h   = o[0] + alpha * (n[0] - o[0]),
                sigma      = o[1] + alpha * (n[1] - o[1]),
                xy_std     = o[2] + alpha * (n[2] - o[2]),
                vel_xy_std = o[3] + alpha * (n[3] - o[3]),
            )
            if state["trans_iter"] >= _BJ_TRANSITION:
                state["old_stage"] = None

        # ── extract time_out fraction and ball-apex reward ────────────────────
        ep_infos = locs.get("ep_infos", [])
        time_out_frac = None
        ball_apex_frac = None
        if ep_infos:
            for key in ep_infos[0]:
                if "time_out" in key.lower() or "timeout" in key.lower():
                    vals = [float(ep[key]) for ep in ep_infos if key in ep]
                    if vals:
                        time_out_frac = sum(vals) / len(vals)
                if "ball_apex" in key.lower():
                    vals = [float(ep[key]) for ep in ep_infos if key in ep]
                    if vals:
                        ball_apex_frac = sum(vals) / len(vals)

        # ── threshold + sustain check ─────────────────────────────────────────
        if not final and time_out_frac is not None:
            juggling_ok = (ball_apex_frac is None) or (ball_apex_frac >= _BJ_APEX_THRESHOLD)
            if time_out_frac >= _BJ_THRESHOLD and juggling_ok:
                state["above_count"] += 1
            else:
                state["above_count"] = 0

            if state["above_count"] >= _BJ_SUSTAIN:
                state["old_stage"]   = s
                state["trans_iter"]  = 0
                state["stage"]      += 1
                state["above_count"] = 0
                state["stage_iter"]  = 0
                # Reset early stop — task just got harder
                state["best_reward"] = float("-inf")
                state["no_improve"]  = 0
                s     = state["stage"]
                label = _STAGE_LETTERS[s] if s < len(_STAGE_LETTERS) else str(s)
                final = s >= len(_BJ_STAGES) - 1
                n = _BJ_STAGES[s]
                print(
                    f"\n[HIER-JUGGLE-CURRICULUM] Stage {s-1}→{s} ({_STAGE_LETTERS[s-1]}→{label})  "
                    f"blending over {_BJ_TRANSITION} iters  "
                    f"target: {n[0]:.2f} m  sigma={n[1]:.3f} m  "
                    f"xy={n[2]:.3f} m  vel_xy={n[3]:.3f} m/s\n"
                )

        # ── early stopping ─────────────────────────────────────────────────────
        # RSL-RL stores episode returns in locs["rewbuffer"] (list), not "mean_reward"
        rew_buf = locs.get("rewbuffer", [])
        if rew_buf:
            import statistics
            mean_reward = statistics.mean(rew_buf)
        else:
            mean_reward = 0.0
        if mean_reward > state["best_reward"] + _ES_MIN_DELTA:
            state["best_reward"] = mean_reward
            state["no_improve"] = 0
            if log_dir:
                best_path = os.path.join(log_dir, "model_best.pt")
                runner.save(best_path)
        else:
            state["no_improve"] += 1

        # ── per-iteration status line ─────────────────────────────────────────
        if state["old_stage"] is not None:
            alpha = min(state["trans_iter"] / _BJ_TRANSITION, 1.0)
            o = _BJ_STAGES[state["old_stage"]]
            n = _BJ_STAGES[s]
            cur_tgt = o[0] + alpha * (n[0] - o[0])
            cur_sig = o[1] + alpha * (n[1] - o[1])
            cur_xy  = o[2] + alpha * (n[2] - o[2])
            cur_vel = o[3] + alpha * (n[3] - o[3])
            blend_tag     = f" [→{label} blend {state['trans_iter']}/{_BJ_TRANSITION}]"
            s_display     = state["old_stage"]
            label_display = _STAGE_LETTERS[s_display] if s_display < len(_STAGE_LETTERS) else str(s_display)
        else:
            cur_tgt, cur_sig, cur_xy, cur_vel = _BJ_STAGES[s]
            blend_tag     = ""
            s_display     = s
            label_display = label

        to_str   = f"{time_out_frac:.0%}" if time_out_frac is not None else "n/a"
        apex_str = f"{ball_apex_frac:.2f}" if ball_apex_frac is not None else "n/a"
        if final and state["old_stage"] is None:
            suffix = "FINAL STAGE"
        else:
            juggling_ok = (ball_apex_frac is None) or (ball_apex_frac >= _BJ_APEX_THRESHOLD)
            suffix = (
                f"threshold {state['above_count']:>2}/{_BJ_SUSTAIN}  "
                f"timeout={to_str}/{_BJ_THRESHOLD:.0%}  "
                f"apex={apex_str}/{_BJ_APEX_THRESHOLD:.1f} ({'OK' if juggling_ok else 'NO'})"
            )
        print(
            f"[HIER-JUGGLE-CURRICULUM] Stage {s_display}/{len(_BJ_STAGES) - 1} ({label_display}) | "
            f"iter {s_iter:>4} in stage | "
            f"target={cur_tgt:.2f} m  sigma={cur_sig:.3f} m  "
            f"xy={cur_xy:.3f} m  vel_xy={cur_vel:.3f} m/s | "
            f"{suffix}{blend_tag} | "
            f"ES {state['no_improve']}/{_ES_PATIENCE}"
        )

        if state["no_improve"] >= _ES_PATIENCE:
            print(
                f"\n[EARLY-STOP] No improvement for {_ES_PATIENCE} iterations.  "
                f"Best reward: {state['best_reward']:.3f}.  Stopping training.\n"
            )
            raise EarlyStopException()

    runner.log = _wrapped_log
    print(
        f"[HIER-JUGGLE-CURRICULUM] Installed.  "
        f"Stages: {len(_BJ_STAGES)}  "
        f"Threshold: {_BJ_THRESHOLD:.0%}  "
        f"Sustain: {_BJ_SUSTAIN} iters  "
        f"Transition: {_BJ_TRANSITION} iters  "
        f"Early-stop patience: {_ES_PATIENCE} iters"
    )
# ─────────────────────────────────────────────────────────────────────────────


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train pi1 with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError("Distributed training requires GPU device.")

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device   = f"cuda:{app_launcher.local_rank}"
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # ── Inject pi2 checkpoint into action config ────────────────────────────
    pi2_path = os.path.abspath(args_cli.pi2_checkpoint)
    if not os.path.isfile(pi2_path):
        raise FileNotFoundError(f"pi2 checkpoint not found: {pi2_path}")
    env_cfg.actions.torso_cmd.pi2_checkpoint = pi2_path
    print(f"[INFO] pi2 checkpoint: {pi2_path}")

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors

    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Install juggling curriculum + early stopping
    _bj_install_curriculum(runner)

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except EarlyStopException:
        runner.save(os.path.join(log_dir, "model_early_stop.pt"))
        print("[INFO] Saved early-stop checkpoint.")

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
