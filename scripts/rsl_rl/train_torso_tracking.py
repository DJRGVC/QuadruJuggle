# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train RL agent for the torso-tracking task (pi2) with command-range curriculum
and early stopping.

Curriculum advances through 4 stages (A→D) when mean tracking reward ≥ 0.7
for 10 consecutive iterations.  Each stage widens the command ranges from
narrow (easy) to full (hard).

Early stopping: if mean reward does not improve by min_delta=0.5 for 100
consecutive iterations, training halts and saves the best checkpoint.

  Stage  h_range        roll/pitch     h_dot          omega
  A      [0.36, 0.42]   [-0.1, 0.1]   [-0.1, 0.1]   [-0.5, 0.5]
  B      [0.32, 0.46]   [-0.2, 0.2]   [-0.4, 0.4]   [-1.0, 1.0]
  C      [0.28, 0.48]   [-0.3, 0.3]   [-0.7, 0.7]   [-2.0, 2.0]
  D      [0.25, 0.50]   [-0.4, 0.4]   [-1.0, 1.0]   [-3.0, 3.0]

Usage:
    uv run --active python scripts/rsl_rl/train_torso_tracking.py \\
        --task Isaac-TorsoTracking-Go1-v0 --num_envs 12288 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train an RL agent for torso tracking (pi2).")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Isaac-TorsoTracking-Go1-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--distributed", action="store_true", default=False)
parser.add_argument("--export_io_descriptors", action="store_true", default=False)
parser.add_argument("--ray-proc-id", "-rid", type=int, default=None)
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


# ── Torso-tracking command-range curriculum ──────────────────────────────────
# Stages define the command ranges for each dimension.
# Format: (h_lo, h_hi, rp_lo, rp_hi, hd_lo, hd_hi, om_lo, om_hi)
_TT_STAGES = [
    # h_lo   h_hi   rp_lo  rp_hi  hd_lo  hd_hi  om_lo  om_hi
    (0.36,   0.42,  -0.1,  0.1,   -0.1,  0.1,   -0.5,  0.5),    # A — narrow
    (0.32,   0.46,  -0.2,  0.2,   -0.4,  0.4,   -1.0,  1.0),    # B — medium
    (0.28,   0.48,  -0.3,  0.3,   -0.7,  0.7,   -2.0,  2.0),    # C — wide
    (0.25,   0.50,  -0.4,  0.4,   -1.0,  1.0,   -3.0,  3.0),    # D — full
]
_TT_THRESHOLD = 0.7    # mean tracking reward fraction to advance
_TT_HEIGHT_THRESHOLD = 0.3  # minimum height_tracking reward to advance (prevents ignoring height)
_TT_SUSTAIN = 10       # consecutive iterations at threshold
_TT_TRANSITION = 5     # iterations to blend between stages

# Early stopping config
_ES_PATIENCE = 99999   # effectively disabled
_ES_MIN_DELTA = 0.5    # minimum total reward improvement to count


class EarlyStopException(Exception):
    """Raised to cleanly halt training when early stopping triggers."""
    pass


def _tt_apply_stage(rl_env, stage_idx: int) -> None:
    """Set command ranges to a specific stage."""
    s = _TT_STAGES[stage_idx]
    rl_env._torso_cmd_ranges["h"] = [s[0], s[1]]
    rl_env._torso_cmd_ranges["roll"] = [s[2], s[3]]
    rl_env._torso_cmd_ranges["pitch"] = [s[2], s[3]]
    rl_env._torso_cmd_ranges["h_dot"] = [s[4], s[5]]
    rl_env._torso_cmd_ranges["omega_roll"] = [s[6], s[7]]
    rl_env._torso_cmd_ranges["omega_pitch"] = [s[6], s[7]]
    print(
        f"\n[TORSO-CURRICULUM] Stage {stage_idx}/{len(_TT_STAGES) - 1}  "
        f"h=[{s[0]:.2f},{s[1]:.2f}]  roll/pitch=[{s[2]:.1f},{s[3]:.1f}]  "
        f"h_dot=[{s[4]:.1f},{s[5]:.1f}]  omega=[{s[6]:.1f},{s[7]:.1f}]\n"
    )


def _tt_blend_stages(rl_env, old_idx: int, new_idx: int, alpha: float) -> None:
    """Linearly interpolate command ranges between two stages."""
    o = _TT_STAGES[old_idx]
    n = _TT_STAGES[new_idx]
    blended = tuple(o[i] + alpha * (n[i] - o[i]) for i in range(8))
    rl_env._torso_cmd_ranges["h"] = [blended[0], blended[1]]
    rl_env._torso_cmd_ranges["roll"] = [blended[2], blended[3]]
    rl_env._torso_cmd_ranges["pitch"] = [blended[2], blended[3]]
    rl_env._torso_cmd_ranges["h_dot"] = [blended[4], blended[5]]
    rl_env._torso_cmd_ranges["omega_roll"] = [blended[6], blended[7]]
    rl_env._torso_cmd_ranges["omega_pitch"] = [blended[6], blended[7]]


def _tt_install_curriculum(runner) -> None:
    """Monkey-patch runner.log() for curriculum + early stopping."""
    try:
        rl_env = runner.env.unwrapped
    except AttributeError:
        print("[TORSO-CURRICULUM] Could not access unwrapped env — curriculum disabled.")
        return

    # Ensure command buffer exists with Stage A ranges
    from go1_ball_balance.tasks.torso_tracking.mdp.commands import _ensure_cmd_buffer
    _ensure_cmd_buffer(rl_env)
    _tt_apply_stage(rl_env, 0)

    original_log = runner.log
    _STAGE_LETTERS = "ABCD"
    state = {
        "stage":       0,
        "above_count": 0,
        "stage_iter":  0,
        "old_stage":   None,
        "trans_iter":  0,
        # Early stopping
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
        final  = s >= len(_TT_STAGES) - 1

        # ── continue in-progress parameter transition ─────────────────────────
        if state["old_stage"] is not None:
            state["trans_iter"] += 1
            alpha = min(state["trans_iter"] / _TT_TRANSITION, 1.0)
            _tt_blend_stages(rl_env, state["old_stage"], s, alpha)
            if state["trans_iter"] >= _TT_TRANSITION:
                state["old_stage"] = None

        # ── extract per-term tracking rewards + total reward ────────────────
        ep_infos = locs.get("ep_infos", [])
        mean_reward = locs.get("mean_reward", None)

        tracking_keys = ["height_tracking", "roll_tracking", "pitch_tracking",
                         "height_vel_tracking", "roll_rate_tracking", "pitch_rate_tracking"]
        tracking_vals = {}
        time_out_frac = None
        height_tracking_val = None

        if ep_infos:
            for key in ep_infos[0]:
                if "time_out" in key.lower() or "timeout" in key.lower():
                    vals = [float(ep[key]) for ep in ep_infos if key in ep]
                    if vals:
                        time_out_frac = sum(vals) / len(vals)
                for tk in tracking_keys:
                    if tk in key.lower():
                        vals = [float(ep[key]) for ep in ep_infos if key in ep]
                        if vals:
                            tracking_vals[tk] = sum(vals) / len(vals)

        # Height tracking is the hardest — extract it specifically
        height_tracking_val = tracking_vals.get("height_tracking", None)

        # Compute mean of all tracking terms for curriculum signal
        tracking_mean = None
        if tracking_vals:
            tracking_mean = sum(tracking_vals.values()) / len(tracking_vals)

        # ── curriculum: advance only if ALL tracking is good, including height ─
        if not final and tracking_mean is not None:
            height_ok = (height_tracking_val is None) or (height_tracking_val >= _TT_HEIGHT_THRESHOLD)
            if tracking_mean >= _TT_THRESHOLD and height_ok:
                state["above_count"] += 1
            else:
                state["above_count"] = 0

            if state["above_count"] >= _TT_SUSTAIN:
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
                n = _TT_STAGES[s]
                print(
                    f"\n[TORSO-CURRICULUM] Stage {s-1}→{s} ({_STAGE_LETTERS[s-1]}→{label})  "
                    f"blending over {_TT_TRANSITION} iters  "
                    f"h=[{n[0]:.2f},{n[1]:.2f}]  "
                    f"roll/pitch=[{n[2]:.1f},{n[3]:.1f}]  "
                    f"h_dot=[{n[4]:.1f},{n[5]:.1f}]  "
                    f"omega=[{n[6]:.1f},{n[7]:.1f}]\n"
                )

        # ── early stopping ─────────────────────────────────────────────────────
        # RSL-RL stores episode returns in locs["rewbuffer"] (list)
        if mean_reward is None:
            rew_buf = locs.get("rewbuffer", [])
            if rew_buf:
                import statistics
                mean_reward = statistics.mean(rew_buf)
        reward_for_es = mean_reward if mean_reward is not None else 0.0
        if reward_for_es > state["best_reward"] + _ES_MIN_DELTA:
            state["best_reward"] = reward_for_es
            state["no_improve"] = 0
            # Save best checkpoint
            if log_dir:
                best_path = os.path.join(log_dir, "model_best.pt")
                runner.save(best_path)
        else:
            state["no_improve"] += 1

        # ── per-iteration status line ─────────────────────────────────────────
        trk_str = f"{tracking_mean:.3f}" if tracking_mean is not None else "n/a"
        ht_str  = f"{height_tracking_val:.3f}" if height_tracking_val is not None else "n/a"
        to_str  = f"{time_out_frac:.0%}" if time_out_frac is not None else "n/a"
        rew_str = f"{mean_reward:.1f}" if mean_reward is not None else "n/a"
        blend_tag = ""
        if state["old_stage"] is not None:
            blend_tag = f" [→{label} blend {state['trans_iter']}/{_TT_TRANSITION}]"
        print(
            f"[TORSO-CURRICULUM] Stage {s}/{len(_TT_STAGES) - 1} ({label}) | "
            f"iter {s_iter:>4} | "
            f"trk={trk_str}/{_TT_THRESHOLD:.1f}  "
            f"ht={ht_str}/{_TT_HEIGHT_THRESHOLD:.1f}  "
            f"timeout={to_str}  rew={rew_str} | "
            f"advance {state['above_count']:>2}/{_TT_SUSTAIN} | "
            f"ES {state['no_improve']}/{_ES_PATIENCE}{blend_tag}"
        )

        if state["no_improve"] >= _ES_PATIENCE:
            print(
                f"\n[EARLY-STOP] No improvement for {_ES_PATIENCE} iterations.  "
                f"Best reward: {state['best_reward']:.3f}.  Stopping training.\n"
            )
            raise EarlyStopException()

    runner.log = _wrapped_log
    print(
        f"[TORSO-CURRICULUM] Installed.  "
        f"Stages: {len(_TT_STAGES)}  "
        f"Threshold: {_TT_THRESHOLD}  "
        f"Sustain: {_TT_SUSTAIN} iters  "
        f"Transition: {_TT_TRANSITION} iters  "
        f"Early-stop patience: {_ES_PATIENCE} iters"
    )
# ─────────────────────────────────────────────────────────────────────────────


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
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

    # Install torso-tracking curriculum + early stopping
    _tt_install_curriculum(runner)

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except EarlyStopException:
        # Save final checkpoint
        runner.save(os.path.join(log_dir, "model_early_stop.pt"))
        print("[INFO] Saved early-stop checkpoint.")

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
