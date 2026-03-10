# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# check minimum supported rsl-rl version
RSL_RL_VERSION = "3.0.1"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

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

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)
import go1_ball_balance  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# ─── Ball-balance automatic spawn + sigma curriculum ─────────────────────────
# Advance A → G when time_out% ≥ threshold for a sustained run of consecutive
# iterations.  One-way: never rolls back.
#
# Academic basis:
#   ROGER (RSS 2025)           — threshold-based one-way sigma curriculum
#   ANYmal Parkour (2024)      — stage-gated, sustained sustain iters above threshold
#   ADR / OpenAI (2019)        — auto-expand boundary sampling on success
#
# Stages — each entry: (xy_std, drop_height_mean, drop_height_std, sigma)
#   sigma = std for ball_on_paddle_exp Gaussian centering kernel.
#   drop_height in metres above the paddle surface.
#   Ball arrival speed = sqrt(2 * 9.81 * drop_height).
# Columns: (xy_std, drop_height_mean, drop_height_std, sigma)
#
# xy_std:          geometric ~1.3× per stage  (30→50→65→85→105→130→150 mm)
# drop_height_std: ~20 % of mean throughout (consistent spread across stages)
# sigma:           geometric ~0.80× per stage, floor 80 mm  (ROGER RSS 2025 floor)
#
# Note on Stage G: 1.0 m drop → ball bounces to ~72 cm, lateral drift can
# exceed the 300 mm ball_off termination radius during the arc.  If Stage G
# proves unreachable, widen ball_off radius to ~450 mm before attempting it.
_BB_STAGES = [
    # xy_std  drop_mean  drop_std  sigma
    (0.030,   0.08,      0.016,    0.25),  # A — 1.3 m/s,  drop_std=20% of mean
    (0.050,   0.15,      0.030,    0.20),  # B — 1.7 m/s,  xy step ~1.33× (was 1.5× — fixed)
    (0.065,   0.25,      0.050,    0.16),  # C — 2.2 m/s
    (0.085,   0.40,      0.080,    0.13),  # D — 2.8 m/s
    (0.105,   0.60,      0.120,    0.10),  # E — 3.4 m/s
    (0.130,   0.80,      0.160,    0.08),  # F — 4.0 m/s  (beyond current literature)
    (0.150,   1.00,      0.200,    0.08),  # G — 4.4 m/s  (aspirational; sigma floor=80mm)
]
_BB_THRESHOLD  = 0.75   # time_out fraction required to advance
                        # 76% training success (with std=3 action noise) = near-perfect in play
_BB_SUSTAIN    = 15     # consecutive iterations required (was 20)
_BB_TRANSITION = 10     # iterations to linearly blend old → new stage parameters
                        # prevents the 70→50% performance dip from a hard parameter snap


def _bb_set_params(rl_env, xy_std: float, drop_h_mean: float,
                   drop_h_std: float, sigma: float) -> None:
    """Write spawn and reward params directly (used by both hard-set and blend)."""
    for term_cfg in rl_env.event_manager._mode_term_cfgs.get("reset", []):
        if hasattr(term_cfg, "params") and "xy_std" in (term_cfg.params or {}):
            term_cfg.params["xy_std"]           = xy_std
            term_cfg.params["drop_height_mean"] = drop_h_mean
            term_cfg.params["drop_height_std"]  = drop_h_std
    for i, name in enumerate(rl_env.reward_manager._term_names):
        if name == "ball_on_paddle":
            rl_env.reward_manager._term_cfgs[i].params["std"] = sigma
            break


def _bb_apply_stage(rl_env, stage_idx: int) -> None:
    """Hard-set params to a specific stage (used at curriculum install time)."""
    xy_std, drop_h_mean, drop_h_std, sigma = _BB_STAGES[stage_idx]
    _bb_set_params(rl_env, xy_std, drop_h_mean, drop_h_std, sigma)
    print(
        f"\n[CURRICULUM] ▶ Stage {stage_idx}/{len(_BB_STAGES) - 1}  "
        f"xy_std={xy_std:.3f} m  "
        f"drop={drop_h_mean:.2f} m ± {drop_h_std:.3f} m  "
        f"sigma={sigma:.3f} m\n"
    )


def _bb_install_curriculum(runner) -> None:
    """Monkey-patch runner.log() to auto-advance the spawn+sigma curriculum.

    Intercepts the per-iteration log call inside runner.learn(), reads the
    time_out fraction from ep_infos, and advances from Stage A to G when the
    threshold has been sustained for the required number of consecutive iterations.

    Stage transitions use linear parameter interpolation over _BB_TRANSITION
    iterations to avoid the sharp distribution shift that causes the
    ~70% → 50% performance dip seen with an instantaneous parameter snap.
    """
    try:
        rl_env = runner.env.unwrapped
    except AttributeError:
        print("[CURRICULUM] Could not access unwrapped env — curriculum disabled.")
        return

    original_log = runner.log
    _STAGE_LETTERS = "ABCDEFGH"
    state = {
        "stage":       0,
        "above_count": 0,
        "stage_iter":  0,
        "old_stage":   None,   # set during transition; None outside one
        "trans_iter":  0,      # iterations elapsed since transition started
    }

    def _wrapped_log(locs, *args, **kwargs):
        # Always call the original log first (TensorBoard / console output).
        original_log(locs, *args, **kwargs)

        state["stage_iter"] += 1
        s      = state["stage"]
        s_iter = state["stage_iter"]
        label  = _STAGE_LETTERS[s] if s < len(_STAGE_LETTERS) else str(s)
        final  = s >= len(_BB_STAGES) - 1

        # ── continue in-progress parameter transition ─────────────────────────
        if state["old_stage"] is not None:
            state["trans_iter"] += 1
            alpha = min(state["trans_iter"] / _BB_TRANSITION, 1.0)
            o = _BB_STAGES[state["old_stage"]]
            n = _BB_STAGES[s]
            _bb_set_params(
                rl_env,
                xy_std      = o[0] + alpha * (n[0] - o[0]),
                drop_h_mean = o[1] + alpha * (n[1] - o[1]),
                drop_h_std  = o[2] + alpha * (n[2] - o[2]),
                sigma       = o[3] + alpha * (n[3] - o[3]),
            )
            if state["trans_iter"] >= _BB_TRANSITION:
                state["old_stage"] = None   # transition complete

        # ── extract time_out fraction from RSL-RL ep_infos ───────────────────
        ep_infos = locs.get("ep_infos", [])
        time_out_frac = None
        if ep_infos:
            for key in ep_infos[0]:
                if "time_out" in key.lower() or "timeout" in key.lower():
                    vals = [float(ep[key]) for ep in ep_infos if key in ep]
                    if vals:
                        time_out_frac = sum(vals) / len(vals)
                    break

        # ── threshold + sustain check (skip if already at final stage) ────────
        if not final and time_out_frac is not None:
            if time_out_frac >= _BB_THRESHOLD:
                state["above_count"] += 1
            else:
                state["above_count"] = 0

            if state["above_count"] >= _BB_SUSTAIN:
                state["old_stage"]   = s           # begin gradual transition
                state["trans_iter"]  = 0
                state["stage"]      += 1
                state["above_count"] = 0
                state["stage_iter"]  = 0
                s     = state["stage"]
                label = _STAGE_LETTERS[s] if s < len(_STAGE_LETTERS) else str(s)
                final = s >= len(_BB_STAGES) - 1
                n = _BB_STAGES[s]
                print(
                    f"\n[CURRICULUM] ▶ Stage {s-1}→{s} ({_STAGE_LETTERS[s-1]}→{label})  "
                    f"blending over {_BB_TRANSITION} iters  "
                    f"target: xy={n[0]:.3f} m  drop={n[1]:.2f} m  σ={n[3]:.3f} m\n"
                )

        # ── per-iteration status line ─────────────────────────────────────────
        # Show current *actual* blended params during transition.
        if state["old_stage"] is not None:
            alpha = min(state["trans_iter"] / _BB_TRANSITION, 1.0)
            o = _BB_STAGES[state["old_stage"]]
            n = _BB_STAGES[s]
            cur_xy  = o[0] + alpha * (n[0] - o[0])
            cur_dh  = o[1] + alpha * (n[1] - o[1])
            cur_sig = o[3] + alpha * (n[3] - o[3])
            blend_tag = f" [→{label} blend {state['trans_iter']}/{_BB_TRANSITION}]"
            # Stage counter shows OLD stage during blend
            s_display = state["old_stage"]
            label_display = _STAGE_LETTERS[s_display] if s_display < len(_STAGE_LETTERS) else str(s_display)
        else:
            cur_xy, cur_dh, _, cur_sig = _BB_STAGES[s]
            blend_tag     = ""
            s_display     = s
            label_display = label

        to_str = f"{time_out_frac:.0%}" if time_out_frac is not None else "n/a"
        if final and state["old_stage"] is None:
            suffix = "FINAL STAGE"
        else:
            suffix = f"threshold {state['above_count']:>2}/{_BB_SUSTAIN} ({to_str} vs {_BB_THRESHOLD:.0%})"
        print(
            f"[CURRICULUM] Stage {s_display}/{len(_BB_STAGES) - 1} ({label_display}) | "
            f"iter {s_iter:>4} in stage | "
            f"xy={cur_xy:.3f} m  drop={cur_dh:.2f} m  σ={cur_sig:.3f} m | "
            f"{suffix}{blend_tag}"
        )

    runner.log = _wrapped_log
    print(
        f"[CURRICULUM] Installed.  "
        f"Stages: {len(_BB_STAGES)}  "
        f"Threshold: {_BB_THRESHOLD:.0%}  "
        f"Sustain: {_BB_SUSTAIN} iters  "
        f"Transition: {_BB_TRANSITION} iters"
    )
# ─────────────────────────────────────────────────────────────────────────────


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
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

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Install automatic spawn + sigma curriculum (ball-balance task only).
    # No-op for other tasks — _bb_apply_stage() inspects event param names.
    _bb_install_curriculum(runner)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
