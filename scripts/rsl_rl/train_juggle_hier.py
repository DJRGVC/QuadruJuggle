# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train pi1 (ball planner) for the hierarchical ball-juggle task.

pi1 outputs 8D torso commands; frozen pi2 (torso tracker) converts them
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
parser.add_argument(
    "--start-stage", type=int, default=0,
    help="Curriculum stage to start from (0=A, 13=N, etc.). Use with --resume for hot restarts.",
)
parser.add_argument(
    "--noise-mode", type=str, default="oracle",
    choices=["oracle", "d435i", "ekf"],
    help="Ball observation noise mode: 'oracle' (ground truth), 'd435i' (structured camera noise), or 'ekf' (d435i→EKF filter).",
)
parser.add_argument(
    "--wandb", action="store_true", default=False,
    help="Log to Weights & Biases (project: quadrujuggle). Requires WANDB_API_KEY.",
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


# ── Juggling curriculum (tight σ + target randomization) ──────────────────────
# Key invariant: target/σ ≥ 2.0 at every stage.  When target < 2σ the Gaussian
# reward at h=0 (ball sitting on paddle) is > 0.13 — high enough that the policy
# learns to balance instead of juggle, then stalls when σ eventually tightens.
#
# Per-env target randomization: each stage specifies (target_min, target_max).
# Early stages use fixed targets (min=max) to learn basic bouncing.  Starting
# at Stage I, the range expands so the policy learns to handle varied heights.
# By Stage P the full [0.30, 1.00] range is active.  σ = target / sigma_ratio
# per env, maintaining the target/σ invariant regardless of sampled height.
_BJ_STAGES = [
    # tgt_min  tgt_max  σ_ratio  xy_std  vel_xy_std  noise_scale
    # σ_ratio=3.5 makes ball-at-rest earn only 0.2% of max apex reward (was 4.4% with 2.5),
    # forcing active throwing to earn meaningful reward. Root cause of Stage F plateau fixed.
    (0.05,     0.05,    2.5,     0.020,  0.00,       0.00),  # A  — oracle, bootstrap bounce (wider for initial learning)
    (0.10,     0.10,    3.0,     0.022,  0.00,       0.00),  # B  — oracle, tighten sigma
    (0.15,     0.15,    3.5,     0.025,  0.00,       0.00),  # C  — oracle, target sigma ratio
    (0.20,     0.20,    3.5,     0.028,  0.00,       0.25),  # D  — 25% d435i noise
    (0.25,     0.25,    3.5,     0.030,  0.00,       0.50),  # E  — 50% d435i noise
    (0.30,     0.30,    3.5,     0.033,  0.00,       0.75),  # F  — 75% d435i noise
    (0.36,     0.36,    3.5,     0.036,  0.00,       1.00),  # G  — full d435i noise
    (0.42,     0.42,    3.5,     0.040,  0.00,       1.00),  # H  — full d435i noise
    (0.30,     0.48,    3.5,     0.045,  0.02,       1.00),  # I  — range begins + lateral vel
    (0.30,     0.55,    3.5,     0.050,  0.04,       1.00),  # J
    (0.30,     0.62,    3.5,     0.055,  0.06,       1.00),  # K
    (0.30,     0.70,    3.5,     0.060,  0.08,       1.00),  # L
    (0.30,     0.78,    3.5,     0.068,  0.10,       1.00),  # M
    (0.30,     0.86,    3.5,     0.075,  0.12,       1.00),  # N
    (0.30,     0.92,    3.5,     0.082,  0.15,       1.00),  # O
    (0.30,     1.00,    3.5,     0.090,  0.18,       1.00),  # P  — full range, full noise
]
_BJ_THRESHOLD      = 0.75
_BJ_APEX_THRESHOLD = 0.5     # with sigma_ratio=3.5, ball at rest earns ~0.05/step × 1500 = ~75 total
                              # so natural bounce (h≈5cm) earns ~0.1/step; threshold=0.5 requires active throwing
_BJ_SUSTAIN    = 20    # was 15 — require stronger mastery before advancing
_BJ_TRANSITION = 15    # was 10 — slower parameter blending

# Early stopping
_ES_PATIENCE = 700     # longer patience for hierarchical (pi2 adds lag)
_ES_MIN_DELTA = 0.5


class EarlyStopException(Exception):
    pass


def _bj_set_noise_scale(rl_env, noise_scale: float) -> None:
    """Update noise_scale on live perception pipeline or obs term params."""
    # Prefer the stateful pipeline (EKF/d435i modes — created on first obs call)
    try:
        from go1_ball_balance.perception.ball_obs_spec import update_perception_noise_scale
        update_perception_noise_scale(rl_env, noise_scale)
    except ImportError:
        pass
    # Also update static noise_cfg params (stateless d435i fallback)
    obs_group = getattr(rl_env.observation_manager, "_group_obs_term_cfgs", {})
    for _group, term_cfgs in obs_group.items():
        for cfg in term_cfgs:
            params = getattr(cfg, "params", None) or {}
            noise_cfg = params.get("noise_cfg")
            if noise_cfg is not None and hasattr(noise_cfg, "noise_scale"):
                noise_cfg.noise_scale = noise_scale


def _bj_set_params(
    rl_env,
    target_min: float, target_max: float, sigma_ratio: float,
    xy_std: float, vel_xy_std: float, noise_scale: float = 1.0,
) -> None:
    """Write per-env target range, event params, and noise scale directly."""
    # Update the randomize_target_apex event params
    for term_cfg in rl_env.event_manager._mode_term_cfgs.get("reset", []):
        if hasattr(term_cfg, "params") and "target_min" in (term_cfg.params or {}):
            term_cfg.params["target_min"] = target_min
            term_cfg.params["target_max"] = target_max
            term_cfg.params["sigma_ratio"] = sigma_ratio
        if hasattr(term_cfg, "params") and "xy_std" in (term_cfg.params or {}):
            term_cfg.params["xy_std"] = xy_std
        if hasattr(term_cfg, "params") and "vel_xy_std" in (term_cfg.params or {}):
            term_cfg.params["vel_xy_std"] = vel_xy_std
    # Also update scalar fallbacks in reward params (used if per-env buffers not yet created)
    mid = (target_min + target_max) / 2.0
    # Dynamic ball_low threshold: 40% of current target height forces ball to bounce meaningfully.
    # At Stage A (target=0.05m): threshold=0.02m. At Stage E (0.25m): threshold=0.10m.
    ball_low_threshold = 0.4 * mid
    for i, name in enumerate(rl_env.reward_manager._term_names):
        if name == "ball_apex_height":
            rl_env.reward_manager._term_cfgs[i].params["target_height"] = mid
            rl_env.reward_manager._term_cfgs[i].params["std"] = mid / sigma_ratio
        elif name == "ball_low":
            rl_env.reward_manager._term_cfgs[i].params["low_threshold"] = ball_low_threshold
    # Update noise scale on perception obs terms (only active in d435i mode)
    _bj_set_noise_scale(rl_env, noise_scale)


def _bj_apply_stage(rl_env, stage_idx: int) -> None:
    tgt_min, tgt_max, sigma_ratio, xy_std, vel_xy_std, noise_scale = _BJ_STAGES[stage_idx]
    _bj_set_params(rl_env, tgt_min, tgt_max, sigma_ratio, xy_std, vel_xy_std, noise_scale)
    if tgt_min == tgt_max:
        tgt_str = f"target={tgt_min:.2f} m (fixed)"
    else:
        tgt_str = f"target=[{tgt_min:.2f}, {tgt_max:.2f}] m"
    low_thr = 0.4 * (tgt_min + tgt_max) / 2.0
    print(
        f"\n[HIER-JUGGLE-CURRICULUM] Stage {stage_idx}/{len(_BJ_STAGES) - 1}  "
        f"{tgt_str}  σ_ratio={sigma_ratio:.1f}  "
        f"xy_std={xy_std:.3f} m  vel_xy={vel_xy_std:.3f} m/s  "
        f"noise_scale={noise_scale:.2f}  ball_low_thr={low_thr:.3f} m\n"
    )


def _bj_install_curriculum(runner, start_stage: int = 0) -> None:
    """Monkey-patch runner.log() for juggling curriculum + early stopping."""
    try:
        rl_env = runner.env.unwrapped
    except AttributeError:
        print("[HIER-JUGGLE-CURRICULUM] Could not access unwrapped env — curriculum disabled.")
        return

    start_stage = max(0, min(start_stage, len(_BJ_STAGES) - 1))

    original_log = runner.log
    _STAGE_LETTERS = "ABCDEFGHIJKLMNOP"
    state = {
        "stage":       start_stage,
        "above_count": 0,
        "stage_iter":  0,
        "old_stage":   None,
        "trans_iter":  0,
        "best_reward": float("-inf"),
        "no_improve":  0,
    }

    # Apply start stage parameters immediately
    if start_stage > 0:
        _bj_apply_stage(rl_env, start_stage)

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
                target_min  = o[0] + alpha * (n[0] - o[0]),
                target_max  = o[1] + alpha * (n[1] - o[1]),
                sigma_ratio = o[2] + alpha * (n[2] - o[2]),
                xy_std      = o[3] + alpha * (n[3] - o[3]),
                vel_xy_std  = o[4] + alpha * (n[4] - o[4]),
                noise_scale = o[5] + alpha * (n[5] - o[5]),
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
                if n[0] == n[1]:
                    tgt_str = f"target={n[0]:.2f} m"
                else:
                    tgt_str = f"target=[{n[0]:.2f}, {n[1]:.2f}] m"
                print(
                    f"\n[HIER-JUGGLE-CURRICULUM] Stage {s-1}→{s} ({_STAGE_LETTERS[s-1]}→{label})  "
                    f"blending over {_BJ_TRANSITION} iters  "
                    f"{tgt_str}  σ_ratio={n[2]:.1f}  "
                    f"xy={n[3]:.3f} m  vel_xy={n[4]:.3f} m/s  "
                    f"noise_scale={n[5]:.2f}\n"
                )

        # ── early stopping ─────────────────────────────────────────────────────
        # Skip ES during blend transitions — reward is inflated by easier
        # previous-stage parameters, setting an unreachable best_reward.
        # RSL-RL stores episode returns in locs["rewbuffer"] (list), not "mean_reward"
        blending = state["old_stage"] is not None
        rew_buf = locs.get("rewbuffer", [])
        if rew_buf:
            import statistics
            mean_reward = statistics.mean(rew_buf)
        else:
            mean_reward = 0.0
        if not blending:
            if mean_reward > state["best_reward"] + _ES_MIN_DELTA:
                state["best_reward"] = mean_reward
                state["no_improve"] = 0
                if log_dir:
                    best_path = os.path.join(log_dir, "model_best.pt")
                    runner.save(best_path)
            else:
                state["no_improve"] += 1

        # ── per-iteration status line ─────────────────────────────────────────
        # Always display CURRENT stage; show blend progress as suffix
        s_display     = s
        label_display = _STAGE_LETTERS[s] if s < len(_STAGE_LETTERS) else str(s)
        if state["old_stage"] is not None:
            alpha = min(state["trans_iter"] / _BJ_TRANSITION, 1.0)
            o = _BJ_STAGES[state["old_stage"]]
            n = _BJ_STAGES[s]
            cur_tmin = o[0] + alpha * (n[0] - o[0])
            cur_tmax = o[1] + alpha * (n[1] - o[1])
            cur_xy   = o[3] + alpha * (n[3] - o[3])
            cur_vel  = o[4] + alpha * (n[4] - o[4])
            old_label = _STAGE_LETTERS[state["old_stage"]] if state["old_stage"] < len(_STAGE_LETTERS) else str(state["old_stage"])
            blend_tag = f" [blend {old_label}→{label_display} {state['trans_iter']}/{_BJ_TRANSITION}]"
        else:
            cur_tmin, cur_tmax, _, cur_xy, cur_vel, _ns = _BJ_STAGES[s]
            blend_tag = ""

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
        if cur_tmin == cur_tmax:
            tgt_display = f"target={cur_tmin:.2f} m"
        else:
            tgt_display = f"target=[{cur_tmin:.2f},{cur_tmax:.2f}] m"
        print(
            f"[HIER-JUGGLE-CURRICULUM] Stage {s_display + 1}/{len(_BJ_STAGES)} ({label_display}) | "
            f"iter {s_iter:>4} in stage | "
            f"{tgt_display}  "
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
    label = _STAGE_LETTERS[start_stage] if start_stage < len(_STAGE_LETTERS) else str(start_stage)
    print(
        f"[HIER-JUGGLE-CURRICULUM] Installed.  "
        f"Start: {start_stage} ({label})  "
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

    # ── Inject ball observation noise mode ─────────────────────────────────
    if args_cli.noise_mode != "oracle":
        from go1_ball_balance.perception.ball_obs_spec import BallObsNoiseCfg
        noise_cfg = BallObsNoiseCfg(mode=args_cli.noise_mode)
        env_cfg.observations.policy.ball_pos.params["noise_cfg"] = noise_cfg
        env_cfg.observations.policy.ball_vel.params["noise_cfg"] = noise_cfg
        print(f"[INFO] Ball obs noise mode: {args_cli.noise_mode}")
    else:
        print("[INFO] Ball obs noise mode: oracle (ground truth)")

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

    runner_cfg = agent_cfg.to_dict()
    if args_cli.wandb:
        runner_cfg["logger"] = "wandb"
        runner_cfg["wandb_project"] = "quadrujuggle"
        runner_cfg["wandb_entity"] = "d-grant-uc-berkeley"
        # Descriptive run name: noise_mode-envs-pi2dim-maxiters
        pi2_ckpt_tag = os.path.basename(os.path.dirname(args_cli.pi2_checkpoint or "unknown"))
        noise_tag = args_cli.noise_mode if hasattr(args_cli, "noise_mode") else "oracle"
        runner_cfg["wandb_name"] = (
            f"pi1_{noise_tag}_{args_cli.num_envs or 4096}envs_{pi2_ckpt_tag[:10]}"
            f"_{'resume' if agent_cfg.resume else 'fresh'}"
            f"_{agent_cfg.max_iterations}iters"
        )
        print(f"[INFO] Logging to wandb: {runner_cfg['wandb_project']} / {runner_cfg['wandb_name']}")

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, runner_cfg, log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, runner_cfg, log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Install juggling curriculum + early stopping
    _bj_install_curriculum(runner, start_stage=args_cli.start_stage)

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except EarlyStopException:
        runner.save(os.path.join(log_dir, "model_early_stop.pt"))
        print("[INFO] Saved early-stop checkpoint.")

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # Upload training videos to wandb (if both --wandb and --video are set)
    if args_cli.wandb and args_cli.video:
        import glob as _glob
        import wandb as _wandb
        if _wandb.run is not None:
            video_dir = os.path.join(log_dir, "videos", "train")
            mp4_files = sorted(_glob.glob(os.path.join(video_dir, "*.mp4")))
            if mp4_files:
                print(f"[INFO] Uploading {len(mp4_files)} video(s) to wandb.")
                for mp4 in mp4_files[-3:]:  # upload at most 3 most recent
                    _wandb.log({"training_video": _wandb.Video(mp4, fps=30, format="mp4")})
            else:
                print("[INFO] No training videos found to upload.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
