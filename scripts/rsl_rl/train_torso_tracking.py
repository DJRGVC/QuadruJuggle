# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train RL agent for the torso-tracking task (pi2) with 3-stage curriculum.

Follows Isaac Lab's approach: learn to WALK first, then add pose tracking.

  Stage  h_range        roll/pitch       h_dot            omega             vxy
  A      0.38 (fixed)   0.0 (fixed)     0.0 (fixed)     0.0 (fixed)      [-0.50, 0.50]
  B      [0.32, 0.44]   [-0.15, 0.15]   [-0.3,  0.3]   [-1.0, 1.0]      [-0.50, 0.50]
  C      [0.20, 0.50]   [-0.50, 0.50]   [-1.5,  1.5]   [-4.0, 4.0]      [-0.50, 0.50]

A→B: timeout ≥ 85% AND vxy_error > -0.80 (robot is actually walking)
B→C: timeout ≥ 85% (pose tracking working)
Early stopping: 500 iterations without reward improvement on final stage.

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


# ── Torso-tracking 3-stage curriculum ─────────────────────────────────────────
#
# Isaac Lab approach: NO curriculum on velocity.  Full vxy range from the start.
# The robot must learn to walk first (Stage A), then mild pose (B), then full (C).
#
# Format: (h_lo, h_hi, rp_lo, rp_hi, hd_lo, hd_hi, om_lo, om_hi, vxy_lo, vxy_hi)
_TT_STAGES = [
    # h_lo   h_hi   rp_lo  rp_hi  hd_lo  hd_hi  om_lo  om_hi  vxy_lo vxy_hi
    (0.38,   0.38,   0.0,  0.0,    0.0,  0.0,    0.0,  0.0,  -0.50,  0.50),   # A — walk (full speed)
    (0.32,   0.44,  -0.15, 0.15,  -0.3,  0.3,   -1.0,  1.0,  -0.50,  0.50),   # B — walk + mild pose
    (0.20,   0.50,  -0.50, 0.50,  -1.5,  1.5,   -4.0,  4.0,  -0.50,  0.50),   # C — full 8D
]
_STAGE_LABELS = "ABC"
_TT_TIMEOUT_THRESHOLD = 0.85   # timeout% >= 85% to advance
_TT_VXY_ERROR_GATE = -0.80    # vxy_error must be > this to advance from A (proves walking)
_TT_SUSTAIN = 100              # consecutive iterations at threshold before advancing
_TT_TRANSITION = 20            # iterations to blend between stages

# Early stopping config (only active on final stage)
_ES_PATIENCE = 500     # iterations without improvement
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
    rl_env._torso_cmd_ranges["vx"] = [s[8], s[9]]
    rl_env._torso_cmd_ranges["vy"] = [s[8], s[9]]
    label = _STAGE_LABELS[stage_idx] if stage_idx < len(_STAGE_LABELS) else str(stage_idx)
    print(
        f"\n[TORSO-CURRICULUM] Stage {label}  "
        f"h=[{s[0]:.2f},{s[1]:.2f}]  roll/pitch=[{s[2]:.2f},{s[3]:.2f}]  "
        f"h_dot=[{s[4]:.1f},{s[5]:.1f}]  omega=[{s[6]:.1f},{s[7]:.1f}]  "
        f"vxy=[{s[8]:.2f},{s[9]:.2f}]\n"
    )


def _tt_blend_stages(rl_env, old_idx: int, new_idx: int, alpha: float) -> None:
    """Linearly interpolate command ranges between two stages."""
    o = _TT_STAGES[old_idx]
    n = _TT_STAGES[new_idx]
    blended = tuple(o[i] + alpha * (n[i] - o[i]) for i in range(10))
    rl_env._torso_cmd_ranges["h"] = [blended[0], blended[1]]
    rl_env._torso_cmd_ranges["roll"] = [blended[2], blended[3]]
    rl_env._torso_cmd_ranges["pitch"] = [blended[2], blended[3]]
    rl_env._torso_cmd_ranges["h_dot"] = [blended[4], blended[5]]
    rl_env._torso_cmd_ranges["omega_roll"] = [blended[6], blended[7]]
    rl_env._torso_cmd_ranges["omega_pitch"] = [blended[6], blended[7]]
    rl_env._torso_cmd_ranges["vx"] = [blended[8], blended[9]]
    rl_env._torso_cmd_ranges["vy"] = [blended[8], blended[9]]


def _tt_install_curriculum(runner) -> None:
    """Monkey-patch runner.log() for 3-stage curriculum + early stopping."""
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
    _FINAL_SUSTAIN = 750  # iterations at threshold on final stage before auto-stop
    state = {
        "stage":       0,
        "above_count": 0,
        "final_count": 0,
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
        label  = _STAGE_LABELS[s] if s < len(_STAGE_LABELS) else str(s)
        final  = s >= len(_TT_STAGES) - 1

        # ── continue in-progress parameter transition ─────────────────────────
        if state["old_stage"] is not None:
            state["trans_iter"] += 1
            alpha = min(state["trans_iter"] / _TT_TRANSITION, 1.0)
            _tt_blend_stages(rl_env, state["old_stage"], s, alpha)
            if state["trans_iter"] >= _TT_TRANSITION:
                state["old_stage"] = None

        # ── extract metrics ───────────────────────────────────────────────────
        ep_infos = locs.get("ep_infos", [])
        mean_reward = locs.get("mean_reward", None)

        display_keys = ["height_error", "vx_tracking", "vy_tracking", "vxy_error"]
        display_vals = {}
        time_out_frac = None

        if ep_infos:
            for key in ep_infos[0]:
                if "time_out" in key.lower() or "timeout" in key.lower():
                    vals = [float(ep[key]) for ep in ep_infos if key in ep]
                    if vals:
                        time_out_frac = sum(vals) / len(vals)
                for dk in display_keys:
                    if dk in key.lower():
                        vals = [float(ep[key]) for ep in ep_infos if key in ep]
                        if vals:
                            display_vals[dk] = sum(vals) / len(vals)

        height_error_val = display_vals.get("height_error", None)
        vx_tracking_val = display_vals.get("vx_tracking", None)
        vy_tracking_val = display_vals.get("vy_tracking", None)
        vxy_error_val = display_vals.get("vxy_error", None)

        # ── advancement check ─────────────────────────────────────────────────
        # Stage A→B requires: timeout ≥ 85% AND vxy_error > -0.80
        #   (proves the robot is actually walking, not standing still)
        # Stage B→C requires: timeout ≥ 85% only
        #   (mild pose tracking is working)
        if not final and time_out_frac is not None:
            timeout_ok = time_out_frac >= _TT_TIMEOUT_THRESHOLD

            if s == 0:
                # Stage A: must prove walking via velocity tracking quality
                vxy_ok = (vxy_error_val is not None) and (vxy_error_val > _TT_VXY_ERROR_GATE)
                advance_ok = timeout_ok and vxy_ok
            else:
                # Stage B+: survival is sufficient
                advance_ok = timeout_ok

            if advance_ok:
                state["above_count"] += 1
            else:
                state["above_count"] = 0

            if state["above_count"] >= _TT_SUSTAIN:
                old_label = label
                state["old_stage"]   = s
                state["trans_iter"]  = 0
                state["stage"]      += 1
                state["above_count"] = 0
                state["stage_iter"]  = 0
                # Reset early stop — task just got harder
                state["best_reward"] = float("-inf")
                state["no_improve"]  = 0
                s     = state["stage"]
                label = _STAGE_LABELS[s] if s < len(_STAGE_LABELS) else str(s)
                n = _TT_STAGES[s]
                print(
                    f"\n[TORSO-CURRICULUM] ═══ {old_label} → {label} ═══  "
                    f"Blending over {_TT_TRANSITION} iters.  "
                    f"h=[{n[0]:.2f},{n[1]:.2f}]  "
                    f"roll/pitch=[{n[2]:.2f},{n[3]:.2f}]  "
                    f"h_dot=[{n[4]:.1f},{n[5]:.1f}]  "
                    f"omega=[{n[6]:.1f},{n[7]:.1f}]  "
                    f"vxy=[{n[8]:.2f},{n[9]:.2f}]\n"
                )

        # ── final stage completion (auto-stop after sustained convergence) ────
        if final and time_out_frac is not None:
            if time_out_frac >= _TT_TIMEOUT_THRESHOLD:
                state["final_count"] += 1
            else:
                state["final_count"] = 0
            if state["final_count"] >= _FINAL_SUSTAIN:
                print(
                    f"\n[TORSO-CURRICULUM] Stage {label} sustained {_FINAL_SUSTAIN} iters.  "
                    f"Training complete.\n"
                )
                if log_dir:
                    runner.save(os.path.join(log_dir, "model_best.pt"))
                raise EarlyStopException()

        # ── early stopping (paused during blends) ─────────────────────────────
        if mean_reward is None:
            rew_buf = locs.get("rewbuffer", [])
            if rew_buf:
                import statistics
                mean_reward = statistics.mean(rew_buf)
        if state["old_stage"] is None:  # only update ES when not blending
            reward_for_es = mean_reward if mean_reward is not None else 0.0
            if reward_for_es > state["best_reward"] + _ES_MIN_DELTA:
                state["best_reward"] = reward_for_es
                state["no_improve"] = 0
                if log_dir:
                    best_path = os.path.join(log_dir, "model_best.pt")
                    runner.save(best_path)
            else:
                state["no_improve"] += 1

        # ── per-iteration status line ─────────────────────────────────────────
        he_str   = f"{height_error_val:.3f}" if height_error_val is not None else "n/a"
        vx_str   = f"{vx_tracking_val:.3f}" if vx_tracking_val is not None else "n/a"
        vy_str   = f"{vy_tracking_val:.3f}" if vy_tracking_val is not None else "n/a"
        vxye_str = f"{vxy_error_val:.3f}" if vxy_error_val is not None else "n/a"
        to_str   = f"{time_out_frac:.0%}" if time_out_frac is not None else "n/a"
        rew_str  = f"{mean_reward:.1f}" if mean_reward is not None else "n/a"
        blend_tag = ""
        if state["old_stage"] is not None:
            next_label = _STAGE_LABELS[s] if s < len(_STAGE_LABELS) else str(s)
            blend_tag = f" [→{next_label} blend {state['trans_iter']}/{_TT_TRANSITION}]"
        # Show vxy gate status on Stage A
        vxy_gate_str = ""
        if s == 0 and vxy_error_val is not None:
            gate_status = "OK" if vxy_error_val > _TT_VXY_ERROR_GATE else "NEED WALKING"
            vxy_gate_str = f"  [{gate_status}]"
        print(
            f"[TORSO] {label} | "
            f"iter {s_iter:>4} | "
            f"timeout={to_str}/{_TT_TIMEOUT_THRESHOLD:.0%}  "
            f"vx={vx_str}  vy={vy_str}  vxy_err={vxye_str}{vxy_gate_str}  "
            f"h_err={he_str}  "
            f"rew={rew_str} | "
            f"adv {state['above_count']:>3}/{_TT_SUSTAIN} | "
            f"{'FINAL ' + str(state['final_count']) + '/' + str(_FINAL_SUSTAIN) + ' | ' if final else ''}"
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
        f"[TORSO-CURRICULUM] 3-stage curriculum installed.\n"
        f"  A = walk only (vxy ±0.50, pose fixed)\n"
        f"  B = walk + mild pose (h=[0.32,0.44], rp=±0.15, omega=±1.0)\n"
        f"  C = full 8D (h=[0.20,0.50], rp=±0.50, omega=±4.0)\n"
        f"  A→B: timeout ≥ {_TT_TIMEOUT_THRESHOLD:.0%} AND vxy_error > {_TT_VXY_ERROR_GATE}  "
        f"for {_TT_SUSTAIN} iters\n"
        f"  B→C: timeout ≥ {_TT_TIMEOUT_THRESHOLD:.0%} for {_TT_SUSTAIN} iters\n"
        f"  ES patience: {_ES_PATIENCE}"
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

    banner = (
        "\n"
        "╔══════════════════════════════════════════════════════════════════╗\n"
        f"║  TORSO TRACKING (pi2) — TRAINING START                         ║\n"
        f"║  Log dir: {os.path.basename(log_dir):<54}║\n"
        f"║  Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<54}║\n"
        f"║  Envs:    {env_cfg.scene.num_envs:<54}║\n"
        "╚══════════════════════════════════════════════════════════════════╝"
    )
    print(banner)

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    except EarlyStopException:
        # Save final checkpoint
        runner.save(os.path.join(log_dir, "model_early_stop.pt"))
        print("[INFO] Saved early-stop checkpoint.")

    elapsed = round(time.time() - start_time, 2)
    banner_end = (
        "\n"
        "╔══════════════════════════════════════════════════════════════════╗\n"
        f"║  TORSO TRACKING (pi2) — TRAINING COMPLETE                      ║\n"
        f"║  Log dir: {os.path.basename(log_dir):<54}║\n"
        f"║  Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S'):<54}║\n"
        f"║  Elapsed: {elapsed:<54}║\n"
        "╚══════════════════════════════════════════════════════════════════╝"
    )
    print(banner_end)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
