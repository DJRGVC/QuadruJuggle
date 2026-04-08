"""Teleoperated play — user controls robot vx/vy while pi1 handles ball juggling.

Usage:
    uv run --active python scripts/rsl_rl/play_teleop.py \
        --task Isaac-BallJuggleHier-Go1-Play-v0 \
        --pi2-checkpoint <path> \
        --num_envs 1 \
        --backend keyboard

Backends:
    pygame   — USB gamepad (left stick, default)
    keyboard — WASD keys (requires pynput)
    zero     — no input; pi1 controls everything (baseline comparison)

Blend modes:
    override    — user fully controls vx/vy (default)
    blend       — weighted mix of pi1 + user (set --blend-alpha)
    passthrough — pi1 controls everything (ablation)
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# ── Pre-parse teleop-specific args (before argparse/Hydra) ──────────────
_pi2_checkpoint_path = None
_teleop_backend = "keyboard"
_teleop_mode = "override"
_teleop_blend_alpha = 0.0
_clean_argv = []
_i = 0
while _i < len(sys.argv):
    if sys.argv[_i] in ("--pi2-checkpoint", "--pi2_checkpoint") and _i + 1 < len(sys.argv):
        _pi2_checkpoint_path = sys.argv[_i + 1]
        _i += 2
    elif sys.argv[_i] in ("--backend",) and _i + 1 < len(sys.argv):
        _teleop_backend = sys.argv[_i + 1]
        _i += 2
    elif sys.argv[_i] in ("--blend-mode", "--blend_mode") and _i + 1 < len(sys.argv):
        _teleop_mode = sys.argv[_i + 1]
        _i += 2
    elif sys.argv[_i] in ("--blend-alpha", "--blend_alpha") and _i + 1 < len(sys.argv):
        _teleop_blend_alpha = float(sys.argv[_i + 1])
        _i += 2
    else:
        _clean_argv.append(sys.argv[_i])
        _i += 1
sys.argv = _clean_argv

parser = argparse.ArgumentParser(description="Teleoperated play — user vx/vy + pi1 ball juggling.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (default: 1 for teleop).")
parser.add_argument("--task", type=str, default="Isaac-BallJuggleHier-Go1-Play-v0", help="Task name.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run at real-time speed.")
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument("--video_length", type=int, default=400, help="Video length in steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric."
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import go1_ball_balance  # noqa: F401

from go1_ball_balance.vel_cmd import (
    CommandMixer,
    CommandMixerCfg,
    UserVelocityInput,
    UserVelocityInputCfg,
)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Teleoperated play with user velocity commands."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint:
        from isaaclab.utils.assets import retrieve_file_path
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # Inject pi2 checkpoint
    if _pi2_checkpoint_path is not None:
        if hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "torso_cmd"):
            env_cfg.actions.torso_cmd.pi2_checkpoint = os.path.abspath(_pi2_checkpoint_path)
            print(f"[INFO] pi2 checkpoint: {env_cfg.actions.torso_cmd.pi2_checkpoint}")

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "teleop"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── Banner ───────────────────────────────────────────────────────────
    from datetime import datetime as _dt
    _ckpt_mtime = os.path.getmtime(resume_path)
    _run_dir = os.path.basename(os.path.dirname(resume_path))
    _ckpt_name = os.path.basename(resume_path)
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════════════╗\n"
        f"║  TELEOP PLAY                                                   ║\n"
        f"║  Task:    {task_name:<54}║\n"
        f"║  Run:     {_run_dir:<54}║\n"
        f"║  File:    {_ckpt_name:<54}║\n"
        f"║  Backend: {_teleop_backend:<54}║\n"
        f"║  Mode:    {_teleop_mode:<54}║\n"
        "╠══════════════════════════════════════════════════════════════════╣\n"
        f"║  Controls: {'WASD = vx/vy' if _teleop_backend == 'keyboard' else 'Left stick = vx/vy':<54}║\n"
        f"║  vx/vy max: ±0.30 m/s (60% pi2 training max)                  ║\n"
        "╚══════════════════════════════════════════════════════════════════╝"
    )

    # ── Load policy ──────────────────────────────────────────────────────
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # ── Setup velocity input + mixer ─────────────────────────────────────
    vel_input = UserVelocityInput(UserVelocityInputCfg(backend=_teleop_backend))
    mixer = CommandMixer(CommandMixerCfg(mode=_teleop_mode, blend_alpha=_teleop_blend_alpha))

    vel_input.start()

    dt = env.unwrapped.step_dt * env.unwrapped.cfg.decimation
    num_envs = env.unwrapped.num_envs
    device = env.unwrapped.device

    # Reset environment
    obs = env.get_observations()
    timestep = 0

    try:
        while simulation_app.is_running():
            start_time = time.time()

            with torch.inference_mode():
                # Pi1 produces 8D commands
                actions = policy(obs)

                # Mix in user velocity
                vel_user = vel_input.get_cmd_tensor(num_envs, device)
                actions = mixer.mix(actions, vel_user)

                # Step
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)

            timestep += 1

            # Telemetry every 20 steps (~0.4s)
            if timestep % 20 == 0:
                vx_ms, vy_ms = vel_input.get_cmd()
                _uw = env.unwrapped
                _robot = _uw.scene["robot"]
                _vx_act = _robot.data.root_lin_vel_b[0, 0].item()
                _vy_act = _robot.data.root_lin_vel_b[0, 1].item()
                # Show ball state if available
                _ball_str = ""
                if "ball" in _uw.scene:
                    _ball = _uw.scene["ball"]
                    _bz = _ball.data.root_pos_w[0, 2].item()
                    _ball_str = f" | ball_z={_bz:.3f}"
                print(
                    f"[TELEOP] user vx={vx_ms:+.2f} vy={vy_ms:+.2f} m/s | "
                    f"actual vx={_vx_act:+.2f} vy={_vy_act:+.2f} m/s"
                    f"{_ball_str}"
                )

            # Video exit
            if args_cli.video and timestep >= args_cli.video_length:
                break

            # Real-time pacing
            elapsed = time.time() - start_time
            sleep_time = dt - elapsed
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl-C — stopping teleop.")
    finally:
        vel_input.stop()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
