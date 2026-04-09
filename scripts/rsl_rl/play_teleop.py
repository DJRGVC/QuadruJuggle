# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Teleop play script for hierarchical ball juggling (pi1 + frozen pi2).

User controls:
    WASD  — velocity commands (vx forward/back, vy left/right)
    P/L   — increase/decrease target ball apex height
    R     — reset target height to default
    Q/ESC — quit

The script overrides pi1's velocity output channels (slots 6-7) with user
commands and adjusts the target_apex_height observation dynamically via P/L.

Usage:
    uv run --active python scripts/rsl_rl/play_teleop.py \
        --task Isaac-BallJuggleHier-Go1-Play-v0 \
        --pi2-checkpoint <path> \
        --num_envs 1
"""

import argparse
import os
import signal
import sys
import termios
import threading
import tty

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Teleop play for hierarchical ball juggling.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments (default: 1 for teleop).")
parser.add_argument("--task", type=str, default="Isaac-BallJuggleHier-Go1-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time.")
parser.add_argument("--vel-step", type=float, default=0.15, help="Velocity increment per keypress (m/s).")
parser.add_argument("--height-step", type=float, default=0.05, help="Height increment per keypress (m).")
parser.add_argument("--height-min", type=float, default=0.05, help="Minimum target apex height (m).")
parser.add_argument("--height-max", type=float, default=0.80, help="Maximum target apex height (m).")
parser.add_argument("--height-default", type=float, default=0.30, help="Default target apex height (m).")
parser.add_argument("--vel-max", type=float, default=0.50, help="Maximum velocity command (m/s).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
)

# Pre-strip custom args before argparse sees them
_pi2_checkpoint_path = None
_clean_argv = []
_i = 0
while _i < len(sys.argv):
    if sys.argv[_i] in ("--pi2-checkpoint", "--pi2_checkpoint") and _i + 1 < len(sys.argv):
        _pi2_checkpoint_path = sys.argv[_i + 1]
        _i += 2
    else:
        _clean_argv.append(sys.argv[_i])
        _i += 1
sys.argv = _clean_argv

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports after AppLauncher ─────────────────────────────────────────────────

import time

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import go1_ball_balance  # noqa: F401


# ── Non-blocking keyboard input ──────────────────────────────────────────────

class KeyboardController:
    """Non-blocking keyboard input via a background thread reading stdin."""

    def __init__(self, vel_step=0.15, height_step=0.05, vel_max=0.50,
                 height_min=0.05, height_max=0.80, height_default=0.30):
        self.vel_step = vel_step
        self.height_step = height_step
        self.vel_max = vel_max
        self.height_min = height_min
        self.height_max = height_max

        # State
        self.user_vx = 0.0
        self.user_vy = 0.0
        self.target_height = height_default
        self.quit_requested = False
        self._lock = threading.Lock()

        # Save terminal settings
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)

        # Start background thread
        self._thread = threading.Thread(target=self._read_keys, daemon=True)
        self._thread.start()

    def _read_keys(self):
        """Background thread: read single characters from stdin."""
        try:
            tty.setcbreak(self._fd)
            while not self.quit_requested:
                ch = sys.stdin.read(1)
                if not ch:
                    break
                self._handle_key(ch.lower())
        except Exception:
            pass

    def _handle_key(self, key):
        with self._lock:
            if key == 'w':
                self.user_vx = min(self.user_vx + self.vel_step, self.vel_max)
            elif key == 's':
                self.user_vx = max(self.user_vx - self.vel_step, -self.vel_max)
            elif key == 'a':
                self.user_vy = min(self.user_vy + self.vel_step, self.vel_max)
            elif key == 'd':
                self.user_vy = max(self.user_vy - self.vel_step, -self.vel_max)
            elif key == 'p':
                self.target_height = min(self.target_height + self.height_step, self.height_max)
            elif key == 'l':
                self.target_height = max(self.target_height - self.height_step, self.height_min)
            elif key == 'r':
                self.user_vx = 0.0
                self.user_vy = 0.0
            elif key == 'x':
                self.user_vx = 0.0
                self.user_vy = 0.0
                self.target_height = 0.30
            elif key in ('q', '\x1b'):  # q or ESC
                self.quit_requested = True

    def get_state(self):
        with self._lock:
            return self.user_vx, self.user_vy, self.target_height

    def restore_terminal(self):
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)
        except Exception:
            pass


# ── HUD display ──────────────────────────────────────────────────────────────

def print_hud(user_vx, user_vy, target_h, ball_h, pi1_cmd, step):
    """Print a compact terminal HUD with user commands and ball state."""
    # Clear line and print
    lines = [
        f"\r\033[K[TELEOP] step={step:>5d} | "
        f"vx={user_vx:+.2f} vy={user_vy:+.2f} | "
        f"target_h={target_h:.2f}m | "
        f"ball_h={ball_h:.3f}m | "
        f"pi1_cmd=[h={pi1_cmd[0]:+.2f} hd={pi1_cmd[1]:+.2f} "
        f"r={pi1_cmd[2]:+.2f} p={pi1_cmd[3]:+.2f}]",
    ]
    sys.stdout.write(lines[0])
    sys.stdout.flush()


def print_controls_banner():
    print("\n" + "=" * 72)
    print("  TELEOP CONTROLS — Hierarchical Ball Juggling (pi1 + pi2)")
    print("=" * 72)
    print("  W/S     — forward/backward velocity (vx)")
    print("  A/D     — left/right velocity (vy)")
    print("  P/L     — increase/decrease target ball apex height")
    print("  R       — zero velocity")
    print("  X       — reset all (zero vel + default height)")
    print("  Q/ESC   — quit")
    print("=" * 72 + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with teleop controls."""

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Inject pi2 checkpoint
    if _pi2_checkpoint_path is not None:
        if hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "torso_cmd"):
            env_cfg.actions.torso_cmd.pi2_checkpoint = os.path.abspath(_pi2_checkpoint_path)
            print(f"[INFO] pi2 checkpoint: {env_cfg.actions.torso_cmd.pi2_checkpoint}")

    # Log directory
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env_cfg.log_dir = os.path.dirname(resume_path)

    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load policy
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # Banner
    print(f"\n[INFO] Playing checkpoint: {resume_path}")
    print(f"[INFO] Num envs: {env_cfg.scene.num_envs}")
    print_controls_banner()

    # Initialize keyboard controller
    kb = KeyboardController(
        vel_step=args_cli.vel_step,
        height_step=args_cli.height_step,
        vel_max=args_cli.vel_max,
        height_min=args_cli.height_min,
        height_max=args_cli.height_max,
        height_default=args_cli.height_default,
    )

    # Get env internals
    uw = env.unwrapped
    device = uw.device

    # Command scaling (from action_term.py)
    # pi1 outputs [-1, 1]; slots 6,7 are vx,vy with scale=0.5, offset=0
    vxy_scale = 0.5  # maps [-1,1] to [-0.5, 0.5] m/s

    dt = uw.step_dt * env_cfg.decimation
    timestep = 0

    obs = env.get_observations()

    try:
        while simulation_app.is_running() and not kb.quit_requested:
            start_time = time.time()

            with torch.inference_mode():
                # Get user commands
                user_vx, user_vy, target_h = kb.get_state()

                # Update target apex height on the environment
                # This modifies what the obs term returns next step
                if hasattr(uw, "_target_apex_heights"):
                    uw._target_apex_heights[:] = target_h
                else:
                    # Create per-env target buffer
                    uw._target_apex_heights = torch.full(
                        (uw.num_envs,), target_h, device=device, dtype=torch.float32,
                    )
                # Also update the reward term's target/sigma so reward matches
                for i, name in enumerate(uw.reward_manager._term_names):
                    if name == "ball_apex_height":
                        uw.reward_manager._term_cfgs[i].params["target_height"] = target_h
                        uw.reward_manager._term_cfgs[i].params["std"] = target_h / 3.5
                        break

                # Run pi1 policy
                actions = policy(obs)

                # Override vx/vy channels (slots 6, 7) with user commands
                # Convert m/s to [-1,1] range: action = velocity / scale
                actions[:, 6] = user_vx / vxy_scale
                actions[:, 7] = user_vy / vxy_scale
                # Clamp to valid range
                actions[:, 6:8] = actions[:, 6:8].clamp(-1.0, 1.0)

                # Step environment
                obs, _, dones, _ = env.step(actions)
                policy_nn.reset(dones)

            timestep += 1

            # Get ball height above paddle for HUD
            ball_h = 0.0
            try:
                ball_pos = uw.scene["ball"].data.root_pos_w[0]
                robot_pos = uw.scene["robot"].data.root_pos_w[0]
                ball_h = (ball_pos[2] - robot_pos[2] - 0.070).item()  # height above paddle
            except Exception:
                pass

            # Get pi1's raw commands for display
            pi1_cmd = actions[0, :4].cpu().numpy()

            # Print HUD every 10 steps
            if timestep % 10 == 0:
                print_hud(user_vx, user_vy, target_h, ball_h, pi1_cmd, timestep)

            # Real-time sleep
            elapsed = time.time() - start_time
            if args_cli.real_time and elapsed < dt:
                time.sleep(dt - elapsed)

    finally:
        kb.restore_terminal()
        print("\n[INFO] Teleop session ended.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
