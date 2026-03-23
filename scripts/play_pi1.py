"""Play script: learned pi1 + frozen pi2 hierarchical ball juggling.

Loads a trained pi1 checkpoint and runs the full hierarchical system:
    Learned pi1 (RL) → 6D torso command → frozen pi2 → 12D joints → Go1

Usage:
    cd /home/frank/berkeley_mde/QuadruJuggle
    PYTHONPATH=/home/frank/IsaacLab/scripts/reinforcement_learning/rsl_rl:$PYTHONPATH \\
    python scripts/play_pi1.py \\
        --pi1_checkpoint logs/rsl_rl/go1_ball_juggle_hier/TIMESTAMP/model_best.pt \\
        --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/TIMESTAMP/model_best.pt \\
        --num_envs 4
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Learned pi1 ball juggling play.")
parser.add_argument("--pi1_checkpoint", type=str, required=True,
                    help="Path to trained pi1 checkpoint .pt")
parser.add_argument("--pi2_checkpoint", type=str, required=True,
                    help="Path to frozen pi2 (torso-tracking) checkpoint .pt")
parser.add_argument("--num_envs",       type=int,   default=4)
parser.add_argument("--apex_height",    type=float, default=None,
                    help="Fix target apex height [m] for all envs (e.g. 0.20). "
                         "If omitted, uses random heights from 0.10–0.60 m.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
import go1_ball_balance  # noqa: F401

from go1_ball_balance.tasks.ball_juggle_hier.ball_juggle_pi1_env_cfg import (
    BallJugglePi1EnvCfg_PLAY,
)
from go1_ball_balance.tasks.ball_juggle_hier.agents.rsl_rl_ppo_cfg import (
    BallJuggleHierPPORunnerCfg,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# ── Build env config ────────────────────────────────────────────────────────
env_cfg = BallJugglePi1EnvCfg_PLAY()
env_cfg.scene.num_envs = args.num_envs
env_cfg.actions.torso_cmd.pi2_checkpoint = os.path.abspath(args.pi2_checkpoint)
env_cfg.observations.policy.enable_corruption = False

# ── Create environment ──────────────────────────────────────────────────────
env = gym.make("Isaac-BallJugglePi1-Go1-v0", cfg=env_cfg)

device = env.unwrapped.device

# ── Load trained pi1 policy ─────────────────────────────────────────────────
agent_cfg = BallJuggleHierPPORunnerCfg()
env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=str(device))
pi1_path = os.path.abspath(args.pi1_checkpoint)
print(f"[play_pi1] Loading pi1 from: {pi1_path}")
runner.load(pi1_path)
policy = runner.get_inference_policy(device=str(device))

print(f"\n[play_pi1] pi1 checkpoint : {args.pi1_checkpoint}")
print(f"[play_pi1] pi2 checkpoint : {args.pi2_checkpoint}")
print(f"[play_pi1] num_envs       : {args.num_envs}")
if args.apex_height is not None:
    print(f"[play_pi1] apex_height    : {args.apex_height:.3f} m (fixed)")
else:
    print("[play_pi1] apex_height    : random 0.10–0.60 m per episode")
print("[play_pi1] Running — close window or Ctrl+C to stop.\n")

obs_raw, _ = env.reset()

# Override apex height for play if requested
if args.apex_height is not None:
    env.unwrapped._apex_target_h = torch.full(
        (args.num_envs,), args.apex_height, device=device
    )
    env.unwrapped._apex_target_std = torch.full(
        (args.num_envs,), 0.10, device=device
    )
step = 0
episode_rewards = torch.zeros(args.num_envs, device=device)
episode_lengths = torch.zeros(args.num_envs, device=device)

try:
    while simulation_app.is_running():
        with torch.no_grad():
            obs_tensor = obs_raw["policy"].to(device)   # (N, 46)
            actions = policy({"policy": obs_tensor})    # (N, 6)
            obs_raw, rew, terminated, truncated, _ = env.step(actions)
            dones = (terminated | truncated).long()

        episode_rewards += rew
        episode_lengths += 1

        if dones.any():
            for i in dones.nonzero(as_tuple=False).squeeze(-1).tolist():
                print(
                    f"  env {i:3d} | ep_len={int(episode_lengths[i].item()):4d} "
                    f"| ep_rew={episode_rewards[i].item():.1f}"
                )
                episode_rewards[i] = 0.0
                episode_lengths[i] = 0.0

        step += 1
        if step % 200 == 0:
            ball  = env.unwrapped.scene["ball"]
            robot = env.unwrapped.scene["robot"]
            paddle_z = robot.data.root_pos_w[0, 2].item() + 0.07
            ball_z   = ball.data.root_pos_w[0, 2].item()
            ball_vz  = ball.data.root_lin_vel_w[0, 2].item()
            print(
                f"  step {step:5d} | "
                f"paddle_z={paddle_z:.3f} ball_z={ball_z:.3f} ball_vz={ball_vz:+.2f} | "
                f"ep_rew(mean)={episode_rewards.mean().item():.1f}"
            )

except KeyboardInterrupt:
    print("\n[play_pi1] Stopped by user.")

env.close()
simulation_app.close()
