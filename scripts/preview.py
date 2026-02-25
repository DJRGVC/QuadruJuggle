"""Preview the ball-balance scene with zero-action stepping.

Opens Isaac Sim with the GUI so you can visually inspect the scene,
ball positioning, and reward values.  No checkpoint required.

Run:
    cd ~/IsaacLab
    python ~/Research/QuadruJuggle/scripts/preview.py [--num_envs 4]
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Preview the ball-balance scene.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=500, help="Steps to run before exiting (0 = run forever).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports (require Isaac Sim to be running) ────────────────────
import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv

# Register our custom gym environments
import go1_ball_balance  # noqa: F401

from go1_ball_balance.tasks.ball_balance.ball_balance_env_cfg import BallBalanceEnvCfg


def main():
    env_cfg = BallBalanceEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.scene.env_spacing = 3.5
    # Disable observation noise for a clean preview
    env_cfg.observations.policy.enable_corruption = False

    env = ManagerBasedRLEnv(cfg=env_cfg)

    obs, _ = env.reset()
    zero_action = torch.zeros(env.num_envs, env.action_space.shape[1], device=env.device)

    print("\n=== Ball-Balance Scene Preview ===")
    print(f"  Num envs:      {env.num_envs}")
    print(f"  Obs dim:       {obs['policy'].shape[-1]}")
    print(f"  Action dim:    {env.action_space.shape[1]}")
    print(f"  Steps planned: {args_cli.steps if args_cli.steps > 0 else '∞'}")
    print()
    print("Ball-obs layout (first 6 dims):")
    print("  [0:3]  ball XYZ in paddle frame (m)")
    print("  [3:6]  ball vel in trunk frame (m/s)")
    print()

    step = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            obs, rew, terminated, truncated, info = env.step(zero_action)

        if step % 50 == 0:
            ball_pos_rel = obs["policy"][0, 0:3]
            ball_vel_rel = obs["policy"][0, 3:6]
            print(
                f"  step {step:>4d} | "
                f"ball_pos_rel (cm): [{ball_pos_rel[0]*100:+5.1f}, {ball_pos_rel[1]*100:+5.1f}, {ball_pos_rel[2]*100:+5.1f}] | "
                f"reward[0]: {rew[0].item():.3f}"
            )

        step += 1
        if args_cli.steps > 0 and step >= args_cli.steps:
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
