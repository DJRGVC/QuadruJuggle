"""Debug variant of play_pi2.py — dumps raw obs, action, and robot state
every N steps so you can compare directly against test_pi2_isaaclab.py.

Usage:
    python scripts/play_pi2_debug.py
    python scripts/play_pi2_debug.py --print_every 1   # print every step
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--pi2_checkpoint", type=str,
                    default="logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt")
parser.add_argument("--num_envs",    type=int, default=1)
parser.add_argument("--max_steps",   type=int, default=500)
parser.add_argument("--print_every", type=int, default=50,
                    help="Print debug info every N steps (default 50)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import isaaclab.utils.math as math_utils

import isaaclab_tasks  # noqa: F401
import go1_ball_balance  # noqa: F401

from go1_ball_balance.tasks.torso_tracking.torso_tracking_env_cfg import TorsoTrackingEnvCfg_PLAY


def _build_actor(checkpoint_path, device):
    ck = torch.load(checkpoint_path, map_location=device, weights_only=True)
    sd = ck.get("model_state_dict", ck)
    actor_keys = sorted(k for k in sd if k.startswith("actor.") and "weight" in k)
    layers = [(sd[k].shape[1], sd[k].shape[0]) for k in actor_keys]
    modules = []
    for i, (in_dim, out_dim) in enumerate(layers):
        modules.append(nn.Linear(in_dim, out_dim))
        if i < len(layers) - 1:
            modules.append(nn.ELU())
    actor = nn.Sequential(*modules).to(device)
    actor_sd = {}
    for key in actor_keys:
        seq_idx = int(key.split(".")[1])
        actor_sd[f"{seq_idx}.weight"] = sd[key]
        bias_key = key.replace("weight", "bias")
        if bias_key in sd:
            actor_sd[f"{seq_idx}.bias"] = sd[bias_key]
    actor.load_state_dict(actor_sd)
    actor.eval()
    for p in actor.parameters():
        p.requires_grad = False
    arch = " → ".join(f"{i}→{o}" for i, o in layers)
    print(f"[debug] actor arch: {arch}")
    return actor


env_cfg = TorsoTrackingEnvCfg_PLAY()
env_cfg.scene.num_envs = args.num_envs
env = gym.make("Isaac-TorsoTracking-Go1-Play-v0", cfg=env_cfg)
device = env.unwrapped.device

actor = _build_actor(os.path.abspath(args.pi2_checkpoint), str(device))

# two resets + a zero-action step to let the USD stage settle before recording
obs_dict, _ = env.reset()
obs_dict, _ = env.reset()
with torch.no_grad():
    zero_act = torch.zeros(args.num_envs, 12, device=device)
    obs_dict, _, _, _, _ = env.step(zero_act)
step = 0

print("\n[debug] Obs breakdown legend (matches test_pi2_isaaclab.py layout):")
print("  obs[0:6]   = torso_cmd_norm")
print("  obs[6:9]   = base_lin_vel_b")
print("  obs[9:12]  = base_ang_vel_b")
print("  obs[12:15] = projected_gravity_b")
print("  obs[15:27] = joint_pos_rel")
print("  obs[27:39] = joint_vel_rel\n")

try:
    while simulation_app.is_running():
        robot  = env.unwrapped.scene["robot"]
        obs_t  = obs_dict["policy"][0].cpu()   # env 0, shape (39,)

        with torch.no_grad():
            actions = actor(obs_dict["policy"].to(device))

        action_np = actions[0].cpu().numpy()
        obs_np    = obs_t.numpy()

        if step % args.print_every == 0:
            trunk_z  = robot.data.root_pos_w[0, 2].item()
            grav_b   = robot.data.projected_gravity_b[0].cpu().numpy()
            tilt_deg = np.degrees(np.arccos(np.clip(-grav_b[2], -1, 1)))
            torques  = robot.data.applied_torque[0].cpu().numpy()

            print(f"  step {step:5d} | trunk_z={trunk_z:.3f}m  tilt={tilt_deg:.1f}deg")
            print(f"    obs[0:6]  cmd_norm  : {obs_np[0:6].round(3)}")
            print(f"    obs[6:9]  lin_vel_b : {obs_np[6:9].round(3)}")
            print(f"    obs[9:12] ang_vel_b : {obs_np[9:12].round(3)}")
            print(f"    obs[12:15] grav_b   : {obs_np[12:15].round(3)}")
            print(f"    obs[15:27] jp_rel   : {obs_np[15:27].round(3)}")
            print(f"    obs[27:39] jv_rel   : {obs_np[27:39].round(3)}")
            print(f"    action              : {action_np.round(3)}  |max|={np.abs(action_np).max():.3f}")
            print(f"    torques             : {torques.round(1)}  |max|={np.abs(torques).max():.1f} Nm")

        obs_dict, _, terminated, truncated, _ = env.step(actions)
        if terminated.any() or truncated.any():
            reset_envs = (terminated | truncated).nonzero(as_tuple=False).squeeze(-1).tolist()
            print(f"  ── RESET at step {step} (envs {reset_envs}) ──────────────────────")
        step += 1
        if args.max_steps > 0 and step >= args.max_steps:
            break

except KeyboardInterrupt:
    print("\n[debug] Stopped.")

env.close()
simulation_app.close()
