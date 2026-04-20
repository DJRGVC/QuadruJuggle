"""Minimal Isaac Lab spawn test.

Spawns the Go1 via TorsoTrackingEnvCfg and runs the pi2 policy (same as
BallJuggleMirror does internally).  This is the apples-to-apples comparison:
both envs use pi2 to fight gravity, so if TorsoTracking still ghosts it is a
scene/config bug rather than a missing policy.

Usage:
    cd /home/frank/berkeley_mde/QuadruJuggle
    PYTHONPATH=/home/frank/IsaacLab/scripts/reinforcement_learning/rsl_rl:$PYTHONPATH
    python scripts/tests/test_spawn_isaaclab.py           # opens viewport
    python scripts/tests/test_spawn_isaaclab.py --headless # no window
"""

import argparse, os, sys
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_steps", type=int, default=500)
parser.add_argument("--pi2_checkpoint", type=str,
    default=os.path.join(os.path.dirname(__file__), "../..",
        "logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import go1_ball_balance  # noqa: F401

from go1_ball_balance.tasks.torso_tracking.torso_tracking_env_cfg import TorsoTrackingEnvCfg_PLAY

env_cfg = TorsoTrackingEnvCfg_PLAY()
env_cfg.scene.num_envs = 1
env_cfg.terminations.trunk_collapsed.params["minimum_height"] = -10.0
env_cfg.episode_length_s = 500.0

env = gym.make("Isaac-TorsoTracking-Go1-Play-v0", cfg=env_cfg)
device = env.unwrapped.device
robot  = env.unwrapped.scene["robot"]

# Load pi2 (RSL-RL checkpoint) — same active stance control as BallJuggleMirror
def _load_pi2(path, device):
    ckpt = torch.load(path, map_location="cpu")
    sd   = ckpt.get("model_state_dict", ckpt)
    keys = sorted(k for k in sd if k.startswith("actor.") and "weight" in k)
    layers = [(sd[k].shape[1], sd[k].shape[0]) for k in keys]
    modules = []
    for i, (in_d, out_d) in enumerate(layers):
        modules.append(nn.Linear(in_d, out_d))
        if i < len(layers) - 1:
            modules.append(nn.ELU())
    net = nn.Sequential(*modules)
    actor_sd = {}
    for k, v in sd.items():
        if not k.startswith("actor."):
            continue
        parts = k[len("actor."):].split(".")
        actor_sd[f"{int(parts[0])}.{parts[1]}"] = v
    net.load_state_dict(actor_sd)
    return net.eval().to(device)

pi2_path = os.path.abspath(args.pi2_checkpoint)
pi2 = _load_pi2(pi2_path, device)
print(f"[spawn_test] Loaded pi2: {pi2_path}  input_dim={pi2[0].in_features}")

obs_dict, _ = env.reset()

print("[spawn_test] Running pi2 in TorsoTracking. trunk_z should stay ~0.35m.\n")

for step in range(args.num_steps):
    obs = obs_dict["policy"]                      # (1, 39)
    with torch.no_grad():
        action = pi2(obs)                         # (1, 12)

    obs_dict, _, terminated, truncated, _ = env.step(action)

    if step % 10 == 0:
        trunk_z = robot.data.root_pos_w[0, 2].item()
        print(f"  step {step:4d} | trunk_z = {trunk_z:.4f} m")

    if terminated.any() or truncated.any():
        print(f"  [reset at step {step}]")
        obs_dict, _ = env.reset()

    if not simulation_app.is_running():
        break

env.close()
simulation_app.close()
