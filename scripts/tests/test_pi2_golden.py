"""Collect golden test cases for pi2: 10 fixed 6D commands → (obs, action) pairs.

Runs the gym env (same as play_pi2.py), injects each command, waits 300 steps
for the robot to reach steady state, then records the obs and action.

Output:
    tests_out/pi2_golden.csv   — one row per test case

Columns:
    cmd_*           raw 6D torso command fed to pi2
    obs_cmd_norm_*  normalized command (obs[0:6])
    obs_linvel_*    body-frame linear velocity (obs[6:9])
    obs_angvel_*    body-frame angular velocity (obs[9:12])
    obs_grav_*      projected gravity (obs[12:15])
    obs_jp_*        joint pos relative to default (obs[15:27])
    obs_jv_*        joint vel (obs[27:39])
    act_*           raw policy output (12D)
    tgt_*           joint position targets sent to actuator = default + 0.25*act

Usage:
    python scripts/tests/test_pi2_golden.py --headless
"""

import argparse, os, sys

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--pi2_checkpoint",
                default=os.path.join(os.path.dirname(__file__),
                    "../../logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt"))
ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                    "../../tests_out/pi2_golden.csv"))
ap.add_argument("--settle_steps", type=int, default=300,
                help="Steps to let robot settle before recording each case")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import gymnasium as gym

import isaaclab_tasks   # noqa: F401
import go1_ball_balance # noqa: F401

from go1_ball_balance.tasks.torso_tracking.torso_tracking_env_cfg import TorsoTrackingEnvCfg_PLAY

# ── 10 test cases: [h, h_dot, roll, pitch, roll_rate, pitch_rate] ─────────────
TEST_CASES = [
    {"name": "neutral",          "cmd": [0.375, 0.0,  0.0,  0.0,  0.0,  0.0]},
    {"name": "low",              "cmd": [0.300, 0.0,  0.0,  0.0,  0.0,  0.0]},
    {"name": "mid_low",          "cmd": [0.350, 0.0,  0.0,  0.0,  0.0,  0.0]},
    {"name": "mid_high",         "cmd": [0.420, 0.0,  0.0,  0.0,  0.0,  0.0]},
    {"name": "high",             "cmd": [0.470, 0.0,  0.0,  0.0,  0.0,  0.0]},
    {"name": "roll_pos",         "cmd": [0.375, 0.0,  0.15, 0.0,  0.0,  0.0]},
    {"name": "roll_neg",         "cmd": [0.375, 0.0, -0.15, 0.0,  0.0,  0.0]},
    {"name": "pitch_pos",        "cmd": [0.375, 0.0,  0.0,  0.15, 0.0,  0.0]},
    {"name": "pitch_neg",        "cmd": [0.375, 0.0,  0.0, -0.15, 0.0,  0.0]},
    {"name": "mixed",            "cmd": [0.420, 0.0,  0.10, 0.10, 0.0,  0.0]},
]

# ── Normalization (observations.py) ──────────────────────────────────────────
_NORM   = np.array([1/0.125, 1/1.0, 1/0.4, 1/0.4, 1/3.0, 1/3.0])
_OFFSET = np.array([-0.375,   0.0,   0.0,   0.0,   0.0,   0.0])
ACTION_SCALE = 0.25


def _build_actor(path, device):
    ck = torch.load(path, map_location=device, weights_only=True)
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
        idx = int(key.split(".")[1])
        actor_sd[f"{idx}.weight"] = sd[key]
        bkey = key.replace("weight", "bias")
        if bkey in sd:
            actor_sd[f"{idx}.bias"] = sd[bkey]
    actor.load_state_dict(actor_sd)
    actor.eval()
    for p in actor.parameters(): p.requires_grad = False
    return actor


# ── Env ───────────────────────────────────────────────────────────────────────
env_cfg = TorsoTrackingEnvCfg_PLAY()
env_cfg.scene.num_envs = 1
env = gym.make("Isaac-TorsoTracking-Go1-Play-v0", cfg=env_cfg)
device = env.unwrapped.device

actor = _build_actor(os.path.abspath(args.pi2_checkpoint), str(device))
robot = env.unwrapped.scene["robot"]
jnames = list(robot.joint_names)   # Isaac Lab joint ordering

# default joint pos for computing targets
default_jp = robot.data.default_joint_pos[0].cpu().numpy()   # (12,)

print(f"[golden] joint_names : {jnames}")
print(f"[golden] default_pos : {np.round(default_jp, 3)}")
print(f"[golden] settle_steps: {args.settle_steps}")
print(f"[golden] {len(TEST_CASES)} test cases\n")


def inject_cmd(cmd_t):
    env.unwrapped._torso_cmd      = cmd_t.clone()
    env.unwrapped._torso_cmd_goal = cmd_t.clone()


def run_case(cmd_list):
    """Reset env, inject fixed cmd, settle, return (obs_np, action_np)."""
    cmd_t = torch.tensor(cmd_list, dtype=torch.float32, device=device).unsqueeze(0)

    # warmup: double reset + zero step (same as play_pi2_debug)
    obs_dict, _ = env.reset(); inject_cmd(cmd_t)
    obs_dict, _ = env.reset(); inject_cmd(cmd_t)
    with torch.no_grad():
        zero = torch.zeros(1, 12, device=device)
        obs_dict, _, _, _, _ = env.step(zero); inject_cmd(cmd_t)

    # settle for N steps under the fixed command
    for _ in range(args.settle_steps - 1):
        inject_cmd(cmd_t)
        with torch.no_grad():
            actions = actor(obs_dict["policy"].to(device))
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        if terminated.any() or truncated.any():
            inject_cmd(cmd_t)

    # record: obs and action at the same timestep (before stepping)
    inject_cmd(cmd_t)
    obs_np = obs_dict["policy"][0].cpu().numpy()              # (39,) at time T
    with torch.no_grad():
        actions = actor(obs_dict["policy"].to(device))
    action_np = actions[0].cpu().numpy()                      # actor(obs at T)
    return obs_np, action_np


# ── Run all cases ─────────────────────────────────────────────────────────────
rows = []
for i, case in enumerate(TEST_CASES):
    cmd = case["cmd"]
    print(f"  [{i+1:2d}/10] {case['name']:20s} cmd={np.round(cmd, 3)}", flush=True)

    obs_np, action_np = run_case(cmd)
    cmd_np     = np.array(cmd)
    cmd_norm   = (cmd_np + _OFFSET) * _NORM
    targets_np = default_jp + ACTION_SCALE * action_np

    trunk_z  = robot.data.root_pos_w[0, 2].item()
    grav_b   = robot.data.projected_gravity_b[0].cpu().numpy()
    tilt_deg = np.degrees(np.arccos(np.clip(-grav_b[2], -1, 1)))
    print(f"           trunk_z={trunk_z:.3f}m  tilt={tilt_deg:.1f}deg  "
          f"|act|_max={np.abs(action_np).max():.3f}", flush=True)

    row = {"case": case["name"]}
    for j, k in enumerate(["h", "h_dot", "roll", "pitch", "roll_rate", "pitch_rate"]):
        row[f"cmd_{k}"]          = cmd_np[j]
        row[f"obs_cmd_norm_{k}"] = cmd_norm[j]
    for j, k in enumerate(["x", "y", "z"]):
        row[f"obs_linvel_{k}"] = obs_np[6+j]
        row[f"obs_angvel_{k}"] = obs_np[9+j]
        row[f"obs_grav_{k}"]   = obs_np[12+j]
    for j, jn in enumerate(jnames):
        row[f"obs_jp_{jn}"]  = obs_np[15+j]
        row[f"obs_jv_{jn}"]  = obs_np[27+j]
        row[f"act_{jn}"]     = action_np[j]
        row[f"tgt_{jn}"]     = targets_np[j]

    rows.append(row)

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
df = pd.DataFrame(rows)
df.to_csv(args.out, index=False, float_format="%.6f")
print(f"\n[golden] saved {len(rows)} rows → {args.out}")
print(df[["case"] + [c for c in df.columns if c.startswith("cmd_")]].to_string(index=False))

env.close()
simulation_app.close()
