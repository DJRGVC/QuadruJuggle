"""Play script: pi2 torso-tracking policy (standalone, no ball).

Loads a trained pi2 checkpoint and runs it in the TorsoTracking-Play env.
The env resamples random 6D commands (h, h_dot, roll, pitch, ω_roll, ω_pitch)
with smooth blending so the robot must continuously track changing targets.

After the run, saves a 4-panel plot comparing desired vs actual for env 0:
    videos/pi2_tracking.png  (always overwritten)

Usage:
    cd /home/frank/berkeley_mde/QuadruJuggle
    PYTHONPATH=/home/frank/IsaacLab/scripts/reinforcement_learning/rsl_rl:$PYTHONPATH \\
    python scripts/play_pi2.py \\
        --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \\
        --num_envs 4

Optional flags:
    --num_envs      Number of parallel envs (default 4)
    --max_steps     Stop after N steps (default 1000, 0 = run forever)
    --video         Record replay as MP4 → videos/pi2_latest.mp4 (always overwritten)
    --video_length  Steps to record when --video is set (default 500)
"""

import argparse
import glob
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pi2 torso-tracking standalone play.")
parser.add_argument("--pi2_checkpoint", type=str,
                    default="logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt",
                    help="Path to trained pi2 checkpoint .pt")
parser.add_argument("--num_envs",    type=int, default=4)
parser.add_argument("--max_steps",   type=int, default=1000,
                    help="Stop after N steps (0 = run forever)")
parser.add_argument("--video",       action="store_true", default=False,
                    help="Record replay as MP4. Always overwrites videos/pi2_latest.mp4.")
parser.add_argument("--video_length", type=int, default=500,
                    help="Steps to record when --video is set (default 500).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.video:
    args.headless = True
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import gymnasium as gym
import isaaclab.utils.math as math_utils

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import isaaclab_tasks  # noqa: F401
import go1_ball_balance  # noqa: F401

from go1_ball_balance.tasks.torso_tracking.torso_tracking_env_cfg import TorsoTrackingEnvCfg_PLAY


def _build_actor(checkpoint_path: str, device: str) -> nn.Module:
    """Load actor directly from checkpoint, bypassing rsl_rl's OnPolicyRunner."""
    ck = torch.load(checkpoint_path, map_location=device, weights_only=True)
    sd = ck.get("model_state_dict", ck)

    actor_keys = sorted([k for k in sd if k.startswith("actor.") and "weight" in k])
    layers = [(sd[k].shape[1], sd[k].shape[0]) for k in actor_keys]

    modules: list[nn.Module] = []
    for i, (in_dim, out_dim) in enumerate(layers):
        modules.append(nn.Linear(in_dim, out_dim))
        if i < len(layers) - 1:
            modules.append(nn.ELU())
    actor = nn.Sequential(*modules).to(device)

    actor_sd: dict[str, torch.Tensor] = {}
    for key in actor_keys:
        seq_idx = int(key.split(".")[1])
        actor_sd[f"{seq_idx}.weight"] = sd[key]
        bias_key = key.replace("weight", "bias")
        if bias_key in sd:
            actor_sd[f"{seq_idx}.bias"] = sd[bias_key]
    actor.load_state_dict(actor_sd)

    for p in actor.parameters():
        p.requires_grad = False
    actor.eval()

    arch = " → ".join(f"{i}→{o}" for i, o in layers)
    print(f"[play_pi2] actor loaded: {arch}")
    return actor

# ── Video folder ────────────────────────────────────────────────────────────
_video_folder = os.path.join(os.path.dirname(__file__), "..", "videos", "pi2")
os.makedirs(_video_folder, exist_ok=True)

# ── Env ─────────────────────────────────────────────────────────────────────
env_cfg = TorsoTrackingEnvCfg_PLAY()
env_cfg.scene.num_envs = args.num_envs

env = gym.make("Isaac-TorsoTracking-Go1-Play-v0", cfg=env_cfg,
               render_mode="rgb_array" if args.video else None)
device = env.unwrapped.device

if args.video:
    for old in glob.glob(os.path.join(_video_folder, "pi2_latest*.mp4")):
        os.remove(old)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=_video_folder,
        name_prefix="pi2_latest",
        step_trigger=lambda step: step == 0,
        video_length=args.video_length,
        disable_logger=True,
    )
    print(f"[play_pi2] Recording {args.video_length} steps → {_video_folder}/pi2_latest-episode-0.mp4")

# ── Load pi2 policy ─────────────────────────────────────────────────────────
_actor = _build_actor(os.path.abspath(args.pi2_checkpoint), str(device))

def policy(obs_dict: dict) -> torch.Tensor:
    return _actor(obs_dict["policy"])

print(f"\n[play_pi2] checkpoint : {args.pi2_checkpoint}")
print(f"[play_pi2] num_envs   : {args.num_envs}")
print("[play_pi2] Running — Ctrl+C to stop.\n")

env.unwrapped._torso_smooth_enabled = True   # activate EMA blending (cfg sets it on cfg obj, not env)
obs_dict, _ = env.reset()
step = 0

# ── Tracking logs for env 0 ─────────────────────────────────────────────────
_keys6d = ["h", "h_dot", "roll", "pitch", "omega_roll", "omega_pitch"]
_goal: dict[str, list[float]] = {k: [] for k in _keys6d}  # before EMA (_torso_cmd_goal)
_ema:  dict[str, list[float]] = {k: [] for k in _keys6d}  # after  EMA (_torso_cmd)
_act:  dict[str, list[float]] = {k: [] for k in _keys6d}  # actual robot state

try:
    while simulation_app.is_running():
        robot = env.unwrapped.scene["robot"]

        # ── Commands: before EMA (goal) and after EMA (what policy sees) ──
        cmd_ema  = env.unwrapped._torso_cmd[0]                                    # after EMA
        cmd_goal = getattr(env.unwrapped, "_torso_cmd_goal", env.unwrapped._torso_cmd)[0]  # before EMA

        for ki, k in enumerate(_keys6d):
            _ema[k].append(cmd_ema[ki].item())
            _goal[k].append(cmd_goal[ki].item())

        # ── Actual robot state for env 0 ────────────────────────────────
        quat_w    = robot.data.root_quat_w[0]
        ang_vel_b = robot.data.root_ang_vel_b[0]
        roll_act, pitch_act, _ = math_utils.euler_xyz_from_quat(quat_w.unsqueeze(0))

        _act["h"].append(robot.data.root_pos_w[0, 2].item())
        _act["h_dot"].append(robot.data.root_lin_vel_w[0, 2].item())
        _act["roll"].append(roll_act[0].item())
        _act["pitch"].append(pitch_act[0].item())
        _act["omega_roll"].append(ang_vel_b[0].item())
        _act["omega_pitch"].append(ang_vel_b[1].item())

        # ── Policy step ─────────────────────────────────────────────────
        with torch.no_grad():
            obs_tensor = obs_dict["policy"].to(device)
            actions = policy({"policy": obs_tensor})
            obs_dict, _, _, _, _ = env.step(actions)

        step += 1
        if args.video and step >= args.video_length:
            print(f"[play_pi2] Video recorded ({args.video_length} steps). Exiting.")
            break
        if args.max_steps > 0 and step >= args.max_steps:
            break

        if step % 200 == 0:
            h_err = abs(_act["h"][-1]     - _ema["h"][-1])
            r_err = abs(_act["roll"][-1]  - _ema["roll"][-1])
            p_err = abs(_act["pitch"][-1] - _ema["pitch"][-1])
            print(f"  step {step:4d} | h_err={h_err:.3f}m  roll_err={r_err:.3f}rad  pitch_err={p_err:.3f}rad")

except KeyboardInterrupt:
    print("\n[play_pi2] Stopped by user.")

# ── Comparison plot ─────────────────────────────────────────────────────────
if _ema["h"]:
    steps_arr = list(range(len(_ema["h"])))
    panels = [
        ("h",           "Height [m]",      (0.20, 0.55)),
        ("h_dot",       "h_dot [m/s]",     (-1.2,  1.2)),
        ("roll",        "Roll [rad]",       (-0.5,  0.5)),
        ("pitch",       "Pitch [rad]",      (-0.5,  0.5)),
        ("omega_roll",  "ω_roll [rad/s]",   (-3.5,  3.5)),
        ("omega_pitch", "ω_pitch [rad/s]",  (-3.5,  3.5)),
    ]

    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    for ax, (key, ylabel, ylim) in zip(axes, panels):
        ax.plot(steps_arr, _goal[key], color="gray",      linewidth=0.7, linestyle="--", label="before EMA (goal)", alpha=0.8)
        ax.plot(steps_arr, _ema[key],  color="red",       linewidth=0.8, label="after EMA (cmd to policy)")
        ax.plot(steps_arr, _act[key],  color="steelblue", linewidth=0.8, label="actual", alpha=0.85)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(ylim)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Step")
    fig.suptitle("Pi2 Torso Tracking — Before EMA / After EMA / Actual (env 0)", fontsize=11)
    fig.tight_layout()

    plot_path = os.path.join(_video_folder, "pi2_tracking.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[play_pi2] Tracking plot saved → {plot_path}")

env.close()
simulation_app.close()
