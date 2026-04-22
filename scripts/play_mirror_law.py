"""Play script: mirror-law pi1 + frozen pi2 hierarchical ball juggling.

The mirror-law controller computes the paddle orientation from ball state
to reflect the ball toward a target apex height.  No pi1 training needed —
only the frozen pi2 (torso-tracking) checkpoint is required.

Usage:
    cd /home/frank/berkeley_mde/QuadruJuggle
    PYTHONPATH=/home/frank/IsaacLab/scripts/reinforcement_learning/rsl_rl:$PYTHONPATH \\
    python scripts/play_mirror_law.py \\
        --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/TIMESTAMP/model_best.pt \\
        --apex_height 0.20 \\
        --num_envs 4

Optional flags:
    --num_envs       Number of parallel envs to visualise (default 4)
    --apex_height    Target ball apex height in metres above paddle (default 0.20)
    --h_nominal      Trunk height target in metres (default 0.38)
    --centering_gain Lateral correction gain (default 2.0)
    --video          Save replay as MP4 (always overwrites videos/mirror_law_latest.mp4)
    --video_length   Number of steps to record (default 500)
"""

import argparse
import glob
import math
import os
import sys

# AppLauncher MUST be imported and launched before any Isaac Sim / Isaac Lab imports
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Mirror-law ball juggling play.")
_BEST_PI2 = os.path.join(
    os.path.dirname(__file__), "..",
    "logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt"
)
parser.add_argument("--pi2_checkpoint", type=str, default=_BEST_PI2,
                    help="Path to trained pi2 (torso-tracking) checkpoint .pt")
parser.add_argument("--num_envs",       type=int,   default=4)
parser.add_argument("--apex_height",    type=float, default=0.20,
                    help="Target ball apex height above paddle [m]")
parser.add_argument("--h_nominal",      type=float, default=0.38,
                    help="Nominal trunk height command [m]")
parser.add_argument("--centering_gain", type=float, default=2.0)
parser.add_argument("--ball_pos_noise", type=float, default=0.0,
                    help="Gaussian std [m] added to ball position (simulates perception noise)")
parser.add_argument("--ball_vel_noise", type=float, default=0.0,
                    help="Gaussian std [m/s] added to ball velocity (simulates perception noise)")
parser.add_argument("--cmd_smooth_alpha", type=float, default=1.0,
                    help="EMA alpha for roll/pitch/h_dot commands [0.3–1.0]. "
                         "Use 0.3–0.5 with noisy velocity to reduce body shaking.")
parser.add_argument("--impact_tilt_gain", type=float, default=1.0,
                    help="Multiply roll/pitch during impact for stronger bounce. "
                         "1.0=mirror law only, 1.5-2.5=diagonal energy injection.")
parser.add_argument("--video", action="store_true", default=False,
                    help="Record replay as MP4. Always overwrites videos/mirror_law_latest.mp4.")
parser.add_argument("--video_length", type=int, default=500,
                    help="Number of steps to record when --video is set (default 500).")
parser.add_argument("--user-cmd", dest="user_cmd", choices=["none", "keyboard", "fixed"],
                    default="none",
                    help="Drive the 3D user slice (vx, vy, omega_yaw) of the 9D torso command. "
                         "'keyboard' uses Se2Keyboard (arrows + Z/X); 'fixed' uses --user-vx/vy/yaw.")
parser.add_argument("--user-vx",  dest="user_vx",  type=float, default=0.0,
                    help="Normalized [-1,1] forward velocity (±0.5 m/s at ±1). --user-cmd must be 'fixed'.")
parser.add_argument("--user-vy",  dest="user_vy",  type=float, default=0.0,
                    help="Normalized [-1,1] lateral velocity (±0.5 m/s at ±1). --user-cmd must be 'fixed'.")
parser.add_argument("--user-yaw", dest="user_yaw", type=float, default=0.0,
                    help="Normalized [-1,1] yaw rate (±1.5 rad/s at ±1). --user-cmd must be 'fixed'.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# --video needs rgb_array rendering; --headless otherwise disables it.
if args.video:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Isaac Sim / Isaac Lab imports (after AppLauncher) ───────────────────────
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401 — registers all Isaac Lab built-in tasks
import go1_ball_balance  # noqa: F401 — registers our custom tasks
import isaaclab.utils.math as math_utils

from go1_ball_balance.tasks.ball_juggle_hier.ball_juggle_mirror_env_cfg import (
    BallJuggleMirrorEnvCfg,
)

# ── Build env config ────────────────────────────────────────────────────────
env_cfg = BallJuggleMirrorEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env_cfg.scene.env_spacing = 3.5

# Frame the whole env grid instead of a single robot. This also affects recorded video.
grid_cols = max(1, math.ceil(math.sqrt(args.num_envs)))
grid_rows = max(1, math.ceil(args.num_envs / grid_cols))
grid_w = max(0.0, (grid_cols - 1) * env_cfg.scene.env_spacing)
grid_h = max(0.0, (grid_rows - 1) * env_cfg.scene.env_spacing)
cx = 0.5 * grid_w
cy = 0.5 * grid_h
env_cfg.viewer.lookat = (cx, cy, 0.55)
env_cfg.viewer.eye = (
    cx + max(4.5, 0.75 * grid_w + 3.0),
    cy + max(4.5, 0.75 * grid_h + 3.0),
    3.2 + 0.15 * max(grid_w, grid_h),
)

# Set pi2 checkpoint and mirror-law hyper-params
env_cfg.actions.torso_cmd.pi2_checkpoint = os.path.abspath(args.pi2_checkpoint)
env_cfg.actions.torso_cmd.apex_height_max = args.apex_height
env_cfg.actions.torso_cmd.apex_height_min = max(0.02, args.apex_height * 0.25)
env_cfg.actions.torso_cmd.h_nominal          = args.h_nominal
env_cfg.actions.torso_cmd.centering_gain     = args.centering_gain
env_cfg.actions.torso_cmd.ball_pos_noise_std = args.ball_pos_noise
env_cfg.actions.torso_cmd.ball_vel_noise_std = args.ball_vel_noise
env_cfg.actions.torso_cmd.cmd_smooth_alpha   = args.cmd_smooth_alpha
env_cfg.actions.torso_cmd.impact_tilt_gain  = args.impact_tilt_gain

env_cfg.observations.policy.enable_corruption = False

# ── Create environment ──────────────────────────────────────────────────────
# Isaac Lab's ManagerBasedRLEnv uses 'cfg' kwarg (not 'env_cfg')
env = gym.make("Isaac-BallJuggleMirror-Go1-v0", cfg=env_cfg,
               render_mode="rgb_array" if args.video else None)

if args.video:
    video_folder = os.path.join(os.path.dirname(__file__), "..", "videos", "mirror_law")
    os.makedirs(video_folder, exist_ok=True)
    # Delete previous recording so the file is always overwritten
    for old in glob.glob(os.path.join(video_folder, "mirror_law_latest*.mp4")):
        os.remove(old)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix="mirror_law_latest",
        step_trigger=lambda step: step == 0,
        video_length=args.video_length,
        disable_logger=True,
    )
    print(f"[play_mirror_law] Recording {args.video_length} steps → {video_folder}/mirror_law_latest-episode-0.mp4")

device = env.unwrapped.device
_uw = env.unwrapped

# Mirror law is deterministic — feed a fixed apex height (normalised 1.0 = max)
# Changing this to a value in [0, 1] scales the apex toward apex_height_min.
apex_action = torch.ones(args.num_envs, 1, device=device)
# apex_action = torch.tensor([[0.3], [0.2], [2.0], [1.0]], device=device)

# ── User-command injection (drives env._user_cmd_vel; 9D pi2 reads it) ──────
_user_keyboard = None
_user_fixed = None
if args.user_cmd == "keyboard":
    try:
        from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg
        _user_keyboard = Se2Keyboard(Se2KeyboardCfg(
            v_x_sensitivity=1.0,
            v_y_sensitivity=1.0,
            omega_z_sensitivity=1.0,
            sim_device=device,
        ))
        print("[USER-CMD] keyboard mode ON")
        print(str(_user_keyboard))
    except Exception as e:
        print(f"[USER-CMD] keyboard init failed: {e}; falling back to 'none'")
elif args.user_cmd == "fixed":
    _user_fixed = torch.tensor(
        [args.user_vx, args.user_vy, args.user_yaw],
        device=device,
    ).clamp(-1.0, 1.0)
    print(f"[USER-CMD] fixed mode: vx={_user_fixed[0]:.2f} vy={_user_fixed[1]:.2f} yaw={_user_fixed[2]:.2f} (normalized)")

def _apply_user_commands():
    """Write user commands into env._user_cmd_vel every step.

    TorsoCommandAction (parent of MirrorLawTorsoAction) lazily creates this
    buffer on first call. We seed it after reset below and refresh per-step.
    """
    cmd = None
    if _user_keyboard is not None:
        cmd = _user_keyboard.advance().clamp(-1.0, 1.0)
    elif _user_fixed is not None:
        cmd = _user_fixed
    if cmd is not None and hasattr(_uw, "_user_cmd_vel"):
        _uw._user_cmd_vel[:, 0] = cmd[0]
        _uw._user_cmd_vel[:, 1] = cmd[1]
        _uw._user_cmd_vel[:, 2] = cmd[2]


print(f"\n[play_mirror_law] pi2 checkpoint : {args.pi2_checkpoint}")
print(f"[play_mirror_law] apex_height    : {args.apex_height:.3f} m")
print(f"[play_mirror_law] h_nominal      : {args.h_nominal:.3f} m")
print(f"[play_mirror_law] centering_gain : {args.centering_gain:.2f}")
print(f"[play_mirror_law] ball_pos_noise : {args.ball_pos_noise:.4f} m")
print(f"[play_mirror_law] ball_vel_noise : {args.ball_vel_noise:.4f} m/s")
print(f"[play_mirror_law] cmd_smooth_alpha: {args.cmd_smooth_alpha:.2f}")
print(f"[play_mirror_law] num_envs       : {args.num_envs}")
print("[play_mirror_law] Running — close window or Ctrl+C to stop.\n")

obs, _ = env.reset()

# Ensure _user_cmd_vel buffer exists before the first step so fixed/keyboard
# commands take effect on iteration 0 (TorsoCommandAction would otherwise
# lazily create it mid-step).
if args.user_cmd != "none":
    from go1_ball_balance.tasks.torso_tracking.action_term import ensure_user_cmd_buffer
    ensure_user_cmd_buffer(_uw)
    _apply_user_commands()

step = 0
episode_rewards = torch.zeros(args.num_envs, device=device)
episode_lengths = torch.zeros(args.num_envs, device=device)

try:
    while simulation_app.is_running():
        _apply_user_commands()
        with torch.no_grad():
            obs, rew, terminated, truncated, info = env.step(apex_action)

        episode_rewards += rew
        episode_lengths += 1

        done = terminated | truncated
        if done.any():
            for i in done.nonzero(as_tuple=False).squeeze(-1).tolist():
                print(
                    f"  env {i:3d} | ep_len={int(episode_lengths[i].item()):4d} "
                    f"| ep_rew={episode_rewards[i].item():.1f}"
                )
                episode_rewards[i] = 0.0
                episode_lengths[i] = 0.0

        step += 1
        if args.video and step >= args.video_length:
            print(f"[play_mirror_law] Video recorded ({args.video_length} steps). Exiting.")
            break


except KeyboardInterrupt:
    print("\n[play_mirror_law] Stopped by user.")

env.close()
simulation_app.close()
