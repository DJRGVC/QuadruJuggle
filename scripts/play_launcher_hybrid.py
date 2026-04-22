"""V4 Hybrid: Launcher pi1 → Mirror Law.

V4 improvement over V3 play_hybrid.py:
  - Uses launcher pi1 (trained to hit target ± 0.05 m precisely, not juggle forever)
  - Cleaner handoffs — launcher stops building energy once in window
  - Mirror law receives ball at the right height from the start, less decay

Pipeline:
    Launcher pi1 (precision) ──► Mirror Law (sustainer)
            ▲                           │
            └─── fallback if apex ──────┘
                  drops below threshold

Usage:
    cd /home/frank/berkeley_mde/QuadruJuggle
    PYTHONPATH=/home/frank/IsaacLab/scripts/reinforcement_learning/rsl_rl:$PYTHONPATH \\
    python scripts/play_launcher_hybrid.py \\
        --launcher_checkpoint logs/rsl_rl/go1_ball_launcher/TIMESTAMP/model_best.pt \\
        --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \\
        --apex_height 0.30 \\
        --num_envs 4
"""

import argparse
import glob
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="V4 Launcher pi1 + mirror-law hybrid.")
parser.add_argument("--launcher_checkpoint", type=str,
                    default=os.path.join(os.path.dirname(__file__),
                        "../logs/rsl_rl/go1_ball_launcher/2026-04-05_21-42-11/model_best.pt"),
                    help="Path to trained V4 launcher pi1 checkpoint .pt")
parser.add_argument("--pi2_checkpoint",      type=str,
                    default=os.path.join(os.path.dirname(__file__),
                        "../logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt"),
                    help="Path to frozen pi2 (torso-tracking) checkpoint .pt")
parser.add_argument("--num_envs",            type=int,   default=4)
parser.add_argument("--apex_height",         type=float, default=0.30,
                    help="Target apex height [m] (default 0.30)")
parser.add_argument("--switch_window",       type=float, default=0.10,
                    help="Pi1→mirror-law when |last_apex - target| < window [m] (default 0.10)")
parser.add_argument("--fallback_threshold",  type=float, default=0.50,
                    help="Mirror-law→pi1 when apex < fallback_threshold * target (default 0.50)")
parser.add_argument("--centering_gain",      type=float, default=2.0)
parser.add_argument("--h_nominal",           type=float, default=0.38)
parser.add_argument("--max_steps",           type=int,   default=0,
                    help="Stop after N steps (0 = run forever)")
parser.add_argument("--video",               action="store_true", default=False,
                    help="Record replay as MP4. Always overwrites videos/hybrid_latest.mp4.")
parser.add_argument("--video_length",        type=int,   default=500,
                    help="Number of steps to record when --video is set (default 500).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Video recording needs offscreen camera rendering.
# AppLauncher reads args.__dict__ directly, so mutate the namespace here.
# headless=True + enable_cameras=True → PARTIAL_RENDERING (supports rgb_array).
if args.video:
    args.headless = True
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import isaaclab.utils.math as math_utils
import matplotlib
matplotlib.use("Agg")   # headless-safe backend
import matplotlib.pyplot as plt

import isaaclab_tasks  # noqa: F401
import go1_ball_balance  # noqa: F401

from go1_ball_balance.tasks.ball_juggle_hier.ball_juggle_launcher_env_cfg import (
    BallJuggleLauncherEnvCfg_PLAY,
)


def _build_pi1_actor(checkpoint_path: str, device: str) -> nn.Module:
    """Load pi1 actor directly from checkpoint, bypassing rsl_rl's OnPolicyRunner.

    Works with any rsl_rl version — we just read the actor weights directly.
    """
    ck = torch.load(checkpoint_path, map_location=device, weights_only=True)
    sd = ck.get("model_state_dict", ck)

    # Infer architecture from weight shapes
    actor_keys = sorted([k for k in sd if k.startswith("actor.") and "weight" in k])
    layers = []
    for key in actor_keys:
        out_dim, in_dim = sd[key].shape
        layers.append((in_dim, out_dim))

    # Build MLP: linear → ELU → ... → linear (no final activation)
    modules: list[nn.Module] = []
    for i, (in_dim, out_dim) in enumerate(layers):
        modules.append(nn.Linear(in_dim, out_dim))
        if i < len(layers) - 1:
            modules.append(nn.ELU())
    actor = nn.Sequential(*modules).to(device)

    # Load weights (RSL-RL uses skip-2 indices: 0,2,4,6 for linear layers)
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
    print(f"[play_launcher_hybrid] pi1 actor loaded: {arch}")
    return actor

# Mirror-law command scaling (must match action_term.py)
_CMD_SCALES  = torch.tensor([0.125, 1.0, 0.4, 0.4, 3.0, 3.0])
_CMD_OFFSETS = torch.tensor([0.375, 0.0, 0.0, 0.0, 0.0, 0.0])
_GRAVITY     = 9.81
_RESTITUTION = 0.85


def mirror_law_cmd(
    ball_pos_w:   torch.Tensor,
    ball_vel_w:   torch.Tensor,
    robot_pos_w:  torch.Tensor,
    robot_quat_w: torch.Tensor,
    apex:         torch.Tensor,
    h_nominal:    float,
    centering_gain: float,
    device: str,
) -> torch.Tensor:
    """Return (N, 6) normalised torso command using mirror law geometry."""
    N = ball_pos_w.shape[0]
    paddle_offset_b = torch.tensor([0.0, 0.0, 0.070], device=device)
    offset_b = paddle_offset_b.unsqueeze(0).expand(N, -1)
    paddle_pos_w = robot_pos_w + math_utils.quat_apply(robot_quat_w, offset_b)
    p_rel_w = ball_pos_w - paddle_pos_w

    v_out_z = (2.0 * _GRAVITY * apex).sqrt().clamp(min=0.5)
    v_out_x = -centering_gain * p_rel_w[:, 0]
    v_out_y = -centering_gain * p_rel_w[:, 1]
    v_out_w = torch.stack([v_out_x, v_out_y, v_out_z], dim=-1)

    v_out_eff = v_out_w / max(_RESTITUTION, 0.1)
    n_raw = v_out_eff - ball_vel_w
    n_w = F.normalize(n_raw, dim=-1)
    flip = (n_w[:, 2] < 0).unsqueeze(-1).float()
    n_w = n_w * (1.0 - 2.0 * flip)

    quat_w2b = math_utils.quat_conjugate(robot_quat_w)
    n_b = math_utils.quat_apply(quat_w2b, n_w)
    nz_safe = n_b[:, 2].clamp(min=0.15)

    pitch_tgt = torch.atan2( n_b[:, 0], nz_safe).clamp(-0.4, 0.4)
    roll_tgt  = torch.atan2(-n_b[:, 1], nz_safe).clamp(-0.4, 0.4)

    ball_descending = (ball_vel_w[:, 2] < 0.0).float()
    near_impact     = (p_rel_w[:, 2] < 0.50).float()
    # v_in_z_abs = ball_vel_w[:, 2].abs()
    # v_paddle_target = (v_out_z - _RESTITUTION * v_in_z_abs) / (1.0 + _RESTITUTION)
    # v_paddle_cmd = (v_paddle_target / 0.5).clamp(0.0, 1.0)
    # h_dot_impulse = (v_paddle_cmd * ball_descending * near_impact).clamp(0.0, 1.0)



    ############Debugging see if clamp is triggered too much###########
    v_in_z_abs = ball_vel_w[:, 2].abs()
    
    # Calculate the mathematical target
    v_paddle_target = (v_out_z + _RESTITUTION * v_in_z_abs) / (1.0 + _RESTITUTION)
    
    # 1. Calculate the raw command WITHOUT chaining the .clamp()
    raw_paddle_cmd = v_paddle_target / 0.5
    
    # 2. Debug Print (Only for Environment 0)
    # if N > 0: 
    #     raw_val = raw_paddle_cmd[0].item()
    #     clamped_val = max(0.0, min(raw_val, 1.0)) # Simulate the clamp in Python
        
    #     # Only print when the clamp is actually doing something
    #     if raw_val != clamped_val:
    #         print(f"🚨 CLAMPED! Raw Math: {raw_val:.3f} | Forced to: {clamped_val:.3f} | Ball Vel Z: {ball_vel_w[0, 2].item():.3f}")
    
    # 3. Now actually apply the clamp for the simulation to use
    v_paddle_cmd = raw_paddle_cmd.clamp(0.0, 1.0)
    
    h_dot_impulse = (v_paddle_cmd * ball_descending * near_impact).clamp(0.0, 1.0)

    ####################################################################

    not_impacting = 1.0 - (ball_descending * near_impact).clamp(0.0, 1.0)
    h_dot_cmd = (h_dot_impulse + 0.15 * not_impacting).clamp(0.0, 1.0)

    h_cmd = torch.full((N,), h_nominal, device=device)
    zeros = torch.zeros(N, device=device)

    # Clamp to pi2 reliable tracking range (from eval_pi2 sweep)
    _clamp_checks = [
        ("h",     h_cmd,     0.32,  0.42),
        ("h_dot", h_dot_cmd, -0.8,  1.0),
        ("roll",  roll_tgt,  -0.4,  0.4),
        ("pitch", pitch_tgt, -0.4,  0.4),
    ]
    for _name, _val, _lo, _hi in _clamp_checks:
        _raw = _val[0].item()
        _clamped = max(_lo, min(_raw, _hi))
        if abs(_raw - _clamped) > 1e-6:
            print(f"⚠️  CLAMP [{_name}] raw={_raw:.3f} → {_clamped:.3f}")

    h_cmd     = h_cmd.clamp(0.32, 0.42)
    h_dot_cmd = h_dot_cmd.clamp(-0.8, 1.0)
    roll_tgt  = roll_tgt.clamp(-0.4, 0.4)
    pitch_tgt = pitch_tgt.clamp(-0.4, 0.4)

    cmd_phys = torch.stack([h_cmd, h_dot_cmd, roll_tgt, pitch_tgt, zeros, zeros], dim=-1)
    scales  = _CMD_SCALES.to(device)
    offsets = _CMD_OFFSETS.to(device)
    return (cmd_phys - offsets) / scales


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
env_cfg = BallJuggleLauncherEnvCfg_PLAY()
env_cfg.scene.num_envs = args.num_envs
env_cfg.actions.torso_cmd.pi2_checkpoint = os.path.abspath(args.pi2_checkpoint)
env_cfg.observations.policy.enable_corruption = False

_video_folder = os.path.join(os.path.dirname(__file__), "..", "videos", "hybrid")
os.makedirs(_video_folder, exist_ok=True)

env = gym.make("Isaac-BallJuggleLauncher-Go1-v0", cfg=env_cfg,
               render_mode="rgb_array" if args.video else None)
device = env.unwrapped.device

if args.video:
    for old in glob.glob(os.path.join(_video_folder, "hybrid_latest*.mp4")):
        os.remove(old)
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=_video_folder,
        name_prefix="hybrid_latest",
        step_trigger=lambda step: step == 0,
        video_length=args.video_length,
        disable_logger=True,
    )
    print(f"[play_launcher_hybrid] Recording {args.video_length} steps → {_video_folder}/hybrid_latest-episode-0.mp4")

_pi1_actor = _build_pi1_actor(os.path.abspath(args.launcher_checkpoint), str(device))

def policy(obs_dict: dict) -> torch.Tensor:
    return _pi1_actor(obs_dict["policy"])

# Fix apex height
env.unwrapped._apex_target_h   = torch.full((args.num_envs,), args.apex_height, device=device)
env.unwrapped._apex_target_std = torch.full((args.num_envs,), 0.03, device=device)

apex_tensor      = torch.full((args.num_envs,), args.apex_height, device=device)
fallback_apex    = args.fallback_threshold * args.apex_height

using_mirror     = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
prev_ball_vz     = torch.zeros(args.num_envs, device=device)
last_bounce_apex = torch.zeros(args.num_envs, device=device)
mode_steps       = torch.zeros(args.num_envs, device=device)
MIN_MODE_STEPS   = 20   # switch quickly — mirror law is more robust

print(f"\n[play_launcher_hybrid] launcher : {args.launcher_checkpoint}")
print(f"[play_launcher_hybrid] pi2      : {args.pi2_checkpoint}")
print(f"[play_launcher_hybrid] target   : {args.apex_height:.2f} m")
print(f"[play_launcher_hybrid] window   : ±{args.switch_window:.3f} m")
print(f"[play_launcher_hybrid] fallback : {args.fallback_threshold:.2f} × target = {fallback_apex:.2f} m")
print("[play_launcher_hybrid] Running — Ctrl+C to stop.\n")

obs_raw, _ = env.reset()
step = 0
episode_rewards = torch.zeros(args.num_envs, device=device)
episode_lengths = torch.zeros(args.num_envs, device=device)

# Ball height relative to paddle for env 0 (ball_z_world - paddle_z_world)
_ball_rel_z_log: list[float] = []
_mode_log:       list[int]   = []  # 0 = pi1, 1 = mirror law

# 6D torso command desired vs actual (physical units) for env 0
_CMD_SCALES_T  = _CMD_SCALES.clone()
_CMD_OFFSETS_T = _CMD_OFFSETS.clone()
_keys6d = ["h", "h_dot", "roll", "pitch", "omega_roll", "omega_pitch"]
_torso_des: dict[str, list[float]] = {k: [] for k in _keys6d}
_torso_act: dict[str, list[float]] = {k: [] for k in _keys6d}

# First-5s snapshot: ball xyz + 6D torso cmd (desired only), env 0
_POLICY_DT   = 0.02          # 4 × 0.005 s
_PLOT_STEPS  = int(10.0 / _POLICY_DT)   # 250 steps
_snap_ball:  list[list[float]] = []   # [[x,y,z], ...]
_snap_cmd:   list[list[float]] = []   # [[h,h_dot,roll,pitch,wr,wp], ...]

try:
    while simulation_app.is_running():
        ball  = env.unwrapped.scene["ball"]
        robot = env.unwrapped.scene["robot"]

        ball_pos_w   = ball.data.root_pos_w.clone()
        ball_vel_w   = ball.data.root_lin_vel_w.clone()
        robot_pos_w  = robot.data.root_pos_w.clone()
        robot_quat_w = robot.data.root_quat_w.clone()

        ball_vz = ball_vel_w[:, 2]
        paddle_z_vec = robot_pos_w[:, 2] + 0.07
        ball_above_paddle = (ball_pos_w[:, 2] - paddle_z_vec).clamp(min=0.0)

        # Detect bounce apex
        just_peaked = (prev_ball_vz > 0) & (ball_vz <= 0)
        last_bounce_apex = torch.where(just_peaked, ball_above_paddle, last_bounce_apex)
        prev_ball_vz = ball_vz.clone()

        mode_steps += 1

        # V4 switch: apex within ±switch_window of target (tighter than V3's ±0.08)
        switch_to_mirror = (
            (~using_mirror)
            & (last_bounce_apex >= apex_tensor - args.switch_window)
            & (last_bounce_apex <= apex_tensor + args.switch_window)
            & (last_bounce_apex > 0)
            & (mode_steps >= MIN_MODE_STEPS)
        )
        fallback_to_pi1 = (
            using_mirror
            & (last_bounce_apex < fallback_apex)
            & (last_bounce_apex > 0)
            & (mode_steps >= MIN_MODE_STEPS)
        )

        mode_steps = torch.where(switch_to_mirror | fallback_to_pi1,
                                 torch.zeros_like(mode_steps), mode_steps)
        using_mirror = (using_mirror | switch_to_mirror) & ~fallback_to_pi1

        with torch.no_grad():
            obs_tensor  = obs_raw["policy"].to(device)
            if step % 100 == 0:
                x = obs_raw["policy"][0].detach().cpu()
                # print("ball_pos:", x[0:3])
                # print("ball_vel:", x[3:6])
                # print("base_lin_vel:", x[6:9])
                # print("base_ang_vel:", x[9:12])
                # print("projected_gravity:", x[12:15])
                # print("joint_pos:", x[15:27])
                # print("joint_vel:", x[27:39])
                # print("target_apex:", x[39:40])
                # print("last_action:", x[40:46])

            pi1_actions = policy({"policy": obs_tensor})

            ml_actions = mirror_law_cmd(
                ball_pos_w, ball_vel_w, robot_pos_w, robot_quat_w,
                apex_tensor, args.h_nominal, args.centering_gain, device,
            )

            mask    = using_mirror.unsqueeze(-1).float()
            actions = ml_actions * mask + pi1_actions * (1.0 - mask)


            obs_raw, rew, terminated, truncated, _ = env.step(actions)

        dones = terminated | truncated
        episode_rewards += rew
        episode_lengths += 1

        if dones.any():
            reset_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            using_mirror[reset_ids]      = False
            last_bounce_apex[reset_ids]  = 0.0
            prev_ball_vz[reset_ids]      = 0.0
            mode_steps[reset_ids]        = 0.0
            env.unwrapped._apex_target_h[reset_ids] = args.apex_height
            for i in reset_ids.tolist():
                print(
                    f"  env {i:3d} | ep_len={int(episode_lengths[i].item()):4d} "
                    f"| ep_rew={episode_rewards[i].item():.1f}"
                )
                episode_rewards[i] = 0.0
                episode_lengths[i] = 0.0

        _ball_rel_z_log.append(ball_pos_w[0, 2].item() - (robot_pos_w[0, 2].item() + 0.07))
        _mode_log.append(int(using_mirror[0].item()))

        # Desired: de-normalize actions[0] → physical 6D torso command
        cmd_phys = actions[0].cpu() * _CMD_SCALES_T + _CMD_OFFSETS_T
        for ki, k in enumerate(_keys6d):
            _torso_des[k].append(cmd_phys[ki].item())

        # Actual: read robot state for env 0
        roll_a, pitch_a, _ = math_utils.euler_xyz_from_quat(robot_quat_w[0].unsqueeze(0))
        ang_vel_b = robot.data.root_ang_vel_b[0]   # body-frame angular velocity
        _torso_act["h"].append(robot_pos_w[0, 2].item())
        _torso_act["h_dot"].append(robot.data.root_lin_vel_w[0, 2].item())
        _torso_act["roll"].append(roll_a[0].item())
        _torso_act["pitch"].append(pitch_a[0].item())
        _torso_act["omega_roll"].append(ang_vel_b[0].item())
        _torso_act["omega_pitch"].append(ang_vel_b[1].item())

        if step < _PLOT_STEPS:
            _snap_ball.append(ball_pos_w[0].cpu().tolist())
            _snap_cmd.append(cmd_phys.tolist())
            if step == _PLOT_STEPS - 1:
                print(f"[play_launcher_hybrid] 5s snapshot complete ({_PLOT_STEPS} steps) — saving plot.")
                break

        step += 1
        if args.video and step >= args.video_length:
            print(f"[play_launcher_hybrid] Video recorded ({args.video_length} steps). Exiting.")
            break
        if args.max_steps > 0 and step >= args.max_steps:
            break

        if step % 200 == 0:
            paddle_z = robot_pos_w[0, 2].item() + 0.07
            bz  = ball_pos_w[0, 2].item()
            bvz = ball_vel_w[0, 2].item()
            n_mirror = using_mirror.sum().item()
            apex0 = last_bounce_apex[0].item()
            print(
                f"  step {step:5d} | paddle_z={paddle_z:.3f} ball_z={bz:.3f} "
                f"ball_vz={bvz:+.2f} | last_apex={apex0:.2f}m "
                f"| mirror={n_mirror}/{args.num_envs} envs "
                f"| ep_rew(mean)={episode_rewards.mean().item():.1f}"
            )

except KeyboardInterrupt:
    print("\n[play_launcher_hybrid] Stopped by user.")

# ── Helper: shade mirror-law regions on an axes ─────────────────────────────
def _shade_mirror(ax, mode_arr):
    in_mirror, seg_start, labeled = False, 0, False
    for i, m in enumerate(mode_arr):
        if m == 1 and not in_mirror:
            seg_start, in_mirror = i, True
        elif m == 0 and in_mirror:
            ax.axvspan(seg_start, i, alpha=0.15, color="orange",
                       label="mirror law" if not labeled else "")
            in_mirror, labeled = False, True
    if in_mirror:
        ax.axvspan(seg_start, len(mode_arr), alpha=0.15, color="orange",
                   label="mirror law" if not labeled else "")


# ── Helper: shade both modes with time x-axis ────────────────────────────────
def _shade_modes_t(ax, mode_arr, t):
    """Orange = mirror law, cornflower-blue = RL (pi1)."""
    lbl = {0: True, 1: True}
    colors = {0: ("cornflowerblue", "RL (pi1)"), 1: ("orange", "mirror law")}
    cur_mode, seg_start = mode_arr[0], 0
    def _flush(mode, start, end):
        c, name = colors[mode]
        ax.axvspan(t[start], t[min(end, len(t)-1)], alpha=0.18, color=c,
                   label=name if lbl[mode] else "")
        lbl[mode] = False
    for i, m in enumerate(mode_arr):
        if m != cur_mode:
            _flush(cur_mode, seg_start, i - 1)
            cur_mode, seg_start = m, i
    _flush(cur_mode, seg_start, len(mode_arr) - 1)


# ── Plot 1: Ball trajectory ──────────────────────────────────────────────────
if _ball_rel_z_log:
    steps_arr = list(range(len(_ball_rel_z_log)))
    mode_arr  = _mode_log

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(steps_arr, _ball_rel_z_log, linewidth=0.8, color="steelblue", label="ball z (rel. paddle)")
    _shade_mirror(ax, mode_arr)
    ax.axhline(args.apex_height, color="red", linestyle="--", linewidth=0.8,
               label=f"target apex {args.apex_height:.2f} m")
    ax.set_xlabel("Step")
    ax.set_ylabel("Ball height above paddle [m]")
    ax.set_title("V4 Hybrid — Ball Height Trajectory (env 0)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(_video_folder, "hybrid_ball_traj.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[play_launcher_hybrid] Ball trajectory saved → {path}")

# ── Plot 2: 6D torso desired vs actual ──────────────────────────────────────
if _torso_des["h"]:
    steps_arr = list(range(len(_torso_des["h"])))
    mode_arr  = _mode_log

    panels = [
        ("h",           "Height [m]",      (0.20, 0.55)),
        ("h_dot",       "h_dot [m/s]",     (-1.2, 1.2)),
        ("roll",        "Roll [rad]",       (-0.50, 0.50)),
        ("pitch",       "Pitch [rad]",      (-0.50, 0.50)),
        ("omega_roll",  "ω_roll [rad/s]",   (-3.5, 3.5)),
        ("omega_pitch", "ω_pitch [rad/s]",  (-3.5, 3.5)),
    ]

    fig, axes = plt.subplots(6, 1, figsize=(12, 14), sharex=True)
    for ax, (key, ylabel, ylim) in zip(axes, panels):
        ax.plot(steps_arr, _torso_des[key], linewidth=0.8, color="red",       label="desired")
        ax.plot(steps_arr, _torso_act[key], linewidth=0.8, color="steelblue", label="actual", alpha=0.85)
        _shade_mirror(ax, mode_arr)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)

    axes[-1].set_xlabel("Step")
    fig.suptitle("V4 Hybrid — 6D Torso: Desired vs Actual (env 0)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(_video_folder, "hybrid_torso_cmd.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[play_launcher_hybrid] Torso command saved → {path}")

# ── Plot 3: First-5s ball position + 6D torso command ────────────────────────
if _snap_ball:
    import numpy as np
    n   = len(_snap_ball)
    t   = np.arange(n) * _POLICY_DT
    bp  = np.array(_snap_ball)    # (n, 3)
    cmd = np.array(_snap_cmd)     # (n, 6)
    cmd_labels = ["h [m]", "h_dot [m/s]", "roll [rad]", "pitch [rad]",
                  "ω_roll [rad/s]", "ω_pitch [rad/s]"]

    mode_snap = _mode_log[:n]
    fig, axes = plt.subplots(9, 1, figsize=(12, 18), sharex=True)

    for ai, (label, data, color) in enumerate([
        ("ball x [m]", bp[:, 0], "steelblue"),
        ("ball y [m]", bp[:, 1], "darkorange"),
        ("ball z [m]", bp[:, 2], "green"),
    ]):
        axes[ai].plot(t, data, linewidth=0.9, color=color)
        _shade_modes_t(axes[ai], mode_snap, t)
        axes[ai].set_ylabel(label, fontsize=8)
        axes[ai].legend(loc="upper right", fontsize=7)
        axes[ai].grid(True, alpha=0.3)

    for ai, (label, col) in enumerate(zip(cmd_labels, range(6))):
        ax = axes[3 + ai]
        ax.plot(t, cmd[:, col], linewidth=0.9, color="crimson")
        _shade_modes_t(ax, mode_snap, t)
        ax.set_ylabel(label, fontsize=8)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"V4 Hybrid — First {n*_POLICY_DT:.1f}s: Ball Position & 6D Torso Cmd (env 0)", fontsize=11)
    fig.tight_layout()
    path = os.path.join(_video_folder, "hybrid_first5s.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[play_launcher_hybrid] First-5s plot saved → {path}")

env.close()
simulation_app.close()
