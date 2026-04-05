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
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="V4 Launcher pi1 + mirror-law hybrid.")
parser.add_argument("--launcher_checkpoint", type=str, required=True,
                    help="Path to trained V4 launcher pi1 checkpoint .pt")
parser.add_argument("--pi2_checkpoint",      type=str, required=True,
                    help="Path to frozen pi2 (torso-tracking) checkpoint .pt")
parser.add_argument("--num_envs",            type=int,   default=4)
parser.add_argument("--apex_height",         type=float, default=0.30,
                    help="Target apex height [m] (default 0.30)")
parser.add_argument("--switch_window",       type=float, default=0.05,
                    help="Pi1→mirror-law when |last_apex - target| < window [m] (default 0.05)")
parser.add_argument("--fallback_threshold",  type=float, default=0.50,
                    help="Mirror-law→pi1 when apex < fallback_threshold * target (default 0.50)")
parser.add_argument("--centering_gain",      type=float, default=2.0)
parser.add_argument("--h_nominal",           type=float, default=0.38)
parser.add_argument("--max_steps",           type=int,   default=0,
                    help="Stop after N steps (0 = run forever)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import torch.nn.functional as F
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
import isaaclab.utils.math as math_utils

import isaaclab_tasks  # noqa: F401
import go1_ball_balance  # noqa: F401

from go1_ball_balance.tasks.ball_juggle_hier.ball_juggle_launcher_env_cfg import (
    BallJuggleLauncherEnvCfg_PLAY,
)
from go1_ball_balance.tasks.ball_juggle_hier.agents.rsl_rl_ppo_cfg import BallJuggleHierPPORunnerCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

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
    v_in_z_abs = ball_vel_w[:, 2].abs()
    v_paddle_target = (v_out_z + _RESTITUTION * v_in_z_abs) / (1.0 + _RESTITUTION)
    v_paddle_cmd = (v_paddle_target / 0.5).clamp(0.0, 1.0)
    h_dot_impulse = (v_paddle_cmd * ball_descending * near_impact).clamp(0.0, 1.0)
    not_impacting = 1.0 - (ball_descending * near_impact).clamp(0.0, 1.0)
    h_dot_cmd = (h_dot_impulse + 0.15 * not_impacting).clamp(0.0, 1.0)

    h_cmd = torch.full((N,), h_nominal, device=device)
    zeros = torch.zeros(N, device=device)

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

env = gym.make("Isaac-BallJuggleLauncher-Go1-v0", cfg=env_cfg)
device = env.unwrapped.device

agent_cfg = BallJuggleHierPPORunnerCfg()
env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=str(device))
runner.load(os.path.abspath(args.launcher_checkpoint))
policy = runner.get_inference_policy(device=str(device))

# Fix apex height
env.unwrapped._apex_target_h   = torch.full((args.num_envs,), args.apex_height, device=device)
env.unwrapped._apex_target_std = torch.full((args.num_envs,), 0.03, device=device)

apex_tensor      = torch.full((args.num_envs,), args.apex_height, device=device)
fallback_apex    = args.fallback_threshold * args.apex_height

using_mirror     = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
prev_ball_vz     = torch.zeros(args.num_envs, device=device)
last_bounce_apex = torch.zeros(args.num_envs, device=device)
mode_steps       = torch.zeros(args.num_envs, device=device)
MIN_MODE_STEPS   = 50   # shorter than V3 — launcher handoffs are cleaner

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

        step += 1
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

env.close()
simulation_app.close()
