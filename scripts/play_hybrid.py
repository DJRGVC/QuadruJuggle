"""Hybrid play: learned pi1 builds energy, mirror law maintains it.

Strategy
--------
1. Pi1 phase  — RL policy runs freely, building ball apex height each bounce.
2. Mirror law phase — once estimated apex >= switch_threshold * target,
   hand off to mirror law to hold the ball at that height precisely.
3. If apex drops below fallback_threshold * target (e.g. bad bounce), revert
   to pi1 to rebuild energy.

Both policies output 6D normalised torso commands, so the env never needs
to change — we just swap which tensor gets passed to env.step().

Usage
-----
    python scripts/play_hybrid.py \\
        --pi1_checkpoint logs/rsl_rl/go1_ball_juggle_hier/TIMESTAMP/model_best.pt \\
        --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \\
        --apex_height 0.50 \\
        --switch_threshold 0.80 \\
        --fallback_threshold 0.50 \\
        --num_envs 4
"""

import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Hybrid pi1 → mirror-law juggling.")
parser.add_argument("--pi1_checkpoint", type=str, required=True)
parser.add_argument("--pi2_checkpoint", type=str, required=True)
parser.add_argument("--num_envs",            type=int,   default=4)
parser.add_argument("--apex_height",         type=float, default=0.40,
                    help="Target apex height [m]")
parser.add_argument("--switch_threshold",    type=float, default=0.80,
                    help="Switch pi1→mirror-law when estimated apex >= "
                         "switch_threshold * apex_height  (default 0.80)")
parser.add_argument("--fallback_threshold",  type=float, default=0.50,
                    help="Revert mirror-law→pi1 when estimated apex drops below "
                         "fallback_threshold * apex_height  (default 0.50)")
parser.add_argument("--centering_gain",      type=float, default=2.0)
parser.add_argument("--impact_tilt_gain",    type=float, default=1.0)
parser.add_argument("--h_nominal",           type=float, default=0.38)
parser.add_argument("--max_steps",           type=int,   default=0,
                    help="Stop after this many steps and print SWEEP_RESULT. 0=run forever.")
parser.add_argument("--apex_schedule",       type=str,   default=None,
                    help="Comma-separated apex heights to cycle through, e.g. '0.15,0.20,0.25,0.30'. "
                         "Each height runs for --schedule_steps steps before advancing.")
parser.add_argument("--schedule_steps",      type=int,   default=1500,
                    help="Steps to spend at each apex height in the schedule (default 1500).")
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

from go1_ball_balance.tasks.ball_juggle_hier.ball_juggle_pi1_env_cfg import BallJugglePi1EnvCfg_PLAY
from go1_ball_balance.tasks.ball_juggle_hier.agents.rsl_rl_ppo_cfg import BallJuggleHierPPORunnerCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# Mirror-law command scaling constants (must match action_term.py)
_CMD_SCALES  = torch.tensor([0.125, 1.0, 0.4, 0.4, 3.0, 3.0])
_CMD_OFFSETS = torch.tensor([0.375, 0.0, 0.0, 0.0, 0.0, 0.0])
_GRAVITY = 9.81
_RESTITUTION = 0.85


# ---------------------------------------------------------------------------
# Mirror law: compute 6D normalised torso command from raw ball/robot state
# ---------------------------------------------------------------------------
def mirror_law_cmd(
    ball_pos_w:  torch.Tensor,   # (N, 3)
    ball_vel_w:  torch.Tensor,   # (N, 3)
    robot_pos_w: torch.Tensor,   # (N, 3)
    robot_quat_w: torch.Tensor,  # (N, 4) wxyz
    apex:        torch.Tensor,   # (N,)   target apex per env
    h_nominal:   float,
    centering_gain: float,
    impact_tilt_gain: float,
    device: str,
) -> torch.Tensor:
    """Return (N, 6) normalised torso command using mirror law geometry."""
    N = ball_pos_w.shape[0]
    paddle_offset_b = torch.tensor([0.0, 0.0, 0.070], device=device)
    offset_b = paddle_offset_b.unsqueeze(0).expand(N, -1)
    paddle_pos_w = robot_pos_w + math_utils.quat_apply(robot_quat_w, offset_b)
    p_rel_w = ball_pos_w - paddle_pos_w  # (N, 3)

    # Desired outgoing velocity
    v_out_z = (2.0 * _GRAVITY * apex).sqrt().clamp(min=0.5)
    v_out_x = -centering_gain * p_rel_w[:, 0]
    v_out_y = -centering_gain * p_rel_w[:, 1]
    v_out_w = torch.stack([v_out_x, v_out_y, v_out_z], dim=-1)

    # Mirror law normal
    v_out_eff = v_out_w / max(_RESTITUTION, 0.1)
    n_raw = v_out_eff - ball_vel_w
    n_w = F.normalize(n_raw, dim=-1)
    flip = (n_w[:, 2] < 0).unsqueeze(-1).float()
    n_w = n_w * (1.0 - 2.0 * flip)

    # Rotate into robot body frame
    quat_w2b = math_utils.quat_conjugate(robot_quat_w)
    n_b = math_utils.quat_apply(quat_w2b, n_w)
    nz_safe = n_b[:, 2].clamp(min=0.15)

    pitch_tgt = torch.atan2( n_b[:, 0], nz_safe).clamp(-0.4, 0.4)
    roll_tgt  = torch.atan2(-n_b[:, 1], nz_safe).clamp(-0.4, 0.4)

    # Impact tilt boost
    if impact_tilt_gain > 1.0:
        impact = ((ball_vel_w[:, 2] < 0).float() * (p_rel_w[:, 2] < 0.50).float())
        gain = 1.0 + (impact_tilt_gain - 1.0) * impact
        pitch_tgt = (pitch_tgt * gain).clamp(-0.4, 0.4)
        roll_tgt  = (roll_tgt  * gain).clamp(-0.4, 0.4)

    # h_dot impulse
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
    return (cmd_phys - offsets) / scales   # (N, 6) normalised


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
env_cfg = BallJugglePi1EnvCfg_PLAY()
env_cfg.scene.num_envs = args.num_envs
env_cfg.actions.torso_cmd.pi2_checkpoint = os.path.abspath(args.pi2_checkpoint)
env_cfg.observations.policy.enable_corruption = False

env = gym.make("Isaac-BallJugglePi1-Go1-v0", cfg=env_cfg)
device = env.unwrapped.device

agent_cfg = BallJuggleHierPPORunnerCfg()
env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=str(device))
runner.load(os.path.abspath(args.pi1_checkpoint))
policy = runner.get_inference_policy(device=str(device))

# Apex schedule
if args.apex_schedule is not None:
    _schedule = [float(x) for x in args.apex_schedule.split(",")]
else:
    _schedule = [args.apex_height]
_sched_idx       = 0
_sched_step_start = 0
current_apex = _schedule[0]

def _set_apex(h: float) -> tuple:
    env.unwrapped._apex_target_h.fill_(h)
    env.unwrapped._apex_target_std.fill_(0.10)
    return (
        torch.full((args.num_envs,), h, device=device),
        args.switch_threshold  * h,
        args.fallback_threshold * h,
    )

env.unwrapped._apex_target_h   = torch.full((args.num_envs,), current_apex, device=device)
env.unwrapped._apex_target_std = torch.full((args.num_envs,), 0.10, device=device)

apex_tensor, switch_apex, fallback_apex = _set_apex(current_apex)

# Per-env controller mode: False=pi1, True=mirror law
using_mirror = torch.zeros(args.num_envs, dtype=torch.bool, device=device)

# Apex tracking: detect bounce apex when ball transitions from up→down
prev_ball_vz    = torch.zeros(args.num_envs, device=device)
last_bounce_apex = torch.zeros(args.num_envs, device=device)  # peak height above paddle per env
# Minimum steps to stay in each mode before switching (hysteresis)
mode_steps      = torch.zeros(args.num_envs, device=device)
MIN_MODE_STEPS  = 100  # stay in current mode for at least this many steps

print(f"\n[play_hybrid] pi1     : {args.pi1_checkpoint}")
print(f"[play_hybrid] pi2     : {args.pi2_checkpoint}")
if len(_schedule) > 1:
    print(f"[play_hybrid] apex schedule : {_schedule} (each {args.schedule_steps} steps)")
else:
    print(f"[play_hybrid] target  : {current_apex:.2f} m")
print(f"[play_hybrid] switch  : {args.switch_threshold:.2f}  fallback: {args.fallback_threshold:.2f}")
print("[play_hybrid] Running — Ctrl+C to stop.\n")

obs_raw, _ = env.reset()
step = 0
episode_rewards = torch.zeros(args.num_envs, device=device)
episode_lengths = torch.zeros(args.num_envs, device=device)

# Sweep metrics
_apex_samples   = []   # last_bounce_apex samples (env 0) while in mirror mode
_mirror_steps   = 0    # total steps any env spent in mirror-law mode
_total_steps    = 0

try:
    while simulation_app.is_running():
        ball  = env.unwrapped.scene["ball"]
        robot = env.unwrapped.scene["robot"]

        ball_pos_w  = ball.data.root_pos_w.clone()
        ball_vel_w  = ball.data.root_lin_vel_w.clone()
        robot_pos_w = robot.data.root_pos_w.clone()
        robot_quat_w = robot.data.root_quat_w.clone()

        ball_vz = ball_vel_w[:, 2]
        paddle_z_vec = robot_pos_w[:, 2] + 0.07
        ball_above_paddle = (ball_pos_w[:, 2] - paddle_z_vec).clamp(min=0.0)

        # Detect bounce apex: ball transitions from ascending (vz>0) to descending (vz<0)
        just_peaked = (prev_ball_vz > 0) & (ball_vz <= 0)
        last_bounce_apex = torch.where(just_peaked, ball_above_paddle, last_bounce_apex)
        prev_ball_vz = ball_vz.clone()

        mode_steps += 1

        # Switch pi1 → mirror law: apex within ±switch_band of target AND stable
        # Use absolute window rather than relative threshold so low targets aren't
        # triggered by overshoot (e.g. switch_band=0.05 means target±0.05 m).
        _switch_band = 0.08   # m — how close to target before handing off
        switch_to_mirror = (
            (~using_mirror)
            & (last_bounce_apex >= apex_tensor - _switch_band)
            & (last_bounce_apex <= apex_tensor + _switch_band)
            & (mode_steps >= MIN_MODE_STEPS)
        )
        # Fall back mirror law → pi1: bounce apex dropped too low AND mode has been stable
        fallback_to_pi1 = (
            using_mirror
            & (last_bounce_apex < fallback_apex)
            & (last_bounce_apex > 0)        # only after a real bounce is recorded
            & (mode_steps >= MIN_MODE_STEPS)
        )

        mode_steps = torch.where(switch_to_mirror | fallback_to_pi1,
                                 torch.zeros_like(mode_steps), mode_steps)

        using_mirror = (using_mirror | switch_to_mirror) & ~fallback_to_pi1

        with torch.no_grad():
            # Pi1 actions for all envs
            obs_tensor = obs_raw["policy"].to(device)
            pi1_actions = policy({"policy": obs_tensor})   # (N, 6)

            # Mirror law actions for all envs
            ml_actions = mirror_law_cmd(
                ball_pos_w, ball_vel_w, robot_pos_w, robot_quat_w,
                apex_tensor, args.h_nominal, args.centering_gain,
                args.impact_tilt_gain, device,
            )                                              # (N, 6)

            # Select per env
            mask = using_mirror.unsqueeze(-1).float()      # (N, 1)
            actions = ml_actions * mask + pi1_actions * (1.0 - mask)

            obs_raw, rew, terminated, truncated, _ = env.step(actions)

        dones = terminated | truncated
        episode_rewards += rew
        episode_lengths += 1

        # Reset mode on episode end
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
        _total_steps += 1
        _mirror_steps += using_mirror.sum().item()
        if using_mirror[0] and last_bounce_apex[0].item() > 0:
            _apex_samples.append(last_bounce_apex[0].item())

        # Advance apex schedule
        if len(_schedule) > 1 and (step - _sched_step_start) >= args.schedule_steps:
            _sched_idx = (_sched_idx + 1) % len(_schedule)
            current_apex = _schedule[_sched_idx]
            apex_tensor, switch_apex, fallback_apex = _set_apex(current_apex)
            _sched_step_start = step
            # Reset mode so pi1 rebuilds to new target
            using_mirror[:] = False
            last_bounce_apex[:] = 0.0
            mode_steps[:] = 0.0
            print(f"  [schedule] step {step} → apex={current_apex:.2f} m")

        if args.max_steps > 0 and step >= args.max_steps:
            break

        if step % 200 == 0:
            paddle_z = robot_pos_w[0, 2].item() + 0.07
            bz = ball_pos_w[0, 2].item()
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
    print("\n[play_hybrid] Stopped by user.")

import statistics
mirror_frac   = _mirror_steps / max(_total_steps * args.num_envs, 1)
mean_apex     = statistics.mean(_apex_samples)   if _apex_samples else 0.0
median_apex   = statistics.median(_apex_samples) if _apex_samples else 0.0
print(
    f"SWEEP_RESULT apex={args.apex_height:.2f} "
    f"sw={args.switch_threshold:.2f} fb={args.fallback_threshold:.2f} | "
    f"mirror_frac={mirror_frac:.3f} "
    f"mean_apex={mean_apex:.3f} median_apex={median_apex:.3f}"
)

env.close()
simulation_app.close()
