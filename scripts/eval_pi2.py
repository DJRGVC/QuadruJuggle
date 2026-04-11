"""1D sweep evaluation of pi2 torso-tracking accuracy.

For each of the four dimensions (h, h_dot, roll, pitch), sweeps a range of
fixed command values while holding the other three at nominal.  Runs N envs
in parallel — one per sweep point — measures mean absolute error after a
warmup period, and saves one error-vs-command plot per dimension.

Output: videos/pi2/eval_pi2_sweep.png  (always overwritten)

Usage:
    cd /home/frank/berkeley_mde/QuadruJuggle
    PYTHONPATH=/home/frank/IsaacLab/scripts/reinforcement_learning/rsl_rl:$PYTHONPATH \\
    python scripts/eval_pi2.py \\
        --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \\
        --n_points 16 \\
        --warmup_steps 150 \\
        --measure_steps 200
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pi2 1D sweep accuracy evaluation.")
parser.add_argument("--pi2_checkpoint", type=str, required=True)
parser.add_argument("--n_points",      type=int, default=16,
                    help="Number of sweep points per dimension (= num_envs needed)")
parser.add_argument("--warmup_steps",  type=int, default=150,
                    help="Steps to discard before measuring (let robot settle)")
parser.add_argument("--measure_steps", type=int, default=200,
                    help="Steps to average error over")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
import gymnasium as gym
from rsl_rl.runners import OnPolicyRunner
import isaaclab.utils.math as math_utils

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import isaaclab_tasks   # noqa: F401
import go1_ball_balance  # noqa: F401

from go1_ball_balance.tasks.torso_tracking.torso_tracking_env_cfg import TorsoTrackingEnvCfg_PLAY
from go1_ball_balance.tasks.torso_tracking.agents.rsl_rl_ppo_cfg import TorsoTrackingPPORunnerCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

N = args.n_points

# ── Sweep definitions ────────────────────────────────────────────────────────
# Each entry: (dim_name, dim_index, sweep_values, nominal_6d_cmd, ylabel, unit)
# Nominal: [h=0.38, h_dot=0.0, roll=0.0, pitch=0.0, ω_r=0.0, ω_p=0.0]
_NOMINAL = [0.38, 0.0, 0.0, 0.0, 0.0, 0.0]

SWEEPS = [
    ("h",     0, np.linspace(0.25, 0.50, N), "Height [m]",   "m"),
    ("h_dot", 1, np.linspace(-1.0,  1.0, N), "h_dot [m/s]",  "m/s"),
    ("roll",  2, np.linspace(-0.4,  0.4, N), "Roll [rad]",   "rad"),
    ("pitch", 3, np.linspace(-0.4,  0.4, N), "Pitch [rad]",  "rad"),
]

# ── Output folder ────────────────────────────────────────────────────────────
_out = os.path.join(os.path.dirname(__file__), "..", "videos", "pi2")
os.makedirs(_out, exist_ok=True)

# ── Build env (N envs — one per sweep point) ─────────────────────────────────
env_cfg = TorsoTrackingEnvCfg_PLAY()
env_cfg.scene.num_envs   = N
env_cfg.scene.env_spacing = 1.5

env = gym.make("Isaac-TorsoTracking-Go1-Play-v0", cfg=env_cfg)
device = env.unwrapped.device

# Disable automatic command resampling — we drive _torso_cmd manually
env.unwrapped._torso_smooth_enabled = False

# ── Load pi2 policy ──────────────────────────────────────────────────────────
agent_cfg  = TorsoTrackingPPORunnerCfg()
env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=str(device))
runner.load(os.path.abspath(args.pi2_checkpoint))
policy = runner.get_inference_policy(device=str(device))

print(f"\n[eval_pi2] checkpoint    : {args.pi2_checkpoint}")
print(f"[eval_pi2] n_points      : {N}")
print(f"[eval_pi2] warmup_steps  : {args.warmup_steps}")
print(f"[eval_pi2] measure_steps : {args.measure_steps}")
print(f"[eval_pi2] total_steps   : {args.warmup_steps + args.measure_steps}\n")

# ── Helper: read actual 6D robot state ───────────────────────────────────────
def read_actual(robot) -> torch.Tensor:
    """Return (N, 6) actual state: [h, h_dot, roll, pitch, ω_roll, ω_pitch]."""
    h     = robot.data.root_pos_w[:, 2]
    h_dot = robot.data.root_lin_vel_w[:, 2]
    quat  = robot.data.root_quat_w          # (N, 4)
    roll, pitch, _ = math_utils.euler_xyz_from_quat(quat)
    ang_b = robot.data.root_ang_vel_b       # (N, 3) body frame
    return torch.stack([h, h_dot, roll, pitch, ang_b[:, 0], ang_b[:, 1]], dim=-1)


# ── Run all sweeps ────────────────────────────────────────────────────────────
results = {}   # dim_name → (sweep_values, mean_abs_err, std_abs_err)

for dim_name, dim_idx, sweep_vals, ylabel, unit in SWEEPS:
    print(f"  Sweeping {dim_name} ({N} points) ...")

    # Build fixed command matrix: each env gets one sweep value, rest nominal
    cmd_fixed = torch.tensor(_NOMINAL, dtype=torch.float32, device=device)
    cmd_fixed = cmd_fixed.unsqueeze(0).expand(N, -1).clone()  # (N, 6)
    cmd_fixed[:, dim_idx] = torch.tensor(sweep_vals, dtype=torch.float32, device=device)

    # Reset and pin commands
    obs_dict, _ = env.reset()
    env.unwrapped._torso_cmd[:]      = cmd_fixed
    env.unwrapped._torso_cmd_goal[:] = cmd_fixed

    errors = torch.zeros(N, device=device)   # accumulated |actual - desired|
    total  = args.warmup_steps + args.measure_steps

    for step in range(total):
        # Pin command every step (suppress any internal resampling)
        env.unwrapped._torso_cmd[:]      = cmd_fixed
        env.unwrapped._torso_cmd_goal[:] = cmd_fixed

        robot = env.unwrapped.scene["robot"]
        actual = read_actual(robot)          # (N, 6)

        if step >= args.warmup_steps:
            errors += (actual[:, dim_idx] - cmd_fixed[:, dim_idx]).abs()

        with torch.no_grad():
            obs_tensor = obs_dict["policy"].to(device)
            actions    = policy({"policy": obs_tensor})
            obs_dict, _, _, _, _ = env.step(actions)

    mean_err = (errors / args.measure_steps).cpu().numpy()
    results[dim_name] = (sweep_vals, mean_err)
    print(f"    {dim_name}: mean err = {mean_err.mean():.4f} {unit}  "
          f"(min={mean_err.min():.4f}, max={mean_err.max():.4f})")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

sweep_meta = {s[0]: s for s in SWEEPS}
for ax, (dim_name, (sweep_vals, mean_err)) in zip(axes, results.items()):
    _, _, _, ylabel, unit = sweep_meta[dim_name]

    ax.plot(sweep_vals, mean_err, color="steelblue", linewidth=1.5, marker="o", markersize=3)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.fill_between(sweep_vals, mean_err, alpha=0.15, color="steelblue")

    # Mark ±5% of range as "good" threshold
    cmd_range = sweep_vals[-1] - sweep_vals[0]
    good_thresh = 0.05 * cmd_range
    ax.axhline(good_thresh, color="red", linewidth=0.8, linestyle="--",
               label=f"5% range = {good_thresh:.3f} {unit}")

    ax.set_xlabel(f"Commanded {dim_name} [{unit}]")
    ax.set_ylabel(f"Mean |error| [{unit}]")
    ax.set_title(f"{dim_name} tracking accuracy")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("Pi2 1D Sweep — Tracking Accuracy per Dimension", fontsize=12)
fig.tight_layout()

plot_path = os.path.join(_out, "eval_pi2_sweep.png")
fig.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"\n[eval_pi2] Sweep plot saved → {plot_path}")

env.close()
simulation_app.close()
