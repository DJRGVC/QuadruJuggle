"""Test pi2 torso-tracking in Isaac Lab — pitch sweep to find tracking limit.

Runs the same SWEEP_TARGETS as test_pi2_mujoco.py so results are directly comparable.
Saves a 4-panel tracking plot (h, h_dot, roll, pitch) to videos/pi2_tracking_isaaclab.png.

Usage:
    python scripts/tests/test_pi2_isaaclab.py --headless
    python scripts/tests/test_pi2_isaaclab.py --headless --sweep --dwell 150
"""
import argparse, os, sys, math

from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser()
ap.add_argument("--pi2_checkpoint",
                default=os.path.join(os.path.dirname(__file__),
                    "../../logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt"))
ap.add_argument("--sweep",      action="store_true",
                help="Cycle through pitch targets; otherwise hold --pitch fixed.")
ap.add_argument("--pitch",      type=float, default=0.0,
                help="Fixed pitch command (rad) when not using --sweep.")
ap.add_argument("--roll",       type=float, default=0.0)
ap.add_argument("--dwell",      type=int,   default=200,
                help="Policy steps per sweep phase (default 200 = 4 s at 50 Hz).")
ap.add_argument("--warmup",     type=int,   default=200,
                help="Sim steps of warmup before each trial.")
ap.add_argument("--print_every",type=int,   default=50)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG

# ── Constants (must match torso_tracking/mdp/observations.py) ────────────────
_NORM   = torch.tensor([1/0.125, 1/1.0, 1/0.4, 1/0.4, 1/3.0, 1/3.0])
_OFFSET = torch.tensor([-0.375,  0.0,   0.0,   0.0,   0.0,   0.0  ])
NEUTRAL = torch.tensor([0.375, 0.0, 0.0, 0.0, 0.0, 0.0])

ACTION_SCALE = 0.25
SIM_DT       = 1.0 / 200.0
DECIMATION   = 4
POLICY_DT    = DECIMATION * SIM_DT   # 0.02 s

# EMA taus — must match commands.py _SMOOTH_TAUS
_SMOOTH_TAUS       = torch.tensor([0.40, 0.30, 0.02, 0.02, 0.01, 0.01])
_EMA_ALPHAS_POLICY = 1.0 - torch.exp(-torch.tensor(POLICY_DT) / _SMOOTH_TAUS)

# ── SWEEP TARGETS (pitch focus — same as test_pi2_mujoco.py) ─────────────────
SWEEP_TARGETS = [
    torch.tensor([0.375, 0.0,  0.0,  0.10, 0.0, 0.0]),   # pitch+ 0.10 rad
    torch.tensor([0.375, 0.0,  0.0,  0.15, 0.0, 0.0]),   # pitch+ 0.15 rad
    torch.tensor([0.375, 0.0,  0.0,  0.20, 0.0, 0.0]),   # pitch+ 0.20 rad
    torch.tensor([0.375, 0.0,  0.0,  0.25, 0.0, 0.0]),   # pitch+ 0.25 rad
    torch.tensor([0.375, 0.0,  0.0,  0.28, 0.0, 0.0]),   # pitch+ 0.28 rad
    torch.tensor([0.375, 0.0,  0.0,  0.30, 0.0, 0.0]),   # pitch+ 0.30 rad
    torch.tensor([0.375, 0.0,  0.0,  0.32, 0.0, 0.0]),   # pitch+ 0.32 rad
    torch.tensor([0.375, 0.0,  0.0,  0.35, 0.0, 0.0]),   # pitch+ 0.35 rad
    torch.tensor([0.375, 0.0,  0.0,  0.40, 0.0, 0.0]),   # pitch+ 0.40 rad
    torch.tensor([0.375, 0.0,  0.0, -0.10, 0.0, 0.0]),   # pitch- 0.10 rad
    torch.tensor([0.375, 0.0,  0.0, -0.15, 0.0, 0.0]),   # pitch- 0.15 rad
    torch.tensor([0.375, 0.0,  0.0, -0.20, 0.0, 0.0]),   # pitch- 0.20 rad
    torch.tensor([0.375, 0.0,  0.0, -0.25, 0.0, 0.0]),   # pitch- 0.25 rad
    torch.tensor([0.375, 0.0,  0.0, -0.28, 0.0, 0.0]),   # pitch- 0.28 rad
    torch.tensor([0.375, 0.0,  0.0, -0.30, 0.0, 0.0]),   # pitch- 0.30 rad
    torch.tensor([0.375, 0.0,  0.0, -0.32, 0.0, 0.0]),   # pitch- 0.32 rad
    torch.tensor([0.375, 0.0,  0.0, -0.35, 0.0, 0.0]),   # pitch- 0.35 rad
    torch.tensor([0.375, 0.0,  0.0, -0.40, 0.0, 0.0]),   # pitch- 0.40 rad
]


# ── Scene ─────────────────────────────────────────────────────────────────────
@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground",    spawn=sim_utils.GroundPlaneCfg())
    dome   = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=500.0))
    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ── Actor ─────────────────────────────────────────────────────────────────────
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
        if bkey in sd: actor_sd[f"{idx}.bias"] = sd[bkey]
    actor.load_state_dict(actor_sd)
    actor.eval()
    for p in actor.parameters(): p.requires_grad = False
    print(f"[pi2_il] actor: {' → '.join(f'{i}→{o}' for i,o in layers)}", flush=True)
    return actor


# ── Sim setup ─────────────────────────────────────────────────────────────────
sim   = SimulationContext(SimulationCfg(dt=SIM_DT, render_interval=DECIMATION))
scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
sim.reset()
scene.update(SIM_DT)

device      = sim.device
robot       = scene["robot"]
actor       = _build_actor(os.path.abspath(args.pi2_checkpoint), str(device))
default_pos = robot.data.default_joint_pos.clone()
default_vel = robot.data.default_joint_vel.clone()
env_ids     = torch.tensor([0], device=device)

_spawn_pose = torch.zeros(1, 7, device=device)
_spawn_pose[0, 2] = 0.42
_spawn_pose[0, 3] = 1.0


def do_warmup():
    robot.write_joint_state_to_sim(default_pos, default_vel, env_ids=env_ids)
    robot.write_root_pose_to_sim(_spawn_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=env_ids)
    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(SIM_DT)
    for _ in range(args.warmup - 1):
        robot.set_joint_position_target(default_pos)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(SIM_DT)
    trunk_z = robot.data.root_pos_w[0, 2].item()
    jp_err  = (robot.data.joint_pos[0] - default_pos[0]).abs().max().item()
    print(f"  [warmup done] trunk_z={trunk_z:.3f}m  jp_err_max={jp_err:.4f}", flush=True)


# ── Recording ─────────────────────────────────────────────────────────────────
_rec = {"cmd_h": [], "act_h": [],
        "cmd_h_dot": [], "act_h_dot": [],
        "cmd_roll": [], "act_roll": [],
        "cmd_pitch": [], "act_pitch": []}

_neutral_norm = ((NEUTRAL + _OFFSET) * _NORM).to(device)

# EMA state
cmd_ema  = NEUTRAL.clone()
cmd_goal = NEUTRAL.clone()

sweep_idx   = 0
sweep_phase = "to_target"
sweep_count = 0

if not args.sweep:
    fixed_cmd      = torch.tensor([0.375, 0.0, args.roll, args.pitch, 0.0, 0.0])
    torso_cmd_norm = ((fixed_cmd + _OFFSET) * _NORM).to(device).unsqueeze(0)
    print(f"[pi2_il] fixed cmd: {fixed_cmd.numpy()}", flush=True)

print(f"[pi2_il] sweep={args.sweep}  dwell={args.dwell}", flush=True)
print("[pi2_il] starting warmup...", flush=True)
do_warmup()

step = 0
while simulation_app.is_running():
    # ── Sweep EMA ────────────────────────────────────────────────────────────
    if args.sweep:
        if sweep_phase == "to_target":
            if sweep_count == 0:
                cmd_goal = SWEEP_TARGETS[sweep_idx].clone()
                t = SWEEP_TARGETS[sweep_idx]
                label = f"pitch={t[3]:+.2f}" if t[3] != 0 else f"roll={t[2]:+.2f}"
                print(f"\n  [sweep {sweep_idx+1}/{len(SWEEP_TARGETS)}] → {label}", flush=True)
            sweep_count += 1
            if sweep_count >= args.dwell:
                sweep_phase = "to_neutral"
                sweep_count = 0
                print("  [sweep] → neutral", flush=True)
        else:  # to_neutral
            if sweep_count == 0:
                cmd_goal = NEUTRAL.clone()
            sweep_count += 1
            if sweep_count >= args.dwell:
                sweep_idx += 1
                sweep_count = 0
                if sweep_idx >= len(SWEEP_TARGETS):
                    print("[pi2_il] all targets done — stopping.", flush=True)
                    break
                else:
                    sweep_phase = "to_target"
                    # reset between targets
                    print("  [sweep] resetting for next target...", flush=True)
                    cmd_ema  = NEUTRAL.clone()
                    cmd_goal = NEUTRAL.clone()
                    do_warmup()
                    continue

        cmd_ema = cmd_ema + _EMA_ALPHAS_POLICY * (cmd_goal - cmd_ema)
        torso_cmd_norm = ((cmd_ema + _OFFSET) * _NORM).to(device).unsqueeze(0)

    # ── Build 39D obs ─────────────────────────────────────────────────────────
    lin_vel_b = robot.data.root_lin_vel_b[0:1]
    ang_vel_b = robot.data.root_ang_vel_b[0:1]
    grav_b    = robot.data.projected_gravity_b[0:1]
    jp_rel    = robot.data.joint_pos[0:1] - default_pos
    jv_rel    = robot.data.joint_vel[0:1] - default_vel

    obs = torch.cat([torso_cmd_norm, lin_vel_b, ang_vel_b, grav_b, jp_rel, jv_rel], dim=1)

    with torch.no_grad():
        actions = actor(obs)

    targets = default_pos + ACTION_SCALE * actions
    for i in range(DECIMATION):
        robot.set_joint_position_target(targets)
        scene.write_data_to_sim()
        sim.step(render=False)
        if i == DECIMATION - 1:
            sim.render()
        scene.update(SIM_DT)

    # ── Measure actual state ──────────────────────────────────────────────────
    trunk_z   = robot.data.root_pos_w[0, 2].item()
    trunk_zdot = robot.data.root_lin_vel_w[0, 2].item()
    quat_w    = robot.data.root_quat_w[0]
    roll_act, pitch_act, _ = math_utils.euler_xyz_from_quat(quat_w.unsqueeze(0))
    roll_act  = roll_act[0].item()
    pitch_act = pitch_act[0].item()
    grav_np   = robot.data.projected_gravity_b[0].cpu().numpy()
    tilt_deg  = float(np.degrees(np.arccos(np.clip(-grav_np[2], -1, 1))))

    # decode cmd
    cn = torso_cmd_norm[0].cpu()
    cmd_h     = float(cn[0] / _NORM[0] - _OFFSET[0])
    cmd_h_dot = float(cn[1] / _NORM[1])
    cmd_roll  = float(cn[2] / _NORM[2])
    cmd_pitch = float(cn[3] / _NORM[3])

    _rec["cmd_h"].append(cmd_h)
    _rec["act_h"].append(trunk_z)
    _rec["cmd_h_dot"].append(cmd_h_dot)
    _rec["act_h_dot"].append(trunk_zdot)
    _rec["cmd_roll"].append(cmd_roll)
    _rec["act_roll"].append(roll_act)
    _rec["cmd_pitch"].append(cmd_pitch)
    _rec["act_pitch"].append(pitch_act)

    if step % args.print_every == 0:
        phase = f"  [{sweep_phase} {sweep_count}/{args.dwell}]" if args.sweep else ""
        print(f"  step {step:5d} | "
              f"h: cmd={cmd_h:.3f} act={trunk_z:.3f} | "
              f"pitch: cmd={np.degrees(cmd_pitch):+.1f}° act={np.degrees(pitch_act):+.1f}° "
              f"err={np.degrees(pitch_act-cmd_pitch):+.1f}°{phase}", flush=True)

    # Fall detection — reset current target trial
    if trunk_z < 0.20 or tilt_deg > 55:
        print(f"  [fall] trunk_z={trunk_z:.3f}  tilt={tilt_deg:.1f}° — skipping to next target",
              flush=True)
        # advance to next target immediately
        sweep_idx += 1
        sweep_count = 0
        sweep_phase = "to_target"
        cmd_ema  = NEUTRAL.clone()
        cmd_goal = NEUTRAL.clone()
        if sweep_idx >= len(SWEEP_TARGETS):
            break
        do_warmup()

    step += 1

# ── Plot ──────────────────────────────────────────────────────────────────────
if _rec["cmd_h"]:
    n = len(_rec["cmd_h"])
    t = np.arange(n) * POLICY_DT

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Pi2 Torso Tracking — Isaac Lab", fontsize=13)
    panels = [
        ("cmd_h",     "act_h",     "Height [m]",    (0.20, 0.55), "tab:orange"),
        ("cmd_h_dot", "act_h_dot", "h_dot [m/s]",   (-1.2,  1.2), "tab:purple"),
        ("cmd_roll",  "act_roll",  "Roll [rad]",    (-0.5,  0.5), "tab:green"),
        ("cmd_pitch", "act_pitch", "Pitch [rad]",   (-0.5,  0.5), "tab:red"),
    ]
    for ax, (ck, ak, ylabel, ylim, acolor) in zip(axes, panels):
        ax.plot(t, _rec[ck], label="cmd",    color="tab:blue", linestyle="--", linewidth=0.9)
        ax.plot(t, _rec[ak], label="actual", color=acolor,     linewidth=0.9, alpha=0.85)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(ylim)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()

    out = os.path.join(os.path.dirname(__file__), "../../videos/pi2_tracking_isaaclab.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n[pi2_il] plot saved → {os.path.normpath(out)}")

simulation_app.close()
