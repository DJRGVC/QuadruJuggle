"""Sim-to-Sim validation: run pi1+pi2 juggling in Isaac Lab without the gym env.

Uses SimulationContext + InteractiveScene directly (same as test_pi2_isaaclab.py).
Pipeline mirrors play_mujoco.py and play_launcher_hybrid.py exactly:
    Isaac Lab state → 46D pi1 obs → pi1 → 6D torso cmd
                    → 39D pi2 obs → pi2 → 12D joint targets
                    → GO1 actuator net → joint torques → physics step
                    → paddle tracked kinematically at 200 Hz

Usage:
    python scripts/play_isaaclab.py --headless
    python scripts/play_isaaclab.py  (with viewer)
"""
import argparse, os, sys, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser()
ap.add_argument("--launcher_checkpoint",
                default=os.path.join(os.path.dirname(__file__),
                    "../logs/rsl_rl/go1_ball_launcher/2026-04-05_21-42-11/model_best.pt"))
ap.add_argument("--pi2_checkpoint",
                default=os.path.join(os.path.dirname(__file__),
                    "../logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt"))
ap.add_argument("--apex_height", type=float, default=0.50)
ap.add_argument("--max_steps",   type=int,   default=0, help="0 = run forever")
ap.add_argument("--print_every", type=int,   default=100)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
import torch
import torch.nn as nn

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG

# ── Constants (must match Isaac Lab env cfgs) ─────────────────────────────────
PADDLE_OFFSET_B  = torch.tensor([0.0, 0.0, 0.070])   # trunk body frame
BALL_RADIUS      = 0.020
MAX_APEX_HEIGHT  = 1.00

# Pi1 action → physical torso command
CMD_SCALES  = torch.tensor([0.125, 1.0, 0.4, 0.4, 3.0, 3.0])
CMD_OFFSETS = torch.tensor([0.375, 0.0, 0.0, 0.0, 0.0, 0.0])

# Pi2 obs normalisation (torso_tracking/mdp/observations.py)
OBS_NORM   = torch.tensor([8.0, 1.0, 2.5, 2.5, 1/3, 1/3])
OBS_OFFSET = torch.tensor([-0.375, 0.0, 0.0, 0.0, 0.0, 0.0])

SIM_DT     = 1.0 / 200.0
DECIMATION = 4
POLICY_DT  = DECIMATION * SIM_DT


_ASSETS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__),
    "../assets/paddle"))


# ── Scene ─────────────────────────────────────────────────────────────────────
@configclass
class SceneCfg(InteractiveSceneCfg):
    ground     = AssetBaseCfg(prim_path="/World/ground",
                              spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(prim_path="/World/DomeLight",
                              spawn=sim_utils.DomeLightCfg(intensity=500.0))

    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.01, angular_damping=0.01,
                max_linear_velocity=10.0, max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0027),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=0.99, restitution_combine_mode="max",
                static_friction=0.3, dynamic_friction=0.3,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.4, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.54)),
    )

    paddle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Paddle",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{_ASSETS_DIR}/dumbbell.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.53)),
    )


# ── Actor loader (shared by pi1 and pi2) ──────────────────────────────────────
def _build_actor(path: str, device: str) -> nn.Module:
    ck = torch.load(path, map_location=device, weights_only=True)
    sd = ck.get("model_state_dict", ck)
    keys = sorted(k for k in sd if k.startswith("actor.") and "weight" in k)
    layers = [(sd[k].shape[1], sd[k].shape[0]) for k in keys]
    modules = []
    for i, (in_d, out_d) in enumerate(layers):
        modules.append(nn.Linear(in_d, out_d))
        if i < len(layers) - 1:
            modules.append(nn.ELU())
    actor = nn.Sequential(*modules).to(device)
    sd2 = {}
    for k in keys:
        idx = int(k.split(".")[1])
        sd2[f"{idx}.weight"] = sd[k]
        bk = k.replace("weight", "bias")
        if bk in sd: sd2[f"{idx}.bias"] = sd[bk]
    actor.load_state_dict(sd2)
    actor.eval()
    for p in actor.parameters(): p.requires_grad = False
    print(f"  loaded: {' → '.join(f'{i}→{o}' for i,o in layers)}")
    return actor


# ── Sim setup ─────────────────────────────────────────────────────────────────
sim   = SimulationContext(SimulationCfg(dt=SIM_DT, render_interval=DECIMATION))
scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=4.0))
sim.reset()
scene.update(SIM_DT)

device = sim.device
robot  = scene["robot"]
ball   = scene["ball"]
paddle = scene["paddle"]

print("[play_il] loading pi1...")
pi1 = _build_actor(os.path.abspath(args.launcher_checkpoint), str(device))
print("[play_il] loading pi2...")
pi2 = _build_actor(os.path.abspath(args.pi2_checkpoint),      str(device))

default_pos = robot.data.default_joint_pos.clone()   # (1, 12)
default_vel = robot.data.default_joint_vel.clone()
env_ids     = torch.tensor([0], device=device)

_spawn_pose = torch.zeros(1, 7, device=device)
_spawn_pose[0, 2] = 0.42
_spawn_pose[0, 3] = 1.0   # quat w=1

apex_norm   = args.apex_height / MAX_APEX_HEIGHT


# ── Paddle tracking (kinematic, 200 Hz) ───────────────────────────────────────
def update_paddle():
    trunk_pos  = robot.data.root_pos_w            # (1,3)
    trunk_quat = robot.data.root_quat_w           # (1,4) wxyz
    off = PADDLE_OFFSET_B.to(device).unsqueeze(0) # (1,3)
    pad_pos  = trunk_pos + math_utils.quat_apply(trunk_quat, off)
    pad_pose = torch.cat([pad_pos, trunk_quat], dim=-1)   # (1,7)
    paddle.write_root_pose_to_sim(pad_pose)
    # Pass trunk velocity so PhysX sees correct contact velocity (enables h_dot energy transfer)
    pad_vel = torch.cat([robot.data.root_lin_vel_w,
                         robot.data.root_ang_vel_w], dim=-1)   # (1,6)
    paddle.write_root_velocity_to_sim(pad_vel)


# ── Reset ─────────────────────────────────────────────────────────────────────
def do_reset():
    # Hard-write robot to default
    robot.write_joint_state_to_sim(default_pos, default_vel, env_ids=env_ids)
    robot.write_root_pose_to_sim(_spawn_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=env_ids)

    # Place ball just above paddle
    trunk_z = _spawn_pose[0, 2].item()
    ball_init_pos = torch.tensor([[0.0, 0.0, trunk_z + 0.070 + 0.12]], device=device)
    ball_init_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    ball.write_root_pose_to_sim(torch.cat([ball_init_pos, ball_init_quat], dim=-1))
    ball.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))

    scene.write_data_to_sim()
    sim.step(render=False)
    scene.update(SIM_DT)

    ball_pinned_pose = torch.cat([ball_init_pos, ball_init_quat], dim=-1)
    ball_pinned_vel  = torch.zeros(1, 6, device=device)

    # Warmup: let physics settle at default pose; keep ball pinned above paddle
    for _ in range(199):
        robot.set_joint_position_target(default_pos)
        update_paddle()
        ball.write_root_pose_to_sim(ball_pinned_pose)
        ball.write_root_velocity_to_sim(ball_pinned_vel)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(SIM_DT)

    trunk_z = robot.data.root_pos_w[0, 2].item()
    jp_err  = (robot.data.joint_pos[0] - default_pos[0]).abs().max().item()
    print(f"  [reset] trunk_z={trunk_z:.3f}  jp_err={jp_err:.4f}", flush=True)


# ── Main loop ─────────────────────────────────────────────────────────────────
print(f"\n[play_il] apex_height={args.apex_height}m  max_steps={args.max_steps}", flush=True)
print("[play_il] resetting...", flush=True)
do_reset()

last_action  = torch.zeros(1, 6, device=device)
prev_ball_vz = 0.0

_PLOT_STEPS = int(5.0 / POLICY_DT)   # 250 steps = 5s
_snap_ball: list[list[float]] = []
_snap_cmd:  list[list[float]] = []
last_apex    = 0.0
bounce_count = 0
episode      = 0
ep_start     = 0
ep_bounces   = 0
step         = 0

while simulation_app.is_running():
    if args.max_steps > 0 and step >= args.max_steps:
        break

    # ── Pi1 obs (46D) ──────────────────────────────────────────────────────
    trunk_quat = robot.data.root_quat_w        # (1,4)
    trunk_pos  = robot.data.root_pos_w         # (1,3)
    off = PADDLE_OFFSET_B.to(device).unsqueeze(0)
    paddle_pos_w = trunk_pos + math_utils.quat_apply(trunk_quat, off)

    ball_pos_w   = ball.data.root_pos_w        # (1,3)
    ball_vel_w   = ball.data.root_lin_vel_w    # (1,3)

    # ball pos in paddle frame
    diff_w       = ball_pos_w - paddle_pos_w
    ball_pos_pad = math_utils.quat_apply_inverse(trunk_quat, diff_w)   # (1,3)
    # ball vel in trunk frame
    ball_vel_b   = math_utils.quat_apply_inverse(trunk_quat, ball_vel_w)  # (1,3)

    lin_vel_b  = robot.data.root_lin_vel_b[0:1]       # (1,3)
    ang_vel_b  = robot.data.root_ang_vel_b[0:1]       # (1,3)
    grav_b     = robot.data.projected_gravity_b[0:1]  # (1,3)
    jp_rel     = robot.data.joint_pos[0:1] - default_pos
    jv_rel     = robot.data.joint_vel[0:1] - default_vel
    apex_t     = torch.tensor([[apex_norm]], device=device)

    pi1_obs = torch.cat([
        ball_pos_pad, ball_vel_b,
        lin_vel_b, ang_vel_b, grav_b,
        jp_rel, jv_rel,
        apex_t, last_action,
    ], dim=1)   # (1, 46)

    with torch.no_grad():
        pi1_raw = pi1(pi1_obs)                         # (1, 6)
    pi1_raw = pi1_raw.clamp(-2.0, 2.0)

    # Scale to physical torso command
    torso_cmd = pi1_raw * CMD_SCALES.to(device) + CMD_OFFSETS.to(device)  # (1,6)

    # Clamp to ranges pi2 can track in Isaac Lab
    torso_cmd[:, 0] = torso_cmd[:, 0].clamp(0.25, 0.50)   # height
    
    torso_cmd[:, 2] = torso_cmd[:, 2].clamp(-0.4, 0.4)    # roll
    torso_cmd[:, 3] = torso_cmd[:, 3].clamp(-0.4, 0.4)    # pitch

    last_action = pi1_raw.clone()

    # ── Pi2 obs (39D) ──────────────────────────────────────────────────────
    torso_cmd_norm = (torso_cmd + OBS_OFFSET.to(device)) * OBS_NORM.to(device)

    pi2_obs = torch.cat([
        torso_cmd_norm,
        lin_vel_b, ang_vel_b, grav_b,
        jp_rel, jv_rel,
    ], dim=1)   # (1, 39)

    with torch.no_grad():
        pi2_action = pi2(pi2_obs)   # (1, 12)

    targets = default_pos + 0.25 * pi2_action

    # ── Physics decimation (4 × 0.005 s = 0.02 s) ──────────────────────────
    for i in range(DECIMATION):
        robot.set_joint_position_target(targets)
        update_paddle()
        scene.write_data_to_sim()
        sim.step(render=False)
        if i == DECIMATION - 1:
            sim.render()
        scene.update(SIM_DT)

    # ── Logging ──────────────────────────────────────────────────────────────
    trunk_z   = robot.data.root_pos_w[0, 2].item()
    ball_pos  = ball.data.root_pos_w[0].cpu().numpy()
    ball_vz   = ball.data.root_lin_vel_w[0, 2].item()
    pad_z     = trunk_z + 0.070
    grav_np   = robot.data.projected_gravity_b[0].cpu().numpy()
    tilt_deg  = float(np.degrees(np.arccos(np.clip(-grav_np[2], -1, 1))))

    # Apex detection
    if prev_ball_vz > 0.02 and ball_vz <= 0.02:
        apex = max(0.0, ball_pos[2] - pad_z - BALL_RADIUS)
        last_apex = apex
        bounce_count += 1
        ep_bounces  += 1
    prev_ball_vz = ball_vz

    if step < _PLOT_STEPS:
        _snap_ball.append(ball_pos_w[0].cpu().tolist())
        _snap_cmd.append(torso_cmd[0].cpu().tolist())
        if step == _PLOT_STEPS - 1:
            print(f"[play_il] 5s snapshot complete — saving plot.", flush=True)
            break

    if step % args.print_every == 0:
        tc = torso_cmd[0].cpu().numpy()
        jp = jp_rel[0].cpu().numpy()
        print(f"  step {step:5d} | trunk_z={trunk_z:.3f}  tilt={tilt_deg:.1f}°  "
              f"bounces={bounce_count}  last_apex={last_apex:.2f}m", flush=True)
        print(f"    ball_pos : x={ball_pos[0]:+.3f}  y={ball_pos[1]:+.3f}  "
              f"z={ball_pos[2]:+.3f}  vz={ball_vz:+.2f}")
        print(f"    6d_cmd   : h={tc[0]:.3f}  h_dot={tc[1]:+.2f}  "
              f"roll={tc[2]:+.3f}  pitch={tc[3]:+.3f}  wr={tc[4]:+.2f}  wp={tc[5]:+.2f}")
        print(f"    jp_rel   : {np.round(jp, 3)}", flush=True)

    # ── Reset conditions ─────────────────────────────────────────────────────
    ball_off   = (ball_pos[2] < pad_z - 0.15 or
                  abs(ball_pos[0]) > 1.0 or abs(ball_pos[1]) > 1.0)
    robot_fell = trunk_z < 0.15 or tilt_deg > 60

    if ball_off or robot_fell:
        reason = "ball_off" if ball_off else "robot_fell"
        print(f"\n  [episode {episode}] len={step-ep_start}  bounces={ep_bounces}  "
              f"reason={reason}\n", flush=True)
        episode    += 1
        ep_start    = step
        ep_bounces  = 0
        prev_ball_vz = 0.0
        last_action  = torch.zeros(1, 6, device=device)

        do_reset()

    step += 1

print(f"\n[play_il] done. total bounces: {bounce_count}")

if _snap_ball:
    n   = len(_snap_ball)
    t   = np.arange(n) * POLICY_DT
    bp  = np.array(_snap_ball)
    cmd = np.array(_snap_cmd)
    cmd_labels = ["h [m]", "h_dot [m/s]", "roll [rad]", "pitch [rad]",
                  "ω_roll [rad/s]", "ω_pitch [rad/s]"]

    fig, axes = plt.subplots(9, 1, figsize=(12, 18), sharex=True)
    for ai, (label, data, color) in enumerate([
        ("ball x [m]", bp[:, 0], "steelblue"),
        ("ball y [m]", bp[:, 1], "darkorange"),
        ("ball z [m]", bp[:, 2], "green"),
    ]):
        axes[ai].plot(t, data, linewidth=0.9, color=color)
        axes[ai].set_ylabel(label, fontsize=8)
        axes[ai].grid(True, alpha=0.3)

    for ai, label in enumerate(cmd_labels):
        ax = axes[3 + ai]
        ax.plot(t, cmd[:, ai], linewidth=0.9, color="crimson")
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"play_isaaclab — First {n*POLICY_DT:.1f}s: Ball Position & 6D Torso Cmd", fontsize=11)
    fig.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "../videos")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "isaaclab_first5s.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[play_il] plot saved → {path}")

simulation_app.close()
