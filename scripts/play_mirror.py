"""Standalone mirror-law paddle juggling demo.

A floating kinematic paddle juggles a ping-pong ball to a target bounce
height using a trained NN controller (or the analytical mirror law fallback).

Usage:
    uv run --active python scripts/play_mirror.py --target-height 0.30
    uv run --active python scripts/play_mirror.py --controller logs/mirror_law/best_controller.pt
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Mirror-law paddle juggling (standalone)")
parser.add_argument("--target-height", type=float, default=0.30,
                    help="Target bounce height above paddle rest (m)")
parser.add_argument("--controller", type=str, default=None,
                    help="Path to trained NN controller checkpoint")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import random
import torch
import torch.nn as nn
import numpy as np

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DT = 1.0 / 200.0
G = 9.81
BALL_RADIUS = 0.020
RESTITUTION = 0.85
PADDLE_REST_Z = 0.5
PADDLE_RADIUS = 0.085
PADDLE_HALF_THICKNESS = 0.005
SURFACE_Z = PADDLE_REST_Z + PADDLE_HALF_THICKNESS + BALL_RADIUS
MAX_TILT = 0.30
ALPHA_NEUTRAL = (1.0 - RESTITUTION) / (1.0 + RESTITUTION)
K_ALPHA_BLEND = 0.3
KP_LAT = 3.0
KD_LAT = 0.8
DROP_HEIGHT_MEAN = 0.10
DROP_HEIGHT_STD = 0.05
XY_OFFSET_MAX = 0.04
VEL_XY_MAX = 0.3


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@configclass
class MirrorSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    paddle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Paddle",
        spawn=sim_utils.CylinderCfg(
            radius=PADDLE_RADIUS,
            height=PADDLE_HALF_THICKNESS * 2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=0.5, static_friction=0.3, dynamic_friction=0.3,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.85)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, PADDLE_REST_Z)),
    )
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.1, angular_damping=0.1,
                max_linear_velocity=10.0, max_angular_velocity=50.0,
                max_depenetration_velocity=1.0, disable_gravity=False,
                sleep_threshold=0.0,  # prevent PhysX from sleeping the ball
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0027),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=RESTITUTION, restitution_combine_mode="max",
                static_friction=0.3, dynamic_friction=0.3,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.5, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, SURFACE_Z + 0.08)),
    )
    target_marker = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetMarker",
        spawn=sim_utils.CylinderCfg(
            radius=0.15,
            height=0.005,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 1.0, 0.2), opacity=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.8)),  # updated in main
    )


# ---------------------------------------------------------------------------
# NN Controller (must match train_mirror.py)
# ---------------------------------------------------------------------------
class PaddleController(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 3), nn.Tanh(),
        )

    def forward(self, x):
        raw = self.net(x)
        z_offset = raw[:, 0:1] * 0.15
        roll = raw[:, 1:2] * MAX_TILT
        pitch = raw[:, 2:3] * MAX_TILT
        return torch.cat([z_offset, roll, pitch], dim=-1)


# ---------------------------------------------------------------------------
# Analytical mirror law (fallback if no NN loaded)
# ---------------------------------------------------------------------------
def mirror_controller(bp, bv, pp, target_h):
    ball_z = bp[2].item()
    ball_vz = bv[2].item()
    v_desired = math.sqrt(2.0 * G * target_h)
    ball_speed = abs(ball_vz)
    if ball_speed > 0.05:
        alpha_needed = (v_desired / ball_speed - RESTITUTION) / (1.0 + RESTITUTION)
        alpha_needed = max(0.0, min(1.0, alpha_needed))
        alpha = ALPHA_NEUTRAL + K_ALPHA_BLEND * (alpha_needed - ALPHA_NEUTRAL)
    else:
        alpha = 0.3
    alpha = max(0.0, min(0.8, alpha))
    target_z = PADDLE_REST_Z - alpha * (ball_z - PADDLE_REST_Z)
    target_z = max(0.2, min(0.8, target_z))

    ball_xy = bp[:2] - pp[:2]
    ball_vxy = bv[:2]
    roll = -(KP_LAT * ball_xy[1] + KD_LAT * ball_vxy[1]).item()
    pitch = -(KP_LAT * ball_xy[0] + KD_LAT * ball_vxy[0]).item()
    roll = max(-MAX_TILT, min(MAX_TILT, roll))
    pitch = max(-MAX_TILT, min(MAX_TILT, pitch))
    return target_z, roll, pitch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    target_h = args_cli.target_height

    sim_cfg = sim_utils.SimulationCfg(dt=DT, render_interval=4)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(1.0, 1.0, 0.8), target=(0.0, 0.0, PADDLE_REST_Z))

    scene_cfg = MirrorSceneCfg(num_envs=1, env_spacing=2.0)
    # Set target marker to correct height
    scene_cfg.target_marker.init_state.pos = (0.0, 0.0, SURFACE_Z + target_h)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    dev = sim_cfg.device
    origins = scene.env_origins  # (1, 3)
    ball = scene["ball"]
    paddle = scene["paddle"]

    # Warmup: first sim.step() initializes PhysX. Writes don't work before this.
    sim.step()
    ball.update(DT)
    paddle.update(DT)

    # Load NN controller or use analytical fallback
    nn_controller = None
    if args_cli.controller is not None:
        nn_controller = PaddleController().to(dev)
        ckpt = torch.load(args_cli.controller, map_location=dev, weights_only=True)
        nn_controller.load_state_dict(ckpt["state_dict"])
        nn_controller.eval()
        print(f"[INFO] Loaded NN controller from {args_cli.controller}")
        print(f"       Trained for target_height={ckpt.get('target_height', '?')}, fitness={ckpt.get('fitness', '?')}")
    else:
        print("[INFO] No --controller specified, using analytical mirror law")

    # Randomize initial ball state (now works after warmup step)
    bx = random.uniform(-XY_OFFSET_MAX, XY_OFFSET_MAX)
    by = random.uniform(-XY_OFFSET_MAX, XY_OFFSET_MAX)
    bz = SURFACE_Z + max(0.02, random.gauss(DROP_HEIGHT_MEAN, DROP_HEIGHT_STD))
    vx = random.uniform(-VEL_XY_MAX, VEL_XY_MAX)
    vy = random.uniform(-VEL_XY_MAX, VEL_XY_MAX)
    ball_pose = torch.tensor([[bx, by, bz, 1.0, 0.0, 0.0, 0.0]], device=dev)
    ball_pose[:, :3] += origins
    ball_vel = torch.tensor([[vx, vy, 0.0, 0.0, 0.0, 0.0]], device=dev)
    ball.write_root_pose_to_sim(ball_pose)
    ball.write_root_velocity_to_sim(ball_vel)
    sim.step()
    ball.update(DT)
    paddle.update(DT)

    print(f"\n  Target bounce height: {target_h:.2f} m")
    print(f"  Restitution: {RESTITUTION}\n")

    prev_vz = 0.0
    max_z_this_bounce = 0.0
    step = 0

    while simulation_app.is_running():
        with torch.inference_mode():
            bp = ball.data.root_pos_w[0] - origins[0]
            bv = ball.data.root_lin_vel_w[0]
            pp = paddle.data.root_pos_w[0] - origins[0]

            if nn_controller is not None:
                obs = torch.tensor([[
                    bp[0].item() - pp[0].item(),
                    bp[1].item() - pp[1].item(),
                    bp[2].item() - pp[2].item(),
                    bv[0].item(), bv[1].item(), bv[2].item(),
                ]], device=dev)
                cmd = nn_controller(obs)
                target_z = PADDLE_REST_Z + cmd[0, 0].item()
                target_z = max(0.25, min(0.75, target_z))
                roll_cmd = cmd[0, 1].item()
                pitch_cmd = cmd[0, 2].item()
            else:
                target_z, roll_cmd, pitch_cmd = mirror_controller(bp, bv, pp, target_h)

            quat = math_utils.quat_from_euler_xyz(
                torch.tensor([roll_cmd], device=dev),
                torch.tensor([pitch_cmd], device=dev),
                torch.tensor([0.0], device=dev),
            )
            new_pos = torch.tensor([[0.0, 0.0, target_z]], device=dev) + origins
            paddle.write_root_pose_to_sim(torch.cat([new_pos, quat], dim=-1))

            sim.step()
            ball.update(DT)
            paddle.update(DT)
            step += 1

            ball_z = bp[2].item()
            ball_vz = bv[2].item()
            max_z_this_bounce = max(max_z_this_bounce, ball_z)

            if prev_vz > 0 and ball_vz <= 0:
                apex_above_surface = max_z_this_bounce - SURFACE_Z
                print(
                    f"[step {step:5d}]  apex = {apex_above_surface:.3f} m  "
                    f"(target {target_h:.3f} m)  "
                    f"ball_xy = ({bp[0]:.3f}, {bp[1]:.3f})  "
                    f"tilt = ({math.degrees(roll_cmd):.1f}, {math.degrees(pitch_cmd):.1f})"
                )
                max_z_this_bounce = 0.0
            prev_vz = ball_vz

            # Reset if ball falls off
            if ball_z < 0.1 or abs(bp[0].item()) > 0.5 or abs(bp[1].item()) > 0.5:
                print("[RESET] Ball fell off — resetting.")
                bx = random.uniform(-XY_OFFSET_MAX, XY_OFFSET_MAX)
                by = random.uniform(-XY_OFFSET_MAX, XY_OFFSET_MAX)
                bz = SURFACE_Z + max(0.02, random.gauss(DROP_HEIGHT_MEAN, DROP_HEIGHT_STD))
                vx = random.uniform(-VEL_XY_MAX, VEL_XY_MAX)
                vy = random.uniform(-VEL_XY_MAX, VEL_XY_MAX)
                r_pose = torch.tensor([[bx, by, bz, 1.0, 0.0, 0.0, 0.0]], device=dev)
                r_pose[:, :3] += origins
                r_vel = torch.tensor([[vx, vy, 0.0, 0.0, 0.0, 0.0]], device=dev)
                ball.write_root_pose_to_sim(r_pose)
                ball.write_root_velocity_to_sim(r_vel)
                max_z_this_bounce = 0.0


if __name__ == "__main__":
    main()
    simulation_app.close()
