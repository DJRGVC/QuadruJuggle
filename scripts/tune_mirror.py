"""Headless grid search over mirror-law controller parameters.

Tests all sign/axis combinations and a range of PD gains.
Fitness = number of physics steps before ball falls off.
Prints the best parameters at the end.

Usage:
    uv run --active python scripts/tune_mirror.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Tune mirror-law controller")
parser.add_argument("--target-height", type=float, default=0.30)
parser.add_argument("--steps", type=int, default=2000, help="Max steps per trial")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import itertools
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import RigidObject, RigidObjectCfg

# Constants (same as play_mirror.py)
DT = 1.0 / 200.0
G = 9.81
BALL_RADIUS = 0.020
RESTITUTION = 0.85
PADDLE_REST_Z = 0.5
PADDLE_RADIUS = 0.085
PADDLE_HALF_THICKNESS = 0.005
ALPHA_NEUTRAL = (1.0 - RESTITUTION) / (1.0 + RESTITUTION)
MAX_TILT = 0.30


def build_scene():
    """Spawn ground + paddle + ball."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0)
    light_cfg.func("/World/DomeLight", light_cfg)

    paddle = RigidObject(RigidObjectCfg(
        prim_path="/World/Paddle",
        spawn=sim_utils.CylinderCfg(
            radius=PADDLE_RADIUS,
            height=PADDLE_HALF_THICKNESS * 2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=0.5, static_friction=0.3, dynamic_friction=0.3,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, PADDLE_REST_Z)),
    ))

    ball = RigidObject(RigidObjectCfg(
        prim_path="/World/Ball",
        spawn=sim_utils.SphereCfg(
            radius=BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                linear_damping=0.1, angular_damping=0.1,
                max_linear_velocity=10.0, max_angular_velocity=50.0,
                max_depenetration_velocity=1.0, disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0027),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=RESTITUTION, restitution_combine_mode="max",
                static_friction=0.3, dynamic_friction=0.3,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, PADDLE_REST_Z + PADDLE_HALF_THICKNESS + BALL_RADIUS + 0.10),
        ),
    ))

    return paddle, ball


def run_trial(sim, paddle, ball, params, target_h, max_steps, dev):
    """Run one trial, return (steps_alive, mean_apex_error)."""
    kp = params["kp"]
    kd = params["kd"]
    roll_sign = params["roll_sign"]
    pitch_sign = params["pitch_sign"]
    # swap: if True, roll controls X offset, pitch controls Y offset
    swap = params["swap"]
    k_alpha = params["k_alpha"]

    # Reset ball: small offset + small lateral velocity
    surface_z = PADDLE_REST_Z + PADDLE_HALF_THICKNESS + BALL_RADIUS
    ball_pos0 = torch.tensor([[0.02, -0.01, surface_z + 0.08]], device=dev)
    ball_quat0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev)
    ball_vel0 = torch.tensor([[0.15, -0.10, 0.0, 0.0, 0.0, 0.0]], device=dev)

    # Reset paddle
    paddle_pose0 = torch.tensor([[0.0, 0.0, PADDLE_REST_Z, 1.0, 0.0, 0.0, 0.0]], device=dev)

    # Write reset states and step a few times to let PhysX commit them
    ball.write_root_pose_to_sim(torch.cat([ball_pos0, ball_quat0], dim=-1))
    ball.write_root_velocity_to_sim(ball_vel0)
    paddle.write_root_pose_to_sim(paddle_pose0)
    for _ in range(5):
        sim.step()
        # Re-write each step to hold the reset state until PhysX commits
        ball.write_root_pose_to_sim(torch.cat([ball_pos0, ball_quat0], dim=-1))
        ball.write_root_velocity_to_sim(ball_vel0)
        paddle.write_root_pose_to_sim(paddle_pose0)
    ball.update(DT)
    paddle.update(DT)

    v_desired = math.sqrt(2.0 * G * target_h)
    apex_errors = []
    prev_vz = 0.0
    max_z = 0.0

    for step in range(max_steps):
        # Read state first
        bp = ball.data.root_pos_w[0]
        bv = ball.data.root_lin_vel_w[0]
        pp = paddle.data.root_pos_w[0]

        ball_z = bp[2].item()
        ball_vz = bv[2].item()

        # Check termination
        if ball_z < 0.1 or abs(bp[0].item()) > 0.5 or abs(bp[1].item()) > 0.5:
            break

        # Track apex
        max_z = max(max_z, ball_z)
        if prev_vz > 0 and ball_vz <= 0:
            apex_h = max_z - surface_z
            apex_errors.append(abs(apex_h - target_h))
            max_z = 0.0
        prev_vz = ball_vz

        # --- Vertical mirror law ---
        ball_speed = abs(ball_vz)
        if ball_speed > 0.05:
            alpha_needed = (v_desired / ball_speed - RESTITUTION) / (1.0 + RESTITUTION)
            alpha_needed = max(0.0, min(1.0, alpha_needed))
            alpha = ALPHA_NEUTRAL + k_alpha * (alpha_needed - ALPHA_NEUTRAL)
        else:
            alpha = 0.3
        alpha = max(0.0, min(0.8, alpha))

        target_z = PADDLE_REST_Z - alpha * (ball_z - PADDLE_REST_Z)
        target_z = max(0.2, min(0.8, target_z))

        # --- Lateral PD ---
        ball_xy = bp[:2] - pp[:2]
        ball_vxy = bv[:2]

        if swap:
            raw_roll = kp * ball_xy[0].item() + kd * ball_vxy[0].item()
            raw_pitch = kp * ball_xy[1].item() + kd * ball_vxy[1].item()
        else:
            raw_roll = kp * ball_xy[1].item() + kd * ball_vxy[1].item()
            raw_pitch = kp * ball_xy[0].item() + kd * ball_vxy[0].item()

        roll_cmd = max(-MAX_TILT, min(MAX_TILT, roll_sign * raw_roll))
        pitch_cmd = max(-MAX_TILT, min(MAX_TILT, pitch_sign * raw_pitch))

        quat = math_utils.quat_from_euler_xyz(
            torch.tensor([roll_cmd], device=dev),
            torch.tensor([pitch_cmd], device=dev),
            torch.tensor([0.0], device=dev),
        )
        new_pos = torch.tensor([[pp[0].item(), pp[1].item(), target_z]], device=dev)
        paddle.write_root_pose_to_sim(torch.cat([new_pos, quat], dim=-1))

        # Step physics, then update buffers
        sim.step()
        ball.update(DT)
        paddle.update(DT)

    mean_apex_err = sum(apex_errors) / len(apex_errors) if apex_errors else 999.0
    return step + 1, mean_apex_err


def main():
    target_h = args_cli.target_height
    max_steps = args_cli.steps

    sim_cfg = sim_utils.SimulationCfg(dt=DT, render_interval=max_steps + 1)  # no rendering
    sim = SimulationContext(sim_cfg)

    paddle, ball = build_scene()
    sim.reset()
    paddle.reset()
    ball.reset()

    dev = sim_cfg.device

    # Grid search parameters
    roll_signs = [+1, -1]
    pitch_signs = [+1, -1]
    swaps = [False, True]
    kps = [1.0, 3.0, 6.0, 10.0]
    kds = [0.3, 0.8, 1.5, 3.0]
    k_alphas = [0.3, 0.5, 0.8, 1.0]

    results = []
    total = len(roll_signs) * len(pitch_signs) * len(swaps) * len(kps) * len(kds) * len(k_alphas)
    print(f"Running {total} trials, {max_steps} steps each...\n")

    trial_num = 0
    for rs, ps, sw, kp, kd, ka in itertools.product(roll_signs, pitch_signs, swaps, kps, kds, k_alphas):
        trial_num += 1
        params = {"roll_sign": rs, "pitch_sign": ps, "swap": sw, "kp": kp, "kd": kd, "k_alpha": ka}
        steps_alive, apex_err = run_trial(sim, paddle, ball, params, target_h, max_steps, dev)

        # Fitness: primarily survival, secondarily apex accuracy
        fitness = steps_alive + (1.0 / (1.0 + apex_err)) * 100
        results.append((fitness, steps_alive, apex_err, params))

        if trial_num % 50 == 0 or steps_alive > max_steps * 0.5:
            print(f"  [{trial_num}/{total}] alive={steps_alive:4d}  apex_err={apex_err:.3f}  "
                  f"rs={rs:+d} ps={ps:+d} sw={sw} kp={kp:.1f} kd={kd:.1f} ka={ka:.1f}")

    # Sort by fitness
    results.sort(key=lambda x: -x[0])

    print(f"\n{'='*70}")
    print(f"  TOP 10 RESULTS (out of {total} trials)")
    print(f"{'='*70}")
    for i, (fit, alive, aerr, p) in enumerate(results[:10]):
        print(f"  #{i+1:2d}  alive={alive:4d}  apex_err={aerr:.3f}  "
              f"rs={p['roll_sign']:+d} ps={p['pitch_sign']:+d} swap={p['swap']}  "
              f"kp={p['kp']:.1f}  kd={p['kd']:.1f}  k_alpha={p['k_alpha']:.1f}")

    best = results[0][3]
    print(f"\n  BEST: roll_sign={best['roll_sign']:+d}  pitch_sign={best['pitch_sign']:+d}  "
          f"swap={best['swap']}  kp={best['kp']:.1f}  kd={best['kd']:.1f}  k_alpha={best['k_alpha']:.1f}")
    print()


if __name__ == "__main__":
    main()
    simulation_app.close()
