"""Joint command comparison test — Isaac Lab side.

Mirrors test_joint_cmd_mujoco.py exactly: same step-input protocol,
same output CSV format.  Compare the two CSVs to debug joint ordering,
default positions, and PD gain mismatches between simulators.

Protocol
--------
Phase 1 (steps 0 .. HOLD_STEPS-1):   action = 0  → target = default positions
Phase 2 (steps HOLD_STEPS .. end):
        action[joint_idx] = STEP_AMP / ACTION_SCALE → target[joint_idx] += STEP_AMP

Joint indices are in Isaac Lab order (FL, FR, RL, RR):
    0: FL_hip    1: FL_thigh    2: FL_calf
    3: FR_hip    4: FR_thigh    5: FR_calf
    6: RL_hip    7: RL_thigh    8: RL_calf
    9: RR_hip   10: RR_thigh   11: RR_calf

Action scale: 0.25  (JointPositionActionCfg scale=0.25 in TorsoTrackingEnvCfg)

Usage
-----
cd /home/frank/berkeley_mde/QuadruJuggle
PYTHONPATH=/home/frank/IsaacLab/scripts/reinforcement_learning/rsl_rl:$PYTHONPATH

# Watch robot move (Omniverse viewport):
python scripts/tests/test_joint_cmd_isaaclab.py --joint_idx 3 --viewer

# Cycle all 12 joints with viewer:
python scripts/tests/test_joint_cmd_isaaclab.py --viewer

# Save CSV (headless):
python scripts/tests/test_joint_cmd_isaaclab.py \\
    --joint_idx 3 --headless --out tests_out/isaaclab_joint3.csv
"""

import argparse
import os
import sys
import time

# Isaac Lab AppLauncher must be created BEFORE any torch/isaac imports
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Joint step-response test in Isaac Lab")
parser.add_argument("--joint_idx", type=int, default=None,
                    help="Isaac Lab joint index to perturb (0-11). "
                         "Default: cycle through all 12 in viewer mode, or 3 (FR_hip) headless.")
parser.add_argument("--step_amp", type=float, default=0.30,
                    help="Step amplitude in radians (default 0.30)")
parser.add_argument("--hold_steps", type=int, default=100,
                    help="Policy steps at default before step input (default 100)")
parser.add_argument("--step_steps", type=int, default=300,
                    help="Policy steps at step target (default 300)")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Number of parallel Isaac Lab envs (default 1)")
parser.add_argument("--viewer", action="store_true",
                    help="Open Omniverse viewport. Cycles all 12 joints unless --joint_idx set.")
parser.add_argument("--out", type=str, default=None,
                    help="Write CSV to this file (avoids Isaac Lab log lines in stdout).")
parser.add_argument("--no_header", action="store_true",
                    help="Suppress CSV header")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# --viewer overrides --headless; otherwise default to headless
if args.viewer:
    args.headless = False
elif not hasattr(args, "headless") or args.headless is None:
    args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Now safe to import torch and Isaac Lab ────────────────────────────────────
import torch
import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import go1_ball_balance  # noqa: F401

from go1_ball_balance.tasks.torso_tracking.torso_tracking_env_cfg import TorsoTrackingEnvCfg_PLAY

# Action scale from TorsoTrackingEnvCfg ActionsCfg (scale=0.25)
ACTION_SCALE = 0.25

# Real-time pacing: env step_dt = 0.02 s (50 Hz policy)
ENV_DT = 0.02


WARMUP_STEPS = 20   # physics steps at default pose before test begins


def run_joint(env, robot, device, default_pos, joint_names, joint_idx, step_amp,
              hold_steps, step_steps, out_file, no_header, first_joint,
              use_viewer):
    """Run one joint's step-input sequence."""
    total_steps = hold_steps + step_steps
    jname = joint_names[joint_idx]

    print(f"\n[isaaclab] ── Joint [{joint_idx}] {jname}  "
          f"(default: {default_pos[joint_idx]:+.3f} → target: "
          f"{default_pos[joint_idx]+step_amp:+.3f}) ──", file=sys.stderr)

    # Reset once, then run warmup steps at zero action so robot settles
    obs_dict, _ = env.reset()
    zero_action = torch.zeros(args.num_envs, 12, device=device)
    for _ in range(WARMUP_STEPS):
        obs_dict, _, _, _, _ = env.step(zero_action)
        if not simulation_app.is_running():
            return False

    if not no_header and first_joint and out_file:
        print("step,sim,joint_name,joint_idx_isaac,target_rad,actual_rad,error_rad,torque_Nm",
              file=out_file)

    for step in range(total_steps):
        t0 = time.perf_counter()

        action = torch.zeros(args.num_envs, 12, device=device)
        if step >= hold_steps:
            action[:, joint_idx] = step_amp / ACTION_SCALE

        obs_dict, _, terminated, truncated, _ = env.step(action)

        # Log
        if out_file:
            actual_pos      = robot.data.joint_pos[0].cpu().numpy()
            applied_torques = robot.data.applied_torque[0].cpu().numpy()
            target_pos      = default_pos + ACTION_SCALE * action[0].cpu().numpy()
            for ji, jn in enumerate(joint_names):
                print(
                    f"{step},isaaclab,{jn},{ji},"
                    f"{target_pos[ji]:.6f},{actual_pos[ji]:.6f},"
                    f"{actual_pos[ji]-target_pos[ji]:.6f},{applied_torques[ji]:.4f}",
                    file=out_file,
                )

        # Termination disabled above — just warn if it somehow fires.
        if terminated.any() or truncated.any():
            print(f"[isaaclab] WARNING: unexpected termination at step {step}", file=sys.stderr)

        # Real-time pacing in viewer mode
        if use_viewer:
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, ENV_DT - elapsed))

        if not simulation_app.is_running():
            return False

    return True


def main():
    # ── Joint sequence ────────────────────────────────────────────────────────
    if args.joint_idx is not None:
        joint_sequence = [args.joint_idx]
    elif args.viewer:
        joint_sequence = list(range(12))   # cycle all 12 in viewer mode
    else:
        joint_sequence = [3]               # default: FR_hip for CSV mode

    # ── Build env ─────────────────────────────────────────────────────────────
    env_cfg = TorsoTrackingEnvCfg_PLAY()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.observations.policy.enable_corruption = False

    # Disable trunk_collapsed — a single joint deflection can drop the trunk
    # below 0.15 m and trigger spurious saw-tooth resets in the CSV.
    env_cfg.terminations.trunk_collapsed.params["minimum_height"] = -10.0

    # Long fixed episode so time_out never fires during the test.
    env_cfg.episode_length_s = 500.0  # >> any test duration

    # Explicit spawn height (mirrors UNITREE_GO1_CFG default; set here so it's
    # easy to adjust if the robot clips through the ground on your setup).
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.40)

    env = gym.make("Isaac-TorsoTracking-Go1-Play-v0", cfg=env_cfg)
    device = env.unwrapped.device
    robot  = env.unwrapped.scene["robot"]

    # ── Query actual joint names from the articulation — DO NOT hardcode ──────
    # Isaac Lab's joint ordering depends on the USD hierarchy and is NOT the
    # same as the per-leg (FL hip+thigh+calf, FR, RL, RR) order assumed by
    # play_mujoco.py.  Querying at runtime is the only reliable approach.
    joint_names = list(robot.joint_names)   # actual order from articulation
    num_joints   = len(joint_names)

    print(f"[isaaclab] Device    : {device}", file=sys.stderr)
    print(f"[isaaclab] Viewer    : {args.viewer}", file=sys.stderr)
    print(f"[isaaclab] Step amp  : {args.step_amp:+.3f} rad  hold={args.hold_steps}  step={args.step_steps}", file=sys.stderr)
    print(f"[isaaclab] Num joints: {num_joints}", file=sys.stderr)

    # Read default positions from articulation
    default_pos = robot.data.default_joint_pos[0].cpu().numpy().copy()

    print("[isaaclab] Actual joint ordering from articulation:", file=sys.stderr)
    for ji, (name, val) in enumerate(zip(joint_names, default_pos)):
        print(f"  [{ji:2d}] {name:25s}  default={val:+.4f} rad", file=sys.stderr)

    # Resolve joint_sequence indices to names for logging
    print(f"[isaaclab] Joints to test: {[joint_names[i] for i in joint_sequence]}", file=sys.stderr)

    # ── Open output file ──────────────────────────────────────────────────────
    if args.out:
        out_file = open(args.out, "w")
        print(f"[isaaclab] Writing CSV → {args.out}", file=sys.stderr)
    elif args.viewer:
        out_file = None    # no CSV spam in viewer mode unless --out given
    else:
        out_file = sys.stdout

    # ── Run joint sequence ────────────────────────────────────────────────────
    first = True
    for ji in joint_sequence:
        if not simulation_app.is_running():
            break
        ok = run_joint(
            env, robot, device, default_pos, joint_names,
            ji, args.step_amp, args.hold_steps, args.step_steps,
            out_file=out_file,
            no_header=args.no_header,
            first_joint=first,
            use_viewer=args.viewer,
        )
        first = False
        if not ok:
            break

    if args.out and out_file:
        out_file.close()

    env.close()
    
    simulation_app.close()


if __name__ == "__main__":
    main()
