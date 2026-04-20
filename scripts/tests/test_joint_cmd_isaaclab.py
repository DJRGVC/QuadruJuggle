"""Joint command step-response test — Isaac Lab (raw sim, no ManagerBasedRLEnv).

play_mirror_law.py is the official reference (BallJuggleMirrorEnvCfg):
  sim.dt = 1/200 s (200 Hz physics), decimation=4 (50 Hz policy).
Both test scripts use SIM_DT = 1/200 to match.

  - ActuatorNetMLP (same as training) holding default positions
  - Steps at Isaac Lab physics rate: 200 Hz (dt=0.005 s)
  - 50-step warm-up to fill the actuator-net history before recording

Protocol
--------
Phase 1 (steps 0 .. HOLD_STEPS-1):   target = default positions
Phase 2 (steps HOLD_STEPS .. end):   target[joint_idx] += STEP_AMP

Usage
-----
cd /home/frank/berkeley_mde/QuadruJuggle
PYTHONPATH=/home/frank/IsaacLab/scripts/reinforcement_learning/rsl_rl:$PYTHONPATH

# Viewer:
python scripts/tests/test_joint_cmd_isaaclab.py --joint_idx 3 --viewer

# Headless CSV:
python scripts/tests/test_joint_cmd_isaaclab.py \\
    --joint_idx 3 --headless --out tests_out/isaaclab_joint3.csv
"""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Joint step-response test — raw Isaac Lab sim")
parser.add_argument("--joint_idx", type=int, default=None,
                    help="Joint index to perturb (0-11). Default: 3 headless, all viewer.")
parser.add_argument("--step_amp", type=float, default=0.30,
                    help="Step amplitude in radians (default 0.30)")
parser.add_argument("--hold_steps", type=int, default=100,
                    help="Steps at default before step input (default 100)")
parser.add_argument("--step_steps", type=int, default=300,
                    help="Steps at step target (default 300)")
parser.add_argument("--warmup_steps", type=int, default=50,
                    help="Physics steps to warm up actuator-net history (default 50)")
parser.add_argument("--aggressive", action="store_true",
                    help="Demo mode: oscillate ALL joints through large ±1.2 rad steps "
                         "in sequence, cycling forever. Requires --viewer.")
parser.add_argument("--viewer", action="store_true",
                    help="Open Omniverse viewport.")
parser.add_argument("--out", type=str, default=None,
                    help="Write CSV to this file.")
parser.add_argument("--no_header", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

if args.viewer:
    args.headless = False
elif not hasattr(args, "headless") or args.headless is None:
    args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ── Imports (after AppLauncher) ───────────────────────────────────────────────
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # isort: skip

# ── Constants ─────────────────────────────────────────────────────────────────
SIM_DT = 1.0 / 200.0   # Must match play_mirror_law.py (BallJuggleMirrorEnvCfg sim.dt)

# Spawn at z=0.35m (correct Isaac Lab standing height) so feet touch ground immediately.
# UNITREE_GO1_CFG default is z=0.4m which leaves feet ~0.05m in the air;
# with the cold actuator-net (near-zero torques at zero history) the robot
# then falls before contact is established.
_GO1_CFG = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
_GO1_CFG.init_state.pos = (0.0, 0.0, 0.35)


# ── Scene config ──────────────────────────────────────────────────────────────
@configclass
class MinimalSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    robot: ArticulationCfg = _GO1_CFG


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Simulation context (no decimation — step at physics rate, same as MuJoCo) ──
    # render_interval=4 → viewer updates at 50 Hz (real-time feel without
    # rendering every 200 Hz physics step, which would be GPU-limited slow motion)
    sim_cfg = SimulationCfg(dt=SIM_DT, render_interval=4)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 1.5], target=[0.0, 0.0, 0.4])

    # ── Scene + reset ─────────────────────────────────────────────────────────
    scene = InteractiveScene(MinimalSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.update(SIM_DT)

    robot = scene["robot"]
    num_joints  = robot.num_joints
    joint_names = list(robot.joint_names)
    default_pos = robot.data.default_joint_pos.clone()   # (1, 12)

    print(f"[isaaclab] Device     : {device}", file=sys.stderr)
    print(f"[isaaclab] Num joints : {num_joints}", file=sys.stderr)
    print(f"[isaaclab] Joint ordering:", file=sys.stderr)
    for ji, (name, val) in enumerate(zip(joint_names, default_pos[0].cpu().tolist())):
        print(f"  [{ji:2d}] {name:25s}  default={val:+.4f} rad", file=sys.stderr)

    # ── Joint sequence ────────────────────────────────────────────────────────
    if args.joint_idx is not None:
        joint_sequence = [args.joint_idx]
    elif args.viewer:
        joint_sequence = list(range(num_joints))
    else:
        joint_sequence = [3]

    print(f"[isaaclab] Joints to test: {[joint_names[i] for i in joint_sequence]}",
          file=sys.stderr)
    print(f"[isaaclab] Step amp  : {args.step_amp:+.3f} rad  "
          f"hold={args.hold_steps}  step={args.step_steps}", file=sys.stderr)
    print(f"[isaaclab] Spawn z   : {_GO1_CFG.init_state.pos[2]:.3f} m (feet on ground)", file=sys.stderr)
    print(f"[isaaclab] Warmup    : {args.warmup_steps} physics steps (fills actuator-net history)", file=sys.stderr)

    # ── Output file ───────────────────────────────────────────────────────────
    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_file = open(args.out, "w")
        print(f"[isaaclab] Writing CSV → {args.out}", file=sys.stderr)
    elif args.viewer:
        out_file = None
    else:
        out_file = sys.stdout

    if out_file and not args.no_header:
        print("step,sim,actuator,joint_name,joint_idx_isaac,"
              "target_rad,actual_rad,error_rad,torque_Nm",
              file=out_file)

    # ── Actuator warm-up: fill net history at default positions ───────────────
    # Mirrors test_joint_cmd_mujoco.py reset_sim() pre-warm loop.
    print(f"[isaaclab] Warming up actuator net ({args.warmup_steps} steps)...",
          file=sys.stderr)
    for _ in range(args.warmup_steps):
        robot.set_joint_position_target(default_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(SIM_DT)
        if not simulation_app.is_running():
            break

    z0 = robot.data.root_pos_w[0, 2].item()
    print(f"[isaaclab] Post-warmup trunk_z = {z0:.4f} m", file=sys.stderr)

    # ── Aggressive demo mode ──────────────────────────────────────────────────
    if args.aggressive:
        AMP   = 1.2    # rad — large enough to be very visible
        HOLD  = 80     # steps per phase (0.4s at 200 Hz)
        print(f"\n[isaaclab] AGGRESSIVE mode: ±{AMP} rad on all joints, cycling forever.",
              file=sys.stderr)
        phase = 0
        while simulation_app.is_running():
            # Every HOLD steps, advance to next joint / direction
            ji     = (phase // 2) % num_joints
            sign   = +1.0 if (phase % 2 == 0) else -1.0
            targets = default_pos.clone()
            targets[0, ji] = default_pos[0, ji] + sign * AMP
            for _ in range(HOLD):
                if not simulation_app.is_running():
                    break
                t0 = time.perf_counter()
                robot.set_joint_position_target(targets)
                scene.write_data_to_sim()
                sim.step()
                scene.update(SIM_DT)
            phase += 1
        simulation_app.close()
        return

    # ── Per-joint step-response test ──────────────────────────────────────────
    for joint_idx in joint_sequence:
        if not simulation_app.is_running():
            break

        jname = joint_names[joint_idx]
        total_steps = args.hold_steps + args.step_steps
        default_val = default_pos[0, joint_idx].item()
        print(f"\n[isaaclab] ── Joint [{joint_idx}] {jname}  "
              f"(default: {default_val:+.3f} → "
              f"target: {default_val + args.step_amp:+.3f}) ──",
              file=sys.stderr)

        targets = default_pos.clone()   # (1, 12)

        for step in range(total_steps):
            if not simulation_app.is_running():
                break
            t0 = time.perf_counter()

            if step >= args.hold_steps:
                targets[0, joint_idx] = default_pos[0, joint_idx] + args.step_amp

            # Capture state BEFORE the step (matches MuJoCo: log then step)
            actual_pos      = robot.data.joint_pos[0].cpu()
            applied_torques = robot.data.applied_torque[0].cpu()

            robot.set_joint_position_target(targets)
            scene.write_data_to_sim()
            sim.step()
            scene.update(SIM_DT)

            if out_file:
                for ji, jn in enumerate(joint_names):
                    print(
                        f"{step},isaaclab,actnet,{jn},{ji},"
                        f"{targets[0, ji].item():.6f},"
                        f"{actual_pos[ji].item():.6f},"
                        f"{actual_pos[ji].item() - targets[0, ji].item():.6f},"
                        f"{applied_torques[ji].item():.4f}",
                        file=out_file,
                    )

            if args.viewer:
                elapsed = time.perf_counter() - t0
                time.sleep(max(0.0, SIM_DT - elapsed))

        # Reset between joints: return to default and re-warm
        if len(joint_sequence) > 1:
            targets = default_pos.clone()
            for _ in range(args.warmup_steps):
                robot.set_joint_position_target(default_pos)
                scene.write_data_to_sim()
                sim.step()
                scene.update(SIM_DT)
                if not simulation_app.is_running():
                    break

    if args.out and out_file:
        out_file.close()

    simulation_app.close()


if __name__ == "__main__":
    main()
