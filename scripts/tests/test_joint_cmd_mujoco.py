"""Joint command comparison test — MuJoCo side.

Applies a fixed step-input to each joint in sequence and logs the actual
joint response.  Run alongside test_joint_cmd_isaaclab.py to compare
joint ordering, PD gains, and default positions across simulators.

Two actuator modes
------------------
  PD (default)        KP=100, KD=3 — fast but not matching Isaac Lab
  --use_actuator_net  Same unitree_go1.pt MLP that Isaac Lab uses internally
                      → this is the apples-to-apples comparison

Protocol
--------
Phase 1 (steps 0 .. HOLD_STEPS-1):   target = default positions
Phase 2 (steps HOLD_STEPS .. end):   target = default + STEP_AMP on active joint

Usage
-----
cd /home/frank/berkeley_mde/QuadruJuggle

# Watch with actuator net (matches Isaac Lab):
python scripts/tests/test_joint_cmd_mujoco.py --viewer --use_actuator_net

# Single joint, viewer:
python scripts/tests/test_joint_cmd_mujoco.py --viewer --joint_idx 3 --use_actuator_net

# Save CSV (headless, actuator net):
python scripts/tests/test_joint_cmd_mujoco.py \\
    --joint_idx 0 --use_actuator_net --out tests_out/mujoco_actnet_joint0.csv

# Save CSV (headless, plain PD):
python scripts/tests/test_joint_cmd_mujoco.py \\
    --joint_idx 0 --out tests_out/mujoco_pd_joint0.csv
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import mujoco
import mujoco.viewer

# ── Constants (must match play_mujoco.py exactly) ─────────────────────────────

DEFAULT_JOINT_POS_ISAAC = np.array([
     0.1,  0.8, -1.5,   # FL: hip, thigh, calf
    -0.1,  0.8, -1.5,   # FR
     0.1,  1.0, -1.5,   # RL
    -0.1,  1.0, -1.5,   # RR
])

REINDEX = np.array([3, 4, 5,  0, 1, 2,  9, 10, 11,  6, 7, 8])
DEFAULT_JOINT_POS_MJCF = DEFAULT_JOINT_POS_ISAAC[REINDEX]

JOINT_NAMES_MJCF = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]

# PD fallback gains
KP = 100.0
KD = 3.0
EFFORT_LIMIT = 23.7  # N·m  (matches actuator net limit)

# Default actuator net path (same file Isaac Lab uses)
DEFAULT_ACTUATOR_NET_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__),
    "../../../walk-these-ways/resources/actuator_nets/unitree_go1.pt",
))

SIM_DT = 0.002   # MuJoCo timestep


# ── Actuator net wrapper (copied from play_mujoco.py) ─────────────────────────

class GoActuatorNet:
    """Wraps unitree_go1.pt with 3-step pos-error + vel history.

    Matches Isaac Lab ActuatorNetMLPCfg:
        input_idx=[0,1,2], pos_scale=-1.0, vel_scale=1.0,
        input_order="pos_vel", effort_limit=23.7 N·m
    """
    HISTORY_LEN  = 3
    POS_SCALE    = -1.0
    VEL_SCALE    =  1.0
    EFFORT_LIMIT = 23.7

    def __init__(self, net_path: str):
        net_path = os.path.abspath(net_path)
        if not os.path.exists(net_path):
            raise FileNotFoundError(f"Actuator net not found: {net_path}")
        self.net = torch.jit.load(net_path, map_location="cpu").eval()
        self._pos_hist = np.zeros((self.HISTORY_LEN, 12), dtype=np.float32)
        self._vel_hist = np.zeros((self.HISTORY_LEN, 12), dtype=np.float32)
        print(f"[ActuatorNet] Loaded: {net_path}", file=sys.stderr)

    def reset(self):
        self._pos_hist[:] = 0.0
        self._vel_hist[:] = 0.0

    def compute(self, targets_mjcf, joint_pos_mjcf, joint_vel_mjcf):
        """Return torques (MJCF order), clipped to ±23.7 N·m."""
        pos_err = targets_mjcf - joint_pos_mjcf
        self._pos_hist = np.roll(self._pos_hist, 1, axis=0)
        self._vel_hist = np.roll(self._vel_hist, 1, axis=0)
        self._pos_hist[0] = pos_err
        self._vel_hist[0] = joint_vel_mjcf

        pos_in  = (self._pos_hist * self.POS_SCALE).T   # (12, 3)
        vel_in  = (self._vel_hist * self.VEL_SCALE).T   # (12, 3)
        net_in  = np.concatenate([pos_in, vel_in], axis=1)  # (12, 6)

        with torch.no_grad():
            torques = self.net(torch.from_numpy(net_in)).squeeze(-1).numpy()
        return np.clip(torques, -self.EFFORT_LIMIT, self.EFFORT_LIMIT)


# ── Helpers ───────────────────────────────────────────────────────────────────

WARMUP_STEPS = 50   # steps at default pose before test — warms actuator net history


def compute_torques(target, data, actuator_net):
    """Compute torques via actuator net or PD fallback."""
    pos = data.qpos[7:19]
    vel = data.qvel[6:18]
    if actuator_net is not None:
        return actuator_net.compute(target, pos, vel)
    else:
        return np.clip(KP * (target - pos) + KD * (-vel), -EFFORT_LIMIT, EFFORT_LIMIT)


def reset_sim(model, data, actuator_net=None):
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.35]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:19] = DEFAULT_JOINT_POS_MJCF
    data.qpos[19:22] = [0.0, 0.0, 0.53]
    data.qpos[22:26] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    if actuator_net is not None:
        actuator_net.reset()
        # Pre-warm history at default pose so step 0 torque is already correct.
        for _ in range(WARMUP_STEPS):
            t = compute_torques(DEFAULT_JOINT_POS_MJCF, data, actuator_net)
            data.ctrl[:] = t
            mujoco.mj_step(model, data)


def run_steps(model, data, actuator_net, joint_idx, step_amp, hold_steps, step_steps,
              viewer=None, out_file=None, no_header=False):
    total_steps = hold_steps + step_steps

    if out_file and not no_header:
        mode = "actnet" if actuator_net else "pd"
        print(f"step,sim,actuator,joint_name,joint_idx_mjcf,"
              f"target_rad,actual_rad,error_rad,torque_Nm",
              file=out_file)

    mode = "actnet" if actuator_net else "pd"

    for step in range(total_steps):
        t0 = time.perf_counter()

        target = DEFAULT_JOINT_POS_MJCF.copy()
        if step >= hold_steps:
            target[joint_idx] += step_amp

        torques = compute_torques(target, data, actuator_net)
        data.ctrl[:] = torques
        mujoco.mj_step(model, data)

        if out_file:
            actual = data.qpos[7:19].copy()
            for ji in range(12):
                print(
                    f"{step},mujoco,{mode},{JOINT_NAMES_MJCF[ji]},{ji},"
                    f"{target[ji]:.6f},{actual[ji]:.6f},"
                    f"{actual[ji]-target[ji]:.6f},{torques[ji]:.4f}",
                    file=out_file,
                )

        if viewer is not None:
            viewer.sync()
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, SIM_DT - elapsed))
            if not viewer.is_running():
                return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Joint step-response test in MuJoCo")
    parser.add_argument("--joint_idx", type=int, default=None,
                        help="MJCF joint index (0-11). Default: all 12 in viewer, 0 headless.")
    parser.add_argument("--step_amp", type=float, default=0.30,
                        help="Step amplitude in radians (default 0.30)")
    parser.add_argument("--hold_steps", type=int, default=100,
                        help="Steps at default before step input (default 100)")
    parser.add_argument("--step_steps", type=int, default=300,
                        help="Steps at step target (default 300)")
    parser.add_argument("--use_actuator_net", action="store_true",
                        help="Use unitree_go1.pt MLP actuator (same as Isaac Lab) "
                             "instead of plain PD (KP=100, KD=3).")
    parser.add_argument("--actuator_net_path", type=str,
                        default=DEFAULT_ACTUATOR_NET_PATH,
                        help=f"Path to unitree_go1.pt (default: {DEFAULT_ACTUATOR_NET_PATH})")
    parser.add_argument("--viewer", action="store_true",
                        help="Open interactive MuJoCo viewer.")
    parser.add_argument("--xml", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "../../mujoco/go1_juggle.xml"))
    parser.add_argument("--out", type=str, default=None,
                        help="Write CSV to file (default: stdout; suppressed in viewer mode).")
    parser.add_argument("--no_header", action="store_true")
    args = parser.parse_args()

    xml_path = os.path.abspath(args.xml)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    # ── Actuator ──────────────────────────────────────────────────────────────
    actuator_net = None
    if args.use_actuator_net:
        actuator_net = GoActuatorNet(args.actuator_net_path)
        print(f"[mujoco] Actuator  : MLP net (unitree_go1.pt)", file=sys.stderr)
    else:
        print(f"[mujoco] Actuator  : PD  KP={KP}  KD={KD}", file=sys.stderr)

    # ── Joint sequence ────────────────────────────────────────────────────────
    if args.joint_idx is not None:
        joint_sequence = [args.joint_idx]
    elif args.viewer:
        joint_sequence = list(range(12))
    else:
        joint_sequence = [0]

    print(f"[mujoco] XML       : {xml_path}", file=sys.stderr)
    print(f"[mujoco] Joints    : {[JOINT_NAMES_MJCF[i] for i in joint_sequence]}", file=sys.stderr)
    print(f"[mujoco] Step amp  : {args.step_amp:+.3f} rad  hold={args.hold_steps}  step={args.step_steps}", file=sys.stderr)

    # ── Output file ───────────────────────────────────────────────────────────
    if args.out:
        out_file = open(args.out, "w")
    elif args.viewer:
        out_file = None
    else:
        out_file = sys.stdout

    # ── Viewer mode ───────────────────────────────────────────────────────────
    if args.viewer:
        reset_sim(model, data, actuator_net)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            first = True
            for ji in joint_sequence:
                if not viewer.is_running():
                    break
                jname = JOINT_NAMES_MJCF[ji]
                print(f"\n[mujoco] ── Joint [{ji}] {jname}  "
                      f"(default: {DEFAULT_JOINT_POS_MJCF[ji]:+.3f} → "
                      f"target: {DEFAULT_JOINT_POS_MJCF[ji]+args.step_amp:+.3f}) ──",
                      file=sys.stderr)
                reset_sim(model, data, actuator_net)
                ok = run_steps(model, data, actuator_net, ji, args.step_amp,
                               args.hold_steps, args.step_steps,
                               viewer=viewer, out_file=out_file,
                               no_header=(args.no_header or not first))
                first = False
                if not ok:
                    break

        if out_file and out_file is not sys.stdout:
            out_file.close()
        return

    # ── Headless / CSV mode ───────────────────────────────────────────────────
    first = True
    for ji in joint_sequence:
        reset_sim(model, data, actuator_net)
        run_steps(model, data, actuator_net, ji, args.step_amp,
                  args.hold_steps, args.step_steps,
                  viewer=None, out_file=out_file,
                  no_header=(args.no_header or not first))
        first = False

    if out_file and out_file is not sys.stdout:
        out_file.close()


if __name__ == "__main__":
    main()
