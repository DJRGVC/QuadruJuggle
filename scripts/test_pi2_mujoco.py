"""Test pi2 in isolation in MuJoCo using the Go1 actuator net.

Robot should stand stably at ~0.375m with fixed neutral torso command.

Run:
    python scripts/test_pi2_mujoco.py --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt
"""

import argparse, sys, os
import numpy as np
import torch
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(__file__))
from play_mujoco import (
    load_policies, reset_sim, quat_to_rot,
    DEFAULT_JOINT_POS_ISAAC, DEFAULT_JOINT_POS_MJCF,
    REINDEX, CMD_OFFSETS, OBS_OFFSET, OBS_NORM,
    GoActuatorNet, ACTUATOR_NET_PATH,
)

parser = argparse.ArgumentParser()
parser.add_argument("--pi2_checkpoint", required=True)
parser.add_argument("--xml", type=str,
                    default=os.path.join(os.path.dirname(__file__), "../mujoco/go1_juggle.xml"))
args = parser.parse_args()

_, pi2 = load_policies(
    # pi1 not needed, just pass pi2 path twice
    args.pi2_checkpoint, args.pi2_checkpoint
)

model = mujoco.MjModel.from_xml_path(os.path.abspath(args.xml))
data  = mujoco.MjData(model)

actuator = GoActuatorNet(ACTUATOR_NET_PATH)

# Fixed neutral torso command: h=0.375, rest=0
torso_cmd = CMD_OFFSETS.copy()
torso_cmd_norm = (torso_cmd + OBS_OFFSET) * OBS_NORM
print(f"Torso cmd: {torso_cmd}")
print(f"Torso cmd norm: {torso_cmd_norm}")
print(f"Default joints (MJCF order): {np.round(DEFAULT_JOINT_POS_MJCF, 3)}")

def run(viewer=None):
    reset_sim(model, data)
    actuator.reset()
    mujoco.mj_forward(model, data)

    step = 0
    while True:
        R = quat_to_rot(data.body("trunk").xquat)
        lin_vel_b = R.T @ data.qvel[0:3]
        ang_vel_b = R.T @ data.qvel[3:6]
        grav_b    = R.T @ np.array([0., 0., -1.])

        # Convert MJCF joint data → Isaac order
        isaac_jp = data.qpos[7:19][REINDEX]
        isaac_jv = data.qvel[6:18][REINDEX]
        jp_rel   = isaac_jp - DEFAULT_JOINT_POS_ISAAC

        obs = np.concatenate([torso_cmd_norm, lin_vel_b, ang_vel_b, grav_b,
                               jp_rel, isaac_jv]).astype(np.float32)
        t = torch.from_numpy(obs).unsqueeze(0)
        with torch.no_grad():
            a = pi2(t).squeeze(0).numpy()

        # Targets in Isaac order → convert to MJCF
        isaac_targets = DEFAULT_JOINT_POS_ISAAC + 0.25 * a
        mjcf_targets  = isaac_targets[REINDEX]

        for _ in range(4):
            torques = actuator.compute(mjcf_targets,
                                       data.qpos[7:19].copy(),
                                       data.qvel[6:18].copy())
            data.ctrl[:] = torques
            mujoco.mj_step(model, data)

        if step % 50 == 0:
            trunk_z = data.body("trunk").xpos[2]
            tilt = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))
            print(f"  step {step:4d} | trunk_z={trunk_z:.3f} | tilt={tilt:.1f}deg "
                  f"| jp={np.round(data.qpos[7:19], 2)}")

        if viewer:
            viewer.sync()

        step += 1
        if step > 2000:
            break

with mujoco.viewer.launch_passive(model, data) as viewer:
    run(viewer)
