"""Run a step command on one joint in MuJoCo and plot the response.

Usage:
    python scripts/tests/compare_joint_mj.py             # FR_hip, viewer off
    python scripts/tests/compare_joint_mj.py --joint 0   # MJCF joint index
    python scripts/tests/compare_joint_mj.py --out out.png
"""
import argparse, os, sys
import numpy as np
import matplotlib.pyplot as plt
import mujoco

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mujoco_utils import build_reindex

DEFAULT_JOINT_POS_ISAAC = np.array([
     0.1, -0.1,  0.1, -0.1,
     0.8,  0.8,  1.0,  1.0,
    -1.5, -1.5, -1.5, -1.5,
])
JOINT_NAMES_MJCF = [
    "FR_hip","FR_thigh","FR_calf",
    "FL_hip","FL_thigh","FL_calf",
    "RR_hip","RR_thigh","RR_calf",
    "RL_hip","RL_thigh","RL_calf",
]
SIM_DT   = 1/200
HOLD     = 50
STEPS    = 150
AMP      = 0.3
WARMUP   = 50

# Actuator net
import torch
NET_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__),
    "../../../walk-these-ways/resources/actuator_nets/unitree_go1.pt"))

class ActNet:
    EFFORT = 23.7
    def __init__(self):
        self.net = torch.jit.load(NET_PATH, map_location="cpu").eval()
        self._ph = np.zeros((3,12), np.float32)
        self._vh = np.zeros((3,12), np.float32)
    def reset(self): self._ph[:]=0; self._vh[:]=0
    def __call__(self, tgt, pos, vel):
        e = tgt - pos
        self._ph = np.roll(self._ph, 1, 0); self._ph[0] = e
        self._vh = np.roll(self._vh, 1, 0); self._vh[0] = vel
        x = np.concatenate([-self._ph.T, self._vh.T], 1)  # (12,6)
        with torch.no_grad():
            t = self.net(torch.from_numpy(x)).squeeze(-1).numpy()
        return np.clip(t, -self.EFFORT, self.EFFORT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", type=int, default=0)
    ap.add_argument("--xml", default=os.path.join(os.path.dirname(__file__),
                                                   "../../mujoco/go1_juggle.xml"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(os.path.abspath(args.xml))
    data  = mujoco.MjData(model)
    reindex = build_reindex(model)
    default = DEFAULT_JOINT_POS_ISAAC[reindex]

    net = ActNet()
    jname = JOINT_NAMES_MJCF[args.joint]

    # reset + warmup
    mujoco.mj_resetData(model, data)
    data.qpos[0:3]  = [0,0,0.35]
    data.qpos[3:7]  = [1,0,0,0]
    data.qpos[7:19] = default
    mujoco.mj_forward(model, data)
    net.reset()
    for _ in range(WARMUP):
        data.ctrl[:] = net(default, data.qpos[7:19], data.qvel[6:18])
        mujoco.mj_step(model, data)

    # run
    pos_log, tgt_log = [], []
    for step in range(HOLD + STEPS):
        tgt = default.copy()
        if step >= HOLD:
            tgt[args.joint] += AMP
        data.ctrl[:] = net(tgt, data.qpos[7:19], data.qvel[6:18])
        mujoco.mj_step(model, data)
        pos_log.append(data.qpos[7+args.joint])
        tgt_log.append(tgt[args.joint])

    t = np.arange(len(pos_log)) * SIM_DT
    plt.figure(figsize=(8,4))
    plt.plot(t, tgt_log, "k--", label="target")
    plt.plot(t, pos_log, label="MuJoCo")
    plt.axvline(HOLD*SIM_DT, color="gray", lw=0.8)
    plt.xlabel("time (s)"); plt.ylabel("angle (rad)")
    plt.title(f"MuJoCo  {jname}  (MJCF idx={args.joint})")
    plt.legend(); plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=120)
        print(f"saved → {args.out}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
