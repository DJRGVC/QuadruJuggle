"""Step all 12 joints simultaneously in Isaac Lab and MuJoCo, plot 4×3 grid.

Joint pairing is derived at runtime from the MuJoCo XML via build_reindex —
no hardcoded index lists.

Usage:
    conda run -n isaaclab env PYTHONPATH="/home/frank/IsaacLab/scripts/reinforcement_learning/rsl_rl:$PYTHONPATH" \
        python scripts/tests/compare_joint.py --headless
"""
import argparse, os, sys

from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser()
ap.add_argument("--out", default="tests_out/compare_joint.png")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
if not getattr(args, "headless", None):
    args.headless = True

app_launcher = AppLauncher(args)
app = app_launcher.app

# ── imports after AppLauncher ─────────────────────────────────────────────────
import numpy as np
import torch
import mujoco
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mujoco_utils import build_reindex

# ── shared config ─────────────────────────────────────────────────────────────
SIM_DT = 1/200
HOLD   = 50
STEPS  = 150
AMP    = 0.3
WARMUP = 200

DEFAULT_JOINT_POS_ISAAC = np.array([
     0.1, -0.1,  0.1, -0.1,
     0.8,  0.8,  1.0,  1.0,
    -1.5, -1.5, -1.5, -1.5,
])
NET_PATH = os.path.normpath(os.path.join(os.path.dirname(__file__),
    "../../../walk-these-ways/resources/actuator_nets/unitree_go1.pt"))

# Build reindex once from the XML so joint pairing is never hardcoded
_xml = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../mujoco/go1_juggle.xml"))
_model_tmp = mujoco.MjModel.from_xml_path(_xml)
REINDEX = build_reindex(_model_tmp)   # shape (12,): REINDEX[mjcf_i] = isaac_i
del _model_tmp

# Derive joint names in MJCF order directly from the XML model
_model_tmp2 = mujoco.MjModel.from_xml_path(_xml)
MJCF_JOINT_NAMES = [
    mujoco.mj_id2name(_model_tmp2, mujoco.mjtObj.mjOBJ_JOINT,
                      _model_tmp2.actuator_trnid[i, 0])
    for i in range(_model_tmp2.nu)
]
del _model_tmp2


# ── MuJoCo actuator net ───────────────────────────────────────────────────────
class ActNet:
    EFFORT = 23.7
    def __init__(self, reindex):
        self.net = torch.jit.load(NET_PATH, map_location="cpu").eval()
        self._ph = np.zeros((3, 12), np.float32)
        self._vh = np.zeros((3, 12), np.float32)
        self._reindex = reindex
    def reset(self): self._ph[:] = 0; self._vh[:] = 0
    def __call__(self, tgt_mjcf, pos_mjcf, vel_mjcf):
        e_mjcf = tgt_mjcf - pos_mjcf
        e_isaac = np.zeros(12, np.float32); e_isaac[self._reindex] = e_mjcf
        v_isaac = np.zeros(12, np.float32); v_isaac[self._reindex] = vel_mjcf
        self._ph = np.roll(self._ph, 1, 0); self._ph[0] = e_isaac
        self._vh = np.roll(self._vh, 1, 0); self._vh[0] = v_isaac
        x = np.concatenate([-self._ph.T, self._vh.T], 1)
        with torch.no_grad():
            t_isaac = self.net(torch.from_numpy(x)).squeeze(-1).numpy()
        return np.clip(t_isaac[self._reindex], -self.EFFORT, self.EFFORT)


# ── Isaac Lab scene (must be module-level for @configclass) ───────────────────
_robot_cfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
_robot_cfg.init_state.pos = (0.0, 0.0, 1.0)
_robot_cfg.spawn.articulation_props.fix_root_link = True

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground",    spawn=sim_utils.GroundPlaneCfg())
    dome   = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(intensity=500.0))
    robot: ArticulationCfg = _robot_cfg


# ── Isaac Lab run — records all 12 joints at once ────────────────────────────
def run_isaaclab():
    sim = SimulationContext(SimulationCfg(dt=SIM_DT, render_interval=4))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset(); scene.update(SIM_DT)
    robot   = scene["robot"]
    default = robot.data.default_joint_pos.clone()   # (1, 12) in Isaac order

    print(f"[IL] joint names: {robot.joint_names}", file=sys.stderr, flush=True)

    for _ in range(WARMUP):
        robot.set_joint_position_target(default)
        scene.write_data_to_sim(); sim.step(render=False); scene.update(SIM_DT)

    targets = default.clone()
    # pos_logs[i] = time series for Isaac joint i
    pos_logs = [[] for _ in range(12)]
    tgt_logs = [[] for _ in range(12)]

    for step in range(HOLD + STEPS):
        if step == HOLD:
            targets[0, :] = default[0, :] + AMP
        for i in range(12):
            pos_logs[i].append(robot.data.joint_pos[0, i].item())
            tgt_logs[i].append(targets[0, i].item())
        robot.set_joint_position_target(targets)
        scene.write_data_to_sim(); sim.step(render=False); scene.update(SIM_DT)

    # Return arrays indexed by Isaac joint index
    return (
        robot.joint_names,
        [np.array(pos_logs[i]) for i in range(12)],
        [np.array(tgt_logs[i]) for i in range(12)],
    )


# ── MuJoCo run — records all 12 joints at once ───────────────────────────────
def run_mujoco():
    model = mujoco.MjModel.from_xml_path(_xml)
    data  = mujoco.MjData(model)
    # reindex[mjcf_i] = isaac_i  →  use REINDEX computed at module load
    default_mjcf = DEFAULT_JOINT_POS_ISAAC[REINDEX]   # default in MJCF order
    net = ActNet(REINDEX)

    BASE_POS  = np.array([0, 0, 1.0])
    BASE_QUAT = np.array([1, 0, 0, 0])

    def pin_base():
        data.qpos[0:3] = BASE_POS
        data.qpos[3:7] = BASE_QUAT
        data.qvel[0:6] = 0.0

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = BASE_POS; data.qpos[3:7] = BASE_QUAT
    data.qpos[7:19] = default_mjcf; mujoco.mj_forward(model, data)
    net.reset()
    for _ in range(WARMUP):
        pin_base()
        data.ctrl[:] = net(default_mjcf, data.qpos[7:19], data.qvel[6:18])
        mujoco.mj_step(model, data)

    # pos_logs[i] = time series for MJCF joint i
    pos_logs = [[] for _ in range(12)]
    tgt_logs = [[] for _ in range(12)]

    for step in range(HOLD + STEPS):
        pin_base()
        tgt = default_mjcf.copy()
        if step >= HOLD:
            tgt[:] += AMP
        data.ctrl[:] = net(tgt, data.qpos[7:19], data.qvel[6:18])
        mujoco.mj_step(model, data)
        for i in range(12):
            pos_logs[i].append(data.qpos[7 + i])
            tgt_logs[i].append(tgt[i])

    # Return arrays indexed by MJCF joint index
    return (
        MJCF_JOINT_NAMES,
        [np.array(pos_logs[i]) for i in range(12)],
        [np.array(tgt_logs[i]) for i in range(12)],
    )


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("[IL] Running Isaac Lab...", file=sys.stderr, flush=True)
    _, il_pos, il_tgt = run_isaaclab()

    print("[MJ] Running MuJoCo...", file=sys.stderr, flush=True)
    mj_names, mj_pos, _ = run_mujoco()

    t = np.arange(HOLD + STEPS) * SIM_DT

    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharey=False)
    axes = axes.flatten()

    # REINDEX[mjcf_i] = isaac_i → iterate MJCF order, look up Isaac index for overlay
    for mjcf_i in range(12):
        isaac_i = REINDEX[mjcf_i]
        ax = axes[mjcf_i]
        ax.plot(t, il_tgt[isaac_i], "k--", lw=1.0, label="target")
        ax.plot(t, il_pos[isaac_i], label="Isaac Lab")
        ax.plot(t, mj_pos[mjcf_i],  label="MuJoCo")
        ax.axvline(HOLD * SIM_DT, color="gray", lw=0.6, ls=":")
        ax.set_title(mj_names[mjcf_i], fontsize=9)
        ax.set_xlabel("t (s)", fontsize=7)
        ax.set_ylabel("rad", fontsize=7)
        ax.tick_params(labelsize=7)
        if mjcf_i == 0:
            ax.legend(fontsize=7)

    fig.suptitle("Joint step response — all 12 joints (Isaac Lab vs MuJoCo)", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=130)
    print(f"saved → {args.out}")
    app.close()

if __name__ == "__main__":
    main()
