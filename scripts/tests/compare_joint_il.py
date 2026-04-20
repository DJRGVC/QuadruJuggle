"""Run a step command on one joint in Isaac Lab and plot the response.

Usage:
    python scripts/tests/compare_joint_il.py --headless           # joint 1 (FR_hip)
    python scripts/tests/compare_joint_il.py --joint 1 --headless
    python scripts/tests/compare_joint_il.py --joint 1 --out out.png --headless
"""
import argparse, os, sys

from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser()
ap.add_argument("--joint", type=int, default=1)
ap.add_argument("--out",   default=None)
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
if not hasattr(args, "headless") or args.headless is None:
    args.headless = True

app_launcher = AppLauncher(args)
app = app_launcher.app

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg") if args.headless else None
import matplotlib.pyplot as plt

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG

SIM_DT  = 1/200
HOLD    = 50
STEPS   = 150
AMP     = 0.3
WARMUP  = 50

_CFG = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
_CFG.init_state.pos = (0.0, 0.0, 0.35)

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground    = AssetBaseCfg(prim_path="/World/ground",     spawn=sim_utils.GroundPlaneCfg())
    dome      = AssetBaseCfg(prim_path="/World/DomeLight",  spawn=sim_utils.DomeLightCfg(intensity=500.0))
    robot: ArticulationCfg = _CFG

def main():
    sim = SimulationContext(SimulationCfg(dt=SIM_DT, render_interval=4))
    sim.set_camera_view(eye=[2,2,1.5], target=[0,0,0.4])
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset(); scene.update(SIM_DT)

    robot   = scene["robot"]
    default = robot.data.default_joint_pos.clone()
    jname   = robot.joint_names[args.joint]

    print(f"[IL] joint {args.joint} = {jname}", file=sys.stderr)
    

    # warmup
    for _ in range(WARMUP):
        robot.set_joint_position_target(default)
        scene.write_data_to_sim(); sim.step(render=False); scene.update(SIM_DT)

    # run
    targets   = default.clone()
    pos_log, tgt_log = [], []
    for step in range(HOLD + STEPS):
        if step >= HOLD:
            targets[0, args.joint] = default[0, args.joint] + AMP
        pos = robot.data.joint_pos[0, args.joint].item()
        pos_log.append(pos)
        tgt_log.append(targets[0, args.joint].item())
        robot.set_joint_position_target(targets)
        scene.write_data_to_sim(); sim.step(render=False); scene.update(SIM_DT)


    t = np.arange(len(pos_log)) * SIM_DT
    plt.figure(figsize=(8,4))
    plt.plot(t, tgt_log, "k--", label="target")
    plt.plot(t, pos_log, label="Isaac Lab")
    plt.axvline(HOLD*SIM_DT, color="gray", lw=0.8)
    plt.xlabel("time (s)"); plt.ylabel("angle (rad)")
    plt.title(f"Isaac Lab  {jname}  (idx={args.joint})")
    plt.legend(); plt.tight_layout()
    out = args.out or f"tests_out/il_joint{args.joint}.png"
    plt.savefig(out, dpi=120)
    print(f"saved → {out}")
    app.close()

if __name__ == "__main__":
    main()
