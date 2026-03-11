"""Minimal physics test — diagnose ball/paddle state initialization.

Usage:
    uv run --active python scripts/test_physics.py --headless
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObject
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

DT = 1.0 / 200.0

@configclass
class TestSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    paddle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Paddle",
        spawn=sim_utils.CylinderCfg(
            radius=0.085, height=0.01,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
    )
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.02,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, linear_damping=0.1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.0027),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=0.85, restitution_combine_mode="max",
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.7)),
    )


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT))
    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    ball = scene["ball"]
    paddle = scene["paddle"]

    print(f"\n1. After InteractiveScene created (before sim.reset)")
    print(f"   env_origins = {scene.env_origins}")

    sim.reset()
    print(f"\n2. After sim.reset() (before scene.reset)")
    ball.update(DT)
    paddle.update(DT)
    print(f"   ball  pos_w = {ball.data.root_pos_w}")
    print(f"   paddle pos_w = {paddle.data.root_pos_w}")

    scene.reset()
    print(f"\n3. After scene.reset() (before any sim.step)")
    ball.update(DT)
    paddle.update(DT)
    print(f"   ball  pos_w = {ball.data.root_pos_w}")
    print(f"   paddle pos_w = {paddle.data.root_pos_w}")

    sim.step()
    print(f"\n4. After first sim.step()")
    ball.update(DT)
    paddle.update(DT)
    print(f"   ball  pos_w = {ball.data.root_pos_w}")
    print(f"   paddle pos_w = {paddle.data.root_pos_w}")

    sim.step()
    print(f"\n5. After second sim.step()")
    ball.update(DT)
    paddle.update(DT)
    print(f"   ball  pos_w = {ball.data.root_pos_w}")
    print(f"   paddle pos_w = {paddle.data.root_pos_w}")

    # Try explicit write
    origins = scene.env_origins
    print(f"\n6. Writing ball to (0, 0, 0.8) + origins")
    ball_pose = torch.tensor([[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]], device=sim.device)
    ball_pose[:, :3] += origins
    ball.write_root_pose_to_sim(ball_pose)
    sim.step()
    ball.update(DT)
    print(f"   ball  pos_w = {ball.data.root_pos_w}")

    # Run 50 steps and watch ball fall
    print(f"\n7. Running 50 steps (ball should fall under gravity):")
    for i in range(50):
        sim.step()
        ball.update(DT)
        if i % 10 == 0:
            bp = ball.data.root_pos_w[0]
            print(f"   step {i:3d}  ball_z = {bp[2].item():.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
    simulation_app.close()
