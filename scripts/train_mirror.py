"""Train a small NN controller for mirror-law paddle juggling.

Uses OpenAI-ES with parallel environments — one env per population member.
All candidates are evaluated simultaneously in a single sim.step() call.

Usage:
    uv run --active python scripts/train_mirror.py --headless
    uv run --active python scripts/train_mirror.py --headless --target-height 0.50
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train mirror-law NN controller")
parser.add_argument("--target-height", type=float, default=0.30)
parser.add_argument("--generations", type=int, default=50)
parser.add_argument("--pop-size", type=int, default=64)
parser.add_argument("--ep-steps", type=int, default=1500)
parser.add_argument("--sigma", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=0.03)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import os
import time
import numpy as np
import torch
import torch.nn as nn

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
MAX_TILT = 0.30
SURFACE_Z = PADDLE_REST_Z + PADDLE_HALF_THICKNESS + BALL_RADIUS


@configclass
class MirrorSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    paddle = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Paddle",
        spawn=sim_utils.CylinderCfg(
            radius=PADDLE_RADIUS, height=PADDLE_HALF_THICKNESS * 2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.15),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                restitution=0.5, static_friction=0.3, dynamic_friction=0.3,
            ),
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
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, SURFACE_Z + 0.08)),
    )


# ---------------------------------------------------------------------------
# NN Controller (batched — runs all envs at once)
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
        return torch.cat([
            raw[:, 0:1] * 0.15,      # z_offset ±15cm
            raw[:, 1:2] * MAX_TILT,   # roll ±0.30 rad
            raw[:, 2:3] * MAX_TILT,   # pitch ±0.30 rad
        ], dim=-1)

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([p.data.cpu().numpy().ravel() for p in self.parameters()])

    def set_flat_params(self, flat: np.ndarray):
        idx = 0
        for p in self.parameters():
            n = p.numel()
            p.data.copy_(torch.from_numpy(flat[idx:idx + n]).reshape(p.shape))
            idx += n

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Parallel population of controllers
# ---------------------------------------------------------------------------
class PopulationControllers:
    """Maintains pop_size copies of PaddleController with different weights.

    Uses a single batched forward pass by stacking weight matrices.
    For simplicity, we just loop over controllers (tiny NN, negligible cost).
    """

    def __init__(self, pop_size: int, device: str):
        self.controllers = [PaddleController().to(device) for _ in range(pop_size)]
        self.pop_size = pop_size

    def set_params(self, idx: int, flat: np.ndarray):
        self.controllers[idx].set_flat_params(flat)

    def forward_all(self, obs_per_env: torch.Tensor) -> torch.Tensor:
        """obs_per_env: (pop_size, 6) → returns (pop_size, 3)."""
        results = []
        with torch.no_grad():
            for i in range(self.pop_size):
                results.append(self.controllers[i](obs_per_env[i:i+1]))
        return torch.cat(results, dim=0)


# ---------------------------------------------------------------------------
# Parallel reset & step
# ---------------------------------------------------------------------------
def reset_all_balls(ball, paddle, origins, N, dev):
    """Reset all N balls to random initial conditions above their paddles."""
    bx = torch.empty(N, device=dev).uniform_(-0.02, 0.02)
    by = torch.empty(N, device=dev).uniform_(-0.02, 0.02)
    bz = SURFACE_Z + torch.empty(N, device=dev).uniform_(0.05, 0.12)
    vx = torch.empty(N, device=dev).uniform_(-0.15, 0.15)
    vy = torch.empty(N, device=dev).uniform_(-0.15, 0.15)

    env_ids = torch.arange(N, device=dev)

    ball_pose = torch.zeros(N, 7, device=dev)
    ball_pose[:, 0] = bx
    ball_pose[:, 1] = by
    ball_pose[:, 2] = bz
    ball_pose[:, 3] = 1.0  # quat_w
    ball_pose[:, :3] += origins[:, :3]
    ball_vel = torch.zeros(N, 6, device=dev)
    ball_vel[:, 0] = vx
    ball_vel[:, 1] = vy
    ball.write_root_pose_to_sim(ball_pose, env_ids)
    ball.write_root_velocity_to_sim(ball_vel, env_ids)

    paddle_pose = torch.zeros(N, 7, device=dev)
    paddle_pose[:, 2] = PADDLE_REST_Z
    paddle_pose[:, 3] = 1.0  # quat_w
    paddle_pose[:, :3] += origins[:, :3]
    paddle.write_root_pose_to_sim(paddle_pose, env_ids)


def run_generation(sim, ball, paddle, origins, pop_controllers, target_h, max_steps, dev):
    """Run one generation: all pop_size envs in parallel. Returns fitness array."""
    N = pop_controllers.pop_size

    reset_all_balls(ball, paddle, origins, N, dev)
    sim.step()
    ball.update(DT)
    paddle.update(DT)

    # Per-env tracking
    alive = torch.ones(N, dtype=torch.bool, device=dev)
    steps_alive = torch.zeros(N, device=dev)
    apex_bonus = torch.zeros(N, device=dev)
    centering_reward = torch.zeros(N, device=dev)
    prev_vz = torch.zeros(N, device=dev)
    max_z = torch.zeros(N, device=dev)

    for step in range(max_steps):
        bp = ball.data.root_pos_w[:, :3] - origins  # (N, 3)
        bv = ball.data.root_lin_vel_w[:, :3]        # (N, 3)
        pp = paddle.data.root_pos_w[:, :3] - origins  # (N, 3)

        ball_z = bp[:, 2]
        ball_vz = bv[:, 2]

        # Termination check
        fell = (ball_z < 0.2) | (bp[:, 0].abs() > 0.4) | (bp[:, 1].abs() > 0.4)
        alive &= ~fell

        if not alive.any():
            break

        # Accumulate per-step rewards (only for alive envs)
        steps_alive += alive.float()

        xy_dist = ((bp[:, :2] - pp[:, :2]).pow(2).sum(dim=1)).sqrt()
        centering_reward += alive.float() * (1.0 - xy_dist / 0.1).clamp(min=0.0)

        # Apex detection
        max_z = torch.max(max_z, ball_z)
        apex_mask = alive & (prev_vz > 0) & (ball_vz <= 0)
        if apex_mask.any():
            apex_h = max_z[apex_mask] - SURFACE_Z
            apex_bonus[apex_mask] += torch.exp(-((apex_h - target_h) ** 2) / (2 * 0.1 ** 2))
            max_z[apex_mask] = 0.0
        prev_vz = ball_vz.clone()

        # Compute observations: relative ball pos + ball vel
        obs = torch.cat([bp - pp, bv], dim=1)  # (N, 6)

        # Get commands from all controllers
        cmds = pop_controllers.forward_all(obs)  # (N, 3)

        # Apply commands to paddles
        target_z = (PADDLE_REST_Z + cmds[:, 0]).clamp(0.25, 0.75)
        roll_cmd = cmds[:, 1]
        pitch_cmd = cmds[:, 2]

        quat = math_utils.quat_from_euler_xyz(roll_cmd, pitch_cmd, torch.zeros(N, device=dev))
        new_pos = torch.zeros(N, 3, device=dev)
        new_pos[:, 2] = target_z
        new_pos += origins
        paddle_pose = torch.cat([new_pos, quat], dim=-1)

        # Only write alive envs to avoid disturbing terminated paddles
        alive_ids = torch.where(alive)[0]
        if len(alive_ids) > 0:
            paddle.write_root_pose_to_sim(paddle_pose[alive_ids], alive_ids)

        sim.step()
        ball.update(DT)
        paddle.update(DT)

    fitness = steps_alive + 0.5 * centering_reward + 50.0 * apex_bonus
    return fitness.cpu().numpy()


# ---------------------------------------------------------------------------
# Sanity check (quick, using env 0 only)
# ---------------------------------------------------------------------------
def sanity_check(sim, ball, paddle, origins, N, dev):
    print("\n--- Sanity check (parallel envs) ---")
    print(f"  num_envs = {N}")
    print(f"  origins[0] = {origins[0]}")

    env_ids = torch.arange(N, device=dev)

    # Write all balls to z=0.8
    ball_pose = torch.zeros(N, 7, device=dev)
    ball_pose[:, 2] = 0.8
    ball_pose[:, 3] = 1.0
    ball_pose[:, :3] += origins
    ball.write_root_pose_to_sim(ball_pose, env_ids)
    sim.step()
    ball.update(DT)

    bp = ball.data.root_pos_w - origins
    print(f"  after write z=0.8: ball[0] z={bp[0, 2].item():.4f}, ball[-1] z={bp[-1, 2].item():.4f}")

    if abs(bp[0, 2].item() - 0.8) > 0.05:
        print("  FAIL: ball position write didn't work!")
        return False

    # Let balls fall for 100 steps
    paddle_pose = torch.zeros(N, 7, device=dev)
    paddle_pose[:, 2] = PADDLE_REST_Z
    paddle_pose[:, 3] = 1.0
    paddle_pose[:, :3] += origins
    paddle.write_root_pose_to_sim(paddle_pose, env_ids)

    bounced = False
    for step in range(100):
        sim.step()
        ball.update(DT)
        bv = ball.data.root_lin_vel_w[0]
        bp0 = ball.data.root_pos_w[0] - origins[0]
        if step % 25 == 0:
            print(f"    step {step:3d}  ball[0] z={bp0[2].item():.4f}  vz={bv[2].item():.3f}")
        if bv[2].item() > 0.1:
            bounced = True

    if bounced:
        print("  OK: ball bounced!\n")
    else:
        print("  WARN: ball didn't bounce but survived 100 steps\n")
    return True


# ---------------------------------------------------------------------------
# ES
# ---------------------------------------------------------------------------
def es_step(fitnesses, noise_seeds, sigma, lr, base_params, n_params):
    fitnesses = np.array(fitnesses)
    n = len(fitnesses)
    sorted_idx = np.argsort(fitnesses)
    ranks = np.zeros(n)
    for i, idx in enumerate(sorted_idx):
        ranks[idx] = i
    ranks = (ranks - (n - 1) / 2.0) / ((n - 1) / 2.0 + 1e-8)

    grad = np.zeros(n_params)
    for i in range(n):
        noise = np.random.RandomState(noise_seeds[i]).randn(n_params)
        grad += ranks[i] * noise
    grad /= (n * sigma)
    return base_params + lr * grad


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    target_h = args_cli.target_height
    pop_size = args_cli.pop_size
    n_gens = args_cli.generations
    ep_steps = args_cli.ep_steps
    sigma = args_cli.sigma
    lr = args_cli.lr

    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT, render_interval=ep_steps + 1))
    scene = InteractiveScene(MirrorSceneCfg(num_envs=pop_size, env_spacing=2.0))
    sim.reset()

    ball = scene["ball"]
    paddle = scene["paddle"]
    origins = scene.env_origins  # (pop_size, 3)
    dev = sim.device

    # Warmup
    sim.step()
    ball.update(DT)
    paddle.update(DT)

    if not sanity_check(sim, ball, paddle, origins, pop_size, dev):
        print("Physics sanity check failed!")
        return

    # Create population of controllers
    pop_ctrl = PopulationControllers(pop_size, dev)
    ref_controller = PaddleController().to(dev)
    n_params = ref_controller.num_params()
    base_params = ref_controller.get_flat_params()

    print(f"\n{'='*60}")
    print(f"  ES Training — Mirror-law NN Controller (PARALLEL)")
    print(f"  Target height: {target_h:.2f} m  |  NN params: {n_params}")
    print(f"  Pop: {pop_size}  Gens: {n_gens}  Sigma: {sigma}  LR: {lr}")
    print(f"  Max steps/episode: {ep_steps}")
    print(f"  Envs: {pop_size} (1 per candidate, all parallel)")
    print(f"{'='*60}\n")

    best_ever_fitness = -float("inf")
    best_ever_params = base_params.copy()
    gen_times = []

    for gen in range(n_gens):
        t0 = time.time()

        # Generate perturbed params and load into controllers
        noise_seeds = []
        for i in range(pop_size):
            seed = np.random.randint(0, 2**31)
            noise = np.random.RandomState(seed).randn(n_params)
            noise_seeds.append(seed)
            pop_ctrl.set_params(i, base_params + sigma * noise)

        # Run all envs in parallel for ep_steps
        fitnesses = run_generation(sim, ball, paddle, origins, pop_ctrl, target_h, ep_steps, dev)

        # ES update
        base_params = es_step(fitnesses, noise_seeds, sigma, lr, base_params, n_params)

        gen_best = fitnesses.max()
        gen_mean = fitnesses.mean()
        gen_median = np.median(fitnesses)
        if gen_best > best_ever_fitness:
            best_ever_fitness = gen_best
            best_ever_params = base_params.copy()

        elapsed = time.time() - t0
        gen_times.append(elapsed)
        eta_total = np.mean(gen_times) * (n_gens - gen - 1)
        eta_min, eta_sec = int(eta_total // 60), int(eta_total % 60)

        # Compute alive stats
        alive_steps = fitnesses  # rough proxy
        max_alive = int(alive_steps.max())

        print(
            f"[Gen {gen+1:3d}/{n_gens}]  "
            f"mean={gen_mean:7.1f}  median={gen_median:7.1f}  best={gen_best:7.1f}  "
            f"best_ever={best_ever_fitness:7.1f}  "
            f"({elapsed:.1f}s)  ETA: {eta_min}m{eta_sec:02d}s"
        )

    # Save best
    ref_controller.set_flat_params(best_ever_params)
    save_dir = "logs/mirror_law"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_controller.pt")
    torch.save({
        "state_dict": ref_controller.state_dict(),
        "flat_params": best_ever_params,
        "fitness": best_ever_fitness,
        "target_height": target_h,
    }, save_path)
    print(f"\nSaved to: {save_path}  (fitness={best_ever_fitness:.1f})")


if __name__ == "__main__":
    main()
    simulation_app.close()
