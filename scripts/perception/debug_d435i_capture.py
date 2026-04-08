"""Debug script: mount simulated D435i on Go1, drop a ball, save RGB + depth frames.

Usage (from project root):
    uv run --active python scripts/perception/debug_d435i_capture.py \
        --task Isaac-BallJuggleHier-Go1-Play-v0 --num_envs 1 --headless --enable_cameras

Saves frames to: source/go1_ball_balance/go1_ball_balance/perception/debug/
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="D435i debug capture — save RGB + depth from simulated camera.")
parser.add_argument("--task", type=str, default="Isaac-BallJuggleHier-Go1-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=100, help="Sim steps before capture (let ball drop).")
# Strip --pi2-checkpoint before AppLauncher sees it
_pi2_checkpoint_path = None
_clean_argv = []
_i = 0
while _i < len(sys.argv):
    if sys.argv[_i] in ("--pi2-checkpoint", "--pi2_checkpoint") and _i + 1 < len(sys.argv):
        _pi2_checkpoint_path = sys.argv[_i + 1]
        _i += 2
    else:
        _clean_argv.append(sys.argv[_i])
        _i += 1
sys.argv = _clean_argv

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # required for TiledCamera

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- post-launch imports ---
import os

import numpy as np
import torch

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import go1_ball_balance  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg


def main():
    # If hierarchical task, set pi2 checkpoint
    if _pi2_checkpoint_path is not None:
        from go1_ball_balance.tasks.torso_tracking.action_term import TorsoCommandActionCfg
        TorsoCommandActionCfg._pi2_checkpoint_override = _pi2_checkpoint_path

    env = gym.make(args_cli.task, cfg_entry_point="env_cfg_entry_point")
    obs, _ = env.reset()

    # Step the environment to let the ball drop onto the paddle
    print(f"[debug_capture] Stepping {args_cli.steps} steps to let ball settle...")
    for i in range(args_cli.steps):
        action = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
        obs, _, _, _, _ = env.step(action)

    # Access the TiledCamera sensor
    unwrapped = env.unwrapped
    scene = unwrapped.scene
    if not hasattr(scene, "d435i"):
        print("[debug_capture] ERROR: scene has no 'd435i' camera. Is this a PLAY config?")
        env.close()
        simulation_app.close()
        return

    cam = scene["d435i"]
    print(f"[debug_capture] Camera shape: {cam.image_shape}, data types: {cam.cfg.data_types}")

    # Force a camera update
    cam.update(dt=unwrapped.step_dt)

    # Save output directory
    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "source", "go1_ball_balance", "go1_ball_balance", "perception", "debug",
    )
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Save RGB frame (env 0)
    if "rgb" in cam.data.output:
        rgb = cam.data.output["rgb"][0].cpu().numpy()  # (H, W, 3) or (H, W, 4)
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        # Scale to uint8 if needed
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
        from PIL import Image
        img = Image.fromarray(rgb)
        rgb_path = os.path.join(out_dir, "frame_000_rgb.png")
        img.save(rgb_path)
        print(f"[debug_capture] Saved RGB: {rgb_path}")

    # Save depth frame (env 0)
    depth_key = "distance_to_image_plane"
    if depth_key in cam.data.output:
        depth = cam.data.output[depth_key][0].cpu().numpy()  # (H, W, 1) or (H, W)
        if depth.ndim == 3:
            depth = depth[..., 0]
        # Normalise for visualization
        valid = np.isfinite(depth) & (depth > 0)
        if valid.any():
            d_min, d_max = depth[valid].min(), depth[valid].max()
            depth_vis = np.zeros_like(depth, dtype=np.uint8)
            depth_vis[valid] = ((depth[valid] - d_min) / max(d_max - d_min, 1e-6) * 255).astype(np.uint8)
        else:
            depth_vis = np.zeros_like(depth, dtype=np.uint8)
        from PIL import Image
        img = Image.fromarray(depth_vis)
        depth_path = os.path.join(out_dir, "frame_000_depth.png")
        img.save(depth_path)
        print(f"[debug_capture] Saved depth: {depth_path}  (range {d_min:.3f}–{d_max:.3f} m)")

        # Also save raw .npy
        np.save(os.path.join(out_dir, "frame_000_depth_raw.npy"), depth)
        print(f"[debug_capture] Saved raw depth: frame_000_depth_raw.npy")

    print("[debug_capture] Done.")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
