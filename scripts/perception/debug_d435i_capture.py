"""Debug script: mount simulated D435i on Go1, drop a ball, save RGB + depth frames.

Usage (from project root):
    uv run --active python scripts/perception/debug_d435i_capture.py \
        --task Isaac-BallJuggleHier-Go1-Play-v0 --num_envs 1 --headless --enable_cameras

Saves frames to: source/go1_ball_balance/go1_ball_balance/perception/debug/
"""

import argparse
import os
import sys

# Prepend our worktree's source dir so our version of go1_ball_balance is found first
# (the shared venv's editable install may point to a sibling agent's worktree)
_OUR_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "source", "go1_ball_balance"))
sys.path.insert(0, _OUR_SRC)

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

from go1_ball_balance.tasks.ball_juggle_hier.ball_juggle_hier_env_cfg import BallJuggleHierEnvCfg_PLAY


def main():
    # Build the env config directly (avoids Hydra wiring)
    env_cfg = BallJuggleHierEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs

    # Inject pi2 checkpoint for hierarchical tasks
    if hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "torso_cmd"):
        pi2_path = _pi2_checkpoint_path
        if pi2_path is None:
            # Auto-detect: find the latest torso-tracking checkpoint
            import glob
            candidates = sorted(glob.glob(
                os.path.join(os.path.dirname(__file__), "..", "..",
                             "logs", "rsl_rl", "go1_torso_tracking", "*", "model_best.pt")
            ))
            if not candidates:
                # Try the main QuadruJuggle repo
                candidates = sorted(glob.glob(
                    os.path.expanduser("~/Research/QuadruJuggle/logs/rsl_rl/go1_torso_tracking/*/model_best.pt")
                ))
            if candidates:
                pi2_path = candidates[-1]
                print(f"[debug_capture] Auto-detected pi2 checkpoint: {pi2_path}")
            else:
                print("[debug_capture] WARNING: no pi2 checkpoint found. Use --pi2-checkpoint <path>")
        if pi2_path is not None:
            env_cfg.actions.torso_cmd.pi2_checkpoint = os.path.abspath(pi2_path)
            print(f"[debug_capture] pi2 checkpoint: {env_cfg.actions.torso_cmd.pi2_checkpoint}")

    import sys
    # Verify our worktree's version of the config was loaded (with D435i camera)
    print(f"[debug_capture] Scene type: {type(env_cfg.scene).__name__}", flush=True)
    print(f"[debug_capture] Scene has d435i: {hasattr(env_cfg.scene, 'd435i')}", flush=True)
    if not hasattr(env_cfg.scene, 'd435i'):
        # Fallback: force DEBUG scene if __post_init__ override didn't take effect
        from go1_ball_balance.tasks.ball_juggle_hier.ball_juggle_hier_env_cfg import BallJuggleHierSceneCfg_DEBUG
        env_cfg.scene = BallJuggleHierSceneCfg_DEBUG(num_envs=args_cli.num_envs, env_spacing=3.5)
        print(f"[debug_capture] Forced DEBUG scene, has d435i: {hasattr(env_cfg.scene, 'd435i')}", flush=True)
    print("[debug_capture] Creating env...", flush=True)
    sys.stdout.flush()
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[debug_capture] Env created. Resetting...", flush=True)
    sys.stdout.flush()
    obs, _ = env.reset()
    print("[debug_capture] Env reset complete.", flush=True)
    sys.stdout.flush()

    # Step the environment to let the ball drop onto the paddle
    print(f"[debug_capture] Stepping {args_cli.steps} steps to let ball settle...", flush=True)
    unwrapped = env.unwrapped
    print(f"[debug_capture] Action space: {unwrapped.action_space.shape}, device: {unwrapped.device}", flush=True)
    for i in range(args_cli.steps):
        try:
            action = torch.zeros(unwrapped.action_space.shape, device=unwrapped.device)
            obs, _, _, _, _ = env.step(action)
            if i == 0:
                print(f"[debug_capture] First step OK, obs keys: {list(obs.keys()) if isinstance(obs, dict) else obs.shape}", flush=True)
        except Exception as e:
            print(f"[debug_capture] ERROR at step {i}: {e}", flush=True)
            import traceback; traceback.print_exc()
            break

    # Access the TiledCamera sensor
    print("[debug_capture] Accessing camera sensor...", flush=True)
    scene = unwrapped.scene
    print(f"[debug_capture] Scene entities: {list(scene.keys()) if hasattr(scene, 'keys') else dir(scene)}", flush=True)

    try:
        cam = scene["d435i"]
    except (KeyError, AttributeError) as e:
        print(f"[debug_capture] ERROR: no 'd435i' camera in scene: {e}", flush=True)
        env.close()
        simulation_app.close()
        return

    print(f"[debug_capture] Camera found: {cam}, data types: {cam.cfg.data_types}", flush=True)

    # Force a camera update
    cam.update(dt=unwrapped.step_dt)
    print("[debug_capture] Camera updated.", flush=True)

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
        if valid.any():
            print(f"[debug_capture] Saved depth: {depth_path}  (range {d_min:.3f}–{d_max:.3f} m)")
        else:
            print(f"[debug_capture] Saved depth: {depth_path}  (all invalid/zero — camera may have no objects in view)")

        # Also save raw .npy
        np.save(os.path.join(out_dir, "frame_000_depth_raw.npy"), depth)
        print(f"[debug_capture] Saved raw depth: frame_000_depth_raw.npy")

    print("[debug_capture] Done.")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
