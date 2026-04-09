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

    # Step the environment to let the ball drop, then give it upward velocity
    # so it's in the air (visible to the 75° tilted camera) during capture.
    print(f"[debug_capture] Stepping {args_cli.steps} steps to let ball settle...", flush=True)
    unwrapped = env.unwrapped
    print(f"[debug_capture] Action space: {unwrapped.action_space.shape}, device: {unwrapped.device}", flush=True)

    settle_steps = min(args_cli.steps, 50)
    for i in range(settle_steps):
        try:
            action = torch.zeros(unwrapped.action_space.shape, device=unwrapped.device)
            obs, _, _, _, _ = env.step(action)
            if i == 0:
                print(f"[debug_capture] First step OK, obs keys: {list(obs.keys()) if isinstance(obs, dict) else obs.shape}", flush=True)
        except Exception as e:
            print(f"[debug_capture] ERROR at step {i}: {e}", flush=True)
            import traceback; traceback.print_exc()
            break

    # Give ball upward velocity so it's airborne during capture
    try:
        ball = unwrapped.scene["ball"]
        ball_vel = ball.data.root_vel_w.clone()  # (N, 6): [vx, vy, vz, wx, wy, wz]
        ball_vel[:, 2] = 3.0  # 3 m/s upward — reaches ~0.46 m apex in ~0.3 s
        ball.write_root_velocity_to_sim(ball_vel)
        print("[debug_capture] Applied 3 m/s upward velocity to ball.", flush=True)
    except Exception as e:
        print(f"[debug_capture] WARNING: could not set ball velocity: {e}", flush=True)

    # Step only 5-8 steps (~0.1-0.16 s at 50 Hz) so ball is still rising / near apex.
    # At 3 m/s up, apex is at t≈0.31s, h≈0.46m above paddle. Capture at ~0.1s → h≈0.25m.
    capture_delay = 8
    for i in range(capture_delay):
        action = torch.zeros(unwrapped.action_space.shape, device=unwrapped.device)
        obs, _, _, _, _ = env.step(action)

    # Report ball position for debugging
    try:
        ball = unwrapped.scene["ball"]
        ball_pos_w = ball.data.root_pos_w[0].cpu().numpy()
        print(f"[debug_capture] Ball world pos at capture: ({ball_pos_w[0]:.3f}, {ball_pos_w[1]:.3f}, {ball_pos_w[2]:.3f})", flush=True)
    except Exception:
        pass

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

    # Force multiple camera updates to ensure rendering catches up
    for _ in range(5):
        cam.update(dt=unwrapped.step_dt)
    print("[debug_capture] Camera updated (5x).", flush=True)

    # Print camera world pose for debugging
    try:
        cam_pos_w = cam.data.pos_w[0].cpu().numpy()
        cam_quat_w = cam.data.quat_w_ros[0].cpu().numpy() if hasattr(cam.data, 'quat_w_ros') else cam.data.quat_w[0].cpu().numpy()
        print(f"[debug_capture] Camera world pos: ({cam_pos_w[0]:.4f}, {cam_pos_w[1]:.4f}, {cam_pos_w[2]:.4f})", flush=True)
        print(f"[debug_capture] Camera world quat: ({cam_quat_w[0]:.4f}, {cam_quat_w[1]:.4f}, {cam_quat_w[2]:.4f}, {cam_quat_w[3]:.4f})", flush=True)
        # Compute look direction from quaternion
        import numpy as np_diag
        w, x, y, z = cam_quat_w
        # Forward direction (Z axis in ROS or camera convention)
        fwd_x = 2 * (x * z + w * y)
        fwd_y = 2 * (y * z - w * x)
        fwd_z = 1 - 2 * (x * x + y * y)
        pitch_deg = np_diag.degrees(np_diag.arcsin(np_diag.clip(fwd_z, -1, 1)))
        print(f"[debug_capture] Camera forward: ({fwd_x:.3f}, {fwd_y:.3f}, {fwd_z:.3f}), pitch ≈ {pitch_deg:.1f}°", flush=True)
    except Exception as e:
        print(f"[debug_capture] Could not read camera pose: {e}", flush=True)

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
