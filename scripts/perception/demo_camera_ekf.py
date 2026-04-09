"""Demo: camera → detect → EKF pipeline in Isaac Lab simulation.

Runs the hierarchical juggle env with a D435i camera, detects the ball
in each depth frame, feeds detections into the EKF, and saves annotated
frames showing GT vs detected vs EKF-estimated ball positions.

All positions are in world frame. Detection pipeline:
  depth image → SimBallDetector (camera frame) → cam-to-world transform → EKF (world frame)

Usage (from project root):
    uv run --active python scripts/perception/demo_camera_ekf.py \
        --task Isaac-BallJuggleHier-Go1-Play-v0 --num_envs 1 --headless --enable_cameras

Outputs to: source/go1_ball_balance/go1_ball_balance/perception/debug/demo/
"""

import argparse
import os
import sys

# Prepend our worktree's source dir
_OUR_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "source", "go1_ball_balance"))
sys.path.insert(0, _OUR_SRC)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Camera → Detect → EKF demo pipeline")
parser.add_argument("--task", type=str, default="Isaac-BallJuggleHier-Go1-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=200, help="Total sim steps to run.")
parser.add_argument("--capture_interval", type=int, default=10, help="Save annotated frame every N steps.")

# Strip --pi2-checkpoint
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
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- post-launch imports ---
import glob
import os

import numpy as np
import torch

import gymnasium as gym

import isaaclab_tasks  # noqa: F401
import go1_ball_balance  # noqa: F401

from go1_ball_balance.tasks.ball_juggle_hier.ball_juggle_hier_env_cfg import BallJuggleHierEnvCfg_PLAY
from go1_ball_balance.perception.sim_detector import SimBallDetector
from go1_ball_balance.perception.ball_ekf import BallEKF, BallEKFConfig
from go1_ball_balance.perception.frame_transforms import quat_to_rotmat, cam_detection_to_world


# Keep _quat_to_rotmat alias for backward compat in _save_annotated_frame
_quat_to_rotmat = quat_to_rotmat


def main():
    env_cfg = BallJuggleHierEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs

    # pi2 checkpoint
    if hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "torso_cmd"):
        pi2_path = _pi2_checkpoint_path
        if pi2_path is None:
            candidates = sorted(glob.glob(
                os.path.join(os.path.dirname(__file__), "..", "..",
                             "logs", "rsl_rl", "go1_torso_tracking", "*", "model_best.pt")
            ))
            if not candidates:
                candidates = sorted(glob.glob(
                    os.path.expanduser("~/Research/QuadruJuggle/logs/rsl_rl/go1_torso_tracking/*/model_best.pt")
                ))
            if candidates:
                pi2_path = candidates[-1]
                print(f"[demo] Auto-detected pi2: {pi2_path}")
        if pi2_path is not None:
            env_cfg.actions.torso_cmd.pi2_checkpoint = os.path.abspath(pi2_path)

    # Force DEBUG scene for camera
    from go1_ball_balance.tasks.ball_juggle_hier.ball_juggle_hier_env_cfg import BallJuggleHierSceneCfg_DEBUG
    env_cfg.scene = BallJuggleHierSceneCfg_DEBUG(num_envs=args_cli.num_envs, env_spacing=3.5)

    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    unwrapped = env.unwrapped

    # Setup detector and EKF (both in world frame)
    detector = SimBallDetector.from_tiled_camera_cfg()
    ekf_cfg = BallEKFConfig(contact_aware=True, q_vel=0.40)
    ekf = BallEKF(num_envs=1, device="cpu", cfg=ekf_cfg)

    # Output directory
    out_dir = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..",
        "source", "go1_ball_balance", "go1_ball_balance", "perception", "debug", "demo"
    ))
    os.makedirs(out_dir, exist_ok=True)

    ball = unwrapped.scene["ball"]

    # Give ball initial upward velocity
    try:
        ball_vel = ball.data.root_vel_w.clone()
        ball_vel[:, 2] = 3.0
        ball.write_root_velocity_to_sim(ball_vel)
        print("[demo] Applied 3 m/s upward velocity to ball.")
    except Exception as e:
        print(f"[demo] WARNING: could not set ball velocity: {e}")

    # Initialize EKF with GT ball position in world frame
    ball_pos_w_init = ball.data.root_pos_w[0].cpu()
    ball_vel_w_init = ball.data.root_vel_w[0, :3].cpu()
    ekf.reset(torch.tensor([0]), ball_pos_w_init.unsqueeze(0), ball_vel_w_init.unsqueeze(0))
    print(f"[demo] EKF initialized at world pos: {ball_pos_w_init.numpy()}")

    cam = unwrapped.scene.get("d435i")
    if cam is None:
        print("[demo] ERROR: no d435i camera in scene!")
        env.close()
        simulation_app.close()
        return

    # Print camera world pose for diagnostic
    cam.update(dt=0.02)
    cam_pos_w = cam.data.pos_w[0].cpu().numpy()
    cam_quat_w_ros = cam.data.quat_w_ros[0].cpu().numpy()
    R_cam = _quat_to_rotmat(cam_quat_w_ros)
    cam_fwd = R_cam @ np.array([0, 0, 1])  # ROS Z = optical axis
    cam_elev = np.degrees(np.arcsin(np.clip(cam_fwd[2], -1, 1)))
    print(f"[demo] Camera world pos: {cam_pos_w}")
    print(f"[demo] Camera world quat (ROS): {cam_quat_w_ros}")
    print(f"[demo] Camera forward (world): {cam_fwd}, elevation: {cam_elev:.1f}°")

    dt = 0.02  # 50 Hz policy rate

    metrics = {"detected": 0, "missed": 0, "rmse_det": [], "rmse_ekf": []}
    # Trajectory tracking for summary visualizations
    traj = {"gt": [], "ekf": [], "det": [], "det_steps": [], "steps": []}

    print(f"[demo] Running {args_cli.steps} steps...")
    for step in range(args_cli.steps):
        action = torch.zeros(unwrapped.action_space.shape, device=unwrapped.device)
        obs, _, _, _, _ = env.step(action)

        # GT ball position in world frame
        ball_pos_w = ball.data.root_pos_w[0].cpu().numpy()

        # Camera update + detection
        cam.update(dt=dt)
        depth_key = "distance_to_image_plane"
        detection = None
        if depth_key in cam.data.output:
            depth = cam.data.output[depth_key][0].cpu().numpy()
            if depth.ndim == 3:
                depth = depth[..., 0]
            detection = detector.detect(depth)

        # Get camera world pose for coordinate transform
        cam_pos_w = cam.data.pos_w[0].cpu().numpy()
        cam_quat_w_ros = cam.data.quat_w_ros[0].cpu().numpy()

        # Transform detection to world frame and run EKF
        if detection is not None:
            metrics["detected"] += 1
            # Camera frame → world frame (using camera's reported world pose)
            pos_world = cam_detection_to_world(detection.pos_cam, cam_pos_w, cam_quat_w_ros)
            z_meas = torch.tensor(pos_world, dtype=torch.float32).unsqueeze(0)
            detected = torch.tensor([True])

            # Detection RMSE vs GT (world frame)
            det_err = np.linalg.norm(pos_world - ball_pos_w)
            metrics["rmse_det"].append(det_err)
            traj["det"].append(pos_world.copy())
            traj["det_steps"].append(step)
        else:
            metrics["missed"] += 1
            z_meas = torch.zeros(1, 3)
            detected = torch.tensor([False])

        ekf.step(z_meas, detected, dt=dt)

        # EKF RMSE vs GT (world frame)
        ekf_pos = ekf.pos[0].numpy()
        ekf_err = np.linalg.norm(ekf_pos - ball_pos_w)
        metrics["rmse_ekf"].append(ekf_err)

        traj["gt"].append(ball_pos_w.copy())
        traj["ekf"].append(ekf_pos.copy())
        traj["steps"].append(step)

        # Save annotated frame periodically
        if step % args_cli.capture_interval == 0 and "rgb" in cam.data.output:
            _save_annotated_frame(cam, detection, out_dir, step, ekf_pos, ball_pos_w)

        if step % 50 == 0:
            det_rate = metrics["detected"] / max(1, step + 1) * 100
            ekf_rmse_recent = np.mean(metrics["rmse_ekf"][-10:]) if metrics["rmse_ekf"] else 0
            det_rmse_recent = np.mean(metrics["rmse_det"][-10:]) if metrics["rmse_det"] else 0
            print(f"[demo] Step {step}: det_rate={det_rate:.0f}%, "
                  f"det_rmse={det_rmse_recent:.4f}m, ekf_rmse={ekf_rmse_recent:.4f}m",
                  flush=True)

    # Summary
    total = metrics["detected"] + metrics["missed"]
    print(f"\n[demo] SUMMARY ({total} steps):")
    print(f"  Detection rate: {metrics['detected']}/{total} ({metrics['detected']/max(1,total)*100:.1f}%)")
    if metrics["rmse_det"]:
        print(f"  Detection RMSE: {np.mean(metrics['rmse_det']):.4f} +/- {np.std(metrics['rmse_det']):.4f} m")
    if metrics["rmse_ekf"]:
        print(f"  EKF RMSE: {np.mean(metrics['rmse_ekf']):.4f} +/- {np.std(metrics['rmse_ekf']):.4f} m")

    # Generate summary visualizations
    _save_summary_plots(traj, metrics, out_dir, dt)
    _compile_video(out_dir)

    env.close()
    simulation_app.close()


def _save_summary_plots(traj, metrics, out_dir, dt):
    """Generate summary figure: height trajectory + RMSE over time."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        gt = np.array(traj["gt"])
        ekf = np.array(traj["ekf"])
        steps = np.array(traj["steps"])
        t = steps * dt  # seconds

        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        # Panel 1: Z (height) trajectory — GT vs EKF vs detections
        ax1 = axes[0]
        ax1.plot(t, gt[:, 2], "k-", linewidth=1.5, label="GT")
        ax1.plot(t, ekf[:, 2], "b-", linewidth=1.2, alpha=0.8, label="EKF")
        if traj["det"]:
            det = np.array(traj["det"])
            det_t = np.array(traj["det_steps"]) * dt
            ax1.scatter(det_t, det[:, 2], s=8, c="green", alpha=0.5, label="Detection", zorder=5)
        ax1.set_ylabel("Height z (m)")
        ax1.set_title("Ball Height: GT vs EKF vs Camera Detection")
        ax1.legend(loc="upper right", fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Panel 2: RMSE over time (EKF and detection)
        ax2 = axes[1]
        rmse_ekf = np.array(metrics["rmse_ekf"])
        ax2.plot(t, rmse_ekf * 1000, "b-", linewidth=1.0, alpha=0.7, label="EKF error")
        if traj["det"]:
            # Sparse detection errors at det_steps
            rmse_det = np.array(metrics["rmse_det"])
            det_t = np.array(traj["det_steps"]) * dt
            ax2.scatter(det_t, rmse_det * 1000, s=8, c="green", alpha=0.5, label="Det error")
        ax2.set_ylabel("Position error (mm)")
        ax2.set_xlabel("Time (s)")
        ax2.set_title("Position Error vs Ground Truth")
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        path = os.path.join(out_dir, "summary.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[demo] Summary plot saved: {path}")
    except Exception as e:
        print(f"[demo] Could not generate summary plot: {e}")


def _compile_video(out_dir):
    """Compile saved frames into an mp4 video using ffmpeg."""
    import subprocess
    import shutil

    if shutil.which("ffmpeg") is None:
        print("[demo] ffmpeg not found — skipping video compilation.")
        return

    pattern = os.path.join(out_dir, "frame_%04d.png")
    # Check if any frames exist
    if not os.path.exists(os.path.join(out_dir, "frame_0000.png")):
        print("[demo] No frames found — skipping video compilation.")
        return

    video_path = os.path.join(out_dir, "demo.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", "10",
        "-i", pattern,
        "-c:v", "libx264", "-crf", "23", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        video_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, timeout=30)
        if os.path.exists(video_path):
            size_kb = os.path.getsize(video_path) / 1024
            print(f"[demo] Video saved: {video_path} ({size_kb:.0f} KB)")
        else:
            print("[demo] ffmpeg ran but no video produced.")
    except Exception as e:
        print(f"[demo] Video compilation failed: {e}")


def _save_annotated_frame(cam, detection, out_dir, step, ekf_pos_w, ball_pos_w):
    """Save RGB frame with detection overlay and EKF/GT info."""
    try:
        from PIL import Image, ImageDraw

        rgb = cam.data.output["rgb"][0].cpu().numpy()
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

        img = Image.fromarray(rgb).convert("RGB")
        draw = ImageDraw.Draw(img)

        if detection is not None:
            u, v = detection.pixel_uv
            r = max(3, 10)
            draw.ellipse([u - r, v - r, u + r, v + r], outline=(0, 255, 0), width=2)
            draw.text((10, 10), f"Step {step} | det d={detection.depth_m:.2f}m", fill=(0, 255, 0))
        else:
            draw.text((10, 10), f"Step {step} | NO DETECTION", fill=(255, 0, 0))

        # Show GT vs EKF position
        gt_z = ball_pos_w[2]
        ekf_z = ekf_pos_w[2]
        draw.text((10, 25), f"GT z={gt_z:.3f}m | EKF z={ekf_z:.3f}m", fill=(255, 255, 0))

        img.save(os.path.join(out_dir, f"frame_{step:04d}.png"))
    except Exception as e:
        print(f"[demo] Frame save error at step {step}: {e}")


if __name__ == "__main__":
    main()
