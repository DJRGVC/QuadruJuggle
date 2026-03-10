# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Offline slow-motion renderer for recorded ball-balance trajectories.

Pipeline
--------
1.  Record a trajectory during play:
      python play.py --task Isaac-BallBalance-Go1-Play-v0 --num_envs 1
                     --record-traj /tmp/traj.npz

2.  Render at a high frame rate with cubic-spline / SLERP interpolation:
      python render_slow.py --traj /tmp/traj.npz --render-fps 480

3.  Play back at 30 FPS → 16× slow motion:
      ffplay slow_motion.mp4
    (the script prints the exact ffplay command at the end)

Physics setup (our Go1 env)
  sim.dt = 1/200 = 5 ms  →  physics_fps = 200 Hz
  decimation = 4 (training)  →  but recording forces decimation=1

Interpolation factor examples
  render-fps=400 Hz  →  2.0×  upsample  →  400/30 ≈ 13×  slow
  render-fps=480 Hz  →  2.4×  upsample  →  480/30  = 16×  slow
  render-fps=600 Hz  →  3.0×  upsample  →  600/30  = 20×  slow
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# ── argparse ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Offline slow-motion renderer.")
parser.add_argument("--traj",       required=True,  type=str,   help="Path to trajectory .npz produced by play.py --record-traj.")
parser.add_argument("--env-idx",    default=0,       type=int,   help="Which env from the .npz to render (default 0).")
parser.add_argument("--render-fps",   default=480.0, type=float, help="Interpolation FPS. slowdown = render-fps / playback-fps (default 480).")
parser.add_argument("--playback-fps", default=30.0,  type=float, help="Container FPS of the output video (default 30). Use 240 for 240 Hz displays.")
parser.add_argument("--output",       default="slow_motion.mp4", type=str, help="Output video file (default slow_motion.mp4).")
parser.add_argument("--duration",   default=None,    type=float, help="Limit render to first N seconds of the trajectory.")
parser.add_argument(
    "--task", type=str, default="Isaac-BallBalance-Go1-Play-v0",
    help="Isaac Lab task name (default Isaac-BallBalance-Go1-Play-v0).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# cameras always needed for rgb_array render mode
args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── post-launch imports ───────────────────────────────────────────────────────
import numpy as np
import torch

try:
    import imageio
except ImportError:
    raise ImportError("imageio is required: pip install imageio[ffmpeg]")

try:
    from scipy.interpolate import CubicSpline
    from scipy.spatial.transform import Rotation, Slerp
except ImportError:
    raise ImportError("scipy is required: pip install scipy")

import gymnasium as gym
import isaaclab.utils.math as math_utils
import isaaclab_tasks  # noqa: F401

import go1_ball_balance  # noqa: F401
from go1_ball_balance.tasks.ball_balance.ball_balance_env_cfg import (
    BallBalanceEnvCfg_PLAY,
    _PADDLE_OFFSET_B,
)


# ── interpolation ─────────────────────────────────────────────────────────────

def _interpolate(traj: dict, env_idx: int, render_fps: float) -> tuple[dict, int]:
    """Cubic-spline + SLERP interpolation from physics rate to render_fps.

    Args:
        traj:       Raw trajectory dict loaded from .npz.
        env_idx:    Which environment index to extract.
        render_fps: Target frames-per-second after interpolation.

    Returns:
        (interp_dict, T_new) where interp_dict contains (T_new, D) arrays
        and T_new is the number of output frames.
    """
    physics_fps = 1.0 / float(traj["physics_dt"])
    T = traj["root_pos"].shape[0]
    factor = render_fps / physics_fps
    T_new = max(int(round(T * factor)), 2)

    t_orig = np.arange(T, dtype=float)
    t_new  = np.linspace(0.0, float(T - 1), T_new)

    out: dict = {}
    e = env_idx

    # ── vector fields: cubic spline ───────────────────────────────────────────
    vec_keys = ("root_pos", "root_lin_vel", "root_ang_vel",
                "joint_pos", "joint_vel",
                "ball_pos",  "ball_lin_vel")
    for key in vec_keys:
        raw = traj[key][:, e, :]   # (T, D)
        if T < 4:
            # Fallback: linear interp when trajectory is very short
            out[key] = np.column_stack(
                [np.interp(t_new, t_orig, raw[:, d]) for d in range(raw.shape[1])]
            )
        else:
            cs = CubicSpline(t_orig, raw, axis=0, extrapolate=False)
            result = cs(t_new)
            # CubicSpline may return NaN at the boundary; clamp to nearest valid
            result = np.nan_to_num(result, nan=0.0)
            out[key] = result

    # ── quaternion fields: SLERP ──────────────────────────────────────────────
    # Isaac Lab convention: (w, x, y, z)
    # scipy Rotation convention: (x, y, z, w)
    for key in ("root_quat", "ball_quat"):
        raw = traj[key][:, e, :]           # (T, 4) in wxyz
        raw_xyzw = raw[:, [1, 2, 3, 0]]   # → xyzw for scipy
        r = Rotation.from_quat(raw_xyzw)
        slerp = Slerp(t_orig, r)
        interp_xyzw = slerp(t_new).as_quat()          # (T_new, 4) xyzw
        out[key] = interp_xyzw[:, [3, 0, 1, 2]]       # back to wxyz

    print(f"[INFO] Interpolated {T} → {T_new} frames  "
          f"({physics_fps:.0f} Hz → {render_fps:.0f} Hz,  {factor:.2f}× upsample)")
    return out, T_new


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── load trajectory ───────────────────────────────────────────────────────
    traj = {k: v for k, v in np.load(args_cli.traj, allow_pickle=True).items()}
    N_envs = traj["root_pos"].shape[0]   # first dim is time
    # actually shape is (T, N_envs, D)
    N_envs = traj["root_pos"].shape[1]
    env_idx = args_cli.env_idx
    if env_idx >= N_envs:
        print(f"[ERROR] --env-idx {env_idx} out of range (trajectory has {N_envs} envs).")
        return

    physics_fps = 1.0 / float(traj["physics_dt"])
    T_total = traj["root_pos"].shape[0]
    print(f"[INFO] Loaded: {T_total} frames × {N_envs} envs  "
          f"|  physics_fps={physics_fps:.0f} Hz  "
          f"|  {T_total/physics_fps:.1f}s total")

    # ── interpolate ───────────────────────────────────────────────────────────
    interp, T_new = _interpolate(traj, env_idx, args_cli.render_fps)

    # optionally limit duration
    if args_cli.duration is not None:
        max_frames = max(1, int(args_cli.duration * args_cli.render_fps))
        if T_new > max_frames:
            T_new = max_frames
            interp = {k: v[:T_new] for k, v in interp.items()}
            print(f"[INFO] Duration limited to {args_cli.duration}s → {T_new} frames")

    # ── create environment ────────────────────────────────────────────────────
    env_cfg = BallBalanceEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    if args_cli.device:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env.reset()

    device  = env.unwrapped.device
    scene   = env.unwrapped.scene
    sim     = env.unwrapped.sim

    robot  = scene["robot"]
    ball   = scene["ball"]
    paddle = scene["paddle"]

    # env origin for env 0 in this 1-env setup (usually near world origin)
    replay_origin = scene.env_origins[0].to(device)   # (3,)

    pad_off = torch.tensor(list(_PADDLE_OFFSET_B), dtype=torch.float32, device=device)

    # ── kinematic replay loop ─────────────────────────────────────────────────
    frames: list = []
    print(f"[INFO] Rendering {T_new} frames at {args_cli.render_fps:.0f} Hz interpolation, "
          f"encoded at {int(args_cli.playback_fps)} FPS ...")

    with torch.no_grad():
        for t_idx in range(T_new):
            def _t(key: str, shape: tuple | None = None) -> torch.Tensor:
                arr = interp[key][t_idx]
                if shape is not None:
                    arr = arr.reshape(shape)
                return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)

            root_pos_w  = _t("root_pos")  + replay_origin.unsqueeze(0)   # (1, 3)
            root_quat_w = _t("root_quat")                                  # (1, 4) wxyz
            root_lin    = _t("root_lin_vel")
            root_ang    = _t("root_ang_vel")
            jpos        = _t("joint_pos")
            jvel        = _t("joint_vel")
            ball_pos_w  = _t("ball_pos")  + replay_origin.unsqueeze(0)
            ball_quat_w = _t("ball_quat")
            ball_lin    = _t("ball_lin_vel")

            # Paddle: compute from interpolated robot pose + body-frame offset.
            # This matches exactly what update_paddle_pose does in simulation and
            # is smooth because it follows the spline-interpolated root_pos/quat.
            pad_off_w  = math_utils.quat_apply(root_quat_w,
                                                pad_off.unsqueeze(0).expand(1, -1))
            paddle_pos_w = root_pos_w + pad_off_w

            # ── set states ────────────────────────────────────────────────────
            robot.write_root_pose_to_sim(torch.cat([root_pos_w, root_quat_w], dim=-1))
            robot.write_root_velocity_to_sim(torch.cat([root_lin, root_ang], dim=-1))
            robot.write_joint_state_to_sim(jpos, jvel)

            ball.write_root_pose_to_sim(torch.cat([ball_pos_w, ball_quat_w], dim=-1))
            ball.write_root_velocity_to_sim(
                torch.cat([ball_lin, torch.zeros_like(ball_lin)], dim=-1)
            )

            paddle.write_root_pose_to_sim(torch.cat([paddle_pos_w, root_quat_w], dim=-1))

            # ── propagate transforms (no physics step) ────────────────────────
            sim.forward()

            # ── capture frame ─────────────────────────────────────────────────
            frame = env.render()
            if frame is not None:
                # shape may be (H, W, 3) or (1, H, W, 3) — normalise
                if isinstance(frame, np.ndarray):
                    if frame.ndim == 4:
                        frame = frame[0]
                    frames.append(frame.astype(np.uint8))

            if (t_idx + 1) % 500 == 0 or t_idx == T_new - 1:
                print(f"  {t_idx + 1:>6}/{T_new}  ({100*(t_idx+1)/T_new:.1f}%)")

    env.close()

    # ── write video ───────────────────────────────────────────────────────────
    if not frames:
        print("[ERROR] No frames captured — check that render_mode='rgb_array' is working.")
        return

    out_path     = args_cli.output
    render_fps   = args_cli.render_fps
    playback_fps = args_cli.playback_fps
    slowdown     = render_fps / playback_fps

    # Encode at playback_fps — slowdown = render_fps / playback_fps.
    # e.g. render_fps=480, playback_fps=30  → 1440 frames / 30 FPS = 48 s → 16× slow
    #      render_fps=3840, playback_fps=240 → 11520 frames / 240 FPS = 48 s → 16× slow (smooth)
    encode_fps = int(playback_fps)
    video_dur_s = len(frames) / encode_fps
    phys_dur_s  = len(frames) / render_fps

    print(f"\n[INFO] Writing video → {out_path}  "
          f"({len(frames)} frames, encoded @ {encode_fps} FPS)")
    try:
        writer = imageio.get_writer(out_path, fps=encode_fps, codec="libx264", quality=9)
        for f in frames:
            writer.append_data(f)
        writer.close()
    except Exception as exc:
        print(f"[WARN] imageio.get_writer failed ({exc}), trying imageio.mimwrite ...")
        imageio.mimwrite(out_path, frames, fps=encode_fps)

    print(f"\n[INFO] Done.")
    print(f"       Action   : {phys_dur_s:.2f}s of physics  →  {video_dur_s:.1f}s of video  "
          f"({slowdown:.1f}× slow motion)")
    print(f"\n  Play:")
    print(f"    ffplay '{out_path}'")


if __name__ == "__main__":
    main()
    simulation_app.close()
