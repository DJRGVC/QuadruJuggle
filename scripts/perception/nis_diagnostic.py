#!/usr/bin/env python3
"""Standalone NIS (Normalized Innovation Squared) diagnostic for EKF tuning.

Runs the env with EKF mode for N steps (no training), logs NIS every
`log_interval` steps, and reports whether NIS is within the 95% χ²(3)
consistency band [0.35, 7.81].

Target: mean NIS ≈ 3.0 (the mean of χ²(3) with 3 DoF).
  - NIS > 7.81: filter is overconfident (Q or R too small)
  - NIS < 0.35: filter is over-conservative (Q or R too large)

Usage:
    $C3R_BIN/gpu_lock.sh uv run --active python scripts/perception/nis_diagnostic.py \
        --pi2-checkpoint <path> --num_envs 2048 --steps 500 --headless
"""

import argparse
import os
import sys


def _print(*args, **kwargs):
    """Unbuffered print for subprocess visibility."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="EKF NIS diagnostic")
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=500, help="Number of env steps")
    parser.add_argument("--log_interval", type=int, default=50, help="Log NIS every N steps")
    parser.add_argument("--pi2-checkpoint", type=str, required=True)
    parser.add_argument("--headless", action="store_true")
    # EKF tuning overrides
    parser.add_argument("--q_vel", type=float, default=None, help="Override EKF q_vel")
    parser.add_argument("--q_pos", type=float, default=None, help="Override EKF q_pos")
    parser.add_argument("--r_xy", type=float, default=None, help="Override EKF r_xy")
    parser.add_argument("--r_z", type=float, default=None, help="Override EKF r_z")
    parser.add_argument("--world-frame", action="store_true",
                        help="Run EKF in world frame (transforms meas body→world)")
    parser.add_argument("--no-contact-aware", action="store_true",
                        help="Disable contact-aware adaptive process noise (use uniform q_vel)")
    parser.add_argument("--no-imu", action="store_true",
                        help="Disable IMU-aided Coriolis/centrifugal corrections")
    parser.add_argument("--enable-spin", action="store_true",
                        help="Enable 9D spin estimation (Magnus effect)")
    args = parser.parse_args()

    # Isaac Lab AppLauncher
    from isaaclab.app import AppLauncher

    inner_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(inner_parser)
    launcher_args, _ = inner_parser.parse_known_args(
        ["--headless"] if args.headless else []
    )
    launcher_args.enable_cameras = False
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    # Worktree isolation
    _OUR_SRC = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..",
        "source", "go1_ball_balance",
    ))
    sys.path.insert(0, _OUR_SRC)

    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg

    import go1_ball_balance  # noqa: F401

    from go1_ball_balance.perception import (
        BallObsNoiseCfg,
        ball_pos_perceived,
        ball_vel_perceived,
        reset_perception_pipeline,
    )
    from go1_ball_balance.perception.ball_ekf import BallEKFConfig
    from go1_ball_balance.tasks.ball_juggle_hier.ball_juggle_hier_env_cfg import (
        BallJuggleHierEnvCfg,
        _PADDLE_OFFSET_B,
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Build EKF config with optional overrides
    ekf_cfg = BallEKFConfig()
    if args.q_vel is not None:
        ekf_cfg.q_vel = args.q_vel
    if args.q_pos is not None:
        ekf_cfg.q_pos = args.q_pos
    if args.r_xy is not None:
        ekf_cfg.r_xy = args.r_xy
    if args.r_z is not None:
        ekf_cfg.r_z = args.r_z
    if getattr(args, "no_contact_aware", False):
        ekf_cfg.contact_aware = False

    world_frame = getattr(args, "world_frame", False)
    enable_imu = not getattr(args, "no_imu", False)
    enable_spin = getattr(args, "enable_spin", False)
    _print(f"\nEKF config: q_pos={ekf_cfg.q_pos}, q_vel={ekf_cfg.q_vel}, "
           f"r_xy={ekf_cfg.r_xy}, r_z={ekf_cfg.r_z}, r_z_per_m={ekf_cfg.r_z_per_metre}, "
           f"contact_aware={ekf_cfg.contact_aware}, world_frame={world_frame}, "
           f"enable_imu={enable_imu}, enable_spin={enable_spin}")

    # Build env config
    noise_cfg = BallObsNoiseCfg(
        mode="ekf", ekf_cfg=ekf_cfg, world_frame=world_frame,
        enable_imu=enable_imu, enable_spin=enable_spin,
    )

    env_cfg = BallJuggleHierEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"

    pi2_path = os.path.abspath(args.pi2_checkpoint)
    if not os.path.isfile(pi2_path):
        raise FileNotFoundError(f"pi2 checkpoint not found: {pi2_path}")
    env_cfg.actions.torso_cmd.pi2_checkpoint = pi2_path

    # Patch obs for EKF mode
    env_cfg.observations.policy.ball_pos = ObsTerm(
        func=ball_pos_perceived,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "noise_cfg": noise_cfg,
        },
    )
    env_cfg.observations.policy.ball_vel = ObsTerm(
        func=ball_vel_perceived,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
            "noise_cfg": noise_cfg,
        },
    )
    env_cfg.events.reset_perception = EventTerm(
        func=reset_perception_pipeline,
        mode="reset",
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "robot_cfg": SceneEntityCfg("robot"),
            "paddle_offset_b": _PADDLE_OFFSET_B,
        },
    )

    env = gym.make("Isaac-BallJuggleHier-Go1-v0", cfg=env_cfg)
    # Pipeline is created during gym.make() — delete and recreate with diagnostics
    base_env = env.unwrapped
    base_env._perception_diagnostics_enabled = True
    if hasattr(base_env, "_perception_pipeline"):
        base_env._perception_pipeline = None  # force recreation with diag enabled

    obs, _ = env.reset()
    action_dim = env.action_space.shape[-1]

    _print(f"\nRunning NIS diagnostic: {args.num_envs} envs × {args.steps} steps")
    _print(f"{'='*70}")
    _print(f"  {'Step':>6s}  {'NIS':>8s}  {'Band':>12s}  "
           f"{'EKF mm':>8s}  {'Raw mm':>8s}  {'Impr%':>7s}  {'Det%':>6s}")
    _print(f"  {'-'*6}  {'-'*8}  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*6}")

    all_nis = []

    for step in range(1, args.steps + 1):
        # Random actions (untrained — just want to measure EKF consistency)
        actions = torch.randn(args.num_envs, action_dim, device="cuda:0") * 0.1
        obs, _, _, _, _ = env.step(actions)

        if step % args.log_interval == 0:
            pipeline = getattr(base_env, "_perception_pipeline", None)
            if pipeline is None:
                _print(f"  {step:6d}  PIPELINE NOT FOUND — check env wrapper chain")
                continue

            diag = pipeline.diagnostics
            if diag is None:
                _print(f"  {step:6d}  DIAGNOSTICS DISABLED — enable_diag not set?")
                continue

            nis = diag.get("mean_nis", 0.0)
            all_nis.append(nis)

            # Classify NIS
            if 0.35 <= nis <= 7.81:
                band = "✓ OK"
            elif nis < 0.35:
                band = "⚠ LOW (over-conservative)"
            else:
                band = "⚠ HIGH (overconfident)"

            _print(f"  {step:6d}  {nis:8.3f}  {band:>12s}  "
                   f"{diag.get('pos_rmse_ekf_mm', 0):8.2f}  "
                   f"{diag.get('pos_rmse_raw_mm', 0):8.2f}  "
                   f"{diag.get('ekf_improvement_pct', 0):7.1f}  "
                   f"{diag.get('detection_rate', 0)*100:6.1f}")

    _print(f"{'='*70}")
    if all_nis:
        mean_nis = sum(all_nis) / len(all_nis)
        in_band = sum(1 for n in all_nis if 0.35 <= n <= 7.81)
        _print(f"\nSummary:")
        _print(f"  Overall mean NIS: {mean_nis:.3f}  (target ≈ 3.0)")
        _print(f"  In 95% band [0.35, 7.81]: {in_band}/{len(all_nis)} intervals")

        if mean_nis < 0.35:
            _print(f"  → DIAGNOSIS: Q/R too large. Try reducing q_vel (currently {ekf_cfg.q_vel})")
        elif mean_nis > 7.81:
            _print(f"  → DIAGNOSIS: Q/R too small. Try increasing q_vel or r_xy/r_z")
        elif mean_nis < 2.0:
            _print(f"  → DIAGNOSIS: Slightly over-conservative but OK. Consider reducing q_vel slightly.")
        elif mean_nis > 5.0:
            _print(f"  → DIAGNOSIS: Slightly overconfident but OK. Consider increasing q_vel slightly.")
        else:
            _print(f"  → DIAGNOSIS: Well-tuned! NIS close to χ²(3) mean = 3.0")
    else:
        _print("\n  No NIS data collected!")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
