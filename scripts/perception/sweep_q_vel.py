#!/usr/bin/env python3
"""Sweep q_vel values for EKF tuning under trained policy.

Runs eval_perception_live.py at multiple q_vel values and collects results
into a summary table. Each q_vel is evaluated in a single env session to
avoid repeated env creation overhead.

Usage:
    $C3R_BIN/gpu_lock.sh uv run --active python scripts/perception/sweep_q_vel.py \
        --pi2-checkpoint <path> --pi1-checkpoint <path> --headless
"""

import argparse
import json
import os
import sys


def _print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="q_vel sweep for EKF tuning")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=600,
                        help="Steps per q_vel setting (shorter per-point)")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--target-height", type=float, default=0.10)
    parser.add_argument("--sigma-ratio", type=float, default=3.5)
    parser.add_argument("--output", type=str, default=None)

    # Pre-strip checkpoints before AppLauncher
    _pi2_path = None
    _pi1_path = None
    _clean = []
    _i = 0
    while _i < len(sys.argv):
        if sys.argv[_i] in ("--pi2-checkpoint", "--pi2_checkpoint") and _i + 1 < len(sys.argv):
            _pi2_path = sys.argv[_i + 1]
            _i += 2
        elif sys.argv[_i] in ("--pi1-checkpoint", "--pi1_checkpoint") and _i + 1 < len(sys.argv):
            _pi1_path = sys.argv[_i + 1]
            _i += 2
        else:
            _clean.append(sys.argv[_i])
            _i += 1
    sys.argv = _clean

    # q_vel values to sweep (+ q_vel_contact values)
    parser.add_argument("--q-vels", type=str, default="0.4,2.0,5.0,10.0,20.0,50.0",
                        help="Comma-separated q_vel values to sweep")
    parser.add_argument("--q-vel-contact", type=float, default=50.0,
                        help="Fixed q_vel_contact for all runs")
    parser.add_argument("--q-vel-post-contact", type=float, default=20.0,
                        help="q_vel during post-contact window")
    parser.add_argument("--post-contact-steps", type=int, default=10,
                        help="Steps after contact to keep q_vel elevated")
    parser.add_argument("--warmup-steps", type=int, default=50,
                        help="Steps to run before collecting metrics (EKF convergence)")
    parser.add_argument("--no-post-contact", action="store_true",
                        help="Disable post-contact inflation (2-level only)")

    args = parser.parse_args()

    if args.no_post_contact:
        args.post_contact_steps = 0

    if _pi2_path is None:
        parser.error("--pi2-checkpoint is required")
    if _pi1_path is None:
        parser.error("--pi1-checkpoint is required")

    q_vel_list = [float(x) for x in args.q_vels.split(",")]
    _print(f"\nSweeping q_vel = {q_vel_list}")
    _print(f"q_vel_contact = {args.q_vel_contact} (fixed)")
    if args.post_contact_steps > 0:
        _print(f"q_vel_post_contact = {args.q_vel_post_contact}, post_contact_steps = {args.post_contact_steps}")
    else:
        _print(f"post-contact inflation DISABLED (2-level q_vel only)")
    _print(f"target_height = {args.target_height}m, {args.num_envs} envs × {args.steps} steps/setting\n")

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
    from rsl_rl.runners import OnPolicyRunner

    _OUR_SRC = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..",
        "source", "go1_ball_balance",
    ))
    sys.path.insert(0, _OUR_SRC)

    from isaaclab.envs import ManagerBasedRLEnvCfg
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
    from go1_ball_balance.perception.ball_ekf import BallEKF, BallEKFConfig
    from go1_ball_balance.tasks.ball_juggle_hier.ball_juggle_hier_env_cfg import (
        BallJuggleHierEnvCfg,
        _PADDLE_OFFSET_B,
    )
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- Build env ONCE ---
    # Start with first q_vel; we'll swap the EKF between sweeps
    ekf_cfg = BallEKFConfig()
    ekf_cfg.q_vel = q_vel_list[0]
    ekf_cfg.q_vel_contact = args.q_vel_contact
    ekf_cfg.q_vel_post_contact = args.q_vel_post_contact
    ekf_cfg.post_contact_steps = args.post_contact_steps

    noise_cfg = BallObsNoiseCfg(mode="ekf", ekf_cfg=ekf_cfg, enable_imu=True)

    env_cfg = BallJuggleHierEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"

    pi2_path = os.path.abspath(_pi2_path)
    if not os.path.isfile(pi2_path):
        raise FileNotFoundError(f"pi2 checkpoint: {pi2_path}")
    env_cfg.actions.torso_cmd.pi2_checkpoint = pi2_path

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
    base_env = env.unwrapped
    base_env._perception_diagnostics_enabled = True

    target_h = args.target_height
    sigma = target_h / args.sigma_ratio
    base_env._target_apex_heights = torch.full(
        (args.num_envs,), target_h, device=base_env.device, dtype=torch.float32,
    )
    base_env._target_apex_sigmas = torch.full(
        (args.num_envs,), sigma, device=base_env.device, dtype=torch.float32,
    )

    env_wrapped = RslRlVecEnvWrapper(env)

    # --- Load policy ---
    pi1_path = os.path.abspath(_pi1_path)
    if not os.path.isfile(pi1_path):
        raise FileNotFoundError(f"pi1 checkpoint: {pi1_path}")

    _print(f"Loading pi1: {pi1_path}")
    run_dir = os.path.dirname(pi1_path)
    agent_cfg_path = os.path.join(run_dir, "params", "agent.yaml")
    if os.path.isfile(agent_cfg_path):
        from omegaconf import OmegaConf
        agent_dict = OmegaConf.to_container(OmegaConf.load(agent_cfg_path), resolve=True)
    else:
        from go1_ball_balance.agents.rsl_rl_ppo_cfg import BallJuggleHierPPORunnerCfg
        agent_dict = BallJuggleHierPPORunnerCfg().to_dict()

    runner = OnPolicyRunner(env_wrapped, agent_dict, log_dir=None, device="cuda:0")
    runner.load(pi1_path)
    policy = runner.get_inference_policy(device=base_env.device)

    _print(f"Target: {target_h:.2f}m  σ={sigma:.3f}m\n")

    # --- Sweep ---
    all_results = []

    for q_vel in q_vel_list:
        _print(f"\n{'='*80}")
        _print(f"  q_vel = {q_vel}  (q_vel_contact = {args.q_vel_contact})")
        _print(f"{'='*80}")

        # Swap q_vel in the EKF config
        pipeline = getattr(base_env, "_perception_pipeline", None)
        if pipeline is not None and hasattr(pipeline, "ekf"):
            pipeline.ekf.cfg.q_vel = q_vel
            pipeline.ekf.cfg.q_vel_contact = args.q_vel_contact
            pipeline.ekf.cfg.q_vel_post_contact = args.q_vel_post_contact
            pipeline.ekf.cfg.post_contact_steps = args.post_contact_steps
            # Reset EKF state with current positions (resets P to initial values)
            all_ids = torch.arange(args.num_envs, device=base_env.device)
            init_pos = pipeline.ekf.pos.clone()
            init_vel = pipeline.ekf.vel.clone()
            pipeline.ekf.reset(all_ids, init_pos, init_vel)
            # Flush diagnostic accumulators (diagnostics property resets on access)
            _ = pipeline.diagnostics

        # Reset env for clean start
        obs = env_wrapped.get_observations()

        # Warmup: let EKF converge with new q_vel before collecting stats
        if args.warmup_steps > 0:
            _print(f"  warmup: {args.warmup_steps} steps...")
            for _ in range(args.warmup_steps):
                with torch.inference_mode():
                    actions = policy(obs)
                    obs, _, _, _ = env_wrapped.step(actions)
            # Flush stale diagnostics accumulated during warmup
            pipeline = getattr(base_env, "_perception_pipeline", None)
            if pipeline is not None:
                _ = pipeline.diagnostics

        nis_vals = []
        nis_flight_vals = []
        nis_contact_vals = []
        ekf_rmse_vals = []
        raw_rmse_vals = []
        total_ep = 0
        total_to = 0
        log_interval = 100

        for step in range(1, args.steps + 1):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, infos = env_wrapped.step(actions)

            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                time_outs = infos.get("time_outs", torch.zeros_like(dones))
                total_ep += done_ids.numel()
                total_to += time_outs[done_ids].sum().item()

            if step % log_interval == 0:
                pipeline = getattr(base_env, "_perception_pipeline", None)
                if pipeline is None:
                    continue
                diag = pipeline.diagnostics
                if diag is None:
                    continue

                nis = diag.get("mean_nis", 0.0)
                nis_f = diag.get("mean_nis_flight", 0.0)
                nis_c = diag.get("mean_nis_contact", 0.0)
                ekf_mm = diag.get("pos_rmse_ekf_mm", 0)
                raw_mm = diag.get("pos_rmse_raw_mm", 0)

                if nis > 0:
                    nis_vals.append(nis)
                if nis_f > 0:
                    nis_flight_vals.append(nis_f)
                if nis_c > 0:
                    nis_contact_vals.append(nis_c)
                if ekf_mm > 0:
                    ekf_rmse_vals.append(ekf_mm)
                if raw_mm > 0:
                    raw_rmse_vals.append(raw_mm)

                _print(f"  step {step:4d}  NIS={nis:6.2f}  flight={nis_f:6.2f}  "
                       f"contact={nis_c:6.2f}  EKF={ekf_mm:5.1f}mm  raw={raw_mm:5.1f}mm")

        # Aggregate
        mean_nis = sum(nis_vals) / len(nis_vals) if nis_vals else 0
        mean_f = sum(nis_flight_vals) / len(nis_flight_vals) if nis_flight_vals else 0
        mean_c = sum(nis_contact_vals) / len(nis_contact_vals) if nis_contact_vals else 0
        mean_ekf = sum(ekf_rmse_vals) / len(ekf_rmse_vals) if ekf_rmse_vals else 0
        mean_raw = sum(raw_rmse_vals) / len(raw_rmse_vals) if raw_rmse_vals else 0
        impr = (1 - mean_ekf / mean_raw) * 100 if mean_raw > 0 else 0

        result = {
            "q_vel": q_vel,
            "q_vel_contact": args.q_vel_contact,
            "mean_nis": round(mean_nis, 3),
            "flight_nis": round(mean_f, 3),
            "contact_nis": round(mean_c, 3),
            "ekf_rmse_mm": round(mean_ekf, 2),
            "raw_rmse_mm": round(mean_raw, 2),
            "improvement_pct": round(impr, 1),
            "episodes": total_ep,
            "timeout_pct": round(100 * total_to / max(total_ep, 1), 1),
        }
        all_results.append(result)

        _print(f"\n  → NIS: {mean_nis:.2f} (flight={mean_f:.2f}, contact={mean_c:.2f})")
        _print(f"  → RMSE: EKF={mean_ekf:.1f}mm, raw={mean_raw:.1f}mm, impr={impr:.1f}%")

    # --- Summary table ---
    _print(f"\n\n{'='*100}")
    _print(f"  q_vel SWEEP SUMMARY  (target_height={target_h}m, {args.num_envs} envs × {args.steps} steps)")
    _print(f"{'='*100}")
    _print(f"  {'q_vel':>8s}  {'NIS':>8s}  {'Flight':>8s}  {'Contact':>8s}  "
           f"{'EKF mm':>8s}  {'Raw mm':>8s}  {'Impr%':>7s}  {'Eps':>6s}  {'TO%':>6s}")
    _print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  "
           f"{'-'*8}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*6}")

    best_q = None
    best_score = float("inf")
    for r in all_results:
        _print(f"  {r['q_vel']:8.1f}  {r['mean_nis']:8.3f}  {r['flight_nis']:8.3f}  "
               f"{r['contact_nis']:8.3f}  {r['ekf_rmse_mm']:8.2f}  {r['raw_rmse_mm']:8.2f}  "
               f"{r['improvement_pct']:7.1f}  {r['episodes']:6d}  {r['timeout_pct']:5.1f}%")
        # Score: distance from target NIS=3.0 for flight phase
        score = abs(r["flight_nis"] - 3.0)
        if score < best_score:
            best_score = score
            best_q = r["q_vel"]

    _print(f"\n  Best q_vel for flight NIS ≈ 3.0: {best_q}")
    _print(f"  (closest flight NIS distance from target: {best_score:.2f})")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"sweep": all_results, "best_q_vel": best_q}, f, indent=2)
        _print(f"\n  Results saved to: {args.output}")

    _print()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
