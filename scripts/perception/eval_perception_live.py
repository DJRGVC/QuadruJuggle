#!/usr/bin/env python3
"""Evaluate EKF perception accuracy during live pi1 policy execution.

Unlike nis_diagnostic.py (which uses random actions), this script loads a
trained pi1 checkpoint and evaluates EKF accuracy under realistic ball
trajectories. This tests whether the filter is well-tuned for the actual
ball dynamics that occur during juggling.

Reports: NIS (overall, flight, contact), position RMSE, detection rate,
gate rejection rate, and per-target-height breakdowns.

Usage:
    $C3R_BIN/gpu_lock.sh uv run --active python scripts/perception/eval_perception_live.py \
        --pi2-checkpoint logs/rsl_rl/go1_torso_tracking/<run>/model_best.pt \
        --pi1-checkpoint logs/rsl_rl/go1_ball_juggle_hier/<run>/model_best.pt \
        --num_envs 512 --steps 1000 --headless

    # With EKF parameter overrides:
    --q_vel 0.10 --q_vel_contact 20.0
"""

import argparse
import os
import sys


def _print(*args, **kwargs):
    """Unbuffered print for subprocess visibility."""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description="Live-policy EKF perception evaluation")
    parser.add_argument("--num_envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000, help="Number of env steps")
    parser.add_argument("--log_interval", type=int, default=100, help="Log metrics every N steps")
    parser.add_argument("--pi1-checkpoint", type=str, required=True,
                        help="Path to trained pi1 (ball planner) checkpoint")
    parser.add_argument("--headless", action="store_true")

    # Pre-strip --pi2-checkpoint before Hydra/AppLauncher sees it
    _pi2_path = None
    _clean = []
    _i = 0
    while _i < len(sys.argv):
        if sys.argv[_i] in ("--pi2-checkpoint", "--pi2_checkpoint") and _i + 1 < len(sys.argv):
            _pi2_path = sys.argv[_i + 1]
            _i += 2
        else:
            _clean.append(sys.argv[_i])
            _i += 1
    sys.argv = _clean

    # EKF tuning overrides
    parser.add_argument("--q_vel", type=float, default=None)
    parser.add_argument("--q_vel_contact", type=float, default=None)
    parser.add_argument("--q_pos", type=float, default=None)
    parser.add_argument("--r_xy", type=float, default=None)
    parser.add_argument("--r_z", type=float, default=None)
    parser.add_argument("--no-contact-aware", action="store_true")
    parser.add_argument("--no-imu", action="store_true")
    parser.add_argument("--enable-spin", action="store_true")

    # Target height for evaluation
    parser.add_argument("--target-height", type=float, default=0.30,
                        help="Fixed target apex height for all envs (metres)")
    parser.add_argument("--sigma-ratio", type=float, default=3.5,
                        help="sigma = target / sigma_ratio")

    args = parser.parse_args()

    if _pi2_path is None:
        parser.error("--pi2-checkpoint is required")

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

    # Worktree isolation — ensure our source is on sys.path
    _OUR_SRC = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..",
        "source", "go1_ball_balance",
    ))
    sys.path.insert(0, _OUR_SRC)

    from isaaclab.envs import ManagerBasedRLEnvCfg
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg

    import go1_ball_balance  # noqa: F401 — register tasks

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
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- Build EKF config ---
    ekf_cfg = BallEKFConfig()
    if args.q_vel is not None:
        ekf_cfg.q_vel = args.q_vel
    if args.q_pos is not None:
        ekf_cfg.q_pos = args.q_pos
    if args.r_xy is not None:
        ekf_cfg.r_xy = args.r_xy
    if args.r_z is not None:
        ekf_cfg.r_z = args.r_z
    if args.q_vel_contact is not None:
        ekf_cfg.q_vel_contact = args.q_vel_contact
    if getattr(args, "no_contact_aware", False):
        ekf_cfg.contact_aware = False

    enable_imu = not getattr(args, "no_imu", False)
    enable_spin = getattr(args, "enable_spin", False)

    _print(f"\nEKF config: q_pos={ekf_cfg.q_pos}, q_vel={ekf_cfg.q_vel}, "
           f"q_vel_contact={ekf_cfg.q_vel_contact}, "
           f"r_xy={ekf_cfg.r_xy}, r_z={ekf_cfg.r_z}, "
           f"contact_aware={ekf_cfg.contact_aware}, "
           f"imu={enable_imu}, spin={enable_spin}")

    # --- Build noise config + env ---
    noise_cfg = BallObsNoiseCfg(
        mode="ekf", ekf_cfg=ekf_cfg,
        enable_imu=enable_imu, enable_spin=enable_spin,
    )

    env_cfg = BallJuggleHierEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"

    pi2_path = os.path.abspath(_pi2_path)
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
    base_env = env.unwrapped
    base_env._perception_diagnostics_enabled = True
    if hasattr(base_env, "_perception_pipeline"):
        base_env._perception_pipeline = None  # force recreation with diag enabled

    # Set fixed target height for all envs
    target_h = args.target_height
    sigma = target_h / args.sigma_ratio
    base_env._target_apex_heights = torch.full(
        (args.num_envs,), target_h, device=base_env.device, dtype=torch.float32,
    )
    base_env._target_apex_sigmas = torch.full(
        (args.num_envs,), sigma, device=base_env.device, dtype=torch.float32,
    )
    # Also update reset event params
    for term_cfg in base_env.event_manager._mode_term_cfgs.get("reset", []):
        if hasattr(term_cfg, "params") and "target_min" in (term_cfg.params or {}):
            term_cfg.params["target_min"] = target_h
            term_cfg.params["target_max"] = target_h

    # Wrap for RSL-RL policy loading
    env_wrapped = RslRlVecEnvWrapper(env)

    # --- Load pi1 policy ---
    pi1_path = os.path.abspath(args.pi1_checkpoint)
    if not os.path.isfile(pi1_path):
        raise FileNotFoundError(f"pi1 checkpoint not found: {pi1_path}")

    _print(f"Loading pi1 checkpoint: {pi1_path}")

    # Load agent config from the run directory
    run_dir = os.path.dirname(pi1_path)
    agent_cfg_path = os.path.join(run_dir, "params", "agent.yaml")
    if os.path.isfile(agent_cfg_path):
        from omegaconf import OmegaConf
        agent_dict = OmegaConf.to_container(OmegaConf.load(agent_cfg_path), resolve=True)
    else:
        # Fallback: use default agent config
        from go1_ball_balance.agents.rsl_rl_ppo_cfg import (
            BallJuggleHierPPORunnerCfg,
        )
        agent_dict = BallJuggleHierPPORunnerCfg().to_dict()

    runner = OnPolicyRunner(env_wrapped, agent_dict, log_dir=None, device="cuda:0")
    runner.load(pi1_path)
    policy = runner.get_inference_policy(device=base_env.device)

    _print(f"\nTarget height: {target_h:.2f}m  σ={sigma:.3f}m")
    _print(f"Running: {args.num_envs} envs × {args.steps} steps with trained policy")

    obs = env_wrapped.get_observations()

    # --- Tracking ---
    all_nis = []
    all_nis_flight = []
    all_nis_contact = []
    total_gate_rejected = 0
    total_gate_total = 0
    total_timeouts = 0
    total_episodes = 0

    _print(f"\n{'='*100}")
    _print(f"  {'Step':>6s}  {'NIS':>8s}  {'Flight':>8s}  {'Contact':>8s}  "
           f"{'EKF mm':>8s}  {'Raw mm':>8s}  {'Impr%':>7s}  {'Det%':>6s}  "
           f"{'Gate%':>6s}  {'TO%':>6s}  {'EpLen':>7s}")
    _print(f"  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  "
           f"{'-'*8}  {'-'*8}  {'-'*7}  {'-'*6}  "
           f"{'-'*6}  {'-'*6}  {'-'*7}")

    step_ep_lens = torch.zeros(args.num_envs, device=base_env.device)
    interval_timeouts = 0
    interval_episodes = 0

    for step in range(1, args.steps + 1):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, infos = env_wrapped.step(actions)

        step_ep_lens += 1

        if dones.any():
            done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
            time_outs = infos.get("time_outs", torch.zeros_like(dones))
            for idx in done_ids:
                total_episodes += 1
                interval_episodes += 1
                if time_outs[idx].item():
                    total_timeouts += 1
                    interval_timeouts += 1
            step_ep_lens[done_ids] = 0

        if step % args.log_interval == 0:
            pipeline = getattr(base_env, "_perception_pipeline", None)
            if pipeline is None:
                _print(f"  {step:6d}  PIPELINE NOT FOUND")
                continue

            diag = pipeline.diagnostics
            if diag is None:
                _print(f"  {step:6d}  DIAGNOSTICS DISABLED")
                continue

            nis = diag.get("mean_nis", 0.0)
            all_nis.append(nis)

            nis_flight = diag.get("mean_nis_flight", 0.0)
            nis_contact = diag.get("mean_nis_contact", 0.0)
            if nis_flight > 0:
                all_nis_flight.append(nis_flight)
            if nis_contact > 0:
                all_nis_contact.append(nis_contact)

            gate_rate = diag.get("gate_rejection_rate", 0.0)
            gate_count = diag.get("gate_rejected", 0)
            total_gate_rejected += gate_count
            total_gate_total += diag.get("gate_total", 0)

            to_pct = (100.0 * interval_timeouts / interval_episodes
                      if interval_episodes > 0 else 0.0)
            mean_ep_len = step_ep_lens.mean().item()

            _print(f"  {step:6d}  {nis:8.3f}  {nis_flight:8.3f}  {nis_contact:8.3f}  "
                   f"{diag.get('pos_rmse_ekf_mm', 0):8.2f}  "
                   f"{diag.get('pos_rmse_raw_mm', 0):8.2f}  "
                   f"{diag.get('ekf_improvement_pct', 0):7.1f}  "
                   f"{diag.get('detection_rate', 0)*100:6.1f}  "
                   f"{gate_rate*100:6.2f}  "
                   f"{to_pct:5.1f}%  "
                   f"{mean_ep_len:7.1f}")

            interval_timeouts = 0
            interval_episodes = 0

    # --- Summary ---
    _print(f"\n{'='*100}")
    _print(f"  PERCEPTION ACCURACY UNDER TRAINED POLICY")
    _print(f"  Target: {target_h:.2f}m  |  Policy: {os.path.basename(pi1_path)}")
    _print(f"  Episodes: {total_episodes}  |  Timeout: {100*total_timeouts/max(total_episodes,1):.1f}%")
    _print(f"{'='*100}")

    if all_nis:
        mean_nis = sum(all_nis) / len(all_nis)
        in_band = sum(1 for n in all_nis if 0.35 <= n <= 7.81)
        _print(f"\n  Overall NIS: {mean_nis:.3f}  (target ≈ 3.0, in-band: {in_band}/{len(all_nis)})")

        if all_nis_flight:
            mean_f = sum(all_nis_flight) / len(all_nis_flight)
            _print(f"  Flight NIS:  {mean_f:.3f}  (q_vel={ekf_cfg.q_vel})")
            if mean_f < 1.0:
                _print(f"    → Over-conservative: reduce q_vel")
            elif mean_f > 5.0:
                _print(f"    → Overconfident: increase q_vel")
            else:
                _print(f"    → Well-tuned")

        if all_nis_contact:
            mean_c = sum(all_nis_contact) / len(all_nis_contact)
            _print(f"  Contact NIS: {mean_c:.3f}  (q_vel_contact={ekf_cfg.q_vel_contact})")
            if mean_c < 1.0:
                _print(f"    → Over-conservative: reduce q_vel_contact")
            elif mean_c > 5.0:
                _print(f"    → Overconfident: increase q_vel_contact")
            else:
                _print(f"    → Well-tuned")

        overall_gate_rate = (total_gate_rejected / total_gate_total * 100
                             if total_gate_total > 0 else 0)
        _print(f"\n  Gate rejections: {total_gate_rejected}/{total_gate_total} ({overall_gate_rate:.2f}%)")

        # Last interval's RMSE for final summary
        if pipeline is not None and pipeline.diagnostics is not None:
            d = pipeline.diagnostics
            _print(f"  EKF RMSE: {d.get('pos_rmse_ekf_mm', 0):.2f} mm  "
                   f"(raw: {d.get('pos_rmse_raw_mm', 0):.2f} mm, "
                   f"improvement: {d.get('ekf_improvement_pct', 0):.1f}%)")
            _print(f"  Detection rate: {d.get('detection_rate', 0)*100:.1f}%")
    else:
        _print("\n  No NIS data collected!")

    _print()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
