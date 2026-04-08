#!/usr/bin/env python3
"""Compare perception modes (oracle vs d435i vs ekf) on ball_juggle_hier.

Each mode runs in a SEPARATE subprocess to avoid Isaac Lab's inability to
cleanly re-create simulation envs within a single process.

Usage:
    $C3R_BIN/gpu_lock.sh uv run --active python scripts/perception/compare_perception_modes.py \
        --pi2-checkpoint logs/rsl_rl/go1_torso_tracking/<run>/model_best.pt \
        --num_envs 4096 --headless --max_iterations 50

    # Specific modes only:
    ... --modes ekf d435i

    # Single-mode (called by subprocess, not by user):
    ... --single-mode oracle
"""

import argparse
import json
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Compare perception modes.")
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=50)
    parser.add_argument(
        "--pi2-checkpoint", type=str, required=True,
        help="Path to the trained pi2 (torso-tracking) checkpoint.",
    )
    parser.add_argument(
        "--modes", nargs="+", default=["oracle", "d435i", "ekf"],
        help="Perception modes to compare.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--single-mode", type=str, default=None,
        help="(internal) Run a single mode and write results to JSON.",
    )
    args = parser.parse_args()

    results_dir = os.path.join("logs", "rsl_rl", "perception_compare")
    os.makedirs(results_dir, exist_ok=True)

    if args.single_mode:
        # ── Single-mode subprocess: run training, write results ──
        _run_single_mode(args)
        return

    # ── Multi-mode orchestrator: spawn subprocesses ──
    all_results = {}
    for mode in args.modes:
        print(f"\n{'='*60}")
        print(f"  Launching subprocess for mode: {mode}")
        print(f"{'='*60}\n")

        result_file = os.path.join(results_dir, f"{mode}_result.json")
        cmd = [
            sys.executable, __file__,
            "--pi2-checkpoint", args.pi2_checkpoint,
            "--num_envs", str(args.num_envs),
            "--max_iterations", str(args.max_iterations),
            "--single-mode", mode,
        ]
        if args.headless:
            cmd.append("--headless")

        proc = subprocess.run(cmd, timeout=1800)  # 30 min per mode max
        if proc.returncode != 0:
            print(f"  [ERROR] Mode {mode} failed with return code {proc.returncode}")
            all_results[mode] = {"mode": mode, "error": f"exit code {proc.returncode}"}
            continue

        if os.path.isfile(result_file):
            with open(result_file) as f:
                all_results[mode] = json.load(f)
        else:
            all_results[mode] = {"mode": mode, "error": "no result file"}

    # Print comparison table
    _print_comparison_table(all_results)

    # Save combined results
    combined_path = os.path.join(results_dir, "results.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {combined_path}")


def _run_single_mode(args):
    """Run training for one mode inside an Isaac Lab process."""
    mode = args.single_mode

    # Isaac Lab requires AppLauncher before any sim imports
    from isaaclab.app import AppLauncher

    inner_parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(inner_parser)
    launcher_args, _ = inner_parser.parse_known_args(
        ["--headless"] if args.headless else []
    )
    launcher_args.enable_cameras = False
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    import time

    import gymnasium as gym
    import torch
    from rsl_rl.runners import OnPolicyRunner

    # Worktree isolation
    _OUR_SRC = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..",
        "source", "go1_ball_balance",
    ))
    sys.path.insert(0, _OUR_SRC)

    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import go1_ball_balance  # noqa: F401 — registers gym envs

    from go1_ball_balance.perception import (
        BallObsNoiseCfg,
        ball_pos_perceived,
        ball_vel_perceived,
        reset_perception_pipeline,
    )
    from go1_ball_balance.tasks.ball_juggle_hier.ball_juggle_hier_env_cfg import (
        BallJuggleHierEnvCfg,
        _PADDLE_OFFSET_B,
    )
    from go1_ball_balance.tasks.ball_juggle_hier.agents.rsl_rl_ppo_cfg import (
        BallJuggleHierPPORunnerCfg,
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    DIAG_LOG_INTERVAL = 10

    # Build env config
    env_cfg = BallJuggleHierEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = "cuda:0"

    pi2_path = os.path.abspath(args.pi2_checkpoint)
    if not os.path.isfile(pi2_path):
        raise FileNotFoundError(f"pi2 checkpoint not found: {pi2_path}")
    env_cfg.actions.torso_cmd.pi2_checkpoint = pi2_path

    # Patch obs for perception mode
    noise_cfg = BallObsNoiseCfg(mode=mode)
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
            "noise_cfg": noise_cfg,
        },
    )
    if mode == "ekf":
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

    if mode == "ekf":
        env.unwrapped._perception_diagnostics_enabled = True

    agent_cfg = BallJuggleHierPPORunnerCfg()
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.device = "cuda:0"
    agent_cfg.experiment_name = f"perception_compare_{mode}"

    log_dir = os.path.join("logs", "rsl_rl", "perception_compare", mode)
    os.makedirs(log_dir, exist_ok=True)

    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(
        env_wrapped, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device,
    )

    # Metric collection
    metrics = {"mode": mode}
    ep_lengths = []
    reward_sums = []
    timeout_pcts = []
    perception_diags = []
    iteration_count = [0]

    original_log = runner.log

    def _metric_log(locs, *args_log, **kwargs):
        import statistics

        original_log(locs, *args_log, **kwargs)
        iteration_count[0] += 1

        # Read from RSL-RL's rewbuffer/lenbuffer (deques of per-episode values)
        lenbuf = locs.get("lenbuffer", [])
        rewbuf = locs.get("rewbuffer", [])
        if len(lenbuf) > 0:
            ep_lengths.append(statistics.mean(lenbuf))
        if len(rewbuf) > 0:
            reward_sums.append(statistics.mean(rewbuf))

        # Also capture termination breakdown from ep_infos
        ep_infos = locs.get("ep_infos", [])
        if ep_infos:
            for key in ep_infos[0]:
                if "time_out" in key.lower() or "timeout" in key.lower():
                    vals = [float(ep[key]) for ep in ep_infos if key in ep]
                    if vals:
                        timeout_pcts.append(sum(vals) / len(vals))

        if mode == "ekf" and iteration_count[0] % DIAG_LOG_INTERVAL == 0:
            pipeline = getattr(env.unwrapped, "_perception_pipeline", None)
            if pipeline is not None:
                diag = pipeline.diagnostics
                if diag:
                    diag["iteration"] = iteration_count[0]
                    perception_diags.append(diag)
                    nis_str = f", NIS: {diag['mean_nis']:.2f}" if 'mean_nis' in diag else ""
                    print(f"  [DIAG iter {iteration_count[0]}] "
                          f"EKF pos RMSE: {diag['pos_rmse_ekf_mm']:.1f}mm, "
                          f"raw: {diag['pos_rmse_raw_mm']:.1f}mm, "
                          f"improvement: {diag['ekf_improvement_pct']:.0f}%"
                          f"{nis_str}")

    runner.log = _metric_log

    start = time.time()
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
    elapsed = time.time() - start

    metrics["elapsed_s"] = round(elapsed, 1)
    if ep_lengths:
        metrics["mean_ep_len_final10"] = round(sum(ep_lengths[-10:]) / len(ep_lengths[-10:]), 1)
        metrics["mean_ep_len_all"] = round(sum(ep_lengths) / len(ep_lengths), 1)
        metrics["max_ep_len"] = round(max(ep_lengths), 1)
    if reward_sums:
        metrics["mean_reward_final10"] = round(sum(reward_sums[-10:]) / len(reward_sums[-10:]), 1)
    if timeout_pcts:
        metrics["timeout_pct_final10"] = round(100 * sum(timeout_pcts[-10:]) / len(timeout_pcts[-10:]), 1)

    if perception_diags:
        avg_diag = {}
        for key in perception_diags[0]:
            if key in ("iteration", "num_samples"):
                continue
            vals = [d[key] for d in perception_diags if key in d]
            avg_diag[f"avg_{key}"] = round(sum(vals) / len(vals), 2)
        metrics.update(avg_diag)

        diag_path = os.path.join(log_dir, "perception_diagnostics.json")
        with open(diag_path, "w") as f:
            json.dump(perception_diags, f, indent=2)
        print(f"  [DIAG] Full diagnostics saved to {diag_path}")

    env.close()

    # Write per-mode result
    result_file = os.path.join("logs", "rsl_rl", "perception_compare", f"{mode}_result.json")
    with open(result_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Mode {mode} result saved to {result_file}")

    simulation_app.close()


def _print_comparison_table(all_results):
    """Print a formatted comparison table."""
    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*70}")

    all_keys = set()
    for r in all_results.values():
        all_keys.update(r.keys())
    all_keys.discard("mode")
    all_keys.discard("error")
    sorted_keys = sorted(all_keys)

    modes = list(all_results.keys())
    header = f"  {'metric':<30s}" + "".join(f"  {m:>12s}" for m in modes)
    print(header)
    print("  " + "-" * (30 + 14 * len(modes)))

    for key in sorted_keys:
        row = f"  {key:<30s}"
        for m in modes:
            val = all_results[m].get(key, "—")
            if isinstance(val, float):
                row += f"  {val:>12.2f}"
            else:
                row += f"  {str(val):>12s}"
        print(row)

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
