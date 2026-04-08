"""Evaluate a trained pi1 (hierarchical ball-juggle) policy.

Runs the policy headless across a sweep of target apex heights and reports
timeout %, mean apex reward, and mean episode length for each.

Usage:
    uv run --active python scripts/rsl_rl/eval_juggle_hier.py \
        --task Isaac-BallJuggleHier-Go1-Play-v0 \
        --pi2-checkpoint logs/rsl_rl/go1_torso_tracking/<run>/model_best.pt \
        --checkpoint logs/rsl_rl/go1_ball_juggle_hier/<run>/model_best.pt \
        --num_envs 256 --headless
"""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

# Pre-strip --pi2-checkpoint before argparse/Hydra (same workaround as play.py)
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

parser = argparse.ArgumentParser(description="Evaluate pi1 for hierarchical ball juggling.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Isaac-BallJuggleHier-Go1-Play-v0")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--episodes", type=int, default=50,
    help="Minimum episodes to collect per target height.",
)
parser.add_argument(
    "--targets", type=str, default=None,
    help="Comma-separated target heights to evaluate (e.g. '0.30,0.50,0.70,0.86'). "
         "Defaults to all curriculum stage targets.",
)
parser.add_argument(
    "--noise-mode", type=str, default="oracle", choices=["oracle", "d435i"],
    help="Ball observation noise mode: oracle (no noise) or d435i (structured camera noise).",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import go1_ball_balance  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Default eval targets (fixed height, sigma = target/2.5)
_EVAL_TARGETS = [
    0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.36, 0.42,
    0.48, 0.55, 0.62, 0.70, 0.78, 0.86, 0.92, 1.00,
]
_SIGMA_RATIO = 2.5


def _set_apex_params(rl_env, target_h: float, sigma: float, num_envs: int) -> None:
    """Set a fixed target height for all envs (creates per-env buffers)."""
    if not hasattr(rl_env, "_target_apex_heights"):
        rl_env._target_apex_heights = torch.full(
            (num_envs,), target_h, device=rl_env.device, dtype=torch.float32,
        )
        rl_env._target_apex_sigmas = torch.full(
            (num_envs,), sigma, device=rl_env.device, dtype=torch.float32,
        )
    else:
        rl_env._target_apex_heights[:] = target_h
        rl_env._target_apex_sigmas[:] = sigma
    # Also update the event params so resets use the same fixed target
    for term_cfg in rl_env.event_manager._mode_term_cfgs.get("reset", []):
        if hasattr(term_cfg, "params") and "target_min" in (term_cfg.params or {}):
            term_cfg.params["target_min"] = target_h
            term_cfg.params["target_max"] = target_h


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.observations.policy.enable_corruption = False

    # Inject pi2 checkpoint
    if _pi2_checkpoint_path is not None and hasattr(env_cfg.actions, "torso_cmd"):
        env_cfg.actions.torso_cmd.pi2_checkpoint = os.path.abspath(_pi2_checkpoint_path)

    # Inject noise mode
    if args_cli.noise_mode != "oracle":
        from go1_ball_balance.perception.ball_obs_spec import BallObsNoiseCfg
        noise_cfg = BallObsNoiseCfg(mode=args_cli.noise_mode)
        env_cfg.observations.policy.ball_pos.params["noise_cfg"] = noise_cfg
        env_cfg.observations.policy.ball_vel.params["noise_cfg"] = noise_cfg

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))

    # Resolve pi1 checkpoint
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[EVAL] Loading checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    rl_env = env.unwrapped

    # Determine target heights to evaluate
    if args_cli.targets:
        targets = [(float(t.strip()), float(t.strip()) / _SIGMA_RATIO)
                    for t in args_cli.targets.split(",")]
    else:
        targets = [(t, t / _SIGMA_RATIO) for t in _EVAL_TARGETS]

    min_episodes = args_cli.episodes
    max_steps_per_target = min_episodes * 1600  # generous upper bound

    print(f"\n{'='*80}")
    print(f"  HIERARCHICAL JUGGLE EVALUATION — {len(targets)} target heights, {min_episodes} episodes each")
    print(f"  Envs: {args_cli.num_envs}  |  Checkpoint: {os.path.basename(resume_path)}")
    print(f"{'='*80}\n")

    all_results = []

    apex_idx = rl_env.reward_manager._term_names.index("ball_apex_height")

    for target_h, sigma in targets:
        _set_apex_params(rl_env, target_h, sigma, args_cli.num_envs)

        # Flush stale episodes: step until every env has reset at least once
        # so no episode straddles a target-height change.
        reset_seen = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=rl_env.device)
        obs = env.get_observations()
        flush_limit = 1600  # > max episode length
        for _ in range(flush_limit):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, infos = env.step(actions)
            if dones.any():
                reset_seen |= dones.bool()
            if reset_seen.all():
                break

        ep_lengths = []
        ep_timeouts = []
        ep_apex_rewards = []
        step_apex_accum = torch.zeros(args_cli.num_envs, device=rl_env.device)
        step_counts = torch.zeros(args_cli.num_envs, device=rl_env.device)

        steps = 0
        while len(ep_lengths) < min_episodes and steps < max_steps_per_target:
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, infos = env.step(actions)

            steps += 1
            step_counts += 1

            # Accumulate per-step apex reward (read from _step_reward computed during env.step)
            step_apex_accum += rl_env.reward_manager._step_reward[:, apex_idx]

            # Check for done envs
            if dones.any():
                done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                time_outs = infos.get("time_outs", torch.zeros_like(dones))

                for idx in done_ids:
                    idx_int = idx.item()
                    ep_len = step_counts[idx_int].item()
                    if ep_len > 1:  # skip degenerate resets
                        ep_lengths.append(ep_len)
                        ep_timeouts.append(time_outs[idx_int].item())
                        ep_apex_rewards.append(
                            step_apex_accum[idx_int].item() / max(ep_len, 1)
                        )

                    step_apex_accum[idx_int] = 0.0
                    step_counts[idx_int] = 0.0

        n = len(ep_lengths)
        if n == 0:
            print(f"  target={target_h:.2f}m  σ={sigma:.3f}m  |  NO EPISODES COMPLETED")
            all_results.append((target_h, sigma, 0, 0, 0, 0))
            continue

        timeout_pct = 100.0 * sum(ep_timeouts) / n
        mean_len = sum(ep_lengths) / n
        mean_apex = sum(ep_apex_rewards) / n

        letter_idx = next(
            (i for i, t in enumerate(_EVAL_TARGETS) if abs(t - target_h) < 0.005), -1
        )
        letter = chr(ord('A') + letter_idx) if 0 <= letter_idx < 16 else "?"

        print(
            f"  Stage {letter}  target={target_h:.2f}m  σ={sigma:.3f}m  |  "
            f"timeout={timeout_pct:5.1f}%  apex_rew={mean_apex:6.2f}  "
            f"mean_len={mean_len:7.1f}  episodes={n}"
        )
        all_results.append((target_h, sigma, timeout_pct, mean_apex, mean_len, n))

    # Summary table
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Target':>7s}  {'σ':>6s}  {'Timeout%':>8s}  {'Apex Rew':>8s}  {'Mean Len':>8s}  {'Episodes':>8s}")
    print(f"  {'-'*7}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")
    for target_h, sigma, to_pct, apex, ml, n in all_results:
        print(f"  {target_h:7.2f}  {sigma:6.3f}  {to_pct:7.1f}%  {apex:8.2f}  {ml:8.1f}  {n:8d}")
    print()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
