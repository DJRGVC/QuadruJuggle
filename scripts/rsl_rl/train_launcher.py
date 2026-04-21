"""Train V4 launcher pi1 for hybrid ball juggling.

The launcher pi1 has a single job: get the ball to exactly [target ± 0.05 m]
above the paddle in as few bounces as possible, then stop.
Mirror law takes over once the handoff happens.

Key differences from train_pi1.py:
  - Task: Isaac-BallJuggleLauncher-Go1-v0 (tighter apex reward, overshoot penalty,
    success termination)
  - experiment_name: go1_ball_launcher (separate log dir from V3 pi1)
  - Supports --resume_checkpoint to warm-start from existing pi1 weights

Usage:
    cd /home/frank/berkeley_mde/QuadruJuggle
    PYTHONPATH=/home/frank/IsaacLab/scripts/reinforcement_learning/rsl_rl:$PYTHONPATH \\
    python scripts/rsl_rl/train_launcher.py \\
        --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \\
        --num_envs 4096 --headless

    # Warm-start from existing V3 pi1 (recommended — saves ~30% training time):
    python scripts/rsl_rl/train_launcher.py \\
        --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \\
        --resume_checkpoint logs/rsl_rl/go1_ball_juggle_hier/2026-03-22_19-46-52/model_best.pt \\
        --num_envs 4096 --headless
"""

import argparse
import os
import sys
import time
from datetime import datetime

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train V4 launcher pi1.")
parser.add_argument("--task",             type=str, default="Isaac-BallJuggleLauncher-Go1-v0")
parser.add_argument("--pi2_checkpoint",   type=str, required=True,
                    help="Path to frozen pi2 (torso-tracking) checkpoint .pt")
parser.add_argument("--resume_checkpoint", type=str, default=None,
                    help="Warm-start actor from this .pt checkpoint (e.g. V3 pi1 model_best.pt)")
parser.add_argument("--num_envs",         type=int, default=None)
parser.add_argument("--max_iterations",   type=int, default=None)
parser.add_argument("--seed",             type=int, default=None)
parser.add_argument("--video",           action="store_true", default=False)
parser.add_argument("--video_length",    type=int, default=200)
parser.add_argument("--video_interval",  type=int, default=2000)
parser.add_argument("--agent",           type=str, default="rsl_rl_cfg_entry_point")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import isaaclab_tasks  # noqa: F401
import go1_ball_balance  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Train launcher pi1 with RSL-RL PPO."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Inject frozen pi2 checkpoint
    env_cfg.actions.torso_cmd.pi2_checkpoint = os.path.abspath(args_cli.pi2_checkpoint)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging to: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, _build_runner_cfg(agent_cfg), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    # Warm-start actor from V3 pi1 or any previous checkpoint
    if args_cli.resume_checkpoint:
        resume_path = os.path.abspath(args_cli.resume_checkpoint)
        print(f"[INFO] Warm-starting actor from: {resume_path}")
        loaded = torch.load(resume_path, map_location=agent_cfg.device)
        actor = runner.alg.actor if hasattr(runner.alg, "actor") else runner.alg.policy
        key = "actor_state_dict" if "actor_state_dict" in loaded else "model_state_dict"
        actor.load_state_dict(loaded[key], strict=False)
        print(f"[INFO] Actor weights loaded from key '{key}'. Critic re-initializes fresh (new reward scale).")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    _install_best_checkpoint(runner, log_dir)
    _install_std_clamp(runner)

    start_time = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


def _build_runner_cfg(agent_cfg) -> dict:
    """Convert agent config to dict, stripping deprecated rsl_rl < 5.0 fields."""
    _deprecated = {"stochastic", "init_noise_std", "noise_std_type", "state_dependent_std"}
    cfg = agent_cfg.to_dict()
    for model_key in ("actor", "critic"):
        if isinstance(cfg.get(model_key), dict):
            for k in _deprecated:
                cfg[model_key].pop(k, None)
    return cfg


def _install_std_clamp(runner, min_std: float = 0.02) -> None:
    """Clamp actor distribution std >= min_std after every gradient step."""
    actor = runner.alg.actor if hasattr(runner.alg, "actor") else runner.alg.policy
    dist = getattr(actor, "distribution", None)
    if dist is None:
        print("[LAUNCHER-TRAIN] No distribution found on actor — std clamp skipped.")
        return

    std_param = getattr(dist, "std_param", None)
    if std_param is None:
        std_param = getattr(dist, "log_std_param", None)
    if std_param is None:
        print("[LAUNCHER-TRAIN] No std_param found — std clamp skipped.")
        return

    original_update = runner.alg.update

    def _safe_update(*args, **kwargs):
        result = original_update(*args, **kwargs)
        with torch.no_grad():
            std_param.clamp_(min=min_std)
        return result

    runner.alg.update = _safe_update
    print(f"[LAUNCHER-TRAIN] std clamp installed (min={min_std}).")


def _install_best_checkpoint(runner, log_dir: str) -> None:
    """Save model_best.pt whenever mean episode reward improves."""
    original_log = runner.logger.log
    state = {"best_reward": float("-inf")}

    def _wrapped_log(*args, **kwargs):
        original_log(*args, **kwargs)
        rew_buf = runner.logger.rewbuffer
        if len(rew_buf) > 0:
            import statistics
            mean_reward = statistics.mean(rew_buf)
            if mean_reward > state["best_reward"] + 0.5:
                state["best_reward"] = mean_reward
                best_path = os.path.join(log_dir, "model_best.pt")
                runner.save(best_path)
                print(f"[LAUNCHER-TRAIN] New best reward: {mean_reward:.1f} → saved {best_path}")

    runner.logger.log = _wrapped_log
    print("[LAUNCHER-TRAIN] model_best saving installed.")


if __name__ == "__main__":
    main()
    simulation_app.close()
