"""Train learned pi1 for hierarchical ball juggling.

Pi1 outputs a 6D normalized torso command directly to frozen pi2.
The RL policy learns to juggle by discovering good roll/pitch/h_dot patterns.

No curriculum needed — the env provides graded difficulty via episode resets.
Ball bouncing reward penalises cradling and drives active juggling.

Usage:
    uv run --active python scripts/rsl_rl/train_pi1.py \\
        --task Isaac-BallJugglePi1-Go1-v0 \\
        --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/TIMESTAMP/model_best.pt \\
        --num_envs 4096 --headless
"""

import argparse
import os
import sys
import time
from datetime import datetime

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train learned pi1 for ball juggling.")
parser.add_argument("--task",            type=str,   default="Isaac-BallJugglePi1-Go1-v0")
parser.add_argument("--pi2_checkpoint",  type=str,   required=True,
                    help="Path to frozen pi2 (torso-tracking) checkpoint .pt")
parser.add_argument("--resume_checkpoint", type=str, default=None,
                    help="Path to a pi1 checkpoint .pt to warm-start actor weights from")
parser.add_argument("--num_envs",        type=int,   default=None)
parser.add_argument("--max_iterations",  type=int,   default=None)
parser.add_argument("--seed",            type=int,   default=None)
parser.add_argument("--video",          action="store_true", default=False)
parser.add_argument("--video_length",   type=int,   default=200)
parser.add_argument("--video_interval", type=int,   default=2000)
parser.add_argument("--agent",          type=str,   default="rsl_rl_cfg_entry_point")
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
    """Train pi1 with RSL-RL PPO."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed
        agent_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # Inject frozen pi2 checkpoint path
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
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    # Warm-start from a previous checkpoint (actor only — critic re-initializes fresh)
    if args_cli.resume_checkpoint:
        resume_path = os.path.abspath(args_cli.resume_checkpoint)
        print(f"[INFO] Loading actor weights from: {resume_path}")
        loaded = torch.load(resume_path, map_location=agent_cfg.device)
        # Load only actor_critic weights; skip optimizer so critic re-learns from scratch
        runner.alg.policy.load_state_dict(loaded["model_state_dict"])
        print("[INFO] Weights loaded. Optimizer re-initialized (new reward scale).")

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Install model_best saving
    _install_best_checkpoint(runner, log_dir)

    start_time = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"Training time: {round(time.time() - start_time, 2)} seconds")
    env.close()


def _install_best_checkpoint(runner, log_dir: str) -> None:
    """Save model_best.pt whenever reward improves."""
    original_log = runner.log
    state = {"best_reward": float("-inf")}

    def _wrapped_log(locs, *args, **kwargs):
        original_log(locs, *args, **kwargs)
        import statistics
        rew_buf = locs.get("rewbuffer", [])
        if rew_buf:
            mean_reward = statistics.mean(rew_buf)
            if mean_reward > state["best_reward"] + 0.5:
                state["best_reward"] = mean_reward
                best_path = os.path.join(log_dir, "model_best.pt")
                runner.save(best_path)
                print(f"[PI1-TRAIN] New best reward: {mean_reward:.1f} → saved {best_path}")

    runner.log = _wrapped_log
    print("[PI1-TRAIN] model_best saving installed.")


if __name__ == "__main__":
    main()
    simulation_app.close()
