# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import signal
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-BallBalance-Go1-Play-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--slow",
    type=float,
    default=None,
    metavar="N",
    help="Slow-motion multiplier: sleep N × the policy step duration per step (e.g. --slow 4 = 4× slower than real-time).",
)
# 9D pi2: user locomotion commands (vx, vy, omega_yaw) written into env._user_cmd_vel
parser.add_argument(
    "--user-cmd",
    choices=["none", "keyboard", "fixed"],
    default="none",
    help="How to drive user locomotion commands (vx/vy/yaw). 'keyboard' uses Se2Keyboard "
    "(arrow keys / Z/X for yaw); 'fixed' uses --user-vx/vy/yaw CLI values; 'none' keeps zero.",
)
parser.add_argument("--user-vx", type=float, default=0.0,
                    help="Normalized [-1,1] forward velocity (0.5 m/s at ±1). --user-cmd must be 'fixed'.")
parser.add_argument("--user-vy", type=float, default=0.0,
                    help="Normalized [-1,1] lateral velocity. --user-cmd must be 'fixed'.")
parser.add_argument("--user-yaw", type=float, default=0.0,
                    help="Normalized [-1,1] yaw rate (1.5 rad/s at ±1). --user-cmd must be 'fixed'.")
parser.add_argument("--target-apex", type=float, default=None,
                    help="Override juggle apex target [m above paddle] — used by hier / mirror-law tasks.")
# --record-traj and --record-steps are stripped from sys.argv BEFORE argparse/Hydra
# ever see them.  Reason: argparse's parse_known_args() can occasionally leak the
# VALUE of a hyphenated flag (e.g. "/tmp/traj.npz") into hydra_args.  Hydra then
# tries to parse "/tmp/traj.npz" as a config override and throws
# LexerNoViableAltException because "/" is not a valid character in its grammar.
# Pre-stripping avoids this.
_record_traj_path = None
_record_steps = 600          # default: 3 s at 200 Hz physics rate; override with --record-steps N
_pi2_checkpoint_path = None  # for hierarchical tasks: path to frozen pi2 checkpoint
_clean_argv = []
_i = 0
while _i < len(sys.argv):
    if sys.argv[_i] in ("--record-traj", "--record_traj") and _i + 1 < len(sys.argv):
        _record_traj_path = sys.argv[_i + 1]
        _i += 2
    elif sys.argv[_i] in ("--record-steps", "--record_steps") and _i + 1 < len(sys.argv):
        _record_steps = int(sys.argv[_i + 1])
        _i += 2
    elif sys.argv[_i] in ("--pi2-checkpoint", "--pi2_checkpoint") and _i + 1 < len(sys.argv):
        _pi2_checkpoint_path = sys.argv[_i + 1]
        _i += 2
    else:
        _clean_argv.append(sys.argv[_i])
        _i += 1
sys.argv = _clean_argv

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
import go1_ball_balance  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # Slow-motion / trajectory-recording setup: reduce decimation to 1 so each
    # env.step() advances exactly one 5 ms physics timestep.  The policy still runs
    # at its training frequency — we call env.step() original_decimation times per
    # policy update, holding the action constant across sub-steps.
    if args_cli.slow is not None or _record_traj_path is not None:
        _n_sub_steps = env_cfg.decimation   # typically 4
        env_cfg.decimation = 1
    else:
        _n_sub_steps = 1

    # Inject pi2 checkpoint for hierarchical tasks
    if _pi2_checkpoint_path is not None:
        if hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "torso_cmd"):
            env_cfg.actions.torso_cmd.pi2_checkpoint = os.path.abspath(_pi2_checkpoint_path)
            print(f"[INFO] pi2 checkpoint: {env_cfg.actions.torso_cmd.pi2_checkpoint}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Enable smooth command interpolation for torso-tracking play
    if hasattr(env_cfg, "_torso_smooth_enabled") and env_cfg._torso_smooth_enabled:
        env.unwrapped._torso_smooth_enabled = True

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    sub_dt = env.unwrapped.step_dt          # 5 ms when decimation=1, 20 ms otherwise
    dt = sub_dt * _n_sub_steps              # true policy-step duration (always ~20 ms)
    _FRAME_S = 1.0 / 60.0                   # display cadence between sub-steps

    # trajectory recording buffer (populated inside the loop when --record-traj is set)
    _rec = _record_traj_path is not None
    _buf: dict = {}
    if _rec:
        _buf = {k: [] for k in (
            "root_pos", "root_quat", "root_lin_vel", "root_ang_vel",
            "joint_pos", "joint_vel",
            "ball_pos", "ball_quat", "ball_lin_vel",
        )}
        print(f"[INFO] Recording trajectory to: {_record_traj_path}  "
              f"(physics_dt={sub_dt*1000:.2f} ms, every sub-step, "
              f"limit={_record_steps} frames = {_record_steps * sub_dt:.1f}s)")

    # Ctrl-C handler: set a flag so the while loop exits cleanly and the save runs.
    # A second Ctrl-C restores the default handler (hard kill).
    _stop = False
    if _rec:
        def _on_sigint(sig, frame):
            nonlocal _stop
            _stop = True
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            print("\n[INFO] Ctrl-C caught — finishing current step then saving. "
                  "Press Ctrl-C again to force-quit (data may be lost).")
        signal.signal(signal.SIGINT, _on_sigint)

    # ── User-command injection (9D pi2 hierarchy) ───────────────────────────
    # _user_cmd_vel is a (N, 3) tensor written by TorsoCommandAction.
    # Keyboard and fixed modes both feed into the same buffer, overwriting any
    # value that ball_juggle_hier randomization would otherwise drive.
    _uw = env.unwrapped
    _user_keyboard = None
    _user_fixed = None
    if args_cli.user_cmd == "keyboard":
        try:
            from isaaclab.devices.keyboard import Se2Keyboard, Se2KeyboardCfg
            _user_keyboard = Se2Keyboard(Se2KeyboardCfg(
                v_x_sensitivity=1.0,
                v_y_sensitivity=1.0,
                omega_z_sensitivity=1.0,
                sim_device=_uw.device,
            ))
            print("[USER-CMD] keyboard mode ON")
            print(str(_user_keyboard))
        except Exception as e:
            print(f"[USER-CMD] keyboard init failed: {e}; falling back to 'none'")
    elif args_cli.user_cmd == "fixed":
        _user_fixed = torch.tensor(
            [args_cli.user_vx, args_cli.user_vy, args_cli.user_yaw],
            device=_uw.device,
        ).clamp(-1.0, 1.0)
        print(f"[USER-CMD] fixed mode: vx={_user_fixed[0]:.2f} vy={_user_fixed[1]:.2f} yaw={_user_fixed[2]:.2f} (normalized)")

    # Target-apex override (hier/mirror-law juggling tasks)
    _target_apex_override = args_cli.target_apex
    if _target_apex_override is not None:
        print(f"[USER-CMD] target apex override: {_target_apex_override:.3f} m above paddle")

    def _apply_user_commands():
        """Write user commands into env buffers every step.

        Two paths:
        - hier task: `_user_cmd_vel` exists (lazy-created by TorsoCommandAction);
          process_actions() concats it into _torso_cmd[:, 6:9] next step.
        - torso-tracking task (pi2-only): no TorsoCommandAction; the 9D
          `_torso_cmd` is populated by resample events. We overwrite the
          last 3 dims directly so keyboard/fixed commands take precedence
          over random resampling.
        """
        cmd = None
        if _user_keyboard is not None:
            cmd = _user_keyboard.advance().clamp(-1.0, 1.0)
        elif _user_fixed is not None:
            cmd = _user_fixed
        if cmd is not None:
            if hasattr(_uw, "_user_cmd_vel"):
                _uw._user_cmd_vel[:, 0] = cmd[0]
                _uw._user_cmd_vel[:, 1] = cmd[1]
                _uw._user_cmd_vel[:, 2] = cmd[2]
            elif hasattr(_uw, "_torso_cmd") and _uw._torso_cmd.shape[-1] >= 9:
                # Torso-tracking task: write both _torso_cmd and _torso_cmd_goal so
                # update_torso_commands_smooth() doesn't drift the user command
                # back toward the resampled random goal.
                _uw._torso_cmd[:, 6] = cmd[0]
                _uw._torso_cmd[:, 7] = cmd[1]
                _uw._torso_cmd[:, 8] = cmd[2]
                if hasattr(_uw, "_torso_cmd_goal"):
                    _uw._torso_cmd_goal[:, 6] = cmd[0]
                    _uw._torso_cmd_goal[:, 7] = cmd[1]
                    _uw._torso_cmd_goal[:, 8] = cmd[2]
        if _target_apex_override is not None and hasattr(_uw, "_target_apex_heights"):
            _uw._target_apex_heights[:] = _target_apex_override

    # reset environment
    obs = env.get_observations()
    _apply_user_commands()
    timestep = 0
    # simulate environment
    while simulation_app.is_running() and not _stop:
        start_time = time.time()
        # refresh user commands for this step (keyboard reads latest key state)
        _apply_user_commands()
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)
            for _ in range(_n_sub_steps):
                sub_start = time.time()
                # env.step() advances one physics sub-step and renders it
                obs, _, dones, _ = env.step(actions)
                # reset recurrent states for episodes that have terminated
                policy_nn.reset(dones)

                # record state at every physics sub-step
                if _rec:
                    _scene = env.unwrapped.scene
                    _robot = _scene["robot"]
                    _ball  = _scene["ball"]
                    _orig  = _scene.env_origins          # (N_envs, 3)
                    _buf["root_pos"].append((_robot.data.root_pos_w - _orig).cpu().numpy().copy())
                    _buf["root_quat"].append(_robot.data.root_quat_w.cpu().numpy().copy())
                    _buf["root_lin_vel"].append(_robot.data.root_lin_vel_w.cpu().numpy().copy())
                    _buf["root_ang_vel"].append(_robot.data.root_ang_vel_w.cpu().numpy().copy())
                    _buf["joint_pos"].append(_robot.data.joint_pos.cpu().numpy().copy())
                    _buf["joint_vel"].append(_robot.data.joint_vel.cpu().numpy().copy())
                    _buf["ball_pos"].append((_ball.data.root_pos_w - _orig).cpu().numpy().copy())
                    _buf["ball_quat"].append(_ball.data.root_quat_w.cpu().numpy().copy())
                    _buf["ball_lin_vel"].append(_ball.data.root_lin_vel_w.cpu().numpy().copy())
                    if len(_buf["root_pos"]) >= _record_steps:
                        break   # exit inner for-loop; outer while checks the limit below

                # slow-motion: pad each sub-step to the target wall-clock duration,
                # filling the gap with additional render calls for a smooth display
                if args_cli.slow is not None:
                    deadline = sub_start + sub_dt * args_cli.slow
                    while time.time() < deadline:
                        frame_start = time.time()
                        simulation_app.update()
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            break
                        sleep_s = min(_FRAME_S - (time.time() - frame_start), remaining)
                        if sleep_s > 0:
                            time.sleep(sleep_s)

        timestep += 1

        # Torso-tracking telemetry: print commanded vs actual values for env 0
        _uw = env.unwrapped
        if hasattr(_uw, "_torso_cmd") and timestep % 20 == 0:
            import isaaclab.utils.math as _math_utils
            _robot = _uw.scene["robot"]
            _cmd = _uw._torso_cmd[0].cpu()
            _z = _robot.data.root_pos_w[0, 2].item()
            _zd = _robot.data.root_lin_vel_w[0, 2].item()
            _r, _p, _ = _math_utils.euler_xyz_from_quat(_robot.data.root_quat_w[0:1])
            _wr = _robot.data.root_ang_vel_b[0, 0].item()
            _wp = _robot.data.root_ang_vel_b[0, 1].item()
            _vx_b = _robot.data.root_lin_vel_b[0, 0].item()
            _vy_b = _robot.data.root_lin_vel_b[0, 1].item()
            _wy_b = _robot.data.root_ang_vel_b[0, 2].item()
            print(
                f"[TORSO] h={_cmd[0]:.2f}→{_z:.2f}  hd={_cmd[1]:+.1f}→{_zd:+.1f}  "
                f"r={_cmd[2]:+.2f}→{_r[0]:+.2f}  p={_cmd[3]:+.2f}→{_p[0]:+.2f}  "
                f"wr={_cmd[4]:+.1f}→{_wr:+.1f}  wp={_cmd[5]:+.1f}→{_wp:+.1f}  "
                f"| vx={_cmd[6]:+.2f}→{_vx_b:+.2f}  vy={_cmd[7]:+.2f}→{_vy_b:+.2f}  "
                f"wy={_cmd[8]:+.1f}→{_wy_b:+.1f}"
            )

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # trajectory recording: auto-exit after the first episode ends
        # (dones fires when any env times out or terminates; with num_envs=1
        # this is exactly one clean episode, with no cross-episode teleports)
        if _rec and dones.any():
            n_frames = len(_buf["root_pos"])
            print(f"[INFO] Episode complete ({n_frames} frames recorded). "
                  f"Saving and exiting ...")
            break

        # trajectory recording: auto-exit after hitting the step limit
        if _rec and len(_buf["root_pos"]) >= _record_steps:
            print(f"[INFO] Step limit reached ({_record_steps} frames = "
                  f"{_record_steps * sub_dt:.1f}s). Saving and exiting ...")
            break

        # real-time sleep (non-slow mode only)
        if args_cli.slow is None:
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

    # save trajectory if requested
    if _rec and _buf.get("root_pos"):
        save_path = _record_traj_path
        np.savez_compressed(
            save_path,
            **{k: np.stack(v) for k, v in _buf.items()},
            physics_dt=np.float64(sub_dt),
        )
        T  = len(_buf["root_pos"])
        N  = _buf["root_pos"][0].shape[0]
        print(f"[INFO] Trajectory saved → {save_path}")
        print(f"       {T} frames × {N} envs  |  physics_dt={sub_dt*1000:.2f} ms  "
              f"|  {T/N:.0f} frames per env  |  {T*sub_dt/N:.1f}s per env")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
