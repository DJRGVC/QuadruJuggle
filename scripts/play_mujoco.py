"""Sim-to-Sim: Run Isaac Lab pi1+pi2 policies in MuJoCo.

Validates that juggling policies transfer across physics engines.
No Isaac Lab / CUDA required — pure NumPy + PyTorch + MuJoCo.

Pipeline (matches play_launcher_hybrid.py exactly):
    MuJoCo state → 46D pi1 obs → pi1 MLP → 6D torso cmd
                 → 39D pi2 obs → pi2 MLP → 12D joint targets
                 → Go1 actuator net (TorchScript MLP) → torques → MuJoCo step

Actuator model:
    Uses the same Isaac Lab unitree_go1.pt actuator net (3-step history MLP).
    input_idx=[0,1,2], pos_scale=-1.0, vel_scale=1.0, input_order="pos_vel"
    effort_limit=23.7 N·m (same as Go1 real hardware).

Joint ordering note:
    Isaac Lab: type-grouped — hips(0-3), thighs(4-7), calves(8-11)
               [FL FR RL RR]_hip, [FL FR RL RR]_thigh, [FL FR RL RR]_calf
    MJCF:      leg-grouped  — FR(0-2), FL(3-5), RR(6-8), RL(9-11)
    Reindex built at runtime by mujoco_utils.build_reindex() — do not hardcode.

Usage:
    cd /home/frank/berkeley_mde/QuadruJuggle
    conda run -n isaaclab python scripts/play_mujoco.py \\
        --launcher_checkpoint logs/rsl_rl/go1_ball_launcher/2026-04-05_17-48-40/model_best.pt \\
        --pi2_checkpoint      logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \\
        --apex_height 0.30
"""

import argparse
import os
import time

import numpy as np
import sys, os as _os
sys.path.insert(0, _os.path.dirname(__file__))
from mujoco_utils import build_reindex
import torch
import torch.nn as nn
import mujoco
import mujoco.viewer
import imageio

# ── Constants matching Isaac Lab env cfg ──────────────────────────────────────
PADDLE_OFFSET_B   = np.array([0.0, 0.0, 0.070])   # paddle in trunk body frame
BALL_RADIUS       = 0.020                           # metres
MAX_TARGET_HEIGHT = 1.00                            # matches Isaac Lab max_target_height=1.00

# Pi1 action scaling: physical = action * scale + offset
CMD_SCALES  = np.array([0.125, 1.0, 0.4, 0.4, 3.0, 3.0])
CMD_OFFSETS = np.array([0.375, 0.0, 0.0, 0.0, 0.0, 0.0])

# Pi2 obs normalisation for torso command
OBS_NORM   = np.array([8.0, 1.0, 2.5, 2.5, 1/3, 1/3])   # 1/CMD_SCALES
OBS_OFFSET = np.array([-0.375, 0.0, 0.0, 0.0, 0.0, 0.0])

# Isaac Lab Go1 default joint positions — type-grouped order (confirmed by
# test_joint_cmd_isaaclab.py): all hips, then all thighs, then all calves.
DEFAULT_JOINT_POS_ISAAC = np.array([
     0.1, -0.1,  0.1, -0.1,   # FL_hip  FR_hip  RL_hip  RR_hip
     0.8,  0.8,  1.0,  1.0,   # FL_thigh FR_thigh RL_thigh RR_thigh
    -1.5, -1.5, -1.5, -1.5,   # FL_calf  FR_calf  RL_calf  RR_calf
])

# REINDEX and DEFAULT_JOINT_POS_MJCF are computed at runtime after model load.
# See build_reindex() in mujoco_utils.py — queried from model joint metadata.
# Placeholders here so module-level functions can reference them before main() runs.
REINDEX = None
DEFAULT_JOINT_POS_MJCF = None

# Actuator net path (same model Isaac Lab uses, locally available)
ACTUATOR_NET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../walk-these-ways/resources/actuator_nets/unitree_go1.pt"
)

# MuJoCo action scale: reduces pi2 action magnitude to compensate for physics
# mismatch (MuJoCo WTW MJCF ≠ Isaac Lab USD geometry). alpha=1.0 in Isaac Lab.
# In MuJoCo, alpha=0.20 gives 1.4° tilt (near pure-PD stability of 0.9°);
# alpha=0.40 gives 4.6° tilt which deflects the ball ~0.1m per bounce.
MUJOCO_ACTION_SCALE = 0.20

# PD gains for MuJoCo fallback (used when actuator net is unavailable/unstable)
KP_FALLBACK = 100.0
KD_FALLBACK = 3.0


# ── Policy loading ─────────────────────────────────────────────────────────────

def _build_actor(state_dict: dict, prefix: str) -> nn.Sequential:
    """Build actor MLP from RSL-RL checkpoint state dict."""
    keys = sorted(k for k in state_dict if k.startswith(prefix) and "weight" in k)
    layers = []
    for k in keys:
        out_dim, in_dim = state_dict[k].shape
        layers.append((in_dim, out_dim))

    modules = []
    for i, (in_dim, out_dim) in enumerate(layers):
        modules.append(nn.Linear(in_dim, out_dim))
        if i < len(layers) - 1:
            modules.append(nn.ELU())

    net = nn.Sequential(*modules)

    # Load weights (weight keys and corresponding bias keys)
    actor_state = {}
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        parts = k[len(prefix):].split(".")  # e.g. ["0", "weight"] or ["0", "bias"]
        idx   = int(parts[0])
        ptype = parts[1]
        actor_state[f"{idx}.{ptype}"] = v

    net.load_state_dict(actor_state)
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    return net


def load_policies(pi1_path: str, pi2_path: str):
    pi1_ckpt = torch.load(pi1_path, map_location="cpu")
    pi2_ckpt = torch.load(pi2_path, map_location="cpu")
    # rsl_rl 5.0.1: checkpoint has separate actor_state_dict with "mlp." prefix
    # rsl_rl 4.x:   checkpoint has model_state_dict with "actor." prefix
    if "actor_state_dict" in pi1_ckpt:
        pi1_sd, pi1_pfx = pi1_ckpt["actor_state_dict"], "mlp."
    else:
        pi1_sd, pi1_pfx = pi1_ckpt.get("model_state_dict", pi1_ckpt), "actor."
    if "actor_state_dict" in pi2_ckpt:
        pi2_sd, pi2_pfx = pi2_ckpt["actor_state_dict"], "mlp."
    else:
        pi2_sd, pi2_pfx = pi2_ckpt.get("model_state_dict", pi2_ckpt), "actor."
    pi1 = _build_actor(pi1_sd, pi1_pfx)
    pi2 = _build_actor(pi2_sd, pi2_pfx)
    print(f"[play_mujoco] Pi1 input dim: {pi1[0].in_features}  (key={pi1_pfx})")
    print(f"[play_mujoco] Pi2 input dim: {pi2[0].in_features}  (key={pi2_pfx})")
    return pi1, pi2


# ── Quaternion helpers ─────────────────────────────────────────────────────────

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion (w,x,y,z) → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


# ── Observation construction ───────────────────────────────────────────────────

def build_pi1_obs(data: mujoco.MjData,
                  last_action: np.ndarray,
                  apex_height_norm: float) -> np.ndarray:
    """Build 46D pi1 observation from MuJoCo state.

    Order must match Isaac Lab BallJugglePi1EnvCfg ObservationsCfg exactly:
      ball_pos_in_paddle_frame(3), ball_vel_in_paddle_frame(3),
      base_lin_vel(3), base_ang_vel(3), projected_gravity(3),
      joint_pos_rel(12), joint_vel(12), target_apex_height(1), last_action(6)
    """
    # Trunk state
    trunk_pos_w  = data.body("trunk").xpos.copy()             # (3,) world pos
    trunk_quat_w = data.body("trunk").xquat.copy()            # (4,) w,x,y,z
    R = quat_to_rot(trunk_quat_w)                              # body→world

    # Paddle position in world frame
    paddle_pos_w = trunk_pos_w + R @ PADDLE_OFFSET_B

    # Ball position in paddle frame (body frame, relative to paddle)
    ball_pos_w = data.body("ball").xpos.copy()
    ball_pos_paddle = R.T @ (ball_pos_w - paddle_pos_w)        # (3,)

    # Ball velocity in trunk (body) frame
    ball_vel_w = data.body("ball").cvel[3:6].copy()            # linear vel
    ball_vel_b = R.T @ ball_vel_w                              # (3,)

    # Trunk velocities in body frame
    trunk_linvel_w  = data.qvel[0:3].copy()
    trunk_angvel_w  = data.qvel[3:6].copy()
    trunk_linvel_b  = R.T @ trunk_linvel_w
    trunk_angvel_b  = R.T @ trunk_angvel_w

    # Projected gravity in body frame
    gravity_w = np.array([0.0, 0.0, -1.0])
    proj_gravity = R.T @ gravity_w                             # (3,)

    # Joint positions: scatter MJCF→Isaac (result[reindex]=mjcf); gather mjcf[reindex] is wrong dir.
    joint_pos_isaac = np.zeros(12)
    joint_pos_isaac[REINDEX] = data.qpos[7:19]
    joint_pos_rel = joint_pos_isaac - DEFAULT_JOINT_POS_ISAAC  # (12,)

    # Joint velocities: same scatter pattern
    joint_vel_isaac = np.zeros(12)
    joint_vel_isaac[REINDEX] = data.qvel[6:18]

    obs = np.concatenate([
        ball_pos_paddle,    # 3
        ball_vel_b,         # 3
        trunk_linvel_b,     # 3
        trunk_angvel_b,     # 3
        proj_gravity,       # 3
        joint_pos_rel,      # 12
        joint_vel_isaac,    # 12
        [apex_height_norm], # 1
        last_action,        # 6
    ])  # total: 46
    return obs.astype(np.float32)


def build_pi2_obs(torso_cmd: np.ndarray, data: mujoco.MjData) -> np.ndarray:
    """Build 39D pi2 observation from torso command + MuJoCo state.

    Order must match Isaac Lab TorsoCommandAction.process_actions exactly:
      torso_cmd_norm(6), base_lin_vel(3), base_ang_vel(3),
      projected_gravity(3), joint_pos_rel(12), joint_vel(12)
    """
    # Normalise torso command (matches action_term.py)
    torso_cmd_norm = (torso_cmd + OBS_OFFSET) * OBS_NORM      # (6,)

    trunk_quat_w = data.body("trunk").xquat.copy()
    R = quat_to_rot(trunk_quat_w)

    trunk_linvel_b = R.T @ data.qvel[0:3]
    trunk_angvel_b = R.T @ data.qvel[3:6]
    proj_gravity   = R.T @ np.array([0.0, 0.0, -1.0])

    joint_pos_isaac = np.zeros(12)
    joint_pos_isaac[REINDEX] = data.qpos[7:19]
    joint_pos_rel   = joint_pos_isaac - DEFAULT_JOINT_POS_ISAAC
    joint_vel_isaac = np.zeros(12)
    joint_vel_isaac[REINDEX] = data.qvel[6:18]

    obs = np.concatenate([
        torso_cmd_norm,  # 6
        trunk_linvel_b,  # 3
        trunk_angvel_b,  # 3
        proj_gravity,    # 3
        joint_pos_rel,   # 12
        joint_vel_isaac, # 12
    ])  # total: 39
    return obs.astype(np.float32)


# ── Actuator net (replaces simple PD) ─────────────────────────────────────────

class GoActuatorNet:
    """Wraps Isaac Lab's unitree_go1.pt actuator net with 3-step history.

    Matches Isaac Lab ActuatorNetMLPCfg:
        input_idx  = [0, 1, 2]    →  3-step pos-error and vel history
        pos_scale  = -1.0         →  input is -(target - current)
        vel_scale  = 1.0
        input_order = "pos_vel"   →  concat [pos_history, vel_history]
        effort_limit = 23.7 N·m
    """

    HISTORY_LEN  = 3
    POS_SCALE    = -1.0
    VEL_SCALE    =  1.0
    EFFORT_LIMIT = 23.7   # N·m — matches Go1 hardware
    TORQUE_SCALE = 0.93    # compensates for MuJoCo vs PhysX physics gap (~2.5x weaker response)
    def __init__(self, net_path: str, reindex: np.ndarray):
        net_path = os.path.abspath(net_path)
        self.net = torch.jit.load(net_path, map_location="cpu").eval()
        print(f"[ActuatorNet] Loaded from {net_path}")
        self._pos_hist = np.zeros((self.HISTORY_LEN, 12), dtype=np.float32)
        self._vel_hist = np.zeros((self.HISTORY_LEN, 12), dtype=np.float32)
        self._reindex  = reindex   # MJCF→Isaac: reindex[mjcf_i] = isaac_i

    def reset(self):
        self._pos_hist[:] = 0.0
        self._vel_hist[:] = 0.0

    def compute(self, targets_mjcf: np.ndarray,
                joint_pos_mjcf: np.ndarray,
                joint_vel_mjcf: np.ndarray) -> np.ndarray:
        """Return torques (MJCF order, clipped to ±23.7 N·m)."""
        pos_err_mjcf = targets_mjcf - joint_pos_mjcf

        # Reorder MJCF → Isaac before feeding net
        e_isaac = np.zeros(12, dtype=np.float32); e_isaac[self._reindex] = pos_err_mjcf
        v_isaac = np.zeros(12, dtype=np.float32); v_isaac[self._reindex] = joint_vel_mjcf

        self._pos_hist = np.roll(self._pos_hist, 1, axis=0)
        self._vel_hist = np.roll(self._vel_hist, 1, axis=0)
        self._pos_hist[0] = e_isaac
        self._vel_hist[0] = v_isaac

        pos_in = (self._pos_hist * self.POS_SCALE).T   # (12, 3)
        vel_in = (self._vel_hist * self.VEL_SCALE).T   # (12, 3)
        net_in = np.concatenate([pos_in, vel_in], axis=1)  # (12, 6)

        with torch.no_grad():
            torques_isaac = self.net(torch.from_numpy(net_in)).squeeze(-1).numpy()

        # Convert Isaac torques → MJCF order for data.ctrl, scaled for MuJoCo physics gap
        return np.clip(torques_isaac[self._reindex] * self.TORQUE_SCALE,
                       -self.EFFORT_LIMIT, self.EFFORT_LIMIT)


# ── Apex detection ─────────────────────────────────────────────────────────────

def detect_apex(prev_vz: float, curr_vz: float,
                ball_z: float, paddle_z: float):
    """Return apex height above paddle if ball just crossed vz=0 going down."""
    if prev_vz > 0.02 and curr_vz <= 0.02:
        return max(0.0, ball_z - paddle_z - BALL_RADIUS)
    return None


# ── Reset ─────────────────────────────────────────────────────────────────────

def reset_sim(model: mujoco.MjModel, data: mujoco.MjData,
              actuator=None, warmup_steps: int = 500,
              actuator_warmup: int = 20,
              kp: float = 100.0, kd: float = 3.0) -> None:
    """Reset to standing pose with ball above paddle.

    Three-phase warmup (mirrors Isaac Lab fix_root_link + write_joint_state_to_sim):
      1. PD warmup with base pinned — joints settle to default without gravity drift.
      2. Hard-set joints to exact default — jp_err_max = 0.
      3. Actuator net warmup with base pinned — fills 3-step history with near-zero errors.
    """
    mujoco.mj_resetData(model, data)

    _spawn_pos  = np.array([0.0, 0.0, 0.42])   # matches Isaac Lab init_state.pos
    _spawn_quat = np.array([1.0, 0.0, 0.0, 0.0])

    data.qpos[0:3]  = _spawn_pos
    data.qpos[3:7]  = _spawn_quat
    data.qpos[7:19] = DEFAULT_JOINT_POS_MJCF
    data.qpos[19:22] = [0.0, 0.0, 0.53]
    data.qpos[22:26] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    if actuator is None:
        return

    _ball_spawn = np.array([0.0, 0.0, 0.63])

    # Phase 1: PD warmup with base + ball pinned.
    # Without pinning, gravity pulls the robot down and the ball falls to the floor
    # during the ~2.5 s warmup, causing an immediate reset when the main loop starts.
    actuator.reset()
    for _ in range(warmup_steps):
        err = DEFAULT_JOINT_POS_MJCF - data.qpos[7:19]
        data.ctrl[:] = np.clip(kp * err - kd * data.qvel[6:18], -23.7, 23.7)
        mujoco.mj_step(model, data)
        data.qpos[0:3]  = _spawn_pos
        data.qpos[3:7]  = _spawn_quat
        data.qvel[0:6]  = 0.0
        data.qpos[19:22] = _ball_spawn   # keep ball from falling during warmup
        data.qvel[19:25] = 0.0

    # Phase 2: hard-set joints to exact default (mirrors write_joint_state_to_sim).
    data.qpos[7:19]  = DEFAULT_JOINT_POS_MJCF
    data.qpos[19:22] = _ball_spawn
    data.qvel[:]     = 0.0
    mujoco.mj_forward(model, data)

    # Phase 3: actuator net warmup — fills 3-step history with near-zero errors.
    actuator.reset()
    for _ in range(actuator_warmup):
        data.ctrl[:] = actuator.compute(
            DEFAULT_JOINT_POS_MJCF, data.qpos[7:19], data.qvel[6:18])
        mujoco.mj_step(model, data)
        data.qpos[0:3]  = _spawn_pos
        data.qpos[3:7]  = _spawn_quat
        data.qvel[0:6]  = 0.0
        data.qpos[19:22] = _ball_spawn
        data.qvel[19:25] = 0.0

    data.qpos[7:19]  = DEFAULT_JOINT_POS_MJCF
    data.qpos[19:22] = _ball_spawn
    data.qvel[:]     = 0.0
    mujoco.mj_forward(model, data)

    jp_err = np.abs(data.qpos[7:19] - DEFAULT_JOINT_POS_MJCF).max()
    print(f"  [reset_sim] trunk_z={data.body('trunk').xpos[2]:.3f}  jp_err_max={jp_err:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sim-to-Sim: Isaac Lab policies in MuJoCo")
    _root = os.path.dirname(os.path.abspath(__file__)) + "/.."
    parser.add_argument("--launcher_checkpoint",
                        default=f"{_root}/logs/rsl_rl/go1_ball_launcher/2026-04-20_11-03-50/model_best.pt",
                        help="Path to trained pi1 launcher checkpoint .pt")
    parser.add_argument("--pi2_checkpoint",
                        default=f"{_root}/logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt",
                        help="Path to frozen pi2 torso-tracking checkpoint .pt")
    parser.add_argument("--apex_height", type=float, default=0.30,
                        help="Target apex height in metres (default 0.30)")
    parser.add_argument("--max_steps", type=int, default=3000,
                        help="Steps to run (0 = run forever, default 3000)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without viewer (log only)")
    parser.add_argument("--xml", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "../mujoco/go1_juggle.xml"),
                        help="MuJoCo scene XML path")
    parser.add_argument("--action_scale", type=float, default=MUJOCO_ACTION_SCALE,
                        help=f"Scale pi2 action magnitude (default {MUJOCO_ACTION_SCALE}). "
                             "Compensates for Isaac Lab USD vs MuJoCo MJCF geometry difference. "
                             "Must be < ~0.42 for stability; use 1.0 to match Isaac Lab exactly.")
    parser.add_argument("--video", action="store_true", default=False,
                        help="Record replay as MP4. Saves to videos/mujoco_latest.mp4.")
    parser.add_argument("--video_length", type=int, default=600,
                        help="Number of policy steps to record (default 600 ≈ 12 s).")
    parser.add_argument("--video_fps", type=int, default=50,
                        help="Output video frame rate (default 50 = real-time).")
    args = parser.parse_args()

    print(f"[play_mujoco] launcher     : {args.launcher_checkpoint}")
    print(f"[play_mujoco] pi2          : {args.pi2_checkpoint}")
    print(f"[play_mujoco] target       : {args.apex_height:.2f} m")
    print(f"[play_mujoco] action_scale : {args.action_scale}")
    print(f"[play_mujoco] xml          : {os.path.abspath(args.xml)}")

    # Load policies
    pi1, pi2 = load_policies(args.launcher_checkpoint, args.pi2_checkpoint)

    # Load MuJoCo model
    xml_path = os.path.abspath(args.xml)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    # Build joint reindex from model metadata (replaces hardcoded array)
    global REINDEX, DEFAULT_JOINT_POS_MJCF
    REINDEX = build_reindex(model)
    DEFAULT_JOINT_POS_MJCF = DEFAULT_JOINT_POS_ISAAC[REINDEX]

    actuator = GoActuatorNet(ACTUATOR_NET_PATH, REINDEX)

    # Simulation parameters
    DECIMATION   = 4     # pi1/pi2 run at 50 Hz, physics at 200 Hz (0.005s × 4 = 0.02s)
    apex_norm    = args.apex_height / MAX_TARGET_HEIGHT
    last_action  = np.zeros(6, dtype=np.float32)
    last_apex    = 0.0
    prev_ball_vz = 0.0
    bounce_count = 0
    episode      = 0

    reset_sim(model, data, actuator)

    def run_loop(viewer=None, renderer=None):
        nonlocal last_action, last_apex, prev_ball_vz, bounce_count, episode

        step = 0
        ep_start = step
        ep_bounces = 0
        frames = [] if renderer is not None else None

        print(f"\n[play_mujoco] Running — {'Ctrl+C' if viewer is None else 'close window'} to stop.\n")

        while True:
            if args.max_steps > 0 and step >= args.max_steps:
                break
            if renderer is not None and step >= args.video_length:
                break

            # ── Policy step (every 4 physics steps = 50 Hz) ───────────────
            pi1_obs_np = build_pi1_obs(data, last_action, apex_norm)
            pi1_obs    = torch.from_numpy(pi1_obs_np).unsqueeze(0)  # (1, 46)

            with torch.no_grad():
                pi1_action = pi1(pi1_obs).squeeze(0).numpy()        # (6,)

            pi1_action = np.clip(pi1_action, -2.0, 2.0)
            last_action = pi1_action.copy()

            # Scale to physical torso command
            torso_cmd = pi1_action * CMD_SCALES + CMD_OFFSETS       # (6,)

            # Clamp commands to ranges pi2 can reliably track in MuJoCo:
            #   height:     [0.34, 0.41]  (MuJoCo equilibrium ~0.41m)
            #   roll/pitch: ±0.25 rad     (pi2 stable up to ~0.30 rad in MuJoCo)
            #   h_dot / rates: zeroed     (height velocity not tracked well in MuJoCo)
            # 1. Capture the 'raw' intent from the policy/controller
            raw_torso_cmd = torso_cmd.copy()

            # 2. Apply your clipping logic (with the fix for index [1])
            torso_cmd[0] = np.clip(raw_torso_cmd[0], 0.34, 0.41)
            torso_cmd[1] = np.clip(raw_torso_cmd[1], -1.0, 1.0)
            torso_cmd[2] = np.clip(raw_torso_cmd[2], -0.4, 0.4)
            torso_cmd[3] = np.clip(raw_torso_cmd[3], -0.4, 0.4)
            torso_cmd[4:6] = 0.0

            # 3. Print the comparison (formatted for readability)
            # Only print every N steps to avoid flooding the terminal
            if step % 50 == 0:
                print(f"\n--- Cmd Clipping Check (Step {step}) ---")
                # Height (index 0)
                print(f"Height | Raw: {raw_torso_cmd[0]:.3f} -> Clip: {torso_cmd[0]:.3f} {'[SATURATED]' if abs(raw_torso_cmd[0]-torso_cmd[0]) > 1e-5 else ''}")
                
                # Roll (index 2)
                print(f"Roll   | Raw: {raw_torso_cmd[2]:.3f} -> Clip: {torso_cmd[2]:.3f} {'[SATURATED]' if abs(raw_torso_cmd[2]-torso_cmd[2]) > 1e-5 else ''}")
                
                # Pitch (index 3)
                print(f"Pitch  | Raw: {raw_torso_cmd[3]:.3f} -> Clip: {torso_cmd[3]:.3f} {'[SATURATED]' if abs(raw_torso_cmd[3]-torso_cmd[3]) > 1e-5 else ''}")
                
                # Yaw / Extras (indices 4:6)
                print(f"Yaw/R  | Forced to 0.0 (Raw was: {raw_torso_cmd[4:6]})")

            # Build pi2 obs and run
            pi2_obs_np = build_pi2_obs(torso_cmd, data)
            pi2_obs    = torch.from_numpy(pi2_obs_np).unsqueeze(0)  # (1, 39)

            with torch.no_grad():
                pi2_action = pi2(pi2_obs).squeeze(0).numpy()        # (12,)

            # Joint position targets in MJCF order.
            # action_scale < 1.0 compensates for physics gap (Isaac Lab USD ≠ MJCF geometry):
            #   - Isaac Lab USD stands at 0.375m with default joints; MJCF settles at 0.316m
            #   - Pi2 actions sized for Isaac Lab dynamics → too large for MuJoCo
            #   - Scaling to 40% keeps robot stable while preserving torso tracking direction
            joint_targets_isaac = DEFAULT_JOINT_POS_ISAAC + args.action_scale * 0.25 * pi2_action
            targets_mjcf = joint_targets_isaac[REINDEX]

            # ── Inner physics loop (4 × 0.005s = 0.02s) ──────────────────
            for _ in range(DECIMATION):
                data.ctrl[:] = actuator.compute(
                    targets_mjcf, data.qpos[7:19], data.qvel[6:18])
                mujoco.mj_step(model, data)

            # ── Video frame capture ───────────────────────────────────────
            if renderer is not None:
                renderer.update_scene(data)
                frames.append(renderer.render().copy())

            # ── Logging & apex detection ──────────────────────────────────
            trunk_z  = data.body("trunk").xpos[2]
            ball_z   = data.body("ball").xpos[2]
            ball_vz  = data.body("ball").cvel[5]   # world z velocity
            paddle_z = trunk_z + PADDLE_OFFSET_B[2]

            apex = detect_apex(prev_ball_vz, ball_vz, ball_z, paddle_z)
            if apex is not None:
                last_apex = apex
                bounce_count += 1
                ep_bounces  += 1

            prev_ball_vz = ball_vz

            if step % 100 == 0:
                ball_pos  = data.body("ball").xpos.copy()
                jp_isaac  = np.zeros(12); jp_isaac[REINDEX] = data.qpos[7:19]
                jp_rel    = jp_isaac - DEFAULT_JOINT_POS_ISAAC
                print(f"  step {step:5d} | trunk_z={trunk_z:.3f} | bounces={bounce_count} | last_apex={last_apex:.2f}m")
                print(f"    ball_pos  : x={ball_pos[0]:+.3f}  y={ball_pos[1]:+.3f}  z={ball_pos[2]:+.3f}  vz={ball_vz:+.2f}")
                print(f"    6d_cmd    : h={torso_cmd[0]:.3f}  h_dot={torso_cmd[1]:+.2f}  "
                      f"roll={torso_cmd[2]:+.3f}  pitch={torso_cmd[3]:+.3f}  "
                      f"wr={torso_cmd[4]:+.2f}  wp={torso_cmd[5]:+.2f}")
                print(f"    jp_rel    : {np.round(jp_rel, 3)}")

            # Episode reset: ball fell below paddle or robot fell over
            ball_off   = ball_z < (paddle_z - 0.15) or \
                         abs(data.body("ball").xpos[0]) > 1.0 or \
                         abs(data.body("ball").xpos[1]) > 1.0
            robot_fell = trunk_z < 0.10

            if ball_off or robot_fell:
                reason = "ball_off" if ball_off else "robot_fell"
                ep_len = step - ep_start
                print(f"\n  [episode {episode}] len={ep_len} bounces={ep_bounces} "
                      f"reason={reason}\n")
                episode  += 1
                ep_start  = step
                ep_bounces = 0
                last_action[:] = 0
                prev_ball_vz   = 0
                reset_sim(model, data, actuator)

            if viewer is not None:
                viewer.sync()
                # Real-time pacing: 50 Hz policy = 0.02 s per step
                time.sleep(max(0.0, 0.02 - 0.001))

            step += 1

        if renderer is not None and frames:
            video_dir = os.path.join(os.path.dirname(__file__), "..", "videos")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.abspath(os.path.join(video_dir, "mujoco_latest.mp4"))
            imageio.mimwrite(video_path, frames, fps=args.video_fps)
            print(f"[play_mujoco] Video saved → {video_path}  ({len(frames)} frames @ {args.video_fps} fps)")

    if args.video:
        renderer = mujoco.Renderer(model, height=480, width=640)
        run_loop(renderer=renderer)
        renderer.close()
    elif args.headless:
        run_loop(viewer=None)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run_loop(viewer=viewer)

    print(f"\n[play_mujoco] Done. Total bounces: {bounce_count}")


if __name__ == "__main__":
    main()
