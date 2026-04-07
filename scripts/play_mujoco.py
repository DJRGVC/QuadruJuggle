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
    Isaac Lab: FL(0-2), FR(3-5), RL(6-8), RR(9-11)
    MJCF:      FR(0-2), FL(3-5), RR(6-8), RL(9-11)
    Reindex:   MJCF_idx = Isaac_idx[[3,4,5, 0,1,2, 9,10,11, 6,7,8]]
               (same permutation used in both directions)

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
import torch
import torch.nn as nn
import mujoco
import mujoco.viewer

# ── Constants matching Isaac Lab env cfg ──────────────────────────────────────
PADDLE_OFFSET_B   = np.array([0.0, 0.0, 0.070])   # paddle in trunk body frame
BALL_RADIUS       = 0.020                           # metres
MAX_TARGET_HEIGHT = 0.60                            # normalisation ceiling

# Pi1 action scaling: physical = action * scale + offset
CMD_SCALES  = np.array([0.125, 1.0, 0.4, 0.4, 3.0, 3.0])
CMD_OFFSETS = np.array([0.375, 0.0, 0.0, 0.0, 0.0, 0.0])

# Pi2 obs normalisation for torso command
OBS_NORM   = np.array([8.0, 1.0, 2.5, 2.5, 1/3, 1/3])   # 1/CMD_SCALES
OBS_OFFSET = np.array([-0.375, 0.0, 0.0, 0.0, 0.0, 0.0])

# Isaac Lab Go1 default joint positions (FL, FR, RL, RR order)
DEFAULT_JOINT_POS_ISAAC = np.array([
     0.1,  0.8, -1.5,   # FL: hip, thigh, calf
    -0.1,  0.8, -1.5,   # FR
     0.1,  1.0, -1.5,   # RL
    -0.1,  1.0, -1.5,   # RR
])

# Reindex: converts between Isaac (FL,FR,RL,RR) and MJCF (FR,FL,RR,RL) order.
# Same permutation applies in both directions.
REINDEX = np.array([3, 4, 5,  0, 1, 2,  9, 10, 11,  6, 7, 8])

# MJCF default joint positions (MJCF order: FR,FL,RR,RL)
DEFAULT_JOINT_POS_MJCF = DEFAULT_JOINT_POS_ISAAC[REINDEX]

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
    pi1_sd = pi1_ckpt.get("model_state_dict", pi1_ckpt)
    pi2_sd = pi2_ckpt.get("model_state_dict", pi2_ckpt)
    pi1 = _build_actor(pi1_sd, "actor.")
    pi2 = _build_actor(pi2_sd, "actor.")
    print(f"[play_mujoco] Pi1 input dim: {pi1[0].in_features}")
    print(f"[play_mujoco] Pi2 input dim: {pi2[0].in_features}")
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

    # Joint positions (MJCF order FR,FL,RR,RL → reindex to Isaac FL,FR,RL,RR)
    # qpos layout: [trunk_xyz(3), trunk_quat(4), joints(12), ball_xyz(3), ball_quat(4)]
    joint_pos_mjcf = data.qpos[7:19].copy()
    joint_pos_isaac = joint_pos_mjcf[REINDEX]
    joint_pos_rel = joint_pos_isaac - DEFAULT_JOINT_POS_ISAAC  # (12,)

    # Joint velocities (MJCF order → Isaac order)
    joint_vel_mjcf  = data.qvel[6:18].copy()
    joint_vel_isaac = joint_vel_mjcf[REINDEX]                  # (12,)

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

    joint_pos_mjcf  = data.qpos[7:19].copy()
    joint_pos_rel   = joint_pos_mjcf[REINDEX] - DEFAULT_JOINT_POS_ISAAC
    joint_vel_isaac = data.qvel[6:18][REINDEX]

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

    def __init__(self, net_path: str):
        net_path = os.path.abspath(net_path)
        self.net = torch.jit.load(net_path, map_location="cpu").eval()
        print(f"[ActuatorNet] Loaded from {net_path}")
        self._pos_hist = np.zeros((self.HISTORY_LEN, 12), dtype=np.float32)
        self._vel_hist = np.zeros((self.HISTORY_LEN, 12), dtype=np.float32)

    def reset(self):
        self._pos_hist[:] = 0.0
        self._vel_hist[:] = 0.0

    def compute(self, targets_mjcf: np.ndarray,
                joint_pos_mjcf: np.ndarray,
                joint_vel_mjcf: np.ndarray) -> np.ndarray:
        """Return torques (MJCF order, clipped to ±23.7 N·m)."""
        pos_err = targets_mjcf - joint_pos_mjcf   # (12,)
        # Roll history: newest at index 0
        self._pos_hist = np.roll(self._pos_hist, 1, axis=0)
        self._vel_hist = np.roll(self._vel_hist, 1, axis=0)
        self._pos_hist[0] = pos_err
        self._vel_hist[0] = joint_vel_mjcf

        # Build (12, 6) input: [pos_err * pos_scale × 3, vel × vel_scale × 3]
        pos_in = (self._pos_hist * self.POS_SCALE).T   # (12, 3)
        vel_in = (self._vel_hist * self.VEL_SCALE).T   # (12, 3)
        net_in = np.concatenate([pos_in, vel_in], axis=1)  # (12, 6)

        with torch.no_grad():
            torques = self.net(torch.from_numpy(net_in)).squeeze(-1).numpy()  # (12,)

        return np.clip(torques, -self.EFFORT_LIMIT, self.EFFORT_LIMIT)


# ── Apex detection ─────────────────────────────────────────────────────────────

def detect_apex(prev_vz: float, curr_vz: float,
                ball_z: float, paddle_z: float):
    """Return apex height above paddle if ball just crossed vz=0 going down."""
    if prev_vz > 0.02 and curr_vz <= 0.02:
        return max(0.0, ball_z - paddle_z - BALL_RADIUS)
    return None


# ── Reset ─────────────────────────────────────────────────────────────────────

def reset_sim(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Reset to standing pose with ball above paddle."""
    mujoco.mj_resetData(model, data)

    data.qpos[0:3] = [0, 0, 0.35]        # trunk position (near MuJoCo natural height)
    data.qpos[3:7] = [1, 0, 0, 0]        # trunk quaternion (identity)
    data.qpos[7:19] = DEFAULT_JOINT_POS_MJCF  # joints (MJCF order)

    # Ball above paddle: trunk at 0.315, paddle at +0.07 = 0.385, ball 0.15m above
    data.qpos[19:22] = [0.0, 0.0, 0.53]  # ball xyz
    data.qpos[22:26] = [1, 0, 0, 0]      # ball quaternion

    mujoco.mj_forward(model, data)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sim-to-Sim: Isaac Lab policies in MuJoCo")
    parser.add_argument("--launcher_checkpoint", required=True,
                        help="Path to trained pi1 launcher checkpoint .pt")
    parser.add_argument("--pi2_checkpoint", required=True,
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

    # Actuator net still loaded for reference, but PD fallback used for stability
    actuator = GoActuatorNet(ACTUATOR_NET_PATH)

    # Simulation parameters
    DECIMATION   = 4     # pi1/pi2 run at 50 Hz, physics at 200 Hz (0.002s × 4 = 0.02s)
    apex_norm    = args.apex_height / MAX_TARGET_HEIGHT
    last_action  = np.zeros(6, dtype=np.float32)
    last_apex    = 0.0
    prev_ball_vz = 0.0
    bounce_count = 0
    episode      = 0

    reset_sim(model, data)

    def run_loop(viewer=None):
        nonlocal last_action, last_apex, prev_ball_vz, bounce_count, episode

        step = 0
        ep_start = step
        ep_bounces = 0

        print(f"\n[play_mujoco] Running — {'Ctrl+C' if viewer is None else 'close window'} to stop.\n")

        while True:
            if args.max_steps > 0 and step >= args.max_steps:
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

            # MuJoCo geometry gap: Isaac Lab USD equilibrium 0.375m ≠ MJCF 0.316m.
            # Pi1's velocity/angular commands (cmd[1..5]) cause MuJoCo pi2 to walk
            # away from the ball. Zero them out; keep only the height cmd[0] so pi1
            # can still inject energy via timed height changes.
            # Height clamped to [0.34, 0.41]: range pi2 can actually track in MuJoCo.
            torso_cmd[0] = np.clip(torso_cmd[0], 0.34, 0.41)
            torso_cmd[1:] = 0.0   # zero velocity/angular commands → robot stands still

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
                pos_err = targets_mjcf - data.qpos[7:19]
                data.ctrl[:] = np.clip(KP_FALLBACK * pos_err + KD_FALLBACK * (-data.qvel[6:18]),
                                       -23.7, 23.7)
                mujoco.mj_step(model, data)

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
                print(f"  step {step:5d} | trunk_z={trunk_z:.3f} "
                      f"ball_z={ball_z:.3f} ball_vz={ball_vz:+.2f} "
                      f"| last_apex={last_apex:.2f}m "
                      f"| bounces={bounce_count}")

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
                reset_sim(model, data)

            if viewer is not None:
                viewer.sync()
                # Real-time pacing: 50 Hz policy = 0.02 s per step
                time.sleep(max(0.0, 0.02 - 0.001))

            step += 1

    if args.headless:
        run_loop(viewer=None)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run_loop(viewer=viewer)

    print(f"\n[play_mujoco] Done. Total bounces: {bounce_count}")


if __name__ == "__main__":
    main()
