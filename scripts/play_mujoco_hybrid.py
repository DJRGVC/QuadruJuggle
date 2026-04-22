"""Sim-to-Sim: Launcher pi1 → Mirror Law hybrid in MuJoCo.

Same as play_mujoco.py but replaces pure pi1 control with the V4 hybrid:
  - pi1 (launcher) runs until ball apex enters ±switch_window of target
  - Mirror law sustains the juggle after handoff
  - Falls back to pi1 if apex drops below fallback_threshold × target

Pipeline:
    MuJoCo state → 46D pi1 obs → pi1 MLP → 6D torso cmd  (launcher phase)
    MuJoCo state → mirror_law_cmd()         → 6D torso cmd  (sustain phase)
                 → 39D pi2 obs → pi2 MLP → 12D joint targets
                 → Go1 actuator net → torques → MuJoCo step

Usage:
    cd /home/frank/berkeley_mde/QuadruJuggle
    conda run -n isaaclab python scripts/play_mujoco_hybrid.py \\
        --launcher_checkpoint logs/rsl_rl/go1_ball_launcher/2026-04-20_11-03-50/model_best.pt \\
        --pi2_checkpoint      logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \\
        --apex_height 0.30
"""

import argparse
import os
import time

import numpy as np
import sys
sys.path.insert(0, os.path.dirname(__file__))
from mujoco_utils import build_reindex
import torch
import torch.nn as nn
import mujoco
import mujoco.viewer
import imageio

# ── Constants matching Isaac Lab env cfg ──────────────────────────────────────
PADDLE_OFFSET_B   = np.array([0.0, 0.0, 0.070])
BALL_RADIUS       = 0.020
MAX_TARGET_HEIGHT = 1.00

CMD_SCALES  = np.array([0.125, 1.0, 0.4, 0.4, 3.0, 3.0])
CMD_OFFSETS = np.array([0.375, 0.0, 0.0, 0.0, 0.0, 0.0])

OBS_NORM   = np.array([8.0, 1.0, 2.5, 2.5, 1/3, 1/3])
OBS_OFFSET = np.array([-0.375, 0.0, 0.0, 0.0, 0.0, 0.0])

DEFAULT_JOINT_POS_ISAAC = np.array([
     0.1, -0.1,  0.1, -0.1,
     0.8,  0.8,  1.0,  1.0,
    -1.5, -1.5, -1.5, -1.5,
])

REINDEX              = None
DEFAULT_JOINT_POS_MJCF = None

ACTUATOR_NET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../walk-these-ways/resources/actuator_nets/unitree_go1.pt"
)

MUJOCO_ACTION_SCALE = 0.20
KP_FALLBACK = 100.0
KD_FALLBACK = 3.0

_GRAVITY     = 9.81
_RESTITUTION = 0.85


# ── Policy loading ─────────────────────────────────────────────────────────────

def _build_actor(state_dict: dict, prefix: str) -> nn.Sequential:
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
    actor_state = {}
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        parts = k[len(prefix):].split(".")
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
    print(f"[play_mujoco_hybrid] Pi1 input dim: {pi1[0].in_features}  (key={pi1_pfx})")
    print(f"[play_mujoco_hybrid] Pi2 input dim: {pi2[0].in_features}  (key={pi2_pfx})")
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


# ── Mirror law (numpy) ─────────────────────────────────────────────────────────

def mirror_law_cmd(
    ball_pos_w:    np.ndarray,   # (3,)
    ball_vel_w:    np.ndarray,   # (3,)
    trunk_pos_w:   np.ndarray,   # (3,)
    trunk_quat_w:  np.ndarray,   # (4,) w,x,y,z
    apex:          float,
    h_nominal:     float,
    centering_gain: float,
) -> np.ndarray:
    """Return normalised 6D torso command using mirror law. Returns (6,) array."""
    R = quat_to_rot(trunk_quat_w)                        # body → world

    paddle_pos_w = trunk_pos_w + R @ PADDLE_OFFSET_B
    p_rel_w      = ball_pos_w - paddle_pos_w             # ball relative to paddle

    v_out_z = np.sqrt(max(2.0 * _GRAVITY * apex, 0.25))
    v_out_x = -centering_gain * p_rel_w[0]
    v_out_y = -centering_gain * p_rel_w[1]
    v_out_w = np.array([v_out_x, v_out_y, v_out_z])

    v_out_eff = v_out_w / max(_RESTITUTION, 0.1)
    n_raw = v_out_eff - ball_vel_w
    norm  = np.linalg.norm(n_raw)
    n_w   = n_raw / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
    if n_w[2] < 0:
        n_w = -n_w

    n_b    = R.T @ n_w                                   # world → body
    nz_safe = max(n_b[2], 0.15)

    pitch_tgt = float(np.clip(np.arctan2( n_b[0], nz_safe), -0.4, 0.4))
    roll_tgt  = float(np.clip(np.arctan2(-n_b[1], nz_safe), -0.4, 0.4))

    ball_descending = float(ball_vel_w[2] < 0.0)
    near_impact     = float(p_rel_w[2] < 0.50)

    v_in_z_abs      = abs(ball_vel_w[2])
    v_paddle_target = (v_out_z + _RESTITUTION * v_in_z_abs) / (1.0 + _RESTITUTION)
    v_paddle_cmd    = float(np.clip(v_paddle_target / 0.5, 0.0, 1.0))
    h_dot_impulse   = float(np.clip(v_paddle_cmd * ball_descending * near_impact, 0.0, 1.0))

    not_impacting = 1.0 - float(np.clip(ball_descending * near_impact, 0.0, 1.0))
    h_dot_cmd     = float(np.clip(h_dot_impulse + 0.15 * not_impacting, 0.0, 1.0))

    h_cmd      = float(np.clip(h_nominal, 0.32, 0.42))
    h_dot_cmd  = float(np.clip(h_dot_cmd, -0.8, 1.0))

    cmd_phys = np.array([h_cmd, h_dot_cmd, roll_tgt, pitch_tgt, 0.0, 0.0])
    return (cmd_phys - CMD_OFFSETS) / CMD_SCALES          # normalised (6,)


# ── Observation construction ───────────────────────────────────────────────────

def build_pi1_obs(data: mujoco.MjData,
                  last_action: np.ndarray,
                  apex_height_norm: float) -> np.ndarray:
    trunk_pos_w  = data.body("trunk").xpos.copy()
    trunk_quat_w = data.body("trunk").xquat.copy()
    R = quat_to_rot(trunk_quat_w)

    paddle_pos_w = trunk_pos_w + R @ PADDLE_OFFSET_B

    ball_pos_w   = data.body("ball").xpos.copy()
    ball_pos_paddle = R.T @ (ball_pos_w - paddle_pos_w)

    ball_vel_w = data.body("ball").cvel[3:6].copy()
    ball_vel_b = R.T @ ball_vel_w

    trunk_linvel_b = R.T @ data.qvel[0:3].copy()
    trunk_angvel_b = R.T @ data.qvel[3:6].copy()
    proj_gravity   = R.T @ np.array([0.0, 0.0, -1.0])

    joint_pos_isaac = np.zeros(12)
    joint_pos_isaac[REINDEX] = data.qpos[7:19]
    joint_pos_rel = joint_pos_isaac - DEFAULT_JOINT_POS_ISAAC

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
    ])
    return obs.astype(np.float32)


def build_pi2_obs(torso_cmd: np.ndarray, data: mujoco.MjData) -> np.ndarray:
    torso_cmd_norm = (torso_cmd + OBS_OFFSET) * OBS_NORM

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
    ])
    return obs.astype(np.float32)


# ── Actuator net ───────────────────────────────────────────────────────────────

class GoActuatorNet:
    HISTORY_LEN  = 3
    POS_SCALE    = -1.0
    VEL_SCALE    =  1.0
    EFFORT_LIMIT = 23.7
    TORQUE_SCALE = 0.93

    def __init__(self, net_path: str, reindex: np.ndarray):
        net_path = os.path.abspath(net_path)
        self.net = torch.jit.load(net_path, map_location="cpu").eval()
        print(f"[ActuatorNet] Loaded from {net_path}")
        self._pos_hist = np.zeros((self.HISTORY_LEN, 12), dtype=np.float32)
        self._vel_hist = np.zeros((self.HISTORY_LEN, 12), dtype=np.float32)
        self._reindex  = reindex

    def reset(self):
        self._pos_hist[:] = 0.0
        self._vel_hist[:] = 0.0

    def compute(self, targets_mjcf, joint_pos_mjcf, joint_vel_mjcf):
        pos_err_mjcf = targets_mjcf - joint_pos_mjcf
        e_isaac = np.zeros(12, dtype=np.float32); e_isaac[self._reindex] = pos_err_mjcf
        v_isaac = np.zeros(12, dtype=np.float32); v_isaac[self._reindex] = joint_vel_mjcf

        self._pos_hist = np.roll(self._pos_hist, 1, axis=0)
        self._vel_hist = np.roll(self._vel_hist, 1, axis=0)
        self._pos_hist[0] = e_isaac
        self._vel_hist[0] = v_isaac

        pos_in = (self._pos_hist * self.POS_SCALE).T
        vel_in = (self._vel_hist * self.VEL_SCALE).T
        net_in = np.concatenate([pos_in, vel_in], axis=1)

        with torch.no_grad():
            torques_isaac = self.net(torch.from_numpy(net_in)).squeeze(-1).numpy()

        return np.clip(torques_isaac[self._reindex] * self.TORQUE_SCALE,
                       -self.EFFORT_LIMIT, self.EFFORT_LIMIT)


# ── Apex detection ─────────────────────────────────────────────────────────────

def detect_apex(prev_vz, curr_vz, ball_z, paddle_z):
    if prev_vz > 0.02 and curr_vz <= 0.02:
        return max(0.0, ball_z - paddle_z - BALL_RADIUS)
    return None


# ── Reset ──────────────────────────────────────────────────────────────────────

def reset_sim(model, data, actuator=None, warmup_steps=500, actuator_warmup=20,
              kp=100.0, kd=3.0):
    mujoco.mj_resetData(model, data)
    _spawn_pos  = np.array([0.0, 0.0, 0.42])
    _spawn_quat = np.array([1.0, 0.0, 0.0, 0.0])

    data.qpos[0:3]   = _spawn_pos
    data.qpos[3:7]   = _spawn_quat
    data.qpos[7:19]  = DEFAULT_JOINT_POS_MJCF
    data.qpos[19:22] = [0.0, 0.0, 0.53]
    data.qpos[22:26] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    if actuator is None:
        return

    _ball_spawn = np.array([0.0, 0.0, 0.63])
    actuator.reset()

    for _ in range(warmup_steps):
        err = DEFAULT_JOINT_POS_MJCF - data.qpos[7:19]
        data.ctrl[:] = np.clip(kp * err - kd * data.qvel[6:18], -23.7, 23.7)
        mujoco.mj_step(model, data)
        data.qpos[0:3]   = _spawn_pos
        data.qpos[3:7]   = _spawn_quat
        data.qvel[0:6]   = 0.0
        data.qpos[19:22] = _ball_spawn
        data.qvel[19:25] = 0.0

    data.qpos[7:19]  = DEFAULT_JOINT_POS_MJCF
    data.qpos[19:22] = _ball_spawn
    data.qvel[:]     = 0.0
    mujoco.mj_forward(model, data)

    actuator.reset()
    for _ in range(actuator_warmup):
        data.ctrl[:] = actuator.compute(
            DEFAULT_JOINT_POS_MJCF, data.qpos[7:19], data.qvel[6:18])
        mujoco.mj_step(model, data)
        data.qpos[0:3]   = _spawn_pos
        data.qpos[3:7]   = _spawn_quat
        data.qvel[0:6]   = 0.0
        data.qpos[19:22] = _ball_spawn
        data.qvel[19:25] = 0.0

    data.qpos[7:19]  = DEFAULT_JOINT_POS_MJCF
    data.qpos[19:22] = _ball_spawn
    data.qvel[:]     = 0.0
    mujoco.mj_forward(model, data)

    jp_err = np.abs(data.qpos[7:19] - DEFAULT_JOINT_POS_MJCF).max()
    print(f"  [reset_sim] trunk_z={data.body('trunk').xpos[2]:.3f}  jp_err_max={jp_err:.4f}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sim-to-Sim: Launcher + Mirror Law hybrid in MuJoCo")
    _root = os.path.dirname(os.path.abspath(__file__)) + "/.."
    parser.add_argument("--launcher_checkpoint",
                        default=f"{_root}/logs/rsl_rl/go1_ball_launcher/2026-04-20_11-03-50/model_best.pt")
    parser.add_argument("--pi2_checkpoint",
                        default=f"{_root}/logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt")
    parser.add_argument("--apex_height",        type=float, default=0.30)
    parser.add_argument("--mirror_only",          action="store_true", default=False,
                        help="Skip pi1 entirely — run pure mirror law from step 0")
    parser.add_argument("--switch_window",       type=float, default=0.10,
                        help="Pi1→mirror when |last_apex - target| < window [m]")
    parser.add_argument("--fallback_threshold",  type=float, default=0.50,
                        help="Mirror→pi1 when apex < fallback × target")
    parser.add_argument("--centering_gain",      type=float, default=2.0)
    parser.add_argument("--h_nominal",           type=float, default=0.38)
    parser.add_argument("--max_steps",           type=int,   default=3000)
    parser.add_argument("--headless",            action="store_true")
    parser.add_argument("--xml",                 type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "../mujoco/go1_juggle.xml"))
    parser.add_argument("--action_scale",        type=float, default=MUJOCO_ACTION_SCALE)
    parser.add_argument("--video",               action="store_true", default=False)
    parser.add_argument("--video_length",        type=int,   default=600)
    parser.add_argument("--video_fps",           type=int,   default=50)
    args = parser.parse_args()

    print(f"[play_mujoco_hybrid] launcher     : {args.launcher_checkpoint}")
    print(f"[play_mujoco_hybrid] pi2          : {args.pi2_checkpoint}")
    print(f"[play_mujoco_hybrid] target       : {args.apex_height:.2f} m")
    print(f"[play_mujoco_hybrid] switch window: ±{args.switch_window:.3f} m")
    print(f"[play_mujoco_hybrid] fallback     : {args.fallback_threshold:.2f} × target")
    print(f"[play_mujoco_hybrid] action_scale : {args.action_scale}")

    pi1, pi2 = load_policies(args.launcher_checkpoint, args.pi2_checkpoint)

    xml_path = os.path.abspath(args.xml)
    model    = mujoco.MjModel.from_xml_path(xml_path)
    data     = mujoco.MjData(model)

    global REINDEX, DEFAULT_JOINT_POS_MJCF
    REINDEX              = build_reindex(model)
    DEFAULT_JOINT_POS_MJCF = DEFAULT_JOINT_POS_ISAAC[REINDEX]

    actuator = GoActuatorNet(ACTUATOR_NET_PATH, REINDEX)

    DECIMATION   = 4
    apex_norm    = args.apex_height / MAX_TARGET_HEIGHT
    fallback_apex = args.fallback_threshold * args.apex_height

    last_action    = np.zeros(6,  dtype=np.float32)
    prev_ball_vz   = 0.0
    last_bounce_apex = 0.0
    mode_steps     = 0
    MIN_MODE_STEPS = 20
    using_mirror   = False

    bounce_count = 0
    episode      = 0

    if args.mirror_only:
        print("[play_mujoco_hybrid] --mirror_only: pure mirror law, pi1 disabled.")

    reset_sim(model, data, actuator)

    def run_loop(viewer=None, renderer=None):
        nonlocal last_action, prev_ball_vz, last_bounce_apex
        nonlocal mode_steps, using_mirror, bounce_count, episode

        step      = 0
        ep_start  = 0
        ep_bounces = 0
        frames    = [] if renderer is not None else None

        print(f"\n[play_mujoco_hybrid] Running — {'Ctrl+C' if viewer is None else 'close window'} to stop.\n")

        while True:
            if args.max_steps > 0 and step >= args.max_steps:
                break
            if renderer is not None and step >= args.video_length:
                break

            trunk_pos_w  = data.body("trunk").xpos.copy()
            trunk_quat_w = data.body("trunk").xquat.copy()
            ball_pos_w   = data.body("ball").xpos.copy()
            ball_vel_w   = data.body("ball").cvel[3:6].copy()
            paddle_z     = trunk_pos_w[2] + PADDLE_OFFSET_B[2]

            # Apex detection and mode switching
            ball_vz  = ball_vel_w[2]
            apex     = detect_apex(prev_ball_vz, ball_vz, ball_pos_w[2], paddle_z)
            if apex is not None:
                last_bounce_apex = apex
                bounce_count    += 1
                ep_bounces      += 1

            prev_ball_vz = ball_vz
            mode_steps  += 1

            if not args.mirror_only and mode_steps >= MIN_MODE_STEPS:
                if not using_mirror and last_bounce_apex > 0:
                    if abs(last_bounce_apex - args.apex_height) < args.switch_window:
                        using_mirror = True
                        mode_steps   = 0
                        print(f"  [step {step}] → MIRROR LAW  (apex={last_bounce_apex:.3f}m)")
                if using_mirror and last_bounce_apex > 0:
                    if last_bounce_apex < fallback_apex:
                        using_mirror = False
                        mode_steps   = 0
                        print(f"  [step {step}] → PI1 LAUNCHER (apex={last_bounce_apex:.3f}m)")

            # ── Compute torso command ─────────────────────────────────────────
            if using_mirror:
                pi1_action = mirror_law_cmd(
                    ball_pos_w, ball_vel_w, trunk_pos_w, trunk_quat_w,
                    args.apex_height, args.h_nominal, args.centering_gain,
                )
            else:
                pi1_obs_np = build_pi1_obs(data, last_action, apex_norm)
                pi1_obs    = torch.from_numpy(pi1_obs_np).unsqueeze(0)
                with torch.no_grad():
                    pi1_action = pi1(pi1_obs).squeeze(0).numpy()
                pi1_action = np.clip(pi1_action, -2.0, 2.0)

            last_action = pi1_action.copy()

            # Scale to physical torso command
            torso_cmd    = pi1_action * CMD_SCALES + CMD_OFFSETS
            torso_cmd[0] = np.clip(torso_cmd[0], 0.34, 0.41)
            torso_cmd[1] = np.clip(torso_cmd[1], -1.0, 1.0)
            torso_cmd[2] = np.clip(torso_cmd[2], -0.4, 0.4)
            torso_cmd[3] = np.clip(torso_cmd[3], -0.4, 0.4)
            torso_cmd[4:6] = 0.0

            # ── Pi2 → joint targets ───────────────────────────────────────────
            pi2_obs_np = build_pi2_obs(torso_cmd, data)
            pi2_obs    = torch.from_numpy(pi2_obs_np).unsqueeze(0)
            with torch.no_grad():
                pi2_action = pi2(pi2_obs).squeeze(0).numpy()

            joint_targets_isaac = DEFAULT_JOINT_POS_ISAAC + args.action_scale * 0.25 * pi2_action
            targets_mjcf = joint_targets_isaac[REINDEX]

            # ── Physics step ──────────────────────────────────────────────────
            for _ in range(DECIMATION):
                data.ctrl[:] = actuator.compute(
                    targets_mjcf, data.qpos[7:19], data.qvel[6:18])
                mujoco.mj_step(model, data)

            if renderer is not None:
                renderer.update_scene(data)
                frames.append(renderer.render().copy())

            # ── Logging ───────────────────────────────────────────────────────
            trunk_z = data.body("trunk").xpos[2]
            if step % 100 == 0:
                mode_str = "MIRROR" if using_mirror else "PI1"
                print(f"  step {step:5d} [{mode_str}] | trunk_z={trunk_z:.3f} "
                      f"| bounces={bounce_count} | last_apex={last_bounce_apex:.2f}m")
                print(f"    ball_pos: x={ball_pos_w[0]:+.3f}  y={ball_pos_w[1]:+.3f}  "
                      f"z={ball_pos_w[2]:+.3f}  vz={ball_vz:+.2f}")
                print(f"    cmd6d:  h={torso_cmd[0]:.3f}  h_dot={torso_cmd[1]:+.2f}  "
                      f"roll={torso_cmd[2]:+.3f}  pitch={torso_cmd[3]:+.3f}")

            # ── Episode reset ─────────────────────────────────────────────────
            ball_off   = (ball_pos_w[2] < paddle_z - 0.15 or
                          abs(ball_pos_w[0]) > 1.0 or
                          abs(ball_pos_w[1]) > 1.0)
            robot_fell = trunk_z < 0.10

            if ball_off or robot_fell:
                reason = "ball_off" if ball_off else "robot_fell"
                print(f"\n  [episode {episode}] len={step - ep_start} "
                      f"bounces={ep_bounces} reason={reason}\n")
                episode       += 1
                ep_start       = step
                ep_bounces     = 0
                last_action[:] = 0
                prev_ball_vz   = 0.0
                last_bounce_apex = 0.0
                mode_steps     = 0
                using_mirror   = args.mirror_only
                reset_sim(model, data, actuator)

            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, 0.02 - 0.001))

            step += 1

        if renderer is not None and frames:
            video_dir = os.path.join(os.path.dirname(__file__), "..", "videos")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.abspath(os.path.join(video_dir, "mujoco_hybrid_latest.mp4"))
            imageio.mimwrite(video_path, frames, fps=args.video_fps)
            print(f"[play_mujoco_hybrid] Video saved → {video_path}  ({len(frames)} frames @ {args.video_fps} fps)")

    if args.video:
        renderer = mujoco.Renderer(model, height=480, width=640)
        run_loop(renderer=renderer)
        renderer.close()
    elif args.headless:
        run_loop(viewer=None)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run_loop(viewer=viewer)

    print(f"\n[play_mujoco_hybrid] Done. Total bounces: {bounce_count}")
    os._exit(0)  # bypass MuJoCo viewer destructor segfault on exit


if __name__ == "__main__":
    main()
