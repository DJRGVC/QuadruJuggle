"""Sim-to-Sim: Paper-faithful mirror law in MuJoCo (no pi1).

Implements equation (14) from the juggling mirror law paper faithfully:

    m(w) = φ_s                                          (i)   azimuth setpoint
         + [-π/2 - (κ₀ + κ₁(η-η̄))(θ_s + π/2)]        (ii)  energy regulation
         + [κ₀₀(ρ_b - ρ̄_s) + κ₀₁ρ̇_b]                (iii) radial PD centering
         + [κ₁₀(φ_b - φ_s) + κ₁₁φ̇_b]                 (iv)  azimuthal PD centering

Key differences from play_mujoco_mirror.py (the approximate version):
  - Energy η = g·h_ball + ½·v²_ball_z is measured directly at every step
    (conserved quantity, same along whole trajectory — no one-bounce delay)
  - η̄ = g·apex_target (desired apex energy)
  - v_out_z is scaled by η̄/η so the paddle absorbs/adds energy within the
    same bounce rather than one bounce later
  - Centering adds velocity feedback (κ₀₁·ρ̇_b, κ₁₁·φ̇_b) in polar coords,
    replacing the pure-proportional Cartesian gain in the old version

Pipeline:
    MuJoCo state → mirror_law_paper_cmd() → 6D torso cmd
                 → 39D pi2 obs → pi2 MLP → 12D joint targets
                 → Go1 actuator net → torques → MuJoCo step

Usage:
    cd /home/frank/berkeley_mde/QuadruJuggle
    conda run -n isaaclab python scripts/play_mujoco_mirror_paper.py \\
        --pi2_checkpoint logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt \\
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

# ── Constants ─────────────────────────────────────────────────────────────────
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

REINDEX               = None
DEFAULT_JOINT_POS_MJCF = None

ACTUATOR_NET_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../walk-these-ways/resources/actuator_nets/unitree_go1.pt"
)

MUJOCO_ACTION_SCALE = 0.20

_GRAVITY     = 9.81
_RESTITUTION = 0.85


# ── Policy loading (pi2 only) ──────────────────────────────────────────────────

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
        actor_state[f"{int(parts[0])}.{parts[1]}"] = v

    net.load_state_dict(actor_state)
    for p in net.parameters():
        p.requires_grad = False
    net.eval()
    return net


def load_pi2(pi2_path: str) -> nn.Sequential:
    ckpt = torch.load(pi2_path, map_location="cpu")
    if "actor_state_dict" in ckpt:
        sd, pfx = ckpt["actor_state_dict"], "mlp."
    else:
        sd, pfx = ckpt.get("model_state_dict", ckpt), "actor."
    pi2 = _build_actor(sd, pfx)
    print(f"[play_mujoco_mirror_paper] Pi2 input dim: {pi2[0].in_features}  (key={pfx})")
    return pi2


# ── Quaternion helpers ─────────────────────────────────────────────────────────

def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """MuJoCo quaternion (w,x,y,z) → 3×3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


# ── Paper mirror law ───────────────────────────────────────────────────────────

def mirror_law_paper_cmd(
    ball_pos_w:   np.ndarray,
    ball_vel_w:   np.ndarray,
    trunk_pos_w:  np.ndarray,
    trunk_quat_w: np.ndarray,
    apex:         float,
    h_nominal:    float,
    k00:  float = 2.0,   # radial position gain  (iii)
    k01:  float = 0.5,   # radial velocity gain   (iii)
    k10:  float = 1.0,   # azimuthal position gain (iv)
    k11:  float = 0.3,   # azimuthal velocity gain  (iv)
    k1:   float = 1.0,   # energy error gain κ₁    (ii)
) -> np.ndarray:
    """Return normalised 6D torso command matching paper eq. (14).

    Term (ii) — energy regulation:
        η  = g·h_ball + ½·v_ball_z²   (conserved along flight, no bounce delay)
        η̄  = g·apex_target
        v_out_z is scaled by η̄/η → if ball carries too much energy, the
        commanded outgoing z-velocity is reduced within the SAME bounce.

    Terms (iii)+(iv) — PD centering in polar coords:
        Radial:    -(k00·ρ_b + k01·ρ̇_b)  where ρ_b = lateral distance from centre
        Azimuthal: -(k10·φ_err·ρ_b + k11·φ̇_b·ρ_b)  where φ_err = atan2(y,x)
        These feed into lateral components of the desired outgoing velocity,
        replacing the pure-proportional Cartesian centering in the old version.
    """
    R = quat_to_rot(trunk_quat_w)
    paddle_pos_w = trunk_pos_w + R @ PADDLE_OFFSET_B
    p_rel_w      = ball_pos_w - paddle_pos_w   # ball relative to paddle

    # ── Term (ii): Energy regulation ─────────────────────────────────────────
    # η is conserved along flight → measure it continuously, no apex needed
    eta        = _GRAVITY * ball_pos_w[2] + 0.5 * ball_vel_w[2] ** 2
    eta_target = _GRAVITY * apex
    # Clamp so we never amplify wildly when ball is near ground (eta very small)
    eta_safe   = max(eta, _GRAVITY * 0.02)
    # Scale factor: < 1 when over-energised, > 1 when under-energised
    # κ₀ = 1 (nominal gain), κ₁ scales the error correction
    energy_error = eta - eta_target
    energy_scale = eta_target / (eta_safe + k1 * max(energy_error, 0.0))
    energy_scale = float(np.clip(energy_scale, 0.2, 1.5))

    # Desired outgoing vertical speed (for target apex), modulated by energy
    v_out_z_nom = np.sqrt(max(2.0 * _GRAVITY * apex, 0.25))
    v_out_z     = v_out_z_nom * energy_scale

    # ── Terms (iii) + (iv): Radial and azimuthal PD centering ────────────────
    bx, by = ball_pos_w[0], ball_pos_w[1]
    vx, vy = ball_vel_w[0], ball_vel_w[1]

    rho_b = np.sqrt(bx**2 + by**2)   # radial distance from centre (ρ_b)

    if rho_b > 1e-6:
        # Unit radial and azimuthal directions in world XY
        e_r   = np.array([bx, by]) / rho_b            # points outward
        e_phi = np.array([-by, bx]) / rho_b           # CCW tangent

        rho_dot = float(np.dot([vx, vy], e_r))        # radial velocity ρ̇_b
        phi_b   = float(np.arctan2(by, bx))           # azimuthal angle φ_b
        phi_s   = 0.0                                  # target azimuth (centre)
        phi_dot = float(np.dot([vx, vy], e_phi)) / rho_b  # φ̇_b

        # Radial correction (term iii): push toward centre + damp radial vel
        v_radial  = -(k00 * rho_b + k01 * rho_dot)
        # Azimuthal correction (term iv): correct azimuthal drift + damp
        v_azimuth = -(k10 * (phi_b - phi_s) * rho_b + k11 * phi_dot * rho_b)

        v_out_x = v_radial * e_r[0] + v_azimuth * e_phi[0]
        v_out_y = v_radial * e_r[1] + v_azimuth * e_phi[1]
    else:
        v_out_x = 0.0
        v_out_y = 0.0

    # ── Mirror law: desired paddle normal from v_in and v_out ────────────────
    v_out_w   = np.array([v_out_x, v_out_y, v_out_z])
    v_out_eff = v_out_w / max(_RESTITUTION, 0.1)   # account for energy loss
    n_raw     = v_out_eff - ball_vel_w              # bisector → desired normal
    norm      = np.linalg.norm(n_raw)
    n_w       = n_raw / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
    if n_w[2] < 0:
        n_w = -n_w

    # Express in body frame → roll and pitch targets
    n_b     = R.T @ n_w
    nz_safe = max(n_b[2], 0.15)
    pitch_tgt = float(np.clip(np.arctan2( n_b[0], nz_safe), -0.4, 0.4))
    roll_tgt  = float(np.clip(np.arctan2(-n_b[1], nz_safe), -0.4, 0.4))

    # ── h_dot: paddle impulse at impact ──────────────────────────────────────
    # Uses energy-regulated v_out_z so the impulse is already corrected
    ball_descending = float(ball_vel_w[2] < 0.0)
    near_impact     = float(p_rel_w[2] < 0.50)
    v_in_z_abs      = abs(ball_vel_w[2])
    v_paddle_target = (v_out_z + _RESTITUTION * v_in_z_abs) / (1.0 + _RESTITUTION)
    v_paddle_cmd    = float(np.clip(v_paddle_target / 0.5, 0.0, 1.0))
    h_dot_impulse   = float(np.clip(v_paddle_cmd * ball_descending * near_impact, 0.0, 1.0))
    not_impacting   = 1.0 - float(np.clip(ball_descending * near_impact, 0.0, 1.0))
    h_dot_cmd       = float(np.clip(h_dot_impulse + 0.15 * not_impacting, 0.0, 1.0))

    h_cmd = float(np.clip(h_nominal, 0.32, 0.42))

    cmd_phys = np.array([h_cmd, h_dot_cmd, roll_tgt, pitch_tgt, 0.0, 0.0])
    return (cmd_phys - CMD_OFFSETS) / CMD_SCALES


# ── Pi2 observation ────────────────────────────────────────────────────────────

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

    return np.concatenate([
        torso_cmd_norm,  # 6
        trunk_linvel_b,  # 3
        trunk_angvel_b,  # 3
        proj_gravity,    # 3
        joint_pos_rel,   # 12
        joint_vel_isaac, # 12
    ]).astype(np.float32)


# ── Actuator net ───────────────────────────────────────────────────────────────

class GoActuatorNet:
    HISTORY_LEN  = 3
    POS_SCALE    = -1.0
    VEL_SCALE    =  1.0
    EFFORT_LIMIT = 23.7
    TORQUE_SCALE = 0.93

    def __init__(self, net_path: str, reindex: np.ndarray):
        self.net = torch.jit.load(os.path.abspath(net_path), map_location="cpu").eval()
        self._pos_hist = np.zeros((self.HISTORY_LEN, 12), dtype=np.float32)
        self._vel_hist = np.zeros((self.HISTORY_LEN, 12), dtype=np.float32)
        self._reindex  = reindex

    def reset(self):
        self._pos_hist[:] = 0.0
        self._vel_hist[:] = 0.0

    def compute(self, targets_mjcf, joint_pos_mjcf, joint_vel_mjcf):
        pos_err = targets_mjcf - joint_pos_mjcf
        e_isaac = np.zeros(12, dtype=np.float32); e_isaac[self._reindex] = pos_err
        v_isaac = np.zeros(12, dtype=np.float32); v_isaac[self._reindex] = joint_vel_mjcf

        self._pos_hist = np.roll(self._pos_hist, 1, axis=0)
        self._vel_hist = np.roll(self._vel_hist, 1, axis=0)
        self._pos_hist[0] = e_isaac
        self._vel_hist[0] = v_isaac

        net_in = np.concatenate(
            [(self._pos_hist * self.POS_SCALE).T,
             (self._vel_hist * self.VEL_SCALE).T], axis=1)
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

def reset_sim(model, data, actuator, warmup_steps=500, actuator_warmup=20,
              kp=100.0, kd=3.0):
    mujoco.mj_resetData(model, data)
    _spawn_pos  = np.array([0.0, 0.0, 0.42])
    _spawn_quat = np.array([1.0, 0.0, 0.0, 0.0])
    _ball_spawn = np.array([0.0, 0.0, 0.63])

    data.qpos[0:3]   = _spawn_pos
    data.qpos[3:7]   = _spawn_quat
    data.qpos[7:19]  = DEFAULT_JOINT_POS_MJCF
    data.qpos[19:22] = _ball_spawn
    data.qpos[22:26] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

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
    parser = argparse.ArgumentParser(
        description="Sim-to-Sim: Paper-faithful mirror law in MuJoCo")
    _root = os.path.dirname(os.path.abspath(__file__)) + "/.."
    parser.add_argument("--pi2_checkpoint",
                        default=f"{_root}/logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt")
    parser.add_argument("--apex_height",  type=float, default=0.30)
    parser.add_argument("--h_nominal",    type=float, default=0.38)
    parser.add_argument("--max_steps",    type=int,   default=3000)
    parser.add_argument("--headless",     action="store_true")
    parser.add_argument("--xml",          type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "../mujoco/go1_juggle.xml"))
    parser.add_argument("--action_scale", type=float, default=MUJOCO_ACTION_SCALE)
    parser.add_argument("--video",        action="store_true", default=False)
    parser.add_argument("--video_length", type=int,   default=600)
    parser.add_argument("--video_fps",    type=int,   default=50)
    # Paper gains
    parser.add_argument("--k00", type=float, default=2.0,
                        help="Radial position gain κ₀₀ (term iii)")
    parser.add_argument("--k01", type=float, default=0.5,
                        help="Radial velocity gain κ₀₁ (term iii)")
    parser.add_argument("--k10", type=float, default=1.0,
                        help="Azimuthal position gain κ₁₀ (term iv)")
    parser.add_argument("--k11", type=float, default=0.3,
                        help="Azimuthal velocity gain κ₁₁ (term iv)")
    parser.add_argument("--k1",  type=float, default=1.0,
                        help="Energy error gain κ₁ (term ii)")
    args = parser.parse_args()

    print(f"[mirror_paper] pi2          : {args.pi2_checkpoint}")
    print(f"[mirror_paper] target       : {args.apex_height:.2f} m")
    print(f"[mirror_paper] h_nominal    : {args.h_nominal}")
    print(f"[mirror_paper] action_scale : {args.action_scale}")
    print(f"[mirror_paper] gains        : k00={args.k00} k01={args.k01} "
          f"k10={args.k10} k11={args.k11} k1={args.k1}")

    pi2 = load_pi2(args.pi2_checkpoint)

    xml_path = os.path.abspath(args.xml)
    model    = mujoco.MjModel.from_xml_path(xml_path)
    data     = mujoco.MjData(model)

    global REINDEX, DEFAULT_JOINT_POS_MJCF
    REINDEX               = build_reindex(model)
    DEFAULT_JOINT_POS_MJCF = DEFAULT_JOINT_POS_ISAAC[REINDEX]

    actuator = GoActuatorNet(ACTUATOR_NET_PATH, REINDEX)

    prev_ball_vz = 0.0
    last_apex    = 0.0
    bounce_count = 0
    episode      = 0

    reset_sim(model, data, actuator)

    def run_loop(viewer=None, renderer=None):
        nonlocal prev_ball_vz, last_apex, bounce_count, episode

        step       = 0
        ep_start   = 0
        ep_bounces = 0
        frames     = [] if renderer is not None else None

        print(f"\n[mirror_paper] Running — "
              f"{'Ctrl+C' if viewer is None else 'close window'} to stop.\n")

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

            # Apex detection (for logging only — not used for energy regulation)
            ball_vz = ball_vel_w[2]
            apex    = detect_apex(prev_ball_vz, ball_vz, ball_pos_w[2], paddle_z)
            if apex is not None:
                last_apex    = apex
                bounce_count += 1
                ep_bounces   += 1
            prev_ball_vz = ball_vz

            # Paper mirror law torso command
            torso_cmd_norm = mirror_law_paper_cmd(
                ball_pos_w, ball_vel_w, trunk_pos_w, trunk_quat_w,
                args.apex_height, args.h_nominal,
                k00=args.k00, k01=args.k01,
                k10=args.k10, k11=args.k11,
                k1=args.k1,
            )
            torso_cmd = torso_cmd_norm * CMD_SCALES + CMD_OFFSETS
            torso_cmd[0] = np.clip(torso_cmd[0], 0.34, 0.41)
            torso_cmd[1] = np.clip(torso_cmd[1], -0.8, 1.0)
            torso_cmd[2] = np.clip(torso_cmd[2], -0.4, 0.4)
            torso_cmd[3] = np.clip(torso_cmd[3], -0.4, 0.4)
            torso_cmd[4:6] = 0.0

            # Pi2 → joint targets
            pi2_obs = torch.from_numpy(build_pi2_obs(torso_cmd, data)).unsqueeze(0)
            with torch.no_grad():
                pi2_action = pi2(pi2_obs).squeeze(0).numpy()

            joint_targets_isaac = DEFAULT_JOINT_POS_ISAAC + args.action_scale * 0.25 * pi2_action
            targets_mjcf = joint_targets_isaac[REINDEX]

            for _ in range(4):  # decimation
                data.ctrl[:] = actuator.compute(
                    targets_mjcf, data.qpos[7:19], data.qvel[6:18])
                mujoco.mj_step(model, data)

            if renderer is not None:
                renderer.update_scene(data)
                frames.append(renderer.render().copy())

            trunk_z = data.body("trunk").xpos[2]
            if step % 100 == 0:
                eta = _GRAVITY * ball_pos_w[2] + 0.5 * ball_vel_w[2]**2
                eta_target = _GRAVITY * args.apex_height
                print(f"  step {step:5d} | trunk_z={trunk_z:.3f} | bounces={bounce_count}"
                      f" | last_apex={last_apex:.2f}m")
                print(f"    ball: x={ball_pos_w[0]:+.3f}  y={ball_pos_w[1]:+.3f}"
                      f"  z={ball_pos_w[2]:+.3f}  vz={ball_vz:+.2f}")
                print(f"    η={eta:.3f}  η̄={eta_target:.3f}  Δη={eta-eta_target:+.3f}"
                      f"  scale={eta_target/max(eta,0.01):.2f}")
                print(f"    cmd:  h={torso_cmd[0]:.3f}  h_dot={torso_cmd[1]:+.2f}"
                      f"  roll={torso_cmd[2]:+.3f}  pitch={torso_cmd[3]:+.3f}")

            ball_off   = (ball_pos_w[2] < paddle_z - 0.15 or
                          abs(ball_pos_w[0]) > 1.0 or
                          abs(ball_pos_w[1]) > 1.0)
            robot_fell = trunk_z < 0.10

            if ball_off or robot_fell:
                reason = "ball_off" if ball_off else "robot_fell"
                print(f"\n  [episode {episode}] len={step - ep_start}"
                      f" bounces={ep_bounces} reason={reason}\n")
                episode    += 1
                ep_start    = step
                ep_bounces  = 0
                prev_ball_vz = 0.0
                last_apex   = 0.0
                reset_sim(model, data, actuator)

            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, 0.02 - 0.001))

            step += 1

        if renderer is not None and frames:
            video_dir  = os.path.join(os.path.dirname(__file__), "..", "videos")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.abspath(
                os.path.join(video_dir, "mujoco_mirror_paper_latest.mp4"))
            imageio.mimwrite(video_path, frames, fps=args.video_fps)
            print(f"[mirror_paper] Video → {video_path}  ({len(frames)} frames @ {args.video_fps} fps)")

    if args.video:
        renderer = mujoco.Renderer(model, height=480, width=640)
        run_loop(renderer=renderer)
        renderer.close()
    elif args.headless:
        run_loop(viewer=None)
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            run_loop(viewer=viewer)

    print(f"\n[mirror_paper] Done. Total bounces: {bounce_count}")
    os._exit(0)


if __name__ == "__main__":
    main()
