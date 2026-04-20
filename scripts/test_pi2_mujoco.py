"""Test pi2 torso-tracking in isolation in MuJoCo.

Sends a fixed torso command to pi2 and watches the robot track it.
No ball, no pi1 — just pi2 + actuator net.

Observation layout (matches TorsoTracking gym env observations.py exactly):
  obs[0:6]   = torso_command_obs  → (cmd + [-0.375,0,0,0,0,0]) * [8,1,2.5,2.5,1/3,1/3]
  obs[6:9]   = base_lin_vel_b     → R.T @ qvel[0:3]
  obs[9:12]  = base_ang_vel_b     → R.T @ qvel[3:6]
  obs[12:15] = projected_gravity_b→ R.T @ [0,0,-1]
  obs[15:27] = joint_pos_rel      → qpos[7:19][reindex] - DEFAULT_JOINT_POS_ISAAC
  obs[27:39] = joint_vel_rel      → qvel[6:18][reindex]  (default_joint_vel=0)

Usage:
    python scripts/test_pi2_mujoco.py                             # neutral stand
    python scripts/test_pi2_mujoco.py --height 0.40               # taller stance
    python scripts/test_pi2_mujoco.py --height 0.35 --h_dot 0.1   # rising
    python scripts/test_pi2_mujoco.py --roll 0.1 --pitch -0.05    # tilt cmd
"""

import argparse, sys, os, time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(__file__))
from mujoco_utils import build_reindex
from play_mujoco import GoActuatorNet, ACTUATOR_NET_PATH

# ── Constants (must match Isaac Lab env cfg) ──────────────────────────────────
DEFAULT_JOINT_POS_ISAAC = np.array([
     0.1, -0.1,  0.1, -0.1,
     0.8,  0.8,  1.0,  1.0,
    -1.5, -1.5, -1.5, -1.5,
])
CMD_SCALES  = np.array([0.125, 1.0,  0.4,  0.4,  3.0,  3.0])
CMD_OFFSETS = np.array([0.375, 0.0,  0.0,  0.0,  0.0,  0.0])
OBS_NORM    = np.array([8.0,   1.0,  2.5,  2.5,  1/3,  1/3])
OBS_OFFSET  = np.array([-0.375, 0.0, 0.0,  0.0,  0.0,  0.0])
DECIMATION  = 4
SIM_DT      = 0.005  # seconds per physics step

# EMA taus from commands.py — must match Isaac Lab's update_torso_commands_smooth.
# Applied once per policy step (dt = DECIMATION * SIM_DT = 0.02 s), which is
# mathematically identical to 4 × per-physics-step updates at 200 Hz.
# [h, h_dot, roll, pitch, omega_roll, omega_pitch]
_SMOOTH_TAUS       = np.array([0.40, 0.30, 0.02, 0.02, 0.01, 0.01])
_EMA_ALPHAS_POLICY = 1.0 - np.exp(-(DECIMATION * SIM_DT) / _SMOOTH_TAUS)

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)],
    ])


def load_pi2(path):
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("model_state_dict", sd)
    keys = sorted(k for k in sd if k.startswith("actor.") and "weight" in k)
    modules = []
    for i, k in enumerate(keys):
        out_dim, in_dim = sd[k].shape
        modules.append(nn.Linear(in_dim, out_dim))
        if i < len(keys) - 1:
            modules.append(nn.ELU())
    net = nn.Sequential(*modules)
    state = {}
    for k, v in sd.items():
        if not k.startswith("actor."): continue
        parts = k[len("actor."):].split(".")
        state[f"{int(parts[0])}.{parts[1]}"] = v
    net.load_state_dict(state)
    net.eval()
    for p in net.parameters(): p.requires_grad = False
    return net


def reset(model, data, default_mjcf, actuator, warmup_steps=500, kp=100.0, kd=3.0,
          actuator_warmup=20):
    mujoco.mj_resetData(model, data)
    data.qpos[0:3]  = [0, 0, 0.42]   # Isaac Lab spawn height
    data.qpos[3:7]  = [1, 0, 0, 0]
    data.qpos[7:19] = default_mjcf
    mujoco.mj_forward(model, data)

    _spawn_pos  = np.array([0.0, 0.0, 0.42])
    _spawn_quat = np.array([1.0, 0.0, 0.0, 0.0])

    # Phase 1: PD warmup with base pinned — mirrors Isaac Lab's fix_root_link=True.
    # Gravity would pull the robot down and drift joints; pinning keeps them at default.
    actuator.reset()
    for _ in range(warmup_steps):
        err = default_mjcf - data.qpos[7:19]
        data.ctrl[:] = np.clip(kp * err - kd * data.qvel[6:18], -23.7, 23.7)
        mujoco.mj_step(model, data)
        data.qpos[0:3] = _spawn_pos
        data.qpos[3:7] = _spawn_quat
        data.qvel[0:6] = 0.0

    # Phase 2: hard-set joints to exact default — mirrors write_joint_state_to_sim.
    data.qpos[7:19] = default_mjcf
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    # Phase 3: fill actuator net history with base still pinned.
    # After the hard-set, run the actuator net targeting default so the 3-step
    # history buffer sees near-zero position errors — no cold-start artefact.
    actuator.reset()
    for _ in range(actuator_warmup):
        data.ctrl[:] = actuator.compute(default_mjcf, data.qpos[7:19], data.qvel[6:18])
        mujoco.mj_step(model, data)
        data.qpos[0:3] = _spawn_pos
        data.qpos[3:7] = _spawn_quat
        data.qvel[0:6] = 0.0

    data.qpos[7:19] = default_mjcf
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    jp_err = np.abs(data.qpos[7:19] - default_mjcf).max()
    print(f"  [warmup done] trunk_z={data.body('trunk').xpos[2]:.3f}  jp_err_max={jp_err:.4f}")


PI2_CHECKPOINT = f"{_ROOT}/logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt"
XML_PATH       = f"{_ROOT}/mujoco/go1_juggle.xml"
ACTION_SCALE   = 0.25
PD_WARMUP_STEPS   = 500
CALIB_STEPS       = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="Run EMA-smoothed sweep: neutral → SWEEP_TARGET → neutral → reset")
    ap.add_argument("--dwell", type=int, default=200,
                    help="Policy steps to hold each phase (default 200 = 4 s at 50 Hz)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Playback speed multiplier (1=real-time, 2=2x, 0=unlimited)")
    args = ap.parse_args()

    # ── SWEEP TARGETS — add/remove/reorder lines to change what gets tested ──
    # Format: [height(m), h_dot(m/s), roll(rad), pitch(rad), roll_rate, pitch_rate]
    SWEEP_TARGETS = [
        np.array([0.375, 0.0,  0.0,  0.10, 0.0, 0.0]),   # pitch+ 0.10 rad
        np.array([0.375, 0.0,  0.0,  0.15, 0.0, 0.0]),   # pitch+ 0.15 rad
        np.array([0.375, 0.0,  0.0,  0.20, 0.0, 0.0]),   # pitch+ 0.20 rad
        np.array([0.375, 0.0,  0.0,  0.25, 0.0, 0.0]),   # pitch+ 0.25 rad
        np.array([0.375, 0.0,  0.0,  0.28, 0.0, 0.0]),   # pitch+ 0.28 rad
        np.array([0.375, 0.0,  0.0,  0.30, 0.0, 0.0]),   # pitch+ 0.30 rad
        np.array([0.375, 0.0,  0.0,  0.32, 0.0, 0.0]),   # pitch+ 0.32 rad
        np.array([0.375, 0.0,  0.0,  0.35, 0.0, 0.0]),   # pitch+ 0.35 rad
        np.array([0.375, 0.0,  0.0,  0.40, 0.0, 0.0]),   # pitch+ 0.40 rad
        np.array([0.375, 0.0,  0.0, -0.10, 0.0, 0.0]),   # pitch- 0.10 rad
        np.array([0.375, 0.0,  0.0, -0.15, 0.0, 0.0]),   # pitch- 0.15 rad
        np.array([0.375, 0.0,  0.0, -0.20, 0.0, 0.0]),   # pitch- 0.20 rad
        np.array([0.375, 0.0,  0.0, -0.25, 0.0, 0.0]),   # pitch- 0.25 rad
        np.array([0.375, 0.0,  0.0, -0.28, 0.0, 0.0]),   # pitch- 0.28 rad
        np.array([0.375, 0.0,  0.0, -0.30, 0.0, 0.0]),   # pitch- 0.30 rad
        np.array([0.375, 0.0,  0.0, -0.32, 0.0, 0.0]),   # pitch- 0.32 rad
        np.array([0.375, 0.0,  0.0, -0.35, 0.0, 0.0]),   # pitch- 0.35 rad
        np.array([0.375, 0.0,  0.0, -0.40, 0.0, 0.0]),   # pitch- 0.40 rad
    ]
    # ─────────────────────────────────────────────────────────────────────────

    NEUTRAL = np.array([0.375, 0.0, 0.0, 0.0, 0.0, 0.0])

    pi2   = load_pi2(PI2_CHECKPOINT)
    model = mujoco.MjModel.from_xml_path(os.path.abspath(XML_PATH))
    data  = mujoco.MjData(model)

    reindex      = build_reindex(model)
    default_mjcf = DEFAULT_JOINT_POS_ISAAC[reindex]
    actuator     = GoActuatorNet(ACTUATOR_NET_PATH, reindex)

    KP, KD = 100.0, 3.0

    def apply_ctrl(targets_mjcf):
        data.ctrl[:] = actuator.compute(targets_mjcf, data.qpos[7:19], data.qvel[6:18])

    ACTUATOR_WARMUP = 20
    _spawn_pos  = np.array([0.0, 0.0, 0.42])
    _spawn_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def begin_reset(reason=""):
        """Initialise state for an inline warmup (viewer stays live)."""
        print(f"  [reset] {reason}")
        mujoco.mj_resetData(model, data)
        data.qpos[0:3]  = _spawn_pos
        data.qpos[3:7]  = _spawn_quat
        data.qpos[7:19] = default_mjcf
        mujoco.mj_forward(model, data)
        actuator.reset()

    print(f"[pi2_test] sweep={args.sweep}  dwell={args.dwell}  action_scale={ACTION_SCALE}")

    # State machine: "pd_warmup" → "act_warmup" → "calibrate" → "running"
    state        = "pd_warmup"
    warmup_count = 0
    begin_reset("initial warmup")

    policy_step = 0   # counts only RUNNING steps (drives sweep timing)

    # Sweep state machine using EMA smoothing (matches Isaac Lab's update_torso_commands_smooth).
    # Goal snaps between NEUTRAL and SWEEP_TARGET; cmd_ema tracks it with per-dimension taus.
    _neutral    = NEUTRAL.copy()
    cmd_ema     = _neutral.copy()   # EMA-smoothed command seen by pi2 (physical units)
    cmd_goal    = _neutral.copy()   # current goal before EMA
    sweep_phase = "to_target"       # to_target | to_neutral
    sweep_count = 0                 # policy steps in current phase
    sweep_idx   = 0                 # index into SWEEP_TARGETS

    # Height offset: shifts all height commands to match MuJoCo equilibrium.
    # Calibrated once on startup; reused after falls (equilibrium is stable).
    ISAAC_NEUTRAL_H = 0.375   # height pi2 was trained to treat as "zero error"
    height_offset   = None   # auto-calibrated each run
    calib_samples   = []

    _neutral_norm  = (_neutral + OBS_OFFSET) * OBS_NORM  # fixed for calibration
    torso_cmd_norm = _neutral_norm.copy()               # default until calibration updates it

    _rec = {"cmd_h": [], "act_h": [],
            "cmd_h_dot": [], "act_h_dot": [],
            "cmd_roll": [], "act_roll": [],
            "cmd_pitch": [], "act_pitch": []}

    POLICY_DT   = DECIMATION * SIM_DT   # 0.02 s per policy step (50 Hz real time)
    _step_start = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            if state == "pd_warmup":
                # Phase 1: PD with base pinned
                err = default_mjcf - data.qpos[7:19]
                data.ctrl[:] = np.clip(KP * err - KD * data.qvel[6:18], -23.7, 23.7)
                mujoco.mj_step(model, data)
                data.qpos[0:3] = _spawn_pos
                data.qpos[3:7] = _spawn_quat
                data.qvel[0:6] = 0.0
                warmup_count += 1
                if warmup_count >= PD_WARMUP_STEPS:
                    # Phase 2: hard-set joints
                    data.qpos[7:19] = default_mjcf
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                    actuator.reset()
                    warmup_count = 0
                    state = "act_warmup"

            elif state == "act_warmup":
                # Phase 3: actuator net warmup with base pinned
                data.ctrl[:] = actuator.compute(default_mjcf, data.qpos[7:19], data.qvel[6:18])
                mujoco.mj_step(model, data)
                data.qpos[0:3] = _spawn_pos
                data.qpos[3:7] = _spawn_quat
                data.qvel[0:6] = 0.0
                warmup_count += 1
                if warmup_count >= ACTUATOR_WARMUP:
                    data.qpos[7:19] = default_mjcf
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                    jp_err = np.abs(data.qpos[7:19] - default_mjcf).max()
                    print(f"  [warmup done] trunk_z={data.body('trunk').xpos[2]:.3f}  jp_err_max={jp_err:.4f}")
                    if height_offset is None:
                        print(f"  [calibrate] running {CALIB_STEPS} steps at neutral to measure equilibrium...")
                        calib_samples.clear()
                        warmup_count = 0
                        state = "calibrate"
                    else:
                        state = "running"

            elif state == "calibrate":
                # Run policy at neutral, collect trunk_z to find MuJoCo equilibrium height.
                R = quat_to_rot(data.body("trunk").xquat)
                jp_isaac = np.zeros(12); jp_isaac[reindex] = data.qpos[7:19]
                jv_isaac = np.zeros(12); jv_isaac[reindex] = data.qvel[6:18]
                obs = np.concatenate([
                    _neutral_norm,
                    R.T @ data.qvel[0:3], R.T @ data.qvel[3:6],
                    R.T @ np.array([0., 0., -1.]),
                    jp_isaac - DEFAULT_JOINT_POS_ISAAC, jv_isaac,
                ]).astype(np.float32)
                with torch.no_grad():
                    action = pi2(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
                targets_mjcf = (DEFAULT_JOINT_POS_ISAAC + ACTION_SCALE * action)[reindex]
                for _ in range(DECIMATION):
                    apply_ctrl(targets_mjcf)
                    mujoco.mj_step(model, data)
                calib_samples.append(data.body("trunk").xpos[2])
                warmup_count += 1
                if warmup_count >= CALIB_STEPS:
                    eq_z          = float(np.mean(calib_samples))
                    height_offset = eq_z - ISAAC_NEUTRAL_H
                    print(f"  [calibrate] equilibrium trunk_z={eq_z:.4f}m  "
                          f"height_offset={height_offset:+.4f}m")
                    state = "running"

            else:  # running
                # ── Sweep: EMA-smoothed goal transitions matching Isaac Lab ───
                if args.sweep:
                    if sweep_phase == "to_target":
                        if sweep_count == 0:
                            cmd_goal[:] = SWEEP_TARGETS[sweep_idx]
                            t = SWEEP_TARGETS[sweep_idx]
                            label = (f"roll={t[2]:+.2f}" if t[2] != 0 else f"pitch={t[3]:+.2f}")
                            print(f"\n  [sweep {sweep_idx+1}/{len(SWEEP_TARGETS)}] → {label}  {SWEEP_TARGETS[sweep_idx]}")
                        sweep_count += 1
                        if sweep_count >= args.dwell:
                            sweep_phase = "to_neutral"
                            sweep_count = 0
                            print(f"  [sweep] → neutral")

                    else:  # to_neutral
                        if sweep_count == 0:
                            cmd_goal[:] = _neutral
                        sweep_count += 1
                        if sweep_count >= args.dwell:
                            sweep_idx += 1
                            sweep_count = 0
                            if sweep_idx >= len(SWEEP_TARGETS):
                                print(f"  [sweep] all targets done — resetting")
                                begin_reset("sweep cycle complete")
                                cmd_ema[:] = _neutral; cmd_goal[:] = _neutral
                                state = "pd_warmup"; warmup_count = 0
                                sweep_phase = "to_target"; sweep_idx = 0
                                policy_step += 1
                                viewer.sync()
                                continue
                            else:
                                sweep_phase = "to_target"

                    # EMA blend — one step per policy step, mathematically equal
                    # to 4× per physics step (see _EMA_ALPHAS_POLICY derivation)
                    cmd_ema += _EMA_ALPHAS_POLICY * (cmd_goal - cmd_ema)

                    cmd_shifted     = cmd_ema.copy()
                    cmd_shifted[0] += height_offset
                    torso_cmd_norm  = (cmd_shifted + OBS_OFFSET) * OBS_NORM

                R = quat_to_rot(data.body("trunk").xquat)

                lin_vel_b = R.T @ data.qvel[0:3]
                ang_vel_b = R.T @ data.qvel[3:6]
                grav_b    = R.T @ np.array([0., 0., -1.])

                jp_isaac = np.zeros(12, dtype=np.float64); jp_isaac[reindex] = data.qpos[7:19]
                jv_isaac = np.zeros(12, dtype=np.float64); jv_isaac[reindex] = data.qvel[6:18]
                jp_rel   = jp_isaac - DEFAULT_JOINT_POS_ISAAC

                obs = np.concatenate([
                    torso_cmd_norm, lin_vel_b, ang_vel_b, grav_b, jp_rel, jv_isaac
                ]).astype(np.float32)

                with torch.no_grad():
                    action = pi2(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()

                targets_mjcf = (DEFAULT_JOINT_POS_ISAAC + ACTION_SCALE * action)[reindex]

                for _ in range(DECIMATION):
                    apply_ctrl(targets_mjcf)
                    mujoco.mj_step(model, data)

                trunk_z = data.body("trunk").xpos[2]
                tilt    = np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))

                if policy_step % 50 == 0:
                    cmd_h     = torso_cmd_norm[0] / OBS_NORM[0] - OBS_OFFSET[0]
                    cmd_roll  = torso_cmd_norm[2] / OBS_NORM[2]
                    cmd_pitch = torso_cmd_norm[3] / OBS_NORM[3]
                    # actual roll = atan2(R[2,1], R[2,2]), pitch = -asin(R[2,0])
                    act_roll  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
                    act_pitch = np.degrees(-np.arcsin(np.clip(R[2, 0], -1, 1)))
                    err_h     = trunk_z - cmd_h
                    err_roll  = act_roll  - np.degrees(cmd_roll)
                    err_pitch = act_pitch - np.degrees(cmd_pitch)
                    phase = f"  [{sweep_phase} {sweep_count}/{args.dwell}]" if args.sweep else ""
                    print(f"  step {policy_step:5d} | "
                          f"h: cmd={cmd_h:.3f} act={trunk_z:.3f} err={err_h:+.3f}m | "
                          f"roll: cmd={np.degrees(cmd_roll):+.1f}° act={act_roll:+.1f}° err={err_roll:+.1f}° | "
                          f"pitch: cmd={np.degrees(cmd_pitch):+.1f}° act={act_pitch:+.1f}° err={err_pitch:+.1f}°"
                          f"{phase}")

                # Record for plot — radians, matching play_pi2.py
                _rec["cmd_h"].append(torso_cmd_norm[0] / OBS_NORM[0] - OBS_OFFSET[0])
                _rec["act_h"].append(trunk_z)
                _rec["cmd_h_dot"].append(torso_cmd_norm[1] / OBS_NORM[1])
                _rec["act_h_dot"].append(float(data.qvel[2]))
                _rec["cmd_roll"].append(torso_cmd_norm[2] / OBS_NORM[2])
                _rec["act_roll"].append(np.arctan2(R[2, 1], R[2, 2]))
                _rec["cmd_pitch"].append(torso_cmd_norm[3] / OBS_NORM[3])
                _rec["act_pitch"].append(-np.arcsin(np.clip(R[2, 0], -1, 1)))

                # Fall / stuck detection — reset without blocking the viewer
                if trunk_z < 0.20:
                    begin_reset("robot fell (trunk_z < 0.20)")
                    state = "pd_warmup"; warmup_count = 0
                elif tilt > 55:
                    begin_reset(f"robot tilted (tilt={tilt:.0f}°)")
                    state = "pd_warmup"; warmup_count = 0

                policy_step += 1

                # Real-time pacing (skipped when --speed 0)
                if args.speed > 0:
                    budget = POLICY_DT / args.speed
                    elapsed = time.time() - _step_start
                    if elapsed < budget:
                        time.sleep(budget - elapsed)
                _step_start = time.time()

            viewer.sync()


    # ── Save tracking plot ────────────────────────────────────────────────────
    if _rec["cmd_h"]:
        n = len(_rec["cmd_h"])
        t = np.arange(n) * DECIMATION * 0.005   # seconds

        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        fig.suptitle("Pi2 Torso Tracking — MuJoCo", fontsize=13)

        panels = [
            ("cmd_h",     "act_h",     "Height [m]",    (0.20, 0.55), "tab:orange"),
            ("cmd_h_dot", "act_h_dot", "h_dot [m/s]",   (-1.2,  1.2), "tab:purple"),
            ("cmd_roll",  "act_roll",  "Roll [rad]",    (-0.5,  0.5), "tab:green"),
            ("cmd_pitch", "act_pitch", "Pitch [rad]",   (-0.5,  0.5), "tab:red"),
        ]
        for ax, (ck, ak, ylabel, ylim, acolor) in zip(axes, panels):
            ax.plot(t, _rec[ck], label="cmd",    color="tab:blue", linestyle="--", linewidth=0.9)
            ax.plot(t, _rec[ak], label="actual", color=acolor,     linewidth=0.9, alpha=0.85)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_ylim(ylim)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)")

        plt.tight_layout()
        out_path = os.path.join(_ROOT, "videos", "pi2_tracking_mujoco.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"\n[pi2_test] saved tracking plot → {out_path}")
    else:
        print("\n[pi2_test] no running steps recorded — plot skipped")


if __name__ == "__main__":
    main()
