"""Verify that load_pi2() from test_pi2_mujoco.py gives identical output to the Isaac Lab golden data.

Uses golden (obs, action) pairs from tests_out/pi2_golden.csv collected by test_pi2_golden.py.
Feeds each obs through the MuJoCo loader and compares to stored Isaac Lab actions.
A max error < 1e-4 confirms the two loaders are bit-equivalent.

No MuJoCo sim required — pure torch + pandas.

Usage:
    python scripts/tests/verify_pi2_golden_mujoco.py
"""
import os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Pull in load_pi2 from test_pi2_mujoco.py
_SCRIPTS = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _SCRIPTS)
from test_pi2_mujoco import load_pi2

_ROOT      = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))
CHECKPOINT = os.path.join(_ROOT, "logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt")
GOLDEN_CSV = os.path.join(_ROOT, "tests_out/pi2_golden.csv")

JOINT_NAMES = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint",  "FR_calf_joint",  "RL_calf_joint",  "RR_calf_joint",
]

OBS_COLS = (
    ["obs_cmd_norm_h", "obs_cmd_norm_h_dot", "obs_cmd_norm_roll",
     "obs_cmd_norm_pitch", "obs_cmd_norm_roll_rate", "obs_cmd_norm_pitch_rate"]   # [0:6]
  + ["obs_linvel_x", "obs_linvel_y", "obs_linvel_z"]                              # [6:9]
  + ["obs_angvel_x", "obs_angvel_y", "obs_angvel_z"]                              # [9:12]
  + ["obs_grav_x",   "obs_grav_y",   "obs_grav_z"]                               # [12:15]
  + [f"obs_jp_{j}" for j in JOINT_NAMES]                                          # [15:27]
  + [f"obs_jv_{j}" for j in JOINT_NAMES]                                          # [27:39]
)
ACT_COLS = [f"act_{j}" for j in JOINT_NAMES]


actor = load_pi2(CHECKPOINT)
df    = pd.read_csv(GOLDEN_CSV)

print(f"[verify_mujoco] loaded golden CSV: {len(df)} cases from {GOLDEN_CSV}\n")
print(f"{'Case':<20}  {'|act_err|_max':>13}  {'|act_err|_mean':>14}  {'match?':>8}")
print("-" * 62)

all_max_err = []
for _, row in df.iterrows():
    obs_np  = row[OBS_COLS].values.astype(np.float32)   # (39,)
    act_ref = row[ACT_COLS].values.astype(np.float32)   # (12,) from Isaac Lab

    obs_t = torch.from_numpy(obs_np).unsqueeze(0)       # (1, 39)
    with torch.no_grad():
        act_pred = actor(obs_t).squeeze(0).numpy()      # (12,) from MuJoCo loader

    err     = np.abs(act_pred - act_ref)
    max_err = err.max()
    all_max_err.append(max_err)
    match   = "✓" if max_err < 1e-4 else "✗"

    print(f"{row['case']:<20}  {max_err:>13.6f}  {err.mean():>14.6f}  {match:>8}")

    if max_err >= 1e-4:
        print("  per-joint breakdown:")
        for jn, ref, pred, e in zip(JOINT_NAMES, act_ref, act_pred, err):
            if e > 1e-5:
                print(f"    {jn:<25} ref={ref:+.6f}  pred={pred:+.6f}  err={e:.2e}")

print("-" * 62)
overall = max(all_max_err)
verdict = "PASS" if overall < 1e-4 else "FAIL"
print(f"{'Max across all cases':<20}  {overall:>13.6f}  [{verdict}]")
print()
if verdict == "PASS":
    print("load_pi2() is bit-equivalent to the Isaac Lab actor loader.")
else:
    print("WARNING: loaders differ — check key mapping in load_pi2().")
