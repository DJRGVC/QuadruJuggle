"""Verify pi2 golden data: reconstruct obs from CSV, run actor, compare to stored actions.

No Isaac Lab needed — pure torch + pandas.

Usage:
    python scripts/tests/verify_pi2_golden.py
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../.."))

CHECKPOINT = os.path.join(_ROOT, "logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt")
GOLDEN_CSV = os.path.join(_ROOT, "tests_out/pi2_golden.csv")

JOINT_NAMES = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint",  "FR_calf_joint",  "RL_calf_joint",  "RR_calf_joint",
]

# obs column order (must match training layout exactly)
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


def build_actor(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck.get("model_state_dict", ck)
    actor_keys = sorted(k for k in sd if k.startswith("actor.") and "weight" in k)
    layers = [(sd[k].shape[1], sd[k].shape[0]) for k in actor_keys]
    modules = []
    for i, (in_dim, out_dim) in enumerate(layers):
        modules.append(nn.Linear(in_dim, out_dim))
        if i < len(layers) - 1:
            modules.append(nn.ELU())
    actor = nn.Sequential(*modules)
    actor_sd = {}
    for key in actor_keys:
        idx = int(key.split(".")[1])
        actor_sd[f"{idx}.weight"] = sd[key]
        bkey = key.replace("weight", "bias")
        if bkey in sd:
            actor_sd[f"{idx}.bias"] = sd[bkey]
    actor.load_state_dict(actor_sd)
    actor.eval()
    for p in actor.parameters(): p.requires_grad = False
    arch = " → ".join(f"{i}→{o}" for i, o in layers)
    print(f"[verify] actor arch : {arch}")
    return actor


actor = build_actor(CHECKPOINT)
df    = pd.read_csv(GOLDEN_CSV)

print(f"[verify] loaded {len(df)} golden cases from {GOLDEN_CSV}\n")
print(f"{'Case':<20}  {'|act_err|_max':>13}  {'|act_err|_mean':>14}  {'match?':>8}")
print("-" * 62)

all_max_err = []
for _, row in df.iterrows():
    obs_np  = row[OBS_COLS].values.astype(np.float32)        # (39,)
    act_ref = row[ACT_COLS].values.astype(np.float32)        # (12,)

    obs_t = torch.from_numpy(obs_np).unsqueeze(0)            # (1, 39)
    with torch.no_grad():
        act_pred = actor(obs_t).squeeze(0).numpy()           # (12,)

    err     = np.abs(act_pred - act_ref)
    max_err = err.max()
    all_max_err.append(max_err)
    match   = "✓" if max_err < 1e-4 else "✗"

    print(f"{row['case']:<20}  {max_err:>13.6f}  {err.mean():>14.6f}  {match:>8}")

    if max_err >= 1e-4:
        print(f"  joint errors:")
        for jn, ref, pred, e in zip(JOINT_NAMES, act_ref, act_pred, err):
            if e > 1e-5:
                print(f"    {jn:<25} ref={ref:+.6f}  pred={pred:+.6f}  err={e:.2e}")

print("-" * 62)
print(f"{'Max across all cases':<20}  {max(all_max_err):>13.6f}")
