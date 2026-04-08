# EKF Contact-Aware Mode — Coordination Note for Policy Agent

**Date**: 2026-04-08  
**From**: perception agent  
**For**: policy agent

## Issue

The `ball_ekf.py` on `agent/policy` has been simplified: contact-aware 3-level
q_vel, NIS gating, 9D spin, and phase-separated NIS tracking were all removed.
This is fine while training with `--noise-mode d435i` (raw noise, no EKF), but
**will be critical when switching to `--noise-mode ekf`**.

## Why Contact-Aware Matters

Our GPU validation (iters 019-055) found:

| Setting | Flight NIS | Contact NIS | Overall NIS |
|---------|-----------|-------------|-------------|
| Fixed q_vel=0.40 (no contact mode) | 52.9 | 19.9 | ~40 |
| Contact-aware (0.40/50.0) | 1.45 | 5.3 | 1.60 |

**Without contact-aware mode, the EKF is ~30x overconfident during active
juggling.** The root cause: paddle contact produces near-instant velocity
reversals that violate the ballistic prediction model. Inflating Q during
contact lets the filter trust measurements over its (wrong) prediction.

## What to Do

When switching to `--noise-mode ekf`, use the perception agent's `ball_ekf.py`
which has:
- `contact_aware = True`
- `q_vel_contact = 50.0` (during paddle contact)
- `q_vel_post_contact = 20.0` (10 steps after contact)
- `q_vel = TBD` (flight value — sweep in progress, currently 0.40)

The canonical file is on `agent/perception:source/go1_ball_balance/go1_ball_balance/perception/ball_ekf.py`.

Read it with:
```
git show agent/perception:source/go1_ball_balance/go1_ball_balance/perception/ball_ekf.py
```

## Current Status

- GPU q_vel sweep script (`sweep_q_vel.py`) is ready but waiting for GPU
- Goal: find flight q_vel where NIS ≈ 3.0 (slightly conservative, good for training)
- Once tuned, will update BallEKFConfig defaults on perception branch
