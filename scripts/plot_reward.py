"""Plot the ball-balance reward function shape.

Run:
    uv run python scripts/plot_reward.py
"""

import math
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Use non-interactive backend if no display
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Parameters (must match ball_balance_env_cfg.py)
# ---------------------------------------------------------------------------
STD = 0.10          # Gaussian std (metres)
WEIGHT = 2.0        # reward weight
PADDLE_RADIUS = 0.15  # termination radius (metres)
BALL_RADIUS = 0.020   # ping-pong ball (metres)
EPISODE_LENGTH_S = 10.0
SIM_DT = 1.0 / 200.0
DECIMATION = 4
POLICY_DT = SIM_DT * DECIMATION  # 0.02 s
STEPS = int(EPISODE_LENGTH_S / POLICY_DT)

# ---------------------------------------------------------------------------
# Reward landscape
# ---------------------------------------------------------------------------
r = np.linspace(0, PADDLE_RADIUS * 1.4, 500)  # XY distance from paddle centre (m)
reward = WEIGHT * np.exp(-(r**2) / (2 * STD**2))

# ---------------------------------------------------------------------------
# Time-series: reward over a perfect episode vs a drifting ball
# ---------------------------------------------------------------------------
t = np.arange(STEPS) * POLICY_DT

# Scenario A: ball perfectly centred (r=0 always) → reward = WEIGHT each step
r_perfect = np.zeros(len(t))

# Scenario B: ball drifts linearly from 0 to the termination boundary
drift_rate = PADDLE_RADIUS / EPISODE_LENGTH_S  # m/s
r_drift = drift_rate * t
# Terminate when ball exits paddle
terminate_idx = np.searchsorted(r_drift, PADDLE_RADIUS)

r_rew_perfect = WEIGHT * np.exp(-(r_perfect**2) / (2 * STD**2))
r_rew_drift   = WEIGHT * np.exp(-(r_drift**2) / (2 * STD**2))

# Clamp terminate_idx so it never exceeds array length
terminate_idx = min(terminate_idx, len(t) - 1)

cumsum_perfect = np.cumsum(r_rew_perfect)
cumsum_drift   = np.cumsum(r_rew_drift[:terminate_idx])

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Ball-Balance Reward Function Analysis", fontsize=13, fontweight="bold")

# ── Panel 1: reward vs XY distance ──────────────────────────────────────────
ax = axes[0]
ax.plot(r * 100, reward, lw=2, color="steelblue")
ax.axvline(PADDLE_RADIUS * 100, color="red", ls="--", lw=1.5, label=f"Termination ({PADDLE_RADIUS*100:.0f} cm)")
ax.axvline(STD * 100, color="orange", ls=":", lw=1.5, label=f"σ = {STD*100:.0f} cm  (r={WEIGHT*math.exp(-0.5):.2f})")
ax.axvline(BALL_RADIUS * 100, color="green", ls=":", lw=1.2, label=f"Ball radius ({BALL_RADIUS*100:.0f} mm)")
ax.set_xlabel("XY distance from paddle centre (cm)")
ax.set_ylabel("Reward per step")
ax.set_title("Reward landscape")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, PADDLE_RADIUS * 1.4 * 100)

# ── Panel 2: reward over time (two scenarios) ────────────────────────────────
ax = axes[1]
ax.plot(t, r_rew_perfect, lw=2, color="steelblue", label="Perfect balance (r=0)")
ax.plot(t[:terminate_idx], r_rew_drift[:terminate_idx], lw=2, color="tomato",
        label=f"Linear drift → termination at t={t[terminate_idx]:.1f}s")
ax.axhline(WEIGHT, color="steelblue", ls=":", alpha=0.5)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Reward per step")
ax.set_title("Per-step reward over episode")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Panel 3: cumulative return ────────────────────────────────────────────────
ax = axes[2]
ax.plot(t, cumsum_perfect, lw=2, color="steelblue",
        label=f"Perfect: {cumsum_perfect[-1]:.0f} total")
ax.plot(t[:terminate_idx], cumsum_drift, lw=2, color="tomato",
        label=f"Drift→term: {cumsum_drift[-1]:.0f} total ({terminate_idx} steps)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Cumulative return")
ax.set_title("Episode return")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "reward_analysis.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# Print summary
print()
print("=== Reward summary ===")
print(f"  At r=0 cm   (perfect centre): {WEIGHT * math.exp(0):.3f} / step")
print(f"  At r={STD*100:.0f} cm   (1σ):           {WEIGHT * math.exp(-0.5):.3f} / step")
print(f"  At r={STD*2*100:.0f} cm   (2σ):           {WEIGHT * math.exp(-2.0):.3f} / step")
print(f"  At r={PADDLE_RADIUS*100:.0f} cm  (termination): {WEIGHT * math.exp(-(PADDLE_RADIUS**2)/(2*STD**2)):.4f} / step")
print()
print(f"  Episode length:    {EPISODE_LENGTH_S:.0f} s  ({STEPS} policy steps at {1/POLICY_DT:.0f} Hz)")
print(f"  Max return:        {WEIGHT * STEPS:.0f}  (ball centred for full episode)")
print(f"  Drift return:      {cumsum_drift[-1]:.0f}  (linear drift to boundary)")

if os.environ.get("DISPLAY"):
    plt.show()
