"""Simple 1D mirror-law ball juggling demo — no robot.

Ball bounces on a controlled paddle. Mirror law sets the paddle velocity
at each impact to achieve the target apex height above the paddle.

Usage:
    python testing_codes/mirror_law_test.py
    python testing_codes/mirror_law_test.py --apex_height 0.60
    python testing_codes/mirror_law_test.py --apex_height 0.20 --restitution 0.90
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

parser = argparse.ArgumentParser(description="Mirror-law ball juggling demo")
parser.add_argument("--apex_height", type=float, default=0.40,
                    help="Target apex above resting paddle [m] (default 0.40)")
parser.add_argument("--restitution", type=float, default=0.85,
                    help="Coefficient of restitution (default 0.85)")
args = parser.parse_args()

# ── Constants ──────────────────────────────────────────────────────────────────
G    = 9.81
E    = args.restitution
APEX = args.apex_height
DT   = 0.001                        # physics step [s]
FPS  = 50
STEPS_PER_FRAME = int(1.0 / FPS / DT)

BALL_R  = 0.02                      # ball radius [m]
PAD_W   = 0.20                      # paddle width (visual) [m]
PAD_H   = 0.015                     # paddle thickness [m]
PAD_NOM = 0.0                       # resting height [m]
K_PAD   = 300.0                     # spring stiffness for paddle return
C_PAD   = 30.0                      # damping

# Desired post-impact ball velocity to reach apex
v_out = np.sqrt(2 * G * APEX)

# ── Simulation state ───────────────────────────────────────────────────────────
bz        = PAD_NOM + BALL_R + APEX * 0.85   # ball starts near apex
bvz       = 0.0
pz        = PAD_NOM
pvz       = 0.0
triggered = False    # True from "near+descending" trigger until post-contact
t_sim     = 0.0


def step():
    global bz, bvz, pz, pvz, triggered, t_sim

    descending = bvz < 0
    near       = (bz - pz - BALL_R) < 0.06   # within 6 cm of paddle surface

    # Mirror law: trigger on first step ball is descending and close
    if descending and near and not triggered:
        pvz = max(0.0, (v_out - E * abs(bvz)) / (1 + E))
        triggered = True

    # Paddle spring-damper return when idle
    if not triggered:
        pvz += (-K_PAD * (pz - PAD_NOM) - C_PAD * pvz) * DT

    # Integrate
    pz  += pvz * DT
    bvz -= G * DT
    bz  += bvz * DT
    t_sim += DT

    # Contact resolution
    if bz <= pz + BALL_R:
        bz  = pz + BALL_R
        bvz = pvz + E * (pvz - bvz)
        triggered = False


# ── Matplotlib setup ───────────────────────────────────────────────────────────
ymax = max(APEX * 1.8, 0.5)
fig, ax = plt.subplots(figsize=(5, 8))
ax.set_xlim(-0.4, 0.4)
ax.set_ylim(-0.12, ymax)
ax.set_aspect("equal")
ax.set_xlabel("x  [m]")
ax.set_ylabel("height  [m]")
ax.set_title(f"Mirror Law  —  apex target = {APEX:.2f} m  (e = {E:.2f})", fontsize=11)

ax.axhline(PAD_NOM + APEX, color="tomato", linestyle="--", linewidth=1.5,
           label=f"apex target  {APEX:.2f} m")
ax.axhline(PAD_NOM, color="black", linewidth=0.6, linestyle=":")
ax.legend(loc="upper right", fontsize=9)

ball_patch = plt.Circle((0, bz), BALL_R, color="steelblue", zorder=4)
pad_patch  = mpatches.FancyBboxPatch(
    (-PAD_W / 2, pz - PAD_H), PAD_W, PAD_H,
    boxstyle="round,pad=0.003", color="dimgray", zorder=3,
)
ax.add_patch(ball_patch)
ax.add_patch(pad_patch)

info = ax.text(0.03, 0.97, "", transform=ax.transAxes, fontsize=9,
               va="top", family="monospace")


def animate(__frame):
    for _ in range(STEPS_PER_FRAME):
        step()
    ball_patch.set_center((0, bz))
    pad_patch.set_y(pz - PAD_H)
    info.set_text(
        f"t      = {t_sim:5.2f} s\n"
        f"ball z = {bz:.3f} m\n"
        f"ball vz= {bvz:+.2f} m/s\n"
        f"pad vz = {pvz:+.2f} m/s"
    )
    return ball_patch, pad_patch, info


ani = animation.FuncAnimation(
    fig, animate, interval=int(1000 / FPS), blit=True, cache_frame_data=False
)
plt.tight_layout()
plt.show()
