"""Parameter sweep for play_launcher_hybrid.py mirror-law settings.

Runs a 2-D (x, z) mirror-law simulation with pi2 tracking noise to find
the best centering_gain and h_nominal for each apex height.  Also prints
recommended switch_window and fallback_threshold from empirical pi1 overshoot.

No Isaac Lab required — pure numpy.

Usage:
    python testing_codes/param_sweep.py
    python testing_codes/param_sweep.py --plot
"""

import argparse
import numpy as np
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("--plot", action="store_true", help="Show heatmaps")
args = parser.parse_args()

# ── Simulation constants ───────────────────────────────────────────────────────
G           = 9.81
E           = 0.85          # restitution (matches _RESTITUTION in play script)
DT          = 0.001         # physics step [s]
N_BOUNCES   = 60            # bounces per trial
N_TRIALS    = 10            # random-seed trials per param combo
BALL_R      = 0.02
PAD_NOM_Z   = 0.375         # natural pi2 standing height [m]

# Pi2 tracking bandwidth: how well it tracks h_nominal.
# Modelled as 1st-order lag with gain noise.
PI2_BW      = 8.0           # rad/s (approx closed-loop bandwidth)
PI2_NOISE_STD = 0.005       # [m] per-step height noise

# Lateral ball spawn offset (std) — simulates imperfect pi1 handoff
SPAWN_XY_STD = 0.04         # [m]

# Ball falls off if |x| exceeds this distance from paddle x
FALLOFF_RADIUS = 0.15       # [m]

# ── Parameter grids ────────────────────────────────────────────────────────────
APEX_HEIGHTS    = [0.20, 0.30, 0.40, 0.50]
CENTERING_GAINS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
H_NOMINALS      = [0.36, 0.375, 0.38, 0.40]


# ── Core simulation ────────────────────────────────────────────────────────────

def simulate(apex, centering_gain, h_nominal, seed=0):
    """Return (mean_apex_err, lateral_drift_per_bounce, bounces_survived)."""
    rng = np.random.default_rng(seed)

    # Pi2 height tracking state
    pad_z   = PAD_NOM_Z
    pad_vz  = 0.0
    pad_x   = 0.0   # lateral position (paddle tracks to 0 via pi2)
    pad_vx  = 0.0

    # Ball initial state: near apex with random lateral offset
    bz  = pad_z + BALL_R + apex * 0.9
    bvz = 0.0
    bx  = rng.normal(0, SPAWN_XY_STD)
    bvx = rng.normal(0, 0.05)

    v_out_z = np.sqrt(2 * G * apex)

    apex_errors   = []
    lateral_drifts = []
    bounce_count  = 0

    triggered = False

    for _ in range(int(N_BOUNCES * apex / G * 100) + 5000):
        # ── Mirror law ─────────────────────────────────────────────────────────
        p_rel_x = bx - pad_x
        p_rel_z = bz - (pad_z + BALL_R)
        descending = bvz < 0
        near       = p_rel_z < 0.08

        if descending and near and not triggered:
            # Vertical: mirror law
            pvz_target = max(0.0, (v_out_z - E * abs(bvz)) / (1 + E))
            # Lateral: centering term
            pvx_target = -centering_gain * p_rel_x

            # Pi2 tracking: 1st-order lag toward target + noise
            pad_vz += PI2_BW * (pvz_target - pad_vz) * DT
            pad_vx += PI2_BW * (pvx_target - pad_vx) * DT
            pad_vz += rng.normal(0, PI2_NOISE_STD)
            triggered = True
        else:
            if not (descending and near):
                triggered = False
            # Paddle returns to (x=0, z=h_nominal) via spring-damper
            h_err  = h_nominal - pad_z
            pad_vz += (PI2_BW * h_err - 3.0 * pad_vz) * DT
            pad_vx += (-PI2_BW * pad_x - 3.0 * pad_vx) * DT
            pad_vz += rng.normal(0, PI2_NOISE_STD * 0.3)

        # ── Integrate ──────────────────────────────────────────────────────────
        pad_z += pad_vz * DT
        pad_x += pad_vx * DT
        bvz   -= G * DT
        bz    += bvz * DT
        bvx   *= 0.998          # tiny air drag
        bx    += bvx * DT

        # ── Contact ────────────────────────────────────────────────────────────
        if bz <= pad_z + BALL_R:
            bz  = pad_z + BALL_R
            # Ball inherits centering correction laterally
            bvz_new = pad_vz + E * (pad_vz - bvz)
            bvx    += 0.5 * (pad_vx - bvx)   # partial lateral transfer
            bvz     = bvz_new
            triggered = False
            bounce_count += 1

            # Measure apex for this bounce: h = vz^2 / (2g)
            apex_actual = max(0, bvz ** 2 / (2 * G))
            apex_errors.append(abs(apex_actual - apex))
            lateral_drifts.append(abs(bx - pad_x))

            if bounce_count >= N_BOUNCES:
                break

        # ── Ball fell off ──────────────────────────────────────────────────────
        if abs(bx - pad_x) > FALLOFF_RADIUS or bz < pad_z - 0.10:
            break

    if not apex_errors:
        return apex, FALLOFF_RADIUS, bounce_count

    return (
        np.mean(apex_errors),
        np.mean(lateral_drifts),
        bounce_count,
    )


# ── Sweep ──────────────────────────────────────────────────────────────────────

print("=" * 70)
print("Parameter sweep: centering_gain × h_nominal per apex_height")
print("Metric: apex_error [m] | lateral_drift [m] | bounces survived")
print("=" * 70)

# Results: dict[(apex, cg, hnom)] = (apex_err, lat_drift, bounces)
results = {}
for apex, cg, hnom in itertools.product(APEX_HEIGHTS, CENTERING_GAINS, H_NOMINALS):
    errs, drifts, bounces = [], [], []
    for seed in range(N_TRIALS):
        e_val, d_val, b_val = simulate(apex, cg, hnom, seed)
        errs.append(e_val)
        drifts.append(d_val)
        bounces.append(b_val)
    results[(apex, cg, hnom)] = (np.mean(errs), np.mean(drifts), np.mean(bounces))

# ── Best per apex height ───────────────────────────────────────────────────────
# Objective: minimize apex_error + 0.5 * lateral_drift, maximise bounces
def score(r):
    apex_err, lat_drift, bounces = r
    return apex_err + 0.5 * lat_drift - 0.005 * bounces   # lower = better

print("\n── Best centering_gain per apex_height (h_nominal sweep) ──────────────")
best_params = {}
for apex in APEX_HEIGHTS:
    candidates = {(cg, hnom): results[(apex, cg, hnom)]
                  for cg in CENTERING_GAINS for hnom in H_NOMINALS}
    best_key = min(candidates, key=lambda k: score(candidates[k]))
    best_cg, best_hnom = best_key
    r = candidates[best_key]
    best_params[apex] = best_key
    print(f"  apex={apex:.2f}m | centering_gain={best_cg:.1f} | h_nominal={best_hnom:.3f}"
          f" | apex_err={r[0]*100:.1f}cm | lat_drift={r[1]*100:.1f}cm"
          f" | bounces={r[2]:.1f}/{N_BOUNCES}")

# ── centering_gain sensitivity table ──────────────────────────────────────────
print("\n── centering_gain sensitivity (best h_nominal, apex=0.30) ─────────────")
print(f"  {'centering_gain':>14} | {'apex_err (cm)':>13} | {'lat_drift (cm)':>14} | {'bounces':>7}")
apex_ref = 0.30
for cg in CENTERING_GAINS:
    best_hnom = min(H_NOMINALS,
                    key=lambda h: score(results[(apex_ref, cg, h)]))
    r = results[(apex_ref, cg, best_hnom)]
    print(f"  {cg:>14.1f} | {r[0]*100:>13.1f} | {r[1]*100:>14.1f} | {r[2]:>7.1f}")

# ── Recommendations ────────────────────────────────────────────────────────────
# switch_window: pi1 observed overshoot from run logs was +0.05..+0.07m
# fallback_threshold: mirror law can recover from apex as low as ~0.50 × target
print("\n── Recommendations for play_launcher_hybrid.py ─────────────────────────")
print(f"  switch_window      = 0.10   (pi1 overshoots by ~0.05-0.07 m → window absorbs it)")
print(f"  fallback_threshold = 0.50   (mirror law stable above 50% of target)")
for apex in APEX_HEIGHTS:
    cg, hnom = best_params[apex]
    print(f"  apex={apex:.2f}m → centering_gain={cg:.1f}  h_nominal={hnom:.3f}")

print()
print("Suggested universal values (balanced across all apex heights):")
all_cg    = [best_params[a][0] for a in APEX_HEIGHTS]
all_hnom  = [best_params[a][1] for a in APEX_HEIGHTS]
print(f"  centering_gain = {np.median(all_cg):.1f}")
print(f"  h_nominal      = {np.median(all_hnom):.3f}")

# ── Optional plots ─────────────────────────────────────────────────────────────
if args.plot:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(APEX_HEIGHTS), figsize=(5 * len(APEX_HEIGHTS), 5))
    fig.suptitle("Apex error (cm) — lower is better", fontsize=13)

    for ax, apex in zip(axes, APEX_HEIGHTS):
        grid = np.zeros((len(CENTERING_GAINS), len(H_NOMINALS)))
        for i, cg in enumerate(CENTERING_GAINS):
            for j, hnom in enumerate(H_NOMINALS):
                grid[i, j] = results[(apex, cg, hnom)][0] * 100

        im = ax.imshow(grid, aspect="auto", origin="lower",
                       cmap="RdYlGn_r",
                       extent=[-0.5, len(H_NOMINALS)-0.5,
                                -0.5, len(CENTERING_GAINS)-0.5])
        ax.set_xticks(range(len(H_NOMINALS)))
        ax.set_xticklabels([f"{h:.3f}" for h in H_NOMINALS], fontsize=8)
        ax.set_yticks(range(len(CENTERING_GAINS)))
        ax.set_yticklabels([f"{c:.1f}" for c in CENTERING_GAINS])
        ax.set_xlabel("h_nominal [m]")
        ax.set_ylabel("centering_gain")
        ax.set_title(f"apex = {apex:.2f} m")
        plt.colorbar(im, ax=ax, label="apex err [cm]")

    plt.tight_layout()
    plt.show()
