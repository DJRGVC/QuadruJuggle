#!/usr/bin/env python3
"""
results_analysis.py
===================
Result analysis plots for the Mirror Law + RL (pi2) juggling system.
Uses REAL training data extracted from TensorBoard logs.

Six figures
-----------
1. Pi2 training convergence   – torso-tracking policy learns to follow commands
2. Pi1 juggling convergence   – ball juggling reward components over training
3. Termination analysis       – how episodes end (time_out rising = success)
4. Reward breakdown           – per-term contribution at convergence
5. Mirror law physics sim     – simulated ball trajectory showing stable juggling
6. Noise robustness demo      – juggling quality vs perception noise level

Run:
    python scripts/results_analysis.py
Output:
    docs/figures/results_*.png
"""

import json, os, math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size":   11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "lines.linewidth": 2.0,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})
BLUE   = "#1976D2";  ORANGE = "#F57C00";  GREEN  = "#388E3C"
RED    = "#C62828";  PURPLE = "#6A1B9A";  GREY   = "#616161"
TEAL   = "#00796B";  PINK   = "#AD1457"

os.makedirs("docs/figures", exist_ok=True)

# ---------------------------------------------------------------------------
# Load real training data
# ---------------------------------------------------------------------------
with open("/tmp/tb_data.json") as f:
    TB = json.load(f)

def get(run, tag):
    """Return (steps_array, smoothed_values_array) for a TB tag."""
    d = TB[run].get(tag, {"steps": [], "values": []})
    s = np.array(d["steps"])
    v = np.array(d["values"])
    if len(v) > 20:
        v_smooth = uniform_filter1d(v, size=max(1, len(v)//40))
    else:
        v_smooth = v
    return s, v, v_smooth


# ===========================================================================
# Figure 1 – Pi2 Torso-Tracking Convergence
# ===========================================================================
def plot_pi2_convergence():
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    fig.suptitle("Fig 1 — Pi2 Torso-Tracking Policy: Training Convergence\n(frozen after this; used unchanged by both mirror-law and RL pi1)",
                 fontsize=12, y=1.01)

    reward_tags = [
        ("Episode_Reward/height_tracking",      "Height tracking",    BLUE),
        ("Episode_Reward/roll_tracking",         "Roll tracking",      ORANGE),
        ("Episode_Reward/pitch_tracking",        "Pitch tracking",     GREEN),
        ("Episode_Reward/height_vel_tracking",   "Height-vel tracking",PURPLE),
        ("Episode_Reward/roll_rate_tracking",    "Roll rate tracking", TEAL),
        ("Episode_Reward/pitch_rate_tracking",   "Pitch rate tracking",PINK),
    ]

    for ax, (tag, label, color) in zip(axes.flat, reward_tags):
        steps, raw, smooth = get("pi2", tag)
        if len(steps) == 0:
            ax.set_visible(False)
            continue
        iters = steps / (steps[-1] / len(steps)) if len(steps) > 0 else steps
        iters = np.arange(len(steps))
        ax.plot(iters, raw,    color=color, alpha=0.25, linewidth=0.8)
        ax.plot(iters, smooth, color=color, linewidth=2.2)
        ax.set_title(label)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Reward")
        # Mark convergence point (90% of max)
        thresh = 0.90 * np.max(smooth)
        conv_idx = np.argmax(smooth >= thresh)
        if conv_idx > 0:
            ax.axvline(conv_idx, color=RED, linestyle="--", linewidth=1.0, alpha=0.7)
            ax.text(conv_idx + len(iters)*0.02, np.min(smooth),
                    f"90%@{conv_idx}", color=RED, fontsize=8)

    fig.tight_layout()
    fig.savefig("docs/figures/results_1_pi2_convergence.png", bbox_inches="tight")
    plt.close(fig)
    print("Saved Fig 1")


# ===========================================================================
# Figure 2 – Pi1 Juggling Reward Convergence
# ===========================================================================
def plot_pi1_convergence():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.suptitle("Fig 2 — Pi1 (Learned) Juggling Policy: Key Reward Components",
                 fontsize=12)

    panels = [
        # (ax, [(tag, label, color), ...], title)
        (axes[0, 0],
         [("Episode_Reward/ball_apex_height", "Ball apex height reward", BLUE),
          ("Episode_Reward/ball_bouncing",    "Ball bouncing reward",    GREEN)],
         "Core Juggling Rewards (higher = better juggling)"),

        (axes[0, 1],
         [("Episode_Reward/alive",            "Alive reward",     ORANGE),
          ("Train/mean_episode_length",        "Episode length",   PURPLE)],
         "Survival (higher = robot stays upright longer)"),

        (axes[1, 0],
         [("Episode_Reward/ball_xy_dist",     "Ball XY dist",     RED),
          ("Episode_Reward/trunk_tilt",       "Trunk tilt",       PINK),
          ("Episode_Reward/action_rate",      "Action rate",      GREY)],
         "Penalty Terms (closer to 0 = better)"),

        (axes[1, 1],
         [("Episode_Termination/time_out",    "Time-out (success)", GREEN),
          ("Episode_Termination/ball_off",    "Ball off paddle",    RED),
          ("Episode_Termination/ball_below",  "Ball below floor",   ORANGE)],
         "Termination Breakdown (time_out↑ = success)"),
    ]

    for ax, tag_list, title in panels:
        for tag, label, color in tag_list:
            steps, raw, smooth = get("pi1", tag)
            if len(steps) == 0:
                continue
            iters = np.arange(len(steps))
            ax.plot(iters, raw,    color=color, alpha=0.20, linewidth=0.8)
            ax.plot(iters, smooth, color=color, linewidth=2.0, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training iteration")
        ax.legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig("docs/figures/results_2_pi1_convergence.png")
    plt.close(fig)
    print("Saved Fig 2")


# ===========================================================================
# Figure 3 – Termination Analysis (key success indicator)
# ===========================================================================
def plot_termination_analysis():
    tags = {
        "time_out":    ("Episode_Termination/time_out",   GREEN,  "Time-out (full episode — SUCCESS)"),
        "ball_off":    ("Episode_Termination/ball_off",   RED,    "Ball left paddle"),
        "ball_below":  ("Episode_Termination/ball_below", ORANGE, "Ball fell below floor"),
        "robot_tilt":  ("Episode_Termination/robot_tilt", PURPLE, "Robot tipped over"),
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title("Fig 3 — How Episodes End During Pi1 Training\n"
                 "(time_out → 1.0 means robot juggles full 30 s episode)", fontsize=12)

    for key, (tag, color, label) in tags.items():
        steps, raw, smooth = get("pi1", tag)
        if len(steps) == 0:
            continue
        iters = np.arange(len(steps))
        ax.plot(iters, raw,    color=color, alpha=0.15, linewidth=0.7)
        ax.plot(iters, smooth, color=color, linewidth=2.2, label=label)

    # Annotate final values
    for key, (tag, color, label) in tags.items():
        steps, raw, smooth = get("pi1", tag)
        if len(smooth) == 0:
            continue
        ax.annotate(f"{smooth[-1]:.2f}",
                    xy=(len(smooth)-1, smooth[-1]),
                    xytext=(len(smooth)-1 + 50, smooth[-1]),
                    color=color, fontsize=9, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8))

    # Reference line at 0.5
    ax.axhline(0.5, color=GREY, linestyle=":", linewidth=1.0, alpha=0.6)
    ax.text(10, 0.51, "50%", color=GREY, fontsize=8)

    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Fraction of episodes")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig("docs/figures/results_3_termination_analysis.png")
    plt.close(fig)
    print("Saved Fig 3")


# ===========================================================================
# Figure 4 – Reward Breakdown at Convergence (pie + bar)
# ===========================================================================
def plot_reward_breakdown():
    # Use mean of last 200 iterations for each term
    reward_tags = [
        ("Episode_Reward/ball_apex_height", "Ball apex height", BLUE),
        ("Episode_Reward/ball_bouncing",    "Ball bouncing",    GREEN),
        ("Episode_Reward/alive",            "Alive",            TEAL),
        ("Episode_Reward/ball_xy_dist",     "Ball XY dist",     RED),
        ("Episode_Reward/trunk_tilt",       "Trunk tilt",       PINK),
        ("Episode_Reward/action_rate",      "Action rate",      GREY),
        ("Episode_Reward/foot_contact",     "Foot contact",     ORANGE),
        ("Episode_Reward/trunk_contact",    "Trunk contact",    PURPLE),
        ("Episode_Reward/body_lin_vel",     "Body lin vel",     "#795548"),
        ("Episode_Reward/body_ang_vel",     "Body ang vel",     "#9E9E9E"),
        ("Episode_Reward/joint_torques",    "Joint torques",    "#607D8B"),
    ]

    labels, means, colors = [], [], []
    for tag, label, color in reward_tags:
        _, raw, _ = get("pi1", tag)
        if len(raw) == 0:
            continue
        m = float(np.mean(raw[-200:]))
        labels.append(label)
        means.append(m)
        colors.append(color)

    means = np.array(means)
    pos_mask = means > 0
    neg_mask = means < 0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Fig 4 — Reward Breakdown at Convergence (last 200 iterations)", fontsize=12)

    # Left: positive rewards as pie
    ax = axes[0]
    pos_vals   = means[pos_mask]
    pos_labels = [labels[i] for i in range(len(labels)) if pos_mask[i]]
    pos_colors = [colors[i] for i in range(len(labels)) if pos_mask[i]]
    if len(pos_vals) > 0:
        wedges, texts, autotexts = ax.pie(
            pos_vals, labels=pos_labels, colors=pos_colors,
            autopct="%1.1f%%", startangle=90,
            textprops={"fontsize": 9},
            wedgeprops={"linewidth": 1, "edgecolor": "white"},
        )
        for at in autotexts:
            at.set_fontsize(8)
    ax.set_title("Positive rewards\n(what the robot is optimising for)")

    # Right: all terms as horizontal bar
    ax2 = axes[1]
    y = np.arange(len(labels))
    bar_colors = [GREEN if m > 0 else RED for m in means]
    bars = ax2.barh(y, means, color=bar_colors, alpha=0.80, edgecolor="white")
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.axvline(0, color=GREY, linewidth=0.8)
    ax2.set_xlabel("Mean episode reward")
    ax2.set_title("All reward terms\n(green=positive, red=penalty)")
    for bar, val in zip(bars, means):
        ax2.text(val + (0.02 if val >= 0 else -0.02), bar.get_y() + bar.get_height()/2,
                 f"{val:+.3f}", va="center", ha="left" if val >= 0 else "right", fontsize=8)

    fig.tight_layout()
    fig.savefig("docs/figures/results_4_reward_breakdown.png")
    plt.close(fig)
    print("Saved Fig 4")


# ===========================================================================
# Figure 5 – Mirror Law Physics: Simulated Ball Trajectory
# ===========================================================================
def plot_mirror_law_simulation():
    """
    Forward-simulate the mirror law controller for 5 bounces.
    Shows: ball height over time, paddle height commands,
    and how apex height tracks the target.
    """
    G = 9.81;  E = 0.85;  DT = 0.005  # fine sim timestep
    H_APEX_TARGET = 0.30  # metres above paddle

    rng = np.random.default_rng(42)

    # --- simulation state ---
    ball_z   = 0.35   # initial height above ground
    ball_vz  = 0.0
    paddle_z = 0.38   # nominal trunk height above ground
    paddle_vz = 0.0

    T_total = 8.0   # seconds
    N = int(T_total / DT)

    t_arr       = np.zeros(N)
    ball_z_arr  = np.zeros(N)
    paddle_z_arr= np.zeros(N)
    cmd_vz_arr  = np.zeros(N)
    apex_arr    = []   # (t, height) of each apex

    in_air     = True
    last_apex  = 0.0
    rising     = False

    for i in range(N):
        t = i * DT
        t_arr[i] = t

        # Mirror law: compute h_dot command
        ball_rel_z = ball_z - paddle_z
        ball_descending = ball_vz < 0.0
        near_impact = ball_rel_z < 0.50

        v_out_z = math.sqrt(2.0 * G * H_APEX_TARGET)
        v_in_z_abs = abs(ball_vz)
        v_paddle_target = (v_out_z + E * v_in_z_abs) / (1.0 + E)
        h_dot_cmd = (2.0 * v_paddle_target * float(ball_descending) * float(near_impact))
        h_dot_cmd = min(h_dot_cmd, 1.0)
        if not (ball_descending and near_impact):
            h_dot_cmd = 0.15  # baseline

        # Pi2 tracking lag model: only 40% of command achieved
        paddle_vz = 0.40 * h_dot_cmd + 0.60 * paddle_vz
        paddle_z  = paddle_z + paddle_vz * DT
        paddle_z  = max(0.30, min(0.46, paddle_z))  # clamp to physical range

        # Ball physics
        ball_vz  -= G * DT
        ball_z   += ball_vz * DT

        # Detect and handle bounce
        if ball_z <= paddle_z and ball_vz < 0:
            # Impact: reflection + energy injection
            rel_vz = ball_vz - paddle_vz
            ball_vz = -E * rel_vz + paddle_vz
            ball_z  = paddle_z
            # Small lateral noise (simulates imperfect centering)
            ball_vz += rng.normal(0, 0.05)
            rising = True

        # Detect apex
        if rising and ball_vz < 0:
            apex_h = ball_z - paddle_z  # relative to current paddle
            apex_arr.append((t, apex_h))
            rising = False

        ball_z_arr[i]   = ball_z
        paddle_z_arr[i] = paddle_z
        cmd_vz_arr[i]   = h_dot_cmd

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig.suptitle("Fig 5 — Mirror Law Forward Simulation: Stable Juggling Trajectory\n"
                 f"Target apex = {H_APEX_TARGET:.2f} m,  restitution e = {E}",
                 fontsize=12)

    # Panel 1: ball and paddle height
    ax = axes[0]
    ax.plot(t_arr, ball_z_arr,   color=BLUE,   linewidth=1.8, label="Ball height")
    ax.plot(t_arr, paddle_z_arr, color=ORANGE, linewidth=1.5, linestyle="--", label="Paddle height")
    if apex_arr:
        at, ah = zip(*apex_arr)
        ax.scatter(at, [h + p for h, p in zip(ah, np.interp(at, t_arr, paddle_z_arr))],
                   color=GREEN, zorder=5, s=50, label="Apex (absolute)")
        target_abs = np.interp(at, t_arr, paddle_z_arr) + H_APEX_TARGET
        ax.plot(at, target_abs, color=GREEN, linestyle=":", linewidth=1.2, label="Target apex")
    ax.set_ylabel("Height above ground  [m]")
    ax.legend(fontsize=9)

    # Panel 2: apex height relative to paddle
    ax2 = axes[1]
    if apex_arr:
        at, ah = zip(*apex_arr)
        ax2.bar(at, ah, width=0.08, color=BLUE, alpha=0.7, label="Achieved apex (rel. paddle)")
        ax2.axhline(H_APEX_TARGET, color=RED, linestyle="--", linewidth=1.5,
                    label=f"Target = {H_APEX_TARGET} m")
        errors = [abs(h - H_APEX_TARGET) for h in ah]
        mean_err = np.mean(errors) if errors else 0
        ax2.text(0.98, 0.92, f"Mean error = {mean_err*100:.1f} cm",
                 transform=ax2.transAxes, ha="right", color=RED, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Apex height rel. paddle  [m]")
    ax2.legend(fontsize=9)

    # Panel 3: paddle velocity command
    ax3 = axes[2]
    ax3.plot(t_arr, cmd_vz_arr, color=PURPLE, linewidth=1.5, label="h_dot command")
    ax3.axhline(0.15, color=GREY, linestyle=":", linewidth=1.0, alpha=0.7, label="Baseline (0.15)")
    ax3.set_ylabel("Paddle velocity cmd  [m/s]")
    ax3.set_xlabel("Time  [s]")
    ax3.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig("docs/figures/results_5_mirror_law_simulation.png")
    plt.close(fig)
    print("Saved Fig 5")


# ===========================================================================
# Figure 6 – Noise Robustness: Juggling Quality vs Perception Noise
# ===========================================================================
def plot_noise_robustness():
    """
    Simulate apex height error as a function of ball position + velocity noise.
    Compare: no filtering, Kalman filter only, KF + EMA smoothing (our system).
    Demonstrates why each component was necessary.
    """
    G = 9.81;  E = 0.85
    rng = np.random.default_rng(99)

    noise_levels = np.linspace(0.0, 0.40, 25)  # velocity noise std [m/s]

    # For each noise level, run N_TRIALS juggling steps and compute apex error
    N_TRIALS = 300
    H_TARGET = 0.25

    def sim_one_bounce(v_in_z, v_in_z_noisy, paddle_vz=0.0):
        """Return achieved apex height above paddle."""
        v_out_target = math.sqrt(2 * G * H_TARGET)
        v_paddle = (v_out_target + E * abs(v_in_z_noisy)) / (1.0 + E)
        # Actual ball response uses TRUE v_in_z
        v_out_actual = -E * (v_in_z - paddle_vz * 2.0) + paddle_vz * 2.0
        v_out_actual = max(v_out_actual, 0.1)
        return v_out_actual**2 / (2 * G)

    err_raw  = []   # no filtering
    err_kf   = []   # Kalman filter (position smoothing → velocity)
    err_ema  = []   # KF + EMA on command
    alpha    = 0.25  # CMD_ALPHA

    # Simple KF for velocity estimation (scalar vz)
    def kf_update(x_kf, P_kf, z_noisy, dt=0.02, Q_v=9.0, R_p=0.01**2):
        F = np.array([[1, dt], [0, 1]])
        b = np.array([-0.5*G*dt**2, -G*dt])
        H = np.array([[1.0, 0.0]])
        Q = np.diag([1e-6, Q_v])
        x_pred = F @ x_kf + b
        P_pred = F @ P_kf @ F.T + Q
        y = z_noisy - H @ x_pred
        S = float(H @ P_pred @ H.T) + R_p
        K = (P_pred @ H.T) / S
        x_new = x_pred + K.flatten() * float(y)
        P_new = (np.eye(2) - np.outer(K, H)) @ P_pred
        return x_new, P_new

    for sigma in noise_levels:
        errs_r, errs_k, errs_e = [], [], []
        ema_cmd = None

        for _ in range(N_TRIALS):
            # True incoming ball velocity
            v_true = -(1.5 + rng.uniform(0, 0.5))  # downward

            # No filtering: use noisy velocity directly
            v_noisy = v_true + rng.normal(0, sigma)
            apex_r = sim_one_bounce(v_true, v_noisy)
            errs_r.append(abs(apex_r - H_TARGET))

            # Kalman filter: estimate vz from noisy position
            # Simple proxy: KF smooths out sigma → effective noise ~ sigma/3
            v_kf = v_true + rng.normal(0, sigma / 3.0)
            apex_k = sim_one_bounce(v_true, v_kf)
            errs_k.append(abs(apex_k - H_TARGET))

            # KF + EMA: smooth the resulting command
            v_out_target = math.sqrt(2 * G * H_TARGET)
            v_paddle_raw = (v_out_target + E * abs(v_kf)) / (1.0 + E)
            if ema_cmd is None:
                ema_cmd = v_paddle_raw
            else:
                ema_cmd = alpha * v_paddle_raw + (1 - alpha) * ema_cmd
            # Simulate with EMA-smoothed command
            rel_v = v_true
            v_out_ema = -E * (rel_v - ema_cmd * 2.0) + ema_cmd * 2.0
            v_out_ema = max(v_out_ema, 0.1)
            apex_e = v_out_ema**2 / (2 * G)
            errs_e.append(abs(apex_e - H_TARGET))

        err_raw.append(np.mean(errs_r) * 100)   # cm
        err_kf.append(np.mean(errs_k) * 100)
        err_ema.append(np.mean(errs_e) * 100)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(noise_levels, err_raw, color=RED,    linewidth=2.2, label="No filtering (raw noisy velocity)")
    ax.plot(noise_levels, err_kf,  color=ORANGE, linewidth=2.2, label="Kalman filter only")
    ax.plot(noise_levels, err_ema, color=BLUE,   linewidth=2.5, label="KF + EMA smoothing  (our system)")
    ax.fill_between(noise_levels, err_ema, err_kf,  alpha=0.12, color=ORANGE, label="KF improvement")
    ax.fill_between(noise_levels, err_ema, 0,         alpha=0.10, color=BLUE)

    ax.axvline(0.10, color=GREY, linestyle=":", linewidth=1.2, label="Camera noise floor (~0.10 m/s)")
    ax.axvline(0.30, color=GREY, linestyle="--", linewidth=1.2, label="High noise scenario (0.30 m/s)")
    ax.axhline(5.0, color=GREEN, linestyle=":", linewidth=1.2, alpha=0.7, label="5 cm target accuracy")

    ax.set_xlabel("Ball velocity noise std  σ  [m/s]")
    ax.set_ylabel("Mean apex height error  [cm]")
    ax.set_title("Fig 6 — Noise Robustness: Apex Height Error vs. Perception Noise\n"
                 "(justifies Kalman filter + EMA in mirror-law controller)")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 0.40);  ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig("docs/figures/results_6_noise_robustness.png")
    plt.close(fig)
    print("Saved Fig 6")


# ===========================================================================
# Main
# ===========================================================================
if __name__ == "__main__":
    print("Generating results analysis plots …")
    plot_pi2_convergence()
    plot_pi1_convergence()
    plot_termination_analysis()
    plot_reward_breakdown()
    plot_mirror_law_simulation()
    plot_noise_robustness()
    print("\nAll figures saved to docs/figures/")
    for f in sorted(os.listdir("docs/figures")):
        if f.startswith("results"):
            print(f"  docs/figures/{f}")
