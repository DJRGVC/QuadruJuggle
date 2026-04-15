"""Visualize joint command comparison: MuJoCo vs Isaac Lab.

Reads CSVs produced by test_joint_cmd_mujoco.py and test_joint_cmd_isaaclab.py
and plots target, actual, error, and torque for each joint side-by-side.

Usage
-----
# Compare FR_hip (MuJoCo idx 0, Isaac Lab idx 3):
conda run -n isaaclab python scripts/tests/plot_joint_comparison.py \\
    --mujoco   tests_out/mujoco_joint0.csv \\
    --isaaclab tests_out/isaaclab_joint3.csv \\
    --joint    FR_hip

# Show all joints from one run (no Isaac Lab):
conda run -n isaaclab python scripts/tests/plot_joint_comparison.py \\
    --mujoco tests_out/mujoco_joint0.csv

# All 12 joints grid (both sims):
conda run -n isaaclab python scripts/tests/plot_joint_comparison.py \\
    --mujoco   tests_out/mujoco_joint0.csv \\
    --isaaclab tests_out/isaaclab_joint3.csv \\
    --all_joints
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# ── Joint name cross-reference ────────────────────────────────────────────────
# Maps MuJoCo short name → Isaac Lab short name (strip _joint suffix)
MJCF_TO_ISAAC = {
    "FR_hip": "FR_hip_joint",  "FR_thigh": "FR_thigh_joint",  "FR_calf": "FR_calf_joint",
    "FL_hip": "FL_hip_joint",  "FL_thigh": "FL_thigh_joint",  "FL_calf": "FL_calf_joint",
    "RR_hip": "RR_hip_joint",  "RR_thigh": "RR_thigh_joint",  "RR_calf": "RR_calf_joint",
    "RL_hip": "RL_hip_joint",  "RL_thigh": "RL_thigh_joint",  "RL_calf": "RL_calf_joint",
}

# Ordered list of all 12 joints (MJCF short names, top-to-bottom in plot)
ALL_JOINTS_MJCF = [
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
]


def load_csv(path: str) -> pd.DataFrame:
    """Read CSV, skipping Isaac Lab log lines that bleed into stdout."""
    import io
    header = None
    rows = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip lines that look like Isaac Lab log output (start with [, whitespace, etc.)
            if line.startswith("[") or line.startswith("\x1b"):
                continue
            parts = line.split(",")
            if header is None:
                # First clean line must be the header (starts with 'step' or a digit)
                if parts[0].strip() == "step" or parts[0].strip().lstrip("-").isdigit():
                    if parts[0].strip() == "step":
                        header = [p.strip() for p in parts]
                    else:
                        # No header row — use default column names
                        header = ["step", "sim", "joint_name", "joint_idx",
                                  "target_rad", "actual_rad", "error_rad", "torque_Nm"]
                        rows.append(parts)
            else:
                if len(parts) == len(header):
                    rows.append(parts)
    df = pd.DataFrame(rows, columns=header)
    # Cast numeric columns
    for col in ["step", "target_rad", "actual_rad", "error_rad", "torque_Nm"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def plot_one_joint(ax_top, ax_err, ax_torq,
                   mj_df: pd.DataFrame | None, il_df: pd.DataFrame | None,
                   mjcf_name: str, hold_steps: int | None = None):
    """Plot position tracking + error + torque for one joint."""
    il_name = MJCF_TO_ISAAC[mjcf_name]

    has_mj = mj_df is not None and mjcf_name in mj_df["joint_name"].values
    has_il = il_df is not None and il_name in il_df["joint_name"].values

    # ── Position tracking ────────────────────────────────────────────────────
    if has_mj:
        mj = mj_df[mj_df["joint_name"] == mjcf_name].sort_values("step")
        ax_top.plot(mj["step"], mj["target_rad"],
                    color="black", lw=1.2, ls="--", label="target")
        ax_top.plot(mj["step"], mj["actual_rad"],
                    color="#1f77b4", lw=1.2, label="MuJoCo actual")

    if has_il:
        il = il_df[il_df["joint_name"] == il_name].sort_values("step")
        if not has_mj:
            ax_top.plot(il["step"], il["target_rad"],
                        color="black", lw=1.2, ls="--", label="target")
        ax_top.plot(il["step"], il["actual_rad"],
                    color="#d62728", lw=1.2, label="Isaac Lab actual")

    ax_top.set_ylabel("angle [rad]", fontsize=7)
    ax_top.legend(fontsize=6, loc="lower right")
    ax_top.grid(True, alpha=0.3)
    ax_top.set_title(mjcf_name, fontsize=8, fontweight="bold")

    # ── Error ────────────────────────────────────────────────────────────────
    if has_mj:
        mj = mj_df[mj_df["joint_name"] == mjcf_name].sort_values("step")
        ax_err.plot(mj["step"], mj["error_rad"],
                    color="#1f77b4", lw=1.0, label="MuJoCo err")

    if has_il:
        il = il_df[il_df["joint_name"] == il_name].sort_values("step")
        ax_err.plot(il["step"], il["error_rad"],
                    color="#d62728", lw=1.0, label="Isaac err")

    ax_err.axhline(0, color="gray", lw=0.6, ls=":")
    ax_err.set_ylabel("error [rad]", fontsize=7)
    ax_err.legend(fontsize=6, loc="lower right")
    ax_err.grid(True, alpha=0.3)

    # ── Torque ───────────────────────────────────────────────────────────────
    if has_mj:
        mj = mj_df[mj_df["joint_name"] == mjcf_name].sort_values("step")
        ax_torq.plot(mj["step"], mj["torque_Nm"],
                     color="#1f77b4", lw=1.0, label="MuJoCo τ")

    if has_il:
        il = il_df[il_df["joint_name"] == il_name].sort_values("step")
        ax_torq.plot(il["step"], il["torque_Nm"],
                     color="#d62728", lw=1.0, label="Isaac τ")

    ax_torq.axhline(0, color="gray", lw=0.6, ls=":")
    ax_torq.set_ylabel("torque [N·m]", fontsize=7)
    ax_torq.set_xlabel("step", fontsize=7)
    ax_torq.legend(fontsize=6, loc="lower right")
    ax_torq.grid(True, alpha=0.3)

    # Mark step boundary
    if hold_steps is not None:
        for ax in [ax_top, ax_err, ax_torq]:
            ax.axvline(hold_steps, color="orange", lw=0.8, ls="--", alpha=0.7)


def infer_hold_steps(df: pd.DataFrame, joint_col: str) -> int | None:
    """Detect step at which target changes."""
    first_joint = df[joint_col].iloc[0]
    sub = df[df[joint_col] == first_joint].sort_values("step")
    if "target_rad" not in sub.columns:
        return None
    tgt = sub["target_rad"].values
    for i in range(1, len(tgt)):
        if abs(tgt[i] - tgt[0]) > 1e-4:
            return int(sub["step"].iloc[i])
    return None


def main():
    parser = argparse.ArgumentParser(description="Visualize joint command comparison")
    parser.add_argument("--mujoco",   type=str, default=None,
                        help="MuJoCo CSV from test_joint_cmd_mujoco.py")
    parser.add_argument("--isaaclab", type=str, default=None,
                        help="Isaac Lab CSV from test_joint_cmd_isaaclab.py")
    parser.add_argument("--joint",    type=str, default=None,
                        help="Single MJCF joint name to plot (e.g. FR_hip). "
                             "Default: auto-detect active joint.")
    parser.add_argument("--all_joints", action="store_true",
                        help="Plot all 12 joints in a grid (ignores --joint)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output PNG path. Default: tests_out/joint_comparison.png")
    parser.add_argument("--show", action="store_true",
                        help="Open interactive window instead of saving")
    args = parser.parse_args()

    if args.mujoco is None and args.isaaclab is None:
        parser.error("Provide at least one of --mujoco or --isaaclab")

    mj_df = load_csv(args.mujoco)   if args.mujoco   else None
    il_df = load_csv(args.isaaclab) if args.isaaclab else None

    # ── Detect hold boundary ─────────────────────────────────────────────────
    hold_steps = None
    if mj_df is not None:
        hold_steps = infer_hold_steps(mj_df, "joint_name")
    elif il_df is not None:
        hold_steps = infer_hold_steps(il_df, "joint_name")

    # ── Detect active joint (the one with a step input) ───────────────────────
    def find_active_joint_mj(df):
        for name in df["joint_name"].unique():
            sub = df[df["joint_name"] == name]
            if sub["target_rad"].std() > 1e-4:
                return name
        return df["joint_name"].iloc[0]

    def find_active_joint_il(df):
        for name in df["joint_name"].unique():
            sub = df[df["joint_name"] == name]
            if sub["target_rad"].std() > 1e-4:
                # Convert to MJCF name
                for mj, il in MJCF_TO_ISAAC.items():
                    if il == name:
                        return mj
        return None

    if args.all_joints:
        joints_to_plot = ALL_JOINTS_MJCF
    elif args.joint:
        joints_to_plot = [args.joint]
    else:
        # Auto-detect from whichever CSV is available
        active = None
        if mj_df is not None:
            active = find_active_joint_mj(mj_df)
        elif il_df is not None:
            active = find_active_joint_il(il_df)
        joints_to_plot = [active] if active else [ALL_JOINTS_MJCF[0]]
        print(f"[plot] Auto-detected active joint: {joints_to_plot[0]}")

    # ── Build figure ─────────────────────────────────────────────────────────
    n = len(joints_to_plot)
    # 3 subplots per joint (position, error, torque) arranged in columns
    cols = min(n, 4)
    rows_per_col = (n + cols - 1) // cols
    fig_h = max(8, rows_per_col * 5)
    fig_w = cols * 5

    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = gridspec.GridSpec(rows_per_col, cols, figure=fig,
                              hspace=0.55, wspace=0.40)

    for ji, jname in enumerate(joints_to_plot):
        row = ji // cols
        col = ji %  cols
        inner = gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer[row, col], hspace=0.08, height_ratios=[3, 2, 2]
        )
        ax_top  = fig.add_subplot(inner[0])
        ax_err  = fig.add_subplot(inner[1], sharex=ax_top)
        ax_torq = fig.add_subplot(inner[2], sharex=ax_top)
        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_err.get_xticklabels(), visible=False)

        plot_one_joint(ax_top, ax_err, ax_torq, mj_df, il_df, jname, hold_steps)

    # ── Legend sims ───────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="black",   lw=1.2, ls="--", label="target"),
        Line2D([0], [0], color="#1f77b4", lw=1.2, label="MuJoCo"),
        Line2D([0], [0], color="#d62728", lw=1.2, label="Isaac Lab"),
        Line2D([0], [0], color="orange",  lw=0.8, ls="--", label="step input"),
    ]
    fig.legend(handles=legend_handles, loc="upper center",
               ncol=4, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 1.01))

    title_parts = []
    if mj_df is not None:   title_parts.append(f"MuJoCo: {os.path.basename(args.mujoco)}")
    if il_df is not None:   title_parts.append(f"Isaac: {os.path.basename(args.isaaclab)}")
    fig.suptitle("Joint Command Comparison — " + " | ".join(title_parts),
                 fontsize=10, y=1.03)

    # ── Save or show ─────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "../../tests_out")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out or os.path.join(out_dir, "joint_comparison.png")
    out_path = os.path.abspath(out_path)

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()
    else:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] Saved → {out_path}")


if __name__ == "__main__":
    main()
