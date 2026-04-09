#!/usr/bin/env python3
"""Perception gap decomposition — break down EKF observation error by ball phase.

Takes oracle and d435i eval directories and produces a figure showing WHERE
the observation error lives: contact vs ascending vs descending vs near-apex.
This helps diagnose whether the perception gap (d435i vs oracle timeout %)
is caused by:
  - Position noise during critical moments (launch decision, apex detection)
  - Dropout-induced stale measurements during flight
  - EKF lag during phase transitions
  - Or is simply a policy robustness issue

Usage:
    python scripts/perception/analyze_perception_gap.py \
        --oracle-dir logs/perception/eval_stage_g_final_oracle \
        --d435i-dir logs/perception/eval_stage_g_final_d435i \
        --out images/perception/perception_gap_decomposition.png

    # JSON only (no figure):
    python scripts/perception/analyze_perception_gap.py \
        --oracle-dir logs/perception/eval_stage_g_final_oracle \
        --d435i-dir logs/perception/eval_stage_g_final_d435i \
        --json-out logs/perception/gap_decomposition.json
"""

import argparse
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from analyze_eval_trajectory import compute_phase_metrics, load_npz

# Phase names for display
_PHASE_NAMES = ["contact", "ascending", "descending"]
_PHASE_COLORS = {"contact": "#4477AA", "ascending": "#EE6677", "descending": "#228833"}


def discover_targets(eval_dir: str) -> list[tuple[float, str]]:
    """Find target_X_XX subdirs and return sorted (target_height, npz_path)."""
    targets = []
    if not os.path.isdir(eval_dir):
        raise FileNotFoundError(f"Eval directory not found: {eval_dir}")
    for entry in os.listdir(eval_dir):
        m = re.match(r"target_(\d+)_(\d+)", entry)
        if m:
            npz = os.path.join(eval_dir, entry, "trajectory.npz")
            if os.path.isfile(npz):
                h = float(f"{m.group(1)}.{m.group(2)}")
                targets.append((h, npz))
    targets.sort(key=lambda x: x[0])
    return targets


def compute_obs_staleness(traj: dict) -> dict:
    """Compute observation staleness metrics — how many steps since last measurement."""
    det_steps = traj["det_steps"].astype(int)
    T = traj["gt"].shape[0]
    phase = traj.get("phase")
    anchored = traj.get("anchored_step")

    # Steps since last measurement (detection or anchor)
    last_meas = -1
    staleness = np.zeros(T, dtype=int)
    for t in range(T):
        if t in det_steps:
            last_meas = t
        elif anchored is not None and anchored[t]:
            last_meas = t
        staleness[t] = t - last_meas if last_meas >= 0 else t

    result = {"mean_staleness": float(np.mean(staleness))}

    # Staleness by phase
    if phase is not None:
        for pval, pname in enumerate(_PHASE_NAMES):
            mask = phase == pval
            if mask.sum() > 0:
                result[f"staleness_{pname}"] = float(np.mean(staleness[mask]))
                result[f"max_staleness_{pname}"] = int(np.max(staleness[mask]))
            else:
                result[f"staleness_{pname}"] = float("nan")
                result[f"max_staleness_{pname}"] = 0
    return result


def compute_velocity_error(traj: dict, paddle_z: float = 0.47) -> dict:
    """Estimate EKF velocity error via finite differences of position error."""
    gt = traj["gt"]
    ekf = traj["ekf"]
    dt = float(traj["dt"]) if "dt" in traj else 0.02
    T = gt.shape[0]

    if T < 3:
        return {"vz_rmse_mps": float("nan")}

    # Ground truth velocity (central differences)
    gt_vz = np.zeros(T)
    gt_vz[1:-1] = (gt[2:, 2] - gt[:-2, 2]) / (2 * dt)
    gt_vz[0] = (gt[1, 2] - gt[0, 2]) / dt
    gt_vz[-1] = (gt[-1, 2] - gt[-2, 2]) / dt

    # EKF velocity (central differences of EKF position estimate)
    ekf_vz = np.zeros(T)
    ekf_vz[1:-1] = (ekf[2:, 2] - ekf[:-2, 2]) / (2 * dt)
    ekf_vz[0] = (ekf[1, 2] - ekf[0, 2]) / dt
    ekf_vz[-1] = (ekf[-1, 2] - ekf[-2, 2]) / dt

    vz_err = ekf_vz - gt_vz
    result = {"vz_rmse_mps": float(np.sqrt(np.mean(vz_err ** 2)))}

    # By phase
    phase = traj.get("phase")
    gt_h = gt[:, 2] - paddle_z
    if phase is not None:
        for pval, pname in enumerate(_PHASE_NAMES):
            mask = phase == pval
            if mask.sum() > 2:
                result[f"vz_rmse_{pname}"] = float(np.sqrt(np.mean(vz_err[mask] ** 2)))
            else:
                result[f"vz_rmse_{pname}"] = float("nan")

    # Near-apex accuracy (top 10% of ball height)
    max_h = gt_h.max()
    if max_h > 0.02:  # only if ball actually leaves paddle
        apex_mask = gt_h > 0.9 * max_h
        if apex_mask.sum() > 0:
            apex_pos_err = np.linalg.norm(ekf[apex_mask] - gt[apex_mask], axis=1)
            result["apex_pos_rmse_mm"] = float(np.sqrt(np.mean(apex_pos_err ** 2)) * 1000)
            if apex_mask.sum() > 2:
                result["apex_vz_rmse"] = float(np.sqrt(np.mean(vz_err[apex_mask] ** 2)))
            result["apex_steps"] = int(apex_mask.sum())
        else:
            result["apex_pos_rmse_mm"] = float("nan")
    else:
        result["apex_pos_rmse_mm"] = float("nan")

    return result


def decompose_gap(oracle_dir: str, d435i_dir: str, paddle_z: float = 0.47) -> dict:
    """Full gap decomposition across all matching targets."""
    oracle_targets = discover_targets(oracle_dir)
    d435i_targets = discover_targets(d435i_dir)

    # Match targets
    oracle_dict = {h: p for h, p in oracle_targets}
    d435i_dict = {h: p for h, p in d435i_targets}
    common = sorted(set(oracle_dict.keys()) & set(d435i_dict.keys()))

    if not common:
        raise ValueError(f"No matching targets. Oracle: {list(oracle_dict)}, D435i: {list(d435i_dict)}")

    results = {}
    for h in common:
        o_traj = load_npz(oracle_dict[h])
        d_traj = load_npz(d435i_dict[h])

        o_phase = compute_phase_metrics(o_traj, paddle_z)
        d_phase = compute_phase_metrics(d_traj, paddle_z)

        o_stale = compute_obs_staleness(o_traj)
        d_stale = compute_obs_staleness(d_traj)

        o_vel = compute_velocity_error(o_traj, paddle_z)
        d_vel = compute_velocity_error(d_traj, paddle_z)

        # Ball dynamics summary
        o_gt = o_traj["gt"]
        d_gt = d_traj["gt"]
        o_ball_h = o_traj.get("ball_h", o_gt[:, 2] - paddle_z)
        d_ball_h = d_traj.get("ball_h", d_gt[:, 2] - paddle_z)

        results[str(h)] = {
            "target_m": h,
            "oracle": {
                "phase_metrics": o_phase,
                "staleness": o_stale,
                "velocity": o_vel,
                "max_ball_h_mm": float(np.max(o_ball_h) * 1000),
                "flight_frac_pct": float(
                    ((o_traj.get("phase", np.zeros(len(o_gt))) > 0).sum() / len(o_gt)) * 100
                ),
                "n_detections": int(o_traj["det"].shape[0]),
            },
            "d435i": {
                "phase_metrics": d_phase,
                "staleness": d_stale,
                "velocity": d_vel,
                "max_ball_h_mm": float(np.max(d_ball_h) * 1000),
                "flight_frac_pct": float(
                    ((d_traj.get("phase", np.zeros(len(d_gt))) > 0).sum() / len(d_gt)) * 100
                ),
                "n_detections": int(d_traj["det"].shape[0]),
            },
        }

    return results


def plot_gap_decomposition(results: dict, out_path: str) -> None:
    """Generate publication-quality figure showing perception gap breakdown."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    targets = sorted(results.keys(), key=lambda x: float(x))
    n_targets = len(targets)

    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.30)

    target_vals = [float(t) for t in targets]
    x = np.arange(n_targets)
    bar_w = 0.35

    # ── Panel 1: Position RMSE by phase (d435i only) ──
    ax1 = fig.add_subplot(gs[0, 0])
    for i, pname in enumerate(_PHASE_NAMES):
        vals = []
        for t in targets:
            v = results[t]["d435i"]["phase_metrics"].get(pname, {}).get("ekf_rmse_mm", float("nan"))
            vals.append(v)
        ax1.bar(x + i * bar_w / len(_PHASE_NAMES), vals, bar_w / len(_PHASE_NAMES),
                label=pname, color=_PHASE_COLORS[pname], alpha=0.8)
    ax1.set_xlabel("Target height (m)")
    ax1.set_ylabel("EKF position RMSE (mm)")
    ax1.set_title("(a) D435i position error by phase")
    ax1.set_xticks(x + bar_w / 3)
    ax1.set_xticklabels([f"{v:.2f}" for v in target_vals])
    ax1.legend(fontsize=8)
    ax1.set_ylim(bottom=0)

    # ── Panel 2: Velocity error by phase ──
    ax2 = fig.add_subplot(gs[0, 1])
    for mode, offset, color, alpha in [("oracle", -bar_w / 2, "#999999", 0.5),
                                        ("d435i", bar_w / 2, "#CC6677", 0.8)]:
        vals = [results[t][mode]["velocity"].get("vz_rmse_mps", float("nan")) for t in targets]
        ax2.bar(x + offset / 2, vals, bar_w, label=mode.upper(), color=color, alpha=alpha)
    ax2.set_xlabel("Target height (m)")
    ax2.set_ylabel("Vz RMSE (m/s)")
    ax2.set_title("(b) Vertical velocity error")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{v:.2f}" for v in target_vals])
    ax2.legend(fontsize=8)
    ax2.set_ylim(bottom=0)

    # ── Panel 3: Observation staleness by phase ──
    ax3 = fig.add_subplot(gs[0, 2])
    for pname in _PHASE_NAMES:
        vals = [results[t]["d435i"]["staleness"].get(f"staleness_{pname}", float("nan"))
                for t in targets]
        ax3.plot(target_vals, vals, "o-", label=pname, color=_PHASE_COLORS[pname])
    ax3.set_xlabel("Target height (m)")
    ax3.set_ylabel("Mean steps since measurement")
    ax3.set_title("(c) D435i observation staleness")
    ax3.legend(fontsize=8)
    ax3.set_ylim(bottom=0)

    # ── Panel 4: Detection rate by phase ──
    ax4 = fig.add_subplot(gs[1, 0])
    for pname in _PHASE_NAMES:
        o_vals = [results[t]["oracle"]["phase_metrics"].get(pname, {}).get("det_rate_pct", 0)
                  for t in targets]
        d_vals = [results[t]["d435i"]["phase_metrics"].get(pname, {}).get("det_rate_pct", 0)
                  for t in targets]
        ax4.plot(target_vals, d_vals, "o-", label=f"d435i {pname}",
                 color=_PHASE_COLORS[pname])
        ax4.plot(target_vals, o_vals, "x--", label=f"oracle {pname}",
                 color=_PHASE_COLORS[pname], alpha=0.4)
    ax4.set_xlabel("Target height (m)")
    ax4.set_ylabel("Detection rate (%)")
    ax4.set_title("(d) Detection rate by phase")
    ax4.legend(fontsize=7, ncol=2)
    ax4.set_ylim(-5, 105)

    # ── Panel 5: Flight fraction & max ball height ──
    ax5 = fig.add_subplot(gs[1, 1])
    ax5_twin = ax5.twinx()
    o_flight = [results[t]["oracle"]["flight_frac_pct"] for t in targets]
    d_flight = [results[t]["d435i"]["flight_frac_pct"] for t in targets]
    o_maxh = [results[t]["oracle"]["max_ball_h_mm"] for t in targets]
    d_maxh = [results[t]["d435i"]["max_ball_h_mm"] for t in targets]

    ax5.bar(x - bar_w / 2, o_flight, bar_w, label="Oracle flight%", color="#999999", alpha=0.5)
    ax5.bar(x + bar_w / 2, d_flight, bar_w, label="D435i flight%", color="#CC6677", alpha=0.8)
    ax5_twin.plot(target_vals, o_maxh, "s--", color="#333333", label="Oracle max h", alpha=0.6)
    ax5_twin.plot(target_vals, d_maxh, "D-", color="#882255", label="D435i max h")

    ax5.set_xlabel("Target height (m)")
    ax5.set_ylabel("Flight fraction (%)")
    ax5_twin.set_ylabel("Max ball height (mm)")
    ax5.set_title("(e) Ball dynamics")
    ax5.set_xticks(x)
    ax5.set_xticklabels([f"{v:.2f}" for v in target_vals])
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="upper left")

    # ── Panel 6: Summary table ──
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    headers = ["Target", "D435i\nPos RMSE", "D435i\nVz RMSE", "Det\nRate", "Flight\n%", "Stale\n(flight)"]
    table_data = []
    for t in targets:
        r = results[t]
        # Overall d435i EKF RMSE
        d_phases = r["d435i"]["phase_metrics"]
        # Weighted average RMSE across phases
        total_steps = sum(d_phases.get(p, {}).get("count", 0) for p in _PHASE_NAMES)
        if total_steps > 0:
            weighted_rmse = sum(
                d_phases.get(p, {}).get("ekf_rmse_mm", 0) ** 2 * d_phases.get(p, {}).get("count", 0)
                for p in _PHASE_NAMES
            ) / total_steps
            overall_rmse = f"{np.sqrt(weighted_rmse):.0f}mm"
        else:
            overall_rmse = "N/A"

        vz = r["d435i"]["velocity"].get("vz_rmse_mps", float("nan"))
        vz_str = f"{vz:.3f}" if not np.isnan(vz) else "N/A"

        det_n = r["d435i"]["n_detections"]
        flight = r["d435i"]["flight_frac_pct"]

        # Flight staleness
        stale_asc = r["d435i"]["staleness"].get("staleness_ascending", float("nan"))
        stale_desc = r["d435i"]["staleness"].get("staleness_descending", float("nan"))
        if not np.isnan(stale_asc) and not np.isnan(stale_desc):
            flight_stale = f"{(stale_asc + stale_desc) / 2:.0f}"
        else:
            flight_stale = "N/A"

        table_data.append([f"{float(t):.2f}m", overall_rmse, vz_str, str(det_n), f"{flight:.1f}%", flight_stale])

    table = ax6.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    ax6.set_title("(f) D435i pipeline summary", pad=20)

    fig.suptitle("Perception Gap Decomposition — D435i vs Oracle", fontsize=14, fontweight="bold", y=0.98)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Perception gap decomposition analysis")
    parser.add_argument("--oracle-dir", required=True, help="Oracle eval directory")
    parser.add_argument("--d435i-dir", required=True, help="D435i eval directory")
    parser.add_argument("--paddle-z", type=float, default=0.47, help="Paddle Z in world frame")
    parser.add_argument("--out", help="Output figure path")
    parser.add_argument("--json-out", help="Output JSON path")
    args = parser.parse_args()

    results = decompose_gap(args.oracle_dir, args.d435i_dir, args.paddle_z)

    # Print summary
    print("\n=== Perception Gap Decomposition ===\n")
    for t in sorted(results.keys(), key=float):
        r = results[t]
        print(f"Target {t}m:")
        for mode in ["oracle", "d435i"]:
            m = r[mode]
            print(f"  {mode.upper()}:")
            print(f"    Flight: {m['flight_frac_pct']:.1f}%, max_h: {m['max_ball_h_mm']:.0f}mm, "
                  f"detections: {m['n_detections']}")
            print(f"    Vz RMSE: {m['velocity'].get('vz_rmse_mps', float('nan')):.4f} m/s")
            for pname in _PHASE_NAMES:
                pm = m["phase_metrics"].get(pname, {})
                sm = m["staleness"]
                cnt = pm.get("count", 0)
                rmse = pm.get("ekf_rmse_mm", float("nan"))
                det_r = pm.get("det_rate_pct", float("nan"))
                stale = sm.get(f"staleness_{pname}", float("nan"))
                print(f"    {pname:12s}: {cnt:5d} steps, RMSE {rmse:7.1f}mm, "
                      f"det {det_r:5.1f}%, stale {stale:.1f}")
        print()

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved JSON: {args.json_out}")

    if args.out:
        plot_gap_decomposition(results, args.out)


if __name__ == "__main__":
    main()
