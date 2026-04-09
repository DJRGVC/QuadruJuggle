#!/usr/bin/env python3
"""Parse oracle eval log output from demo_camera_ekf.py.

Usage:
    python scripts/perception/parse_oracle_eval.py logs/perception/oracle_eval.log

Extracts: detection rate, RMSE, episode stats, per-step ball height, camera config.
Outputs a concise summary suitable for pasting into RESEARCH_LOG.md.
"""
import re
import sys
from pathlib import Path


def parse_log(path: str) -> dict:
    text = Path(path).read_text()
    result = {}

    # Detection rate
    m = re.search(r"Detection rate:\s*(\d+)/(\d+)\s*\((\d+\.\d+)%\)", text)
    if m:
        result["det_count"] = int(m.group(1))
        result["total_steps"] = int(m.group(2))
        result["det_rate_pct"] = float(m.group(3))

    # Detection RMSE
    m = re.search(r"Detection RMSE:\s*(\d+\.\d+)\s*\+/-\s*(\d+\.\d+)\s*m", text)
    if m:
        result["det_rmse_m"] = float(m.group(1))
        result["det_rmse_std_m"] = float(m.group(2))
        result["det_rmse_mm"] = result["det_rmse_m"] * 1000
        result["det_rmse_std_mm"] = result["det_rmse_std_m"] * 1000

    # EKF RMSE
    m = re.search(r"EKF RMSE:\s*(\d+\.\d+)\s*\+/-\s*(\d+\.\d+)\s*m", text)
    if m:
        result["ekf_rmse_m"] = float(m.group(1))
        result["ekf_rmse_std_m"] = float(m.group(2))
        result["ekf_rmse_mm"] = result["ekf_rmse_m"] * 1000
        result["ekf_rmse_std_mm"] = result["ekf_rmse_std_m"] * 1000

    # Episode stats
    m = re.search(r"Episodes:\s*(\d+)\s*\|\s*Timeout:\s*(\d+\.\d+)%", text)
    if m:
        result["episodes"] = int(m.group(1))
        result["timeout_pct"] = float(m.group(2))

    # Camera elevation
    m = re.search(r"elevation:\s*([\-\d.]+)°", text)
    if m:
        result["cam_elevation_deg"] = float(m.group(1))

    # Noise mode
    m = re.search(r"Ball obs noise mode:\s*(\w+)", text)
    if m:
        result["noise_mode"] = m.group(1)

    # Target height
    m = re.search(r"Fixed target apex height:\s*(\d+\.\d+)m", text)
    if m:
        result["target_height_m"] = float(m.group(1))

    # Per-step logs: extract ball heights and detection rates at milestones
    step_logs = re.findall(
        r"\[demo\] Step (\d+): det_rate=(\d+)%, ball_h=([\-\d.]+)m, "
        r"det_rmse=(\d+\.\d+)m, ekf_rmse=(\d+\.\d+)m"
        r"(?: episodes=(\d+) TO=(\d+)%)?",
        text,
    )
    if step_logs:
        result["step_logs"] = []
        for sl in step_logs:
            entry = {
                "step": int(sl[0]),
                "det_rate_pct": int(sl[1]),
                "ball_h_m": float(sl[2]),
                "det_rmse_m": float(sl[3]),
                "ekf_rmse_m": float(sl[4]),
            }
            if sl[5]:
                entry["episodes"] = int(sl[5])
                entry["timeout_pct"] = int(sl[6])
            result["step_logs"].append(entry)

    # Trajectory NPZ path
    m = re.search(r"Trajectory data saved: (.+\.npz)", text)
    if m:
        result["trajectory_npz"] = m.group(1)

    return result


def print_summary(r: dict) -> None:
    print("=== Oracle Eval Summary ===")
    if "noise_mode" in r:
        print(f"  Noise mode: {r['noise_mode']}")
    if "target_height_m" in r:
        print(f"  Target height: {r['target_height_m']:.2f}m")
    if "cam_elevation_deg" in r:
        print(f"  Camera elevation: {r['cam_elevation_deg']:.1f}°")
    print()

    if "det_rate_pct" in r:
        print(f"  Detection rate: {r['det_count']}/{r['total_steps']} ({r['det_rate_pct']:.1f}%)")
    if "det_rmse_mm" in r:
        print(f"  Detection RMSE: {r['det_rmse_mm']:.1f} ± {r['det_rmse_std_mm']:.1f} mm")
    if "ekf_rmse_mm" in r:
        print(f"  EKF RMSE: {r['ekf_rmse_mm']:.1f} ± {r['ekf_rmse_std_mm']:.1f} mm")
    if "episodes" in r:
        print(f"  Episodes: {r['episodes']}, Timeout: {r['timeout_pct']:.1f}%")
    print()

    # Ball height distribution from step logs
    if "step_logs" in r:
        heights = [s["ball_h_m"] for s in r["step_logs"]]
        above_02 = sum(1 for h in heights if h > 0.20)
        print(f"  Step samples with ball >0.2m above paddle: {above_02}/{len(heights)}")
        if heights:
            print(f"  Ball height range: {min(heights):.3f}m to {max(heights):.3f}m")

    if "trajectory_npz" in r:
        print(f"  Trajectory: {r['trajectory_npz']}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <log_file>")
        sys.exit(1)
    results = parse_log(sys.argv[1])
    if not results:
        print("ERROR: Could not parse any results from log file.")
        print("(File may be empty or GPU run hasn't completed yet.)")
        sys.exit(1)
    print_summary(results)
