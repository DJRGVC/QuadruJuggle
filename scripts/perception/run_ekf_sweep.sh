#!/usr/bin/env bash
# GPU sweep: run extended bounce demo then analyze EKF vs raw by height.
# Usage: $C3R_BIN/gpu_lock.sh bash scripts/perception/run_ekf_sweep.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

STEP_TIMEOUT=${STEP_TIMEOUT:-300}
LOG="$REPO_ROOT/logs/perception/ekf_sweep_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"

echo "=== EKF vs Raw Sweep — $(date -Iseconds) ===" | tee "$LOG"

# Run extended bounce demo (500 steps = 10s sim, ~8-10 bounces for good height coverage)
echo "--- Step 1: Extended bounce demo (500 steps) ---" | tee -a "$LOG"
if timeout "$STEP_TIMEOUT" uv run --active python -u scripts/perception/demo_camera_ekf.py \
    --task Isaac-BallJuggleHier-Go1-Play-v0 --num_envs 1 --headless \
    --steps 500 --capture_interval 25 \
    2>&1 | tee -a "$LOG"; then
    echo "--- Step 1: SUCCESS ---" | tee -a "$LOG"
else
    rc=$?
    echo "--- Step 1: FAILED (exit $rc) ---" | tee -a "$LOG"
    exit 1
fi

# Run analysis (CPU only, no timeout needed)
echo "" | tee -a "$LOG"
echo "--- Step 2: Height-binned analysis ---" | tee -a "$LOG"
uv run --active python scripts/perception/analyze_ekf_vs_raw.py 2>&1 | tee -a "$LOG"

echo "" | tee -a "$LOG"
echo "=== Sweep Complete — $(date -Iseconds) ===" | tee -a "$LOG"

# Write sentinel
echo "$(date -Iseconds) log=$LOG" > "$REPO_ROOT/logs/perception/ekf_sweep_DONE"
echo "Sentinel: logs/perception/ekf_sweep_DONE"
