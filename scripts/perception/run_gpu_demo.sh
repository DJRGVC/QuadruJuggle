#!/usr/bin/env bash
# Combined GPU runner: debug capture (smoke test) + full demo.
# Run via: $C3R_BIN/gpu_lock.sh bash scripts/perception/run_gpu_demo.sh
# Captures output to a log file and runs both scripts in sequence.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

LOG="$REPO_ROOT/logs/perception/gpu_demo_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG")"

echo "=== GPU Demo Run — $(date -Iseconds) ===" | tee "$LOG"

# Step 1: Debug capture (smoke test — validates camera sees ball)
echo "" | tee -a "$LOG"
echo "--- Step 1: debug_d435i_capture.py ---" | tee -a "$LOG"
if uv run --active python scripts/perception/debug_d435i_capture.py \
    --task Isaac-BallJuggleHier-Go1-Play-v0 --num_envs 1 --headless --steps 50 \
    2>&1 | tee -a "$LOG"; then
    echo "--- Step 1: SUCCESS ---" | tee -a "$LOG"
else
    echo "--- Step 1: FAILED (exit $?) ---" | tee -a "$LOG"
    echo "Aborting — debug capture failed." | tee -a "$LOG"
    exit 1
fi

# Step 2: Full demo (camera → detect → EKF)
echo "" | tee -a "$LOG"
echo "--- Step 2: demo_camera_ekf.py ---" | tee -a "$LOG"
if uv run --active python scripts/perception/demo_camera_ekf.py \
    --task Isaac-BallJuggleHier-Go1-Play-v0 --num_envs 1 --headless --steps 300 \
    --capture_interval 5 \
    2>&1 | tee -a "$LOG"; then
    echo "--- Step 2: SUCCESS ---" | tee -a "$LOG"
else
    echo "--- Step 2: FAILED (exit $?) ---" | tee -a "$LOG"
    exit 1
fi

echo "" | tee -a "$LOG"
echo "=== GPU Demo Complete — $(date -Iseconds) ===" | tee -a "$LOG"
echo "Log: $LOG"
