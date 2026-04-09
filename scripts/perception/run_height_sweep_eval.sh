#!/bin/bash
# Sweep target apex heights to measure detection rate vs target height.
# Usage: $C3R_BIN/gpu_lock.sh bash scripts/perception/run_height_sweep_eval.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

D435I_PI1="/home/daniel-grant/Research/QuadruJuggle-policy/logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_22-51-56/model_best.pt"
PI2="/home/daniel-grant/Research/QuadruJuggle/logs/rsl_rl/go1_torso_tracking/2026-03-12_17-16-01/model_best.pt"
STEPS=1500
NUM_ENVS=4
TARGETS="0.42 0.50 0.70 1.00"

SWEEP_DIR="logs/perception/height_sweep"
mkdir -p "$SWEEP_DIR"

echo "============================================"
echo "=== Height sweep eval (d435i policy)     ==="
echo "============================================"
echo "Targets: ${TARGETS}"
echo "Steps: ${STEPS}, Envs: ${NUM_ENVS}"
echo ""

for TARGET in $TARGETS; do
    TAG=$(echo "$TARGET" | tr '.' '_')
    OUT_DIR="${SWEEP_DIR}/target_${TAG}"
    echo "=== Target ${TARGET}m → ${OUT_DIR} ==="
    mkdir -p "$OUT_DIR"
    uv run --active python -u scripts/perception/demo_camera_ekf.py \
        --task Isaac-BallJuggleHier-Go1-Play-v0 \
        --num_envs "$NUM_ENVS" \
        --steps "$STEPS" \
        --headless \
        --pi1-checkpoint "$D435I_PI1" \
        --pi2-checkpoint "$PI2" \
        --target-height "$TARGET" \
        --noise-mode d435i \
        --out-dir "$OUT_DIR" \
        2>&1 | tail -n 30
    echo ""
    echo "=== Target ${TARGET}m done ==="
    echo ""
done

# Summary: extract detection rate from each run
echo "============================================"
echo "=== SUMMARY: Detection rate by target    ==="
echo "============================================"
for TARGET in $TARGETS; do
    TAG=$(echo "$TARGET" | tr '.' '_')
    NPZ="${SWEEP_DIR}/target_${TAG}/trajectory.npz"
    if [ -f "$NPZ" ]; then
        uv run --active python -c "
import numpy as np
d = np.load('$NPZ')
total = len(d['steps'])
num_det = len(d['det_steps'])
rate = num_det / max(total, 1) * 100
# Ball height above paddle: gt[:,2] (z component)
gt_z = d['gt'][:, 2]
mean_h = gt_z.mean()
max_h = gt_z.max()
print(f'Target ${TARGET}m: det_rate={rate:.1f}% ({num_det}/{total}), mean_h={mean_h:.3f}m, max_h={max_h:.3f}m')
" 2>&1
    else
        echo "Target ${TARGET}m: no trajectory.npz found"
    fi
done

echo ""
echo "=== Height sweep complete ==="
