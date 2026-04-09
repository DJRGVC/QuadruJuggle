#!/bin/bash
# Run camera pipeline eval with d435i-trained pi1 checkpoint.
# Uses d435i noise mode so we see the perception pipeline under
# the noise conditions it was trained with.
set -euo pipefail

D435I_PI1="/home/daniel-grant/Research/QuadruJuggle-policy/logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_22-51-56/model_best.pt"
PI2="/home/daniel-grant/Research/QuadruJuggle/logs/rsl_rl/go1_torso_tracking/2026-03-12_17-16-01/model_best.pt"
STEPS=1500
NUM_ENVS=4
TARGET=0.42

OUT_DIR="logs/perception/d435i_eval"
LOG_FILE="logs/perception/d435i_eval_$(date +%Y%m%d_%H%M%S).log"

echo "=== D435i pi1 eval: ${STEPS} steps, ${NUM_ENVS} envs, target=${TARGET}m ==="
echo "=== Log: ${LOG_FILE}, Out: ${OUT_DIR} ==="
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
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=== Done ==="
echo "LOG=$LOG_FILE" > logs/perception/d435i_eval_DONE
echo "OUT_DIR=$OUT_DIR" >> logs/perception/d435i_eval_DONE
touch logs/perception/d435i_eval_DONE
