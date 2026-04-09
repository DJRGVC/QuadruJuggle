#!/bin/bash
# Run camera pipeline eval with oracle-trained pi1 checkpoint.
# This model achieves 100% timeout (stable juggling), so the ball
# is frequently in-flight and visible to the D435i camera.
set -euo pipefail

ORACLE_PI1="/home/daniel-grant/Research/QuadruJuggle-policy/logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_19-19-41/model_best.pt"
PI2="/home/daniel-grant/Research/QuadruJuggle/logs/rsl_rl/go1_torso_tracking/2026-03-12_17-16-01/model_best.pt"
STEPS=1500
NUM_ENVS=4
# Use target=0.42m where policy cross-eval showed 83% timeout
TARGET=0.42

LOG_FILE="logs/perception/oracle_eval_$(date +%Y%m%d_%H%M%S).log"

echo "=== Oracle pi1 eval: ${STEPS} steps, ${NUM_ENVS} envs, target=${TARGET}m ==="
echo "=== Log: ${LOG_FILE} ==="
uv run --active python -u scripts/perception/demo_camera_ekf.py \
    --task Isaac-BallJuggleHier-Go1-Play-v0 \
    --num_envs "$NUM_ENVS" \
    --steps "$STEPS" \
    --headless \
    --pi1-checkpoint "$ORACLE_PI1" \
    --pi2-checkpoint "$PI2" \
    --target-height "$TARGET" \
    --noise-mode oracle \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "=== Done ==="
echo "LOG=$LOG_FILE" > logs/perception/oracle_eval_DONE
touch logs/perception/oracle_eval_DONE
