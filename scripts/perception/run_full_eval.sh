#!/bin/bash
# Run BOTH oracle and d435i eval back-to-back, then generate comparison figure.
# Designed to run inside a single GPU lock to minimize lock overhead.
# Usage: $C3R_BIN/gpu_lock.sh bash scripts/perception/run_full_eval.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

ORACLE_PI1="/home/daniel-grant/Research/QuadruJuggle-policy/logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_19-19-41/model_best.pt"
D435I_PI1="/home/daniel-grant/Research/QuadruJuggle-policy/logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_22-51-56/model_best.pt"
PI2="/home/daniel-grant/Research/QuadruJuggle/logs/rsl_rl/go1_torso_tracking/2026-03-12_17-16-01/model_best.pt"
STEPS=1500
NUM_ENVS=4
TARGET=0.42

ORACLE_OUT="logs/perception/oracle_eval"
D435I_OUT="logs/perception/d435i_eval"
COMPARISON_OUT="images/perception/oracle_vs_d435i_comparison.png"

echo "============================================"
echo "=== Full eval: oracle + d435i + comparison ==="
echo "============================================"
echo "Steps: ${STEPS}, Envs: ${NUM_ENVS}, Target: ${TARGET}m"
echo ""

# ── 1. Oracle eval ──
echo "=== [1/3] Oracle pi1 eval (noise-mode=oracle) ==="
mkdir -p "$ORACLE_OUT"
uv run --active python -u scripts/perception/demo_camera_ekf.py \
    --task Isaac-BallJuggleHier-Go1-Play-v0 \
    --num_envs "$NUM_ENVS" \
    --steps "$STEPS" \
    --headless \
    --pi1-checkpoint "$ORACLE_PI1" \
    --pi2-checkpoint "$PI2" \
    --target-height "$TARGET" \
    --noise-mode oracle \
    --out-dir "$ORACLE_OUT" \
    2>&1 | tail -n 50
echo ""
echo "=== Oracle eval done ==="
echo ""

# ── 2. D435i eval ──
echo "=== [2/3] D435i pi1 eval (noise-mode=d435i) ==="
mkdir -p "$D435I_OUT"
uv run --active python -u scripts/perception/demo_camera_ekf.py \
    --task Isaac-BallJuggleHier-Go1-Play-v0 \
    --num_envs "$NUM_ENVS" \
    --steps "$STEPS" \
    --headless \
    --pi1-checkpoint "$D435I_PI1" \
    --pi2-checkpoint "$PI2" \
    --target-height "$TARGET" \
    --noise-mode d435i \
    --out-dir "$D435I_OUT" \
    2>&1 | tail -n 50
echo ""
echo "=== D435i eval done ==="
echo ""

# ── 3. Comparison analysis ──
echo "=== [3/3] Generating comparison figure ==="
if [ -f "${ORACLE_OUT}/trajectory.npz" ] && [ -f "${D435I_OUT}/trajectory.npz" ]; then
    uv run --active python scripts/perception/analyze_eval_trajectory.py \
        --npz "${ORACLE_OUT}/trajectory.npz" \
        --compare "${D435I_OUT}/trajectory.npz" \
        --labels "Oracle policy" "D435i policy" \
        --out "$COMPARISON_OUT" \
        --quarto-copy 2>&1
    echo "Comparison figure: $COMPARISON_OUT"
else
    echo "WARNING: Missing trajectory.npz, running individual analysis"
    [ -f "${ORACLE_OUT}/trajectory.npz" ] && \
        uv run --active python scripts/perception/analyze_eval_trajectory.py \
            --npz "${ORACLE_OUT}/trajectory.npz" --labels "Oracle" 2>&1
    [ -f "${D435I_OUT}/trajectory.npz" ] && \
        uv run --active python scripts/perception/analyze_eval_trajectory.py \
            --npz "${D435I_OUT}/trajectory.npz" --labels "D435i" 2>&1
fi

echo ""
echo "============================================"
echo "=== Full eval complete ==="
echo "============================================"

# Write sentinel files
echo "DONE=$(date -Iseconds)" > logs/perception/oracle_eval_DONE
echo "DONE=$(date -Iseconds)" > logs/perception/d435i_eval_DONE
echo "COMPARISON=$COMPARISON_OUT" > logs/perception/full_eval_DONE
