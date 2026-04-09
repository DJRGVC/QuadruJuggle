#!/bin/bash
# A/B comparison: paddle anchor ON vs OFF for EKF accuracy during contact phases.
# Usage: $C3R_BIN/gpu_lock.sh bash scripts/perception/run_anchor_ablation.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

D435I_PI1="/home/daniel-grant/Research/QuadruJuggle-policy/logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_22-51-56/model_best.pt"
PI2="/home/daniel-grant/Research/QuadruJuggle/logs/rsl_rl/go1_torso_tracking/2026-03-12_17-16-01/model_best.pt"
STEPS=1500
NUM_ENVS=4
TARGET=0.42

ANCHOR_ON_OUT="logs/perception/anchor_on_eval"
ANCHOR_OFF_OUT="logs/perception/anchor_off_eval"

echo "============================================"
echo "=== Anchor ablation: ON vs OFF ==="
echo "============================================"
echo "Steps: ${STEPS}, Envs: ${NUM_ENVS}, Target: ${TARGET}m"
echo ""

# ── 1. Anchor ON (default) ──
echo "=== [1/2] Anchor ON ==="
mkdir -p "$ANCHOR_ON_OUT"
uv run --active python -u scripts/perception/demo_camera_ekf.py \
    --task Isaac-BallJuggleHier-Go1-Play-v0 \
    --num_envs "$NUM_ENVS" \
    --steps "$STEPS" \
    --headless \
    --pi1-checkpoint "$D435I_PI1" \
    --pi2-checkpoint "$PI2" \
    --target-height "$TARGET" \
    --noise-mode d435i \
    --out-dir "$ANCHOR_ON_OUT" \
    2>&1 | tail -n 50
echo ""

# ── 2. Anchor OFF ──
echo "=== [2/2] Anchor OFF ==="
mkdir -p "$ANCHOR_OFF_OUT"
uv run --active python -u scripts/perception/demo_camera_ekf.py \
    --task Isaac-BallJuggleHier-Go1-Play-v0 \
    --num_envs "$NUM_ENVS" \
    --steps "$STEPS" \
    --headless \
    --pi1-checkpoint "$D435I_PI1" \
    --pi2-checkpoint "$PI2" \
    --target-height "$TARGET" \
    --noise-mode d435i \
    --no-anchor \
    --out-dir "$ANCHOR_OFF_OUT" \
    2>&1 | tail -n 50
echo ""

# ── 3. Comparison ──
echo "=== Comparison ==="
if [ -f "${ANCHOR_ON_OUT}/trajectory.npz" ] && [ -f "${ANCHOR_OFF_OUT}/trajectory.npz" ]; then
    # General comparison (height-binned)
    uv run --active python scripts/perception/analyze_eval_trajectory.py \
        --npz "${ANCHOR_ON_OUT}/trajectory.npz" \
        --compare "${ANCHOR_OFF_OUT}/trajectory.npz" \
        --labels "Anchor ON" "Anchor OFF" \
        --out "images/perception/anchor_ablation.png" \
        2>&1
    echo "Figure: images/perception/anchor_ablation.png"

    # Detailed anchor-specific analysis (phase RMSE, cumulative divergence)
    uv run --active python scripts/perception/analyze_anchor_ablation.py \
        --on "${ANCHOR_ON_OUT}/trajectory.npz" \
        --off "${ANCHOR_OFF_OUT}/trajectory.npz" \
        --out "images/perception/anchor_ablation_detail.png" \
        2>&1
    echo "Figure: images/perception/anchor_ablation_detail.png"
fi

echo ""
echo "=== Anchor ablation complete ==="
echo "DONE=$(date -Iseconds)" > logs/perception/anchor_ablation_DONE
