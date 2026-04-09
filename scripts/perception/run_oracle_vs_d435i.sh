#!/bin/bash
# Oracle vs D435i full comparison pipeline.
#
# Runs both oracle-trained and d435i-trained checkpoints through the perception
# eval at matching target heights, then generates a comparison dashboard.
#
# Usage:
#   $C3R_BIN/gpu_lock.sh bash scripts/perception/run_oracle_vs_d435i.sh \
#     --oracle-pi1 <path> --d435i-pi1 <path> [--targets "0.10 0.30 0.50 0.70 1.00"] \
#     [--label my_comparison] [--steps 1500] [--num-envs 4]
#
# The oracle checkpoint runs with --noise-mode oracle; the d435i checkpoint runs
# with --noise-mode d435i + anchor + camera-scheduling. This matches each model's
# training regime for a fair comparison.

set -euo pipefail
cd "$(dirname "$0")/../.."

# ── Defaults ──
ORACLE_PI1=""
D435I_PI1=""
PI2="/home/daniel-grant/Research/QuadruJuggle/logs/rsl_rl/go1_torso_tracking/2026-03-12_17-16-01/model_best.pt"
TARGETS="0.10 0.30 0.50 0.70 1.00"
LABEL="comparison_$(date +%Y%m%d_%H%M)"
STEPS=1500
NUM_ENVS=4

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --oracle-pi1) ORACLE_PI1="$2"; shift 2 ;;
        --d435i-pi1) D435I_PI1="$2"; shift 2 ;;
        --pi2) PI2="$2"; shift 2 ;;
        --targets) TARGETS="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --num-envs) NUM_ENVS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$ORACLE_PI1" ] || [ -z "$D435I_PI1" ]; then
    echo "ERROR: Both --oracle-pi1 and --d435i-pi1 are required"
    echo "Usage: $0 --oracle-pi1 <path> --d435i-pi1 <path>"
    exit 1
fi

for ckpt in "$ORACLE_PI1" "$D435I_PI1" "$PI2"; do
    if [ ! -f "$ckpt" ]; then
        echo "ERROR: Checkpoint not found: $ckpt"
        exit 1
    fi
done

ORACLE_DIR="logs/perception/eval_${LABEL}_oracle"
D435I_DIR="logs/perception/eval_${LABEL}_d435i"
FIGURE="images/perception/oracle_vs_d435i_${LABEL}.png"

echo "============================================"
echo "=== Oracle vs D435i Comparison           ==="
echo "============================================"
echo "Oracle pi1: ${ORACLE_PI1}"
echo "D435i pi1:  ${D435I_PI1}"
echo "Pi2:        ${PI2}"
echo "Targets:    ${TARGETS}"
echo "Steps:      ${STEPS}, Envs: ${NUM_ENVS}"
echo "Oracle dir: ${ORACLE_DIR}"
echo "D435i dir:  ${D435I_DIR}"
echo "Figure:     ${FIGURE}"
echo ""

# ── Step 1: Oracle eval (noise-mode oracle, no anchor/scheduling) ──
echo ">>> STEP 1: Oracle eval..."
bash scripts/perception/run_perception_eval.sh \
    --pi1 "$ORACLE_PI1" \
    --pi2 "$PI2" \
    --noise-mode oracle \
    --targets "$TARGETS" \
    --label "${LABEL}_oracle" \
    --steps "$STEPS" \
    --num-envs "$NUM_ENVS"

echo ""

# ── Step 2: D435i eval (noise-mode d435i, anchor + camera scheduling) ──
echo ">>> STEP 2: D435i eval..."
bash scripts/perception/run_perception_eval.sh \
    --pi1 "$D435I_PI1" \
    --pi2 "$PI2" \
    --noise-mode d435i \
    --targets "$TARGETS" \
    --label "${LABEL}_d435i" \
    --anchor \
    --camera-scheduling \
    --starve-limit 10 \
    --steps "$STEPS" \
    --num-envs "$NUM_ENVS"

echo ""

# ── Step 3: Comparison figure ──
echo ">>> STEP 3: Generating comparison figure..."
mkdir -p "$(dirname "$FIGURE")"
uv run --active python scripts/perception/compare_multi_target.py \
    --dir-a "$ORACLE_DIR" \
    --dir-b "$D435I_DIR" \
    --labels "Oracle" "D435i" \
    --out "$FIGURE"

echo ""
echo "============================================"
echo "=== Comparison complete                  ==="
echo "=== Figure: ${FIGURE}                    ==="
echo "============================================"
