#!/bin/bash
# Full Stage G comparison: d435i vs oracle, anchor ON vs OFF.
# Runs 4 eval sweeps sequentially and generates comparison summary.
#
# Usage:
#   $C3R_BIN/gpu_lock.sh bash scripts/perception/run_stage_g_comparison.sh \
#     --pi1 <checkpoint>
#
# Runs:
#   1. d435i + anchor + camera-scheduling  (production config)
#   2. d435i + no-anchor                   (anchor ablation)
#   3. oracle + anchor                     (oracle baseline)
#   4. oracle + no-anchor                  (oracle no-anchor)

set -euo pipefail
cd "$(dirname "$0")/../.."

PI1=""
PI2="/home/daniel-grant/Research/QuadruJuggle/logs/rsl_rl/go1_torso_tracking/2026-03-12_09-04-32/model_best.pt"
TARGETS="0.10 0.30 0.50 0.70 1.00"
STEPS=1500
NUM_ENVS=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --pi1) PI1="$2"; shift 2 ;;
        --pi2) PI2="$2"; shift 2 ;;
        --targets) TARGETS="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --num-envs) NUM_ENVS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$PI1" ]; then
    echo "ERROR: --pi1 <checkpoint> is required"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COMPARISON_DIR="logs/perception/comparison_${TIMESTAMP}"
mkdir -p "$COMPARISON_DIR"

echo "============================================================"
echo "=== Stage G Comparison — ${TIMESTAMP}                    ==="
echo "============================================================"
echo "Pi1: ${PI1}"
echo "Targets: ${TARGETS}"
echo "Output: ${COMPARISON_DIR}"
echo ""

# Helper to run one eval variant
run_eval() {
    local label="$1"
    local noise_mode="$2"
    local anchor_flag="$3"
    local sched_flag="$4"

    echo ""
    echo "============================================================"
    echo "=== Running: ${label}                                    ==="
    echo "============================================================"

    local extra=""
    [ "$anchor_flag" = "true" ] && extra="--anchor" || extra=""
    [ "$sched_flag" = "true" ] && extra="$extra --camera-scheduling"

    bash scripts/perception/run_perception_eval.sh \
        --pi1 "$PI1" \
        --pi2 "$PI2" \
        --targets "$TARGETS" \
        --noise-mode "$noise_mode" \
        --label "${label}" \
        --steps "$STEPS" \
        --num-envs "$NUM_ENVS" \
        $extra

    # Copy results into comparison directory
    cp -r "logs/perception/eval_${label}" "${COMPARISON_DIR}/${label}"
}

# ── Run all 4 configurations ──
run_eval "d435i_anchor"    "d435i"  "true"  "true"
run_eval "d435i_no_anchor" "d435i"  "false" "false"
run_eval "oracle_anchor"   "oracle" "true"  "false"
run_eval "oracle_baseline" "oracle" "false" "false"

# ── Generate combined summary ──
echo ""
echo "============================================================"
echo "=== COMBINED SUMMARY                                     ==="
echo "============================================================"
echo ""

for VARIANT in d435i_anchor d435i_no_anchor oracle_anchor oracle_baseline; do
    echo "--- ${VARIANT} ---"
    if [ -f "${COMPARISON_DIR}/${VARIANT}/eval_config.txt" ]; then
        printf "%-10s %-10s %-10s %-10s %-10s %-10s\n" "Target" "Det%" "Steps" "MeanH" "MaxH" "RMSE"
        for TARGET in $TARGETS; do
            TAG=$(echo "$TARGET" | tr '.' '_')
            NPZ="${COMPARISON_DIR}/${VARIANT}/target_${TAG}/trajectory.npz"
            if [ -f "$NPZ" ]; then
                uv run --active python -c "
import numpy as np
d = np.load('$NPZ')
total = len(d['steps'])
num_det = len(d['det_steps'])
rate = num_det / max(total, 1) * 100
gt = d['gt']
mean_h = gt[:, 2].mean()
max_h = gt[:, 2].max()
ekf = d['ekf']
if len(ekf) > 0 and len(gt) > 0:
    n = min(len(ekf), len(gt))
    rmse = np.sqrt(np.mean((ekf[:n] - gt[:n])**2))
else:
    rmse = float('nan')
print(f'$TARGET     {rate:6.1f}%  {total:8d}  {mean_h:8.3f}  {max_h:8.3f}  {rmse:8.4f}')
" 2>&1
            fi
        done
    fi
    echo ""
done

# Save comparison manifest
cat > "${COMPARISON_DIR}/manifest.txt" << EOF
pi1_checkpoint: ${PI1}
pi2_checkpoint: ${PI2}
targets: ${TARGETS}
steps: ${STEPS}
num_envs: ${NUM_ENVS}
timestamp: ${TIMESTAMP}
variants: d435i_anchor d435i_no_anchor oracle_anchor oracle_baseline
EOF

echo "Manifest: ${COMPARISON_DIR}/manifest.txt"
echo "=== Comparison complete ==="
