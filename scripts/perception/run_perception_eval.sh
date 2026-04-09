#!/bin/bash
# Parameterized perception pipeline evaluation: camera → detect → EKF.
# Evaluates a pi1 checkpoint across multiple target heights with optional
# anchor ablation and camera scheduling.
#
# Usage:
#   $C3R_BIN/gpu_lock.sh bash scripts/perception/run_perception_eval.sh \
#     --pi1 <checkpoint> [--pi2 <checkpoint>] [--targets "0.10 0.30 0.50"] \
#     [--noise-mode d435i] [--label my_run] [--anchor] [--camera-scheduling] \
#     [--steps 1500] [--num-envs 4]
#
# Examples:
#   # Quick eval at single target:
#   run_perception_eval.sh --pi1 path/model_best.pt --targets "0.50"
#
#   # Full sweep with anchor + scheduling:
#   run_perception_eval.sh --pi1 path/model_best.pt --targets "0.10 0.20 0.30 0.50 0.70 1.00" \
#     --anchor --camera-scheduling --label stage_g_full
#
#   # Compare two checkpoints (run script twice with different --label):
#   run_perception_eval.sh --pi1 oracle_model.pt --label oracle --targets "0.10 0.30 0.50"
#   run_perception_eval.sh --pi1 d435i_model.pt --label d435i --targets "0.10 0.30 0.50"
#   # Then compare: python scripts/perception/analyze_eval_trajectory.py --npz <a> --compare <b>

set -euo pipefail
cd "$(dirname "$0")/../.."

# ── Defaults ──
PI1=""
PI2="/home/daniel-grant/Research/QuadruJuggle/logs/rsl_rl/go1_torso_tracking/2026-03-12_17-16-01/model_best.pt"
TARGETS="0.10 0.30 0.50"
NOISE_MODE="d435i"
LABEL=""
STEPS=1500
NUM_ENVS=4
ANCHOR=false
CAMERA_SCHED=false
STARVE_LIMIT=10

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case $1 in
        --pi1) PI1="$2"; shift 2 ;;
        --pi2) PI2="$2"; shift 2 ;;
        --targets) TARGETS="$2"; shift 2 ;;
        --noise-mode) NOISE_MODE="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --num-envs) NUM_ENVS="$2"; shift 2 ;;
        --anchor) ANCHOR=true; shift ;;
        --camera-scheduling) CAMERA_SCHED=true; shift ;;
        --starve-limit) STARVE_LIMIT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [ -z "$PI1" ]; then
    echo "ERROR: --pi1 <checkpoint> is required"
    exit 1
fi

if [ ! -f "$PI1" ]; then
    echo "ERROR: pi1 checkpoint not found: $PI1"
    exit 1
fi

# Auto-generate label from checkpoint path if not specified
if [ -z "$LABEL" ]; then
    LABEL=$(basename "$(dirname "$PI1")")_$(basename "$PI1" .pt)
fi

SWEEP_DIR="logs/perception/eval_${LABEL}"
mkdir -p "$SWEEP_DIR"

# Build extra flags
EXTRA_FLAGS=""
if [ "$ANCHOR" = true ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS"
else
    EXTRA_FLAGS="$EXTRA_FLAGS --no-anchor"
fi
if [ "$CAMERA_SCHED" = true ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --camera-scheduling --starve-limit $STARVE_LIMIT"
fi

echo "============================================"
echo "=== Perception eval: ${LABEL}            ==="
echo "============================================"
echo "Pi1:       ${PI1}"
echo "Pi2:       ${PI2}"
echo "Noise:     ${NOISE_MODE}"
echo "Targets:   ${TARGETS}"
echo "Steps:     ${STEPS}, Envs: ${NUM_ENVS}"
echo "Anchor:    ${ANCHOR}, Camera scheduling: ${CAMERA_SCHED}, Starve limit: ${STARVE_LIMIT}"
echo "Output:    ${SWEEP_DIR}"
echo ""

# ── Run each target ──
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
        --pi1-checkpoint "$PI1" \
        --pi2-checkpoint "$PI2" \
        --target-height "$TARGET" \
        --noise-mode "$NOISE_MODE" \
        --out-dir "$OUT_DIR" \
        $EXTRA_FLAGS \
        2>&1 | tail -n 30
    echo ""
done

# ── Summary table ──
echo "============================================"
echo "=== SUMMARY: ${LABEL}                    ==="
echo "============================================"
printf "%-10s %-10s %-10s %-10s %-10s %-10s\n" "Target" "Det%" "Steps" "MeanH" "MaxH" "RMSE_pos"
printf "%-10s %-10s %-10s %-10s %-10s %-10s\n" "------" "-----" "------" "------" "------" "--------"

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
gt = d['gt']
mean_h = gt[:, 2].mean()
max_h = gt[:, 2].max()
# Position RMSE (EKF vs GT, XYZ)
ekf = d['ekf']
if len(ekf) > 0 and len(gt) > 0:
    n = min(len(ekf), len(gt))
    rmse = np.sqrt(np.mean((ekf[:n] - gt[:n])**2))
else:
    rmse = float('nan')
target_str = '$TARGET'
print(f'{target_str:10s} {rate:8.1f}%  {total:8d}  {mean_h:8.3f}  {max_h:8.3f}  {rmse:8.4f}')
" 2>&1
    else
        echo "${TARGET}       --        --        --        --        --"
    fi
done

# ── Save config for reproducibility ──
cat > "${SWEEP_DIR}/eval_config.txt" << CFGEOF
pi1_checkpoint: ${PI1}
pi2_checkpoint: ${PI2}
noise_mode: ${NOISE_MODE}
targets: ${TARGETS}
steps: ${STEPS}
num_envs: ${NUM_ENVS}
anchor: ${ANCHOR}
camera_scheduling: ${CAMERA_SCHED}
starve_limit: ${STARVE_LIMIT}
label: ${LABEL}
date: $(date -Iseconds)
CFGEOF

echo ""
echo "Config saved: ${SWEEP_DIR}/eval_config.txt"
echo "=== Eval complete: ${LABEL} ==="
