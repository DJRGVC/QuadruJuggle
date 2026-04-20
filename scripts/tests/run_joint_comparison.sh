#!/usr/bin/env bash
# Run joint command comparison: MuJoCo vs Isaac Lab.
#
# Runs both simulators for a specified joint (or all 12), saves CSVs to
# tests_out/, then prints a side-by-side comparison of steady-state error.
#
# Usage:
#   # Compare FR_hip only (MuJoCo idx=0, Isaac Lab idx=3):
#   bash scripts/tests/run_joint_comparison.sh --joint_pair 0 3
#
#   # Run all 12 joints (slow — requires Isaac Lab for each):
#   bash scripts/tests/run_joint_comparison.sh --all
#
# Output files:
#   tests_out/mujoco_joint{N}.csv
#   tests_out/isaaclab_joint{N}.csv

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/tests_out"
mkdir -p "$OUT"

MUJOCO_IDXS=()
ISAAC_IDXS=()
ALL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --joint_pair)
            MUJOCO_IDXS+=("$2")
            ISAAC_IDXS+=("$3")
            shift 3
            ;;
        --all)
            # MJCF leg-grouped (FR 0-2, FL 3-5, RR 6-8, RL 9-11)
            # → Isaac type-grouped (all hips 0-3, all thighs 4-7, all calves 8-11)
            # [FL FR RL RR]_hip=0-3, [FL FR RL RR]_thigh=4-7, [FL FR RL RR]_calf=8-11
            MUJOCO_IDXS=(0 1 2   3 4 5   6 7 8    9 10 11)
            ISAAC_IDXS=( 1 5 9   0 4 8   3 7 11   2  6 10)
            ALL=true
            shift
            ;;
        *)
            echo "Unknown arg: $1" >&2
            exit 1
            ;;
    esac
done

if [[ ${#MUJOCO_IDXS[@]} -eq 0 ]]; then
    # Default: FR_hip  (MJCF idx=0 → Isaac idx=1)
    MUJOCO_IDXS=(0)
    ISAAC_IDXS=(1)
fi

HOLD=50
STEPS=150

echo "=== Running MuJoCo tests ==="
for i in "${!MUJOCO_IDXS[@]}"; do
    MIDX="${MUJOCO_IDXS[$i]}"
    echo "  MuJoCo joint_idx=$MIDX"
    conda run -n isaaclab python "$ROOT/scripts/tests/test_joint_cmd_mujoco.py" \
        --joint_idx "$MIDX" \
        --use_actuator_net \
        --hold_steps "$HOLD" \
        --step_steps "$STEPS" \
        > "$OUT/mujoco_joint${MIDX}.csv" 2>"$OUT/mujoco_joint${MIDX}.log"
    echo "    → $OUT/mujoco_joint${MIDX}.csv"
done

echo ""
echo "=== Running Isaac Lab tests ==="
for i in "${!ISAAC_IDXS[@]}"; do
    IIDX="${ISAAC_IDXS[$i]}"
    echo "  Isaac Lab joint_idx=$IIDX"
    conda run -n isaaclab env \
        PYTHONPATH="/home/frank/IsaacLab/scripts/reinforcement_learning/rsl_rl:${PYTHONPATH:-}" \
        python "$ROOT/scripts/tests/test_joint_cmd_isaaclab.py" \
        --joint_idx "$IIDX" \
        --hold_steps "$HOLD" \
        --step_steps "$STEPS" \
        --headless \
        > "$OUT/isaaclab_joint${IIDX}.csv" 2>"$OUT/isaaclab_joint${IIDX}.log"
    echo "    → $OUT/isaaclab_joint${IIDX}.csv"
done

echo ""
echo "=== Steady-state error comparison (last 20 steps of step phase) ==="
printf "%-20s  %-10s  %-12s  %-12s\n" "Joint" "Sim" "Target(rad)" "SS-Error(rad)"
echo "----------------------------------------------------------------------"

compare_joint() {
    local mujoco_csv="$1"
    local isaac_csv="$2"
    local joint_name="$3"

    if [[ -f "$mujoco_csv" ]]; then
        # Last 20 steps, only the active joint (first column matches step, grab error col)
        SS_MJ=$(awk -F',' -v j="$joint_name" \
            'NR>1 && $3==j {err+=$7; n++} END {if(n>0) printf "%.5f", err/n}' \
            <(tail -n $((20*12)) "$mujoco_csv"))
        TGT_MJ=$(awk -F',' -v j="$joint_name" \
            'NR>1 && $3==j {t=$5} END {printf "%.4f", t}' \
            <(tail -n $((20*12)) "$mujoco_csv"))
        printf "%-20s  %-10s  %-12s  %-12s\n" "$joint_name" "mujoco" "$TGT_MJ" "$SS_MJ"
    fi

    if [[ -f "$isaac_csv" ]]; then
        SS_IL=$(awk -F',' -v j="${joint_name}_joint" \
            'NR>1 && $3==j {err+=$7; n++} END {if(n>0) printf "%.5f", err/n}' \
            <(tail -n $((20*12)) "$isaac_csv"))
        TGT_IL=$(awk -F',' -v j="${joint_name}_joint" \
            'NR>1 && $3==j {t=$5} END {printf "%.4f", t}' \
            <(tail -n $((20*12)) "$isaac_csv"))
        printf "%-20s  %-10s  %-12s  %-12s\n" "$joint_name" "isaaclab" "$TGT_IL" "$SS_IL"
    fi
}

# Map MJCF name → Isaac name (strip _joint suffix for comparison)
for i in "${!MUJOCO_IDXS[@]}"; do
    MIDX="${MUJOCO_IDXS[$i]}"
    IIDX="${ISAAC_IDXS[$i]}"
    JOINT_NAMES=(FR_hip FR_thigh FR_calf FL_hip FL_thigh FL_calf RR_hip RR_thigh RR_calf RL_hip RL_thigh RL_calf)
    JNAME="${JOINT_NAMES[$MIDX]}"
    compare_joint "$OUT/mujoco_joint${MIDX}.csv" "$OUT/isaaclab_joint${IIDX}.csv" "$JNAME"
done

echo ""
echo "Done. Full CSVs in: $OUT/"
