#!/usr/bin/env bash
# Sweep hybrid controller hyper-parameters and rank by performance.
#
# Usage:
#   bash scripts/sweep_hybrid.sh \
#       logs/rsl_rl/go1_ball_juggle_hier/2026-03-22_19-46-52/model_best.pt \
#       logs/rsl_rl/go1_torso_tracking/2026-03-13_03-02-24/model_best.pt
#
# Metrics printed per run (SWEEP_RESULT line):
#   mirror_frac  — fraction of total env-steps spent in mirror-law mode (higher = more stable)
#   mean_apex    — mean bounce apex while in mirror mode (closer to target = better)
#   median_apex  — median bounce apex while in mirror mode

PI1=$1
PI2=$2

if [[ -z "$PI1" || -z "$PI2" ]]; then
    echo "Usage: $0 <pi1_checkpoint> <pi2_checkpoint>"
    exit 1
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LOG="$REPO/logs/sweep_hybrid_$(date +%Y%m%d_%H%M%S).txt"
MAX_STEPS=2000
NUM_ENVS=2   # keep small for speed

APEX_HEIGHTS=(0.15 0.20 0.25 0.30)
SWITCH_THRESHOLDS=(0.70 0.80 0.90)
FALLBACK_THRESHOLDS=(0.40 0.50 0.60)

echo "Sweep log: $LOG"
echo "apex_heights    : ${APEX_HEIGHTS[*]}"
echo "switch_thresholds : ${SWITCH_THRESHOLDS[*]}"
echo "fallback_thresholds: ${FALLBACK_THRESHOLDS[*]}"
echo "max_steps=$MAX_STEPS  num_envs=$NUM_ENVS"
echo ""

mkdir -p "$(dirname "$LOG")"
> "$LOG"

total=$(( ${#APEX_HEIGHTS[@]} * ${#SWITCH_THRESHOLDS[@]} * ${#FALLBACK_THRESHOLDS[@]} ))
run=0

for apex in "${APEX_HEIGHTS[@]}"; do
for sw in "${SWITCH_THRESHOLDS[@]}"; do
for fb in "${FALLBACK_THRESHOLDS[@]}"; do
    run=$(( run + 1 ))
    echo -n "[$run/$total] apex=$apex sw=$sw fb=$fb  ... "

    result=$(
        cd "$REPO" && \
        PYTHONPATH=/home/frank/IsaacLab/scripts/reinforcement_learning/rsl_rl:$PYTHONPATH \
        python scripts/play_hybrid.py \
            --pi1_checkpoint "$PI1" \
            --pi2_checkpoint "$PI2" \
            --apex_height "$apex" \
            --switch_threshold "$sw" \
            --fallback_threshold "$fb" \
            --num_envs $NUM_ENVS \
            --max_steps $MAX_STEPS \
            --headless 2>/dev/null \
        | grep "^SWEEP_RESULT"
    )

    echo "$result"
    echo "$result" >> "$LOG"
done
done
done

echo ""
echo "=== Results ranked by mirror_frac (desc) ==="
grep "^SWEEP_RESULT" "$LOG" \
    | sort -t= -k5 -rn \
    | head -10

echo ""
echo "=== Results ranked by mean_apex proximity to target ==="
# Sort by abs(mean_apex - target_apex): awk extracts apex and mean_apex, computes diff
grep "^SWEEP_RESULT" "$LOG" \
    | awk '{
        for(i=1;i<=NF;i++){
            if($i ~ /^apex=/)    { split($i,a,"="); apex=a[2] }
            if($i ~ /^mean_apex=/){ split($i,m,"="); ma=m[2]  }
        }
        diff = apex - ma; if(diff<0) diff=-diff
        print diff, $0
    }' \
    | sort -n \
    | head -10 \
    | cut -d' ' -f2-

echo ""
echo "Full results in: $LOG"
