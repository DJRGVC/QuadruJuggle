#!/bin/bash
# Cross-eval with fixed eval script (partial-episode bug fixed in iter 29).
# Tests d435i-trained and oracle-trained checkpoints under both noise modes.

set -euo pipefail

PI2="/home/daniel-grant/Research/QuadruJuggle/logs/rsl_rl/go1_torso_tracking/2026-03-12_17-16-01/model_best.pt"
D435I_CKPT="logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_22-51-56/model_best.pt"
ORACLE_CKPT="logs/rsl_rl/go1_ball_juggle_hier/2026-04-08_19-19-41/model_best.pt"
TARGETS="0.10,0.20,0.30,0.42,0.50"
OUTDIR="experiments/iter_029_cross_eval_fixed"

mkdir -p "$OUTDIR"

echo "=== Cross-eval (fixed): 4 combinations ==="

for CKPT_NAME in oracle d435i; do
    if [ "$CKPT_NAME" = "d435i" ]; then
        CKPT="$D435I_CKPT"
    else
        CKPT="$ORACLE_CKPT"
    fi

    for NOISE in oracle d435i; do
        LABEL="${CKPT_NAME}_trained__${NOISE}_obs"
        echo ""
        echo ">>> Running: $LABEL"
        echo "    checkpoint=$CKPT  noise_mode=$NOISE"

        uv run --active python scripts/rsl_rl/eval_juggle_hier.py \
            --task Isaac-BallJuggleHier-Go1-Play-v0 \
            --pi2-checkpoint "$PI2" \
            --checkpoint "$CKPT" \
            --noise-mode "$NOISE" \
            --num_envs 256 \
            --episodes 30 \
            --targets "$TARGETS" \
            --headless \
            > "$OUTDIR/${LABEL}.log" 2>&1

        echo "    Done. Output: $OUTDIR/${LABEL}.log"
        tail -15 "$OUTDIR/${LABEL}.log"
        echo ""
    done
done

echo "=== All 4 cross-eval runs complete ==="
echo "Results in $OUTDIR/"
