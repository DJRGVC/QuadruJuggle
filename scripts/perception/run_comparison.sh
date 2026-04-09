#!/usr/bin/env bash
# Run comparison between oracle and d435i camera eval runs.
# Uses trajectory.npz files from dedicated eval output directories.
set -euo pipefail
cd "$(dirname "$0")/../.."

ORACLE_NPZ="logs/perception/oracle_eval/trajectory.npz"
D435I_NPZ="logs/perception/d435i_eval/trajectory.npz"
OUT="images/perception/oracle_vs_d435i_comparison.png"

# Single-run analysis (oracle only)
if [ -f "$ORACLE_NPZ" ] && [ ! -f "$D435I_NPZ" ]; then
    echo "=== Analyzing oracle eval trajectory ==="
    uv run --active python scripts/perception/analyze_eval_trajectory.py \
        --npz "$ORACLE_NPZ" \
        --labels "Oracle policy" \
        --out "$OUT" \
        --quarto-copy
    echo "Done. Figure at: $OUT"
    exit 0
fi

# Two-run comparison
if [ -f "$ORACLE_NPZ" ] && [ -f "$D435I_NPZ" ]; then
    echo "=== Comparing oracle vs d435i eval trajectories ==="
    uv run --active python scripts/perception/analyze_eval_trajectory.py \
        --npz "$ORACLE_NPZ" \
        --compare "$D435I_NPZ" \
        --labels "Oracle policy" "D435i policy" \
        --out "$OUT" \
        --quarto-copy
    echo "Done. Figure at: $OUT"
    exit 0
fi

# Fallback: check legacy demo dir
LEGACY_NPZ="source/go1_ball_balance/go1_ball_balance/perception/debug/demo/trajectory.npz"
if [ -f "$LEGACY_NPZ" ]; then
    echo "=== Analyzing legacy demo trajectory ==="
    uv run --active python scripts/perception/analyze_eval_trajectory.py \
        --npz "$LEGACY_NPZ" \
        --labels "Demo run" \
        --out "$OUT" \
        --quarto-copy
    echo "Done. Figure at: $OUT"
    exit 0
fi

echo "ERROR: No trajectory.npz found."
echo "Run run_oracle_eval.sh and/or run_d435i_eval.sh first."
echo "Expected: $ORACLE_NPZ and/or $D435I_NPZ"
exit 1
