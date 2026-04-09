#!/usr/bin/env bash
# Run comparison between oracle and d435i camera eval runs.
# Uses trajectory.npz files (robust, stdout-independent) as primary path,
# falls back to text log parsing if npz not available.
set -euo pipefail
cd "$(dirname "$0")/../.."

DEMO_DIR="source/go1_ball_balance/go1_ball_balance/perception/debug/demo"
ORACLE_NPZ="${DEMO_DIR}/trajectory.npz"
OUT="images/perception/oracle_vs_d435i_comparison.png"

# Primary path: trajectory.npz analysis (works even if stdout was lost)
if [ -f "$ORACLE_NPZ" ]; then
    echo "=== Analyzing trajectory.npz (stdout-independent) ==="
    uv run --active python scripts/perception/analyze_eval_trajectory.py \
        --npz "$ORACLE_NPZ" \
        --labels "Oracle policy" \
        --out "$OUT" \
        --quarto-copy
    echo ""
    echo "Done. Figure at: $OUT"
    exit 0
fi

# Fallback: text log parsing
ORACLE_LOG="logs/perception/oracle_eval.log"
echo "=== No trajectory.npz found, falling back to text log parsing ==="

if [ ! -f "$ORACLE_LOG" ] || [ "$(wc -c < "$ORACLE_LOG")" -lt 200 ]; then
    # Try timestamped log files
    ORACLE_LOG=$(ls -t logs/perception/oracle_eval_*.log 2>/dev/null | head -1 || true)
    if [ -z "$ORACLE_LOG" ] || [ "$(wc -c < "$ORACLE_LOG")" -lt 200 ]; then
        echo "ERROR: No usable oracle eval log found."
        echo "Run run_oracle_eval.sh first, or check if trajectory.npz exists at:"
        echo "  $DEMO_DIR/"
        exit 1
    fi
fi

uv run --active python scripts/perception/compare_eval_runs.py \
    --logs "$ORACLE_LOG" \
    --labels "Oracle policy" \
    --out "$OUT"

echo ""
echo "Done. Figure at: $OUT"
