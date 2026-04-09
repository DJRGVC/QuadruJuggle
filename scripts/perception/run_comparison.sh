#!/usr/bin/env bash
# Run comparison between oracle and d435i camera eval runs.
# Prerequisites: both eval logs must exist.
set -euo pipefail
cd "$(dirname "$0")/../.."

ORACLE_LOG="logs/perception/oracle_eval.log"
D435I_LOG="logs/perception/d435i_eval.log"
OUT="images/perception/oracle_vs_d435i_comparison.png"

# Check prerequisites
if [ ! -f "$ORACLE_LOG" ] || [ "$(wc -c < "$ORACLE_LOG")" -lt 200 ]; then
    echo "ERROR: Oracle eval log missing or too short: $ORACLE_LOG"
    exit 1
fi

# D435i log is optional — from iter 92 we have the numbers in RESEARCH_LOG
# but no separate log file. Skip if missing.
ARGS=(--logs "$ORACLE_LOG" --labels "Oracle policy")
if [ -f "$D435I_LOG" ] && [ "$(wc -c < "$D435I_LOG")" -gt 200 ]; then
    ARGS=(--logs "$ORACLE_LOG" "$D435I_LOG" --labels "Oracle policy" "D435i policy")
fi

uv run --active python scripts/perception/compare_eval_runs.py \
    "${ARGS[@]}" \
    --out "$OUT"

echo ""
echo "Done. Figure at: $OUT"
