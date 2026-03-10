#!/usr/bin/env bash
# Play the latest (or specified) Go1 ball-balance checkpoint
#   Usage: ./scripts/play.sh [--slow N] [run-name]
#
#   --slow N   Slow-motion multiplier (e.g. --slow 4 = 4× slower than real-time)
#   run-name   Optional: name of the log directory under go1_ball_balance/
#              Defaults to the most recent run.
set -e

ISAACLAB_DIR="${HOME}/IsaacLab"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_DIR}/logs/rsl_rl/go1_ball_balance"

source "${ISAACLAB_DIR}/.venv311/bin/activate"

# Parse arguments
SLOW_ARG=""
RUN_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --slow)
            SLOW_ARG="--slow $2"
            shift 2
            ;;
        *)
            RUN_ARG="$1"
            shift
            ;;
    esac
done

if [ -n "$RUN_ARG" ]; then
    RUN="$RUN_ARG"
else
    RUN=$(ls -1t "${LOG_DIR}" | head -1)
fi

echo "Playing run: ${RUN}"
echo "Log dir:     ${LOG_DIR}/${RUN}"
[ -n "$SLOW_ARG" ] && echo "Slow motion: ${SLOW_ARG}"
echo ""

cd "${REPO_DIR}"

ISAACSIM_RENDERING_MODE=performance \
    uv run --active python "${REPO_DIR}/scripts/rsl_rl/play.py" \
    --task Isaac-BallBalance-Go1-Play-v0 \
    --num_envs 16 \
    --load_run "${RUN}" \
    ${SLOW_ARG}
