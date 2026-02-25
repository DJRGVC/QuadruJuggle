#!/usr/bin/env bash
# Play the latest (or specified) Go1 ball-balance checkpoint
#   Usage: ./scripts/play.sh [run-name]
set -e

ISAACLAB_DIR="${HOME}/IsaacLab"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ISAACLAB_DIR}/logs/rsl_rl/go1_ball_balance"

source "${ISAACLAB_DIR}/.venv311/bin/activate"

if [ -n "$1" ]; then
    RUN="$1"
else
    RUN=$(ls -1t "${LOG_DIR}" | head -1)
fi

echo "Playing run: ${RUN}"
echo "Log dir:     ${LOG_DIR}/${RUN}"
echo ""

cd "${ISAACLAB_DIR}"

ISAACSIM_RENDERING_MODE=performance \
    python "${REPO_DIR}/scripts/rsl_rl/play.py" \
    --task Isaac-BallBalance-Go1-Play-v0 \
    --num_envs 16 \
    --load_run "${RUN}"
