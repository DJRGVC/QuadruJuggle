#!/usr/bin/env bash
# Open Isaac Sim GUI and preview the ball-balance scene.
# No checkpoint required — runs with zero actions.
#
# Usage:
#   ./scripts/preview.sh              # 4 envs, 500 steps
#   ./scripts/preview.sh --num_envs 1 # single env (cleaner view)
set -e

ISAACLAB_DIR="${HOME}/IsaacLab"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "${ISAACLAB_DIR}/.venv311/bin/activate"

# Must cd to IsaacLab so hydra resolves log paths correctly
cd "${ISAACLAB_DIR}"

ISAACSIM_RENDERING_MODE=performance \
    python "${REPO_DIR}/scripts/preview.py" "$@"
