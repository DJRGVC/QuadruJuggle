#!/usr/bin/env bash
# Train Go1 ball-balance — Phase 1 (privileged state PPO)
set -e

ISAACLAB_DIR="${HOME}/IsaacLab"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ISAACLAB_DIR}/logs/rsl_rl/go1_ball_balance"

source "${ISAACLAB_DIR}/.venv311/bin/activate"

NUM_ENVS=4096
MAX_ITERATIONS=3000
TASK="Isaac-BallBalance-Go1-v0"

echo "=================================================="
echo " QuadruJuggle — Go1 Ball Balance Training"
echo " envs: ${NUM_ENVS}  |  iters: ${MAX_ITERATIONS}"
echo "=================================================="
echo ""

echo "GPU status at launch:"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""
echo "TensorBoard:  tensorboard --logdir ${LOG_DIR}"
echo ""

cd "${ISAACLAB_DIR}"

ISAACSIM_RENDERING_MODE=performance \
    python "${REPO_DIR}/scripts/rsl_rl/train.py" \
    --task "${TASK}" \
    --headless \
    --num_envs "${NUM_ENVS}" \
    --max_iterations "${MAX_ITERATIONS}"
