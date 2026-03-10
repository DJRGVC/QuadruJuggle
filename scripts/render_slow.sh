#!/usr/bin/env bash
# Three-step slow-motion recording pipeline for the Go1 ball-balance policy.
#
# Step 1 — Record trajectory at physics rate (200 Hz) while running the policy.
# Step 2 — Interpolate offline with cubic spline + SLERP and render to video.
# Step 3 — Re-encode at 30 FPS targeting <10 MB for Discord / sharing.
#
# Slowdown formula:  slowdown = --fps / --playback-fps
#
# Usage:
#   ./scripts/render_slow.sh [options]
#
# Options:
#   --run  NAME         Log-run directory under go1_ball_balance/ (default: latest)
#   --mult N            Slowdown multiplier (default: 16)
#   --playback-fps N    Container / display FPS of the output video (default: 30)
#                       render_fps is computed automatically as mult × playback-fps
#   --dur  S            Limit recording to first S seconds (default: full episode)
#   --record-steps N    Physics frames to record (default: 600 = 3 s at 200 Hz)
#   --out  FILE         Output video filename (default: slow_motion.mp4)
#   --env-idx N         Which env to render from the trajectory (default: 0)
#
# Examples:
#   ./scripts/render_slow.sh                                    # 16× slow, 30 FPS
#   ./scripts/render_slow.sh --mult 8                           # 8× slow, 30 FPS
#   ./scripts/render_slow.sh --mult 4                           # 4× slow, 30 FPS
#   ./scripts/render_slow.sh --mult 16 --playback-fps 240       # 16× slow, 240 FPS
#   ./scripts/render_slow.sh --mult 16 --record-steps 1200      # 16× slow, 6 s clip
#   ./scripts/render_slow.sh --run 2026-02-27_12-00-00 --mult 16
set -e

ISAACLAB_DIR="${HOME}/IsaacLab"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
LOG_DIR="${REPO_DIR}/logs/rsl_rl/go1_ball_balance"



source "${ISAACLAB_DIR}/.venv311/bin/activate"

# ── defaults ──────────────────────────────────────────────────────────────────
RUN=""
MULT=16            # slowdown multiplier
PLAYBACK_FPS=30    # container / display FPS
DURATION=""
OUTPUT="slow_motion.mp4"
ENV_IDX=0
TRAJ_FILE="/tmp/bb_traj.npz"
RECORD_STEPS=600   # physics frames to record (default 600 = 3 s at 200 Hz)

# ── parse args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)           RUN="$2";           shift 2 ;;
        --mult)          MULT="$2";          shift 2 ;;
        --playback-fps)  PLAYBACK_FPS="$2";  shift 2 ;;
        --dur)           DURATION="$2";      shift 2 ;;
        --out)           OUTPUT="$2";        shift 2 ;;
        --env-idx)       ENV_IDX="$2";       shift 2 ;;
        --record-steps)  RECORD_STEPS="$2";  shift 2 ;;
        *) echo "[ERROR] Unknown option: $1"; exit 1 ;;
    esac
done

# render_fps = slowdown × playback_fps  (e.g. 16 × 240 = 3840)
RENDER_FPS=$(python3 -c "print(${MULT} * ${PLAYBACK_FPS})")

# pick latest run if not specified
if [ -z "$RUN" ]; then
    RUN=$(ls -1t "${LOG_DIR}" | head -1)
fi

DUR_ARG=""
[ -n "$DURATION" ] && DUR_ARG="--duration ${DURATION}"

# Discord output: same path but with _discord suffix before extension
DISCORD_OUT="${OUTPUT%.mp4}_discord.mp4"

echo "========================================================================"
echo "  Slow-motion recording pipeline"
echo "  Run          : ${RUN}"
echo "  Slowdown     : ${MULT}×  (render ${RENDER_FPS} Hz → playback ${PLAYBACK_FPS} FPS)"
echo "  Output       : ${OUTPUT}"
echo "  Discord(<10M): ${DISCORD_OUT}"
echo "========================================================================"
echo ""

# Run from repo root so that play.py resolves logs/ relative to the project
cd "${REPO_DIR}"

# ── Step 1: record trajectory ─────────────────────────────────────────────────
echo "▶ Step 1/3 — Recording trajectory (1 env, physics rate = 200 Hz) ..."
uv run --active python "${REPO_DIR}/scripts/rsl_rl/play.py" \
    --task Isaac-BallBalance-Go1-Play-v0 \
    --num_envs 1 \
    --load_run "${RUN}" \
    --record-traj "${TRAJ_FILE}" \
    --record-steps "${RECORD_STEPS}"

echo ""
echo "▶ Step 2/3 — Rendering interpolated slow-motion video ..."
uv run --active python "${REPO_DIR}/scripts/rsl_rl/render_slow.py" \
    --traj "${TRAJ_FILE}" \
    --render-fps "${RENDER_FPS}" \
    --playback-fps "${PLAYBACK_FPS}" \
    --env-idx "${ENV_IDX}" \
    --output "${OUTPUT}" \
    ${DUR_ARG}

# ── Step 3: re-encode for Discord (<10 MB) ────────────────────────────────────
echo ""
echo "▶ Step 3/3 — Re-encoding for Discord (<10 MB, 30 FPS) ..."

# Get the actual video duration, then calculate the bitrate needed for 9 MB
# (1 MB headroom below the 10 MB Discord limit).
DUR=$(ffprobe -v error \
    -show_entries format=duration \
    -of default=noprint_wrappers=1:nokey=1 \
    "${OUTPUT}")
KBPS=$(python3 -c "print(max(200, int(9*1024*1024*8/${DUR}/1000)))")

ffmpeg -y -i "${OUTPUT}" \
    -vf fps=30 \
    -c:v libx264 \
    -b:v "${KBPS}k" \
    -maxrate "$((KBPS * 2))k" \
    -bufsize "$((KBPS * 4))k" \
    -movflags +faststart \
    "${DISCORD_OUT}"

ACTUAL_MB=$(python3 -c "import os; print(f'{os.path.getsize(\"${DISCORD_OUT}\")/1024/1024:.1f}')")

echo ""
echo "========================================================================"
echo "  Done."
echo ""
echo "  Full-quality  : ${OUTPUT}  (${PLAYBACK_FPS} FPS, ${MULT}× slow)"
echo "  Discord(<10MB): ${DISCORD_OUT}  (${ACTUAL_MB} MB, 30 FPS, ${MULT}× slow)"
echo ""
echo "  Play:"
echo "    ffplay \"${OUTPUT}\""
echo "========================================================================"
