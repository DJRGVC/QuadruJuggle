#!/usr/bin/env bash
# Convert the most recent GNOME screencast to a ~9.9MB MP4 for Discord.
#
# Usage:
#   ./encode_for_discord.sh              # auto-picks latest screencast
#   ./encode_for_discord.sh /path/to/file.webm

set -e

SCREENCAST_DIR="${HOME}/Videos/Screencasts"
OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/myrecordings"
TARGET_MB=9.9
# Use 9.4MB as the bitrate target — container/muxer overhead pushes actual size ~5% higher
TARGET_BYTES=$(echo "9.4 * 1024 * 1024" | bc | cut -d. -f1)

mkdir -p "${OUT_DIR}"

# ── Pick input file ──────────────────────────────────────────────────────────
if [ -n "$1" ]; then
    INPUT="$1"
else
    INPUT=$(ls -1t "${SCREENCAST_DIR}"/*.webm "${SCREENCAST_DIR}"/*.mkv \
                   "${SCREENCAST_DIR}"/*.mp4 2>/dev/null | head -1)
    if [ -z "${INPUT}" ]; then
        echo "No screencasts found in ${SCREENCAST_DIR}"
        exit 1
    fi
fi

echo "Input:  ${INPUT}"

# ── Get duration ─────────────────────────────────────────────────────────────
DURATION=$(ffprobe -v error -show_entries format=duration \
           -of default=noprint_wrappers=1:nokey=1 "${INPUT}")
echo "Duration: ${DURATION}s"

# ── Check for audio stream ───────────────────────────────────────────────────
HAS_AUDIO=$(ffprobe -v error -select_streams a \
            -show_entries stream=codec_type \
            -of default=noprint_wrappers=1:nokey=1 "${INPUT}")

# ── Compute bitrate for target file size ─────────────────────────────────────
# total_bitrate (kbps) = (target_bytes * 8) / (duration * 1000)
TOTAL_KBPS=$(echo "scale=0; (${TARGET_BYTES} * 8) / (${DURATION} * 1000)" | bc)
if [ -n "${HAS_AUDIO}" ]; then
    AUDIO_KBPS=64
    VIDEO_KBPS=$((TOTAL_KBPS - AUDIO_KBPS))
    echo "Target bitrate: ${VIDEO_KBPS}kbps video + ${AUDIO_KBPS}kbps audio"
else
    AUDIO_KBPS=0
    VIDEO_KBPS=${TOTAL_KBPS}
    echo "Target bitrate: ${VIDEO_KBPS}kbps video (no audio stream)"
fi

if [ "${VIDEO_KBPS}" -le 0 ]; then
    echo "Error: video is too long for target size."
    exit 1
fi

# ── Output filename ───────────────────────────────────────────────────────────
BASENAME=$(basename "${INPUT}")
STEM="${BASENAME%.*}"
OUTPUT="${OUT_DIR}/${STEM}.mp4"

# ── Scale filter: fps=30 normalises VFR→CFR so pass1/pass2 see identical frames
VF="fps=30,scale=trunc(iw/2)*2:trunc(ih/2)*2"

# ── Audio flags ───────────────────────────────────────────────────────────────
if [ -n "${HAS_AUDIO}" ]; then
    AUDIO_FLAGS="-c:a aac -b:a ${AUDIO_KBPS}k"
else
    AUDIO_FLAGS="-an"
fi

# ── Two-pass encode ───────────────────────────────────────────────────────────
echo ""
echo "Pass 1/2..."
ffmpeg -y -fflags +genpts -i "${INPUT}" \
    -vf "${VF}" \
    -c:v libx264 -preset slow -b:v "${VIDEO_KBPS}k" \
    -pass 1 -an \
    -f null /dev/null 2>&1 | grep -E "frame=|time=|Error" || true

echo "Pass 2/2..."
ffmpeg -y -fflags +genpts -i "${INPUT}" \
    -vf "${VF}" \
    -c:v libx264 -preset slow -b:v "${VIDEO_KBPS}k" \
    -pass 2 \
    ${AUDIO_FLAGS} \
    -movflags +faststart \
    "${OUTPUT}" 2>&1 | grep -E "frame=|time=|Error" || true

# Clean up
rm -f ffmpeg2pass-0.log ffmpeg2pass-0.log.mbtree

# ── Report ────────────────────────────────────────────────────────────────────
ACTUAL_BYTES=$(stat -c%s "${OUTPUT}")
ACTUAL_MB=$(echo "scale=2; ${ACTUAL_BYTES} / 1048576" | bc)
echo ""
echo "Done!"
echo "Output: ${OUTPUT}"
echo "Size:   ${ACTUAL_MB}MB (target: ${TARGET_MB}MB)"
