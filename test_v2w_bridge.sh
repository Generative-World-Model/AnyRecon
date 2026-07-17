#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash test_v2w_bridge.sh /path/to/video_to_world_scene [output_dir]
#
# Optional environment overrides:
#   BASE_DIR=/path/to/hf_models
#   TRAJECTORY_JSON=/path/to/0000_extend_transforms.json
#   NUM_COND_FRAMES=6
#   NUM_RENDER_FRAMES=35
#   CONF_PERCENTILE=40
#   MAX_POINTS=1000000
#   PIXEL_STRIDE=1
#   USE_ALL_STAGE0_POINTS=0
#   USE_RETRIEVAL=0
#   RUN_GENERATION=1
#   LORA_PATH=/path/to/AnyRecon_full_attention.ckpt
#   PYTHON_BIN=python

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash $0 /path/to/video_to_world_scene [output_dir]" >&2
    exit 2
fi

SCENE_ROOT="$(realpath "$1")"
OUTPUT_DIR="${2:-${SCENE_ROOT}/anyrecon_v2w_test}"
BASE_DIR="${BASE_DIR:-/home/dataset/data/z84450662/hf_models}"
PYTHON_BIN="${PYTHON_BIN:-python}"

STAGE0_NPZ="${SCENE_ROOT}/exports/npz/results.npz"
TRAJECTORY_JSON="${TRAJECTORY_JSON:-${SCENE_ROOT}/gs_video/0000_extend_transforms.json}"
WAN_MODEL_DIR="${WAN_MODEL_DIR:-${BASE_DIR}/Wan-AI/Wan2.1-I2V-14B-720P}"
LORA_PATH="${LORA_PATH:-${BASE_DIR}/Yutian10/AnyRecon/AnyRecon_full_attention.ckpt}"

NUM_COND_FRAMES="${NUM_COND_FRAMES:-6}"
NUM_RENDER_FRAMES="${NUM_RENDER_FRAMES:-35}"
CONF_PERCENTILE="${CONF_PERCENTILE:-40}"
MAX_POINTS="${MAX_POINTS:-1000000}"
PIXEL_STRIDE="${PIXEL_STRIDE:-1}"
USE_ALL_STAGE0_POINTS="${USE_ALL_STAGE0_POINTS:-0}"
USE_RETRIEVAL="${USE_RETRIEVAL:-0}"
RUN_GENERATION="${RUN_GENERATION:-1}"

if [[ ! -f "${STAGE0_NPZ}" ]]; then
    echo "Missing Stage 0 NPZ: ${STAGE0_NPZ}" >&2
    exit 1
fi
if [[ ! -f "${TRAJECTORY_JSON}" ]]; then
    echo "Missing trajectory JSON: ${TRAJECTORY_JSON}" >&2
    exit 1
fi
if [[ "${RUN_GENERATION}" == "1" && ! -d "${WAN_MODEL_DIR}" ]]; then
    echo "Missing Wan model directory: ${WAN_MODEL_DIR}" >&2
    exit 1
fi
if [[ "${RUN_GENERATION}" == "1" && ! -f "${LORA_PATH}" ]]; then
    echo "Missing AnyRecon LoRA: ${LORA_PATH}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Scene root:       ${SCENE_ROOT}"
echo "Stage 0 NPZ:      ${STAGE0_NPZ}"
echo "Trajectory JSON:  ${TRAJECTORY_JSON}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Condition frames: ${NUM_COND_FRAMES}"
echo "Render frames:    ${NUM_RENDER_FRAMES}"

BRIDGE_ARGS=(
    --stage0_npz "${STAGE0_NPZ}"
    --trajectory_json "${TRAJECTORY_JSON}"
    --output_dir "${OUTPUT_DIR}"
    --num_cond_frames "${NUM_COND_FRAMES}"
    --num_render_frames "${NUM_RENDER_FRAMES}"
    --conf_percentile "${CONF_PERCENTILE}"
    --max_points "${MAX_POINTS}"
    --pixel_stride "${PIXEL_STRIDE}"
)

if [[ "${USE_ALL_STAGE0_POINTS}" == "1" ]]; then
    BRIDGE_ARGS+=(--use_all_stage0_points)
fi
if [[ "${USE_RETRIEVAL}" == "1" ]]; then
    BRIDGE_ARGS+=(--use_retrieval)
fi

"${PYTHON_BIN}" run_v2w_stage0_bridge.py "${BRIDGE_ARGS[@]}"

mapfile -t CONDITION_VIDEOS < <(
    find "${OUTPUT_DIR}/condition" -maxdepth 1 -type f -name "*.mp4" | sort
)
mapfile -t MASK_VIDEOS < <(
    find "${OUTPUT_DIR}/mask" -maxdepth 1 -type f -name "*.mp4" | sort
)

if [[ ${#CONDITION_VIDEOS[@]} -eq 0 ]]; then
    echo "Bridge produced no condition videos." >&2
    exit 1
fi
if [[ ${#CONDITION_VIDEOS[@]} -ne ${#MASK_VIDEOS[@]} ]]; then
    echo "Condition/mask video count mismatch: ${#CONDITION_VIDEOS[@]} vs ${#MASK_VIDEOS[@]}." >&2
    exit 1
fi

echo "Generated ${#CONDITION_VIDEOS[@]} condition/mask video pair(s)."

if command -v ffprobe >/dev/null 2>&1; then
    for video in "${CONDITION_VIDEOS[@]}"; do
        frames="$(
            ffprobe -v error                 -count_frames                 -select_streams v:0                 -show_entries stream=nb_read_frames                 -of default=nokey=1:noprint_wrappers=1                 "${video}"
        )"
        echo "Condition video: ${video} (${frames} frames)"
    done
    for video in "${MASK_VIDEOS[@]}"; do
        frames="$(
            ffprobe -v error                 -count_frames                 -select_streams v:0                 -show_entries stream=nb_read_frames                 -of default=nokey=1:noprint_wrappers=1                 "${video}"
        )"
        echo "Mask video:      ${video} (${frames} frames)"
    done
else
    echo "ffprobe not found; skipping encoded-frame count checks."
fi

if [[ "${RUN_GENERATION}" != "1" ]]; then
    echo "RUN_GENERATION=${RUN_GENERATION}; bridge-only test completed."
    exit 0
fi

RESULT_DIR="${OUTPUT_DIR}/results"
mkdir -p "${RESULT_DIR}"

"${PYTHON_BIN}" run_AnyRecon.py     --root_dir "${OUTPUT_DIR}"     --output_dir "${RESULT_DIR}"     --wan_model_dir "${WAN_MODEL_DIR}"     --lora_path "${LORA_PATH}"

mapfile -t RESULT_VIDEOS < <(
    find "${RESULT_DIR}" -maxdepth 1 -type f -name "*.mp4" | sort
)
if [[ ${#RESULT_VIDEOS[@]} -eq 0 ]]; then
    echo "AnyRecon produced no result videos." >&2
    exit 1
fi

echo "AnyRecon generated ${#RESULT_VIDEOS[@]} result video(s):"
printf "  %s\n" "${RESULT_VIDEOS[@]}"
echo "End-to-end video_to_world Stage 0 bridge test completed."
