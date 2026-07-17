#!/usr/bin/env bash
set -e

# ============================================================
# Only edit these paths.
# ============================================================
ANYRECON_DIR="/home/dataset/data/z84450662/AnyRecon"
PYTHON_BIN="/home/ma-user/anaconda3/envs/anyrecon/bin/python"

STAGE0_NPZ="/path/to/video_to_world_scene/exports/npz/results_pi3_raw.npz"
TRAJECTORY_JSON="/path/to/video_to_world_scene/gs_video/0000_extend_transforms.json"
OUTPUT_DIR="/home/dataset/data/z84450662/AnyRecon/example/v2w_bridge_test"

# ============================================================
# Test settings.
# ============================================================
NUM_COND_FRAMES=6
NUM_RENDER_FRAMES=35
CONF_THRESHOLD=0.1
MAX_POINTS=0
PIXEL_STRIDE=1

cd "$ANYRECON_DIR"

"$PYTHON_BIN" "$ANYRECON_DIR/run_v2w_stage0_bridge.py" \
    --stage0_npz "$STAGE0_NPZ" \
    --trajectory_json "$TRAJECTORY_JSON" \
    --output_dir "$OUTPUT_DIR" \
    --num_cond_frames "$NUM_COND_FRAMES" \
    --num_render_frames "$NUM_RENDER_FRAMES" \
    --conf_threshold "$CONF_THRESHOLD" \
    --max_points "$MAX_POINTS" \
    --pixel_stride "$PIXEL_STRIDE"

echo "Finished. Outputs are in: $OUTPUT_DIR"
