#!/bin/bash
set -x

# source activate /mnt/afs/wangkaibin/.conda/envs/llava

WORK_DIR="/mnt/afs/wangkaibin/VLMEvalKit"
cd "$WORK_DIR"

DATA="Video-MME"
# DATA="MLVU_MCQ"
# DATA="LongVideoBench"

MODEL="llava_video_qwen2_7b"

NFRAME=64

SCRIPT_PATH=$(readlink -f "$0")
EXPR_NAME=$(basename "$0" | sed 's/\.[^.]*$//')
OUTPUT_DIR="$WORK_DIR/outputs/$EXPR_NAME"

mkdir -p "$OUTPUT_DIR"
cp "$SCRIPT_PATH" "$OUTPUT_DIR/run.sh"

python run.py \
    --model "$MODEL" \
    --data "$DATA" \
    --work-dir "$OUTPUT_DIR" \
    --nframe "$NFRAME" \
    --reuse \
    --use_sketch \
    --sketch_type "dummy" \
    --num_sketch_tokens 728 \
    2>&1 | tee "$OUTPUT_DIR/out.log"
