#!/bin/bash

WORK_DIR="."
cd "$WORK_DIR"

MODEL="t3s_fastv_mcq"

read -r -d '' model_config <<EOF
nframe: 256
agg_layer: 2
attention_rank_ratio: 0.5
EOF

SCRIPT_PATH=$(readlink -f "$0")
EXPR_NAME=$(basename "$0" | sed 's/\.[^.]*$//')
OUTPUT_DIR="$WORK_DIR/outputs/${EXPR_NAME}"

mkdir -p "$OUTPUT_DIR"
cp "$SCRIPT_PATH" "$OUTPUT_DIR/run.sh"


CUDA_VISIBLE_DEVICES=0 \
torchrun \
    --nproc_per_node=1 \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_port=$MASTER_PORT \
    --master_addr=$MASTER_ADDR \
run.py \
    --model "$MODEL" \
    --data "Video-MME" "MLVU_MCQ" "LongVideoBench" \
    --work-dir "$OUTPUT_DIR" \
    --reuse \
    --model_config "$model_config" \
    2>&1 | tee "$OUTPUT_DIR/out.log"
