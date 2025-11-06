#!/bin/bash

source activate /mnt/afs/wangkaibin/.conda/envs/oryx

WORK_DIR="/mnt/afs/wangkaibin/VLMEvalKit"
cd "$WORK_DIR"

# DATA="Video-MME"
# DATA="MLVU_MCQ"
# DATA="LongVideoBench"

MODEL="baseline_oryx"

read -r -d '' model_config <<EOF
nframe: 256
EOF

export LOWRES_RESIZE=384x32
export VIDEO_RESIZE="0x64"
export HIGHRES_BASE="0x32"
export MAXRES=1536
export MINRES=0
export VIDEO_MAXRES=480
export VIDEO_MINRES=288

python /mnt/afs/wangkaibin/VLMEvalKit/test.py
