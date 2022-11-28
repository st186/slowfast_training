#!/bin/bash
echo "--------------- Model inference is Initiated ---------------"
cd /app

python3 model_execution.py \
    --s3Path="$2" --checkpoint_folder_name="$3" \
    --cfg cfg/C2D_8x8_R50.yaml \
    MODEL.NUM_CLASSES $1