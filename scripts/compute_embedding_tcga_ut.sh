#!/bin/bash

# Set the working directory to the project root
cd "$(dirname "$0")/.."

TCGA_UT_PATH=/mnt/nas7/data/Personal/Valentin/tcga-ut
MODEL_NAME=UNI2

# Run the script with the same arguments as in the launch configuration
python histopatseg/data/compute_embeddings_tcga_ut.py \
    --tcga-ut-dir $TCGA_UT_PATH \
    --output-h5 data/processed/embeddings/tcga_ut/${MODEL_NAME}_wo_normal.h5 \
    --magnification-key 0 \
    --model-name $MODEL_NAME \
    --gpu-id 0 \
    --batch-size 256 \
    --exclude-cancers "Lung_normal"

echo "Embeddings computation completed."
