#!/bin/bash

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_ROOT"

# Source the .env file to get environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
  source "$PROJECT_ROOT/.env"
else
  echo "Error: .env file not found in $PROJECT_ROOT"
  exit 1
fi

MAGNIFICATION=20
MODEL_NAME=H-optimus-1
TILE_SIZE=1204

# TILES_DIR=data/processed/LungHist700/LungHist700_${MAGNIFICATION}x

TILES_DIR=data/raw/LungHist700
# OUTPUT_FILE=data/processed/embeddings/lunghist700_${MAGNIFICATION}x_${MODEL_NAME}_centercrop_embeddings.npz
OUTPUT_FILE=data/processed/embeddings/lunghist700_raw_${MODEL_NAME}_centercrop_ts_${TILE_SIZE}_embeddings.npz


python histopatseg/data/compute_embeddings_lunghist700.py \
  --tiles-dir $TILES_DIR \
  --output-file $OUTPUT_FILE\
  --model-name $MODEL_NAME \
  --gpu-id 0 \
  --batch-size 256 \
  --num-workers 24 \
  --tile-size $TILE_SIZE \
  --is-raw-data