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
MODEL_NAME=UNI2
TILE_SIZE=256


python histopatseg/data/compute_embeddings_lunghist700.py \
  --tiles-dir data/processed/LungHist700_tiled/LungHist700_${MAGNIFICATION}x_TS_${TILE_SIZE}/tiles \
  --output-file data/processed/embeddings/lunghist700_${MAGNIFICATION}x_${MODEL_NAME}_TS_${TILE_SIZE}_embeddings.npz \
  --model-name $MODEL_NAME \
  --gpu-id 0 \
  --batch-size 256 \
  --num-workers 24