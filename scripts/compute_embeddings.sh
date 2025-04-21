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


python histopatseg/data/compute_embeddings_lunghist700.py \
  --tiles-dir data/processed/LungHist700_tiled/LungHist700_${MAGNIFICATION}x/tiles \
  --output-file data/processed/embeddings/lunghist700_${MAGNIFICATION}x_${MODEL_NAME}_embeddings.npz \
  --model-name $MODEL_NAME \
  --gpu-id 0 \
  --batch-size 256 \
  --num-workers 24