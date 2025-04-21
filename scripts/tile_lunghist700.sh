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

INPUT_DIR=$LUNGHIST700_RAW_PATH
MAGNIFICATION=20


python histopatseg/data/tile_LungHist700.py \
  --input-dir $INPUT_DIR \
  --output-dir data/processed/LungHist700_tiled/LungHist700_${MAGNIFICATION}x\
  --tile-size 224 \
  --desired-magnification $MAGNIFICATION \
  --generate-outlines \
  --num-workers 8