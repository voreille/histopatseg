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
MAGNIFICATION=10


python histopatseg/data/resize_lunghist700.py \
  --raw-data-path $INPUT_DIR \
  --output-dir data/processed/LungHist700/LungHist700_${MAGNIFICATION}x \
  --magnification $MAGNIFICATION \
  --num-workers 24 \