#!/bin/bash

# Get the project root directory (parent of the scripts folder)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Save the current conda environment if one is active
CURRENT_CONDA_ENV=$CONDA_DEFAULT_ENV

# Ensure conda commands work in non-interactive shells
eval "$(conda shell.bash hook)"

# Activate the gigapath environment 
conda activate gigapath

# Set input and output paths relative to project root
INPUT_DIR="${PROJECT_ROOT}/data/processed/LungHist700/LungHist700_20x"
OUTPUT_DIR="${PROJECT_ROOT}/data/processed/LungHist700_embeddings/gigapath/"
OUTPUT_PREFIX="lunghist700_"
CACHE_DIR="${PROJECT_ROOT}/models/cache/gigapath"
USE_GLOBAL_POOL=false  # Set this to true or false as needed

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CACHE_DIR"

# Run the embedding extraction from project root
cd "$PROJECT_ROOT"
python histopatseg/gigapath_wrapper/cli/extract_dataset_embeddings.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size 6 \
  --num-workers 2 \
  --gpu-id 0 \
  --output-prefix "$OUTPUT_PREFIX" \
  --file-extensions ".jpg,.png" \
  --global-pool-tile-encoder "$USE_GLOBAL_POOL" \
  --model-cache-dir "$CACHE_DIR"

# Restore the previous conda environment if one was active
if [ -n "$CURRENT_CONDA_ENV" ]; then
  conda activate $CURRENT_CONDA_ENV
else
  conda deactivate
fi

echo "Embedding extraction complete!"