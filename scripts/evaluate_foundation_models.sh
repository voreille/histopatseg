#!/bin/bash
# filepath: /home/valentin/workspaces/histopatseg/scripts/evaluate_foundation_models.sh

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_ROOT"

# Required parameters
EMBEDDINGS_PATH="data/processed/embeddings/lunghist700_20x_UNI2_embeddings.npz"
METADATA_PATH="data/processed/LungHist700_tiled/LungHist700_20x/metadata.csv"

# Optional parameters (using defaults from the script)
TASK="superclass"
SUPERCLASS_TO_KEEP="aca"
AGGREGATION_METHOD="none"
N_SPLITS=4
CLASSIFIERS="histogram_cluster"

# Extract model name from embeddings filename
MODEL_NAME=$(basename "$EMBEDDINGS_PATH" | sed 's/_embeddings.npz//')

# Extract dataset name from metadata path
DATASET_NAME=$(echo "$METADATA_PATH" | grep -oP '(?<=data/processed/)[^/]+' || echo "dataset")
DATASET_NAME=${DATASET_NAME%%_*}  # Take only the first part before underscore

# Construct dynamic OUTPUT_DIR based on parameters
OUTPUT_DIR="reports/results/${DATASET_NAME}/UNI2/${MODEL_NAME}_task_${TASK}__${SUPERCLASS_TO_KEEP}__agg_${AGGREGATION_METHOD}__n_splits_${N_SPLITS}"

echo "Output will be saved to: ${OUTPUT_DIR}"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the evaluation script
python -m histopatseg.evaluation.evaluate_foundation_models \
  --embeddings-path "${EMBEDDINGS_PATH}" \
  --metadata-path "${METADATA_PATH}" \
  --task "${TASK}" \
  --superclass-to-keep "${SUPERCLASS_TO_KEEP}" \
  --aggregation-method "${AGGREGATION_METHOD}" \
  --output-dir "${OUTPUT_DIR}" \
  --n-splits "${N_SPLITS}" \
  --classifiers "${CLASSIFIERS}"