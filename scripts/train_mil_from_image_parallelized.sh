#!/bin/bash
# filename: train_mil_cv.sh

# Configuration
MODEL_NAME="UNI2"           # Options: UNI2, DINOv2, etc.
SUPERCLASS="aca"            # Options: aca, scc, all
MAX_EPOCHS=50
BATCH_SIZE=32
LR=0.001
DROPOUT=0.2
WEIGHT_DECAY=0.0001
NUM_WORKERS=4               # Adjust based on your CPU
N_SPLITS=4
RANDOM_STATE=42
GPU_ID=0                    # Change based on available GPU

# Paths
IMAGES_DIR="/home/valentin/workspaces/histopatseg/data/processed/LungHist700_tiles"
BASE_OUTPUT_DIR="models/mil"

# Create output directory
mkdir -p $BASE_OUTPUT_DIR

# Generate timestamp for this run (shared across folds)
TIMESTAMP=$(date +%Y%m%d_%H%M)
RUN_NAME="${MODEL_NAME}_${SUPERCLASS}_${TIMESTAMP}"

echo "====================================================================="
echo "Starting MIL training with ${MODEL_NAME} on ${SUPERCLASS} dataset"
echo "Running ${N_SPLITS}-fold cross-validation"
echo "====================================================================="

# Train models for all folds
for FOLD in $(seq 0 $(($N_SPLITS-1))); do
    echo "====================================================================="
    echo "Training fold $FOLD of $N_SPLITS"
    echo "====================================================================="
    
    # Output path for this fold's model
    OUTPUT_PATH="${BASE_OUTPUT_DIR}/${RUN_NAME}_fold${FOLD}.ckpt"
    
    # Custom experiment name for this fold
    EXPERIMENT_NAME="${RUN_NAME}_fold${FOLD}"
    
    echo "Model will be saved to: $OUTPUT_PATH"
    echo "Starting training..."
    
    # Run the training
    python histopatseg/training/train_mil_from_tiles_new.py \
        --output-path "$OUTPUT_PATH" \
        --model-name "$MODEL_NAME" \
        --images-dir "$IMAGES_DIR" \
        --max-epochs $MAX_EPOCHS \
        --gpu-id $GPU_ID \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --dropout $DROPOUT \
        --weight-decay $WEIGHT_DECAY \
        --num-workers $NUM_WORKERS \
        --superclass "$SUPERCLASS" \
        --fold $FOLD \
        --n-splits $N_SPLITS \
        --random-state $RANDOM_STATE \
        --experiment-name "$EXPERIMENT_NAME"
    
    TRAINING_STATUS=$?
    
    if [ $TRAINING_STATUS -eq 0 ]; then
        echo "Successfully completed training for fold $FOLD"
    else
        echo "ERROR: Training failed for fold $FOLD with exit code $TRAINING_STATUS"
        echo "Continuing with next fold..."
    fi
    
    echo "====================================================================="
    echo "Completed fold $FOLD"
    echo "====================================================================="
done

echo "====================================================================="
echo "Cross-validation training complete!"
echo "All models saved to $BASE_OUTPUT_DIR"
echo "====================================================================="