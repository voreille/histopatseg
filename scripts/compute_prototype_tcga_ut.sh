#!/bin/bash

# Set the working directory to the project root
cd "$(dirname "$0")/.."

# Run the script with the same arguments as in the launch configuration
python histopatseg/data/compute_prototype_tcga_ut.py \
    --tcga-ut-dir data/tcga-ut \
    --n-wsi 32 \
    --output-h5 data/processed/prototypes_tcga_ut/uni2_prototypes__n_wsi_32.h5 \
    --magnification-key 0 \
    --model-name UNI2 \
    --gpu-id 0 \
    --batch-size 256 \
    --random-state 42

echo "Prototype computation completed."