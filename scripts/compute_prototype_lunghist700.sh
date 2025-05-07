#!/bin/bash

# Set the working directory to the project root
cd "$(dirname "$0")/.."

LUNGHIST700_PATH=data/processed/LungHist700/LungHist700_20x

# Run the script with the same arguments as in the launch configuration
python histopatseg/data/compute_prototype_LungHist700.py \
    --lunghist700-dir $LUNGHIST700_PATH \
    --output-h5 data/processed/prototypes_lunghist700/uni2_prototypes_precentercrop.h5 \
    --model-name UNI2 \
    --gpu-id 0 \
    --batch-size 8 \
    --num-workers 8 
    # --exclude-cancers Lung_normal \

echo "Prototype computation completed."