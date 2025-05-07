#!/bin/bash

# Set the working directory to the project root
cd "$(dirname "$0")/.."

TCGA_UT_PATH=/mnt/nas7/data/Personal/Valentin/tcga-ut
N_WSI=32

# Run the script with the same arguments as in the launch configuration
python histopatseg/data/compute_prototype_tcga_ut.py \
    --tcga-ut-dir $TCGA_UT_PATH \
    --n-wsi $N_WSI \
    --output-h5 data/processed/prototypes_tcga_ut/uni2_prototypes__n_wsi_${N_WSI}_precentercrop__no_ft.h5 \
    --magnification-key 0 \
    --model-name UNI2 \
    --gpu-id 0 \
    --batch-size 256 \
    --random-state 42 \
    --pre-centercrop-size 224 \
    --feature-transformation "none"
    # --exclude-cancers Lung_normal \

echo "Prototype computation completed."