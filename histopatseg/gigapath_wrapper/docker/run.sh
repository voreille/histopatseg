#!/bin/bash

# Run the docker container with necessary volume mounts
docker run --gpus all \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  gigapath-cli \
  --input-image /app/input/your_image.png \
  --output-dir /app/output \
  --tile-model-name "hf_hub:prov-gigapath/prov-gigapath" \
  --slide-model-path /app/input/slide_model.pth \
  --batch-size 32 \
  --device cuda