#!/bin/bash

# Load Hugging Face token from project root .env
ROOT_DIR=$(git rev-parse --show-toplevel)
source "$ROOT_DIR/.env"

# Check if token is available
if [ -z "$HUGGING_FACE_TOKEN" ]; then
  echo "HUGGING_FACE_TOKEN not set in .env"
  exit 1
fi

# Build the Docker image from the project root for proper COPY context
cd "$ROOT_DIR"

# Build the Docker image with the Hugging Face token
docker build \
  --build-arg HF_TOKEN=$HUGGING_FACE_TOKEN \
  -f histopatseg/gigapath/docker/Dockerfile \
  -t gigapath-cli .

echo "Docker image 'gigapath-cli' built successfully"