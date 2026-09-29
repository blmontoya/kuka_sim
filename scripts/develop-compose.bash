#!/bin/bash

echo "🔍 Checking for available NVIDIA GPUs in Docker passthrough..."

export BASE_IMAGE=nvidia/cuda:12.6.3-base-ubuntu22.04

if docker run --rm --gpus all $BASE_IMAGE nvidia-smi > /dev/null 2>&1; then
  echo "✅ NVIDIA GPUs detected. Using CUDA-enabled base image."
  export GPU_COUNT=all
else
  echo "⚠️  No NVIDIA GPUs detected in Docker passthrough. Falling back to CPU-only base image."
  echo "If you do have GPUs, please consult the guide at https://github.com/garylvov/dev_env/tree/main/setup_scripts/nvidia"
  echo "... for proper configuration."
  export BASE_IMAGE=ubuntu:22.04
  export GPU_COUNT=0
fi

echo "📦 Using base image: $BASE_IMAGE"
echo "🎯 GPU count setting: $GPU_COUNT"

echo "⚠️ Disabling access control for X server. This may have security implications."
echo "⚠️ When done with use, run docker compose down && sudo xhost -"
sudo xhost +local:docker

echo "🚀 Starting Docker Compose build..."
echo "This may take a minute, grab a snack ;)"
docker compose build
docker compose up -d
docker exec -it kuka_sim_dev bash
