#!/bin/bash

# 1. Initialize Conda
source /home/giorgio6846/miniconda3/etc/profile.d/conda.sh

# 2. Activate your Sign env
conda activate Sign

# Create output directory if it doesn't exist
mkdir -p ../outputs/profile

# Get current timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")

echo "NVIDIA profiling has started"
# Run Poe task with nsys profiling
nsys profile \
  --output="../outputs/profile/nsys_profile_$(date +%Y%m%d_%H%M%S).qdrep" \
  --trace=cuda,osrt,nvtx \
  --gpu-metrics-devices=0 \
  --gpu-metrics-set=ad10x \
  --gpu-metrics-frequency=10000 \
  --enable=nvml_metrics,-i100 \
  --cuda-event-trace=true \
  poe profile_nvidia