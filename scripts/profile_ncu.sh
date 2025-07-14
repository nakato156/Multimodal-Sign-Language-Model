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
ncu --set full \
    --target-processes all \
    -o "../outputs/profile/nsys_profile_$(date +%Y%m%d_%H%M%S).ncu-rep" \
    --call-stack \
    poe profile_nvidia --epochs 1 --batch_size 2