#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p ../outputs/profile

# Get current timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")

# Name for the output file using the timestamp
output_file="../outputs/profile/nsys_profile_$timestamp.qdrep"

echo "NVIDIA profiling has started"
# Run Poe task with nsys profiling
nsys profile \
    --output="${output_base}" \
    --trace=cuda,osrt,nvtx \
    --gpu-metrics-devices=0 \
    --gpu-metrics-set=tu10x-gfxt \
    --gpu-metrics-frequency=10000 \
    --enable=nvml_metrics,-i100 \
    poe profile_nvidia
    
# Confirmation
echo "NVIDIA profiling saved to $output_file"

echo "Pytorch profiling has started"
poe profile_pytorch
echo "Pytorch profiling has finished"