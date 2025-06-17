#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p ../outputs/profile

# Get current timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")

# Name for the output file using the timestamp
output_file="../outputs/profile/nsys_profile_$timestamp.qdrep"

echo "NVIDIA profiling has started"
# Run Poe task with nsys profiling
sudo env \
    PATH="/home/giorgio6846/Code/Sign-AI/Sign-Multimodal-Language-Model/.conda/bin:$PATH" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    nsys profile \
    --output="${output_file}" \
    --trace=cuda,osrt,nvtx \
    --gpu-metrics-devices=0 \
    --gpu-metrics-set=ad10x \
    --gpu-metrics-frequency=10000 \
    --enable=nvml_metrics,-i100 \
    poe profile_nvidia
    
# Confirmation
echo "NVIDIA profiling saved to $output_file"

echo "Pytorch profiling has started"
sudo env \
    PATH="/home/giorgio6846/Code/Sign-AI/Sign-Multimodal-Language-Model/.conda/bin:$PATH" \
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
    poe profile_pytorch
echo "Pytorch profiling has finished"