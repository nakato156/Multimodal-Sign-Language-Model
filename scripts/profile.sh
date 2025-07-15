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
    poe profile_nvidia --epochs 1 --batch_size 2
    
#ncu \
#    --target-processes all \
#    --set full \
#    --nvtx \
#    --nvtx-merge true \
#    --nvtx-range model_forward \
#  -o ../outputs/profile/ncu_profile_$(date +%Y%m%d_%H%M%S) \
#  poe profile_nvidia

#echo "Pytorch profiling has started"
#sudo env \
#    PATH="/home/giorgio6846/Code/Sign-AI/Sign-Multimodal-Language-Model/.conda/bin:$PATH" \
#    LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
#    poe profile_pytorch
#echo "Pytorch profiling has finished"