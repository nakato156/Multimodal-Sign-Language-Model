#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p ../outputs/profile

# Get current timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")

# Name for the output file using the timestamp
output_file="../outputs/profile/nsys_profile_$timestamp.qdrep"

# Run Poe task with nsys profiling
nsys profile --output="$output_file" poe profile

# Confirmation
echo "NVIDIA profiling saved to $output_file"