#!/bin/bash
set -e

# Build Docker image
# docker build -t dataset-dm .

# Run Docker container
docker run -it \
    --gpus all \
    -v "$(pwd)":/app \
    -v /extra_space2/romanus/p:/app/data \
    cnstark/pytorch:1.9.1-py3.9.12-cuda11.1.1-ubuntu20.04
    # dataset-dm