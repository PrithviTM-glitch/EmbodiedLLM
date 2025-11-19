FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

# Install essentials
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Create a workspace
RUN mkdir /workspace
WORKDIR /workspace

# Defaults: Only install pip, you install requirements as you need
