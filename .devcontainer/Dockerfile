ARG TAG
FROM ubuntu:${TAG}

# Ensure bash is available
SHELL ["/bin/bash", "-c"]

# Install system dependencies and NVIDIA CUDA repo
RUN apt-get update && apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    bash \
    coreutils \
    ca-certificates \
    curl \
    git \
    wget \
    build-essential \
    gnupg \
    zlib1g \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get upgrade -y && \
    rm -rf /var/lib/apt/lists/*