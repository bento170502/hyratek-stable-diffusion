# docker/Dockerfile
FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /hyratek-stable-diffusion

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    bzip2 \
    ca-certificates \
    curl \
    python3-dev \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Add conda to path
ENV PATH=/opt/conda/bin:$PATH

# Copy environment file
COPY environment.yaml .

# Create conda environment and clean up
RUN conda env create -f environment.yaml && \
    conda clean -a -y && \
    conda run -n ldm pip cache purge

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "ldm", "/bin/bash", "-c"]

# Copy the package files
COPY . /HyratekSD/


# Make directory for models and outputs
RUN mkdir -p /HyratekSD/generated_images

# Set entrypoint script
RUN chmod +x /HyratekSD/entrypoint.sh

# Set environment variables

# # Activate conda environment by default
# RUN echo "conda activate ldm" >> ~/.bashrc

# ENTRYPOINT ["/hyratek-stable-diffusion/docker/entrypoint.sh"]