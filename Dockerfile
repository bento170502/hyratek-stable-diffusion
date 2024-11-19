# Sử dụng base image từ NVIDIA CUDA
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

# Thiết lập biến môi trường để tránh prompt trong apt
ENV DEBIAN_FRONTEND=noninteractive

# Cập nhật hệ thống và cài đặt các công cụ cần thiết
RUN apt-get update && apt-get install -y \
    wget \
    bzip2 \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Tải và cài đặt Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Linux-x86_64.sh -O /tmp/anaconda.sh && \
    bash /tmp/anaconda.sh -b -p /opt/conda && \
    rm /tmp/anaconda.sh

# Thêm Anaconda vào PATH
ENV PATH="/opt/conda/bin:$PATH"

# Copy toàn bộ thư mục hiện tại vào container
WORKDIR /workspace
COPY . .

# Cài đặt môi trường từ file environment.yaml
RUN conda env create -f environment.yaml

# Kích hoạt môi trường khi container chạy
ENV CONDA_DEFAULT_ENV=my_env
ENV PATH="/opt/conda/envs/my_env/bin:$PATH"

# Đặt shell mặc định là bash và kích hoạt môi trường Conda
SHELL ["conda", "run", "-n", "my_env", "/bin/bash", "-c"]
