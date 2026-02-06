# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TOKENIZERS_PARALLELISM=false

# Remove conda ffmpeg (missing libx264) and install system ffmpeg with x264 support
RUN conda uninstall -y ffmpeg || true
ENV PATH=/usr/bin:$PATH
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavcodec-extra \
    libx264-dev \
    x264 \
    git \
    build-essential \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml .

# Sanitize pyproject.toml to avoid duplicate [build-system] sections that break pip parsing.
# Keep only the first [build-system] block if duplicates exist.
RUN awk 'BEGIN{keep=1} /^\[build-system\]/{ if(++seen>1) keep=0 } { if(keep) print }' pyproject.toml > pyproject.tmp && mv pyproject.tmp pyproject.toml

# Install Python dependencies
# Pre-install pinned pandas and transformers; torch is already in the base image.
# Skip PyG pre-install since wheels may not be available; let it install via pyproject.toml.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pandas==2.1.2 transformers==4.37.0 && \
    pip install --no-cache-dir .

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "vidsum_gnn.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# # Base image: PyTorch with CUDA 12.1 and cuDNN
# FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# # --- Environment Variables ---
# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1 \
#     DEBIAN_FRONTEND=noninteractive \
#     # Force GPU-only execution
#     CUDA_VISIBLE_DEVICES=0 \
#     # Disable CPU fallback for torch
#     TORCH_USE_CUDA_DSA=1 \
#     # RTX 3070 compute capability
#     TORCH_CUDA_ARCH_LIST="8.6" \
#     # Avoid CPU-only kernels
#     NVIDIA_VISIBLE_DEVICES=all \
#     NVIDIA_DRIVER_CAPABILITIES=compute,utility

# # --- System dependencies ---
# RUN apt-get update && apt-get install -y \
#     ffmpeg \
#     git \
#     build-essential \
#     libsndfile1 \
#     && rm -rf /var/lib/apt/lists/*

# # --- Working directory ---
# WORKDIR /app

# # --- Copy pyproject ---
# COPY pyproject.toml .

# # --- Install Python dependencies ---
# RUN pip install --upgrade pip

# # Install PyTorch Geometric (GPU wheels)
# RUN pip install torch-scatter torch-sparse torch-geometric \
#     --extra-index-url https://data.pyg.org/whl/torch-2.1.0+cu121.html

# # Install rest of dependencies
# RUN pip install --no-cache-dir .

# # --- Copy application code ---
# COPY . .

# # Expose port
# EXPOSE 8000

# # --- Start API ---
# CMD ["uvicorn", "vidsum_gnn.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
