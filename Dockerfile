# ==============================================================================
# ChatBotColab — Docker Image
# ==============================================================================
# Build:  docker build -t chatbot-colab .
# Run:    docker run --gpus all -p 8000:8000 -p 8501:8501 --env-file .env chatbot-colab
# ==============================================================================

# --- Stage 1: CUDA Base ---
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS base

# Tránh interactive prompts khi cài đặt
ENV DEBIAN_FRONTEND=noninteractive

# Cài Python 3.11 + các tools cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    python3.11-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# --- Stage 2: Dependencies ---
FROM base AS deps

WORKDIR /app

# Copy requirements trước để tận dụng Docker cache
COPY requirements.txt .

# Cài PyTorch với CUDA 12.4 trước, sau đó cài các deps còn lại
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 3: Application ---
FROM deps AS app

WORKDIR /app

# Copy mã nguồn
COPY backend.py .
COPY chatbot.py .
COPY startup.sh .

# Tạo thư mục data
RUN mkdir -p /app/data/vector_db /app/data/documents /app/data/uploads

# Quyền chạy cho startup script
RUN chmod +x startup.sh

# Expose ports: 8000 (FastAPI), 8501 (Streamlit)
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entry point
CMD ["./startup.sh"]
