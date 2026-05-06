# ==============================================================================
# ChatBotColab — Docker Image (Vertex AI Edition)
# ==============================================================================
# No GPU required! Much lighter than the CUDA-based image.
#
# Build:  docker build -t chatbot-colab .
# Run:    docker run -p 8000:8000 -p 8501:8501 --env-file .env chatbot-colab
# ==============================================================================

FROM python:3.11-slim AS base

# Cài các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements trước để tận dụng Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entry point
CMD ["./startup.sh"]
