#!/bin/bash
# ==============================================================================
# ChatBotColab — Startup Script
# ==============================================================================
# Khởi chạy cả Backend (FastAPI) và Frontend (Streamlit) trong cùng container.
# ==============================================================================

set -e

echo ""
echo "============================================================"
echo "🚀 ChatBotColab — Khởi động hệ thống..."
echo "============================================================"
echo ""

# --- 1. Khởi chạy Backend (FastAPI) ở background ---
echo "🔧 [1/2] Khởi chạy Backend (FastAPI) trên port ${BACKEND_PORT:-8000}..."
python3 backend.py &
BACKEND_PID=$!

# Đợi backend sẵn sàng (tối đa 10 phút vì model loading rất lâu)
echo "⏳ Đang đợi Backend sẵn sàng (model loading có thể mất 5-10 phút)..."
MAX_WAIT=600
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf http://localhost:${BACKEND_PORT:-8000}/health > /dev/null 2>&1; then
        echo "✅ Backend đã sẵn sàng!"
        break
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    echo "   ... đã đợi ${WAITED}s / ${MAX_WAIT}s"
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo "⚠️ Backend chưa sẵn sàng sau ${MAX_WAIT}s. Vẫn khởi chạy Frontend..."
fi

# --- 2. Khởi chạy Frontend (Streamlit) ở foreground ---
echo ""
echo "🎨 [2/2] Khởi chạy Frontend (Streamlit) trên port ${STREAMLIT_PORT:-8501}..."
echo ""
exec streamlit run chatbot.py \
    --server.port ${STREAMLIT_PORT:-8501} \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false
