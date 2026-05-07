#!/bin/bash
# ==============================================================================
# ChatBotColab — Startup Script (Vertex AI Edition)
# ==============================================================================
# Khởi chạy cả Backend (FastAPI) và Frontend (Streamlit) trong cùng container.
# Không cần GPU — LLM inference qua Vertex AI API.
# ==============================================================================

set -e

# Graceful shutdown: forward signals đến backend process
cleanup() {
    echo "🛑 Đang tắt hệ thống..."
    kill $BACKEND_PID 2>/dev/null
    wait $BACKEND_PID 2>/dev/null
    echo "✅ Đã tắt sạch."
    exit 0
}
trap cleanup SIGTERM SIGINT

echo ""
echo "============================================================"
echo "🚀 ChatBotColab — Khởi động hệ thống (Vertex AI Edition)"
echo "============================================================"
echo ""

# --- 1. Khởi chạy Backend (FastAPI) ở background ---
echo "🔧 [1/2] Khởi chạy Backend (FastAPI) trên port ${BACKEND_PORT:-8000}..."
python3 backend.py &
BACKEND_PID=$!

# Đợi backend sẵn sàng (tối đa 2 phút — nhanh hơn nhiều vì không load model)
echo "⏳ Đang đợi Backend sẵn sàng..."
MAX_WAIT=120
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf http://localhost:${BACKEND_PORT:-8000}/health > /dev/null 2>&1; then
        echo "✅ Backend đã sẵn sàng!"
        break
    fi
    sleep 3
    WAITED=$((WAITED + 3))
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
