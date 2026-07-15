# 🤖 ChatBotColab

> **AI Chatbot thông minh kết hợp Tree of Thought + Graph RAG + Vector RAG — powered by Gemini API.**
> Upload tài liệu → Chatbot tự học → Trả lời câu hỏi dựa trên nội dung tài liệu.

<p align="center">
  <a href="#-tổng-quan">Tiếng Việt</a> •
  <a href="#-overview-english">English</a>
</p>

---

# 🇻🇳 Tiếng Việt

## 📖 Tổng Quan

**ChatBotColab** là trợ lý AI có khả năng:

1. 📄 **Đọc hiểu tài liệu** — Upload file PDF/TXT/DOCX, chatbot tự phân tích và lưu trữ
2. 🕸️ **Xây đồ thị tri thức** — Tự động trích xuất thực thể và mối quan hệ (ai → làm gì → cho ai)
3. 🌳 **Suy luận sâu** — Dùng kỹ thuật Tree of Thought: nghĩ nhiều hướng → chấm điểm → chọn hướng tốt nhất
4. ✅ **Tự kiểm tra** — LLM tự đọc lại câu trả lời, loại bỏ thông tin bịa đặt

### Tại sao dự án này đặc biệt?

| Tính năng | ChatBotColab | Chatbot thông thường |
|---|---|---|
| Nguồn kiến thức | 📄 Tài liệu của **bạn** | 🌐 Dữ liệu huấn luyện chung |
| Tìm kiếm | 🔍 Vector Search **+ Graph Search** | 🔍 Chỉ Vector Search |
| Suy luận | 🌳 Tree of Thought (4 hướng → chọn best) | 💬 Trả lời thẳng |
| Kiểm tra | ✅ Self-Reflection (tự sửa lỗi) | ❌ Không |
| Bịa đặt | 🛡️ Giảm mạnh (có ngữ cảnh + tự kiểm tra) | ⚠️ Hay bịa |

---

## 🏗️ Kiến Trúc Hệ Thống

```
                        ┌─────────────────────────────────────────────────────┐
                        │              Docker Container                       │
 ┌──────────┐           │                                                     │
 │ 🌐 Trình  │  :8501   │  ┌──────────┐         ┌─────────────────────────┐  │
 │  duyệt    │◄────────►│  │chatbot.py│  HTTP   │  backend.py             │  │
 │ (bất kỳ)  │          │  │Streamlit │ ──────► │  FastAPI :8000          │  │
 └──────────┘  :8000   │  │  UI      │         │                         │  │
               ◄────────►│  └──────────┘         │  ┌───────────────────┐ │  │
                        │                        │  │SmartKnowledgeBase│ │  │
                        │                        │  │ • Vector DB      │ │  │
                        │                        │  │ • Graph DB       │ │  │
                        │                        │  │ • Reranker (GPU) │ │  │
                        │                        │  └───────────────────┘ │  │
                        │                        │  ┌───────────────────┐ │  │
                        │                        │  │ReasoningAgent    │ │  │
                        │                        │  │ • Gemini API ☁️  │ │  │
                        │                        │  │ • Tree of Thought│ │  │
                        │                        │  │ • Self-Reflection│ │  │
                        │                        │  └───────────────────┘ │  │
                        │                        └─────────────────────────┘  │
                        │                                                     │
                        │  📂 Volume: ./data/                                 │
                        │     ├── vector_db/  (ChromaDB)                      │
                        │     ├── graph.json  (Knowledge Graph)               │
                        │     └── uploads/    (Tài liệu upload)               │
                        └─────────────────────────────────────────────────────┘
```

---

## 🛠️ Công Nghệ Sử Dụng

| Tầng | Công nghệ | Vai trò | Ghi chú |
|---|---|---|---|
| **🧠 LLM** | Gemini 2.0 Flash | Suy luận, sinh câu trả lời | Gọi qua API — không cần GPU |
| **📊 Embedding** | `vietnamese-bi-encoder` (BKAI) | Chuyển văn bản → vector | Tối ưu cho tiếng Việt |
| **🔄 Reranker** | `bge-reranker-v2-m3` (BAAI) | Chấm điểm lại kết quả tìm kiếm | Chạy trên GPU nếu có |
| **💾 Vector DB** | ChromaDB | Lưu và tìm kiếm vector | Persistent trên disk |
| **🕸️ Graph DB** | NetworkX + JSON | Đồ thị tri thức (thực thể + quan hệ) | Lưu dạng `graph.json` |
| **✂️ Chunking** | LangChain `RecursiveCharacterTextSplitter` | Chia nhỏ tài liệu thông minh | Chunk 1000 chars, overlap 200 |
| **📄 Loaders** | PyPDF, TextLoader, Docx2txt | Đọc file PDF/TXT/DOCX | — |
| **⚡ API** | FastAPI + Uvicorn | REST API server | Async + Threadpool |
| **🎨 Frontend** | Streamlit | Giao diện chat + upload | Session-based |
| **🐳 Deploy** | Docker + Docker Compose | Đóng gói + triển khai | Pre-cache models |
| **☁️ Runtime** | Google AI Studio API Key | Gemini inference trên cloud | **Không cần GPU cho LLM** |

---

## 📁 Cấu Trúc Dự Án

```
chatbotwithcolab/
│
├── 🧠 backend.py           # Server API chính — RAG + AI + Graph DB
├── 🎨 chatbot.py           # Giao diện chat Streamlit
├── 📜 startup.sh           # Script khởi động (chạy backend → frontend)
│
├── 🐳 Dockerfile           # Build Docker image (pre-cache models)
├── 🐳 docker-compose.yml   # Triển khai 1 lệnh (ports, volumes, GPU)
│
├── 📋 requirements.txt     # Danh sách Python packages
├── 🔑 .env                 # API key + cấu hình (KHÔNG commit lên Git)
├── 🚫 .gitignore           # Loại trừ .env, data/, __pycache__
│
├── 📖 README.md            # Hướng dẫn tổng quan (bạn đang đọc)
├── 🧭 CODE_FLOW.md         # Giải thích chi tiết luồng code
│
├── 📓 backend.ipynb        # Notebook Colab (phiên bản demo cũ)
│
└── 📂 data/                # (Runtime — không commit)
    ├── vector_db/          # ChromaDB embeddings
    ├── graph.json          # Knowledge Graph
    ├── documents/          # Tài liệu gốc
    └── uploads/            # File upload từ user
```

---

## 🔄 Cách Hoạt Động — 3 Bước Chính

### Khi bạn hỏi một câu hỏi:

```
                    "Ai là giám đốc công ty XYZ?"
                                │
              ╔═════════════════╧════════════════╗
              ║    BƯỚC 1: TRA CỨU (Hybrid RAG)  ║
              ║                                   ║
              ║  📊 Vector Search (ChromaDB)      ║
              ║     → Tìm 15 đoạn văn gần nhất   ║
              ║                                   ║
              ║  🕸️ Graph Search (NetworkX)        ║
              ║     → Tìm quan hệ liên quan       ║
              ║     "XYZ → giám đốc → Ng. Văn A"  ║
              ║                                   ║
              ║  🔄 Reranker (CrossEncoder)        ║
              ║     → Chấm điểm tất cả → top 5    ║
              ╚═════════════════╤════════════════╝
                                │
              ╔═════════════════╧════════════════╗
              ║  BƯỚC 2: SUY LUẬN (Tree of Thought)║
              ║                                   ║
              ║  🌳 Sinh 4 hướng phân tích        ║
              ║     Hướng 1: Dựa theo tên...  8đ  ║
              ║     Hướng 2: Dựa theo chức vụ 6đ  ║
              ║     Hướng 3: Dựa theo tổ chức 7đ  ║
              ║     Hướng 4: Tổng hợp...      9đ  ║
              ║                                   ║
              ║  ⭐ Chọn Hướng 4 (9 điểm)         ║
              ╚═════════════════╤════════════════╝
                                │
              ╔═════════════════╧════════════════╗
              ║  BƯỚC 3: KIỂM TRA (Self-Reflection)║
              ║                                   ║
              ║  ✍️ LLM đọc lại câu trả lời       ║
              ║  🔍 Kiểm tra với ngữ cảnh gốc     ║
              ║  ✅ Loại bỏ thông tin bịa đặt      ║
              ║  📝 Viết bản hoàn chỉnh (~500 từ)  ║
              ╚═════════════════╤════════════════╝
                                │
                                ▼
                     📋 Câu Trả Lời Cuối Cùng
                  + 🧠 Quá trình suy nghĩ (xem được)
                  + 📚 Nguồn tài liệu (xem được)
```

> 💡 **Tổng cộng chỉ gọi Gemini API 3 lần** cho 1 câu hỏi (đã tối ưu từ 7 lần).

---

## 🚀 Hướng Dẫn Cài Đặt

### Yêu Cầu

| Yêu cầu | Mục đích | Cách lấy |
|---|---|---|
| **Gemini API Key** | Gọi LLM Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (miễn phí) |
| **Docker** | Đóng gói và chạy ứng dụng | [docs.docker.com](https://docs.docker.com/get-docker/) |
| **Git** | Clone source code | Thường có sẵn |

> 💡 **Không cần GPU** cho phần LLM — Gemini chạy trên cloud. GPU (nếu có) chỉ giúp Embedding + Reranker nhanh hơn.

### Bước 1 — Clone dự án

```bash
git clone https://github.com/Quangvu256/chatbotwithcolab.git
cd chatbotwithcolab
```

### Bước 2 — Cấu hình API Key

```bash
# Tạo file .env (hoặc sửa file .env có sẵn)
cat > .env << 'EOF'
# BẮT BUỘC: Lấy tại https://aistudio.google.com/apikey
GEMINI_API_KEY=your_api_key_here

# TUỲ CHỌN: Đổi model nếu muốn
GEMINI_MODEL=gemini-2.0-flash

# Server config (giữ mặc định nếu không cần đổi)
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
STREAMLIT_PORT=8501
ALLOWED_ORIGINS=http://localhost:8501
DATABASE_PATH=./data/vector_db

# TUỲ CHỌN: Auto-index tài liệu khi khởi động
# DOC_PATH=./data/documents/tai_lieu_cua_ban.pdf
EOF
```

### Bước 3 — Khởi chạy

```bash
docker compose up -d --build
```

> ⏳ Lần đầu build mất ~5-10 phút (tải Python packages + cache AI models).
> Các lần sau chỉ mất ~30 giây.

### Bước 4 — Truy cập

| Giao diện | URL | Mô tả |
|---|---|---|
| 🎨 **Chat UI** | `http://localhost:8501` | Giao diện chat + upload tài liệu |
| 📚 **API Docs** | `http://localhost:8000/docs` | Swagger UI — test API trực tiếp |
| 💚 **Health** | `http://localhost:8000/health` | Kiểm tra trạng thái hệ thống |

### Bước 5 — Sử dụng

1. Mở `http://localhost:8501` trên trình duyệt
2. Upload file tài liệu (PDF/TXT/DOCX) qua sidebar bên trái
3. Bấm **"📥 Index tài liệu"** và đợi hệ thống xử lý (chạy nền)
4. Bắt đầu hỏi đáp! 🎉

---

## 🌐 Triển Khai Trên Server (GCP / VPS)

### Bước thêm cho máy chủ từ xa:

```bash
# SSH vào máy chủ
ssh user@your-server-ip

# Clone + cấu hình (giống Bước 1-2 ở trên)
git clone https://github.com/Quangvu256/chatbotwithcolab.git
cd chatbotwithcolab
# Sửa file .env với API key thật

# Mở firewall cho ports
# GCP: VPC Network > Firewall > Create Rule → TCP: 8000,8501
# VPS: sudo ufw allow 8000 && sudo ufw allow 8501

# Khởi chạy
docker compose up -d --build
```

Truy cập: `http://<IP_MÁY_CHỦ>:8501`

### Lệnh hữu ích

```bash
docker compose logs -f          # Xem logs realtime
docker compose restart          # Khởi động lại
docker compose down             # Dừng hoàn toàn
docker compose up -d            # Khởi động lại (không build)
```

---

## 🧪 Chế Độ Colab (Demo)

> Dùng `backend.ipynb` để chạy demo nhanh trên Google Colab với GPU miễn phí.

1. Upload `backend.ipynb` lên Google Drive tại `/MyDrive/chatbotcolab/`
2. Mở bằng Colab → đổi runtime sang **GPU** → chạy tất cả cell
3. Sao chép URL Ngrok hiển thị
4. Chạy trên máy local: `streamlit run chatbot.py`
5. Dán URL Ngrok vào sidebar → bắt đầu chat

> ⚠️ URL Ngrok thay đổi mỗi lần restart Colab — nhớ cập nhật.

---

## 📡 API Endpoints

| Phương thức | Endpoint | Mô tả | Rate Limit |
|---|---|---|---|
| `GET` | `/health` | Kiểm tra trạng thái hệ thống | Không |
| `POST` | `/api/v1/ask` | Hỏi đáp (RAG + ToT + Self-Reflection) | 10 req/phút |
| `POST` | `/api/v1/index` | Upload & index tài liệu mới | Không |
| `POST` | `/api/v1/summarize` | Tóm tắt tài liệu (Map-Reduce) | Không |

### Ví dụ gọi API:

```bash
# Hỏi đáp
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Nội dung chính của tài liệu là gì?"}'

# Upload tài liệu
curl -X POST http://localhost:8000/api/v1/index \
  -F "file=@tai_lieu.pdf"

# Tóm tắt
curl -X POST http://localhost:8000/api/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{"query": "Tóm tắt toàn bộ tài liệu"}'
```

---

## 📄 Định Dạng File Hỗ Trợ

| Định dạng | Phần mở rộng | Loader |
|---|---|---|
| PDF | `.pdf` | `PyPDFLoader` |
| Plain Text | `.txt` | `TextLoader` (UTF-8) |
| Microsoft Word | `.docx` | `Docx2txtLoader` |

---

## ⚠️ Lưu Ý Quan Trọng

- 🕐 **Thời gian khởi động:** ~30 giây (models đã pre-cached trong Docker image)
- 🕐 **Thời gian trả lời:** 10-40 giây/câu hỏi (3 lần gọi Gemini API)
- 🕐 **Thời gian index:** 2-5 phút cho PDF 50 trang (chạy nền, không block UI)
- 📤 **Upload tài liệu mới:** Không cần restart — upload qua sidebar, index tự động
- 💾 **Dữ liệu được lưu:** Vector DB + Graph DB lưu trên disk, không mất khi restart container
- 🔑 **API Key:** Lấy miễn phí tại [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

---

## 📖 Tài Liệu Thêm

- 📘 **[CODE_FLOW.md](CODE_FLOW.md)** — Giải thích chi tiết từng dòng code, luồng xử lý, sơ đồ Mermaid
- 📋 **[implementation_plan.md](implementation_plan.md)** — Kế hoạch tối ưu hóa ban đầu (lịch sử phát triển)

---

---

# 🇬🇧 Overview (English)

## 📖 What is ChatBotColab?

**ChatBotColab** is an AI assistant that combines multiple advanced techniques to answer questions based on **your own documents**:

- **Hybrid RAG** — Vector Search + Knowledge Graph Search + GPU Reranker
- **Tree of Thought** — Generates multiple reasoning paths, scores them, selects the best
- **Self-Reflection** — LLM reviews its own answer to reduce hallucinations
- **Map-Reduce Summarization** — Summarize large documents using divide-and-conquer

### Key Features

| Feature | Description |
|---|---|
| 📄 Document Upload | PDF, TXT, DOCX — index via UI or API |
| 🕸️ Knowledge Graph | Auto-extract entities & relationships using Gemini API |
| 🌳 Tree of Thought | 4 reasoning paths → batch scoring → best path |
| ✅ Self-Reflection | LLM self-corrects before final answer |
| ☁️ Cloud LLM | Gemini 2.0 Flash via Google AI Studio — **no GPU needed for LLM** |
| 🔍 Hybrid Search | Vector (ChromaDB) + Graph (NetworkX) + Reranker (CrossEncoder) |

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Gemini 2.0 Flash | Reasoning & generation (via Google AI Studio API) |
| **Embedding** | `vietnamese-bi-encoder` (BKAI) | Vietnamese text → vector embeddings |
| **Reranker** | `bge-reranker-v2-m3` (BAAI) | Re-score retrieved chunks for relevance |
| **Vector DB** | ChromaDB | Store & search document embeddings |
| **Graph DB** | NetworkX + JSON | Knowledge graph (entities + relationships) |
| **Chunking** | LangChain `RecursiveCharacterTextSplitter` | Smart document splitting (1000 chars, 200 overlap) |
| **API** | FastAPI + Uvicorn | REST API server |
| **Frontend** | Streamlit | Chat UI with session history & file upload |
| **Container** | Docker + Docker Compose | Reproducible deployment |

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/Quangvu256/chatbotwithcolab.git
cd chatbotwithcolab

# 2. Configure (get free API key at https://aistudio.google.com/apikey)
echo "GEMINI_API_KEY=your_key_here" > .env

# 3. Launch
docker compose up -d --build

# 4. Access
# Chat UI:  http://localhost:8501
# API Docs: http://localhost:8000/docs
# Health:   http://localhost:8000/health
```

## 📡 API Endpoints

| Method | Endpoint | Description | Rate Limit |
|---|---|---|---|
| `GET` | `/health` | System health check | None |
| `POST` | `/api/v1/ask` | Ask a question (RAG + ToT + Self-Reflection) | 10 req/min |
| `POST` | `/api/v1/index` | Upload & index a document (PDF/TXT/DOCX) | None |
| `POST` | `/api/v1/summarize` | Summarize documents (Map-Reduce) | None |

## 🔄 How It Works

```
User Question → RAG Retrieval → Tree of Thought → Self-Reflection → Final Answer
                 (Vector+Graph     (4 paths →        (LLM self-
                  +Reranker)        score → best)      corrects)
```

1. **RAG Retrieval**: Search Vector DB (15 chunks) + Knowledge Graph → Rerank → Top 5
2. **Tree of Thought**: Generate 4 reasoning paths → Batch score all → Select best
3. **Self-Reflection**: LLM reviews & refines the draft → Final answer (~500 words)

> 📖 For detailed code flow with diagrams, see **[CODE_FLOW.md](CODE_FLOW.md)**

---

## 📜 License

This project is for educational and research purposes.

## 👤 Author

**Quangvu256** — Made with ❤️ using Gemini API, Docker & Streamlit
