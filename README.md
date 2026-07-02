# 🤖 ChatBotColab

> **An AI chatbot combining Tree of Thought reasoning with RAG — powered by Gemma-3-4B. Deploy on Google Colab (demo) or GCP with Docker (production).**

<p align="center">
  <a href="#-overview">English</a> •
  <a href="#-tổng-quan">Tiếng Việt</a>
</p>

---

# English

## 📖 Overview

**ChatBotColab** is an AI assistant that uses **Tree of Thought (ToT)** reasoning and **Retrieval-Augmented Generation (RAG)** to answer questions based on your own documents.

Two deployment modes:

| Mode | Runtime | Use Case |
|---|---|---|
| **🧪 Colab Demo** | Google Colab + Ngrok | Quick prototyping with free GPU |
| **🚀 GCP Production** | GCE VM + Docker | Stable, always-on deployment |

## 🏗️ Architecture

### GCP Production (Docker)

```
┌───────────────────┐     HTTP :8501     ┌──────────────────────────────────┐
│  Browser          │ ◄────────────────► │  GCE VM (CPU-only via Vertex AI) │
│  (any device)     │                    │  ┌────────────────────────────┐  │
└───────────────────┘     HTTP :8000     │  │  Docker Container         │  │
                      ◄────────────────► │  │                            │  │
                                         │  │  Streamlit UI (:8501)      │  │
                                         │  │  FastAPI API  (:8000)      │  │
                                         │  │  backend.py                │  │
                                         │  │  • SmartKnowledgeBuilder   │  │
                                         │  │  • AdvancedReasoningAgent  │  │
                                         │  │  • Gemini 2.0 Flash API    │  │
                                         │  └────────────────────────────┘  │
                                         │  Volume: ./data (vector DB)     │
                                         └──────────────────────────────────┘
```

### Colab Demo (Legacy)

```
┌──────────────────────┐        HTTPS (Ngrok)        ┌─────────────────────────────┐
│  LOCAL MACHINE       │  ◄────────────────────────►  │  GOOGLE COLAB (GPU)         │
│  chatbot.py          │      POST /api/v1/ask        │  backend.ipynb              │
│  (Streamlit UI)      │                              │  (FastAPI + LLM + RAG)      │
└──────────────────────┘                              └─────────────────────────────┘
```

## 🛠️ Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **LLM** | `gemini-2.0-flash` | Main language model (Google Cloud Vertex AI) |
| **Embedding** | `bkai-foundation-models/vietnamese-bi-encoder` | Encode documents & queries into vectors (optimized for Vietnamese) |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | Re-score retrieved chunks for higher relevance |
| **Vector DB** | ChromaDB | Store and search document embeddings |
| **Chunking** | LangChain `SemanticChunker` | Split documents by meaning, not by fixed size |
| **Document Loaders** | PyPDFLoader, TextLoader, Docx2txtLoader | Read `.pdf`, `.txt`, `.docx` files |
| **API** | FastAPI + Uvicorn | Serve the backend as a REST API |
| **Frontend** | Streamlit | Chat UI with session history & file upload |
| **Quantization** | None | No local GPU required, API handles it |
| **Container** | Docker + Docker Compose | Reproducible deployment without GPU requirements |
| **Runtime** | GCE VM (CPU-only) or Google Colab | Cloud API for model inference |

## 📁 Project Structure

```
ChatBotColab/
├── backend.py          # Standalone backend (GCP production)
├── backend.ipynb       # Google Colab notebook (demo/reference)
├── chatbot.py          # Streamlit frontend (chat + file upload)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker image build
├── docker-compose.yml  # One-command deployment with GPU
├── startup.sh          # Container entry point
├── .env                # Environment variables (secrets & config)
├── .gitignore          # Git exclusions
├── data/               # (runtime) Vector DB + uploaded documents
│   ├── vector_db/
│   ├── documents/
│   └── uploads/
└── README.md
```

## 🔄 How It Works

When a user asks a question, the system follows this pipeline:

```
User Question
      │
      ▼
┌─────────────────────────────────────────┐
│  1. RAG RETRIEVAL                       │
│     Vector search → top 30 chunks       │
│     Reranker (BGE-M3) → top 5 chunks    │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  2. TREE OF THOUGHT                     │
│     Generate 5 reasoning paths          │
│     Score each path (1-10) vs context   │
│     Select the best path                │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  3. SELF-REFLECTION                     │
│     LLM reviews & refines the answer    │
│     Removes hallucinations              │
└─────────────────┬───────────────────────┘
                  │
                  ▼
          Final Answer
   + Thought Process (expandable)
   + RAG Context (expandable)
```

## 🚀 Getting Started

### 🚀 Option A — GCP Production (Docker)

#### Prerequisites

| Requirement | Purpose |
|---|---|
| [Google Cloud](https://console.cloud.google.com/) account with billing | Run GCE VM and Vertex AI |
| [Vertex AI API](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com) | Enable Vertex AI API for Gemini 2.0 Flash |
| [Docker](https://docs.docker.com/get-docker/) | Run containerized app |

#### Step 1 — Create a GCE VM with GPU

1. Go to **Compute Engine > VM instances > Create Instance**
2. Configuration:
   - **Machine type:** `e2-standard-2` (CPU-only)
   - **Boot disk:** Debian or Ubuntu, **30 GB**
   - **Firewall:** ✅ Allow HTTP, ✅ Allow HTTPS
3. Add firewall rule for ports `8000` (API) and `8501` (UI):
   - **VPC Network > Firewall > Create Rule** → TCP: `8000, 8501`, Source: `0.0.0.0/0`

#### Step 2 — Setup on the VM

SSH into the VM and run:

```bash
# Clone the repository
git clone https://github.com/Quangvu256/chatbotwithcolab.git
cd chatbotwithcolab

# Create .env
cat > .env << 'EOF'
GCP_PROJECT_ID=your-gcp-project-id
GCP_LOCATION=us-central1
GEMINI_MODEL=gemini-2.0-flash
DATABASE_PATH=./data/vector_db
DOC_PATH=./data/documents/your_document.pdf
API_URL=http://localhost:8000/api/v1/ask
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
STREAMLIT_PORT=8501
EOF

# Put your document in data/documents/
mkdir -p data/documents
# Upload your PDF/TXT/DOCX here
```

#### Step 3 — Launch with Docker Compose

```bash
docker compose up -d --build
```

Wait 5-10 minutes for model loading, then access:
- 🎨 **Chat UI:** `http://<VM_EXTERNAL_IP>:8501`
- 📚 **API Docs:** `http://<VM_EXTERNAL_IP>:8000/docs`
- 💚 **Health:** `http://<VM_EXTERNAL_IP>:8000/health`

#### Useful Commands

```bash
docker compose logs -f        # Watch logs (model loading progress)
docker compose restart         # Restart services
docker compose down            # Stop everything
```

#### 💰 Estimated Cost (GCP)

| Resource | Spec | Cost (approx.) |
|---|---|---|
| `e2-standard-2` | 2 vCPU, 8GB RAM | ~$0.07/hr |
| Boot disk | 30 GB | ~$1.50/mo |
| Vertex AI | Gemini API calls | Variable (very cheap) |

> 💡 **Tip:** Use [Spot VMs](https://cloud.google.com/compute/docs/instances/spot) to reduce compute cost by ~60-70%.

---

### 🧪 Option B — Google Colab (Demo)

> Use `backend.ipynb` for quick demos with free GPU. See the notebook for step-by-step instructions.

1. Upload `backend.ipynb` and `.env` to Google Drive at `/MyDrive/chatbotcolab/`
2. Open in Colab, set runtime to **GPU (H100/A100)**, run all cells
3. Copy the Ngrok URL and paste into the Streamlit sidebar
4. Run `streamlit run chatbot.py` on your local machine

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health check (GPU, LLM, DB status) |
| `POST` | `/api/v1/ask` | Ask a question (ToT + RAG reasoning) |
| `POST` | `/api/v1/index` | Upload & index a new document (PDF/TXT/DOCX) |
| `POST` | `/api/v1/summarize` | Summarize all documents using Map-Reduce |

## 📄 Supported Documents

| Format | Extension | Loader Used |
|---|---|---|
| PDF | `.pdf` | `PyPDFLoader` |
| Plain Text | `.txt` | `TextLoader` |
| Microsoft Word | `.docx` | `Docx2txtLoader` |

## ⚠️ Notes

- The backend takes **~30 seconds** to initialize (models are pre-cached)
- Each question may take **30–120 seconds** to answer due to the multi-step ToT reasoning
- You can upload new documents via the Streamlit sidebar (no restart needed)
- For Colab mode: The Ngrok URL changes every restart — update it in the frontend

---

# Tiếng Việt

## 📖 Tổng Quan

**ChatBotColab** là trợ lý AI sử dụng kỹ thuật **Tree of Thought (Cây Suy Nghĩ)** kết hợp **RAG (Truy Xuất Tăng Cường)** để trả lời câu hỏi dựa trên tài liệu của bạn.

Hai chế độ triển khai:

| Chế độ | Môi trường | Mục đích |
|---|---|---|
| **🚀 GCP Production** | GCE VM + Docker | Triển khai ổn định, chạy 24/7 |
| **🧪 Colab Demo** | Google Colab + Ngrok | Prototype nhanh với GPU miễn phí |

## 🏗️ Kiến Trúc Hệ Thống

### GCP Production (Docker)

```
┌───────────────────┐     HTTP :8501     ┌──────────────────────────────────┐
│  Trình duyệt      │ ◄────────────────► │  GCE VM (CPU-only via Vertex AI) │
│  (bất kỳ đâu)     │                    │  ┌────────────────────────────┐  │
└───────────────────┘     HTTP :8000     │  │  Docker Container         │  │
                      ◄────────────────► │  │                            │  │
                                         │  │  Streamlit UI (:8501)      │  │
                                         │  │  FastAPI API  (:8000)      │  │
                                         │  │  backend.py                │  │
                                         │  │  • SmartKnowledgeBuilder   │  │
                                         │  │  • AdvancedReasoningAgent  │  │
                                         │  │  • Gemini 2.0 Flash API    │  │
                                         │  └────────────────────────────┘  │
                                         │  Volume: ./data (vector DB)     │
                                         └──────────────────────────────────┘
```

## 🔄 Quy Trình Hoạt Động

```
Câu hỏi người dùng
        │
        ▼
┌─────────────────────────────────────────────┐
│  1. TRUY XUẤT RAG                           │
│     Tìm kiếm vector → lấy 30 đoạn gần nhất│
│     Reranker (BGE-M3) → chọn 5 đoạn tốt nhất│
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  2. CÂY SUY NGHĨ (Tree of Thought)         │
│     Tạo 5 hướng suy luận khác nhau         │
│     Chấm điểm từng hướng (1-10)            │
│     Chọn hướng tốt nhất                    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  3. TỰ PHẢN BIỆN (Self-Reflection)         │
│     LLM xem xét & tinh chỉnh câu trả lời  │
│     Loại bỏ thông tin bịa đặt              │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
          Câu trả lời cuối cùng
   + Quá trình suy nghĩ (mở rộng được)
   + Ngữ cảnh RAG (mở rộng được)
```

## 🚀 Hướng Dẫn Cài Đặt

### 🚀 Cách A — GCP Production (Docker)

#### Yêu Cầu

| Yêu cầu | Mục đích |
|---|---|
| Tài khoản [Google Cloud](https://console.cloud.google.com/) có billing | Chạy GCE VM và Vertex AI |
| [Vertex AI API](https://console.cloud.google.com/apis/library/aiplatform.googleapis.com) | Enable Vertex AI API cho Gemini 2.0 Flash |
| [Docker](https://docs.docker.com/get-docker/) | Chạy container |

#### Bước 1 — Tạo VM có GPU trên GCE

1. Vào **Compute Engine > VM instances > Create Instance**
2. Cấu hình:
   - **Machine type:** `e2-standard-2` (CPU-only)
   - **Boot disk:** Debian hoặc Ubuntu, **30 GB**
   - **Firewall:** ✅ Allow HTTP, ✅ Allow HTTPS
3. Thêm firewall rule cho port `8000` (API) và `8501` (UI):
   - **VPC Network > Firewall > Create Rule** → TCP: `8000, 8501`, Source: `0.0.0.0/0`

#### Bước 2 — Cài đặt trên VM

SSH vào VM và chạy:

```bash
# Clone dự án
git clone https://github.com/Quangvu256/chatbotwithcolab.git
cd chatbotwithcolab

# Tạo file .env
cat > .env << 'EOF'
GCP_PROJECT_ID=your-gcp-project-id
GCP_LOCATION=us-central1
GEMINI_MODEL=gemini-2.0-flash
DATABASE_PATH=./data/vector_db
DOC_PATH=./data/documents/tai_lieu_cua_ban.pdf
API_URL=http://localhost:8000/api/v1/ask
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
STREAMLIT_PORT=8501
EOF

# Đặt tài liệu vào data/documents/
mkdir -p data/documents
# Upload file PDF/TXT/DOCX vào đây
```

#### Bước 3 — Khởi chạy bằng Docker Compose

```bash
docker compose up -d --build
```

Đợi 5-10 phút để nạp mô hình, sau đó truy cập:
- 🎨 **Giao diện Chat:** `http://<IP_NGOÀI_CỦA_VM>:8501`
- 📚 **API Docs:** `http://<IP_NGOÀI_CỦA_VM>:8000/docs`
- 💚 **Health Check:** `http://<IP_NGOÀI_CỦA_VM>:8000/health`

#### Lệnh hữu ích

```bash
docker compose logs -f        # Xem logs (tiến trình nạp model)
docker compose restart         # Khởi động lại
docker compose down            # Dừng hệ thống
```

#### 💰 Chi phí ước tính (GCP)

| Tài nguyên | Cấu hình | Chi phí (ước tính) |
|---|---|---|
| `e2-standard-2` | 2 vCPU, 8GB RAM | ~$0.07/giờ |
| Boot disk | 30 GB | ~$1.50/tháng |
| Vertex AI | Gọi Gemini API | Thay đổi (rất rẻ) |

> 💡 **Mẹo:** Dùng [Spot VM](https://cloud.google.com/compute/docs/instances/spot) để giảm chi phí compute ~60-70%.

---

### 🧪 Cách B — Google Colab (Demo)

> Dùng `backend.ipynb` để demo nhanh với GPU miễn phí.

1. Upload `backend.ipynb` và `.env` lên Google Drive tại `/MyDrive/chatbotcolab/`
2. Mở bằng Colab, đổi runtime sang **GPU (H100/A100)**, chạy tất cả cell
3. Sao chép URL Ngrok và dán vào sidebar của Streamlit
4. Chạy `streamlit run chatbot.py` trên máy local

---

## 📡 API Endpoints

| Phương thức | Endpoint | Mô tả |
|---|---|---|
| `GET` | `/health` | Kiểm tra trạng thái hệ thống (GPU, LLM, DB) |
| `POST` | `/api/v1/ask` | Đặt câu hỏi (ToT + RAG reasoning) |
| `POST` | `/api/v1/index` | Upload & index tài liệu mới (PDF/TXT/DOCX) |
| `POST` | `/api/v1/summarize` | Tóm tắt tài liệu bằng Map-Reduce |

## ⚠️ Lưu Ý

- Backend mất khoảng **~30 giây** để khởi tạo (models đã được pre-cached)
- Mỗi câu hỏi có thể mất **30–120 giây** để trả lời do quy trình ToT nhiều bước
- Có thể upload tài liệu mới qua giao diện Streamlit (không cần khởi động lại)
- Chế độ Colab: URL Ngrok thay đổi mỗi lần khởi động lại — nhớ cập nhật ở frontend

---

## 📜 License

This project is for educational and research purposes.

## 👤 Author

Made with ❤️ using Google Cloud, Docker & Streamlit



