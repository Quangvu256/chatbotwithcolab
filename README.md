# 🤖 ChatBotColab

> **An AI chatbot combining Tree of Thought reasoning with RAG — powered by Gemma-2-27B on Google Colab**

<p align="center">
  <a href="#-overview">English</a> •
  <a href="#-tổng-quan">Tiếng Việt</a>
</p>

---

# English

## 📖 Overview

**ChatBotColab** is an AI assistant that uses **Tree of Thought (ToT)** reasoning and **Retrieval-Augmented Generation (RAG)** to answer questions based on your own documents.

- The **backend** runs entirely on **Google Colab** with a free GPU (A100 / H100), using the **Gemma-2-27B-IT** model compressed to 4-bit via BitsAndBytes.
- The **frontend** is a **Streamlit** chat interface that runs on your local machine and connects to the backend through an **Ngrok** tunnel.

## 🏗️ Architecture

```
┌──────────────────────┐        HTTPS (Ngrok)        ┌─────────────────────────────┐
│  LOCAL MACHINE       │  ◄────────────────────────►  │  GOOGLE COLAB (GPU)         │
│                      │      POST /api/v1/ask        │                             │
│  chatbot.py          │                              │  backend.ipynb              │
│  ┌────────────────┐  │                              │  ┌───────────────────────┐  │
│  │ Streamlit UI   │  │                              │  │ FastAPI + Uvicorn     │  │
│  │ • Chat input   │  │                              │  │                       │  │
│  │ • Chat history │  │                              │  │ SmartKnowledgeBuilder │  │
│  │ • ToT viewer   │  │                              │  │ • Vietnamese Encoder  │  │
│  │ • RAG viewer   │  │                              │  │ • BGE-M3 Reranker     │  │
│  │ • API config   │  │                              │  │ • ChromaDB            │  │
│  └────────────────┘  │                              │  │ • Semantic Chunking   │  │
│                      │                              │  │                       │  │
│  .env                │                              │  │ AdvancedReasoningAgent│  │
│  • API_URL           │                              │  │ • Gemma-2-27B (4-bit) │  │
│  • HF_TOKEN          │                              │  │ • Tree of Thought     │  │
│  • NGROK_TOKEN       │                              │  │ • Self-Reflection     │  │
│                      │                              │  │ • Map-Reduce          │  │
└──────────────────────┘                              │  └───────────────────────┘  │
                                                      └─────────────────────────────┘
```

## 🛠️ Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **LLM** | `google/gemma-2-27b-it` | Main language model (4-bit quantized via BitsAndBytes NF4) |
| **Embedding** | `bkai-foundation-models/vietnamese-bi-encoder` | Encode documents & queries into vectors (optimized for Vietnamese) |
| **Reranker** | `BAAI/bge-reranker-v2-m3` | Re-score retrieved chunks for higher relevance |
| **Vector DB** | ChromaDB | Store and search document embeddings |
| **Chunking** | LangChain `SemanticChunker` | Split documents by meaning, not by fixed size |
| **Document Loaders** | PyPDFLoader, TextLoader, Docx2txtLoader | Read `.pdf`, `.txt`, `.docx` files |
| **API** | FastAPI + Uvicorn | Serve the backend as a REST API |
| **Tunnel** | pyngrok (Ngrok) | Expose the Colab server to the internet |
| **Frontend** | Streamlit | Chat UI with session history |
| **Quantization** | BitsAndBytes | Compress 27B model to fit in a single GPU's VRAM |
| **Runtime** | Google Colab (H100 / A100 GPU) | Free GPU for model inference |

## 📁 Project Structure

```
ChatBotColab/
├── backend.ipynb   # Google Colab notebook (entire backend)
│                   #   Cell 1: Mount Google Drive
│                   #   Cell 2: Install all dependencies
│                   #   Cell 3: SmartKnowledgeBuilder (RAG engine)
│                   #   Cell 4: AdvancedReasoningAgent (LLM + ToT)
│                   #   Cell 5: Load LLM with HF_TOKEN
│                   #   Cell 6: Define FastAPI endpoints
│                   #   Cell 7: Process documents into vector DB
│                   #   Cell 8: Start Ngrok + Uvicorn server
├── chatbot.py      # Streamlit frontend (local chat interface)
├── .env            # Environment variables (secrets & config)
└── .gitignore      # Excludes .env, __pycache__, vector DB, checkpoints
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

### Prerequisites

| Requirement | Purpose |
|---|---|
| [Google Colab](https://colab.research.google.com/) account | Run backend with free GPU |
| [Hugging Face](https://huggingface.co/settings/tokens) token | Download Gemma-2-27B model |
| [Ngrok](https://dashboard.ngrok.com/get-started/your-authtoken) auth token | Tunnel Colab to the internet |
| Python 3.8+ (local) | Run Streamlit frontend |

### Step 1 — Configure `.env`

Create a `.env` file in the project root:

```env
HF_TOKEN=hf_your_huggingface_token
NGROK_TOKEN=your_ngrok_auth_token
GIT_TOKEN=ghp_your_github_token
DATABASE_PATH=/content/drive/MyDrive/chatbotcolab/my_vector_db
DOC_PATH=/content/drive/MyDrive/chatbotcolab/your_document.pdf
API_URL=https://your-url.ngrok-free.dev/api/v1/ask
```

| Variable | Description |
|---|---|
| `HF_TOKEN` | Hugging Face access token (to download gated models) |
| `NGROK_TOKEN` | Ngrok auth token (to create public tunnels) |
| `GIT_TOKEN` | GitHub personal access token (optional, for version control) |
| `DATABASE_PATH` | Path to persist the ChromaDB vector database on Google Drive |
| `DOC_PATH` | Path to the document you want the chatbot to learn from |
| `API_URL` | The Ngrok public URL of your running backend (updated each session) |

### Step 2 — Run the Backend on Google Colab

1. Upload `backend.ipynb` and `.env` to Google Drive at `/MyDrive/chatbotcolab/`
2. Open `backend.ipynb` in Google Colab
3. Change runtime to **GPU** → select **H100** or **A100**
4. **Run all cells** from top to bottom
5. Wait for the last cell to print a Ngrok URL like:
   ```
   🟢 MÁY CHỦ ĐÃ SẴN SÀNG!
   🔗 URL Của API: https://xxxx.ngrok-free.dev/api/v1/ask
   📚 Swagger UI:  https://xxxx.ngrok-free.dev/docs
   ```
6. Copy the API URL — you'll need it for the frontend

### Step 3 — Run the Frontend Locally

```bash
# Install dependencies
pip install streamlit requests python-dotenv

# Start the chat interface
streamlit run chatbot.py
```

Then either:
- **Paste the Ngrok URL** into the sidebar text input, or
- **Set `API_URL`** in your local `.env` file (it auto-loads on startup)

### Step 4 — Ask Questions!

Type your question in the chat box. The AI will:
1. Search your documents for relevant context
2. Reason through multiple approaches (Tree of Thought)
3. Refine the answer (Self-Reflection)
4. Show you the final answer, plus expandable sections for the reasoning process and retrieved context

## 📄 Supported Documents

| Format | Extension | Loader Used |
|---|---|---|
| PDF | `.pdf` | `PyPDFLoader` |
| Plain Text | `.txt` | `TextLoader` |
| Microsoft Word | `.docx` | `Docx2txtLoader` |

## ⚠️ Notes

- The **Ngrok URL changes every time** you restart the Colab notebook — remember to update it in the frontend
- The backend takes **5–10 minutes** to initialize (downloading & loading the 27B model)
- Each question may take **30–120 seconds** to answer due to the multi-step ToT reasoning
- A **Colab Pro** subscription is recommended for longer runtime and better GPUs

---

# Tiếng Việt

## 📖 Tổng Quan

**ChatBotColab** là trợ lý AI sử dụng kỹ thuật **Tree of Thought (Cây Suy Nghĩ)** kết hợp **RAG (Truy Xuất Tăng Cường)** để trả lời câu hỏi dựa trên tài liệu của bạn.

- **Backend** chạy hoàn toàn trên **Google Colab** với GPU miễn phí (A100 / H100), sử dụng mô hình **Gemma-2-27B-IT** nén xuống 4-bit bằng BitsAndBytes.
- **Frontend** là giao diện chat **Streamlit** chạy trên máy tính cá nhân, kết nối với backend qua đường hầm **Ngrok**.

## 🏗️ Kiến Trúc Hệ Thống

```
┌──────────────────────┐        HTTPS (Ngrok)        ┌─────────────────────────────┐
│  MÁY CÁ NHÂN        │  ◄────────────────────────►  │  GOOGLE COLAB (GPU)         │
│                      │      POST /api/v1/ask        │                             │
│  chatbot.py          │                              │  backend.ipynb              │
│  ┌────────────────┐  │                              │  ┌───────────────────────┐  │
│  │ Giao diện      │  │                              │  │ FastAPI + Uvicorn     │  │
│  │ Streamlit      │  │                              │  │                       │  │
│  │ • Nhập câu hỏi │  │                              │  │ SmartKnowledgeBuilder │  │
│  │ • Lịch sử chat │  │                              │  │ • Vietnamese Encoder  │  │
│  │ • Xem ToT      │  │                              │  │ • BGE-M3 Reranker     │  │
│  │ • Xem RAG      │  │                              │  │ • ChromaDB            │  │
│  │ • Cấu hình URL │  │                              │  │ • Semantic Chunking   │  │
│  └────────────────┘  │                              │  │                       │  │
│                      │                              │  │ AdvancedReasoningAgent│  │
│  .env                │                              │  │ • Gemma-2-27B (4-bit) │  │
│  • API_URL           │                              │  │ • Tree of Thought     │  │
│  • HF_TOKEN          │                              │  │ • Self-Reflection     │  │
│  • NGROK_TOKEN       │                              │  │ • Map-Reduce          │  │
└──────────────────────┘                              │  └───────────────────────┘  │
                                                      └─────────────────────────────┘
```

## 🛠️ Công Nghệ Sử Dụng

| Tầng | Công nghệ | Vai trò |
|---|---|---|
| **Mô hình ngôn ngữ** | `google/gemma-2-27b-it` | LLM chính (nén 4-bit bằng BitsAndBytes NF4) |
| **Nhúng văn bản** | `bkai-foundation-models/vietnamese-bi-encoder` | Mã hóa tài liệu & câu hỏi thành vector (tối ưu cho tiếng Việt) |
| **Chấm điểm lại** | `BAAI/bge-reranker-v2-m3` | Đánh giá lại độ liên quan của các đoạn trích xuất |
| **CSDL Vector** | ChromaDB | Lưu trữ và tìm kiếm vector nhúng |
| **Cắt văn bản** | LangChain `SemanticChunker` | Chia tài liệu theo ngữ nghĩa, không theo kích thước cố định |
| **Đọc tài liệu** | PyPDFLoader, TextLoader, Docx2txtLoader | Đọc file `.pdf`, `.txt`, `.docx` |
| **API** | FastAPI + Uvicorn | Phục vụ backend dưới dạng REST API |
| **Đường hầm** | pyngrok (Ngrok) | Mở Colab ra internet công khai |
| **Giao diện** | Streamlit | Giao diện chat với lịch sử trò chuyện |
| **Nén mô hình** | BitsAndBytes | Nén mô hình 27B để vừa VRAM của 1 GPU |
| **GPU Runtime** | Google Colab (H100 / A100) | GPU miễn phí cho suy luận mô hình |

## 📁 Cấu Trúc Dự Án

```
ChatBotColab/
├── backend.ipynb   # Notebook Google Colab (toàn bộ backend)
│                   #   Cell 1: Mount Google Drive
│                   #   Cell 2: Cài đặt tất cả thư viện
│                   #   Cell 3: SmartKnowledgeBuilder (lõi RAG)
│                   #   Cell 4: AdvancedReasoningAgent (LLM + ToT)
│                   #   Cell 5: Nạp LLM bằng HF_TOKEN
│                   #   Cell 6: Định nghĩa endpoint FastAPI
│                   #   Cell 7: Xử lý tài liệu vào vector DB
│                   #   Cell 8: Khởi động Ngrok + Uvicorn
├── chatbot.py      # Giao diện Streamlit (chạy trên máy local)
├── .env            # Biến môi trường (token, đường dẫn, cấu hình)
└── .gitignore      # Bỏ qua .env, __pycache__, vector DB, checkpoints
```

## 🔄 Quy Trình Hoạt Động

Khi người dùng đặt câu hỏi, hệ thống xử lý theo pipeline sau:

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

### Yêu Cầu

| Yêu cầu | Mục đích |
|---|---|
| Tài khoản [Google Colab](https://colab.research.google.com/) | Chạy backend với GPU miễn phí |
| [Hugging Face](https://huggingface.co/settings/tokens) token | Tải mô hình Gemma-2-27B |
| [Ngrok](https://dashboard.ngrok.com/get-started/your-authtoken) auth token | Tạo đường hầm internet từ Colab |
| Python 3.8+ (trên máy local) | Chạy giao diện Streamlit |

### Bước 1 — Cấu Hình `.env`

Tạo file `.env` ở thư mục gốc dự án:

```env
HF_TOKEN=hf_token_huggingface_cua_ban
NGROK_TOKEN=ngrok_auth_token_cua_ban
GIT_TOKEN=ghp_github_token_cua_ban
DATABASE_PATH=/content/drive/MyDrive/chatbotcolab/my_vector_db
DOC_PATH=/content/drive/MyDrive/chatbotcolab/tai_lieu_cua_ban.pdf
API_URL=https://url-cua-ban.ngrok-free.dev/api/v1/ask
```

| Biến | Mô tả |
|---|---|
| `HF_TOKEN` | Token truy cập Hugging Face (để tải model bị giới hạn) |
| `NGROK_TOKEN` | Auth token của Ngrok (để tạo đường hầm công khai) |
| `GIT_TOKEN` | GitHub personal access token (tùy chọn, cho quản lý phiên bản) |
| `DATABASE_PATH` | Đường dẫn lưu trữ ChromaDB trên Google Drive |
| `DOC_PATH` | Đường dẫn đến tài liệu bạn muốn chatbot học |
| `API_URL` | URL Ngrok công khai của backend đang chạy (cập nhật mỗi phiên) |

### Bước 2 — Chạy Backend trên Google Colab

1. Upload `backend.ipynb` và `.env` lên Google Drive tại `/MyDrive/chatbotcolab/`
2. Mở `backend.ipynb` bằng Google Colab
3. Đổi runtime sang **GPU** → chọn **H100** hoặc **A100**
4. **Chạy tất cả cell** từ trên xuống dưới
5. Chờ cell cuối in ra URL Ngrok:
   ```
   🟢 MÁY CHỦ ĐÃ SẴN SÀNG!
   🔗 URL Của API: https://xxxx.ngrok-free.dev/api/v1/ask
   📚 Swagger UI:  https://xxxx.ngrok-free.dev/docs
   ```
6. Sao chép URL API — bạn sẽ cần nó cho frontend

### Bước 3 — Chạy Frontend trên Máy Local

```bash
# Cài đặt thư viện
pip install streamlit requests python-dotenv

# Khởi chạy giao diện chat
streamlit run chatbot.py
```

Sau đó:
- **Dán URL Ngrok** vào ô nhập liệu ở thanh bên trái, hoặc
- **Đặt `API_URL`** trong file `.env` trên máy local (tự động nạp khi khởi chạy)

### Bước 4 — Bắt Đầu Hỏi!

Gõ câu hỏi vào ô chat. AI sẽ:
1. Tìm kiếm ngữ cảnh liên quan từ tài liệu của bạn
2. Suy luận qua nhiều hướng tiếp cận (Tree of Thought)
3. Tinh chỉnh câu trả lời (Self-Reflection)
4. Hiển thị câu trả lời cuối cùng, kèm phần mở rộng xem quá trình suy nghĩ và ngữ cảnh trích xuất

## 📄 Định Dạng Tài Liệu Hỗ Trợ

| Định dạng | Phần mở rộng | Loader sử dụng |
|---|---|---|
| PDF | `.pdf` | `PyPDFLoader` |
| Văn bản thuần | `.txt` | `TextLoader` |
| Microsoft Word | `.docx` | `Docx2txtLoader` |

## ⚠️ Lưu Ý

- **URL Ngrok thay đổi mỗi lần** khởi động lại notebook Colab — nhớ cập nhật lại ở frontend
- Backend mất khoảng **5–10 phút** để khởi tạo (tải & nạp mô hình 27B)
- Mỗi câu hỏi có thể mất **30–120 giây** để trả lời do quy trình ToT nhiều bước
- Khuyến nghị dùng **Colab Pro** để có thời gian chạy dài hơn và GPU tốt hơn

---

## 📜 License

This project is for educational and research purposes.

## 👤 Author

Made with ❤️ using Google Colab & Streamlit
