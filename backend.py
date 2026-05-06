"""
ChatBotColab — Standalone Backend Server (Vertex AI Edition)
=============================================================
Production deployment on GCP using Gemini API via Vertex AI.
No GPU required — all LLM inference is handled by Google Cloud.

Usage:
    python backend.py

Endpoints:
    GET  /health          — Health check
    POST /api/v1/ask      — Ask a question (ToT + RAG)
    POST /api/v1/index    — Upload & index a new document
"""

import os
from pathlib import Path

from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

from google import genai
from google.genai import types

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ==========================================
# 1. NẠP BIẾN MÔI TRƯỜNG
# ==========================================
load_dotenv()

DATABASE_PATH = os.getenv("DATABASE_PATH", "./data/vector_db")
DOC_PATH = os.getenv("DOC_PATH", "")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))

# Tạo thư mục data nếu chưa có
os.makedirs("./data/documents", exist_ok=True)
os.makedirs("./data/uploads", exist_ok=True)


# ==========================================
# 2. SMART KNOWLEDGE BUILDER (RAG Engine)
# ==========================================
class SmartKnowledgeBuilder:
    """
    Lõi RAG: Vector Search + Semantic Chunking + Reranker.
    Chuyển đổi tài liệu thành vector embeddings và truy xuất ngữ cảnh.
    """

    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path

        # Embedding chạy trên CPU (model nhỏ, không cần GPU)
        print("📥 [1/2] Đang tải mô hình Vector Embedding...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="bkai-foundation-models/vietnamese-bi-encoder",
            model_kwargs={"device": "cpu"},
        )

        print("🕵️ [2/2] Đang tải mô hình Reranker (Thám tử chấm điểm)...")
        self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=2048)

        self.vector_db = None
        self._load_existing_db()

    def _load_existing_db(self):
        """Khôi phục lại DB nếu đã có trên ổ cứng"""
        if os.path.exists(self.db_path) and os.listdir(self.db_path):
            self.vector_db = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embedding_model,
            )
            print(f"📁 Đã khôi phục Vector DB từ {self.db_path}")

    def process_and_save(self, file_path):
        """Chỉ tập trung vào Vector Search + Semantic Chunking"""
        _, ext = os.path.splitext(file_path)
        if ext.lower() == ".pdf":
            loader = PyPDFLoader(file_path)
        elif ext.lower() == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")
        elif ext.lower() == ".docx":
            loader = Docx2txtLoader(file_path)
        else:
            raise ValueError(f"Chưa hỗ trợ định dạng: {ext}")

        documents = loader.load()
        print("✂️ Đang cắt tài liệu bằng Recursive Chunking (Tốc độ siêu tốc)...")

        # Đổi từ Semantic sang Recursive để chạy mượt trên CPU
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)
        print(f"✅ Cắt xong! Hệ thống đã chia thành {len(chunks)} đoạn ngữ nghĩa.")

        print(f"🔄 Đang nhúng {len(chunks)} chunks vào ChromaDB...")
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=self.db_path,
        )
        print("✅ Đã hoàn tất xây dựng Smart Vector RAG!")

    def retrieve_context(self, query, top_k_vector=30, final_k=5):
        """
        Quy trình lấy thông tin chuẩn:
        1. Lấy 30 đoạn liên quan nhất (Vector Search).
        2. Dùng Reranker chấm điểm lại.
        3. Lấy 5 đoạn điểm cao nhất trả về cho LLM.
        """
        if not self.vector_db:
            return "Chưa có dữ liệu."

        raw_docs = self.vector_db.similarity_search(query, k=top_k_vector)
        chunk_texts = [doc.page_content for doc in raw_docs]

        if not chunk_texts:
            return "Không tìm thấy thông tin liên quan."

        pairs = [[query, text] for text in chunk_texts]
        scores = self.reranker.predict(pairs)

        paired_results = list(zip(chunk_texts, scores))
        paired_results.sort(key=lambda x: x[1], reverse=True)
        best_chunks = [item[0] for item in paired_results[:final_k]]

        return "\n---\n".join(best_chunks)


# ==========================================
# 3. ADVANCED REASONING AGENT (Vertex AI Gemini + ToT)
# ==========================================
class AdvancedReasoningAgent:
    """
    Lõi suy luận: Gemini 2.0 Flash (via Vertex AI) + Tree of Thought + Self-Reflection.
    Không cần GPU — gọi API qua Google Cloud.
    """

    def __init__(self, project_id, location="us-central1", model_name="gemini-2.0-flash"):
        print(f"🧠 [LLM] Kết nối Gemini API qua Vertex AI...")
        print(f"   → Project: {project_id}")
        print(f"   → Location: {location}")
        print(f"   → Model: {model_name}")

        self.model_name = model_name
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )

        # Test kết nối
        try:
            test_response = self.client.models.generate_content(
                model=self.model_name,
                contents="Xin chào, trả lời ngắn gọn: 1+1=?",
                config=types.GenerateContentConfig(
                    max_output_tokens=50,
                    temperature=0.0,
                ),
            )
            print(f"✅ [LLM] Kết nối Gemini thành công! Test: {test_response.text.strip()}")
        except Exception as e:
            print(f"⚠️ [LLM] Cảnh báo: Không thể kết nối Gemini: {e}")

    def _call_llm(self, prompt, temperature=0.1, max_output_tokens=2048):
        """Gọi Gemini API qua Vertex AI"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=max_output_tokens,
                    temperature=temperature,
                ),
            )
            return response.text.strip()
        except Exception as e:
            print(f"❌ [LLM] Lỗi khi gọi Gemini: {e}")
            return f"Lỗi khi gọi mô hình: {str(e)}"

    # ==========================================
    # LOGIC 1: TREE OF THOUGHT (Tư duy cục bộ)
    # ==========================================
    def generate_thoughts(self, query, context, num_thoughts=5):
        prompt = f"Ngữ cảnh: {context}\nCâu hỏi: {query}\nHãy đưa ra {num_thoughts} hướng phân tích ngắn gọn và khác biệt để trả lời. Liệt kê bắt đầu bằng 'Hướng 1:', 'Hướng 2:'..."
        raw_text = self._call_llm(prompt, temperature=0.7)
        thoughts = [t.strip() for t in raw_text.split("\n") if len(t.strip()) > 10]
        return thoughts[:num_thoughts] if thoughts else [raw_text]

    def evaluate_thoughts_parallel(self, query, thoughts, context):
        best_thought = thoughts[0]
        best_score = -1

        for i, t in enumerate(thoughts):
            prompt = f"Ngữ cảnh: {context}\nCâu hỏi: {query}\nHướng giải quyết: '{t}'\nĐánh giá hướng này có đúng ngữ cảnh không. Chấm điểm (1-10). Chỉ xuất ra 1 con số nguyên."
            score_text = self._call_llm(
                prompt, temperature=0.0, max_output_tokens=10
            )
            try:
                score = int("".join(filter(str.isdigit, score_text)))
                if score > best_score:
                    best_score, best_thought = score, thoughts[i]
            except ValueError:
                pass
        return best_thought

    def self_reflect(self, query, best_thought, context):
        prompt = f"Bạn là AI cẩn thận. Ngữ cảnh: {context}\nCâu hỏi: {query}\nCâu trả lời nháp: {best_thought}\nNhiệm vụ: Sửa lại câu trả lời nháp sao cho mượt mà, không bịa đặt thông tin ngoài ngữ cảnh. Trả lời trực tiếp."
        return self._call_llm(prompt, temperature=0.1)

    # ==========================================
    # LOGIC 2: MAP-REDUCE (Đọc Toàn cục)
    # ==========================================
    def map_reduce_summarize(self, global_query, chunks_list=None):
        if chunks_list is None:
            chunks_list = []

        print(
            f"🔄 [Map-Reduce] Bắt đầu đọc và tóm tắt {len(chunks_list)} phần tài liệu..."
        )

        partial_summaries = []
        for i, chunk in enumerate(chunks_list):
            prompt_map = f"Đọc đoạn tài liệu sau:\n{chunk}\n\nHãy tóm tắt ngắn gọn các ý chính liên quan đến: '{global_query}'."
            summary = self._call_llm(
                prompt_map, temperature=0.1, max_output_tokens=2048
            )
            partial_summaries.append(summary)
            print(f"  -> Đã tóm tắt phần {i+1}/{len(chunks_list)}")

        print(
            "🧠 [Map-Reduce] Đang tổng hợp các bản tóm tắt thành bài viết hoàn chỉnh..."
        )
        combined_text = "\n".join(partial_summaries)

        prompt_reduce = f"Dưới đây là các bản tóm tắt từ nhiều phần của một tài liệu lớn:\n{combined_text}\n\nDựa trên các thông tin trên, hãy viết một câu trả lời hoàn chỉnh, mạch lạc cho yêu cầu: '{global_query}'."

        final_answer = self._call_llm(
            prompt_reduce, temperature=0.3, max_output_tokens=2048
        )
        return final_answer


# ==========================================
# 4. KHỞI TẠO CÁC MODULE TOÀN CỤC
# ==========================================
print("\n" + "=" * 60)
print("🚀 KHỞI TẠO HỆ THỐNG CHATBOT COLAB (Vertex AI Edition)")
print("=" * 60 + "\n")

global_knowledge_base = SmartKnowledgeBuilder()

global_agent = None
if GCP_PROJECT_ID:
    global_agent = AdvancedReasoningAgent(
        project_id=GCP_PROJECT_ID,
        location=GCP_LOCATION,
        model_name=GEMINI_MODEL,
    )
else:
    print("⚠️ Cảnh báo: Không tìm thấy GCP_PROJECT_ID! LLM sẽ không hoạt động.")
    print("   → Hãy đặt GCP_PROJECT_ID trong file .env và khởi động lại.")

# Tự động index tài liệu nếu có DOC_PATH và chưa có DB
if DOC_PATH and os.path.exists(DOC_PATH) and global_knowledge_base.vector_db is None:
    print(f"\n📄 Tự động index tài liệu: {DOC_PATH}")
    global_knowledge_base.process_and_save(DOC_PATH)


# ==========================================
# 5. FASTAPI APP & ENDPOINTS
# ==========================================
app = FastAPI(
    title="Hệ thống Vistral ToT RAG API (Vertex AI)",
    description="AI Chatbot với Tree of Thought reasoning + RAG — Powered by Gemini via Vertex AI",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schema ---
class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    final_answer: str
    thoughts_process: list
    context_used: str


class IndexResponse(BaseModel):
    status: str
    message: str
    chunks_count: int


class HealthResponse(BaseModel):
    status: str
    gemini_connected: bool
    model_name: str
    vector_db_ready: bool


# --- Endpoints ---
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Kiểm tra trạng thái hệ thống"""
    return HealthResponse(
        status="ok",
        gemini_connected=global_agent is not None,
        model_name=GEMINI_MODEL,
        vector_db_ready=global_knowledge_base.vector_db is not None,
    )


@app.post("/api/v1/ask", response_model=QueryResponse)
async def ask_vistral(request: QueryRequest):
    """Hỏi chatbot — sử dụng RAG + Tree of Thought + Self-Reflection"""
    try:
        if global_agent is None:
            raise HTTPException(
                status_code=503,
                detail="Gemini API chưa được kết nối. Hãy kiểm tra GCP_PROJECT_ID trong .env",
            )

        query = request.question
        print(f"\n📨 [API] Nhận câu hỏi: '{query}'")

        # 1. Trích xuất kiến thức (RAG)
        context = global_knowledge_base.retrieve_context(query)

        # 2. Suy luận (ToT)
        thoughts = global_agent.generate_thoughts(query, context, num_thoughts=5)
        best_thought = global_agent.evaluate_thoughts_parallel(
            query, thoughts, context
        )

        # 3. Chốt đáp án (Self-Reflection)
        final_answer = global_agent.self_reflect(query, best_thought, context)

        print("✅ [API] Đã xử lý xong, trả kết quả cho Client.")
        return QueryResponse(
            final_answer=final_answer,
            thoughts_process=thoughts,
            context_used=context,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}


@app.post("/api/v1/index", response_model=IndexResponse)
async def index_document(file: UploadFile = File(...)):
    """Upload và index một tài liệu mới vào Vector DB"""
    try:
        # Kiểm tra định dạng file
        _, ext = os.path.splitext(file.filename)
        if ext.lower() not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Định dạng '{ext}' không được hỗ trợ. Chỉ chấp nhận: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        # Lưu file tạm
        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / file.filename

        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        print(f"\n📄 [INDEX] Đã nhận file: {file.filename} ({len(content)} bytes)")

        # Index vào Vector DB
        global_knowledge_base.process_and_save(str(file_path))

        # Đếm số chunks trong DB
        chunks_count = 0
        if global_knowledge_base.vector_db:
            chunks_count = global_knowledge_base.vector_db._collection.count()

        return IndexResponse(
            status="success",
            message=f"Đã index thành công file '{file.filename}'",
            chunks_count=chunks_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================
# 6. MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🟢 MÁY CHỦ ĐÃ SẴN SÀNG!")
    print(f"🔗 API URL:     http://{BACKEND_HOST}:{BACKEND_PORT}/api/v1/ask")
    print(f"📚 Swagger UI:  http://{BACKEND_HOST}:{BACKEND_PORT}/docs")
    print(f"💚 Health:      http://{BACKEND_HOST}:{BACKEND_PORT}/health")
    print(f"🤖 Model:       {GEMINI_MODEL} (via Vertex AI)")
    print("=" * 60 + "\n")

    uvicorn.run(app, host=BACKEND_HOST, port=BACKEND_PORT)
