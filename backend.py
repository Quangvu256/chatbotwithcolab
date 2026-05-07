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
    POST /api/v1/summarize - Summarize documents (Map-Reduce)
"""

import os
import re
import logging
from pathlib import Path

from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

from google import genai
from google.genai import types

from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ==========================================
# 0. CẤU HÌNH LOGGING
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("chatbot-colab")

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
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")

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
        logger.info("Loading Vector Embedding model...")
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="bkai-foundation-models/vietnamese-bi-encoder",
            model_kwargs={"device": "cpu"},
        )

        logger.info("Loading Reranker model...")
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
            logger.info("Restored Vector DB from %s", self.db_path)

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
        logger.info("Splitting documents using Recursive Chunking...")

        # Đổi từ Semantic sang Recursive để chạy mượt trên CPU
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)
        logger.info("Split complete! Generated %d semantic chunks.", len(chunks))

        logger.info("Embedding %d chunks into ChromaDB...", len(chunks))
        if self.vector_db:
            # Incremental: thêm chunks mới vào DB đã có
            self.vector_db.add_documents(chunks)
        else:
            # Lần đầu: tạo DB mới
            self.vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=self.db_path,
            )
        logger.info("Smart Vector RAG build completed successfully!")

    def retrieve_context(self, query, top_k_vector=15, final_k=5):
        """
        Quy trình lấy thông tin chuẩn:
        1. Lấy 15 đoạn liên quan nhất (Vector Search).
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
        logger.info("Connecting to Gemini API via Vertex AI...")
        logger.info("Project: %s, Location: %s, Model: %s", project_id, location, model_name)

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
            logger.info("Gemini connection successful! Test: %s", test_response.text.strip())
        except Exception as e:
            logger.warning("Failed to connect to Gemini: %s", e)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        before_sleep=lambda retry_state: logger.warning(
            "Gemini API retry %d/3...", retry_state.attempt_number
        ),
    )
    def _call_llm(self, prompt, temperature=0.1, max_output_tokens=2048):
        """Gọi Gemini API qua Vertex AI (with Retry logic)"""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            ),
        )
        return response.text.strip()

    # ==========================================
    # LOGIC 1: TREE OF THOUGHT (Tư duy cục bộ)
    # ==========================================
    def generate_thoughts(self, query, context, num_thoughts=3):
        prompt = f"Ngữ cảnh: {context}\nCâu hỏi: {query}\nHãy đưa ra {num_thoughts} hướng phân tích ngắn gọn và khác biệt để trả lời. Liệt kê bắt đầu bằng 'Hướng 1:', 'Hướng 2:'..."
        try:
            raw_text = self._call_llm(prompt, temperature=0.7)
        except Exception as e:
            logger.error("Error generating thoughts", exc_info=True)
            return ["Lỗi khi sinh hướng suy nghĩ."]
            
        thoughts = [t.strip() for t in raw_text.split("\n") if len(t.strip()) > 10]
        return thoughts[:num_thoughts] if thoughts else [raw_text]

    def evaluate_thoughts_batch(self, query, thoughts, context):
        """Chấm điểm tất cả thoughts trong 1 lần gọi API duy nhất."""
        thoughts_text = "\n".join(
            f"Hướng {i+1}: {t}" for i, t in enumerate(thoughts)
        )
        prompt = (
            f"Ngữ cảnh: {context}\n"
            f"Câu hỏi: {query}\n\n"
            f"Các hướng giải quyết:\n{thoughts_text}\n\n"
            f"Nhiệm vụ: Đánh giá mỗi hướng có phù hợp với ngữ cảnh không. "
            f"Chấm điểm 1-10 cho từng hướng.\n"
            f"Xuất ra ĐÚNG FORMAT: mỗi dòng là 'Hướng X: Y điểm' (Y là số nguyên).\n"
            f"Ví dụ:\nHướng 1: 8 điểm\nHướng 2: 5 điểm"
        )
        
        try:
            raw_text = self._call_llm(prompt, temperature=0.0, max_output_tokens=200)
        except Exception as e:
            logger.error("Error evaluating thoughts", exc_info=True)
            return thoughts[0]
        
        # Parse scores
        best_idx, best_score = 0, -1
        for line in raw_text.strip().split("\n"):
            digits = re.findall(r'\d+', line)
            if len(digits) >= 2:
                idx = int(digits[0]) - 1
                score = min(int(digits[1]), 10)
                if 0 <= idx < len(thoughts) and score > best_score:
                    best_score = score
                    best_idx = idx
        
        return thoughts[best_idx]

    def self_reflect(self, query, best_thought, context):
        prompt = f"Bạn là AI cẩn thận. Ngữ cảnh: {context}\nCâu hỏi: {query}\nCâu trả lời nháp: {best_thought}\nNhiệm vụ: Sửa lại câu trả lời nháp sao cho mượt mà, không bịa đặt thông tin ngoài ngữ cảnh. Trả lời trực tiếp."
        try:
            return self._call_llm(prompt, temperature=0.1)
        except Exception as e:
            logger.error("Error in self reflection", exc_info=True)
            return best_thought

    # ==========================================
    # LOGIC 2: MAP-REDUCE (Đọc Toàn cục)
    # ==========================================
    def map_reduce_summarize(self, global_query, chunks_list=None):
        if chunks_list is None:
            chunks_list = []

        logger.info("[Map-Reduce] Start reading and summarizing %d parts...", len(chunks_list))

        partial_summaries = []
        for i, chunk in enumerate(chunks_list):
            prompt_map = f"Đọc đoạn tài liệu sau:\n{chunk}\n\nHãy tóm tắt ngắn gọn các ý chính liên quan đến: '{global_query}'."
            try:
                summary = self._call_llm(
                    prompt_map, temperature=0.1, max_output_tokens=2048
                )
                partial_summaries.append(summary)
                logger.info("  -> Summarized part %d/%d", i+1, len(chunks_list))
            except Exception as e:
                logger.error("Error in map phase %d", i+1, exc_info=True)

        logger.info("[Map-Reduce] Combining summaries into final response...")
        combined_text = "\n".join(partial_summaries)

        prompt_reduce = f"Dưới đây là các bản tóm tắt từ nhiều phần của một tài liệu lớn:\n{combined_text}\n\nDựa trên các thôngত্তি trên, hãy viết một bài tóm tắt hoàn chỉnh, mạch lạc cho yêu cầu: '{global_query}'."

        try:
            final_answer = self._call_llm(
                prompt_reduce, temperature=0.3, max_output_tokens=2048
            )
            return final_answer
        except Exception as e:
            logger.error("Error in reduce phase", exc_info=True)
            return "Lỗi tổng hợp dữ liệu."


# ==========================================
# 4. KHỞI TẠO CÁC MODULE TOÀN CỤC
# ==========================================
logger.info("=" * 60)
logger.info("🚀 INITIALIZING CHATBOT COLAB SYSTEM (Vertex AI Edition)")
logger.info("=" * 60)

global_knowledge_base = SmartKnowledgeBuilder()

global_agent = None
if GCP_PROJECT_ID:
    global_agent = AdvancedReasoningAgent(
        project_id=GCP_PROJECT_ID,
        location=GCP_LOCATION,
        model_name=GEMINI_MODEL,
    )
else:
    logger.warning("Missing GCP_PROJECT_ID! LLM will not function.")

# Tự động index tài liệu nếu có DOC_PATH và chưa có DB
if DOC_PATH and os.path.exists(DOC_PATH) and global_knowledge_base.vector_db is None:
    logger.info("Auto-indexing document: %s", DOC_PATH)
    global_knowledge_base.process_and_save(DOC_PATH)


# ==========================================
# 5. FASTAPI APP & ENDPOINTS
# ==========================================
app = FastAPI(
    title="Hệ thống Vistral ToT RAG API (Vertex AI)",
    description="AI Chatbot với Tree of Thought reasoning + RAG — Powered by Gemini via Vertex AI",
    version="3.0.0",
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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

class SummarizeRequest(BaseModel):
    query: str

class SummarizeResponse(BaseModel):
    summary: str
    chunks_processed: int

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
@limiter.limit("10/minute")
def ask_vistral(request: Request, body: QueryRequest):
    """Hỏi chatbot — sử dụng RAG + Tree of Thought + Self-Reflection (Sync handler)"""
    try:
        if global_agent is None:
            raise HTTPException(
                status_code=503,
                detail="Gemini API chưa được kết nối. Hãy kiểm tra GCP_PROJECT_ID trong .env",
            )

        query = body.question
        logger.info("📨 [API] Nhận câu hỏi: '%s'", query[:100])

        # 1. Trích xuất kiến thức (RAG)
        context = global_knowledge_base.retrieve_context(query, top_k_vector=15)

        # 2. Suy luận (ToT)
        thoughts = global_agent.generate_thoughts(query, context, num_thoughts=3)
        best_thought = global_agent.evaluate_thoughts_batch(
            query, thoughts, context
        )

        # 3. Chốt đáp án (Self-Reflection)
        final_answer = global_agent.self_reflect(query, best_thought, context)

        logger.info("✅ [API] Xử lý xong, trả kết quả cho Client.")
        return QueryResponse(
            final_answer=final_answer,
            thoughts_process=thoughts,
            context_used=context,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing ask request", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}


@app.post("/api/v1/index", response_model=IndexResponse)
def index_document(file: UploadFile = File(...)):
    """Upload và index một tài liệu mới vào Vector DB (Sync handler)"""
    try:
        # Kiểm tra định dạng file
        _, ext = os.path.splitext(file.filename)
        if ext.lower() not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Định dạng '{ext}' không được hỗ trợ. Chỉ chấp nhận: {', '.join(ALLOWED_EXTENSIONS)}",
            )

        # Sanitize filename: chỉ giữ alphanumeric, dấu chấm, gạch ngang, gạch dưới
        safe_name = re.sub(r'[^\w\-.]', '_', os.path.basename(file.filename))
        if not safe_name or safe_name.startswith('.'):
            raise HTTPException(status_code=400, detail="Tên file không hợp lệ")

        # Lưu file tạm
        upload_dir = Path("./data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = upload_dir / safe_name

        with open(file_path, "wb") as f:
            content = file.file.read()
            f.write(content)

        logger.info("📄 [INDEX] Đã nhận file: %s (%d bytes)", safe_name, len(content))

        # Index vào Vector DB
        global_knowledge_base.process_and_save(str(file_path))

        # Đếm số chunks trong DB
        chunks_count = 0
        if global_knowledge_base.vector_db:
            chunks_count = global_knowledge_base.vector_db._collection.count()

        return IndexResponse(
            status="success",
            message=f"Đã index thành công file '{safe_name}'",
            chunks_count=chunks_count,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing index request", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/v1/summarize", response_model=SummarizeResponse)
def summarize_documents(request: SummarizeRequest):
    """Tóm tắt toàn bộ tài liệu theo yêu cầu (Map-Reduce)"""
    try:
        if global_agent is None:
            raise HTTPException(status_code=503, detail="Gemini API chưa kết nối")
        if not global_knowledge_base.vector_db:
            raise HTTPException(status_code=404, detail="Chưa có tài liệu nào")
        
        # Lấy tất cả chunks từ DB
        all_docs = global_knowledge_base.vector_db.similarity_search(
            request.query, k=50
        )
        chunk_texts = [doc.page_content for doc in all_docs]
        
        summary = global_agent.map_reduce_summarize(request.query, chunk_texts)
        return SummarizeResponse(
            summary=summary,
            chunks_processed=len(chunk_texts)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing summarize request", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

# ==========================================
# 6. MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🟢 SERVER READY!")
    logger.info("🔗 API URL:     http://%s:%d/api/v1/ask", BACKEND_HOST, BACKEND_PORT)
    logger.info("📚 Swagger UI:  http://%s:%d/docs", BACKEND_HOST, BACKEND_PORT)
    logger.info("💚 Health:      http://%s:%d/health", BACKEND_HOST, BACKEND_PORT)
    logger.info("🤖 Model:       %s (via Vertex AI)", GEMINI_MODEL)
    logger.info("=" * 60)

    uvicorn.run(app, host=BACKEND_HOST, port=BACKEND_PORT)
