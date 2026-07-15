"""
ChatBotColab — Standalone Backend Server (Google AI Studio Edition)
====================================================================
Production deployment using Gemini API via Google AI Studio API Key.
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
import json
from pathlib import Path
import networkx as nx

from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

from google import genai
from google.genai import types

from fastapi import FastAPI, HTTPException, UploadFile, File, Request, BackgroundTasks
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")

# Tạo thư mục data nếu chưa có
os.makedirs("./data/documents", exist_ok=True)
os.makedirs("./data/uploads", exist_ok=True)


# ==========================================
# 1.5. GRAPH RAG BUILDER
# ==========================================
class GraphRAGBuilder:
    """
    Quản lý đồ thị tri thức (Knowledge Graph) sử dụng NetworkX.
    Trích xuất thực thể/quan hệ thông qua Gemini API và thực hiện
    Entity Linking bằng ChromaDB.
    """
    def __init__(self, db_path="./data/vector_db", graph_path="./data/graph.json", embedding_model=None, client=None, model_name="gemini-2.0-flash"):
        self.graph_path = graph_path
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.model_name = model_name
        self.graph = nx.DiGraph()
        
        # Cấu hình Gemini client riêng để tránh phụ thuộc thứ tự khởi tạo
        if not client and GEMINI_API_KEY:
            self.client = genai.Client(api_key=GEMINI_API_KEY)
        else:
            self.client = client
        
        # Tạo Chroma collection riêng cho Entity Linking
        self.entities_collection = None
        self._init_entities_collection()
        self.load_graph()

    def _init_entities_collection(self):
        """Khởi tạo collection lưu tên thực thể phục vụ Entity Linking"""
        try:
            self.entities_collection = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embedding_model,
                collection_name="graph_entities"
            )
            logger.info("Initialized Graph Entities collection for Entity Linking")
        except Exception as e:
            logger.error("Failed to initialize Graph Entities collection: %s", e)

    def load_graph(self):
        """Khôi phục đồ thị từ file JSON và chuyển list thành set"""
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data)
                
                # Khôi phục các sets từ lists để sử dụng trong code
                for node, attrs in self.graph.nodes(data=True):
                    if "chunks" in attrs and isinstance(attrs["chunks"], list):
                        attrs["chunks"] = set(attrs["chunks"])
                for u, v, attrs in self.graph.edges(data=True):
                    if "chunks" in attrs and isinstance(attrs["chunks"], list):
                        attrs["chunks"] = set(attrs["chunks"])
                        
                logger.info("Restored Graph DB from %s with %d nodes and %d edges", 
                            self.graph_path, self.graph.number_of_nodes(), self.graph.number_of_edges())
            except Exception as e:
                logger.error("Failed to load Graph DB from %s: %s", self.graph_path, e)
                self.graph = nx.DiGraph()
        else:
            self.graph = nx.DiGraph()

    def save_graph(self):
        """Lưu đồ thị thành file JSON và chuyển set thành list để serialize"""
        try:
            os.makedirs(os.path.dirname(self.graph_path), exist_ok=True)
            
            # Sao chép đồ thị để chuyển đổi set thành list trước khi lưu
            temp_graph = self.graph.copy()
            for node, attrs in temp_graph.nodes(data=True):
                if "chunks" in attrs and isinstance(attrs["chunks"], set):
                    attrs["chunks"] = list(attrs["chunks"])
            for u, v, attrs in temp_graph.edges(data=True):
                if "chunks" in attrs and isinstance(attrs["chunks"], set):
                    attrs["chunks"] = list(attrs["chunks"])
                    
            data = nx.node_link_data(temp_graph)
            with open(self.graph_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info("Saved Graph DB to %s with %d nodes and %d edges", 
                        self.graph_path, self.graph.number_of_nodes(), self.graph.number_of_edges())
        except Exception as e:
            logger.error("Failed to save Graph DB to %s: %s", self.graph_path, e)

    def extract_triplets(self, text_chunk: str) -> list:
        """Sử dụng Gemini API để trích xuất danh sách thực thể và quan hệ từ văn bản"""
        if not self.client:
            logger.warning("Gemini Client is not configured in GraphRAGBuilder")
            return []

        prompt = f"""Bạn là một chuyên gia về trích xuất Đồ thị tri thức (Knowledge Graph).
Nhiệm vụ của bạn là đọc đoạn văn bản tiếng Việt sau và trích xuất tất cả các thực thể (entities) cùng với mối quan hệ (relationships) giữa chúng.

Đoạn văn bản:
\"\"\"
{text_chunk}
\"\"\"

Yêu cầu xuất ra định dạng JSON duy nhất, có cấu trúc như sau (KHÔNG viết thêm bất kỳ lời dẫn nào ngoài JSON):
{{
  "triplets": [
    {{
      "subject": "Tên thực thể 1",
      "subject_type": "Loại thực thể (ví dụ: Người, Tổ chức, Địa điểm, Khái niệm, ...)",
      "relation": "Mối quan hệ ngắn gọn (ví dụ: là giám đốc của, thành lập vào, nằm ở, ...)",
      "object": "Tên thực thể 2",
      "object_type": "Loại thực thể 2"
    }}
  ]
}}

Lưu ý quan trọng:
1. Trích xuất chính xác từ văn bản, không tự bịa đặt thông tin.
2. Tên thực thể nên chuẩn hóa ngắn gọn (ví dụ: thay vì "ông Nguyễn Văn A" hãy dùng "Nguyễn Văn A").
3. Nếu không có thực thể nào, trả về danh sách triplets trống.
"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=15000,
                    temperature=0.2,
                    response_mime_type="application/json"
                ),
            )
            if response.text:
                result = json.loads(response.text.strip())
                return result.get("triplets", [])
        except Exception as e:
            logger.error("Error extracting triplets from text chunk: %s", e)
        return []

    def add_to_graph(self, triplets: list, chunk_id: str):
        """Thêm các bộ ba (triplets) vào đồ thị và cập nhật Entity Linking collection"""
        new_entities = []
        for trip in triplets:
            subj = trip.get("subject", "").strip()
            obj = trip.get("object", "").strip()
            rel = trip.get("relation", "").strip()
            subj_type = trip.get("subject_type", "Chưa rõ").strip()
            obj_type = trip.get("object_type", "Chưa rõ").strip()

            if not subj or not obj or not rel:
                continue

            # Thêm Node Subject
            if not self.graph.has_node(subj):
                self.graph.add_node(subj, type=subj_type, chunks={chunk_id})
                new_entities.append(subj)
            else:
                self.graph.nodes[subj].setdefault("chunks", set()).add(chunk_id)

            # Thêm Node Object
            if not self.graph.has_node(obj):
                self.graph.add_node(obj, type=obj_type, chunks={chunk_id})
                new_entities.append(obj)
            else:
                self.graph.nodes[obj].setdefault("chunks", set()).add(chunk_id)

            # Thêm hoặc cập nhật Edge
            if self.graph.has_edge(subj, obj):
                existing_rels = self.graph.edges[subj, obj].get("relations", [])
                if rel not in existing_rels:
                    existing_rels.append(rel)
                    self.graph.edges[subj, obj]["relations"] = existing_rels
                self.graph.edges[subj, obj].setdefault("chunks", set()).add(chunk_id)
            else:
                self.graph.add_edge(subj, obj, relations=[rel], chunks={chunk_id})

        # Loại bỏ các thực thể trùng lặp trong đợt thêm này
        new_entities = list(set(new_entities))

        # Lưu các thực thể mới vào ChromaDB để tìm kiếm thực thể bằng GPU
        if new_entities and self.entities_collection:
            try:
                self.entities_collection.add_texts(
                    texts=new_entities,
                    ids=new_entities,
                    metadatas=[{"name": name} for name in new_entities]
                )
                logger.info("Added %d new entities to ChromaDB for Entity Linking", len(new_entities))
            except Exception as e:
                logger.error("Failed to add entities to ChromaDB: %s", e)

    def retrieve_graph_context(self, query: str, top_k_entities: int = 5) -> str:
        """Truy xuất các quan hệ liên quan đến thực thể trong câu hỏi bằng GPU embeddings"""
        if not self.graph or self.graph.number_of_nodes() == 0:
            return ""

        matched_entities = []
        if self.entities_collection:
            try:
                # Kiểm tra số lượng phần tử trước khi search để tránh lỗi Chroma trống
                if self.entities_collection._collection.count() > 0:
                    results = self.entities_collection.similarity_search(query, k=top_k_entities)
                    matched_entities = [doc.page_content for doc in results]
                    logger.info("Entity Linking matched: %s", matched_entities)
            except Exception as e:
                logger.error("Error during Entity Linking similarity search: %s", e)

        # Fallback: tìm kiếm từ khóa trực tiếp nếu Chroma rỗng hoặc lỗi
        if not matched_entities:
            for node in self.graph.nodes:
                if node.lower() in query.lower():
                    matched_entities.append(node)
            matched_entities = matched_entities[:top_k_entities]

        if not matched_entities:
            return ""

        facts = []
        visited_edges = set()

        for entity in matched_entities:
            if not self.graph.has_node(entity):
                continue
            
            entity_type = self.graph.nodes[entity].get("type", "Chưa rõ")
            # Tạo dòng fact chính về loại thực thể
            facts.append(f"- '{entity}' là một '{entity_type}'")

            # Lấy các quan hệ đi ra (out-edges)
            for neighbor in self.graph.successors(entity):
                edge_data = self.graph.edges[entity, neighbor]
                relations = edge_data.get("relations", [])
                for rel_str in relations:
                    edge_key = (entity, neighbor, rel_str)
                    if edge_key not in visited_edges:
                        facts.append(f"- '{entity}' -> {rel_str} -> '{neighbor}'")
                        visited_edges.add(edge_key)

            # Lấy các quan hệ đi vào (in-edges)
            for predecessor in self.graph.predecessors(entity):
                edge_data = self.graph.edges[predecessor, entity]
                relations = edge_data.get("relations", [])
                for rel_str in relations:
                    edge_key = (predecessor, entity, rel_str)
                    if edge_key not in visited_edges:
                        facts.append(f"- '{predecessor}' -> {rel_str} -> '{entity}'")
                        visited_edges.add(edge_key)

        return "\n".join(facts)


# ==========================================
# 2. SMART KNOWLEDGE BUILDER (RAG Engine)
# ==========================================
class SmartKnowledgeBuilder:
    """
    Lõi RAG: Vector Search + Graph Triples + Reranker chạy trên GPU.
    Chuyển đổi tài liệu thành vector embeddings và truy xuất ngữ cảnh lai.
    """

    def __init__(self, db_path=DATABASE_PATH):
        self.db_path = db_path

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("⚡ RAG Engine Device: %s (GPU CUDA available: %s)", device, torch.cuda.is_available())
        if torch.cuda.is_available():
            logger.info("🎮 GPU Device Name: %s", torch.cuda.get_device_name(0))

        logger.info("Loading Vector Embedding model on %s...", device)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="bkai-foundation-models/vietnamese-bi-encoder",
            model_kwargs={"device": device},
            encode_kwargs={"batch_size": 32},
        )

        logger.info("Loading Reranker model on %s...", device)
        self.reranker = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            max_length=15000,
            device=device,
        )

        self.vector_db = None
        self._load_existing_db()

        # Khởi tạo Graph RAG Builder
        GRAPH_PATH = os.path.join(os.path.dirname(self.db_path), "graph.json")
        logger.info("Initializing Graph RAG Builder (Graph path: %s)...", GRAPH_PATH)
        self.graph_db = GraphRAGBuilder(
            db_path=self.db_path,
            graph_path=GRAPH_PATH,
            embedding_model=self.embedding_model,
            model_name=GEMINI_MODEL
        )

    def _load_existing_db(self):
        """Khôi phục lại DB nếu đã có trên ổ cứng"""
        if os.path.exists(self.db_path) and os.listdir(self.db_path):
            self.vector_db = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embedding_model,
            )
            logger.info("Restored Vector DB from %s", self.db_path)

    def process_and_save(self, file_path):
        """Lưu trữ tài liệu vào cả Vector DB và Graph DB"""
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
        logger.info("Vector RAG chunking and indexing completed.")

        # Trích xuất và thêm các thực thể/quan hệ vào đồ thị tri thức (gộp chunk để tăng tốc)
        logger.info("Extracting triplets for Graph RAG using Gemini API (with Super-Chunking)...")
        super_chunk_size = 4
        total_batches = (len(chunks) + super_chunk_size - 1) // super_chunk_size
        
        for i in range(0, len(chunks), super_chunk_size):
            batch_chunks = chunks[i : i + super_chunk_size]
            merged_text = "\n\n".join([c.page_content for c in batch_chunks])
            chunk_id = f"{os.path.basename(file_path)}_superchunk_{i // super_chunk_size}"
            
            triplets = self.graph_db.extract_triplets(merged_text)
            if triplets:
                self.graph_db.add_to_graph(triplets, chunk_id)
                
            logger.info("  -> Processed graph extraction for batch %d/%d (chunks %d-%d)", 
                        (i // super_chunk_size) + 1, 
                        total_batches,
                        i, min(i + super_chunk_size, len(chunks)) - 1)
        
        # Lưu đồ thị xuống ổ cứng
        self.graph_db.save_graph()
        logger.info("Smart Hybrid Graph + Vector RAG build completed successfully!")

    def retrieve_context(self, query, top_k_vector=15, final_k=5):
        """
        Quy trình lấy thông tin chuẩn (Lai):
        1. Lấy 15 đoạn liên quan nhất từ Vector Search (Chroma).
        2. Lấy các quan hệ/thực thể liên quan từ Graph Search (NetworkX + GPU Entity Linking).
        3. Ghép chung cả 2 loại ngữ cảnh lại.
        4. Dùng Reranker chạy trên GPU để chấm điểm và chọn ra `final_k` kết quả tốt nhất.
        """
        vector_chunks = []
        if self.vector_db:
            try:
                raw_docs = self.vector_db.similarity_search(query, k=top_k_vector)
                vector_chunks = [doc.page_content for doc in raw_docs]
            except Exception as e:
                logger.error("Error during vector similarity search: %s", e)

        # 2. Graph Search (Lấy các facts từ đồ thị)
        graph_facts_str = self.graph_db.retrieve_graph_context(query, top_k_entities=5)
        # Tách các dòng facts ra thành danh sách để rerank
        graph_facts = [line.strip() for line in graph_facts_str.split("\n") if line.strip().startswith("-")]

        logger.info("Retrieved %d vector chunks and %d graph facts", len(vector_chunks), len(graph_facts))

        # Gộp chung candidates
        candidates = vector_chunks + graph_facts
        if not candidates:
            return "Không tìm thấy thông tin liên quan."

        # 3. Reranking bằng CrossEncoder chạy trên GPU (CUDA)
        pairs = [[query, candidate] for candidate in candidates]
        scores = self.reranker.predict(pairs)

        paired_results = list(zip(candidates, scores))
        paired_results.sort(key=lambda x: x[1], reverse=True)
        
        # Chọn ra các candidates tốt nhất
        best_candidates = [item[0] for item in paired_results[:final_k]]

        # Trả về ngữ cảnh kết hợp
        return "\n---\n".join(best_candidates)


# ==========================================
# 3. ADVANCED REASONING AGENT (Google AI Studio Gemini + ToT)
# ==========================================
class AdvancedReasoningAgent:
    """
    Lõi suy luận: Gemini 2.0 Flash (via Google AI Studio) + Tree of Thought + Self-Reflection.
    Không cần GPU — gọi API qua Google AI Studio API Key.
    """

    def __init__(self, api_key, model_name="gemini-2.0-flash"):
        logger.info("Connecting to Gemini API via Google AI Studio...")
        logger.info("Model: %s", model_name)

        self.model_name = model_name
        self.client = genai.Client(
            api_key=api_key,
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
    def _call_llm(self, prompt, temperature=0.1, max_output_tokens=15000
    ):
        """Gọi Gemini API qua Google AI Studio (with Retry logic)"""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_output_tokens,
                temperature=temperature,
            ),
        )
        
        # Xử lý trường hợp response.text bị None (do Safety Filters của Google chặn)
        if response.text is not None:
            return response.text.strip()
            
        logger.warning("LLM response text is None. Candidate finish reason: %s", 
                       response.candidates[0].finish_reason if response.candidates else "Unknown")
        return "Xin lỗi, nội dung bị bộ lọc an toàn của Google từ chối trả lời."
    # ==========================================
    # LOGIC 1: TREE OF THOUGHT (Tư duy cục bộ)
    # ==========================================
    def generate_thoughts(self, query, context, num_thoughts=4):
        prompt = f"Ngữ cảnh: {context}\nCâu hỏi: {query}\nHãy đưa ra {num_thoughts} hướng phân tích ngắn gọn và khác biệt để trả lời. KHÔNG viết lời mở đầu hay dẫn nhập. Viết NGAY LẬP TỨC dưới dạng danh sách:\nHướng 1: ...\nHướng 2: ...\nHướng 3: ...\nHướng 4: ..."
        try:
            raw_text = self._call_llm(prompt, temperature=0.7)
        except Exception as e:
            logger.error("Error generating thoughts", exc_info=True)
            return ["Lỗi khi sinh hướng suy nghĩ."]
            
        thoughts = []
        for line in raw_text.split("\n"):
            line = line.strip()
            if line.lower().startswith("hướng") and len(line) > 10:
                thoughts.append(line)
                
        # Fallback nếu AI không dùng từ "Hướng"
        if not thoughts:
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
            raw_text = self._call_llm(prompt, temperature=0.0, max_output_tokens=15000)
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
        prompt = (
            f"Bạn là AI cẩn thận. Ngữ cảnh: {context}\n"
            f"Câu hỏi: {query}\n"
            f"Câu trả lời nháp: {best_thought}\n"
            f"Nhiệm vụ: Sửa lại câu trả lời nháp sao cho mượt mà, không bịa đặt thông tin ngoài ngữ cảnh. "
            f"Hãy viết câu trả lời chi tiết và đầy đủ thông tin, độ dài khoảng 500 từ tiếng Việt. "
            f"Trả lời trực tiếp vào nội dung."
        )
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
                    prompt_map, temperature=0.1, max_output_tokens=15000
                )
                partial_summaries.append(summary)
                logger.info("  -> Summarized part %d/%d", i+1, len(chunks_list))
            except Exception as e:
                logger.error("Error in map phase %d", i+1, exc_info=True)

        logger.info("[Map-Reduce] Combining summaries into final response...")
        combined_text = "\n".join(partial_summaries)

        prompt_reduce = (
            f"Dưới đây là các bản tóm tắt từ nhiều phần của một tài liệu lớn:
{combined_text}

"
            f"Dựa trên các thông tin trên, hãy viết một bài tóm tắt hoàn chỉnh, mạch lạc cho yêu cầu: '{global_query}'. "
            f"Độ dài bài tóm tắt khoảng 500 từ tiếng Việt."
        )

        try:
            final_answer = self._call_llm(
                prompt_reduce, temperature=0.3, max_output_tokens=15000
            )
            return final_answer
        except Exception as e:
            logger.error("Error in reduce phase", exc_info=True)
            return "Lỗi tổng hợp dữ liệu."


# ==========================================
# 4. KHỞI TẠO CÁC MODULE TOÀN CỤC
# ==========================================
logger.info("=" * 60)
logger.info("🚀 INITIALIZING CHATBOT COLAB SYSTEM (Google AI Studio Edition)")
logger.info("=" * 60)

global_knowledge_base = SmartKnowledgeBuilder()

global_agent = None
if GEMINI_API_KEY:
    global_agent = AdvancedReasoningAgent(
        api_key=GEMINI_API_KEY,
        model_name=GEMINI_MODEL,
    )
else:
    logger.warning("Missing GEMINI_API_KEY! LLM will not function. Get one at https://aistudio.google.com/apikey")

# Tự động index tài liệu nếu có DOC_PATH và chưa có DB
if DOC_PATH and os.path.exists(DOC_PATH) and global_knowledge_base.vector_db is None:
    logger.info("Auto-indexing document: %s", DOC_PATH)
    global_knowledge_base.process_and_save(DOC_PATH)


# ==========================================
# 5. FASTAPI APP & ENDPOINTS
# ==========================================
app = FastAPI(
    title="Hệ thống Vistral ToT RAG API (Google AI Studio)",
    description="AI Chatbot với Tree of Thought reasoning + RAG — Powered by Gemini via Google AI Studio",
    version="3.1.0",
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
                detail="Gemini API chưa được kết nối. Hãy kiểm tra GEMINI_API_KEY trong .env",
            )

        query = body.question
        logger.info("📨 [API] Nhận câu hỏi: '%s'", query[:100])

        # 1. Trích xuất kiến thức (RAG)
        context = global_knowledge_base.retrieve_context(query, top_k_vector=15)

        # 2. Suy luận (ToT)
        thoughts = global_agent.generate_thoughts(query, context, num_thoughts=4)
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
def index_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload và index một tài liệu mới vào Vector DB (Sync handler chạy nền)"""
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

        # Thêm việc index vào BackgroundTasks để tránh timeout
        background_tasks.add_task(global_knowledge_base.process_and_save, str(file_path))

        # Đếm số chunks hiện tại trong DB
        chunks_count = 0
        if global_knowledge_base.vector_db:
            chunks_count = global_knowledge_base.vector_db._collection.count()

        return IndexResponse(
            status="success",
            message=f"Tài liệu '{safe_name}' đang được lập chỉ mục (Vector & Graph RAG) trong nền. Bạn có thể kiểm tra logs hoặc bắt đầu đặt câu hỏi.",
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
    logger.info("🤖 Model:       %s (via Google AI Studio)", GEMINI_MODEL)
    logger.info("=" * 60)

    uvicorn.run(app, host=BACKEND_HOST, port=BACKEND_PORT)
