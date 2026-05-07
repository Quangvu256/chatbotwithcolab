# Mô tả luồng mã (Code Flow)

Tài liệu này tóm tắt luồng chính của dự án ChatBotColab (frontend `chatbot.py` và backend trong `backend.ipynb`).

## Tổng quan
- Frontend: giao diện Streamlit (`chatbot.py`) chạy trên trình duyệt (kết nối trực tiếp API).
- Backend: standalone server (`backend.py`) chạy trên GCE / Docker khởi tạo FastAPI + kết nối mô hình LLM qua Google Cloud Vertex AI.
- Giao tiếp: frontend gửi POST tới `API_URL` (ví dụ `.../api/v1/ask`) và nhận về JSON chứa `final_answer`, `thoughts_process`, `context_used`. Hoặc gửi POST tới `/api/v1/summarize` để tóm tắt.

## Frontend (`chatbot.py`) — luồng xử lý chính
1. Khởi tạo
   - Nạp `.env` (sử dụng `load_dotenv()`), cấu hình `API_URL` mặc định.
   - `st.set_page_config(...)` và dựng phần sidebar để nhập `API_URL` (server IP).
   - Tái sử dụng HTTP connection pool qua `requests.Session()` trong `st.session_state` để tối ưu kết nối.
2. Quản lý trạng thái
   - Dùng `st.session_state.messages` để lưu lịch sử chat vì Streamlit rerun toàn bộ script mỗi lần tương tác.
   - Hiển thị lại các tin nhắn trong `session_state` khi load giao diện.
3. Khi người dùng gửi câu hỏi (via `st.chat_input`)
   - Kiểm tra `API_URL` đã được cấu hình chưa; nếu chưa thì show error.
   - Thêm tin nhắn user vào `session_state.messages` và hiển thị ngay trên UI.
   - Gọi HTTP POST tới `API_URL` với payload `{"question": prompt}` (timeout = 120s).
   - Nếu response 200:
     - Đọc `final_answer`, `thoughts_process`, `context_used` từ JSON.
     - Hiển thị `final_answer` và hai expander: một cho quá trình suy nghĩ (Tree of Thought) và một cho ngữ cảnh RAG.
     - Lưu tin nhắn assistant vào `session_state`.
   - Nếu lỗi kết nối hoặc mã trả về != 200 thì hiển thị lỗi thích hợp.

## Backend (`backend.py`) — luồng xử lý chính (tóm tắt)
1. Môi trường & cài đặt
   - Container Docker tải sẵn model vector và dependencies.
   - Nạp biến môi trường GCP_PROJECT_ID, DATABASE_PATH, DOC_PATH, v.v.
2. Xây dựng RAG
   - Chạy SmartKnowledgeBuilder: load tài liệu, semantic chunking, tạo embedding bằng Vietnamese bi-encoder, lưu vào ChromaDB.
   - Reranker (BGE-M3) để sắp xếp lại các đoạn được truy xuất.
3. Advanced Reasoning (Tree of Thought & Map-Reduce)
   - AdvancedReasoningAgent thực hiện ToT: sinh nhiều hướng suy luận (ví dụ 3 paths), chấm điểm tất cả path trong một lần gọi LLM (batch scoring) tiết kiệm API calls.
   - Self-Reflection: LLM kiểm tra, làm sạch và tinh chỉnh câu trả lời để giảm hallucination.
   - Map-Reduce: Có hàm `map_reduce_summarize` để tóm tắt tài liệu từ nhiều chunks bằng phương pháp chia để trị.
   - Gọi Gemini API qua Vertex AI (có retry logic và timeout tự động).
4. API (FastAPI)
   - Endpoint `/api/v1/ask` nhận `question` (có rate limit 10 requests/phút).
   - Pipeline xử lý: truy xuất vector → rerank → ToT → self-reflection → trả về JSON:
     - `final_answer`: chuỗi kết quả cuối cùng
     - `thoughts_process`: danh sách/lưu trình suy nghĩ (dùng cho UI expander)
     - `context_used`: đoạn trích RAG được sử dụng
   - Endpoint `/api/v1/summarize` cho phép tóm tắt toàn bộ tài liệu hiện có.
5. Expose
   - Docker Compose expose server FastAPI ở port 8000 và Streamlit ở port 8501.

## Biến môi trường quan trọng
- `GCP_PROJECT_ID` — ID dự án trên Google Cloud Platform để gọi Vertex AI.
- `API_URL` — URL public của backend (dán vào frontend hoặc `.env` local).
- `DATABASE_PATH`, `DOC_PATH` — nơi lưu vector DB và đường dẫn tài liệu.

## Dòng thời gian xử lý cho một câu hỏi
1. Người dùng nhập câu hỏi trên Streamlit → frontend gửi POST /api/v1/ask.
2. Backend tìm kiếm vector (top N) → reranker chọn top K.
3. ToT: sinh nhiều hướng suy nghĩ → chấm điểm → chọn hướng tốt nhất.
4. Self-reflection tinh chỉnh câu trả lời.
5. Backend trả về JSON (final_answer, thoughts, context).
6. Frontend hiển thị câu trả lời và phần mở rộng cho quá trình suy nghĩ & ngữ cảnh; lưu lịch sử vào session_state.

## Chạy nhanh (tóm tắt)
- Triển khai Docker: SSH vào GCE VM → `docker compose up -d --build` → chờ model preload ~30s.
- Truy cập vào giao diện web Streamlit thông qua port 8501 của IP máy chủ đó. Dùng port 8000/docs để xem API docs.

---

Nếu bạn muốn, tôi có thể:
- cập nhật README để thêm link tới file này, hoặc
- mở rộng CODE_FLOW.md với sơ đồ ASCII hoặc hình ảnh, hoặc
- viết hướng dẫn debug khi server báo lỗi kết nối.

