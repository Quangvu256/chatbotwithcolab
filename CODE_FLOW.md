# Mô tả luồng mã (Code Flow)

Tài liệu này tóm tắt luồng chính của dự án ChatBotColab (frontend `chatbot.py` và backend trong `backend.ipynb`).

## Tổng quan
- Frontend: giao diện Streamlit (`chatbot.py`) chạy trên máy local.
- Backend: notebook Google Colab (`backend.ipynb`) khởi tạo FastAPI + mô hình LLM, exposed qua Ngrok.
- Giao tiếp: frontend gửi POST tới `API_URL` (ví dụ `.../api/v1/ask`) và nhận về JSON chứa `final_answer`, `thoughts_process`, `context_used`.

## Frontend (`chatbot.py`) — luồng xử lý chính
1. Khởi tạo
   - Nạp `.env` (sử dụng `load_dotenv()`), cấu hình `API_URL` mặc định.
   - `st.set_page_config(...)` và dựng phần sidebar để nhập `API_URL` (Ngrok).
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

## Backend (`backend.ipynb`) — luồng xử lý chính (tóm tắt)
1. Môi trường & cài đặt
   - Mount Google Drive, cài dependencies (transformers, bitsandbytes, chromadb, fastapi, pyngrok, v.v.).
   - Nạp biến môi trường `HF_TOKEN`, `NGROK_TOKEN`, `DATABASE_PATH`, `DOC_PATH`, v.v.
2. Xây dựng RAG
   - Chạy SmartKnowledgeBuilder: load tài liệu, semantic chunking, tạo embedding bằng Vietnamese bi-encoder, lưu vào ChromaDB.
   - Reranker (BGE-M3) để sắp xếp lại các đoạn được truy xuất.
3. Advanced Reasoning (Tree of Thought)
   - AdvancedReasoningAgent thực hiện ToT: sinh nhiều hướng suy luận (ví dụ 5 paths), chấm điểm từng path, chọn path tốt nhất.
   - Self-Reflection: LLM kiểm tra, làm sạch và tinh chỉnh câu trả lời để giảm hallucination.
4. API
   - FastAPI endpoint (ví dụ `/api/v1/ask`) nhận `question`.
   - Pipeline xử lý: truy xuất vector → rerank → ToT → self-reflection → trả về JSON:
     - `final_answer`: chuỗi kết quả cuối cùng
     - `thoughts_process`: danh sách/lưu trình suy nghĩ (dùng cho UI expander)
     - `context_used`: đoạn trích RAG được sử dụng
5. Expose
   - Khởi động Uvicorn + Ngrok để cung cấp `API_URL` công khai.

## Biến môi trường quan trọng
- `HF_TOKEN` — token Hugging Face (tải model).
- `NGROK_TOKEN` — token Ngrok (tạo tunnel).
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
- Backend: mở `backend.ipynb` trên Google Colab → chạy tất cả cell → copy Ngrok URL.
- Frontend (local): cài `streamlit requests python-dotenv` → `streamlit run chatbot.py` → dán Ngrok URL vào sidebar hoặc `.env`.

---

Nếu bạn muốn, tôi có thể:
- cập nhật README để thêm link tới file này, hoặc
- mở rộng CODE_FLOW.md với sơ đồ ASCII hoặc hình ảnh, hoặc
- viết hướng dẫn debug khi server báo lỗi kết nối.

