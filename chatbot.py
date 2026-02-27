import streamlit as st
import requests
import json
import os
from dotenv import load_dotenv

# ==========================================
# 1. CẤU HÌNH KẾT NỐI
# ==========================================
load_dotenv()

# set_page_config PHẢI là lệnh Streamlit đầu tiên
st.set_page_config(page_title="Vistral ToT RAG Assistant", page_icon="🤖", layout="wide")

# Ưu tiên: .env → sidebar input
_default_url = os.getenv("API_URL", "")

with st.sidebar:
    st.header("⚙️ Cấu hình")
    API_URL = st.text_input(
        "API URL (từ Ngrok)",
        value=_default_url,
        placeholder="https://xxx.ngrok-free.dev/api/v1/ask",
        help="Dán link ngrok từ Colab vào đây (giữ đuôi /api/v1/ask)"
    )

st.title("🤖 Trợ lý AI Vistral (Tree of Thought + RAG)")

# ==========================================
# 2. QUẢN LÝ TRÍ NHỚ GIAO DIỆN (SESSION STATE)
# ==========================================
# Streamlit có một đặc điểm: Mỗi khi bạn bấm nút, nó sẽ chạy lại code từ đầu.
# Nên ta phải dùng session_state để lưu lại lịch sử chat, nếu không nó sẽ quên sạch.
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lại các tin nhắn cũ mỗi khi giao diện load lại
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================
# 3. XỬ LÝ KHI NGƯỜI DÙNG NHẬP CÂU HỎI
# ==========================================
# Khung nhập liệu ở dưới cùng màn hình
if prompt := st.chat_input("Hãy hỏi tôi về Data Science..."):

    # Kiểm tra API_URL trước khi gửi
    if not API_URL:
        st.error("⚠️ Chưa cấu hình API URL! Hãy dán link Ngrok vào thanh bên trái.")
    else:
        # In câu hỏi của người dùng ra màn hình và lưu vào state
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Hiển thị hiệu ứng AI đang "suy nghĩ"
        with st.chat_message("assistant"):
            with st.spinner("Đang suy luận Tree of Thought trên A100..."):
                try:
                    # GÓI HÀNG VÀ GỬI ĐI
                    payload = {"question": prompt}
                    response = requests.post(API_URL, json=payload, timeout=120) # Chờ tối đa 2 phút vì ToT chạy lâu

                    if response.status_code == 200:
                        # MỞ GÓI HÀNG NHẬN ĐƯỢC
                        data = response.json()
                        final_answer = data.get("final_answer", "")
                        thoughts = data.get("thoughts_process", [])
                        context = data.get("context_used", "")

                        # 1. In ra câu trả lời chính
                        st.markdown("### 🎯 Trả lời:")
                        st.markdown(final_answer)

                        # 2. In ra hậu trường suy nghĩ (Dùng expander để thu gọn cho đẹp)
                        with st.expander("🧠 Xem quá trình suy nghĩ (Tree of Thought)"):
                            for i, t in enumerate(thoughts):
                                st.write(f"**Hướng {i+1}:** {t}")

                        with st.expander("📚 Xem tài liệu đã trích xuất (RAG Context)"):
                            st.info(context)

                        # Lưu câu trả lời của AI vào lịch sử chat
                        st.session_state.messages.append({"role": "assistant", "content": final_answer})

                    else:
                        st.error(f"Lỗi từ Server: {response.status_code} - {response.text}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Không thể kết nối đến máy chủ Colab. Hãy kiểm tra lại link Ngrok! Lỗi: {e}")
