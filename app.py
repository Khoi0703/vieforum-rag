import streamlit as st
import requests

API_URL = "http://localhost:8000/chatbot/"

st.set_page_config(page_title="vieforum-rag", layout="wide")
st.title("🤖 vieforum-rag — Local Forum RAG Chatbot")
st.caption(
    "Chạy hoàn toàn nội bộ bằng Ollama — không gọi API bên ngoài, không mất phí."
)

# Dùng requests.Session() để GIỮ cookie session_id giữa các lượt gọi.
# (Bản trước dùng requests.post() rời rạc từng lần -> cookie session_id server set
#  không bao giờ được gửi lại -> lịch sử hội thoại phía server không hoạt động.)
if "http_session" not in st.session_state:
    st.session_state.http_session = requests.Session()

# Sidebar: chọn chế độ truy vấn
mode = st.sidebar.selectbox(
    "Chế độ truy vấn",
    options=["hybrid", "local", "global", "naive"],
    help=(
        "hybrid: Kết hợp vector + graph (khuyên dùng)\n"
        "local: Tìm kiếm cục bộ, câu hỏi cụ thể\n"
        "global: Tổng hợp toàn bộ đồ thị\n"
        "naive: Vector search thuần túy"
    ),
)

if st.sidebar.button("🗑️ Xóa lịch sử"):
    st.session_state.chat_history = []
    st.session_state.http_session = requests.Session()  # session mới -> cookie mới
    st.rerun()

with st.sidebar.expander("ℹ️ Trạng thái hệ thống"):
    if st.button("Kiểm tra kết nối"):
        try:
            r = st.session_state.http_session.get(
                API_URL.replace("/chatbot/", "/health"), timeout=5
            )
            st.json(r.json())
        except Exception as e:
            st.error(f"Không kết nối được server: {e}")

# Khởi tạo lịch sử chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Hiển thị lịch sử hội thoại bằng st.chat_message
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Ô nhập câu hỏi
user_input = st.chat_input("Nhập câu hỏi của bạn...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý (mô hình local có thể chậm hơn API cloud)..."):
            try:
                resp = st.session_state.http_session.post(
                    API_URL,
                    json={"text": user_input, "mode": mode},
                    timeout=180,  # model local trên CPU có thể mất nhiều thời gian hơn
                )
                if resp.status_code == 200:
                    data = resp.json()
                    bot_response = data.get("error") or data.get(
                        "response", "Không có phản hồi."
                    )
                else:
                    bot_response = f"Lỗi {resp.status_code}: {resp.text}"
            except requests.exceptions.ConnectionError:
                bot_response = "❌ Không thể kết nối đến server. Hãy chắc chắn `server.py` đang chạy."
            except requests.exceptions.Timeout:
                bot_response = "⏱️ Server phản hồi quá lâu (model local có thể cần máy mạnh hơn hoặc timeout dài hơn)."
            except Exception as e:
                bot_response = f"❌ Lỗi không xác định: {e}"

        st.markdown(bot_response)

    st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
