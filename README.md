# 🇻🇳 vieforum-rag — Local Vietnamese Forum RAG Chatbot

## 📌 Tổng Quan

Pipeline gồm:

1. Thu thập / chuẩn bị dữ liệu diễn đàn (CSV) — xem `notebooks/`
2. Làm sạch và tiền xử lý văn bản tiếng Việt
3. Phân tích dữ liệu khám phá (EDA)
4. Xây dựng hệ thống RAG lai (vector search + knowledge graph) — **chạy 100% local qua Ollama**
5. Giao diện hội thoại đa lượt với bộ nhớ ngữ cảnh

Người dùng có thể đặt câu hỏi tự nhiên như:
- *"Mọi người nói gì về vấn đề kinh tế?"*
- *"Quan điểm chung về chủ đề đó là gì?"*
- *"Tiếp tục nói thêm về điều đó"* ← hệ thống hiểu ngữ cảnh từ lượt trước

---

## 🏗️ Kiến Trúc Hệ Thống

```
CSV Dataset (dữ liệu diễn đàn)
     │
     ▼
[Preprocessing]         Làm sạch, stopword, định dạng ngữ cảnh
     │
     ▼
[EDA]                   Phân tích tần suất từ, WordCloud, chủ đề
     │
     ▼
[LightRAG Indexing]     Ollama Embedding → NanoVectorDB (file local)
     │                  + Knowledge Graph → NetworkX (file local, KHÔNG Neo4j)
     │
     ▼
[Ollama LLM]            Chạy local (qwen2.5 / llama3.1 / vinallama...)
     │
     ▼
[FastAPI Backend]       Xử lý query, session memory (có TTL), hybrid retrieval
     │
     ▼
[Streamlit UI]          Giao diện chat đa lượt
```

**Vì sao chạy được nội bộ hoàn toàn?**
- LLM & embedding gọi tới Ollama qua `http://localhost:11434` — đây là **server chạy ngay trên
  máy bạn**, không phải API cloud, không mất phí, không gửi dữ liệu ra ngoài.
- Đồ thị tri thức dùng `NetworkXStorage` (mặc định có sẵn trong `lightrag/`) — lưu thẳng file
  JSON trong `working_dir`, không cần cài/chạy Neo4j hay bất kỳ database server nào.
- Vector index dùng `NanoVectorDB` — cũng là file JSON local.
- `lightrag/llm.py` đã được chỉnh để **không bắt buộc cài** `torch`, `transformers`, `openai`,
  `aioboto3` nữa (các thư viện này chỉ cần khi dùng backend HuggingFace-local/OpenAI/Bedrock,
  không dùng ở bản Ollama-only này) → cài đặt nhẹ và nhanh hơn nhiều.

> ⚠️ Lưu ý duy nhất về mạng: `tiktoken` cần tải một bảng mã hoá (~2MB) từ internet **một lần**
> ở lần chạy đầu tiên (không phải mỗi lần chạy, không liên quan tới việc trả lời câu hỏi). Nếu
> máy chạy hoàn toàn cách ly mạng, hãy chạy lần đầu ở máy có mạng rồi copy cache của `tiktoken`
> (`~/.cache/tiktoken` hoặc biến `TIKTOKEN_CACHE_DIR`) sang máy offline.

---

## ⚙️ Chế Độ Truy Vấn

| Chế độ | Mô tả |
|--------|-------|
| `local` | Tìm kiếm theo ngữ nghĩa cục bộ, phù hợp câu hỏi cụ thể |
| `global` | Tổng hợp thông tin toàn bộ đồ thị, phù hợp câu hỏi tổng quan |
| `hybrid` | Kết hợp local + global, cân bằng tốt nhất (mặc định) |
| `naive` | Vector search thuần túy, không dùng graph — nhanh nhất |

---

## 🛠️ Công Nghệ Sử Dụng

| Thành phần | Công nghệ | Ghi chú |
|---|---|---|
| LLM | **Ollama** (qwen2.5 / llama3.1 / vinallama...) | Chạy local, đổi model tuỳ ý |
| Embedding | **Ollama** (nomic-embed-text / mxbai-embed-large) | Chạy local |
| RAG framework | `LightRAG` (trong `lightrag/`) | Đã patch để không cần torch/transformers |
| Vector storage | `NanoVectorDB` | File JSON local |
| Graph storage | `NetworkX` | File JSON local — **không cần Neo4j** |
| API backend | `FastAPI` + `Uvicorn` | |
| Frontend | `Streamlit` | |
| Xử lý dữ liệu | `Pandas`, `Regex` | |

---

## 📂 Cấu Trúc Thư Mục

```
LightRAG/
├── lightrag/                  # Core LightRAG library
│   ├── lightrag.py            # Main RAG engine
│   ├── llm.py                 # LLM/embedding integrations (đã patch lazy-import)
│   ├── operate.py             # Graph operations
│   ├── prompt.py              # Prompt templates
│   ├── storage.py             # Storage backends (NetworkX, NanoVectorDB, Json)
│   └── kg/                    # Backend đồ thị khác (Neo4j, Chroma...) - KHÔNG dùng mặc định
├── notebooks/
│   ├── ScrapeData.ipynb
│   └── FINAL_NOTEBOOK.ipynb
├── app.py                     # Streamlit frontend (đã sửa giữ session cookie)
├── server.py                  # FastAPI backend (dùng Ollama, không Neo4j)
├── requirements.txt           # Đã bỏ torch/transformers/neo4j, thêm ollama
├── .env.example
└── LICENSE
```

---

## 🚀 Hướng Dẫn Cài Đặt

### Yêu Cầu Hệ Thống

- Python 3.9+
- **Ollama** đã cài (https://ollama.com/download) — Windows/Mac/Linux đều có
- RAM tối thiểu 8GB (16GB nếu chạy model 7B trên CPU sẽ mượt hơn); có GPU thì càng nhanh
- Không cần Docker, không cần Neo4j, không cần API key nào

### 1. Cài Ollama và tải model

```bash
# Cài Ollama: xem hướng dẫn tại https://ollama.com/download

# Tải model LLM (chọn 1, tuỳ cấu hình máy):
ollama pull qwen2.5:7b-instruct     # cân bằng tốt, hiểu tiếng Việt khá
# ollama pull llama3.1:8b
# ollama pull vinallama-7b-chat     # nếu có sẵn trên Ollama registry

# Tải model embedding:
ollama pull nomic-embed-text

# Chạy Ollama server (thường tự chạy nền sau khi cài, hoặc chạy tay):
ollama serve
```

### 2. Clone và cài dependencies Python

```bash
git clone https://github.com/<your-username>/vieforum-rag.git
cd vieforum-rag
pip install -r requirements.txt
```

### 3. Cấu hình biến môi trường

```bash
cp .env.example .env
```

Không cần điền API key nào — chỉnh nếu muốn đổi model hoặc địa chỉ Ollama:

```env
WORKING_DIR=./workdir
OLLAMA_HOST=http://localhost:11434
OLLAMA_LLM_MODEL=qwen2.5:7b-instruct
OLLAMA_EMBED_MODEL=nomic-embed-text
EMBEDDING_DIM=768
```

### 4. Chạy Backend (FastAPI)

```bash
python server.py
# API chạy tại: http://localhost:8000
# Kiểm tra nhanh: curl http://localhost:8000/health
```

### 5. Chạy Frontend (Streamlit)

```bash
streamlit run app.py
# UI chạy tại: http://localhost:8501
```

---

## 📊 Thu Thập & Chuẩn Bị Dữ Liệu

Xem `notebooks/ScrapeData.ipynb` để cào dữ liệu từ diễn đàn (nếu cần) và xuất CSV. Sau đó xem
`notebooks/FINAL_NOTEBOOK.ipynb` để tiền xử lý, EDA và insert dữ liệu vào LightRAG.

> Dataset gốc (CSV) không đi kèm repo này — bạn cần tự chuẩn bị file CSV theo đúng cấu trúc cột
> mà notebook kỳ vọng (User, Reply_Content, Reply_To, Original_Comment, Title).

---

## 🔗 API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `POST` | `/chatbot/` | Gửi câu hỏi, nhận trả lời |
| `GET` | `/health` | Kiểm tra cấu hình đang chạy (model, host...) |

**Request body `/chatbot/`:**
```json
{
  "text": "Mọi người nghĩ gì về vấn đề X?",
  "mode": "hybrid"
}
```

**Response thành công:**
```json
{ "response": "Dựa trên các cuộc thảo luận..." }
```

**Response khi lỗi** (vd Ollama chưa chạy, input không hợp lệ):
```json
{ "error": "Lỗi khi truy vấn RAG. Kiểm tra Ollama đã chạy chưa (...)" }
```

---

## 🧪 Đánh giá (Evaluation)

Repo gốc chưa có bộ đánh giá số liệu nào. Có thể dùng script `run_eval.py` (đo Precision@k,
Recall@k, MRR trên tập câu hỏi có ground-truth tự định nghĩa) làm điểm khởi đầu, thay embedding/
LLM giả lập trong script đó bằng `ollama_embed`/`ollama_model_complete` thật để có số liệu phản
ánh đúng hệ thống local này.

---

## ⚠️ Lưu Ý

- **Session memory** lưu in-memory kèm TTL (mặc định 1 giờ không hoạt động sẽ bị dọn), sẽ mất
  khi restart server. Với production nhiều người dùng/nhiều worker, nên dùng Redis.
- Model local trên CPU sẽ **chậm hơn** API cloud — nếu có GPU, Ollama sẽ tự dùng để tăng tốc.
- Chất lượng câu trả lời phụ thuộc vào model bạn chọn (`qwen2.5:7b` sẽ khác `llama3.1:70b`) —
  nên thử vài model để chọn cái phù hợp giữa tốc độ và chất lượng cho máy của bạn.
- Dữ liệu diễn đàn thu thập được chỉ nên dùng cho mục đích nghiên cứu và học thuật, tuân thủ
  điều khoản sử dụng của nguồn dữ liệu.

---

## 📄 License

MIT License — xem file [LICENSE](./LICENSE)
