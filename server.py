"""
Local RAG Chatbot Server — CHẠY HOÀN TOÀN NỘI BỘ, KHÔNG DÙNG API TRẢ PHÍ NÀO.

- LLM: Ollama (chạy local, ví dụ qwen2.5, llama3.1, vinallama...)
- Embedding: Ollama embedding model (ví dụ nomic-embed-text)
- Lưu trữ đồ thị tri thức: NetworkX (file JSON cục bộ trong working_dir),
  KHÔNG cần cài/chạy Neo4j.
- Lưu trữ vector: NanoVectorDB (file JSON cục bộ trong working_dir).

Yêu cầu duy nhất: đã cài và đang chạy Ollama (https://ollama.com) trên máy
(hoặc máy khác trong mạng nội bộ, chỉnh OLLAMA_HOST), và đã `ollama pull`
model LLM + model embedding mong muốn.
"""
from fastapi import FastAPI, Request, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from lightrag import LightRAG, QueryParam
from lightrag.llm import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc
from dotenv import load_dotenv
import os
import time
from uuid import uuid4

load_dotenv()

WORKING_DIR = os.getenv("WORKING_DIR", "./workdir")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b-instruct")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))  # nomic-embed-text = 768
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "4000"))

os.makedirs(WORKING_DIR, exist_ok=True)


async def local_embedding(texts: list[str]):
    return await ollama_embed(texts, embed_model=OLLAMA_EMBED_MODEL, host=OLLAMA_HOST)


embedding_func = EmbeddingFunc(
    embedding_dim=EMBEDDING_DIM,
    max_token_size=8192,
    func=local_embedding,
)

# graph_storage / kv_storage / vector_storage để mặc định (NetworkX + Json + NanoVectorDB)
# => không cần Neo4j, không cần server DB nào khác ngoài Ollama.
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name=OLLAMA_LLM_MODEL,
    llm_model_kwargs={"host": OLLAMA_HOST, "options": {"num_ctx": 8192}},
    embedding_func=embedding_func,
    addon_params={"language": "Tiếng Việt"},
)

app = FastAPI(title="vieforum-rag — Local RAG Chatbot (Ollama, không dùng API ngoài)")

# Cho phép gọi từ Streamlit / frontend khác (điều chỉnh allow_origins khi deploy thật)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VALID_MODES = {"local", "global", "hybrid", "naive"}

# Session lịch sử hội thoại trong RAM, có TTL để tránh phình bộ nhớ vô hạn.
# Với production nhiều worker, nên thay bằng Redis.
session_store: dict[str, dict] = {}


class Query(BaseModel):
    text: str
    mode: str = "hybrid"


async def get_session_id(request: Request, response: Response) -> str:
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid4())
        response.set_cookie(
            key="session_id", value=session_id, httponly=True, samesite="lax"
        )
    return session_id


def _cleanup_expired_sessions():
    now = time.time()
    expired = [
        sid
        for sid, data in session_store.items()
        if now - data["updated_at"] > SESSION_TTL_SECONDS
    ]
    for sid in expired:
        session_store.pop(sid, None)


@app.post("/chatbot/")
async def chatbot(input: Query, session_id: str = Depends(get_session_id)):
    text = (input.text or "").strip()
    if not text:
        return {"error": "text không được để trống"}
    if len(text) > MAX_QUESTION_CHARS:
        return {"error": f"Câu hỏi quá dài (tối đa {MAX_QUESTION_CHARS} ký tự)"}
    if input.mode not in VALID_MODES:
        return {"error": f"mode không hợp lệ, chọn 1 trong {sorted(VALID_MODES)}"}

    _cleanup_expired_sessions()
    session = session_store.get(session_id, {"history": [], "updated_at": time.time()})
    history_messages = session["history"]

    try:
        response = await rag.aquery(
            text,
            param=QueryParam(mode=input.mode),
            history_chat=history_messages,
        )
    except Exception as e:
        # Không để lỗi thô (vd Ollama chưa chạy, model chưa pull) văng thẳng ra client
        return {
            "error": (
                "Lỗi khi truy vấn RAG. Kiểm tra Ollama đã chạy chưa "
                f"(OLLAMA_HOST={OLLAMA_HOST}) và đã `ollama pull {OLLAMA_LLM_MODEL}` "
                f"chưa. Chi tiết: {e}"
            )
        }

    history_messages.append({"user": text, "bot": response})
    session["history"] = history_messages[-MAX_HISTORY_TURNS:]
    session["updated_at"] = time.time()
    session_store[session_id] = session

    return {"response": response}


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "llm_model": OLLAMA_LLM_MODEL,
        "embedding_model": OLLAMA_EMBED_MODEL,
        "ollama_host": OLLAMA_HOST,
        "graph_storage": "NetworkX (local file, không dùng Neo4j)",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
