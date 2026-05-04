"""
main.py – FastAPI приложение для RAG-сервиса (версия для дообученной модели).
Без истории диалога в промпте, с проверкой чисел, обрезкой контекста.
"""

import logging
import re
import time
import asyncio
import hashlib
import threading
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from pydantic import BaseModel
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from backend.config import (
    EMBEDDING_MODEL_NAME,
    GGUF_MODEL_PATH,
    MAX_TOKENS,
    SESSION_TIMEOUT,
    TEMPERATURE,
)
from backend.document_utils import (
    extract_text_from_docx,
    extract_text_from_pdf,
    _is_docx_content,
    _is_pdf_content,
)
from backend.session_manager import SessionManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

embedding_model: Optional[SentenceTransformer] = None
llm: Optional[Llama] = None
llm_lock = threading.Lock()
session_manager: Optional[SessionManager] = None

MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
SOURCE_SNIPPET_CHARS = 200
TOKEN_SAFETY_MARGIN = 64

app = FastAPI(title="Grant Assistant RAG API")


class AskRequest(BaseModel):
    session_id: str
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str] = []


def _sanitize(text: str) -> str:
    """Удаление потенциально опасных символов."""
    if not text:
        return text
    return text.replace("<", "").replace(">", "").replace("</", "")


def build_prompt(context: str, question: str) -> str:
    safe_question = _sanitize(question)
    return (
        "Ты — эксперт по грантовой документации Российского научного фонда.\n"
        "Отвечай на вопрос пользователя, используя ИСКЛЮЧИТЕЛЬНО приведённый ниже контекст.\n"
        "Правила:\n"
        "- Отвечай только точными фактами из контекста.\n"
        "- Если нужно перечислить несколько пунктов, скопируй их точно как в контексте.\n"
        "- Не изменяй числа, даты, проценты, суммы.\n"
        "- Если в контексте нет ответа, скажи ровно: «Информация не найдена» и остановись.\n"
        "- Не добавляй ничего от себя.\n\n"
        "Контекст:\n" + context + "\n\n"
        "Вопрос: " + safe_question + "\n\n"
        "Ответ:"
    )


def _validate_numbers_in_answer(answer: str, chunks: List[str]) -> bool:
    """Проверяет, что все числа в ответе присутствуют хотя бы в одном чанке."""
    nums_in_answer = set(re.findall(r'\b\d+(?:\.\d+)?\b', answer))
    if not nums_in_answer:
        return True
    nums_in_context = set()
    for chunk in chunks:
        nums_in_context.update(re.findall(r'\b\d+(?:\.\d+)?\b', chunk))
    return nums_in_answer.issubset(nums_in_context)


def _remove_source_reference(answer: str) -> str:
    """Удаляет скобочные ссылки типа (Источник: ...)."""
    answer = re.sub(r'\s*\(Источник[^)]*\)', '', answer, flags=re.IGNORECASE)
    return answer.strip()


def _truncate_context(question: str, chunks: List[str], max_context_tokens: int) -> str:
    if llm is None:
        raise RuntimeError("LLM is not loaded")
    max_symbols = max_context_tokens * 2
    working = list(chunks)
    while working and sum(len(c) for c in working) > max_symbols:
        working.pop()
    while True:
        context = "\n---\n".join(working) if working else ""
        prompt = build_prompt(context, question)
        try:
            token_count = len(llm.tokenize(prompt))
        except Exception:
            token_count = int(len(prompt) / 2.0)
        if token_count <= max_context_tokens or len(working) <= 1:
            return context
        working.pop()
    return context if context else (chunks[0] if chunks else "")


def _generate_llm(prompt: str) -> str:
    with llm_lock:
        output = llm(
            prompt,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            repeat_penalty=1.1,
            top_p=0.88,
            echo=False,
            stop=["\n\n", "Информация не найдена"],
        )
    return output["choices"][0]["text"].strip()


@app.get("/health")
async def health_check():
    active = session_manager.get_active_sessions_count() if session_manager else 0
    return {
        "status": "ok",
        "embedding_model_loaded": embedding_model is not None,
        "llm_loaded": llm is not None,
        "active_sessions": active,
    }


@app.post("/upload")
async def upload_document(request: Request, file: UploadFile = File(...)):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Filename is missing")
    filename = file.filename.lower()
    if not (filename.endswith(".pdf") or filename.endswith(".docx")):
        raise HTTPException(status_code=400, detail="Only PDF or DOCX files are allowed")
    try:
        file_bytes = await file.read(MAX_FILE_SIZE_BYTES + 1)
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to read file")
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")
    if filename.endswith(".pdf") and not _is_pdf_content(file_bytes):
        raise HTTPException(status_code=400, detail="Invalid PDF file")
    if filename.endswith(".docx") and not _is_docx_content(file_bytes):
        raise HTTPException(status_code=400, detail="Invalid DOCX file")
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    try:
        if filename.endswith(".pdf"):
            text = await asyncio.to_thread(extract_text_from_pdf, file_bytes)
        else:
            text = await asyncio.to_thread(extract_text_from_docx, file_bytes)
    except Exception:
        logger.exception("Text extraction failed")
        raise HTTPException(status_code=422, detail="Document extraction error")
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Document contains no extractable text")
    try:
        session_id = session_manager.create_session(
            text, file.filename, content_hash=content_hash
        )
    except Exception:
        logger.exception("Session creation failed")
        raise HTTPException(status_code=500, detail="Indexing error")
    logger.info(f"Session {session_id} created for {file.filename} (hash={content_hash[:12]})")
    return {"session_id": session_id}


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    engine = session_manager.get_engine(request.session_id)
    if engine is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    try:
        t0 = time.time()
        chunks = await asyncio.to_thread(engine.retrieve, request.question)
        logger.info(f"Retrieval took {time.time() - t0:.2f}s, found {len(chunks)} chunks")
    except Exception:
        logger.exception("Retrieval error")
        raise HTTPException(status_code=500, detail="Search failed")
    if not chunks:
        answer = "Информация не найдена"
        session_manager.add_message(request.session_id, "user", request.question)
        session_manager.add_message(request.session_id, "assistant", answer)
        return AskResponse(answer=answer, sources=[])
    safe_window = llm.n_ctx() - MAX_TOKENS - TOKEN_SAFETY_MARGIN
    if safe_window < 512:
        safe_window = 512
    context = _truncate_context(request.question, chunks, safe_window)
    prompt = build_prompt(context, request.question)
    try:
        t1 = time.time()
        raw_answer = await asyncio.to_thread(_generate_llm, prompt)
        logger.info(f"LLM generation took {time.time() - t1:.2f}s")
    except Exception:
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=500, detail="Answer generation failed")
    answer = raw_answer.strip()
    answer = _remove_source_reference(answer)
    if answer and answer != "Информация не найдена":
        if not _validate_numbers_in_answer(answer, chunks):
            logger.warning("Ответ содержит числа, отсутствующие в контексте – заменён на 'Информация не найдена'")
            answer = "Информация не найдена"
    if not answer:
        answer = "Информация не найдена"
    session_manager.add_message(request.session_id, "user", request.question)
    session_manager.add_message(request.session_id, "assistant", answer)
    sources_snippets = [
        (s[:SOURCE_SNIPPET_CHARS] + "…") if len(s) > SOURCE_SNIPPET_CHARS else s
        for s in chunks
    ]
    return AskResponse(answer=answer, sources=sources_snippets)


@app.on_event("startup")
async def startup_event():
    global embedding_model, llm, session_manager
    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Embedding model loaded.")
    logger.info(f"Loading LLM from {GGUF_MODEL_PATH}...")
    try:
        llm = Llama(
            model_path=str(GGUF_MODEL_PATH),
            n_ctx=2048,
            n_gpu_layers=36,
            verbose=False,
        )
    except Exception:
        logger.exception("Failed to load LLM")
        raise RuntimeError("Cannot start server without LLM")
    logger.info(f"LLM loaded. n_ctx={llm.n_ctx()}")
    logger.info("Initializing session manager...")
    session_manager = SessionManager(embedding_model, session_timeout=SESSION_TIMEOUT)
    logger.info("Startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    if session_manager:
        session_manager.shutdown()
    logger.info("Shutdown complete.")