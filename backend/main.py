"""
main.py – FastAPI приложение для RAG-сервиса.
Поддерживает загрузку PDF и DOCX.
"""

import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

from backend.config import EMBEDDING_MODEL_NAME, GGUF_MODEL_PATH, SESSION_TIMEOUT
from backend.document_utils import extract_text_from_pdf, extract_text_from_docx
from backend.session_manager import SessionManager
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
import time

logger = logging.getLogger(__name__)

# Глобальные объекты
embedding_model = None
llm = None
session_manager = None

app = FastAPI(title="Grant Assistant RAG API")


class AskRequest(BaseModel):
    session_id: str
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: List[str] = []


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "embedding_model_loaded": embedding_model is not None,
        "llm_loaded": llm is not None
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Проверка расширения
    filename = file.filename.lower()
    if not (filename.endswith('.pdf') or filename.endswith('.docx')):
        raise HTTPException(status_code=400, detail="Only PDF or DOCX files are allowed")

    try:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        # Извлечение текста в зависимости от типа
        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_bytes)
        else:  # .docx
            text = extract_text_from_docx(file_bytes)

        if not text.strip():
            raise HTTPException(status_code=400, detail="Document contains no extractable text")

        session_id = session_manager.create_session(text, file.filename)
        return {"session_id": session_id}

    except Exception as e:
        logger.exception("Error in /upload")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    try:
        engine = session_manager.get_engine(request.session_id)
        if engine is None:
            raise HTTPException(status_code=404, detail="Session not found or expired")

        chunks = engine.retrieve(request.question)
        if not chunks:
            logger.warning(f"No chunks found for question: {request.question[:100]}")
            return AskResponse(answer="Информация по вашему вопросу не найдена в документе.", sources=[])

        context = "\n---\n".join(chunks)

        # Новый промпт с чёткими инструкциями
        prompt = f"""Ты — эксперт по грантовой документации. Отвечай на вопрос, используя только приведённый ниже контекст. Контекст содержит текст документа без служебных пометок.

        Если в контексте нет информации, ответь ровно одной фразой: "Информация отсутствует".

        Ответ должен быть кратким, содержать только факты из контекста. Не добавляй никаких пояснений, не цитируй контекст, не повторяй вопрос. Не используй слова "контекст", "документ", "согласно" и т.п. Только сухой ответ.

        Контекст:
        {context}

        Вопрос: {request.question}

        Ответ:"""

        import time
        start = time.time()

        # Добавляем параметры для предотвращения зацикливания
        output = llm(
            prompt,
            max_tokens=256,
            temperature=0.1,
            echo=False,
            repeat_penalty=1.2,
            stop=["\nОтвет:", "\nВопрос:", "\nКонтекст:", "Информация отсутствует"]
        )

        elapsed = time.time() - start
        logger.info(f"LLM generation took {elapsed:.2f} seconds for session {request.session_id}")

        answer = output["choices"][0]["text"].strip()
        return AskResponse(answer=answer, sources=chunks)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in /ask")
        raise HTTPException(status_code=500, detail=str(e))
@app.on_event("startup")
async def startup_event():
    global embedding_model, llm, session_manager

    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Embedding model loaded.")

    logger.info(f"Loading LLM from {GGUF_MODEL_PATH}...")
    llm = Llama(
        model_path=str(GGUF_MODEL_PATH),
        n_ctx=2048,
        n_gpu_layers=-1,  # все слои на GPU
        verbose=False
    )
    logger.info("LLM loaded.")

    logger.info("Initializing session manager...")
    session_manager = SessionManager(embedding_model, session_timeout=SESSION_TIMEOUT)
    logger.info("Startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")
    # При необходимости закрыть ресурсы (например, LLM)