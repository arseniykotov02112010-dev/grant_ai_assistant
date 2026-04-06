"""
main.py – FastAPI приложение для RAG-сервиса.
Исправлено: устранение повторений, жёсткий промпт, постобработка,
все слои модели на GPU (Tesla T4).
"""

import logging
import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import EMBEDDING_MODEL_NAME, GGUF_MODEL_PATH, SESSION_TIMEOUT
from backend.document_utils import extract_text_from_pdf, extract_text_from_docx
from backend.session_manager import SessionManager
from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

logger = logging.getLogger(__name__)

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


def clean_repetitions(text: str) -> str:
    """
    Удаляет повторяющиеся предложения и длинные подстроки.
    Возвращает текст, обрезанный до первого повторения.
    """
    if not text:
        return text

    # 1. Удаляем последовательные дублирующиеся предложения
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    cleaned_sentences = []
    for sent in sentences:
        sent_clean = sent.strip()
        if len(sent_clean) < 10:
            cleaned_sentences.append(sent)
            continue
        if sent_clean in seen:
            break
        seen.add(sent_clean)
        cleaned_sentences.append(sent)
    text = ' '.join(cleaned_sentences)

    # 2. Ищем повторяющиеся подстроки любой длины (>40 символов)
    pattern = r'(.{40,}?)\1'
    match = re.search(pattern, text)
    if match:
        text = text[:match.start()] + match.group(1)

    return text.strip()


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "embedding_model_loaded": embedding_model is not None,
        "llm_loaded": llm is not None
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename.lower()
    if not (filename.endswith('.pdf') or filename.endswith('.docx')):
        raise HTTPException(status_code=400, detail="Only PDF or DOCX files are allowed")

    try:
        file_bytes = await file.read()
        if len(file_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file")

        if filename.endswith('.pdf'):
            text = extract_text_from_pdf(file_bytes)
        else:
            text = extract_text_from_docx(file_bytes)

        if not text or not text.strip():
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

        # Жёсткий промпт: краткость, запрет повторов
        prompt = f"""Ты — эксперт по грантовой документации. Отвечай на вопрос, используя только контекст.

Правила:
1. Ответ должен быть кратким — одно или два предложения.
2. Не повторяй одну и ту же информацию.
3. Не используй слова "контекст", "документ", "согласно".
4. Если ответа нет, скажи: "Информация отсутствует".

Контекст:
{context}

Вопрос: {request.question}

Ответ (только факты, не более двух предложений):"""

        import time
        start = time.time()

        output = llm(
            prompt,
            max_tokens=128,               # ограничиваем длину
            temperature=0.0,              # минимальная креативность
            echo=False,
            repeat_penalty=1.5,           # жёсткий штраф за повторы
            top_p=0.9,
            stop=["\n\n", "Информация отсутствует", "Ответ:", "Вопрос:"]
        )

        elapsed = time.time() - start
        logger.info(f"LLM generation took {elapsed:.2f} seconds")

        answer = output["choices"][0]["text"].strip()
        answer = clean_repetitions(answer)

        if not answer:
            answer = "Информация отсутствует."

        # Если ответ всё ещё слишком длинный, обрезаем до первого предложения
        if len(answer) > 300:
            first_sentence = re.split(r'(?<=[.!?])\s+', answer)[0]
            if len(first_sentence) > 50:
                answer = first_sentence

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
        n_gpu_layers=36,          # все 36 слоёв модели на GPU
        verbose=False
    )
    logger.info("LLM loaded.")

    logger.info("Initializing session manager...")
    session_manager = SessionManager(embedding_model, session_timeout=SESSION_TIMEOUT)
    logger.info("Startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down...")