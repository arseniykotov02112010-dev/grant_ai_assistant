"""
rag_engine.py – RAG-движок для индексации документов и поиска релевантных чанков.
Теперь принимает готовый текст (извлечённый из любого формата).
"""

import logging
import uuid
from typing import List, Optional

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from backend.config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    RAG-движок для одного документа.
    При инициализации получает модель эмбеддингов (глобальную).
    """

    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.client = chromadb.EphemeralClient()
        self.collection: Optional[chromadb.Collection] = None
        self.document_text: Optional[str] = None

    def index_document(self, text: str) -> None:
        """
        Индексирует текст документа:
        1. Разбивает на чанки.
        2. Вычисляет эмбеддинги чанков.
        3. Сохраняет чанки в коллекцию ChromaDB.
        """
        if not text.strip():
            raise ValueError("Текст документа пуст")
        self.document_text = text
        logger.info(f"Текст получен, длина: {len(text)} символов")

        # Разбивка на чанки
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(text)
        logger.info(f"Получено {len(chunks)} чанков")
        if not chunks:
            raise RuntimeError("Не удалось разбить текст на чанки")

        # Вычисление эмбеддингов
        logger.info("Вычисление эмбеддингов для чанков...")
        embeddings = self.embedding_model.encode(chunks, normalize_embeddings=True).tolist()
        logger.info(f"Эмбеддинги вычислены, размерность: {len(embeddings[0]) if embeddings else 0}")

        # Создание коллекции и добавление чанков
        collection_name = f"doc_{uuid.uuid4().hex[:8]}"
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=None  # передаём эмбеддинги сами
        )
        ids = [f"chunk_{i:04d}" for i in range(len(chunks))]
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )
        logger.info(f"Чанки добавлены в коллекцию {collection_name}")

    def retrieve(self, question: str, top_k: int = TOP_K_RESULTS) -> List[str]:
        if not self.collection:
            logger.warning("retrieve called before indexing")
            return []
        q_emb = self.embedding_model.encode([question], normalize_embeddings=True).tolist()[0]
        results = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        if results['documents']:
            logger.info(f"Retrieved {len(results['documents'][0])} chunks for question: {question[:100]}")
            return results['documents'][0]
        else:
            logger.warning("No chunks retrieved")
            return []