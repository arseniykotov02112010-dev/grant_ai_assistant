import logging
import uuid
import re
from typing import List, Optional

import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from backend.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Признаки заголовков для семантического чанкинга
HEADING_PATTERNS = [
    r"\d+(\.\d+)+\.?",          # 2.1.1
    r"\d+\.\s",                   # 2. Требования
    r"\d+\.[А-ЯЁа-яё]",          # 2.Общие (без пробела)
    r"^[а-яё]{3,}\s?\d+\.",     # Раздел 1.
]

HEADING_KEYWORDS = [
    "раздел", "требования", "условия", "порядок", "критерии",
    "сроки", "финансирование", "заявитель", "участник",
    "конкурс", "грант", "подача", "заявка", "заявки",
    "заявку", "заявок", "индикатор", "результатив",
    "бюджет", "софинанс",
]

MIN_CHUNK_LENGTH = 15   # порог для не-табличных чанков без цифр и без заголовка


def _contains_table(chunk: str) -> bool:
    return "[ТАБЛИЦА]" in chunk


def _is_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False

    lower = stripped.lower()

    for pat in HEADING_PATTERNS:
        if re.match(pat, lower):
            return True

    if stripped.endswith('.') and not re.match(r'\d+\.', stripped):
        return False

    if len(stripped) > 80 and ':' not in stripped:
        return False

    start = lower[:30]
    for kw in HEADING_KEYWORDS:
        if start.startswith(kw):
            after = stripped[len(kw):].lstrip()
            if not after or after[0] in (':', '.', ' ', '\t') or re.match(r'^\d', after):
                return True
    return False


def _has_meaningful_word(text: str) -> bool:
    words = re.findall(r'[а-яёa-z]{5,}', text, re.IGNORECASE)
    return len(words) > 0


class RAGEngine:
    """
    RAG-движок с поддержкой внешнего клиента (PersistentClient).
    Если client и collection_name переданы, используется существующая коллекция.
    """
    def __init__(self, embedding_model: SentenceTransformer,
                 client: Optional[chromadb.Client] = None,
                 collection_name: Optional[str] = None):
        self.embedding_model = embedding_model
        self.client = client
        self.collection: Optional[chromadb.Collection] = None
        self.document_text: Optional[str] = None
        self._owns_client = False

        if collection_name and client:
            try:
                self.collection = client.get_collection(collection_name)
                logger.info(f"Подключена существующая коллекция: {collection_name}")
            except Exception as e:
                logger.error(f"Коллекция {collection_name} не найдена: {e}")
                raise ValueError(f"Коллекция {collection_name} не существует") from e

    def index_document(self, text: str) -> None:
        if self.collection is not None:
            raise RuntimeError("Индексация невозможна: коллекция уже загружена извне")
        if not text.strip():
            raise ValueError("Текст документа пуст")

        self.document_text = text
        logger.info(f"Индексация текста, длина: {len(text)} символов")

        # Разбиение на разделы с учётом таблиц и заголовков
        pages = text.split("[PAGE_BREAK]")
        sections = []          # {'type': 'text'|'table', 'heading': str, 'body': str}
        current_heading = ""
        current_lines = []

        table_pattern = re.compile(r'\[ТАБЛИЦА\].*?\[/ТАБЛИЦА\]', re.DOTALL)

        for page_text in pages:
            page_text = page_text.strip()
            if not page_text:
                continue

            parts = []
            last_end = 0
            for match in table_pattern.finditer(page_text):
                start, end = match.span()
                if start > last_end:
                    parts.append(('text', page_text[last_end:start]))
                parts.append(('table', match.group()))
                last_end = end
            if last_end < len(page_text):
                parts.append(('text', page_text[last_end:]))

            for ptype, content in parts:
                if ptype == 'table':
                    sections.append({
                        'type': 'table',
                        'heading': current_heading,
                        'body': content
                    })
                else:
                    lines = content.splitlines()
                    for line in lines:
                        if line.strip() and _is_heading(line):
                            if current_lines:
                                sections.append({
                                    'type': 'text',
                                    'heading': current_heading,
                                    'body': '\n'.join(current_lines)
                                })
                                current_lines = []
                            current_heading = line.strip()
                        else:
                            current_lines.append(line)

        if current_lines:
            sections.append({
                'type': 'text',
                'heading': current_heading,
                'body': '\n'.join(current_lines)
            })
        elif current_heading and not any(
            sec['heading'] == current_heading and sec['type'] == 'text'
            for sec in sections
        ):
            sections.append({
                'type': 'text',
                'heading': current_heading,
                'body': ''
            })

        if not sections:
            sections.append({'type': 'text', 'heading': '', 'body': text})

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = []
        for sec in sections:
            heading = sec['heading']
            body = sec['body']

            if sec['type'] == 'table':
                chunks.append(f"Раздел: {heading}\n{body}" if heading else body)
                continue

            body = re.sub(r'\n{3,}', '\n\n', body).strip()
            if not body:
                if heading:
                    chunks.append(f"Раздел: {heading}")
                continue

            if len(body) < CHUNK_SIZE / 2:
                chunk_text = f"Раздел: {heading}\n{body}" if heading else body
                chunks.append(chunk_text)
            else:
                sub_chunks = splitter.split_text(body)
                for sub in sub_chunks:
                    chunk_text = f"Раздел: {heading}\n{sub}" if heading else sub
                    chunks.append(chunk_text)

        # Фильтрация мусорных чанков
        filtered = []
        for ch in chunks:
            if _contains_table(ch):
                filtered.append(ch)
            elif len(ch) >= MIN_CHUNK_LENGTH:
                filtered.append(ch)
            elif re.search(r'\d', ch):
                filtered.append(ch)
            elif ch.startswith("Раздел:"):
                filtered.append(ch)
            elif _has_meaningful_word(ch):
                filtered.append(ch)
            else:
                logger.debug(f"Отброшен короткий чанк без значимых признаков: {ch[:80]}...")
        chunks = filtered

        logger.info(f"Получено {len(chunks)} чанков после фильтрации")
        if not chunks:
            raise RuntimeError("Не удалось разбить текст на чанки")

        logger.info("Вычисление эмбеддингов...")
        try:
            embeddings = self.embedding_model.encode(
                chunks,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False
            ).tolist()
        except Exception as e:
            logger.exception("Ошибка при вычислении эмбеддингов")
            raise RuntimeError("Ошибка кодирования чанков") from e

        # Создание клиента, если не передан
        if self.client is None:
            self.client = chromadb.EphemeralClient()
            self._owns_client = True
            logger.info("Используется EphemeralClient")

        # Создание коллекции с уникальным именем
        collection_name = f"doc_{uuid.uuid4().hex[:8]}"
        try:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=None,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            logger.exception("Не удалось создать коллекцию ChromaDB")
            raise RuntimeError("Ошибка инициализации индекса") from e

        ids = [f"chunk_{i:04d}" for i in range(len(chunks))]
        try:
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                ids=ids
            )
        except Exception as e:
            logger.exception("Ошибка при добавлении чанков в коллекцию")
            raise RuntimeError("Ошибка наполнения индекса") from e

        logger.info(f"Чанки добавлены в коллекцию {collection_name}")

    def retrieve(self, question: str, top_k: int = TOP_K_RESULTS) -> List[str]:
        if not self.collection:
            logger.warning("retrieve вызван до индексации")
            return []
        if not question or not question.strip():
            logger.warning("Пустой вопрос")
            return []

        try:
            q_emb = self.embedding_model.encode(
                [question], normalize_embeddings=True
            )
        except Exception as e:
            logger.exception("Ошибка эмбеддинга вопроса")
            return []

        try:
            results = self.collection.query(
                query_embeddings=q_emb.tolist(),
                n_results=top_k * 3,
                include=["documents", "distances"]
            )
        except Exception as e:
            logger.exception("Ошибка запроса к ChromaDB")
            return []

        if not results.get('documents') or not results.get('distances'):
            logger.warning("Пустой ответ ChromaDB")
            return []

        doc_list = results['documents'][0]
        dist_list = results['distances'][0]
        if not doc_list or not dist_list:
            return []

        similarities = [1.0 - d for d in dist_list]
        scored = [(sim, doc) for sim, doc in zip(similarities, doc_list) if sim >= SIMILARITY_THRESHOLD]
        scored.sort(key=lambda x: x[0], reverse=True)
        selected = [doc for _, doc in scored[:top_k]]

        if not selected:
            logger.info(f"Все чанки ниже порога {SIMILARITY_THRESHOLD} для вопроса: {question[:100]}")
            return []

        logger.info(f"Отобрано {len(selected)} чанков (из {len(doc_list)} кандидатов)")
        return selected

    def close(self):
        """Освобождение ресурсов. Временные коллекции удаляются, персистентные сохраняются."""
        if self.collection and self.client:
            try:
                if self._owns_client:
                    self.client.delete_collection(self.collection.name)
                    logger.info(f"Коллекция {self.collection.name} удалена")
            except Exception as e:
                logger.warning(f"Ошибка при удалении коллекции: {e}")
            finally:
                self.collection = None
        if self.client and self._owns_client:
            self.client = None
            self._owns_client = False
        logger.debug("RAGEngine.close() выполнен")