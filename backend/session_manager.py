"""
session_manager.py – управление сессиями пользователей.
Каждая сессия соответствует одному загруженному документу и содержит RAGEngine.
Сессии автоматически удаляются после периода неактивности.
"""

import time
import logging
import uuid
from typing import Dict, Optional

from backend.rag_engine import RAGEngine
from backend.config import SESSION_TIMEOUT

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Менеджер сессий, хранящий RAGEngine для каждого загруженного документа.
    """

    def __init__(self, embedding_model, session_timeout: int = SESSION_TIMEOUT):
        self.embedding_model = embedding_model
        self.timeout = session_timeout
        self.sessions: Dict[str, dict] = {}  # session_id -> {"engine": RAGEngine, "filename": str, "created": float, "last_used": float}

    def create_session(self, text: str, filename: str) -> str:
        """
        Создаёт новую сессию: индексирует текст и возвращает уникальный session_id.
        """
        session_id = uuid.uuid4().hex
        engine = RAGEngine(self.embedding_model)
        try:
            engine.index_document(text)
        except Exception as e:
            logger.exception(f"Ошибка при индексации документа для сессии {session_id}")
            raise RuntimeError("Не удалось обработать документ") from e

        now = time.time()
        self.sessions[session_id] = {
            "engine": engine,
            "filename": filename,
            "created": now,
            "last_used": now
        }
        logger.info(f"Создана сессия {session_id} для файла {filename}")
        return session_id

    def get_engine(self, session_id: str) -> Optional[RAGEngine]:
        """
        Возвращает RAGEngine для указанной сессии, если она существует и не истекла.
        При успешном получении обновляет время последнего использования.
        """
        # Сначала удалим все истекшие сессии (чтобы не копить мусор)
        self.cleanup_expired_sessions()

        session = self.sessions.get(session_id)
        if not session:
            logger.warning(f"Сессия {session_id} не найдена")
            return None

        now = time.time()
        # Проверка на истечение срока (хотя cleanup уже удалил, но на всякий случай)
        if now - session["last_used"] > self.timeout:
            logger.info(f"Сессия {session_id} истекла, удаляем")
            del self.sessions[session_id]
            return None

        session["last_used"] = now
        logger.debug(f"Сессия {session_id} активна")
        return session["engine"]

    def cleanup_expired_sessions(self) -> int:
        """
        Удаляет все истекшие сессии.
        Возвращает количество удалённых сессий.
        """
        now = time.time()
        expired = [
            sid for sid, data in self.sessions.items()
            if now - data["last_used"] > self.timeout
        ]
        for sid in expired:
            del self.sessions[sid]
        if expired:
            logger.info(f"Удалено {len(expired)} истекших сессий")
        return len(expired)

    def get_active_sessions_count(self) -> int:
        return len(self.sessions)