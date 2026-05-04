import time
import logging
import uuid
import threading
from collections import deque
from typing import Dict, Optional, List

import chromadb
from chromadb.config import Settings
from backend.rag_engine import RAGEngine
from backend.config import SESSION_TIMEOUT

logger = logging.getLogger(__name__)

MAX_HISTORY_LENGTH = 50
CHROMA_PERSIST_DIR = "./chroma_db"


class SessionManager:
    def __init__(self, embedding_model, session_timeout: int = SESSION_TIMEOUT):
        self.embedding_model = embedding_model
        self.timeout = session_timeout
        self.lock = threading.Lock()
        self.sessions: Dict[str, dict] = {}
        self._hash_to_collection: Dict[str, str] = {}

        self._chroma_client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=Settings(anonymized_telemetry=False)
        )
        self._chroma_lock = threading.Lock()

        self._stop_cleanup = threading.Event()
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()

    def _cleanup_loop(self):
        while not self._stop_cleanup.wait(60):
            with self.lock:
                self._cleanup_expired_sessions()

    def _delete_collection_safe(self, name: str):
        """Удаляет коллекцию по имени, игнорируя ошибки."""
        try:
            self._chroma_client.delete_collection(name)
            logger.info(f"Коллекция {name} удалена")
        except Exception as e:
            logger.warning(f"Не удалось удалить коллекцию {name}: {e}")

    def create_session(self, text: str, filename: str,
                       content_hash: Optional[str] = None) -> str:
        # 0. Если есть активная сессия с этим хэшем, сразу возвращаем её
        if content_hash:
            with self.lock:
                for sid, data in self.sessions.items():
                    if data.get("content_hash") == content_hash:
                        if time.time() - data["last_used"] <= self.timeout:
                            data["last_used"] = time.time()
                            logger.info(f"Найдена активная сессия {sid} для хэша {content_hash[:12]}")
                            return sid
                        else:
                            # Истекшая – удалим позже, пока игнорируем
                            pass

        # 1. Проверяем наличие готовой коллекции по хэшу
        if content_hash:
            with self._chroma_lock:
                existing_name = self._hash_to_collection.get(content_hash)
                if existing_name:
                    try:
                        collection = self._chroma_client.get_collection(existing_name)
                        engine = RAGEngine(self.embedding_model,
                                           client=self._chroma_client,
                                           collection_name=existing_name)
                        session_id = self._add_session(engine, filename, content_hash)
                        logger.info(f"Использована существующая коллекция {existing_name}")
                        return session_id
                    except Exception as e:
                        logger.warning(f"Коллекция {existing_name} не загрузилась: {e}. Создаём новую.")
                        self._delete_collection_safe(existing_name)
                        del self._hash_to_collection[content_hash]

        # 2. Индексация нового документа
        engine = RAGEngine(self.embedding_model, client=self._chroma_client)
        try:
            engine.index_document(text)
        except Exception:
            if engine.collection:
                self._delete_collection_safe(engine.collection.name)
            try:
                engine.close()
            except Exception:
                pass
            logger.exception("Ошибка индексации")
            raise RuntimeError("Не удалось обработать документ")

        # 3. Регистрируем коллекцию
        new_collection_name = engine.collection.name
        if content_hash:
            with self._chroma_lock:
                if content_hash in self._hash_to_collection:
                    existing_name = self._hash_to_collection[content_hash]
                    try:
                        existing_collection = self._chroma_client.get_collection(existing_name)
                        self._delete_collection_safe(new_collection_name)
                        engine.close()
                        engine = RAGEngine(self.embedding_model,
                                           client=self._chroma_client,
                                           collection_name=existing_name)
                        session_id = self._add_session(engine, filename, content_hash)
                        logger.info(f"Гонка: коллекция {existing_name} уже существует")
                        return session_id
                    except Exception:
                        self._delete_collection_safe(existing_name)
                self._hash_to_collection[content_hash] = new_collection_name

        session_id = self._add_session(engine, filename, content_hash)
        logger.info(f"Создана сессия {session_id} для файла {filename}")
        return session_id

    def _add_session(self, engine: RAGEngine, filename: str,
                     content_hash: Optional[str] = None) -> str:
        session_id = uuid.uuid4().hex
        now = time.time()
        with self.lock:
            self.sessions[session_id] = {
                "engine": engine,
                "filename": filename,
                "content_hash": content_hash,
                "created": now,
                "last_used": now,
                "history": deque(maxlen=MAX_HISTORY_LENGTH)
            }
        return session_id

    def get_engine(self, session_id: str) -> Optional[RAGEngine]:
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                return None
            if time.time() - session["last_used"] > self.timeout:
                self._remove_session(session_id)
                return None
            session["last_used"] = time.time()
            return session["engine"]

    def add_message(self, session_id: str, role: str, content: str) -> bool:
        with self.lock:
            session = self.sessions.get(session_id)
            if session is None:
                return False
            session["history"].append({"role": role, "content": content, "time": time.time()})
            return True

    def get_history(self, session_id: str, last_n: int = 4) -> List[dict]:
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                return []
            items = list(session.get("history", []))
            return items[-last_n:] if len(items) > last_n else items

    def shutdown(self):
        self._stop_cleanup.set()
        self._cleanup_thread.join(timeout=5)
        with self.lock:
            for sid in list(self.sessions.keys()):
                self._remove_session(sid)
        logger.info("Менеджер сессий остановлен")

    def _cleanup_expired_sessions(self):
        now = time.time()
        expired = [sid for sid, data in self.sessions.items()
                   if now - data["last_used"] > self.timeout]
        for sid in expired:
            self._remove_session(sid)
        if expired:
            logger.info(f"Фоновая очистка: удалено {len(expired)} сессий")

    def _remove_session(self, session_id: str):
        session = self.sessions.pop(session_id, None)
        if not session:
            return
        engine = session.get("engine")
        if engine:
            try:
                engine.close()
            except Exception as e:
                logger.warning(f"Ошибка при закрытии движка сессии {session_id}: {e}")
        # Коллекция НЕ удаляется – она может использоваться другими сессиями
        logger.debug(f"Сессия {session_id} удалена")

    def get_active_sessions_count(self) -> int:
        with self.lock:
            self._cleanup_expired_sessions()
            return len(self.sessions)