"""
config.py – конфигурация для RAG-движка и сервера.
Все параметры собраны в одном месте для удобства.
"""

from pathlib import Path

# ========== ПУТИ ==========
BASE_DIR = Path(__file__).parent.parent

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
GGUF_MODEL_PATH = BASE_DIR / "model" / "qvikhr-grant-q4_k_m.gguf"

# ========== ЧАНКИНГ ==========
CHUNK_SIZE = 700
CHUNK_OVERLAP = 250

# ========== ГЕНЕРАЦИЯ ==========
MAX_TOKENS = 256          # достаточно для коротких точных ответов
TEMPERATURE = 0.0         # строгая детерминированность
REPEAT_PENALTY = 1.1      # умеренный штраф за повторы
TOP_P = 0.9               # исключение маловероятных токенов
STOP_WORDS = ["\n\n", "Информация не найдена"]

# ========== СЕССИИ ==========
SESSION_TIMEOUT = 3600    # 1 час

# ========== RAG ==========
SIMILARITY_THRESHOLD = 0.6   # минимальное косинусное сходство
TOP_K_RESULTS = 10           # количество возвращаемых чанков