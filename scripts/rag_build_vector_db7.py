# scripts/build_vector_db.py
"""
Скрипт для создания векторной базы данных (ChromaDB) из подготовленных JSON-документов.
Выполняет: загрузку, чанкирование, создание эмбеддингов и сохранение в персистентную БД.
"""

import json
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path, чтобы импортировать как модуль (опционально)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import logging

# --- Конфигурация ---
# Путь к JSON-файлу с документами
INPUT_JSON_PATH = project_root / "data" / "processed" / "documents_for_qa_generation.json"
# Директория для сохранения векторной БД
PERSIST_DIRECTORY = project_root / "data" / "vector_db"

# Параметры чанкирования (размер фрагмента и перекрытие)
CHUNK_SIZE = 700  # Размер чанка в символах (приблизительно)
CHUNK_OVERLAP = 150  # Перекрытие между чанками для сохранения контекста

# Модель для эмбеддингов. 'intfloat/multilingual-e5-small' хорошо поддерживает русский и эффективна.
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

# Настройка логирования для отслеживания процесса
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_documents(json_path: Path):
    """Загружает и валидирует документы из JSON-файла."""
    logger.info(f"Загрузка документов из {json_path}")
    if not json_path.exists():
        logger.error(f"Файл не найден: {json_path}")
        raise FileNotFoundError(f"Файл не найден: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Валидация структуры данных
    if not isinstance(data, list):
        logger.error("JSON должен содержать список документов.")
        raise ValueError("Неверный формат JSON: ожидается список.")

    required_keys = {"doc_id", "content"}
    for i, doc in enumerate(data):
        if not all(key in doc for key in required_keys):
            logger.error(f"Документ с индексом {i} не содержит обязательных ключей: {required_keys}")
            raise ValueError(f"Документ {i} имеет неполную структуру.")

    logger.info(f"Успешно загружено {len(data)} документов.")
    return data

def create_and_save_vector_db(documents: list, persist_directory: Path):
    """
    Разбивает документы на чанки, создает эмбеддинги и сохраняет в ChromaDB.
    """
    logger.info("Начало создания векторной базы данных...")

    # 1. Подготовка текстов и метаданных
    logger.info("Подготовка текстов...")
    texts = []
    metadatas = []

    for doc in documents:
        doc_id = doc["doc_id"]
        content = doc["content"]

        # Используем текстовый сплиттер из LangChain
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Приоритеты разделения
        )
        doc_chunks = text_splitter.split_text(content)

        # Для каждого чанка сохраняем текст и метаданные (идентификатор исходного документа)
        for chunk_num, chunk in enumerate(doc_chunks):
            texts.append(chunk)
            metadatas.append({"source_doc_id": doc_id, "chunk_id": chunk_num})

        logger.debug(f"Документ '{doc_id}' разбит на {len(doc_chunks)} чанков.")

    logger.info(f"Общее количество текстовых чанков для индексации: {len(texts)}")

    # 2. Инициализация модели для создания эмбеддингов
    logger.info(f"Загрузка модели эмбеддингов: {EMBEDDING_MODEL_NAME}")
    # Указываем device='cpu', так как работаем локально. Для большой БД можно использовать 'cuda', если GPU доступен.
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # Нормализация улучшает качество косинусного сходства
    )

    # 3. Создание и сохранение векторной базы данных
    logger.info("Создание векторного хранилища Chroma. Это может занять несколько минут...")
    # Параметр persist_directory обеспечивает сохранение БД на диск.
    vector_db = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        persist_directory=str(persist_directory),
        collection_name="grant_documents_collection"
    )

    # Явное сохранение на диск
    vector_db.persist()
    logger.info(f"✅ Векторная база данных успешно создана и сохранена в: {persist_directory}")
    logger.info(f"Количество проиндексированных чанков: {vector_db._collection.count()}")

    return vector_db


if __name__ == "__main__":
    try:
        # Загружаем документы
        docs = load_documents(INPUT_JSON_PATH)

        # Создаем директорию для БД, если её нет
        PERSIST_DIRECTORY.mkdir(parents=True, exist_ok=True)

        # Создаем и сохраняем векторную БД
        db = create_and_save_vector_db(docs, PERSIST_DIRECTORY)

        print("\n" + "=" * 50)
        print("ВЕКТОРНАЯ БАЗА ДАННЫХ УСПЕШНО СОЗДАНА.")
        print("=" * 50)
        print(f"Расположение: {PERSIST_DIRECTORY}")
        print(f"Для проверки можно запустить скрипт: `scripts/test_rag_search.py` (будет создан на следующем этапе).")

    except Exception as e:
        logger.exception(f"❌ Критическая ошибка при создании векторной БД: {e}")
        sys.exit(1)