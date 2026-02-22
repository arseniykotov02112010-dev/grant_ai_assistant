"""
rag_query9.py – Демонстрационный скрипт RAG-поиска.

Использование:
    python scripts/rag_query9.py <путь_к_текстовому_файлу> "Ваш вопрос"

Пример:
    python scripts/rag_query9.py data/processed/extracted_texts/example.txt "Каков максимальный размер гранта?"
"""

import sys
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Конфигурация
CHUNK_SIZE = 700          # размер чанка в символах
CHUNK_OVERLAP = 150       # перекрытие
TOP_K = 3                 # количество выводимых чанков
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

def read_file(file_path: Path) -> str:
    """Читает содержимое текстового файла."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def split_into_chunks(text: str) -> list[str]:
    """Разбивает текст на чанки с перекрытием."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_text(text)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Вычисляет косинусное сходство между двумя векторами."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    if len(sys.argv) < 3:
        print("Ошибка: укажите путь к файлу и вопрос.")
        print(__doc__)
        sys.exit(1)

    file_path = Path(sys.argv[1])
    question = sys.argv[2]

    if not file_path.exists():
        print(f"Ошибка: файл {file_path} не найден.")
        sys.exit(1)

    print(f"Загружаем модель эмбеддингов {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Читаем файл {file_path}...")
    text = read_file(file_path)

    print("Разбиваем на чанки...")
    chunks = split_into_chunks(text)
    print(f"Получено {len(chunks)} чанков.")

    print("Вычисляем эмбеддинги для чанков...")
    chunk_embeddings = model.encode(chunks, normalize_embeddings=True)

    print("Вычисляем эмбеддинг вопроса...")
    question_embedding = model.encode([question], normalize_embeddings=True)[0]

    print("Вычисляем сходство...")
    similarities = [cosine_similarity(question_embedding, chunk_emb) for chunk_emb in chunk_embeddings]

    # Сортируем чанки по убыванию сходства
    sorted_indices = np.argsort(similarities)[::-1]

    print("\n🔍 Топ-3 наиболее релевантных чанка:\n")
    for rank, idx in enumerate(sorted_indices[:TOP_K]):
        score = similarities[idx]
        print(f"--- Результат {rank+1} (сходство: {score:.4f}) ---")
        print(chunks[idx][:500] + ("..." if len(chunks[idx]) > 500 else ""))
        print()

if __name__ == "__main__":
    main()