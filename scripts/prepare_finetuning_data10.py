
"""
prepare_finetuning_data.py (УЛУЧШЕННАЯ ВЕРСИЯ)

Скрипт для обогащения датасета sft_dataset.jsonl контекстами из векторной БД.
Для каждой QA-пары находит топ-3 чанка из того же документа (по doc_id),
объединяет их и сохраняет в новый датасет finetuning_dataset.jsonl.

"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# ==================== КОНФИГУРАЦИЯ ====================
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_JSONL = PROJECT_ROOT / "data" / "processed" / "sft_dataset.jsonl"
VECTOR_DB_PATH = PROJECT_ROOT / "data" / "vector_db"
OUTPUT_JSONL = PROJECT_ROOT / "data" / "processed" / "finetuning_dataset.jsonl"

TOP_K = 3                      # количество чанков для объединения
SIMILARITY_THRESHOLD = 0.5     # минимальное среднее сходство для сохранения
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"

# ==================== ПРОВЕРКИ ====================
print("🔍 Проверка путей...")
if not INPUT_JSONL.exists():
    print(f"❌ ОШИБКА: Входной файл не найден: {INPUT_JSONL}")
    sys.exit(1)
print(f"✅ Входной файл: {INPUT_JSONL}")

if not VECTOR_DB_PATH.exists():
    print(f"❌ ОШИБКА: Векторная БД не найдена: {VECTOR_DB_PATH}")
    sys.exit(1)
print(f"✅ Векторная БД: {VECTOR_DB_PATH}")

# ==================== ЗАГРУЗКА МОДЕЛИ ====================
print(f"\n🔄 Загрузка модели эмбеддингов: {EMBEDDING_MODEL_NAME}...")
try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print("✅ Модель загружена.")
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    sys.exit(1)

# ==================== ПОДКЛЮЧЕНИЕ К БД ====================
print("\n🔄 Подключение к векторной БД...")
try:
    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    collection = client.get_collection(name="grant_documents_collection")
    print(f"✅ Подключено к коллекции '{collection.name}', чанков: {collection.count()}")
except Exception as e:
    print(f"❌ Ошибка подключения к БД: {e}")
    sys.exit(1)

# ==================== ЧТЕНИЕ ВХОДНОГО ФАЙЛА ====================
print("\n📖 Чтение входного датасета...")
records = []
with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            records.append(record)
        except json.JSONDecodeError as e:
            print(f"⚠️ Предупреждение: строка {i+1} не является JSON, пропущена. Ошибка: {e}")

print(f"✅ Загружено записей: {len(records)}")

# Покажем первые 3 записи для проверки
print("\n🔎 Первые 3 записи из датасета:")
for i, rec in enumerate(records[:3]):
    print(f"  {i+1}. doc_id: {rec.get('doc_id', 'НЕТ')}, вопрос: {rec.get('question', '')[:60]}...")

# ==================== ОСНОВНАЯ ОБРАБОТКА ====================
print("\n🚀 Начинаем обработку...")

# Открываем выходной файл
OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
f_out = open(OUTPUT_JSONL, 'w', encoding='utf-8')

total = len(records)
saved = 0
similarities = []

for idx, record in enumerate(records):
    # Выводим прогресс каждые 100 записей
    if (idx + 1) % 100 == 0:
        print(f"  Прогресс: {idx + 1}/{total} ({(idx + 1)/total*100:.1f}%)")

    doc_id = record.get('doc_id')
    question = record.get('question')
    answer = record.get('answer')

    if not doc_id or not question or not answer:
        print(f"⚠️ Пропуск записи {idx+1}: отсутствуют обязательные поля (doc_id, question, answer)")
        continue

    # Получаем эмбеддинг ответа
    answer_emb = embedding_model.encode(answer, normalize_embeddings=True)

    # Ищем чанки в том же документе
    try:
        results = collection.query(
            query_embeddings=[answer_emb.tolist()],
            n_results=TOP_K * 2,  # запрашиваем с запасом
            where={"source_doc_id": doc_id}
        )
    except Exception as e:
        print(f"⚠️ Ошибка запроса к ChromaDB для записи {idx+1}: {e}")
        continue

    if not results['documents'] or not results['documents'][0]:
        print(f"ℹ️ Для записи {idx+1} не найдено чанков в документе {doc_id}")
        continue

    # Получаем тексты чанков
    chunk_texts = results['documents'][0]
    chunk_metadatas = results['metadatas'][0]

    # Кодируем чанки
    chunk_embs = embedding_model.encode(chunk_texts, normalize_embeddings=True)

    # Вычисляем косинусное сходство
    chunk_similarities = []
    for i, chunk_emb in enumerate(chunk_embs):
        sim = float(np.dot(answer_emb, chunk_emb) / (np.linalg.norm(answer_emb) * np.linalg.norm(chunk_emb)))
        chunk_similarities.append(sim)

    # Сортируем по убыванию сходства и берём TOP_K
    sorted_indices = np.argsort(chunk_similarities)[::-1][:TOP_K]
    top_texts = [chunk_texts[i] for i in sorted_indices]
    top_sims = [chunk_similarities[i] for i in sorted_indices]

    avg_sim = np.mean(top_sims)
    if avg_sim < SIMILARITY_THRESHOLD:
        print(f"ℹ️ Запись {idx+1} пропущена: среднее сходство {avg_sim:.3f} < порога {SIMILARITY_THRESHOLD}")
        continue

    # Объединяем тексты чанков
    combined_context = "\n---\n".join(top_texts)

    # Записываем результат
    new_record = {
        "doc_id": doc_id,
        "context": combined_context,
        "question": question,
        "answer": answer,
        "similarity": avg_sim
    }
    f_out.write(json.dumps(new_record, ensure_ascii=False) + '\n')
    saved += 1
    similarities.append(avg_sim)

f_out.close()

# ==================== ИТОГИ ====================
print("\n" + "=" * 60)
print("ОБРАБОТКА ЗАВЕРШЕНА")
print("=" * 60)
print(f"Всего записей в исходном датасете: {total}")
print(f"Сохранено в новый датасет: {saved}")

if similarities:
    print(f"\n📊 Статистика сходства (similarity):")
    print(f"  Среднее: {np.mean(similarities):.4f}")
    print(f"  Медиана: {np.median(similarities):.4f}")
    print(f"  Минимум: {np.min(similarities):.4f}")
    print(f"  Максимум: {np.max(similarities):.4f}")
    print(f"  Стандартное отклонение: {np.std(similarities):.4f}")

print(f"\n✅ Новый датасет сохранён в: {OUTPUT_JSONL.absolute()}")