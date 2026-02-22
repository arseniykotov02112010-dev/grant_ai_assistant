"""
ИСПРАВЛЕННЫЙ ЭТАП 0.3: Тест качества семантического поиска RAG.
Проверяет, находит ли система чанки, смысл которых близок к эталонным ответам.
"""
import torch
import json
from pathlib import Path
import chromadb
import os
from sentence_transformers import SentenceTransformer, util
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# --- КОНФИГУРАЦИЯ (ПОДСТАВЬТЕ СВОИ ПУТИ) ---
project_root = Path(__file__).parent.parent
VECTOR_DB_PATH = project_root / "data" / "vector_db"
VAL_DATASET_PATH = project_root / "data" / "splits" / "val.jsonl"

# Параметры теста
TEST_SAMPLE_SIZE = 150  # Сколько QA-пар проверить
SEARCH_TOP_K = 3  # Сколько чанков искать
SIMILARITY_THRESHOLD = 0.5

# Модель для сравнения смысла
SEMANTIC_MODEL_NAME = "intfloat/multilingual-e5-small"


def main():
    print("=" * 70)
    print("ТЕСТ СЕМАНТИЧЕСКОГО ПОИСКА RAG (ИСПРАВЛЕННЫЙ)")
    print("=" * 70)

    # 1. Загрузка тестовых вопросов и ответов
    print("[1/5] Загружаем QA-пары...")
    qa_pairs = []
    with open(VAL_DATASET_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= TEST_SAMPLE_SIZE:
                break
            qa_pairs.append(json.loads(line.strip()))
    print(f"   Загружено: {len(qa_pairs)} пар")

    # 2. Подключение к векторной базе
    print("[2/5] Подключаемся к векторной базе...")
    client = chromadb.PersistentClient(path=str(VECTOR_DB_PATH))
    collection = client.get_collection(name="grant_documents_collection")
    print(f"   База: '{collection.name}', чанков: {collection.count()}")

    # 3. Загрузка модели для сравнения смысла
    print(f"[3/5] Загружаем модель для сравнения смысла ({SEMANTIC_MODEL_NAME})...")
    similarity_model = SentenceTransformer(SEMANTIC_MODEL_NAME)

    # 4. Основной цикл проверки
    print("[4/5] Проверяем поиск...")
    results = []

    for idx, qa in enumerate(qa_pairs):
        question = qa["question"]
        true_answer = qa["answer"]
        doc_id = qa["doc_id"]

        # Поиск релевантных фрагментов документа с помощью RAG
        # Ищем чанки из ТОГО ЖЕ документа
        search_result = collection.query(
            query_texts=[question],
            n_results=SEARCH_TOP_K,
            where={"source_doc_id": doc_id}  # Ключевой фильтр!
        )

        # Если чанки найдены, сравниваем смысл с эталонным ответом
        best_similarity = 0.0
        if search_result['documents'][0]:
            # Кодируем эталонный ответ и все найденные чанки
            true_answer_vec = similarity_model.encode(true_answer, convert_to_tensor=True)
            chunks_vec = similarity_model.encode(search_result['documents'][0], convert_to_tensor=True)

            # Вычисляем близость по смыслу
            similarities = util.cos_sim(true_answer_vec, chunks_vec)[0]
            best_similarity = float(torch.max(similarities)) if hasattr(similarities, '__iter__') else float(
                similarities)

        # Запоминаем результат
        is_success = best_similarity >= SIMILARITY_THRESHOLD
        results.append({
            "question": question[:60] + "..." if len(question) > 60 else question,
            "true_answer": true_answer[:80] + "..." if len(true_answer) > 80 else true_answer,
            "best_similarity": best_similarity,
            "success": is_success
        })

        # Прогресс
        if (idx + 1) % 50 == 0:
            print(f"   Проверено {idx + 1}/{len(qa_pairs)}")

    # 5. Считаем и выводим итоги
    print("[5/5] Анализируем результаты...")
    print("=" * 70)

    success_count = sum(1 for r in results if r["success"])
    total_count = len(results)
    accuracy = (success_count / total_count) * 100 if total_count > 0 else 0

    print(f"ИТОГО:")
    print(f"  Проверено вопросов: {total_count}")
    print(f"  Успешных поисков: {success_count}")
    print(f"  Точность (Accuracy): {accuracy:.1f}%")
    print(f"  Средняя близость: {np.mean([r['best_similarity'] for r in results]):.3f}")

    # Примеры: лучший и худший результат
    if results:
        best = max(results, key=lambda x: x["best_similarity"])
        worst = min(results, key=lambda x: x["best_similarity"])

        print(f"\nЛУЧШИЙ РЕЗУЛЬТАТ (близость: {best['best_similarity']:.3f}):")
        print(f"  В: {best['question']}")
        print(f"  О: {best['true_answer']}")

        print(f"\nХУДШИЙ РЕЗУЛЬТАТ (близость: {worst['best_similarity']:.3f}):")
        print(f"  В: {worst['question']}")
        print(f"  О: {worst['true_answer']}")

    # Рекомендации
    print("=" * 70)
    if accuracy >= 70:
        print("✅ Отлично! Поиск хорошо находит смыслово близкие фрагменты.")
        print("   Можно переходить к тесту с нулевой моделью.")
    elif accuracy >= 50:
        print("⚠️  Приемлемо, но есть куда улучшать.")
        print("   Совет: увеличьте SEARCH_TOP_K до 5 и порог SIMILARITY_THRESHOLD до 0.6.")
    else:
        print("❌ Качество поиска низкое. Нужна настройка.")
        print("   Действия: 1) Увеличьте CHUNK_SIZE при создании базы")
        print("             2) Проверьте, не обрезаны ли тексты в документах")


if __name__ == "__main__":
    main()