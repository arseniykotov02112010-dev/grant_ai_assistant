import json
from pathlib import Path
import random


def prepare_documents_for_generation(texts_dir, output_path, sample_size=None):
    """
    Подготавливает документы для генерации QA-пар.
    Если sample_size указан, берет только случайную выборку документов.
    """
    base_dir = Path(texts_dir)
    text_files = list(base_dir.glob("*.txt"))

    if sample_size and sample_size < len(text_files):
        text_files = random.sample(text_files, sample_size)

    documents = []

    for txt_file in text_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Определяем язык (простая эвристика)
        russian_chars = sum(1 for c in content if 'а' <= c <= 'я' or 'А' <= c <= 'Я')
        english_chars = sum(1 for c in content if 'a' <= c <= 'z' or 'A' <= c <= 'Z')
        language = "russian" if russian_chars > english_chars else "english"

        # Убираем наш маркер страниц для чистого текста
        clean_content = content.replace("=== Страница", "")

        # Ограничиваем размер (если очень большой документ)
        max_chars = 100000  # ~2000 токенов * 50
        if len(clean_content) > max_chars:
            # Берем начало, середину и конец для сохранения структуры
            part_len = max_chars // 3
            clean_content = (
                    clean_content[:part_len] +
                    "\n[...пропущена середина документа...]\n" +
                    clean_content[len(clean_content) // 2 - part_len // 2:len(clean_content) // 2 + part_len // 2] +
                    "\n[...пропущен конец документа...]\n" +
                    clean_content[-part_len:]
            )

        documents.append({
            "doc_id": txt_file.stem,
            "language": language,
            "content": clean_content,
            "original_length": len(content),
            "prepared_length": len(clean_content)
        })

    # Сохраняем в JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"Подготовлено {len(documents)} документов для генерации.")
    print(f"Сохранено в: {output_path}")

    # Статистика
    total_chars = sum(d['prepared_length'] for d in documents)
    avg_chars = total_chars / len(documents)

    print(f"\nСтатистика:")
    print(f"  Всего символов: {total_chars:,}")
    print(f"  Средний размер документа: {avg_chars:,.0f} символов")
    print(f"  Примерный объем в токенах (приблизительно): {total_chars // 4:,}")

    return documents


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    TEXTS_DIR = BASE_DIR / "data" / "processed" / "extracted_texts"
    OUTPUT_PATH = BASE_DIR / "data" / "processed" / "documents_for_qa_generation.json"

    # Для тестирования можно взять 5 документов
    # documents = prepare_documents_for_generation(TEXTS_DIR, OUTPUT_PATH, sample_size=5)

    # Для реальной работы - все документы
    documents = prepare_documents_for_generation(TEXTS_DIR, OUTPUT_PATH)

    # Выводим пример первого документа
    if documents:
        print(f"\nПример первого документа (первые 500 символов):")
        print("-" * 60)
        print(documents[0]['content'][:500])
        print("...")