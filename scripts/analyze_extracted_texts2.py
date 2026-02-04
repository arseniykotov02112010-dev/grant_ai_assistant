import json
from pathlib import Path
from collections import Counter


def analyze_texts(texts_dir):
    """Анализирует все текстовые файлы и собирает статистику."""
    base_dir = Path(texts_dir)
    text_files = list(base_dir.glob("*.txt"))

    stats = {
        "total_files": len(text_files),
        "by_page_count": Counter(),
        "by_language": Counter(),
        "size_distribution": {"small": 0, "medium": 0, "large": 0},
        "content_samples": []
    }

    for i, txt_file in enumerate(text_files, 1):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Подсчет страниц (по нашему разделителю)
            page_count = content.count("=== Страница") if "=== Страница" in content else 1

            # Определение языка (простая эвристика)
            russian_chars = sum(1 for c in content if 'а' <= c <= 'я' or 'А' <= c <= 'Я')
            english_chars = sum(1 for c in content if 'a' <= c <= 'z' or 'A' <= c <= 'Z')
            language = "rus" if russian_chars > english_chars else "eng"

            # Размер документа
            char_count = len(content)
            if char_count < 5000:
                size_category = "small"
            elif char_count < 30000:
                size_category = "medium"
            else:
                size_category = "large"

            # Собираем статистику
            stats["by_page_count"][page_count] += 1
            stats["by_language"][language] += 1
            stats["size_distribution"][size_category] += 1

            # Сохраняем примеры для каждого типа
            if i <= 3:  # первые 3 файла как примеры
                stats["content_samples"].append({
                    "filename": txt_file.name,
                    "pages": page_count,
                    "language": language,
                    "size_chars": char_count,
                    "preview": content[:500] + "..." if len(content) > 500 else content
                })

            # Прогресс
            if i % 50 == 0:
                print(f"Проанализировано {i}/{len(text_files)} файлов...")

        except Exception as e:
            print(f"Ошибка при анализе {txt_file}: {e}")

    return stats


def save_analysis_report(stats, output_path):
    """Сохраняет детальный отчет анализа."""
    report = {
        "summary": {
            "total_documents": stats["total_files"],
            "documents_per_language": dict(stats["by_language"]),
            "size_distribution": stats["size_distribution"],
            "avg_pages": sum(k * v for k, v in stats["by_page_count"].items()) / stats["total_files"] if stats[
                                                                                                             "total_files"] > 0 else 0
        },
        "detailed_page_stats": dict(stats["by_page_count"]),
        "content_samples": stats["content_samples"]
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nОтчет сохранен в: {output_path}")

    # Выводим краткую сводку
    print("\n" + "=" * 60)
    print("КРАТКАЯ СВОДКА:")
    print(f"Всего документов: {report['summary']['total_documents']}")
    print(
        f"Русских/Английских: {report['summary']['documents_per_language'].get('rus', 0)}/{report['summary']['documents_per_language'].get('eng', 0)}")
    print(f"Среднее количество страниц: {report['summary']['avg_pages']:.1f}")
    print(f"Распределение по размеру: {report['summary']['size_distribution']}")

    print("\nРаспределение по количеству страниц:")
    for pages in sorted(stats["by_page_count"].keys()):
        count = stats["by_page_count"][pages]
        print(f"  {pages} страниц: {count} документов ({count / stats['total_files'] * 100:.1f}%)")


if __name__ == "__main__":
    # Пути
    BASE_DIR = Path(__file__).parent.parent
    TEXTS_DIR = BASE_DIR / "data" / "processed" / "extracted_texts"
    REPORT_PATH = BASE_DIR / "data" / "processed" / "text_analysis_report.json"

    # Анализируем
    print("Начинаю анализ текстовых файлов...")
    stats = analyze_texts(TEXTS_DIR)

    # Сохраняем отчет
    save_analysis_report(stats, REPORT_PATH)

    # Дополнительная информация
    print("\n" + "=" * 60)
    print("РЕКОМЕНДАЦИИ НА ОСНОВЕ АНАЛИЗА:")

    # Анализ коротких документов
    short_docs = sum(count for pages, count in stats["by_page_count"].items() if pages < 6)
    if short_docs > 0:
        print(f"1. Обнаружено {short_docs} коротких документов (<6 страниц).")
        print("   Это могут быть формы заявок или краткие объявления.")
        print("   Они будут полезны для обучения работе с формализованными данными.")

    # Анализ языков
    rus_count = stats["by_language"].get("rus", 0)
    eng_count = stats["by_language"].get("eng", 0)
    if eng_count > 0:
        print(f"2. Обнаружено {eng_count} документов на английском ({eng_count / stats['total_files'] * 100:.1f}%).")
        print("   Оставляем их - это усилит мультиязычные возможности модели.")

    # Общая оценка
    print("3. Общая оценка датасета: ПРИГОДЕН для fine-tuning.")
    print("   Ключевой этап - создание качественных синтетических QA-пар.")