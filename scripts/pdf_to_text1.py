import os
import fitz  # PyMuPDF
import re
from pathlib import Path
import sys


def clean_text(text):
    """Базовая очистка текста: удаление лишних пробелов, нормализация переносов."""
    # Заменяем множественные пробелы и табы на один пробел
    text = re.sub(r'\s+', ' ', text)
    # Убираем "висящие" дефисы в конце строк (которые появились из-за переноса слов в PDF)
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    # Удаляем контрольные символы, кроме нормальных пунктуации и букв
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    return text.strip()


def pdf_to_text(pdf_path, output_dir):
    """Конвертирует один PDF-файл в текстовый."""
    try:
        doc = fitz.open(pdf_path)
        full_text = []

        for page_num, page in enumerate(doc, start=1):
            # Извлекаем текст с сохранением порядка
            page_text = page.get_text("text")
            if page_text.strip():
                cleaned = clean_text(page_text)
                full_text.append(f"=== Страница {page_num} ===\n{cleaned}")

        doc.close()

        if not full_text:
            return f"ВНИМАНИЕ: {pdf_path} не содержит извлекаемого текста (возможно, сканы?)"

        # Сохраняем в файл
        output_filename = Path(pdf_path).stem + ".txt"
        output_path = os.path.join(output_dir, output_filename)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(full_text))

        return f"УСПЕХ: {pdf_path} -> {output_path} ({len(full_text)} страниц)"

    except Exception as e:
        return f"ОШИБКА: {pdf_path} - {str(e)}"


def batch_convert(input_dir, output_dir):
    """Пакетная конвертация всех PDF-файлов в директории."""
    # Создаем выходную директорию, если не существует
    os.makedirs(output_dir, exist_ok=True)

    # Счетчики для статистики
    success_count = 0
    warning_count = 0
    error_count = 0
    results = []

    # Получаем список PDF-файлов
    pdf_files = list(Path(input_dir).glob("*.pdf"))

    print(f"Найдено {len(pdf_files)} PDF-файлов для обработки...")
    print("-" * 60)

    for pdf_file in pdf_files:
        result = pdf_to_text(str(pdf_file), output_dir)
        results.append(result)

        if result.startswith("УСПЕХ"):
            success_count += 1
        elif result.startswith("ВНИМАНИЕ"):
            warning_count += 1
        else:
            error_count += 1

        # Выводим прогресс
        current = success_count + warning_count + error_count
        if current % 10 == 0:
            print(f"Обработано: {current}/{len(pdf_files)}")

    # Сохраняем полный лог
    log_path = os.path.join(output_dir, "conversion_log.txt")
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("\n".join(results))

    print("\n" + "=" * 60)
    print("КОНВЕРТАЦИЯ ЗАВЕРШЕНА:")
    print(f"  Успешно: {success_count}")
    print(f"  С предупреждениями: {warning_count}")
    print(f"  С ошибками: {error_count}")
    print(f"Полный лог сохранен в: {log_path}")

    return results


if __name__ == "__main__":
    # Параметры путей
    BASE_DIR = Path(__file__).parent.parent
    INPUT_DIR = BASE_DIR / "data" / "raw_pdfs"
    OUTPUT_DIR = BASE_DIR / "data" / "processed" / "extracted_texts"

    # Проверяем существование входной директории
    if not INPUT_DIR.exists():
        print(f"ОШИБКА: Входная директория не найдена: {INPUT_DIR}")
        print("Убедитесь, что вы поместили PDF-файлы в правильную папку.")
        sys.exit(1)

    print(f"Входная директория: {INPUT_DIR}")
    print(f"Выходная директория: {OUTPUT_DIR}")

    # Запускаем пакетную конвертацию
    results = batch_convert(INPUT_DIR, OUTPUT_DIR)

    # Быстрая проверка первых нескольких результатов
    print("\nПервые 5 результатов:")
    for i, result in enumerate(results[:5]):
        print(f"{i + 1}. {result}")