import csv
from pathlib import Path


def create_catalog(texts_dir, output_path):
    """Создает CSV-каталог всех документов."""
    base_dir = Path(texts_dir)
    text_files = list(base_dir.glob("*.txt"))

    catalog = []
    for txt_file in text_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Извлекаем первую строку (часто это заголовок)
        first_line = content.split('\n')[0][:200]

        # Определяем тип по содержимому
        if "конкурс" in content[:1000].lower():
            doc_type = "конкурсная_документация"
        elif "заявк" in content[:1000].lower():
            doc_type = "форма_заявки"
        elif "титульн" in content[:500].lower():
            doc_type = "титульный_лист"
        else:
            doc_type = "прочий_документ"

        catalog.append({
            "filename": txt_file.name,
            "original_pdf": txt_file.stem + ".pdf",
            "pages": content.count("=== Страница") if "=== Страница" in content else 1,
            "size_chars": len(content),
            "type": doc_type,
            "preview": first_line
        })

    # Сохраняем в CSV
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=catalog[0].keys())
        writer.writeheader()
        writer.writerows(catalog)

    print(f"Каталог создан: {output_path}")
    print(f"Записей: {len(catalog)}")


if __name__ == "__main__":
    BASE_DIR = Path(__file__).parent.parent
    TEXTS_DIR = BASE_DIR / "data" / "processed" / "extracted_texts"
    CATALOG_PATH = BASE_DIR / "data" / "processed" / "document_catalog.csv"

    create_catalog(TEXTS_DIR, CATALOG_PATH)