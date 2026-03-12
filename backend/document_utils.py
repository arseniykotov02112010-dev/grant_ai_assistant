"""
document_utils.py – извлечение и очистка текста из различных форматов документов.
Поддерживаются: PDF, DOCX.
"""

import re
import logging
from io import BytesIO
from typing import Optional

import fitz  # PyMuPDF
import docx

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Комплексная очистка извлечённого текста:
    - Удаление лишних пробелов и символов перевода строки
    - Склеивание слов, разорванных дефисом при переносе
    - Нормализация пунктуации
    - Удаление нечитаемых символов (сохраняем только буквы, цифры, знаки препинания)
    """
    if not text:
        return ""

    # 1. Заменяем мягкие переносы (soft hyphen) на пустоту
    text = text.replace('\xad', '')

    # 2. Убираем символы, которые не являются буквами/цифрами/знаками препинания/пробельными
    #    Оставляем кириллицу, латиницу, цифры, пробелы, переводы строк, основные знаки.
    allowed = r'[^\w\s\.,!?;:\-–—()\[\]{}«»"\'@#$%^&*+=\\|/]'
    text = re.sub(allowed, '', text, flags=re.UNICODE)

    # 3. Нормализуем дефисы и тире: разные варианты приводим к простому дефису (U+002D)
    text = re.sub(r'[–—]', '-', text)

    # 4. Склеиваем слова, разорванные переносом (дефис + конец строки)
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)

    # 5. Убираем лишние пробелы в начале/конце строк и множественные пробелы
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            line = re.sub(r'\s+', ' ', line)
            lines.append(line)
    text = '\n'.join(lines)

    # 6. Убираем пробелы перед знаками препинания
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)

    return text


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    if not pdf_bytes:
        raise ValueError("PDF-файл пуст")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    logger.info(f"PDF открыт, страниц: {len(doc)}")
    full_text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        cleaned_page = clean_text(page_text)
        if cleaned_page:
            full_text.append(cleaned_page)   # без маркера страницы
        else:
            full_text.append("")
    doc.close()
    return "\n\n".join(full_text)


def extract_text_from_docx(docx_bytes: bytes) -> str:
    """
    Извлекает текст из DOCX-файла, переданного в виде байтов.
    Возвращает очищенный текст (без разделения на страницы, так как DOCX не имеет страниц).
    """
    if not docx_bytes:
        raise ValueError("DOCX-файл пуст")

    try:
        doc = docx.Document(BytesIO(docx_bytes))
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        # Извлекаем текст из таблиц (опционально)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        paragraphs.append(cell.text)

        raw_text = "\n".join(paragraphs)
        cleaned = clean_text(raw_text)
        logger.info(f"DOCX обработан, извлечено {len(cleaned)} символов")
        return cleaned
    except Exception as e:
        logger.exception("Ошибка при извлечении текста из DOCX")
        raise RuntimeError(f"Не удалось обработать DOCX: {e}") from e