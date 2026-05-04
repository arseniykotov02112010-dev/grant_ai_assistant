"""
document_utils.py – извлечение и очистка текста из PDF и DOCX с сохранением таблиц.
"""

import re
import logging
from io import BytesIO
from collections import Counter
from typing import Optional, List, Tuple

import fitz
import docx
import pdfplumber

logger = logging.getLogger(__name__)

Y_TOLERANCE = 3.0
TAB_GAP = 50.0
HEADER_FOOTER_RATIO = 0.9

PDFPLUMBER_TABLE_SETTINGS = {
    "vertical_strategy": "lines",
    "horizontal_strategy": "lines",
    "snap_y_tolerance": 3,
    "intersection_x_tolerance": 5,
    "text_x_tolerance": 3,
    "text_y_tolerance": 3,
}


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('===') and stripped.endswith('==='):
            continue
        if re.fullmatch(r'[\s\.,;:!?…\-–—]+', stripped):
            continue
        if stripped:
            cleaned_lines.append(stripped)
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    lines = [line.strip() for line in text.splitlines()]
    return '\n'.join(lines)


def _table_to_markdown_from_pdfplumber(table_data: List[List[str]]) -> str:
    if not table_data or len(table_data) < 2:
        return ""
    rows = []
    for row in table_data:
        rows.append(["" if cell is None else str(cell).replace('\n', ' ') for cell in row])
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append("")
    valid_cols = []
    for col_idx in range(max_cols):
        if any(rows[row_idx][col_idx].strip() for row_idx in range(len(rows))):
            valid_cols.append(col_idx)
    if not valid_cols:
        return ""
    filtered_rows = []
    for row in rows:
        filtered_rows.append([row[c] for c in valid_cols])
    filtered_rows = [r for r in filtered_rows if any(cell.strip() for cell in r)]
    if len(filtered_rows) < 2:
        return ""
    header = "| " + " | ".join(filtered_rows[0]) + " |"
    separator = "|" + "|".join(["---"] * len(valid_cols)) + "|"
    body = "\n".join("| " + " | ".join(row) + " |" for row in filtered_rows[1:])
    return f"{header}\n{separator}\n{body}"


def _is_pdf_content(file_bytes: bytes) -> bool:
    return file_bytes.lstrip(b'\x00').startswith(b'%PDF')


def _is_docx_content(file_bytes: bytes) -> bool:
    import zipfile
    try:
        with zipfile.ZipFile(BytesIO(file_bytes)) as z:
            return '[Content_Types].xml' in z.namelist()
    except Exception:
        return False


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    if not pdf_bytes:
        raise ValueError("PDF-файл пуст")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pdf = pdfplumber.open(BytesIO(pdf_bytes))
    all_pages_text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_elements = []
        mu_tables = page.find_tables()
        table_zones = []
        if mu_tables:
            for t in mu_tables:
                try:
                    bbox_raw = t.bbox if hasattr(t, 'bbox') and t.bbox else getattr(t, 'rect', None)
                    if bbox_raw:
                        bbox = fitz.Rect(bbox_raw) if isinstance(bbox_raw, tuple) else bbox_raw
                        if bbox.height > 0:
                            table_zones.append(bbox)
                except Exception:
                    pass
        pl_page = pdf.pages[page_num]
        pl_raw_tables = pl_page.find_tables(PDFPLUMBER_TABLE_SETTINGS)
        pl_tables_data = []
        for pl_table in pl_raw_tables:
            bbox_pl = pl_table.bbox
            if bbox_pl:
                bbox_fitz = fitz.Rect(*bbox_pl)
                cells = pl_table.extract()
                if cells:
                    md = _table_to_markdown_from_pdfplumber(cells)
                    if md:
                        pl_tables_data.append((bbox_fitz, md))
        used_pl = set()
        for bbox_mu in table_zones:
            best_md = None
            best_iou = 0.0
            best_idx = -1
            for i, (bbox_pl, md) in enumerate(pl_tables_data):
                if i in used_pl:
                    continue
                x0 = max(bbox_mu.x0, bbox_pl.x0)
                y0 = max(bbox_mu.y0, bbox_pl.y0)
                x1 = min(bbox_mu.x1, bbox_pl.x1)
                y1 = min(bbox_mu.y1, bbox_pl.y1)
                if x0 < x1 and y0 < y1:
                    inter_area = (x1 - x0) * (y1 - y0)
                    union_area = abs(bbox_mu) + abs(bbox_pl) - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                        best_md = md
                        best_idx = i
            if best_md and best_iou > 0.4:
                page_elements.append((bbox_mu.y0, "table", best_md))
                used_pl.add(best_idx)
        for i, (bbox_pl, md) in enumerate(pl_tables_data):
            if i not in used_pl:
                page_elements.append((bbox_pl.y0, "table", md))
                used_pl.add(i)
        words = page.get_text("words", sort=True)
        if words:
            current_line_y = None
            current_line_words = []
            lines = []
            for w in words:
                x0, y0, x1, y1, word = w[:5]
                in_table = any(
                    (x0 < bbox.x1 and x1 > bbox.x0 and y0 < bbox.y1 and y1 > bbox.y0)
                    for bbox in table_zones
                )
                if in_table:
                    continue
                if current_line_y is None or abs(y0 - current_line_y) > Y_TOLERANCE:
                    if current_line_words:
                        lines.append((current_line_y, current_line_words))
                    current_line_y = y0
                    current_line_words = [(x0, word, x1)]
                else:
                    current_line_words.append((x0, word, x1))
            if current_line_words:
                lines.append((current_line_y, current_line_words))
            for y_line, word_triples in lines:
                sorted_words = sorted(word_triples, key=lambda t: t[0])
                line_parts = []
                prev_x_end = None
                for x0, word, x1 in sorted_words:
                    if prev_x_end is not None and (x0 - prev_x_end) > TAB_GAP:
                        line_parts.append("\t")
                    line_parts.append(word)
                    prev_x_end = x1
                line_text = " ".join(line_parts)
                page_elements.append((y_line, "text", line_text))
        page_elements.sort(key=lambda e: e[0])
        page_lines = []
        for _, etype, content in page_elements:
            if etype == "table":
                page_lines.append(f"\n[ТАБЛИЦА]\n{content}\n[/ТАБЛИЦА]")
            else:
                page_lines.append(content)
        page_text = "\n".join(page_lines)
        page_text = clean_text(page_text)
        all_pages_text.append(page_text if page_text.strip() else "")
    doc.close()
    pdf.close()
    total_pages = len(all_pages_text)
    if total_pages <= 5:
        return "\n[PAGE_BREAK]\n".join(all_pages_text)
    normalized_lines_per_page = []
    for page_txt in all_pages_text:
        lines = [line.strip().lower() for line in page_txt.splitlines() if line.strip()]
        normalized_lines_per_page.append(lines)
    line_counter = Counter()
    for lines in normalized_lines_per_page:
        line_counter.update(set(lines))
    threshold = max(2, int(total_pages * HEADER_FOOTER_RATIO))
    header_footer_candidates = {l for l, c in line_counter.items() if c >= threshold}
    number_lines = {l for l in header_footer_candidates if re.fullmatch(r'\d+', l) and len(l) < 5}
    final_remove = set(number_lines)
    for candidate in header_footer_candidates:
        if candidate in number_lines:
            continue
        words = candidate.split()
        has_long_words = any(len(w) > 3 for w in words)
        if not has_long_words or "страница" in candidate or "page" in candidate:
            final_remove.add(candidate)
    cleaned_pages = []
    for page_txt in all_pages_text:
        lines = page_txt.splitlines()
        filtered = [line for line in lines if line.strip().lower() not in final_remove]
        cleaned_pages.append("\n".join(filtered))
    return "\n[PAGE_BREAK]\n".join(cleaned_pages)


def extract_text_from_docx(docx_bytes: bytes) -> str:
    if not docx_bytes:
        raise ValueError("DOCX-файл пуст")
    doc = docx.Document(BytesIO(docx_bytes))
    elements = []
    for element in doc.element.body:
        if element.tag.endswith('}p'):
            para = docx.text.paragraph.Paragraph(element, doc)
            if para.text.strip():
                elements.append(para.text)
        elif element.tag.endswith('}tbl'):
            table = docx.table.Table(element, doc)
            rows = []
            col_count = 0
            for row in table.rows:
                cells = [cell.text.replace("\n", " ").replace("|", " ").replace("\t", " ") for cell in row.cells]
                col_count = max(col_count, len(cells))
                rows.append(cells)
            if not rows:
                continue
            for row in rows:
                while len(row) < col_count:
                    row.append("")
            md_lines = []
            header = "| " + " | ".join(rows[0]) + " |"
            separator = "|" + "|".join(["---"] * col_count) + "|"
            md_lines.append(header)
            md_lines.append(separator)
            for row in rows[1:]:
                md_lines.append("| " + " | ".join(row) + " |")
            elements.append("[ТАБЛИЦА]\n" + "\n".join(md_lines) + "\n[/ТАБЛИЦА]")
    raw_text = "\n".join(elements)
    return clean_text(raw_text)