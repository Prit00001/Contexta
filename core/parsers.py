"""File-format parsers producing a flat list of chunking.Block (page number, section
heading, text). PDF text extraction falls back to OCR (tesseract) for pages with
little/no extractable text (scanned pages).

ponytail: PDF section headings are not detected (would need font-size heuristics
over PyMuPDF's "dict" text layout); only page_number is populated for PDFs. Upgrade
path: inspect span font sizes in page.get_text("dict") if PDF headings are needed.
"""
from __future__ import annotations

import io
import logging
import re
from typing import List, Optional

from core.chunking import Block

log = logging.getLogger("contexta.parsers")

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*)")
_MIN_PAGE_TEXT = 20  # shorter than this -> treat the PDF page as scanned, try OCR


def parse_pdf(data: bytes) -> List[Block]:
    import fitz  # PyMuPDF

    blocks: List[Block] = []
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if len(text) < _MIN_PAGE_TEXT:
                text = _ocr_page(page) or text
            if text:
                blocks.append(Block(text=text, page_number=i))
    finally:
        doc.close()
    return blocks


def _ocr_page(page) -> str:
    try:
        import pytesseract
        from PIL import Image

        pix = page.get_pixmap(dpi=200)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return pytesseract.image_to_string(img).strip()
    except Exception as exc:  # noqa: BLE001 - OCR is best-effort; a page with no text just yields no chunk
        log.warning("OCR failed for a PDF page: %s", exc)
        return ""


def parse_docx(data: bytes) -> List[Block]:
    import docx

    blocks: List[Block] = []
    doc = docx.Document(io.BytesIO(data))
    heading: Optional[str] = None
    buffer: List[str] = []

    def flush():
        text = "\n".join(buffer).strip()
        if text:
            blocks.append(Block(text=text, section_heading=heading))
        buffer.clear()

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        if para.style and para.style.name and para.style.name.lower().startswith("heading"):
            flush()
            heading = text
        else:
            buffer.append(text)
    flush()
    return blocks


def parse_text(data: bytes) -> List[Block]:
    """Shared by .txt and .md: paragraphs grouped under the most recent markdown heading."""
    content = data.decode("utf-8", errors="replace")
    blocks: List[Block] = []
    heading: Optional[str] = None
    buffer: List[str] = []

    def flush():
        text = "\n".join(buffer).strip()
        if text:
            blocks.append(Block(text=text, section_heading=heading))
        buffer.clear()

    for line in content.splitlines():
        m = _MD_HEADING.match(line.strip())
        if m:
            flush()
            heading = m.group(2).strip()
        elif line.strip():
            buffer.append(line.strip())
        else:
            flush()
    flush()
    return blocks


_PARSERS = {
    "application/pdf": parse_pdf,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": parse_docx,
    "text/plain": parse_text,
    "text/markdown": parse_text,
}


def parse_document(data: bytes, content_type: str) -> List[Block]:
    parser = _PARSERS.get(content_type)
    if parser is None:
        raise ValueError(f"Unsupported content type: {content_type}")
    return parser(data)
