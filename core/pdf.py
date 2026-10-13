"""PDF text extraction and sentence-aware chunking for the /ask-pdf endpoint."""

from __future__ import annotations

import re
from io import BytesIO

from pypdf import PdfReader

_WS = re.compile(r"\s+")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")  # "retrie-\nval" -> "retrieval"
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def extract_pages(data: bytes) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for pages that have extractable text."""
    reader = PdfReader(BytesIO(data))
    if (
        reader.is_encrypted and reader.decrypt("") == 0
    ):  # many "encrypted" PDFs use an empty password
        raise ValueError("PDF is password-protected")
    pages = []
    for i, page in enumerate(reader.pages, 1):
        raw = page.extract_text() or ""
        text = _WS.sub(" ", _HYPHEN_BREAK.sub(r"\1\2", raw)).strip()
        if text:
            pages.append((i, text))
    return pages


def _sentences(text: str, max_len: int) -> list[str]:
    out = []
    for s in _SENT.split(text):
        # tables / references often have no sentence punctuation; hard-split them
        while len(s) > max_len:
            cut = s.rfind(" ", 0, max_len)
            cut = cut if cut > max_len // 2 else max_len
            out.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            out.append(s)
    return out


def chunk(
    pages: list[tuple[int, str]], size: int = 900, overlap: int = 1, min_len: int = 60
) -> list[dict]:
    """Group sentences into ~size-char chunks per page, carrying `overlap` sentences forward."""
    out: list[dict] = []
    for page, text in pages:
        buf: list[str] = []
        length = fresh = 0
        for s in _sentences(text, size):
            if buf and length + len(s) > size:
                out.append({"page": page, "text": " ".join(buf)})
                buf = buf[-overlap:] if overlap else []
                length, fresh = sum(len(x) + 1 for x in buf), 0
            buf.append(s)
            length += len(s) + 1
            fresh += 1
        if buf and fresh:
            out.append({"page": page, "text": " ".join(buf)})
    return [c for c in out if len(c["text"]) >= min_len]
