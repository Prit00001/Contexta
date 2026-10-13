"""Sentence-aware chunker: packs sentences into ~max_chars chunks without splitting
mid-sentence, carrying each source block's page number and section heading as metadata."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from core.text import split_sentences


@dataclass
class Block:
    text: str
    page_number: Optional[int] = None
    section_heading: Optional[str] = None


@dataclass
class Chunk:
    chunk_index: int
    content: str
    page_number: Optional[int] = None
    section_heading: Optional[str] = None


def chunk_blocks(blocks: List[Block], max_chars: int = 900) -> List[Chunk]:
    chunks: List[Chunk] = []
    idx = 0
    for block in blocks:
        sentences = split_sentences(block.text) or ([block.text] if block.text.strip() else [])
        current = ""
        for sent in sentences:
            candidate = f"{current} {sent}".strip() if current else sent
            if current and len(candidate) > max_chars:
                chunks.append(Chunk(idx, current.strip(), block.page_number, block.section_heading))
                idx += 1
                current = sent
            else:
                current = candidate
        if current.strip():
            chunks.append(Chunk(idx, current.strip(), block.page_number, block.section_heading))
            idx += 1
    return chunks
