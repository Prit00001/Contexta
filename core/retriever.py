"""
core/retriever.py
Bridges the gap between a raw FAISS search and typed SourceDocument output.
Kept separate so it can be reused by both the main store and per-request
temp stores (arXiv).
"""
from __future__ import annotations
import logging
from typing import List
from app.schemas import SourceDocument
from core.embeddings import get_embedding
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, store: VectorStore) -> None:
        self._store = store

    def retrieve(self, query: str, top_k: int = 5) -> List[SourceDocument]:
        vec  = get_embedding(query)
        hits = self._store.search(vec, top_k=top_k)
        return [
            SourceDocument(
                id      = doc.get("id", "unknown"),
                title   = doc.get("title", "Untitled"),
                content = doc.get("content", ""),
                source  = doc.get("source", "local"),
                url     = doc.get("url"),
                score   = round(score, 4),
            )
            for doc, score in hits
        ]
