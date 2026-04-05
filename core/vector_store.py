"""
core/vector_store.py
FAISS index wrapper.  IndexFlatIP on L2-normalised vectors = exact cosine search.
Swap to IndexIVFFlat / HNSW for million-scale corpora.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Tuple
import faiss
import numpy as np
from core.embeddings import EMBEDDING_DIM

logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self) -> None:
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._id_map: Dict[int, dict]  = {}

    # ── write ─────────────────────────────────────────────────────────
    def add_documents(self, docs: List[dict], embeddings: np.ndarray) -> None:
        assert len(docs) == embeddings.shape[0]
        start = self._index.ntotal
        self._index.add(embeddings)
        for i, doc in enumerate(docs):
            self._id_map[start + i] = doc
        logger.info("FAISS: +%d docs  (total %d)", len(docs), self._index.ntotal)

    # ── read ──────────────────────────────────────────────────────────
    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[dict, float]]:
        if self._index.ntotal == 0:
            return []
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vec.reshape(1, -1).astype(np.float32), k)
        return [
            (self._id_map[int(idx)], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1 and int(idx) in self._id_map
        ]

    @property
    def size(self) -> int:
        return self._index.ntotal

    def reset(self) -> None:
        self._index.reset()
        self._id_map.clear()
