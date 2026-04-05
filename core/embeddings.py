"""
core/embeddings.py
Lazy-loads all-MiniLM-L6-v2 once, exposes single + batch embedding helpers.
Vectors are L2-normalised → FAISS IndexFlatIP == cosine similarity.
"""
from __future__ import annotations
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model '%s'…", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model ready.")
    return _model


def _normalise(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)


def get_embedding(text: str) -> np.ndarray:
    """Single string → normalised (DIM,) vector."""
    vecs = _get_model().encode([text], convert_to_numpy=True)
    return _normalise(vecs)[0]


def get_embeddings_batch(texts: list[str]) -> np.ndarray:
    """List of strings → normalised (N, DIM) matrix."""
    vecs = _get_model().encode(texts, convert_to_numpy=True,
                               batch_size=32, show_progress_bar=False)
    return _normalise(vecs)
