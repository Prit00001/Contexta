"""Local knowledge base: embeds documents once, caches vectors on disk, searches with FAISS."""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np

from core.embeddings import Embedder
from core.vector_store import VectorStore

log = logging.getLogger("contexta.retriever")


class Retriever:
    def __init__(self, embedder: Embedder, cache_path: Path):
        self.embedder = embedder
        self.cache_path = cache_path
        self.store = VectorStore(embedder.dim)

    # Persistent embedding cache, keyed by content hash: editing a document re-embeds only that one.
    def _load_cache(self) -> dict[str, np.ndarray]:
        try:
            with np.load(self.cache_path, allow_pickle=False) as z:
                return dict(zip(z["keys"].tolist(), z["vecs"]))
        except Exception:  # missing/corrupt cache is fine
            return {}

    def _save_cache(self, cache: dict[str, np.ndarray]) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cache_path.with_name(self.cache_path.stem + ".tmp.npz")
            np.savez(tmp, keys=np.array(list(cache)), vecs=np.vstack(list(cache.values())))
            os.replace(tmp, self.cache_path)
        except OSError as exc:
            log.warning("Could not write embedding cache: %s", exc)

    def build(self, docs: list[dict]) -> None:
        texts = [f"{d['title']}. {d['content']}" for d in docs]
        tag = f"{self.embedder.backend}:{self.embedder.model_name}:"
        keys = [hashlib.sha1((tag + t).encode()).hexdigest() for t in texts]
        cache = self._load_cache()
        missing = [i for i, k in enumerate(keys) if k not in cache]
        if missing:
            log.info("Embedding %d/%d documents (rest from cache)", len(missing), len(docs))
            vecs = self.embedder.embed_sync([texts[i] for i in missing])
            for i, v in zip(missing, vecs):
                cache[keys[i]] = v
            self._save_cache({k: cache[k] for k in keys})
        self.store = VectorStore(self.embedder.dim)
        self.store.add(np.vstack([cache[k] for k in keys]), [{**d, "text": t} for d, t in zip(docs, texts)])

    @property
    def size(self) -> int:
        return len(self.store)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[dict, float]]:
        return self.store.search(query_vec, k)
