"""Runs the full ingestion pipeline for one document: fetch bytes -> parse -> chunk
-> embed -> replace chunks -> mark ready/failed. Shared by the worker loop and tests."""
from __future__ import annotations

import logging
from typing import Optional

from app.config import Settings
from core import db
from core.chunking import chunk_blocks
from core.embeddings import Embedder
from core.parsers import parse_document
from core.storage import get_storage

log = logging.getLogger("contexta.ingestion")

_embedder: Optional[Embedder] = None


def _get_embedder(settings: Settings) -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder(settings.embed_backend, settings.embed_model, settings.model_cache_dir)
    return _embedder


def run_ingestion(conn, settings: Settings, document_id: str) -> None:
    doc = db.get_document(conn, document_id)
    db.set_document_status(conn, document_id, "processing")
    storage = get_storage(settings)
    data = storage.load(doc["storage_path"])
    blocks = parse_document(data, doc["content_type"])
    chunks = chunk_blocks(blocks)
    if not chunks:
        db.set_document_status(conn, document_id, "failed", error="No extractable text found in this file.")
        return
    embedder = _get_embedder(settings)
    vectors = embedder.embed_sync([c.content for c in chunks])
    db.replace_chunks(conn, document_id, doc["workspace_id"], doc["access_group_id"], chunks, vectors)
    db.set_document_status(conn, document_id, "ready")
    log.info("Ingested %s: %d chunks", doc["filename"], len(chunks))
