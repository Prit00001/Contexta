"""
services/search_service.py — Top-level orchestrator.

Pipeline per request:
  1. (optional) fetch arXiv → embed → temp FAISS
  2. retrieve from permanent local FAISS
  3. merge + de-duplicate + re-rank by cosine score
  4. call OpenRouter (async) with RAG prompt
  5. return AskResponse
"""
from __future__ import annotations
import logging
from typing import List
from app.schemas import AskResponse, SourceDocument
from core.embeddings import get_embeddings_batch, EMBEDDING_MODEL
from core.generator import generate_answer, OPENROUTER_MODEL
from core.retriever import Retriever
from core.vector_store import VectorStore
from data.documents import SAMPLE_DOCUMENTS
from services.arxiv_client import fetch_arxiv_papers

logger = logging.getLogger(__name__)


class SearchService:
    def __init__(self) -> None:
        self._store     = VectorStore()
        self._retriever = Retriever(self._store)
        self._ready     = False

    # ── startup ───────────────────────────────────────────────────────
    def load_local_documents(self) -> None:
        logger.info("Embedding %d local documents…", len(SAMPLE_DOCUMENTS))
        texts      = [d["content"] for d in SAMPLE_DOCUMENTS]
        embeddings = get_embeddings_batch(texts)
        self._store.add_documents(SAMPLE_DOCUMENTS, embeddings)
        self._ready = True
        logger.info("SearchService ready  |  FAISS size: %d  |  LLM: %s",
                    self._store.size, OPENROUTER_MODEL)

    # ── request ───────────────────────────────────────────────────────
    async def ask(self, question: str, top_k: int = 5, use_arxiv: bool = True) -> AskResponse:
        if not self._ready:
            raise RuntimeError("Call load_local_documents() first.")

        # 1. arXiv live papers (per-request temp store)
        arxiv_docs: List[SourceDocument] = []
        if use_arxiv:
            arxiv_docs = await self._arxiv_retrieve(question, top_k=3)

        # 2. local FAISS
        local_docs = self._retriever.retrieve(question, top_k=top_k)

        # 3. merge → de-dup → re-rank
        seen:   set               = set()
        merged: List[SourceDocument] = []
        for doc in arxiv_docs + local_docs:
            if doc.id not in seen:
                seen.add(doc.id)
                merged.append(doc)
        merged.sort(key=lambda d: d.score, reverse=True)
        top_docs = merged[: top_k * 2]

        # 4. OpenRouter RAG answer (fully async)
        answer = await generate_answer(question, top_docs)

        return AskResponse(
            question      = question,
            answer        = answer,
            sources       = top_docs,
            total_sources = len(top_docs),
            model_used    = OPENROUTER_MODEL,
        )

    async def _arxiv_retrieve(self, query: str, top_k: int = 3) -> List[SourceDocument]:
        papers = await fetch_arxiv_papers(query, max_results=8)
        if not papers:
            return []
        tmp = VectorStore()
        tmp.add_documents(papers, get_embeddings_batch([p["content"] for p in papers]))
        return Retriever(tmp).retrieve(query, top_k=top_k)

    @property
    def index_size(self) -> int:
        return self._store.size


# module-level singleton — imported by routes.py
search_service = SearchService()
