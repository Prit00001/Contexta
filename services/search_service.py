"""Pipeline: embed -> (local FAISS || live arXiv || uploaded PDF) -> merge + re-rank -> grounded answer."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections import OrderedDict

import numpy as np

from app.config import Settings
from core.embeddings import Embedder
from core.generator import Generator
from core.retriever import Retriever
from core.text import clip, keywords, lexical_score
from services.arxiv_client import ArxivClient

log = logging.getLogger("contexta.search")
_ARXIV_ID = re.compile(r"arxiv\.org/abs/([\w.\-/]+?)(v\d+)?/?$")


def _dedupe_key(s: dict) -> str:
    m = _ARXIV_ID.search(s["url"].lower())
    return f"arxiv:{m.group(1)}" if m else re.sub(r"\W+", "", s["title"].lower())


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def _hybrid(terms: set[str], text: str, cos: float) -> float:
    return 0.8 * cos + 0.2 * lexical_score(terms, text)  # semantic + keyword


def _source_out(s: dict) -> dict:
    snippet = s["text"] if s["source"] == "pdf" else s["text"].split(". ", 1)[-1]
    return {
        "id": s["id"], "title": s["title"], "source": s["source"], "url": s.get("url", ""),
        "score": round(s["score"], 4), "snippet": clip(snippet, 300),
        "authors": s.get("authors", []), "published": s.get("published", ""), "page": s.get("page"),
    }


class SearchService:
    def __init__(self, settings: Settings, embedder: Embedder, retriever: Retriever, arxiv: ArxivClient, generator: Generator):
        self.s = settings
        self.embedder = embedder
        self.retriever = retriever
        self.arxiv = arxiv
        self.generator = generator
        self._answers: OrderedDict[tuple, tuple[float, dict]] = OrderedDict()
        # Per-instance cache of PDF chunk embeddings, so follow-up questions on the same
        # file skip re-embedding when they land on a warm instance. Best-effort on serverless.
        self._pdf_vecs: OrderedDict[str, np.ndarray] = OrderedDict()

    # ---- candidate gathering -------------------------------------------------
    async def _arxiv_candidates(self, question: str, qvec) -> list[tuple[dict, float]]:
        papers = await self.arxiv.search(question)
        if not papers:
            return []
        vecs = await self.embedder.embed([p["text"] for p in papers])
        return [(p, float(v @ qvec)) for p, v in zip(papers, vecs)]

    @staticmethod
    def _rank(candidates: list[tuple[dict, float]], question: str) -> list[dict]:
        terms = set(keywords(question, 8))
        best: dict[str, dict] = {}
        for src, cos in candidates:
            cand = {**src, "score": _hybrid(terms, src["text"], cos)}
            k = _dedupe_key(cand)
            if k not in best or cand["score"] > best[k]["score"]:
                best[k] = cand
        return sorted(best.values(), key=lambda x: x["score"], reverse=True)

    # ---- shared helpers ------------------------------------------------------
    def _cached(self, key: tuple) -> dict | None:
        hit = self._answers.get(key)
        if hit and time.monotonic() - hit[0] < self.s.answer_cache_ttl:
            return {**hit[1], "cached": True}
        return None

    def _store(self, key: tuple, response: dict) -> None:
        if response["llm_used"] or not response["sources"]:  # don't pin a degraded answer in cache
            self._answers[key] = (time.monotonic(), response)
            while len(self._answers) > 256:
                self._answers.popitem(last=False)

    async def _answer(self, question: str, sources: list[dict], diagnostics: list[str], empty_msg: str) -> dict:
        if not sources:
            return {"answer": empty_msg, "provider": "none", "model_used": "none", "llm_used": False}
        gen = await self.generator.generate(question, sources)
        diagnostics += gen.warnings
        if not gen.llm_used:
            diagnostics.append(
                "All AI providers failed; answer assembled from top sources."
                if self.generator.providers else "No AI provider configured; answer assembled from top sources."
            )
        return {"answer": gen.answer, "provider": gen.provider, "model_used": gen.model, "llm_used": gen.llm_used}

    # ---- public API ----------------------------------------------------------
    async def ask(self, question: str, top_k: int, use_arxiv: bool) -> dict:
        question = " ".join(question.split())
        use_arxiv = use_arxiv and self.s.arxiv_enabled
        cache_key = (question.lower(), top_k, use_arxiv)
        if hit := self._cached(cache_key):
            return hit

        t0 = time.perf_counter()
        warnings: list[str] = []
        diagnostics: list[str] = []
        qvec = (await self.embedder.embed([question]))[0]
        candidates = self.retriever.search(qvec, top_k * 2)
        t_local = time.perf_counter()

        if use_arxiv:
            try:
                candidates += await asyncio.wait_for(self._arxiv_candidates(question, qvec), self.s.arxiv_budget_s)
            except Exception as exc:  # noqa: BLE001 - arXiv is best-effort
                log.warning("arXiv unavailable: %s", exc)
                warnings.append("Live arXiv search was unavailable, so only library papers were used.")
        t_arxiv = time.perf_counter()

        ranked = self._rank(candidates, question)
        sources = [s for s in ranked if s["score"] >= self.s.min_score][:top_k]
        result = await self._answer(
            question, sources, diagnostics,
            "I couldn't find relevant sources for that question. Try rephrasing it with more specific ML/NLP terms.",
        )
        t_end = time.perf_counter()

        response = {
            "question": question,
            **result,
            "sources": [_source_out(s) for s in sources],
            "total_sources": len(sources),
            "cached": False,
            "warnings": warnings,
            "diagnostics": diagnostics,
            "timings_ms": {
                "local": round((t_local - t0) * 1000), "arxiv": round((t_arxiv - t_local) * 1000),
                "generate": round((t_end - t_arxiv) * 1000), "total": round((t_end - t0) * 1000),
            },
        }
        self._store(cache_key, response)
        return response

    async def ask_pdf(self, question: str, chunks: list[dict], top_k: int, filename: str) -> dict:
        """Answer from an uploaded PDF. `chunks` are {"page": int, "text": str} from core.pdf.chunk."""
        question = " ".join(question.split())
        warnings: list[str] = []
        diagnostics: list[str] = []

        if len(chunks) > self.s.max_pdf_chunks:
            chunks = chunks[: self.s.max_pdf_chunks]
            warnings.append(f"This PDF is long, so only pages 1–{chunks[-1]['page']} were searched.")

        doc_key = hashlib.sha1("\x00".join(c["text"] for c in chunks).encode()).hexdigest()
        cache_key = ("pdf", doc_key, question.lower(), top_k)
        if hit := self._cached(cache_key):
            return hit

        t0 = time.perf_counter()
        vecs = self._pdf_vecs.get(doc_key)
        if vecs is None:
            vecs = _unit(np.asarray(await self.embedder.embed([c["text"] for c in chunks]), dtype=np.float32))
            self._pdf_vecs[doc_key] = vecs
            while len(self._pdf_vecs) > 8:
                self._pdf_vecs.popitem(last=False)
        else:
            self._pdf_vecs.move_to_end(doc_key)
        qvec = _unit(np.asarray((await self.embedder.embed([question]))[0], dtype=np.float32))
        t_embed = time.perf_counter()

        # Rank every chunk; no min_score cut, since the user chose this document deliberately.
        terms = set(keywords(question, 8))
        cos = vecs @ qvec
        pool = np.argsort(-cos)[: max(top_k * 3, 10)]
        scored = sorted(
            (
                {
                    "id": f"pdf-{i}", "title": f"{filename} · p. {chunks[i]['page']}", "source": "pdf",
                    "url": "", "text": chunks[i]["text"], "page": chunks[i]["page"],
                    "score": _hybrid(terms, chunks[i]["text"], float(cos[i])),
                }
                for i in pool
            ),
            key=lambda x: x["score"],
            reverse=True,
        )
        sources = scored[:top_k]
        result = await self._answer(question, sources, diagnostics, "No readable text was found in this PDF.")
        t_end = time.perf_counter()

        response = {
            "question": question,
            **result,
            "sources": [_source_out(s) for s in sources],
            "total_sources": len(sources),
            "cached": False,
            "warnings": warnings,
            "diagnostics": diagnostics,
            "timings_ms": {
                "embed": round((t_embed - t0) * 1000), "generate": round((t_end - t_embed) * 1000),
                "total": round((t_end - t0) * 1000),
            },
        }
        self._store(cache_key, response)
        return response