"""Pipeline: embed -> (local FAISS || live arXiv) -> merge + re-rank -> grounded answer."""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import OrderedDict

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


class SearchService:
    def __init__(self, settings: Settings, embedder: Embedder, retriever: Retriever, arxiv: ArxivClient, generator: Generator):
        self.s = settings
        self.embedder = embedder
        self.retriever = retriever
        self.arxiv = arxiv
        self.generator = generator
        self._answers: OrderedDict[tuple, tuple[float, dict]] = OrderedDict()

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
            score = 0.8 * cos + 0.2 * lexical_score(terms, src["text"])  # hybrid: semantic + keyword
            cand = {**src, "score": score}
            k = _dedupe_key(cand)
            if k not in best or score > best[k]["score"]:
                best[k] = cand
        return sorted(best.values(), key=lambda x: x["score"], reverse=True)

    # ---- public API ----------------------------------------------------------
    async def ask(self, question: str, top_k: int, use_arxiv: bool) -> dict:
        question = " ".join(question.split())
        use_arxiv = use_arxiv and self.s.arxiv_enabled
        cache_key = (question.lower(), top_k, use_arxiv)
        hit = self._answers.get(cache_key)
        if hit and time.monotonic() - hit[0] < self.s.answer_cache_ttl:
            return {**hit[1], "cached": True}

        t0 = time.perf_counter()
        warnings: list[str] = []
        qvec = (await self.embedder.embed([question]))[0]
        candidates = self.retriever.search(qvec, top_k * 2)
        t_local = time.perf_counter()

        if use_arxiv:
            try:
                candidates += await asyncio.wait_for(self._arxiv_candidates(question, qvec), self.s.arxiv_budget_s)
            except Exception as exc:  # noqa: BLE001 - arXiv is best-effort
                log.warning("arXiv unavailable: %s", exc)
                warnings.append("Live arXiv search was unavailable; showing knowledge-base results only.")
        t_arxiv = time.perf_counter()

        ranked = self._rank(candidates, question)
        sources = [s for s in ranked if s["score"] >= self.s.min_score][:top_k]

        if not sources:
            result = {
                "answer": "I couldn't find relevant sources for that question. Try rephrasing it with more specific ML/NLP terms.",
                "provider": "none", "model_used": "none", "llm_used": False,
            }
        else:
            gen = await self.generator.generate(question, sources)
            result = {"answer": gen.answer, "provider": gen.provider, "model_used": gen.model, "llm_used": gen.llm_used}
            warnings += gen.warnings
            if not gen.llm_used and self.generator.providers:
                warnings.append("The AI model is busy right now, so this answer is assembled directly from the top sources.")
            elif not gen.llm_used:
                warnings.append("No AI model is configured, so this answer is assembled directly from the top sources.")
        t_end = time.perf_counter()

        response = {
            "question": question,
            **result,
            "sources": [
                {
                    "id": s["id"], "title": s["title"], "source": s["source"], "url": s["url"],
                    "score": round(s["score"], 4), "snippet": clip(s["text"].split(". ", 1)[-1], 300),
                    "authors": s.get("authors", []), "published": s.get("published", ""),
                }
                for s in sources
            ],
            "total_sources": len(sources),
            "cached": False,
            "warnings": warnings,
            "timings_ms": {
                "local": round((t_local - t0) * 1000), "arxiv": round((t_arxiv - t_local) * 1000),
                "generate": round((t_end - t_arxiv) * 1000), "total": round((t_end - t0) * 1000),
            },
        }
        if result["llm_used"] or not sources:  # don't pin a degraded answer in cache
            self._answers[cache_key] = (time.monotonic(), response)
            while len(self._answers) > 256:
                self._answers.popitem(last=False)
        return response
