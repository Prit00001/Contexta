"""Grounded Q&A over one workspace's ingested documents: hybrid retrieval -> generation
with citations -> page-accurate source list. Reuses core.generator.Generator unmodified,
so the existing provider fallback chain and extractive fallback apply here too."""
from __future__ import annotations

import asyncio
from typing import List, Optional

from core.generator import Generator
from core.text import clip, keywords, lexical_score, split_sentences
from services.retrieval import hybrid_search


def _highlight(content: str, question: str) -> str:
    terms = set(keywords(question, 8))
    sentences = split_sentences(content)
    if not sentences:
        return clip(content, 300)
    best = max(sentences, key=lambda s: (lexical_score(terms, s), -len(s)))
    return clip(best, 300)


async def ask_workspace(
    conn, generator: Generator, embedder, workspace_id: str, question: str, top_k: int = 5,
    allowed_group_ids: Optional[List[str]] = None,
) -> dict:
    question = " ".join(question.split())
    qvec = (await embedder.embed([question]))[0]
    hits = await asyncio.to_thread(hybrid_search, conn, workspace_id, question, qvec, top_k, allowed_group_ids)

    if not hits:
        return {
            "question": question,
            "answer": "The documents in this workspace don't contain information to answer that.",
            "sources": [], "provider": "none", "model_used": "none", "llm_used": False, "warnings": [],
        }

    sources = [{**h, "title": h["filename"], "source": "local", "text": h["content"]} for h in hits]
    gen = await generator.generate(question, sources)
    citations = [
        {
            "document_id": str(h["document_id"]), "filename": h["filename"], "page_number": h["page_number"],
            "section_heading": h["section_heading"], "snippet": _highlight(h["content"], question),
            "score": h["score"],
        }
        for h in hits
    ]
    return {
        "question": question, "answer": gen.answer, "sources": citations,
        "provider": gen.provider, "model_used": gen.model, "llm_used": gen.llm_used, "warnings": gen.warnings,
    }
