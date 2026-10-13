"""Async arXiv search with caching, polite rate limiting and defensive XML parsing."""
from __future__ import annotations

import asyncio
import logging
import re
import time

import httpx
from defusedxml import ElementTree as ET

from core.text import keywords

log = logging.getLogger("contexta.arxiv")
API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}


class ArxivClient:
    def __init__(self, client: httpx.AsyncClient, max_results: int = 12, ttl: int = 3600, min_interval: float = 3.0):
        self.client = client
        self.max_results = max_results
        self.ttl = ttl
        self.min_interval = min_interval  # arXiv asks for >= 3s between calls
        self._cache: dict[str, tuple[float, list[dict]]] = {}
        self._lock = asyncio.Lock()
        self._last = 0.0

    @staticmethod
    def build_query(question: str) -> str:
        kws = keywords(question, 5)
        joiner = " AND " if len(kws) <= 2 else " OR "  # OR for recall; we re-rank by embedding anyway
        return joiner.join(f"all:{k}" for k in kws)

    async def search(self, question: str) -> list[dict]:
        query = self.build_query(question)
        if not query:
            return []
        hit = self._cache.get(query)
        if hit and time.monotonic() - hit[0] < self.ttl:
            return hit[1]

        async with self._lock:
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                r = await self.client.get(
                    API,
                    params={"search_query": query, "start": 0, "max_results": self.max_results, "sortBy": "relevance"},
                    timeout=8.0,
                )
            finally:
                self._last = time.monotonic()
        r.raise_for_status()
        papers = self.parse(r.text)
        if len(self._cache) >= 128:
            self._cache.pop(next(iter(self._cache)))
        self._cache[query] = (time.monotonic(), papers)
        return papers

    @staticmethod
    def parse(xml_text: str) -> list[dict]:
        papers = []
        for e in ET.fromstring(xml_text).findall("a:entry", NS):
            title = " ".join((e.findtext("a:title", "", NS) or "").split())
            summary = " ".join((e.findtext("a:summary", "", NS) or "").split())
            url = (e.findtext("a:id", "", NS) or "").strip().replace("http://", "https://")
            if not title or not summary or not url:
                continue
            authors = [n.text for n in e.findall("a:author/a:name", NS) if n.text][:3]
            papers.append(
                {
                    "id": "arxiv_" + re.sub(r"v\d+$", "", url.rsplit("/abs/", 1)[-1]),
                    "source": "arxiv",
                    "title": title,
                    "url": url,
                    "text": f"{title}. {summary}",
                    "authors": authors,
                    "published": (e.findtext("a:published", "", NS) or "")[:10],
                }
            )
        return papers
