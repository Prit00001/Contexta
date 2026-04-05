"""
services/arxiv_client.py — Robust arXiv paper fetching (no auth needed).
Handles retries, headers, and parsing safely.
"""

from __future__ import annotations
import hashlib
import logging
import asyncio
import xml.etree.ElementTree as ET
from typing import List
import httpx

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}


# ─────────────────────────────────────────────────────────────
# Fetch papers with retry + headers
# ─────────────────────────────────────────────────────────────
async def fetch_arxiv_papers(query: str, max_results: int = 8) -> List[dict]:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    headers = {
        "User-Agent": "semantic-search-app/1.0 (contact: your_email@example.com)"
    }

    retries = 3

    for attempt in range(retries):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    ARXIV_API,
                    params=params,
                    headers=headers
                )

            if response.status_code == 200:
                papers = _parse(response.text)
                logger.info("arXiv: fetched %d papers", len(papers))
                return papers

            logger.warning(
                "arXiv returned %s (attempt %d)",
                response.status_code,
                attempt + 1
            )

        except httpx.TimeoutException:
            logger.warning("arXiv timeout (attempt %d)", attempt + 1)

        except Exception as exc:
            logger.warning("arXiv error (attempt %d): %s", attempt + 1, exc)

        # wait before retry
        await asyncio.sleep(1)

    logger.warning("arXiv failed after %d attempts for query: %r", retries, query)
    return []


# ─────────────────────────────────────────────────────────────
# Parse XML response
# ─────────────────────────────────────────────────────────────
def _parse(xml_text: str) -> List[dict]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.warning("arXiv XML parse error")
        return []

    papers = []

    for entry in root.findall("atom:entry", NS):
        try:
            title_elem = entry.find("atom:title", NS)
            summary_elem = entry.find("atom:summary", NS)
            id_elem = entry.find("atom:id", NS)

            title = (title_elem.text or "").strip().replace("\n", " ") if title_elem is not None else ""
            summary = (summary_elem.text or "").strip().replace("\n", " ") if summary_elem is not None else ""
            url = (id_elem.text or "").strip() if id_elem is not None else ""

            if not summary:
                continue

            papers.append({
                "id": "arxiv_" + hashlib.md5(url.encode()).hexdigest()[:8],
                "title": title,
                "content": summary,
                "source": "arxiv",
                "url": url,
            })

        except Exception as e:
            logger.warning("Error parsing entry: %s", e)
            continue

    return papers