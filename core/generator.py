"""
core/generator.py
─────────────────────────────────────────────────────────────────────────────
OpenRouter LLM integration — production-ready, async, RAG-ready.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
import os
from typing import Any, List

import httpx
from dotenv import load_dotenv
from pathlib import Path

# Load .env if present locally — on Render this is a no-op (env vars are
# injected by the platform directly into os.environ before the app starts)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

# ── Static config (model, URLs) — these don't need to be secret ──────────
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _get_api_key() -> str:
    """Read key fresh from os.environ every call — never cache at import time."""
    return os.environ.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY", "")


def _get_model() -> str:
    return os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct:free")


def _get_site_url() -> str:
    return os.getenv("APP_SITE_URL", "http://localhost:8000")


def _get_site_name() -> str:
    return os.getenv("APP_SITE_NAME", "contexta-semantic-search")


# ── Main public function ──────────────────────────────────────────────────

async def generate_answer(question: str, context_docs: List[Any]) -> str:
    """
    Generate a grounded answer using OpenRouter.

    Args:
        question:     The user's natural-language question.
        context_docs: List of SourceDocument objects from FAISS retrieval.

    Returns:
        Plain text answer string.
    """
    api_key = _get_api_key()

    if not api_key:
        logger.warning(
            "OPENROUTER_API_KEY not found in environment. "
            "Check Render → Environment → OPENROUTER_API_KEY is set and redeployed."
        )
        return _extractive_fallback(question, context_docs)

    logger.info("OpenRouter ready → model: %s", _get_model())
    prompt = _build_prompt(question, context_docs)

    try:
        answer = await _call_openrouter(prompt, api_key)
        return answer
    except httpx.TimeoutException:
        logger.error("OpenRouter request timed out.")
        return _extractive_fallback(question, context_docs)
    except Exception as exc:
        logger.error("OpenRouter error: %s", exc)
        return _extractive_fallback(question, context_docs)


# ── Prompt builder ────────────────────────────────────────────────────────

def _build_prompt(question: str, context_docs: List[Any]) -> str:
    if not context_docs:
        return question

    blocks: list[str] = []
    for i, doc in enumerate(context_docs, 1):
        title   = getattr(doc, "title",   f"Document {i}")
        content = getattr(doc, "content", "")
        url     = getattr(doc, "url",     "") or ""
        blocks.append(f"[{i}] {title}\n{content}\nSource: {url}")

    context_text = "\n\n---\n\n".join(blocks)

    return (
        "You are a precise research assistant. "
        "Answer the question using ONLY the context documents below. "
        "Cite sources inline with their bracketed numbers e.g. [1], [2]. "
        "If the context lacks enough information, say so clearly.\n\n"
        f"### Context Documents\n{context_text}\n\n"
        f"### Question\n{question}\n\n"
        "### Answer"
    )


# ── OpenRouter API call ───────────────────────────────────────────────────

async def _call_openrouter(user_message: str, api_key: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  _get_site_url(),
        "X-Title":       _get_site_name(),
    }

    payload = {
        "model": _get_model(),
        "messages": [
            {
                "role":    "system",
                "content": (
                    "You are a knowledgeable, concise research assistant. "
                    "Always respond in clear, well-structured markdown."
                ),
            },
            {
                "role":    "user",
                "content": user_message,
            },
        ],
        "temperature": 0.2,
        "max_tokens":  1024,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(OPENROUTER_URL, json=payload, headers=headers)
        response.raise_for_status()

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        logger.error("Unexpected OpenRouter response shape: %s", data)
        raise ValueError(f"Could not parse OpenRouter response: {exc}") from exc


# ── Extractive fallback ───────────────────────────────────────────────────

def _extractive_fallback(question: str, docs: List[Any]) -> str:
    if not docs:
        return "No relevant documents found for your question."
    top     = docs[0]
    title   = getattr(top, "title",   "")
    content = getattr(top, "content", "")
    return (
        "**Note:** OpenRouter API key not configured — showing extractive result.\n\n"
        f"Based on **{title}**:\n{content[:600]}…\n\n"
        "_To enable AI-generated answers, set OPENROUTER_API_KEY in your .env file._"
    )
