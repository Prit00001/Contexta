"""
core/generator.py
─────────────────────────────────────────────────────────────────────────────
OpenRouter LLM integration — production-ready, async, RAG-ready.

Why OpenRouter?
  • Single endpoint → access to 200+ models (LLaMA, Mistral, Gemma, Claude…)
  • Free tier available (models marked :free)
  • Drop-in replacement for any OpenAI-compatible client

RAG-ready design:
  generate_answer(question, context_docs)
      ↑ context_docs can be [] for pure LLM mode
      ↑ or a list of SourceDocument objects from FAISS retrieval

How to get your API key:
  1. Go to  https://openrouter.ai
  2. Sign up (free, no credit card for free-tier models)
  3. Dashboard → Keys → "Create Key"
  4. Copy the key
  5. In your project:  cp .env.example .env
  6. Open .env and set:  OPENROUTER_API_KEY=sk-or-v1-...
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
import os
from typing import Any, List

import httpx
from dotenv import load_dotenv

from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

# ── Config (read from environment — never hardcode) ───────────────────────

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL:   str = os.getenv(
    "OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct:free"
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Shown in OpenRouter dashboard — helps with rate-limit debugging
APP_SITE_URL  = os.getenv("APP_SITE_URL",  "http://localhost:8000")
APP_SITE_NAME = os.getenv("APP_SITE_NAME", "shhhhh-semantic-search")

if OPENROUTER_API_KEY:
    logger.info("OpenRouter ready  →  model: %s", OPENROUTER_MODEL)
else:
    logger.warning(
        "OPENROUTER_API_KEY not set. "
        "Answers will use extractive fallback. "
        "Get a free key at https://openrouter.ai"
    )


# ── Main public function ──────────────────────────────────────────────────

async def generate_answer(question: str, context_docs: List[Any]) -> str:
    """
    Generate a grounded answer using OpenRouter.

    Args:
        question:     The user's natural-language question.
        context_docs: List of SourceDocument objects from FAISS retrieval.
                      Pass [] to use the LLM without RAG context.

    Returns:
        Plain text answer string.
    """
    if not OPENROUTER_API_KEY:
        return _extractive_fallback(question, context_docs)

    prompt = _build_prompt(question, context_docs)

    try:
        answer = await _call_openrouter(prompt)
        return answer
    except httpx.TimeoutException:
        logger.error("OpenRouter request timed out.")
        return _extractive_fallback(question, context_docs)
    except Exception as exc:
        logger.error("OpenRouter error: %s", exc)
        return _extractive_fallback(question, context_docs)


# ── Prompt builder (RAG-aware) ────────────────────────────────────────────

def _build_prompt(question: str, context_docs: List[Any]) -> str:
    """
    Build the user message.

    With context  → RAG prompt: answer ONLY from provided documents.
    Without context → open-ended LLM prompt.

    This function is the single place to change prompt strategy — 
    swap to chain-of-thought, step-back prompting, etc. here.
    """
    if not context_docs:
        return question   # pure LLM mode, no RAG

    # Build numbered context blocks from retrieved docs
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


# ── OpenRouter API call (async, httpx) ────────────────────────────────────

async def _call_openrouter(user_message: str) -> str:
    """
    Send a single-turn chat request to OpenRouter and return clean text.

    Uses httpx.AsyncClient so this is non-blocking inside FastAPI's
    async event loop — no thread-pool overhead.
    """
    headers = {
        "Authorization":  f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":   "application/json",
        # Optional but recommended by OpenRouter for analytics
        "HTTP-Referer":   APP_SITE_URL,
        "X-Title":        APP_SITE_NAME,
    }

    payload = {
        "model": OPENROUTER_MODEL,
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
        "temperature": 0.2,          # low = factual, deterministic
        "max_tokens":  1024,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(OPENROUTER_URL, json=payload, headers=headers)
        response.raise_for_status()   # raises HTTPStatusError on 4xx/5xx

    data = response.json()

    # ── Parse response ────────────────────────────────────────────────
    # OpenRouter follows OpenAI schema:
    # data["choices"][0]["message"]["content"]
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        logger.error("Unexpected OpenRouter response shape: %s", data)
        raise ValueError(f"Could not parse OpenRouter response: {exc}") from exc


# ── Extractive fallback (no API key / API down) ───────────────────────────

def _extractive_fallback(question: str, docs: List[Any]) -> str:
    """
    Returns a simple snippet from the top-ranked document.
    Used when OpenRouter is unavailable — keeps the API functional.
    """
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
