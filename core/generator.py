"""
core/generator.py
─────────────────────────────────────────────────────────────────────────────
Groq LLM integration — production-ready, async, RAG-ready.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import os
from typing import Any, List

import httpx
from dotenv import load_dotenv
from pathlib import Path

# Load .env locally
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

# ── Groq Configuration ────────────────────────────────────────────────────

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


def _get_api_key() -> str:
    """Read key fresh from environment every call."""
    return os.environ.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY", "")


def _get_model() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ── Main public function ──────────────────────────────────────────────────

async def generate_answer(question: str, context_docs: List[Any]) -> str:
    """
    Generate a grounded answer using Groq.

    Args:
        question: User question.
        context_docs: Retrieved documents from FAISS.

    Returns:
        Generated answer.
    """

    api_key = _get_api_key()

    if not api_key:
        logger.warning(
            "GROQ_API_KEY not found in environment."
        )
        return _extractive_fallback(question, context_docs)

    logger.info("Groq ready → model: %s", _get_model())

    prompt = _build_prompt(question, context_docs)

    try:
        answer = await _call_groq(prompt, api_key)
        return answer

    except httpx.TimeoutException:
        logger.error("Groq request timed out.")
        return _extractive_fallback(question, context_docs)

    except Exception as exc:
        logger.error("Groq error: %s", exc)
        return _extractive_fallback(question, context_docs)


# ── Prompt Builder ────────────────────────────────────────────────────────

def _build_prompt(question: str, context_docs: List[Any]) -> str:

    if not context_docs:
        return question

    blocks = []

    for i, doc in enumerate(context_docs, 1):
        title = getattr(doc, "title", f"Document {i}")
        content = getattr(doc, "content", "")
        url = getattr(doc, "url", "") or ""

        blocks.append(
            f"[{i}] {title}\n{content}\nSource: {url}"
        )

    context_text = "\n\n---\n\n".join(blocks)

    return (
        "You are a precise research assistant. "
        "Answer ONLY using the provided documents. "
        "Cite sources inline like [1], [2]. "
        "If the answer is not contained in the context, clearly say so.\n\n"
        f"### Context Documents\n{context_text}\n\n"
        f"### Question\n{question}\n\n"
        "### Answer"
    )


# ── Groq API Call ─────────────────────────────────────────────────────────

async def _call_groq(user_message: str, api_key: str) -> str:

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": _get_model(),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable research assistant. "
                    "Respond clearly using markdown."
                ),
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            GROQ_URL,
            headers=headers,
            json=payload,
        )

        response.raise_for_status()

    data = response.json()

    try:
        return data["choices"][0]["message"]["content"].strip()

    except (KeyError, IndexError) as exc:
        logger.error("Unexpected Groq response shape: %s", data)
        raise ValueError(
            f"Could not parse Groq response: {exc}"
        ) from exc


# ── Extractive Fallback ───────────────────────────────────────────────────

def _extractive_fallback(question: str, docs: List[Any]) -> str:

    if not docs:
        return "No relevant documents found for your question."

    top = docs[0]

    title = getattr(top, "title", "")
    content = getattr(top, "content", "")

    return (
        "**Note:** GROQ_API_KEY not configured — showing extractive result.\n\n"
        f"Based on **{title}**:\n\n"
        f"{content[:600]}...\n\n"
        "_To enable AI-generated answers, set GROQ_API_KEY in your environment._"
    )
