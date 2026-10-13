"""Tiny text helpers shared by retrieval, arXiv querying and extractive answers."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a about above after again all also am an and any are as at be because been before being
    below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers him his how i if in into is it its just me more most my
    no nor not now of off on once only or other our out over own same she should so some such than
    that the their theirs them then there these they this those through to too under until up very
    was we were what when where which while who whom why will with would you your explain tell give
    describe work works use used using vs versus difference""".split()
)

_WORD = re.compile(r"[a-z0-9]+")
_SENT = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def keywords(text: str, limit: int = 6) -> list[str]:
    seen: dict[str, None] = {}
    for t in tokenize(text):
        if len(t) > 2 and t not in STOPWORDS:
            seen.setdefault(t)
    return sorted(seen, key=len, reverse=True)[:limit]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text.strip()) if s.strip()]


def lexical_score(query_terms: set[str], text: str) -> float:
    if not query_terms:
        return 0.0
    return len(query_terms & set(tokenize(text))) / len(query_terms)


def clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"
