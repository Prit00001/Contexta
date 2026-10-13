from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(5, ge=1, le=10)
    use_arxiv: bool = True


class Source(BaseModel):
    id: str
    title: str
    source: str  # "local" | "arxiv" | "pdf"
    url: str = ""  # empty for uploaded PDFs
    score: float
    snippet: str = ""
    authors: list[str] = []
    published: str = ""
    page: Optional[int] = (
        None  # set for uploaded PDFs (Optional, not `int | None`, for Python 3.9)
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    total_sources: int
    model_used: str
    provider: str
    llm_used: bool
    cached: bool
    warnings: list[str] = []  # user-facing notes (arXiv down, PDF truncated, ...)
    diagnostics: list[str] = (
        []
    )  # provider errors for debugging; the UI logs these to the console
    timings_ms: dict[str, int] = {}


class HealthResponse(BaseModel):
    status: str
    index_size: int
    model: str
    embedding_backend: str
    llm_providers: list[str]
    arxiv_enabled: bool
