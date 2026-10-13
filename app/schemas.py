from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(5, ge=1, le=10)
    use_arxiv: bool = True


class Source(BaseModel):
    id: str
    title: str
    source: str  # "local" | "arxiv"
    url: str
    score: float
    snippet: str = ""
    authors: list[str] = []
    published: str = ""


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    total_sources: int
    model_used: str
    provider: str
    llm_used: bool
    cached: bool
    warnings: list[str] = []
    timings_ms: dict[str, int] = {}


class HealthResponse(BaseModel):
    status: str
    index_size: int
    model: str
    embedding_backend: str
    llm_providers: list[str]
    arxiv_enabled: bool
