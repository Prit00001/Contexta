"""
app/schemas.py
Pydantic models shared across the entire application.
Adding a new field here automatically propagates to docs, validation, and serialisation.
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


# ── Inbound ───────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str  = Field(..., min_length=3, max_length=1000,
                           description="Natural-language question from the user")
    top_k:    int  = Field(5,  ge=1, le=20,
                           description="Number of documents to retrieve from FAISS")
    use_arxiv: bool = Field(True,
                            description="Augment local results with live arXiv papers")


# ── Outbound ──────────────────────────────────────────────────────────

class SourceDocument(BaseModel):
    id:      str
    title:   str
    content: str
    source:  str            # "local" | "arxiv"
    url:     Optional[str] = None
    score:   float


class AskResponse(BaseModel):
    question:      str
    answer:        str
    sources:       List[SourceDocument]
    total_sources: int
    model_used:    str      # which LLM produced the answer


class HealthResponse(BaseModel):
    status:     str
    index_size: int
    model:      str         # embedding model
    llm:        str         # LLM / OpenRouter model
