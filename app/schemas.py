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


# ---- Workspace document Q&A (Phase 1c) ----


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str


class DocumentOut(BaseModel):
    id: str
    filename: str
    status: str
    error: Optional[str] = None
    created_at: str


class Citation(BaseModel):
    document_id: str
    filename: str
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    snippet: str
    score: float


class WorkspaceAskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    top_k: int = Field(5, ge=1, le=10)


class WorkspaceAskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Citation]
    provider: str
    model_used: str
    llm_used: bool
    warnings: list[str] = []


# ---- Workspace bootstrap, membership, API keys (Phase 1d) ----


class CreateWorkspaceRequest(BaseModel):
    org_name: str = Field(..., min_length=1, max_length=200)
    workspace_name: str = Field(..., min_length=1, max_length=200)


class CreateWorkspaceResponse(BaseModel):
    org_id: str
    workspace_id: str


class AddMemberRequest(BaseModel):
    email: str
    role: str = Field("member", pattern="^(admin|member|viewer)$")


class MemberOut(BaseModel):
    user_id: str
    role: str
    created_at: str


class ApiKeyCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)


class ApiKeyCreateResponse(BaseModel):
    id: str
    name: str
    api_key: str  # shown once; not retrievable again


class ApiKeyOut(BaseModel):
    id: str
    name: str
    created_at: str
    last_used_at: Optional[str] = None
    revoked_at: Optional[str] = None
