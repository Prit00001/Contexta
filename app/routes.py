from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import APIRouter, HTTPException, Request

from app.schemas import AskRequest, AskResponse, HealthResponse

router = APIRouter()


class RateLimiter:
    """Sliding-window limiter per client IP. Protects your free-tier API keys from abuse."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, ip: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic()
        q = self._hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False
        q.append(now)
        if len(self._hits) > 5000:  # bound memory
            self._hits = defaultdict(deque, {k: v for k, v in self._hits.items() if v})
        return True


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")  # Render / HF sit behind a proxy
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request):
    if not request.app.state.limiter.allow(_client_ip(request)):
        raise HTTPException(429, "Too many requests. Please wait a minute and try again.", headers={"Retry-After": "60"})
    return await request.app.state.service.ask(body.question, body.top_k, body.use_arxiv)


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    st = request.app.state
    return HealthResponse(
        status="ok",
        index_size=st.retriever.size,
        model=st.embedder.model_name,
        embedding_backend=st.embedder.backend,
        llm_providers=st.generator.provider_names,
        arxiv_enabled=st.settings.arxiv_enabled,
    )
