from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.schemas import AskRequest, AskResponse, HealthResponse
from core.pdf import chunk, extract_pages

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
    fwd = request.headers.get(
        "x-forwarded-for"
    )  # Vercel / Render / HF sit behind a proxy
    return (
        fwd.split(",")[0].strip()
        if fwd
        else (request.client.host if request.client else "unknown")
    )


def _check_rate_limit(request: Request) -> None:
    if not request.app.state.limiter.allow(_client_ip(request)):
        raise HTTPException(
            429,
            "Too many requests. Please wait a minute and try again.",
            headers={"Retry-After": "60"},
        )


@router.post("/ask", response_model=AskResponse)
async def ask(body: AskRequest, request: Request):
    _check_rate_limit(request)
    return await request.app.state.service.ask(
        body.question, body.top_k, body.use_arxiv
    )


@router.post("/ask-pdf", response_model=AskResponse)
async def ask_pdf(
    request: Request,
    question: str = Form(..., min_length=3, max_length=1000),
    file: UploadFile = File(...),
    top_k: int = Form(5, ge=1, le=10),
):
    _check_rate_limit(request)

    max_bytes = request.app.state.settings.max_pdf_bytes
    data = await file.read(max_bytes + 1)  # never buffer more than the limit

    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > max_bytes:
        raise HTTPException(
            413, f"PDF too large. Max size is {max_bytes // (1024 * 1024)} MB."
        )
    if not data.startswith(b"%PDF"):
        raise HTTPException(415, "That file isn't a PDF.")

    # pypdf parsing is CPU-bound and synchronous; keep it off the event loop.
    try:
        pages = await run_in_threadpool(extract_pages, data)
    except Exception:  # noqa: BLE001 - pypdf raises many types on bad input
        raise HTTPException(
            422, "Couldn't read this PDF. It may be password-protected or corrupted."
        )

    chunks = await run_in_threadpool(chunk, pages)
    if not chunks:
        raise HTTPException(
            422,
            "No extractable text found. Scanned PDFs need OCR, which isn't supported yet.",
        )

    return await request.app.state.service.ask_pdf(
        question=question,
        chunks=chunks,
        top_k=top_k,
        filename=file.filename or "uploaded.pdf",
    )


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
