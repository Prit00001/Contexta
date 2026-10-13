from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.routes import RateLimiter, router
from core.embeddings import Embedder
from core.generator import Generator
from core.retriever import Retriever
from data.documents import DOCUMENTS
from services.arxiv_client import ArxivClient
from services.search_service import SearchService

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    s = get_settings()
    embedder = Embedder(s.embed_backend, s.embed_model, s.model_cache_dir)
    retriever = Retriever(embedder, s.cache_dir / "embeddings.npz")
    retriever.build(DOCUMENTS)
    client = httpx.AsyncClient(
        headers={"User-Agent": "Contexta/2.0 (+https://github.com/Prit00001/Contexta)"}, follow_redirects=True
    )
    generator = Generator(s.providers, client, s.llm_timeout)
    arxiv = ArxivClient(client, s.arxiv_max_results, s.arxiv_cache_ttl)
    app.state.settings = s
    app.state.embedder, app.state.retriever, app.state.generator = embedder, retriever, generator
    app.state.service = SearchService(s, embedder, retriever, arxiv, generator)
    app.state.limiter = RateLimiter(s.rate_limit_per_min)
    logging.getLogger("contexta").info(
        "Ready: %d docs, LLM providers=%s", retriever.size, generator.provider_names or "none (extractive mode)"
    )
    yield
    await client.aclose()


def create_app() -> FastAPI:
    app = FastAPI(title="Contexta", version="2.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=list(get_settings().allowed_origins), allow_methods=["GET", "POST"], allow_headers=["*"]
    )
    app.include_router(router)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()
