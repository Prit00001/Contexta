"""
app/main.py — FastAPI entry point.

Run:
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()   # must be before any env-reading imports

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.routes import router
from services.search_service import search_service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== shhhhh- starting up ===")
    search_service.load_local_documents()
    yield
    logger.info("=== shhhhh- shutting down ===")


app = FastAPI(
    title       = "shhhhh- Semantic Search",
    description = "FAISS · sentence-transformers · arXiv · OpenRouter (LLaMA 3)",
    version     = "2.0.0",
    lifespan    = lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.include_router(router)


@app.get("/", include_in_schema=False)
async def serve_ui():
    f = STATIC_DIR / "index.html"
    if not f.exists():
        return JSONResponse({"message": "UI not found — use /docs"})
    return FileResponse(str(f), media_type="text/html")
