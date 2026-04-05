"""
app/routes.py
POST /ask   → semantic search + OpenRouter answer
GET  /health → liveness + model info
"""
from __future__ import annotations
import logging
from fastapi import APIRouter, HTTPException
from app.schemas import AskRequest, AskResponse, HealthResponse
from core.embeddings import EMBEDDING_MODEL
from core.generator import OPENROUTER_MODEL
from services.search_service import search_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ask", response_model=AskResponse, summary="Semantic search + AI answer")
async def ask_question(request: AskRequest) -> AskResponse:
    logger.info("POST /ask  question=%r", request.question[:80])
    try:
        return await search_service.ask(
            question  = request.question,
            top_k     = request.top_k,
            use_arxiv = request.use_arxiv,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unhandled error in /ask")
        raise HTTPException(status_code=500, detail="Internal server error.") from exc


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    return HealthResponse(
        status     = "ok",
        index_size = search_service.index_size,
        model      = EMBEDDING_MODEL,
        llm        = OPENROUTER_MODEL,
    )
