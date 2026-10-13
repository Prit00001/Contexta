"""Central configuration, read once from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


def _num(name: str, default, cast=int):
    try:
        return cast(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    base_url: str
    model: str
    api_key: str = ""
    headers: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Settings:
    embed_backend: str
    embed_model: str
    cache_dir: Path
    model_cache_dir: str | None
    providers: tuple[ProviderSpec, ...]
    llm_timeout: float
    arxiv_enabled: bool
    arxiv_max_results: int
    arxiv_budget_s: float
    arxiv_cache_ttl: int
    answer_cache_ttl: int
    rate_limit_per_min: int
    min_score: float
    allowed_origins: tuple[str, ...]
    site_url: str


def _build_providers() -> tuple[ProviderSpec, ...]:
    site = os.getenv("APP_SITE_URL", "http://localhost:8000")
    catalog = {
        "groq": ProviderSpec(
            "groq",
            "https://api.groq.com/openai/v1",
            os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            os.getenv("GROQ_API_KEY", ""),
        ),
        "gemini": ProviderSpec(
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta/openai",
            os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            os.getenv("GEMINI_API_KEY", ""),
        ),
        "openrouter": ProviderSpec(
            "openrouter",
            "https://openrouter.ai/api/v1",
            os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
            os.getenv("OPENROUTER_API_KEY", ""),
            {"HTTP-Referer": site, "X-Title": "Contexta"},
        ),
        "custom": ProviderSpec(
            "custom",
            os.getenv("LLM_BASE_URL", "").rstrip("/"),
            os.getenv("LLM_MODEL", ""),
            os.getenv("LLM_API_KEY", ""),
        ),
    }
    order = [p.strip() for p in os.getenv("LLM_PROVIDERS", "groq,gemini,openrouter,custom").split(",")]
    out = []
    for name in order:
        spec = catalog.get(name)
        if not spec:
            continue
        # custom endpoints (e.g. Ollama) may not need a key, but need URL + model
        ready = bool(spec.base_url and spec.model) if name == "custom" else bool(spec.api_key)
        if ready:
            out.append(spec)
    return tuple(out)


@lru_cache
def get_settings() -> Settings:
    return Settings(
        embed_backend=os.getenv("EMBED_BACKEND", "fastembed"),
        embed_model=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        cache_dir=Path(os.getenv("CACHE_DIR", ".cache")),
        model_cache_dir=os.getenv("FASTEMBED_CACHE_PATH"),
        providers=_build_providers(),
        llm_timeout=_num("LLM_TIMEOUT", 25.0, float),
        arxiv_enabled=_bool("ARXIV_ENABLED", True),
        arxiv_max_results=_num("ARXIV_MAX_RESULTS", 12),
        arxiv_budget_s=_num("ARXIV_BUDGET_S", 9.0, float),
        arxiv_cache_ttl=_num("ARXIV_CACHE_TTL", 3600),
        answer_cache_ttl=_num("ANSWER_CACHE_TTL", 3600),
        rate_limit_per_min=_num("RATE_LIMIT_PER_MIN", 20),
        min_score=_num("MIN_SCORE", 0.2, float),
        allowed_origins=tuple(o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()),
        site_url=os.getenv("APP_SITE_URL", "http://localhost:8000"),
    )
