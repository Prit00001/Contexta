# Contexta

Ask a question about ML/NLP → get a cited answer grounded in a local paper library **plus live arXiv search**.

FastAPI · FAISS · fastembed (all-MiniLM-L6-v2) · arXiv API · Groq / Gemini / OpenRouter (free tiers)

## What changed in v2 (why the deployed demo now stays up)

| Problem before | Fix |
|---|---|
| One free LLM → rate limits kill the app | **Provider chain** (Groq → Gemini → OpenRouter → custom). A failing provider gets a cooldown and the next one is tried |
| All providers down = broken app | **Extractive fallback**: answer assembled from the top passages with citations. The app never errors because of the LLM |
| torch + sentence-transformers ≈ 1 GB RAM | **fastembed (ONNX)**, same model, ~150 MB. Fits Render's 512 MB free tier |
| Re-embedded everything on each start | Embeddings **cached on disk** by content hash |
| Repeated questions burn API quota | Answer **cache** (1h) + arXiv cache + per-IP **rate limit** |
| arXiv could hang a request | 9s time budget, polite 3s spacing, falls back to local-only with a notice |
| Cold start downloads the model | Model **baked into the Docker image** |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add GROQ_API_KEY (free, no card: console.groq.com/keys)
uvicorn app.main:app --reload
```
Open http://localhost:8000. No key? It still works in source-based mode.

## Deploy on Render (free)

1. Push this folder to GitHub.
2. Render → **New → Blueprint** → pick the repo (it reads `render.yaml`).
3. Add `GROQ_API_KEY` (and optionally `GEMINI_API_KEY`, `OPENROUTER_API_KEY`) in the dashboard.
4. Deploy. Health check is `/health`.

Free web services sleep after ~15 min idle. A free uptime pinger (UptimeRobot, cron-job.org) hitting `/health` every 10 min keeps it warm.
Hugging Face Spaces (Docker SDK) also works: add `app_port: 8000` to the Space README front matter.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` / `GEMINI_API_KEY` / `OPENROUTER_API_KEY` | – | Set one or more |
| `GROQ_MODEL` / `GEMINI_MODEL` / `OPENROUTER_MODEL` | `llama-3.1-8b-instant` / `gemini-2.5-flash` / `meta-llama/llama-3.3-70b-instruct:free` | Model IDs change; override here |
| `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` | – | Any OpenAI-compatible server (hosted Ollama, Cerebras…) |
| `LLM_PROVIDERS` | `groq,gemini,openrouter,custom` | Order of the fallback chain |
| `RATE_LIMIT_PER_MIN` | `20` | Per IP; `0` disables |
| `ARXIV_ENABLED` | `true` | |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |

## API

`POST /ask` `{ "question": "...", "top_k": 5, "use_arxiv": true }` → `answer`, `sources[]`, `provider`, `model_used`, `llm_used`, `cached`, `warnings[]`, `timings_ms`
`GET /health` → index size, embedding backend, configured providers. Interactive docs at `/docs`.

## Layout

```
app/        main.py (lifespan, static) · routes.py (/ask, /health, rate limiter) · config.py · schemas.py · static/index.html
core/       embeddings.py · vector_store.py (FAISS) · retriever.py (disk-cached index) · generator.py (provider chain) · text.py
services/   search_service.py (pipeline + re-rank) · arxiv_client.py
data/       documents.py  ← add your own entries here
tests/      pytest suite (no network, no model download needed)
```

## Tests

```bash
pip install -r requirements-dev.txt && pytest
```

## Roadmap
- PDF upload · cross-encoder re-ranking (needs a bigger instance) · streaming answers
