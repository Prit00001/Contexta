# Contexta

> Semantic search with grounded AI answers — FAISS · sentence-transformers · arXiv · OpenRouter

Contexta is a RAG (Retrieval-Augmented Generation) search engine. You ask a question in plain English, it semantically retrieves the most relevant documents from a local knowledge base and live arXiv papers, then passes them to an LLM to generate a grounded, cited answer.

---

## How it works

```
Question
   │
   ▼
Embed (all-MiniLM-L6-v2)
   │
   ├──▶ FAISS local index ──────────────┐
   │                                    │
   └──▶ arXiv live search → embed ──────┤
                                        │
                                   Merge + re-rank
                                        │
                                        ▼
                               OpenRouter (LLaMA-3)
                                        │
                                        ▼
                            Grounded answer + sources
```

---

## Stack

| Layer | Technology |
|---|---|
| API | FastAPI + uvicorn |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector store | FAISS `IndexFlatIP` (exact cosine search) |
| Live papers | arXiv API |
| LLM | OpenRouter → LLaMA-3-8B (free tier) |
| Frontend | Vanilla HTML/CSS/JS (served at `/`) |

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/Prit00001/Contexta.git
cd Contexta
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
```

Open `.env` and add your OpenRouter key:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

Get a free key at [openrouter.ai](https://openrouter.ai) — no credit card needed for free-tier models.

### 3. Run

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) — you should see the search UI.

---

## Project structure

```
Contexta/
├── app/
│   ├── main.py          # FastAPI app, lifespan, static serving
│   ├── routes.py        # POST /ask, GET /health
│   ├── schemas.py       # Pydantic request/response models
│   └── static/
│       └── index.html   # Frontend UI
├── core/
│   ├── embeddings.py    # sentence-transformers wrapper (lazy load + L2 norm)
│   ├── vector_store.py  # FAISS IndexFlatIP wrapper
│   ├── retriever.py     # query → embedding → FAISS → SourceDocuments
│   └── generator.py     # OpenRouter async LLM call + RAG prompt builder
├── services/
│   ├── search_service.py  # orchestrates the full pipeline
│   └── arxiv_client.py    # async arXiv fetch + XML parse
├── data/
│   └── documents.py       # 12 local ML/NLP documents (knowledge base)
├── .env.example
└── requirements.txt
```

---

## API

### `POST /ask`

```json
{
  "question": "How does retrieval-augmented generation work?",
  "top_k": 5,
  "use_arxiv": true
}
```

**Response:**

```json
{
  "question": "...",
  "answer": "...",
  "sources": [
    {
      "id": "doc_001",
      "title": "Transformer Architecture",
      "content": "...",
      "source": "local",
      "url": "https://arxiv.org/abs/1706.03762",
      "score": 0.8731
    }
  ],
  "total_sources": 5,
  "model_used": "meta-llama/llama-3-8b-instruct:free"
}
```

### `GET /health`

```json
{
  "status": "ok",
  "index_size": 12,
  "model": "all-MiniLM-L6-v2",
  "llm": "meta-llama/llama-3-8b-instruct:free"
}
```

Interactive docs available at [/docs](http://localhost:8000/docs).

---

## Extending the knowledge base

Add documents to `data/documents.py` — they are embedded and indexed automatically on startup:

```python
{
    "id":      "doc_013",
    "source":  "local",
    "title":   "Your Document Title",
    "url":     "https://source-url.com",
    "content": "The text content to embed and retrieve...",
},
```

---

## Deployment

### Render / Railway / Heroku

1. Add a `Procfile` to the repo root:

```
web: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

2. Set environment variables in your platform dashboard:

```
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=meta-llama/llama-3-8b-instruct:free
```

3. For free-tier platforms, use CPU-only PyTorch to reduce build size. In `requirements.txt`:

```
--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.3.0+cpu
```

> **Never commit your `.env` file.** It is gitignored by default. Always set secrets through your platform's environment variable dashboard.

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `OPENROUTER_API_KEY` | Yes | — | Your OpenRouter API key |
| `OPENROUTER_MODEL` | No | `meta-llama/llama-3-8b-instruct:free` | Any model from [openrouter.ai/models](https://openrouter.ai/models) |
| `APP_SITE_URL` | No | `http://localhost:8000` | Shown in OpenRouter dashboard |
| `APP_SITE_NAME` | No | `shhhhh-semantic-search` | Shown in OpenRouter dashboard |

---

## License

MIT
