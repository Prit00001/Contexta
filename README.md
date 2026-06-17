<div align="center">

# 🔍 Contexta

**Semantic search engine with grounded AI answers**

`FAISS` · `sentence-transformers` · `arXiv` · `FastAPI` · `OpenRouter`

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-blue?style=flat)](https://github.com/facebookresearch/faiss)

</div>

---

## ⚠️ Deployment Status

> **Works perfectly on local machine. Cloud deployment has a known limitation.**

Contexta is **fully functional when run locally**. The cloud-deployed version (Render) hits a wall with the **free-tier AI API** — OpenRouter's free models (`llama-3-8b-instruct:free`) are rate-limited and often return errors under sustained usage on shared hosting.

**The tradeoff I hit:**

| Option | Problem |
|--------|---------|
| OpenRouter free tier | Rate limits break the deployed version |
| Paid API key | Costs money — not feasible for a free demo |
| Ollama (local LLM) | Works great locally, but if my laptop is off, the server dies with it |

**Bottom line:** The search and retrieval pipeline works end-to-end. The bottleneck is purely the AI generation step on free-tier cloud infrastructure. If you run it locally with your own OpenRouter key, it works flawlessly.

---

## What it does

You ask a question in plain English. Contexta:

1. Embeds your query using `all-MiniLM-L6-v2`
2. Searches a local FAISS index of ML/NLP documents
3. Fetches live relevant papers from arXiv
4. Merges and re-ranks results
5. Sends top sources to an LLM to generate a grounded, cited answer

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
                              OpenRouter (LLaMA-3-8B)
                                        │
                                        ▼
                           Grounded answer + sources
```

---

## Screenshots

**Search Interface**

![Contexta Search UI](assets/screenshot1.png)

**Results with Sources**

![Contexta Results](assets/screenshot2.png)

---

## Stack

| Layer | Technology |
|-------|------------|
| API | FastAPI + uvicorn |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector store | FAISS `IndexFlatIP` (exact cosine search) |
| Live papers | arXiv API |
| LLM | OpenRouter → LLaMA-3-8B (free tier) |
| Frontend | Vanilla HTML/CSS/JS |

---

## Run Locally

### 1. Clone & install

```bash
git clone https://github.com/Prit00001/Contexta.git
cd Contexta
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up your API key

```bash
cp .env.example .env
```

Open `.env` and add your key:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

Get a free key at [openrouter.ai](https://openrouter.ai) — no credit card needed.

### 3. Start the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) — you should see the search UI.

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

Interactive docs at [/docs](http://localhost:8000/docs).

---

## Project Structure

```
Contexta/
├── app/
│   ├── main.py          # FastAPI app, lifespan, static serving
│   ├── routes.py        # POST /ask, GET /health
│   ├── schemas.py       # Pydantic request/response models
│   └── static/
│       └── index.html   # Frontend UI
├── core/
│   ├── embeddings.py    # sentence-transformers wrapper
│   ├── vector_store.py  # FAISS IndexFlatIP wrapper
│   ├── retriever.py     # query → embedding → FAISS → sources
│   └── generator.py     # OpenRouter async LLM call + RAG prompt
├── services/
│   ├── search_service.py  # full pipeline orchestration
│   └── arxiv_client.py    # async arXiv fetch + XML parse
├── data/
│   └── documents.py       # local ML/NLP knowledge base (12 docs)
└── requirements.txt
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | ✅ | — | Your OpenRouter API key |
| `OPENROUTER_MODEL` | ❌ | `meta-llama/llama-3-8b-instruct:free` | Any model from openrouter.ai/models |
| `APP_SITE_URL` | ❌ | `http://localhost:8000` | Shown in OpenRouter dashboard |

---

## Extending the Knowledge Base

Add entries to `data/documents.py` — they get embedded and indexed automatically on startup:

```python
{
    "id":      "doc_013",
    "source":  "local",
    "title":   "Your Document Title",
    "url":     "https://source-url.com",
    "content": "The text to embed and retrieve...",
},
```

---

## Known Issues / Roadmap

- [ ] **Free API rate limits** — looking into a paid tier or self-hosted LLM solution that doesn't depend on my laptop being on
- [ ] Add support for uploading custom PDF documents
- [ ] Persistent vector store (currently re-indexes on every startup)
- [ ] Better re-ranking (cross-encoder)

---

## License

MIT — built by [Pratyush Pandey](https://github.com/Prit00001)
