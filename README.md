<div align="center">

# Contexta 🔍

**Semantic search with grounded AI answers**

`FAISS` · `Sentence Transformers` · `arXiv` · `FastAPI` · `OpenRouter`

> ⚠️ **Deployment Notice** — See the [Known Limitations](#known-limitations) section before trying the live demo.

</div>

---

## What is Contexta?

Contexta is a **RAG (Retrieval-Augmented Generation)** search engine built for research. Ask a question in plain English — it semantically retrieves the most relevant documents from a local knowledge base and live arXiv papers, then passes them to an LLM to generate a grounded, cited answer.

No hallucinations. Every answer is backed by real sources.

---

## How It Works

```
Your Question
      │
      ▼
  Embed (all-MiniLM-L6-v2)
      │
      ├──▶ FAISS local index ──────────┐
      │                                │
      └──▶ arXiv live search ──────────┤
                                       │
                                  Merge + Re-rank
                                       │
                                       ▼
                              OpenRouter → LLaMA-3
                                       │
                                       ▼
                          Grounded answer + cited sources
```

---

## Tech Stack

| Layer        | Technology                                   |
|--------------|----------------------------------------------|
| API          | FastAPI + uvicorn                            |
| Embeddings   | `all-MiniLM-L6-v2` (sentence-transformers)   |
| Vector Store | FAISS `IndexFlatIP` (exact cosine search)    |
| Live Papers  | arXiv API                                    |
| LLM          | OpenRouter → LLaMA-3-8B (free tier)          |
| Frontend     | Vanilla HTML/CSS/JS                          |

---

## ⚠️ Known Limitations

### Deployment Status

**Contexta is fully functional locally. The live deployed version has a known issue.**

The app is deployed on Render, but the **AI answer generation is currently broken in the deployed environment** due to OpenRouter's free-tier API restrictions:

- Free-tier models on OpenRouter often get **rate-limited or blocked** when requests come from cloud server IPs (Render, Railway, Heroku, etc.)
- This means the retrieval pipeline works fine, but the final LLM answer generation fails in production
- Switching to a **paid OpenRouter model** would fix this immediately

### Why Not Ollama?

An alternative was considered — running a local model via **Ollama** instead of the OpenRouter API. This works perfectly on a local machine, but has an obvious catch:

> If the laptop running Ollama is turned off, the entire deployed app stops working.

That's not a real deployment. So Ollama is not used in the hosted version.

### What Works Right Now

| Feature | Local | Deployed |
|---------|-------|----------|
| Semantic search (FAISS) | ✅ | ✅ |
| arXiv live paper fetch | ✅ | ✅ |
| AI answer generation | ✅ | ❌ (API limit) |
| Source citations | ✅ | ✅ |

---

## Run It Locally (Fully Works)

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

Add your OpenRouter key to `.env`:

```
OPENROUTER_API_KEY=sk-or-v1-...
```

Get a free key at [openrouter.ai](https://openrouter.ai) — no credit card needed.

### 3. Start the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) — the search UI will be running.

---

## API Reference

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

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

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
│   └── generator.py     # OpenRouter LLM call + RAG prompt
├── services/
│   ├── search_service.py  # full pipeline orchestration
│   └── arxiv_client.py    # async arXiv fetch + XML parse
├── data/
│   └── documents.py       # local knowledge base (12 ML/NLP docs)
├── .env.example
└── requirements.txt
```

---

## Extending the Knowledge Base

Add entries to `data/documents.py` — they are embedded and indexed automatically on startup:

```python
{
    "id":      "doc_013",
    "source":  "local",
    "title":   "Your Document Title",
    "url":     "https://source-url.com",
    "content": "Text content to embed and retrieve...",
}
```

---

## Environment Variables

| Variable             | Required | Default                               | Description                          |
|----------------------|----------|---------------------------------------|--------------------------------------|
| `OPENROUTER_API_KEY` | Yes      | —                                     | Your OpenRouter API key              |
| `OPENROUTER_MODEL`   | No       | `meta-llama/llama-3-8b-instruct:free` | Any model on openrouter.ai/models    |
| `APP_SITE_URL`       | No       | `http://localhost:8000`               | Shown in OpenRouter dashboard        |
| `APP_SITE_NAME`      | No       | `contexta`                            | Shown in OpenRouter dashboard        |

---

## Roadmap

- [ ] Fix production deployment — switch to a paid OpenRouter model or find a free-tier alternative that works from server IPs
- [ ] Add support for user-uploaded PDFs to the knowledge base
- [ ] Persistent vector store (save/load FAISS index)
- [ ] Chat history / multi-turn conversations

---

## License

MIT — built by [Pratyush Pandey](https://github.com/Prit00001)
