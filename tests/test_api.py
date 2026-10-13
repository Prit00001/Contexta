from fastapi.testclient import TestClient

from app.main import create_app


def client():
    return TestClient(create_app())


def test_health_and_index():
    with client() as c:
        h = c.get("/health").json()
        assert h["status"] == "ok" and h["index_size"] >= 20
        assert c.get("/").status_code == 200


def test_ask_extractive_when_no_llm():
    with client() as c:
        r = c.post("/ask", json={"question": "How does retrieval-augmented generation work?", "top_k": 3, "use_arxiv": False})
        d = r.json()
        assert r.status_code == 200
        assert d["llm_used"] is False and d["model_used"] == "extractive"
        assert d["total_sources"] == 3 and "[1]" in d["answer"]
        assert any("No AI model" in w for w in d["warnings"])
        assert "RAG" in d["sources"][0]["title"] or "Retrieval" in d["sources"][0]["title"]


def test_validation_and_rate_limit(monkeypatch):
    with client() as c:
        assert c.post("/ask", json={"question": "hi"}).status_code == 422
        c.app.state.limiter.per_minute = 1
        c.post("/ask", json={"question": "what is faiss", "use_arxiv": False})
        assert c.post("/ask", json={"question": "what is faiss", "use_arxiv": False}).status_code == 429
