import os

# Must be set before the app is imported: tiny hash embeddings (no model download), no API keys.
os.environ.update(
    EMBED_BACKEND="hash", CACHE_DIR="/tmp/contexta_test_cache", ARXIV_ENABLED="false",
    MIN_SCORE="0.0", RATE_LIMIT_PER_MIN="1000",
)
for k in ("GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "LLM_BASE_URL"):
    os.environ.pop(k, None)
