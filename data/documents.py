"""
data/documents.py — Local knowledge base (12 ML/NLP documents).
Add more dicts here — indexed automatically on startup.
Each dict: id, title, content, source, url (optional).
"""

SAMPLE_DOCUMENTS = [
    {
        "id": "doc_001", "source": "local",
        "title": "Transformer Architecture",
        "url": "https://arxiv.org/abs/1706.03762",
        "content": (
            "The Transformer (Vaswani et al., 2017) replaces recurrence with self-attention. "
            "Encoder-decoder stacks use multi-head attention, feed-forward layers, residual "
            "connections, and layer normalisation. Positional encodings inject sequence order. "
            "Transformers train in parallel, enabling massive scale. Foundation for BERT, GPT, T5."
        ),
    },
    {
        "id": "doc_002", "source": "local",
        "title": "BERT: Bidirectional Encoder Representations",
        "url": "https://arxiv.org/abs/1810.04805",
        "content": (
            "BERT (Devlin et al., 2018) pre-trains deep bidirectional Transformers using "
            "Masked Language Modelling (MLM) and Next Sentence Prediction (NSP). Fine-tuning "
            "on QA, NLI, and NER achieves SOTA with minimal task-specific changes."
        ),
    },
    {
        "id": "doc_003", "source": "local",
        "title": "Retrieval-Augmented Generation (RAG)",
        "url": "https://arxiv.org/abs/2005.11401",
        "content": (
            "RAG (Lewis et al., 2020) combines a seq2seq model with a dense vector index. "
            "For each input, it retrieves top-k passages via MIPS, then conditions generation "
            "on query + retrieved text. Reduces hallucinations by grounding answers in evidence."
        ),
    },
    {
        "id": "doc_004", "source": "local",
        "title": "FAISS: Efficient Similarity Search",
        "url": "https://github.com/facebookresearch/faiss",
        "content": (
            "FAISS (Facebook AI Similarity Search) provides optimised ANN algorithms for "
            "high-dimensional vectors. Supports exact (IndexFlatL2/IP) and approximate "
            "(IVF, HNSW, PQ) search. GPU versions achieve 10–100× speed-up. Used in "
            "production RAG, recommendation, and image search systems."
        ),
    },
    {
        "id": "doc_005", "source": "local",
        "title": "Sentence-BERT: Semantic Sentence Embeddings",
        "url": "https://arxiv.org/abs/1908.10084",
        "content": (
            "Sentence-BERT (Reimers & Gurevych, 2019) fine-tunes BERT with siamese networks "
            "to produce sentence embeddings where cosine similarity reflects semantic similarity. "
            "all-MiniLM-L6-v2 outputs 384-dim embeddings in <10 ms — ideal for real-time search."
        ),
    },
    {
        "id": "doc_006", "source": "local",
        "title": "GPT and Large Language Models",
        "url": "https://arxiv.org/abs/2005.14165",
        "content": (
            "GPT uses decoder-only Transformers trained on next-token prediction. GPT-3 (Brown "
            "et al., 2020) demonstrated few-shot learning: task performance from prompt examples "
            "alone without gradient updates. Scaling laws show predictable improvement with "
            "model size, data, and compute."
        ),
    },
    {
        "id": "doc_007", "source": "local",
        "title": "Vector Databases and Semantic Search",
        "url": "https://www.pinecone.io/learn/vector-database/",
        "content": (
            "Vector databases (Pinecone, Weaviate, Qdrant, Milvus) store embeddings and "
            "support ANN queries at scale. Unlike BM25/TF-IDF, semantic search captures meaning: "
            "'car' and 'automobile' are neighbours. Hybrid search (dense + sparse) maximises "
            "recall. Production systems add metadata filters, re-ranking, and caching."
        ),
    },
    {
        "id": "doc_008", "source": "local",
        "title": "Self-Attention and Multi-Head Attention",
        "url": "https://arxiv.org/abs/1706.03762",
        "content": (
            "Self-attention computes a weighted sum of values, with weights from query-key "
            "dot products. Multi-head attention runs h parallel attention functions and "
            "concatenates outputs — attending to different representation subspaces. "
            "Cross-attention in the decoder attends to encoder outputs for seq2seq generation."
        ),
    },
    {
        "id": "doc_009", "source": "local",
        "title": "Fine-Tuning and Parameter-Efficient Methods",
        "url": "https://arxiv.org/abs/2106.09685",
        "content": (
            "Transfer learning: pre-train on unlabelled text, fine-tune on labelled tasks. "
            "LoRA, Adapters, and Prefix Tuning update <1% of parameters, matching full "
            "fine-tuning on most benchmarks while fitting on consumer hardware."
        ),
    },
    {
        "id": "doc_010", "source": "local",
        "title": "Cosine Similarity and Distance Metrics",
        "url": "https://en.wikipedia.org/wiki/Cosine_similarity",
        "content": (
            "Cosine similarity = (A·B)/(|A||B|), range [-1, 1]. For L2-normalised vectors "
            "it equals the dot product — what FAISS IndexFlatIP computes. Magnitude-invariant, "
            "making it the standard metric for text embeddings and semantic search."
        ),
    },
    {
        "id": "doc_011", "source": "local",
        "title": "OpenRouter: Unified LLM Gateway",
        "url": "https://openrouter.ai",
        "content": (
            "OpenRouter provides a single OpenAI-compatible endpoint for 200+ LLMs including "
            "LLaMA 3, Mistral, Gemma, Claude, and GPT-4. Free-tier models (marked :free) "
            "require no credit card. The API follows the /v1/chat/completions schema. "
            "Ideal for cost-optimised RAG pipelines — swap models without code changes."
        ),
    },
    {
        "id": "doc_012", "source": "local",
        "title": "Prompt Engineering for LLMs",
        "url": "https://www.promptingguide.ai/",
        "content": (
            "Prompt engineering elicits desired LLM behaviour without weight updates. "
            "Techniques: zero-shot, few-shot, chain-of-thought, retrieval-augmented prompting. "
            "System prompts, temperature (0=deterministic, 1=creative), and max_tokens control "
            "output style. RAG prompts inject retrieved context before the question."
        ),
    },
]
