# ContextForge

> A production-grade RAG platform that doesn't just answer questions — it evaluates how trustworthy its own answers are.

[![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-teal?style=flat-square)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.1-purple?style=flat-square)](https://langchain-ai.github.io/langgraph)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.17-orange?style=flat-square)](https://qdrant.tech)
[![License](https://img.shields.io/badge/License-MIT-gray?style=flat-square)](LICENSE)

---

## What makes this different

Most RAG systems answer questions and stop there. ContextForge goes a step further: every response comes with an **evaluation score** — a measure of how well the answer is grounded in the source documents. This is what separates a demo from a production system.

Built as a deliberate exercise in production-grade AI engineering: every architectural decision is intentional, every library is chosen over its alternatives, and every abstraction earns its place.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FastAPI + Uvicorn                     │
│                   (Async REST + SSE endpoints)               │
└────────────────────────┬────────────────────────────────────┘
                         │
          ┌──────────────▼───────────────┐
          │      LangGraph Supervisor     │
          │   StateGraph orchestration    │
          └────────┬──────────┬──────────┘
                   │          │
          ┌────────▼───┐  ┌───▼────────┐
          │  RAG Agent  │  │ Eval Agent │
          │             │  │            │
          │ • retrieve  │  │ • LLM judge│
          │ • generate  │  │ • scoring  │
          └────────┬────┘  └────────────┘
                   │
     ┌─────────────▼──────────────┐
     │          Qdrant             │
     │  Dense vectors (OpenAI)     │
     │  Sparse vectors (BM25)      │
     │  Hybrid search + RRF fusion │
     └─────────────────────────────┘
```

---

## Key engineering decisions

### Hybrid search over pure semantic search

Semantic search alone misses exact-match queries — proper nouns, model names, version numbers. Pure keyword search misses paraphrased intent. ContextForge uses **both in parallel**:

- **Dense vectors** (OpenAI `text-embedding-3-small`) for semantic similarity
- **Sparse vectors** (BM25 via FastEmbed) for keyword precision
- **Reciprocal Rank Fusion** to merge results without scale dependency

### LangGraph for multi-agent orchestration

The RAG pipeline and evaluation pipeline run as separate agents in a `StateGraph`. This isn't over-engineering — it's the foundation for extending to query decomposition, self-correction loops, and human-in-the-loop approval in future iterations.

### Deterministic chunk IDs

Every chunk is identified by `sha256(filename + chunk_index)` — not a random UUID. Upload the same document twice and you get an idempotent upsert, not duplicate vectors. A small detail that makes the system production-safe.

### Singleton resource management via lifespan

Both the Qdrant client and OpenAI client are created once at startup and injected via `app.state`. No connection pool rebuilt per request. Follows FastAPI's recommended lifespan pattern over deprecated `@app.on_event`.

---

## Tech stack

| Layer              | Technology                      | Why this, not X                                                                 |
| ------------------ | ------------------------------- | ------------------------------------------------------------------------------- |
| Package management | `uv`                            | 10-100× faster than pip; PEP 621 compliant; lockfile guarantees reproducibility |
| Web framework      | FastAPI 0.135                   | Native async; auto OpenAPI docs; Pydantic v2 integration                        |
| Vector database    | Qdrant 1.17                     | Native hybrid search; self-hostable; Docker-ready                               |
| Embedding          | OpenAI `text-embedding-3-small` | Best cost/quality ratio for dense retrieval                                     |
| Sparse embedding   | FastEmbed BM25                  | CPU-native; no API call; ONNX runtime                                           |
| Orchestration      | LangGraph 1.1                   | Stateful agent workflows; native async; production-grade                        |
| PDF parsing        | pypdf 6.9                       | Lightweight; no Java dependency (vs Apache Tika)                                |
| Chunking           | langchain-text-splitters        | Recursive splitting respects paragraph boundaries                               |
| Monorepo           | `uv` workspaces                 | Single lockfile across `backend` and future `mcp-server` packages               |

---

## Project structure

```
contextforge/
├── packages/
│   └── backend/
│       └── src/contextforge/
│           ├── api/
│           │   ├── ingest.py      # Document upload endpoint
│           │   └── query.py       # Question answering endpoint
│           ├── core/
│           │   └── config.py      # Pydantic Settings — single source of truth
│           ├── ingestion/
│           │   ├── parser.py      # PDF → structured text
│           │   ├── chunker.py     # Text → overlapping chunks
│           │   ├── embedder.py    # Chunks → vectors → Qdrant
│           │   └── pipeline.py    # Orchestrates ingestion steps
│           ├── rag/
│           │   ├── retriever.py   # Hybrid search
│           │   ├── generator.py   # LLM answer generation
│           │   └── graph.py       # LangGraph multi-agent pipeline
│           └── main.py            # FastAPI app + lifespan
├── frontend/                      # Next.js 16 (Faz 7)
├── pyproject.toml                 # uv workspace root
├── uv.lock
└── .env                           # Never committed
```

---

## Getting started

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) — `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`
- [Docker](https://www.docker.com/) — for Qdrant
- OpenAI API key

### 1. Clone and install

```bash
git clone https://github.com/tahaasann/ContextForge.git
cd contextforge
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

```env
OPENAI_API_KEY=sk-proj-...
APP_ENV=development
```

### 3. Start Qdrant

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

### 4. Run the backend

```bash
uvicorn contextforge.main:app --reload --port 8001
```

API docs available at `http://localhost:8001/docs`

---

## API reference

### `POST /ingest/upload`

Upload a PDF document to the knowledge base.

```bash
curl -X POST http://localhost:8001/ingest/upload \
  -F "file=@your-document.pdf;type=application/pdf"
```

```json
{
  "status": "success",
  "filename": "your-document.pdf",
  "total_pages": 42,
  "total_chunks": 156
}
```

### `POST /query/ask`

Ask a question against all ingested documents.

```bash
curl -X POST http://localhost:8001/query/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main argument of the paper?", "top_k": 5}'
```

```json
{
  "answer": "The paper argues that... [Source: document.pdf, page 12]",
  "sources": [
    {"filename": "document.pdf", "page": 12, "score": 0.8821}
  ],
  "eval_score": 0.92,
  "eval_reasoning": "Answer is directly grounded in retrieved context with proper citation."
}
```

### `GET /health`

```json
{"status": "ok", "env": "development", "model": "gpt-4o"}
```

---

## Evaluation

Every response includes an `eval_score` (0.0–1.0) computed by a second LLM agent using the **LLM-as-a-judge** pattern. The score reflects:

- **Groundedness** — Is the answer traceable to the retrieved chunks? (no hallucination)
- **Relevance** — Does it directly address the question?
- **Citation** — Are source documents referenced?

This is not a vanity metric. It's the foundation for automated quality gates in CI/CD pipelines and user-facing confidence indicators.

---

## Roadmap

- [x] Faz 1 — Ingestion pipeline (parse, chunk, embed, store)
- [x] Faz 2 — Hybrid RAG (dense + sparse + RRF)
- [x] Faz 3 — Multi-agent evaluation (LangGraph)
- [ ] Faz 4 — RAGAS evaluation pipeline
- [ ] Faz 5 — Observability (Langfuse tracing)
- [ ] Faz 6 — MCP server (expose as AI tool)
- [ ] Faz 7 — Next.js 16 frontend with streaming UI

---

## Why this project exists

In 2026, the bar for "AI engineer" has moved. Building a chatbot that wraps an LLM is table stakes. What companies actually need — and what this project demonstrates — is the ability to build systems that are **observable**, **reliable**, and **self-correcting**.

ContextForge is a deliberate attempt to build every layer of that stack from first principles.

---

## License

MIT — see [LICENSE](LICENSE) for details.