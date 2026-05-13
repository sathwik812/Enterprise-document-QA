# Enterprise Document Q&A — RAG Pipeline Design Document

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Vector Database: ChromaDB vs Pinecone](#2-vector-database-chromadb-vs-pinecone)
3. [Chunking Strategy and Chunk Size Decisions](#3-chunking-strategy-and-chunk-size-decisions)
4. [Embedding Model Selection](#4-embedding-model-selection)
5. [LLM Selection: Gemini vs Ollama](#5-llm-selection-gemini-vs-ollama)
6. [RAGAS Evaluation Approach](#6-ragas-evaluation-approach)
7. [Faithfulness Threshold Tuning](#7-faithfulness-threshold-tuning)
8. [Prompt Injection Defence](#8-prompt-injection-defence)
9. [Hallucination Detection Architecture](#9-hallucination-detection-architecture)
10. [API Design Decisions](#10-api-design-decisions)
11. [Observability and Monitoring](#11-observability-and-monitoring)
12. [Infrastructure and Deployment](#12-infrastructure-and-deployment)
13. [Known Limitations and Future Work](#13-known-limitations-and-future-work)

---

## 1. System Overview

This system implements a Retrieval-Augmented Generation (RAG) pipeline for enterprise document question-answering. The core flow is:

```
User Question
     │
     ▼
Prompt Injection Sanitisation
     │
     ▼
Query Embedding (Gemini / Sentence-Transformers)
     │
     ▼
ChromaDB Vector Search (cosine similarity)
     │
     ▼
Cosine Reranking (top-k selection)
     │
     ▼
LLM Generation (Gemini 2.0 Flash / Ollama llama3)
     │
     ▼
Hallucination Guardrail (RAGAS faithfulness / keyword overlap)
     │
     ▼
Response (with sources, faithfulness score, latency)
```

The system is designed for enterprise deployment with a focus on answer grounding, auditability, and operational observability.

---

## 2. Vector Database: ChromaDB vs Pinecone

### Decision: ChromaDB (local persistent)

**ChromaDB was chosen** for the following reasons:

**Operational simplicity.** ChromaDB runs as an embedded library with no external service dependency. For enterprise deployments that need to keep data on-premises or within a private VPC, this eliminates a network hop and a managed service dependency. The persistent client stores data on disk in a SQLite + HNSW index format that is straightforward to back up and restore.

**Cost.** ChromaDB is free and open-source. Pinecone's managed service charges per vector stored and per query. For a document corpus of millions of chunks, Pinecone costs can become significant. ChromaDB's cost is bounded by the cost of the host machine's storage and compute.

**Latency.** For collections up to ~10 million vectors, ChromaDB's in-process HNSW index delivers sub-10ms query latency on commodity hardware. Pinecone adds network round-trip latency (typically 20–100ms) even for simple queries.

**Privacy.** Enterprise document corpora often contain sensitive information. Keeping embeddings and document chunks in a local ChromaDB instance avoids transmitting proprietary content to a third-party managed service.

**When Pinecone would be preferred:**
- Collections exceeding ~50 million vectors where HNSW memory footprint becomes prohibitive.
- Multi-region deployments requiring globally distributed vector search.
- Teams that need managed scaling, replication, and SLA guarantees without operational overhead.
- Hybrid search (dense + sparse BM25) at scale, which Pinecone supports natively.

**ChromaDB limitations acknowledged:**
- No built-in replication or high availability. For production, the `chroma_db` directory should be on a replicated volume (e.g., EBS with snapshots, or a shared NFS mount).
- Single-node only. Horizontal scaling requires sharding at the application layer.
- The HNSW index is loaded entirely into memory, which limits collection size to available RAM.

---

## 3. Chunking Strategy and Chunk Size Decisions

### Decision: RecursiveCharacterTextSplitter, chunk_size=512, chunk_overlap=50

**Why RecursiveCharacterTextSplitter?**

This splitter attempts to split on natural boundaries in order: paragraphs (`\n\n`), then sentences (`\n`), then words (` `), then characters. This preserves semantic coherence better than a fixed-character splitter, which can cut mid-sentence.

**Why chunk_size=512?**

The 512-token chunk size is a deliberate balance between several competing factors:

- **Retrieval precision.** Smaller chunks are more topically focused, which improves the signal-to-noise ratio when retrieved. A 512-character chunk typically corresponds to 100–150 tokens, which is enough to contain a complete policy statement or procedure step.
- **Context window budget.** With top_k=5, five 512-character chunks consume approximately 500–750 tokens of context, leaving ample room in the LLM's context window for the system prompt, question, and generated answer.
- **Embedding quality.** Sentence-transformer models (and Gemini embeddings) produce better embeddings for focused, coherent passages than for long, multi-topic passages. Chunks that span multiple topics produce embeddings that are "averaged" across topics, reducing retrieval precision.
- **Enterprise document structure.** HR policies, SLAs, and compliance documents are typically structured as numbered clauses or short paragraphs. A 512-character chunk aligns well with this structure.

**Why chunk_overlap=50?**

A 50-character overlap ensures that sentences split across chunk boundaries are represented in both adjacent chunks. Without overlap, a question about a concept that spans a chunk boundary would fail to retrieve the relevant chunk. The overlap is kept small (50 chars ≈ 8–10 words) to avoid excessive duplication in the context window.

**Alternative considered:** Semantic chunking (splitting on embedding similarity drops) was considered but rejected for the initial implementation due to its higher computational cost during ingestion and its dependency on a running embedding model. It can be added as an enhancement.

---

## 4. Embedding Model Selection

### Primary: Gemini text-embedding-004

Google's `text-embedding-004` model produces 768-dimensional embeddings and is optimised for retrieval tasks. It supports a task type parameter (`RETRIEVAL_DOCUMENT` for ingestion, `RETRIEVAL_QUERY` for queries) that improves asymmetric retrieval performance — the model is trained to match short queries against longer document passages.

### Fallback: sentence-transformers/all-MiniLM-L6-v2

When `GOOGLE_API_KEY` is not set, the system falls back to `all-MiniLM-L6-v2`, a 384-dimensional model that runs entirely locally. It is fast (CPU-friendly), small (~80MB), and performs well on English-language retrieval benchmarks. The trade-off is lower embedding quality compared to Gemini, particularly for domain-specific enterprise vocabulary.

**Critical consistency requirement:** The same embedding model must be used for both ingestion and retrieval. The system detects the available model at startup and uses it consistently. If the model changes between ingestion and retrieval runs, the collection must be re-ingested.

---

## 5. LLM Selection: Gemini vs Ollama

### Primary: Gemini 2.0 Flash (`gemini-2.0-flash`)

**Rationale:**
- **Speed.** Gemini 2.0 Flash is optimised for low-latency inference, typically returning responses in 1–3 seconds for RAG-length prompts.
- **Context window.** The model supports a 1M token context window, which is far more than needed for RAG but provides headroom for large document collections.
- **Instruction following.** The model reliably follows the "answer only from context" instruction, which is critical for reducing hallucinations in enterprise Q&A.
- **Cost.** Flash-tier models are significantly cheaper than Pro-tier models, making them suitable for high-volume enterprise deployments.

**Trade-offs:**
- Requires an internet connection and a Google API key.
- Data is sent to Google's infrastructure, which may be a concern for highly sensitive documents.
- API rate limits apply; high-volume deployments need quota management.

### Fallback: Ollama with llama3

When `USE_OLLAMA=true`, the system uses a locally-running Ollama instance with the `llama3` model. This is appropriate for:
- Air-gapped environments with no internet access.
- Deployments where data sovereignty requires all processing to remain on-premises.
- Development and testing without API key costs.

**Trade-offs of Ollama:**
- Requires a GPU for acceptable inference speed (CPU inference is 5–30x slower).
- `llama3` (8B parameters) has lower instruction-following quality than Gemini 2.0 Flash for complex enterprise prompts.
- The developer is responsible for model updates and security patches.

### Prompt Design

The system prompt explicitly instructs the LLM to:
1. Answer **only** from the provided context.
2. State when the context is insufficient rather than guessing.
3. Cite the source document and page number for each claim.

This prompt design is the first line of defence against hallucinations, complementing the RAGAS guardrail.

---

## 6. RAGAS Evaluation Approach

RAGAS (Retrieval-Augmented Generation Assessment) provides a framework for evaluating RAG pipelines without requiring human-labelled answers for every question. The system uses three metrics:

### Faithfulness

Measures whether the claims in the generated answer are supported by the retrieved context. RAGAS decomposes the answer into atomic claims and checks each claim against the context using an LLM judge. Score range: [0, 1].

**Why this is the primary metric:** Faithfulness directly measures hallucination. An answer with faithfulness=1.0 makes no claims that are not supported by the retrieved documents. This is the most important property for enterprise Q&A where incorrect information can have legal or operational consequences.

### Context Precision

Measures whether the retrieved context chunks are relevant to the question. A high context precision score indicates that the retriever is returning focused, relevant chunks rather than noisy, off-topic content.

### Answer Relevance

Measures whether the generated answer actually addresses the question asked. This catches cases where the LLM produces a faithful but tangential response.

### Evaluation Dataset

The `golden_qa.json` dataset contains 50 question/answer pairs covering realistic enterprise document topics: HR policies, IT SLAs, compliance requirements, financial controls, and legal policies. Each entry includes a reference context that represents the ideal retrieved passage.

### CI Gate

The evaluation script exits with code 1 if the mean faithfulness score across all evaluated questions falls below 0.85. This threshold is set higher than the runtime guardrail threshold (0.7) because the CI evaluation uses the reference context directly, which should produce near-perfect faithfulness scores. A score below 0.85 in CI indicates a regression in the LLM's instruction-following or a problem with the prompt template.

---

## 7. Faithfulness Threshold Tuning

### Runtime Threshold: 0.7 (configurable via `FAITHFULNESS_THRESHOLD`)

The 0.7 threshold was chosen based on the following reasoning:

**Precision vs recall trade-off.** A higher threshold (e.g., 0.9) would block more responses, reducing the risk of hallucinations but also blocking legitimate answers where the LLM paraphrases context in ways that reduce the keyword overlap score. A lower threshold (e.g., 0.5) would allow more responses through but would permit answers with significant unsupported claims.

**RAGAS score distribution.** In practice, well-grounded answers from instruction-tuned models typically score 0.8–1.0 on RAGAS faithfulness. Hallucinated answers typically score 0.2–0.5. The 0.7 threshold sits in the gap between these distributions, minimising both false positives (blocking good answers) and false negatives (allowing hallucinations).

**Fallback heuristic calibration.** When RAGAS is unavailable, the keyword-overlap heuristic is used. This heuristic is less precise than RAGAS — it measures lexical overlap rather than semantic entailment. The same 0.7 threshold is applied, but users should be aware that the heuristic has higher false-positive and false-negative rates.

**Tuning guidance:**
- For high-stakes domains (legal, medical, financial): lower the threshold to 0.8–0.9.
- For exploratory or informational use cases: raise the threshold to 0.5–0.6 to reduce blocking.
- Monitor the `blocked` rate in production. If >20% of responses are blocked, the threshold may be too high or the retrieval quality may be poor.

---

## 8. Prompt Injection Defence

Enterprise RAG systems are vulnerable to prompt injection attacks where a user crafts a question designed to override the system prompt and cause the LLM to ignore its instructions or reveal sensitive information.

### Defence Layers

**Layer 1: Input sanitisation (`sanitize_query` in `retriever.py`)**

The `sanitize_query` function applies a set of regex patterns to strip common injection phrases before the query is processed. Patterns include:
- "ignore previous instructions"
- "disregard all prior instructions"
- "you are now a [different persona]"
- "act as if you are"
- System prompt markers (`<system>`, `[INST]`, `### instruction`)
- Jailbreak keywords ("DAN", "do anything now")

If the entire query is consumed by injection patterns, an empty string is returned and the pipeline returns a blocked response without calling the LLM.

**Layer 2: Structured prompt template**

The system prompt and user question are passed as separate `SystemMessage` and `HumanMessage` objects to the LangChain chat interface. This uses the LLM's native role separation, making it harder for injected instructions in the user turn to override the system turn.

**Layer 3: Context isolation**

The retrieved context is injected into the system prompt, not the user prompt. This means the LLM sees the context as part of its instructions rather than as user-provided content, reducing the risk of context-based injection.

**Layer 4: Output validation (guardrail)**

The faithfulness guardrail acts as a post-generation check. If an injection attack causes the LLM to generate an answer that is not grounded in the retrieved context, the faithfulness score will be low and the response will be blocked.

**Limitations:** Regex-based sanitisation is not foolproof. Sophisticated adversarial prompts using encoding tricks, Unicode homoglyphs, or multi-turn context manipulation may bypass the regex layer. For high-security deployments, consider adding an LLM-based prompt injection classifier as an additional layer.

---

## 9. Hallucination Detection Architecture

The `HallucinationGuard` class implements a two-tier detection strategy:

**Tier 1: RAGAS faithfulness (preferred)**

RAGAS uses an LLM judge to decompose the answer into atomic claims and verify each claim against the context. This is semantically aware and handles paraphrasing, synonyms, and implicit entailment. The main cost is an additional LLM call per query.

**Tier 2: Keyword overlap heuristic (fallback)**

When RAGAS is unavailable (import error) or fails (API error, timeout), the system falls back to a keyword overlap score: the fraction of unique words in the answer that also appear in the context. This is a crude but fast approximation. It works reasonably well for factual answers but struggles with:
- Answers that paraphrase context using different vocabulary.
- Short answers where a few keywords dominate the score.
- Answers that correctly say "I don't know" (which will score low against any context).

The fallback ensures the system degrades gracefully rather than failing open (allowing all responses) or failing closed (blocking all responses).

---

## 10. API Design Decisions

### FastAPI over Flask/Django

FastAPI was chosen for its native async support, automatic OpenAPI documentation generation, and Pydantic-based request/response validation. The async support is important for a RAG API where LLM calls can take several seconds — async allows the server to handle other requests while waiting for the LLM.

### Chain Caching

The `_get_chain` function caches `RAGChain` instances by `(collection, top_k)` key. This avoids re-initialising the ChromaDB client and embedding model on every request, which would add 1–5 seconds of overhead per query. The cache is in-process memory, so it is reset on server restart.

### Synchronous LLM calls in async handlers

The current implementation calls the LLM synchronously within async FastAPI handlers. For production, these calls should be wrapped with `asyncio.get_event_loop().run_in_executor()` or replaced with async LangChain interfaces to avoid blocking the event loop. This is noted as a future improvement.

### Error handling

All endpoints return structured error responses via `HTTPException`. The `/ingest` endpoint distinguishes between `404` (file not found) and `500` (ingestion failure) to give clients actionable error information.

---

## 11. Observability and Monitoring

### Prometheus Metrics

Three metrics are exposed at `/metrics`:

- **`query_latency_seconds` (Histogram):** End-to-end latency from question receipt to response. Buckets are set at 0.1s, 0.25s, 0.5s, 1s, 2.5s, 5s, 10s to capture the typical LLM response time distribution.
- **`llm_response_time_seconds` (Histogram):** LLM-only generation time, isolated from retrieval and guardrail overhead. Useful for detecting LLM API degradation.
- **`retrieval_hit_rate_total` (Counter):** Counts queries that returned at least one document. A declining hit rate may indicate collection drift or query distribution shift.

### Grafana

The docker-compose stack includes Grafana pre-configured to use Prometheus as a data source. Recommended dashboards:
- Query latency percentiles (p50, p95, p99).
- LLM response time vs total query latency (to isolate bottlenecks).
- Retrieval hit rate over time.
- Blocked response rate (requires adding a `blocked_responses_total` counter — noted as future work).

---

## 12. Infrastructure and Deployment

### Multi-stage Dockerfile

The Dockerfile uses a two-stage build:
1. **Builder stage:** Installs all Python dependencies (including native extensions like `chromadb` and `sentence-transformers`) into `/install`.
2. **Final stage:** Copies only the installed packages and application code into a slim Python 3.11 image. This reduces the final image size by ~60% compared to a single-stage build.

The application runs as a non-root user (`appuser`) for security.

### Volume Mounts

The `chroma_db` directory is mounted as a Docker volume to persist the vector store across container restarts. In production, this volume should be backed by a durable storage backend (e.g., AWS EBS, Azure Managed Disk) with automated snapshots.

### Environment Variables

All configuration is injected via environment variables (`.env` file or Docker secrets). The `.env.example` file documents all required and optional variables. Sensitive values (`GOOGLE_API_KEY`) must never be committed to version control.

---

## 13. Known Limitations and Future Work

| Area | Current State | Improvement |
|------|--------------|-------------|
| Async LLM calls | Synchronous (blocks event loop) | Use `run_in_executor` or async LangChain |
| Reranking | Simple cosine score sort | Cross-encoder reranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| Multi-collection queries | Single collection per query | Query federation across collections |
| Streaming responses | Not implemented | SSE streaming for long answers |
| ChromaDB HA | Single node | Chroma distributed mode or migration to Qdrant |
| Injection detection | Regex only | LLM-based injection classifier |
| Blocked response rate metric | Not tracked | Add `blocked_responses_total` Prometheus counter |
| Document update/delete | Re-ingest only | Implement chunk-level upsert and delete by source |
| Multilingual support | English-optimised | Multilingual embedding model (e.g., `paraphrase-multilingual-MiniLM-L12-v2`) |
| Semantic chunking | Fixed-size only | Semantic boundary detection for better chunk coherence |
