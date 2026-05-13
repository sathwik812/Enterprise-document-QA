# 📄 Enterprise Document Q&A — RAG Pipeline with Hallucination Detection

> A production-grade Retrieval-Augmented Generation (RAG) system that lets you ask natural language questions over private PDF/text documents — with built-in hallucination detection, observability, and CI-gated evaluation.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-0.2-yellow)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange)
![Prometheus](https://img.shields.io/badge/Prometheus-monitoring-red?logo=prometheus)
![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)
![AWS](https://img.shields.io/badge/AWS-EC2-orange?logo=amazonaws)

---

## 🔍 What Problem Does This Solve?

Most organisations have vast amounts of knowledge locked in PDFs, SOPs, reports, and internal documents. Employees waste hours searching through them manually. Generic LLMs don't know your private data — and when they guess, they hallucinate.

This system solves that by:
- Ingesting your documents into a vector database
- Retrieving only the most relevant context for each query
- Using an LLM to generate grounded, cited answers
- **Automatically detecting and blocking hallucinated responses** before they reach the user

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                   │
│  PDF/TXT → Chunker → Embedder → ChromaDB Vector Store   │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                      │
│  User Query → Embed → Retrieve Top-K → Rerank →         │
│  LLM (with context) → Guardrail Check → Cited Answer    │
└─────────────────────────────────────────────────────────┘
                            │
┌─────────────────────────────────────────────────────────┐
│                     OBSERVABILITY LAYER                  │
│  Prometheus (latency, hit rate, errors) + Grafana UI     │
│  RAGAS evaluation on CI (faithfulness, precision)        │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

| Feature | Details |
|---|---|
| **Document Ingestion** | PDF, TXT, DOCX support via LangChain document loaders |
| **Vector Storage** | ChromaDB with persistent storage; Pinecone-ready |
| **Embedding Models** | Gemini text-embedding-004 or local Sentence-Transformers |
| **LLM Backend** | Gemini 2.5-Flash-Lite or Ollama (local, offline) |
| **Hallucination Guard** | RAGAS faithfulness check — blocks answers with score < 0.7 |
| **Citation Enforcement** | Every answer includes the source document + page number |
| **Prompt Injection Defence** | Input sanitisation layer rejects adversarial prompts |
| **Observability** | Prometheus metrics: query latency, retrieval hit rate, LLM response time |
| **CI Evaluation** | 50-question golden test set runs on every PR via GitHub Actions |
| **Demo UI** | Streamlit interface for live demonstration |
| **Deployment** | Dockerized; deployed on AWS EC2 with HTTPS via nginx |

---

## 🛠️ Tech Stack

- **Orchestration:** LangChain
- **Vector DB:** ChromaDB (local) / Pinecone (cloud)
- **API Layer:** FastAPI
- **Evaluation:** RAGAS
- **Monitoring:** Prometheus + Grafana
- **UI:** Streamlit
- **LLM:** Google API / Ollama
- **Infra:** Docker, AWS EC2, GitHub Actions

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Google API key (or Ollama installed locally)

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/rag-document-qa
cd rag-document-qa
```

### 2. Set up environment
```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env
pip install -r requirements.txt
```

### 3. Ingest your documents
```bash
python src/ingest.py --input-dir ./docs/sample_docs --collection my-docs
```

### 4. Run the API
```bash
uvicorn src.api.main:app --reload
# API docs: http://localhost:8000/docs
```

### 5. Launch the Streamlit demo
```bash
streamlit run src/ui/app.py
```

### 6. Run with Docker (recommended)
```bash
docker-compose up --build
# API: http://localhost:8000
# Grafana: http://localhost:3000
```

---

## 📁 Project Structure

```
rag-document-qa/
├── src/
│   ├── ingest.py          # Document loading, chunking, embedding
│   ├── retriever.py       # ChromaDB vector search + reranking
│   ├── chain.py           # LangChain RAG chain with guardrails
│   ├── guardrails.py      # Hallucination detection via RAGAS
│   ├── api/
│   │   └── main.py        # FastAPI endpoints
│   └── ui/
│       └── app.py         # Streamlit demo interface
├── eval/
│   ├── golden_qa.json     # 50-question evaluation test set
│   └── evaluate.py        # RAGAS scoring script
├── infra/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── prometheus.yml
├── docs/
│   ├── architecture.png   # System diagram
│   └── design.md          # Full design decisions & trade-offs
├── .github/
│   └── workflows/
│       └── eval.yml       # CI: runs RAGAS evaluation on every PR
├── .env.example
└── README.md
```

---

## 📊 API Reference

### `POST /ingest`
Upload and ingest a document into the vector store.
**Body:** `multipart/form-data`
- `file`: The document file (PDF, TXT, DOCX)
- `collection`: Collection name (string)

```bash
curl -X POST http://localhost:8000/ingest \
  -H "X-API-KEY: dev-key-123" \
  -F "file=@/path/to/doc.pdf" \
  -F "collection=ops-docs"
```

### `POST /query`
Query the RAG pipeline.
```json
{
  "question": "What is the SLA for P1 incidents?",
  "collection": "ops-docs",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "P1 incidents must be resolved within 4 hours...",
  "sources": [{"document": "sla-policy.pdf", "page": 3}],
  "faithfulness_score": 0.92,
  "latency_ms": 1240
}
```

---

## 📈 Evaluation Results

Evaluated on a 50-question golden test set across 3 document collections:

| Metric | Score |
|---|---|
| Faithfulness | 0.91 |
| Context Precision | 0.87 |
| Answer Relevance | 0.89 |
| Hallucination Block Rate | 94% (correctly blocked 17/18 hallucinations) |

---

## 🔑 Design Decisions & Trade-offs

See [`docs/design.md`](docs/design.md) for full write-up. Key decisions:

- **ChromaDB over Pinecone** — local-first for cost; Pinecone adapter available for scale
- **RAGAS over manual eval** — automated, reproducible, CI-compatible
- **Faithfulness threshold 0.7** — tuned to balance precision vs. recall on the golden set
- **Chunk size 512 tokens, overlap 50** — empirically optimal for the document types tested

---

## 🗺️ Roadmap

- [ ] Hybrid search (BM25 + vector) for improved retrieval recall
- [ ] Multi-tenant collection management with auth
- [ ] QLoRA fine-tuning pipeline for domain-specific embedding model
- [ ] LangSmith tracing integration
- [ ] Async ingestion queue with Celery
(BM25 + vector) for improved retrieval recall
- [ ] Multi-tenant collection management with auth
- [ ] QLoRA fine-tuning pipeline for domain-specific embedding model
- [ ] LangSmith tracing integration
- [ ] Async ingestion queue with Celery
