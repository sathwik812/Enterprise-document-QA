"""
FastAPI application for the RAG Document Q&A system.

Endpoints:
    POST /ingest   — ingest a document file into a collection
    POST /query    — run the RAG pipeline for a question
    GET  /health   — liveness check
    GET  /metrics  — Prometheus metrics
"""

import asyncio
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Security, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

load_dotenv()

# Ensure the project root is on sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global Executor for blocking tasks
# ---------------------------------------------------------------------------
executor = ThreadPoolExecutor(max_workers=10)

# ---------------------------------------------------------------------------
# Security: API Key
# ---------------------------------------------------------------------------
API_KEY = os.getenv("API_KEY", "dev-key-123")
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

async def get_api_key(header_key: str = Security(api_key_header)):
    if header_key == API_KEY:
        return header_key
    raise HTTPException(status_code=403, detail="Invalid or missing API Key")

# ---------------------------------------------------------------------------
# Prometheus
# ---------------------------------------------------------------------------
try:
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed; /metrics will return empty.")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RAG Document Q&A API",
    description="Enterprise document question-answering with hallucination detection.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    status: str
    message: str
    collection: str
    file_name: str

class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to answer.")
    collection: str = Field(..., description="ChromaDB collection to query.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of context chunks to retrieve.")

class SourceItem(BaseModel):
    document: str
    page: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    faithfulness_score: float
    latency_ms: int
    blocked: bool

class HealthResponse(BaseModel):
    status: str

# ---------------------------------------------------------------------------
# Chain cache
# ---------------------------------------------------------------------------

_chain_cache: dict[str, object] = {}

def _get_chain(collection: str, top_k: int = 5):
    from src.chain import RAGChain
    cache_key = f"{collection}:{top_k}"
    if cache_key not in _chain_cache:
        _chain_cache[cache_key] = RAGChain(collection=collection, top_k=top_k)
    return _chain_cache[cache_key]

# ---------------------------------------------------------------------------
# Background Task for Ingestion
# ---------------------------------------------------------------------------

def run_ingestion(tmp_path: str, collection: str, original_filename: str):
    """Worker function to run ingestion in a thread."""
    try:
        from src.ingest import ingest_file
        chunks = ingest_file(file_path=tmp_path, collection_name=collection)
        logger.info("Background ingestion success: %s (%d chunks)", original_filename, chunks)
    except Exception as exc:
        logger.exception("Background ingestion failed for %s", original_filename)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    return HealthResponse(status="ok")

@app.get("/metrics", response_class=PlainTextResponse, tags=["System"])
async def metrics() -> PlainTextResponse:
    if not _PROMETHEUS_AVAILABLE:
        return PlainTextResponse("# prometheus_client not installed\n", status_code=200)
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
async def ingest(
    background_tasks: BackgroundTasks,
    collection: str = Form(..., description="ChromaDB collection name."),
    file: UploadFile = File(...),
    api_key: str = Depends(get_api_key)
) -> IngestResponse:
    import tempfile
    import shutil

    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # Offload ingestion to background task
    background_tasks.add_task(run_ingestion, tmp_path, collection, file.filename)

    return IngestResponse(
        status="accepted",
        message="Document received and scheduled for ingestion.",
        collection=collection,
        file_name=file.filename,
    )

@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(
    request: QueryRequest,
    api_key: str = Depends(get_api_key)
) -> QueryResponse:
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    try:
        loop = asyncio.get_event_loop()
        chain = _get_chain(collection=request.collection, top_k=request.top_k)
        
        # Offload blocking RAG pipeline to the thread pool
        result = await loop.run_in_executor(
            executor, 
            chain.query, 
            request.question
        )

        sources = [
            SourceItem(document=s.get("document", ""), page=str(s.get("page", "0")))
            for s in result.get("sources", [])
        ]

        return QueryResponse(
            answer=result["answer"],
            sources=sources,
            faithfulness_score=result.get("faithfulness_score", 0.0),
            latency_ms=result.get("latency_ms", 0),
            blocked=result.get("blocked", False),
        )
    except Exception as exc:
        logger.exception("Query failed: %s", request.question)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {exc}")

