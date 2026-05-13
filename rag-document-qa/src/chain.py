"""
RAG chain: ties together retrieval, LLM generation, and hallucination guardrails.
"""

import logging
import os
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------
try:
    from prometheus_client import Counter, Histogram

    query_latency_seconds = Histogram(
        "query_latency_seconds",
        "End-to-end query latency in seconds",
        buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    llm_response_time_seconds = Histogram(
        "llm_response_time_seconds",
        "LLM generation latency in seconds",
        buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
    retrieval_hit_rate = Counter(
        "retrieval_hit_rate_total",
        "Number of queries that returned at least one document",
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not available; metrics disabled.")


def _record_query_latency(seconds: float) -> None:
    if _PROMETHEUS_AVAILABLE:
        query_latency_seconds.observe(seconds)


def _record_llm_latency(seconds: float) -> None:
    if _PROMETHEUS_AVAILABLE:
        llm_response_time_seconds.observe(seconds)


def _increment_hit_rate() -> None:
    if _PROMETHEUS_AVAILABLE:
        retrieval_hit_rate.inc()


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a precise enterprise document assistant.
Answer the user's question using ONLY the information provided in the context below.
If the context does not contain enough information to answer the question, say:
"I don't have enough information in the provided documents to answer this question."

Do NOT make up facts, figures, or policies that are not explicitly stated in the context.
Always cite the source document and page number when referencing specific information.

Context:
{context}
"""

_HUMAN_PROMPT = "Question: {question}\n\nAnswer (cite sources):"


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _build_llm():
    """Return the appropriate LLM based on environment configuration."""
    use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

    if use_ollama:
        from langchain_community.llms import Ollama

        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3")
        logger.info("Using Ollama (%s) at %s", model, base_url)
        return Ollama(model=model, base_url=base_url)

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set and USE_OLLAMA is not true. "
            "Set one of these to use the RAG chain."
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    model = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    logger.info("Using Gemini %s", model)
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0.1,
        convert_system_message_to_human=True,
    )


# ---------------------------------------------------------------------------
# RAGChain
# ---------------------------------------------------------------------------

class RAGChain:
    """
    Full RAG pipeline: sanitise → retrieve → generate → guardrail.

    Parameters
    ----------
    collection : str
        ChromaDB collection name.
    top_k : int
        Number of context chunks to retrieve.
    """

    def __init__(self, collection: str, top_k: int = 5) -> None:
        from src.guardrails import HallucinationGuard
        from src.retriever import DocumentRetriever

        self.collection = collection
        self.top_k = top_k
        self._retriever = DocumentRetriever(collection=collection, top_k=top_k)
        self._guard = HallucinationGuard()
        self._llm = _build_llm()
        logger.info(
            "RAGChain initialised — collection='%s', top_k=%d", collection, top_k
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context_string(self, hits: list[dict]) -> str:
        parts = []
        for i, hit in enumerate(hits, start=1):
            source = hit.get("source", "unknown")
            page = hit.get("page", "?")
            content = hit.get("content", "")
            parts.append(f"[{i}] Source: {source} (page {page})\n{content}")
        return "\n\n---\n\n".join(parts)

    def _call_llm(self, context: str, question: str) -> str:
        """Invoke the LLM and return the answer text."""
        from langchain.schema import HumanMessage, SystemMessage

        system_content = _SYSTEM_PROMPT.format(context=context)
        human_content = _HUMAN_PROMPT.format(question=question)

        use_ollama = os.getenv("USE_OLLAMA", "false").lower() == "true"

        if use_ollama:
            # Ollama LLM (non-chat) — concatenate into a single prompt
            full_prompt = f"{system_content}\n\n{human_content}"
            response = self._llm(full_prompt)
            return response if isinstance(response, str) else str(response)
        else:
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=human_content),
            ]
            response = self._llm(messages)
            return response.content

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(self, question: str) -> dict[str, Any]:
        """
        Run the full RAG pipeline for *question*.

        Returns
        -------
        dict with keys:
            answer           – str, the LLM answer (or blocked message)
            sources          – list of {document, page}
            faithfulness_score – float
            latency_ms       – int
            blocked          – bool
        """
        pipeline_start = time.perf_counter()

        # 1. Sanitise
        from src.retriever import sanitize_query

        clean_question = sanitize_query(question)
        if not clean_question:
            return {
                "answer": "Your query was flagged as potentially malicious and could not be processed.",
                "sources": [],
                "faithfulness_score": 0.0,
                "latency_ms": 0,
                "blocked": True,
            }

        # 2. Retrieve
        hits = self._retriever.retrieve(clean_question)

        if hits:
            _increment_hit_rate()

        sources = [
            {"document": h.get("source", "unknown"), "page": h.get("page", "0")}
            for h in hits
        ]

        if not hits:
            elapsed_ms = int((time.perf_counter() - pipeline_start) * 1000)
            _record_query_latency(elapsed_ms / 1000)
            return {
                "answer": "No relevant documents were found for your question.",
                "sources": [],
                "faithfulness_score": 0.0,
                "latency_ms": elapsed_ms,
                "blocked": False,
            }

        context_str = self._build_context_string(hits)

        # 3. LLM generation
        llm_start = time.perf_counter()
        answer = self._call_llm(context_str, clean_question)
        llm_elapsed = time.perf_counter() - llm_start
        _record_llm_latency(llm_elapsed)

        # 4. Guardrail
        contexts_text = [h["content"] for h in hits]
        guard_result = self._guard.check(clean_question, answer, contexts_text)

        faithfulness_score = guard_result["score"]
        blocked = not guard_result["passed"]

        if blocked:
            logger.warning(
                "Response blocked by guardrail. Reason: %s", guard_result["reason"]
            )
            answer = (
                "The generated response did not meet the faithfulness threshold "
                "and has been blocked to prevent potential hallucinations. "
                f"Reason: {guard_result['reason']}"
            )

        pipeline_elapsed = time.perf_counter() - pipeline_start
        _record_query_latency(pipeline_elapsed)

        return {
            "answer": answer,
            "sources": sources,
            "faithfulness_score": faithfulness_score,
            "latency_ms": int(pipeline_elapsed * 1000),
            "blocked": blocked,
        }
