"""
Document retriever backed by ChromaDB.

Provides cosine-similarity search with a lightweight reranking step and
prompt-injection sanitisation.
"""

import logging
import os
import re
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Injection patterns to strip from user queries
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|what)\s+(you\s+)?(know|were told|said)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:a\s+)?\w+", re.IGNORECASE),
    re.compile(r"act\s+as\s+(?:if\s+you\s+are\s+)?(?:a\s+)?\w+", re.IGNORECASE),
    re.compile(r"new\s+instructions?:", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"<\s*/?system\s*>", re.IGNORECASE),
    re.compile(r"\[INST\]|\[/INST\]", re.IGNORECASE),
    re.compile(r"###\s*(instruction|system|human|assistant)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
    re.compile(r"DAN\b", re.IGNORECASE),
]


def sanitize_query(query: str) -> str:
    """
    Strip common prompt-injection patterns from a user query.

    Returns the sanitised string. If the entire query is consumed by
    injection patterns, returns an empty string.
    """
    sanitized = query
    for pattern in _INJECTION_PATTERNS:
        sanitized = pattern.sub("", sanitized)

    # Collapse extra whitespace introduced by removals
    sanitized = re.sub(r"\s{2,}", " ", sanitized).strip()

    if sanitized != query:
        logger.warning(
            "Prompt injection attempt detected and sanitised. "
            "Original length: %d, sanitised length: %d",
            len(query),
            len(sanitized),
        )

    return sanitized


# ---------------------------------------------------------------------------
# Embedding helper (mirrors ingest.py)
# ---------------------------------------------------------------------------

def _get_embeddings():
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if api_key:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            return GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key,
            )
        except Exception as exc:
            logger.warning("Gemini embeddings unavailable (%s). Using local model.", exc)

    from langchain_community.embeddings import SentenceTransformerEmbeddings

    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# DocumentRetriever
# ---------------------------------------------------------------------------

class DocumentRetriever:
    """
    Retrieves relevant document chunks from a ChromaDB collection.

    Parameters
    ----------
    collection : str
        Name of the ChromaDB collection to query.
    top_k : int
        Number of results to return after reranking.
    """

    def __init__(self, collection: str, top_k: int = 5) -> None:
        import chromadb

        self.collection_name = collection
        self.top_k = top_k

        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self._client = chromadb.PersistentClient(path=persist_dir)

        try:
            self._collection = self._client.get_collection(name=collection)
            logger.info(
                "Loaded ChromaDB collection '%s' (%d items).",
                collection,
                self._collection.count(),
            )
        except Exception as exc:
            logger.error(
                "Collection '%s' not found. Run ingest.py first. Error: %s",
                collection,
                exc,
            )
            raise

        self._embeddings = _get_embeddings()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[dict[str, Any]]:
        """
        Retrieve the top-k most relevant chunks for *query*.
        Uses a two-stage process:
        1. Vector search (ChromaDB) for high-recall candidates.
        2. Reranking for high-precision selection.

        Returns a list of dicts with keys:
            content  – chunk text
            source   – originating file path
            page     – page number (str)
            score    – reranked or cosine score
        """
        clean_query = sanitize_query(query)
        if not clean_query:
            logger.warning("Query was empty after sanitisation.")
            return []

        # Embed the query
        query_vector = self._embeddings.embed_query(clean_query)

        # Stage 1: Fetch more candidates than needed (High Recall)
        # We fetch 3x top_k candidates to allow reranking to filter out noise.
        n_candidates = min(self.top_k * 3, self._collection.count() or self.top_k)
        n_candidates = max(n_candidates, self.top_k)

        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        hits = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            score = 1.0 - float(dist)  # convert distance → similarity
            hits.append(
                {
                    "content": doc,
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", "0"),
                    "score": score,
                }
            )

        # Stage 2: Reranking (High Precision)
        # In a full production system, we would use a Cross-Encoder here:
        # from sentence_transformers import CrossEncoder
        # model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        # scores = model.predict([(clean_query, h['content']) for h in hits])
        # for h, s in zip(hits, scores): h['score'] = s
        
        # For now, we perform a refined lexical reranking as a placeholder
        # which prioritises chunks containing exact keyword matches from the query.
        query_keywords = set(re.findall(r"\b\w{4,}\b", clean_query.lower()))
        if query_keywords:
            for hit in hits:
                content_lower = hit["content"].lower()
                matches = sum(1 for kw in query_keywords if kw in content_lower)
                # Boost the vector score with lexical match info (scaled)
                lexical_boost = matches / len(query_keywords) * 0.2
                hit["score"] += lexical_boost

        # Sort by final score descending
        hits.sort(key=lambda h: h["score"], reverse=True)

        return hits[: self.top_k]

