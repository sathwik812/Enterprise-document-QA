"""
Document ingestion pipeline for the RAG system.

Usage:
    python src/ingest.py --input-dir ./docs/sample_pdfs --collection my-docs
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
try:
    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
except ImportError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangChain / ChromaDB imports
# ---------------------------------------------------------------------------
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
import chromadb
from chromadb.config import Settings


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _get_embeddings():
    """Return an embeddings object — Gemini if API key present, else local."""
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if api_key:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            logger.info("Using Gemini text-embedding-004 for embeddings.")
            return GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=api_key,
            )
        except Exception as exc:
            logger.warning("Gemini embeddings failed (%s). Falling back to local model.", exc)

    logger.info("Using sentence-transformers/all-MiniLM-L6-v2 for embeddings.")
    from langchain_community.embeddings import SentenceTransformerEmbeddings

    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# Loader selection
# ---------------------------------------------------------------------------

_LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".docx": Docx2txtLoader,
}


def _load_document(file_path: Path):
    """Load a single document using the appropriate LangChain loader."""
    suffix = file_path.suffix.lower()
    loader_cls = _LOADER_MAP.get(suffix)
    if loader_cls is None:
        logger.warning("Unsupported file type '%s' — skipping %s", suffix, file_path)
        return []
    logger.info("Loading %s", file_path)
    loader = loader_cls(str(file_path))
    return loader.load()


# ---------------------------------------------------------------------------
# Core ingestion function
# ---------------------------------------------------------------------------

def ingest_directory(input_dir: str, collection_name: str) -> int:
    """
    Ingest all supported documents from *input_dir* into a ChromaDB collection.

    Returns the total number of chunks stored.
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error("Input directory '%s' does not exist.", input_dir)
        sys.exit(1)

    # Collect all supported files
    supported_extensions = set(_LOADER_MAP.keys())
    files = [
        f
        for f in input_path.rglob("*")
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    if not files:
        logger.warning("No supported documents found in '%s'.", input_dir)
        return 0

    logger.info("Found %d document(s) to ingest.", len(files))

    # Load documents
    all_docs = []
    for file_path in files:
        docs = _load_document(file_path)
        all_docs.extend(docs)

    logger.info("Loaded %d page(s)/section(s) total.", len(all_docs))

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = splitter.split_documents(all_docs)
    logger.info("Split into %d chunk(s).", len(chunks))

    # Embeddings
    embeddings = _get_embeddings()

    # ChromaDB persistent client
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    client = chromadb.PersistentClient(path=persist_dir)

    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Embed and upsert in batches
    batch_size = 100
    total_stored = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [doc.page_content for doc in batch]
        metadatas = []
        for doc in batch:
            meta = dict(doc.metadata)
            # Ensure source and page keys exist
            meta.setdefault("source", meta.get("file_path", "unknown"))
            meta.setdefault("page", str(meta.get("page", 0)))
            # ChromaDB requires all metadata values to be str/int/float/bool
            meta = {k: str(v) for k, v in meta.items()}
            metadatas.append(meta)

        ids = [f"{collection_name}-chunk-{i + j}" for j in range(len(batch))]

        # Compute embeddings
        vectors = embeddings.embed_documents(texts)

        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )
        total_stored += len(batch)
        logger.info("Stored batch %d/%d (%d chunks so far).", i // batch_size + 1, (len(chunks) - 1) // batch_size + 1, total_stored)

    logger.info(
        "Ingestion complete. %d chunk(s) stored in collection '%s'.",
        total_stored,
        collection_name,
    )
    return total_stored


def ingest_file(file_path: str, collection_name: str) -> int:
    """Ingest a single file into a ChromaDB collection."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    docs = _load_document(path)
    if not docs:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split '%s' into %d chunk(s).", file_path, len(chunks))

    embeddings = _get_embeddings()

    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [doc.page_content for doc in chunks]
    metadatas = []
    for doc in chunks:
        meta = dict(doc.metadata)
        meta.setdefault("source", file_path)
        meta.setdefault("page", str(meta.get("page", 0)))
        meta = {k: str(v) for k, v in meta.items()}
        metadatas.append(meta)

    # Use a stable prefix based on file name to allow re-ingestion
    file_stem = path.stem
    ids = [f"{collection_name}-{file_stem}-{j}" for j in range(len(chunks))]

    vectors = embeddings.embed_documents(texts)
    collection.upsert(
        ids=ids,
        embeddings=vectors,
        documents=texts,
        metadatas=metadatas,
    )

    logger.info("Stored %d chunk(s) from '%s'.", len(chunks), file_path)
    return len(chunks)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents into the RAG vector store."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing PDF, TXT, or DOCX files to ingest.",
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="ChromaDB collection name to store documents in.",
    )
    args = parser.parse_args()

    ingest_directory(args.input_dir, args.collection)
