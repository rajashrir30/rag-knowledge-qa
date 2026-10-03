"""BM25 keyword search over the chunks stored in ChromaDB."""

from __future__ import annotations

import hashlib
import pickle
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from config import load_settings
from src.vector_store import VectorStore


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "were", "what", "where", "which", "who", "with",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split, and remove basic stopwords."""
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return [token for token in re.findall(r"\b\w+\b", text) if token not in _STOPWORDS]


@dataclass
class BM25Index:
    chunks: list[dict[str, Any]]
    tokenized_corpus: list[list[str]]
    bm25: BM25Okapi
    source_hash: str
    chunk_signature: str

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if k <= 0 or not self.chunks:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.chunks)), key=lambda index: scores[index], reverse=True)[:k]
        return [
            {
                "text": self.chunks[index].get("text", ""),
                "source": self.chunks[index].get("source", ""),
                "chunk_id": self.chunks[index].get("chunk_id"),
                "start": self.chunks[index].get("start"),
                "end": self.chunks[index].get("end"),
                "bm25_score": float(scores[index]),
            }
            for index in ranked
        ]


def _signatures(chunks: list[dict[str, Any]]) -> tuple[str, str]:
    source_names = sorted({str(chunk.get("source", "")) for chunk in chunks})
    source_hash = hashlib.sha256("\0".join(source_names).encode("utf-8")).hexdigest()
    chunk_values = sorted(
        f"{chunk.get('source', '')}\0{chunk.get('chunk_id', '')}\0{chunk.get('text', '')}"
        for chunk in chunks
    )
    chunk_signature = hashlib.sha256("\0".join(chunk_values).encode("utf-8")).hexdigest()
    return source_hash, chunk_signature


def build_bm25_index(chunks: list[dict[str, Any]], cache_dir: str | Path | None = None) -> BM25Index:
    """Build or load a cached BM25 index keyed by source filenames."""
    source_hash, chunk_signature = _signatures(chunks)
    settings = load_settings()
    cache_path = Path(cache_dir or settings.chroma_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f"bm25_{source_hash[:16]}.pkl"

    if cache_file.exists():
        try:
            with cache_file.open("rb") as handle:
                cached = pickle.load(handle)
            if cached.source_hash == source_hash and cached.chunk_signature == chunk_signature:
                return cached
        except (OSError, AttributeError, EOFError, pickle.PickleError):
            pass

    tokenized_corpus = [tokenize(str(chunk.get("text", ""))) for chunk in chunks]
    index = BM25Index(
        chunks=chunks,
        tokenized_corpus=tokenized_corpus,
        bm25=BM25Okapi(tokenized_corpus),
        source_hash=source_hash,
        chunk_signature=chunk_signature,
    )
    with cache_file.open("wb") as handle:
        pickle.dump(index, handle)
    return index


def _load_chunks(store: VectorStore) -> list[dict[str, Any]]:
    result = store.collection.get(include=["documents", "metadatas"])
    chunks = []
    for text, metadata in zip(result.get("documents") or [], result.get("metadatas") or []):
        metadata = metadata or {}
        chunks.append({
            "text": text,
            "source": metadata.get("source", ""),
            "chunk_id": metadata.get("chunk_id", 0),
            "start": metadata.get("start"),
            "end": metadata.get("end"),
        })
    return chunks


def keyword_search(
    query: str,
    k: int = 5,
    *,
    chunks: list[dict[str, Any]] | None = None,
    index: BM25Index | None = None,
    store: VectorStore | None = None,
) -> list[dict[str, Any]]:
    """Search the stored corpus with BM25, optionally using an injected corpus/index."""
    if k <= 0:
        return []
    if index is None:
        if chunks is None:
            chunks = _load_chunks(store or VectorStore())
        index = build_bm25_index(chunks, cache_dir=(store.path if store else None))
    return index.search(query, k=k)
