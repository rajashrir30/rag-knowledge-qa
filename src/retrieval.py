"""Nearest-neighbor retrieval over the persistent Chroma vector store."""

from __future__ import annotations

import argparse
import math
import os
from typing import Any

from src.embeddings import Embedder, get_embedder
from src.vector_store import VectorStore


def _enabled(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _distance_to_score(distance: float) -> float:
    """Map a non-negative Chroma distance to a bounded similarity score."""
    if not math.isfinite(distance) or distance < 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 / (1.0 + distance)))


class Retriever:
    """Retrieval service with injectable dependencies for tests and applications."""

    def __init__(self, store: VectorStore | None = None, embedder: Embedder | None = None, use_reranker: bool | None = None):
        # Dependencies are lazy so the standalone rerank() helper does not
        # initialize Chroma or an API-backed embedder unnecessarily.
        self.store = store
        self.embedder = embedder
        self.use_reranker = _enabled(os.getenv("USE_RERANKER")) if use_reranker is None else use_reranker
        self._reranker = None

    def _get_reranker(self) -> Any:
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            model_name = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            self._reranker = CrossEncoder(model_name)
        return self._reranker

    def retrieve(self, query: str, k: int = 5, min_score: float = 0.0) -> list[dict[str, Any]]:
        if not query or k <= 0:
            return []
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")

        self.store = self.store or VectorStore()
        self.embedder = self.embedder or get_embedder()
        query_embedding = self.embedder.embed_texts([query])[0]
        result = self.store.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        candidates: list[dict[str, Any]] = []
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for text, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            candidates.append({
                "text": text,
                "source": metadata.get("source", ""),
                "chunk_id": metadata.get("chunk_id", 0),
                "start": metadata.get("start"),
                "end": metadata.get("end"),
                "score": _distance_to_score(float(distance)),
            })

        if self.use_reranker and candidates:
            candidates = self.rerank(query, candidates, k)
        return [candidate for candidate in candidates if candidate["score"] >= min_score][:k]

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
        """Re-rank candidates with a cross-encoder and sigmoid-normalized scores."""
        if not candidates or top_n <= 0:
            return []
        model = self._get_reranker()
        raw_scores = model.predict([(query, candidate["text"]) for candidate in candidates])
        ranked = []
        for candidate, raw_score in zip(candidates, raw_scores):
            updated = dict(candidate)
            updated["score"] = 1.0 / (1.0 + math.exp(-float(raw_score)))
            ranked.append(updated)
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:top_n]


_DEFAULT_RETRIEVER: Retriever | None = None


def retrieve(query: str, k: int = 5, min_score: float = 0.0) -> list[dict[str, Any]]:
    """Retrieve the most relevant stored chunks for a query."""
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = Retriever()
    return _DEFAULT_RETRIEVER.retrieve(query, k=k, min_score=min_score)


def rerank(query: str, candidates: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    """Convenience wrapper for optional cross-encoder re-ranking."""
    return Retriever(use_reranker=True).rerank(query, candidates, top_n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve relevant chunks from ChromaDB.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.0)
    args = parser.parse_args()
    for rank, result in enumerate(retrieve(args.query, args.k, args.min_score), start=1):
        print(f"{rank}. score={result['score']:.4f} source={result['source']} chunk={result['chunk_id']}")
        if result["start"] is not None:
            print(f"   time={result['start']:.3f}-{result['end']:.3f}")
        print(f"   {result['text']}")
