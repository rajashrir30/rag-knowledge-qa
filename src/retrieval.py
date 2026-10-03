"""Nearest-neighbor retrieval over the persistent Chroma vector store."""

from __future__ import annotations

import argparse
import math
import os
from typing import Any

from src.embeddings import Embedder, get_embedder
from src.keyword_search import keyword_search
from src.query_rewrite import rewrite_query
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
        candidates = _vector_search(query, k, self.store, self.embedder)
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


def _vector_search(query: str, k: int, store: VectorStore, embedder: Embedder) -> list[dict[str, Any]]:
    query_embedding = embedder.embed_texts([query])[0]
    result = store.collection.query(
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
    return candidates


def retrieve(
    query: str,
    k: int = 5,
    min_score: float = 0.0,
    *,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the most relevant stored chunks for a query."""
    if not query or k <= 0:
        return []
    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score must be between 0 and 1")
    store = store or VectorStore()
    embedder = embedder or get_embedder()
    candidates = _vector_search(query, k, store, embedder)
    return [candidate for candidate in candidates if candidate["score"] >= min_score][:k]


def _normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    minimum, maximum = min(scores), max(scores)
    if maximum == minimum:
        return [1.0] * len(scores)
    return [(score - minimum) / (maximum - minimum) for score in scores]


def hybrid_retrieve(
    query: str,
    k: int = 5,
    alpha: float = 0.5,
    rewrite: bool = True,
    *,
    store: VectorStore | None = None,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Blend vector and BM25 results, optionally using a rewritten query."""
    if k <= 0:
        return {"query": query, "rewritten_query": query, "results": []}
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    store = store or VectorStore()
    embedder = embedder or get_embedder()
    rewritten = rewrite_query(query) if rewrite else query
    vector_results = _vector_search(rewritten, k, store, embedder)

    if alpha == 1.0:
        return {"query": query, "rewritten_query": rewritten, "results": vector_results[:k]}

    from src.keyword_search import _load_chunks

    chunks = _load_chunks(store)
    keyword_results = keyword_search(rewritten, k=k, chunks=chunks, store=store)
    if alpha == 0.0:
        results = []
        for item in keyword_results:
            result = dict(item)
            result["score"] = _normalize_scores([entry["bm25_score"] for entry in keyword_results])[len(results)]
            results.append(result)
        return {"query": query, "rewritten_query": rewritten, "results": results[:k]}

    vector_norm = _normalize_scores([item["score"] for item in vector_results])
    keyword_norm = _normalize_scores([item["bm25_score"] for item in keyword_results])
    combined: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(vector_results):
        result = dict(item)
        result["vector_score"] = item["score"]
        result["_vector_norm"] = vector_norm[index]
        combined[str(item.get("chunk_id"))] = result
    for index, item in enumerate(keyword_results):
        key = str(item.get("chunk_id"))
        if key not in combined:
            combined[key] = dict(item)
        result = combined[key]
        result["bm25_score"] = item["bm25_score"]
        result["_keyword_norm"] = keyword_norm[index]

    results = []
    for result in combined.values():
        result["score"] = alpha * result.get("_vector_norm", 0.0) + (1.0 - alpha) * result.get("_keyword_norm", 0.0)
        result.pop("_vector_norm", None)
        result.pop("_keyword_norm", None)
        results.append(result)
    results.sort(key=lambda item: item["score"], reverse=True)
    return {"query": query, "rewritten_query": rewritten, "results": results[:k]}


def retrieve_legacy_default(query: str, k: int = 5, min_score: float = 0.0) -> list[dict[str, Any]]:
    """Compatibility helper retained for callers that use the old singleton."""
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
