"""Embedding providers used by the indexing pipeline."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from config import Settings, load_settings


class Embedder(ABC):
    """Interface shared by remote and local embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OpenAIEmbedder(Embedder):
    """Embed text with OpenAI's embeddings endpoint in batches."""

    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 64, max_retries: int = 3, client: Any = None):
        from openai import OpenAI

        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.client = client or OpenAI()

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

        transient_errors = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.embeddings.create(model=self.model, input=batch)
                return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
            except transient_errors:
                if attempt >= self.max_retries:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError("Embedding request failed")

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            embeddings.extend(self._embed_batch(texts[start : start + self.batch_size]))
        return embeddings


class LocalEmbedder(Embedder):
    """Embed text with sentence-transformers/all-MiniLM-L6-v2."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()


def get_embedder(config: Settings | None = None) -> Embedder:
    """Create the provider selected by EMBEDDING_PROVIDER."""
    config = config or load_settings()
    if config.embedding_provider == "openai":
        return OpenAIEmbedder(model=config.embedding_model)
    if config.embedding_provider == "local":
        return LocalEmbedder(model=config.embedding_model)
    raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")
