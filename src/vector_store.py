"""Persistent ChromaDB storage for chunk documents and embeddings."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import chromadb

from config import Settings, load_settings


def chunk_id_for(chunk: dict[str, Any]) -> str:
    """Return a stable ID based on a chunk's source and chunk number."""
    source = str(chunk.get("source", ""))
    chunk_id = str(chunk.get("chunk_id", ""))
    return hashlib.sha256(f"{source}\0{chunk_id}".encode("utf-8")).hexdigest()


class VectorStore:
    """Small wrapper around a persistent Chroma collection."""

    def __init__(self, path: str | Path | None = None, collection_name: str | None = None, config: Settings | None = None):
        config = config or load_settings()
        self.path = str(path or config.chroma_path)
        self.collection_name = collection_name or config.collection_name
        self.client = chromadb.PersistentClient(path=self.path)
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def add_chunks(self, chunks: list[dict[str, Any]], embeddings: list[list[float]]) -> list[str]:
        """Upsert chunks so indexing a file repeatedly does not create duplicates."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return []

        ids = [chunk_id_for(chunk) for chunk in chunks]
        documents = [str(chunk["text"]) for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            metadata: dict[str, str | int | float] = {
                "source": str(chunk.get("source", "")),
                "chunk_id": int(chunk.get("chunk_id", 0)),
            }
            for field in ("start", "end"):
                if field in chunk and chunk[field] is not None:
                    metadata[field] = float(chunk[field])
            metadatas.append(metadata)
        self.collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        return ids

    def count(self) -> int:
        return self.collection.count()

    def list_sources(self) -> list[str]:
        """Return unique source filenames currently stored in the collection."""
        result = self.collection.get(include=["metadatas"])
        sources = {
            str(metadata["source"])
            for metadata in (result.get("metadatas") or [])
            if metadata and metadata.get("source")
        }
        return sorted(sources)
