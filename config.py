"""Application settings loaded from environment variables and .env."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    chroma_path: str = "./chroma_db"
    collection_name: str = "rag_knowledge"


def load_settings() -> Settings:
    load_dotenv()
    provider = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
    if provider not in {"openai", "local"}:
        raise ValueError("EMBEDDING_PROVIDER must be 'openai' or 'local'")
    default_model = "text-embedding-3-small" if provider == "openai" else "sentence-transformers/all-MiniLM-L6-v2"
    return Settings(
        embedding_provider=provider,
        embedding_model=os.getenv("EMBEDDING_MODEL", default_model).strip(),
        chroma_path=os.getenv("CHROMA_PATH", "./chroma_db").strip(),
        collection_name=os.getenv("COLLECTION_NAME", "rag_knowledge").strip(),
    )


settings = load_settings()
