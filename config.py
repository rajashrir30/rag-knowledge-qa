"""Application settings loaded from environment variables and .env."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    embedding_provider: str = "local"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    generation_provider: str = "ollama"
    generation_model: str = "llama3.1:8b"
    chroma_path: str = "./chroma_db"
    collection_name: str = "rag_knowledge"


def load_settings() -> Settings:
    load_dotenv()
    provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    if provider not in {"openai", "local"}:
        raise ValueError("EMBEDDING_PROVIDER must be 'openai' or 'local'")
    default_model = "text-embedding-3-small" if provider == "openai" else "sentence-transformers/all-MiniLM-L6-v2"
    generation_provider = os.getenv("GENERATION_PROVIDER", "ollama").strip().lower()
    if generation_provider not in {"openai", "anthropic", "ollama"}:
        raise ValueError("GENERATION_PROVIDER must be 'openai', 'anthropic', or 'ollama'")
    default_generation_model = {
        "openai": "gpt-4o-mini",
        "anthropic": "claude-sonnet-4-6",
        "ollama": "llama3.1:8b",
    }[generation_provider]
    generation_model_env = "OLLAMA_MODEL" if generation_provider == "ollama" else "GENERATION_MODEL"
    return Settings(
        embedding_provider=provider,
        embedding_model=os.getenv("EMBEDDING_MODEL", default_model).strip(),
        generation_provider=generation_provider,
        generation_model=os.getenv(
            generation_model_env,
            os.getenv("GENERATION_MODEL", default_generation_model),
        ).strip(),
        chroma_path=os.getenv("CHROMA_PATH", "./chroma_db").strip(),
        collection_name=os.getenv("COLLECTION_NAME", "rag_knowledge").strip(),
    )


settings = load_settings()
