"""Grounded answer generation using OpenAI or Anthropic models."""

from __future__ import annotations

import argparse
import os
import re
from abc import ABC, abstractmethod
from typing import Any

from config import Settings, load_settings
from src.prompts import FALLBACK_PHRASE, SYSTEM_PROMPT, build_prompt


_CITATION_RE = re.compile(r"\[(\d+)\]")


def _result_from_answer(answer: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Map valid answer markers back to their retrieved source chunks."""
    citations = []
    seen: set[int] = set()
    for marker in _CITATION_RE.findall(answer):
        index = int(marker)
        if index < 1 or index > len(chunks) or index in seen:
            continue
        seen.add(index)
        chunk = chunks[index - 1]
        citation = {
            "marker": f"[{index}]",
            "source": chunk.get("source", ""),
            "chunk_id": chunk.get("chunk_id"),
            "start": chunk.get("start"),
            "end": chunk.get("end"),
        }
        citations.append(citation)
    return {
        "answer": answer,
        "citations": citations,
        "used_fallback": FALLBACK_PHRASE in answer,
    }


class Generator(ABC):
    """Interface for grounded answer generators."""

    @abstractmethod
    def generate(self, query: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        raise NotImplementedError


class OpenAIGenerator(Generator):
    """Generate grounded answers with the OpenAI Chat Completions API."""

    def __init__(self, model: str | None = None, client: Any = None):
        from openai import OpenAI

        self.model = model or os.getenv("GENERATION_MODEL", "gpt-4o-mini")
        self.client = client or OpenAI()

    def generate(self, query: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(query, chunks)},
            ],
        )
        answer = response.choices[0].message.content or FALLBACK_PHRASE
        return _result_from_answer(answer, chunks)


class AnthropicGenerator(Generator):
    """Generate grounded answers with Anthropic's Messages API."""

    def __init__(self, model: str | None = None, client: Any = None):
        from anthropic import Anthropic

        self.model = model or os.getenv("GENERATION_MODEL", "claude-sonnet-4-6")
        self.client = client or Anthropic()

    def generate(self, query: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=int(os.getenv("GENERATION_MAX_TOKENS", "1000")),
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(query, chunks)}],
        )
        answer = "".join(getattr(block, "text", "") for block in response.content).strip() or FALLBACK_PHRASE
        return _result_from_answer(answer, chunks)


def get_generator(config: Settings | None = None) -> Generator:
    """Create the provider configured by GENERATION_PROVIDER."""
    config = config or load_settings()
    provider = config.generation_provider
    if provider == "openai":
        return OpenAIGenerator(model=config.generation_model)
    if provider == "anthropic":
        return AnthropicGenerator(model=config.generation_model)
    raise ValueError("GENERATION_PROVIDER must be 'openai' or 'anthropic'")


def answer_question(query: str, k: int = 5) -> dict[str, Any]:
    """Retrieve context and generate one grounded answer."""
    # Keep provider imports lazy so prompt formatting and fake generators do
    # not require Chroma or an embedding/API client just to be imported.
    from src.retrieval import retrieve

    chunks = retrieve(query, k=k)
    return get_generator().generate(query, chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Answer a question using retrieved context.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    result = answer_question(args.query, k=args.k)
    print(result["answer"])
    print("\nSources:")
    if not result["citations"]:
        print("None")
    for citation in result["citations"]:
        location = ""
        if citation["start"] is not None:
            location = f" ({citation['start']}s - {citation['end']}s)"
        print(f"{citation['marker']} {citation['source']}{location}")
