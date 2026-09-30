"""Prompt construction for grounded question answering."""

from __future__ import annotations

from typing import Any


FALLBACK_PHRASE = "I don't have enough information to answer that."

SYSTEM_PROMPT = f"""You answer questions using only the provided context chunks.

Rules:
- Answer only with information supported by the context chunks.
- Cite the chunk or chunks supporting each factual claim using numbered markers such as [1] or [2].
- If the context does not contain the answer, respond exactly with: "{FALLBACK_PHRASE}"
- Never make up facts, details, or citations that are not present in the context.
- Do not cite a chunk unless it supports the claim.
"""


def build_prompt(query: str, chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks and a question into the user prompt."""
    sections = ["Context chunks:"]
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.get("source") or "unknown source"
        location = ""
        if chunk.get("start") is not None or chunk.get("end") is not None:
            location = f" | Timestamp: {chunk.get('start', '?')}s - {chunk.get('end', '?')}s"
        sections.append(f"[{index}] Source: {source}{location}\n{chunk.get('text', '')}")
    sections.append(f"\nQuestion: {query}")
    return "\n\n".join(sections)
