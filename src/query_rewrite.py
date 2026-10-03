"""Optional, low-cost LLM query rewriting for retrieval."""

from __future__ import annotations

import os
from typing import Any, Callable

from config import load_settings


_REWRITE_PROMPT = """Rewrite the search query for a document retrieval system.
Make it concise, specific, and self-contained. Fix obvious typos and expand
common abbreviations. If conversation history is provided, resolve pronouns
and references using only that history. Return only the rewritten query,
without quotes or explanation.

Conversation history:
{history}

Original query:
{query}
"""


def _call_llm(prompt: str) -> str:
    from openai import OpenAI

    model = os.getenv("QUERY_REWRITE_MODEL", "gpt-4o-mini")
    response = OpenAI().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You rewrite search queries only."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=64,
        temperature=0,
    )
    return response.choices[0].message.content or ""


def rewrite_query(
    query: str,
    conversation_history: list[dict[str, Any]] | str | None = None,
    llm_call: Callable[[str], str] | None = None,
) -> str:
    """Rewrite a query unless USE_QUERY_REWRITE disables the extra LLM call."""
    if os.getenv("USE_QUERY_REWRITE", "true").strip().lower() in {"0", "false", "no", "off"}:
        return query
    if not query.strip():
        return query

    history = conversation_history or "(none)"
    if isinstance(history, list):
        history = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}" if isinstance(item, dict) else str(item)
            for item in history
        )
    prompt = _REWRITE_PROMPT.format(history=history, query=query)
    rewritten = (llm_call or _call_llm)(prompt).strip()
    return rewritten or query
