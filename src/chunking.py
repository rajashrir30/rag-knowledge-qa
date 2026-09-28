"""Token-aware chunking for plain text and timed subtitles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import tiktoken


_ENCODING = tiktoken.get_encoding("cl100k_base")


def _validate_limits(size: int, overlap: int = 0) -> None:
    if size <= 0:
        raise ValueError("chunk size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than chunk size")


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50, source: str = "") -> list[dict[str, Any]]:
    """Split text into overlapping cl100k_base token windows."""
    _validate_limits(chunk_size, overlap)
    tokens = _ENCODING.encode(text or "")
    if not tokens:
        return []

    step = chunk_size - overlap
    chunks = []
    for chunk_id, start in enumerate(range(0, len(tokens), step)):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunks.append({"text": _ENCODING.decode(window), "chunk_id": chunk_id, "source": source})
        if start + chunk_size >= len(tokens):
            break
    return chunks


def chunk_subtitles(entries: list[dict[str, Any]], max_tokens: int = 300, source: str = "") -> list[dict[str, Any]]:
    """Combine consecutive subtitle cues without exceeding ``max_tokens``."""
    _validate_limits(max_tokens)
    if not entries:
        return []

    chunks: list[dict[str, Any]] = []
    current_tokens: list[int] = []
    current_start = current_end = None

    def flush() -> None:
        nonlocal current_tokens, current_start, current_end
        if current_tokens:
            chunks.append({
                "text": _ENCODING.decode(current_tokens),
                "start": current_start,
                "end": current_end,
                "chunk_id": len(chunks),
                "source": source,
            })
        current_tokens = []
        current_start = current_end = None

    for entry in entries:
        tokens = _ENCODING.encode(str(entry.get("text", "")))
        if not tokens:
            continue
        # A single long cue is split while retaining that cue's timestamps.
        pieces = [tokens[i : i + max_tokens] for i in range(0, len(tokens), max_tokens)]
        for piece_index, piece in enumerate(pieces):
            if current_tokens and len(current_tokens) + len(piece) > max_tokens:
                flush()
            if current_start is None:
                current_start = float(entry["start"])
            current_end = float(entry["end"])
            current_tokens.extend(piece)
            if piece_index < len(pieces) - 1:
                flush()
    flush()
    return chunks


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk a UTF-8 text file.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()
    print(chunk_text(args.path.read_text(encoding="utf-8"), args.chunk_size, args.overlap, args.path.name))
