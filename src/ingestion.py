"""Load and normalize text, PDF, SRT, and simple WebVTT documents."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union


SubtitleEntry = dict[str, float | str]
Document = Union[str, list[SubtitleEntry]]

_TAG_RE = re.compile(r"<[^>]+>|\{\\[^}]*\}")
_WHITESPACE_RE = re.compile(r"\s+")
_TIMESTAMP_RE = re.compile(
    r"(?P<hours>\d{1,2}):(?P<minutes>\d{2}):(?P<seconds>\d{2})[,.](?P<millis>\d{3})"
)


def _clean_text(text: str) -> str:
    """Remove common subtitle markup and normalize whitespace."""
    text = _TAG_RE.sub("", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _timestamp_to_seconds(value: str) -> float:
    match = _TIMESTAMP_RE.search(value.strip())
    if not match:
        raise ValueError(f"Invalid subtitle timestamp: {value!r}")
    return (
        int(match.group("hours")) * 3600
        + int(match.group("minutes")) * 60
        + int(match.group("seconds"))
        + int(match.group("millis")) / 1000
    )


def load_text(path: str | Path) -> str:
    """Load a UTF-8 plain-text document and normalize its whitespace."""
    return _clean_text(Path(path).read_text(encoding="utf-8"))


def load_pdf(path: str | Path) -> str:
    """Extract and normalize text from all pages of a PDF."""
    from pypdf import PdfReader

    pages = []
    for page in PdfReader(str(path)).pages:
        pages.append(page.extract_text() or "")
    return _clean_text("\n".join(pages))


def load_srt(path: str | Path) -> list[SubtitleEntry]:
    """Load SRT or simple WebVTT cues as timed, cleaned subtitle entries."""
    raw = Path(path).read_text(encoding="utf-8-sig")
    blocks = re.split(r"\n\s*\n", raw.replace("\r\n", "\n").replace("\r", "\n"))
    entries: list[SubtitleEntry] = []

    for block in blocks:
        lines = [line.strip() for line in block.splitlines()]
        lines = [line for line in lines if line]
        if not lines or lines[0].upper() == "WEBVTT":
            continue
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        timing = lines[timing_index].split("-->", 1)
        start = _timestamp_to_seconds(timing[0])
        end = _timestamp_to_seconds(timing[1].split()[0])
        text = _clean_text(" ".join(lines[timing_index + 1 :]))
        if text:
            entries.append({"start": start, "end": end, "text": text})
    return entries


def load_document(path: str | Path) -> Document:
    """Dispatch document loading based on the file extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".txt":
        return load_text(path)
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix in {".srt", ".vtt"}:
        return load_srt(path)
    raise ValueError(f"Unsupported document type: {suffix or '<none>'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load a document and print its contents.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(load_document(args.path))
