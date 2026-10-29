"""CLI for ingestion -> chunking -> embedding -> persistent vector storage."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.chunking import chunk_subtitles, chunk_text
from src.embeddings import get_embedder
from src.ingestion import load_document, load_pdf_pages
from src.vector_store import VectorStore


def index_file(path: Path) -> int:
    if path.suffix.lower() == ".pdf":
        chunks = []
        for page_number, page_text in enumerate(load_pdf_pages(path), start=1):
            page_chunks = chunk_text(page_text, source=path.name)
            for chunk in page_chunks:
                chunk["page"] = page_number
                chunk["chunk_id"] = len(chunks)
                chunks.append(chunk)
    else:
        document = load_document(path)
        source = path.name
        if isinstance(document, str):
            chunks = chunk_text(document, source=source)
        else:
            chunks = chunk_subtitles(document, source=source)
    embeddings = get_embedder().embed_texts([chunk["text"] for chunk in chunks])
    store = VectorStore()
    store.add_chunks(chunks, embeddings)
    return len(chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index a text, PDF, SRT, or VTT document.")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(f"Indexed {index_file(args.path)} chunks from {args.path}")
