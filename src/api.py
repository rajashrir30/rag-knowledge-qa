"""FastAPI service for querying and indexing the RAG pipeline."""

from __future__ import annotations

import logging
import os
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any

from fastapi import File, FastAPI, Header, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.generation import get_generator
from src.index import index_file
from src.retrieval import hybrid_retrieve
from src.vector_store import VectorStore


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=3, ge=1, le=10)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    rewrite: bool = True
    sources: list[str] = Field(default_factory=list)


class IndexRequest(BaseModel):
    path: str = Field(..., min_length=1)


app = FastAPI(title="RAG Knowledge Base QA API")
logger = logging.getLogger(__name__)
_DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:8501",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8501",
]
_frontend_url = os.getenv("FRONTEND_URL", "").strip().rstrip("/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_DEFAULT_ORIGINS + ([_frontend_url] if _frontend_url else []),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_api_access(x_api_key: str | None) -> None:
    """Require a shared token when API_ACCESS_TOKEN is configured."""
    expected = os.getenv("API_ACCESS_TOKEN", "").strip()
    if expected and (
        not x_api_key or not secrets.compare_digest(x_api_key, expected)
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing API access token.")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Any, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": "Invalid request: check the supplied fields."})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    try:
        _require_api_access(x_api_key)
        total_started = time.perf_counter()
        question = request.query.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Query must not be empty.")
        retrieval_started = time.perf_counter()
        retrieval_options: dict[str, Any] = {
            "k": request.k,
            "alpha": request.alpha,
            "rewrite": request.rewrite,
        }
        if request.sources:
            retrieval_options["sources"] = request.sources
        hybrid = hybrid_retrieve(question, **retrieval_options)
        retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
        chunks = hybrid["results"]
        generation_started = time.perf_counter()
        result = get_generator().generate(question, chunks)
        generation_ms = (time.perf_counter() - generation_started) * 1000
        # Add chunk text here for the UI's expandable source panel by resolving
        # the numbered citation markers against the hybrid result set.
        if result.get("citations"):
            for citation in result["citations"]:
                marker = str(citation.get("marker", ""))
                try:
                    chunk = chunks[int(marker.strip("[]")) - 1]
                except (ValueError, IndexError):
                    continue
                citation["text"] = chunk.get("text", "")
        result["query"] = hybrid["query"]
        result["rewritten_query"] = hybrid["rewritten_query"]
        total_ms = (time.perf_counter() - total_started) * 1000
        result["timing_ms"] = {
            "retrieval": round(retrieval_ms, 2),
            "generation": round(generation_ms, 2),
            "total": round(total_ms, 2),
        }
        logger.info(
            "query timings: retrieval=%.2fms generation=%.2fms total=%.2fms chunks=%d",
            retrieval_ms, generation_ms, total_ms, len(chunks),
        )
        return result
    except HTTPException:
        raise
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to process the query.") from exc


def _supported_files(path: Path) -> list[Path]:
    supported = {".txt", ".pdf", ".srt", ".vtt"}
    if path.is_file():
        return [path] if path.suffix.lower() in supported else []
    return sorted(file for file in path.rglob("*") if file.is_file() and file.suffix.lower() in supported)


@app.post("/index")
def index(request: IndexRequest, x_api_key: str | None = Header(default=None)) -> dict[str, int]:
    _require_api_access(x_api_key)
    path = Path(request.path).expanduser()
    if not path.exists():
        raise HTTPException(status_code=400, detail="The supplied path does not exist.")
    files = _supported_files(path)
    if not files:
        raise HTTPException(status_code=400, detail="No supported TXT, PDF, SRT, or VTT files found.")

    try:
        chunks_added = sum(index_file(file) for file in files)
        return {"files_processed": len(files), "chunks_added": chunks_added}
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to index the supplied path.") from exc


@app.post("/index/upload")
def index_upload(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> dict[str, int]:
    """Index one uploaded document without relying on a shared filesystem."""
    _require_api_access(x_api_key)
    filename = Path(file.filename or "").name
    supported = {".txt", ".pdf", ".srt", ".vtt"}
    if not filename or Path(filename).suffix.lower() not in supported:
        raise HTTPException(status_code=400, detail="Upload a supported TXT, PDF, SRT, or VTT file.")

    try:
        max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
        contents = file.file.read(max_upload_bytes + 1)
        if len(contents) > max_upload_bytes:
            raise HTTPException(status_code=413, detail="Uploaded file is too large (maximum 25 MB).")
        with tempfile.TemporaryDirectory(prefix="rag-upload-") as temp_dir:
            temporary_path = Path(temp_dir) / filename
            temporary_path.write_bytes(contents)
            chunks_added = index_file(temporary_path)
        return {"files_processed": 1, "chunks_added": chunks_added}
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to index the uploaded file.") from exc


@app.get("/sources")
def sources(x_api_key: str | None = Header(default=None)) -> list[str]:
    try:
        _require_api_access(x_api_key)
        return VectorStore().list_sources()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to list indexed sources.") from exc
