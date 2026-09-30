"""FastAPI service for querying and indexing the RAG pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.generation import answer_question
from src.index import index_file
from src.retrieval import retrieve
from src.vector_store import VectorStore


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=10)


class IndexRequest(BaseModel):
    path: str = Field(..., min_length=1)


app = FastAPI(title="RAG Knowledge Base QA API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:8501",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Any, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": "Invalid request: check the supplied fields."})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query(request: QueryRequest) -> dict[str, Any]:
    try:
        result = answer_question(request.query.strip(), k=request.k)
        # answer_question remains unchanged; add chunk text here for the UI's
        # expandable source panel by resolving the numbered citation markers.
        if result.get("citations"):
            retrieved_chunks = retrieve(request.query.strip(), k=request.k)
            for citation in result["citations"]:
                marker = str(citation.get("marker", ""))
                try:
                    chunk = retrieved_chunks[int(marker.strip("[]")) - 1]
                except (ValueError, IndexError):
                    continue
                citation["text"] = chunk.get("text", "")
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
def index(request: IndexRequest) -> dict[str, int]:
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


@app.get("/sources")
def sources() -> list[str]:
    try:
        return VectorStore().list_sources()
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Unable to list indexed sources.") from exc
