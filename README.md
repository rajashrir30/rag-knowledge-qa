# RAG Knowledge Base QA

A Retrieval-Augmented Generation (RAG) system for answering questions from custom text and video subtitle datasets with grounded answers and source citations.

**Status:** In Progress - ingestion and chunking implemented

## Architecture

Ingestion -> Chunking -> Embedding -> Retrieval -> Generation -> API

## Tech Stack
  
- Python
- ChromaDB
- OpenAI API
- FastAPI
- Uvicorn
- Tiktoken

## Implemented

- Document ingestion
- Text chunking
- Token-aware chunking with configurable overlap
- SRT and simple WebVTT subtitle parsing
- Subtitle timestamp preservation
- PDF text extraction with `pypdf`

## Planned Features

- Embedding generation
- Vector database storage
- Semantic search
- RAG-based question answering
- Source citations
- REST API

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install pytest
```

Copy `.env.example` to `.env` and add your OpenAI API key when the generation layer is added. The current ingestion and chunking modules do not call the OpenAI API.

## Usage

Load a document from Python:

```python
from src.ingestion import load_document
from src.chunking import chunk_subtitles, chunk_text

document = load_document("data/sample/example.txt")
chunks = chunk_text(document, chunk_size=500, overlap=50, source="example.txt")
print(chunks)

subtitles = load_document("data/sample/example.srt")
chunks = chunk_subtitles(subtitles, max_tokens=300, source="example.srt")
print(chunks)
```

The loaders support `.txt`, `.pdf`, `.srt`, and simple `.vtt` files. Text chunks include `text`, `chunk_id`, and `source`; subtitle chunks also include `start` and `end` timestamps in seconds.

Each module can also be run directly:

```bash
python -m src.ingestion data/sample/example.srt
python -m src.chunking data/sample/example.txt --chunk-size 100 --overlap 10
```

## Tests

Run the chunking tests with:

```bash
pytest
```
