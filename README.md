# RAG Knowledge Base QA

A Retrieval-Augmented Generation (RAG) system for answering questions from custom text and video subtitle datasets with grounded answers and source citations.

**Status:** Core RAG pipeline, FastAPI service, and minimal Streamlit UI implemented.

## Architecture

Ingestion -> Chunking -> Embedding -> Vector Store -> Retrieval -> Generation -> API/UI

## Tech Stack

- Python
- ChromaDB
- OpenAI API
- Anthropic API (optional)
- Sentence Transformers (optional local embeddings and reranking)
- Tiktoken
- FastAPI and Uvicorn
- Streamlit

## Implemented

- Text, PDF, SRT, and simple WebVTT ingestion
- Subtitle formatting cleanup and timestamp parsing
- Token-aware text chunking with overlap
- Subtitle chunking with start/end timestamps
- OpenAI embeddings using `text-embedding-3-small`
- Local embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Persistent ChromaDB storage with deterministic IDs
- Duplicate-safe vector upserts when re-indexing files
- Similarity retrieval with score filtering and top-k limits
- Optional cross-encoder reranking
- Grounded generation with OpenAI or Anthropic
- Numbered source citations and timestamp metadata
- Exact fallback response when context is insufficient
- Unit tests for chunking, vector storage, retrieval, and generation

## Project Structure

```text
src/
  ingestion.py       # Load TXT, PDF, SRT, and VTT documents
  chunking.py        # Create token-aware chunks
  embeddings.py      # OpenAI and local embedding providers
  vector_store.py    # Persistent ChromaDB wrapper
  index.py           # Ingestion -> chunking -> embedding -> storage CLI
  retrieval.py       # Similarity search and optional reranking
  prompts.py         # Grounded generation prompt
  generation.py      # OpenAI/Anthropic answer generation
api.py               # FastAPI endpoints
app.py               # Streamlit frontend
config.py            # Environment-backed application settings
data/sample/         # Example TXT and SRT files
data/uploads/        # Streamlit-uploaded files
tests/               # Module tests
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install pytest
```

Copy `.env.example` to `.env` and configure the providers you want to use:

```env
OPENAI_API_KEY=your_api_key_here
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
GENERATION_PROVIDER=openai
GENERATION_MODEL=gpt-4o-mini
CHROMA_PATH=./chroma_db
COLLECTION_NAME=rag_knowledge
USE_RERANKER=false
```

For Anthropic generation, set `GENERATION_PROVIDER=anthropic`, `GENERATION_MODEL=claude-sonnet-4-6`, and provide `ANTHROPIC_API_KEY` in your environment. For local embeddings, set `EMBEDDING_PROVIDER=local`; this downloads the Sentence Transformers model on first use.

## Usage

### Index a document

```bash
python -m src.index data/sample/example.txt
python -m src.index data/sample/example.srt
```

The indexer loads the document, chunks it, creates embeddings, and upserts it into the persistent ChromaDB collection.

### Retrieve relevant chunks

```bash
python -m src.retrieval --query "What is the main idea?" --k 5
```

Enable optional reranking with `USE_RERANKER=true`. Reranking improves ordering in some cases but adds model-loading time, memory usage, and latency.

### Generate an answer

```bash
python -m src.generation --query "What is the main idea?" --k 5
```

Generated answers use only retrieved context. Claims are expected to include numbered citations such as `[1]` and `[2]`. When the context does not contain an answer, the generator returns:

```text
I don't have enough information to answer that.
```

### Use ingestion and chunking from Python

```python
from src.ingestion import load_document
from src.chunking import chunk_subtitles, chunk_text

document = load_document("data/sample/example.txt")
chunks = chunk_text(document, chunk_size=500, overlap=50, source="example.txt")

subtitles = load_document("data/sample/example.srt")
subtitle_chunks = chunk_subtitles(subtitles, max_tokens=300, source="example.srt")
```

## Tests

Run all tests with:

```bash
pytest
```

## API and UI

Start the FastAPI server:

```bash
uvicorn src.api:app --reload
```

In a second terminal, start the Streamlit frontend:

```bash
streamlit run app.py
```

The API provides:

- `GET /health` - health check
- `POST /query` - answer a question using the RAG pipeline
- `POST /index` - index a file or folder of supported documents
- `GET /sources` - list indexed source filenames

Example API requests:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/index -H "Content-Type: application/json" -d "{\"path\": \"data/sample/example.txt\"}"
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d "{\"query\": \"What is the main idea?\", \"k\": 5}"
```

The Streamlit UI uploads documents into `data/uploads/`, indexes them through the API, displays indexed sources, and provides a question-answering interface with cited chunks. Set `API_URL` if the API is running somewhere other than `http://localhost:8000`.
