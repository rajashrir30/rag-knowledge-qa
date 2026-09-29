import pytest

chromadb = pytest.importorskip("chromadb")

from src.vector_store import VectorStore, chunk_id_for


def test_chunk_id_is_deterministic():
    chunk = {"source": "example.txt", "chunk_id": 2, "text": "hello"}
    assert chunk_id_for(chunk) == chunk_id_for(dict(chunk))


def test_add_chunks_upserts_without_duplicates(tmp_path):
    store = VectorStore(path=tmp_path, collection_name="test_collection")
    chunks = [{"text": "hello world", "chunk_id": 0, "source": "example.txt"}]
    embedding = [[0.1, 0.2, 0.3]]

    store.add_chunks(chunks, embedding)
    store.add_chunks(chunks, embedding)

    assert store.count() == 1
    result = store.collection.get(include=["documents", "metadatas"])
    assert result["documents"] == ["hello world"]
    assert result["metadatas"][0]["source"] == "example.txt"


def test_subtitle_metadata_is_stored(tmp_path):
    store = VectorStore(path=tmp_path, collection_name="subtitle_collection")
    chunks = [{"text": "hello", "chunk_id": 0, "source": "example.srt", "start": 1.5, "end": 3.0}]
    store.add_chunks(chunks, [[0.1, 0.2]])

    metadata = store.collection.get(include=["metadatas"])["metadatas"][0]
    assert metadata["start"] == 1.5
    assert metadata["end"] == 3.0
