from fastapi.testclient import TestClient

from src import api


client = TestClient(api.app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_returns_generation_shape(monkeypatch):
    expected = {
        "answer": "The answer is supported [1].",
        "citations": [{"marker": "[1]", "source": "example.txt", "chunk_id": 0, "start": None, "end": None}],
        "used_fallback": False,
    }
    monkeypatch.setattr(api, "answer_question", lambda query, k: expected)
    monkeypatch.setattr(api, "retrieve", lambda query, k: [])

    response = client.post("/query", json={"query": "What is the answer?", "k": 5})
    assert response.status_code == 200
    assert response.json() == expected


def test_query_missing_query_returns_400():
    response = client.post("/query", json={"k": 5})
    assert response.status_code == 400


def test_sources_returns_list(monkeypatch):
    class FakeStore:
        def list_sources(self):
            return ["example.txt", "example.srt"]

    monkeypatch.setattr(api, "VectorStore", lambda: FakeStore())
    response = client.get("/sources")
    assert response.status_code == 200
    assert response.json() == ["example.txt", "example.srt"]
