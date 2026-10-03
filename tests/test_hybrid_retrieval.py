from src.retrieval import hybrid_retrieve


class FakeEmbedder:
    def embed_texts(self, texts):
        return [[1.0, 0.0] for _ in texts]


class FakeCollection:
    def __init__(self):
        self.chunks = [
            {"text": "semantic context", "source": "semantic.txt", "chunk_id": 0},
            {"text": "exact pollution keyword", "source": "keyword.txt", "chunk_id": 1},
        ]

    def query(self, query_embeddings, n_results, include):
        return {
            "documents": [[self.chunks[0]["text"], self.chunks[1]["text"]][:n_results]],
            "metadatas": [[self.chunks[0], self.chunks[1]][:n_results]],
            "distances": [[0.0, 0.8][:n_results]],
        }

    def get(self, include):
        return {
            "documents": [chunk["text"] for chunk in self.chunks],
            "metadatas": self.chunks,
        }


class FakeStore:
    path = "."
    collection = FakeCollection()


def fake_keyword_search(query, k, **kwargs):
    return [
        {"text": "exact pollution keyword", "source": "keyword.txt", "chunk_id": 1, "start": None, "end": None, "bm25_score": 10.0},
        {"text": "semantic context", "source": "semantic.txt", "chunk_id": 0, "start": None, "end": None, "bm25_score": 1.0},
    ][:k]


def test_hybrid_combines_signals_and_surfaces_keyword_match(monkeypatch):
    monkeypatch.setattr("src.retrieval.keyword_search", fake_keyword_search)
    result = hybrid_retrieve("pollution", k=2, alpha=0.1, rewrite=False, store=FakeStore(), embedder=FakeEmbedder())
    assert result["results"][0]["source"] == "keyword.txt"


def test_hybrid_deduplicates_by_chunk_id(monkeypatch):
    monkeypatch.setattr("src.retrieval.keyword_search", fake_keyword_search)
    result = hybrid_retrieve("pollution", k=2, alpha=0.5, rewrite=False, store=FakeStore(), embedder=FakeEmbedder())
    assert len({item["chunk_id"] for item in result["results"]}) == len(result["results"])


def test_alpha_extremes_match_single_signal_search(monkeypatch):
    monkeypatch.setattr("src.retrieval.keyword_search", fake_keyword_search)
    store = FakeStore()
    vector = hybrid_retrieve("pollution", k=2, alpha=1.0, rewrite=False, store=store, embedder=FakeEmbedder())
    keyword = hybrid_retrieve("pollution", k=2, alpha=0.0, rewrite=False, store=store, embedder=FakeEmbedder())
    assert [item["source"] for item in vector["results"]] == ["semantic.txt", "keyword.txt"]
    assert [item["source"] for item in keyword["results"]] == ["keyword.txt", "semantic.txt"]
