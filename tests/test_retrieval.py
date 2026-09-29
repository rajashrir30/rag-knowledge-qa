import pytest

pytest.importorskip("chromadb")

from src.retrieval import Retriever
from src.vector_store import VectorStore


class FakeEmbedder:
    vectors = {
        "cats": [1.0, 0.0, 0.0],
        "dogs": [0.0, 1.0, 0.0],
        "birds": [0.0, 0.0, 1.0],
    }

    def embed_texts(self, texts):
        output = []
        for text in texts:
            vector = [0.0, 0.0, 0.0]
            for keyword, keyword_vector in self.vectors.items():
                if keyword in text.lower():
                    vector = keyword_vector
                    break
            output.append(vector)
        return output


@pytest.fixture
def seeded_retriever(tmp_path):
    store = VectorStore(path=tmp_path, collection_name="retrieval_test")
    chunks = [
        {"text": "Cats are curious and independent animals.", "source": "cats.txt", "chunk_id": 0},
        {"text": "Dogs are loyal companions.", "source": "dogs.txt", "chunk_id": 0},
        {"text": "Birds migrate across long distances.", "source": "birds.txt", "chunk_id": 0},
    ]
    store.add_chunks(chunks, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return Retriever(store=store, embedder=FakeEmbedder(), use_reranker=False)


def test_matching_chunk_is_ranked_first(seeded_retriever):
    results = seeded_retriever.retrieve("cats", k=3)
    assert results[0]["source"] == "cats.txt"
    assert results[0]["score"] > results[1]["score"]


def test_min_score_filters_low_relevance_results(seeded_retriever):
    results = seeded_retriever.retrieve("cats", k=3, min_score=0.6)
    assert [result["source"] for result in results] == ["cats.txt"]


def test_k_limits_results(seeded_retriever):
    assert len(seeded_retriever.retrieve("cats", k=2)) == 2
