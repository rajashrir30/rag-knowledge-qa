from src.keyword_search import build_bm25_index, keyword_search, tokenize


def test_exact_keyword_match_ranks_highly():
    chunks = [
        {"text": "A general discussion of forecasting methods.", "source": "general.txt", "chunk_id": 0},
        {"text": "The pollution level forecasting dataset contains hourly readings.", "source": "dataset.txt", "chunk_id": 1},
    ]
    results = keyword_search("pollution level forecasting dataset", k=1, chunks=chunks)
    assert results[0]["source"] == "dataset.txt"


def test_tokenization_handles_case_punctuation_and_stopwords():
    assert tokenize("The DATASET, is useful!") == ["dataset", "useful"]


def test_bm25_index_can_be_reused(tmp_path):
    chunks = [{"text": "alpha beta", "source": "a.txt", "chunk_id": 0}]
    first = build_bm25_index(chunks, cache_dir=tmp_path)
    second = build_bm25_index(chunks, cache_dir=tmp_path)
    assert first.source_hash == second.source_hash
    assert second.search("alpha", 1)[0]["source"] == "a.txt"
