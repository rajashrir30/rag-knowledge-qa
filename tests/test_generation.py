from src.generation import Generator, _result_from_answer
from src.prompts import FALLBACK_PHRASE, build_prompt


class FakeGenerator(Generator):
    def __init__(self, answer):
        self.answer = answer

    def generate(self, query, chunks):
        return _result_from_answer(self.answer, chunks)


def sample_chunks():
    return [
        {"text": "The library opened in 1901.", "source": "history.txt", "chunk_id": 0},
        {"text": "It stands beside the river.", "source": "tour.srt", "chunk_id": 3, "start": 12.5, "end": 16.0},
    ]


def test_citation_markers_map_to_source_chunks():
    result = FakeGenerator("The library opened in 1901 [1] and stands beside the river [2].").generate("Where is it?", sample_chunks())
    assert result["citations"] == [
        {"marker": "[1]", "source": "history.txt", "chunk_id": 0, "start": None, "end": None},
        {"marker": "[2]", "source": "tour.srt", "chunk_id": 3, "start": 12.5, "end": 16.0},
    ]
    assert result["used_fallback"] is False


def test_fallback_phrase_is_detected():
    result = FakeGenerator(FALLBACK_PHRASE).generate("Unknown?", sample_chunks())
    assert result["used_fallback"] is True
    assert result["citations"] == []


def test_build_prompt_numbers_sources_and_timestamps():
    prompt = build_prompt("Where is the library?", sample_chunks())
    assert "[1] Source: history.txt" in prompt
    assert "[2] Source: tour.srt | Timestamp: 12.5s - 16.0s" in prompt
    assert "Question: Where is the library?" in prompt
