from src.chunking import _ENCODING, chunk_subtitles, chunk_text


def test_chunks_never_exceed_token_limit():
    chunks = chunk_text("word " * 250, chunk_size=32, overlap=4)
    assert chunks
    assert all(len(_ENCODING.encode(chunk["text"])) <= 32 for chunk in chunks)


def test_overlap_appears_at_next_chunk_start():
    chunks = chunk_text(" ".join(f"token{i}" for i in range(100)), chunk_size=20, overlap=5)
    for left, right in zip(chunks, chunks[1:]):
        left_tokens = _ENCODING.encode(left["text"])
        right_tokens = _ENCODING.encode(right["text"])
        assert right_tokens[:5] == left_tokens[-5:]


def test_subtitle_timestamps_are_preserved():
    entries = [
        {"start": 1.5, "end": 3.0, "text": "First line."},
        {"start": 3.0, "end": 5.75, "text": "Second line."},
        {"start": 5.75, "end": 9.0, "text": "Third line."},
    ]
    chunks = chunk_subtitles(entries, max_tokens=100)
    assert len(chunks) == 1
    assert chunks[0]["start"] == 1.5
    assert chunks[0]["end"] == 9.0


def test_empty_input_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_subtitles([]) == []
