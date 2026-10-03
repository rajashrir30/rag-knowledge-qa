from src.query_rewrite import rewrite_query


def test_rewrite_resolves_pronouns_with_history(monkeypatch):
    monkeypatch.setenv("USE_QUERY_REWRITE", "true")
    calls = []

    def fake_llm(prompt):
        calls.append(prompt)
        return "what is the pollution level forecasting dataset?"

    rewritten = rewrite_query(
        "What about the second one?",
        conversation_history=[{"role": "user", "content": "We discussed two datasets."}],
        llm_call=fake_llm,
    )
    assert rewritten == "what is the pollution level forecasting dataset?"
    assert calls
    assert "second one" in calls[0]


def test_rewrite_can_be_disabled(monkeypatch):
    monkeypatch.setenv("USE_QUERY_REWRITE", "false")

    def fail_if_called(prompt):
        raise AssertionError("LLM should not be called when rewriting is disabled")

    query = "What about the second one?"
    assert rewrite_query(query, llm_call=fail_if_called) == query
