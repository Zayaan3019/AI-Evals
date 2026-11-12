"""Groq judge client, the record/replay cassette, and the benchmark agent's retriever.

No network: the Groq tests replace ``httpx.post`` and assert on the request the
client builds and on how it reacts to the responses a real API returns
(success, 429 with Retry-After, a hard error).
"""

from __future__ import annotations

import json

import httpx
import pytest

from agentlens_platform.evaluators import judges
from agentlens_platform.evaluators.judges import (
    GroqJudge,
    JudgeResponse,
    JudgeUnavailable,
    ReplayJudge,
    build_judge,
)


def _ok(text: str = '{"score": 1}', model: str = "m") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": model,
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        },
    )


def test_groq_sends_minimal_reasoning_for_reasoning_models(monkeypatch):
    seen = []
    monkeypatch.setattr(judges.httpx, "post", lambda url, **kw: seen.append(kw["json"]) or _ok())
    GroqJudge("openai/gpt-oss-20b", api_key="k").complete("p", system="s", max_tokens=5, temperature=0)
    GroqJudge("qwen/qwen3.8-27b", api_key="k").complete("p", system="s", max_tokens=5, temperature=0)
    GroqJudge("some-other-model", api_key="k").complete("p", system="s", max_tokens=5, temperature=0)
    assert seen[0]["reasoning_effort"] == "low"
    assert seen[1]["reasoning_effort"] == "none"
    assert "reasoning_effort" not in seen[2]


def test_groq_reports_provider_token_counts(monkeypatch):
    monkeypatch.setattr(judges.httpx, "post", lambda url, **kw: _ok("hi", "served-model"))
    r = GroqJudge("m", api_key="k").complete("p", system="s", max_tokens=5, temperature=0)
    assert (r.text, r.input_tokens, r.output_tokens, r.model) == ("hi", 11, 3, "served-model")


def test_groq_honours_retry_after_then_succeeds(monkeypatch):
    responses = [httpx.Response(429, headers={"retry-after": "2"}), _ok()]
    sleeps: list[float] = []
    monkeypatch.setattr(judges.httpx, "post", lambda url, **kw: responses.pop(0))
    monkeypatch.setattr(judges.time, "sleep", sleeps.append)
    GroqJudge("m", api_key="k").complete("p", system="s", max_tokens=5, temperature=0)
    assert sleeps == [2.0]


def test_groq_hard_error_is_unavailable_not_a_score(monkeypatch):
    monkeypatch.setattr(judges.httpx, "post", lambda url, **kw: httpx.Response(404, text="gone"))
    with pytest.raises(JudgeUnavailable):
        GroqJudge("m", api_key="k").complete("p", system="s", max_tokens=5, temperature=0)


def test_groq_without_a_key_is_unavailable(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(JudgeUnavailable):
        GroqJudge("m").complete("p", system="s", max_tokens=5, temperature=0)


class _Live:
    model = "live"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, prompt, *, system, max_tokens, temperature):
        self.calls += 1
        return JudgeResponse(text=f"answer to {prompt}", input_tokens=7, output_tokens=2, latency_ms=12.5, model="live")


def test_replay_records_once_and_replays_exactly(tmp_path):
    path = tmp_path / "c.json"
    live = _Live()
    first = ReplayJudge("m", str(path), live=live).complete("p", system="s", max_tokens=9, temperature=0)
    replayed = ReplayJudge("m", str(path)).complete("p", system="s", max_tokens=9, temperature=0)
    assert live.calls == 1
    assert (replayed.text, replayed.input_tokens, replayed.output_tokens, replayed.latency_ms) == (
        first.text, 7, 2, 12.5,
    )


@pytest.mark.parametrize(
    "change",
    [
        {"prompt": "other"},
        {"system": "other"},
        {"max_tokens": 10},
        {"temperature": 0.5},
    ],
)
def test_replay_never_serves_a_response_recorded_for_a_different_request(tmp_path, change):
    path = tmp_path / "c.json"
    ReplayJudge("m", str(path), live=_Live()).complete("p", system="s", max_tokens=9, temperature=0)
    request = {"prompt": "p", "system": "s", "max_tokens": 9, "temperature": 0, **change}
    prompt = request.pop("prompt")
    with pytest.raises(JudgeUnavailable):
        ReplayJudge("m", str(path)).complete(prompt, **request)


def test_replay_is_keyed_by_model(tmp_path):
    path = tmp_path / "c.json"
    ReplayJudge("model-a", str(path), live=_Live()).complete("p", system="s", max_tokens=9, temperature=0)
    with pytest.raises(JudgeUnavailable):
        ReplayJudge("model-b", str(path)).complete("p", system="s", max_tokens=9, temperature=0)


def test_two_recorders_sharing_a_cassette_keep_each_others_entries(tmp_path):
    path = tmp_path / "c.json"
    a = ReplayJudge("a", str(path), live=_Live())
    b = ReplayJudge("b", str(path), live=_Live())
    a.complete("p", system="s", max_tokens=9, temperature=0)
    b.complete("p", system="s", max_tokens=9, temperature=0)
    entries = json.loads(path.read_text(encoding="utf-8"))["entries"]
    assert len(entries) == 2


def test_build_judge_wraps_in_replay_only_when_configured(tmp_path, monkeypatch):
    monkeypatch.delenv("AGENTLENS_JUDGE_CASSETTE", raising=False)
    assert isinstance(build_judge("groq", "m"), GroqJudge)
    monkeypatch.setenv("AGENTLENS_JUDGE_CASSETTE", str(tmp_path / "c.json"))
    assert isinstance(build_judge("groq", "m"), ReplayJudge)


# -- benchmark agent retrieval ------------------------------------------------


def test_bm25_finds_the_gold_paragraph_for_most_questions():
    from agentlens_platform.examples.squad_agent import load_data, retriever

    data = load_data()
    hits = sum(
        any(p.id == q["paragraph_id"] for p, _ in retriever().search(q["question"], 2))
        for q in data["questions"]
    )
    # A floor, not a target: BM25 on SQuAD questions is strong but not perfect,
    # and a regression in tokenisation or scoring shows up here first.
    assert hits >= 75, f"BM25 top-2 recall fell to {hits}/100"


def test_bm25_ranking_is_deterministic():
    from agentlens_platform.examples.squad_agent import retriever

    q = "What is the capital of the region?"
    assert [p.id for p, _ in retriever().search(q, 5)] == [p.id for p, _ in retriever().search(q, 5)]


def test_answer_cleaning_strips_reasoning_and_wrapping():
    from agentlens_platform.examples.squad_agent import clean_answer

    assert clean_answer("<think>hmm</think> \"France.\"") == "France"
    assert clean_answer("Denver Broncos") == "Denver Broncos"
