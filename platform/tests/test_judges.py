"""Model-graded evaluators against a real LLM.

These call a real local Ollama. Nothing is mocked, because the thing worth
testing about a judge is precisely the part a mock removes: whether a real
model, given a real rubric, actually separates a faithful answer from an
invented one, and whether its output can be parsed.

Ollama is not part of ``docker-compose.yml`` -- it is a local install with
multi-gigabyte model weights -- so unlike Postgres and ClickHouse these tests
skip when it is absent rather than erroring. The skip is loud, and the
non-network parts of the judge path (prompt rendering, JSON extraction, error
handling) are tested unconditionally below so the coverage does not disappear
along with the daemon.
"""

from __future__ import annotations

import os

import pytest

from agentlens_platform.evaluators import EvalInput, build_evaluator, run_evaluator
from agentlens_platform.evaluators.judges import (
    AnthropicJudge,
    JudgeUnavailable,
    OllamaJudge,
    build_judge,
    extract_json_object,
    ollama_available,
)

#: Smallest model on this machine, chosen so the suite stays runnable.
JUDGE_MODEL = os.environ.get("AGENTLENS_TEST_JUDGE_MODEL", "qwen2.5:3b")

requires_ollama = pytest.mark.skipif(
    not ollama_available(), reason="local Ollama is not running; judge tests need a real model"
)

CONTEXT = (
    "Kaiser Permanente evaluated Limbic, an AI assistant, for mental health "
    "triage. Therapists raised concerns about AI replacing clinical roles.",
)


# ---------------------------------------------------------------------------
# Parsing and error handling -- no network required
# ---------------------------------------------------------------------------


def test_bare_json_is_extracted() -> None:
    assert extract_json_object('{"score": 0.8, "reason": "ok"}')["score"] == 0.8


def test_json_wrapped_in_prose_and_fences_is_extracted() -> None:
    """Models wrap JSON however they like; a judge we cannot parse is no judge."""
    text = 'Here is my assessment:\n```json\n{"reason": "supported", "score": 1.0}\n```\nDone.'
    assert extract_json_object(text)["score"] == 1.0


def test_unparseable_judge_output_raises_rather_than_guessing() -> None:
    with pytest.raises(ValueError):
        extract_json_object("I think it is pretty good, maybe 8 out of 10.")


def test_missing_anthropic_key_is_unavailable_not_a_crash() -> None:
    """Absent credentials must be a clean, reportable condition."""
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        with pytest.raises(JudgeUnavailable):
            AnthropicJudge()
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved


def test_unknown_provider_is_rejected_rather_than_silently_substituted() -> None:
    """A silent fallback would change what scores mean without changing the hash."""
    with pytest.raises(JudgeUnavailable):
        build_judge("some-other-vendor", "a-model")


def test_unreachable_ollama_is_reported_as_unavailable() -> None:
    judge = OllamaJudge(JUDGE_MODEL, base_url="http://127.0.0.1:1", timeout=2.0)
    with pytest.raises(JudgeUnavailable):
        judge.complete("hi", system="s", max_tokens=8, temperature=0.0)


def test_a_dead_judge_yields_an_error_score_not_a_zero() -> None:
    """The single most important property of this tier.

    A judge that cannot be reached has said nothing about the system under
    test. Recording 0.0 would report a healthy system as broken.
    """
    ev = build_evaluator(
        "faithfulness",
        judge=OllamaJudge(JUDGE_MODEL, base_url="http://127.0.0.1:1", timeout=2.0),
    )
    score = run_evaluator(ev, EvalInput(output="anything", context=CONTEXT))
    assert score.failed_to_run
    assert score.value is None
    assert "judge unavailable" in score.error


def test_prompt_renders_context_and_output() -> None:
    ev = build_evaluator("faithfulness")
    prompt = ev.render_prompt(EvalInput(output="Limbic was deployed.", context=CONTEXT))
    assert "Limbic was deployed." in prompt
    assert "Kaiser Permanente" in prompt
    assert "[1]" in prompt


def test_prompt_states_absence_explicitly_rather_than_leaving_a_blank() -> None:
    """An empty slot reads to the model as 'nothing to check', not 'no data'."""
    ev = build_evaluator("faithfulness")
    prompt = ev.render_prompt(EvalInput(output="x"))
    assert "(no context provided)" in prompt


# ---------------------------------------------------------------------------
# Real model calls
# ---------------------------------------------------------------------------


@requires_ollama
def test_judge_separates_a_grounded_answer_from_an_invented_one() -> None:
    """The point of the tier: catch a fluent answer built on nothing.

    Asserts the *ordering*, not absolute values. A small local model's exact
    numbers are not stable enough to pin, and pinning them would make this a
    test of qwen2.5:3b rather than of the evaluator.
    """
    ev = build_evaluator("faithfulness", provider="ollama", model=JUDGE_MODEL)

    grounded = run_evaluator(
        ev,
        EvalInput(
            output="Kaiser Permanente evaluated Limbic for mental health triage.",
            context=CONTEXT,
        ),
    )
    invented = run_evaluator(
        ev,
        EvalInput(
            output=(
                "Kaiser Permanente acquired Limbic for $4.2 billion in 1998 and "
                "deployed it to 300 hospitals in Antarctica."
            ),
            context=CONTEXT,
        ),
    )

    assert not grounded.failed_to_run, grounded.error
    assert not invented.failed_to_run, invented.error
    assert grounded.value > invented.value, (
        f"judge did not separate grounded ({grounded.value}) from invented "
        f"({invented.value}); grounded reason={grounded.explanation!r}; "
        f"invented reason={invented.explanation!r}"
    )


@requires_ollama
def test_judge_returns_usage_and_latency_for_cost_accounting() -> None:
    """A judge is a cost centre; its usage must be auditable."""
    ev = build_evaluator("faithfulness", provider="ollama", model=JUDGE_MODEL)
    score = run_evaluator(
        ev, EvalInput(output="Limbic was evaluated for triage.", context=CONTEXT)
    )
    assert not score.failed_to_run, score.error
    assert score.metadata["judge_input_tokens"] > 0
    assert score.metadata["judge_output_tokens"] > 0
    assert score.metadata["judge_latency_ms"] > 0
    assert score.metadata["judge_model"].startswith(JUDGE_MODEL.split(":")[0])


@requires_ollama
def test_judge_score_is_in_range_and_carries_a_reason() -> None:
    ev = build_evaluator("answer_relevance", provider="ollama", model=JUDGE_MODEL)
    score = run_evaluator(
        ev,
        EvalInput(
            output="The refund window is thirty days from delivery.",
            inputs={"question": "How long do I have to request a refund?"},
        ),
    )
    assert not score.failed_to_run, score.error
    assert 0.0 <= score.value <= 1.0
    assert score.explanation
