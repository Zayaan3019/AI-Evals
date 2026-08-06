"""Evaluator tiers: correctness, containment, and verified determinism."""

from __future__ import annotations

import json

import pytest

from agentlens_platform.evaluators import (
    BUILTIN_EVALUATORS,
    EvalInput,
    NonDeterministicEvaluator,
    Score,
    Tier,
    UnknownEvaluator,
    assert_deterministic,
    build_evaluator,
    describe_registry,
    run_evaluator,
    run_suite,
)
from agentlens_platform.evaluators.base import Evaluator
from agentlens_platform.evaluators.statistical import tokenize


def item(output: str, **kwargs) -> EvalInput:
    return EvalInput(output=output, **kwargs)


# ---------------------------------------------------------------------------
# Tier 1: deterministic
# ---------------------------------------------------------------------------


def test_exact_match_normalises_case_and_whitespace() -> None:
    ev = build_evaluator("exact_match")
    assert run_evaluator(ev, item("  Thirty Days ", expected="thirty days")).value == 1.0
    assert run_evaluator(ev, item("thirty one days", expected="thirty days")).value == 0.0


def test_exact_match_without_a_reference_is_an_error_not_a_zero() -> None:
    """A case this evaluator cannot score must not look like a failing system."""
    score = run_evaluator(build_evaluator("exact_match"), item("anything"))
    assert score.failed_to_run
    assert score.value is None


def test_contains_scores_partial_credit() -> None:
    ev = build_evaluator("contains", substrings=["refund", "30 days", "receipt"])
    score = run_evaluator(ev, item("You get a refund within 30 days."))
    assert score.value == pytest.approx(2 / 3)
    assert score.passed is False
    assert "receipt" in score.explanation


def test_contains_rejects_an_empty_substring_list_at_construction() -> None:
    with pytest.raises(ValueError):
        build_evaluator("contains", substrings=[])


def test_regex_compiles_at_construction_not_per_case() -> None:
    with pytest.raises(Exception):
        build_evaluator("regex_match", pattern="([unclosed")


def test_valid_json_reports_missing_keys_proportionally() -> None:
    ev = build_evaluator("valid_json", required_keys=["a", "b", "c", "d"])
    score = run_evaluator(ev, item(json.dumps({"a": 1, "b": 2})))
    assert score.value == 0.5
    assert score.passed is False


def test_valid_json_rejects_non_objects_when_keys_required() -> None:
    ev = build_evaluator("valid_json", required_keys=["a"])
    assert run_evaluator(ev, item("[1, 2, 3]")).passed is False


def test_no_pii_leak_uses_the_sdk_redactor() -> None:
    """One definition of PII, shared -- two would drift apart."""
    ev = build_evaluator("no_pii_leak")
    clean = run_evaluator(ev, item("Your order shipped on Tuesday."))
    assert clean.value == 1.0
    leaked = run_evaluator(ev, item("Contact alice@example.com about it."))
    assert leaked.value == 0.0
    assert "email" in leaked.explanation


def test_no_pii_leak_does_not_flag_numbers_as_cards() -> None:
    """The float false-positive fixed in Phase 1 must stay fixed here too."""
    ev = build_evaluator("no_pii_leak")
    assert run_evaluator(ev, item('{"score": 0.8231847162893746}')).value == 1.0


def test_output_length_degrades_smoothly_rather_than_cliff_edging() -> None:
    ev = build_evaluator("output_length", min_chars=10, max_chars=100)
    assert run_evaluator(ev, item("x" * 50)).value == 1.0
    slightly_long = run_evaluator(ev, item("x" * 200)).value
    very_long = run_evaluator(ev, item("x" * 1000)).value
    assert 0 < very_long < slightly_long < 1.0


def test_output_length_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError):
        build_evaluator("output_length", min_chars=100, max_chars=10)


def test_tool_was_called_checks_behaviour_not_prose() -> None:
    ev = build_evaluator("tool_was_called", tool_name="lookup_order")
    called = item("done", metadata={"tools_called": ["lookup_order", "notify"]})
    assert run_evaluator(ev, called).value == 1.0
    missing = item("done", metadata={"tools_called": ["notify"]})
    assert run_evaluator(ev, missing).value == 0.0


def test_tool_was_called_supports_forbidding_a_tool() -> None:
    ev = build_evaluator("tool_was_called", tool_name="delete_account", forbidden=True)
    assert run_evaluator(ev, item("ok", metadata={"tools_called": ["read"]})).value == 1.0
    assert run_evaluator(ev, item("ok", metadata={"tools_called": ["delete_account"]})).value == 0.0


def test_tool_was_called_without_trace_data_is_an_error() -> None:
    ev = build_evaluator("tool_was_called", tool_name="x")
    assert run_evaluator(ev, item("ok")).failed_to_run


# ---------------------------------------------------------------------------
# Tier 2: statistical
# ---------------------------------------------------------------------------


def test_tokenizer_handles_non_latin_script() -> None:
    """`[a-z]+` would silently score every non-English case as empty."""
    assert tokenize("café naïve") == ["café", "naïve"]
    assert tokenize("東京 tokyo") == ["東京", "tokyo"]
    assert tokenize("привет мир") == ["привет", "мир"]


def test_tokenizer_uses_full_case_folding_not_lowercasing() -> None:
    """``casefold`` maps German ss to 'ss', which is what we want.

    Full case folding is stronger than ``lower()`` and exists precisely for
    caseless matching, so 'GROSSE' and 'große' compare equal. A token-overlap
    score that treated them as different words would report a spurious
    difference between two spellings of the same answer.
    """
    assert tokenize("große") == tokenize("GROSSE") == ["grosse"]


def test_token_f1_penalises_dumping_everything() -> None:
    """Perfect recall with terrible precision must not score well."""
    ev = build_evaluator("token_f1")
    exact = run_evaluator(ev, item("refunds within 30 days", expected="refunds within 30 days"))
    padded = run_evaluator(
        ev, item("refunds within 30 days " + "filler " * 200, expected="refunds within 30 days")
    )
    assert exact.value == pytest.approx(1.0)
    assert padded.value < 0.2


def test_token_f1_reports_precision_and_recall() -> None:
    score = run_evaluator(build_evaluator("token_f1"), item("a b", expected="a b c d"))
    assert score.metadata["precision"] == pytest.approx(1.0)
    assert score.metadata["recall"] == pytest.approx(0.5)


def test_jaccard_ignores_repetition() -> None:
    ev = build_evaluator("jaccard")
    assert run_evaluator(ev, item("a b a b a", expected="a b")).value == pytest.approx(1.0)


def test_context_groundedness_separates_supported_from_invented() -> None:
    ev = build_evaluator("context_groundedness")
    context = ("Refunds are processed within thirty days of delivery confirmation.",)
    grounded = run_evaluator(
        ev, item("Refunds processed within thirty days of delivery.", context=context)
    )
    invented = run_evaluator(
        ev, item("Refunds require notarised affidavits and biometric verification.", context=context)
    )
    assert grounded.value > 0.8
    assert invented.value < 0.3


def test_context_groundedness_requires_context() -> None:
    assert run_evaluator(build_evaluator("context_groundedness"), item("x")).failed_to_run


def test_response_entropy_detects_degenerate_repetition() -> None:
    """A looping agent is a real failure mode with no reference to disagree with."""
    ev = build_evaluator("response_entropy")
    healthy = run_evaluator(ev, item("The refund window is thirty days from delivery date."))
    looping = run_evaluator(ev, item("yes " * 50))
    assert looping.value == 0.0
    assert healthy.value > 0.9


def test_response_entropy_is_undefined_for_a_single_token() -> None:
    assert run_evaluator(build_evaluator("response_entropy"), item("hi")).failed_to_run


# ---------------------------------------------------------------------------
# Registry behaviour
# ---------------------------------------------------------------------------


class Exploding(Evaluator):
    slug = "exploding"
    tier = Tier.DETERMINISTIC
    version = "1"

    def evaluate(self, item: EvalInput) -> Score:
        raise RuntimeError("evaluator bug")


class WrongType(Evaluator):
    slug = "wrong_type"
    tier = Tier.DETERMINISTIC
    version = "1"

    def evaluate(self, item: EvalInput) -> Score:
        return 0.5  # type: ignore[return-value]


class OutOfRange(Evaluator):
    slug = "out_of_range"
    tier = Tier.DETERMINISTIC
    version = "1"

    def evaluate(self, item: EvalInput) -> Score:
        return Score(value=7.0)


class Flaky(Evaluator):
    slug = "flaky"
    tier = Tier.DETERMINISTIC
    version = "1"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def evaluate(self, item: EvalInput) -> Score:
        self.calls += 1
        return Score(value=float(self.calls % 2))


def test_an_evaluator_bug_is_recorded_as_an_error_not_a_zero() -> None:
    """Conflating the two makes a broken evaluator look like a regression."""
    score = run_evaluator(Exploding(), item("x"))
    assert score.failed_to_run
    assert score.value is None
    assert "evaluator bug" in score.error


def test_a_non_score_return_is_caught() -> None:
    assert run_evaluator(WrongType(), item("x")).failed_to_run


def test_an_out_of_range_score_is_rejected() -> None:
    """A 7.0 silently averaged into a 0-1 metric corrupts every downstream number."""
    score = run_evaluator(OutOfRange(), item("x"))
    assert score.failed_to_run
    assert "out-of-range" in score.error


def test_determinism_is_verified_not_trusted() -> None:
    with pytest.raises(NonDeterministicEvaluator):
        assert_deterministic(Flaky(), [item("x")])


def test_real_deterministic_evaluators_pass_the_check() -> None:
    cases = [
        item("thirty days", expected="thirty days"),
        item('{"a": 1}'),
        item("Contact alice@example.com"),
        item("The refund window is thirty days from the delivery date."),
    ]
    for slug in ("exact_match", "valid_json", "no_pii_leak", "output_length", "token_f1", "jaccard"):
        assert_deterministic(build_evaluator(slug), cases)


def test_model_graded_evaluators_are_not_claimed_deterministic() -> None:
    with pytest.raises(NonDeterministicEvaluator):
        assert_deterministic(build_evaluator("faithfulness"), [item("x")])


def test_unknown_slug_lists_what_is_available() -> None:
    with pytest.raises(UnknownEvaluator) as exc:
        build_evaluator("does_not_exist")
    assert "exact_match" in str(exc.value)


def test_config_hash_changes_when_meaning_changes() -> None:
    """Two judges are only comparable when this hash matches."""
    a = build_evaluator("faithfulness", provider="ollama", model="qwen2.5:3b")
    b = build_evaluator("faithfulness", provider="ollama", model="llama3.1:8b")
    c = build_evaluator("faithfulness", provider="ollama", model="qwen2.5:3b")
    assert a.config_hash() != b.config_hash()
    assert a.config_hash() == c.config_hash()


def test_config_hash_is_stable_across_key_ordering() -> None:
    a = build_evaluator("contains", substrings=["x"], case_sensitive=True)
    b = build_evaluator("contains", case_sensitive=True, substrings=["x"])
    assert a.config_hash() == b.config_hash()


def test_identity_distinguishes_versions() -> None:
    ev = build_evaluator("exact_match")
    assert ev.identity == "exact_match@1"


def test_run_suite_keys_on_identity_so_versions_cannot_collide() -> None:
    results = run_suite(
        [build_evaluator("exact_match"), build_evaluator("token_f1")],
        item("thirty days", expected="thirty days"),
    )
    assert set(results) == {"exact_match@1", "token_f1@1"}
    assert all(s.value == pytest.approx(1.0) for s in results.values())


def test_registry_catalogue_is_complete() -> None:
    """Evaluators requiring config are reported, not omitted."""
    catalogue = describe_registry()
    assert len(catalogue) == len(BUILTIN_EVALUATORS)
    needs_config = {e["slug"] for e in catalogue if e["requires_config"]}
    assert {"contains", "regex_match", "tool_was_called", "custom"} <= needs_config


def test_all_four_tiers_are_represented() -> None:
    tiers = {e["tier"] for e in describe_registry()}
    assert tiers == {"deterministic", "statistical", "model_graded", "human"}


# ---------------------------------------------------------------------------
# Tier 4: human
# ---------------------------------------------------------------------------


def test_human_label_is_keyed_on_case_and_output() -> None:
    """A label is an opinion about one output, not about the case forever."""
    ev = build_evaluator("human_label", labels={("case-1", "thirty days"): 1.0})
    hit = run_evaluator(ev, item("thirty days", metadata={"case_id": "case-1"}))
    assert hit.value == 1.0

    # Same case, different answer -> not labelled, not a silent reuse.
    miss = run_evaluator(ev, item("ninety days", metadata={"case_id": "case-1"}))
    assert miss.failed_to_run


def test_unlabelled_case_is_an_absence_not_a_zero() -> None:
    ev = build_evaluator("human_label", labels={})
    assert run_evaluator(ev, item("x", metadata={"case_id": "c"})).value is None
