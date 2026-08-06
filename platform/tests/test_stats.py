"""Statistics: pairing, bootstrap behaviour, and calibration.

Verified against properties that must hold rather than against numbers copied
from another implementation. Where a closed-form answer exists it is used, and
where the correct answer is "undefined" the code is checked to say so rather
than to return a plausible-looking zero.
"""

from __future__ import annotations

import math
import random

import pytest

from agentlens_platform.stats import (
    align_by_case,
    calibrate,
    cohens_kappa,
    minimum_detectable_effect,
    paired_bootstrap,
    pearson,
    spearman,
)


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------


def test_identical_runs_show_no_effect() -> None:
    scores = [0.1, 0.5, 0.9, 0.3, 0.7] * 10
    result = paired_bootstrap(scores, scores)
    assert result.mean_difference == pytest.approx(0.0)
    assert result.ci_low == pytest.approx(0.0)
    assert result.ci_high == pytest.approx(0.0)
    assert not result.is_significant


def test_a_uniform_improvement_is_detected() -> None:
    baseline = [0.5] * 50
    candidate = [0.6] * 50
    result = paired_bootstrap(baseline, candidate)
    assert result.mean_difference == pytest.approx(0.1)
    assert result.improved
    assert not result.regressed


def test_a_uniform_drop_is_detected_as_a_regression() -> None:
    baseline = [0.8] * 50
    candidate = [0.6] * 50
    result = paired_bootstrap(baseline, candidate)
    assert result.regressed
    assert result.ci_high < 0


def test_pairing_sees_a_consistent_shift_that_unpaired_noise_would_hide() -> None:
    """The reason the test is paired at all.

    Cases differ hugely in difficulty. Here that between-case spread is large
    (0.05..0.95) and the real effect is small (+0.03) and present on every
    case. Paired analysis removes the case-difficulty variance and finds it.
    """
    rng = random.Random(11)
    difficulty = [rng.uniform(0.05, 0.95) for _ in range(80)]
    baseline = difficulty
    candidate = [min(1.0, d + 0.03) for d in difficulty]

    paired = paired_bootstrap(baseline, candidate)
    assert paired.is_significant

    # The same data compared without pairing: the standard deviation of the
    # scores themselves dwarfs a 0.03 shift.
    spread = math.sqrt(sum((d - sum(difficulty) / len(difficulty)) ** 2 for d in difficulty) / len(difficulty))
    assert spread > 0.2
    assert abs(paired.mean_difference) < spread


def test_noise_alone_does_not_produce_a_significant_verdict() -> None:
    """A gate that fires on noise is worse than no gate."""
    rng = random.Random(7)
    baseline = [rng.random() for _ in range(100)]
    candidate = [rng.random() for _ in range(100)]
    result = paired_bootstrap(baseline, candidate)
    assert not result.is_significant, (
        f"CI [{result.ci_low}, {result.ci_high}] excluded zero on pure noise"
    )


def test_results_are_reproducible_for_the_same_seed() -> None:
    """A gate that changes its mind on a re-run is not a gate."""
    rng = random.Random(3)
    baseline = [rng.random() for _ in range(40)]
    candidate = [rng.random() for _ in range(40)]
    first = paired_bootstrap(baseline, candidate, seed=99)
    second = paired_bootstrap(baseline, candidate, seed=99)
    assert (first.ci_low, first.ci_high, first.p_value) == (
        second.ci_low,
        second.ci_high,
        second.p_value,
    )


def test_bimodal_pass_fail_scores_are_handled() -> None:
    """Binary evaluators produce only 0 and 1 -- nothing like a normal curve."""
    baseline = [1.0] * 40 + [0.0] * 10
    candidate = [1.0] * 25 + [0.0] * 25
    result = paired_bootstrap(baseline, candidate)
    assert result.regressed


def test_p_value_is_never_exactly_zero() -> None:
    """A finite number of resamples cannot support a claim of p = 0."""
    result = paired_bootstrap([0.0] * 50, [1.0] * 50, resamples=1000)
    assert result.p_value > 0.0


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2], [0.1])


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([], [])


def test_a_wider_confidence_level_widens_the_interval() -> None:
    rng = random.Random(5)
    baseline = [rng.random() for _ in range(60)]
    candidate = [rng.random() for _ in range(60)]
    narrow = paired_bootstrap(baseline, candidate, confidence=0.80)
    wide = paired_bootstrap(baseline, candidate, confidence=0.99)
    assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------


def test_alignment_keeps_only_cases_scored_in_both_runs() -> None:
    baseline = {"a": 0.5, "b": 0.6, "c": 0.7}
    candidate = {"a": 0.9, "b": 0.8, "d": 0.1}
    base, cand, dropped = align_by_case(baseline, candidate)
    assert base == [0.5, 0.6]
    assert cand == [0.9, 0.8]
    assert dropped == ["c", "d"]


def test_alignment_drops_a_case_unscored_on_either_side() -> None:
    """Keeping it on one side only would silently unpair the comparison."""
    base, cand, dropped = align_by_case({"a": 0.5, "b": None}, {"a": 0.9, "b": 0.4})
    assert base == [0.5]
    assert cand == [0.9]
    assert dropped == ["b"]


def test_alignment_is_order_independent() -> None:
    base, cand, _ = align_by_case({"z": 0.1, "a": 0.9}, {"a": 0.8, "z": 0.2})
    # Sorted by key, so 'a' comes first on both sides -- the pairing is correct.
    assert base == [0.9, 0.1]
    assert cand == [0.8, 0.2]


# ---------------------------------------------------------------------------
# Correlation and calibration
# ---------------------------------------------------------------------------


def test_pearson_matches_a_known_value() -> None:
    assert pearson([1, 2, 3, 4, 5], [2, 4, 6, 8, 10]) == pytest.approx(1.0)
    assert pearson([1, 2, 3, 4, 5], [10, 8, 6, 4, 2]) == pytest.approx(-1.0)


def test_pearson_is_undefined_for_a_constant_series() -> None:
    """None, not 0.0. 'Not computable' and 'no relationship' are different claims."""
    assert pearson([1, 1, 1, 1], [1, 2, 3, 4]) is None


def test_spearman_captures_a_monotonic_but_nonlinear_relationship() -> None:
    """The honest number for a judge that orders correctly but is miscalibrated."""
    xs = [1, 2, 3, 4, 5]
    ys = [1, 4, 9, 16, 25]
    assert spearman(xs, ys) == pytest.approx(1.0)
    assert pearson(xs, ys) < 1.0


def test_spearman_averages_tied_ranks() -> None:
    assert spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)


def test_kappa_is_zero_for_a_judge_that_passes_everything() -> None:
    """Raw agreement would flatter it; kappa reveals it carries no information."""
    judge = [True] * 90 + [True] * 10
    human = [True] * 90 + [False] * 10
    raw = sum(1 for j, h in zip(judge, human) if j == h) / len(judge)
    assert raw == pytest.approx(0.9)
    assert cohens_kappa(judge, human) == pytest.approx(0.0)


def test_kappa_is_one_for_perfect_agreement_with_variance() -> None:
    judge = [True, False, True, False]
    assert cohens_kappa(judge, judge) == pytest.approx(1.0)


def test_kappa_is_undefined_when_both_raters_are_constant() -> None:
    assert cohens_kappa([True] * 10, [True] * 10) is None


def test_calibration_reports_judge_bias() -> None:
    judge = [0.9, 0.8, 0.95, 0.85]
    human = [0.5, 0.4, 0.6, 0.5]
    result = calibrate(judge, human)
    assert result.judge_is_biased_high
    # (0.40 + 0.40 + 0.35 + 0.35) / 4
    assert result.mean_absolute_error == pytest.approx(0.375)
    assert result.n == 4
    # Rank correlation is high but not 1.0: the human scored two items
    # identically (0.5, 0.5) and the judge separated them (0.9, 0.85). Averaged
    # tied ranks are the correct handling and they cap the achievable
    # correlation, so a perfect score here would actually indicate a bug in the
    # tie handling.
    assert result.spearman_r == pytest.approx(0.9486832980505138)
    assert result.spearman_r > result.pearson_r


def test_calibration_of_a_perfect_judge() -> None:
    scores = [0.1, 0.4, 0.6, 0.9, 1.0, 0.0]
    result = calibrate(scores, scores)
    assert result.pearson_r == pytest.approx(1.0)
    assert result.mean_absolute_error == pytest.approx(0.0)
    assert result.binary_agreement == pytest.approx(1.0)
    assert result.cohens_kappa == pytest.approx(1.0)


def test_calibration_rejects_mismatched_input() -> None:
    with pytest.raises(ValueError):
        calibrate([0.5], [0.5, 0.6])


# ---------------------------------------------------------------------------
# Minimum detectable effect
# ---------------------------------------------------------------------------


def test_more_cases_detect_smaller_effects() -> None:
    rng = random.Random(13)
    small = [rng.random() for _ in range(10)]
    large = [rng.random() for _ in range(400)]
    assert minimum_detectable_effect(large) < minimum_detectable_effect(small)


def test_mde_is_infinite_for_a_single_case() -> None:
    """One case can detect nothing, and the number should say so."""
    assert minimum_detectable_effect([0.5]) == float("inf")
