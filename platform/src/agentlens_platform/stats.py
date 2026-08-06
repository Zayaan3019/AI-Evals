"""Statistics for comparing two eval runs.

The question a regression gate must answer is not "did the mean go down" but
"did the mean go down by more than this dataset's noise would produce anyway".
Those are different questions, and answering the first while believing you
answered the second is how teams end up reverting good changes and shipping bad
ones.

**Why paired.** Runs are compared case by case, not mean against mean. Cases
differ enormously in difficulty, and that between-case variance swamps the
effect of a prompt change. Pairing removes it: we resample the *per-case
differences*, so a case that is hard for both versions contributes a difference
near zero rather than a large number to both sides. On correlated data this is
dramatically more powerful than an unpaired test at the same sample size.

**Why bootstrap rather than a t-test.** Eval scores are routinely bounded in
[0, 1], bimodal (a pass/fail evaluator produces only 0 and 1), and skewed. The
t-test assumes approximate normality of the sampling distribution; at the
50-200 case datasets that are typical here, that assumption is doing real work
and is often wrong. The bootstrap assumes only that the sample is
representative, and it gives a confidence interval on the actual statistic of
interest rather than on a normal approximation to it.

**Why a seed.** A gate that returns a different verdict on a re-run of the same
data is not a gate. The seed is explicit and recorded.

Implemented from primitives: scipy is a large dependency for three functions,
and vendoring the definitions means the numbers are auditable and cannot shift
under a transitive upgrade.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

#: Default resample count. 10,000 is the conventional floor for a percentile
#: interval: the Monte Carlo error on a 95% bound is roughly +/-0.005 at this
#: count, which is small relative to any effect worth gating on.
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20260803


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Result of comparing two runs over the same cases."""

    n_pairs: int
    baseline_mean: float
    candidate_mean: float
    mean_difference: float
    ci_low: float
    ci_high: float
    p_value: float
    confidence: float
    resamples: int
    seed: int

    @property
    def is_significant(self) -> bool:
        """Whether the confidence interval excludes zero.

        Equivalent to the two-sided test at ``1 - confidence``, and reported
        alongside the interval rather than instead of it: an interval of
        [-0.001, 0.002] and one of [-0.4, 0.45] are both "not significant" and
        mean entirely different things.
        """
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def regressed(self) -> bool:
        return self.is_significant and self.mean_difference < 0

    @property
    def improved(self) -> bool:
        return self.is_significant and self.mean_difference > 0


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("mean of an empty sequence is undefined")
    return sum(values) / len(values)


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> PairedComparison:
    """Bootstrap confidence interval on the paired mean difference.

    ``baseline[i]`` and ``candidate[i]`` must be scores for the *same case*.
    The caller is responsible for that alignment; :func:`align_by_case` exists
    to do it correctly.

    Returns the interval on ``candidate - baseline``, so a negative interval
    entirely below zero is a regression.
    """
    if len(baseline) != len(candidate):
        raise ValueError(
            f"paired comparison needs equal lengths, got {len(baseline)} and {len(candidate)}"
        )
    if not baseline:
        raise ValueError("paired comparison needs at least one pair")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between 0 and 1")
    if resamples < 1:
        raise ValueError("resamples must be positive")

    differences = [c - b for b, c in zip(baseline, candidate)]
    observed = mean(differences)
    n = len(differences)

    rng = random.Random(seed)
    resampled_means: list[float] = []
    for _ in range(resamples):
        # Resample the *differences*, which is what makes this paired: each draw
        # keeps a case's baseline and candidate score together.
        total = 0.0
        for _ in range(n):
            total += differences[rng.randrange(n)]
        resampled_means.append(total / n)

    resampled_means.sort()
    alpha = 1.0 - confidence
    ci_low = _percentile(resampled_means, alpha / 2)
    ci_high = _percentile(resampled_means, 1 - alpha / 2)

    # Two-sided achieved significance level: how often a resample lands on the
    # far side of zero from the observed effect. The +1 corrections keep the
    # estimate from ever being exactly 0, which would claim more certainty than
    # a finite number of resamples can support.
    if observed >= 0:
        tail = sum(1 for m in resampled_means if m <= 0.0)
    else:
        tail = sum(1 for m in resampled_means if m >= 0.0)
    p_value = min(1.0, 2.0 * (tail + 1) / (resamples + 1))

    return PairedComparison(
        n_pairs=n,
        baseline_mean=mean(baseline),
        candidate_mean=mean(candidate),
        mean_difference=observed,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        confidence=confidence,
        resamples=resamples,
        seed=seed,
    )


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence."""
    if not sorted_values:
        raise ValueError("percentile of an empty sequence is undefined")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def align_by_case(
    baseline: dict[str, float | None], candidate: dict[str, float | None]
) -> tuple[list[float], list[float], list[str]]:
    """Intersect two score maps into aligned, ordered lists.

    Cases missing from either run, or unscored in either (``None``), are
    dropped from **both** sides. Keeping a case on one side only would silently
    unpair the comparison, which is the exact failure the pairing exists to
    avoid. The dropped keys are returned so the caller can report how much data
    the comparison actually rests on.
    """
    usable = sorted(
        key
        for key in baseline.keys() & candidate.keys()
        if baseline[key] is not None and candidate[key] is not None
    )
    dropped = sorted((baseline.keys() | candidate.keys()) - set(usable))
    return (
        [float(baseline[key]) for key in usable],  # type: ignore[arg-type]
        [float(candidate[key]) for key in usable],  # type: ignore[arg-type]
        dropped,
    )


# ---------------------------------------------------------------------------
# Judge calibration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Calibration:
    """How well a model-graded evaluator agrees with human labels."""

    n: int
    pearson_r: float | None
    spearman_r: float | None
    mean_absolute_error: float
    #: Agreement after thresholding both to pass/fail. The number most people
    #: actually mean by "is the judge any good".
    binary_agreement: float | None
    cohens_kappa: float | None
    judge_mean: float
    human_mean: float

    @property
    def judge_is_biased_high(self) -> bool:
        return self.judge_mean > self.human_mean


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation, or ``None`` when it is undefined.

    Undefined when either series is constant -- there is no variance to
    correlate. Returning 0.0 there would read as "no relationship" when the
    truth is "not computable", which is a materially different claim.
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x, mean_y = mean(xs), mean(ys)
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]
    denominator = math.sqrt(sum(d * d for d in dx)) * math.sqrt(sum(d * d for d in dy))
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / denominator


def _rank(values: Sequence[float]) -> list[float]:
    """Ranks with ties averaged, as Spearman requires."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        average = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            ranks[order[position]] = average
        index = stop + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Rank correlation. Robust to a judge that is monotonic but miscalibrated.

    This is usually the more honest number for a judge: what matters is whether
    it *orders* outputs like a human does, not whether its 0.8 means the same as
    a human's 0.8.
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(_rank(xs), _rank(ys))


def cohens_kappa(judge: Sequence[bool], human: Sequence[bool]) -> float | None:
    """Chance-corrected agreement on binary labels.

    Raw agreement is misleading when one class dominates: a judge that passes
    everything scores 90% agreement on a dataset that is 90% passes while
    carrying no information. Kappa subtracts the agreement expected by chance,
    so that judge scores 0.
    """
    if len(judge) != len(human) or not judge:
        return None
    n = len(judge)
    observed = sum(1 for j, h in zip(judge, human) if j == h) / n

    judge_true = sum(judge) / n
    human_true = sum(human) / n
    expected = judge_true * human_true + (1 - judge_true) * (1 - human_true)
    if expected == 1.0:
        # Both raters are constant and identical: agreement is total but there
        # is no variance for chance correction to work on.
        return None
    return (observed - expected) / (1 - expected)


def calibrate(
    judge_scores: Sequence[float],
    human_scores: Sequence[float],
    *,
    threshold: float = 0.5,
) -> Calibration:
    """Compare a judge against human ground truth on the same items."""
    if len(judge_scores) != len(human_scores):
        raise ValueError("calibration needs one judge score per human score")
    if not judge_scores:
        raise ValueError("calibration needs at least one labelled item")

    judge_binary = [s >= threshold for s in judge_scores]
    human_binary = [s >= threshold for s in human_scores]
    agreement = (
        sum(1 for j, h in zip(judge_binary, human_binary) if j == h) / len(judge_binary)
        if judge_binary
        else None
    )

    return Calibration(
        n=len(judge_scores),
        pearson_r=pearson(judge_scores, human_scores),
        spearman_r=spearman(judge_scores, human_scores),
        mean_absolute_error=mean([abs(j - h) for j, h in zip(judge_scores, human_scores)]),
        binary_agreement=agreement,
        cohens_kappa=cohens_kappa(judge_binary, human_binary),
        judge_mean=mean(judge_scores),
        human_mean=mean(human_scores),
    )


def minimum_detectable_effect(
    scores: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2_000,
    seed: int = DEFAULT_SEED,
) -> float:
    """Roughly the smallest paired difference this dataset could detect.

    Bootstraps the half-width of the interval around a zero effect. Reported by
    the gate so a "no significant change" verdict comes with the context needed
    to read it: on 12 cases that verdict is nearly content-free, and the number
    says so.
    """
    if len(scores) < 2:
        return float("inf")
    rng = random.Random(seed)
    n = len(scores)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += scores[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    alpha = 1.0 - confidence
    return (_percentile(means, 1 - alpha / 2) - _percentile(means, alpha / 2)) / 2
