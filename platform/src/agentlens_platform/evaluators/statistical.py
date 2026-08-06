"""Tier 2: statistical evaluators.

Deterministic given identical inputs, but they measure *similarity*, not
correctness. That distinction is the whole reason this is a separate tier: a
high token-overlap score means the output resembles the reference, which is
evidence about phrasing and only weak evidence about truth. They are excellent
at detecting drift and poor at catching a fluent, well-phrased lie.

Implemented from primitives rather than pulled from a metrics library on
purpose: each one is a few lines, and the exact tokenisation and smoothing
choices change the numbers materially. Vendoring the definition means the score
is auditable and stable across dependency upgrades.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .base import DeterministicEvaluator, EvalInput, Score, Tier

#: Unicode-aware word tokenisation. ``\w`` under Python's ``re`` already covers
#: accented and non-Latin letters, so this does not silently drop non-English
#: text the way ``[a-z]+`` would.
_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.casefold())


class _StatisticalEvaluator(DeterministicEvaluator):
    """Deterministic in the mechanical sense, but a different tier of evidence."""

    tier = Tier.STATISTICAL


class TokenF1(_StatisticalEvaluator):
    """Token-overlap F1 between output and expected.

    F1 rather than raw overlap because precision and recall fail differently:
    an output that dumps the whole knowledge base scores perfect recall while
    being useless. The harmonic mean penalises that.
    """

    slug = "token_f1"
    version = "1"

    def evaluate(self, item: EvalInput) -> Score:
        if item.expected is None:
            return Score.failure("token_f1 requires an expected output")

        predicted = Counter(tokenize(item.output))
        reference = Counter(tokenize(item.expected))
        if not predicted or not reference:
            # Both empty is a perfect match; one empty is a total miss.
            both_empty = not predicted and not reference
            return Score(value=1.0 if both_empty else 0.0, passed=both_empty)

        overlap = sum((predicted & reference).values())
        if overlap == 0:
            return Score(value=0.0, passed=False, explanation="no token overlap")

        precision = overlap / sum(predicted.values())
        recall = overlap / sum(reference.values())
        f1 = 2 * precision * recall / (precision + recall)
        return Score(
            value=f1,
            passed=None,  # A similarity has no intrinsic pass threshold.
            explanation=f"precision={precision:.3f} recall={recall:.3f}",
            metadata={"precision": precision, "recall": recall, "overlap_tokens": overlap},
        )


class JaccardSimilarity(_StatisticalEvaluator):
    """Set overlap of unique tokens. Insensitive to repetition and order."""

    slug = "jaccard"
    version = "1"

    def evaluate(self, item: EvalInput) -> Score:
        if item.expected is None:
            return Score.failure("jaccard requires an expected output")
        predicted = set(tokenize(item.output))
        reference = set(tokenize(item.expected))
        union = predicted | reference
        if not union:
            return Score(value=1.0, passed=True)
        value = len(predicted & reference) / len(union)
        return Score(value=value, passed=None)


class ContextGroundedness(_StatisticalEvaluator):
    """Fraction of the output's content tokens that appear in the context.

    A cheap, reference-free proxy for "is this answer supported by what was
    retrieved". It cannot detect a *correct* claim stated in different words, so
    it is a screen, not a verdict -- the model-graded groundedness evaluator
    exists for the cases this one flags.

    Stopwords are excluded because they appear in every context and would inflate
    the score toward 1 regardless of grounding.
    """

    slug = "context_groundedness"
    version = "1"

    #: Deliberately small and English-only. A larger list would be a hidden
    #: model of the language; this covers the tokens that would otherwise
    #: dominate the count.
    STOPWORDS = frozenset(
        """a an the and or but if then than that this these those of to in on at by for with
        from as is are was were be been being it its no not do does did have has had i you he
        she they we them his her their our your my me us can could should would will shall may
        might must so such about into over under again further once here there when where why
        how all any both each few more most other some only own same too very s t just don now""".split()
    )

    def __init__(self, *, min_token_length: int = 3) -> None:
        super().__init__(min_token_length=min_token_length)

    def _content_tokens(self, text: str) -> list[str]:
        minimum = self.config["min_token_length"]
        return [
            token
            for token in tokenize(text)
            if len(token) >= minimum and token not in self.STOPWORDS
        ]

    def evaluate(self, item: EvalInput) -> Score:
        if not item.context:
            return Score.failure("context_groundedness requires retrieved context")

        claim_tokens = self._content_tokens(item.output)
        if not claim_tokens:
            return Score.failure("output has no content tokens to ground")

        supported = set()
        for chunk in item.context:
            supported.update(self._content_tokens(chunk))

        grounded = [token for token in claim_tokens if token in supported]
        value = len(grounded) / len(claim_tokens)
        ungrounded = sorted(set(claim_tokens) - supported)
        return Score(
            value=value,
            passed=None,
            explanation=f"{len(grounded)}/{len(claim_tokens)} content tokens in context"
            + (f"; ungrounded sample: {ungrounded[:8]}" if ungrounded else ""),
            metadata={"ungrounded_token_count": len(ungrounded)},
        )


class ResponseEntropy(_StatisticalEvaluator):
    """Shannon entropy over token frequencies, normalised to [0, 1].

    Detects degenerate output -- looping, repetition, a single token repeated --
    which is a real and recognisable agent failure mode that no reference-based
    metric catches, because there is often no reference for it to disagree with.

    Normalised by ``log(distinct_tokens)`` so length does not dominate.
    """

    slug = "response_entropy"
    version = "1"

    def evaluate(self, item: EvalInput) -> Score:
        tokens = tokenize(item.output)
        if len(tokens) < 2:
            return Score.failure("entropy is undefined for fewer than two tokens")

        counts = Counter(tokens)
        total = len(tokens)
        entropy = -sum(
            (count / total) * math.log(count / total) for count in counts.values()
        )
        # Maximum entropy for this many distinct tokens is log(distinct).
        # With one distinct token the maximum is 0, and so is the entropy.
        maximum = math.log(len(counts)) if len(counts) > 1 else 0.0
        value = 1.0 if maximum == 0 and entropy == 0 else (entropy / maximum if maximum else 0.0)
        # A single repeated token is maximally degenerate, not maximally healthy.
        if len(counts) == 1:
            value = 0.0
        return Score(
            value=value,
            passed=None,
            explanation=f"{len(counts)} distinct of {total} tokens",
            metadata={"distinct_tokens": len(counts), "total_tokens": total},
        )


BUILTIN_STATISTICAL: dict[str, type[_StatisticalEvaluator]] = {
    cls.slug: cls
    for cls in (TokenF1, JaccardSimilarity, ContextGroundedness, ResponseEntropy)
}
