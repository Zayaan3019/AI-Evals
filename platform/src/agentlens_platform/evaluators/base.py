"""Evaluator contract and the four-tier taxonomy.

The tiers exist because the tiers have genuinely different failure modes, and
conflating them is how eval suites end up confidently wrong:

``deterministic``
    Pure functions of (output, expected). Same input, same score, forever. No
    network, no model, no randomness. These are the only evaluators whose score
    can be compared across time without qualification.

``statistical``
    Deterministic given the same inputs, but the score is a similarity or
    distance rather than a judgement. Cheap, no model call. Fine for detecting
    drift; poor at detecting whether an answer is *right*.

``model_graded``
    An LLM judges the output. Non-deterministic, costs money, and -- critically
    -- the scores are only comparable to other scores from the *same judge
    configuration*. Changing the judge model or its prompt changes what the
    number means. That is why ``config_hash`` exists and why the regression gate
    refuses to compare across it.

``human``
    A person's label. The ground truth that calibrates the model-graded tier.
    Never computed at run time; read from stored labels.

Every evaluator returns a :class:`Score`, and a failure of the *evaluator* is
recorded as ``error`` rather than as ``score=0``. Silently scoring zero when the
judge times out understates quality and, worse, does so in a way that looks like
a regression in the system under test.
"""

from __future__ import annotations

import abc
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class Tier(str, Enum):
    DETERMINISTIC = "deterministic"
    STATISTICAL = "statistical"
    MODEL_GRADED = "model_graded"
    HUMAN = "human"


@dataclass(frozen=True, slots=True)
class EvalInput:
    """Everything an evaluator may look at.

    ``inputs`` is the frozen case input; ``output`` is what the system under
    test actually produced. ``expected`` may be absent -- reference-free
    evaluators are the norm for open-ended generation.
    """

    output: str
    inputs: Mapping[str, Any] = field(default_factory=dict)
    expected: str | None = None
    #: Retrieved context, when the case involves retrieval. Used by groundedness
    #: checks, which cannot be computed from output alone.
    context: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Score:
    """One evaluator's verdict on one case.

    ``value`` is conventionally in [0, 1] so scores from different evaluators
    can be averaged and thresholded uniformly; evaluators that are naturally
    binary return exactly 0.0 or 1.0.

    ``error`` being set means the evaluator itself failed. ``value`` is then
    ``None``, never 0.0 -- see the module docstring.
    """

    value: float | None
    passed: bool | None = None
    explanation: str | None = None
    error: str | None = None
    #: Judge tokens, latency, retries -- whatever the caller needs to audit cost.
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def failed_to_run(self) -> bool:
        return self.error is not None

    @classmethod
    def failure(cls, message: str) -> Score:
        return cls(value=None, passed=None, error=message)


class Evaluator(abc.ABC):
    """Base class for all evaluators.

    Subclasses implement :meth:`evaluate`. The registry, not the subclass, is
    responsible for catching exceptions and converting them to
    ``Score.failure`` -- so an evaluator author cannot accidentally swallow a
    bug into a passing score.
    """

    #: Stable identifier. Used as the registry key and stored on results.
    slug: str = ""
    tier: Tier = Tier.DETERMINISTIC
    #: Bumped when the scoring logic changes in a way that alters results.
    #: Comparing scores across a version change is comparing different metrics.
    version: str = "1"

    def __init__(self, **config: Any) -> None:
        self.config = dict(config)

    @abc.abstractmethod
    def evaluate(self, item: EvalInput) -> Score:
        """Score one item. May raise; the registry converts that to a failure."""

    # -- identity ---------------------------------------------------------

    def config_hash(self) -> str:
        """Hash of everything that changes what this evaluator's score means.

        Includes the slug and version, so two evaluators with coincidentally
        identical config still hash differently. Sorted keys and a canonical
        separator make the hash stable across dict ordering and Python runs.
        """
        payload = json.dumps(
            {"slug": self.slug, "version": self.version, "config": self.config},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def identity(self) -> str:
        """Human-readable identity, e.g. ``exact_match@1``."""
        return f"{self.slug}@{self.version}"

    def describe(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "tier": self.tier.value,
            "version": self.version,
            "config": self.config,
            "config_hash": self.config_hash(),
        }


class DeterministicEvaluator(Evaluator):
    """Marker base: no I/O, no randomness, no model.

    Kept as a distinct class so the registry can assert the property rather
    than trust a declaration -- see ``registry.assert_deterministic``.
    """

    tier = Tier.DETERMINISTIC
