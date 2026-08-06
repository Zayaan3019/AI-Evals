"""Tier 4: human labels.

A human evaluator computes nothing. It reads a stored label produced by a
person and returns it as a score. That is the whole point: this tier is the
ground truth the model-graded tier is calibrated against, so it must be
incapable of being derived from a model.

Labels are looked up by ``(case_id, graded_output)``. Keying on the output as
well as the case is deliberate -- a label is an opinion about a *specific
output*, not about the case in the abstract. Reusing yesterday's label for
today's different answer would silently manufacture agreement, which is exactly
the number Phase 5 uses to decide whether a judge can be trusted.
"""

from __future__ import annotations

from typing import Mapping

from .base import EvalInput, Evaluator, Score, Tier


class HumanLabelEvaluator(Evaluator):
    """Returns a stored human label, or an explicit 'not labelled' failure.

    The label index is supplied by the caller (loaded from Postgres) rather than
    queried here, so scoring stays a pure function and a run over 10,000 cases
    is one query rather than 10,000.
    """

    slug = "human_label"
    tier = Tier.HUMAN
    version = "1"

    def __init__(self, *, labels: Mapping[tuple[str, str], float] | None = None) -> None:
        self._labels = dict(labels or {})
        # The labels themselves are data, not configuration: including them in
        # the hash would make every new label look like a different evaluator.
        super().__init__()

    def evaluate(self, item: EvalInput) -> Score:
        case_id = item.metadata.get("case_id")
        if case_id is None:
            return Score.failure("human_label requires case_id in metadata")

        key = (str(case_id), item.output)
        if key not in self._labels:
            # Not a zero and not a pass. An unlabelled case is an absence of
            # evidence, and averaging it in either direction is a fabrication.
            return Score.failure("no human label for this (case, output) pair")

        value = float(self._labels[key])
        return Score(
            value=value,
            passed=value >= 0.5,
            explanation="human label",
            metadata={"source": "human"},
        )
