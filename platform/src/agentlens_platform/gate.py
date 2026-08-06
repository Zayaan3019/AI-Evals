"""The regression gate.

Decides whether a candidate run may ship, given a baseline. Designed around one
principle: **the gate must refuse to answer when it cannot answer honestly.**

Three refusals, all of which are more useful than a verdict:

``INCOMPARABLE``
    The two runs were not scored by the same evaluator configuration. A judge
    model or prompt changed, so the scores are not on the same scale. Comparing
    them produces a number that looks exactly like a real result and means
    nothing. This is why ``config_hash`` is recorded at run time.

``INSUFFICIENT_DATA``
    Too few paired cases to detect an effect anyone would care about. Reported
    together with the minimum detectable effect, so "no significant change" on
    nine cases is not mistaken for evidence of safety.

``ERROR_RATE``
    Too many evaluator failures. A run where a third of the judge calls timed
    out has a mean computed from the two thirds that succeeded, and that subset
    is not random -- long, complex outputs are the ones that time out.

Otherwise the gate applies a paired bootstrap per evaluator and fails if any
gated evaluator regressed significantly, or if any evaluator crossed an absolute
floor. Both checks matter: significance catches "this is worse than noise", and
the floor catches "this was already bad and stayed bad".
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db.models import EvalResult, EvalRun, EvaluatorRecord
from .stats import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    PairedComparison,
    align_by_case,
    minimum_detectable_effect,
    paired_bootstrap,
)


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCOMPARABLE = "incomparable"
    INSUFFICIENT_DATA = "insufficient_data"
    ERROR_RATE = "error_rate"

    @property
    def blocks_release(self) -> bool:
        """Everything except a clean pass blocks.

        A refusal blocks deliberately: "we could not tell" must not be treated
        as "it is fine". That default is the difference between a gate and a
        formality.
        """
        return self is not Verdict.PASS


@dataclass(frozen=True, slots=True)
class GatePolicy:
    """What counts as a regression. Explicit, so a verdict is reproducible."""

    #: Evaluators the gate acts on. Empty means every evaluator present in both
    #: runs. Naming them is safer: an evaluator added later would otherwise
    #: silently start blocking releases.
    gated_evaluators: tuple[str, ...] = ()
    #: A significant drop larger than this fails. Zero means any statistically
    #: significant drop fails.
    max_mean_drop: float = 0.0
    #: Absolute minimum mean score per evaluator, regardless of the baseline.
    score_floors: Mapping[str, float] = field(default_factory=dict)
    #: Below this many paired cases the gate refuses rather than guesses.
    min_pairs: int = 20
    #: Fraction of results that may have failed to evaluate.
    max_error_rate: float = 0.10
    confidence: float = 0.95
    resamples: int = DEFAULT_RESAMPLES
    seed: int = DEFAULT_SEED


@dataclass(frozen=True, slots=True)
class EvaluatorVerdict:
    identity: str
    comparison: PairedComparison | None
    regressed: bool
    floor_breached: bool
    floor: float | None
    dropped_cases: int
    note: str | None = None


@dataclass(frozen=True, slots=True)
class GateReport:
    verdict: Verdict
    reason: str
    baseline_label: str
    candidate_label: str
    per_evaluator: tuple[EvaluatorVerdict, ...]
    minimum_detectable_effect: float | None = None

    @property
    def blocks_release(self) -> bool:
        return self.verdict.blocks_release

    def render(self) -> str:
        """Plain-text report for CI logs.

        Prints the interval, not only the verdict: an engineer looking at a
        failed build needs to see how large the drop was and how certain it is,
        or the only available response is to re-run and hope.
        """
        lines = [
            f"agentlens regression gate: {self.verdict.value.upper()}",
            f"  baseline:  {self.baseline_label}",
            f"  candidate: {self.candidate_label}",
            f"  {self.reason}",
        ]
        if self.minimum_detectable_effect is not None:
            lines.append(
                f"  minimum detectable effect at this sample size: "
                f"{self.minimum_detectable_effect:+.4f}"
            )
        for entry in self.per_evaluator:
            if entry.comparison is None:
                lines.append(f"  - {entry.identity}: {entry.note}")
                continue
            c = entry.comparison
            flags = []
            if entry.regressed:
                flags.append("REGRESSED")
            if entry.floor_breached:
                flags.append(f"BELOW FLOOR {entry.floor}")
            marker = ("  <- " + ", ".join(flags)) if flags else ""
            lines.append(
                f"  - {entry.identity}: {c.baseline_mean:.4f} -> {c.candidate_mean:.4f} "
                f"(diff {c.mean_difference:+.4f}, "
                f"{int(c.confidence * 100)}% CI [{c.ci_low:+.4f}, {c.ci_high:+.4f}], "
                f"p={c.p_value:.4f}, n={c.n_pairs}){marker}"
            )
            if entry.dropped_cases:
                lines.append(
                    f"      {entry.dropped_cases} case(s) excluded: missing or unscored in one run"
                )
        return "\n".join(lines)


async def _run_scores(
    session: AsyncSession, run_id: uuid.UUID
) -> tuple[dict[str, dict[str, float | None]], dict[str, str], dict[str, int]]:
    """Per-evaluator ``{case_id: score}``, config hashes, and error counts."""
    rows = (
        await session.execute(
            select(
                EvaluatorRecord.slug,
                EvaluatorRecord.version,
                EvaluatorRecord.config_hash,
                EvalResult.case_id,
                EvalResult.score,
                EvalResult.error,
            )
            .join(EvaluatorRecord, EvaluatorRecord.id == EvalResult.evaluator_id)
            .where(EvalResult.run_id == run_id)
        )
    ).all()

    scores: dict[str, dict[str, float | None]] = {}
    hashes: dict[str, str] = {}
    errors: dict[str, int] = {}
    for slug, version, config_hash, case_id, score, error in rows:
        identity = f"{slug}@{version}"
        hashes[identity] = config_hash
        bucket = scores.setdefault(identity, {})
        bucket[str(case_id)] = None if error is not None else score
        if error is not None:
            errors[identity] = errors.get(identity, 0) + 1
    return scores, hashes, errors


async def evaluate_gate(
    session: AsyncSession,
    *,
    baseline_run: EvalRun,
    candidate_run: EvalRun,
    policy: GatePolicy | None = None,
) -> GateReport:
    """Compare two runs and return a release verdict."""
    policy = policy or GatePolicy()

    if baseline_run.dataset_id != candidate_run.dataset_id:
        return GateReport(
            verdict=Verdict.INCOMPARABLE,
            reason=(
                "runs are over different datasets; a comparison across datasets "
                "measures the datasets, not the change"
            ),
            baseline_label=baseline_run.system_label,
            candidate_label=candidate_run.system_label,
            per_evaluator=(),
        )

    base_scores, base_hashes, base_errors = await _run_scores(session, baseline_run.id)
    cand_scores, cand_hashes, cand_errors = await _run_scores(session, candidate_run.id)

    shared = sorted(base_scores.keys() & cand_scores.keys())
    if policy.gated_evaluators:
        shared = [identity for identity in shared if identity in policy.gated_evaluators]
        missing = sorted(set(policy.gated_evaluators) - set(shared))
        if missing:
            return GateReport(
                verdict=Verdict.INCOMPARABLE,
                reason=f"gated evaluators absent from one or both runs: {missing}",
                baseline_label=baseline_run.system_label,
                candidate_label=candidate_run.system_label,
                per_evaluator=(),
            )

    if not shared:
        return GateReport(
            verdict=Verdict.INCOMPARABLE,
            reason="the two runs share no evaluator",
            baseline_label=baseline_run.system_label,
            candidate_label=candidate_run.system_label,
            per_evaluator=(),
        )

    # Configuration drift makes the numbers incomparable even when the names
    # match. This is the check that stops a judge swap from being read as a
    # quality change.
    drifted = [
        identity for identity in shared if base_hashes[identity] != cand_hashes[identity]
    ]
    if drifted:
        return GateReport(
            verdict=Verdict.INCOMPARABLE,
            reason=(
                f"evaluator configuration changed between runs for {drifted}; "
                "scores from different configurations are not on the same scale"
            ),
            baseline_label=baseline_run.system_label,
            candidate_label=candidate_run.system_label,
            per_evaluator=(),
        )

    entries: list[EvaluatorVerdict] = []
    any_regressed = False
    any_floor_breached = False
    smallest_mde: float | None = None

    for identity in shared:
        baseline_values, candidate_values, dropped = align_by_case(
            base_scores[identity], cand_scores[identity]
        )

        total = max(len(base_scores[identity]), len(cand_scores[identity]))
        error_count = base_errors.get(identity, 0) + cand_errors.get(identity, 0)
        error_rate = error_count / (2 * total) if total else 0.0
        if error_rate > policy.max_error_rate:
            return GateReport(
                verdict=Verdict.ERROR_RATE,
                reason=(
                    f"{identity} failed to evaluate on {error_rate:.1%} of results "
                    f"(limit {policy.max_error_rate:.1%}); the surviving subset is not "
                    "random, so its mean is not trustworthy"
                ),
                baseline_label=baseline_run.system_label,
                candidate_label=candidate_run.system_label,
                per_evaluator=(),
            )

        if len(baseline_values) < policy.min_pairs:
            mde = (
                minimum_detectable_effect(baseline_values, confidence=policy.confidence)
                if len(baseline_values) >= 2
                else float("inf")
            )
            return GateReport(
                verdict=Verdict.INSUFFICIENT_DATA,
                reason=(
                    f"{identity} has {len(baseline_values)} paired cases, "
                    f"below the minimum of {policy.min_pairs}"
                ),
                baseline_label=baseline_run.system_label,
                candidate_label=candidate_run.system_label,
                per_evaluator=(),
                minimum_detectable_effect=mde,
            )

        comparison = paired_bootstrap(
            baseline_values,
            candidate_values,
            confidence=policy.confidence,
            resamples=policy.resamples,
            seed=policy.seed,
        )
        mde = minimum_detectable_effect(baseline_values, confidence=policy.confidence)
        smallest_mde = mde if smallest_mde is None else min(smallest_mde, mde)

        regressed = comparison.regressed and comparison.mean_difference < -policy.max_mean_drop
        floor = policy.score_floors.get(identity)
        floor_breached = floor is not None and comparison.candidate_mean < floor

        any_regressed = any_regressed or regressed
        any_floor_breached = any_floor_breached or floor_breached
        entries.append(
            EvaluatorVerdict(
                identity=identity,
                comparison=comparison,
                regressed=regressed,
                floor_breached=floor_breached,
                floor=floor,
                dropped_cases=len(dropped),
            )
        )

    if any_regressed or any_floor_breached:
        causes = []
        if any_regressed:
            causes.append("a statistically significant score drop")
        if any_floor_breached:
            causes.append("a score below its absolute floor")
        return GateReport(
            verdict=Verdict.FAIL,
            reason=" and ".join(causes),
            baseline_label=baseline_run.system_label,
            candidate_label=candidate_run.system_label,
            per_evaluator=tuple(entries),
            minimum_detectable_effect=smallest_mde,
        )

    return GateReport(
        verdict=Verdict.PASS,
        reason="no significant regression and no floor breached",
        baseline_label=baseline_run.system_label,
        candidate_label=candidate_run.system_label,
        per_evaluator=tuple(entries),
        minimum_detectable_effect=smallest_mde,
    )
