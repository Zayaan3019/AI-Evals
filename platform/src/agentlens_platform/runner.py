"""Eval run execution.

Takes a dataset, a callable representing the system under test, and a set of
evaluators; produces one :class:`EvalResult` per (case, evaluator). The
uniqueness constraint on that triple is what makes results *paired* -- two runs
over the same dataset can be joined case-by-case, which is the precondition for
the paired bootstrap in Phase 5. An unpaired comparison discards the per-case
correlation and needs far more data to detect the same effect.

The system under test is an arbitrary callable, not a subclass or a plugin. It
receives the case inputs and returns text (or text plus metadata). Anything more
opinionated would force users to restructure their agent to be evaluated, which
is exactly the friction that stops teams from evaluating at all.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Mapping, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db.models import Dataset, EvalCase, EvalResult, EvalRun, EvaluatorRecord, HumanLabel
from .evaluators import EvalInput, Evaluator, Score, run_evaluator

logger = logging.getLogger(__name__)

#: What a system under test may return: text, or text plus metadata.
SystemOutput = str | tuple[str, Mapping[str, Any]]
SystemUnderTest = Callable[[Mapping[str, Any]], SystemOutput | Awaitable[SystemOutput]]


@dataclass(slots=True)
class CaseOutcome:
    """One case's result before persistence."""

    case_id: uuid.UUID
    external_id: str
    output: str
    scores: dict[str, Score] = field(default_factory=dict)
    latency_ms: float = 0.0
    cost_micro_usd: int | None = None
    trace_id: str | None = None
    error: str | None = None


async def _invoke(system: SystemUnderTest, inputs: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    """Call the system under test, tolerating sync or async and either shape."""
    result = system(inputs)
    if inspect.isawaitable(result):
        result = await result
    if isinstance(result, tuple):
        output, metadata = result[0], dict(result[1]) if len(result) > 1 else {}
    else:
        output, metadata = result, {}
    return str(output), metadata


async def register_evaluators(
    session: AsyncSession, evaluators: Sequence[Evaluator], *, tenant_id: uuid.UUID
) -> dict[str, EvaluatorRecord]:
    """Ensure every evaluator has a persisted registration, keyed by identity.

    Registration is keyed on (tenant, slug, version, config_hash). Including
    the hash in the key is what makes the gate's drift check possible: results
    point at the configuration that actually produced them, so two runs graded
    under different judge settings reference different rows and the gate can
    see it. Keying on (slug, version) alone silently reused the first
    registration, leaving every result labelled with a configuration it may not
    have been scored under.
    """
    records: dict[str, EvaluatorRecord] = {}
    for evaluator in evaluators:
        config_hash = evaluator.config_hash()
        existing = (
            await session.execute(
                select(EvaluatorRecord).where(
                    EvaluatorRecord.tenant_id == tenant_id,
                    EvaluatorRecord.slug == evaluator.slug,
                    EvaluatorRecord.version == evaluator.version,
                    EvaluatorRecord.config_hash == config_hash,
                )
            )
        ).scalar_one_or_none()

        if existing is None:
            # Warn when the same version already exists under a different
            # configuration. A new row is still created -- the results must
            # record the configuration that actually produced them -- but
            # reusing a version for a changed config is bad practice and the
            # gate will refuse to compare across it.
            clash = (
                await session.execute(
                    select(EvaluatorRecord).where(
                        EvaluatorRecord.tenant_id == tenant_id,
                        EvaluatorRecord.slug == evaluator.slug,
                        EvaluatorRecord.version == evaluator.version,
                    )
                )
            ).scalars().first()
            if clash is not None:
                logger.warning(
                    "evaluator %s@%s already exists with config_hash %s; this run uses %s. "
                    "Bump the version when the configuration changes -- the gate will "
                    "refuse to compare runs across this difference.",
                    evaluator.slug,
                    evaluator.version,
                    clash.config_hash[:12],
                    config_hash[:12],
                )

            existing = EvaluatorRecord(
                tenant_id=tenant_id,
                slug=evaluator.slug,
                tier=evaluator.tier.value,
                version=evaluator.version,
                config=evaluator.describe()["config"],
                config_hash=config_hash,
            )
            session.add(existing)
            await session.flush()
        records[evaluator.identity] = existing
    return records


async def load_human_labels(
    session: AsyncSession, *, tenant_id: uuid.UUID, dataset_id: uuid.UUID
) -> dict[tuple[str, str], float]:
    """All human labels for a dataset, keyed by ``(case_id, graded_output)``.

    Loaded once per run rather than queried per case: a 10,000-case run must be
    one query, not ten thousand.
    """
    rows = (
        await session.execute(
            select(HumanLabel.case_id, HumanLabel.graded_output, HumanLabel.label)
            .join(EvalCase, EvalCase.id == HumanLabel.case_id)
            .where(HumanLabel.tenant_id == tenant_id, EvalCase.dataset_id == dataset_id)
        )
    ).all()
    return {(str(case_id), output): float(label) for case_id, output, label in rows}


async def run_eval(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    dataset: Dataset,
    system: SystemUnderTest,
    evaluators: Sequence[Evaluator],
    system_label: str,
    concurrency: int = 4,
    metadata: Mapping[str, Any] | None = None,
) -> EvalRun:
    """Execute a dataset and persist one result per (case, evaluator).

    ``concurrency`` bounds in-flight cases. Unbounded parallelism against a
    rate-limited provider produces 429s that look like quality failures, and
    against a local model it simply thrashes.
    """
    if not evaluators:
        raise ValueError("an eval run needs at least one evaluator")

    cases = list(
        (
            await session.execute(
                select(EvalCase)
                .where(EvalCase.dataset_id == dataset.id)
                .order_by(EvalCase.external_id)
            )
        )
        .scalars()
        .all()
    )
    if not cases:
        raise ValueError(f"dataset {dataset.slug!r} has no cases")

    records = await register_evaluators(session, evaluators, tenant_id=tenant_id)

    run = EvalRun(
        tenant_id=tenant_id,
        dataset_id=dataset.id,
        system_label=system_label,
        status="running",
        metadata_=dict(metadata or {}),
    )
    session.add(run)
    await session.flush()

    semaphore = asyncio.Semaphore(concurrency)

    async def execute(case: EvalCase) -> CaseOutcome:
        async with semaphore:
            started = time.perf_counter()
            try:
                output, call_metadata = await _invoke(system, case.inputs)
            except Exception as exc:  # noqa: BLE001
                # The system under test failing is a real result, not a crash of
                # the harness. Recorded so the run completes and the failure is
                # visible per case.
                logger.warning("system under test failed on case %s", case.external_id, exc_info=True)
                return CaseOutcome(
                    case_id=case.id,
                    external_id=case.external_id,
                    output="",
                    latency_ms=(time.perf_counter() - started) * 1000,
                    error=f"{type(exc).__name__}: {exc}",
                )

            latency_ms = (time.perf_counter() - started) * 1000
            item = EvalInput(
                output=output,
                inputs=case.inputs,
                expected=case.expected_output,
                context=tuple(case.inputs.get("context") or ()),
                metadata={
                    "case_id": str(case.id),
                    "external_id": case.external_id,
                    "tools_called": call_metadata.get(
                        "tools_called", case.inputs.get("tools_called")
                    ),
                    **call_metadata,
                },
            )
            # Evaluators are synchronous and may be CPU- or network-bound; run
            # them off the event loop so one slow judge does not serialise the
            # whole run.
            scores = await asyncio.to_thread(
                lambda: {ev.identity: run_evaluator(ev, item) for ev in evaluators}
            )
            return CaseOutcome(
                case_id=case.id,
                external_id=case.external_id,
                output=output,
                scores=scores,
                latency_ms=latency_ms,
                cost_micro_usd=call_metadata.get("cost_micro_usd"),
                trace_id=call_metadata.get("trace_id"),
            )

    outcomes = await asyncio.gather(*(execute(case) for case in cases))

    for outcome in outcomes:
        for identity, score in outcome.scores.items():
            session.add(
                EvalResult(
                    tenant_id=tenant_id,
                    run_id=run.id,
                    case_id=outcome.case_id,
                    evaluator_id=records[identity].id,
                    score=score.value,
                    passed=score.passed,
                    explanation=score.explanation,
                    error=score.error,
                    actual_output=outcome.output,
                    latency_ms=outcome.latency_ms,
                    cost_micro_usd=outcome.cost_micro_usd,
                    trace_id=outcome.trace_id,
                )
            )
        if outcome.error is not None and not outcome.scores:
            # The system failed before any evaluator ran. Record the failure
            # against every evaluator so the run has a complete result matrix
            # and the case is not silently missing from comparisons.
            for identity, record in records.items():
                session.add(
                    EvalResult(
                        tenant_id=tenant_id,
                        run_id=run.id,
                        case_id=outcome.case_id,
                        evaluator_id=record.id,
                        score=None,
                        passed=None,
                        error=outcome.error,
                        actual_output="",
                        latency_ms=outcome.latency_ms,
                    )
                )

    run.status = "succeeded"
    run.finished_at = datetime.now(timezone.utc)
    await session.flush()
    return run


@dataclass(frozen=True, slots=True)
class RunSummary:
    """Aggregate view of a completed run."""

    run_id: uuid.UUID
    system_label: str
    case_count: int
    per_evaluator: dict[str, dict[str, Any]]

    def mean_score(self, evaluator_identity: str) -> float | None:
        entry = self.per_evaluator.get(evaluator_identity)
        return entry["mean_score"] if entry else None


async def summarise_run(session: AsyncSession, run: EvalRun) -> RunSummary:
    """Mean score, pass rate and error count per evaluator.

    Scores that failed to run are excluded from the mean and counted separately.
    Averaging a failure in as zero would understate quality; averaging it in as
    a pass would overstate it. Neither is acceptable, so it is reported apart.
    """
    rows = (
        await session.execute(
            select(
                EvaluatorRecord.slug,
                EvaluatorRecord.version,
                EvalResult.score,
                EvalResult.passed,
                EvalResult.error,
            )
            .join(EvaluatorRecord, EvaluatorRecord.id == EvalResult.evaluator_id)
            .where(EvalResult.run_id == run.id)
        )
    ).all()

    grouped: dict[str, list[tuple[float | None, bool | None, str | None]]] = {}
    for slug, version, score, passed, error in rows:
        grouped.setdefault(f"{slug}@{version}", []).append((score, passed, error))

    per_evaluator: dict[str, dict[str, Any]] = {}
    for identity, entries in grouped.items():
        scored = [s for s, _, e in entries if s is not None and e is None]
        judged = [p for _, p, e in entries if p is not None and e is None]
        errors = [e for _, _, e in entries if e is not None]
        per_evaluator[identity] = {
            "n": len(entries),
            "n_scored": len(scored),
            "n_errors": len(errors),
            "mean_score": sum(scored) / len(scored) if scored else None,
            "pass_rate": sum(judged) / len(judged) if judged else None,
        }

    case_count = len({row for row in (await session.execute(
        select(EvalResult.case_id).where(EvalResult.run_id == run.id)
    )).scalars().all()})

    return RunSummary(
        run_id=run.id,
        system_label=run.system_label,
        case_count=case_count,
        per_evaluator=per_evaluator,
    )
