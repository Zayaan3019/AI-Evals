"""Judge calibration against stored human labels.

Answers the only question that makes a model-graded evaluator trustworthy: does
this judge agree with a person, on this dataset, right now?

Calibration is per (judge configuration, dataset). A judge that agrees well on
support tickets may be useless on clinical summaries, and a judge whose prompt
changed is a different judge. Reporting one global "judge accuracy" would hide
both facts.

The labels are joined on ``(case_id, graded_output)``, so a judge is only
credited for agreeing about the *specific output* a human graded. Matching on
case alone would let yesterday's label validate today's different answer, which
manufactures agreement out of nothing -- and this number's entire job is to tell
you whether the agreement is real.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .db.models import EvalCase, EvalResult, EvaluatorRecord, HumanLabel
from .stats import Calibration, calibrate


@dataclass(frozen=True, slots=True)
class JudgeCalibration:
    """Calibration of one judge configuration on one dataset."""

    evaluator_identity: str
    config_hash: str
    dataset_id: uuid.UUID
    calibration: Calibration | None
    n_labelled: int
    note: str | None = None

    def render(self) -> str:
        if self.calibration is None:
            return (
                f"{self.evaluator_identity} (config {self.config_hash[:12]}): "
                f"{self.note or 'not calibratable'}"
            )
        c = self.calibration
        def fmt(value: float | None) -> str:
            return "n/a" if value is None else f"{value:+.3f}"

        direction = "generous" if c.judge_is_biased_high else "harsh"
        return (
            f"{self.evaluator_identity} (config {self.config_hash[:12]}) on {c.n} labelled outputs\n"
            f"  spearman (rank agreement) : {fmt(c.spearman_r)}\n"
            f"  pearson                   : {fmt(c.pearson_r)}\n"
            f"  cohen's kappa (binary)    : {fmt(c.cohens_kappa)}\n"
            f"  raw binary agreement      : {fmt(c.binary_agreement)}\n"
            f"  mean absolute error       : {c.mean_absolute_error:.3f}\n"
            f"  judge mean {c.judge_mean:.3f} vs human mean {c.human_mean:.3f} "
            f"({direction} by {abs(c.judge_mean - c.human_mean):.3f})"
        )


async def calibrate_judge(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    dataset_id: uuid.UUID,
    evaluator_slug: str,
    evaluator_version: str | None = None,
    threshold: float = 0.5,
) -> list[JudgeCalibration]:
    """Calibrate every registered configuration of one judge on one dataset.

    Returns one entry per ``config_hash``, because different configurations are
    different judges and averaging them together would describe none of them.
    """
    query = select(EvaluatorRecord).where(
        EvaluatorRecord.tenant_id == tenant_id,
        EvaluatorRecord.slug == evaluator_slug,
    )
    if evaluator_version is not None:
        query = query.where(EvaluatorRecord.version == evaluator_version)
    records = (await session.execute(query)).scalars().all()

    results: list[JudgeCalibration] = []
    for record in records:
        rows = (
            await session.execute(
                select(EvalResult.score, HumanLabel.label)
                .join(EvalCase, EvalCase.id == EvalResult.case_id)
                .join(
                    HumanLabel,
                    (HumanLabel.case_id == EvalResult.case_id)
                    # The output identity is load-bearing: a label is an opinion
                    # about one specific answer, not about the case forever.
                    & (HumanLabel.graded_output == EvalResult.actual_output),
                )
                .where(
                    EvalResult.evaluator_id == record.id,
                    EvalResult.error.is_(None),
                    EvalResult.score.is_not(None),
                    EvalCase.dataset_id == dataset_id,
                )
            )
        ).all()

        identity = f"{record.slug}@{record.version}"
        if not rows:
            results.append(
                JudgeCalibration(
                    evaluator_identity=identity,
                    config_hash=record.config_hash,
                    dataset_id=dataset_id,
                    calibration=None,
                    n_labelled=0,
                    note="no human-labelled outputs overlap this judge's results",
                )
            )
            continue

        judge_scores = [float(score) for score, _ in rows]
        human_scores = [float(label) for _, label in rows]
        results.append(
            JudgeCalibration(
                evaluator_identity=identity,
                config_hash=record.config_hash,
                dataset_id=dataset_id,
                calibration=calibrate(judge_scores, human_scores, threshold=threshold),
                n_labelled=len(rows),
            )
        )
    return results
