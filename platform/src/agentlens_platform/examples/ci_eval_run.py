"""Reference eval run for CI, referenced by ``.github/workflows/eval-gate.yml``.

This exists so the workflow is runnable as written rather than aspirational. It
seeds a dataset and produces the two runs the gate compares.

**The system under test here is a placeholder** -- a deterministic function, not
a model. Replace :func:`system_under_test` with a real call into your agent. It
is a placeholder in the honest sense: the eval machinery, the persistence, the
pairing and the gate are all real and are exercised end to end. What is stubbed
is only the thing that is necessarily yours.

Labels come from the environment so the gate can find the runs by the same names
the workflow passes it. A mismatch there is the most common way a correctly
built gate ends up comparing nothing.
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Mapping

from sqlalchemy import select

from ..db.models import Dataset, Tenant
from ..db.postgres import dispose_engine, session_scope
from ..tenancy import tenant_session
from ..evaluators import build_evaluator
from ..promotion import ExtractedCase, get_or_create_dataset, promote_cases
from ..runner import run_eval, summarise_run

DATASET_SLUG = "ci-regression-set"
CASE_COUNT = int(os.environ.get("AGENTLENS_CI_CASES", "40"))


def _cases() -> list[ExtractedCase]:
    """A small deterministic dataset.

    In a real deployment these are promoted from production traces
    (``agentlens_platform.promotion``) rather than generated. Generated here so
    the workflow has something to run on a fresh database.
    """
    return [
        ExtractedCase(
            external_id=f"case-{index:03d}",
            inputs={"question": f"What is the answer to item {index}?"},
            expected_output=f"The answer to item {index}.",
            tags={"source": "ci_seed"},
        )
        for index in range(CASE_COUNT)
    ]


def system_under_test(inputs: Mapping[str, object]) -> str:
    """Placeholder. Replace with a call into your agent.

    Must be deterministic here so the workflow's baseline and candidate runs
    differ only when the code under test differs -- otherwise CI would fail at
    random and the gate would be ignored within a week.
    """
    question = str(inputs["question"])
    index = question.rsplit(" ", 1)[-1].rstrip("?")
    return f"The answer to item {index}."


async def main() -> int:
    tenant_slug = os.environ.get("AGENTLENS_CI_TENANT", "ci")
    base_label = os.environ.get("BASE_LABEL", "baseline")
    head_label = os.environ.get("HEAD_LABEL", "candidate")

    if base_label == head_label:
        print(
            "BASE_LABEL and HEAD_LABEL are identical; the gate would compare a run "
            "against itself and always pass",
            file=sys.stderr,
        )
        return 2

    # tenants is not RLS-scoped; everything after it is.
    async with session_scope() as session:
        tenant = (
            await session.execute(select(Tenant).where(Tenant.slug == tenant_slug))
        ).scalar_one_or_none()
        if tenant is None:
            print(f"unknown tenant {tenant_slug!r}; create it first", file=sys.stderr)
            return 2
        tenant_id = tenant.id

    async with tenant_session(tenant_id) as session:
        dataset = await get_or_create_dataset(
            session, tenant_id=tenant_id, slug=DATASET_SLUG, name="CI regression set"
        )
        await promote_cases(session, _cases(), tenant_id=tenant_id, dataset=dataset)
        dataset_id = dataset.id

    # Both runs use identical evaluator configurations. The gate refuses to
    # compare across a configuration change, so varying them here would produce
    # INCOMPARABLE rather than a verdict.
    def evaluators():
        return [build_evaluator("exact_match"), build_evaluator("token_f1")]

    for label in (base_label, head_label):
        async with tenant_session(tenant_id) as session:
            dataset = await session.get(Dataset, dataset_id)
            run = await run_eval(
                session,
                tenant_id=tenant_id,
                dataset=dataset,
                system=system_under_test,
                evaluators=evaluators(),
                system_label=label,
                metadata={"source": "ci_eval_run"},
            )
            summary = await summarise_run(session, run)
        print(f"{label}: {summary.case_count} cases")
        for identity, entry in sorted(summary.per_evaluator.items()):
            mean = entry["mean_score"]
            print(
                f"  {identity}: mean={'n/a' if mean is None else f'{mean:.4f}'} "
                f"scored={entry['n_scored']} errors={entry['n_errors']}"
            )

    await dispose_engine()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
