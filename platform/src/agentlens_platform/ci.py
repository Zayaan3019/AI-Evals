"""CI entrypoint for the regression gate.

Exits non-zero when a release should be blocked, which is the entire contract a
CI system needs. Everything else -- the interval, the p-value, the sample size,
the reason for a refusal -- goes to stdout so a failed build is diagnosable from
the log without re-running anything.

Exit codes are distinct per verdict rather than a blanket 1. A pipeline can then
treat "regressed" differently from "we could not tell", which are genuinely
different situations: the first is a signal to fix the change, the second is a
signal to fix the eval set.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from typing import Sequence

from sqlalchemy import select

from .db.models import EvalRun, Tenant
from .db.postgres import dispose_engine, session_scope
from .tenancy import tenant_session
from .gate import GatePolicy, GateReport, Verdict, evaluate_gate

#: Distinct exit codes so a pipeline can branch on the reason.
EXIT_CODES: dict[Verdict, int] = {
    Verdict.PASS: 0,
    Verdict.FAIL: 1,
    Verdict.INCOMPARABLE: 2,
    Verdict.INSUFFICIENT_DATA: 3,
    Verdict.ERROR_RATE: 4,
}


async def _latest_run(session, *, tenant_id: uuid.UUID, label: str) -> EvalRun | None:
    return (
        await session.execute(
            select(EvalRun)
            .where(
                EvalRun.tenant_id == tenant_id,
                EvalRun.system_label == label,
                EvalRun.status == "succeeded",
            )
            .order_by(EvalRun.started_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def _resolve_run(session, *, tenant_id: uuid.UUID, reference: str) -> EvalRun:
    """Accept either a run UUID or a system label (latest succeeded run)."""
    try:
        run_id = uuid.UUID(reference)
    except ValueError:
        run = await _latest_run(session, tenant_id=tenant_id, label=reference)
        if run is None:
            raise SystemExit(
                f"no succeeded run found with system_label {reference!r}"
            )
        return run

    run = await session.get(EvalRun, run_id)
    if run is None or run.tenant_id != tenant_id:
        raise SystemExit(f"no run {reference} for this tenant")
    return run


def _report_json(report: GateReport) -> str:
    return json.dumps(
        {
            "verdict": report.verdict.value,
            "blocks_release": report.blocks_release,
            "reason": report.reason,
            "baseline": report.baseline_label,
            "candidate": report.candidate_label,
            "minimum_detectable_effect": report.minimum_detectable_effect,
            "evaluators": [
                {
                    "identity": entry.identity,
                    "regressed": entry.regressed,
                    "floor_breached": entry.floor_breached,
                    "dropped_cases": entry.dropped_cases,
                    **(
                        {
                            "baseline_mean": entry.comparison.baseline_mean,
                            "candidate_mean": entry.comparison.candidate_mean,
                            "mean_difference": entry.comparison.mean_difference,
                            "ci_low": entry.comparison.ci_low,
                            "ci_high": entry.comparison.ci_high,
                            "p_value": entry.comparison.p_value,
                            "n_pairs": entry.comparison.n_pairs,
                        }
                        if entry.comparison is not None
                        else {"note": entry.note}
                    ),
                }
                for entry in report.per_evaluator
            ],
        },
        indent=2,
    )


def parse_floors(pairs: Sequence[str] | None) -> dict[str, float]:
    """Parse ``--floor identity=value`` arguments.

    Malformed input is rejected loudly. A silently ignored floor is worse than
    no floor: the operator believes a minimum is being enforced when nothing is.
    """
    floors: dict[str, float] = {}
    for pair in pairs or ():
        identity, separator, raw = pair.partition("=")
        if not separator or not identity:
            raise SystemExit(f"--floor expects IDENTITY=VALUE, got {pair!r}")
        try:
            floors[identity] = float(raw)
        except ValueError:
            raise SystemExit(f"--floor value for {identity!r} is not a number: {raw!r}") from None
    return floors


async def _main(args: argparse.Namespace) -> int:
    # ``tenants`` is not RLS-scoped -- it is the table that resolves a slug to
    # the id every policy compares against, so it is read without a context.
    async with session_scope() as session:
        tenant = (
            await session.execute(select(Tenant).where(Tenant.slug == args.tenant))
        ).scalar_one_or_none()
        if tenant is None:
            print(f"unknown tenant {args.tenant!r}", file=sys.stderr)
            return 2
        tenant_id = tenant.id

    # Everything below reads eval_runs / eval_results / evaluators, all of which
    # are RLS-scoped. Without the tenant context these queries return zero rows
    # and the gate would report INCOMPARABLE on perfectly good data.
    async with tenant_session(tenant_id) as session:
        baseline = await _resolve_run(session, tenant_id=tenant_id, reference=args.baseline)
        candidate = await _resolve_run(session, tenant_id=tenant_id, reference=args.candidate)

        policy = GatePolicy(
            gated_evaluators=tuple(args.evaluator or ()),
            max_mean_drop=args.max_drop,
            score_floors=parse_floors(args.floor),
            min_pairs=args.min_pairs,
            max_error_rate=args.max_error_rate,
            confidence=args.confidence,
            seed=args.seed,
        )
        report = await evaluate_gate(
            session, baseline_run=baseline, candidate_run=candidate, policy=policy
        )

    print(_report_json(report) if args.json else report.render())
    return EXIT_CODES[report.verdict]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agentlens-gate",
        description="Block a release when an eval run regressed against its baseline.",
    )
    parser.add_argument("--tenant", required=True)
    parser.add_argument(
        "--baseline", required=True, help="run UUID, or a system label (latest succeeded run)"
    )
    parser.add_argument("--candidate", required=True, help="run UUID or system label")
    parser.add_argument(
        "--evaluator",
        action="append",
        help="restrict gating to this evaluator identity (slug@version); repeatable. "
        "Naming them is safer than the default of gating on everything, because a "
        "newly added evaluator would otherwise silently start blocking releases.",
    )
    parser.add_argument(
        "--floor",
        action="append",
        metavar="IDENTITY=VALUE",
        help="absolute minimum mean score for an evaluator; repeatable",
    )
    parser.add_argument("--max-drop", type=float, default=0.0)
    parser.add_argument("--min-pairs", type=int, default=20)
    parser.add_argument("--max-error-rate", type=float, default=0.10)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--json", action="store_true", help="machine-readable output")

    args = parser.parse_args(argv)

    async def run() -> int:
        try:
            return await _main(args)
        finally:
            await dispose_engine()

    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())
