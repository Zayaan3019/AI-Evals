"""Read APIs over the evaluation side: runs, the regression gate, calibration.

Until this module existed the evaluation half of the platform was reachable
only from the command line (``agentlens_platform.ci``), so a dashboard could
show what an agent did but not whether a new version of it was better. These
handlers expose the same objects the CLI prints, computed by the same
functions -- nothing here re-implements a statistic.

All handlers are GETs and run inside :func:`~..tenancy.tenant_session`, so
Postgres row-level security scopes every query to the caller's tenant.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import func, select

from ..calibration import calibrate_judge
from ..db.models import Dataset, EvalCase, EvalResult, EvalRun, EvaluatorRecord, HumanLabel
from ..gate import GatePolicy, GateReport, evaluate_gate
from ..runner import summarise_run
from ..tenancy import tenant_session
from .deps import ReaderDep

router = APIRouter(prefix="/v1/evals", tags=["evals"])

#: Gate reports are deterministic for a pair of finished runs (seeded bootstrap
#: over stored scores), so a computed report is cached for the process
#: lifetime. Bounded so a tenant with many runs cannot grow it without limit.
_GATE_CACHE: dict[tuple[str, str, str, float], dict[str, Any]] = {}
_GATE_CACHE_MAX = 256
_gate_lock = asyncio.Lock()


def _iso(value) -> str | None:
    return value.isoformat() if value is not None else None


def _gate_payload(report: GateReport) -> dict[str, Any]:
    evaluators: list[dict[str, Any]] = []
    for entry in report.per_evaluator:
        item: dict[str, Any] = {
            "identity": entry.identity,
            "regressed": entry.regressed,
            "floor_breached": entry.floor_breached,
            "floor": entry.floor,
            "dropped_cases": entry.dropped_cases,
            "note": entry.note,
            "comparison": None,
        }
        if entry.comparison is not None:
            c = entry.comparison
            item["comparison"] = {
                "baseline_mean": c.baseline_mean,
                "candidate_mean": c.candidate_mean,
                "mean_difference": c.mean_difference,
                "ci_low": c.ci_low,
                "ci_high": c.ci_high,
                "p_value": c.p_value,
                "n_pairs": c.n_pairs,
            }
        evaluators.append(item)
    return {
        "verdict": report.verdict.value,
        "blocks_release": report.blocks_release,
        "reason": report.reason,
        "baseline": report.baseline_label,
        "candidate": report.candidate_label,
        "minimum_detectable_effect": report.minimum_detectable_effect,
        "evaluators": evaluators,
    }


def _parse_uuid(value: str, what: str) -> uuid.UUID:
    try:
        return uuid.UUID(value)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"{what} is not a run id") from None


@router.get("/runs")
async def list_runs(
    tenant: ReaderDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 30,
) -> dict[str, Any]:
    """Eval runs, newest first, each with its per-evaluator summary."""
    async with tenant_session(tenant.tenant_id) as session:
        rows = (
            await session.execute(
                select(EvalRun, Dataset.slug, Dataset.name)
                .join(Dataset, Dataset.id == EvalRun.dataset_id)
                .order_by(EvalRun.started_at.desc())
                .limit(limit)
            )
        ).all()
        runs: list[dict[str, Any]] = []
        for run, dataset_slug, dataset_name in rows:
            summary = await summarise_run(session, run)
            runs.append(
                {
                    "run_id": str(run.id),
                    "system_label": run.system_label,
                    "status": run.status,
                    "dataset": {"slug": dataset_slug, "name": dataset_name},
                    "started_at": _iso(run.started_at),
                    "finished_at": _iso(run.finished_at),
                    "metadata": run.metadata_ or {},
                    "case_count": summary.case_count,
                    "evaluators": [
                        {"identity": identity, **entry}
                        for identity, entry in sorted(summary.per_evaluator.items())
                    ],
                }
            )
    return {"runs": runs, "count": len(runs)}


@router.get("/runs/{run_id}/results")
async def run_results(
    tenant: ReaderDep,
    run_id: str,
    limit: Annotated[int, Query(ge=1, le=500)] = 200,
) -> dict[str, Any]:
    """Per-case outputs and scores for one run."""
    rid = _parse_uuid(run_id, "run_id")
    async with tenant_session(tenant.tenant_id) as session:
        run = await session.get(EvalRun, rid)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        rows = (
            await session.execute(
                select(
                    EvalCase.external_id,
                    EvalCase.inputs,
                    EvalCase.expected_output,
                    EvalResult.actual_output,
                    EvaluatorRecord.slug,
                    EvaluatorRecord.version,
                    EvalResult.score,
                    EvalResult.passed,
                    EvalResult.explanation,
                    EvalResult.error,
                )
                .join(EvalCase, EvalCase.id == EvalResult.case_id)
                .join(EvaluatorRecord, EvaluatorRecord.id == EvalResult.evaluator_id)
                .where(EvalResult.run_id == rid)
                .order_by(EvalCase.external_id)
            )
        ).all()

    cases: dict[str, dict[str, Any]] = {}
    for external_id, inputs, expected, actual, slug, version, score, passed, explanation, error in rows:
        case = cases.setdefault(
            external_id,
            {
                "external_id": external_id,
                "inputs": inputs,
                "expected_output": expected,
                "actual_output": actual,
                "scores": [],
            },
        )
        case["scores"].append(
            {
                "identity": f"{slug}@{version}",
                "score": score,
                "passed": passed,
                "explanation": explanation,
                "error": error,
            }
        )
    ordered = [cases[key] for key in sorted(cases)][:limit]
    return {"run_id": run_id, "system_label": run.system_label, "cases": ordered, "count": len(ordered)}


@router.get("/gate")
async def gate(
    tenant: ReaderDep,
    baseline: str,
    candidate: str,
    max_drop: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
) -> dict[str, Any]:
    """The regression gate's verdict for one baseline/candidate pair.

    Computed by :func:`~..gate.evaluate_gate` with its default policy: a paired
    bootstrap over cases, 10,000 resamples, 95% interval. The gate may also
    refuse to give a verdict (INCOMPARABLE, INSUFFICIENT_DATA, ERROR_RATE);
    that is returned as-is, never coerced to a pass.
    """
    base_id = _parse_uuid(baseline, "baseline")
    cand_id = _parse_uuid(candidate, "candidate")
    if base_id == cand_id:
        raise HTTPException(status_code=400, detail="baseline and candidate are the same run")

    key = (tenant.tenant_id, str(base_id), str(cand_id), max_drop)
    cached = _GATE_CACHE.get(key)
    if cached is not None:
        return cached

    # One bootstrap at a time: 10,000 resamples is CPU-bound, and a small host
    # asked for several at once would serve none of them quickly.
    async with _gate_lock:
        cached = _GATE_CACHE.get(key)
        if cached is not None:
            return cached
        async with tenant_session(tenant.tenant_id) as session:
            base_run = await session.get(EvalRun, base_id)
            cand_run = await session.get(EvalRun, cand_id)
            if base_run is None or cand_run is None:
                raise HTTPException(status_code=404, detail="run not found")
            report = await evaluate_gate(
                session,
                baseline_run=base_run,
                candidate_run=cand_run,
                policy=GatePolicy(max_mean_drop=max_drop),
            )
        payload = _gate_payload(report)
        payload["baseline_run_id"] = str(base_id)
        payload["candidate_run_id"] = str(cand_id)
        payload["policy"] = {
            "max_mean_drop": max_drop,
            "confidence": GatePolicy().confidence,
            "resamples": GatePolicy().resamples,
            "min_pairs": GatePolicy().min_pairs,
            "max_error_rate": GatePolicy().max_error_rate,
        }
        if len(_GATE_CACHE) >= _GATE_CACHE_MAX:
            _GATE_CACHE.pop(next(iter(_GATE_CACHE)))
        _GATE_CACHE[key] = payload
        return payload


@router.get("/calibration")
async def calibration(tenant: ReaderDep) -> dict[str, Any]:
    """Agreement between each model-graded judge and the stored labels.

    One entry per (judge configuration, dataset): a judge whose model or prompt
    changed is a different judge, and one that agrees on one dataset may not on
    another.
    """
    out: list[dict[str, Any]] = []
    async with tenant_session(tenant.tenant_id) as session:
        tenant_uuid = uuid.UUID(tenant.tenant_id)
        judge_slugs = (
            await session.execute(
                select(EvaluatorRecord.slug)
                .where(EvaluatorRecord.tier == "model_graded")
                .distinct()
            )
        ).scalars().all()
        datasets = (
            await session.execute(
                select(Dataset.id, Dataset.slug, Dataset.name, func.count(HumanLabel.id))
                .join(EvalCase, EvalCase.dataset_id == Dataset.id)
                .join(HumanLabel, HumanLabel.case_id == EvalCase.id)
                .group_by(Dataset.id, Dataset.slug, Dataset.name)
            )
        ).all()
        labellers = (
            await session.execute(select(HumanLabel.labeller).distinct())
        ).scalars().all()
        configs = {
            (record.slug, record.version, record.config_hash): record.config
            for record in (
                await session.execute(
                    select(EvaluatorRecord).where(EvaluatorRecord.tier == "model_graded")
                )
            ).scalars()
        }

        for dataset_id, dataset_slug, dataset_name, label_count in datasets:
            for slug in sorted(judge_slugs):
                for item in await calibrate_judge(
                    session, tenant_id=tenant_uuid, dataset_id=dataset_id, evaluator_slug=slug
                ):
                    version = item.evaluator_identity.split("@", 1)[-1]
                    config = configs.get((slug, version, item.config_hash), {})
                    entry: dict[str, Any] = {
                        "judge": item.evaluator_identity,
                        "config_hash": item.config_hash,
                        "provider": config.get("provider"),
                        "model": config.get("model"),
                        "threshold": config.get("threshold"),
                        "dataset": {"slug": dataset_slug, "name": dataset_name},
                        "labels_in_dataset": label_count,
                        "n_labelled": item.n_labelled,
                        "note": item.note,
                        "calibration": None,
                    }
                    if item.calibration is not None:
                        c = item.calibration
                        entry["calibration"] = {
                            "n": c.n,
                            "cohens_kappa": c.cohens_kappa,
                            "binary_agreement": c.binary_agreement,
                            "spearman_r": c.spearman_r,
                            "pearson_r": c.pearson_r,
                            "mean_absolute_error": c.mean_absolute_error,
                            "judge_mean": c.judge_mean,
                            "human_mean": c.human_mean,
                            "judge_is_biased_high": c.judge_is_biased_high,
                        }
                    out.append(entry)
    return {"judges": out, "labellers": sorted(labellers)}
