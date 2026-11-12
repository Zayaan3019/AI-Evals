"""Benchmark runs for a tenant: model comparison on SQuAD, judge calibration on SciEntsBank.

Two experiments, both on public data with human-written ground truth.

**1. Regression gate on a real question: can a cheaper model replace the baseline?**
The QA agent in :mod:`.squad_agent` is run in four configurations on the same
100 SQuAD v1.1 questions. Each answer is scored by exact match and token F1
against a gold answer (the SQuAD metrics) and by an LLM judge. The gate then
compares configurations pairwise. Nothing about the outcome is arranged: the
runs are real model calls and the verdicts are whatever the paired bootstrap
says. One configuration removes retrieval entirely, as an ablation.

Scores use one gold answer per question. Official SQuAD scoring takes the
maximum over all gold answers, so absolute numbers here are lower than
leaderboard numbers; comparisons between runs are unaffected because every run
is scored against the same gold answer.

**2. Judge calibration against human annotators.**
150 student answers from SciEntsBank (SemEval-2013 Task 7), each graded
correct/incorrect against a reference answer by the dataset's human expert
annotators. The LLM judge grades the same answers, and agreement is reported as
Cohen's kappa. Two judge models are calibrated, because a judge is only as
trustworthy as its measured agreement on data like yours.

Model calls go through the record/replay cassette when
``AGENTLENS_JUDGE_CASSETTE`` is set, so a completed benchmark can be reproduced
without calling the provider again::

    GROQ_API_KEY=... AGENTLENS_JUDGE_RECORD=1 \\
        python -m agentlens_platform.examples.benchmark --tenant live
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Callable, Mapping

import agentlens
from sqlalchemy import func, select

from ..db.models import Dataset, EvalCase, EvalRun, HumanLabel, Tenant
from ..db.postgres import dispose_engine, session_scope
from ..evaluators import build_evaluator
from ..promotion import ExtractedCase, get_or_create_dataset, promote_cases
from ..runner import run_eval, summarise_run
from ..tenancy import tenant_session
from .squad_agent import AgentConfig, QAAgent, load_data

DATA_DIR = Path(__file__).parent / "data"
CASSETTE = DATA_DIR / "benchmark_cassette.json"

SQUAD_SLUG = "squad-v1.1-dev-100"
SQUAD_SMOKE_SLUG = "squad-v1.1-dev-10"
SEB_SLUG = "scientsbank-graded-150"
LABELLER = "SemEval-2013 Task 7 expert annotators (2-way scheme)"

BASELINE_MODEL = os.environ.get("AGENTLENS_BENCH_BASELINE", "openai/gpt-oss-120b")
SMALL_MODEL = os.environ.get("AGENTLENS_BENCH_SMALL", "openai/gpt-oss-20b")
THIRD_MODEL = os.environ.get("AGENTLENS_BENCH_THIRD", "qwen/qwen3.8-27b")
JUDGE_MODEL = os.environ.get("AGENTLENS_BENCH_JUDGE", BASELINE_MODEL)

CONFIGS: list[tuple[AgentConfig, str]] = [
    (AgentConfig(model=BASELINE_MODEL), "Baseline: the largest model, with retrieval."),
    (AgentConfig(model=SMALL_MODEL), "Candidate: a model at half the price, same retrieval."),
    (AgentConfig(model=THIRD_MODEL), "Candidate: a different model family, same retrieval."),
    (
        AgentConfig(model=SMALL_MODEL, retrieve=False),
        "Ablation: retrieval removed, so the model answers from memory.",
    ),
]

SystemFn = Callable[[Mapping[str, object]], str]


def _judge(model: str, threshold: float = 0.7):
    return build_evaluator(
        "correctness_vs_reference",
        provider="groq",
        model=model,
        threshold=threshold,
        max_tokens=300,
    )


def _gold(answers: list[str]) -> str:
    """One gold answer per question: the shortest, ties broken alphabetically."""
    return sorted(answers, key=lambda a: (len(a), a))[0]


def squad_cases(limit: int | None = None) -> list[ExtractedCase]:
    questions = load_data()["questions"]
    if limit is not None:
        questions = questions[:limit]
    return [
        ExtractedCase(
            external_id=q["id"],
            inputs={"question": q["question"]},
            expected_output=_gold(q["answers"]),
            tags={"source": "squad_v1.1_dev", "paragraph_id": q["paragraph_id"]},
        )
        for q in questions
    ]


def scientsbank_items() -> list[dict]:
    return json.loads((DATA_DIR / "scientsbank_sample.json").read_text(encoding="utf-8"))["items"]


def scientsbank_cases() -> list[ExtractedCase]:
    return [
        ExtractedCase(
            external_id=item["id"],
            inputs={"question": item["question"], "item": item["id"]},
            expected_output=item["reference_answer"],
            tags={"source": "scientsbank_test_ud", "human_label_5way": item["label_5way"]},
        )
        for item in scientsbank_items()
    ]


def student_answers(inputs: Mapping[str, object]) -> str:
    """The "system" in the calibration runs: the real student answer on record."""
    return _STUDENT[str(inputs["item"])]


_STUDENT = {item["id"]: item["student_answer"] for item in scientsbank_items()}


async def _tenant_id(slug: str):
    async with session_scope() as session:
        tenant = (
            await session.execute(select(Tenant).where(Tenant.slug == slug))
        ).scalar_one_or_none()
        return None if tenant is None else tenant.id


async def _seed_human_labels(tenant_id, dataset_id) -> int:
    items = {item["id"]: item for item in scientsbank_items()}
    added = 0
    async with tenant_session(tenant_id) as session:
        cases = (
            await session.execute(select(EvalCase).where(EvalCase.dataset_id == dataset_id))
        ).scalars().all()
        labelled = set(
            (
                await session.execute(
                    select(HumanLabel.case_id).where(HumanLabel.tenant_id == tenant_id)
                )
            ).scalars().all()
        )
        for case in cases:
            if case.id in labelled:
                continue
            item = items[case.external_id]
            session.add(
                HumanLabel(
                    tenant_id=tenant_id,
                    case_id=case.id,
                    graded_output=item["student_answer"],
                    label=1.0 if item["label_2way"] == "correct" else 0.0,
                    labeller=LABELLER,
                    notes=f"5-way label: {item['label_5way']}",
                )
            )
            added += 1
    return added


async def _run(tenant_id, dataset_id, label: str, system: SystemFn, evaluators, metadata) -> None:
    async with tenant_session(tenant_id) as session:
        dataset = await session.get(Dataset, dataset_id)
        run = await run_eval(
            session,
            tenant_id=tenant_id,
            dataset=dataset,
            system=system,
            evaluators=evaluators,
            system_label=label,
            # Sequential: the provider is rate limited per minute, and a run
            # should not depend on which concurrent call happened to win.
            concurrency=1,
            metadata=metadata,
        )
        summary = await summarise_run(session, run)
    print(f"{label}: {summary.case_count} cases", flush=True)
    for identity, entry in sorted(summary.per_evaluator.items()):
        mean = entry["mean_score"]
        print(
            f"  {identity}: mean={'n/a' if mean is None else f'{mean:.3f}'} "
            f"scored={entry['n_scored']} errors={entry['n_errors']}",
            flush=True,
        )


def _system(config: AgentConfig) -> SystemFn:
    agent = QAAgent(config)
    return lambda inputs: agent.answer(str(inputs["question"]))


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SQuAD and SciEntsBank benchmarks.")
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--force", action="store_true", help="run even if runs already exist")
    parser.add_argument("--only", choices=["squad", "calibration"], default=None)
    args = parser.parse_args(argv)

    os.environ.setdefault("AGENTLENS_JUDGE_CASSETTE", str(CASSETTE))

    # Trace the agent into the platform when a collector is configured, so the
    # benchmark's own model calls appear on the Traces and Analytics pages.
    tracing = bool(os.environ.get("AGENTLENS_ENDPOINT") and os.environ.get("AGENTLENS_API_KEY"))
    if tracing:
        agentlens.init(service_name="squad-qa-agent")

    tenant_id = await _tenant_id(args.tenant)
    if tenant_id is None:
        print(f"unknown tenant {args.tenant!r}; create it first", file=sys.stderr)
        return 2

    async with tenant_session(tenant_id) as session:
        squad = await get_or_create_dataset(
            session, tenant_id=tenant_id, slug=SQUAD_SLUG, name="SQuAD v1.1 dev, 100 questions"
        )
        await promote_cases(session, squad_cases(), tenant_id=tenant_id, dataset=squad)
        smoke = await get_or_create_dataset(
            session, tenant_id=tenant_id, slug=SQUAD_SMOKE_SLUG, name="SQuAD v1.1 dev, 10 questions"
        )
        await promote_cases(session, squad_cases(10), tenant_id=tenant_id, dataset=smoke)
        seb = await get_or_create_dataset(
            session,
            tenant_id=tenant_id,
            slug=SEB_SLUG,
            name="SciEntsBank, 150 human-graded student answers",
        )
        await promote_cases(session, scientsbank_cases(), tenant_id=tenant_id, dataset=seb)
        squad_id, smoke_id, seb_id = squad.id, smoke.id, seb.id
        existing = (await session.execute(select(func.count(EvalRun.id)))).scalar_one()

    print(f"human labels added: {await _seed_human_labels(tenant_id, seb_id)}", flush=True)

    if existing and not args.force:
        print(f"{existing} runs already exist; nothing to do (use --force to add more)")
        await dispose_engine()
        return 0

    def standard():
        # Identical evaluator configurations across comparable runs: the gate
        # refuses to compare runs graded differently.
        return [build_evaluator("exact_match"), build_evaluator("token_f1"), _judge(JUDGE_MODEL)]

    if args.only in (None, "calibration"):
        for model in (JUDGE_MODEL, SMALL_MODEL):
            await _run(
                tenant_id, seb_id, f"SciEntsBank student answers | judge {model}",
                student_answers, [_judge(model)],
                {"purpose": "judge_calibration", "judge_model": model,
                 "description": "Real student answers, graded by the LLM judge and compared "
                                "with the dataset's human annotators."},
            )

    if args.only in (None, "squad"):
        for config, description in CONFIGS:
            await _run(
                tenant_id, squad_id, config.label, _system(config), standard(),
                {"purpose": "model_comparison", "description": description,
                 "model": config.model, "retrieve": config.retrieve, "top_k": config.top_k},
            )

        # Same answers as the baseline, graded with a stricter judge threshold.
        # A different evaluator configuration: the gate must refuse to compare
        # this run with the others (INCOMPARABLE) rather than answer.
        baseline = CONFIGS[0][0]
        await _run(
            tenant_id, squad_id, f"{baseline.label} | judge threshold 0.9", _system(baseline),
            [build_evaluator("exact_match"), build_evaluator("token_f1"), _judge(JUDGE_MODEL, 0.9)],
            {"purpose": "gate_refusal_incomparable",
             "description": "Baseline answers graded with a stricter judge threshold. "
                            "Not comparable with the standard runs."},
        )

        # Two configurations on 10 questions: too few pairs for a verdict, so
        # the gate must report INSUFFICIENT_DATA.
        for config, _ in (CONFIGS[0], CONFIGS[3]):
            await _run(
                tenant_id, smoke_id, f"{config.label} | 10 questions", _system(config), standard(),
                {"purpose": "gate_refusal_insufficient_data",
                 "description": "Ten questions only. Too few for the gate to give a verdict."},
            )

    if tracing:
        agentlens.shutdown()
    await dispose_engine()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
