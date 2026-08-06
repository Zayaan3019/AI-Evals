"""The regression gate, against real runs in real Postgres.

The refusals matter more than the verdicts. A gate that answers when it cannot
answer honestly is worse than no gate, because it launders a guess into a
release decision.
"""

from __future__ import annotations

import uuid

import pytest

from agentlens_platform.db.models import Dataset, Tenant
from agentlens_platform.db.postgres import session_scope
from agentlens_platform.tenancy import tenant_session
from agentlens_platform.evaluators import build_evaluator
from agentlens_platform.gate import GatePolicy, Verdict, evaluate_gate
from agentlens_platform.promotion import ExtractedCase, get_or_create_dataset, promote_cases
from agentlens_platform.runner import run_eval


async def _tenant() -> uuid.UUID:
    async with session_scope() as session:
        row = Tenant(slug=f"g{uuid.uuid4().hex[:14]}", name="gate test")
        session.add(row)
        await session.flush()
        return row.id


async def _cleanup(tenant_id: uuid.UUID) -> None:
    async with session_scope() as session:
        obj = await session.get(Tenant, tenant_id)
        if obj is not None:
            await session.delete(obj)


async def _dataset(tenant_id: uuid.UUID, n: int, slug: str = "d") -> uuid.UUID:
    cases = [
        ExtractedCase(
            external_id=f"case-{i:03d}",
            inputs={"question": f"q{i}"},
            expected_output=f"answer {i}",
        )
        for i in range(n)
    ]
    async with tenant_session(tenant_id) as session:
        dataset = await get_or_create_dataset(session, tenant_id=tenant_id, slug=slug)
        await promote_cases(session, cases, tenant_id=tenant_id, dataset=dataset)
        return dataset.id


async def _run(tenant_id, dataset_id, system, label, evaluators=None):
    async with tenant_session(tenant_id) as session:
        dataset = await session.get(Dataset, dataset_id)
        return await run_eval(
            session,
            tenant_id=tenant_id,
            dataset=dataset,
            system=system,
            evaluators=evaluators or [build_evaluator("exact_match")],
            system_label=label,
        )


def _correct(inputs):
    return f"answer {inputs['question'][1:]}"


def _wrong_for(fraction: float):
    """A system that answers the first ``fraction`` of cases incorrectly."""

    def system(inputs):
        index = int(inputs["question"][1:])
        return "totally wrong" if index < fraction else _correct(inputs)

    return system


async def test_an_identical_system_passes(pg_engine) -> None:
    tenant_id = await _tenant()
    try:
        dataset_id = await _dataset(tenant_id, 40)
        baseline = await _run(tenant_id, dataset_id, _correct, "v1")
        candidate = await _run(tenant_id, dataset_id, _correct, "v2")
        async with tenant_session(tenant_id) as session:
            report = await evaluate_gate(
                session, baseline_run=baseline, candidate_run=candidate
            )
        assert report.verdict is Verdict.PASS
        assert not report.blocks_release
    finally:
        await _cleanup(tenant_id)


async def test_a_real_regression_fails(pg_engine) -> None:
    tenant_id = await _tenant()
    try:
        dataset_id = await _dataset(tenant_id, 60)
        baseline = await _run(tenant_id, dataset_id, _correct, "good")
        # Half the cases now answer incorrectly.
        candidate = await _run(tenant_id, dataset_id, _wrong_for(30), "regressed")
        async with tenant_session(tenant_id) as session:
            report = await evaluate_gate(
                session, baseline_run=baseline, candidate_run=candidate
            )
        assert report.verdict is Verdict.FAIL
        assert report.blocks_release
        entry = report.per_evaluator[0]
        assert entry.regressed
        assert entry.comparison.mean_difference < 0
        assert entry.comparison.ci_high < 0  # the whole interval is below zero
        assert "REGRESSED" in report.render()
    finally:
        await _cleanup(tenant_id)


async def test_an_improvement_passes(pg_engine) -> None:
    tenant_id = await _tenant()
    try:
        dataset_id = await _dataset(tenant_id, 40)
        baseline = await _run(tenant_id, dataset_id, _wrong_for(20), "bad")
        candidate = await _run(tenant_id, dataset_id, _correct, "fixed")
        async with tenant_session(tenant_id) as session:
            report = await evaluate_gate(
                session, baseline_run=baseline, candidate_run=candidate
            )
        assert report.verdict is Verdict.PASS
        assert report.per_evaluator[0].comparison.improved
    finally:
        await _cleanup(tenant_id)


async def test_too_few_cases_refuses_rather_than_guessing(pg_engine) -> None:
    """'No significant change' on nine cases is not evidence of safety."""
    tenant_id = await _tenant()
    try:
        dataset_id = await _dataset(tenant_id, 9)
        baseline = await _run(tenant_id, dataset_id, _correct, "v1")
        candidate = await _run(tenant_id, dataset_id, _correct, "v2")
        async with tenant_session(tenant_id) as session:
            report = await evaluate_gate(
                session, baseline_run=baseline, candidate_run=candidate
            )
        assert report.verdict is Verdict.INSUFFICIENT_DATA
        assert report.blocks_release
        assert "9 paired cases" in report.reason
        # The report says how small an effect this dataset could even detect.
        assert report.minimum_detectable_effect is not None
    finally:
        await _cleanup(tenant_id)


async def test_a_changed_judge_configuration_is_incomparable(pg_engine) -> None:
    """The check that stops a judge swap being read as a quality change."""
    tenant_id = await _tenant()
    try:
        dataset_id = await _dataset(tenant_id, 30)
        baseline = await _run(
            tenant_id,
            dataset_id,
            _correct,
            "v1",
            evaluators=[build_evaluator("contains", substrings=["answer"])],
        )
        # Same slug, same version, different config -> different config_hash.
        candidate = await _run(
            tenant_id,
            dataset_id,
            _correct,
            "v2",
            evaluators=[build_evaluator("contains", substrings=["totally different"])],
        )
        async with tenant_session(tenant_id) as session:
            report = await evaluate_gate(
                session, baseline_run=baseline, candidate_run=candidate
            )
        assert report.verdict is Verdict.INCOMPARABLE
        assert report.blocks_release
        assert "configuration changed" in report.reason
    finally:
        await _cleanup(tenant_id)


async def test_runs_over_different_datasets_are_incomparable(pg_engine) -> None:
    tenant_id = await _tenant()
    try:
        first = await _dataset(tenant_id, 30, slug="first")
        second = await _dataset(tenant_id, 30, slug="second")
        baseline = await _run(tenant_id, first, _correct, "v1")
        candidate = await _run(tenant_id, second, _correct, "v2")
        async with tenant_session(tenant_id) as session:
            report = await evaluate_gate(
                session, baseline_run=baseline, candidate_run=candidate
            )
        assert report.verdict is Verdict.INCOMPARABLE
        assert "different datasets" in report.reason
    finally:
        await _cleanup(tenant_id)


async def test_no_shared_evaluator_is_incomparable(pg_engine) -> None:
    tenant_id = await _tenant()
    try:
        dataset_id = await _dataset(tenant_id, 30)
        baseline = await _run(
            tenant_id, dataset_id, _correct, "v1", [build_evaluator("exact_match")]
        )
        candidate = await _run(
            tenant_id, dataset_id, _correct, "v2", [build_evaluator("token_f1")]
        )
        async with tenant_session(tenant_id) as session:
            report = await evaluate_gate(
                session, baseline_run=baseline, candidate_run=candidate
            )
        assert report.verdict is Verdict.INCOMPARABLE
        assert "share no evaluator" in report.reason
    finally:
        await _cleanup(tenant_id)


async def test_a_high_evaluator_error_rate_refuses(pg_engine) -> None:
    """The surviving subset is not random, so its mean is not trustworthy."""
    tenant_id = await _tenant()
    try:
        # No expected_output -> exact_match cannot score anything.
        cases = [
            ExtractedCase(external_id=f"c{i:03d}", inputs={"question": f"q{i}"}, expected_output=None)
            for i in range(30)
        ]
        async with tenant_session(tenant_id) as session:
            dataset = await get_or_create_dataset(session, tenant_id=tenant_id, slug="broken")
            await promote_cases(session, cases, tenant_id=tenant_id, dataset=dataset)
            dataset_id = dataset.id

        baseline = await _run(tenant_id, dataset_id, _correct, "v1")
        candidate = await _run(tenant_id, dataset_id, _correct, "v2")
        async with tenant_session(tenant_id) as session:
            report = await evaluate_gate(
                session, baseline_run=baseline, candidate_run=candidate
            )
        assert report.verdict is Verdict.ERROR_RATE
        assert report.blocks_release
        assert "not random" in report.reason
    finally:
        await _cleanup(tenant_id)


async def test_an_absolute_floor_fails_even_without_a_regression(pg_engine) -> None:
    """Catches 'this was already bad and stayed bad'."""
    tenant_id = await _tenant()
    try:
        dataset_id = await _dataset(tenant_id, 40)
        baseline = await _run(tenant_id, dataset_id, _wrong_for(30), "bad-v1")
        candidate = await _run(tenant_id, dataset_id, _wrong_for(30), "bad-v2")
        async with tenant_session(tenant_id) as session:
            report = await evaluate_gate(
                session,
                baseline_run=baseline,
                candidate_run=candidate,
                policy=GatePolicy(score_floors={"exact_match@1": 0.9}),
            )
        assert report.verdict is Verdict.FAIL
        assert report.per_evaluator[0].floor_breached
        assert not report.per_evaluator[0].regressed  # no change between runs
        assert "BELOW FLOOR" in report.render()
    finally:
        await _cleanup(tenant_id)


async def test_a_tolerance_allows_a_small_accepted_drop(pg_engine) -> None:
    tenant_id = await _tenant()
    try:
        dataset_id = await _dataset(tenant_id, 100)
        baseline = await _run(tenant_id, dataset_id, _correct, "v1")
        # 8 of 100 cases regress. Chosen to be genuinely detectable: with only
        # 3 failures, ~4.7% of bootstrap resamples contain none at all, so the
        # 97.5th percentile sits exactly on zero and the drop is correctly
        # *not* significant. The tolerance must be tested against an effect the
        # gate can actually see, or it proves nothing.
        candidate = await _run(tenant_id, dataset_id, _wrong_for(8), "v2")
        async with tenant_session(tenant_id) as session:
            strict = await evaluate_gate(
                session, baseline_run=baseline, candidate_run=candidate
            )
            lenient = await evaluate_gate(
                session,
                baseline_run=baseline,
                candidate_run=candidate,
                policy=GatePolicy(max_mean_drop=0.10),
            )
        assert strict.verdict is Verdict.FAIL
        assert lenient.verdict is Verdict.PASS
    finally:
        await _cleanup(tenant_id)


async def test_gating_a_named_evaluator_that_is_absent_is_incomparable(pg_engine) -> None:
    """An explicitly required evaluator missing must not be silently ignored."""
    tenant_id = await _tenant()
    try:
        dataset_id = await _dataset(tenant_id, 30)
        baseline = await _run(tenant_id, dataset_id, _correct, "v1")
        candidate = await _run(tenant_id, dataset_id, _correct, "v2")
        async with tenant_session(tenant_id) as session:
            report = await evaluate_gate(
                session,
                baseline_run=baseline,
                candidate_run=candidate,
                policy=GatePolicy(gated_evaluators=("faithfulness@1",)),
            )
        assert report.verdict is Verdict.INCOMPARABLE
        assert "absent" in report.reason
    finally:
        await _cleanup(tenant_id)


async def test_report_shows_the_interval_not_only_the_verdict(pg_engine) -> None:
    """A failed build must be diagnosable without re-running it."""
    tenant_id = await _tenant()
    try:
        dataset_id = await _dataset(tenant_id, 50)
        baseline = await _run(tenant_id, dataset_id, _correct, "main@abc123")
        candidate = await _run(tenant_id, dataset_id, _wrong_for(25), "pr-42@def456")
        async with tenant_session(tenant_id) as session:
            report = await evaluate_gate(
                session, baseline_run=baseline, candidate_run=candidate
            )
        text = report.render()
        assert "main@abc123" in text
        assert "pr-42@def456" in text
        assert "95% CI" in text
        assert "p=" in text
        assert "n=50" in text
    finally:
        await _cleanup(tenant_id)


async def test_every_non_pass_verdict_blocks_release() -> None:
    """'We could not tell' must never be treated as 'it is fine'."""
    for verdict in Verdict:
        assert verdict.blocks_release is (verdict is not Verdict.PASS)
