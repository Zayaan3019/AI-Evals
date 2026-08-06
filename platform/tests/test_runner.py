"""Eval run execution against real Postgres.

The properties under test are the ones a regression gate depends on: that
results are paired one-per-(run, case, evaluator), that a failing system under
test still produces a complete result matrix, and that evaluator failures are
kept distinct from low scores in every aggregate.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from agentlens_platform.db.models import (
    Dataset,
    EvalCase,
    EvalResult,
    EvalRun,
    EvaluatorRecord,
    HumanLabel,
    Tenant,
)
from agentlens_platform.db.postgres import session_scope
from agentlens_platform.tenancy import tenant_session
from agentlens_platform.evaluators import build_evaluator
from agentlens_platform.promotion import ExtractedCase, get_or_create_dataset, promote_cases
from agentlens_platform.runner import (
    load_human_labels,
    register_evaluators,
    run_eval,
    summarise_run,
)


async def _make_tenant() -> uuid.UUID:
    async with session_scope() as session:
        row = Tenant(slug=f"r{uuid.uuid4().hex[:14]}", name="runner test")
        session.add(row)
        await session.flush()
        return row.id


async def _cleanup(tenant_id: uuid.UUID) -> None:
    async with session_scope() as session:
        obj = await session.get(Tenant, tenant_id)
        if obj is not None:
            await session.delete(obj)


def _cases(count: int = 3) -> list[ExtractedCase]:
    return [
        ExtractedCase(
            external_id=f"case-{i}",
            inputs={"question": f"q{i}"},
            expected_output=f"answer {i}",
        )
        for i in range(count)
    ]


async def _seed(tenant_id: uuid.UUID, cases=None):
    async with tenant_session(tenant_id) as session:
        dataset = await get_or_create_dataset(session, tenant_id=tenant_id, slug="d", name="D")
        await promote_cases(session, cases or _cases(), tenant_id=tenant_id, dataset=dataset)
        return dataset.id


async def test_a_run_produces_one_result_per_case_and_evaluator(pg_engine) -> None:
    """Pairing is the precondition for the paired bootstrap in Phase 5."""
    tenant_id = await _make_tenant()
    try:
        dataset_id = await _seed(tenant_id)
        async with tenant_session(tenant_id) as session:
            dataset = await session.get(Dataset, dataset_id)
            run = await run_eval(
                session,
                tenant_id=tenant_id,
                dataset=dataset,
                system=lambda inputs: f"answer {inputs['question'][1:]}",
                evaluators=[build_evaluator("exact_match"), build_evaluator("token_f1")],
                system_label="v1",
            )
            run_id = run.id

        async with tenant_session(tenant_id) as session:
            results = (
                await session.execute(select(EvalResult).where(EvalResult.run_id == run_id))
            ).scalars().all()
            assert len(results) == 3 * 2
            triples = {(r.case_id, r.evaluator_id) for r in results}
            assert len(triples) == 6  # no duplicates
    finally:
        await _cleanup(tenant_id)


async def test_a_perfect_system_scores_one(pg_engine) -> None:
    tenant_id = await _make_tenant()
    try:
        dataset_id = await _seed(tenant_id)
        async with tenant_session(tenant_id) as session:
            dataset = await session.get(Dataset, dataset_id)
            run = await run_eval(
                session,
                tenant_id=tenant_id,
                dataset=dataset,
                system=lambda inputs: f"answer {inputs['question'][1:]}",
                evaluators=[build_evaluator("exact_match")],
                system_label="perfect",
            )
            summary = await summarise_run(session, run)

        assert summary.case_count == 3
        assert summary.mean_score("exact_match@1") == pytest.approx(1.0)
        assert summary.per_evaluator["exact_match@1"]["pass_rate"] == pytest.approx(1.0)
        assert summary.per_evaluator["exact_match@1"]["n_errors"] == 0
    finally:
        await _cleanup(tenant_id)


async def test_an_async_system_under_test_is_supported(pg_engine) -> None:
    tenant_id = await _make_tenant()
    try:
        dataset_id = await _seed(tenant_id)

        async def system(inputs):
            return f"answer {inputs['question'][1:]}"

        async with tenant_session(tenant_id) as session:
            dataset = await session.get(Dataset, dataset_id)
            run = await run_eval(
                session,
                tenant_id=tenant_id,
                dataset=dataset,
                system=system,
                evaluators=[build_evaluator("exact_match")],
                system_label="async",
            )
            summary = await summarise_run(session, run)
        assert summary.mean_score("exact_match@1") == pytest.approx(1.0)
    finally:
        await _cleanup(tenant_id)


async def test_a_crashing_system_still_yields_a_complete_result_matrix(pg_engine) -> None:
    """A missing case would silently drop out of every later comparison."""
    tenant_id = await _make_tenant()
    try:
        dataset_id = await _seed(tenant_id)

        def system(inputs):
            if inputs["question"] == "q1":
                raise RuntimeError("provider exploded")
            return f"answer {inputs['question'][1:]}"

        async with tenant_session(tenant_id) as session:
            dataset = await session.get(Dataset, dataset_id)
            run = await run_eval(
                session,
                tenant_id=tenant_id,
                dataset=dataset,
                system=system,
                evaluators=[build_evaluator("exact_match")],
                system_label="flaky",
            )
            run_id = run.id
            summary = await summarise_run(session, run)

        async with tenant_session(tenant_id) as session:
            results = (
                await session.execute(select(EvalResult).where(EvalResult.run_id == run_id))
            ).scalars().all()
            assert len(results) == 3  # every case represented
            errored = [r for r in results if r.error]
            assert len(errored) == 1
            assert "provider exploded" in errored[0].error

        # The failure is excluded from the mean, not averaged in as zero.
        assert summary.per_evaluator["exact_match@1"]["n_errors"] == 1
        assert summary.per_evaluator["exact_match@1"]["n_scored"] == 2
        assert summary.mean_score("exact_match@1") == pytest.approx(1.0)
    finally:
        await _cleanup(tenant_id)


async def test_evaluator_errors_are_excluded_from_the_mean(pg_engine) -> None:
    """An unscoreable case must not look like a failing one."""
    tenant_id = await _make_tenant()
    try:
        # No expected_output, so exact_match cannot score any case.
        cases = [
            ExtractedCase(external_id=f"c{i}", inputs={"question": f"q{i}"}, expected_output=None)
            for i in range(3)
        ]
        dataset_id = await _seed(tenant_id, cases)
        async with tenant_session(tenant_id) as session:
            dataset = await session.get(Dataset, dataset_id)
            run = await run_eval(
                session,
                tenant_id=tenant_id,
                dataset=dataset,
                system=lambda inputs: "anything",
                evaluators=[build_evaluator("exact_match")],
                system_label="unscoreable",
            )
            summary = await summarise_run(session, run)

        entry = summary.per_evaluator["exact_match@1"]
        assert entry["n_errors"] == 3
        assert entry["n_scored"] == 0
        # None, not 0.0 -- there is no evidence, and a zero would be a fabrication.
        assert entry["mean_score"] is None
    finally:
        await _cleanup(tenant_id)


async def test_evaluators_are_registered_with_their_config_hash(pg_engine) -> None:
    """Without the recorded hash the gate cannot refuse an invalid comparison."""
    tenant_id = await _make_tenant()
    try:
        evaluator = build_evaluator("contains", substrings=["refund"])
        async with tenant_session(tenant_id) as session:
            records = await register_evaluators(session, [evaluator], tenant_id=tenant_id)
            assert records[evaluator.identity].config_hash == evaluator.config_hash()
            assert records[evaluator.identity].tier == "deterministic"

        # Re-registering the same evaluator reuses the row.
        async with tenant_session(tenant_id) as session:
            again = await register_evaluators(session, [evaluator], tenant_id=tenant_id)
            rows = (
                await session.execute(
                    select(EvaluatorRecord).where(EvaluatorRecord.tenant_id == tenant_id)
                )
            ).scalars().all()
            assert len(rows) == 1
    finally:
        await _cleanup(tenant_id)


async def test_changed_config_gets_its_own_registration_and_a_warning(
    pg_engine, caplog
) -> None:
    """Results must reference the configuration that actually produced them.

    Reusing the existing row would leave every result labelled with the first
    configuration ever registered, which is what made the gate's drift check
    impossible to trigger. A new row is created instead, and the version reuse
    is warned about because it is still bad practice.
    """
    tenant_id = await _make_tenant()
    try:
        async with tenant_session(tenant_id) as session:
            first = await register_evaluators(
                session, [build_evaluator("contains", substrings=["a"])], tenant_id=tenant_id
            )
            first_hash = first["contains@1"].config_hash

        with caplog.at_level("WARNING"):
            async with tenant_session(tenant_id) as session:
                second = await register_evaluators(
                    session, [build_evaluator("contains", substrings=["b"])], tenant_id=tenant_id
                )
                second_hash = second["contains@1"].config_hash

        assert first_hash != second_hash
        assert any("Bump the version" in r.message for r in caplog.records)

        async with tenant_session(tenant_id) as session:
            rows = (
                await session.execute(
                    select(EvaluatorRecord).where(EvaluatorRecord.tenant_id == tenant_id)
                )
            ).scalars().all()
            # Two registrations, one per configuration -- not one overwritten row.
            assert len(rows) == 2
            assert {r.config_hash for r in rows} == {first_hash, second_hash}
    finally:
        await _cleanup(tenant_id)


async def test_human_labels_load_keyed_on_case_and_output(pg_engine) -> None:
    tenant_id = await _make_tenant()
    try:
        dataset_id = await _seed(tenant_id, _cases(1))
        async with tenant_session(tenant_id) as session:
            case = (
                await session.execute(select(EvalCase).where(EvalCase.tenant_id == tenant_id))
            ).scalar_one()
            session.add(
                HumanLabel(
                    tenant_id=tenant_id,
                    case_id=case.id,
                    graded_output="answer 0",
                    label=1.0,
                    labeller="reviewer@example.com",
                )
            )
            case_id = case.id

        async with tenant_session(tenant_id) as session:
            labels = await load_human_labels(
                session, tenant_id=tenant_id, dataset_id=dataset_id
            )
        assert labels[(str(case_id), "answer 0")] == 1.0
    finally:
        await _cleanup(tenant_id)


async def test_an_empty_dataset_is_rejected(pg_engine) -> None:
    tenant_id = await _make_tenant()
    try:
        async with tenant_session(tenant_id) as session:
            dataset = await get_or_create_dataset(session, tenant_id=tenant_id, slug="empty")
            with pytest.raises(ValueError, match="no cases"):
                await run_eval(
                    session,
                    tenant_id=tenant_id,
                    dataset=dataset,
                    system=lambda inputs: "x",
                    evaluators=[build_evaluator("exact_match")],
                    system_label="v",
                )
    finally:
        await _cleanup(tenant_id)


async def test_a_run_with_no_evaluators_is_rejected(pg_engine) -> None:
    tenant_id = await _make_tenant()
    try:
        dataset_id = await _seed(tenant_id)
        async with tenant_session(tenant_id) as session:
            dataset = await session.get(Dataset, dataset_id)
            with pytest.raises(ValueError):
                await run_eval(
                    session,
                    tenant_id=tenant_id,
                    dataset=dataset,
                    system=lambda inputs: "x",
                    evaluators=[],
                    system_label="v",
                )
    finally:
        await _cleanup(tenant_id)


async def test_run_records_status_and_finish_time(pg_engine) -> None:
    tenant_id = await _make_tenant()
    try:
        dataset_id = await _seed(tenant_id)
        async with tenant_session(tenant_id) as session:
            dataset = await session.get(Dataset, dataset_id)
            run = await run_eval(
                session,
                tenant_id=tenant_id,
                dataset=dataset,
                system=lambda inputs: "x",
                evaluators=[build_evaluator("output_length", min_chars=1)],
                system_label="v1",
                metadata={"git_sha": "abc123"},
            )
            run_id = run.id

        async with tenant_session(tenant_id) as session:
            stored = await session.get(EvalRun, run_id)
            assert stored.status == "succeeded"
            assert stored.finished_at is not None
            assert stored.metadata_["git_sha"] == "abc123"
    finally:
        await _cleanup(tenant_id)


async def test_system_metadata_reaches_behavioural_evaluators(pg_engine) -> None:
    """tool_was_called needs what the agent did, not just what it said."""
    tenant_id = await _make_tenant()
    try:
        dataset_id = await _seed(tenant_id, _cases(1))
        async with tenant_session(tenant_id) as session:
            dataset = await session.get(Dataset, dataset_id)
            run = await run_eval(
                session,
                tenant_id=tenant_id,
                dataset=dataset,
                system=lambda inputs: ("done", {"tools_called": ["lookup_order"]}),
                evaluators=[build_evaluator("tool_was_called", tool_name="lookup_order")],
                system_label="agentic",
            )
            summary = await summarise_run(session, run)
        assert summary.mean_score("tool_was_called@1") == pytest.approx(1.0)
    finally:
        await _cleanup(tenant_id)
