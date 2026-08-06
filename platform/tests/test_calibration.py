"""Judge calibration against real stored labels."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select

from agentlens_platform.calibration import calibrate_judge
from agentlens_platform.db.models import Dataset, EvalCase, HumanLabel, Tenant
from agentlens_platform.db.postgres import session_scope
from agentlens_platform.tenancy import tenant_session
from agentlens_platform.evaluators import build_evaluator
from agentlens_platform.promotion import ExtractedCase, get_or_create_dataset, promote_cases
from agentlens_platform.runner import run_eval


async def _tenant() -> uuid.UUID:
    async with session_scope() as session:
        row = Tenant(slug=f"c{uuid.uuid4().hex[:14]}", name="calibration test")
        session.add(row)
        await session.flush()
        return row.id


async def _cleanup(tenant_id: uuid.UUID) -> None:
    async with session_scope() as session:
        obj = await session.get(Tenant, tenant_id)
        if obj is not None:
            await session.delete(obj)


async def _seed(tenant_id: uuid.UUID, n: int = 20) -> uuid.UUID:
    cases = [
        ExtractedCase(
            external_id=f"case-{i:03d}",
            inputs={"question": f"q{i}"},
            expected_output="the right answer",
        )
        for i in range(n)
    ]
    async with tenant_session(tenant_id) as session:
        dataset = await get_or_create_dataset(session, tenant_id=tenant_id, slug="cal")
        await promote_cases(session, cases, tenant_id=tenant_id, dataset=dataset)
        return dataset.id


async def test_a_judge_agreeing_with_humans_calibrates_well(pg_engine) -> None:
    """Half the cases answer correctly; humans label exactly that."""
    tenant_id = await _tenant()
    try:
        dataset_id = await _seed(tenant_id, 20)

        def system(inputs):
            index = int(inputs["question"][1:])
            return "the right answer" if index % 2 == 0 else "a wrong answer"

        async with tenant_session(tenant_id) as session:
            dataset = await session.get(Dataset, dataset_id)
            await run_eval(
                session,
                tenant_id=tenant_id,
                dataset=dataset,
                system=system,
                evaluators=[build_evaluator("exact_match")],
                system_label="v1",
            )

        # Humans label the same outputs, agreeing with exact_match.
        async with tenant_session(tenant_id) as session:
            cases = (
                await session.execute(
                    select(EvalCase).where(EvalCase.dataset_id == dataset_id)
                )
            ).scalars().all()
            for case in cases:
                index = int(case.inputs["question"][1:])
                output = "the right answer" if index % 2 == 0 else "a wrong answer"
                session.add(
                    HumanLabel(
                        tenant_id=tenant_id,
                        case_id=case.id,
                        graded_output=output,
                        label=1.0 if index % 2 == 0 else 0.0,
                        labeller="reviewer",
                    )
                )

        async with tenant_session(tenant_id) as session:
            results = await calibrate_judge(
                session,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                evaluator_slug="exact_match",
            )

        assert len(results) == 1
        entry = results[0]
        assert entry.n_labelled == 20
        assert entry.calibration.cohens_kappa == pytest.approx(1.0)
        assert entry.calibration.binary_agreement == pytest.approx(1.0)
        assert entry.calibration.mean_absolute_error == pytest.approx(0.0)
        assert "spearman" in entry.render()
    finally:
        await _cleanup(tenant_id)


async def test_a_generous_judge_is_identified_as_biased_high(pg_engine) -> None:
    """The judge passes everything; humans do not. Kappa must expose it."""
    tenant_id = await _tenant()
    try:
        dataset_id = await _seed(tenant_id, 20)

        async with tenant_session(tenant_id) as session:
            dataset = await session.get(Dataset, dataset_id)
            await run_eval(
                session,
                tenant_id=tenant_id,
                dataset=dataset,
                # output_length passes every non-empty answer -> a 'generous judge'.
                system=lambda inputs: "the right answer",
                evaluators=[build_evaluator("output_length", min_chars=1)],
                system_label="v1",
            )

        async with tenant_session(tenant_id) as session:
            cases = (
                await session.execute(
                    select(EvalCase).where(EvalCase.dataset_id == dataset_id)
                )
            ).scalars().all()
            for index, case in enumerate(cases):
                session.add(
                    HumanLabel(
                        tenant_id=tenant_id,
                        case_id=case.id,
                        graded_output="the right answer",
                        # Humans think only half are acceptable.
                        label=1.0 if index % 2 == 0 else 0.0,
                        labeller="reviewer",
                    )
                )

        async with tenant_session(tenant_id) as session:
            results = await calibrate_judge(
                session,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                evaluator_slug="output_length",
            )

        entry = results[0]
        assert entry.calibration.judge_is_biased_high
        assert entry.calibration.binary_agreement == pytest.approx(0.5)
        # No information beyond chance, despite 50% raw agreement.
        assert entry.calibration.cohens_kappa == pytest.approx(0.0)
        assert "generous" in entry.render()
    finally:
        await _cleanup(tenant_id)


async def test_labels_only_count_for_the_output_they_graded(pg_engine) -> None:
    """A label about one answer must not validate a different answer."""
    tenant_id = await _tenant()
    try:
        dataset_id = await _seed(tenant_id, 5)
        async with tenant_session(tenant_id) as session:
            dataset = await session.get(Dataset, dataset_id)
            await run_eval(
                session,
                tenant_id=tenant_id,
                dataset=dataset,
                system=lambda inputs: "todays answer",
                evaluators=[build_evaluator("output_length", min_chars=1)],
                system_label="v1",
            )

        async with tenant_session(tenant_id) as session:
            cases = (
                await session.execute(
                    select(EvalCase).where(EvalCase.dataset_id == dataset_id)
                )
            ).scalars().all()
            for case in cases:
                session.add(
                    HumanLabel(
                        tenant_id=tenant_id,
                        case_id=case.id,
                        # A label about yesterday's different answer.
                        graded_output="yesterdays answer",
                        label=1.0,
                        labeller="reviewer",
                    )
                )

        async with tenant_session(tenant_id) as session:
            results = await calibrate_judge(
                session,
                tenant_id=tenant_id,
                dataset_id=dataset_id,
                evaluator_slug="output_length",
            )

        assert results[0].n_labelled == 0
        assert results[0].calibration is None
        assert "no human-labelled outputs" in results[0].note
    finally:
        await _cleanup(tenant_id)


async def test_each_judge_configuration_is_calibrated_separately(pg_engine) -> None:
    """Averaging two configurations together would describe neither."""
    tenant_id = await _tenant()
    try:
        dataset_id = await _seed(tenant_id, 6)
        for min_chars in (1, 5):
            async with tenant_session(tenant_id) as session:
                dataset = await session.get(Dataset, dataset_id)
                await run_eval(
                    session,
                    tenant_id=tenant_id,
                    dataset=dataset,
                    system=lambda inputs: "abc",
                    evaluators=[build_evaluator("output_length", min_chars=min_chars)],
                    system_label=f"v-{min_chars}",
                )

        async with tenant_session(tenant_id) as session:
            cases = (
                await session.execute(
                    select(EvalCase).where(EvalCase.dataset_id == dataset_id)
                )
            ).scalars().all()
            for case in cases:
                session.add(
                    HumanLabel(
                        tenant_id=tenant_id, case_id=case.id, graded_output="abc",
                        label=1.0, labeller="reviewer",
                    )
                )

        async with tenant_session(tenant_id) as session:
            results = await calibrate_judge(
                session, tenant_id=tenant_id, dataset_id=dataset_id,
                evaluator_slug="output_length",
            )

        # Two configurations -> two registrations -> two calibrations.
        assert len(results) == 2
        assert len({r.config_hash for r in results}) == 2
    finally:
        await _cleanup(tenant_id)
