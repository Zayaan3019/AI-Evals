"""The full loop: SDK -> ClickHouse -> promoted case -> eval run -> scores.

This is the test that justifies the architecture. Everything before it verifies
one boundary; this verifies that a real trace produced by the real SDK, stored
in real ClickHouse, can be turned into a permanent eval case in real Postgres
and scored -- which is the thing the platform exists to do.

It also demonstrates the snapshot rule concretely: after promotion the case is
evaluated with the span store never consulted again.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from sqlalchemy import select

from agentlens_platform.db.models import Dataset, EvalCase, Tenant
from agentlens_platform.db.postgres import session_scope
from agentlens_platform.tenancy import tenant_session
from agentlens_platform.evaluators import build_evaluator
from agentlens_platform.promotion import (
    extract_case_from_trace,
    get_or_create_dataset,
    promote_cases,
)
from agentlens_platform.queries import fetch_trace
from agentlens_platform.runner import run_eval, summarise_run

from .test_e2e import TEST_GRPC_PORT, _export_via_sdk, _wait_for_spans, ingest_server  # noqa: F401


async def _tenant_row(slug: str) -> uuid.UUID:
    async with session_scope() as session:
        row = (
            await session.execute(select(Tenant).where(Tenant.slug == slug))
        ).scalar_one()
        return row.id


async def test_a_real_trace_becomes_a_scored_eval_case(
    ingest_server, ch_client, clickhouse_database, tenant
):
    tenant_slug, api_key = tenant
    tenant_id = await _tenant_row(tenant_slug)

    # 1. A real agent run, exported over real OTLP/gRPC.
    trace_id = await asyncio.get_running_loop().run_in_executor(
        None, lambda: _export_via_sdk(api_key, service_name=f"loop-{uuid.uuid4().hex[:8]}")
    )
    spans = await _wait_for_spans(
        ingest_server, ch_client, clickhouse_database, tenant_slug, trace_id, expected=5
    )

    # 2. Promote it. This is the only time ClickHouse is read.
    candidate = extract_case_from_trace(spans)
    assert candidate.source_trace_id == trace_id
    assert candidate.inputs["question"] == "refund window?"
    assert candidate.context, "retrieved documents must be snapshotted onto the case"
    assert "lookup_order" in candidate.tools_called

    async with tenant_session(tenant_id) as session:
        dataset = await get_or_create_dataset(
            session, tenant_id=tenant_id, slug="from-production", name="From production"
        )
        result = await promote_cases(
            session, [candidate], tenant_id=tenant_id, dataset=dataset
        )
        dataset_id = dataset.id
    assert result.created_count == 1

    # 3. Score it. Note there is no ClickHouse client in scope from here on --
    #    the case is self-sufficient, which is the whole point of the snapshot.
    async with tenant_session(tenant_id) as session:
        dataset = await session.get(Dataset, dataset_id)
        run = await run_eval(
            session,
            tenant_id=tenant_id,
            dataset=dataset,
            # Replay the captured production answer as the system under test.
            system=lambda inputs: ("Thirty days.", {"tools_called": ["lookup_order"]}),
            evaluators=[
                build_evaluator("exact_match"),
                build_evaluator("token_f1"),
                build_evaluator("context_groundedness"),
                build_evaluator("tool_was_called", tool_name="lookup_order"),
                build_evaluator("no_pii_leak"),
            ],
            system_label="replay-of-production",
        )
        summary = await summarise_run(session, run)

    assert summary.case_count == 1
    # The replayed answer matches what production said, so exact_match is 1.0.
    assert summary.mean_score("exact_match@1") == pytest.approx(1.0)
    # The agent did call the tool the trace recorded.
    assert summary.mean_score("tool_was_called@1") == pytest.approx(1.0)
    assert summary.mean_score("no_pii_leak@1") == pytest.approx(1.0)
    # Groundedness scored against the documents captured in the trace.
    grounded = summary.per_evaluator["context_groundedness@1"]
    assert grounded["n_errors"] == 0, "context must have survived promotion"
    assert grounded["mean_score"] is not None


async def test_a_regression_is_visible_as_a_score_drop(
    ingest_server, ch_client, clickhouse_database, tenant
):
    """Two runs over the same promoted case, one degraded.

    This is the shape the Phase 5 gate consumes: identical dataset, identical
    evaluators, two system labels, paired per case.
    """
    tenant_slug, api_key = tenant
    tenant_id = await _tenant_row(tenant_slug)

    trace_id = await asyncio.get_running_loop().run_in_executor(
        None, lambda: _export_via_sdk(api_key, service_name=f"loop-{uuid.uuid4().hex[:8]}")
    )
    spans = await _wait_for_spans(
        ingest_server, ch_client, clickhouse_database, tenant_slug, trace_id, expected=5
    )

    async with tenant_session(tenant_id) as session:
        dataset = await get_or_create_dataset(session, tenant_id=tenant_id, slug="regress")
        await promote_cases(
            session, [extract_case_from_trace(spans)], tenant_id=tenant_id, dataset=dataset
        )
        dataset_id = dataset.id

    evaluators = [build_evaluator("token_f1"), build_evaluator("no_pii_leak")]

    async with tenant_session(tenant_id) as session:
        dataset = await session.get(Dataset, dataset_id)
        good = await run_eval(
            session,
            tenant_id=tenant_id,
            dataset=dataset,
            system=lambda inputs: "Thirty days.",
            evaluators=evaluators,
            system_label="v1-good",
        )
        good_summary = await summarise_run(session, good)

    async with tenant_session(tenant_id) as session:
        dataset = await session.get(Dataset, dataset_id)
        bad = await run_eval(
            session,
            tenant_id=tenant_id,
            dataset=dataset,
            # A plausible regression: wrong answer, and it leaks an email.
            system=lambda inputs: "Contact support@example.com for details.",
            evaluators=evaluators,
            system_label="v2-regressed",
        )
        bad_summary = await summarise_run(session, bad)

    assert good_summary.mean_score("token_f1@1") > bad_summary.mean_score("token_f1@1")
    assert good_summary.mean_score("no_pii_leak@1") == pytest.approx(1.0)
    assert bad_summary.mean_score("no_pii_leak@1") == pytest.approx(0.0)


async def test_promoting_the_same_trace_twice_does_not_duplicate_the_case(
    ingest_server, ch_client, clickhouse_database, tenant
):
    """Datasets get built by re-running a query; that must be safe."""
    tenant_slug, api_key = tenant
    tenant_id = await _tenant_row(tenant_slug)

    trace_id = await asyncio.get_running_loop().run_in_executor(
        None, lambda: _export_via_sdk(api_key, service_name=f"loop-{uuid.uuid4().hex[:8]}")
    )
    spans = await _wait_for_spans(
        ingest_server, ch_client, clickhouse_database, tenant_slug, trace_id, expected=5
    )

    for _ in range(3):
        # Re-read from ClickHouse each time, as a scheduled promotion job would.
        fresh = fetch_trace(
            ch_client,
            database=clickhouse_database,
            tenant_slug=tenant_slug,
            trace_id=trace_id,
        )
        async with tenant_session(tenant_id) as session:
            dataset = await get_or_create_dataset(session, tenant_id=tenant_id, slug="idem")
            await promote_cases(
                session, [extract_case_from_trace(fresh)], tenant_id=tenant_id, dataset=dataset
            )

    async with tenant_session(tenant_id) as session:
        cases = (
            await session.execute(
                select(EvalCase).where(
                    EvalCase.tenant_id == tenant_id,
                    EvalCase.source_trace_id == trace_id,
                )
            )
        ).scalars().all()
    assert len(cases) == 1
