"""Trace-to-eval promotion, against real Postgres.

The extraction half is a pure function and tested as one. The persistence half
runs against the real database because the properties that matter -- the
uniqueness constraint that makes promotion idempotent, the cascade, the JSONB
round-trip -- are enforced by Postgres.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from agentlens_platform.db.models import Dataset, EvalCase, Tenant
from agentlens_platform.db.postgres import session_scope
from agentlens_platform.tenancy import tenant_session
from agentlens_platform.promotion import (
    ExtractedCase,
    extract_case_from_trace,
    get_or_create_dataset,
    promote_cases,
)


def span(
    *,
    name: str,
    operation: str = "",
    parent: str = "",
    span_id: str = "aa",
    trace_id: str = "t" * 32,
    attributes: dict | None = None,
    start: datetime | None = None,
    **extra,
) -> dict:
    base = {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent,
        "name": name,
        "gen_ai_operation_name": operation,
        "attributes": attributes or {},
        "start_time": start or datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
        "service_name": "svc",
        "status_code": "ok",
    }
    base.update(extra)
    return base


WORKFLOW = span(
    name="invoke_workflow answer",
    operation="invoke_workflow",
    span_id="01",
    attributes={
        "gen_ai.input.messages": '[{"role":"user","content":"What is the refund window?"}]',
        "gen_ai.output.messages": '[{"role":"assistant","content":"Thirty days from delivery."}]',
    },
)
RETRIEVAL = span(
    name="retrieval kb",
    operation="retrieval",
    parent="01",
    span_id="02",
    attributes={
        "gen_ai.retrieval.documents": '[{"id":"d1","score":0.9,"content":"Refunds within 30 days."}]'
    },
)
TOOL = span(name="execute_tool lookup", operation="execute_tool", parent="01", span_id="03",
            gen_ai_tool_name="lookup_order")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def test_workflow_span_is_preferred_as_the_case_source() -> None:
    case = extract_case_from_trace([RETRIEVAL, TOOL, WORKFLOW])
    assert case.inputs["question"] == "What is the refund window?"
    assert case.expected_output == "Thirty days from delivery."


def test_retrieved_context_is_snapshotted_into_the_case() -> None:
    """The evaluator needs the documents as they were, not as retrieval is now."""
    case = extract_case_from_trace([WORKFLOW, RETRIEVAL])
    assert case.context == ("Refunds within 30 days.",)
    assert case.to_case_inputs()["context"] == ["Refunds within 30 days."]


def test_tool_calls_are_captured_so_behaviour_can_be_evaluated() -> None:
    case = extract_case_from_trace([WORKFLOW, TOOL])
    assert case.tools_called == ("lookup_order",)
    assert case.to_case_inputs()["tools_called"] == ["lookup_order"]


def test_production_answer_is_not_marked_as_verified_ground_truth() -> None:
    """The most dangerous possible default would be trusting production output."""
    case = extract_case_from_trace([WORKFLOW])
    assert case.expected_output == "Thirty days from delivery."
    assert case.tags["reference_reviewed"] is False
    assert case.tags["source"] == "production_trace"


def test_external_id_is_deterministic_so_promotion_is_idempotent() -> None:
    first = extract_case_from_trace([WORKFLOW])
    second = extract_case_from_trace([WORKFLOW])
    assert first.external_id == second.external_id
    assert first.external_id.startswith("trace-")


def test_external_id_differs_across_traces() -> None:
    other = span(name="invoke_workflow answer", operation="invoke_workflow",
                 span_id="01", trace_id="b" * 32)
    assert extract_case_from_trace([WORKFLOW]).external_id != extract_case_from_trace([other]).external_id


def test_provenance_is_recorded_for_a_trace_that_will_expire() -> None:
    case = extract_case_from_trace([WORKFLOW])
    assert case.source_trace_id == "t" * 32
    assert case.source_span_id == "01"
    assert case.source_captured_at == datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)


def test_trace_cost_and_errors_are_tagged_for_slicing() -> None:
    spans = [
        span(name="w", operation="invoke_workflow", span_id="01", cost_micro_usd=100),
        span(name="c", operation="chat", parent="01", span_id="02",
             cost_micro_usd=250, status_code="error", gen_ai_request_model="claude-opus-5"),
    ]
    case = extract_case_from_trace(spans)
    assert case.tags["trace_cost_micro_usd"] == 350
    assert case.tags["trace_had_error"] is True
    assert case.tags["models"] == ["claude-opus-5"]
    assert case.tags["span_count"] == 2


def test_a_chat_only_trace_still_promotes() -> None:
    """Not every traced system has a workflow wrapper."""
    chat = span(
        name="chat claude-opus-5",
        operation="chat",
        span_id="09",
        attributes={
            "gen_ai.input.messages": '[{"role":"user","content":"hi"}]',
            "gen_ai.output.messages": '[{"role":"assistant","content":"hello"}]',
        },
    )
    case = extract_case_from_trace([chat])
    assert case.inputs["question"] == "hi"
    assert case.expected_output == "hello"


def test_plain_string_messages_are_handled() -> None:
    """The SDK writes plain text for workflow-level input."""
    plain = span(
        name="invoke_workflow w",
        operation="invoke_workflow",
        span_id="01",
        attributes={"gen_ai.input.messages": "What is the refund window?"},
    )
    assert extract_case_from_trace([plain]).inputs["question"] == "What is the refund window?"


def test_a_trace_with_no_messages_yields_an_empty_input_not_a_crash() -> None:
    bare = span(name="invoke_workflow w", operation="invoke_workflow", span_id="01")
    case = extract_case_from_trace([bare])
    assert case.inputs == {}
    assert case.expected_output is None


def test_promoting_an_empty_trace_is_rejected() -> None:
    with pytest.raises(ValueError):
        extract_case_from_trace([])


def test_context_can_be_excluded() -> None:
    case = extract_case_from_trace([WORKFLOW, RETRIEVAL], include_context=False)
    assert case.context == ()
    assert "context" not in case.to_case_inputs()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


async def _tenant_id(slug: str) -> uuid.UUID:
    async with session_scope() as session:
        row = Tenant(slug=slug, name=slug)
        session.add(row)
        await session.flush()
        return row.id


async def _cleanup(tenant_id: uuid.UUID) -> None:
    async with session_scope() as session:
        obj = await session.get(Tenant, tenant_id)
        if obj is not None:
            await session.delete(obj)


async def test_cases_persist_with_their_snapshot(pg_engine) -> None:
    tenant_id = await _tenant_id(f"p{uuid.uuid4().hex[:14]}")
    try:
        async with tenant_session(tenant_id) as session:
            dataset = await get_or_create_dataset(session, tenant_id=tenant_id, slug="regressions")
            result = await promote_cases(
                session,
                [extract_case_from_trace([WORKFLOW, RETRIEVAL, TOOL])],
                tenant_id=tenant_id,
                dataset=dataset,
            )
            assert result.created_count == 1

        async with tenant_session(tenant_id) as session:
            case = (
                await session.execute(select(EvalCase).where(EvalCase.tenant_id == tenant_id))
            ).scalar_one()
            assert case.inputs["question"] == "What is the refund window?"
            assert case.inputs["context"] == ["Refunds within 30 days."]
            assert case.inputs["tools_called"] == ["lookup_order"]
            assert case.expected_output == "Thirty days from delivery."
            assert case.source_trace_id == "t" * 32
    finally:
        await _cleanup(tenant_id)


async def test_promotion_is_idempotent(pg_engine) -> None:
    """Building a dataset by re-running a query must not duplicate cases."""
    tenant_id = await _tenant_id(f"p{uuid.uuid4().hex[:14]}")
    try:
        candidates = [extract_case_from_trace([WORKFLOW])]
        async with tenant_session(tenant_id) as session:
            dataset = await get_or_create_dataset(session, tenant_id=tenant_id, slug="d")
            first = await promote_cases(session, candidates, tenant_id=tenant_id, dataset=dataset)
        async with tenant_session(tenant_id) as session:
            dataset = await get_or_create_dataset(session, tenant_id=tenant_id, slug="d")
            second = await promote_cases(session, candidates, tenant_id=tenant_id, dataset=dataset)

        assert first.created_count == 1
        assert second.created_count == 0
        assert second.skipped_count == 1

        async with tenant_session(tenant_id) as session:
            total = len(
                (await session.execute(select(EvalCase).where(EvalCase.tenant_id == tenant_id)))
                .scalars()
                .all()
            )
            assert total == 1
    finally:
        await _cleanup(tenant_id)


async def test_a_duplicate_within_one_batch_is_handled(pg_engine) -> None:
    tenant_id = await _tenant_id(f"p{uuid.uuid4().hex[:14]}")
    try:
        candidate = extract_case_from_trace([WORKFLOW])
        async with tenant_session(tenant_id) as session:
            dataset = await get_or_create_dataset(session, tenant_id=tenant_id, slug="d")
            result = await promote_cases(
                session, [candidate, candidate], tenant_id=tenant_id, dataset=dataset
            )
        assert result.created_count == 1
        assert result.skipped_count == 1
    finally:
        await _cleanup(tenant_id)


async def test_get_or_create_dataset_is_stable(pg_engine) -> None:
    tenant_id = await _tenant_id(f"p{uuid.uuid4().hex[:14]}")
    try:
        async with tenant_session(tenant_id) as session:
            first = await get_or_create_dataset(session, tenant_id=tenant_id, slug="d", name="D")
            first_id = first.id
        async with tenant_session(tenant_id) as session:
            again = await get_or_create_dataset(session, tenant_id=tenant_id, slug="d")
            assert again.id == first_id
    finally:
        await _cleanup(tenant_id)


async def test_promoting_nothing_is_a_noop(pg_engine) -> None:
    tenant_id = await _tenant_id(f"p{uuid.uuid4().hex[:14]}")
    try:
        async with tenant_session(tenant_id) as session:
            dataset = await get_or_create_dataset(session, tenant_id=tenant_id, slug="d")
            result = await promote_cases(session, [], tenant_id=tenant_id, dataset=dataset)
        assert result.created_count == 0
    finally:
        await _cleanup(tenant_id)


async def test_a_case_survives_its_source_trace_expiring(pg_engine) -> None:
    """The snapshot rule, stated as a test.

    Spans live under a ClickHouse TTL; cases are permanent. Deleting every span
    -- simulated here by simply never consulting ClickHouse -- must leave the
    case fully usable, with only its provenance pointer dangling.
    """
    tenant_id = await _tenant_id(f"p{uuid.uuid4().hex[:14]}")
    try:
        async with tenant_session(tenant_id) as session:
            dataset = await get_or_create_dataset(session, tenant_id=tenant_id, slug="d")
            await promote_cases(
                session,
                [extract_case_from_trace([WORKFLOW, RETRIEVAL])],
                tenant_id=tenant_id,
                dataset=dataset,
            )

        async with tenant_session(tenant_id) as session:
            case = (
                await session.execute(select(EvalCase).where(EvalCase.tenant_id == tenant_id))
            ).scalar_one()
            # Everything an evaluator needs is present without any span store.
            assert case.inputs["question"]
            assert case.inputs["context"]
            assert case.expected_output
            assert case.source_trace_id is not None  # provenance, possibly dangling
    finally:
        await _cleanup(tenant_id)


def test_content_is_taken_from_the_innermost_span_that_has_it() -> None:
    """Regression: a workflow wrapper often records structure but no content.

    Ranking purely by nesting produced empty cases from real SDK traces -- the
    question and answer were present in ClickHouse on the chat span and none of
    it reached the eval case. Provenance still comes from the outermost span.
    """
    bare_workflow = span(name="invoke_workflow w", operation="invoke_workflow", span_id="01")
    inner_chat = span(
        name="chat claude-opus-5",
        operation="chat",
        parent="01",
        span_id="02",
        attributes={
            "gen_ai.input.messages": '[{"role":"user","content":"refund window?"}]',
            "gen_ai.output.messages": '[{"role":"assistant","content":"Thirty days."}]',
        },
    )
    case = extract_case_from_trace([bare_workflow, inner_chat])
    assert case.inputs["question"] == "refund window?"
    assert case.expected_output == "Thirty days."
    # Provenance still points at the outermost span, not the one content came from.
    assert case.source_span_id == "01"


def test_outer_content_wins_when_both_spans_have_it() -> None:
    """The workflow's view is the user-facing one when it exists."""
    outer = span(
        name="invoke_workflow w",
        operation="invoke_workflow",
        span_id="01",
        attributes={
            "gen_ai.input.messages": "the real user question",
            "gen_ai.output.messages": "the final answer",
        },
    )
    inner = span(
        name="chat m",
        operation="chat",
        parent="01",
        span_id="02",
        attributes={
            "gen_ai.input.messages": "an internal rewritten prompt",
            "gen_ai.output.messages": "an intermediate draft",
        },
    )
    case = extract_case_from_trace([outer, inner])
    assert case.inputs["question"] == "the real user question"
    assert case.expected_output == "the final answer"
