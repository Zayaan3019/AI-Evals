"""End-to-end: agentlens SDK -> OTLP/gRPC -> ClickHouse -> query API.

Nothing in this path is stubbed. The SDK is the real package from ``sdk/``, the
exporter is the real OTLP gRPC exporter, the server is the real
``grpc.aio`` service, the storage is the real ClickHouse, and the read is the
real query layer. That is the point: each of those boundaries has already
produced a defect in this project, and none of them would have been caught by a
test that faked the other side.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import pytest_asyncio
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

import agentlens
from agentlens.config import AgentLensConfig
from agentlens_platform.db.clickhouse import build_client
from agentlens_platform.ingest.otlp import API_KEY_METADATA, serve_otlp
from agentlens_platform.ingest.writer import SpanWriter
from agentlens_platform.queries import TimeWindow, fetch_trace, list_traces, model_usage_summary

#: A port unlikely to collide with the developer's own stack. The real OTLP
#: port (4317) is deliberately avoided so running the suite cannot interfere
#: with a collector the user has running.
TEST_GRPC_PORT = 14317


@pytest_asyncio.fixture()
async def ingest_server(ch_client, clickhouse_database, settings):
    """A real OTLP gRPC server writing into the scratch database."""
    # The writer gets its own client: the same separation the service makes,
    # for the same reason (one client is one connection pool).
    write_client = build_client(settings, database=clickhouse_database)
    writer = SpanWriter(
        write_client,
        database=clickhouse_database,
        batch_size=100,
        flush_interval_seconds=0.2,
        max_buffered_spans=10_000,
    )
    await writer.start()
    server = await serve_otlp(writer, host="127.0.0.1", port=TEST_GRPC_PORT)
    try:
        yield writer
    finally:
        await server.stop(grace=1.0)
        await writer.aclose()
        write_client.close()


def _export_via_sdk(api_key: str, *, service_name: str) -> str:
    """Run a small agent workload through the real SDK. Returns the trace id."""
    exporter = OTLPSpanExporter(
        endpoint=f"http://127.0.0.1:{TEST_GRPC_PORT}",
        insecure=True,
        headers=((API_KEY_METADATA, api_key),),
        timeout=10,
    )
    agentlens.init(
        AgentLensConfig(
            service_name=service_name,
            tenant="ignored-by-server",
            custom_exporter=exporter,
            schedule_delay_millis=100,
            set_global_tracer_provider=False,
        )
    )
    try:
        with agentlens.workflow("answer_question") as root:
            trace_id = format(root.span.get_span_context().trace_id, "032x")
            with agentlens.agent_step("planner"):
                with agentlens.retrieval("kb", query="what is the refund window?", top_k=3) as r:
                    r.set_documents(
                        [
                            agentlens.Document(id="d1", score=0.91, content="Refunds within 30 days."),
                            agentlens.Document(id="d2", score=0.44, content="Shipping policy."),
                        ]
                    )
                with agentlens.llm_call(
                    "claude-opus-5", provider="anthropic", temperature=0.2
                ) as span:
                    span.set_messages([{"role": "user", "content": "refund window?"}])
                    span.set_response("Thirty days.", finish_reasons=["end_turn"])
                    span.set_usage(148, 32)
            with agentlens.tool_call("lookup_order", arguments={"order_id": "A-1"}) as tool:
                tool.set_result({"status": "shipped"})
        return trace_id
    finally:
        agentlens.shutdown(timeout_millis=5000)


async def _wait_for_spans(writer, ch_client, database, tenant_slug, trace_id, expected: int):
    """Poll until the spans land, or fail with what actually arrived.

    Ingestion is asynchronous by design, so the test must wait rather than
    assume. The assertion message reports the real count so a failure says what
    happened instead of just 'False is not True'.
    """
    for _ in range(60):
        await writer.flush()
        spans = fetch_trace(
            ch_client, database=database, tenant_slug=tenant_slug, trace_id=trace_id
        )
        if len(spans) >= expected:
            return spans
        await asyncio.sleep(0.25)
    spans = fetch_trace(ch_client, database=database, tenant_slug=tenant_slug, trace_id=trace_id)
    raise AssertionError(
        f"expected {expected} spans for trace {trace_id}, got {len(spans)}; "
        f"writer stats: {writer.stats.snapshot()}"
    )


async def test_sdk_trace_reaches_clickhouse_intact(
    ingest_server, ch_client, clickhouse_database, tenant
):
    """The whole pipeline, with the decision tree preserved."""
    tenant_slug, api_key = tenant
    service_name = f"e2e-{uuid.uuid4().hex[:8]}"

    trace_id = await asyncio.get_running_loop().run_in_executor(
        None, lambda: _export_via_sdk(api_key, service_name=service_name)
    )

    spans = await _wait_for_spans(
        ingest_server, ch_client, clickhouse_database, tenant_slug, trace_id, expected=5
    )

    by_name = {span["name"]: span for span in spans}
    assert "invoke_workflow answer_question" in by_name
    assert "invoke_agent planner" in by_name
    assert "chat claude-opus-5" in by_name
    assert "execute_tool lookup_order" in by_name
    assert "retrieval kb" in by_name

    # The tenant is the authenticated one, not what the SDK config claimed.
    llm = by_name["chat claude-opus-5"]
    assert llm["gen_ai_request_model"] == "claude-opus-5"
    assert llm["gen_ai_usage_input_tokens"] == 148
    assert llm["gen_ai_usage_output_tokens"] == 32
    # Cost was computed by the SDK from the versioned pricing table and survived
    # the round trip as an exact integer.
    assert llm["cost_status"] == "ok"
    assert llm["cost_micro_usd"] > 0

    retrieval = by_name["retrieval kb"]
    assert retrieval["retrieval_top_k"] == 3
    assert retrieval["retrieval_document_count"] == 2


async def test_parent_child_structure_survives_the_round_trip(
    ingest_server, ch_client, clickhouse_database, tenant
):
    """A trace whose tree is lost in transit is not a trace."""
    tenant_slug, api_key = tenant
    trace_id = await asyncio.get_running_loop().run_in_executor(
        None, lambda: _export_via_sdk(api_key, service_name=f"e2e-{uuid.uuid4().hex[:8]}")
    )
    spans = await _wait_for_spans(
        ingest_server, ch_client, clickhouse_database, tenant_slug, trace_id, expected=5
    )

    by_id = {span["span_id"]: span for span in spans}
    roots = [s for s in spans if not s["parent_span_id"]]
    assert len(roots) == 1
    assert roots[0]["name"] == "invoke_workflow answer_question"

    # Every non-root parent must resolve to a span in the same trace.
    for span in spans:
        if span["parent_span_id"]:
            assert span["parent_span_id"] in by_id, f"{span['name']} has a dangling parent"

    chat = next(s for s in spans if s["name"] == "chat claude-opus-5")
    planner = next(s for s in spans if s["name"] == "invoke_agent planner")
    assert chat["parent_span_id"] == planner["span_id"]


async def test_an_unauthenticated_export_is_rejected(ingest_server, ch_client, clickhouse_database):
    """A bad key must not write anything, and must fail terminally not retryably."""
    before = ingest_server.stats.snapshot()["spans_accepted"]

    def export_with_bad_key() -> None:
        exporter = OTLPSpanExporter(
            endpoint=f"http://127.0.0.1:{TEST_GRPC_PORT}",
            insecure=True,
            headers=((API_KEY_METADATA, "al_definitely-not-valid"),),
            timeout=5,
        )
        agentlens.init(
            AgentLensConfig(
                service_name="unauthorised",
                custom_exporter=exporter,
                schedule_delay_millis=50,
                set_global_tracer_provider=False,
            )
        )
        try:
            with agentlens.llm_call("claude-opus-5", provider="anthropic") as span:
                span.set_usage(1, 1)
        finally:
            agentlens.shutdown(timeout_millis=3000)

    await asyncio.get_running_loop().run_in_executor(None, export_with_bad_key)
    await asyncio.sleep(0.5)
    await ingest_server.flush()

    assert ingest_server.stats.snapshot()["spans_accepted"] == before


async def test_analytics_reflect_the_ingested_trace(
    ingest_server, ch_client, clickhouse_database, tenant
):
    """The aggregation path, over data that arrived through the real pipeline."""
    tenant_slug, api_key = tenant
    trace_id = await asyncio.get_running_loop().run_in_executor(
        None, lambda: _export_via_sdk(api_key, service_name=f"e2e-{uuid.uuid4().hex[:8]}")
    )
    await _wait_for_spans(
        ingest_server, ch_client, clickhouse_database, tenant_slug, trace_id, expected=5
    )

    window = TimeWindow.last(hours=1)
    models = model_usage_summary(
        ch_client, database=clickhouse_database, tenant_slug=tenant_slug, window=window
    )
    # Not next(): a bare StopIteration inside a coroutine surfaces as an
    # opaque "coroutine raised StopIteration" and hides what actually went
    # wrong. Fail with the rows we did get.
    matching = [m for m in models if m["model"] == "claude-opus-5"]
    assert matching, f"no claude-opus-5 row in model summary; got {models}"
    row = matching[0]
    assert row["call_count"] == 1
    assert row["input_tokens"] == 148
    assert row["cost_micro_usd"] > 0

    traces = list_traces(
        ch_client, database=clickhouse_database, tenant_slug=tenant_slug, window=window
    )
    matched = [t for t in traces if t["trace_id"] == trace_id]
    assert matched, f"trace {trace_id} missing from list_traces; got {traces}"
    match = matched[0]
    assert match["span_count"] == 5
    assert match["root_name"] == "invoke_workflow answer_question"


async def test_redaction_happened_before_the_data_left_the_process(
    ingest_server, ch_client, clickhouse_database, tenant
):
    """PII must never reach the server. Verified by reading what the server stored."""
    tenant_slug, api_key = tenant

    def export_with_pii() -> str:
        exporter = OTLPSpanExporter(
            endpoint=f"http://127.0.0.1:{TEST_GRPC_PORT}",
            insecure=True,
            headers=((API_KEY_METADATA, api_key),),
            timeout=10,
        )
        agentlens.init(
            AgentLensConfig(
                service_name=f"pii-{uuid.uuid4().hex[:8]}",
                custom_exporter=exporter,
                schedule_delay_millis=100,
                set_global_tracer_provider=False,
            )
        )
        try:
            with agentlens.workflow("support") as root:
                trace_id = format(root.span.get_span_context().trace_id, "032x")
                with agentlens.llm_call("claude-opus-5", provider="anthropic") as span:
                    span.set_messages(
                        [{"role": "user", "content": "my email is alice@example.com, card 4111111111111111"}]
                    )
                    span.set_usage(10, 5)
            return trace_id
        finally:
            agentlens.shutdown(timeout_millis=5000)

    trace_id = await asyncio.get_running_loop().run_in_executor(None, export_with_pii)
    spans = await _wait_for_spans(
        ingest_server, ch_client, clickhouse_database, tenant_slug, trace_id, expected=2
    )

    stored = str(spans)
    assert "alice@example.com" not in stored
    assert "4111111111111111" not in stored
    assert "[REDACTED:email]" in stored
    assert "[REDACTED:credit_card]" in stored


async def test_partial_success_is_reported_when_the_buffer_is_full(
    ch_client, clickhouse_database, settings, tenant
):
    """OTLP has a partial_success field; a full buffer must use it, not lie.

    Reporting complete success while dropping spans would make a well-behaved
    client believe data was stored. Reporting total failure would make it retry
    spans that were in fact accepted.
    """
    tenant_slug, api_key = tenant
    write_client = build_client(settings, database=clickhouse_database)
    # A buffer of 2, and a flush interval long enough that nothing drains.
    writer = SpanWriter(
        write_client,
        database=clickhouse_database,
        batch_size=2,
        flush_interval_seconds=3600,
        max_buffered_spans=2,
    )
    server = await serve_otlp(writer, host="127.0.0.1", port=TEST_GRPC_PORT + 1)
    try:
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
            ExportTraceServiceRequest,
        )
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import TraceServiceStub

        import grpc

        from .test_mapper import build_resource_spans
        from .test_storage import _now_ns

        request = ExportTraceServiceRequest(
            resource_spans=[
                build_resource_spans(
                    trace_id=uuid.uuid4().bytes,
                    span_id=uuid.uuid4().bytes[:8],
                    start_ns=_now_ns(-2),
                    end_ns=_now_ns(-1),
                )
                for _ in range(5)
            ]
        )
        async with grpc.aio.insecure_channel(f"127.0.0.1:{TEST_GRPC_PORT + 1}") as channel:
            stub = TraceServiceStub(channel)
            response = await stub.Export(
                request, metadata=((API_KEY_METADATA, api_key),)
            )

        assert response.partial_success.rejected_spans == 3
        assert "buffer full" in response.partial_success.error_message
    finally:
        await server.stop(grace=1.0)
        await writer.aclose()
        write_client.close()
