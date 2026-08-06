"""Span storage against the real ClickHouse server.

These assert on engine behaviour that only exists in ClickHouse: positional
insert against the declared column order, ``Map(String, String)`` round-trips,
t-digest quantiles, and the partition/sort layout the queries depend on.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from agentlens_platform.db.clickhouse import SPANS_TABLE, insert_spans
from agentlens_platform.ingest.mapper import resource_spans_to_rows
from agentlens_platform.queries import (
    TimeWindow,
    count_spans,
    fetch_trace,
    list_traces,
    model_usage_summary,
    operation_latency_summary,
)

from .test_mapper import build_resource_spans


def _now_ns(offset_seconds: float = 0.0) -> int:
    base = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    return int(base.timestamp() * 1e9)


def _insert(ch_client, database: str, tenant: str, *resource_spans) -> None:
    rows = []
    for item in resource_spans:
        rows.extend(resource_spans_to_rows(item, tenant_slug=tenant))
    insert_spans(ch_client, rows, database=database)


@pytest.fixture()
def tenant_slug() -> str:
    return f"s{uuid.uuid4().hex[:12]}"


def test_round_trip_preserves_every_promoted_field(ch_client, clickhouse_database, tenant_slug):
    trace_id = uuid.uuid4().hex
    _insert(
        ch_client,
        clickhouse_database,
        tenant_slug,
        build_resource_spans(
            trace_id=bytes.fromhex(trace_id),
            span_id=b"\xaa" * 8,
            name="chat claude-opus-5",
            start_ns=_now_ns(-2),
            end_ns=_now_ns(-1),
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": "anthropic",
                "gen_ai.request.model": "claude-opus-5",
                "gen_ai.usage.input_tokens": 148,
                "gen_ai.usage.output_tokens": 32,
                "agentlens.cost.micro_usd": 1_234,
                "agentlens.cost.status": "ok",
                "custom.thing": "kept",
            },
        ),
    )

    spans = fetch_trace(
        ch_client, database=clickhouse_database, tenant_slug=tenant_slug, trace_id=trace_id
    )
    assert len(spans) == 1
    span = spans[0]
    assert span["gen_ai_request_model"] == "claude-opus-5"
    assert span["gen_ai_usage_input_tokens"] == 148
    assert span["cost_micro_usd"] == 1_234
    assert span["attributes"]["custom.thing"] == "kept"


def test_a_tenant_cannot_read_another_tenants_trace(ch_client, clickhouse_database, tenant_slug):
    """Isolation is a predicate on every read, and it is load-bearing."""
    trace_id = uuid.uuid4().hex
    _insert(
        ch_client,
        clickhouse_database,
        tenant_slug,
        build_resource_spans(trace_id=bytes.fromhex(trace_id), start_ns=_now_ns(-2), end_ns=_now_ns(-1)),
    )
    assert fetch_trace(
        ch_client, database=clickhouse_database, tenant_slug=tenant_slug, trace_id=trace_id
    )
    assert (
        fetch_trace(
            ch_client,
            database=clickhouse_database,
            tenant_slug="someone-else",
            trace_id=trace_id,
        )
        == []
    )


def test_trace_lookup_needs_no_time_predicate(ch_client, clickhouse_database, tenant_slug):
    """A user pasting a trace id does not know when it ran.

    This is the access pattern the bloom-filter skip index exists to serve; the
    sort order (tenant, service, time, trace) cannot.
    """
    trace_id = uuid.uuid4().hex
    old = _now_ns(-60 * 60 * 24 * 3)  # three days back, a different partition
    _insert(
        ch_client,
        clickhouse_database,
        tenant_slug,
        build_resource_spans(trace_id=bytes.fromhex(trace_id), start_ns=old, end_ns=old + 10**9),
    )
    found = fetch_trace(
        ch_client, database=clickhouse_database, tenant_slug=tenant_slug, trace_id=trace_id
    )
    assert len(found) == 1


def test_list_traces_rolls_up_per_trace(ch_client, clickhouse_database, tenant_slug):
    trace_id = uuid.uuid4().hex
    raw = bytes.fromhex(trace_id)
    _insert(
        ch_client,
        clickhouse_database,
        tenant_slug,
        build_resource_spans(
            trace_id=raw,
            span_id=b"\x01" * 8,
            name="invoke_workflow run",
            start_ns=_now_ns(-10),
            end_ns=_now_ns(-1),
        ),
        build_resource_spans(
            trace_id=raw,
            span_id=b"\x02" * 8,
            parent_span_id=b"\x01" * 8,
            name="chat claude-opus-5",
            start_ns=_now_ns(-9),
            end_ns=_now_ns(-5),
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.usage.input_tokens": 100,
                "gen_ai.usage.output_tokens": 20,
                "agentlens.cost.micro_usd": 500,
            },
        ),
    )
    traces = list_traces(
        ch_client,
        database=clickhouse_database,
        tenant_slug=tenant_slug,
        window=TimeWindow.last(hours=1),
    )
    match = [t for t in traces if t["trace_id"] == trace_id]
    assert len(match) == 1
    assert match[0]["span_count"] == 2
    assert match[0]["input_tokens"] == 100
    assert match[0]["cost_micro_usd"] == 500
    assert match[0]["root_name"] == "invoke_workflow run"


def test_model_summary_aggregates_cost_and_latency(ch_client, clickhouse_database, tenant_slug):
    for index in range(20):
        _insert(
            ch_client,
            clickhouse_database,
            tenant_slug,
            build_resource_spans(
                trace_id=uuid.uuid4().bytes,
                span_id=bytes([index % 256]) * 8,
                start_ns=_now_ns(-30),
                end_ns=_now_ns(-30) + (index + 1) * 10**8,
                attributes={
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": "anthropic",
                    "gen_ai.request.model": "claude-opus-5",
                    "gen_ai.usage.input_tokens": 10,
                    "gen_ai.usage.output_tokens": 5,
                    "agentlens.cost.micro_usd": 100,
                    "agentlens.cost.status": "ok",
                },
            ),
        )

    summary = model_usage_summary(
        ch_client,
        database=clickhouse_database,
        tenant_slug=tenant_slug,
        window=TimeWindow.last(hours=1),
    )
    assert len(summary) == 1
    row = summary[0]
    assert row["model"] == "claude-opus-5"
    assert row["call_count"] == 20
    assert row["cost_micro_usd"] == 2_000
    assert row["input_tokens"] == 200
    # Percentiles must be ordered; an inverted pair means the query is wrong.
    assert row["p50_ms"] <= row["p95_ms"] <= row["p99_ms"]


def test_unpriced_calls_are_counted_separately(ch_client, clickhouse_database, tenant_slug):
    """A model with no rate must be visible as unpriced, not as free."""
    _insert(
        ch_client,
        clickhouse_database,
        tenant_slug,
        build_resource_spans(
            trace_id=uuid.uuid4().bytes,
            start_ns=_now_ns(-5),
            end_ns=_now_ns(-4),
            attributes={
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "qwen2.5:3b",
                "agentlens.cost.status": "no_pricing",
            },
        ),
    )
    summary = model_usage_summary(
        ch_client,
        database=clickhouse_database,
        tenant_slug=tenant_slug,
        window=TimeWindow.last(hours=1),
    )
    row = next(r for r in summary if r["model"] == "qwen2.5:3b")
    assert row["unpriced_calls"] == 1
    assert row["cost_micro_usd"] is None or row["cost_micro_usd"] == 0


def test_operation_summary_groups_by_operation(ch_client, clickhouse_database, tenant_slug):
    for operation in ("chat", "execute_tool", "retrieval"):
        _insert(
            ch_client,
            clickhouse_database,
            tenant_slug,
            build_resource_spans(
                trace_id=uuid.uuid4().bytes,
                start_ns=_now_ns(-5),
                end_ns=_now_ns(-4),
                attributes={"gen_ai.operation.name": operation},
            ),
        )
    rows = operation_latency_summary(
        ch_client,
        database=clickhouse_database,
        tenant_slug=tenant_slug,
        window=TimeWindow.last(hours=1),
    )
    assert {r["operation"] for r in rows} == {"chat", "execute_tool", "retrieval"}


def test_time_window_excludes_spans_outside_it(ch_client, clickhouse_database, tenant_slug):
    old = _now_ns(-60 * 60 * 5)
    _insert(
        ch_client,
        clickhouse_database,
        tenant_slug,
        build_resource_spans(trace_id=uuid.uuid4().bytes, start_ns=old, end_ns=old + 10**9),
    )
    recent = list_traces(
        ch_client,
        database=clickhouse_database,
        tenant_slug=tenant_slug,
        window=TimeWindow.last(hours=1),
    )
    assert recent == []
    wide = list_traces(
        ch_client,
        database=clickhouse_database,
        tenant_slug=tenant_slug,
        window=TimeWindow.last(hours=24),
    )
    assert len(wide) == 1


def test_count_spans_is_tenant_scoped(ch_client, clickhouse_database, tenant_slug):
    assert count_spans(ch_client, database=clickhouse_database, tenant_slug=tenant_slug) == 0
    _insert(
        ch_client,
        clickhouse_database,
        tenant_slug,
        build_resource_spans(trace_id=uuid.uuid4().bytes, start_ns=_now_ns(-2), end_ns=_now_ns(-1)),
    )
    assert count_spans(ch_client, database=clickhouse_database, tenant_slug=tenant_slug) == 1


def test_table_layout_is_what_the_queries_assume(ch_client, clickhouse_database):
    """Guard the physical design, not just the results.

    Every query in queries.py is written against this sort order and these skip
    indexes. If someone changes the DDL the queries still return correct rows --
    just by scanning far more data -- so a correctness test would not catch it.
    """
    ddl = ch_client.query(f"SHOW CREATE TABLE {clickhouse_database}.{SPANS_TABLE}").result_rows[0][0]
    assert "ORDER BY (tenant_id, service_name, start_time, trace_id)" in ddl
    assert "PARTITION BY toDate(start_time)" in ddl
    assert "idx_trace_id" in ddl and "bloom_filter" in ddl
    assert "TTL" in ddl


def test_insert_rejects_a_row_of_the_wrong_width(ch_client, clickhouse_database, tenant_slug):
    """Positional insert means a width bug must fail loudly, not shift columns."""
    rows = list(
        resource_spans_to_rows(
            build_resource_spans(start_ns=_now_ns(-2), end_ns=_now_ns(-1)), tenant_slug=tenant_slug
        )
    )
    truncated = [rows[0][:-3]]
    with pytest.raises(Exception):
        insert_spans(ch_client, truncated, database=clickhouse_database)


def test_root_name_survives_tied_timestamps(ch_client, clickhouse_database, tenant_slug):
    """Regression: root detection must not depend on ordering by start_time.

    Nested spans routinely share an identical start timestamp because the OS
    clock does not resolve them apart. With ``argMin(name, start_time)`` the
    rollup returned a child ('retrieval kb') as the trace root. Here every span
    is given the *same* start time deliberately, which is the degenerate case
    the real data produces anyway.
    """
    trace_id = uuid.uuid4().hex
    raw = bytes.fromhex(trace_id)
    tied = _now_ns(-5)
    _insert(
        ch_client,
        clickhouse_database,
        tenant_slug,
        # Child listed first, so a tie broken by insertion order would pick it.
        build_resource_spans(
            trace_id=raw, span_id=b"\x02" * 8, parent_span_id=b"\x01" * 8,
            name="retrieval kb", start_ns=tied, end_ns=tied + 10**8,
        ),
        build_resource_spans(
            trace_id=raw, span_id=b"\x03" * 8, parent_span_id=b"\x01" * 8,
            name="chat claude-opus-5", start_ns=tied, end_ns=tied + 10**8,
        ),
        build_resource_spans(
            trace_id=raw, span_id=b"\x01" * 8, parent_span_id=b"",
            name="invoke_workflow run", start_ns=tied, end_ns=tied + 2 * 10**8,
        ),
    )
    traces = list_traces(
        ch_client, database=clickhouse_database, tenant_slug=tenant_slug,
        window=TimeWindow.last(hours=1),
    )
    match = [t for t in traces if t["trace_id"] == trace_id]
    assert match, f"trace missing; got {traces}"
    assert match[0]["root_name"] == "invoke_workflow run"


def test_partial_trace_without_a_root_still_reports_something(
    ch_client, clickhouse_database, tenant_slug
):
    """A dropped root must degrade to the earliest span, not to an empty name."""
    trace_id = uuid.uuid4().hex
    raw = bytes.fromhex(trace_id)
    _insert(
        ch_client,
        clickhouse_database,
        tenant_slug,
        build_resource_spans(
            trace_id=raw, span_id=b"\x02" * 8, parent_span_id=b"\x09" * 8,
            name="orphaned child", start_ns=_now_ns(-4), end_ns=_now_ns(-3),
        ),
    )
    traces = list_traces(
        ch_client, database=clickhouse_database, tenant_slug=tenant_slug,
        window=TimeWindow.last(hours=1),
    )
    match = [t for t in traces if t["trace_id"] == trace_id]
    assert match[0]["root_name"] == "orphaned child"


def test_a_span_written_now_falls_inside_a_window_ending_now(
    ch_client, clickhouse_database, tenant_slug
):
    """Regression: bound datetimes were truncated to whole seconds.

    The driver renders a bound datetime as 'YYYY-MM-DD HH:MM:SS', discarding
    sub-second precision. Against DateTime64(9) that put any span written
    during the current second *outside* a window ending 'now'. The bounds are
    now passed as integer nanoseconds.
    """
    trace_id = uuid.uuid4().hex
    now_ns = _now_ns(0)
    _insert(
        ch_client,
        clickhouse_database,
        tenant_slug,
        build_resource_spans(
            trace_id=bytes.fromhex(trace_id), start_ns=now_ns, end_ns=now_ns + 10**6
        ),
    )
    window = TimeWindow.last(hours=1)
    # The window end is strictly after the span, but by less than one second --
    # exactly the case second-truncation used to drop.
    assert window.end_ns > now_ns
    assert window.end_ns - now_ns < 1_000_000_000
    traces = list_traces(
        ch_client, database=clickhouse_database, tenant_slug=tenant_slug, window=window
    )
    assert any(t["trace_id"] == trace_id for t in traces), (
        "span written inside the window was excluded from it"
    )
