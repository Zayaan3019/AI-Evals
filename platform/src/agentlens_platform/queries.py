"""Read paths over the span store.

Every query is parameterised. ClickHouse's Python driver supports server-side
binding, and string interpolation into a query that carries a tenant identifier
would be a tenant-isolation hole, not merely an injection risk: the whole
isolation model rests on the ``tenant_id`` predicate being exactly what the
authenticated session says it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from clickhouse_connect.driver.client import Client

from .db.clickhouse import SPANS_TABLE

#: Columns returned when reading spans back for display or tree reconstruction.
_SPAN_SELECT = """
    trace_id, span_id, parent_span_id, name, kind,
    start_time, end_time, duration_ns,
    status_code, status_message,
    service_name, project,
    gen_ai_operation_name, gen_ai_provider_name,
    gen_ai_request_model, gen_ai_response_model,
    gen_ai_request_temperature,
    gen_ai_usage_input_tokens, gen_ai_usage_output_tokens,
    gen_ai_tool_name, gen_ai_agent_name,
    cost_micro_usd, cost_status, cost_pricing_version,
    retrieval_top_k, retrieval_document_count,
    redaction_applied, redaction_rules,
    error_type, attributes, events
"""


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _to_epoch_nanos(value: datetime) -> int:
    """Exact nanoseconds since the Unix epoch.

    Integer arithmetic on a timedelta rather than ``value.timestamp() * 1e9``:
    the float route has only ~15-16 significant digits, which is not enough to
    hold microsecond precision on a nanosecond-scale epoch value, so it rounds.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    delta = value - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """A closed time range. Every analytics query takes one.

    Required rather than optional on purpose: the span table is partitioned by
    date and sorted by time, so an unbounded query reads every partition. Making
    the window mandatory means a caller cannot accidentally table-scan.

    Bounds are handed to ClickHouse as integer nanoseconds, not as datetimes.
    The driver renders a bound datetime as ``'YYYY-MM-DD HH:MM:SS'`` -- **whole
    seconds, sub-second precision silently discarded**. Against a
    ``DateTime64(9)`` column that truncation is not cosmetic: a window ending
    "now" became a window ending at the top of the current second, so any span
    written during that second was excluded from its own time range. Caught by
    the end-to-end test, where the span is written at the same instant the
    window is computed.
    """

    start: datetime
    end: datetime

    @classmethod
    def last(cls, **delta: float) -> TimeWindow:
        now = datetime.now(timezone.utc)
        return cls(start=now - timedelta(**delta), end=now)

    @property
    def start_ns(self) -> int:
        return _to_epoch_nanos(self.start)

    @property
    def end_ns(self) -> int:
        return _to_epoch_nanos(self.end)


#: Time predicate shared by every windowed query. ``fromUnixTimestamp64Nano``
#: reconstructs an exact DateTime64(9) from the integer bound, so the comparison
#: is precision-preserving and free of any string-parsing or timezone
#: interpretation on the server side.
_TIME_PREDICATE = (
    "start_time >= fromUnixTimestamp64Nano(toInt64(%(start_ns)s)) "
    "AND start_time <= fromUnixTimestamp64Nano(toInt64(%(end_ns)s))"
)


def _rows_to_dicts(result: Any) -> list[dict[str, Any]]:
    columns = result.column_names
    return [dict(zip(columns, row)) for row in result.result_rows]


def fetch_trace(
    client: Client, *, database: str, tenant_slug: str, trace_id: str
) -> list[dict[str, Any]]:
    """All spans of one trace, ordered so parents precede children in time.

    No time predicate: a user pasting a trace id has no idea when it ran. This
    is the access pattern the ``idx_trace_id`` bloom filter exists for -- the
    sort order cannot serve it, and without the skip index this would read every
    partition in the retention window.
    """
    result = client.query(
        f"SELECT {_SPAN_SELECT} FROM {database}.{SPANS_TABLE} "
        "WHERE tenant_id = %(tenant)s AND trace_id = %(trace_id)s "
        "ORDER BY start_time ASC",
        parameters={"tenant": tenant_slug, "trace_id": trace_id},
    )
    return _rows_to_dicts(result)


def list_traces(
    client: Client,
    *,
    database: str,
    tenant_slug: str,
    window: TimeWindow,
    service_name: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Root spans in the window, newest first, with per-trace rollups.

    Aggregating in ClickHouse rather than fetching spans and summing in Python
    is the entire reason spans live there: this touches millions of rows and
    returns tens.
    """
    conditions = ["tenant_id = %(tenant)s", _TIME_PREDICATE]
    parameters: dict[str, Any] = {
        "tenant": tenant_slug,
        "start_ns": window.start_ns,
        "end_ns": window.end_ns,
        "limit": limit,
        "offset": offset,
    }
    if service_name:
        conditions.append("service_name = %(service)s")
        parameters["service"] = service_name

    where = " AND ".join(conditions)
    result = client.query(
        f"""
        SELECT
            trace_id,
            min(start_time)                                    AS started_at,
            max(end_time)                                      AS ended_at,
            count()                                            AS span_count,
            -- The root is defined by structure -- no parent -- not by being
            -- earliest. Ordering by start_time cannot work: nested spans
            -- routinely carry byte-identical start timestamps because the OS
            -- clock does not resolve them apart (measured: three of four
            -- nested spans shared one nanosecond value), so argMin returns an
            -- arbitrary member of the tie. It really did pick a child over the
            -- workflow root. Fall back to earliest only when the root is
            -- genuinely absent, which is what a partial trace looks like after
            -- a dropped span.
            if(
                countIf(parent_span_id = '') > 0,
                argMinIf(name, start_time, parent_span_id = ''),
                argMin(name, start_time)
            )                                                   AS root_name,
            any(service_name)                                   AS service_name,
            sum(gen_ai_usage_input_tokens)                       AS input_tokens,
            sum(gen_ai_usage_output_tokens)                      AS output_tokens,
            sum(cost_micro_usd)                                  AS cost_micro_usd,
            countIf(status_code = 'error')                       AS error_count,
            max(duration_ns)                                     AS root_duration_ns
        FROM {database}.{SPANS_TABLE}
        WHERE {where}
        GROUP BY trace_id
        ORDER BY started_at DESC
        LIMIT %(limit)s OFFSET %(offset)s
        """,
        parameters=parameters,
    )
    return _rows_to_dicts(result)


def model_usage_summary(
    client: Client, *, database: str, tenant_slug: str, window: TimeWindow
) -> list[dict[str, Any]]:
    """Per-model cost, token and latency rollup.

    ``quantileTDigest`` rather than an exact quantile: exact percentiles require
    holding every value, which defeats the point of aggregating server-side.
    t-digest is accurate in the tails, which is where latency questions live.
    """
    result = client.query(
        f"""
        SELECT
            gen_ai_request_model                                AS model,
            gen_ai_provider_name                                AS provider,
            count()                                             AS call_count,
            sum(gen_ai_usage_input_tokens)                      AS input_tokens,
            sum(gen_ai_usage_output_tokens)                     AS output_tokens,
            sum(cost_micro_usd)                                 AS cost_micro_usd,
            countIf(cost_status = 'no_pricing')                 AS unpriced_calls,
            round(quantileTDigest(0.50)(duration_ns) / 1e6, 3)  AS p50_ms,
            round(quantileTDigest(0.95)(duration_ns) / 1e6, 3)  AS p95_ms,
            round(quantileTDigest(0.99)(duration_ns) / 1e6, 3)  AS p99_ms,
            countIf(status_code = 'error')                      AS error_count
        FROM {database}.{SPANS_TABLE}
        WHERE tenant_id = %(tenant)s
          AND {_TIME_PREDICATE}
          AND gen_ai_operation_name IN ('chat', 'text_completion', 'generate_content')
        GROUP BY model, provider
        ORDER BY call_count DESC
        """,
        parameters={
            "tenant": tenant_slug,
            "start_ns": window.start_ns,
            "end_ns": window.end_ns,
        },
    )
    return _rows_to_dicts(result)


def operation_latency_summary(
    client: Client, *, database: str, tenant_slug: str, window: TimeWindow
) -> list[dict[str, Any]]:
    """Latency and error rate per operation type."""
    result = client.query(
        f"""
        SELECT
            gen_ai_operation_name                               AS operation,
            count()                                             AS span_count,
            round(quantileTDigest(0.50)(duration_ns) / 1e6, 3)  AS p50_ms,
            round(quantileTDigest(0.95)(duration_ns) / 1e6, 3)  AS p95_ms,
            round(quantileTDigest(0.99)(duration_ns) / 1e6, 3)  AS p99_ms,
            countIf(status_code = 'error')                      AS error_count
        FROM {database}.{SPANS_TABLE}
        WHERE tenant_id = %(tenant)s
          AND {_TIME_PREDICATE}
          AND gen_ai_operation_name != ''
        GROUP BY operation
        ORDER BY span_count DESC
        """,
        parameters={
            "tenant": tenant_slug,
            "start_ns": window.start_ns,
            "end_ns": window.end_ns,
        },
    )
    return _rows_to_dicts(result)


def count_spans(client: Client, *, database: str, tenant_slug: str) -> int:
    """Total stored spans for a tenant. Used by health checks and tests."""
    result = client.query(
        f"SELECT count() FROM {database}.{SPANS_TABLE} WHERE tenant_id = %(tenant)s",
        parameters={"tenant": tenant_slug},
    )
    return int(result.result_rows[0][0])
