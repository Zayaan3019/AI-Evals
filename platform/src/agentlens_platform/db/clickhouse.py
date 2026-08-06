"""ClickHouse: span storage.

Why spans live here rather than in Postgres is argued in ARCHITECTURE.md
section 1. The short version: spans are append-only, never updated, queried by
aggregate over time windows, and expire wholesale. That is a columnar workload,
and forcing it into a row store costs write amplification, index bloat, and a
vacuum problem that grows with retention.

Schema notes that are load-bearing rather than stylistic:

``ORDER BY (tenant_id, service_name, start_time, trace_id)``
    ClickHouse's primary key is the sort order, and it is sparse -- there is no
    per-row index. Analytics queries ("p95 latency for model X last hour, this
    tenant") all filter tenant first, so tenant leads. Putting ``trace_id``
    first instead would scatter every tenant's rows across the whole table and
    make those scans read everything.

``INDEX idx_trace_id ... TYPE bloom_filter``
    The one access pattern the sort order does *not* serve is "fetch this one
    trace by id", which the UI does constantly. A bloom filter skip index lets
    ClickHouse discard granules that cannot contain the id, turning a full scan
    into a few reads. It is a filter, not an index: false positives are
    possible (harmless, just extra reads), false negatives are not.

``LowCardinality(String)``
    Applied only to columns with genuinely bounded cardinality (tenant, service,
    model, provider, operation). On a high-cardinality column this is slower
    than plain String, so it is not applied to names or ids.

``TTL start_time + INTERVAL N DAY``
    Retention as a background merge. The Postgres equivalent would be a DELETE
    plus vacuum, which competes with live traffic; here expiry is part of the
    merge the engine was going to do anyway.

Materialised GenAI columns
    ``gen_ai.request.model`` and friends are extracted into real columns at
    insert time rather than read out of the attribute map at query time. Map
    lookups cannot use the sparse index and cost a scan of the whole map column;
    a typed column can be filtered and aggregated directly.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import clickhouse_connect
from clickhouse_connect.driver.client import Client

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)

SPANS_TABLE = "spans"

#: Columns written on every insert, in order. The writer builds rows positionally
#: against this tuple, so the order here is the wire contract between
#: ``ingest/mapper.py`` and the table. Changing it requires changing both.
SPAN_COLUMNS: tuple[str, ...] = (
    "tenant_id",
    "project",
    "service_name",
    "service_version",
    "deployment_environment",
    "trace_id",
    "span_id",
    "parent_span_id",
    "name",
    "kind",
    "start_time",
    "end_time",
    "duration_ns",
    "status_code",
    "status_message",
    "gen_ai_operation_name",
    "gen_ai_provider_name",
    "gen_ai_request_model",
    "gen_ai_response_model",
    "gen_ai_request_temperature",
    "gen_ai_usage_input_tokens",
    "gen_ai_usage_output_tokens",
    "gen_ai_tool_name",
    "gen_ai_agent_name",
    "cost_micro_usd",
    "cost_status",
    "cost_pricing_version",
    "prompt_template",
    "prompt_template_version",
    "retrieval_top_k",
    "retrieval_document_count",
    "redaction_applied",
    "redaction_rules",
    "sdk_version",
    "sdk_language",
    "error_type",
    "attributes",
    "resource_attributes",
    "events",
    "ingested_at",
)


def spans_ddl(database: str, retention_days: int) -> str:
    """DDL for the span table.

    Retention is interpolated rather than parameterised because ClickHouse does
    not accept bound parameters in a TTL clause; the value is an int from
    settings, never user input.
    """
    return f"""
    CREATE TABLE IF NOT EXISTS {database}.{SPANS_TABLE}
    (
        tenant_id               LowCardinality(String),
        project                 LowCardinality(String),
        service_name            LowCardinality(String),
        service_version         LowCardinality(String),
        deployment_environment  LowCardinality(String),

        trace_id                String,
        span_id                 String,
        parent_span_id          String,
        name                    String,
        kind                    LowCardinality(String),

        start_time              DateTime64(9, 'UTC'),
        end_time                DateTime64(9, 'UTC'),
        duration_ns             UInt64,

        status_code             LowCardinality(String),
        status_message          String,

        -- GenAI convention fields, promoted to columns so they can be
        -- filtered and aggregated without decoding the attribute map.
        gen_ai_operation_name   LowCardinality(String),
        gen_ai_provider_name    LowCardinality(String),
        gen_ai_request_model    LowCardinality(String),
        gen_ai_response_model   LowCardinality(String),
        gen_ai_request_temperature  Nullable(Float64),
        gen_ai_usage_input_tokens   Nullable(UInt64),
        gen_ai_usage_output_tokens  Nullable(UInt64),
        gen_ai_tool_name        LowCardinality(String),
        gen_ai_agent_name       LowCardinality(String),

        -- Cost is an integer count of micro-USD. Never a float: money that
        -- rounds differently on each aggregation is not auditable.
        cost_micro_usd          Nullable(UInt64),
        cost_status             LowCardinality(String),
        cost_pricing_version    LowCardinality(String),

        prompt_template         String,
        prompt_template_version LowCardinality(String),

        retrieval_top_k         Nullable(UInt32),
        retrieval_document_count Nullable(UInt32),

        redaction_applied       UInt8,
        redaction_rules         Array(String),

        sdk_version             LowCardinality(String),
        sdk_language            LowCardinality(String),
        error_type              LowCardinality(String),

        -- Everything not promoted above. Kept so no captured signal is lost,
        -- including attributes from third-party instrumentation we do not model.
        attributes              Map(String, String),
        resource_attributes     Map(String, String),
        events                  String,

        ingested_at             DateTime64(3, 'UTC') DEFAULT now64(3),

        INDEX idx_trace_id trace_id TYPE bloom_filter(0.01) GRANULARITY 1,
        INDEX idx_span_name name TYPE bloom_filter(0.01) GRANULARITY 4
    )
    ENGINE = MergeTree
    PARTITION BY toDate(start_time)
    ORDER BY (tenant_id, service_name, start_time, trace_id)
    TTL toDateTime(start_time) + INTERVAL {int(retention_days)} DAY
    SETTINGS index_granularity = 8192
    """


def build_client(settings: Settings | None = None, *, database: str | None = None) -> Client:
    """Connect to ClickHouse.

    ``database`` overrides the configured one so tests can target a scratch
    database on the same real server rather than substituting a fake.

    ``autogenerate_session_id=False`` is load-bearing, not tuning. By default
    the driver stamps every request with a session id, and ClickHouse serialises
    requests within a session -- so a second concurrent query on the same client
    raises "Attempt to execute concurrent queries within the same session".
    That is exactly the shape of this service: the ingest writer inserts from a
    worker thread while the API reads on the event loop. Sessions exist for
    temporary tables and session-scoped settings, and we use neither, so
    dropping the session id is the correct fix rather than a workaround.

    Concurrency still wants separate clients per role (see api/app.py), because
    a shared client also shares one HTTP connection pool.
    """
    settings = settings or get_settings()
    return clickhouse_connect.get_client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        username=settings.clickhouse_user,
        password=settings.clickhouse_password,
        database=database or settings.clickhouse_database,
        secure=settings.clickhouse_secure,
        autogenerate_session_id=False,
        # Spans are immutable and inserts are idempotent by (trace_id, span_id)
        # at the query layer, so an async insert that batches server-side is
        # safe and materially reduces part creation.
        settings={"async_insert": 1, "wait_for_async_insert": 1},
    )


def ensure_schema(client: Client, database: str, retention_days: int) -> None:
    """Create the database and span table if absent. Safe to run repeatedly."""
    client.command(f"CREATE DATABASE IF NOT EXISTS {database}")
    client.command(spans_ddl(database, retention_days))
    logger.info("clickhouse schema ready: %s.%s", database, SPANS_TABLE)


def insert_spans(client: Client, rows: Sequence[Sequence[Any]], *, database: str) -> None:
    """Insert positional rows built against :data:`SPAN_COLUMNS`."""
    if not rows:
        return
    client.insert(
        SPANS_TABLE,
        rows,
        column_names=list(SPAN_COLUMNS),
        database=database,
    )
