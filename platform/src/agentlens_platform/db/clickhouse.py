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
Tenant isolation: a row policy, not just a predicate
    The API reads spans as a separate user (``agentlens_reader``) that a row
    policy restricts to ``tenant_id = getSetting('SQL_agentlens_tenant')``.
    Every read passes the tenant as that per-query setting (see
    :func:`tenant_query`), so a query that forgets its ``WHERE tenant_id``
    still returns only the caller's rows, and a query sent with no tenant at
    all returns none -- the reader's default for the setting is ''. The same
    guarantee Postgres RLS gives the relational side (tenancy.py).

    One ClickHouse rule makes this subtle: once ANY row policy exists on a
    table, users without a policy see no rows at all. The admin user (schema,
    ingest) therefore gets an explicit allow-all policy; it is the equivalent
    of the Postgres table owner and is never used by request handlers.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import clickhouse_connect
from clickhouse_connect.driver.client import Client

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)

SPANS_TABLE = "spans"

#: Custom setting the row policy reads. ClickHouse accepts unknown settings
#: only under a configured prefix; ``SQL_`` is the server default.
TENANT_SETTING = "SQL_agentlens_tenant"
TENANT_POLICY = "agentlens_tenant_isolation"
ADMIN_POLICY = "agentlens_admin_all"

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


def build_client(
    settings: Settings | None = None,
    *,
    database: str | None = None,
    reader: bool = False,
) -> Client:
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

    ``reader=True`` connects as the tenant-restricted reader instead of the
    admin user; that is the client request handlers must use.
    """
    settings = settings or get_settings()
    return clickhouse_connect.get_client(
        host=settings.clickhouse_host,
        port=settings.clickhouse_port,
        username=settings.clickhouse_reader_user if reader else settings.clickhouse_user,
        password=settings.clickhouse_reader_secret if reader else settings.clickhouse_password,
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


def ensure_reader(
    client: Client,
    database: str,
    *,
    reader_user: str,
    reader_password: str,
    admin_user: str,
) -> None:
    """Create the tenant-restricted reader and the row policies. Idempotent.

    Run as the admin user (it needs access management). The password is reset
    on every start so a rotated secret takes effect without manual steps.
    """
    for name in (reader_user, admin_user):
        if not name.replace("_", "").isalnum():
            raise ValueError(f"unsafe ClickHouse user name {name!r}")
    pw = reader_password.replace("\\", "\\\\").replace("'", "\\'")
    client.command(
        f"CREATE USER IF NOT EXISTS {reader_user} IDENTIFIED WITH sha256_password BY '{pw}' "
        f"SETTINGS {TENANT_SETTING} = ''"
    )
    client.command(f"ALTER USER {reader_user} IDENTIFIED WITH sha256_password BY '{pw}'")
    client.command(f"GRANT SELECT ON {database}.{SPANS_TABLE} TO {reader_user}")
    client.command(
        f"CREATE ROW POLICY OR REPLACE {TENANT_POLICY} ON {database}.{SPANS_TABLE} "
        f"FOR SELECT USING tenant_id = getSetting('{TENANT_SETTING}') TO {reader_user}"
    )
    client.command(
        f"CREATE ROW POLICY OR REPLACE {ADMIN_POLICY} ON {database}.{SPANS_TABLE} "
        f"FOR SELECT USING 1 TO {admin_user}"
    )
    logger.info("clickhouse reader %s restricted by row policy on %s.%s",
                reader_user, database, SPANS_TABLE)


def drop_row_policies(client: Client, database: str) -> None:
    """Remove this database's policies (tests drop scratch databases)."""
    for policy in (TENANT_POLICY, ADMIN_POLICY):
        client.command(f"DROP ROW POLICY IF EXISTS {policy} ON {database}.{SPANS_TABLE}")


def tenant_query(client: Client, sql: str, *, parameters: dict[str, Any]) -> Any:
    """Run a read with the tenant bound for the row policy.

    The tenant comes from the same ``parameters["tenant"]`` the query's own
    ``WHERE tenant_id = %(tenant)s`` uses, so the predicate (which the sort
    key needs for speed) and the policy (which makes isolation hold even
    without the predicate) cannot disagree.
    """
    tenant = parameters.get("tenant")
    if not tenant:
        raise ValueError("tenant_query needs parameters['tenant']")
    # In the query's own SETTINGS clause rather than the driver's settings=
    # argument: clickhouse-connect validates settings against the server's
    # list on the client side and rejects custom SQL_* settings outright.
    # The value is bound like any other parameter (escaped by the driver).
    return client.query(
        f"{sql.rstrip()} SETTINGS {TENANT_SETTING} = %(tenant)s", parameters=parameters
    )


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
