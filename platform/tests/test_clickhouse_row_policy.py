"""Span isolation enforced by ClickHouse itself, not by query authors.

The predicate ``WHERE tenant_id = ...`` is on every query in queries.py, and
test_storage.py checks it. These tests check the stronger property: the API's
ClickHouse user cannot see another tenant's spans even through a query that
has no tenant predicate at all, because a row policy on the table filters by
the tenant bound to the request. Run against the real server; a fake would not
have row policies.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from agentlens_platform.db.clickhouse import (
    SPANS_TABLE,
    TENANT_SETTING,
    insert_spans,
    tenant_query,
)
from agentlens_platform.ingest.mapper import resource_spans_to_rows
from agentlens_platform.queries import TimeWindow, count_spans, fetch_trace, list_traces

from .test_mapper import build_resource_spans


def _now_ns(offset_seconds: float = 0.0) -> int:
    return int((datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)).timestamp() * 1e9)


@pytest.fixture()
def two_tenants(ch_client, clickhouse_database):
    """Two tenants with one trace each, written by the admin (ingest) user."""
    tenants = {}
    for _ in range(2):
        slug = f"rp{uuid.uuid4().hex[:10]}"
        trace_id = uuid.uuid4().hex
        rows = list(resource_spans_to_rows(
            build_resource_spans(
                trace_id=bytes.fromhex(trace_id), start_ns=_now_ns(-2), end_ns=_now_ns(-1),
            ),
            tenant_slug=slug,
        ))
        insert_spans(ch_client, rows, database=clickhouse_database)
        tenants[slug] = trace_id
    return tenants


def test_a_query_without_a_tenant_predicate_still_sees_one_tenant(
    ch_reader, clickhouse_database, two_tenants
):
    for slug in two_tenants:
        rows = ch_reader.query(
            f"SELECT DISTINCT tenant_id FROM {clickhouse_database}.{SPANS_TABLE} "
            f"SETTINGS {TENANT_SETTING} = %(t)s",
            parameters={"t": slug},
        ).result_rows
        assert rows == [(slug,)]


def test_naming_another_tenant_in_the_query_returns_nothing(
    ch_reader, clickhouse_database, two_tenants
):
    mine, theirs = list(two_tenants)
    rows = ch_reader.query(
        f"SELECT count() FROM {clickhouse_database}.{SPANS_TABLE} WHERE tenant_id = %(t)s "
        f"SETTINGS {TENANT_SETTING} = %(mine)s",
        parameters={"t": theirs, "mine": mine},
    ).result_rows
    assert rows == [(0,)]


def test_no_tenant_bound_means_no_rows(ch_reader, clickhouse_database, two_tenants):
    """The reader's default for the setting is '' -- fail closed, not open."""
    total = int(
        ch_reader.query(f"SELECT count() FROM {clickhouse_database}.{SPANS_TABLE}").result_rows[0][0]
    )
    assert total == 0


def test_admin_still_sees_everything(ch_client, clickhouse_database, two_tenants):
    """Once any row policy exists, ClickHouse hides all rows from users with
    no policy; the admin's allow-all policy is what keeps ingest and schema
    work functional. Pin it, because losing it fails silently (zero rows)."""
    total = int(
        ch_client.query(
            f"SELECT count() FROM {clickhouse_database}.{SPANS_TABLE} "
            "WHERE tenant_id IN %(ts)s",
            parameters={"ts": tuple(two_tenants)},
        ).result_rows[0][0]
    )
    assert total == 2


def test_the_reader_cannot_write(ch_reader, clickhouse_database, two_tenants):
    slug = next(iter(two_tenants))
    rows = list(resource_spans_to_rows(
        build_resource_spans(start_ns=_now_ns(-2), end_ns=_now_ns(-1)), tenant_slug=slug,
    ))
    with pytest.raises(Exception, match="(?i)not enough privileges|ACCESS_DENIED"):
        insert_spans(ch_reader, rows, database=clickhouse_database)


def test_the_reader_cannot_drop_the_policy(ch_reader, clickhouse_database, two_tenants):
    with pytest.raises(Exception, match="(?i)not enough privileges|ACCESS_DENIED"):
        ch_reader.command(
            f"DROP ROW POLICY IF EXISTS agentlens_tenant_isolation ON "
            f"{clickhouse_database}.{SPANS_TABLE}"
        )


def test_the_api_queries_work_through_the_reader(ch_reader, clickhouse_database, two_tenants):
    mine, theirs = list(two_tenants)
    assert fetch_trace(
        ch_reader, database=clickhouse_database, tenant_slug=mine, trace_id=two_tenants[mine]
    )
    assert fetch_trace(
        ch_reader, database=clickhouse_database, tenant_slug=mine, trace_id=two_tenants[theirs]
    ) == []
    assert count_spans(ch_reader, database=clickhouse_database, tenant_slug=mine) == 1
    traces = list_traces(
        ch_reader, database=clickhouse_database, tenant_slug=mine,
        window=TimeWindow.last(hours=1),
    )
    assert [t["trace_id"] for t in traces] == [two_tenants[mine]]


def test_tenant_query_refuses_to_run_without_a_tenant(ch_reader, clickhouse_database):
    with pytest.raises(ValueError):
        tenant_query(ch_reader, "SELECT 1", parameters={})
