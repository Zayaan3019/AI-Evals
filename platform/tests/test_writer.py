"""Buffered writer: batching, bounded loss, and fault containment.

The writer is the component that decides what happens when ClickHouse is slower
than ingestion. Every test here is about a failure mode, because the happy path
is the easy part.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from agentlens_platform.ingest.mapper import resource_spans_to_rows
from agentlens_platform.ingest.writer import SpanWriter
from agentlens_platform.queries import count_spans

from .test_mapper import build_resource_spans
from .test_storage import _now_ns


def _rows(count: int, tenant: str) -> list[list]:
    rows: list[list] = []
    for index in range(count):
        rows.extend(
            resource_spans_to_rows(
                build_resource_spans(
                    trace_id=uuid.uuid4().bytes,
                    span_id=uuid.uuid4().bytes[:8],
                    start_ns=_now_ns(-2),
                    end_ns=_now_ns(-1),
                ),
                tenant_slug=tenant,
            )
        )
    return rows


class ExplodingClient:
    """A client whose insert always fails.

    Not a mock of ClickHouse: it stands in for ClickHouse being down, which is
    the condition under test. The writer must survive it.
    """

    def __init__(self) -> None:
        self.attempts = 0

    def insert(self, *args, **kwargs):
        self.attempts += 1
        raise ConnectionError("clickhouse unreachable")


async def test_spans_reach_clickhouse(writer, ch_client, clickhouse_database):
    tenant = f"w{uuid.uuid4().hex[:12]}"
    accepted = await writer.submit(_rows(10, tenant))
    assert accepted == 10
    await writer.flush()
    assert count_spans(ch_client, database=clickhouse_database, tenant_slug=tenant) == 10
    assert writer.stats.snapshot()["spans_written"] == 10


async def test_buffer_drains_on_the_interval_without_being_asked(
    writer, ch_client, clickhouse_database
):
    """A low-traffic tenant must not wait for a full batch."""
    tenant = f"w{uuid.uuid4().hex[:12]}"
    await writer.submit(_rows(3, tenant))
    # The fixture's flush interval is 200ms; allow generous slack for CI.
    for _ in range(50):
        await asyncio.sleep(0.1)
        if count_spans(ch_client, database=clickhouse_database, tenant_slug=tenant) == 3:
            break
    assert count_spans(ch_client, database=clickhouse_database, tenant_slug=tenant) == 3


async def test_full_buffer_drops_and_counts_rather_than_growing(ch_client, clickhouse_database):
    """An unbounded queue in front of a slow consumer is an OOM waiting to happen."""
    instance = SpanWriter(
        ch_client,
        database=clickhouse_database,
        batch_size=10,
        flush_interval_seconds=3600,  # never fires during the test
        max_buffered_spans=10,
    )
    tenant = f"w{uuid.uuid4().hex[:12]}"
    accepted_first = await instance.submit(_rows(10, tenant))
    accepted_second = await instance.submit(_rows(5, tenant))

    assert accepted_first == 10
    assert accepted_second == 0
    stats = instance.stats.snapshot()
    assert stats["spans_dropped_buffer_full"] == 5
    assert instance.buffered == 10


async def test_partial_acceptance_keeps_what_fits(ch_client, clickhouse_database):
    """A batch that overflows must not lose the spans there was room for."""
    instance = SpanWriter(
        ch_client,
        database=clickhouse_database,
        batch_size=10,
        flush_interval_seconds=3600,
        max_buffered_spans=10,
    )
    tenant = f"w{uuid.uuid4().hex[:12]}"
    accepted = await instance.submit(_rows(4, tenant))
    assert accepted == 4
    accepted = await instance.submit(_rows(9, tenant))
    assert accepted == 6
    assert instance.stats.snapshot()["spans_dropped_buffer_full"] == 3


async def test_write_failure_is_counted_and_does_not_raise(clickhouse_database):
    """ClickHouse being down must not take the ingestion process with it."""
    client = ExplodingClient()
    instance = SpanWriter(
        client, database=clickhouse_database, batch_size=5, flush_interval_seconds=3600
    )
    tenant = f"w{uuid.uuid4().hex[:12]}"
    await instance.submit(_rows(5, tenant))

    written = await instance.flush()

    assert written == 0
    assert client.attempts == 1
    stats = instance.stats.snapshot()
    assert stats["batches_failed"] == 1
    assert stats["spans_failed_write"] == 5
    assert "ConnectionError" in stats["last_error"]


async def test_flush_loop_survives_a_failing_backend(clickhouse_database):
    """The background task must not die on the first error and stop draining."""
    client = ExplodingClient()
    instance = SpanWriter(
        client, database=clickhouse_database, batch_size=2, flush_interval_seconds=0.05
    )
    await instance.start()
    try:
        tenant = f"w{uuid.uuid4().hex[:12]}"
        await instance.submit(_rows(2, tenant))
        await asyncio.sleep(0.3)
        await instance.submit(_rows(2, tenant))
        await asyncio.sleep(0.3)
        # If the loop had died on the first failure, the second batch would
        # still be sitting in the buffer unattempted.
        assert client.attempts >= 2
    finally:
        await instance.aclose()


async def test_close_flushes_the_tail_rather_than_discarding_it(ch_client, clickhouse_database):
    tenant = f"w{uuid.uuid4().hex[:12]}"
    instance = SpanWriter(
        ch_client, database=clickhouse_database, batch_size=1000, flush_interval_seconds=3600
    )
    await instance.start()
    await instance.submit(_rows(7, tenant))
    assert instance.buffered == 7
    await instance.aclose()
    assert count_spans(ch_client, database=clickhouse_database, tenant_slug=tenant) == 7


async def test_batch_size_larger_than_ceiling_is_clamped(ch_client, clickhouse_database):
    """A ceiling below the batch size would drop spans about to be written."""
    instance = SpanWriter(
        ch_client, database=clickhouse_database, batch_size=100, max_buffered_spans=10
    )
    tenant = f"w{uuid.uuid4().hex[:12]}"
    assert await instance.submit(_rows(100, tenant)) == 100
    assert instance.stats.snapshot()["spans_dropped_buffer_full"] == 0


async def test_invalid_batch_size_is_rejected_at_construction(ch_client, clickhouse_database):
    with pytest.raises(ValueError):
        SpanWriter(ch_client, database=clickhouse_database, batch_size=0)


async def test_empty_submit_is_a_noop(writer):
    assert await writer.submit([]) == 0
    assert writer.stats.snapshot()["spans_accepted"] == 0


async def test_reads_and_writes_can_run_concurrently_on_one_client(
    ch_client, clickhouse_database
):
    """Regression: the driver serialises requests sharing a session id.

    By default clickhouse-connect stamps every request with a session id, and
    ClickHouse refuses concurrent queries within one session. That is precisely
    this service's shape -- the writer inserts from a worker thread while the
    API reads on the event loop -- so with the default the two paths raise
    ProgrammingError against each other. Caught by
    test_buffer_drains_on_the_interval_without_being_asked, which queried while
    the background flush was in flight.
    """
    tenant = f"w{uuid.uuid4().hex[:12]}"
    instance = SpanWriter(
        ch_client, database=clickhouse_database, batch_size=20, flush_interval_seconds=3600
    )
    await instance.submit(_rows(60, tenant))

    async def read_repeatedly() -> None:
        for _ in range(20):
            count_spans(ch_client, database=clickhouse_database, tenant_slug=tenant)
            await asyncio.sleep(0)

    # Both directions at once on the same client. Before the fix this raised.
    await asyncio.gather(instance.flush(), read_repeatedly())
    assert count_spans(ch_client, database=clickhouse_database, tenant_slug=tenant) == 60


async def test_large_batch_is_split_into_multiple_inserts(ch_client, clickhouse_database):
    """Batching is a ClickHouse requirement, not a nicety: parts must stay bounded."""
    tenant = f"w{uuid.uuid4().hex[:12]}"
    instance = SpanWriter(
        ch_client, database=clickhouse_database, batch_size=50, flush_interval_seconds=3600
    )
    await instance.submit(_rows(120, tenant))
    await instance.flush()
    stats = instance.stats.snapshot()
    assert stats["batches_written"] == 3  # 50 + 50 + 20
    assert stats["spans_written"] == 120
    assert count_spans(ch_client, database=clickhouse_database, tenant_slug=tenant) == 120
