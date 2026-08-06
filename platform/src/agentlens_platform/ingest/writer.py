"""Buffered span writer.

ClickHouse punishes small frequent inserts: every INSERT creates a part, parts
are merged in the background, and exceeding ``parts_to_throw_insert`` makes the
server reject writes outright. So batching here is not a throughput
optimisation, it is what keeps the engine healthy under sustained ingestion.

The buffer is bounded. When it is full, spans are dropped and counted -- the
same discipline the SDK applies on the client side, for the same reason: an
unbounded queue in front of a slow consumer converts a latency problem into an
OOM kill, and a silent drop converts it into a data-integrity problem you find
out about a week later.

``clickhouse-connect`` is a synchronous driver, so inserts run in a worker
thread. Calling it directly from the event loop would stall every concurrent
gRPC stream for the duration of the write.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Sequence

from clickhouse_connect.driver.client import Client

from ..db.clickhouse import insert_spans

logger = logging.getLogger(__name__)


@dataclass
class WriterStats:
    """Counters. Read by the health endpoint and the ingestion tests."""

    spans_accepted: int = 0
    spans_written: int = 0
    spans_dropped_buffer_full: int = 0
    spans_failed_write: int = 0
    batches_written: int = 0
    batches_failed: int = 0
    last_error: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, name: str, amount: int = 1) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + amount)

    def set_error(self, message: str) -> None:
        with self._lock:
            self.last_error = message

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "spans_accepted": self.spans_accepted,
                "spans_written": self.spans_written,
                "spans_dropped_buffer_full": self.spans_dropped_buffer_full,
                "spans_failed_write": self.spans_failed_write,
                "batches_written": self.batches_written,
                "batches_failed": self.batches_failed,
                "last_error": self.last_error,
            }


class SpanWriter:
    """Accumulates rows and flushes them to ClickHouse in batches.

    Lifecycle: :meth:`start` launches the flush loop, :meth:`submit` is called
    per ingest request, :meth:`aclose` drains and stops. ``aclose`` flushes what
    it holds rather than discarding it, so a clean shutdown does not lose the
    tail of the buffer.
    """

    def __init__(
        self,
        client: Client,
        *,
        database: str,
        batch_size: int = 5_000,
        flush_interval_seconds: float = 2.0,
        max_buffered_spans: int = 200_000,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_buffered_spans < batch_size:
            # A ceiling below the batch size would drop spans that the next
            # flush was about to write anyway.
            max_buffered_spans = batch_size
        self._client = client
        self._database = database
        self._batch_size = batch_size
        self._flush_interval = flush_interval_seconds
        self._max_buffered = max_buffered_spans

        self._buffer: list[Sequence[Any]] = []
        self._lock = asyncio.Lock()
        self._flush_lock = asyncio.Lock()
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._closing = False
        self.stats = WriterStats()

    # -- lifecycle --------------------------------------------------------

    async def start(self) -> None:
        if self._task is None:
            self._closing = False
            self._task = asyncio.create_task(self._run(), name="span-writer")

    async def aclose(self) -> None:
        self._closing = True
        self._wake.set()
        if self._task is not None:
            await self._task
            self._task = None
        await self.flush()

    # -- ingest path ------------------------------------------------------

    async def submit(self, rows: Sequence[Sequence[Any]]) -> int:
        """Buffer rows. Returns how many were accepted.

        Partial acceptance is deliberate: when a batch would overflow the
        ceiling we take what fits rather than rejecting the whole payload, so a
        single large request cannot lose spans that there was room for.
        """
        if not rows:
            return 0
        async with self._lock:
            room = self._max_buffered - len(self._buffer)
            if room <= 0:
                self.stats.add("spans_dropped_buffer_full", len(rows))
                return 0
            accepted = rows[:room] if len(rows) > room else rows
            dropped = len(rows) - len(accepted)
            self._buffer.extend(accepted)
            should_flush = len(self._buffer) >= self._batch_size

        self.stats.add("spans_accepted", len(accepted))
        if dropped:
            self.stats.add("spans_dropped_buffer_full", dropped)
            logger.warning("span buffer full, dropped %d spans", dropped)
        if should_flush:
            self._wake.set()
        return len(accepted)

    # -- flush ------------------------------------------------------------

    async def flush(self) -> int:
        """Write everything currently buffered. Returns rows written.

        Serialised by ``_flush_lock`` so a timer-driven flush and a size-driven
        one cannot both take the buffer and write interleaved batches.
        """
        async with self._flush_lock:
            written = 0
            while True:
                async with self._lock:
                    if not self._buffer:
                        return written
                    batch = self._buffer[: self._batch_size]
                    del self._buffer[: len(batch)]
                written += await self._write(batch)

    async def _write(self, batch: list[Sequence[Any]]) -> int:
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None, lambda: insert_spans(self._client, batch, database=self._database)
            )
        except Exception as exc:  # noqa: BLE001 - must not kill the loop
            # The batch is lost. That is preferable to retrying forever behind a
            # bounded buffer, which would stall every subsequent write; the loss
            # is counted and surfaced, never silent.
            self.stats.add("batches_failed")
            self.stats.add("spans_failed_write", len(batch))
            self.stats.set_error(f"{type(exc).__name__}: {exc}")
            logger.error("clickhouse insert failed for %d spans", len(batch), exc_info=True)
            return 0
        self.stats.add("batches_written")
        self.stats.add("spans_written", len(batch))
        return len(batch)

    async def _run(self) -> None:
        """Flush on whichever comes first: a full batch, or the interval."""
        while not self._closing:
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self._flush_interval)
            except asyncio.TimeoutError:
                pass
            self._wake.clear()
            if self._closing:
                break
            try:
                await self.flush()
            except Exception:  # noqa: BLE001
                # _write already contains its own failures; this guards against
                # anything unexpected so the loop cannot die and silently stop
                # draining the buffer.
                logger.error("span writer flush loop error", exc_info=True)
                await asyncio.sleep(min(self._flush_interval, 1.0))

    # -- introspection ----------------------------------------------------

    @property
    def buffered(self) -> int:
        return len(self._buffer)
