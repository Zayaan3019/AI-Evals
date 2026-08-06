"""Non-blocking export pipeline: bounded queue, drop accounting, redaction.

The caller's request path must never block on telemetry and must never fail
because of it. Three guarantees, each enforced here rather than assumed:

**Bounded memory.** :class:`BoundedQueueSpanProcessor` performs admission
control against an explicit in-flight budget before handing a span to the
OpenTelemetry :class:`~opentelemetry.sdk.trace.export.BatchSpanProcessor`. We do
not reimplement batching or the exporter thread -- ``BatchSpanProcessor`` owns
those. We add only the one thing it does not expose: an exact, readable count of
what was dropped.

**Loud drops.** A full queue drops the span and increments
:attr:`ExportStats.spans_dropped_queue_full`. Nothing is lost silently; the
counter is readable at runtime and is what the Phase 10 benchmark reports.

**Redaction before egress.** :class:`RedactingSpanExporter` wraps the real
exporter, so anything reaching the wire has passed the redaction pipeline --
including spans produced by third-party instrumentation that never went through
:mod:`agentlens.api`.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from opentelemetry.attributes import BoundedAttributes
from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
    SpanProcessor,
)

from .redaction import DEFAULT_REDACTOR, Redactor
from .semconv import REDACTION_APPLIED, REDACTION_RULES

logger = logging.getLogger(__name__)


@dataclass
class ExportStats:
    """Thread-safe counters describing what the pipeline did.

    Exposed rather than logged-and-forgotten: a dropped span is a measurement
    gap, and a platform that reports quality numbers has to be able to say how
    much of its input it discarded.
    """

    spans_submitted: int = 0
    spans_accepted: int = 0
    spans_dropped_queue_full: int = 0
    spans_dropped_shutdown: int = 0
    spans_exported_ok: int = 0
    spans_export_failed: int = 0
    export_batches: int = 0
    export_exceptions: int = 0
    #: Internal SDK faults that were swallowed to protect the caller.
    sdk_errors: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def _add(self, field: str, amount: int = 1) -> None:
        with self._lock:
            setattr(self, field, getattr(self, field) + amount)

    def snapshot(self) -> dict[str, int]:
        """Return a consistent point-in-time copy of every counter."""
        with self._lock:
            return {
                "spans_submitted": self.spans_submitted,
                "spans_accepted": self.spans_accepted,
                "spans_dropped_queue_full": self.spans_dropped_queue_full,
                "spans_dropped_shutdown": self.spans_dropped_shutdown,
                "spans_exported_ok": self.spans_exported_ok,
                "spans_export_failed": self.spans_export_failed,
                "export_batches": self.export_batches,
                "export_exceptions": self.export_exceptions,
                "sdk_errors": self.sdk_errors,
            }

    @property
    def drop_rate(self) -> float:
        """Fraction of submitted spans that were dropped at admission."""
        with self._lock:
            if self.spans_submitted == 0:
                return 0.0
            dropped = self.spans_dropped_queue_full + self.spans_dropped_shutdown
            return dropped / self.spans_submitted


def _redact_attributes(
    attributes: object, redactor: Redactor
) -> tuple[dict[str, object], tuple[str, ...], bool]:
    """Redact a span/event attribute mapping. Returns (attrs, rules, changed)."""
    out: dict[str, object] = {}
    fired: list[str] = []
    changed = False
    if not attributes:
        return out, (), False
    for key, value in dict(attributes).items():  # type: ignore[arg-type]
        new_value, rules = redactor.redact_value(value)
        if rules:
            changed = True
            fired.extend(rules)
        out[key] = new_value
    return out, tuple(dict.fromkeys(fired)), changed


class RedactingSpanExporter(SpanExporter):
    """Wraps an exporter and redacts every span attribute and event attribute.

    Spans that the redactor does not change are forwarded **by reference**, so
    the common case costs one pass over the attributes and preserves the span
    object exactly (including ``dropped_attributes`` bookkeeping).

    Known fidelity limitation: when redaction *does* change a span, the span is
    rebuilt with an unbounded attribute mapping, so the original
    ``dropped_attributes`` / ``dropped_events`` counts reset to zero for that
    span. This is a deliberate trade -- ``BoundedAttributes`` evicts entries on
    mutation, so in-place editing would silently discard real attributes.
    """

    def __init__(self, exporter: SpanExporter, redactor: Redactor = DEFAULT_REDACTOR):
        self._exporter = exporter
        self._redactor = redactor

    def _redact_span(self, span: ReadableSpan) -> ReadableSpan:
        # Spans created through agentlens.api were already redacted at capture
        # time and carry the marker. Re-scanning them is pure waste: a batch of
        # 512 spans costs tens of thousands of regex matches on the exporter
        # thread, and because that work holds the GIL it shows up directly as
        # latency spikes on the caller's thread. Skip them; the pass below
        # exists for third-party spans, which carry no marker.
        if span.attributes and span.attributes.get(REDACTION_APPLIED):
            return span

        attrs, rules, changed = _redact_attributes(span.attributes, self._redactor)

        events: list[Event] = []
        events_changed = False
        for event in span.events or ():
            ev_attrs, ev_rules, ev_changed = _redact_attributes(
                event.attributes, self._redactor
            )
            if ev_changed:
                events_changed = True
                rules = tuple(dict.fromkeys((*rules, *ev_rules)))
                events.append(
                    Event(
                        name=event.name,
                        attributes=ev_attrs,  # type: ignore[arg-type]
                        timestamp=event.timestamp,
                    )
                )
            else:
                events.append(event)

        if not changed and not events_changed:
            return span

        attrs[REDACTION_APPLIED] = True
        if rules:
            attrs[REDACTION_RULES] = list(rules)

        return ReadableSpan(
            name=span.name,
            context=span.context,
            parent=span.parent,
            resource=span.resource,
            attributes=BoundedAttributes(attributes=attrs, immutable=True),  # type: ignore[arg-type]
            events=tuple(events),
            links=span.links,
            kind=span.kind,
            status=span.status,
            start_time=span.start_time,
            end_time=span.end_time,
            instrumentation_scope=span.instrumentation_scope,
        )

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            redacted = [self._redact_span(span) for span in spans]
        except Exception:
            # Failing closed: if redaction raises we must not export the raw
            # spans, because that is exactly the leak this class exists to
            # prevent.
            logger.exception("agentlens: redaction failed; dropping batch unexported")
            return SpanExportResult.FAILURE
        return self._exporter.export(redacted)

    def shutdown(self) -> None:
        self._exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return self._exporter.force_flush(timeout_millis)


class CountingSpanExporter(SpanExporter):
    """Records export outcomes and releases in-flight budget.

    Also the isolation boundary for exporter faults: an exporter that raises
    (unreachable collector, TLS failure, protocol error) is contained here and
    converted into a counter, so the background thread stays alive and the
    caller's process is unaffected.
    """

    def __init__(
        self,
        exporter: SpanExporter,
        stats: ExportStats,
        release: Callable[[int], None] | None = None,
    ):
        self._exporter = exporter
        self._stats = stats
        self._release = release
        self._deadline: float | None = None

    def set_drain_deadline(self, seconds_from_now: float) -> None:
        """Stop attempting network export after ``seconds_from_now``.

        Neither ``force_flush`` nor ``shutdown`` reliably bounds how long the
        backlog takes to drain: against an unreachable collector every pending
        batch costs a full export timeout, so teardown scales with undelivered
        spans and can stall process exit for minutes. Past the deadline we fail
        batches immediately -- without touching the network -- so the drain
        completes promptly and the loss is counted as a shutdown drop.

        First call wins. Teardown runs flush *then* drain, and re-arming between
        the two would let the second phase extend the budget the first phase was
        supposed to bound.
        """
        if self._deadline is None:
            self._deadline = time.monotonic() + seconds_from_now

    def _past_deadline(self) -> bool:
        return self._deadline is not None and time.monotonic() >= self._deadline

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        count = len(spans)

        if self._past_deadline():
            self._stats._add("spans_dropped_shutdown", count)
            self._stats._add("export_batches")
            if self._release is not None:
                self._release(count)
            return SpanExportResult.FAILURE

        try:
            result = self._exporter.export(spans)
        except Exception:
            self._stats._add("export_exceptions")
            self._stats._add("spans_export_failed", count)
            logger.warning(
                "agentlens: span export raised; %d span(s) lost. "
                "The instrumented application is unaffected.",
                count,
                exc_info=True,
            )
            return SpanExportResult.FAILURE
        finally:
            self._stats._add("export_batches")
            if self._release is not None:
                self._release(count)

        if result is SpanExportResult.SUCCESS:
            self._stats._add("spans_exported_ok", count)
        else:
            self._stats._add("spans_export_failed", count)
        return result

    def shutdown(self) -> None:
        try:
            self._exporter.shutdown()
        except Exception:
            self._stats._add("export_exceptions")
            logger.warning("agentlens: exporter shutdown raised", exc_info=True)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        try:
            return self._exporter.force_flush(timeout_millis)
        except Exception:
            self._stats._add("export_exceptions")
            return False


class BoundedQueueSpanProcessor(SpanProcessor):
    """Admission control in front of ``BatchSpanProcessor``.

    ``BatchSpanProcessor`` already batches on a background thread and bounds its
    own queue, but it does not expose how many spans it discarded. We therefore
    keep the authoritative budget here: :meth:`on_end` checks an in-flight
    counter *before* delegating, and the counter is released by
    :class:`CountingSpanExporter` once a batch leaves the pipeline. The inner
    processor's queue is sized above ours so ours is always the binding
    constraint.

    This is composition, not reimplementation -- no queue, thread, or batching
    logic is duplicated from the SDK.
    """

    def __init__(
        self,
        exporter: SpanExporter,
        *,
        stats: ExportStats | None = None,
        max_queue_size: int = 2048,
        max_export_batch_size: int = 512,
        schedule_delay_millis: float = 5000,
        export_timeout_millis: float = 30_000,
    ) -> None:
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        if max_export_batch_size <= 0:
            raise ValueError("max_export_batch_size must be positive")
        # A batch larger than the queue can never fill, and BatchSpanProcessor
        # rejects that combination outright. Clamp rather than raise: shrinking
        # the queue is a normal thing to do and should not additionally require
        # the caller to remember to shrink the batch size.
        max_export_batch_size = min(max_export_batch_size, max_queue_size)

        self.stats = stats if stats is not None else ExportStats()
        self._max_queue_size = max_queue_size
        self._inflight = 0
        self._lock = threading.Lock()
        self._shutdown = False

        self._counting = CountingSpanExporter(
            exporter, self.stats, release=self._release
        )
        counting = self._counting
        # Headroom so the inner queue never rejects first; ours is authoritative.
        self._inner = BatchSpanProcessor(
            counting,
            max_queue_size=max_queue_size * 2,
            max_export_batch_size=max_export_batch_size,
            schedule_delay_millis=schedule_delay_millis,
            export_timeout_millis=export_timeout_millis,
        )

    def _release(self, count: int) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - count)

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    @property
    def max_queue_size(self) -> int:
        return self._max_queue_size

    def on_start(self, span, parent_context=None) -> None:  # noqa: ANN001
        # Required by the SpanProcessor interface; OpenTelemetry's
        # SynchronousMultiSpanProcessor invokes this on every span start. It is
        # intentionally a no-op: admission control happens at on_end, when the
        # span is complete and its real size is known. Doing work here would
        # charge the caller twice per span for no added signal.
        return None

    def on_end(self, span: ReadableSpan) -> None:
        # This runs on the caller's thread. It must not block, must not raise,
        # and must complete in constant time.
        self.stats._add("spans_submitted")

        with self._lock:
            if self._shutdown:
                self.stats._add("spans_dropped_shutdown")
                return
            if self._inflight >= self._max_queue_size:
                self.stats._add("spans_dropped_queue_full")
                return
            self._inflight += 1

        try:
            self._inner.on_end(span)
        except Exception:
            self._release(1)
            self.stats._add("sdk_errors")
            logger.warning("agentlens: span enqueue failed", exc_info=True)
            return
        self.stats._add("spans_accepted")

    def begin_shutdown(self, drain_timeout_millis: float) -> None:
        """Cap how long the final drain may spend attempting network export.

        Called before ``TracerProvider.shutdown()``, which invokes
        :meth:`shutdown` with no arguments and therefore cannot carry a timeout
        itself.
        """
        self._counting.set_drain_deadline(drain_timeout_millis / 1000.0)

    def shutdown(self) -> None:
        with self._lock:
            self._shutdown = True
        try:
            self._inner.shutdown()
        except Exception:
            self.stats._add("sdk_errors")
            logger.warning("agentlens: processor shutdown raised", exc_info=True)

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        try:
            return self._inner.force_flush(timeout_millis)
        except Exception:
            self.stats._add("sdk_errors")
            return False


__all__ = [
    "BoundedQueueSpanProcessor",
    "CountingSpanExporter",
    "ExportStats",
    "RedactingSpanExporter",
]
