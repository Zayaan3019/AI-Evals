"""Export pipeline: bounded memory, loud drops, contained exporter faults."""

from __future__ import annotations

import threading

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from agentlens.export import (
    BoundedQueueSpanProcessor,
    ExportStats,
    RedactingSpanExporter,
)
from agentlens.redaction import DEFAULT_REDACTOR
from agentlens.semconv import REDACTION_APPLIED, REDACTION_RULES


class RecordingExporter(SpanExporter):
    """Captures batches without touching the network."""

    def __init__(self) -> None:
        self.batches: list[list] = []
        self.spans: list = []
        self.lock = threading.Lock()

    def export(self, spans):
        with self.lock:
            self.batches.append(list(spans))
            self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


class BlockingExporter(SpanExporter):
    """Blocks inside ``export`` until released.

    Makes overflow deterministic: in-flight budget is only released once an
    export *returns*, so while this exporter is blocked the queue provably
    cannot drain and drop arithmetic is exact rather than timing-dependent.
    """

    def __init__(self) -> None:
        self.release = threading.Event()
        self.entered = threading.Event()

    def export(self, spans):
        self.entered.set()
        self.release.wait(timeout=30)
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.release.set()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


class RaisingExporter(SpanExporter):
    """Simulates a broken collector: raises on every export."""

    def __init__(self) -> None:
        self.calls = 0

    def export(self, spans):
        self.calls += 1
        raise RuntimeError("collector exploded")

    def shutdown(self):
        raise RuntimeError("shutdown exploded")

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


class FailingExporter(SpanExporter):
    """Returns FAILURE without raising -- the well-behaved-failure path."""

    def export(self, spans):
        return SpanExportResult.FAILURE

    def shutdown(self):
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


def _provider_with(processor) -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": "t"}))
    provider.add_span_processor(processor)
    return provider


def _emit(provider: TracerProvider, count: int, name: str = "s") -> None:
    tracer = provider.get_tracer("test")
    for index in range(count):
        with tracer.start_as_current_span(f"{name}{index}"):
            pass


# -- bounded queue and drop accounting --------------------------------------


def test_queue_is_bounded_and_drops_are_counted_exactly() -> None:
    """Overflow drops the span and increments a counter -- never silent."""
    exporter = BlockingExporter()
    processor = BoundedQueueSpanProcessor(
        exporter, max_queue_size=10, max_export_batch_size=10
    )
    provider = _provider_with(processor)
    try:
        _emit(provider, 100)

        stats = processor.stats.snapshot()
        assert stats["spans_submitted"] == 100
        assert stats["spans_accepted"] == 10
        assert stats["spans_dropped_queue_full"] == 90
        # The invariant that matters: nothing vanished unaccounted for.
        assert (
            stats["spans_accepted"] + stats["spans_dropped_queue_full"]
            == stats["spans_submitted"]
        )
        assert processor.inflight <= processor.max_queue_size
    finally:
        exporter.release.set()
        processor.shutdown()


def test_drop_rate_is_reported() -> None:
    exporter = BlockingExporter()
    processor = BoundedQueueSpanProcessor(
        exporter, max_queue_size=5, max_export_batch_size=5
    )
    provider = _provider_with(processor)
    try:
        _emit(provider, 50)
        assert processor.stats.drop_rate == 45 / 50
    finally:
        exporter.release.set()
        processor.shutdown()


def test_budget_is_released_after_export_so_the_queue_recovers() -> None:
    """A bounded queue that never drains would be a leak, not a bound."""
    exporter = RecordingExporter()
    processor = BoundedQueueSpanProcessor(
        exporter, max_queue_size=10, max_export_batch_size=5, schedule_delay_millis=50
    )
    provider = _provider_with(processor)

    _emit(provider, 5)
    assert processor.force_flush(5_000)
    assert processor.inflight == 0

    # Capacity is available again, so a later burst is accepted.
    _emit(provider, 5)
    processor.force_flush(5_000)
    stats = processor.stats.snapshot()
    assert stats["spans_dropped_queue_full"] == 0
    assert stats["spans_exported_ok"] == 10
    processor.shutdown()


def test_spans_after_shutdown_are_dropped_and_counted_separately() -> None:
    exporter = RecordingExporter()
    processor = BoundedQueueSpanProcessor(exporter, max_queue_size=100)
    provider = _provider_with(processor)
    processor.shutdown()

    _emit(provider, 5)
    stats = processor.stats.snapshot()
    assert stats["spans_dropped_shutdown"] == 5
    assert stats["spans_dropped_queue_full"] == 0


def test_invalid_queue_configuration_is_rejected_at_construction() -> None:
    import pytest

    with pytest.raises(ValueError):
        BoundedQueueSpanProcessor(RecordingExporter(), max_queue_size=0)
    with pytest.raises(ValueError):
        BoundedQueueSpanProcessor(RecordingExporter(), max_export_batch_size=0)


def test_batch_size_larger_than_queue_is_clamped_not_rejected() -> None:
    """Shrinking the queue must not also require shrinking the batch size."""
    processor = BoundedQueueSpanProcessor(
        RecordingExporter(), max_queue_size=8, max_export_batch_size=512
    )
    provider = _provider_with(processor)
    _emit(provider, 3)
    assert processor.force_flush(5_000)
    assert processor.stats.snapshot()["spans_exported_ok"] == 3
    processor.shutdown()


# -- fault containment -------------------------------------------------------


def test_raising_exporter_never_reaches_the_application() -> None:
    exporter = RaisingExporter()
    processor = BoundedQueueSpanProcessor(
        exporter, max_queue_size=100, max_export_batch_size=10, schedule_delay_millis=50
    )
    provider = _provider_with(processor)

    _emit(provider, 20)  # must not raise
    processor.force_flush(5_000)

    stats = processor.stats.snapshot()
    assert stats["export_exceptions"] >= 1
    assert stats["spans_export_failed"] >= 1
    # Budget is still released, so a broken collector cannot wedge the queue.
    assert processor.inflight == 0
    processor.shutdown()  # exporter.shutdown also raises; must be contained


def test_failing_exporter_is_counted_without_exceptions() -> None:
    processor = BoundedQueueSpanProcessor(
        FailingExporter(),
        max_queue_size=100,
        max_export_batch_size=10,
        schedule_delay_millis=50,
    )
    provider = _provider_with(processor)
    _emit(provider, 10)
    processor.force_flush(5_000)

    stats = processor.stats.snapshot()
    assert stats["spans_export_failed"] == 10
    assert stats["spans_exported_ok"] == 0
    assert stats["export_exceptions"] == 0
    processor.shutdown()


# -- redaction at the export boundary ---------------------------------------


def test_redacting_exporter_scrubs_third_party_span_attributes() -> None:
    """Defence in depth for spans that never went through agentlens.api."""
    recorder = RecordingExporter()
    processor = BoundedQueueSpanProcessor(
        RedactingSpanExporter(recorder, DEFAULT_REDACTOR),
        max_queue_size=100,
        max_export_batch_size=10,
        schedule_delay_millis=50,
    )
    provider = _provider_with(processor)

    tracer = provider.get_tracer("third-party")
    with tracer.start_as_current_span("external") as span:
        span.set_attribute("db.statement", "SELECT * FROM users WHERE email='a@b.com'")

    processor.force_flush(5_000)
    processor.shutdown()

    assert len(recorder.spans) == 1
    exported = recorder.spans[0]
    assert "a@b.com" not in exported.attributes["db.statement"]
    assert "[REDACTED:email]" in exported.attributes["db.statement"]
    assert exported.attributes[REDACTION_APPLIED] is True
    assert "email" in exported.attributes[REDACTION_RULES]


def test_redacting_exporter_passes_clean_spans_through_by_reference() -> None:
    """Unchanged spans keep their identity and dropped-attribute bookkeeping."""
    recorder = RecordingExporter()
    redacting = RedactingSpanExporter(recorder, DEFAULT_REDACTOR)
    processor = BoundedQueueSpanProcessor(
        redacting, max_queue_size=100, max_export_batch_size=10, schedule_delay_millis=50
    )
    provider = _provider_with(processor)
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("clean") as span:
        span.set_attribute("http.route", "/healthz")
    processor.force_flush(5_000)
    processor.shutdown()

    exported = recorder.spans[0]
    assert REDACTION_APPLIED not in exported.attributes
    assert exported.attributes["http.route"] == "/healthz"


def test_redaction_failure_drops_the_batch_rather_than_leaking() -> None:
    """Fail closed: a redactor fault must not let raw content out."""

    class ExplodingRedactor:
        def redact_value(self, value):
            raise RuntimeError("boom")

    recorder = RecordingExporter()
    redacting = RedactingSpanExporter(recorder, ExplodingRedactor())  # type: ignore[arg-type]

    provider = TracerProvider()
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("s") as span:
        span.set_attribute("secret", "a@b.com")
    finished = None
    # Build a real ReadableSpan by exporting through a recorder first.
    plain = RecordingExporter()
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider2 = TracerProvider()
    provider2.add_span_processor(SimpleSpanProcessor(plain))
    t2 = provider2.get_tracer("t")
    with t2.start_as_current_span("s2") as span2:
        span2.set_attribute("secret", "a@b.com")
    finished = plain.spans[0]

    result = redacting.export([finished])
    assert result is SpanExportResult.FAILURE
    assert recorder.spans == []  # nothing leaked downstream


def test_stats_snapshot_is_a_consistent_copy() -> None:
    stats = ExportStats()
    stats._add("spans_submitted", 3)
    snapshot = stats.snapshot()
    stats._add("spans_submitted", 1)
    assert snapshot["spans_submitted"] == 3
