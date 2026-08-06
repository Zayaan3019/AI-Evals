"""Graceful degradation: an unreachable collector must not affect the app.

The endpoints used here are genuinely unreachable, not mocked:

* ``192.0.2.1`` is TEST-NET-1 (RFC 5737), reserved for documentation and
  guaranteed not to be routable on the public internet. Connections to it hang
  or are rejected -- exactly the "black hole collector" failure mode.
* ``.invalid`` is the reserved TLD (RFC 2606); it can never resolve.

Each test asserts the *caller's* observable behaviour: correct results, no
exceptions, and bounded wall-clock time.
"""

from __future__ import annotations

import time

import pytest

import agentlens
from agentlens.config import AgentLensConfig

#: Non-routable by definition (RFC 5737 TEST-NET-1).
BLACKHOLE_ENDPOINT = "http://192.0.2.1:4317"
#: Reserved TLD (RFC 2606) -- DNS resolution can never succeed.
UNRESOLVABLE_ENDPOINT = "http://collector.invalid:4317"

#: The hypothesis under test is binary: does the caller block on the network?
#: If it did, 1000 spans against a dead collector would cost on the order of
#: 1000 export timeouts -- minutes, not seconds. Unloaded, the loop takes well
#: under a second. Any threshold between those two regimes is decisive, so this
#: is set loose enough to survive a machine under concurrent load (where CPU
#: starvation, not blocking, slows the loop) while still failing by two orders
#: of magnitude if telemetry ever moves onto the request path.
MAX_CALLER_SECONDS = 8.0


def _blackhole_config(endpoint: str) -> AgentLensConfig:
    return AgentLensConfig(
        service_name="blackhole-test",
        exporter="otlp",
        endpoint=endpoint,
        insecure=True,
        # Small batches and a short delay so the exporter is actively trying to
        # reach the dead collector while the caller works.
        max_export_batch_size=8,
        schedule_delay_millis=50,
        export_timeout_millis=1000,
        set_global_tracer_provider=False,
    )


def _do_work(iterations: int) -> int:
    """Realistic instrumented workload; returns a value the caller depends on."""
    total = 0
    for index in range(iterations):
        with agentlens.workflow(f"run-{index}"):
            with agentlens.agent_step("step"):
                with agentlens.retrieval("kb", query="q", top_k=2) as r:
                    r.set_documents([agentlens.Document(id="d", score=0.5)])
                with agentlens.llm_call("claude-opus-5", provider="anthropic") as llm:
                    llm.set_messages([{"role": "user", "content": "hello"}])
                    llm.set_response("world")
                    llm.set_usage(100, 20)
                with agentlens.tool_call("t", arguments={"i": index}) as tool:
                    tool.set_result(index * 2)
                total += index * 2
    return total


@pytest.mark.parametrize(
    "endpoint", [BLACKHOLE_ENDPOINT, UNRESOLVABLE_ENDPOINT], ids=["blackhole", "dns"]
)
def test_application_is_unaffected_by_an_unreachable_collector(endpoint: str) -> None:
    lens = agentlens.init(_blackhole_config(endpoint))
    try:
        start = time.perf_counter()
        result = _do_work(200)
        elapsed = time.perf_counter() - start

        # 1. The application computed the right answer.
        assert result == sum(index * 2 for index in range(200))
        # 2. No exception surfaced (reaching here proves it).
        # 3. The caller was never blocked on the network.
        assert elapsed < MAX_CALLER_SECONDS, (
            f"caller path took {elapsed:.3f}s against an unreachable collector; "
            f"telemetry is blocking the request path"
        )
        # 4. Spans were produced and handed to the pipeline, not skipped.
        assert lens.stats.snapshot()["spans_submitted"] == 200 * 5
    finally:
        agentlens.shutdown(timeout_millis=2000)


def test_shutdown_is_bounded_regardless_of_backlog_size() -> None:
    """Process exit must not scale with the undelivered backlog.

    Regression test for a real defect: ``TracerProvider.shutdown()`` drains the
    whole backlog with no time bound, so against a dead collector each pending
    batch cost a full export timeout. With 1000 spans at 8 per batch that was
    ~125 sequential 1s timeouts. The drain deadline caps it instead.
    """
    agentlens.init(_blackhole_config(BLACKHOLE_ENDPOINT))
    _do_work(200)  # 1000 spans -> 125 batches, none of which can be delivered

    start = time.perf_counter()
    agentlens.shutdown(timeout_millis=2000)
    elapsed = time.perf_counter() - start

    # Budget is 2s flush + 2s drain, plus room for one in-flight export attempt.
    assert elapsed < 10.0, (
        f"shutdown took {elapsed:.2f}s with a 2s budget; the drain is not bounded "
        f"and process exit stalls proportionally to undelivered spans"
    )


def test_export_failures_are_counted_not_hidden() -> None:
    """Data loss must be measurable, not silent."""
    lens = agentlens.init(_blackhole_config(BLACKHOLE_ENDPOINT))
    try:
        _do_work(10)
        lens.force_flush(2000)
        stats = lens.stats.snapshot()
        assert stats["spans_submitted"] == 50
        # The collector is unreachable, so nothing can have succeeded.
        assert stats["spans_exported_ok"] == 0
        # And the failure is visible in the counters rather than swallowed.
        assert stats["spans_export_failed"] + stats["export_exceptions"] > 0
    finally:
        agentlens.shutdown(timeout_millis=2000)


def test_malformed_endpoint_degrades_to_no_export_without_raising() -> None:
    """A misconfigured endpoint must not crash application startup."""
    lens = agentlens.init(
        AgentLensConfig(
            service_name="bad-endpoint",
            exporter="otlp",
            endpoint="not a valid endpoint at all",
            set_global_tracer_provider=False,
        )
    )
    try:
        # Instrumented code still runs.
        result = _do_work(5)
        assert result == sum(i * 2 for i in range(5))
    finally:
        agentlens.shutdown(timeout_millis=2000)


def test_init_twice_does_not_install_a_second_pipeline() -> None:
    first = agentlens.init(_blackhole_config(BLACKHOLE_ENDPOINT))
    second = agentlens.init(_blackhole_config(BLACKHOLE_ENDPOINT))
    try:
        assert first is second
    finally:
        agentlens.shutdown(timeout_millis=2000)


def test_shutdown_is_idempotent() -> None:
    agentlens.init(_blackhole_config(BLACKHOLE_ENDPOINT))
    agentlens.shutdown(timeout_millis=2000)
    agentlens.shutdown(timeout_millis=2000)  # must not raise
