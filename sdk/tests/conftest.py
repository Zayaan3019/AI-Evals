from __future__ import annotations

import pytest

import agentlens
from agentlens.config import AgentLensConfig


@pytest.fixture(autouse=True)
def _clean_sdk_state():
    """Guarantee no SDK instance leaks between tests.

    ``init`` installs process-global state, so a test that forgets to shut down
    would otherwise silently change the behaviour of every test after it.
    """
    agentlens.shutdown()
    yield
    agentlens.shutdown()


@pytest.fixture
def lens():
    """SDK wired to an in-process exporter with synchronous export.

    ``exporter="memory"`` uses SimpleSpanProcessor, so a span is visible the
    instant it ends -- tests assert on exact span sets without racing a timer.
    """
    instance = agentlens.init(
        AgentLensConfig(
            service_name="test-service",
            tenant="test-tenant",
            project="test-project",
            exporter="memory",
            set_global_tracer_provider=False,
        )
    )
    yield instance
    agentlens.shutdown()


def spans_by_name(lens_instance) -> dict[str, object]:
    assert lens_instance.memory_exporter is not None
    return {s.name: s for s in lens_instance.memory_exporter.get_finished_spans()}
