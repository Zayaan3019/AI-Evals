"""Context propagation across threads.

Regression tests for a real defect found while instrumenting the Multi-Agent
Clinical RAG app: its retrieval fans out across a ThreadPoolExecutor, so the
retrieval spans had no parent and appeared as separate trace roots.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

import agentlens
from agentlens.concurrency import (
    bind_context,
    instrument_thread_pools,
    uninstrument_thread_pools,
)
from agentlens.tree import build_forest


@pytest.fixture
def threaded():
    instrument_thread_pools()
    yield
    uninstrument_thread_pools()


def _fan_out_retrievals(count: int = 3) -> None:
    with agentlens.workflow("parallel_rag"):
        with agentlens.agent_step("planner"):
            with ThreadPoolExecutor(max_workers=count) as pool:
                futures = [
                    pool.submit(_one_retrieval, index) for index in range(count)
                ]
                for future in futures:
                    future.result()


def _one_retrieval(index: int) -> None:
    with agentlens.retrieval(f"shard_{index}") as span:
        span.set_documents([agentlens.Document(id=f"d{index}", score=0.5)])


def test_without_propagation_spans_are_orphaned(lens) -> None:
    """Documents the defect: this is what the app's trace looked like."""
    _fan_out_retrievals()
    roots = build_forest(lens.memory_exporter.get_finished_spans())
    # Workflow plus one orphaned root per worker thread.
    assert len(roots) == 4
    assert sum(1 for r in roots if r.name.startswith("retrieval")) == 3


def test_thread_pool_instrumentation_restores_nesting(lens, threaded) -> None:
    _fan_out_retrievals()
    roots = build_forest(lens.memory_exporter.get_finished_spans())

    assert len(roots) == 1
    root = roots[0]
    assert root.name == "invoke_workflow parallel_rag"
    planner = root.children[0]
    assert planner.name == "invoke_agent planner"
    assert sorted(c.name for c in planner.children) == [
        "retrieval shard_0",
        "retrieval shard_1",
        "retrieval shard_2",
    ]


def test_all_threaded_spans_share_one_trace_id(lens, threaded) -> None:
    _fan_out_retrievals()
    spans = lens.memory_exporter.get_finished_spans()
    trace_ids = {s.context.trace_id for s in spans}
    assert len(trace_ids) == 1


def test_bind_context_works_without_patching(lens) -> None:
    """Escape hatch for schedulers this module does not patch."""
    with agentlens.workflow("manual"):
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(bind_context(lambda: _one_retrieval(9))).result()

    roots = build_forest(lens.memory_exporter.get_finished_spans())
    assert len(roots) == 1
    assert roots[0].children[0].name == "retrieval shard_9"


def test_instrumentation_is_idempotent(lens) -> None:
    instrument_thread_pools()
    first = ThreadPoolExecutor.submit
    instrument_thread_pools()
    assert ThreadPoolExecutor.submit is first
    uninstrument_thread_pools()


def test_uninstrument_restores_original(lens) -> None:
    before = ThreadPoolExecutor.submit
    instrument_thread_pools()
    assert ThreadPoolExecutor.submit is not before
    uninstrument_thread_pools()
    assert ThreadPoolExecutor.submit is before


def test_worker_exception_still_propagates(lens, threaded) -> None:
    """Context binding must not swallow or alter worker failures."""

    def boom() -> None:
        with agentlens.tool_call("explodes"):
            raise ValueError("worker failed")

    with agentlens.workflow("run"):
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(boom)
            with pytest.raises(ValueError, match="worker failed"):
                future.result()

    spans = {s.name: s for s in lens.memory_exporter.get_finished_spans()}
    assert spans["execute_tool explodes"].status.status_code.name == "ERROR"


def test_context_does_not_leak_between_workers(lens, threaded) -> None:
    """Each worker must detach; a leaked token would cross-parent later spans."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        with agentlens.workflow("first"):
            pool.submit(_one_retrieval, 1).result()
        # Submitted outside any workflow: must be a root, not a child of "first".
        pool.submit(_one_retrieval, 2).result()

    roots = build_forest(lens.memory_exporter.get_finished_spans())
    names = {r.name for r in roots}
    assert "retrieval shard_2" in names
