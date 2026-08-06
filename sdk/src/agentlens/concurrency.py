"""Context propagation across thread pools.

Agent systems parallelise constantly -- fan-out retrieval, parallel tool calls,
concurrent sub-agents. Python's :class:`~concurrent.futures.ThreadPoolExecutor`
does **not** copy the caller's context into worker threads, and OpenTelemetry
stores the active span in a :mod:`contextvars` variable. The consequence is
silent and severe: work done in a worker thread starts with no parent, so its
spans become orphaned roots and the trace tree no longer reflects what the agent
actually did.

This was not hypothetical. Instrumenting the Multi-Agent Clinical RAG app
produced a trace whose three retrieval spans sat outside the workflow entirely,
because ``HybridRetriever.search_multi_query`` fans out across a thread pool.

:func:`instrument_thread_pools` fixes this process-wide by attaching the
submitting thread's context inside the worker, so nested spans parent correctly
with no change to the application's code.
"""

from __future__ import annotations

import functools
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from opentelemetry import context as otel_context

_ORIGINAL_SUBMIT: Callable[..., Any] | None = None


def bind_context(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return ``fn`` bound to the *current* OpenTelemetry context.

    Use directly when handing a callable to a scheduler this module does not
    patch (a custom executor, an event loop, a queue consumer)::

        pool.submit(agentlens.bind_context(work), arg)
    """
    captured = otel_context.get_current()

    @functools.wraps(fn)
    def runner(*args: Any, **kwargs: Any) -> Any:
        token = otel_context.attach(captured)
        try:
            return fn(*args, **kwargs)
        finally:
            otel_context.detach(token)

    return runner


def instrument_thread_pools() -> None:
    """Make ``ThreadPoolExecutor.submit`` carry the submitting context.

    Idempotent and process-wide. Only ``submit`` is patched, because
    ``Executor.map`` is implemented in terms of it.
    """
    global _ORIGINAL_SUBMIT
    if _ORIGINAL_SUBMIT is not None:
        return

    original = ThreadPoolExecutor.submit
    _ORIGINAL_SUBMIT = original

    @functools.wraps(original)
    def submit(self, fn, /, *args: Any, **kwargs: Any):
        return original(self, bind_context(fn), *args, **kwargs)

    ThreadPoolExecutor.submit = submit  # type: ignore[method-assign]


def uninstrument_thread_pools() -> None:
    """Restore the original ``submit``. Primarily for tests."""
    global _ORIGINAL_SUBMIT
    if _ORIGINAL_SUBMIT is None:
        return
    ThreadPoolExecutor.submit = _ORIGINAL_SUBMIT  # type: ignore[method-assign]
    _ORIGINAL_SUBMIT = None


__all__ = ["bind_context", "instrument_thread_pools", "uninstrument_thread_pools"]
