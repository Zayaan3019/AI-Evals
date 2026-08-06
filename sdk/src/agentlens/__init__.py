"""agentlens -- non-blocking OpenTelemetry tracing for LLM agent systems.

Minimal integration::

    import agentlens

    agentlens.init(service_name="my-agent", tenant="acme")

    with agentlens.workflow("answer_question"):
        with agentlens.llm_call("claude-opus-5", provider="anthropic") as span:
            response = client.messages.create(...)
            span.set_response(text).set_usage(
                response.usage.input_tokens, response.usage.output_tokens
            )

Design guarantees, each covered by a test in ``sdk/tests``:

* The caller's request path never blocks on export and never sees an SDK
  exception.
* A full queue drops spans and counts them; nothing is lost silently.
* Redaction runs in-process, before anything is serialised for export.
* An unreachable collector leaves the application completely unaffected.
"""

from .api import (
    AgentSpan,
    Document,
    LLMSpan,
    RetrievalSpan,
    ToolSpan,
    agent_step,
    llm_call,
    retrieval,
    tool_call,
    trace_agent,
    trace_llm,
    trace_retrieval,
    trace_tool,
    workflow,
)
from .concurrency import (
    bind_context,
    instrument_thread_pools,
    uninstrument_thread_pools,
)
from .config import AgentLensConfig
from .export import ExportStats
from .pricing import Cost, CostStatus, PricingTable, TokenUsage
from .redaction import (
    DEFAULT_REDACTOR,
    RedactionRule,
    Redactor,
    build_redactor,
)
from .tracer import AgentLens, get_instance, get_tracer, init, shutdown
from .version import __version__

__all__ = [
    "AgentLens",
    "AgentLensConfig",
    "AgentSpan",
    "Cost",
    "CostStatus",
    "DEFAULT_REDACTOR",
    "Document",
    "ExportStats",
    "LLMSpan",
    "PricingTable",
    "RedactionRule",
    "Redactor",
    "RetrievalSpan",
    "TokenUsage",
    "ToolSpan",
    "__version__",
    "agent_step",
    "bind_context",
    "build_redactor",
    "get_instance",
    "instrument_thread_pools",
    "uninstrument_thread_pools",
    "get_tracer",
    "init",
    "llm_call",
    "retrieval",
    "shutdown",
    "tool_call",
    "trace_agent",
    "trace_llm",
    "trace_retrieval",
    "trace_tool",
    "workflow",
]
