"""Zero-edit instrumentation for FinGuru's causal-analysis agent.

FinGuru is a LangGraph multi-agent system: ``market_analyst`` proposes a causal
hypothesis, ``causal_validator`` tests it statistically, ``risk_critic``
challenges it, and ``hypothesis_refiner`` revises -- with a conditional edge
that loops back to the validator. That refine loop is the interesting part:
it means a single request can invoke the same agent several times, and a trace
that cannot show which iteration produced which claim is not much use.

Nothing in FinGuru's source is modified. Every hook attaches from outside by
wrapping functions the application already exposes:

* ``LLMManager.invoke`` -- the single choke point for every model call.
* the four LangGraph node coroutines, wrapped where the graph builder reads
  them, so each shows up as an ``invoke_agent`` span.
* ``run_causal_analysis`` -- the whole request, as one workflow span.

Token usage is read from the LangChain response metadata rather than estimated.
Groq returns real counts and estimating them would put a fabricated number in a
system whose entire purpose is measurement; when the metadata is absent the span
records no usage and the cost status becomes ``no_usage``.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

import agentlens

logger = logging.getLogger(__name__)

#: FinGuru talks to Groq. Not a convention value we invented -- ``gen_ai.system``
#: permits a provider identifier, and "groq" is what the OTel GenAI registry
#: lists for this vendor.
PROVIDER = "groq"

_installed = False


def _usage_from_response(response: Any) -> tuple[int | None, int | None, str | None]:
    """Pull real token counts out of a LangChain AIMessage.

    LangChain normalises provider usage into ``usage_metadata`` on newer
    versions and leaves it in ``response_metadata['token_usage']`` on older
    ones. Both are checked; neither is invented.
    """
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        return usage.get("input_tokens"), usage.get("output_tokens"), None

    metadata = getattr(response, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage") or {}
    if token_usage:
        return (
            token_usage.get("prompt_tokens"),
            token_usage.get("completion_tokens"),
            metadata.get("model_name"),
        )
    return None, None, metadata.get("model_name")


def _messages_to_payload(messages: Any) -> list[dict[str, str]]:
    """Render LangChain messages into the shape the SDK expects."""
    payload: list[dict[str, str]] = []
    for message in messages or ():
        role = getattr(message, "type", None) or getattr(message, "role", "user")
        content = getattr(message, "content", message)
        payload.append({"role": str(role), "content": str(content)})
    return payload


def _wrap_llm_manager() -> None:
    """Trace every model call at FinGuru's single LLM choke point."""
    from app.core.llm import LLMManager
    from app.core.config import settings

    original = LLMManager.invoke
    if getattr(original, "_agentlens_wrapped", False):
        return

    @functools.wraps(original)
    def invoke(self, messages, *args, **kwargs):
        model = getattr(settings, "llm_model", "unknown")
        with agentlens.llm_call(
            model,
            provider=PROVIDER,
            temperature=getattr(settings, "llm_temperature", None),
            max_tokens=getattr(settings, "llm_max_tokens", None),
        ) as span:
            span.set_messages(_messages_to_payload(messages))
            # Call the underlying client directly so the real response object
            # -- and its usage metadata -- is visible here. LLMManager.invoke
            # returns only ``.content`` and discards the counts.
            response = self._llm.invoke(messages)
            text = getattr(response, "content", response)
            span.set_response(str(text))

            input_tokens, output_tokens, response_model = _usage_from_response(response)
            if input_tokens is not None or output_tokens is not None:
                span.set_usage(input_tokens or 0, output_tokens or 0)
            if response_model:
                span.set_response_model(response_model)
            return text

    invoke._agentlens_wrapped = True  # type: ignore[attr-defined]
    LLMManager.invoke = invoke


def _wrap_nodes() -> None:
    """Wrap each LangGraph node so agent steps appear as spans.

    The graph builder reads the node functions from ``app.agents.causal.graph``
    at build time, so rebinding them in that module is what actually takes
    effect. Rebinding only in ``nodes`` would be invisible to an
    already-imported graph module -- the same import-time capture that produced
    a defect in the Clinical RAG instrumentation.
    """
    from app.agents.causal import graph as graph_module
    from app.agents.causal import nodes as nodes_module

    for name in ("market_analyst", "causal_validator", "risk_critic", "hypothesis_refiner"):
        original = getattr(nodes_module, name, None)
        if original is None or getattr(original, "_agentlens_wrapped", False):
            continue

        def make(node_name: str, func):
            @functools.wraps(func)
            async def wrapper(state, *args, **kwargs):
                with agentlens.agent_step(node_name) as span:
                    # The refine loop can invoke a node several times per
                    # request. Recording the iteration is what makes the trace
                    # readable: without it the spans are indistinguishable.
                    iteration = None
                    if isinstance(state, dict):
                        iteration = state.get("iteration_count")
                    elif hasattr(state, "iteration_count"):
                        iteration = getattr(state, "iteration_count")
                    if iteration is not None:
                        span.set_attribute("agentlens.app.iteration", int(iteration))
                    return await func(state, *args, **kwargs)

            wrapper._agentlens_wrapped = True  # type: ignore[attr-defined]
            return wrapper

        wrapped = make(name, original)
        setattr(nodes_module, name, wrapped)
        # The binding the graph builder will actually read.
        if hasattr(graph_module, name):
            setattr(graph_module, name, wrapped)


def _wrap_statistics() -> None:
    """Record the statistical tests as tool calls.

    These are the load-bearing computation in a causal-analysis agent -- the
    part that distinguishes it from a model asserting a correlation exists. A
    trace that showed only the LLM calls would omit the actual analysis.
    """
    from app.agents.causal import nodes as nodes_module

    for name in ("_perform_pearson_correlation", "_perform_granger_causality"):
        original = getattr(nodes_module, name, None)
        if original is None or getattr(original, "_agentlens_wrapped", False):
            continue

        def make(tool_name: str, func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                with agentlens.tool_call(tool_name.lstrip("_")) as span:
                    result = func(*args, **kwargs)
                    span.set_result(result)
                    return result

            wrapper._agentlens_wrapped = True  # type: ignore[attr-defined]
            return wrapper

        setattr(nodes_module, name, make(name, original))


def _wrap_entrypoint() -> None:
    """One workflow span per analysis request."""
    from app.agents.causal import graph as graph_module

    original = getattr(graph_module, "run_causal_analysis", None)
    if original is None or getattr(original, "_agentlens_wrapped", False):
        return

    @functools.wraps(original)
    async def wrapper(query: str, *args, **kwargs):
        with agentlens.workflow("finguru_causal_analysis") as span:
            span.set_input(query)
            result = await original(query, *args, **kwargs)
            if isinstance(result, dict):
                summary = result.get("final_output")
                if summary is not None:
                    span.set_output(str(summary))
                if result.get("status") is not None:
                    span.set_attribute("agentlens.app.status", str(result["status"]))
                if result.get("iteration_count") is not None:
                    span.set_attribute(
                        "agentlens.app.iteration_count", int(result["iteration_count"])
                    )
            return result

    wrapper._agentlens_wrapped = True  # type: ignore[attr-defined]
    graph_module.run_causal_analysis = wrapper


def instrument() -> None:
    """Attach every hook. Idempotent."""
    global _installed
    if _installed:
        return

    # LangGraph executes nodes concurrently for parallel branches, and Python
    # does not copy contextvars into worker threads -- without this, spans from
    # those branches become orphaned roots. Same defect the Clinical RAG trace
    # exposed.
    agentlens.instrument_thread_pools()

    _wrap_llm_manager()
    _wrap_nodes()
    _wrap_statistics()
    _wrap_entrypoint()
    _installed = True
    logger.info("agentlens instrumentation attached to FinGuru")
