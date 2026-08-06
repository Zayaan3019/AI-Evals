"""agentlens instrumentation for the Multi-Agent Clinical RAG app.

Demonstrates the Phase 1 requirement -- instrument an existing agent app with
minimal code change -- in its strongest form: **zero edits to the application's
source**. Everything here attaches from the outside by wrapping the app's own
functions, so the app remains exactly as its author wrote it.

Integration points chosen because they are the app's real seams:

* ``utils.call_llm``            -- the single chokepoint every agent's LLM call
                                  passes through.
* ``utils.requests.post``       -- wrapped only to read Ollama's ``prompt_eval_count``
                                  / ``eval_count`` from the HTTP response, which
                                  ``call_llm`` discards before returning. Token
                                  counts are therefore measured, not estimated.
* ``utils.embed_text``          -- embedding calls, as ``embeddings`` operations.
* ``HybridRetriever.search``    -- retrieval, with real per-chunk scores.
* ``query_agent`` / ``reasoning_agent`` / ``critic_agent`` ``.run``
                                -- the three agent steps.
* ``Pipeline.query``            -- the workflow root.

Because the wrappers nest through the ambient OpenTelemetry context, the
resulting trace reconstructs the pipeline's actual decision tree without the app
passing any context around.
"""

from __future__ import annotations

import contextvars
import functools
from typing import Any

import agentlens
from agentlens import Document

#: Ollama is a local runtime, not one of the providers enumerated by the GenAI
#: conventions. The spec permits values outside the enumeration; "ollama" is the
#: accurate identifier and is recorded as-is rather than forced into a wrong
#: enum member.
PROVIDER = "ollama"

#: Ollama usage counters for the in-flight HTTP call. A ContextVar (not a global)
#: so concurrent requests cannot read each other's token counts.
_last_usage: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "agentlens_ollama_usage", default=None
)

_ORIGINALS: dict[str, Any] = {}
_INSTRUMENTED = False


def _capture_usage_from_response(response: Any) -> None:
    """Record Ollama's token counters, if this response carries them."""
    try:
        payload = response.json()
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    usage: dict[str, int] = {}
    # Ollama reports prompt_eval_count (input) and eval_count (output).
    if isinstance(payload.get("prompt_eval_count"), int):
        usage["input_tokens"] = payload["prompt_eval_count"]
    if isinstance(payload.get("eval_count"), int):
        usage["output_tokens"] = payload["eval_count"]
    if usage:
        _last_usage.set(usage)


def _wrap_requests_post(utils_module: Any) -> None:
    original = utils_module.requests.post
    _ORIGINALS["requests.post"] = original

    @functools.wraps(original)
    def wrapper(url: str, *args: Any, **kwargs: Any):
        response = original(url, *args, **kwargs)
        if isinstance(url, str) and ("/api/chat" in url or "/api/generate" in url):
            _capture_usage_from_response(response)
        return response

    utils_module.requests.post = wrapper


def _wrap_call_llm(utils_module: Any, config_module: Any) -> None:
    original = utils_module.call_llm
    _ORIGINALS["call_llm"] = original

    @functools.wraps(original)
    def wrapper(
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.1,
        json_mode: bool = False,
        num_ctx: int | None = None,
        think: bool = False,
    ) -> str:
        resolved_model = model or config_module.MAIN_LLM
        _last_usage.set(None)
        with agentlens.llm_call(
            resolved_model,
            provider=PROVIDER,
            temperature=temperature,
            max_tokens=num_ctx,
        ) as span:
            span.set_messages(messages)
            span.set_attribute("agentlens.app.json_mode", json_mode)
            result = original(
                messages,
                model=model,
                temperature=temperature,
                json_mode=json_mode,
                num_ctx=num_ctx,
                think=think,
            )
            span.set_response(result, model=resolved_model)
            usage = _last_usage.get()
            if usage:
                span.set_usage(
                    usage.get("input_tokens"), usage.get("output_tokens")
                )
            return result

    utils_module.call_llm = wrapper


def _wrap_embed_text(utils_module: Any, config_module: Any) -> None:
    original = utils_module.embed_text
    _ORIGINALS["embed_text"] = original

    @functools.wraps(original)
    def wrapper(text: str) -> list[float]:
        with agentlens.llm_call(
            config_module.EMBED_MODEL,
            provider=PROVIDER,
            operation="embeddings",
        ) as span:
            vector = original(text)
            span.set_attribute("agentlens.app.embedding_dimensions", len(vector))
            return vector

    utils_module.embed_text = wrapper


def _wrap_retriever(hybrid_module: Any, config_module: Any) -> None:
    original = hybrid_module.HybridRetriever.search
    _ORIGINALS["HybridRetriever.search"] = original

    @functools.wraps(original)
    def wrapper(self, query: str, n_results: int = config_module.TOP_K_RERANK, **kwargs: Any):
        with agentlens.retrieval(
            "hybrid_dense_bm25",
            query=query,
            top_k=n_results,
            data_source_id="chroma+bm25",
        ) as span:
            chunks, stats = original(self, query, n_results, **kwargs)
            span.set_documents(
                [
                    Document(
                        id=chunk.chunk_id,
                        # rrf_score is the fused rank the pipeline actually
                        # ranks on; raw `score` is a cosine distance and is
                        # kept in metadata so both remain inspectable.
                        score=float(chunk.rrf_score),
                        content=chunk.text,
                        source=chunk.article_id,
                        metadata={
                            "article_title": chunk.article_title,
                            "cosine_distance": float(chunk.score),
                            "rank": int(chunk.rank),
                        },
                    )
                    for chunk in chunks
                ]
            )
            for field_name in ("dense_hits", "bm25_hits", "reranked"):
                value = getattr(stats, field_name, None)
                if value is not None:
                    span.set_attribute(f"agentlens.app.retrieval.{field_name}", value)
            return chunks, stats

    hybrid_module.HybridRetriever.search = wrapper


def _wrap_agent(module: Any, key: str, step_name: str) -> None:
    original = module.run
    _ORIGINALS[key] = original

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any):
        with agentlens.agent_step(step_name) as span:
            result = original(*args, **kwargs)
            span.set_output(repr(result))
            return result

    module.run = wrapper


def _wrap_pipeline(pipeline_module: Any) -> None:
    original = pipeline_module.HealthcareRAGPipeline.query
    _ORIGINALS["HealthcareRAGPipeline.query"] = original

    @functools.wraps(original)
    def wrapper(self, user_query: str, *args: Any, **kwargs: Any):
        with agentlens.workflow("clinical_rag_query") as span:
            span.set_input(user_query)
            result = original(self, user_query, *args, **kwargs)
            answer = getattr(result, "final_answer", None)
            if answer:
                span.set_output(answer)
            route = getattr(result, "route", None)
            if route:
                # The app's routing decision is the single most useful thing to
                # filter traces on: simple and complex take different paths.
                span.set_attribute("agentlens.app.route", str(route))
            cited = getattr(result, "cited_articles", None)
            if cited:
                span.set_attribute("agentlens.app.cited_articles", list(cited))
            return result

    pipeline_module.HealthcareRAGPipeline.query = wrapper


def instrument() -> None:
    """Attach agentlens to the Clinical RAG app. Idempotent."""
    global _INSTRUMENTED
    if _INSTRUMENTED:
        return

    import config
    import utils
    from agents import critic_agent, query_agent, reasoning_agent
    import pipeline as pipeline_module
    from retrieval import hybrid as hybrid_module

    # HybridRetriever.search_multi_query fans out across a ThreadPoolExecutor
    # (retrieval/hybrid.py:184). Python does not copy contextvars into worker
    # threads, so without this the retrieval spans start with no parent and
    # appear as separate trace roots instead of nesting under the workflow.
    agentlens.instrument_thread_pools()

    _wrap_requests_post(utils)
    _wrap_call_llm(utils, config)
    _wrap_embed_text(utils, config)
    _wrap_retriever(hybrid_module, config)
    _wrap_agent(query_agent, "query_agent.run", "query_decomposer")
    _wrap_agent(reasoning_agent, "reasoning_agent.run", "reasoner")
    _wrap_agent(critic_agent, "critic_agent.run", "critic")
    _wrap_pipeline(pipeline_module)

    # The agents captured `call_llm` and `embed_text` by value at import time,
    # so rebinding the utils attribute alone would miss them. Rebind in each
    # module that holds its own reference.
    for module in (query_agent, reasoning_agent, critic_agent, pipeline_module):
        if hasattr(module, "call_llm"):
            module.call_llm = utils.call_llm
        if hasattr(module, "embed_text"):
            module.embed_text = utils.embed_text
    if hasattr(hybrid_module, "embed_text"):
        hybrid_module.embed_text = utils.embed_text

    _INSTRUMENTED = True
