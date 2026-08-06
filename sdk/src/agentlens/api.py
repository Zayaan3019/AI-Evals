"""Instrumentation surface: decorators, context managers, and explicit handles.

Three shapes over one implementation, because real agent code has all three:

* ``@trace_llm`` / ``@trace_tool`` / ... when a function *is* the operation.
* ``with llm_call(...) as span:`` when the operation spans several statements.
* ``span.set_usage(...)`` when values only become known after the call returns.

Spans nest through the ambient OpenTelemetry context, so an agent step
containing a retrieval and two tool calls reconstructs as a tree without the
caller passing anything around.

**Contract: these functions never raise on the caller's behalf.** Every
attribute write is guarded; an internal fault increments
``ExportStats.sdk_errors`` and is otherwise invisible. Exceptions raised by the
*wrapped* code are recorded on the span and re-raised unchanged -- swallowing
them would corrupt the application's control flow.
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Generator, Mapping, Sequence, TypeVar

from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from . import semconv as sc
from .config import AgentLensConfig
from .pricing import CostStatus, PricingTable, TokenUsage
from .redaction import Redactor
from .tracer import get_instance, get_tracer

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_DEFAULT_CONFIG = AgentLensConfig()


def _active_config() -> AgentLensConfig:
    instance = get_instance()
    return instance.config if instance is not None else _DEFAULT_CONFIG


def _note_sdk_error(message: str) -> None:
    """Record an internal fault without surfacing it to the caller."""
    instance = get_instance()
    if instance is not None:
        instance.stats._add("sdk_errors")
    logger.debug("agentlens: %s", message, exc_info=True)


def _json_dumps(value: Any) -> str:
    """Serialise for a span attribute; never raise on odd objects."""
    try:
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except Exception:
        return json.dumps({"agentlens_serialization_error": repr(value)[:512]})


@dataclass(frozen=True, slots=True)
class Document:
    """One retrieved document.

    ``score`` is the retriever's raw score. Its scale is retriever-specific, so
    it is recorded verbatim rather than normalised -- normalising here would
    silently destroy the ability to compare against the retriever's own output.
    """

    id: str | None = None
    score: float | None = None
    content: str | None = None
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class _SpanHandle:
    """Base handle. Owns guarded, redacted attribute writes."""

    __slots__ = ("_span", "_config", "_redactor", "_rules_fired")

    def __init__(self, span: Span, config: AgentLensConfig) -> None:
        self._span = span
        self._config = config
        self._redactor: Redactor | None = config.content_redactor()
        self._rules_fired: set[str] = set()

    @property
    def span(self) -> Span:
        """The underlying OpenTelemetry span, for advanced use."""
        return self._span

    @property
    def trace_id(self) -> str:
        """Hex trace id, for correlating with application logs."""
        try:
            return format(self._span.get_span_context().trace_id, "032x")
        except Exception:
            return ""

    @property
    def span_id(self) -> str:
        try:
            return format(self._span.get_span_context().span_id, "016x")
        except Exception:
            return ""

    # -- attribute writing ------------------------------------------------

    def _set(self, key: str, value: Any) -> None:
        """Set a non-content attribute (parameters, counts, identifiers)."""
        if value is None:
            return
        try:
            self._span.set_attribute(key, value)
        except Exception:
            _note_sdk_error(f"failed to set attribute {key!r}")

    def _set_content(self, key: str, value: Any) -> None:
        """Set a free-text attribute: redacted and truncated before it is stored.

        Redacting here -- not at export -- means the raw value never exists on
        the span object, so it cannot leak through a console exporter, a heap
        dump, or a third-party processor.
        """
        if value is None:
            return
        if not self._config.capture_content:
            return
        try:
            if not isinstance(value, str):
                value = _json_dumps(value)
            if self._redactor is not None:
                outcome = self._redactor.redact(value)
                value = outcome.text
                self._rules_fired.update(outcome.rules_fired)
            elif len(value) > self._config.max_attribute_length:
                value = value[: self._config.max_attribute_length] + "...[truncated]"
            self._span.set_attribute(key, value)
        except Exception:
            _note_sdk_error(f"failed to set content attribute {key!r}")

    def set_attribute(self, key: str, value: Any) -> None:
        """Attach a caller-defined attribute. Content is redacted."""
        if isinstance(value, str):
            self._set_content(key, value)
        else:
            self._set(key, value)

    def add_event(self, name: str, attributes: Mapping[str, Any] | None = None) -> None:
        try:
            self._span.add_event(name, dict(attributes or {}))
        except Exception:
            _note_sdk_error(f"failed to add event {name!r}")

    # -- lifecycle --------------------------------------------------------

    def record_exception(self, exc: BaseException) -> None:
        try:
            self._span.record_exception(exc)
            self._span.set_status(Status(StatusCode.ERROR, str(exc)))
            # __name__, not __qualname__: the convention asks for a
            # low-cardinality value, and __qualname__ embeds the enclosing
            # scope for locally-defined exception classes.
            self._span.set_attribute(sc.ERROR_TYPE, type(exc).__name__)
        except Exception:
            _note_sdk_error("failed to record exception")

    def _finalize(self) -> None:
        """Write bookkeeping attributes that depend on everything else."""
        try:
            if self._redactor is not None:
                self._span.set_attribute(sc.REDACTION_APPLIED, True)
                if self._rules_fired:
                    self._span.set_attribute(
                        sc.REDACTION_RULES, sorted(self._rules_fired)
                    )
        except Exception:
            _note_sdk_error("failed to finalize span")


class LLMSpan(_SpanHandle):
    """Handle for a model call."""

    __slots__ = ("_provider", "_model", "_usage", "_pricing")

    def __init__(
        self,
        span: Span,
        config: AgentLensConfig,
        provider: str,
        model: str,
        pricing: PricingTable | None,
    ) -> None:
        super().__init__(span, config)
        self._provider = provider
        self._model = model
        self._usage: TokenUsage | None = None
        self._pricing = pricing

    def set_request(
        self,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        stop_sequences: Sequence[str] | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        seed: int | None = None,
        choice_count: int | None = None,
        stream: bool | None = None,
    ) -> LLMSpan:
        """Record request parameters. All optional; unset values are omitted."""
        self._set(sc.GEN_AI_REQUEST_TEMPERATURE, temperature)
        self._set(sc.GEN_AI_REQUEST_TOP_P, top_p)
        self._set(sc.GEN_AI_REQUEST_TOP_K, top_k)
        self._set(sc.GEN_AI_REQUEST_MAX_TOKENS, max_tokens)
        if stop_sequences is not None:
            self._set(sc.GEN_AI_REQUEST_STOP_SEQUENCES, list(stop_sequences))
        self._set(sc.GEN_AI_REQUEST_FREQUENCY_PENALTY, frequency_penalty)
        self._set(sc.GEN_AI_REQUEST_PRESENCE_PENALTY, presence_penalty)
        self._set(sc.GEN_AI_REQUEST_SEED, seed)
        self._set(sc.GEN_AI_REQUEST_CHOICE_COUNT, choice_count)
        self._set(sc.GEN_AI_REQUEST_STREAM, stream)
        return self

    def set_messages(
        self,
        messages: Sequence[Mapping[str, Any]] | None = None,
        *,
        system_instructions: str | None = None,
    ) -> LLMSpan:
        """Record the rendered input messages (``gen_ai.input.messages``)."""
        if messages is not None:
            self._set_content(sc.GEN_AI_INPUT_MESSAGES, list(messages))
        if system_instructions is not None:
            self._set_content(sc.GEN_AI_SYSTEM_INSTRUCTIONS, system_instructions)
        return self

    def set_prompt_template(
        self,
        template: str | None = None,
        *,
        name: str | None = None,
        version: str | None = None,
        variables: Mapping[str, Any] | None = None,
    ) -> LLMSpan:
        """Record the *un-rendered* template alongside the rendered messages.

        The rendered prompt goes to ``gen_ai.input.messages``; the template and
        its version are what a prompt-change regression gate actually varies, so
        both are needed to attribute a score delta to a prompt revision.
        """
        self._set(sc.GEN_AI_PROMPT_NAME, name)
        self._set(sc.PROMPT_TEMPLATE_VERSION, version)
        self._set_content(sc.PROMPT_TEMPLATE, template)
        if variables is not None:
            self._set_content(sc.PROMPT_VARIABLES, dict(variables))
        return self

    def set_response(
        self,
        output: Any = None,
        *,
        response_id: str | None = None,
        model: str | None = None,
        finish_reasons: Sequence[str] | None = None,
    ) -> LLMSpan:
        """Record the model output and response metadata."""
        self._set(sc.GEN_AI_RESPONSE_ID, response_id)
        self._set(sc.GEN_AI_RESPONSE_MODEL, model)
        if finish_reasons is not None:
            self._set(sc.GEN_AI_RESPONSE_FINISH_REASONS, list(finish_reasons))
        if output is not None:
            if isinstance(output, str):
                output = [{"role": "assistant", "content": output}]
            self._set_content(sc.GEN_AI_OUTPUT_MESSAGES, output)
        return self

    def set_usage(
        self,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        *,
        cache_read_input_tokens: int | None = None,
        cache_creation_input_tokens: int | None = None,
        reasoning_output_tokens: int | None = None,
    ) -> LLMSpan:
        """Record token counts. Cost is computed from these when the span ends."""
        self._usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            reasoning_output_tokens=reasoning_output_tokens,
        )
        self._set(sc.GEN_AI_USAGE_INPUT_TOKENS, input_tokens)
        self._set(sc.GEN_AI_USAGE_OUTPUT_TOKENS, output_tokens)
        self._set(sc.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS, cache_read_input_tokens)
        self._set(
            sc.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS, cache_creation_input_tokens
        )
        self._set(sc.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS, reasoning_output_tokens)
        return self

    def _finalize(self) -> None:
        self._write_cost()
        super()._finalize()

    def _write_cost(self) -> None:
        if self._pricing is None:
            return
        try:
            usage = self._usage if self._usage is not None else TokenUsage()
            cost = self._pricing.compute(self._provider, self._model, usage)
            self._span.set_attribute(sc.COST_PRICING_VERSION, cost.pricing_version)
            self._span.set_attribute(sc.COST_STATUS, cost.status.value)
            if cost.status is CostStatus.OK and cost.micro_usd is not None:
                self._span.set_attribute(sc.COST_MICRO_USD, cost.micro_usd)
                self._span.set_attribute(sc.COST_CURRENCY, cost.currency)
        except Exception:
            _note_sdk_error("failed to compute cost")


class ToolSpan(_SpanHandle):
    """Handle for a tool invocation."""

    __slots__ = ()

    def set_arguments(self, arguments: Any) -> ToolSpan:
        self._set_content(sc.GEN_AI_TOOL_CALL_ARGUMENTS, arguments)
        return self

    def set_result(self, result: Any) -> ToolSpan:
        self._set_content(sc.GEN_AI_TOOL_CALL_RESULT, result)
        return self


class RetrievalSpan(_SpanHandle):
    """Handle for a retrieval call."""

    __slots__ = ()

    def set_query(self, query: str) -> RetrievalSpan:
        self._set_content(sc.GEN_AI_RETRIEVAL_QUERY_TEXT, query)
        return self

    def set_documents(self, documents: Sequence[Document | Mapping[str, Any]]) -> RetrievalSpan:
        """Record retrieved documents with their scores.

        Document text is treated as content: it is omitted when
        ``capture_content`` is off and redacted otherwise. Ids and scores are
        always kept, so retrieval quality remains measurable even when the
        corpus itself may not be stored.
        """
        try:
            payload: list[dict[str, Any]] = []
            for doc in documents:
                item = asdict(doc) if isinstance(doc, Document) else dict(doc)
                if not self._config.capture_content:
                    item.pop("content", None)
                payload.append(item)
            self._set(sc.RETRIEVAL_DOCUMENT_COUNT, len(payload))
            # Serialise then redact as one string: document text is the field
            # most likely to carry PII out of a customer corpus.
            self._set_content(sc.GEN_AI_RETRIEVAL_DOCUMENTS, payload)
        except Exception:
            _note_sdk_error("failed to record retrieval documents")
        return self


class AgentSpan(_SpanHandle):
    """Handle for an agent step."""

    __slots__ = ()

    def set_input(self, value: Any) -> AgentSpan:
        self._set_content(sc.GEN_AI_INPUT_MESSAGES, value)
        return self

    def set_output(self, value: Any) -> AgentSpan:
        if isinstance(value, str):
            value = [{"role": "assistant", "content": value}]
        self._set_content(sc.GEN_AI_OUTPUT_MESSAGES, value)
        return self


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


def _start_span(name: str, kind: SpanKind):
    return get_tracer().start_as_current_span(name, kind=kind, end_on_exit=False)


@contextmanager
def _span_scope(span_name: str, kind: SpanKind, build, /, **kw) -> Generator[Any, None, None]:
    """Shared span lifecycle: start, yield handle, record errors, finalize, end.

    The framing parameters are positional-only so that ``**kw`` -- which carries
    builder arguments such as ``name`` -- can never collide with them.

    ``end_on_exit=False`` plus an explicit ``end()`` lets ``_finalize`` run
    *after* the caller's block, so cost (which depends on usage set inside the
    block) lands on the span before it is exported.
    """
    config = _active_config()
    with _start_span(span_name, kind) as span:
        handle = build(span, config, **kw)
        try:
            yield handle
        except BaseException as exc:
            handle.record_exception(exc)
            raise  # never swallow the caller's exception
        finally:
            handle._finalize()
            try:
                span.end()
            except Exception:
                _note_sdk_error("failed to end span")


def _resolve_pricing(config: AgentLensConfig) -> PricingTable | None:
    try:
        return config.resolved_pricing()
    except Exception:
        _note_sdk_error("failed to load pricing table")
        return None


def _build_llm(span, config, *, provider, model, operation, conversation_id):
    handle = LLMSpan(span, config, provider, model, _resolve_pricing(config))
    handle._set(sc.GEN_AI_OPERATION_NAME, operation)
    # Both are set deliberately: `gen_ai.system` is the long-standing attribute
    # most backends index on, `gen_ai.provider.name` is its newer replacement.
    # Emitting both keeps queries working across the convention transition.
    handle._set(sc.GEN_AI_SYSTEM, provider)
    handle._set(sc.GEN_AI_PROVIDER_NAME, provider)
    handle._set(sc.GEN_AI_REQUEST_MODEL, model)
    handle._set(sc.GEN_AI_CONVERSATION_ID, conversation_id)
    return handle


@contextmanager
def llm_call(
    model: str,
    *,
    provider: str = "unknown",
    operation: str = sc.OperationName.CHAT.value,
    conversation_id: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    messages: Sequence[Mapping[str, Any]] | None = None,
    prompt_template: str | None = None,
    prompt_version: str | None = None,
    prompt_variables: Mapping[str, Any] | None = None,
) -> Generator[LLMSpan, None, None]:
    """Trace a model call.

    Span name follows the GenAI convention ``{operation} {model}``.

        with llm_call("claude-opus-5", provider="anthropic") as span:
            response = client.messages.create(...)
            span.set_response(text).set_usage(in_tok, out_tok)
    """
    with _span_scope(
        f"{operation} {model}",
        SpanKind.CLIENT,
        _build_llm,
        provider=provider,
        model=model,
        operation=operation,
        conversation_id=conversation_id,
    ) as handle:
        handle.set_request(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        if messages is not None:
            handle.set_messages(messages)
        if prompt_template is not None or prompt_version is not None:
            handle.set_prompt_template(
                prompt_template, version=prompt_version, variables=prompt_variables
            )
        yield handle


def _build_tool(span, config, *, name, tool_type, call_id, description):
    handle = ToolSpan(span, config)
    handle._set(sc.GEN_AI_OPERATION_NAME, sc.OperationName.EXECUTE_TOOL.value)
    handle._set(sc.GEN_AI_TOOL_NAME, name)
    handle._set(sc.GEN_AI_TOOL_TYPE, tool_type)
    handle._set(sc.GEN_AI_TOOL_CALL_ID, call_id)
    handle._set(sc.GEN_AI_TOOL_DESCRIPTION, description)
    return handle


@contextmanager
def tool_call(
    name: str,
    *,
    arguments: Any = None,
    tool_type: str = "function",
    call_id: str | None = None,
    description: str | None = None,
) -> Generator[ToolSpan, None, None]:
    """Trace a tool invocation."""
    with _span_scope(
        f"{sc.OperationName.EXECUTE_TOOL.value} {name}",
        SpanKind.INTERNAL,
        _build_tool,
        name=name,
        tool_type=tool_type,
        call_id=call_id,
        description=description,
    ) as handle:
        if arguments is not None:
            handle.set_arguments(arguments)
        yield handle


def _build_retrieval(span, config, *, top_k, data_source_id):
    handle = RetrievalSpan(span, config)
    handle._set(sc.GEN_AI_OPERATION_NAME, sc.OperationName.RETRIEVAL.value)
    handle._set(sc.RETRIEVAL_TOP_K, top_k)
    handle._set(sc.GEN_AI_DATA_SOURCE_ID, data_source_id)
    return handle


@contextmanager
def retrieval(
    name: str = "retrieval",
    *,
    query: str | None = None,
    top_k: int | None = None,
    data_source_id: str | None = None,
) -> Generator[RetrievalSpan, None, None]:
    """Trace a retrieval call."""
    with _span_scope(
        f"{sc.OperationName.RETRIEVAL.value} {name}",
        SpanKind.CLIENT,
        _build_retrieval,
        top_k=top_k,
        data_source_id=data_source_id,
    ) as handle:
        if query is not None:
            handle.set_query(query)
        yield handle


def _build_agent(span, config, *, name, agent_id, description, operation):
    handle = AgentSpan(span, config)
    handle._set(sc.GEN_AI_OPERATION_NAME, operation)
    handle._set(sc.GEN_AI_AGENT_NAME, name)
    handle._set(sc.GEN_AI_AGENT_ID, agent_id)
    handle._set(sc.GEN_AI_AGENT_DESCRIPTION, description)
    return handle


@contextmanager
def agent_step(
    name: str,
    *,
    agent_id: str | None = None,
    description: str | None = None,
    input: Any = None,
    operation: str = sc.OperationName.INVOKE_AGENT.value,
) -> Generator[AgentSpan, None, None]:
    """Trace an agent step. Nest LLM, tool, and retrieval spans inside it."""
    with _span_scope(
        f"{operation} {name}",
        SpanKind.INTERNAL,
        _build_agent,
        name=name,
        agent_id=agent_id,
        description=description,
        operation=operation,
    ) as handle:
        if input is not None:
            handle.set_input(input)
        yield handle


@contextmanager
def workflow(name: str) -> Generator[AgentSpan, None, None]:
    """Trace a whole workflow/run. The natural root of a trace tree."""
    config = _active_config()
    with _start_span(f"{sc.OperationName.INVOKE_WORKFLOW.value} {name}", SpanKind.SERVER) as span:
        handle = AgentSpan(span, config)
        handle._set(sc.GEN_AI_OPERATION_NAME, sc.OperationName.INVOKE_WORKFLOW.value)
        handle._set(sc.GEN_AI_WORKFLOW_NAME, name)
        try:
            yield handle
        except BaseException as exc:
            handle.record_exception(exc)
            raise
        finally:
            handle._finalize()
            try:
                span.end()
            except Exception:
                _note_sdk_error("failed to end workflow span")


# ---------------------------------------------------------------------------
# Decorators (sync + async)
# ---------------------------------------------------------------------------


def trace_tool(
    name: str | None = None,
    *,
    capture_arguments: bool = True,
    capture_result: bool = True,
    tool_type: str = "function",
    description: str | None = None,
) -> Callable[[F], F]:
    """Trace a function as a tool call. Works on sync and async functions.

        @trace_tool()
        def search_flights(origin: str, destination: str) -> list[dict]:
            ...
    """

    def wrap(func: F) -> F:
        tool_name = name or func.__name__
        tool_desc = description or (inspect.getdoc(func) or None)

        def _bind(args: tuple, kwargs: dict) -> Any:
            if not capture_arguments:
                return None
            try:
                bound = inspect.signature(func).bind_partial(*args, **kwargs)
                bound.apply_defaults()
                return dict(bound.arguments)
            except Exception:
                # A signature we cannot bind is not a reason to lose the span.
                return {"args": [repr(a) for a in args], "kwargs": {k: repr(v) for k, v in kwargs.items()}}

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tool_call(
                    tool_name,
                    arguments=_bind(args, kwargs),
                    tool_type=tool_type,
                    description=tool_desc,
                ) as handle:
                    result = await func(*args, **kwargs)
                    if capture_result:
                        handle.set_result(result)
                    return result

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tool_call(
                tool_name,
                arguments=_bind(args, kwargs),
                tool_type=tool_type,
                description=tool_desc,
            ) as handle:
                result = func(*args, **kwargs)
                if capture_result:
                    handle.set_result(result)
                return result

        return sync_wrapper  # type: ignore[return-value]

    return wrap


def trace_agent(
    name: str | None = None,
    *,
    agent_id: str | None = None,
    capture_output: bool = True,
) -> Callable[[F], F]:
    """Trace a function as an agent step. Works on sync and async functions."""

    def wrap(func: F) -> F:
        step_name = name or func.__name__

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with agent_step(step_name, agent_id=agent_id) as handle:
                    result = await func(*args, **kwargs)
                    if capture_output and result is not None:
                        handle.set_output(result)
                    return result

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with agent_step(step_name, agent_id=agent_id) as handle:
                result = func(*args, **kwargs)
                if capture_output and result is not None:
                    handle.set_output(result)
                return result

        return sync_wrapper  # type: ignore[return-value]

    return wrap


def trace_retrieval(
    name: str | None = None, *, top_k: int | None = None
) -> Callable[[F], F]:
    """Trace a function as a retrieval call.

    If the function returns a sequence of :class:`Document` (or dicts with
    ``id``/``score``/``content`` keys) the documents are recorded automatically.
    """

    def wrap(func: F) -> F:
        retrieval_name = name or func.__name__

        def _capture(handle: RetrievalSpan, result: Any) -> None:
            if isinstance(result, Sequence) and not isinstance(result, (str, bytes)):
                if all(isinstance(d, (Document, Mapping)) for d in result):
                    handle.set_documents(result)  # type: ignore[arg-type]

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with retrieval(retrieval_name, top_k=top_k) as handle:
                    result = await func(*args, **kwargs)
                    _capture(handle, result)
                    return result

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with retrieval(retrieval_name, top_k=top_k) as handle:
                result = func(*args, **kwargs)
                _capture(handle, result)
                return result

        return sync_wrapper  # type: ignore[return-value]

    return wrap


def trace_llm(
    model: str,
    *,
    provider: str = "unknown",
    operation: str = sc.OperationName.CHAT.value,
) -> Callable[[F], F]:
    """Trace a function as a model call.

    The wrapped function is not handed a span handle, so this decorator records
    structure, latency and errors but not usage, cost, or content. When you need
    those, use the :func:`llm_call` context manager instead -- it yields a
    handle you can call :meth:`LLMSpan.set_usage` on once the response returns.
    """

    def wrap(func: F) -> F:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with llm_call(model, provider=provider, operation=operation):
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with llm_call(model, provider=provider, operation=operation):
                return func(*args, **kwargs)

        return sync_wrapper  # type: ignore[return-value]

    return wrap


__all__ = [
    "AgentSpan",
    "Document",
    "LLMSpan",
    "RetrievalSpan",
    "ToolSpan",
    "agent_step",
    "llm_call",
    "retrieval",
    "tool_call",
    "trace_agent",
    "trace_llm",
    "trace_retrieval",
    "trace_tool",
    "workflow",
]
