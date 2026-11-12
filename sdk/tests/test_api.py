"""Instrumentation API: trace-tree shape, convention compliance, safety."""

from __future__ import annotations

import json

import pytest

import agentlens
from agentlens import Document, semconv as sc
from agentlens.config import AgentLensConfig


def finished(lens):
    return lens.memory_exporter.get_finished_spans()


def by_name(lens):
    return {s.name: s for s in finished(lens)}


# -- trace tree reconstruction ----------------------------------------------


def test_full_decision_tree_is_reconstructable(lens) -> None:
    """The whole point: parent/child links must rebuild the agent's decisions."""
    with agentlens.workflow("support_request"):
        with agentlens.agent_step("triage"):
            with agentlens.retrieval("kb", query="refund policy", top_k=3) as r:
                r.set_documents([Document(id="d1", score=0.91, content="policy text")])
            with agentlens.llm_call("claude-opus-5", provider="anthropic") as llm:
                llm.set_usage(100, 20)
            with agentlens.tool_call("issue_refund") as tool:
                tool.set_result({"ok": True})

    spans = {s.name: s for s in finished(lens)}
    assert set(spans) == {
        "invoke_workflow support_request",
        "invoke_agent triage",
        "retrieval kb",
        "chat claude-opus-5",
        "execute_tool issue_refund",
    }

    workflow = spans["invoke_workflow support_request"]
    agent = spans["invoke_agent triage"]
    # One trace, correct nesting depth.
    assert agent.parent.span_id == workflow.context.span_id
    for child in ("retrieval kb", "chat claude-opus-5", "execute_tool issue_refund"):
        assert spans[child].parent.span_id == agent.context.span_id
        assert spans[child].context.trace_id == workflow.context.trace_id


def test_sibling_llm_calls_are_ordered_and_distinct(lens) -> None:
    with agentlens.agent_step("react_loop"):
        for _ in range(3):
            with agentlens.llm_call("claude-opus-5", provider="anthropic"):
                pass
    llm_spans = [s for s in finished(lens) if s.name == "chat claude-opus-5"]
    assert len(llm_spans) == 3
    assert len({s.context.span_id for s in llm_spans}) == 3


# -- convention compliance ---------------------------------------------------


def test_llm_span_uses_genai_conventions(lens) -> None:
    with agentlens.llm_call(
        "claude-opus-5",
        provider="anthropic",
        temperature=0.2,
        max_tokens=1024,
        conversation_id="conv-7",
    ) as span:
        span.set_messages([{"role": "user", "content": "hello"}])
        span.set_response("hi there", response_id="msg_1", finish_reasons=["end_turn"])
        span.set_usage(120, 34, cache_read_input_tokens=1000)

    attrs = by_name(lens)["chat claude-opus-5"].attributes
    assert attrs[sc.GEN_AI_OPERATION_NAME] == "chat"
    assert attrs[sc.GEN_AI_REQUEST_MODEL] == "claude-opus-5"
    assert attrs[sc.GEN_AI_REQUEST_TEMPERATURE] == 0.2
    assert attrs[sc.GEN_AI_REQUEST_MAX_TOKENS] == 1024
    assert attrs[sc.GEN_AI_CONVERSATION_ID] == "conv-7"
    assert attrs[sc.GEN_AI_USAGE_INPUT_TOKENS] == 120
    assert attrs[sc.GEN_AI_USAGE_OUTPUT_TOKENS] == 34
    assert attrs[sc.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 1000
    assert attrs[sc.GEN_AI_RESPONSE_ID] == "msg_1"
    assert attrs[sc.GEN_AI_RESPONSE_FINISH_REASONS] == ("end_turn",)


def test_provider_is_emitted_under_both_convention_names(lens) -> None:
    """gen_ai.system is widely indexed; gen_ai.provider.name supersedes it."""
    with agentlens.llm_call("claude-opus-5", provider="anthropic"):
        pass
    attrs = by_name(lens)["chat claude-opus-5"].attributes
    assert attrs[sc.GEN_AI_SYSTEM] == "anthropic"
    assert attrs[sc.GEN_AI_PROVIDER_NAME] == "anthropic"


def test_tool_span_uses_tool_call_conventions(lens) -> None:
    with agentlens.tool_call(
        "get_weather", arguments={"city": "Paris"}, call_id="toolu_1"
    ) as span:
        span.set_result("18C and clear")

    attrs = by_name(lens)["execute_tool get_weather"].attributes
    assert attrs[sc.GEN_AI_OPERATION_NAME] == "execute_tool"
    assert attrs[sc.GEN_AI_TOOL_NAME] == "get_weather"
    assert attrs[sc.GEN_AI_TOOL_CALL_ID] == "toolu_1"
    assert json.loads(attrs[sc.GEN_AI_TOOL_CALL_ARGUMENTS]) == {"city": "Paris"}
    assert attrs[sc.GEN_AI_TOOL_CALL_RESULT] == "18C and clear"


def test_retrieval_span_records_documents_with_scores(lens) -> None:
    with agentlens.retrieval("vector_store", query="how do refunds work", top_k=2) as r:
        r.set_documents(
            [
                Document(id="a", score=0.93, content="Refunds within 30 days."),
                Document(id="b", score=0.71, content="Shipping policy."),
            ]
        )

    attrs = by_name(lens)["retrieval vector_store"].attributes
    assert attrs[sc.GEN_AI_OPERATION_NAME] == "retrieval"
    assert attrs[sc.GEN_AI_RETRIEVAL_QUERY_TEXT] == "how do refunds work"
    assert attrs[sc.RETRIEVAL_TOP_K] == 2
    assert attrs[sc.RETRIEVAL_DOCUMENT_COUNT] == 2
    docs = json.loads(attrs[sc.GEN_AI_RETRIEVAL_DOCUMENTS])
    assert [d["score"] for d in docs] == [0.93, 0.71]
    assert docs[0]["id"] == "a"


def test_agent_span_records_agent_identity(lens) -> None:
    with agentlens.agent_step("planner", agent_id="agent-1") as span:
        span.set_output("plan ready")
    attrs = by_name(lens)["invoke_agent planner"].attributes
    assert attrs[sc.GEN_AI_AGENT_NAME] == "planner"
    assert attrs[sc.GEN_AI_AGENT_ID] == "agent-1"
    assert attrs[sc.GEN_AI_OPERATION_NAME] == "invoke_agent"


# -- prompt template capture -------------------------------------------------


def test_prompt_template_and_rendered_prompt_are_both_captured(lens) -> None:
    """Regression gating needs the template version, not just the rendered text."""
    with agentlens.llm_call("claude-opus-5", provider="anthropic") as span:
        span.set_prompt_template(
            "Answer as {persona}: {question}",
            name="qa_prompt",
            version="v3",
            variables={"persona": "an analyst", "question": "what is EBITDA?"},
        )
        span.set_messages([{"role": "user", "content": "Answer as an analyst: ..."}])

    attrs = by_name(lens)["chat claude-opus-5"].attributes
    assert attrs[sc.PROMPT_TEMPLATE] == "Answer as {persona}: {question}"
    assert attrs[sc.PROMPT_TEMPLATE_VERSION] == "v3"
    assert attrs[sc.GEN_AI_PROMPT_NAME] == "qa_prompt"
    assert "analyst" in attrs[sc.PROMPT_VARIABLES]
    assert sc.GEN_AI_INPUT_MESSAGES in attrs


# -- cost --------------------------------------------------------------------


def test_cost_is_computed_from_usage_and_pricing_table(lens) -> None:
    with agentlens.llm_call("claude-opus-5", provider="anthropic") as span:
        span.set_usage(1_000_000, 0)
    attrs = by_name(lens)["chat claude-opus-5"].attributes
    assert attrs[sc.COST_MICRO_USD] == 5_000_000
    assert attrs[sc.COST_CURRENCY] == "USD"
    assert attrs[sc.COST_STATUS] == "ok"
    assert attrs[sc.COST_PRICING_VERSION] == "2026-11-12"


def test_unpriced_model_records_status_not_a_fake_zero(lens) -> None:
    with agentlens.llm_call("some-unlisted-model", provider="anthropic") as span:
        span.set_usage(100, 10)
    attrs = by_name(lens)["chat some-unlisted-model"].attributes
    assert attrs[sc.COST_STATUS] == "no_pricing"
    assert sc.COST_MICRO_USD not in attrs


def test_missing_usage_records_no_usage_status(lens) -> None:
    with agentlens.llm_call("claude-opus-5", provider="anthropic"):
        pass
    attrs = by_name(lens)["chat claude-opus-5"].attributes
    assert attrs[sc.COST_STATUS] == "no_usage"
    assert sc.COST_MICRO_USD not in attrs


# -- redaction at capture time ----------------------------------------------


def test_pii_is_redacted_before_it_is_stored_on_the_span(lens) -> None:
    """Raw PII must never exist as a span attribute, not merely be scrubbed later."""
    with agentlens.llm_call("claude-opus-5", provider="anthropic") as span:
        span.set_messages(
            [{"role": "user", "content": "my email is alice@example.com"}]
        )
        span.set_response("Confirmed for alice@example.com")

    attrs = by_name(lens)["chat claude-opus-5"].attributes
    assert "alice@example.com" not in attrs[sc.GEN_AI_INPUT_MESSAGES]
    assert "alice@example.com" not in attrs[sc.GEN_AI_OUTPUT_MESSAGES]
    assert "[REDACTED:email]" in attrs[sc.GEN_AI_INPUT_MESSAGES]
    assert attrs[sc.REDACTION_APPLIED] is True
    assert "email" in attrs[sc.REDACTION_RULES]


def test_document_text_is_redacted(lens) -> None:
    with agentlens.retrieval("kb") as r:
        r.set_documents([Document(id="d", score=0.5, content="call 555-123-4567")])
    attrs = by_name(lens)["retrieval kb"].attributes
    assert "555-123-4567" not in attrs[sc.GEN_AI_RETRIEVAL_DOCUMENTS]


# -- content suppression -----------------------------------------------------


def test_capture_content_false_drops_text_but_keeps_metrics() -> None:
    """A privacy-restricted deployment keeps every non-content measurement."""
    lens = agentlens.init(
        AgentLensConfig(
            service_name="t",
            exporter="memory",
            capture_content=False,
            set_global_tracer_provider=False,
        )
    )
    try:
        with agentlens.llm_call("claude-opus-5", provider="anthropic") as span:
            span.set_messages([{"role": "user", "content": "sensitive question"}])
            span.set_response("sensitive answer")
            span.set_usage(1_000_000, 0)
        with agentlens.retrieval("kb") as r:
            r.set_documents([Document(id="d", score=0.4, content="secret corpus text")])

        spans = {s.name: s for s in lens.memory_exporter.get_finished_spans()}
        llm = spans["chat claude-opus-5"].attributes
        assert sc.GEN_AI_INPUT_MESSAGES not in llm
        assert sc.GEN_AI_OUTPUT_MESSAGES not in llm
        # Structure, tokens and cost survive.
        assert llm[sc.GEN_AI_USAGE_INPUT_TOKENS] == 1_000_000
        assert llm[sc.COST_MICRO_USD] == 5_000_000

        ret = spans["retrieval kb"].attributes
        assert ret[sc.RETRIEVAL_DOCUMENT_COUNT] == 1
        assert "secret corpus text" not in ret.get(sc.GEN_AI_RETRIEVAL_DOCUMENTS, "")
    finally:
        agentlens.shutdown()


def test_long_content_is_truncated(lens) -> None:
    with agentlens.llm_call("claude-opus-5", provider="anthropic") as span:
        span.set_response("x" * 50_000)
    attrs = by_name(lens)["chat claude-opus-5"].attributes
    assert len(attrs[sc.GEN_AI_OUTPUT_MESSAGES]) < 50_000
    assert "truncated" in attrs[sc.GEN_AI_OUTPUT_MESSAGES]


# -- error handling ----------------------------------------------------------


def test_exception_is_recorded_and_re_raised_unchanged(lens) -> None:
    """Swallowing the caller's exception would corrupt their control flow."""

    class DomainError(RuntimeError):
        pass

    with pytest.raises(DomainError, match="rate limited"):
        with agentlens.llm_call("claude-opus-5", provider="anthropic"):
            raise DomainError("rate limited")

    span = by_name(lens)["chat claude-opus-5"]
    assert span.status.status_code.name == "ERROR"
    assert span.attributes[sc.ERROR_TYPE] == "DomainError"
    assert any(e.name == "exception" for e in span.events)


def test_failed_child_still_leaves_a_complete_tree(lens) -> None:
    with pytest.raises(ValueError):
        with agentlens.workflow("run"):
            with agentlens.tool_call("bad_tool"):
                raise ValueError("tool blew up")

    spans = by_name(lens)
    assert "invoke_workflow run" in spans
    assert spans["execute_tool bad_tool"].status.status_code.name == "ERROR"
    assert spans["invoke_workflow run"].status.status_code.name == "ERROR"


# -- decorators, sync and async ---------------------------------------------


def test_tool_decorator_captures_bound_arguments_and_result(lens) -> None:
    @agentlens.trace_tool()
    def lookup_order(order_id: str, include_items: bool = False) -> dict:
        return {"order_id": order_id, "status": "shipped"}

    assert lookup_order("A-1")["status"] == "shipped"

    attrs = by_name(lens)["execute_tool lookup_order"].attributes
    args = json.loads(attrs[sc.GEN_AI_TOOL_CALL_ARGUMENTS])
    # Defaults are applied, so the recorded call is reproducible.
    assert args == {"order_id": "A-1", "include_items": False}
    assert "shipped" in attrs[sc.GEN_AI_TOOL_CALL_RESULT]


async def test_async_tool_decorator_works(lens) -> None:
    @agentlens.trace_tool()
    async def fetch_profile(user_id: str) -> dict:
        return {"user_id": user_id}

    assert await fetch_profile("u-9") == {"user_id": "u-9"}
    assert "execute_tool fetch_profile" in by_name(lens)


async def test_async_agent_and_retrieval_decorators_nest(lens) -> None:
    @agentlens.trace_retrieval(top_k=1)
    async def search(q: str):
        return [Document(id="x", score=0.8, content="hit")]

    @agentlens.trace_agent("async_planner")
    async def plan():
        await search("q")
        return "done"

    assert await plan() == "done"

    spans = by_name(lens)
    assert spans["retrieval search"].parent.span_id == (
        spans["invoke_agent async_planner"].context.span_id
    )
    assert spans["retrieval search"].attributes[sc.RETRIEVAL_DOCUMENT_COUNT] == 1


def test_llm_decorator_records_model_and_provider(lens) -> None:
    @agentlens.trace_llm("claude-opus-5", provider="anthropic")
    def summarise(text: str) -> str:
        return text[:5]

    assert summarise("hello world") == "hello"

    attrs = by_name(lens)["chat claude-opus-5"].attributes
    assert attrs[sc.GEN_AI_REQUEST_MODEL] == "claude-opus-5"
    assert attrs[sc.GEN_AI_PROVIDER_NAME] == "anthropic"
    # Documented tradeoff: the wrapped function gets no handle, so there is no
    # usage and therefore no cost. Asserting the absence keeps the docstring
    # honest -- a silent zero here would be a fabricated number.
    assert sc.GEN_AI_USAGE_INPUT_TOKENS not in attrs
    assert attrs[sc.COST_STATUS] == "no_usage"


async def test_async_llm_decorator_nests_under_agent(lens) -> None:
    @agentlens.trace_llm("claude-haiku-4-5", provider="anthropic")
    async def draft() -> str:
        return "ok"

    @agentlens.trace_agent("writer")
    async def run() -> str:
        return await draft()

    assert await run() == "ok"

    spans = by_name(lens)
    assert spans["chat claude-haiku-4-5"].parent.span_id == (
        spans["invoke_agent writer"].context.span_id
    )


def test_llm_decorator_records_exceptions_and_reraises(lens) -> None:
    @agentlens.trace_llm("claude-opus-5", provider="anthropic")
    def overloaded():
        raise RuntimeError("529 overloaded")

    with pytest.raises(RuntimeError):
        overloaded()
    span = by_name(lens)["chat claude-opus-5"]
    assert span.status.status_code.name == "ERROR"
    assert span.attributes[sc.ERROR_TYPE] == "RuntimeError"


def test_decorator_preserves_function_metadata(lens) -> None:
    @agentlens.trace_tool()
    def documented(x: int) -> int:
        """Adds one."""
        return x + 1

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Adds one."


def test_decorated_function_exception_propagates(lens) -> None:
    @agentlens.trace_tool()
    def failing():
        raise KeyError("missing")

    with pytest.raises(KeyError):
        failing()
    assert by_name(lens)["execute_tool failing"].status.status_code.name == "ERROR"


# -- safety when uninitialised ----------------------------------------------


def test_api_works_without_init_and_never_raises() -> None:
    """Instrumented libraries must import and run whether or not init() ran."""
    agentlens.shutdown()
    with agentlens.workflow("no_init"):
        with agentlens.agent_step("step"):
            with agentlens.llm_call("claude-opus-5", provider="anthropic") as span:
                span.set_usage(10, 5)
                span.set_response("ok")
            with agentlens.tool_call("t", arguments={"a": 1}) as tool:
                tool.set_result("r")


def test_disabled_sdk_is_a_no_op_but_code_still_runs() -> None:
    lens = agentlens.init(
        AgentLensConfig(service_name="t", disabled=True, set_global_tracer_provider=False)
    )
    try:
        calls = []
        with agentlens.llm_call("claude-opus-5", provider="anthropic") as span:
            span.set_usage(1, 1)
            calls.append("ran")
        assert calls == ["ran"]
        assert lens.processor is None
    finally:
        agentlens.shutdown()


def test_trace_and_span_ids_are_exposed_for_log_correlation(lens) -> None:
    with agentlens.llm_call("claude-opus-5", provider="anthropic") as span:
        trace_id, span_id = span.trace_id, span.span_id
    assert len(trace_id) == 32 and int(trace_id, 16) != 0
    assert len(span_id) == 16 and int(span_id, 16) != 0


def test_unserialisable_content_does_not_break_the_span(lens) -> None:
    class Weird:
        def __repr__(self):
            raise RuntimeError("no repr for you")

    with agentlens.tool_call("weird") as span:
        span.set_result({"obj": Weird()})
    # The span still exists and ended cleanly.
    assert "execute_tool weird" in by_name(lens)


def test_resource_carries_tenant_and_sdk_provenance(lens) -> None:
    with agentlens.llm_call("claude-opus-5", provider="anthropic"):
        pass
    resource = by_name(lens)["chat claude-opus-5"].resource.attributes
    assert resource["service.name"] == "test-service"
    assert resource["agentlens.tenant"] == "test-tenant"
    assert resource["agentlens.project"] == "test-project"
    assert resource[sc.SDK_VERSION] == agentlens.__version__
    assert resource[sc.SDK_LANGUAGE] == "python"
