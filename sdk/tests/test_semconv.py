"""Semantic-convention discipline.

Two failure modes this guards against:

1. **Silent upstream rename.** We bind to the semconv package rather than
   hard-coding strings, so a rename cannot leave us emitting a stale name -- but
   it *could* silently change what we write into stored traces. Pinning the
   resolved literals turns that into a test failure.
2. **Invented attributes.** A custom attribute that duplicates an existing
   convention fragments queries. Every custom name must be registered with a
   justification and must not shadow a convention.
"""

from __future__ import annotations

import pytest

from agentlens import semconv as sc

# Literal values as of opentelemetry-semantic-conventions 0.65b0. If an upgrade
# changes one of these, that is a deliberate decision with migration
# consequences for stored traces -- not something to silently absorb.
PINNED_CONVENTIONS = {
    "GEN_AI_OPERATION_NAME": "gen_ai.operation.name",
    "GEN_AI_SYSTEM": "gen_ai.system",
    "GEN_AI_PROVIDER_NAME": "gen_ai.provider.name",
    "GEN_AI_REQUEST_MODEL": "gen_ai.request.model",
    "GEN_AI_REQUEST_TEMPERATURE": "gen_ai.request.temperature",
    "GEN_AI_REQUEST_TOP_P": "gen_ai.request.top_p",
    "GEN_AI_REQUEST_MAX_TOKENS": "gen_ai.request.max_tokens",
    "GEN_AI_USAGE_INPUT_TOKENS": "gen_ai.usage.input_tokens",
    "GEN_AI_USAGE_OUTPUT_TOKENS": "gen_ai.usage.output_tokens",
    "GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS": "gen_ai.usage.cache_read.input_tokens",
    "GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS": "gen_ai.usage.cache_creation.input_tokens",
    "GEN_AI_INPUT_MESSAGES": "gen_ai.input.messages",
    "GEN_AI_OUTPUT_MESSAGES": "gen_ai.output.messages",
    "GEN_AI_TOOL_NAME": "gen_ai.tool.name",
    "GEN_AI_TOOL_CALL_ID": "gen_ai.tool.call.id",
    "GEN_AI_TOOL_CALL_ARGUMENTS": "gen_ai.tool.call.arguments",
    "GEN_AI_TOOL_CALL_RESULT": "gen_ai.tool.call.result",
    "GEN_AI_RETRIEVAL_QUERY_TEXT": "gen_ai.retrieval.query.text",
    "GEN_AI_RETRIEVAL_DOCUMENTS": "gen_ai.retrieval.documents",
    "GEN_AI_AGENT_NAME": "gen_ai.agent.name",
    "GEN_AI_AGENT_ID": "gen_ai.agent.id",
    "GEN_AI_WORKFLOW_NAME": "gen_ai.workflow.name",
    "GEN_AI_RESPONSE_ID": "gen_ai.response.id",
    "GEN_AI_RESPONSE_MODEL": "gen_ai.response.model",
    "GEN_AI_RESPONSE_FINISH_REASONS": "gen_ai.response.finish_reasons",
    "GEN_AI_CONVERSATION_ID": "gen_ai.conversation.id",
    "GEN_AI_DATA_SOURCE_ID": "gen_ai.data_source.id",
    "GEN_AI_PROMPT_NAME": "gen_ai.prompt.name",
    "ERROR_TYPE": "error.type",
}


@pytest.mark.parametrize("name,expected", sorted(PINNED_CONVENTIONS.items()))
def test_convention_names_have_not_drifted(name: str, expected: str) -> None:
    assert getattr(sc, name) == expected


def test_operation_name_values_are_convention_enums() -> None:
    assert sc.OperationName.CHAT.value == "chat"
    assert sc.OperationName.EXECUTE_TOOL.value == "execute_tool"
    assert sc.OperationName.INVOKE_AGENT.value == "invoke_agent"
    assert sc.OperationName.RETRIEVAL.value == "retrieval"
    assert sc.OperationName.INVOKE_WORKFLOW.value == "invoke_workflow"


def test_every_custom_attribute_is_namespaced() -> None:
    for name in sc.custom_attribute_names():
        assert name.startswith(f"{sc.NAMESPACE}."), (
            f"{name} is not in the agentlens namespace, so it could collide "
            f"with a current or future OpenTelemetry convention"
        )


def test_no_custom_attribute_shadows_a_convention() -> None:
    overlap = sc.custom_attribute_names() & sc.CONVENTION_ATTRIBUTES
    assert not overlap, f"custom attributes duplicate conventions: {overlap}"


def test_every_custom_attribute_has_a_justification() -> None:
    for name, reason in sc.CUSTOM_ATTRIBUTES.items():
        assert reason and len(reason) > 40, (
            f"{name} lacks a substantive justification; a custom attribute "
            f"without a stated reason is an invented attribute"
        )


def test_custom_attribute_registry_is_immutable() -> None:
    # A mutable registry would let a caller add an unjustified attribute at
    # runtime and defeat the checks above.
    with pytest.raises(TypeError):
        sc.CUSTOM_ATTRIBUTES["agentlens.sneaky"] = "x"  # type: ignore[index]


def test_retrieval_uses_conventions_not_custom_names() -> None:
    """Retrieval had a real temptation to invent names -- assert we did not."""
    assert sc.GEN_AI_RETRIEVAL_QUERY_TEXT in sc.CONVENTION_ATTRIBUTES
    assert sc.GEN_AI_RETRIEVAL_DOCUMENTS in sc.CONVENTION_ATTRIBUTES
    # Only genuinely-missing retrieval concepts are custom.
    assert sc.RETRIEVAL_TOP_K in sc.CUSTOM_ATTRIBUTES
    assert sc.RETRIEVAL_DOCUMENT_COUNT in sc.CUSTOM_ATTRIBUTES
