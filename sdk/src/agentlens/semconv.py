"""OpenTelemetry GenAI semantic conventions, plus the custom attributes we add.

Design rule for this module: **never hard-code a convention string**. Every name
in the ``GEN_AI_*`` / ``ERROR_*`` block below is re-exported from the installed
``opentelemetry-semantic-conventions`` package, so an upstream rename cannot
silently leave us emitting a stale attribute name. ``tests/test_semconv.py``
additionally pins the *resolved literal* values, so an upstream rename is caught
as a test failure rather than propagating silently into stored traces.

Custom attributes are allowed only where the conventions have no equivalent.
Every one is registered in :data:`CUSTOM_ATTRIBUTES` with a written
justification, and a test asserts that (a) every custom attribute uses the
``agentlens.`` namespace, and (b) no custom attribute duplicates a convention
that already exists.
"""

from __future__ import annotations

from types import MappingProxyType

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as _genai,
)
from opentelemetry.semconv.attributes import error_attributes as _error

# ---------------------------------------------------------------------------
# Conventions (re-exported -- do not replace these with string literals)
# ---------------------------------------------------------------------------

# Operation / provider identification
GEN_AI_OPERATION_NAME = _genai.GEN_AI_OPERATION_NAME
GEN_AI_SYSTEM = _genai.GEN_AI_SYSTEM
GEN_AI_PROVIDER_NAME = _genai.GEN_AI_PROVIDER_NAME
GEN_AI_CONVERSATION_ID = _genai.GEN_AI_CONVERSATION_ID

# Request parameters
GEN_AI_REQUEST_MODEL = _genai.GEN_AI_REQUEST_MODEL
GEN_AI_REQUEST_TEMPERATURE = _genai.GEN_AI_REQUEST_TEMPERATURE
GEN_AI_REQUEST_TOP_P = _genai.GEN_AI_REQUEST_TOP_P
GEN_AI_REQUEST_TOP_K = _genai.GEN_AI_REQUEST_TOP_K
GEN_AI_REQUEST_MAX_TOKENS = _genai.GEN_AI_REQUEST_MAX_TOKENS
GEN_AI_REQUEST_STOP_SEQUENCES = _genai.GEN_AI_REQUEST_STOP_SEQUENCES
GEN_AI_REQUEST_FREQUENCY_PENALTY = _genai.GEN_AI_REQUEST_FREQUENCY_PENALTY
GEN_AI_REQUEST_PRESENCE_PENALTY = _genai.GEN_AI_REQUEST_PRESENCE_PENALTY
GEN_AI_REQUEST_SEED = _genai.GEN_AI_REQUEST_SEED
GEN_AI_REQUEST_CHOICE_COUNT = _genai.GEN_AI_REQUEST_CHOICE_COUNT
GEN_AI_REQUEST_STREAM = _genai.GEN_AI_REQUEST_STREAM

# Response
GEN_AI_RESPONSE_ID = _genai.GEN_AI_RESPONSE_ID
GEN_AI_RESPONSE_MODEL = _genai.GEN_AI_RESPONSE_MODEL
GEN_AI_RESPONSE_FINISH_REASONS = _genai.GEN_AI_RESPONSE_FINISH_REASONS

# Content (JSON-encoded; the conventions model these as complex types, and OTel
# span attributes accept only primitives, so serialization is required)
GEN_AI_INPUT_MESSAGES = _genai.GEN_AI_INPUT_MESSAGES
GEN_AI_OUTPUT_MESSAGES = _genai.GEN_AI_OUTPUT_MESSAGES
GEN_AI_SYSTEM_INSTRUCTIONS = _genai.GEN_AI_SYSTEM_INSTRUCTIONS
GEN_AI_PROMPT_NAME = _genai.GEN_AI_PROMPT_NAME

# Usage
GEN_AI_USAGE_INPUT_TOKENS = _genai.GEN_AI_USAGE_INPUT_TOKENS
GEN_AI_USAGE_OUTPUT_TOKENS = _genai.GEN_AI_USAGE_OUTPUT_TOKENS
GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS = _genai.GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS
GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS = (
    _genai.GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS
)
GEN_AI_USAGE_REASONING_OUTPUT_TOKENS = _genai.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS

# Tools
GEN_AI_TOOL_NAME = _genai.GEN_AI_TOOL_NAME
GEN_AI_TOOL_TYPE = _genai.GEN_AI_TOOL_TYPE
GEN_AI_TOOL_DESCRIPTION = _genai.GEN_AI_TOOL_DESCRIPTION
GEN_AI_TOOL_CALL_ID = _genai.GEN_AI_TOOL_CALL_ID
GEN_AI_TOOL_CALL_ARGUMENTS = _genai.GEN_AI_TOOL_CALL_ARGUMENTS
GEN_AI_TOOL_CALL_RESULT = _genai.GEN_AI_TOOL_CALL_RESULT

# Retrieval
GEN_AI_RETRIEVAL_QUERY_TEXT = _genai.GEN_AI_RETRIEVAL_QUERY_TEXT
GEN_AI_RETRIEVAL_DOCUMENTS = _genai.GEN_AI_RETRIEVAL_DOCUMENTS
GEN_AI_DATA_SOURCE_ID = _genai.GEN_AI_DATA_SOURCE_ID

# Agents / workflows
GEN_AI_AGENT_NAME = _genai.GEN_AI_AGENT_NAME
GEN_AI_AGENT_ID = _genai.GEN_AI_AGENT_ID
GEN_AI_AGENT_DESCRIPTION = _genai.GEN_AI_AGENT_DESCRIPTION
GEN_AI_WORKFLOW_NAME = _genai.GEN_AI_WORKFLOW_NAME

# Errors
ERROR_TYPE = _error.ERROR_TYPE

# Enumerated values
OperationName = _genai.GenAiOperationNameValues
ProviderName = _genai.GenAiProviderNameValues

#: Every convention attribute this SDK is capable of emitting. Used by the
#: drift test and by :mod:`agentlens.redaction` to decide which attributes carry
#: free-text content.
CONVENTION_ATTRIBUTES: frozenset[str] = frozenset(
    {
        GEN_AI_OPERATION_NAME,
        GEN_AI_SYSTEM,
        GEN_AI_PROVIDER_NAME,
        GEN_AI_CONVERSATION_ID,
        GEN_AI_REQUEST_MODEL,
        GEN_AI_REQUEST_TEMPERATURE,
        GEN_AI_REQUEST_TOP_P,
        GEN_AI_REQUEST_TOP_K,
        GEN_AI_REQUEST_MAX_TOKENS,
        GEN_AI_REQUEST_STOP_SEQUENCES,
        GEN_AI_REQUEST_FREQUENCY_PENALTY,
        GEN_AI_REQUEST_PRESENCE_PENALTY,
        GEN_AI_REQUEST_SEED,
        GEN_AI_REQUEST_CHOICE_COUNT,
        GEN_AI_REQUEST_STREAM,
        GEN_AI_RESPONSE_ID,
        GEN_AI_RESPONSE_MODEL,
        GEN_AI_RESPONSE_FINISH_REASONS,
        GEN_AI_INPUT_MESSAGES,
        GEN_AI_OUTPUT_MESSAGES,
        GEN_AI_SYSTEM_INSTRUCTIONS,
        GEN_AI_PROMPT_NAME,
        GEN_AI_USAGE_INPUT_TOKENS,
        GEN_AI_USAGE_OUTPUT_TOKENS,
        GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
        GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
        GEN_AI_USAGE_REASONING_OUTPUT_TOKENS,
        GEN_AI_TOOL_NAME,
        GEN_AI_TOOL_TYPE,
        GEN_AI_TOOL_DESCRIPTION,
        GEN_AI_TOOL_CALL_ID,
        GEN_AI_TOOL_CALL_ARGUMENTS,
        GEN_AI_TOOL_CALL_RESULT,
        GEN_AI_RETRIEVAL_QUERY_TEXT,
        GEN_AI_RETRIEVAL_DOCUMENTS,
        GEN_AI_DATA_SOURCE_ID,
        GEN_AI_AGENT_NAME,
        GEN_AI_AGENT_ID,
        GEN_AI_AGENT_DESCRIPTION,
        GEN_AI_WORKFLOW_NAME,
        ERROR_TYPE,
    }
)

# ---------------------------------------------------------------------------
# Custom attributes -- each requires a justification
# ---------------------------------------------------------------------------

NAMESPACE = "agentlens"

# Cost. The GenAI conventions define token usage but no monetary cost attribute,
# and cost is a first-class objective for this platform (regression gates and
# the Pareto frontier both rank on it).
COST_MICRO_USD = "agentlens.cost.micro_usd"
COST_CURRENCY = "agentlens.cost.currency"
COST_PRICING_VERSION = "agentlens.cost.pricing_version"
COST_STATUS = "agentlens.cost.status"

# Prompt templating. `gen_ai.prompt.name` names a prompt but the conventions
# carry no template body, template version, or variable bindings. Regression
# gating attributes a score change to a specific prompt version, so the version
# must travel on the span.
PROMPT_TEMPLATE = "agentlens.prompt.template"
PROMPT_TEMPLATE_VERSION = "agentlens.prompt.template_version"
PROMPT_VARIABLES = "agentlens.prompt.variables"

# Retrieval shape. `gen_ai.retrieval.documents` carries the documents and their
# scores, but there is no attribute for the requested top-k or the returned
# count. Both are needed to filter traces without parsing the document JSON.
RETRIEVAL_TOP_K = "agentlens.retrieval.top_k"
RETRIEVAL_DOCUMENT_COUNT = "agentlens.retrieval.document_count"

# Redaction audit. Records that redaction ran and which rules matched, so a
# reviewer can distinguish "no PII present" from "redaction was misconfigured".
REDACTION_APPLIED = "agentlens.redaction.applied"
REDACTION_RULES = "agentlens.redaction.rules"

# Provenance. Identifies the SDK build that produced a span, so a change in
# capture behaviour is attributable when comparing historical traces.
SDK_VERSION = "agentlens.sdk.version"
SDK_LANGUAGE = "agentlens.sdk.language"

#: Custom attribute name -> justification. The justification is not decoration:
#: ``tests/test_semconv.py`` asserts every custom attribute appears here, and
#: ``docs``/README render this mapping so the docs cannot drift from the code.
CUSTOM_ATTRIBUTES: MappingProxyType[str, str] = MappingProxyType(
    {
        COST_MICRO_USD: (
            "Computed cost in millionths of a currency unit (integer, so money "
            "is never stored as float). The GenAI conventions define token "
            "usage but no cost attribute."
        ),
        COST_CURRENCY: (
            "ISO-4217 code for COST_MICRO_USD. Explicit so a future non-USD "
            "pricing table cannot be misread as USD."
        ),
        COST_PRICING_VERSION: (
            "Version of the pricing table used for the computation. Without it "
            "a vendor price change silently rewrites the meaning of historical "
            "cost comparisons."
        ),
        COST_STATUS: (
            "Why cost is absent or partial: 'ok', 'no_pricing' (model not in "
            "the table), or 'no_usage' (token counts unavailable). Distinguishes "
            "a genuine zero from a missing measurement."
        ),
        PROMPT_TEMPLATE: (
            "The un-rendered prompt template. The conventions capture only the "
            "rendered messages (gen_ai.input.messages); the template is what a "
            "prompt-change regression gate actually varies."
        ),
        PROMPT_TEMPLATE_VERSION: (
            "Version identifier of the template above, so a score delta can be "
            "attributed to a specific prompt revision."
        ),
        PROMPT_VARIABLES: (
            "JSON object of variables bound into the template, so the rendered "
            "prompt is reproducible from template + variables."
        ),
        RETRIEVAL_TOP_K: (
            "Requested number of documents. No convention exists; needed to "
            "distinguish 'retriever returned few' from 'we asked for few'."
        ),
        RETRIEVAL_DOCUMENT_COUNT: (
            "Number of documents actually returned, denormalised out of "
            "gen_ai.retrieval.documents so queries can filter on it without "
            "parsing JSON."
        ),
        REDACTION_APPLIED: (
            "True when the redaction pipeline ran on this span. Lets an auditor "
            "prove redaction was active rather than assuming it."
        ),
        REDACTION_RULES: (
            "Names of redaction rules that matched. Distinguishes 'no PII "
            "found' from 'redaction misconfigured and matched nothing'."
        ),
        SDK_VERSION: (
            "agentlens SDK version that produced the span. Capture behaviour "
            "changes between versions; comparisons across a version boundary "
            "need to be attributable."
        ),
        SDK_LANGUAGE: (
            "Implementation language of the instrumenting SDK, for when "
            "non-Python SDKs emit into the same backend."
        ),
    }
)


def custom_attribute_names() -> frozenset[str]:
    """Return every custom (non-convention) attribute this SDK may emit."""
    return frozenset(CUSTOM_ATTRIBUTES)
