"""OTLP protobuf spans -> ClickHouse rows.

This is the only place that knows the wire format. It is deliberately total: an
attribute the platform does not model is preserved in the ``attributes`` map
rather than discarded, because the whole point of capturing a trace is that you
do not know in advance which field will explain an incident.

Trace and span ids arrive as raw bytes on the wire and are stored as lowercase
hex, matching what every OTel tool and the W3C traceparent header display. A
user pasting an id from a log must be able to find the trace.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, Span, Status

# Span kind enum -> the name OTel tooling uses.
_KIND_NAMES = {
    Span.SPAN_KIND_UNSPECIFIED: "unspecified",
    Span.SPAN_KIND_INTERNAL: "internal",
    Span.SPAN_KIND_SERVER: "server",
    Span.SPAN_KIND_CLIENT: "client",
    Span.SPAN_KIND_PRODUCER: "producer",
    Span.SPAN_KIND_CONSUMER: "consumer",
}

_STATUS_NAMES = {
    Status.STATUS_CODE_UNSET: "unset",
    Status.STATUS_CODE_OK: "ok",
    Status.STATUS_CODE_ERROR: "error",
}

#: Attributes promoted to typed ClickHouse columns. Anything listed here is
#: removed from the generic map so the same value is not stored twice.
_PROMOTED = frozenset(
    {
        "gen_ai.operation.name",
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "gen_ai.response.model",
        "gen_ai.request.temperature",
        "gen_ai.usage.input_tokens",
        "gen_ai.usage.output_tokens",
        "gen_ai.tool.name",
        "gen_ai.agent.name",
        "agentlens.cost.micro_usd",
        "agentlens.cost.status",
        "agentlens.cost.pricing_version",
        "agentlens.prompt.template",
        "agentlens.prompt.template_version",
        "agentlens.retrieval.top_k",
        "agentlens.retrieval.document_count",
        "agentlens.redaction.applied",
        "agentlens.redaction.rules",
        "agentlens.sdk.version",
        "agentlens.sdk.language",
        "error.type",
    }
)


def _anyvalue(value: Any) -> Any:
    """Unwrap an OTLP ``AnyValue`` into a Python value."""
    kind = value.WhichOneof("value")
    if kind == "string_value":
        return value.string_value
    if kind == "bool_value":
        return value.bool_value
    if kind == "int_value":
        return value.int_value
    if kind == "double_value":
        return value.double_value
    if kind == "array_value":
        return [_anyvalue(item) for item in value.array_value.values]
    if kind == "kvlist_value":
        return {kv.key: _anyvalue(kv.value) for kv in value.kvlist_value.values}
    if kind == "bytes_value":
        return value.bytes_value.hex()
    return None


def _attributes(pairs: Any) -> dict[str, Any]:
    return {kv.key: _anyvalue(kv.value) for kv in pairs}


def _as_map_value(value: Any) -> str:
    """Render a leftover attribute for the ``Map(String, String)`` column.

    ClickHouse maps are homogeneous, so non-string values are JSON-encoded
    rather than coerced with ``str()`` -- ``str(True)`` is ``'True'``, which is
    not valid JSON and would not round-trip.
    """
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _opt_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    # The columns are unsigned; a negative token count is corrupt input, and
    # storing it would raise at insert time and fail the whole batch.
    return parsed if parsed >= 0 else None


def _opt_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    """Coerce to the non-nullable String columns, which use '' for absent."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return _as_map_value(value)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [item if isinstance(item, str) else _as_map_value(item) for item in value]
    return [_as_map_value(value)]


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _ns_to_datetime(nanos: int) -> datetime:
    """OTLP timestamps are unsigned nanoseconds since the Unix epoch.

    Integer arithmetic rather than ``datetime.fromtimestamp(nanos / 1e9)``:
    at present-day epoch values a float64 mantissa resolves to roughly 0.4
    microseconds, so the float route can introduce error the input did not
    have. This form cannot.

    It does *not* make sibling spans distinguishable, and nothing here could.
    Measured on this machine, three nested spans came out of the SDK carrying a
    byte-identical ``start_time_unix_nano`` -- the ties are in the OS clock, at
    the source, before any conversion. Both the float and integer forms yield
    the same two distinct values from those four spans. Anything ordering spans
    by timestamp must therefore treat ties as normal; that is why the root-span
    rollup in queries.py keys on ``parent_span_id`` instead.

    Precision is capped at microseconds because that is Python's datetime
    resolution; the exact nanosecond duration survives in ``duration_ns``,
    which is what latency analysis actually reads.
    """
    seconds, remainder = divmod(int(nanos), 1_000_000_000)
    return _EPOCH + timedelta(seconds=seconds, microseconds=remainder // 1_000)


def _events_json(span: Span) -> str:
    if not span.events:
        return "[]"
    return json.dumps(
        [
            {
                "name": event.name,
                "time_unix_nano": event.time_unix_nano,
                "attributes": _attributes(event.attributes),
            }
            for event in span.events
        ],
        ensure_ascii=False,
        default=str,
    )


def resource_spans_to_rows(
    resource_spans: ResourceSpans,
    *,
    tenant_slug: str,
    ingested_at: datetime | None = None,
) -> Iterator[list[Any]]:
    """Yield one positional row per span, ordered to match ``SPAN_COLUMNS``.

    ``tenant_slug`` comes from the authenticated request, never from the payload
    -- a client must not be able to write into another tenant by setting a
    resource attribute.
    """
    ingested_at = ingested_at or datetime.now(timezone.utc)
    resource_attrs = _attributes(resource_spans.resource.attributes)

    service_name = _text(resource_attrs.get("service.name")) or "unknown_service"
    service_version = _text(resource_attrs.get("service.version"))
    environment = _text(resource_attrs.get("deployment.environment.name")) or _text(
        resource_attrs.get("deployment.environment")
    )
    project = _text(resource_attrs.get("agentlens.project")) or service_name

    resource_map = {key: _as_map_value(value) for key, value in resource_attrs.items()}

    for scope_spans in resource_spans.scope_spans:
        for span in scope_spans.spans:
            attrs = _attributes(span.attributes)
            leftover = {
                key: _as_map_value(value)
                for key, value in attrs.items()
                if key not in _PROMOTED
            }

            start = _ns_to_datetime(span.start_time_unix_nano)
            end = _ns_to_datetime(span.end_time_unix_nano)
            # Computed from the raw nanos, not from the truncated datetimes.
            duration_ns = max(0, span.end_time_unix_nano - span.start_time_unix_nano)

            yield [
                tenant_slug,
                project,
                service_name,
                service_version,
                environment,
                span.trace_id.hex(),
                span.span_id.hex(),
                span.parent_span_id.hex(),
                span.name,
                _KIND_NAMES.get(span.kind, "unspecified"),
                start,
                end,
                duration_ns,
                _STATUS_NAMES.get(span.status.code, "unset"),
                span.status.message or "",
                _text(attrs.get("gen_ai.operation.name")),
                _text(attrs.get("gen_ai.provider.name")) or _text(attrs.get("gen_ai.system")),
                _text(attrs.get("gen_ai.request.model")),
                _text(attrs.get("gen_ai.response.model")),
                _opt_float(attrs.get("gen_ai.request.temperature")),
                _opt_int(attrs.get("gen_ai.usage.input_tokens")),
                _opt_int(attrs.get("gen_ai.usage.output_tokens")),
                _text(attrs.get("gen_ai.tool.name")),
                _text(attrs.get("gen_ai.agent.name")),
                _opt_int(attrs.get("agentlens.cost.micro_usd")),
                _text(attrs.get("agentlens.cost.status")),
                _text(attrs.get("agentlens.cost.pricing_version")),
                _text(attrs.get("agentlens.prompt.template")),
                _text(attrs.get("agentlens.prompt.template_version")),
                _opt_int(attrs.get("agentlens.retrieval.top_k")),
                _opt_int(attrs.get("agentlens.retrieval.document_count")),
                1 if attrs.get("agentlens.redaction.applied") else 0,
                _string_list(attrs.get("agentlens.redaction.rules")),
                _text(attrs.get("agentlens.sdk.version")),
                _text(attrs.get("agentlens.sdk.language")),
                _text(attrs.get("error.type")),
                leftover,
                resource_map,
                _events_json(span),
                ingested_at,
            ]
