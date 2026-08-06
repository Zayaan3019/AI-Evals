"""OTLP protobuf -> ClickHouse row mapping.

These build genuine protobuf messages rather than dicts pretending to be them,
because the mapper's job is precisely to handle the wire representation:
``AnyValue`` unions, raw id bytes, unsigned nanosecond timestamps.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, ArrayValue, KeyValue
from opentelemetry.proto.resource.v1.resource_pb2 import Resource
from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span, Status

from agentlens_platform.db.clickhouse import SPAN_COLUMNS
from agentlens_platform.ingest.mapper import resource_spans_to_rows


def _kv(key: str, value) -> KeyValue:
    if isinstance(value, bool):
        any_value = AnyValue(bool_value=value)
    elif isinstance(value, int):
        any_value = AnyValue(int_value=value)
    elif isinstance(value, float):
        any_value = AnyValue(double_value=value)
    elif isinstance(value, (list, tuple)):
        any_value = AnyValue(
            array_value=ArrayValue(values=[AnyValue(string_value=str(v)) for v in value])
        )
    else:
        any_value = AnyValue(string_value=str(value))
    return KeyValue(key=key, value=any_value)


def build_resource_spans(
    *,
    service_name: str = "svc",
    attributes: dict | None = None,
    trace_id: bytes = b"\x01" * 16,
    span_id: bytes = b"\x02" * 8,
    parent_span_id: bytes = b"",
    name: str = "chat claude-opus-5",
    start_ns: int = 1_700_000_000_000_000_000,
    end_ns: int = 1_700_000_000_500_000_000,
    status_code=Status.STATUS_CODE_OK,
    resource_attributes: dict | None = None,
) -> ResourceSpans:
    span = Span(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        name=name,
        kind=Span.SPAN_KIND_CLIENT,
        start_time_unix_nano=start_ns,
        end_time_unix_nano=end_ns,
        status=Status(code=status_code),
        attributes=[_kv(k, v) for k, v in (attributes or {}).items()],
    )
    resource_attrs = {"service.name": service_name, **(resource_attributes or {})}
    return ResourceSpans(
        resource=Resource(attributes=[_kv(k, v) for k, v in resource_attrs.items()]),
        scope_spans=[ScopeSpans(spans=[span])],
    )


def row_dict(row) -> dict:
    return dict(zip(SPAN_COLUMNS, row))


def test_row_width_matches_the_column_contract() -> None:
    """The writer inserts positionally; a width mismatch corrupts every column."""
    rows = list(resource_spans_to_rows(build_resource_spans(), tenant_slug="acme"))
    assert len(rows) == 1
    assert len(rows[0]) == len(SPAN_COLUMNS)


def test_ids_are_lowercase_hex() -> None:
    rows = list(
        resource_spans_to_rows(
            build_resource_spans(trace_id=bytes.fromhex("4bf92f3577b34da6a3ce929d0e0e4736")),
            tenant_slug="acme",
        )
    )
    assert row_dict(rows[0])["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"


def test_tenant_comes_from_the_caller_not_the_payload() -> None:
    """A client must not be able to write into another tenant's data.

    The payload here actively tries to claim a different tenant via a resource
    attribute; the authenticated slug must win.
    """
    rows = list(
        resource_spans_to_rows(
            build_resource_spans(resource_attributes={"tenant_id": "victim", "agentlens.tenant": "victim"}),
            tenant_slug="attacker",
        )
    )
    assert row_dict(rows[0])["tenant_id"] == "attacker"


def test_genai_attributes_are_promoted_to_columns() -> None:
    rows = list(
        resource_spans_to_rows(
            build_resource_spans(
                attributes={
                    "gen_ai.operation.name": "chat",
                    "gen_ai.provider.name": "anthropic",
                    "gen_ai.request.model": "claude-opus-5",
                    "gen_ai.request.temperature": 0.2,
                    "gen_ai.usage.input_tokens": 148,
                    "gen_ai.usage.output_tokens": 32,
                    "agentlens.cost.micro_usd": 1234,
                    "agentlens.cost.status": "ok",
                }
            ),
            tenant_slug="acme",
        )
    )
    row = row_dict(rows[0])
    assert row["gen_ai_operation_name"] == "chat"
    assert row["gen_ai_request_model"] == "claude-opus-5"
    assert row["gen_ai_request_temperature"] == pytest.approx(0.2)
    assert row["gen_ai_usage_input_tokens"] == 148
    assert row["cost_micro_usd"] == 1234
    # Promoted attributes must not also sit in the generic map: storing the same
    # value twice doubles the cost of the widest column in the table.
    assert "gen_ai.request.model" not in row["attributes"]


def test_gen_ai_system_is_accepted_as_a_provider_fallback() -> None:
    """The SDK emits both names during the convention transition."""
    rows = list(
        resource_spans_to_rows(
            build_resource_spans(attributes={"gen_ai.system": "anthropic"}),
            tenant_slug="acme",
        )
    )
    assert row_dict(rows[0])["gen_ai_provider_name"] == "anthropic"


def test_unmodelled_attributes_are_preserved() -> None:
    """Third-party instrumentation must not be silently discarded."""
    rows = list(
        resource_spans_to_rows(
            build_resource_spans(attributes={"http.request.method": "POST", "custom.retries": 3}),
            tenant_slug="acme",
        )
    )
    attrs = row_dict(rows[0])["attributes"]
    assert attrs["http.request.method"] == "POST"
    # Non-strings are JSON-encoded, so they round-trip out of a String map.
    assert json.loads(attrs["custom.retries"]) == 3


def test_booleans_round_trip_as_json_not_python_repr() -> None:
    """``str(True)`` is ``'True'``, which is not valid JSON."""
    rows = list(
        resource_spans_to_rows(
            build_resource_spans(attributes={"cache.hit": True}), tenant_slug="acme"
        )
    )
    assert json.loads(row_dict(rows[0])["attributes"]["cache.hit"]) is True


def test_duration_is_computed_from_raw_nanoseconds() -> None:
    """Datetime conversion loses sub-microsecond precision; duration must not."""
    rows = list(
        resource_spans_to_rows(
            build_resource_spans(start_ns=1_700_000_000_000_000_001, end_ns=1_700_000_000_000_000_124),
            tenant_slug="acme",
        )
    )
    assert row_dict(rows[0])["duration_ns"] == 123


def test_negative_token_counts_become_null_rather_than_failing_the_batch() -> None:
    """The column is unsigned; a corrupt value must not reject 5000 good spans."""
    rows = list(
        resource_spans_to_rows(
            build_resource_spans(attributes={"gen_ai.usage.input_tokens": -5}),
            tenant_slug="acme",
        )
    )
    assert row_dict(rows[0])["gen_ai_usage_input_tokens"] is None


def test_redaction_rules_become_a_string_array() -> None:
    rows = list(
        resource_spans_to_rows(
            build_resource_spans(
                attributes={
                    "agentlens.redaction.applied": True,
                    "agentlens.redaction.rules": ["email", "credit_card"],
                }
            ),
            tenant_slug="acme",
        )
    )
    row = row_dict(rows[0])
    assert row["redaction_applied"] == 1
    assert row["redaction_rules"] == ["email", "credit_card"]


def test_timestamps_are_utc_aware() -> None:
    rows = list(resource_spans_to_rows(build_resource_spans(), tenant_slug="acme"))
    start = row_dict(rows[0])["start_time"]
    assert isinstance(start, datetime)
    assert start.tzinfo is not None
    assert start.astimezone(timezone.utc) == start


def test_missing_service_name_gets_an_explicit_placeholder() -> None:
    """'' would be indistinguishable from a service literally named ''."""
    resource_spans = ResourceSpans(
        resource=Resource(attributes=[]),
        scope_spans=[ScopeSpans(spans=[Span(name="x", trace_id=b"\x01" * 16, span_id=b"\x02" * 8)])],
    )
    rows = list(resource_spans_to_rows(resource_spans, tenant_slug="acme"))
    assert row_dict(rows[0])["service_name"] == "unknown_service"


def test_events_are_serialised_as_json() -> None:
    span = Span(
        trace_id=b"\x01" * 16,
        span_id=b"\x02" * 8,
        name="failing",
        events=[
            Span.Event(
                name="exception",
                time_unix_nano=1_700_000_000_000_000_000,
                attributes=[_kv("exception.type", "RuntimeError")],
            )
        ],
    )
    resource_spans = ResourceSpans(
        resource=Resource(attributes=[_kv("service.name", "svc")]),
        scope_spans=[ScopeSpans(spans=[span])],
    )
    events = json.loads(row_dict(list(resource_spans_to_rows(resource_spans, tenant_slug="a"))[0])["events"])
    assert events[0]["name"] == "exception"
    assert events[0]["attributes"]["exception.type"] == "RuntimeError"


def test_root_span_has_empty_parent() -> None:
    rows = list(resource_spans_to_rows(build_resource_spans(parent_span_id=b""), tenant_slug="a"))
    assert row_dict(rows[0])["parent_span_id"] == ""


def test_error_status_is_mapped() -> None:
    rows = list(
        resource_spans_to_rows(
            build_resource_spans(status_code=Status.STATUS_CODE_ERROR), tenant_slug="a"
        )
    )
    assert row_dict(rows[0])["status_code"] == "error"
