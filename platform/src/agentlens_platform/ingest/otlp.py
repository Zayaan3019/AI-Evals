"""OTLP/gRPC trace ingestion.

Implements the collector's ``TraceService`` from the generated protobuf stubs.
We do not define the wire format -- OpenTelemetry does -- so any OTLP-speaking
producer can point at this endpoint, not just the agentlens SDK.

Two behaviours here are specified by OTLP rather than chosen by us:

* **Partial success is a success.** ``ExportTraceServiceResponse`` carries a
  ``partial_success`` field precisely so a server can accept most of a batch and
  report what it rejected. Returning an error status for a partly-good batch
  would make well-behaved clients retry data we already stored.
* **Retryable vs terminal errors matter.** ``UNAUTHENTICATED`` tells the client
  to stop and fix its credentials; ``UNAVAILABLE`` tells it to back off and
  retry. Getting this backwards either loses data or produces an infinite retry
  storm against a permanently broken request.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import grpc
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTracePartialSuccess,
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.collector.trace.v1.trace_service_pb2_grpc import (
    TraceServiceServicer,
    add_TraceServiceServicer_to_server,
)

from ..auth import authenticate
from ..db.postgres import session_scope
from .mapper import resource_spans_to_rows
from .writer import SpanWriter

logger = logging.getLogger(__name__)

#: Metadata key carrying the API key. Lowercase because gRPC normalises header
#: names, and a mixed-case lookup silently misses.
API_KEY_METADATA = "x-agentlens-api-key"


class TraceService(TraceServiceServicer):
    """Accepts OTLP spans, authenticates the tenant, buffers rows for write."""

    def __init__(self, writer: SpanWriter) -> None:
        self._writer = writer

    async def Export(  # noqa: N802 - name fixed by the generated stub
        self,
        request: ExportTraceServiceRequest,
        context: grpc.aio.ServicerContext,
    ) -> ExportTraceServiceResponse:
        metadata = dict(context.invocation_metadata() or ())
        api_key = metadata.get(API_KEY_METADATA)

        try:
            async with session_scope() as session:
                tenant = await authenticate(session, api_key)
        except Exception:  # noqa: BLE001
            # Postgres being down is our fault, not the client's. UNAVAILABLE
            # is retryable, so the client backs off and keeps its spans instead
            # of discarding them as permanently rejected.
            logger.error("tenant authentication failed", exc_info=True)
            await context.abort(grpc.StatusCode.UNAVAILABLE, "authentication backend unavailable")
            raise AssertionError("unreachable")  # pragma: no cover - abort raises

        if tenant is None:
            await context.abort(grpc.StatusCode.UNAUTHENTICATED, "invalid or missing API key")
            raise AssertionError("unreachable")  # pragma: no cover - abort raises

        ingested_at = datetime.now(timezone.utc)
        rows: list[list[object]] = []
        malformed = 0
        for resource_spans in request.resource_spans:
            try:
                rows.extend(
                    resource_spans_to_rows(
                        resource_spans,
                        tenant_slug=tenant.tenant_slug,
                        ingested_at=ingested_at,
                    )
                )
            except Exception:  # noqa: BLE001
                # One malformed ResourceSpans must not cost us the rest of the
                # batch. Counted and reported through partial_success.
                malformed += sum(len(ss.spans) for ss in resource_spans.scope_spans)
                logger.warning("failed to map resource_spans", exc_info=True)

        accepted = await self._writer.submit(rows)
        rejected = malformed + (len(rows) - accepted)

        response = ExportTraceServiceResponse()
        if rejected:
            response.partial_success.CopyFrom(
                ExportTracePartialSuccess(
                    rejected_spans=rejected,
                    error_message=(
                        f"{rejected} spans rejected "
                        f"({malformed} malformed, {len(rows) - accepted} buffer full)"
                    ),
                )
            )
        return response


async def serve_otlp(
    writer: SpanWriter,
    *,
    host: str,
    port: int,
    max_message_bytes: int = 16 * 1024 * 1024,
) -> grpc.aio.Server:
    """Start the OTLP gRPC server and return it (already started).

    The message ceiling is real backpressure: an unbounded limit lets one client
    hand us a payload large enough to exhaust memory before any of our own
    buffer accounting gets a chance to run.
    """
    server = grpc.aio.server(
        options=[
            ("grpc.max_receive_message_length", max_message_bytes),
            ("grpc.max_send_message_length", max_message_bytes),
        ]
    )
    add_TraceServiceServicer_to_server(TraceService(writer), server)
    bound_port = server.add_insecure_port(f"{host}:{port}")
    if bound_port == 0:
        raise RuntimeError(f"could not bind OTLP gRPC listener on {host}:{port}")
    await server.start()
    logger.info("OTLP gRPC listening on %s:%d", host, bound_port)
    return server
