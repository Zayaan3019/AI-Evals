"""FastAPI application: read APIs over spans, plus lifecycle for ingestion.

The OTLP gRPC listener and the HTTP API run in one process here. They are
separable -- the writer is the only shared state and it is passed explicitly --
but co-locating them means the local stack is one command, and the gRPC server
is asyncio-native so it shares the event loop rather than fighting it.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, AsyncIterator

import grpc
from fastapi import FastAPI, HTTPException, Query

from ..config import Settings, get_settings
from ..db.clickhouse import build_client, ensure_reader, ensure_schema
from ..db.postgres import dispose_engine
from ..ingest.otlp import serve_otlp
from ..ingest.writer import SpanWriter
from ..drift import detect_drift, model_pareto
from .deps import ReaderDep, TenantDep, current_reader, current_tenant  # noqa: F401
from .evals import router as evals_router
from ..queries import (
    TimeWindow,
    count_spans,
    fetch_trace,
    list_traces,
    model_usage_summary,
    operation_latency_summary,
)

logger = logging.getLogger(__name__)


class ServiceState:
    """Process-wide handles, created on startup and closed on shutdown."""

    def __init__(self) -> None:
        self.settings: Settings = get_settings()
        #: Read path, used by the API handlers on the event loop. Connects as
        #: the tenant-restricted reader, so a handler cannot read another
        #: tenant's spans even through a query that forgets its predicate.
        self.clickhouse: Any = None
        #: Admin connection: schema, reader/policy setup and health checks.
        self.clickhouse_admin: Any = None
        #: Write path, used only by the ingest writer from a worker thread.
        #: Deliberately a second client: one client is one HTTP connection pool,
        #: and sharing it would put bulk inserts and interactive reads in
        #: contention for the same connections.
        self.clickhouse_ingest: Any = None
        self.writer: SpanWriter | None = None
        self.grpc_server: grpc.aio.Server | None = None


state = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    state.settings = settings

    state.clickhouse_admin = build_client(settings)
    ensure_schema(state.clickhouse_admin, settings.clickhouse_database, settings.span_retention_days)
    ensure_reader(
        state.clickhouse_admin,
        settings.clickhouse_database,
        reader_user=settings.clickhouse_reader_user,
        reader_password=settings.clickhouse_reader_secret,
        admin_user=settings.clickhouse_user,
    )
    state.clickhouse = build_client(settings, reader=True)

    state.clickhouse_ingest = build_client(settings)
    state.writer = SpanWriter(
        state.clickhouse_ingest,
        database=settings.clickhouse_database,
        batch_size=settings.ingest_batch_size,
        flush_interval_seconds=settings.ingest_flush_interval_seconds,
        max_buffered_spans=settings.ingest_max_buffered_spans,
    )
    await state.writer.start()

    state.grpc_server = await serve_otlp(
        state.writer, host=settings.otlp_grpc_host, port=settings.otlp_grpc_port
    )

    try:
        yield
    finally:
        # Order matters: stop accepting first, then drain. Draining while the
        # listener is still open would race new spans against the final flush.
        if state.grpc_server is not None:
            await state.grpc_server.stop(grace=5.0)
        if state.writer is not None:
            await state.writer.aclose()
        if state.clickhouse_ingest is not None:
            state.clickhouse_ingest.close()
        if state.clickhouse is not None:
            state.clickhouse.close()
        if state.clickhouse_admin is not None:
            state.clickhouse_admin.close()
        await dispose_engine()


app = FastAPI(
    title="agentlens platform",
    version="0.1.0",
    description="Ingestion and query APIs for LLM agent traces.",
    lifespan=lifespan,
)


# Auth lives in .deps: TenantDep needs a valid key, ReaderDep also admits the
# public read-only tenant. Every GET below takes ReaderDep; the one
# state-changing handler (POST /v1/flush) keeps TenantDep.
app.include_router(evals_router)


def _window(hours: float) -> TimeWindow:
    now = datetime.now(timezone.utc)
    return TimeWindow(start=now - timedelta(hours=hours), end=now)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health", tags=["ops"])
async def health() -> dict[str, Any]:
    """Liveness plus real dependency checks.

    Each backend is probed rather than assumed. A health endpoint that returns
    200 while its datastore is unreachable is worse than none: it actively
    suppresses the alert that would have caught the outage.
    """
    checks: dict[str, Any] = {}

    try:
        state.clickhouse.query("SELECT 1")
        checks["clickhouse"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["clickhouse"] = f"error: {type(exc).__name__}"

    try:
        from sqlalchemy import text

        from ..db.postgres import session_scope

        async with session_scope() as session:
            await session.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["postgres"] = f"error: {type(exc).__name__}"

    healthy = all(value == "ok" for value in checks.values())
    return {
        "status": "ok" if healthy else "degraded",
        "checks": checks,
        "ingest": state.writer.stats.snapshot() if state.writer else None,
        "buffered_spans": state.writer.buffered if state.writer else 0,
    }


# ---------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------


@app.get("/v1/traces", tags=["traces"])
async def get_traces(
    tenant: ReaderDep,
    hours: Annotated[float, Query(gt=0, le=24 * 90)] = 24,
    service: str | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    rows = list_traces(
        state.clickhouse,
        database=state.settings.clickhouse_database,
        tenant_slug=tenant.tenant_slug,
        window=_window(hours),
        service_name=service,
        limit=limit,
        offset=offset,
    )
    return {"traces": rows, "count": len(rows)}


@app.get("/v1/traces/{trace_id}", tags=["traces"])
async def get_trace(tenant: ReaderDep, trace_id: str) -> dict[str, Any]:
    spans = fetch_trace(
        state.clickhouse,
        database=state.settings.clickhouse_database,
        tenant_slug=tenant.tenant_slug,
        trace_id=trace_id,
    )
    if not spans:
        raise HTTPException(status_code=404, detail=f"no trace {trace_id!r} for this tenant")
    return {"trace_id": trace_id, "spans": spans, "span_count": len(spans)}


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


@app.get("/v1/analytics/models", tags=["analytics"])
async def get_model_usage(
    tenant: ReaderDep,
    hours: Annotated[float, Query(gt=0, le=24 * 90)] = 24,
) -> dict[str, Any]:
    return {
        "models": model_usage_summary(
            state.clickhouse,
            database=state.settings.clickhouse_database,
            tenant_slug=tenant.tenant_slug,
            window=_window(hours),
        )
    }


@app.get("/v1/analytics/operations", tags=["analytics"])
async def get_operation_latency(
    tenant: ReaderDep,
    hours: Annotated[float, Query(gt=0, le=24 * 90)] = 24,
) -> dict[str, Any]:
    return {
        "operations": operation_latency_summary(
            state.clickhouse,
            database=state.settings.clickhouse_database,
            tenant_slug=tenant.tenant_slug,
            window=_window(hours),
        )
    }


@app.get("/v1/analytics/drift", tags=["analytics"])
async def get_drift(
    tenant: ReaderDep,
    metric: Annotated[str, Query()] = "latency_ms",
    current_hours: Annotated[float, Query(gt=0, le=24 * 30)] = 1,
    reference_hours: Annotated[float, Query(gt=0, le=24 * 90)] = 24,
    operation: str = "chat",
    model: str | None = None,
) -> dict[str, Any]:
    """Compare a metric's distribution now against a reference period.

    The reference window ends where the current window begins, so the two never
    overlap -- an overlapping reference contains the very data being tested and
    biases every statistic toward "no change".
    """
    now = datetime.now(timezone.utc)
    current = TimeWindow(start=now - timedelta(hours=current_hours), end=now)
    reference = TimeWindow(
        start=current.start - timedelta(hours=reference_hours), end=current.start
    )

    try:
        result = detect_drift(
            state.clickhouse,
            database=state.settings.clickhouse_database,
            tenant_slug=tenant.tenant_slug,
            reference_window=reference,
            current_window=current,
            metric=metric,
            operation=operation,
            model=model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if result is None:
        # Not "no drift": there was nothing to compare.
        return {"status": "no_data", "metric": metric, "drift": None}
    return {
        "status": "ok",
        "metric": metric,
        "drift": {
            "psi": result.psi,
            "ks_statistic": result.ks_statistic,
            "severity": result.severity,
            "underpowered": result.is_underpowered,
            "reference_n": result.reference_n,
            "current_n": result.current_n,
            "reference_mean": result.reference_mean,
            "current_mean": result.current_mean,
            "reference_p95": result.reference_p95,
            "current_p95": result.current_p95,
        },
    }


@app.get("/v1/analytics/pareto", tags=["analytics"])
async def get_pareto(
    tenant: ReaderDep,
    hours: Annotated[float, Query(gt=0, le=24 * 90)] = 24,
) -> dict[str, Any]:
    """Cost / quality / latency frontier across models.

    ``quality`` is the non-error rate, a weak proxy named as one -- real quality
    needs an evaluator and lives in the offline path. It is included because
    cost and latency alone would rank the cheapest broken model first.
    """
    points, frontier = model_pareto(
        state.clickhouse,
        database=state.settings.clickhouse_database,
        tenant_slug=tenant.tenant_slug,
        window=_window(hours),
    )
    on_frontier = {p.label for p in frontier}

    def encode(point) -> dict[str, Any]:
        return {
            "model": point.label,
            "quality_success_rate": point.quality,
            "cost_micro_usd_per_call": point.cost_micro_usd_per_call,
            "latency_p95_ms": point.latency_p95_ms,
            "call_count": point.call_count,
            "on_frontier": point.label in on_frontier,
        }

    return {
        "quality_metric": "success_rate (non-error fraction) -- a proxy, not an eval score",
        "models": [encode(p) for p in points],
        "frontier": [encode(p) for p in frontier],
    }


@app.get("/v1/stats", tags=["ops"])
async def get_stats(tenant: ReaderDep) -> dict[str, Any]:
    """Ingestion counters plus this tenant's stored span count."""
    return {
        "tenant": tenant.tenant_slug,
        "stored_spans": count_spans(
            state.clickhouse,
            database=state.settings.clickhouse_database,
            tenant_slug=tenant.tenant_slug,
        ),
        "ingest": state.writer.stats.snapshot() if state.writer else None,
        "buffered_spans": state.writer.buffered if state.writer else 0,
    }


@app.post("/v1/flush", tags=["ops"], status_code=200)
async def force_flush(tenant: TenantDep) -> dict[str, Any]:
    """Drain the ingest buffer synchronously.

    Exists because ingestion is asynchronous by design: a caller that has just
    exported spans and wants to query them needs a way to wait for durability
    without polling. Used by the end-to-end tests and the dogfooding runners.
    """
    if state.writer is None:
        raise HTTPException(status_code=503, detail="writer not running")
    written = await state.writer.flush()
    return {"flushed": written, "buffered": state.writer.buffered}
