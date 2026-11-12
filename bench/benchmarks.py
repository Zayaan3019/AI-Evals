"""Benchmarks for the agentlens platform.

Every number this produces is measured against the real stack -- the real SDK,
the real OTLP gRPC listener, the real ClickHouse. Nothing is extrapolated,
modelled, or copied from a datasheet.

What is deliberately *not* claimed:

* These are single-machine numbers on one developer laptop under Windows. They
  say what this code does here. They are not a capacity plan, and a figure like
  "N spans/sec" would be dishonest as a product claim without describing the
  hardware, which is why every result carries its environment.
* Latency benchmarks measure the machine as much as the code. Where a tail
  matters the harness reports percentiles and the sample size rather than a
  single mean, because a mean hides exactly the behaviour worth knowing.
* Ingestion throughput here excludes network transit: the load generator and
  the server share a host. Real deployments add a network hop this cannot see.

Run with ``python -m bench.benchmarks`` from the repository root, with the
docker stack up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

sys.path.insert(0, "platform/src")
sys.path.insert(0, "sdk/src")

from agentlens_platform.config import get_settings  # noqa: E402
from agentlens_platform.db.clickhouse import build_client, ensure_schema, insert_spans  # noqa: E402
from agentlens_platform.ingest.mapper import resource_spans_to_rows  # noqa: E402
from agentlens_platform.ingest.writer import SpanWriter  # noqa: E402
from agentlens_platform.queries import (  # noqa: E402
    TimeWindow,
    count_spans,
    fetch_trace,
    list_traces,
    model_usage_summary,
)


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return ordered[index]


@dataclass
class Result:
    name: str
    unit: str
    value: float
    detail: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        extras = "  ".join(f"{k}={v}" for k, v in self.detail.items())
        return f"{self.name:42s} {self.value:12.4f} {self.unit:12s} {extras}"


def environment() -> dict[str, Any]:
    """Recorded with every run. A number without its machine is not a result."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "cpu_count": __import__("os").cpu_count(),
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------------------
# 1. SDK caller-path overhead
# ---------------------------------------------------------------------------


def bench_sdk_overhead(iterations: int = 2000) -> Result:
    """Added latency per instrumented LLM call, on the caller's thread.

    Reuses the Phase 1 harness rather than reimplementing it, so the number in
    the report and the number the test suite gates on are produced by the same
    code. Run in a subprocess for the reasons documented there.
    """
    sys.path.insert(0, "sdk/tests")
    from test_overhead import run_isolated  # type: ignore[import-not-found]

    runs = [run_isolated() for _ in range(3)]
    best = min(runs, key=lambda r: r["p99_ms"])
    return Result(
        name="sdk.overhead_p99",
        unit="ms/call",
        value=best["p99_ms"],
        detail={
            "p50": f"{best['p50_ms']:.4f}",
            "p95": f"{best['p95_ms']:.4f}",
            "iterations": best["iterations"],
            "runs": len(runs),
            "all_p99": [round(r["p99_ms"], 4) for r in runs],
        },
    )


# ---------------------------------------------------------------------------
# 2. Ingestion throughput
# ---------------------------------------------------------------------------


def _make_rows(count: int, tenant: str) -> list[list[Any]]:
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
    from opentelemetry.proto.resource.v1.resource_pb2 import Resource
    from opentelemetry.proto.trace.v1.trace_pb2 import ResourceSpans, ScopeSpans, Span

    now_ns = int(time.time() * 1e9)
    rows: list[list[Any]] = []
    for index in range(count):
        span = Span(
            trace_id=uuid.uuid4().bytes,
            span_id=uuid.uuid4().bytes[:8],
            name="chat claude-opus-5",
            kind=Span.SPAN_KIND_CLIENT,
            start_time_unix_nano=now_ns - 500_000_000,
            end_time_unix_nano=now_ns,
            attributes=[
                KeyValue(key="gen_ai.operation.name", value=AnyValue(string_value="chat")),
                KeyValue(key="gen_ai.provider.name", value=AnyValue(string_value="anthropic")),
                KeyValue(key="gen_ai.request.model", value=AnyValue(string_value="claude-opus-5")),
                KeyValue(key="gen_ai.usage.input_tokens", value=AnyValue(int_value=1200 + index % 500)),
                KeyValue(key="gen_ai.usage.output_tokens", value=AnyValue(int_value=80 + index % 60)),
                KeyValue(key="agentlens.cost.micro_usd", value=AnyValue(int_value=6000 + index)),
                KeyValue(key="agentlens.cost.status", value=AnyValue(string_value="ok")),
            ],
        )
        resource_spans = ResourceSpans(
            resource=Resource(
                attributes=[KeyValue(key="service.name", value=AnyValue(string_value="bench"))]
            ),
            scope_spans=[ScopeSpans(spans=[span])],
        )
        rows.extend(resource_spans_to_rows(resource_spans, tenant_slug=tenant))
    return rows


async def bench_ingest_throughput(client, database: str, total: int = 50_000) -> Result:
    """Spans per second through the buffered writer into ClickHouse.

    Measures the write path only -- rows are pre-built so protobuf decoding is
    excluded and the number reflects buffering plus insert, which is the part
    that backpressures.
    """
    tenant = f"bench{uuid.uuid4().hex[:10]}"
    rows = _make_rows(total, tenant)

    writer = SpanWriter(
        client,
        database=database,
        batch_size=5_000,
        flush_interval_seconds=1.0,
        max_buffered_spans=200_000,
    )
    await writer.start()
    started = time.perf_counter()
    # Submit in request-sized chunks, as a real producer would.
    for offset in range(0, len(rows), 500):
        await writer.submit(rows[offset : offset + 500])
    await writer.flush()
    elapsed = time.perf_counter() - started
    await writer.aclose()

    stats = writer.stats.snapshot()
    stored = count_spans(client, database=database, tenant_slug=tenant)
    return Result(
        name="ingest.throughput",
        unit="spans/sec",
        value=stored / elapsed if elapsed > 0 else float("nan"),
        detail={
            "spans": stored,
            "seconds": f"{elapsed:.2f}",
            "batches": stats["batches_written"],
            "dropped": stats["spans_dropped_buffer_full"],
            "failed": stats["spans_failed_write"],
        },
    )


# ---------------------------------------------------------------------------
# 3. Query latency
# ---------------------------------------------------------------------------


def _time_query(fn: Callable[[], Any], repeats: int) -> list[float]:
    timings: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - started) * 1000)
    return timings


def bench_query_latency(client, database: str, tenant: str, repeats: int = 30) -> list[Result]:
    """Latency of the three access patterns the UI actually issues."""
    results: list[Result] = []

    # A trace id that exists, so the point lookup does real work.
    traces = list_traces(
        client, database=database, tenant_slug=tenant, window=TimeWindow.last(hours=24), limit=1
    )
    trace_id = traces[0]["trace_id"] if traces else uuid.uuid4().hex

    cases = [
        (
            "query.fetch_trace_by_id",
            lambda: fetch_trace(client, database=database, tenant_slug=tenant, trace_id=trace_id),
            "point lookup, no time filter (bloom-filter skip index)",
        ),
        (
            "query.list_traces_24h",
            lambda: list_traces(
                client, database=database, tenant_slug=tenant, window=TimeWindow.last(hours=24)
            ),
            "per-trace rollup over a 24h window",
        ),
        (
            "query.model_summary_24h",
            lambda: model_usage_summary(
                client, database=database, tenant_slug=tenant, window=TimeWindow.last(hours=24)
            ),
            "cost/token/latency aggregate with t-digest quantiles",
        ),
    ]

    for name, fn, note in cases:
        # Warm up: the first query pays for connection setup and mark caching,
        # which is a one-off cost and not what a served request looks like.
        _time_query(fn, 3)
        timings = _time_query(fn, repeats)
        results.append(
            Result(
                name=name,
                unit="ms",
                value=percentile(timings, 0.50),
                detail={
                    "p95": f"{percentile(timings, 0.95):.2f}",
                    "p99": f"{percentile(timings, 0.99):.2f}",
                    "n": repeats,
                    "note": note,
                },
            )
        )
    return results


# ---------------------------------------------------------------------------
# 4. Evaluator throughput
# ---------------------------------------------------------------------------


def bench_evaluators(iterations: int = 2000) -> list[Result]:
    """Deterministic and statistical evaluators, per-item cost.

    Model-graded evaluators are excluded: their latency is the judge provider's,
    not ours, and reporting it as a platform number would be misattribution.
    """
    from agentlens_platform.evaluators import EvalInput, build_evaluator, run_evaluator

    item = EvalInput(
        output="The refund window is thirty days from the delivery date, per policy 4.2.",
        inputs={"question": "How long do I have to request a refund?"},
        expected="Refunds are available within thirty days of delivery.",
        context=("Refunds are processed within thirty days of delivery confirmation.",),
        metadata={"case_id": "bench", "tools_called": ["lookup_policy"]},
    )

    results: list[Result] = []
    for slug, kwargs in [
        ("exact_match", {}),
        ("token_f1", {}),
        ("context_groundedness", {}),
        ("no_pii_leak", {}),
        ("response_entropy", {}),
    ]:
        evaluator = build_evaluator(slug, **kwargs)
        run_evaluator(evaluator, item)  # warm up regex compilation etc.

        started = time.perf_counter()
        for _ in range(iterations):
            run_evaluator(evaluator, item)
        elapsed = time.perf_counter() - started
        results.append(
            Result(
                name=f"evaluator.{slug}",
                unit="µs/item",
                value=(elapsed / iterations) * 1e6,
                detail={"items": iterations, "per_sec": f"{iterations / elapsed:,.0f}"},
            )
        )
    return results


# ---------------------------------------------------------------------------
# 5. Statistics
# ---------------------------------------------------------------------------


def bench_bootstrap(resamples: int = 10_000) -> Result:
    """Cost of one paired bootstrap, which the gate runs per evaluator.

    Timed on real annotations rather than generated numbers: the SciEntsBank
    expert labels for 150 student answers, scored two ways by the dataset's
    own schemes (2-way correct/incorrect against 5-way with partial credit).
    """
    from agentlens_platform.examples.benchmark import scientsbank_items
    from agentlens_platform.stats import paired_bootstrap

    partial = {"correct": 1.0, "partially_correct_incomplete": 0.5}
    items = scientsbank_items()
    baseline = [1.0 if i["label_2way"] == "correct" else 0.0 for i in items]
    candidate = [partial.get(i["label_5way"], 0.0) for i in items]
    pairs = len(items)

    started = time.perf_counter()
    paired_bootstrap(baseline, candidate, resamples=resamples)
    elapsed = time.perf_counter() - started
    return Result(
        name="stats.paired_bootstrap",
        unit="ms",
        value=elapsed * 1000,
        detail={"pairs": pairs, "resamples": resamples},
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def run_all(args: argparse.Namespace) -> dict[str, Any]:
    settings = get_settings()
    database = f"agentlens_bench_{uuid.uuid4().hex[:10]}"

    admin = build_client(settings)
    admin.command(f"CREATE DATABASE IF NOT EXISTS {database}")
    client = build_client(settings, database=database)
    ensure_schema(client, database, settings.span_retention_days)

    results: list[Result] = []
    try:
        print("running: sdk overhead (3 isolated subprocess runs)…")
        results.append(bench_sdk_overhead())

        print(f"running: ingestion throughput ({args.spans:,} spans)…")
        ingest = await bench_ingest_throughput(client, database, total=args.spans)
        results.append(ingest)

        tenant = None
        rows = client.query(
            f"SELECT tenant_id, count() c FROM {database}.spans GROUP BY tenant_id ORDER BY c DESC LIMIT 1"
        ).result_rows
        tenant = rows[0][0] if rows else "none"

        print("running: query latency…")
        results.extend(bench_query_latency(client, database, tenant, repeats=args.repeats))

        print("running: evaluator throughput…")
        results.extend(bench_evaluators(iterations=args.eval_iterations))

        print("running: paired bootstrap…")
        results.append(bench_bootstrap())
    finally:
        client.close()
        admin.command(f"DROP DATABASE IF EXISTS {database}")
        admin.close()

    return {"environment": environment(), "results": [asdict(r) for r in results]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench", description="Measure the platform.")
    parser.add_argument("--spans", type=int, default=50_000, help="spans for the ingest benchmark")
    parser.add_argument("--repeats", type=int, default=30, help="query repetitions")
    parser.add_argument("--eval-iterations", type=int, default=2000)
    parser.add_argument("--json", type=str, default=None, help="write results to this path")
    args = parser.parse_args(argv)

    payload = asyncio.run(run_all(args))

    print("\n" + "=" * 96)
    print("agentlens benchmarks")
    print("=" * 96)
    for key, value in payload["environment"].items():
        print(f"  {key:14s} {value}")
    print("-" * 96)
    for entry in payload["results"]:
        print(Result(**entry).render())
    print("=" * 96)

    if args.json:
        from pathlib import Path

        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
