"""Run FinGuru's causal agent under agentlens, exporting to the live platform.

Unlike the Clinical RAG runner (which captured into memory), this exports over
real OTLP/gRPC to the running ingestion service, so the trace lands in
ClickHouse and is visible in the UI. Set ``AGENTLENS_ENDPOINT`` to empty to fall
back to in-memory capture when the platform is not running.

FinGuru's own ``.env`` supplies the Groq credential. It is read by the
application exactly as it would be in normal operation; this script neither
prints it nor sends it anywhere but Groq.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
APP_DIR = Path(os.environ.get("FINGURU_DIR", REPO.parent / "Finguru"))

if not APP_DIR.is_dir():
    raise SystemExit(
        f"FinGuru not found at {APP_DIR}. Set FINGURU_DIR to its location."
    )

sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(APP_DIR)

# FinGuru reads configuration from its own .env, which is where the real Groq
# key lives. Loading it here means the app behaves exactly as it does normally.
try:
    from dotenv import load_dotenv

    load_dotenv(APP_DIR / ".env")
except ImportError:
    pass

import agentlens
from agentlens.config import AgentLensConfig

ENDPOINT = os.environ.get("AGENTLENS_ENDPOINT", "http://127.0.0.1:14400")
API_KEY = os.environ.get("AGENTLENS_API_KEY", "")

QUERY = os.environ.get(
    "FINGURU_QUERY",
    "Does a rise in crude oil prices cause airline stocks to fall?",
)


class TeeExporter:
    """Forwards every batch to a real exporter and keeps a local copy.

    Needed because the SDK retains spans in memory only when the memory
    exporter is selected, and this run wants both: the spans must reach
    ClickHouse *and* be renderable as a tree once the process ends. Retaining a
    reference is safe here -- a single request produces tens of spans, not a
    stream -- and would not be appropriate in a long-lived service.
    """

    def __init__(self, inner):
        self._inner = inner
        self.spans: list = []

    def export(self, spans):
        self.spans.extend(spans)
        return self._inner.export(spans)

    def shutdown(self):
        return self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000):
        return self._inner.force_flush(timeout_millis)


_tee: TeeExporter | None = None


def _build_config() -> AgentLensConfig:
    global _tee
    if ENDPOINT and API_KEY:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        _tee = TeeExporter(
            OTLPSpanExporter(
                endpoint=ENDPOINT,
                insecure=True,
                headers=(("x-agentlens-api-key", API_KEY),),
                timeout=15,
            )
        )
        exporter = _tee
        return AgentLensConfig(
            service_name="finguru-causal",
            custom_exporter=exporter,
            schedule_delay_millis=500,
            set_global_tracer_provider=False,
        )
    # No platform configured: capture in memory so the run still produces a
    # tree. Stated rather than silently degrading to "no tracing".
    print("AGENTLENS_ENDPOINT/API_KEY not set -- capturing in memory only", file=sys.stderr)
    return AgentLensConfig(
        service_name="finguru-causal",
        exporter="memory",
        set_global_tracer_provider=False,
    )


async def main() -> int:
    lens = agentlens.init(_build_config())

    from instrumentation import instrument

    instrument()

    from app.agents.causal.graph import run_causal_analysis

    print(f"query: {QUERY}")
    started = time.perf_counter()
    try:
        result = await run_causal_analysis(QUERY, user_id="agentlens-dogfood", verbose=False)
    except Exception as exc:  # noqa: BLE001
        print(f"FinGuru run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        agentlens.shutdown(timeout_millis=10_000)
        return 1
    wall = time.perf_counter() - started

    # Flush before reading: export is asynchronous by design, so the tee has
    # not necessarily seen the final batch yet.
    lens.force_flush(15_000)
    if _tee is not None:
        spans = list(_tee.spans)
    elif lens.memory_exporter is not None:
        spans = list(lens.memory_exporter.get_finished_spans())
    else:
        spans = []
    trace_id = format(spans[0].context.trace_id, "032x") if spans else None

    # Persist before rendering: a rendering error must not destroy the record of
    # a run that cost real API calls.
    artifact = REPO / "artifacts" / "finguru_trace.md"
    artifact.parent.mkdir(exist_ok=True)
    lines = [
        "# Captured trace: FinGuru causal analysis",
        "",
        f"- Query: `{QUERY}`",
        f"- Status: `{result.get('status')}` | iterations: {result.get('iteration_count')}",
        f"- Wall clock: {wall:.1f}s | spans captured: {len(spans)}",
        f"- Trace id: `{trace_id or 'n/a'}`",
        "",
        "## Final output",
        "",
        "```",
        str(result.get("final_output", ""))[:4000],
        "```",
        "",
    ]
    if spans:
        from agentlens.tree import render_forest, tree_stats

        lines += ["## Trace tree", "", "```", render_forest(spans, ascii_only=True), "```", ""]
        stats = tree_stats(spans)
        lines += ["## Summary", "", "```"]
        for key, value in stats.items():
            lines.append(f"  {key:16s} {value}")
        lines += ["```", ""]
    artifact.write_text("\n".join(lines), encoding="utf-8")

    agentlens.shutdown(timeout_millis=15_000)

    print(f"\nstatus={result.get('status')} iterations={result.get('iteration_count')} wall={wall:.1f}s")
    if spans:
        from agentlens.tree import render_forest, tree_stats

        print(render_forest(spans, ascii_only=True))
        for key, value in tree_stats(spans).items():
            print(f"  {key:16s} {value}")
    stats = lens.stats.snapshot()
    print(
        f"  exported={stats.get('spans_exported_ok')} "
        f"dropped={stats.get('spans_dropped_queue_full')} errors={stats.get('sdk_errors')}"
    )
    print(f"\nartifact: {artifact}")
    if trace_id:
        print(f"UI: http://localhost:5273/traces/{trace_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
