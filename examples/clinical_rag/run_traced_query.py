"""Run a real Clinical RAG query end-to-end and print the captured trace tree.

This is not a simulation. It drives the user's actual Multi-Agent Clinical RAG
app against a live local Ollama server and the app's real ChromaDB + BM25 index.
Every span, token count, score, and latency printed below is measured.

Usage:
    python examples/clinical_rag/run_traced_query.py [question]

Requires: Ollama running with the app's models pulled, and the app's index built.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

APP_DIR = Path(
    os.environ.get("CLINICAL_RAG_DIR", r"C:\Users\Mohamed Zayaan\Downloads\healthcare_rag")
)

DEFAULT_QUESTION = "What did Kaiser Permanente deploy for mental health assessment?"


class _TeeExporter:
    """Ships spans to the platform and keeps a local copy for rendering.

    The SDK only retains spans when the memory exporter is selected, and this
    run wants both: durable storage in ClickHouse and a tree printed at the
    end. Safe for a single request; not something a long-lived service should
    do.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--attributes",
        action="store_true",
        help="print every captured span attribute, not just the summary",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[2] / "artifacts"),
        help="directory to write the captured trace report into",
    )
    args = parser.parse_args()

    if not APP_DIR.is_dir():
        print(f"Clinical RAG app not found at {APP_DIR}", file=sys.stderr)
        print("Set CLINICAL_RAG_DIR to its location.", file=sys.stderr)
        return 2

    # The app imports its modules by bare name (config, utils, agents...), so it
    # must be importable as a top-level package root.
    sys.path.insert(0, str(APP_DIR))
    os.chdir(APP_DIR)

    import agentlens


    from agentlens.config import AgentLensConfig
    from agentlens.tree import render_forest, tree_stats

    import config

    # The cross-encoder reranker needs sentence-transformers, which cannot load
    # in this environment due to a pre-existing protobuf version conflict in the
    # host Anaconda install (the app's own requirements.txt documents this).
    # USE_RERANKER is the app's own supported toggle for skipping it. Retrieval
    # still runs dense + BM25 + RRF fusion; only cross-encoder reranking is off.
    reranker_disabled = False
    if config.USE_RERANKER:
        try:
            import sentence_transformers  # noqa: F401
        except Exception:
            config.USE_RERANKER = False
            reranker_disabled = True

    from instrumentation import instrument  # noqa: E402  (needs sys.path first)

    # Export to the platform when it is configured, and keep a local copy so
    # the tree can still be rendered afterwards. Falls back to memory-only when
    # no collector is reachable, which is stated rather than silent.
    endpoint = os.environ.get("AGENTLENS_ENDPOINT", "")
    api_key = os.environ.get("AGENTLENS_API_KEY", "")
    tee = None
    if endpoint and api_key:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        tee = _TeeExporter(
            OTLPSpanExporter(
                endpoint=endpoint,
                insecure=True,
                headers=(("x-agentlens-api-key", api_key),),
                timeout=15,
            )
        )
        lens = agentlens.init(
            AgentLensConfig(
                service_name="clinical-rag",
                project="dogfood",
                environment="local",
                custom_exporter=tee,
                schedule_delay_millis=500,
                set_global_tracer_provider=False,
            )
        )
    else:
        print("AGENTLENS_ENDPOINT/API_KEY unset -- capturing in memory only")
        lens = agentlens.init(
            AgentLensConfig(
                service_name="clinical-rag",
                project="dogfood",
                environment="local",
                exporter="memory",
                set_global_tracer_provider=False,
            )
        )
    instrument()

    from pipeline import HealthcareRAGPipeline

    print("=" * 78)
    print("Multi-Agent Clinical RAG -- traced with agentlens")
    print("=" * 78)
    print(f"Ollama       : {config.OLLAMA_BASE_URL}")
    print(f"Reason model : {config.MAIN_LLM}")
    print(f"Fast model   : {config.FAST_LLM}")
    print(f"Embed model  : {config.EMBED_MODEL}")
    print(f"Reranker     : {'ON' if config.USE_RERANKER else 'OFF'}"
          + (" (disabled: sentence-transformers unavailable in this env)"
             if reranker_disabled else ""))
    print(f"Question     : {args.question}")
    print()

    print("Loading index and running query (real LLM inference, please wait)...")
    started = time.perf_counter()
    pipeline = HealthcareRAGPipeline()
    result = pipeline.query(args.question)
    elapsed = time.perf_counter() - started

    lens.force_flush(15_000)
    spans = list(tee.spans) if tee is not None else list(
        lens.memory_exporter.get_finished_spans()
    )
    stats = tree_stats(spans)
    export = lens.stats.snapshot()

    # Persist first. A run costs minutes of real inference; a rendering or
    # console-encoding problem must not destroy the captured result.
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tree_text = render_forest(spans, show_attributes=True, ascii_only=True)
    report = out_dir / "clinical_rag_trace.md"
    report.write_text(
        "# Captured trace: Multi-Agent Clinical RAG\n\n"
        f"- Question: `{args.question}`\n"
        f"- Reasoning model: `{config.MAIN_LLM}` | Fast: `{config.FAST_LLM}` "
        f"| Embed: `{config.EMBED_MODEL}`\n"
        f"- Reranker: {'ON' if config.USE_RERANKER else 'OFF'}"
        + (" (sentence-transformers unavailable in this environment)\n"
           if reranker_disabled else "\n")
        + f"- Wall clock: {elapsed:.1f}s | Spans captured: {len(spans)}\n"
        f"- Spans exported: {export['spans_exported_ok']} | "
        f"dropped: {export['spans_dropped_queue_full']} | "
        f"sdk_errors: {export['sdk_errors']}\n\n"
        "## Answer\n\n```\n" + result.final_answer.strip() + "\n```\n\n"
        "## Trace tree\n\n```\n" + tree_text + "\n```\n\n"
        "## Summary\n\n```\n"
        + "\n".join(f"{k:16s} {v}" for k, v in stats.items())
        + "\n```\n",
        encoding="utf-8",
    )

    print()
    print("-" * 78)
    print("ANSWER")
    print("-" * 78)
    print(result.final_answer.strip()[:1200])
    print()

    print("-" * 78)
    print(f"CAPTURED TRACE TREE  ({len(spans)} spans, wall {elapsed:.1f}s)")
    print("-" * 78)
    print(render_forest(spans, show_attributes=args.attributes, ascii_only=True))
    print()

    print("-" * 78)
    print("TRACE SUMMARY")
    print("-" * 78)
    for key, value in stats.items():
        print(f"  {key:16s} {value}")
    print(f"  spans_exported   {export['spans_exported_ok']}")
    print(f"  spans_dropped    {export['spans_dropped_queue_full']}")
    print(f"  sdk_errors       {export['sdk_errors']}")
    print()
    print(f"Full trace with all attributes written to: {report}")

    agentlens.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
