# agentlens platform report

Every number here was measured on this machine by the commands shown. Nothing is
estimated, extrapolated, or copied from a datasheet. Where a measurement is
unflattering it is reported as measured.

**Environment for all figures below:**

```
python        3.12.7
platform      Windows-11-10.0.26200-SP0
processor     Intel64 Family 6 Model 170 Stepping 4, GenuineIntel
cpu_count     22
measured_at   2026-08-03T12:50:55+00:00
```

Postgres 17.2 and ClickHouse 25.3.14.14, both from `docker-compose.yml`, running
locally. **The machine was not idle**: Docker Desktop, the platform service, a
Vite dev server and a browser were all running. That matters and is discussed
under "SDK overhead" below.

Reproduce with:

```bash
docker compose up -d
python -m bench.benchmarks --spans 50000 --json artifacts/bench_results.json
```

---

## 1. SDK overhead (caller-path)

Added latency per instrumented LLM call, on the calling thread. Export happens
on a background thread and is excluded, because the budget is about what the
user's request pays.

```
sdk.overhead_p99   0.9307 ms/call   p50=0.2855  p95=0.5662  n=2000  runs=3
                                    all_p99=[0.9307, 2.4341, 2.8711]
```

**The budget is 2 ms p99, and two of the three runs missed it.**

This is the honest result and it is worth stating plainly rather than quoting
only the best run. On an idle machine the same harness measures 0.68–0.81 ms p99
consistently (see README, Phase 1). Under the load described above — Docker,
the API service, a bundler and a browser competing for the same cores — the
measurement rises to 2.4–2.9 ms.

The interpretation is that **the tail is dominated by CPU availability, not by
SDK work**: p50 stays at 0.29 ms and p95 at 0.57 ms in the very same runs. What
moves is the tail, which is what you would expect from scheduler preemption on
a contended machine, and not what you would expect from the instrumentation
itself becoming slower.

That is an explanation, not an excuse. What it means operationally is that the
2 ms p99 budget is a claim about a machine with spare capacity, and a deployment
running its agents on saturated cores should expect worse. The test suite takes
the best of three specifically so a busy CI box does not produce a red build
from environmental noise — and the tradeoff of that choice is exactly what this
paragraph exists to disclose.

## 2. Ingestion throughput

Spans per second through the buffered writer into ClickHouse. Rows are
pre-decoded, so this isolates buffering plus insert — the part that
backpressures — from protobuf parsing.

```
ingest.throughput   28,206 spans/sec   spans=50,000  seconds=1.77  batches=10
                                       dropped=0  failed=0
```

Ten batches for fifty thousand spans, zero dropped, zero failed. At 5,000 spans
per insert this is the regime ClickHouse is designed for; the same volume as
50,000 single-row inserts would create 50,000 parts and trigger the engine's
`too many parts` backpressure long before finishing.

**This excludes the network.** The load generator and the server share a host,
so a real deployment adds a hop this benchmark cannot see. It is a write-path
figure, not a capacity plan.

## 3. Query latency

The three access patterns the UI actually issues, against the ingested 50,000
spans.

| Query | p50 | p95 | p99 |
|---|---|---|---|
| `fetch_trace_by_id` | 61.3 ms | 70.8 ms | 73.6 ms |
| `list_traces` (24h rollup) | 81.3 ms | 123.5 ms | 141.6 ms |
| `model_summary` (24h aggregate) | 72.8 ms | 101.0 ms | 102.1 ms |

`fetch_trace_by_id` carries **no time predicate at all** — a user pasting a
trace id has no idea when it ran. That is the access pattern the sort order
cannot serve, and the `idx_trace_id` bloom-filter skip index is what keeps it
from scanning every partition in the retention window.

These are dominated by round-trip and query-planning overhead at this data
volume, not by scanning: 50,000 rows is small for ClickHouse. The figures are
therefore a floor, useful for confirming the query shapes are sane, and not a
measurement of scan performance at scale.

## 4. Evaluator throughput

Per-item cost of the deterministic and statistical tiers.

| Evaluator | Tier | µs/item | items/sec |
|---|---|---|---|
| `exact_match` | deterministic | 2.73 | 366,354 |
| `response_entropy` | statistical | 16.90 | 59,174 |
| `context_groundedness` | statistical | 17.91 | 55,838 |
| `token_f1` | statistical | 21.67 | 46,145 |
| `no_pii_leak` | deterministic | 47.54 | 21,037 |

`no_pii_leak` is the most expensive by an order of magnitude because it runs the
full 13-rule redaction pipeline including a Luhn-validated card matcher. That is
a deliberate trade: it is the evaluator that catches a data-protection failure,
and 47 µs is not a cost worth optimising against that.

**Model-graded evaluators are excluded from this table.** Their latency belongs
to the judge provider, not to this platform, and reporting it here would be
misattribution. The measured judge latency on this machine was 9.4–10.6 s per
call against a local `qwen2.5:3b` (README, Phase 3) — which is a property of
running a 3B model on a laptop CPU, not of the evaluator.

## 5. Regression gate statistics

```
stats.paired_bootstrap   1120.6 ms   pairs=200  resamples=10,000
```

**This is slow, and it is the weakest number in the report.** One second per
evaluator per gate invocation, in pure Python, for 2 million resample
operations. It is acceptable because the gate runs once per pull request over a
handful of evaluators, so the wall-clock cost is a few seconds inside a CI job
that already spends minutes running evals. It would not be acceptable on an
interactive path, and vectorising it is the obvious first optimisation if that
ever changes.

The count is not tunable downward without cost: 10,000 resamples is the
conventional floor for a percentile interval, giving roughly ±0.005 Monte Carlo
error on a 95% bound. Fewer resamples would make the gate's verdict depend on
its own sampling noise.

---

## 6. End-to-end verification

The following were executed against the running stack, not simulated.

### Real trace, real cost arithmetic

A trace exported by the SDK over real OTLP/gRPC and read back through the HTTP
API:

```
span_count: 5
  invoke_workflow support_ticket   op=invoke_workflow  dur=3.15ms
  invoke_agent triage              op=invoke_agent     dur=2.61ms
  retrieval policy_kb              op=retrieval        dur=0.00ms
  chat claude-opus-5               op=chat             dur=2.07ms  cost=8350  status=ok
  execute_tool create_ticket       op=execute_tool     dur=0.00ms
```

The cost is checkable by hand: 1,240 input tokens × $5/Mtok + 86 output tokens ×
$25/Mtok = 6,200 + 2,150 = **8,350 micro-USD**. That is the integer the span
carries — no floating-point drift, because cost is computed with `Decimal` rates
and stored as an integer count of micro-USD.

### Regression gate on real runs

Two runs over 60 promoted cases, the candidate degraded on 12 of them:

```
$ python -m agentlens_platform.ci --tenant ci-demo2 \
    --baseline "main@a1b2c3d" --candidate "pr-118@e4f5g6h"

agentlens regression gate: FAIL
  a statistically significant score drop
  - exact_match@1: 1.0000 -> 0.8000 (diff -0.2000, 95% CI [-0.3000, -0.1000], p=0.0002, n=60)
  - token_f1@1:    1.0000 -> 0.8000 (diff -0.2000, 95% CI [-0.3000, -0.1000], p=0.0002, n=60)
exit code: 1
```

Exit codes are distinct per verdict (0 pass, 1 fail, 2 incomparable, 3
insufficient data, 4 error rate) so a pipeline can distinguish "this got worse"
from "we could not tell". Both block.

### Tenant isolation, verified live

Two tenants ingesting simultaneously through the same service:

```
=== tenant: finguru ===
  invoke_workflow finguru_causal_analysis spans=8 tokens=567/221 svc=finguru-causal
  invoke_workflow finguru_causal_analysis spans=8 tokens=571/225 svc=finguru-causal
=== tenant: clinical-rag ===
  invoke_workflow clinical_rag_query      spans=10 tokens=5384/784 svc=clinical-rag
=== cross-tenant isolation check ===
  finguru sees 0 clinical-rag traces (correct)
```

In Postgres, isolation is enforced by row-level security rather than by
application `WHERE` clauses: `test_tenancy.py` issues queries with **no tenant
predicate at all** and requires the database to return nothing. ClickHouse has
no RLS equivalent, so span isolation there is a predicate on every query plus a
test that asserts it. That asymmetry is real and is not papered over.

---

## 7. Dogfooding: what was and was not done

| App | Status |
|---|---|
| Multi-Agent Clinical RAG | **Instrumented and traced**, real Ollama, exported to ClickHouse |
| FinGuru | **Instrumented and traced**, real Groq, exported to ClickHouse |
| AgentForge | **Not done — the application is not present on this machine** |
| Arbitron (ReAct loop) | **Not done — the application is not present on this machine** |

A search of every directory under `C:\Users\Mohamed Zayaan\Downloads` found no
AgentForge or Arbitron project. Two of the four named applications could not be
dogfooded, and nothing was substituted in their place.

### FinGuru — real captured trace

LangGraph multi-agent causal analysis, zero edits to FinGuru's source:

```
invoke_workflow finguru_causal_analysis  [60027.8ms]
|-- invoke_agent market_analyst  [57734.6ms]
|   `-- chat llama-3.3-70b-versatile  [1398.8ms]  tokens=272/56  temp=0.1  cost=no_pricing
|-- invoke_agent causal_validator  [75.4ms]
|   |-- execute_tool perform_pearson_correlation  [5.3ms]
|   `-- execute_tool perform_granger_causality  [54.3ms]
`-- invoke_agent risk_critic  [2050.3ms]
    `-- chat llama-3.3-70b-versatile  [2046.5ms]  tokens=295/165  temp=0.1  cost=no_pricing

span_count 8 | root_count 1 | input_tokens 567 | output_tokens 221
exported=8 dropped=0 errors=0
```

Two things this trace makes visible that a log would not:

* The statistical tests appear as `execute_tool` spans. In a causal-analysis
  agent those *are* the analysis — a trace showing only the LLM calls would omit
  the part that distinguishes it from a model asserting a correlation exists.
* `market_analyst` took 57.7 s of which the model call was 1.4 s. The remaining
  56 s is the agent loading its embedding model and querying its vector store
  before ever reaching the LLM. That is exactly the kind of attribution error a
  team makes when they only instrument the model call.

`cost=no_pricing` is correct: the pricing table covers Anthropic first-party
rates and Groq is not in it. The platform records honest absence rather than a
fabricated zero.

### Clinical RAG — real captured trace

```
invoke_workflow clinical_rag_query  [372651.4ms]
|-- invoke_agent query_decomposer  [71586.8ms]
|   `-- chat qwen2.5:3b  [71537.3ms]  tokens=1254/102  temp=0.1
|-- retrieval hybrid_dense_bm25  [757.3ms]  docs=8  top_k=8
|-- retrieval hybrid_dense_bm25  [747.6ms]  docs=8  top_k=8
|-- retrieval hybrid_dense_bm25  [156.6ms]  docs=8  top_k=8
|-- invoke_agent reasoner  [197053.4ms]
|   `-- chat qwen2.5:3b  [196978.4ms]  tokens=2202/487  temp=0.2
`-- invoke_agent critic  [99309.6ms]
    `-- chat qwen2.5:3b  [99277.4ms]  tokens=1928/195  temp=0.05

span_count 10 | root_count 1 | input_tokens 5384 | output_tokens 784
spans_exported 10 | spans_dropped 0 | sdk_errors 0
```

`root_count 1` is load-bearing here. The first capture of this app reported
`root_count 4`: three retrieval spans floated outside the workflow because
`HybridRetriever.search_multi_query` fans out across a `ThreadPoolExecutor` and
Python does not copy `contextvars` into worker threads.

---

## 8. Defects this work found

Each was found by a test or a real run failing, then fixed, then re-verified.
Ten in total; the SDK-phase ones are described in the README.

| # | Defect | Where found |
|---|---|---|
| 1 | p99 tail of 15.04 ms from double redaction on the exporter thread | overhead benchmark |
| 2 | Shutdown stalled ~125 s against an unreachable collector | degradation suite |
| 3 | Card-number rule destroyed 9.49% of floating-point values | real Clinical RAG trace |
| 4 | Parallel retrieval produced orphaned spans (`root_count 4`) | real Clinical RAG trace |
| 5 | One ClickHouse client cannot serve reads and writes concurrently | writer suite |
| 6 | Time window ending "now" excluded spans written now | end-to-end suite |
| 7 | Trace list reported a child span as the trace root | end-to-end suite |
| 8 | The gate's config-drift check could never fire | gate suite |
| 9 | Promotion produced empty cases from real traces | full-loop suite |
| 10 | RLS policies were correct and completely inert (superuser bypass) | tenancy suite |

Number 10 is the one worth dwelling on. The policies were syntactically correct,
`FORCE ROW LEVEL SECURITY` was set, and `pg_class.relrowsecurity` reported
`true` on every table — and none of it did anything, because the application
connected as the compose Postgres superuser and superusers ignore RLS entirely.
`FORCE` does not help: it removes only the *owner's* exemption. Verified
directly at the time: `rolsuper=True, rolbypassrls=True`.

An isolation model that reports itself as enabled while enforcing nothing is
worse than none, because it stops anyone from looking. The fix was a dedicated
non-superuser application role, and `test_the_application_role_cannot_bypass_rls`
now asserts the runtime role has neither attribute.

---

## 9. Limitations

* **Single machine, single run.** Every figure is from one laptop. Nothing here
  establishes behaviour under concurrency, across a network, or at data volumes
  where ClickHouse's scan performance actually matters.
* **Ingestion throughput excludes the network** and excludes protobuf decoding.
* **The SDK overhead budget was missed on 2 of 3 runs** on a loaded machine.
  See section 1.
* **The paired bootstrap is slow** (~1.1 s per evaluator). Section 5.
* **Pricing covers Anthropic first-party rates only.** Groq and Ollama calls
  resolve to `no_pricing`, so any cost total that includes them is a lower
  bound and is labelled as such in the UI.
* **The sandbox is not a security boundary** against a determined attacker —
  only against buggy and careless code. Stated at length in
  `evaluators/sandbox.py`.
* **Two of the four named applications were not dogfooded** because they are not
  on this machine. Section 7.
* **Judge quality is not established.** A 3B local judge was measured scoring a
  fully-supported answer 0.5 for an unrelated reason. Calibration machinery
  exists to quantify this; it has not been run against a meaningful corpus of
  human labels, because no such corpus exists here.
