# Architecture

Decisions, with the reasoning that produced them. Sections marked **(planned)**
describe work that does not exist in the repository yet; they are recorded here
because the Phase 1 design commits to them, not because they are built.

---

## 1. Storage boundary: Postgres vs ClickHouse

The platform has two data shapes with incompatible access patterns, so it uses
two stores. This is the single most consequential decision in the design.

### Spans go to ClickHouse

Trace data is append-only, high-volume, and read analytically: "p99 latency by
model over 30 days", "cost per tenant per day", "every trace where a tool
errored after a retrieval returned fewer than 3 documents". These are
column-scan-and-aggregate queries over a large immutable log.

Postgres is the wrong shape for this at volume:

* **Row storage reads whole rows.** Aggregating one column over 30 days of spans
  pulls every column of every row through the buffer pool. A columnar store
  reads only the column referenced.
* **Write amplification.** Span ingestion is a firehose of inserts. Postgres
  pays heap inserts plus every index, and MVCC leaves dead tuples for autovacuum
  to reclaim — a sustained cost that competes with the same instance serving
  application queries.
* **Retention is a delete problem.** Expiring old spans in Postgres is a bulk
  `DELETE` plus vacuum. ClickHouse expresses it as a `TTL` clause and drops whole
  parts, which is a metadata operation.
* **Compression.** Span attributes repeat heavily (same model, same tenant, same
  operation across millions of rows). Columnar encoding plus per-column codecs
  compress this far better than row storage.

Planned schema shape: `MergeTree`, `PARTITION BY toDate(start_time)`, `ORDER BY
(tenant_id, trace_id, start_time)`. The sort key puts tenant first so every query
is tenant-pruned, then `trace_id` so a single-trace tree fetch is one contiguous
range read. Materialised views maintain per-tenant daily cost/latency/error
rollups at write time, so dashboards never scan raw spans.

### Relational state goes to Postgres

Tenants, users, API keys, datasets, eval cases, runs, verdicts, judge
calibrations, and the audit log are comparatively small, mutable, and demand
guarantees ClickHouse deliberately does not provide:

* **Transactions across tables.** Promoting a trace into an eval case must write
  the case, a dataset-version row, and an audit entry atomically. ClickHouse has
  no multi-table transactions.
* **Referential integrity.** A verdict must reference a real eval case and a real
  run. Foreign keys enforce that; ClickHouse has none.
* **Real updates and deletes.** Revoking an API key, rotating a refresh token,
  correcting a human label. ClickHouse mutations are asynchronous background
  rewrites, not row updates.
* **Row-Level Security.** Tenant isolation is a database guarantee in Postgres
  (Phase 7 requirement). ClickHouse has no equivalent.
* **Uniqueness.** `MergeTree` does not enforce unique constraints; its
  `ReplacingMergeTree` deduplicates eventually. Eval-case identity cannot be
  eventually consistent.

### The join between them

The two stores are joined by identifier, never by a distributed query. Postgres
holds `trace_id` as an opaque reference on an eval case; fetching the underlying
trace is a second lookup against ClickHouse. Money crosses the boundary as
integer minor units in both directions so no rounding is introduced by the trip.

**Consequence, stated plainly:** there is no cross-store transaction. Promoting a
trace to an eval case commits in Postgres and *references* a span that could
later be expired by ClickHouse TTL. The eval case therefore snapshots the inputs
it needs at promotion time rather than relying on the span still existing. That
is a deliberate trade: eval cases must outlive trace retention.

---

## 2. Phase 1: the tracing SDK

### Compose OpenTelemetry, do not reimplement it

A non-goal is reimplementing OTel. The SDK therefore uses:

* `TracerProvider`, `Resource`, samplers, and span creation from `opentelemetry-sdk`.
* `BatchSpanProcessor` for batching and the exporter thread.
* `OTLPSpanExporter` for transport.
* `opentelemetry-semantic-conventions` for every attribute name.

What is added is only what OTel does not provide:

| Added | Why it is not available upstream |
|---|---|
| `BoundedQueueSpanProcessor` | `BatchSpanProcessor` bounds its queue but exposes no count of what it discarded. A platform reporting quality numbers must know how much input it dropped. |
| `RedactingSpanExporter` | No upstream redaction hook runs before serialisation. |
| `CountingSpanExporter` | Converts exporter faults into counters so the background thread survives and the app is unaffected. |
| Pricing / cost | Outside OTel's scope entirely. |

`BoundedQueueSpanProcessor` deliberately *delegates* to `BatchSpanProcessor`
rather than owning a queue and thread. It keeps an in-flight counter, admits or
drops before delegating, and releases budget when a batch leaves via
`CountingSpanExporter`. The inner queue is sized above ours so ours is always the
binding constraint. No batching logic is duplicated.

### Redaction runs twice, deliberately

* **Capture time** (`api.py`, `_set_content`) is the primary layer. Content is
  redacted *before* it becomes a span attribute, so raw PII never exists on the
  span object — not in a console exporter, not in a heap dump, not visible to a
  third-party span processor.
* **Export time** (`RedactingSpanExporter`) is defence in depth for spans
  produced by instrumentation that never went through this API.

Spans redacted at capture carry `agentlens.redaction.applied`, and the export
layer skips them. This is not just an optimisation: re-scanning a 512-span batch
cost tens of thousands of regex matches on the exporter thread, and because that
holds the GIL it appeared as p99 latency on the *caller's* thread (15.04 ms →
0.97 ms once fixed).

The exporter fails **closed**: if the redactor raises, the batch is dropped
unexported rather than sent raw.

### Teardown is bounded

Neither `force_flush()` nor `TracerProvider.shutdown()` bounds how long the
backlog takes to drain. Against an unreachable collector each pending batch costs
a full export timeout, so process exit scaled with undelivered spans (~125 s for
1000 spans). A drain deadline is armed *before* the flush and honoured by
`CountingSpanExporter`: past the deadline batches fail immediately without
touching the network, and the loss is counted as `spans_dropped_shutdown`. First
call wins, so the drain phase cannot extend the budget the flush phase was meant
to bound.

### Money is never a float

Rates parse as `Decimal` (a `float` rate is rejected with `TypeError`). Because
table rates are quoted per 1,000,000 tokens and the output unit is millionths of
a currency unit, `tokens * rate` is already the micro-unit cost — no division, no
rounding drift. Cost is emitted as an `int`.

Absence is distinguished from zero: `no_pricing` (model not in the table) and
`no_usage` (no token counts) are separate statuses, so an unpriced model never
looks free.

### Conventions are bound, not copied

`semconv.py` re-exports names from the semantic-conventions package rather than
hard-coding strings, so an upstream rename cannot leave the SDK emitting a stale
name. A test additionally pins the *resolved literals*, so a rename surfaces as a
test failure with migration consequences for stored traces rather than silently
changing what is written.

Custom attributes are confined to the `agentlens.` namespace, each carries a
written justification in an immutable registry, and a test asserts none shadows
an existing convention. Retrieval was the real temptation here — `top_k` and
`document_count` are custom, but `gen_ai.retrieval.documents` and
`gen_ai.retrieval.query.text` already exist and are used.

---

## 3. Ingestion and query (planned)

OTLP/gRPC receiver authenticated by scoped API key, writing to ClickHouse
through an async batch writer, idempotent on `(trace_id, span_id)` so a retried
export cannot duplicate. A bounded ingestion queue returns 429 with `Retry-After`
when saturated — the server-side mirror of the SDK's bounded queue, for the same
reason: bounded memory beats unbounded latency.

## 4. Evaluators (planned)

Four tiers behind one interface returning a typed `EvalResult`. The statistical
tier pins the embedding model and records its version on every result: a silent
embedding upgrade invalidates historical comparisons, which is the same failure
mode `pricing_version` guards against for cost.

Custom code evaluators execute untrusted user Python. The threat model is
hostile code, so containment is layered (namespaces, seccomp-bpf allowlist,
cgroups v2 limits, no network, read-only rootfs, wall-clock *and* CPU-time caps)
and asserted by an adversarial suite rather than assumed.

## 5. Judge calibration and regression gating (planned)

The load-bearing idea: an LLM judge is an instrument, and an uncalibrated
instrument may not gate a deployment. Agreement with human labels is measured on
a held-out gold set (Cohen's κ, quadratic-weighted κ for graded scores) with a
bootstrap confidence interval, and a judge below the κ floor may report scores
but is refused gate authority in code.

Regression decisions use a paired bootstrap over cases, nested to also resample
repetitions because the judge is stochastic — resampling cases alone understates
variance and ships noise. Power analysis reports the minimum detectable effect,
so a dataset too small to detect the claimed improvement says so.
