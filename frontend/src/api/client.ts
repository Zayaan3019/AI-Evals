/**
 * Typed client for the agentlens API.
 *
 * The API key is deliberately absent from this file and from the bundle. The
 * Vite dev server attaches it server-side (see vite.config.ts); in a deployment
 * the same job belongs to whatever sits in front of the app. A credential
 * shipped in client JavaScript is readable by anyone who opens devtools, and
 * "it's only a read key" stops being true the first time someone adds a write
 * endpoint.
 */

const BASE = "/api";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function get<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const url = new URL(`${BASE}${path}`, window.location.origin);
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value !== undefined && value !== "") url.searchParams.set(key, String(value));
  }

  const response = await fetch(url, { headers: { accept: "application/json" } });
  if (!response.ok) {
    // Surface the server's reason rather than a generic failure. A 401 and a
    // 404 call for completely different user actions, and collapsing them into
    // "something went wrong" hides which one happened.
    let detail: string | undefined;
    try {
      detail = (await response.json())?.detail;
    } catch {
      detail = undefined;
    }
    throw new ApiError(
      response.status === 401
        ? "Not authenticated: the API key is missing or invalid."
        : `Request failed (${response.status})`,
      response.status,
      detail,
    );
  }
  return (await response.json()) as T;
}

// -- shapes returned by the API ------------------------------------------
// Nullable fields are typed as nullable on purpose. The platform records an
// absent cost as null rather than zero, and flattening that in the UI would
// reintroduce exactly the fabrication the backend avoids.

export interface TraceSummary {
  trace_id: string;
  started_at: string;
  ended_at: string;
  span_count: number;
  root_name: string;
  service_name: string;
  input_tokens: number | null;
  output_tokens: number | null;
  cost_micro_usd: number | null;
  error_count: number;
  root_duration_ns: number;
}

export interface Span {
  trace_id: string;
  span_id: string;
  parent_span_id: string;
  name: string;
  kind: string;
  start_time: string;
  end_time: string;
  duration_ns: number;
  status_code: string;
  status_message: string;
  service_name: string;
  gen_ai_operation_name: string;
  gen_ai_provider_name: string;
  gen_ai_request_model: string;
  gen_ai_request_temperature: number | null;
  gen_ai_usage_input_tokens: number | null;
  gen_ai_usage_output_tokens: number | null;
  gen_ai_tool_name: string;
  gen_ai_agent_name: string;
  cost_micro_usd: number | null;
  cost_status: string;
  retrieval_top_k: number | null;
  retrieval_document_count: number | null;
  redaction_applied: number;
  redaction_rules: string[];
  error_type: string;
  attributes: Record<string, string>;
}

export interface ModelUsage {
  model: string;
  provider: string;
  call_count: number;
  input_tokens: number | null;
  output_tokens: number | null;
  cost_micro_usd: number | null;
  unpriced_calls: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
  error_count: number;
}

export interface OperationLatency {
  operation: string;
  span_count: number;
  p50_ms: number;
  p95_ms: number;
  p99_ms: number;
  error_count: number;
}

export interface ParetoPoint {
  model: string;
  quality_success_rate: number;
  cost_micro_usd_per_call: number;
  latency_p95_ms: number;
  call_count: number;
  on_frontier: boolean;
}

export interface DriftReport {
  status: "ok" | "no_data";
  metric: string;
  drift: {
    psi: number;
    ks_statistic: number;
    severity: "stable" | "moderate" | "significant";
    underpowered: boolean;
    reference_n: number;
    current_n: number;
    reference_mean: number;
    current_mean: number;
    reference_p95: number;
    current_p95: number;
  } | null;
}

export interface Health {
  status: string;
  checks: Record<string, string>;
  ingest: Record<string, number | string | null> | null;
  buffered_spans: number;
}

export const api = {
  health: () => get<Health>("/health"),
  traces: (hours: number, limit = 50) =>
    get<{ traces: TraceSummary[]; count: number }>("/v1/traces", { hours, limit }),
  trace: (traceId: string) =>
    get<{ trace_id: string; spans: Span[]; span_count: number }>(`/v1/traces/${traceId}`),
  models: (hours: number) => get<{ models: ModelUsage[] }>("/v1/analytics/models", { hours }),
  operations: (hours: number) =>
    get<{ operations: OperationLatency[] }>("/v1/analytics/operations", { hours }),
  pareto: (hours: number) =>
    get<{ quality_metric: string; models: ParetoPoint[]; frontier: ParetoPoint[] }>(
      "/v1/analytics/pareto",
      { hours },
    ),
  drift: (metric: string, currentHours: number, referenceHours: number) =>
    get<DriftReport>("/v1/analytics/drift", {
      metric,
      current_hours: currentHours,
      reference_hours: referenceHours,
    }),
};

// -- formatting ------------------------------------------------------------

/** Micro-USD to a currency string, or an explicit marker when unpriced. */
export function formatCost(microUsd: number | null | undefined): string {
  if (microUsd === null || microUsd === undefined) return "—";
  const usd = microUsd / 1_000_000;
  // Costs are frequently sub-cent; two decimal places would render most real
  // spans as "$0.00" and make the column useless.
  if (usd > 0 && usd < 0.01) return `$${usd.toFixed(6)}`;
  return `$${usd.toFixed(4)}`;
}

export function formatDuration(nanos: number): string {
  const ms = nanos / 1e6;
  if (ms < 1) return `${(nanos / 1e3).toFixed(0)}µs`;
  if (ms < 1000) return `${ms.toFixed(1)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return value.toLocaleString();
}
