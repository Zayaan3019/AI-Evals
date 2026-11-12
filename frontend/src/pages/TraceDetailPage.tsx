import { useQuery } from "@tanstack/react-query";
import { Link, useParams } from "react-router-dom";
import { api, formatCost, formatDuration, formatNumber, type Span } from "../api/client";
import { QueryState } from "../components/QueryState";

interface TreeNode {
  span: Span;
  depth: number;
}

/**
 * Reconstruct the span tree.
 *
 * Ordering within a parent is by start time, but that is only a tiebreaker for
 * display -- parentage comes from ``parent_span_id``, never from timestamps.
 * Nested spans routinely share an identical start time because the OS clock
 * does not resolve them apart, so a tree built by sorting on time puts children
 * above their parents.
 *
 * Spans whose parent is missing from the trace are rendered as additional roots
 * rather than dropped: a partial trace is a real thing (the SDK drops spans
 * under load, by design) and hiding the orphans would misrepresent what ran.
 */
function buildTree(spans: Span[]): TreeNode[] {
  const byId = new Map(spans.map((s) => [s.span_id, s]));
  const children = new Map<string, Span[]>();
  const roots: Span[] = [];

  for (const span of spans) {
    if (span.parent_span_id && byId.has(span.parent_span_id)) {
      const bucket = children.get(span.parent_span_id) ?? [];
      bucket.push(span);
      children.set(span.parent_span_id, bucket);
    } else {
      roots.push(span);
    }
  }

  const byStart = (a: Span, b: Span) =>
    new Date(a.start_time).getTime() - new Date(b.start_time).getTime();

  const out: TreeNode[] = [];
  const visit = (span: Span, depth: number) => {
    out.push({ span, depth });
    for (const child of (children.get(span.span_id) ?? []).sort(byStart)) {
      visit(child, depth + 1);
    }
  };
  roots.sort(byStart).forEach((r) => visit(r, 0));
  return out;
}

function SpanRow({ node, traceStart, traceDuration }: {
  node: TreeNode;
  traceStart: number;
  traceDuration: number;
}) {
  const { span, depth } = node;
  const offset = new Date(span.start_time).getTime() - traceStart;
  const width = traceDuration > 0 ? (span.duration_ns / 1e6 / traceDuration) * 100 : 0;
  const left = traceDuration > 0 ? (offset / traceDuration) * 100 : 0;

  return (
    <tr>
      <td>
        <span className="tree-indent">{"  ".repeat(depth)}</span>
        {span.name}
        {span.status_code === "error" && (
          <span className="err"> ⚠ {span.error_type || "error"}</span>
        )}
      </td>
      <td className="muted">{span.gen_ai_operation_name || span.kind}</td>
      <td>
        <div className="bar-track">
          <div
            className="bar"
            style={{ width: `${Math.max(width, 0.5)}%`, marginLeft: `${Math.min(left, 99)}%` }}
          />
        </div>
      </td>
      <td className="num">{formatDuration(span.duration_ns)}</td>
      <td className="num">
        {span.gen_ai_usage_input_tokens === null && span.gen_ai_usage_output_tokens === null
          ? "—"
          : `${formatNumber(span.gen_ai_usage_input_tokens)} / ${formatNumber(span.gen_ai_usage_output_tokens)}`}
      </td>
      <td className="num">
        {/* cost_status distinguishes "we could not price this model" from
            "this span cost nothing", which look the same as a bare number. */}
        {span.cost_status === "no_pricing" ? (
          <span className="muted" title="no pricing table entry for this model">no pricing</span>
        ) : span.cost_status === "no_usage" ? (
          <span className="muted" title="no token usage recorded">no usage</span>
        ) : (
          formatCost(span.cost_micro_usd)
        )}
      </td>
      <td className="muted mono">
        {span.redaction_applied ? span.redaction_rules.join(", ") || "yes" : "—"}
      </td>
    </tr>
  );
}

export function TraceDetailPage() {
  const { traceId = "" } = useParams();
  const { data, isLoading, error } = useQuery({
    queryKey: ["trace", traceId],
    queryFn: () => api.trace(traceId),
    enabled: traceId.length > 0,
  });

  const spans = data?.spans ?? [];
  const tree = buildTree(spans);
  const traceStart = spans.length
    ? Math.min(...spans.map((s) => new Date(s.start_time).getTime()))
    : 0;
  const traceDuration = spans.length
    ? Math.max(...spans.map((s) => new Date(s.end_time).getTime())) - traceStart
    : 0;

  const totalCost = spans.reduce((sum, s) => sum + (s.cost_micro_usd ?? 0), 0);
  const anyUnpriced = spans.some((s) => s.cost_status === "no_pricing");
  const rootCount = tree.filter((n) => n.depth === 0).length;

  return (
    <>
      <div className="controls">
        <Link to="/traces">← Traces</Link>
        <span className="mono muted">{traceId}</span>
      </div>

      <div className="panel">
        <h2>Trace</h2>
        <QueryState
          isLoading={isLoading}
          error={error}
          isEmpty={!isLoading && spans.length === 0}
          emptyMessage="No spans for this trace. It may have expired: spans are kept under a retention TTL."
        >
          <p className="sub">
            {spans.length} spans · {formatDuration(traceDuration * 1e6)} wall ·{" "}
            {anyUnpriced ? (
              <>
                {formatCost(totalCost)} <span className="warn">(partial — some models unpriced)</span>
              </>
            ) : (
              formatCost(totalCost)
            )}
            {rootCount > 1 && (
              <>
                {" · "}
                <span className="warn">
                  {rootCount} roots — some spans have no parent in this trace
                </span>
              </>
            )}
          </p>
          <table>
            <thead>
              <tr>
                <th>Span</th>
                <th>Operation</th>
                <th style={{ width: "26%" }}>Timeline</th>
                <th className="num">Duration</th>
                <th className="num">Tokens</th>
                <th className="num">Cost</th>
                <th>Redacted</th>
              </tr>
            </thead>
            <tbody>
              {tree.map((node) => (
                <SpanRow
                  key={node.span.span_id}
                  node={node}
                  traceStart={traceStart}
                  traceDuration={traceDuration}
                />
              ))}
            </tbody>
          </table>
        </QueryState>
      </div>
    </>
  );
}
