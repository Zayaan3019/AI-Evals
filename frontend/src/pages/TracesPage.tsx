import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api, formatCost, formatDuration, formatNumber } from "../api/client";
import { QueryState } from "../components/QueryState";

const WINDOWS = [
  { label: "Last hour", hours: 1 },
  { label: "Last 6 hours", hours: 6 },
  { label: "Last 24 hours", hours: 24 },
  { label: "Last 7 days", hours: 168 },
];

export function TracesPage() {
  const [hours, setHours] = useState(24);
  const { data, isLoading, error } = useQuery({
    queryKey: ["traces", hours],
    queryFn: () => api.traces(hours),
  });

  return (
    <>
      <div className="controls">
        <select value={hours} onChange={(e) => setHours(Number(e.target.value))}>
          {WINDOWS.map((w) => (
            <option key={w.hours} value={w.hours}>
              {w.label}
            </option>
          ))}
        </select>
        <span className="muted">{data ? `${data.count} traces` : ""}</span>
      </div>

      <div className="panel">
        <h2>Traces</h2>
        <QueryState
          isLoading={isLoading}
          error={error}
          isEmpty={data?.traces.length === 0}
          emptyMessage="No traces captured in this window. If an agent is running, check that it is exporting to this collector."
        >
          <table>
            <thead>
              <tr>
                <th>Root span</th>
                <th>Service</th>
                <th className="num">Spans</th>
                <th className="num">Duration</th>
                <th className="num">Tokens in / out</th>
                <th className="num">Cost</th>
                <th>Status</th>
                <th>Started</th>
              </tr>
            </thead>
            <tbody>
              {data?.traces.map((t) => (
                <tr key={t.trace_id}>
                  <td>
                    <Link to={`/traces/${t.trace_id}`}>{t.root_name}</Link>
                    <div className="mono muted">{t.trace_id.slice(0, 16)}…</div>
                  </td>
                  <td>{t.service_name}</td>
                  <td className="num">{t.span_count}</td>
                  <td className="num">{formatDuration(t.root_duration_ns)}</td>
                  <td className="num">
                    {formatNumber(t.input_tokens)} / {formatNumber(t.output_tokens)}
                  </td>
                  {/* An unpriced trace renders as a dash, never as $0.00 --
                      the backend records absence rather than zero and the UI
                      must not undo that. */}
                  <td className="num">{formatCost(t.cost_micro_usd)}</td>
                  <td>
                    {t.error_count > 0 ? (
                      <span className="err">{t.error_count} error{t.error_count > 1 ? "s" : ""}</span>
                    ) : (
                      <span className="ok">ok</span>
                    )}
                  </td>
                  <td className="muted">{new Date(t.started_at + "Z").toLocaleTimeString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </QueryState>
      </div>
    </>
  );
}
