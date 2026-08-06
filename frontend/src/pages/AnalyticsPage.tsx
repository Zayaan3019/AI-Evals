import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, formatCost, formatNumber } from "../api/client";
import { QueryState } from "../components/QueryState";

const METRICS = ["latency_ms", "input_tokens", "output_tokens", "cost_micro_usd"];

export function AnalyticsPage() {
  const [hours, setHours] = useState(24);
  const [metric, setMetric] = useState("latency_ms");

  const models = useQuery({ queryKey: ["models", hours], queryFn: () => api.models(hours) });
  const operations = useQuery({
    queryKey: ["operations", hours],
    queryFn: () => api.operations(hours),
  });
  const pareto = useQuery({ queryKey: ["pareto", hours], queryFn: () => api.pareto(hours) });
  const drift = useQuery({
    queryKey: ["drift", metric, hours],
    queryFn: () => api.drift(metric, 1, hours),
  });

  return (
    <>
      <div className="controls">
        <select value={hours} onChange={(e) => setHours(Number(e.target.value))}>
          <option value={1}>Last hour</option>
          <option value={6}>Last 6 hours</option>
          <option value={24}>Last 24 hours</option>
          <option value={168}>Last 7 days</option>
        </select>
      </div>

      <div className="panel">
        <h2>Model usage</h2>
        <QueryState
          isLoading={models.isLoading}
          error={models.error}
          isEmpty={models.data?.models.length === 0}
        >
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Provider</th>
                <th className="num">Calls</th>
                <th className="num">Tokens in / out</th>
                <th className="num">Cost</th>
                <th className="num">p50</th>
                <th className="num">p95</th>
                <th className="num">p99</th>
                <th className="num">Errors</th>
              </tr>
            </thead>
            <tbody>
              {models.data?.models.map((m) => (
                <tr key={`${m.provider}/${m.model}`}>
                  <td>{m.model}</td>
                  <td className="muted">{m.provider || "—"}</td>
                  <td className="num">{formatNumber(m.call_count)}</td>
                  <td className="num">
                    {formatNumber(m.input_tokens)} / {formatNumber(m.output_tokens)}
                  </td>
                  <td className="num">
                    {formatCost(m.cost_micro_usd)}
                    {m.unpriced_calls > 0 && (
                      <span className="warn" title={`${m.unpriced_calls} calls have no pricing entry`}>
                        {" "}*
                      </span>
                    )}
                  </td>
                  <td className="num">{m.p50_ms.toFixed(1)}</td>
                  <td className="num">{m.p95_ms.toFixed(1)}</td>
                  <td className="num">{m.p99_ms.toFixed(1)}</td>
                  <td className="num">
                    {m.error_count > 0 ? <span className="err">{m.error_count}</span> : "0"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {models.data?.models.some((m) => m.unpriced_calls > 0) && (
            <p className="sub" style={{ marginTop: 10 }}>
              * Cost is a lower bound: some calls used models with no entry in the pricing
              table and contribute nothing to the total.
            </p>
          )}
        </QueryState>
      </div>

      <div className="panel">
        <h2>Cost / quality / latency frontier</h2>
        <p className="sub">{pareto.data?.quality_metric}</p>
        <QueryState
          isLoading={pareto.isLoading}
          error={pareto.error}
          isEmpty={pareto.data?.models.length === 0}
        >
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th className="num">Success rate</th>
                <th className="num">Cost / call</th>
                <th className="num">p95 latency</th>
                <th className="num">Calls</th>
                <th>Frontier</th>
              </tr>
            </thead>
            <tbody>
              {pareto.data?.models.map((p) => (
                <tr key={p.model}>
                  <td>{p.model}</td>
                  <td className="num">{(p.quality_success_rate * 100).toFixed(1)}%</td>
                  <td className="num">{formatCost(p.cost_micro_usd_per_call)}</td>
                  <td className="num">{p.latency_p95_ms.toFixed(1)}ms</td>
                  <td className="num">{formatNumber(p.call_count)}</td>
                  <td>
                    {/* Everything on the frontier is a defensible choice. There
                        is deliberately no single "best" -- the tradeoff between
                        quality and cost belongs to whoever is choosing. */}
                    {p.on_frontier ? (
                      <span className="badge frontier">non-dominated</span>
                    ) : (
                      <span className="badge">dominated</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </QueryState>
      </div>

      <div className="panel">
        <h2>Drift</h2>
        <p className="sub">
          Last hour against the preceding {hours}h. Distribution comparison, not a mean:
          a bimodal split leaves the mean unchanged.
        </p>
        <div className="controls">
          <select value={metric} onChange={(e) => setMetric(e.target.value)}>
            {METRICS.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </div>
        <QueryState isLoading={drift.isLoading} error={drift.error}>
          {drift.data?.status === "no_data" || !drift.data?.drift ? (
            <div className="notice">
              Not enough data in one of the windows to compare. This is an absence of
              evidence, not evidence of stability.
            </div>
          ) : (
            <table>
              <tbody>
                <tr>
                  <th>PSI</th>
                  <td className="num">{drift.data.drift.psi.toFixed(4)}</td>
                  <td>
                    <span
                      className={
                        drift.data.drift.severity === "significant"
                          ? "err"
                          : drift.data.drift.severity === "moderate"
                            ? "warn"
                            : "ok"
                      }
                    >
                      {drift.data.drift.severity}
                    </span>
                    {drift.data.drift.underpowered && (
                      <span className="warn"> · underpowered (&lt;100 samples)</span>
                    )}
                  </td>
                </tr>
                <tr>
                  <th>KS statistic</th>
                  <td className="num">{drift.data.drift.ks_statistic.toFixed(4)}</td>
                  <td className="muted">max CDF gap</td>
                </tr>
                <tr>
                  <th>Samples</th>
                  <td className="num">
                    {formatNumber(drift.data.drift.reference_n)} →{" "}
                    {formatNumber(drift.data.drift.current_n)}
                  </td>
                  <td className="muted">reference → current</td>
                </tr>
                <tr>
                  <th>Mean</th>
                  <td className="num">
                    {drift.data.drift.reference_mean.toFixed(2)} →{" "}
                    {drift.data.drift.current_mean.toFixed(2)}
                  </td>
                  <td className="muted">p95: {drift.data.drift.reference_p95.toFixed(2)} → {drift.data.drift.current_p95.toFixed(2)}</td>
                </tr>
              </tbody>
            </table>
          )}
        </QueryState>
      </div>

      <div className="panel">
        <h2>Operation latency</h2>
        <QueryState
          isLoading={operations.isLoading}
          error={operations.error}
          isEmpty={operations.data?.operations.length === 0}
        >
          <table>
            <thead>
              <tr>
                <th>Operation</th>
                <th className="num">Spans</th>
                <th className="num">p50</th>
                <th className="num">p95</th>
                <th className="num">p99</th>
                <th className="num">Errors</th>
              </tr>
            </thead>
            <tbody>
              {operations.data?.operations.map((o) => (
                <tr key={o.operation}>
                  <td>{o.operation}</td>
                  <td className="num">{formatNumber(o.span_count)}</td>
                  <td className="num">{o.p50_ms.toFixed(1)}</td>
                  <td className="num">{o.p95_ms.toFixed(1)}</td>
                  <td className="num">{o.p99_ms.toFixed(1)}</td>
                  <td className="num">
                    {o.error_count > 0 ? <span className="err">{o.error_count}</span> : "0"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </QueryState>
      </div>
    </>
  );
}
