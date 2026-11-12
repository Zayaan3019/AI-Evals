import { useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api, type CaseResult } from "../api/client";
import { QueryState } from "../components/QueryState";
import { evaluatorName } from "./EvalsPage";

type Filter = "all" | "judge-disagrees" | "failed" | "errors";

function judgeScore(c: CaseResult) {
  return c.scores.find((s) => s.identity.startsWith("correctness_vs_reference"));
}

function exactScore(c: CaseResult) {
  return c.scores.find((s) => s.identity.startsWith("exact_match"));
}

/** Every case of one run: input, reference, the system's answer, each grade. */
export function EvalRunPage() {
  const { runId = "" } = useParams();
  const [filter, setFilter] = useState<Filter>("all");
  const query = useQuery({
    queryKey: ["eval-results", runId],
    queryFn: () => api.evalResults(runId),
    refetchInterval: false,
    staleTime: Infinity,
  });

  const cases = useMemo(() => {
    const all = query.data?.cases ?? [];
    switch (filter) {
      case "failed":
        return all.filter((c) => c.scores.some((s) => s.passed === false));
      case "errors":
        return all.filter((c) => c.scores.some((s) => s.error));
      case "judge-disagrees":
        // Where string matching and the judge disagree: the cases that show
        // why a judge is used at all, and where it might be wrong.
        return all.filter((c) => {
          const j = judgeScore(c);
          const e = exactScore(c);
          return j?.passed != null && e?.passed != null && j.passed !== e.passed;
        });
      default:
        return all;
    }
  }, [query.data, filter]);

  const identities = useMemo(
    () => [...new Set((query.data?.cases ?? []).flatMap((c) => c.scores.map((s) => s.identity)))].sort(),
    [query.data],
  );

  return (
    <>
      <p>
        <Link to="/evals">← Evaluations</Link>
      </p>
      <div className="panel">
        <h2>{query.data?.system_label ?? "Eval run"}</h2>
        <div className="controls wrap">
          <select value={filter} onChange={(e) => setFilter(e.target.value as Filter)}>
            <option value="all">All cases</option>
            <option value="failed">Failed by any evaluator</option>
            <option value="judge-disagrees">Judge and exact match disagree</option>
            <option value="errors">Evaluation errors</option>
          </select>
          {query.data && (
            <span className="muted">
              {cases.length} of {query.data.count} cases
            </span>
          )}
        </div>
        <QueryState
          isLoading={query.isLoading}
          error={query.error}
          isEmpty={cases.length === 0}
          emptyMessage="No cases match this filter."
        >
          <div className="scroll-x">
            <table>
              <thead>
                <tr>
                  <th>Input</th>
                  <th>Reference</th>
                  <th>Output</th>
                  {identities.map((id) => (
                    <th key={id} className="num">
                      {evaluatorName(id)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {cases.map((c) => (
                  <tr key={c.external_id}>
                    <td className="cell-text">{String(c.inputs.question ?? JSON.stringify(c.inputs))}</td>
                    <td className="cell-text muted">{c.expected_output ?? "—"}</td>
                    <td className="cell-text">{c.actual_output ?? "—"}</td>
                    {identities.map((id) => {
                      const s = c.scores.find((x) => x.identity === id);
                      if (!s) return <td key={id} className="num">—</td>;
                      if (s.error)
                        return (
                          <td key={id} className="num err" title={s.error}>
                            error
                          </td>
                        );
                      return (
                        <td
                          key={id}
                          className={`num ${s.passed === false ? "err" : s.passed ? "ok" : ""}`}
                          title={s.explanation ?? ""}
                        >
                          {s.score === null ? "—" : s.score.toFixed(2)}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="sub" style={{ marginTop: 10 }}>
            Hover a judge score to read the judge's stated reason.
          </p>
        </QueryState>
      </div>
    </>
  );
}
