import { useMemo } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api, type EvalRunSummary, type GateReport, type JudgeCalibration } from "../api/client";
import { QueryState } from "../components/QueryState";
import { ForestPlot, GroupedBars, Legend, Meter, StatTile, seriesVar } from "../components/Charts";

/**
 * The evaluation half of the platform: judge calibration, eval runs and the
 * regression gate, all read from the same objects the CLI gate uses.
 *
 * Nothing here computes a statistic. Means, intervals, p-values and kappa come
 * from the API, which calls the platform's own gate and calibration code.
 */

const EVALUATOR_NAMES: Record<string, string> = {
  exact_match: "Exact match",
  token_f1: "Token F1",
  correctness_vs_reference: "LLM judge",
};

export function evaluatorName(identity: string): string {
  const [slug = identity, version] = identity.split("@");
  return `${EVALUATOR_NAMES[slug] ?? slug}${version && version !== "1" ? ` v${version}` : ""}`;
}

function pct(value: number | null | undefined, digits = 1): string {
  return value === null || value === undefined ? "—" : `${(value * 100).toFixed(digits)}%`;
}

function num(value: number | null | undefined, digits = 3): string {
  return value === null || value === undefined ? "—" : value.toFixed(digits);
}

function signed(value: number, digits = 3): string {
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
}

/** Landis & Koch (1977) bands, the usual reading of Cohen's kappa. */
function kappaBand(kappa: number | null): string {
  if (kappa === null) return "";
  if (kappa < 0) return "poor";
  if (kappa <= 0.2) return "slight";
  if (kappa <= 0.4) return "fair";
  if (kappa <= 0.6) return "moderate";
  if (kappa <= 0.8) return "substantial";
  return "almost perfect";
}

const VERDICT_STYLE: Record<string, { label: string; className: string; explain: string }> = {
  pass: {
    label: "PASS",
    className: "ok",
    explain: "No evaluator's mean dropped by more than the allowed margin at 95% confidence.",
  },
  fail: {
    label: "FAIL",
    className: "err",
    explain: "At least one evaluator regressed: its 95% interval lies entirely below the allowed drop.",
  },
  incomparable: {
    label: "INCOMPARABLE",
    className: "warn",
    explain: "The runs were graded by different evaluator configurations, so their scores measure different things. The gate refuses to compare them.",
  },
  insufficient_data: {
    label: "INSUFFICIENT DATA",
    className: "warn",
    explain: "Too few paired cases for an interval that means anything. The gate refuses rather than guesses.",
  },
  error_rate: {
    label: "ERROR RATE",
    className: "warn",
    explain: "Too many results failed to evaluate; the surviving subset may not represent the dataset.",
  },
};

function runTime(run: EvalRunSummary): number {
  return run.started_at ? Date.parse(run.started_at) : 0;
}

function purpose(run: EvalRunSummary): string {
  return String(run.metadata.purpose ?? "");
}

/** Latest run per system label and dataset: a rerun replaces, not duplicates. */
function latestRuns(runs: EvalRunSummary[]): EvalRunSummary[] {
  const byKey = new Map<string, EvalRunSummary>();
  for (const run of runs) {
    const key = `${run.dataset.slug}|${run.system_label}`;
    const seen = byKey.get(key);
    if (!seen || runTime(run) > runTime(seen)) byKey.set(key, run);
  }
  return [...byKey.values()].sort((a, b) => runTime(a) - runTime(b));
}

interface Preset {
  title: string;
  baseline: EvalRunSummary;
  candidate: EvalRunSummary;
}

/** The comparisons the benchmark was designed to ask, in a fixed order. */
function presets(runs: EvalRunSummary[]): Preset[] {
  const comparison = runs.filter((r) => purpose(r) === "model_comparison");
  const baseline = comparison[0];
  const out: Preset[] = [];
  if (baseline) {
    for (const candidate of comparison.slice(1)) {
      const retrieve = candidate.metadata.retrieve;
      out.push({
        title:
          retrieve === false
            ? "Ablation: does retrieval matter?"
            : `Can ${String(candidate.metadata.model ?? candidate.system_label)} replace the baseline?`,
        baseline,
        candidate,
      });
    }
    const stricter = runs.find((r) => purpose(r) === "gate_refusal_incomparable");
    if (stricter) {
      out.push({ title: "Same answers, stricter judge threshold", baseline, candidate: stricter });
    }
  }
  const [smokeA, smokeB] = runs.filter((r) => purpose(r) === "gate_refusal_insufficient_data");
  if (smokeA && smokeB) {
    out.push({ title: "Ten questions only", baseline: smokeA, candidate: smokeB });
  }
  return out;
}

export function EvalsPage() {
  const [params, setParams] = useSearchParams();
  const runsQuery = useQuery({ queryKey: ["eval-runs"], queryFn: () => api.evalRuns(100) });
  const calibration = useQuery({ queryKey: ["calibration"], queryFn: api.calibration });
  // A judge configuration used only for grading (e.g. the stricter-threshold
  // run) has no human-labelled answers; an empty row would read as a failure.
  const calibrated = (calibration.data?.judges ?? []).filter(
    (j) => (j.calibration ? j.calibration.n : j.n_labelled) > 0,
  );

  const runs = useMemo(() => latestRuns(runsQuery.data?.runs ?? []), [runsQuery.data]);
  const gateRuns = runs.filter((r) => purpose(r) !== "judge_calibration");
  const comparison = runs.filter((r) => purpose(r) === "model_comparison");
  const presetList = useMemo(() => presets(runs), [runs]);

  const baselineId = params.get("baseline") ?? presetList[0]?.baseline.run_id ?? "";
  const candidateId = params.get("candidate") ?? presetList[0]?.candidate.run_id ?? "";

  const select = (baseline: string, candidate: string) =>
    setParams({ baseline, candidate }, { replace: true });

  return (
    <>
      <section className="hero">
        <h2>Evaluating a retrieval QA agent, with a judge that is itself checked</h2>
        <p className="lead">
          A question-answering agent (BM25 retrieval over 400 Wikipedia paragraphs, answered by
          hosted open-weight models) was run on 100 questions from the{" "}
          <a href="https://rajpurkar.github.io/SQuAD-explorer/" target="_blank" rel="noreferrer">
            SQuAD v1.1
          </a>{" "}
          dev set in four configurations, and every answer was graded three ways. Before the LLM
          judge's grades are trusted, the judge is calibrated against human experts on 150 student
          answers from{" "}
          <a href="https://huggingface.co/datasets/nkazi/SciEntsBank" target="_blank" rel="noreferrer">
            SciEntsBank (SemEval-2013 Task 7)
          </a>
          . Every model call was real, and every number below comes from the platform's own gate
          and calibration code.
        </p>
        <KpiRow runs={runs} judges={calibration.data?.judges ?? []} />
      </section>

      <div className="panel">
        <h2>Four configurations, three graders</h2>
        <p className="sub">
          Mean score on the same 100 questions (0 to 1). Exact match and token F1 are the SQuAD
          metrics against one gold answer; the judge reads the answer and the reference.
        </p>
        <QueryState
          isLoading={runsQuery.isLoading}
          error={runsQuery.error}
          isEmpty={comparison.length === 0}
          emptyMessage="No model-comparison runs yet."
        >
          <ComparisonChart runs={comparison} />
        </QueryState>
      </div>

      <div className="panel">
        <h2>Regression gate</h2>
        <p className="sub">
          Would the change be allowed to ship? The gate pairs the two runs question by question
          and bootstraps the difference in means (10,000 resamples, 95% interval). When the runs
          cannot honestly be compared it refuses and says why; a refusal is never turned into a
          pass.
        </p>
        <QueryState
          isLoading={runsQuery.isLoading}
          error={runsQuery.error}
          isEmpty={gateRuns.length < 2}
          emptyMessage="At least two eval runs are needed for a comparison."
        >
          <div className="presets" role="group" aria-label="Comparisons">
            {presetList.map((p) => {
              const active = p.baseline.run_id === baselineId && p.candidate.run_id === candidateId;
              return (
                <button
                  key={`${p.baseline.run_id}-${p.candidate.run_id}`}
                  className={active ? "preset active" : "preset"}
                  aria-pressed={active}
                  onClick={() => select(p.baseline.run_id, p.candidate.run_id)}
                >
                  {p.title}
                </button>
              );
            })}
          </div>
          <div className="controls wrap">
            <label>
              Baseline
              <select value={baselineId} onChange={(e) => select(e.target.value, candidateId)}>
                {gateRuns.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.system_label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Candidate
              <select value={candidateId} onChange={(e) => select(baselineId, e.target.value)}>
                {gateRuns.map((r) => (
                  <option key={r.run_id} value={r.run_id}>
                    {r.system_label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {baselineId && candidateId && <GatePanel baseline={baselineId} candidate={candidateId} />}
        </QueryState>
      </div>

      <div className="panel">
        <h2>Can the judge be trusted?</h2>
        <p className="sub">
          Agreement between each judge model and the human annotators on the same 150 answers.
          Cohen's κ discounts the agreement two graders would reach by chance; 0.61 to 0.80 is
          conventionally read as substantial.
        </p>
        <QueryState
          isLoading={calibration.isLoading}
          error={calibration.error}
          isEmpty={calibrated.length === 0}
          emptyMessage="No judge has been calibrated yet."
        >
          <CalibrationTable judges={calibrated} />
          {calibration.data && calibration.data.labellers.length > 0 && (
            <p className="sub" style={{ marginTop: 10 }}>
              Human labels: {calibration.data.labellers.join("; ")}.
            </p>
          )}
        </QueryState>
      </div>

      <div className="panel">
        <h2>All eval runs</h2>
        <p className="sub">Open a run to read every answer, the reference and each grade.</p>
        <QueryState
          isLoading={runsQuery.isLoading}
          error={runsQuery.error}
          isEmpty={runs.length === 0}
          emptyMessage="No eval runs yet."
        >
          <RunsTable runs={runs} />
        </QueryState>
      </div>
    </>
  );
}

const IDENTITY_ORDER = ["exact_match", "token_f1", "correctness_vs_reference"];

function identityRank(id: string): number {
  const i = IDENTITY_ORDER.indexOf(id.split("@")[0] ?? id);
  return i === -1 ? 99 : i;
}

function orderedIdentities(runs: EvalRunSummary[]): string[] {
  const all = new Set(runs.flatMap((r) => r.evaluators.map((e) => e.identity)));
  return [...all].sort((a, b) => identityRank(a) - identityRank(b) || a.localeCompare(b));
}

function mean(run: EvalRunSummary | undefined, slug: string): number | null {
  return run?.evaluators.find((e) => e.identity.startsWith(`${slug}@`))?.mean_score ?? null;
}

function KpiRow({ runs, judges }: { runs: EvalRunSummary[]; judges: JudgeCalibration[] }) {
  const best = judges
    .filter((j) => j.calibration?.cohens_kappa != null)
    .sort((a, b) => (b.calibration?.cohens_kappa ?? 0) - (a.calibration?.cohens_kappa ?? 0))[0];
  const comparison = runs.filter((r) => purpose(r) === "model_comparison");
  const top = [...comparison].sort(
    (a, b) => (mean(b, "token_f1") ?? -1) - (mean(a, "token_f1") ?? -1),
  )[0];
  const graded = runs.reduce((n, r) => n + r.case_count * r.evaluators.length, 0);
  if (!best && !top) return null;
  return (
    <div className="kpis">
      {best?.calibration && (
        <StatTile
          label="Best judge's agreement with human graders"
          value={<>κ {num(best.calibration.cohens_kappa, 2)}</>}
          detail={`${pct(best.calibration.binary_agreement)} of ${best.calibration.n} answers · ${best.model ?? best.judge}`}
        />
      )}
      {top && (
        <StatTile
          label="Best configuration, token F1"
          value={num(mean(top, "token_f1"), 2)}
          detail={String(top.metadata.model ?? top.system_label)}
        />
      )}
      <StatTile
        label="Grades recorded"
        value={graded.toLocaleString()}
        detail={`${runs.length} runs, each stored with its evaluator configuration`}
      />
    </div>
  );
}

function ComparisonChart({ runs }: { runs: EvalRunSummary[] }) {
  const identities = orderedIdentities(runs);
  return (
    <>
      <Legend
        items={identities.map((id) => ({ key: id, label: evaluatorName(id), color: seriesVar(id) }))}
      />
      <GroupedBars
        seriesLabel={evaluatorName}
        rows={runs.map((r) => ({
          key: r.run_id,
          label: String(r.metadata.model ?? r.system_label),
          sublabel:
            r.metadata.retrieve === false
              ? "no retrieval (ablation)"
              : `BM25 top-${String(r.metadata.top_k ?? 2)}`,
          values: identities.map((id) => ({
            series: id,
            value: r.evaluators.find((e) => e.identity === id)?.mean_score ?? null,
          })),
        }))}
      />
    </>
  );
}

function CalibrationTable({ judges }: { judges: JudgeCalibration[] }) {
  return (
    <div className="scroll-x">
      <table>
        <thead>
          <tr>
            <th>Judge model</th>
            <th className="num">Threshold</th>
            <th className="num">Answers</th>
            <th className="num">Cohen's κ</th>
            <th className="num">Agreement</th>
            <th className="num">Spearman ρ</th>
            <th className="num">Judge / human pass rate</th>
          </tr>
        </thead>
        <tbody>
          {judges.map((j) => {
            const c = j.calibration;
            return (
              <tr key={`${j.config_hash}-${j.dataset.slug}`}>
                <td>
                  {j.model ?? j.judge}
                  <div className="muted small">{j.provider} · {j.dataset.name}</div>
                </td>
                <td className="num">{num(j.threshold, 2)}</td>
                <td className="num">{c ? c.n : j.n_labelled}</td>
                <td className="num">
                  {c ? (
                    <>
                      <div className="kappa">
                        {c.cohens_kappa !== null && (
                          <Meter value={c.cohens_kappa} label={`Cohen's kappa ${c.cohens_kappa.toFixed(2)}`} />
                        )}
                        <strong>{num(c.cohens_kappa, 2)}</strong>
                      </div>
                      <div className="muted small">{kappaBand(c.cohens_kappa)}</div>
                    </>
                  ) : (
                    <span className="muted" title={j.note ?? ""}>not computed</span>
                  )}
                </td>
                <td className="num">{c ? pct(c.binary_agreement) : "—"}</td>
                <td className="num">{c ? num(c.spearman_r, 2) : "—"}</td>
                <td className="num">
                  {c ? (
                    <>
                      {pct(c.judge_mean, 0)} / {pct(c.human_mean, 0)}
                      {c.judge_is_biased_high && (
                        <div className="warn small">judge is lenient</div>
                      )}
                    </>
                  ) : (
                    "—"
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function GatePanel({ baseline, candidate }: { baseline: string; candidate: string }) {
  const same = baseline === candidate;
  const gate = useQuery({
    queryKey: ["gate", baseline, candidate],
    queryFn: () => api.gate(baseline, candidate),
    enabled: !same,
    staleTime: Infinity,
    refetchInterval: false,
  });
  if (same) return <div className="notice">Pick two different runs.</div>;
  return (
    <QueryState isLoading={gate.isLoading} error={gate.error}>
      {gate.data && <GateResult report={gate.data} />}
    </QueryState>
  );
}

function GateResult({ report: raw }: { report: GateReport }) {
  // Same evaluator order everywhere on the page (exact match, F1, judge).
  const report = {
    ...raw,
    evaluators: [...raw.evaluators].sort(
      (a, b) => identityRank(a.identity) - identityRank(b.identity),
    ),
  };
  const style = VERDICT_STYLE[report.verdict] ?? {
    label: report.verdict,
    className: "warn",
    explain: "",
  };
  return (
    <div className="gate">
      <div className="gate-head">
        <span className={`verdict ${style.className}`}>{style.label}</span>
        <span className={report.blocks_release ? "err" : "ok"}>
          {report.blocks_release ? "Release blocked" : "Release allowed"}
        </span>
      </div>
      <p className="sub">{style.explain}</p>
      <p className="mono small reason">{report.reason}</p>
      {report.evaluators.some((e) => e.comparison) && (
        <ForestPlot
          margin={report.policy.max_mean_drop}
          rows={report.evaluators.flatMap((e) =>
            e.comparison
              ? [
                  {
                    key: e.identity,
                    label: evaluatorName(e.identity),
                    estimate: e.comparison.mean_difference,
                    low: e.comparison.ci_low,
                    high: e.comparison.ci_high,
                    regressed: e.regressed,
                  },
                ]
              : [],
          )}
        />
      )}
      {report.evaluators.some((e) => e.comparison) && (
        <div className="scroll-x">
          <table>
            <thead>
              <tr>
                <th>Evaluator</th>
                <th className="num">Baseline</th>
                <th className="num">Candidate</th>
                <th className="num">Difference</th>
                <th className="num">95% interval</th>
                <th className="num">p</th>
                <th className="num">Pairs</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {report.evaluators.map((e) => {
                const c = e.comparison;
                return (
                  <tr key={e.identity}>
                    <td>{evaluatorName(e.identity)}</td>
                    <td className="num">{c ? num(c.baseline_mean) : "—"}</td>
                    <td className="num">{c ? num(c.candidate_mean) : "—"}</td>
                    <td className="num">{c ? signed(c.mean_difference) : "—"}</td>
                    <td className="num">
                      {c ? `[${signed(c.ci_low)}, ${signed(c.ci_high)}]` : "—"}
                    </td>
                    <td className="num">{c ? (c.p_value < 0.0001 ? "<0.0001" : c.p_value.toFixed(4)) : "—"}</td>
                    <td className="num">{c ? c.n_pairs : "—"}</td>
                    <td>
                      {e.regressed ? (
                        <span className="err">regressed</span>
                      ) : c ? (
                        <span className="muted">within margin</span>
                      ) : (
                        <span className="muted">{e.note}</span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {report.minimum_detectable_effect !== null && (
        <p className="sub" style={{ marginTop: 10 }}>
          Smallest drop this comparison could reliably detect: about{" "}
          {num(report.minimum_detectable_effect)}. A pass means "no drop larger than that", not
          "no drop at all". Policy: {report.policy.resamples.toLocaleString()} bootstrap resamples,{" "}
          {Math.round(report.policy.confidence * 100)}% confidence, at least {report.policy.min_pairs} paired
          cases, at most {pct(report.policy.max_error_rate, 0)} evaluation errors.
        </p>
      )}
      <p className="sub">
        <Link to={`/evals/runs/${report.baseline_run_id}`}>Baseline answers</Link> ·{" "}
        <Link to={`/evals/runs/${report.candidate_run_id}`}>Candidate answers</Link>
      </p>
    </div>
  );
}

function RunsTable({ runs }: { runs: EvalRunSummary[] }) {
  const identities = useMemo(() => orderedIdentities(runs), [runs]);

  return (
    <div className="scroll-x">
      <table>
        <thead>
          <tr>
            <th>System</th>
            <th>Dataset</th>
            <th className="num">Cases</th>
            {identities.map((id) => (
              <th key={id} className="num">
                {evaluatorName(id)}
              </th>
            ))}
            <th className="num">Errors</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => {
            const byId = new Map(run.evaluators.map((e) => [e.identity, e]));
            const errors = run.evaluators.reduce((sum, e) => sum + e.n_errors, 0);
            return (
              <tr key={run.run_id}>
                <td>
                  <Link to={`/evals/runs/${run.run_id}`}>{run.system_label}</Link>
                  {typeof run.metadata.description === "string" && (
                    <div className="muted small">{run.metadata.description}</div>
                  )}
                </td>
                <td className="muted">{run.dataset.slug}</td>
                <td className="num">{run.case_count}</td>
                {identities.map((id) => (
                  <td key={id} className="num">
                    {num(byId.get(id)?.mean_score ?? null)}
                  </td>
                ))}
                <td className="num">{errors > 0 ? <span className="err">{errors}</span> : "0"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
