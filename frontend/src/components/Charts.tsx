/**
 * Small, dependency-free chart pieces for the evaluation pages.
 *
 * Conventions (kept deliberately few): marks carry the series colour, text
 * never does; bars are <= 24px with a rounded data end and a square baseline;
 * every mark has a hover title, and every chart sits next to a table that
 * carries the same numbers, so nothing is readable by colour alone.
 */

import type { ReactNode } from "react";

/** Fixed evaluator -> colour slot. Identity follows the evaluator, never its rank. */
const SERIES_SLOT: Record<string, number> = {
  exact_match: 1,
  token_f1: 2,
  correctness_vs_reference: 3,
};

export function seriesVar(identity: string): string {
  const slug = identity.split("@")[0] ?? identity;
  return `var(--series-${SERIES_SLOT[slug] ?? 1})`;
}

export function StatTile({
  label,
  value,
  detail,
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
}) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {detail && <div className="stat-detail">{detail}</div>}
    </div>
  );
}

export function Legend({ items }: { items: { key: string; label: string; color: string }[] }) {
  return (
    <ul className="legend" aria-label="Legend">
      {items.map((item) => (
        <li key={item.key}>
          <span className="swatch" style={{ background: item.color }} aria-hidden="true" />
          {item.label}
        </li>
      ))}
    </ul>
  );
}

export interface GroupedBarRow {
  key: string;
  label: string;
  sublabel?: string;
  values: { series: string; value: number | null }[];
}

/**
 * Horizontal grouped bars on a shared 0..1 axis: one row per system, one bar
 * per evaluator. Values print at the bar tip in text ink.
 */
export function GroupedBars({
  rows,
  seriesLabel,
}: {
  rows: GroupedBarRow[];
  seriesLabel: (series: string) => string;
}) {
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  return (
    <div className="gbars" role="img" aria-label="Mean score per system and evaluator, 0 to 1">
      {rows.map((row) => (
        <div className="gbar-row" key={row.key}>
          <div className="gbar-label">
            <div>{row.label}</div>
            {row.sublabel && <div className="muted small">{row.sublabel}</div>}
          </div>
          <div className="gbar-plot">
            <div className="gbar-layer" aria-hidden="true">
              {ticks.map((t) => (
                <span key={t} className="gridline" style={{ left: `${t * 100}%` }} />
              ))}
            </div>
            {row.values.map((v) => (
              <div className="gbar-track" key={v.series}>
                {v.value !== null && (
                  <>
                    <div
                      className="gbar"
                      style={{ width: `${Math.max(0, Math.min(1, v.value)) * 100}%`, background: seriesVar(v.series) }}
                      title={`${row.label} · ${seriesLabel(v.series)}: ${v.value.toFixed(3)}`}
                    />
                    <span className="gbar-value">{v.value.toFixed(2)}</span>
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
      ))}
      <div className="gbar-row axis" aria-hidden="true">
        <div className="gbar-label" />
        <div className="gbar-plot">
          <div className="gbar-layer">
            {ticks.map((t) => (
              <span key={t} className="tick" style={{ left: `${t * 100}%` }}>
                {t}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export interface ForestRow {
  key: string;
  label: string;
  estimate: number;
  low: number;
  high: number;
  regressed: boolean;
}

/**
 * Difference in means (candidate - baseline) with its 95% interval, one row
 * per evaluator, on a symmetric axis around zero. The allowed-drop margin, if
 * any, is drawn as a second reference line.
 */
export function ForestPlot({ rows, margin }: { rows: ForestRow[]; margin: number }) {
  const extent = Math.max(0.1, margin, ...rows.flatMap((r) => [Math.abs(r.low), Math.abs(r.high)]));
  const span = niceCeil(extent);
  const x = (v: number) => 50 + (v / span) * 50; // percent
  const ticks = [-span, -span / 2, 0, span / 2, span];
  return (
    <div className="forest" role="img" aria-label="Difference in mean score with 95% interval, per evaluator">
      {rows.map((r) => (
        <div className="forest-row" key={r.key}>
          <div className="forest-label">
            {r.label}
            {r.regressed && (
              <span className="status critical small">
                <span aria-hidden="true">▼</span> regressed
              </span>
            )}
          </div>
          <div
            className="forest-plot"
            title={`${r.label}: ${fmt(r.estimate)} [${fmt(r.low)}, ${fmt(r.high)}]`}
          >
            {ticks.map((t) => (
              <span key={t} className={t === 0 ? "zero" : "gridline"} style={{ left: `${x(t)}%` }} aria-hidden="true" />
            ))}
            {margin > 0 && <span className="margin" style={{ left: `${x(-margin)}%` }} aria-hidden="true" />}
            <span
              className={`whisker ${r.regressed ? "bad" : ""}`}
              style={{ left: `${x(r.low)}%`, width: `${x(r.high) - x(r.low)}%` }}
            />
            <span className={`point ${r.regressed ? "bad" : ""}`} style={{ left: `${x(r.estimate)}%` }} />
          </div>
        </div>
      ))}
      <div className="forest-row axis" aria-hidden="true">
        <div className="forest-label" />
        <div className="forest-plot">
          {ticks.map((t) => (
            <span key={t} className="tick" style={{ left: `${x(t)}%` }}>
              {t === 0 ? "0" : fmt(t, 2)}
            </span>
          ))}
        </div>
      </div>
      <div className="forest-caption muted small">
        ← candidate worse · candidate better →
        {margin > 0 && ` · dashed line: allowed drop (${margin})`}
      </div>
    </div>
  );
}

/** A single ratio against [lo, hi], fill on a lighter track of the same hue. */
export function Meter({ value, lo = 0, hi = 1, label }: { value: number; lo?: number; hi?: number; label: string }) {
  const p = Math.max(0, Math.min(1, (value - lo) / (hi - lo)));
  return (
    <span className="meter" role="meter" aria-valuemin={lo} aria-valuemax={hi} aria-valuenow={value} aria-label={label}>
      <span className="meter-fill" style={{ width: `${p * 100}%` }} />
    </span>
  );
}

function niceCeil(v: number): number {
  for (const step of [0.05, 0.1, 0.2, 0.25, 0.5, 1]) if (v <= step) return step;
  return Math.ceil(v);
}

function fmt(v: number, digits = 3): string {
  return `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;
}
