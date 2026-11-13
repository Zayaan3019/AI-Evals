"""Online drift detection and cost/quality/latency Pareto analysis.

Offline evals answer "is the new version better on our dataset". They cannot
answer "is production behaving differently today than last week", because the
dataset does not change and production does. Both questions matter and they need
different machinery.

**Drift is measured on distributions, not means.** A mean is insensitive to the
failure that actually happens: a bimodal split where half the traffic degrades
badly and half improves slightly leaves the mean flat. Population Stability
Index and a two-sample Kolmogorov-Smirnov statistic both see the shape change.

**Drift is not causation, and this module does not pretend otherwise.** A PSI
above threshold means "the input distribution moved", which can be a real
regression, a seasonal shift, a new customer onboarding, or a deploy that
changed nothing but the prompt template. The output is a flag for a human, not
an automatic rollback -- and the thresholds are the conventional ones, cited,
not tuned to make a demo look good.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

from clickhouse_connect.driver.client import Client

from .db.clickhouse import tenant_query, SPANS_TABLE
from .queries import TimeWindow, _TIME_PREDICATE

#: Conventional PSI interpretation from the credit-risk literature, where the
#: index originates. Not tuned here: inventing thresholds and then reporting
#: that the system detects drift at them would be circular.
PSI_NO_SIGNIFICANT_CHANGE = 0.10
PSI_MODERATE_SHIFT = 0.25


@dataclass(frozen=True, slots=True)
class DriftResult:
    """Distribution comparison between a reference and a current window."""

    metric: str
    psi: float
    ks_statistic: float
    reference_n: int
    current_n: int
    reference_mean: float
    current_mean: float
    reference_p95: float
    current_p95: float
    bucket_count: int

    @property
    def severity(self) -> str:
        if self.psi < PSI_NO_SIGNIFICANT_CHANGE:
            return "stable"
        if self.psi < PSI_MODERATE_SHIFT:
            return "moderate"
        return "significant"

    @property
    def is_underpowered(self) -> bool:
        """Whether either sample is too small to trust.

        PSI on tiny samples is dominated by which bucket a handful of points
        happened to land in. Reported rather than silently returning a number
        that looks authoritative.
        """
        return min(self.reference_n, self.current_n) < 100

    def render(self) -> str:
        caveat = "  (UNDERPOWERED: fewer than 100 samples in a window)" if self.is_underpowered else ""
        return (
            f"{self.metric}: PSI={self.psi:.4f} ({self.severity}) KS={self.ks_statistic:.4f}\n"
            f"  reference n={self.reference_n} mean={self.reference_mean:.3f} p95={self.reference_p95:.3f}\n"
            f"  current   n={self.current_n} mean={self.current_mean:.3f} p95={self.current_p95:.3f}"
            f"{caveat}"
        )


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def population_stability_index(
    reference: Sequence[float], current: Sequence[float], *, buckets: int = 10
) -> tuple[float, int]:
    """PSI between two samples, plus the number of buckets actually used.

    Buckets are quantile edges of the *reference*, which is the standard
    construction: it makes the reference roughly uniform across buckets so the
    index measures how far current has moved from it, rather than measuring the
    shape of the reference.

    Empty buckets are the classic way PSI blows up to infinity. Both proportions
    are floored at a small epsilon, which is the conventional fix; the
    alternative -- dropping empty buckets -- discards exactly the signal that a
    whole region of the distribution vanished.
    """
    if not reference or not current:
        raise ValueError("PSI needs a non-empty reference and current sample")

    ordered = sorted(reference)
    # Interior quantile edges. Deduplicated because a heavily tied distribution
    # (a score that is nearly always 1.0) produces repeated edges, and a
    # zero-width bucket contributes a spurious term.
    edges = sorted({_quantile(ordered, i / buckets) for i in range(1, buckets)})
    bucket_count = len(edges) + 1

    def proportions(sample: Sequence[float]) -> list[float]:
        counts = [0] * bucket_count
        for value in sample:
            index = 0
            while index < len(edges) and value > edges[index]:
                index += 1
            counts[index] += 1
        total = len(sample)
        return [count / total for count in counts]

    reference_p = proportions(ordered)
    current_p = proportions(current)

    epsilon = 1e-6
    psi = 0.0
    for r, c in zip(reference_p, current_p):
        r = max(r, epsilon)
        c = max(c, epsilon)
        psi += (c - r) * math.log(c / r)
    return psi, bucket_count


def ks_statistic(reference: Sequence[float], current: Sequence[float]) -> float:
    """Two-sample Kolmogorov-Smirnov statistic: max CDF gap.

    Complements PSI. PSI is bucketed and can miss a shift entirely inside one
    bucket; KS is bucket-free and sees any displacement of the CDF. Reporting
    both means a disagreement between them is itself informative.
    """
    if not reference or not current:
        raise ValueError("KS needs two non-empty samples")

    ordered_ref = sorted(reference)
    ordered_cur = sorted(current)
    i = j = 0
    cdf_ref = cdf_cur = 0.0
    largest = 0.0
    n_ref, n_cur = len(ordered_ref), len(ordered_cur)

    while i < n_ref and j < n_cur:
        # A value present in both samples must advance *both* CDFs before the
        # gap is measured. Advancing one side at a time leaves a transient gap
        # of 1/n at every tie, which reports a KS of 1/n for two identical
        # samples and inflates the statistic on any tied distribution -- which
        # is every bounded score, where ties are the norm rather than the
        # exception.
        if ordered_ref[i] < ordered_cur[j]:
            value = ordered_ref[i]
            while i < n_ref and ordered_ref[i] == value:
                i += 1
            cdf_ref = i / n_ref
        elif ordered_ref[i] > ordered_cur[j]:
            value = ordered_cur[j]
            while j < n_cur and ordered_cur[j] == value:
                j += 1
            cdf_cur = j / n_cur
        else:
            value = ordered_ref[i]
            while i < n_ref and ordered_ref[i] == value:
                i += 1
            while j < n_cur and ordered_cur[j] == value:
                j += 1
            cdf_ref = i / n_ref
            cdf_cur = j / n_cur
        largest = max(largest, abs(cdf_ref - cdf_cur))

    return largest


def compare_distributions(
    metric: str, reference: Sequence[float], current: Sequence[float], *, buckets: int = 10
) -> DriftResult:
    """PSI plus KS plus the summary statistics needed to interpret them."""
    psi, bucket_count = population_stability_index(reference, current, buckets=buckets)
    ordered_ref = sorted(reference)
    ordered_cur = sorted(current)
    return DriftResult(
        metric=metric,
        psi=psi,
        ks_statistic=ks_statistic(reference, current),
        reference_n=len(reference),
        current_n=len(current),
        reference_mean=sum(reference) / len(reference),
        current_mean=sum(current) / len(current),
        reference_p95=_quantile(ordered_ref, 0.95),
        current_p95=_quantile(ordered_cur, 0.95),
        bucket_count=bucket_count,
    )


# ---------------------------------------------------------------------------
# Pulling live samples out of the span store
# ---------------------------------------------------------------------------

#: Metrics drift is tracked on, mapped to their ClickHouse expression.
DRIFT_METRICS: dict[str, str] = {
    "latency_ms": "duration_ns / 1e6",
    "input_tokens": "gen_ai_usage_input_tokens",
    "output_tokens": "gen_ai_usage_output_tokens",
    "cost_micro_usd": "cost_micro_usd",
}


def sample_metric(
    client: Client,
    *,
    database: str,
    tenant_slug: str,
    window: TimeWindow,
    metric: str,
    operation: str = "chat",
    model: str | None = None,
    limit: int = 50_000,
) -> list[float]:
    """Draw a sample of one metric over a window.

    Sampling deterministically by hash rather than with ``rand()``: a
    reproducible sample means two people investigating the same drift alert see
    the same numbers. ``limit`` caps memory, since a busy tenant can produce
    millions of spans per window and PSI needs no more than tens of thousands.
    """
    if metric not in DRIFT_METRICS:
        raise ValueError(f"unknown drift metric {metric!r}; known: {sorted(DRIFT_METRICS)}")

    conditions = [
        "tenant_id = %(tenant)s",
        _TIME_PREDICATE,
        f"{DRIFT_METRICS[metric]} IS NOT NULL",
    ]
    parameters: dict[str, Any] = {
        "tenant": tenant_slug,
        "start_ns": window.start_ns,
        "end_ns": window.end_ns,
        "limit": limit,
    }
    if operation:
        conditions.append("gen_ai_operation_name = %(operation)s")
        parameters["operation"] = operation
    if model:
        conditions.append("gen_ai_request_model = %(model)s")
        parameters["model"] = model

    result = tenant_query(client, 
        f"""
        SELECT toFloat64({DRIFT_METRICS[metric]}) AS value
        FROM {database}.{SPANS_TABLE}
        WHERE {' AND '.join(conditions)}
        ORDER BY cityHash64(span_id)
        LIMIT %(limit)s
        """,
        parameters=parameters,
    )
    return [float(row[0]) for row in result.result_rows]


def detect_drift(
    client: Client,
    *,
    database: str,
    tenant_slug: str,
    reference_window: TimeWindow,
    current_window: TimeWindow,
    metric: str = "latency_ms",
    operation: str = "chat",
    model: str | None = None,
) -> DriftResult | None:
    """Compare a metric's distribution across two windows.

    Returns ``None`` when either window has no data -- an absence, which is
    different from "no drift" and must not be reported as a PSI of zero.
    """
    reference = sample_metric(
        client, database=database, tenant_slug=tenant_slug, window=reference_window,
        metric=metric, operation=operation, model=model,
    )
    current = sample_metric(
        client, database=database, tenant_slug=tenant_slug, window=current_window,
        metric=metric, operation=operation, model=model,
    )
    if not reference or not current:
        return None
    return compare_distributions(metric, reference, current)


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelPoint:
    """One model's aggregate position on cost, quality and latency."""

    label: str
    quality: float
    cost_micro_usd_per_call: float
    latency_p95_ms: float
    call_count: int

    def dominates(self, other: ModelPoint) -> bool:
        """True when this point is at least as good on every axis and better on one.

        Strict Pareto dominance. Quality is maximised; cost and latency are
        minimised. No weighting between the axes, deliberately: a weighted score
        would embed a tradeoff that belongs to whoever is choosing, not to this
        function.
        """
        at_least_as_good = (
            self.quality >= other.quality
            and self.cost_micro_usd_per_call <= other.cost_micro_usd_per_call
            and self.latency_p95_ms <= other.latency_p95_ms
        )
        strictly_better = (
            self.quality > other.quality
            or self.cost_micro_usd_per_call < other.cost_micro_usd_per_call
            or self.latency_p95_ms < other.latency_p95_ms
        )
        return at_least_as_good and strictly_better


def pareto_frontier(points: Sequence[ModelPoint]) -> list[ModelPoint]:
    """The non-dominated subset, ordered by quality descending.

    Everything on the frontier is a defensible choice; everything off it is
    strictly worse than something on it. Presenting the frontier rather than a
    single "best model" is the honest output, because the tradeoff between an
    extra point of quality and twice the cost is a business decision.
    """
    frontier = [
        candidate
        for candidate in points
        if not any(other.dominates(candidate) for other in points if other is not candidate)
    ]
    return sorted(frontier, key=lambda p: (-p.quality, p.cost_micro_usd_per_call))


def model_pareto(
    client: Client, *, database: str, tenant_slug: str, window: TimeWindow
) -> tuple[list[ModelPoint], list[ModelPoint]]:
    """Build cost/latency points per model from live spans.

    Returns ``(all_points, frontier)``.

    Quality here is the **success rate** -- the fraction of calls that did not
    error. That is a weak proxy and is named as one: real quality needs an
    evaluator, which lives in the offline path. It is included because latency
    and cost alone would make the cheapest broken model look optimal.
    """
    result = tenant_query(client, 
        f"""
        SELECT
            gen_ai_request_model                                AS model,
            count()                                             AS call_count,
            countIf(status_code = 'error')                      AS error_count,
            avg(cost_micro_usd)                                 AS avg_cost,
            quantileTDigest(0.95)(duration_ns) / 1e6            AS p95_ms
        FROM {database}.{SPANS_TABLE}
        WHERE tenant_id = %(tenant)s
          AND {_TIME_PREDICATE}
          AND gen_ai_request_model != ''
        GROUP BY model
        HAVING call_count > 0
        """,
        parameters={
            "tenant": tenant_slug,
            "start_ns": window.start_ns,
            "end_ns": window.end_ns,
        },
    )

    points = [
        ModelPoint(
            label=str(model),
            quality=(call_count - error_count) / call_count,
            cost_micro_usd_per_call=float(avg_cost or 0.0),
            latency_p95_ms=float(p95_ms or 0.0),
            call_count=int(call_count),
        )
        for model, call_count, error_count, avg_cost, p95_ms in result.result_rows
    ]
    return points, pareto_frontier(points)
