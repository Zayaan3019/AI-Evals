"""Drift detection and Pareto analysis.

The distribution maths is tested as pure functions against cases with known
answers. The query layer is tested against real ClickHouse.
"""

from __future__ import annotations

import random
import uuid

import pytest

from agentlens_platform.drift import (
    ModelPoint,
    compare_distributions,
    detect_drift,
    ks_statistic,
    model_pareto,
    pareto_frontier,
    population_stability_index,
    sample_metric,
)
from agentlens_platform.queries import TimeWindow

from .test_mapper import build_resource_spans
from .test_storage import _insert, _now_ns


# ---------------------------------------------------------------------------
# PSI and KS
# ---------------------------------------------------------------------------


def test_identical_distributions_have_no_drift() -> None:
    rng = random.Random(1)
    sample = [rng.gauss(100, 15) for _ in range(2000)]
    psi, _ = population_stability_index(sample, sample)
    assert psi == pytest.approx(0.0, abs=1e-9)
    assert ks_statistic(sample, sample) == pytest.approx(0.0)


def test_resampling_the_same_distribution_stays_below_the_stable_threshold() -> None:
    """The false-positive check: a stable system must not raise an alert."""
    rng = random.Random(2)
    reference = [rng.gauss(100, 15) for _ in range(5000)]
    current = [rng.gauss(100, 15) for _ in range(5000)]
    result = compare_distributions("latency_ms", reference, current)
    assert result.severity == "stable"
    assert result.psi < 0.10


def test_a_shifted_distribution_is_flagged() -> None:
    rng = random.Random(3)
    reference = [rng.gauss(100, 15) for _ in range(5000)]
    current = [rng.gauss(160, 15) for _ in range(5000)]
    result = compare_distributions("latency_ms", reference, current)
    assert result.severity == "significant"
    assert result.psi > 0.25
    assert result.ks_statistic > 0.5


def test_a_bimodal_split_with_an_unchanged_mean_is_still_detected() -> None:
    """The failure a mean cannot see.

    Half the traffic degrades badly and half improves slightly, leaving the
    mean almost unmoved. This is exactly the shape a mean-based monitor misses
    and the reason drift is measured on distributions.
    """
    rng = random.Random(4)
    reference = [rng.gauss(100, 5) for _ in range(4000)]
    current = [rng.gauss(40, 5) if i % 2 else rng.gauss(160, 5) for i in range(4000)]

    reference_mean = sum(reference) / len(reference)
    current_mean = sum(current) / len(current)
    assert abs(reference_mean - current_mean) < 3  # a mean monitor sees nothing

    result = compare_distributions("latency_ms", reference, current)
    assert result.severity == "significant"


def test_psi_is_finite_when_a_bucket_empties_entirely() -> None:
    """The classic way PSI blows up to infinity."""
    reference = [float(i) for i in range(1000)]
    current = [999.0] * 1000  # every point in the top bucket
    psi, _ = population_stability_index(reference, current)
    assert psi != float("inf")
    assert psi > 0.25


def test_heavily_tied_values_do_not_create_zero_width_buckets() -> None:
    """A score that is nearly always 1.0 produces repeated quantile edges."""
    reference = [1.0] * 900 + [0.0] * 100
    current = [1.0] * 500 + [0.0] * 500
    psi, bucket_count = population_stability_index(reference, current)
    assert bucket_count <= 3  # deduplicated edges
    assert psi > 0.25
    assert math_is_finite(psi)


def math_is_finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


def test_small_samples_are_reported_as_underpowered() -> None:
    """PSI on tiny samples is dominated by luck; the caller must be told."""
    rng = random.Random(5)
    result = compare_distributions(
        "latency_ms", [rng.gauss(100, 15) for _ in range(30)], [rng.gauss(100, 15) for _ in range(30)]
    )
    assert result.is_underpowered
    assert "UNDERPOWERED" in result.render()


def test_large_samples_are_not_flagged_underpowered() -> None:
    rng = random.Random(6)
    result = compare_distributions(
        "latency_ms", [rng.gauss(100, 15) for _ in range(500)], [rng.gauss(100, 15) for _ in range(500)]
    )
    assert not result.is_underpowered


def test_ks_detects_a_shift_inside_a_single_psi_bucket() -> None:
    """Why both statistics are reported rather than one."""
    reference = [0.0] * 500 + [1.0] * 500
    current = [0.0] * 250 + [1.0] * 750
    assert ks_statistic(reference, current) > 0.2


def test_empty_samples_are_rejected() -> None:
    with pytest.raises(ValueError):
        population_stability_index([], [1.0])
    with pytest.raises(ValueError):
        ks_statistic([1.0], [])


# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------


def _point(label, quality, cost, latency, n=100) -> ModelPoint:
    return ModelPoint(label, quality, cost, latency, n)


def test_a_strictly_worse_model_is_dominated() -> None:
    good = _point("good", 0.95, 100.0, 500.0)
    bad = _point("bad", 0.90, 200.0, 800.0)
    assert good.dominates(bad)
    assert not bad.dominates(good)


def test_a_tradeoff_is_not_dominance() -> None:
    """Cheaper but worse is a choice, not an inferior option."""
    accurate = _point("accurate", 0.95, 500.0, 900.0)
    cheap = _point("cheap", 0.85, 50.0, 200.0)
    assert not accurate.dominates(cheap)
    assert not cheap.dominates(accurate)
    frontier = pareto_frontier([accurate, cheap])
    assert {p.label for p in frontier} == {"accurate", "cheap"}


def test_frontier_excludes_dominated_points() -> None:
    points = [
        _point("best_quality", 0.98, 800.0, 1200.0),
        _point("balanced", 0.93, 200.0, 400.0),
        _point("cheapest", 0.80, 20.0, 150.0),
        _point("pointless", 0.85, 900.0, 1500.0),  # worse than balanced on all axes
    ]
    frontier = {p.label for p in pareto_frontier(points)}
    assert "pointless" not in frontier
    assert frontier == {"best_quality", "balanced", "cheapest"}


def test_frontier_is_ordered_by_quality() -> None:
    points = [_point("a", 0.7, 10.0, 10.0), _point("b", 0.9, 50.0, 50.0)]
    assert [p.label for p in pareto_frontier(points)] == ["b", "a"]


def test_identical_points_do_not_eliminate_each_other() -> None:
    """Equal on every axis is not dominance; both remain valid choices."""
    a = _point("a", 0.9, 100.0, 200.0)
    b = _point("b", 0.9, 100.0, 200.0)
    assert not a.dominates(b)
    assert len(pareto_frontier([a, b])) == 2


def test_a_single_point_is_its_own_frontier() -> None:
    assert len(pareto_frontier([_point("only", 0.5, 1.0, 1.0)])) == 1


def test_empty_input_yields_an_empty_frontier() -> None:
    assert pareto_frontier([]) == []


# ---------------------------------------------------------------------------
# Against real ClickHouse
# ---------------------------------------------------------------------------


@pytest.fixture()
def drift_tenant() -> str:
    return f"drift{uuid.uuid4().hex[:10]}"


def _emit(ch_client, database, tenant, *, count, duration_ms, model="claude-opus-5",
          offset_seconds, errors=0):
    for index in range(count):
        start = _now_ns(-offset_seconds)
        _insert(
            ch_client,
            database,
            tenant,
            build_resource_spans(
                trace_id=uuid.uuid4().bytes,
                span_id=uuid.uuid4().bytes[:8],
                start_ns=start,
                end_ns=start + int(duration_ms * 1e6),
                status_code=2 if index < errors else 1,  # 2 = ERROR, 1 = OK
                attributes={
                    "gen_ai.operation.name": "chat",
                    "gen_ai.request.model": model,
                    "gen_ai.usage.input_tokens": 100,
                    "gen_ai.usage.output_tokens": 20,
                    "agentlens.cost.micro_usd": 500,
                },
            ),
        )


def test_drift_detected_between_two_real_windows(ch_client, clickhouse_database, drift_tenant):
    # Reference window: ~100ms. Current window: ~400ms.
    for index in range(150):
        _emit(ch_client, clickhouse_database, drift_tenant, count=1,
              duration_ms=100 + (index % 20), offset_seconds=7200)
    for index in range(150):
        _emit(ch_client, clickhouse_database, drift_tenant, count=1,
              duration_ms=400 + (index % 20), offset_seconds=60)

    reference = TimeWindow.last(hours=3)
    current = TimeWindow.last(minutes=10)
    result = detect_drift(
        ch_client,
        database=clickhouse_database,
        tenant_slug=drift_tenant,
        reference_window=TimeWindow(start=reference.start, end=current.start),
        current_window=current,
        metric="latency_ms",
    )
    assert result is not None
    assert result.current_mean > result.reference_mean * 3
    assert result.severity == "significant"


def test_no_data_returns_none_rather_than_zero_drift(ch_client, clickhouse_database, drift_tenant):
    """Absence of data is not absence of drift."""
    result = detect_drift(
        ch_client,
        database=clickhouse_database,
        tenant_slug=drift_tenant,
        reference_window=TimeWindow.last(hours=2),
        current_window=TimeWindow.last(minutes=5),
    )
    assert result is None


def test_sampling_is_deterministic(ch_client, clickhouse_database, drift_tenant):
    """Two investigators looking at the same alert must see the same numbers."""
    _emit(ch_client, clickhouse_database, drift_tenant, count=50, duration_ms=120, offset_seconds=60)
    window = TimeWindow.last(hours=1)
    first = sample_metric(ch_client, database=clickhouse_database, tenant_slug=drift_tenant,
                          window=window, metric="latency_ms", limit=20)
    second = sample_metric(ch_client, database=clickhouse_database, tenant_slug=drift_tenant,
                           window=window, metric="latency_ms", limit=20)
    assert first == second
    assert len(first) == 20


def test_unknown_metric_is_rejected(ch_client, clickhouse_database, drift_tenant):
    with pytest.raises(ValueError, match="unknown drift metric"):
        sample_metric(ch_client, database=clickhouse_database, tenant_slug=drift_tenant,
                      window=TimeWindow.last(hours=1), metric="vibes")


def test_pareto_over_real_spans(ch_client, clickhouse_database, drift_tenant):
    # Fast, cheap, and reliable.
    _emit(ch_client, clickhouse_database, drift_tenant, count=40, duration_ms=100,
          model="fast-model", offset_seconds=60)
    # Slow and unreliable -- strictly worse on quality and latency at equal cost.
    _emit(ch_client, clickhouse_database, drift_tenant, count=40, duration_ms=900,
          model="slow-model", offset_seconds=60, errors=10)

    points, frontier = model_pareto(
        ch_client, database=clickhouse_database, tenant_slug=drift_tenant,
        window=TimeWindow.last(hours=1),
    )
    labels = {p.label for p in points}
    assert {"fast-model", "slow-model"} <= labels

    fast = next(p for p in points if p.label == "fast-model")
    slow = next(p for p in points if p.label == "slow-model")
    assert fast.quality == pytest.approx(1.0)
    assert slow.quality == pytest.approx(0.75)
    assert fast.dominates(slow)
    assert "slow-model" not in {p.label for p in frontier}


def test_ks_is_exactly_zero_for_tied_identical_samples() -> None:
    """Regression: advancing one CDF at a time left a transient 1/n gap.

    Every bounded score distribution is heavily tied, so a KS that reports 1/n
    for two identical samples would add a floor of noise to every drift check
    and make small-sample comparisons look like real movement.
    """
    tied = [1.0] * 500 + [0.0] * 500
    assert ks_statistic(tied, tied) == 0.0
    assert ks_statistic([5.0] * 100, [5.0] * 100) == 0.0
    assert ks_statistic([1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 2.0, 3.0]) == 0.0


def test_ks_still_detects_a_real_shift_in_tied_data() -> None:
    """The fix must not cost sensitivity."""
    reference = [0.0] * 500 + [1.0] * 500
    current = [0.0] * 250 + [1.0] * 750
    assert ks_statistic(reference, current) == pytest.approx(0.25)
