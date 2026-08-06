"""Cost computation: exactness, provenance, and honest absence."""

from __future__ import annotations

from decimal import Decimal

import pytest

from agentlens.pricing import (
    Cost,
    CostStatus,
    PricingTable,
    TokenUsage,
    default_table,
)


def test_table_loads_with_provenance() -> None:
    table = default_table()
    assert table.pricing_version == "2026-06-24"
    assert table.currency == "USD"
    # Numbers without a stated source are indistinguishable from invented ones.
    assert "anthropic" in table.provenance
    assert table.provenance["anthropic"]["source"]
    assert table.provenance["anthropic"]["retrieved"]


def test_cost_is_exact_integer_micro_units() -> None:
    """1M input tokens at $5.00/Mtok is exactly $5.00 = 5_000_000 micro-USD."""
    table = default_table()
    cost = table.compute(
        "anthropic", "claude-opus-5", TokenUsage(input_tokens=1_000_000)
    )
    assert cost.status is CostStatus.OK
    assert cost.micro_usd == 5_000_000
    assert isinstance(cost.micro_usd, int)


def test_input_and_output_are_priced_separately() -> None:
    table = default_table()
    # 1000 in @ $5/Mtok = 5000 micro; 500 out @ $25/Mtok = 12500 micro.
    cost = table.compute(
        "anthropic",
        "claude-opus-5",
        TokenUsage(input_tokens=1000, output_tokens=500),
    )
    assert cost.micro_usd == 5_000 + 12_500


def test_cache_tokens_are_priced_at_their_own_rates() -> None:
    table = default_table()
    # Cache reads on opus-5 are $0.50/Mtok -> 10_000 tokens = 5_000 micro.
    cost = table.compute(
        "anthropic",
        "claude-opus-5",
        TokenUsage(cache_read_input_tokens=10_000),
    )
    assert cost.micro_usd == 5_000
    # Cache writes (5m TTL) are $6.25/Mtok -> 10_000 tokens = 62_500 micro.
    write = table.compute(
        "anthropic",
        "claude-opus-5",
        TokenUsage(cache_creation_input_tokens=10_000),
    )
    assert write.micro_usd == 62_500


def test_cache_reads_are_cheaper_than_fresh_input() -> None:
    table = default_table()
    fresh = table.compute("anthropic", "claude-opus-5", TokenUsage(input_tokens=100_000))
    cached = table.compute(
        "anthropic", "claude-opus-5", TokenUsage(cache_read_input_tokens=100_000)
    )
    assert cached.micro_usd is not None and fresh.micro_usd is not None
    assert cached.micro_usd < fresh.micro_usd


def test_unknown_model_reports_no_pricing_rather_than_zero() -> None:
    """A missing price must not masquerade as a free call."""
    table = default_table()
    cost = table.compute(
        "anthropic", "some-model-that-does-not-exist", TokenUsage(input_tokens=1000)
    )
    assert cost.status is CostStatus.NO_PRICING
    assert cost.micro_usd is None


def test_missing_usage_reports_no_usage_rather_than_zero() -> None:
    table = default_table()
    cost = table.compute("anthropic", "claude-opus-5", TokenUsage())
    assert cost.status is CostStatus.NO_USAGE
    assert cost.micro_usd is None


def test_unknown_provider_reports_no_pricing() -> None:
    table = default_table()
    cost = table.compute("some-provider", "some-model", TokenUsage(input_tokens=10))
    assert cost.status is CostStatus.NO_PRICING


def test_dated_snapshot_resolves_to_base_model_and_records_the_match() -> None:
    table = default_table()
    cost = table.compute(
        "anthropic", "claude-haiku-4-5-20251001", TokenUsage(input_tokens=1_000_000)
    )
    assert cost.status is CostStatus.OK
    # Resolution is explicit, not implicit: the caller can see what was used.
    assert cost.matched_model == "claude-haiku-4-5"
    assert cost.micro_usd == 1_000_000  # $1.00/Mtok


def test_every_cost_carries_its_pricing_version() -> None:
    """Without this, a vendor price change silently rewrites history."""
    table = default_table()
    for usage in (TokenUsage(input_tokens=1), TokenUsage()):
        cost = table.compute("anthropic", "claude-opus-5", usage)
        assert cost.pricing_version == table.pricing_version


def test_float_rates_are_rejected() -> None:
    """Money must never round-trip through a binary float."""
    with pytest.raises(TypeError, match="float"):
        PricingTable.from_dict(
            {
                "pricing_version": "test",
                "models": {"p": {"m": {"input_per_mtok": 5.1, "output_per_mtok": "1"}}},
            }
        )


def test_custom_table_overrides_shipped_rates() -> None:
    """Negotiated or introductory rates are supported without a code change."""
    table = PricingTable.from_dict(
        {
            "pricing_version": "negotiated-2026-08",
            "currency": "USD",
            "models": {
                "anthropic": {
                    "claude-sonnet-5": {
                        "input_per_mtok": "2.00",
                        "output_per_mtok": "10.00",
                    }
                }
            },
        }
    )
    cost = table.compute(
        "anthropic", "claude-sonnet-5", TokenUsage(input_tokens=1_000_000)
    )
    assert cost.micro_usd == 2_000_000
    assert cost.pricing_version == "negotiated-2026-08"


def test_missing_pricing_version_raises_with_available_versions() -> None:
    with pytest.raises(FileNotFoundError, match="available"):
        PricingTable.load("1999-01-01")


def test_rates_parse_as_decimal_not_float() -> None:
    table = default_table()
    resolved = table.lookup("anthropic", "claude-opus-5")
    assert resolved is not None
    _, price = resolved
    assert isinstance(price.input_per_mtok, Decimal)
    assert price.input_per_mtok == Decimal("5.00")


def test_from_file_loads_a_negotiated_table_and_wires_into_config(tmp_path) -> None:
    """The README offers a custom table as the override for unpriced models.

    This exercises the whole promised path -- file on disk, through
    ``AgentLensConfig``, to a computed cost -- rather than just the loader, so
    the documented escape hatch is verified end to end.
    """
    import json

    from agentlens.config import AgentLensConfig

    path = tmp_path / "pricing_negotiated.json"
    path.write_text(
        json.dumps(
            {
                "pricing_version": "acme-2026-08",
                "currency": "USD",
                "models": {
                    "openai": {"gpt-4o": {"input_per_mtok": "1.25", "output_per_mtok": "5.00"}}
                },
                "provenance": {"openai": {"source": "acme enterprise agreement"}},
            }
        ),
        encoding="utf-8",
    )

    table = PricingTable.from_file(path)
    assert table.pricing_version == "acme-2026-08"
    assert table.provenance["openai"]["source"] == "acme enterprise agreement"

    cost = table.compute("openai", "gpt-4o", TokenUsage(input_tokens=1_000_000))
    assert cost.status is CostStatus.OK
    assert cost.micro_usd == 1_250_000

    resolved = AgentLensConfig(
        service_name="pricing-override", pricing_table=table
    ).resolved_pricing()
    assert resolved.pricing_version == "acme-2026-08"
