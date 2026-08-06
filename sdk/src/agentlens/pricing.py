"""Versioned pricing table and cost computation.

Money is never represented as a float. Rates are parsed as :class:`~decimal.Decimal`
and cost is emitted as an integer count of **micro-units** (millionths of a
currency unit). Because table rates are quoted per 1,000,000 tokens, the
conversion is exact:

    cost_usd    = tokens * rate_per_mtok / 1e6
    cost_micro  = cost_usd * 1e6
                = tokens * rate_per_mtok          <- no division, no rounding drift

Every computed cost carries the ``pricing_version`` it was derived from. A
vendor price change therefore produces a new version rather than silently
rewriting the meaning of historical cost comparisons.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal
from enum import Enum
from pathlib import Path
from typing import Any

_DATA_DIR = Path(__file__).parent / "data"

#: Pricing table shipped with the SDK. Selected explicitly (not "newest file
#: wins") so adding a file to the data directory cannot silently change costs.
DEFAULT_PRICING_VERSION = "2026-06-24"


class CostStatus(str, Enum):
    """Why a cost value is present, absent, or partial."""

    OK = "ok"
    #: The model is not in the pricing table. Tokens are still recorded.
    NO_PRICING = "no_pricing"
    #: No token counts were supplied, so no cost can be computed.
    NO_USAGE = "no_usage"


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts for a single model call."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    reasoning_output_tokens: int | None = None

    def is_empty(self) -> bool:
        return (
            self.input_tokens is None
            and self.output_tokens is None
            and self.cache_read_input_tokens is None
            and self.cache_creation_input_tokens is None
        )


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Rates for one model, in currency units per 1,000,000 tokens."""

    input_per_mtok: Decimal
    output_per_mtok: Decimal
    cache_read_per_mtok: Decimal | None = None
    cache_write_5m_per_mtok: Decimal | None = None
    cache_write_1h_per_mtok: Decimal | None = None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class Cost:
    """Result of a cost computation."""

    micro_usd: int | None
    currency: str
    pricing_version: str
    status: CostStatus
    #: Table key the lookup resolved to. Differs from the requested model when a
    #: dated snapshot resolved to its base model; ``None`` when unmatched.
    matched_model: str | None = None


def _to_decimal(value: Any) -> Decimal:
    # Rates are stored as JSON strings precisely so they never pass through a
    # binary float. Accept int for convenience but never float.
    if isinstance(value, float):
        raise TypeError(
            "pricing rates must be strings or ints, not float "
            "(floats cannot represent decimal rates exactly)"
        )
    return Decimal(str(value))


class PricingTable:
    """An immutable, versioned set of model rates."""

    def __init__(
        self,
        *,
        pricing_version: str,
        currency: str,
        models: dict[str, dict[str, ModelPrice]],
        provenance: dict[str, Any] | None = None,
    ) -> None:
        self._version = pricing_version
        self._currency = currency
        self._models = models
        self._provenance = provenance or {}

    @property
    def pricing_version(self) -> str:
        return self._version

    @property
    def currency(self) -> str:
        return self._currency

    @property
    def provenance(self) -> dict[str, Any]:
        """Where these numbers came from, per provider."""
        return self._provenance

    # -- loading ----------------------------------------------------------

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PricingTable:
        models: dict[str, dict[str, ModelPrice]] = {}
        for provider, entries in raw.get("models", {}).items():
            models[provider] = {
                name: ModelPrice(
                    input_per_mtok=_to_decimal(spec["input_per_mtok"]),
                    output_per_mtok=_to_decimal(spec["output_per_mtok"]),
                    cache_read_per_mtok=(
                        _to_decimal(spec["cache_read_per_mtok"])
                        if "cache_read_per_mtok" in spec
                        else None
                    ),
                    cache_write_5m_per_mtok=(
                        _to_decimal(spec["cache_write_5m_per_mtok"])
                        if "cache_write_5m_per_mtok" in spec
                        else None
                    ),
                    cache_write_1h_per_mtok=(
                        _to_decimal(spec["cache_write_1h_per_mtok"])
                        if "cache_write_1h_per_mtok" in spec
                        else None
                    ),
                    note=spec.get("note"),
                )
                for name, spec in entries.items()
            }
        return cls(
            pricing_version=raw["pricing_version"],
            currency=raw.get("currency", "USD"),
            models=models,
            provenance=raw.get("provenance"),
        )

    @classmethod
    def load(cls, version: str = DEFAULT_PRICING_VERSION) -> PricingTable:
        path = _DATA_DIR / f"pricing_{version}.json"
        if not path.is_file():
            available = sorted(
                p.stem.removeprefix("pricing_") for p in _DATA_DIR.glob("pricing_*.json")
            )
            raise FileNotFoundError(
                f"no pricing table for version {version!r}; available: {available}"
            )
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    @classmethod
    def from_file(cls, path: str | Path) -> PricingTable:
        """Load a caller-supplied pricing table (e.g. negotiated rates)."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # -- lookup -----------------------------------------------------------

    def lookup(self, provider: str, model: str) -> tuple[str, ModelPrice] | None:
        """Resolve ``model`` within ``provider``.

        Exact match wins. Otherwise the longest table key that ``model`` starts
        with is used, so a dated snapshot (``claude-haiku-4-5-20251001``)
        resolves to its base entry (``claude-haiku-4-5``). The key actually used
        is returned so the resolution is auditable rather than implicit.
        """
        entries = self._models.get(provider)
        if not entries:
            return None
        exact = entries.get(model)
        if exact is not None:
            return model, exact
        candidates = [k for k in entries if model.startswith(k)]
        if not candidates:
            return None
        best = max(candidates, key=len)
        return best, entries[best]

    # -- computation ------------------------------------------------------

    def compute(self, provider: str, model: str, usage: TokenUsage) -> Cost:
        """Compute cost in integer micro-units of :attr:`currency`."""
        if usage.is_empty():
            return Cost(
                micro_usd=None,
                currency=self._currency,
                pricing_version=self._version,
                status=CostStatus.NO_USAGE,
            )

        resolved = self.lookup(provider, model)
        if resolved is None:
            return Cost(
                micro_usd=None,
                currency=self._currency,
                pricing_version=self._version,
                status=CostStatus.NO_PRICING,
            )

        matched_model, price = resolved
        total = Decimal(0)

        # Rates are per 1e6 tokens and the output unit is 1e-6 currency units,
        # so tokens * rate is already the micro-unit cost.
        if usage.input_tokens:
            total += Decimal(usage.input_tokens) * price.input_per_mtok
        if usage.output_tokens:
            total += Decimal(usage.output_tokens) * price.output_per_mtok
        if usage.cache_read_input_tokens and price.cache_read_per_mtok is not None:
            total += (
                Decimal(usage.cache_read_input_tokens) * price.cache_read_per_mtok
            )
        if (
            usage.cache_creation_input_tokens
            and price.cache_write_5m_per_mtok is not None
        ):
            # Default to the 5-minute TTL rate: it is the API default, and
            # assuming the cheaper of the two write rates keeps the figure a
            # lower bound rather than an inflated guess.
            total += (
                Decimal(usage.cache_creation_input_tokens)
                * price.cache_write_5m_per_mtok
            )

        micro = int(total.quantize(Decimal(1), rounding=ROUND_HALF_EVEN))
        return Cost(
            micro_usd=micro,
            currency=self._currency,
            pricing_version=self._version,
            status=CostStatus.OK,
            matched_model=matched_model,
        )


@dataclass(slots=True)
class _Cache:
    table: PricingTable | None = field(default=None)


_default_cache = _Cache()


def default_table() -> PricingTable:
    """Return (and memoise) the pricing table shipped with the SDK."""
    if _default_cache.table is None:
        _default_cache.table = PricingTable.load()
    return _default_cache.table
