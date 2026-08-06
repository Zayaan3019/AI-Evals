"""Evaluator registry: construction, execution, and verified claims.

Two responsibilities beyond a name lookup, both of which exist because the
alternative silently corrupts results:

**Exception containment.** :func:`run_evaluator` catches everything an
evaluator can throw and converts it to ``Score.failure``. Evaluator authors --
including us -- otherwise end up writing ``try/except: return 0.0``, which
reports a bug in the evaluator as a failure of the system under test. Those two
things must never be the same number.

**Determinism is verified, not declared.** An evaluator claiming the
``deterministic`` tier is checked by running it twice on the same input. A
mislabelled tier is not cosmetic: the regression gate treats deterministic
scores as directly comparable across runs, so a non-deterministic evaluator
wearing that label manufactures regressions out of noise.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Sequence

from .base import EvalInput, Evaluator, Score, Tier
from .deterministic import BUILTIN_DETERMINISTIC
from .human import HumanLabelEvaluator
from .model_graded import BUILTIN_MODEL_GRADED
from .sandbox import CustomEvaluator
from .statistical import BUILTIN_STATISTICAL

logger = logging.getLogger(__name__)


class UnknownEvaluator(KeyError):
    """No evaluator is registered under that slug."""


class NonDeterministicEvaluator(AssertionError):
    """An evaluator claiming the deterministic tier did not behave like one."""


#: Every built-in, keyed by slug. Assembled from the per-tier tables so adding
#: an evaluator in one place makes it constructible everywhere.
BUILTIN_EVALUATORS: dict[str, type[Evaluator]] = {
    **BUILTIN_DETERMINISTIC,
    **BUILTIN_STATISTICAL,
    **BUILTIN_MODEL_GRADED,
    HumanLabelEvaluator.slug: HumanLabelEvaluator,
    CustomEvaluator.slug: CustomEvaluator,
}


def _assert_no_slug_collisions() -> None:
    """Two evaluators sharing a slug would silently shadow one another.

    Checked at import so the failure is immediate and total, rather than
    appearing as a mysteriously wrong evaluator on one run.
    """
    tables: list[Mapping[str, type[Evaluator]]] = [
        BUILTIN_DETERMINISTIC,
        BUILTIN_STATISTICAL,
        BUILTIN_MODEL_GRADED,
        {HumanLabelEvaluator.slug: HumanLabelEvaluator},
        {CustomEvaluator.slug: CustomEvaluator},
    ]
    seen: dict[str, str] = {}
    for table in tables:
        for slug, cls in table.items():
            if slug in seen:
                raise RuntimeError(
                    f"evaluator slug {slug!r} is defined twice: "
                    f"{seen[slug]} and {cls.__module__}.{cls.__name__}"
                )
            seen[slug] = f"{cls.__module__}.{cls.__name__}"


_assert_no_slug_collisions()


def build_evaluator(slug: str, **config: Any) -> Evaluator:
    """Construct a registered evaluator by slug.

    Configuration errors surface here -- at registration -- rather than on the
    first case of a long run.
    """
    try:
        cls = BUILTIN_EVALUATORS[slug]
    except KeyError as exc:
        raise UnknownEvaluator(
            f"unknown evaluator {slug!r}; available: {sorted(BUILTIN_EVALUATORS)}"
        ) from exc
    return cls(**config)


def run_evaluator(evaluator: Evaluator, item: EvalInput) -> Score:
    """Score one item, converting any exception into a recorded failure.

    This is the only place evaluators should be invoked from. Calling
    ``evaluator.evaluate`` directly bypasses containment.
    """
    try:
        score = evaluator.evaluate(item)
    except Exception as exc:  # noqa: BLE001 - containment is the purpose
        logger.warning("evaluator %s raised", evaluator.identity, exc_info=True)
        return Score.failure(f"{type(exc).__name__}: {exc}")

    if not isinstance(score, Score):
        return Score.failure(
            f"{evaluator.identity} returned {type(score).__name__}, expected Score"
        )
    if score.value is not None and not 0.0 <= score.value <= 1.0:
        return Score.failure(f"{evaluator.identity} returned out-of-range score {score.value}")
    return score


def run_suite(
    evaluators: Sequence[Evaluator], item: EvalInput
) -> dict[str, Score]:
    """Run several evaluators over one item.

    Keyed by :attr:`Evaluator.identity` (slug@version) rather than slug alone,
    so results from two versions of the same evaluator cannot collide in the
    same dict and silently overwrite each other.
    """
    results: dict[str, Score] = {}
    for evaluator in evaluators:
        results[evaluator.identity] = run_evaluator(evaluator, item)
    return results


def assert_deterministic(evaluator: Evaluator, items: Iterable[EvalInput]) -> None:
    """Verify a deterministic-tier claim by running twice.

    Two runs agreeing does not *prove* determinism, and this does not claim to.
    It reliably catches the realistic ways the claim goes wrong -- a clock, a
    ``set`` iteration order leaking into output, an unseeded sample, a network
    call -- which is what the tier label needs to survive.
    """
    if evaluator.tier not in (Tier.DETERMINISTIC, Tier.STATISTICAL):
        raise NonDeterministicEvaluator(
            f"{evaluator.identity} is tier {evaluator.tier.value}, which is not "
            "expected to be deterministic"
        )

    for item in items:
        first = run_evaluator(evaluator, item)
        second = run_evaluator(evaluator, item)
        if (first.value, first.passed, first.error) != (second.value, second.passed, second.error):
            raise NonDeterministicEvaluator(
                f"{evaluator.identity} is registered as {evaluator.tier.value} but "
                f"returned {first.value!r}/{first.error!r} then "
                f"{second.value!r}/{second.error!r} for the same input"
            )


def describe_registry() -> list[dict[str, Any]]:
    """Every registered evaluator, grouped for display and API responses.

    Constructed with no arguments where possible; evaluators with required
    configuration report their requirement instead of being omitted, so the
    catalogue is complete rather than convenient.
    """
    catalogue: list[dict[str, Any]] = []
    for slug, cls in sorted(BUILTIN_EVALUATORS.items()):
        entry: dict[str, Any] = {
            "slug": slug,
            "tier": cls.tier.value,
            "version": cls.version,
            "doc": (cls.__doc__ or "").strip().split("\n")[0],
        }
        try:
            entry["default_config"] = cls().describe()["config"]  # type: ignore[call-arg]
            entry["requires_config"] = False
        except TypeError as exc:
            entry["default_config"] = None
            entry["requires_config"] = True
            entry["config_error"] = str(exc)
        catalogue.append(entry)
    return catalogue
