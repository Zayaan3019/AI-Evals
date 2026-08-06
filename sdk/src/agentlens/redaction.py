"""Redaction that runs inside the customer process, before export.

Two layers, both in-process:

1. **At capture time** (:mod:`agentlens.api`). Text is redacted *before* it is
   ever set as a span attribute, so raw PII does not exist in the span object,
   in a local console exporter, or in a crash dump.
2. **At export time** (:class:`agentlens.export.RedactingSpanProcessor`).
   Defence in depth for spans produced by third-party instrumentation that
   never went through our API.

Both layers run before the exporter serialises anything, so redacted content
never crosses the process boundary.

The default rule set is deliberately conservative about false positives where a
cheap structural check exists (card numbers are Luhn-validated). It is *not* a
compliance guarantee -- see ``KNOWN_LIMITATIONS``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable, Pattern, Sequence

#: Documented gaps in the default rule set. Stated here rather than omitted so
#: an operator can judge whether the defaults are sufficient for their data.
KNOWN_LIMITATIONS: tuple[str, ...] = (
    "Names, physical addresses, and dates of birth are not detected -- they have "
    "no reliable surface form and would require a model, not a regex.",
    "Phone matching requires a separator or country prefix, so an unformatted "
    "10-digit run is not treated as a phone number.",
    "National identifiers other than US SSN are not covered.",
    "Free-text PII embedded in a language the patterns do not anticipate "
    "(e.g. a credit card written in words) is not detected.",
)


def _luhn_ok(digits: str) -> bool:
    """Luhn checksum. Used to keep long digit runs from matching as cards."""
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        digit = ord(char) - 48
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _card_validator(match: re.Match[str]) -> bool:
    digits = re.sub(r"[^0-9]", "", match.group(0))
    return 13 <= len(digits) <= 19 and _luhn_ok(digits)


@dataclass(frozen=True, slots=True)
class RedactionRule:
    """A named pattern whose matches are replaced before export."""

    name: str
    pattern: Pattern[str]
    replacement: str
    #: Optional structural check. When it returns ``False`` the match is left
    #: alone. Used to suppress false positives (e.g. non-Luhn digit runs).
    validator: Callable[[re.Match[str]], bool] | None = None

    def apply(self, text: str) -> tuple[str, bool]:
        """Return ``(redacted_text, fired)``."""
        fired = False

        def _sub(match: re.Match[str]) -> str:
            nonlocal fired
            if self.validator is not None and not self.validator(match):
                return match.group(0)
            fired = True
            return self.replacement

        return self.pattern.sub(_sub, text), fired


def _rule(
    name: str,
    pattern: str,
    replacement: str,
    *,
    flags: int = 0,
    validator: Callable[[re.Match[str]], bool] | None = None,
) -> RedactionRule:
    return RedactionRule(
        name=name,
        pattern=re.compile(pattern, flags),
        replacement=replacement,
        validator=validator,
    )


# Order matters: structured secrets are matched before generic patterns that
# could partially consume them (e.g. a JWT before a bare token match).
DEFAULT_RULES: tuple[RedactionRule, ...] = (
    _rule(
        "private_key_block",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        "[REDACTED:private_key_block]",
        flags=re.DOTALL,
    ),
    _rule(
        "jwt",
        r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b",
        "[REDACTED:jwt]",
    ),
    _rule(
        "authorization_header",
        r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._~+/=-]{12,}",
        "[REDACTED:authorization_header]",
    ),
    _rule("aws_access_key_id", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", "[REDACTED:aws_access_key_id]"),
    _rule(
        "anthropic_api_key",
        r"\bsk-ant-[A-Za-z0-9_-]{16,}\b",
        "[REDACTED:anthropic_api_key]",
    ),
    _rule("openai_api_key", r"\bsk-[A-Za-z0-9]{20,}\b", "[REDACTED:openai_api_key]"),
    _rule("github_token", r"\bgh[pousr]_[A-Za-z0-9]{20,}\b", "[REDACTED:github_token]"),
    _rule(
        "credit_card",
        # The lookbehind is load-bearing. Without `(?<![\d.])` the digits after
        # a decimal point form a 13-19 digit run, and roughly 1 in 10 such runs
        # passes Luhn by chance -- measured at 9.49% over 20,000 random floats.
        # That silently corrupted retrieval scores, probabilities and latencies:
        # exactly the data this platform exists to measure. A card number is
        # never the fractional part of a decimal.
        r"(?<![\d.])(?:\d[ -]?){13,19}(?!\.\d)\b",
        "[REDACTED:credit_card]",
        validator=_card_validator,
    ),
    _rule("us_ssn", r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED:us_ssn]"),
    _rule(
        "email",
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "[REDACTED:email]",
    ),
    _rule(
        "phone",
        # Requires a country prefix or separators; a bare digit run is not
        # treated as a phone number (see KNOWN_LIMITATIONS).
        r"(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?|\b\d{2,4}[ .-])\d{3,4}[ .-]\d{3,4}\b",
        "[REDACTED:phone]",
    ),
    _rule(
        "ipv4",
        r"\b(?:(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)\b",
        "[REDACTED:ipv4]",
    ),
    _rule("ipv6", r"\b(?:[0-9A-Fa-f]{1,4}:){7}[0-9A-Fa-f]{1,4}\b", "[REDACTED:ipv6]"),
)


@dataclass(frozen=True, slots=True)
class RedactionOutcome:
    """Result of running the pipeline over one value."""

    text: str
    rules_fired: tuple[str, ...]

    @property
    def changed(self) -> bool:
        return bool(self.rules_fired)


class Redactor:
    """An ordered set of rules plus optional caller-supplied transforms.

    Instances are immutable; :meth:`with_rules` and :meth:`with_transform`
    return new instances so a shared default cannot be mutated by one caller.
    """

    __slots__ = ("_rules", "_transforms", "_max_length")

    def __init__(
        self,
        rules: Sequence[RedactionRule] = DEFAULT_RULES,
        transforms: Sequence[Callable[[str], str]] = (),
        *,
        max_length: int | None = None,
    ) -> None:
        self._rules = tuple(rules)
        self._transforms = tuple(transforms)
        self._max_length = max_length

    @property
    def rules(self) -> tuple[RedactionRule, ...]:
        return self._rules

    def with_rules(self, *rules: RedactionRule, prepend: bool = False) -> Redactor:
        """Return a new redactor with extra rules.

        ``prepend=True`` puts the new rules first, which matters when a custom
        rule must claim text a default rule would otherwise consume.
        """
        merged = (*rules, *self._rules) if prepend else (*self._rules, *rules)
        return Redactor(merged, self._transforms, max_length=self._max_length)

    def without_rules(self, *names: str) -> Redactor:
        drop = set(names)
        return Redactor(
            tuple(r for r in self._rules if r.name not in drop),
            self._transforms,
            max_length=self._max_length,
        )

    def with_transform(self, transform: Callable[[str], str]) -> Redactor:
        """Attach an arbitrary text transform, run after all regex rules."""
        return Redactor(
            self._rules, (*self._transforms, transform), max_length=self._max_length
        )

    def with_max_length(self, max_length: int | None) -> Redactor:
        return Redactor(self._rules, self._transforms, max_length=max_length)

    def redact(self, text: str) -> RedactionOutcome:
        fired: list[str] = []
        out = text
        for rule in self._rules:
            out, hit = rule.apply(out)
            if hit:
                fired.append(rule.name)
        for transform in self._transforms:
            transformed = transform(out)
            if transformed != out:
                fired.append("custom_transform")
                out = transformed
        if self._max_length is not None and len(out) > self._max_length:
            out = out[: self._max_length] + f"...[truncated {len(out)} chars]"
            fired.append("truncated")
        return RedactionOutcome(text=out, rules_fired=tuple(fired))

    def redact_value(self, value: object) -> tuple[object, tuple[str, ...]]:
        """Redact ``value`` if it is text; pass other primitives through.

        Sequences of strings are redacted element-wise, since OTel permits
        homogeneous sequence attributes.
        """
        if isinstance(value, str):
            outcome = self.redact(value)
            return outcome.text, outcome.rules_fired
        if isinstance(value, (list, tuple)) and any(
            isinstance(item, str) for item in value
        ):
            fired: list[str] = []
            items: list[object] = []
            for item in value:
                if isinstance(item, str):
                    outcome = self.redact(item)
                    items.append(outcome.text)
                    fired.extend(outcome.rules_fired)
                else:
                    items.append(item)
            return tuple(items), tuple(dict.fromkeys(fired))
        return value, ()


#: Shared default. Immutable, so importing it cannot let one caller's
#: customisation leak into another's.
DEFAULT_REDACTOR = Redactor()


def build_redactor(
    *,
    extra_rules: Iterable[RedactionRule] = (),
    disable: Iterable[str] = (),
    transforms: Iterable[Callable[[str], str]] = (),
    max_length: int | None = None,
) -> Redactor:
    """Convenience constructor for the common customisation cases."""
    redactor = DEFAULT_REDACTOR.without_rules(*disable)
    extras = tuple(extra_rules)
    if extras:
        redactor = redactor.with_rules(*extras)
    for transform in transforms:
        redactor = redactor.with_transform(transform)
    return redactor.with_max_length(max_length)


__all__ = [
    "DEFAULT_REDACTOR",
    "DEFAULT_RULES",
    "KNOWN_LIMITATIONS",
    "RedactionOutcome",
    "RedactionRule",
    "Redactor",
    "build_redactor",
]
