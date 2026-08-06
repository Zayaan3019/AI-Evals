"""Tier 1: deterministic evaluators.

Pure functions of their input. No network, no model, no clock, no randomness.
These are the only evaluators whose scores are comparable across arbitrary
spans of time without qualification, which makes them the backbone of a
regression gate.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import DeterministicEvaluator, EvalInput, Score


class ExactMatch(DeterministicEvaluator):
    """Output equals expected, after optional normalisation."""

    slug = "exact_match"
    version = "1"

    def __init__(self, *, case_sensitive: bool = False, strip: bool = True) -> None:
        super().__init__(case_sensitive=case_sensitive, strip=strip)

    def _normalise(self, text: str) -> str:
        if self.config["strip"]:
            text = text.strip()
        if not self.config["case_sensitive"]:
            text = text.casefold()
        return text

    def evaluate(self, item: EvalInput) -> Score:
        if item.expected is None:
            # Not a zero: the case simply cannot be scored by this evaluator.
            # Returning 0.0 would report a passing system as failing.
            return Score.failure("exact_match requires an expected output")
        hit = self._normalise(item.output) == self._normalise(item.expected)
        return Score(
            value=1.0 if hit else 0.0,
            passed=hit,
            explanation=None if hit else f"expected {item.expected!r}, got {item.output!r}",
        )


class Contains(DeterministicEvaluator):
    """Every required substring appears in the output.

    Scores the *fraction* present rather than all-or-nothing, so a partial
    answer is distinguishable from a completely wrong one -- which matters when
    you are trying to tell a regression from a rewrite.
    """

    slug = "contains"
    version = "1"

    def __init__(self, *, substrings: list[str], case_sensitive: bool = False) -> None:
        if not substrings:
            raise ValueError("contains requires at least one substring")
        super().__init__(substrings=list(substrings), case_sensitive=case_sensitive)

    def evaluate(self, item: EvalInput) -> Score:
        haystack = item.output if self.config["case_sensitive"] else item.output.casefold()
        needles = [
            s if self.config["case_sensitive"] else s.casefold()
            for s in self.config["substrings"]
        ]
        found = [n for n in needles if n in haystack]
        missing = [n for n in needles if n not in haystack]
        value = len(found) / len(needles)
        return Score(
            value=value,
            passed=not missing,
            explanation=None if not missing else f"missing: {missing}",
        )


class RegexMatch(DeterministicEvaluator):
    """The output matches a pattern.

    The pattern is compiled at construction so a bad regex fails when the
    evaluator is registered, not silently on every case at run time.
    """

    slug = "regex_match"
    version = "1"

    def __init__(self, *, pattern: str, flags: int = 0) -> None:
        self._compiled = re.compile(pattern, flags)
        super().__init__(pattern=pattern, flags=flags)

    def evaluate(self, item: EvalInput) -> Score:
        hit = self._compiled.search(item.output) is not None
        return Score(
            value=1.0 if hit else 0.0,
            passed=hit,
            explanation=None if hit else f"no match for {self.config['pattern']!r}",
        )


class ValidJson(DeterministicEvaluator):
    """The output parses as JSON, optionally with required top-level keys.

    A structured-output contract that silently degrades to prose is one of the
    most common agent regressions, and it is free to detect.
    """

    slug = "valid_json"
    version = "1"

    def __init__(self, *, required_keys: list[str] | None = None) -> None:
        super().__init__(required_keys=list(required_keys or []))

    def evaluate(self, item: EvalInput) -> Score:
        try:
            parsed = json.loads(item.output)
        except (json.JSONDecodeError, TypeError) as exc:
            return Score(value=0.0, passed=False, explanation=f"not valid JSON: {exc}")

        required = self.config["required_keys"]
        if not required:
            return Score(value=1.0, passed=True)
        if not isinstance(parsed, dict):
            return Score(
                value=0.0, passed=False, explanation=f"expected an object, got {type(parsed).__name__}"
            )
        missing = [key for key in required if key not in parsed]
        value = (len(required) - len(missing)) / len(required)
        return Score(
            value=value,
            passed=not missing,
            explanation=None if not missing else f"missing keys: {missing}",
        )


class NoPiiLeak(DeterministicEvaluator):
    """The output contains no PII the redaction layer should have caught.

    Reuses the SDK's redactor rather than defining a second set of patterns.
    Two independently-maintained PII definitions drift apart, and the one in the
    evaluator would quietly stop matching what the SDK actually redacts.
    """

    slug = "no_pii_leak"
    version = "1"

    def __init__(self, *, rules: list[str] | None = None) -> None:
        from agentlens.redaction import DEFAULT_REDACTOR

        self._redactor = DEFAULT_REDACTOR
        if rules:
            keep = set(rules)
            self._redactor = DEFAULT_REDACTOR.without_rules(
                *[r.name for r in DEFAULT_REDACTOR.rules if r.name not in keep]
            )
        super().__init__(rules=list(rules or []))

    def evaluate(self, item: EvalInput) -> Score:
        outcome = self._redactor.redact(item.output)
        # 'truncated' is a length policy, not a PII finding.
        fired = [rule for rule in outcome.rules_fired if rule != "truncated"]
        clean = not fired
        return Score(
            value=1.0 if clean else 0.0,
            passed=clean,
            explanation=None if clean else f"PII detected: {sorted(set(fired))}",
        )


class OutputLength(DeterministicEvaluator):
    """Output length falls within bounds.

    Scores 1.0 inside the range and degrades linearly outside it rather than
    dropping to 0, so "slightly too long" and "ten times too long" are
    distinguishable.
    """

    slug = "output_length"
    version = "1"

    def __init__(self, *, min_chars: int = 0, max_chars: int | None = None) -> None:
        if max_chars is not None and max_chars < min_chars:
            raise ValueError("max_chars must be >= min_chars")
        super().__init__(min_chars=min_chars, max_chars=max_chars)

    def evaluate(self, item: EvalInput) -> Score:
        length = len(item.output)
        low = self.config["min_chars"]
        high = self.config["max_chars"]

        if length < low:
            value = length / low if low else 1.0
            return Score(value=value, passed=False, explanation=f"{length} chars < min {low}")
        if high is not None and length > high:
            value = max(0.0, high / length)
            return Score(value=value, passed=False, explanation=f"{length} chars > max {high}")
        return Score(value=1.0, passed=True)


class ToolWasCalled(DeterministicEvaluator):
    """A required tool appears in the trace's recorded tool calls.

    Reads ``metadata['tools_called']``, which the eval runner populates from the
    trace. Checking the agent's *behaviour* rather than only its prose is the
    difference between evaluating an agent and evaluating a chatbot.
    """

    slug = "tool_was_called"
    version = "1"

    def __init__(self, *, tool_name: str, forbidden: bool = False) -> None:
        super().__init__(tool_name=tool_name, forbidden=forbidden)

    def evaluate(self, item: EvalInput) -> Score:
        called = item.metadata.get("tools_called")
        if called is None:
            return Score.failure("no tools_called recorded for this case")
        present = self.config["tool_name"] in set(called)
        want_absent = self.config["forbidden"]
        ok = (not present) if want_absent else present
        verb = "must not call" if want_absent else "must call"
        return Score(
            value=1.0 if ok else 0.0,
            passed=ok,
            explanation=None if ok else f"{verb} {self.config['tool_name']!r}; called {sorted(called)}",
        )


#: Constructors exposed to configuration. Keyed by slug so a stored evaluator
#: definition can be rebuilt from (slug, config) alone.
BUILTIN_DETERMINISTIC: dict[str, type[DeterministicEvaluator]] = {
    cls.slug: cls
    for cls in (
        ExactMatch,
        Contains,
        RegexMatch,
        ValidJson,
        NoPiiLeak,
        OutputLength,
        ToolWasCalled,
    )
}
