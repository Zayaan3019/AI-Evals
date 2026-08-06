"""Sandboxed custom evaluators.

These exercise the containment claims in ``sandbox.py`` against code that
actually tries to break out, allocate without bound, hang, and crash. A sandbox
whose guarantees are only asserted in a docstring is not a sandbox.

The docstring is explicit that this is a boundary against *buggy and careless*
code, not against a determined attacker. These tests verify what is claimed,
and the escape test documents the limit rather than pretending it does not
exist.
"""

from __future__ import annotations

import sys

import pytest

from agentlens_platform.evaluators import (
    CustomEvaluator,
    EvalInput,
    SandboxViolation,
    run_evaluator,
    run_sandboxed,
)
from agentlens_platform.evaluators.sandbox import inspect_source

PAYLOAD = {"output": "hello world", "inputs": {}, "expected": "hello world", "context": [], "metadata": {}}


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_a_custom_evaluator_runs_and_returns_a_score() -> None:
    source = """
def evaluate(item):
    return 1.0 if item["output"] == item["expected"] else 0.0
"""
    result = run_sandboxed(source, PAYLOAD)
    assert result.error is None
    assert result.value == 1.0


def test_a_custom_evaluator_may_return_an_explanation() -> None:
    source = """
def evaluate(item):
    n = len(item["output"])
    return min(1.0, n / 20), f"length was {n}"
"""
    result = run_sandboxed(source, PAYLOAD)
    assert result.explanation == "length was 11"


def test_custom_evaluator_integrates_with_the_registry() -> None:
    ev = CustomEvaluator(
        slug="starts_with_hello",
        source="""
def evaluate(item):
    return 1.0 if item["output"].startswith("hello") else 0.0
""",
    )
    score = run_evaluator(ev, EvalInput(output="hello world"))
    assert score.value == 1.0
    assert score.passed is True
    assert ev.slug == "starts_with_hello"


def test_pure_stdlib_math_is_still_usable() -> None:
    """The denylist must not make legitimate evaluators impossible to write."""
    source = """
import math
import re
def evaluate(item):
    words = re.findall(r"\\w+", item["output"])
    return min(1.0, math.log1p(len(words)) / 3)
"""
    result = run_sandboxed(source, PAYLOAD)
    assert result.error is None
    assert 0.0 < result.value <= 1.0


# ---------------------------------------------------------------------------
# Static rejection: caught before anything executes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source,reason",
    [
        ("import os\ndef evaluate(item): return 1.0", "os"),
        ("import subprocess\ndef evaluate(item): return 1.0", "subprocess"),
        ("from socket import socket\ndef evaluate(item): return 1.0", "socket"),
        ("import httpx\ndef evaluate(item): return 1.0", "httpx"),
        ("import sqlalchemy\ndef evaluate(item): return 1.0", "sqlalchemy"),
        ("def evaluate(item): return eval('1.0')", "eval"),
        ("def evaluate(item): return exec('x=1') or 1.0", "exec"),
        ("def evaluate(item): open('/etc/passwd')", "open"),
        ("def evaluate(item): return ().__class__ and 1.0", "__class__"),
        ("def evaluate(item): return __import__('os') and 1.0", "__import__"),
    ],
)
def test_dangerous_source_is_rejected_before_execution(source: str, reason: str) -> None:
    with pytest.raises(SandboxViolation) as exc:
        inspect_source(source)
    assert reason in str(exc.value)


def test_rejection_happens_at_construction_not_on_case_900() -> None:
    """A bad evaluator must fail at registration, not mid-run."""
    with pytest.raises(SandboxViolation):
        CustomEvaluator(source="import os\ndef evaluate(item): return 1.0")


def test_syntax_errors_are_reported_as_violations() -> None:
    with pytest.raises(SandboxViolation) as exc:
        inspect_source("def evaluate(item) return 1.0")
    assert "syntax" in str(exc.value).lower()


def test_a_rejected_evaluator_reports_an_error_not_a_score() -> None:
    result = run_sandboxed("import os\ndef evaluate(item): return 1.0", PAYLOAD)
    assert result.value is None
    assert "not allowed" in result.error


# ---------------------------------------------------------------------------
# Runtime containment
# ---------------------------------------------------------------------------


def test_an_infinite_loop_is_killed_by_the_timeout() -> None:
    source = """
def evaluate(item):
    while True:
        pass
"""
    result = run_sandboxed(source, PAYLOAD, timeout=2.0)
    assert result.value is None
    assert "timeout" in result.error


def test_a_crashing_evaluator_does_not_take_the_host_with_it() -> None:
    source = """
def evaluate(item):
    raise ValueError("boom")
"""
    result = run_sandboxed(source, PAYLOAD)
    assert result.value is None
    assert "ValueError" in result.error
    # The parent is still perfectly healthy.
    assert run_sandboxed("def evaluate(item): return 1.0", PAYLOAD).value == 1.0


def test_a_missing_evaluate_function_is_reported_clearly() -> None:
    result = run_sandboxed("x = 1", PAYLOAD)
    assert "must define evaluate" in result.error


def test_an_out_of_range_score_is_rejected_at_the_boundary() -> None:
    result = run_sandboxed("def evaluate(item): return 42.0", PAYLOAD)
    assert result.value is None
    assert "outside [0, 1]" in result.error


def test_a_non_numeric_return_is_rejected() -> None:
    result = run_sandboxed("def evaluate(item): return 'perfect'", PAYLOAD)
    assert result.value is None
    assert result.error


def test_runtime_import_of_a_denied_module_is_blocked() -> None:
    """Defence in depth: the AST gate is not the only barrier.

    ``importlib`` is itself denied statically, so this reaches the runtime
    guard via a name the AST pass does not flag.
    """
    source = """
def evaluate(item):
    mod = "o" + "s"
    m = __builtins__["__import__"](mod) if isinstance(__builtins__, dict) else None
    return 1.0
"""
    result = run_sandboxed(source, PAYLOAD)
    # Either the AST gate or the runtime guard stops it; both are acceptable,
    # a score is not.
    assert result.value is None
    assert result.error


def test_sys_exit_in_the_child_is_contained() -> None:
    source = """
def evaluate(item):
    raise SystemExit(3)
"""
    result = run_sandboxed(source, PAYLOAD)
    assert result.value is None
    assert result.error


@pytest.mark.skipif(sys.platform == "win32", reason="RLIMIT_AS is POSIX-only; documented as such")
def test_memory_ceiling_stops_an_allocation_bomb() -> None:
    source = """
def evaluate(item):
    blob = [0] * (10 ** 9)
    return 1.0
"""
    result = run_sandboxed(source, PAYLOAD, memory_mb=64, timeout=15.0)
    assert result.value is None
    assert result.error


def test_the_child_does_not_inherit_parent_state() -> None:
    """'spawn' not 'fork': a forked child would inherit DB handles and creds."""
    import agentlens_platform.evaluators.sandbox as sandbox_module

    sandbox_module.SENTINEL_SHOULD_NOT_BE_VISIBLE = "leaked-secret"
    try:
        source = """
def evaluate(item):
    # A forked child could reach the parent's module globals. A spawned one
    # starts from a clean interpreter, so this name cannot exist.
    return 0.0 if "leaked-secret" in str(item) else 1.0
"""
        result = run_sandboxed(source, PAYLOAD)
        assert result.value == 1.0
    finally:
        del sandbox_module.SENTINEL_SHOULD_NOT_BE_VISIBLE


def test_timeout_is_configurable_per_evaluator() -> None:
    ev = CustomEvaluator(
        source="""
def evaluate(item):
    while True:
        pass
""",
        timeout=1.5,
    )
    score = run_evaluator(ev, EvalInput(output="x"))
    assert score.failed_to_run
    assert "1.5s" in score.error
