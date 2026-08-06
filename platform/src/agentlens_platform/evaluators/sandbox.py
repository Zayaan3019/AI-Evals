"""Sandboxed execution of customer-supplied evaluator code.

Teams need evaluators the platform does not ship -- domain rules, internal
format checks, business logic. Accepting that code means accepting arbitrary
Python from outside the trust boundary, so it runs in a separate process with
limits the parent enforces.

**What this does and does not protect against, stated plainly.**

Enforced here:

* **Wall-clock timeout.** The child is killed if it overruns. An infinite loop
  cannot stall the eval run.
* **Memory ceiling** (POSIX only, via ``RLIMIT_AS``). An allocation bomb fails
  the child instead of the host.
* **Crash containment.** A segfault, a ``sys.exit``, or an unhandled exception
  kills the child, not the platform.
* **No inherited state.** The child is spawned, not forked, so it does not
  start with the parent's database handles, credentials, or open sockets.
* **Import denylist**, applied at compile time by AST inspection *and* at run
  time by a restricted ``__import__``.

**Not enforced**, and therefore not claimed:

* This is not a security boundary against a determined attacker. Python has too
  many escape routes -- ``ctypes``, C extensions, frame walking -- for an
  in-interpreter restriction to be airtight. Untrusted-in-the-adversarial-sense
  code needs an OS-level sandbox (a container with seccomp, gVisor, or a VM).
  What this provides is robust protection against *buggy and careless* code,
  which is the actual common case, plus a meaningful speed bump.
* ``RLIMIT_AS`` does not exist on Windows, so the memory ceiling is silently
  unavailable there. The timeout and process isolation still apply.

Saying this out loud matters more than the mechanism: an operator who believes
they have a security boundary they do not have will run code they should not.
"""

from __future__ import annotations

import ast
import multiprocessing
import os
import sys
import textwrap
from dataclasses import dataclass
from typing import Any

from .base import EvalInput, Evaluator, Score, Tier

#: Modules a custom evaluator has no legitimate need for. Blocking the module
#: is far more reliable than trying to block the dangerous callables inside it.
DENIED_IMPORTS = frozenset(
    {
        "os", "sys", "subprocess", "socket", "shutil", "pathlib", "importlib",
        "ctypes", "multiprocessing", "threading", "asyncio", "signal",
        "http", "urllib", "requests", "httpx", "ftplib", "smtplib", "telnetlib",
        "pickle", "shelve", "marshal", "builtins", "gc", "inspect", "code",
        "sqlite3", "psycopg", "asyncpg", "sqlalchemy", "clickhouse_connect",
        "webbrowser", "tempfile", "glob", "fileinput", "pty", "platform",
    }
)

#: Names that provide an escape hatch even without an import.
DENIED_NAMES = frozenset(
    {"__import__", "eval", "exec", "compile", "open", "globals", "locals", "vars",
     "breakpoint", "input", "memoryview", "exit", "quit"}
)

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MEMORY_MB = 256


class SandboxViolation(Exception):
    """Static inspection rejected the code before it ever ran."""


@dataclass(frozen=True, slots=True)
class SandboxResult:
    value: float | None
    explanation: str | None
    error: str | None


def inspect_source(source: str) -> None:
    """Reject obviously-unsafe code before executing it.

    A compile-time gate is cheap and catches the overwhelming majority of
    accidents. It is a filter, not a proof -- see the module docstring.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SandboxViolation(f"syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in DENIED_IMPORTS:
                    raise SandboxViolation(f"import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in DENIED_IMPORTS:
                raise SandboxViolation(f"import from {node.module!r} is not allowed")
        elif isinstance(node, ast.Name) and node.id in DENIED_NAMES:
            raise SandboxViolation(f"use of {node.id!r} is not allowed")
        elif isinstance(node, ast.Attribute):
            # Dunder attribute access is the standard route out of a restricted
            # namespace (``().__class__.__bases__[0].__subclasses__()``).
            if node.attr.startswith("__") and node.attr.endswith("__"):
                raise SandboxViolation(f"access to {node.attr!r} is not allowed")


def _child_entry(source: str, payload: dict[str, Any], memory_mb: int, pipe) -> None:
    """Runs in the spawned child. Must never raise into the parent."""
    try:
        if hasattr(os, "setsid"):
            # Detach from the parent's process group so a signal aimed at the
            # child cannot travel back up.
            os.setsid()

        try:
            import resource  # POSIX only

            limit = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        except (ImportError, ValueError, OSError):
            # Windows has no RLIMIT_AS. The timeout and isolation still stand.
            pass

        safe_builtins = {
            name: getattr(__builtins__, name, None)
            if not isinstance(__builtins__, dict)
            else __builtins__.get(name)
            for name in (
                "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
                "float", "format", "frozenset", "int", "isinstance", "issubclass",
                "len", "list", "map", "max", "min", "pow", "range", "repr", "reversed",
                "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip",
                "True", "False", "None", "ValueError", "TypeError", "KeyError",
                "IndexError", "ZeroDivisionError", "Exception",
            )
        }

        def guarded_import(name, *args, **kwargs):
            root = name.split(".")[0]
            if root in DENIED_IMPORTS:
                raise ImportError(f"import of {name!r} is not allowed in the sandbox")
            return __import__(name, *args, **kwargs)

        safe_builtins["__import__"] = guarded_import
        namespace: dict[str, Any] = {"__builtins__": safe_builtins}

        exec(compile(source, "<evaluator>", "exec"), namespace)  # noqa: S102
        function = namespace.get("evaluate")
        if not callable(function):
            pipe.send(("error", "custom evaluator must define evaluate(item) -> float"))
            return

        result = function(payload)
        explanation = None
        if isinstance(result, tuple):
            result, explanation = result[0], (str(result[1]) if len(result) > 1 else None)

        value = float(result)
        if not 0.0 <= value <= 1.0:
            pipe.send(("error", f"score {value} is outside [0, 1]"))
            return
        pipe.send(("ok", (value, explanation)))
    except BaseException as exc:  # noqa: BLE001 - the whole point is containment
        pipe.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        try:
            pipe.close()
        except Exception:  # noqa: BLE001
            pass


def run_sandboxed(
    source: str,
    payload: dict[str, Any],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> SandboxResult:
    """Execute ``source`` in a spawned child and return its verdict."""
    try:
        inspect_source(source)
    except SandboxViolation as exc:
        return SandboxResult(None, None, f"rejected before execution: {exc}")

    # 'spawn' rather than 'fork': a forked child inherits every open file
    # descriptor, database connection and credential in the parent. Spawn
    # starts a clean interpreter.
    context = multiprocessing.get_context("spawn")
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(
        target=_child_entry, args=(source, payload, memory_mb, child_conn), daemon=True
    )
    process.start()
    child_conn.close()

    try:
        if not parent_conn.poll(timeout):
            process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
            return SandboxResult(None, None, f"evaluator exceeded {timeout}s timeout")

        status, body = parent_conn.recv()
    except EOFError:
        # The child died without sending: a crash, an OOM kill, or sys.exit.
        process.join(timeout=2.0)
        return SandboxResult(
            None, None, f"evaluator process died (exit code {process.exitcode})"
        )
    finally:
        parent_conn.close()
        process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)

    if status == "error":
        return SandboxResult(None, None, str(body))
    value, explanation = body
    return SandboxResult(value, explanation, None)


class CustomEvaluator(Evaluator):
    """A customer-supplied scoring function, executed in the sandbox.

    Declared ``DETERMINISTIC`` only if the author says so; the registry verifies
    that claim by running the evaluator twice rather than trusting it.
    """

    slug = "custom"
    tier = Tier.DETERMINISTIC
    version = "1"

    def __init__(
        self,
        *,
        source: str,
        slug: str = "custom",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        memory_mb: int = DEFAULT_MEMORY_MB,
        threshold: float = 0.5,
    ) -> None:
        # Validate now so a bad evaluator fails at registration, not on case 900
        # of a 1000-case run.
        inspect_source(textwrap.dedent(source))
        self.slug = slug
        self._source = textwrap.dedent(source)
        super().__init__(
            source=self._source, timeout=timeout, memory_mb=memory_mb, threshold=threshold
        )

    def evaluate(self, item: EvalInput) -> Score:
        payload = {
            "output": item.output,
            "inputs": dict(item.inputs),
            "expected": item.expected,
            "context": list(item.context),
            "metadata": dict(item.metadata),
        }
        result = run_sandboxed(
            self._source,
            payload,
            timeout=self.config["timeout"],
            memory_mb=self.config["memory_mb"],
        )
        if result.error is not None:
            return Score.failure(result.error)
        return Score(
            value=result.value,
            passed=(result.value or 0.0) >= self.config["threshold"],
            explanation=result.explanation,
        )
