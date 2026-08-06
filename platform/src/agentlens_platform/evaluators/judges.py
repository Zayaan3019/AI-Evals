"""Backends for model-graded evaluation.

Two real providers, no stubs:

* **Anthropic** over the public Messages API, used when ``ANTHROPIC_API_KEY`` is
  present in the environment. The key is never read from config or stored in the
  database -- a judge configuration that carried a credential would end up in
  ``config_hash`` and, from there, into query results.
* **Ollama** over its local HTTP API, which needs no credential and is what the
  dogfooding runs on this machine use.

A judge call is a network call to a non-deterministic system, so everything here
is built around that being true: bounded retries, an explicit timeout, and
failures surfaced as failures rather than as a zero score.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
#: Pinned. The Messages API is versioned by date and an unpinned client can
#: change behaviour under you without a code change.
ANTHROPIC_VERSION = "2023-06-01"

DEFAULT_OLLAMA_URL = "http://localhost:11434"


@dataclass(frozen=True, slots=True)
class JudgeResponse:
    """What a judge backend returned, plus what it cost."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float = 0.0
    model: str = ""


class JudgeUnavailable(RuntimeError):
    """The backend could not be reached or is not configured.

    Distinct from a bad response: this means we never got a judgement, so the
    result must be recorded as an evaluator error, not as a low score.
    """


class JudgeBackend(Protocol):
    """Minimal contract a judge backend must satisfy."""

    model: str

    def complete(self, prompt: str, *, system: str, max_tokens: int, temperature: float) -> JudgeResponse:
        ...


class AnthropicJudge:
    """Real Anthropic Messages API client.

    Only constructed when a key is actually present; :func:`build_judge` decides.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        *,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise JudgeUnavailable("ANTHROPIC_API_KEY is not set")
        self._key = key
        self.model = model
        self._timeout = timeout
        self._max_retries = max_retries

    def complete(
        self, prompt: str, *, system: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> JudgeResponse:
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self._key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            started = time.perf_counter()
            try:
                response = httpx.post(
                    ANTHROPIC_API_URL, json=payload, headers=headers, timeout=self._timeout
                )
                if response.status_code in (429, 500, 502, 503, 529):
                    # Retryable. Anything else is a permanent error and retrying
                    # it just burns quota.
                    raise httpx.HTTPStatusError(
                        f"retryable status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                body = response.json()
                text = "".join(
                    block.get("text", "") for block in body.get("content", [])
                    if block.get("type") == "text"
                )
                usage = body.get("usage", {})
                return JudgeResponse(
                    text=text,
                    input_tokens=usage.get("input_tokens"),
                    output_tokens=usage.get("output_tokens"),
                    latency_ms=(time.perf_counter() - started) * 1000,
                    model=body.get("model", self.model),
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < self._max_retries:
                    # Exponential backoff. Judges are called in bulk, so a
                    # tight retry loop against a rate limit makes it worse.
                    time.sleep(2**attempt)
        raise JudgeUnavailable(f"anthropic judge failed: {last_error}") from last_error


class OllamaJudge:
    """Real local Ollama client.

    Uses ``/api/chat`` with streaming disabled. Token counts come from Ollama's
    own ``prompt_eval_count`` / ``eval_count``, so they are measured rather than
    estimated.
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        *,
        base_url: str | None = None,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self._base_url = (base_url or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_URL).rstrip("/")
        self._timeout = timeout

    def complete(
        self, prompt: str, *, system: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> JudgeResponse:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        started = time.perf_counter()
        try:
            response = httpx.post(
                f"{self._base_url}/api/chat", json=payload, timeout=self._timeout
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            raise JudgeUnavailable(f"ollama judge failed: {exc}") from exc

        return JudgeResponse(
            text=body.get("message", {}).get("content", ""),
            input_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
            latency_ms=(time.perf_counter() - started) * 1000,
            model=body.get("model", self.model),
        )


def ollama_available(base_url: str | None = None) -> bool:
    """Whether a local Ollama is reachable right now."""
    url = (base_url or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_URL).rstrip("/")
    try:
        return httpx.get(f"{url}/api/tags", timeout=2.0).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def build_judge(provider: str, model: str, **kwargs: Any) -> JudgeBackend:
    """Construct a judge backend by provider name.

    Raises :class:`JudgeUnavailable` rather than falling back to a different
    provider. A silent fallback would change what the scores mean without
    changing ``config_hash``, which is precisely the failure the hash exists to
    prevent.
    """
    if provider == "anthropic":
        return AnthropicJudge(model, **kwargs)
    if provider == "ollama":
        return OllamaJudge(model, **kwargs)
    raise JudgeUnavailable(f"unknown judge provider {provider!r}")


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response.

    Models wrap JSON in prose and fences however they like, and a judge whose
    output cannot be parsed is a judge that produces no signal. Tries the whole
    string first, then the outermost brace-delimited span.
    """
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise ValueError(f"no JSON object in judge response: {text[:200]!r}")
